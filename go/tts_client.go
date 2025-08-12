package tts

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// TTSClient TTS客户端
type TTSClient struct {
	conn     *websocket.Conn
	url      string
	mu       sync.Mutex
	isClosed bool
}

// AudioFrame 音频帧数据
type AudioFrame struct {
	Type        string  `json:"type"`
	FrameID     int     `json:"frame_id"`
	Data        string  `json:"data"` // base64编码的PCM数据
	TimestampMs float64 `json:"timestamp_ms"`
	IsLast      bool    `json:"is_last"`
}

// SynthesisRequest 合成请求
type SynthesisRequest struct {
	Text            string `json:"text"`
	FrameSize       int    `json:"frame_size,omitempty"`
	Speaker         string `json:"speaker,omitempty"`
	SampleRate      int    `json:"sample_rate,omitempty"`       // 采样率，默认22050
	BitDepth        int    `json:"bit_depth,omitempty"`         // 位深度，默认32
	FrameDurationMs int    `json:"frame_duration_ms,omitempty"` // 帧时长（毫秒），可选
}

// SynthesisResponse 合成响应
type SynthesisResponse struct {
	Type            string  `json:"type"`
	Text            string  `json:"text,omitempty"`
	FrameSize       int     `json:"frame_size,omitempty"`
	Speaker         string  `json:"speaker,omitempty"`
	SampleRate      int     `json:"sample_rate,omitempty"`
	BitDepth        int     `json:"bit_depth,omitempty"`
	FrameDurationMs int     `json:"frame_duration_ms,omitempty"`
	AudioLength     int     `json:"audio_length,omitempty"`
	DurationMs      float64 `json:"duration_ms,omitempty"`
	TotalFrames     int     `json:"total_frames,omitempty"`
	TotalDurationMs float64 `json:"total_duration_ms,omitempty"`
	Error           string  `json:"error,omitempty"`
}

// AudioFrameHandler 音频帧处理函数
type AudioFrameHandler func(frame *AudioFrame) error

// SynthesisHandler 合成事件处理函数
type SynthesisHandler func(response *SynthesisResponse) error

// NewTTSClient 创建新的TTS客户端
func NewTTSClient(url string) *TTSClient {
	return &TTSClient{
		url: url,
	}
}

// Connect 连接到WebSocket服务器
func (c *TTSClient) Connect() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.conn != nil {
		return nil
	}

	conn, _, err := websocket.DefaultDialer.Dial(c.url, nil)
	if err != nil {
		return fmt.Errorf("连接WebSocket失败: %w", err)
	}

	c.conn = conn
	c.isClosed = false
	return nil
}

// Close 关闭连接
func (c *TTSClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.conn != nil {
		err := c.conn.Close()
		c.conn = nil
		c.isClosed = true
		return err
	}
	return nil
}

// SynthesizeStream 流式合成音频
func (c *TTSClient) SynthesizeStream(ctx context.Context, req *SynthesisRequest,
	frameHandler AudioFrameHandler, responseHandler SynthesisHandler) error {

	if err := c.Connect(); err != nil {
		return err
	}

	// 发送合成请求
	if err := c.conn.WriteJSON(req); err != nil {
		return fmt.Errorf("发送请求失败: %w", err)
	}

	// 设置读取超时
	c.conn.SetReadDeadline(time.Now().Add(30 * time.Second))

	// 监听消息
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			_, message, err := c.conn.ReadMessage()
			if err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					return fmt.Errorf("WebSocket连接异常关闭: %w", err)
				}
				return fmt.Errorf("读取消息失败: %w", err)
			}

			// 解析JSON消息
			var response SynthesisResponse
			if err := json.Unmarshal(message, &response); err != nil {
				return fmt.Errorf("解析JSON失败: %w", err)
			}

			// 处理不同类型的消息
			switch response.Type {
			case "start":
				if responseHandler != nil {
					if err := responseHandler(&response); err != nil {
						log.Printf("处理start消息失败: %v", err)
					}
				}
				log.Printf("开始合成: %s", response.Text)

			case "synthesized":
				if responseHandler != nil {
					if err := responseHandler(&response); err != nil {
						log.Printf("处理synthesized消息失败: %v", err)
					}
				}
				log.Printf("音频合成完成: %d 采样点, %.0fms", response.AudioLength, response.DurationMs)

			case "audio_frame":
				// 解析音频帧
				var frame AudioFrame
				if err := json.Unmarshal(message, &frame); err != nil {
					log.Printf("解析音频帧失败: %v", err)
					continue
				}

				if frameHandler != nil {
					if err := frameHandler(&frame); err != nil {
						log.Printf("处理音频帧失败: %v", err)
					}
				}

			case "complete":
				if responseHandler != nil {
					if err := responseHandler(&response); err != nil {
						log.Printf("处理complete消息失败: %v", err)
					}
				}
				log.Printf("合成完成: 共%d帧, 总时长%.0fms", response.TotalFrames, response.TotalDurationMs)
				return nil

			case "error":
				if responseHandler != nil {
					if err := responseHandler(&response); err != nil {
						log.Printf("处理error消息失败: %v", err)
					}
				}
				return fmt.Errorf("合成错误: %s", response.Error)

			default:
				log.Printf("未知消息类型: %s", response.Type)
			}
		}
	}
}

// DecodeAudioFrame 解码音频帧数据
func (c *TTSClient) DecodeAudioFrame(frame *AudioFrame) ([]float32, error) {
	return c.DecodeAudioFrameWithFormat(frame, 32)
}

// DecodeAudioFrameWithFormat 根据位深度解码音频帧数据
func (c *TTSClient) DecodeAudioFrameWithFormat(frame *AudioFrame, bitDepth int) ([]float32, error) {
	// 解码base64数据
	data, err := base64.StdEncoding.DecodeString(frame.Data)
	if err != nil {
		return nil, fmt.Errorf("base64解码失败: %w", err)
	}

	bytesPerSample := bitDepth / 8

	// 检查数据长度是否为字节数的倍数
	if len(data)%bytesPerSample != 0 {
		return nil, fmt.Errorf("数据长度 %d 不是 %d 的倍数", len(data), bytesPerSample)
	}

	sampleCount := len(data) / bytesPerSample
	audioData := make([]float32, sampleCount)

	switch bitDepth {
	case 16:
		// 16位有符号整数
		for i := 0; i < len(data); i += 2 {
			// 小端序读取int16
			sample := int16(data[i]) | int16(data[i+1])<<8
			// 转换为float32，范围[-1.0, 1.0]
			audioData[i/2] = float32(sample) / 32768.0
		}
	case 32:
		// 32位浮点数
		for i := 0; i < len(data); i += 4 {
			// 小端序读取float32
			bits := uint32(data[i]) | uint32(data[i+1])<<8 | uint32(data[i+2])<<16 | uint32(data[i+3])<<24
			audioData[i/4] = float32(bits)
		}
	default:
		return nil, fmt.Errorf("不支持的位深度: %d", bitDepth)
	}

	return audioData, nil
}

// SaveAudioFrameToFile 保存音频帧到文件（用于调试）
func (c *TTSClient) SaveAudioFrameToFile(frame *AudioFrame, filename string) error {
	audioData, err := c.DecodeAudioFrame(frame)
	if err != nil {
		return err
	}

	// 这里可以添加保存到WAV文件的逻辑
	// 为了简化，这里只是打印信息
	log.Printf("音频帧 %d: %d 采样点, 保存到 %s", frame.FrameID, len(audioData), filename)
	return nil
}
