# TTS Go SDK

这是一个用于调用TTS WebSocket流式接口的Go SDK。

## 功能特性

- 🔄 **流式音频合成**: 支持实时接收音频帧数据
- 🎵 **音频解码**: 自动解码base64编码的PCM音频数据
- 📁 **文件保存**: 支持将音频保存为WAV文件
- ⚡ **异步处理**: 支持并发处理和超时控制
- 🛡️ **错误处理**: 完善的错误处理和重连机制

## 安装依赖

```bash
# 初始化Go模块
go mod init tts-client

# 安装WebSocket依赖
go get github.com/gorilla/websocket
```

## 快速开始

### 1. 基本使用

```go
package main

import (
    "context"
    "log"
    "time"
    "tts-client/tts"
)

func main() {
    // 创建TTS客户端
    client := tts.NewTTSClient("ws://localhost:8421/ws/synthesize")
    defer client.Close()

    // 创建合成请求
    req := &tts.SynthesisRequest{
        Text:      "Hello, this is a test.",
        FrameSize: 2048,
        Speaker:   "default",
    }

    // 音频帧处理器
    frameHandler := func(frame *tts.AudioFrame) error {
        // 解码音频数据
        audioData, err := client.DecodeAudioFrame(frame)
        if err != nil {
            return err
        }

        log.Printf("收到音频帧 %d: %d 采样点", frame.FrameID, len(audioData))
        
        // 这里可以添加音频播放逻辑
        return nil
    }

    // 合成事件处理器
    responseHandler := func(response *tts.SynthesisResponse) error {
        switch response.Type {
        case "start":
            log.Printf("开始合成: %s", response.Text)
        case "complete":
            log.Printf("合成完成: 共%d帧", response.TotalFrames)
        }
        return nil
    }

    // 开始流式合成
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    err := client.SynthesizeStream(ctx, req, frameHandler, responseHandler)
    if err != nil {
        log.Fatalf("合成失败: %v", err)
    }
}
```

### 2. 自定义音频参数

```go
// 使用16000采样率、16bit、20ms帧时长
req := &tts.SynthesisRequest{
    Text:            "Hello, this is a test with custom audio parameters.",
    SampleRate:      16000,  // 16kHz采样率
    BitDepth:        16,     // 16位深度
    FrameDurationMs: 20,     // 20ms帧时长
    Speaker:         "default",
}

// 音频帧处理器（16位解码）
frameHandler := func(frame *tts.AudioFrame) error {
    audioData, err := client.DecodeAudioFrameWithFormat(frame, 16)
    if err != nil {
        return err
    }
    
    log.Printf("收到音频帧 %d: %d 采样点", frame.FrameID, len(audioData))
    return nil
}
```

### 2. 保存音频到文件

```go
func saveAudioToFile() {
    client := tts.NewTTSClient("ws://localhost:8421/ws/synthesize")
    defer client.Close()

    req := &tts.SynthesisRequest{
        Text:      "This will be saved to a file.",
        FrameSize: 2048,
    }

    var allAudioData []float32

    frameHandler := func(frame *tts.AudioFrame) error {
        audioData, err := client.DecodeAudioFrame(frame)
        if err != nil {
            return err
        }
        
        // 收集所有音频数据
        allAudioData = append(allAudioData, audioData...)
        return nil
    }

    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    err := client.SynthesizeStream(ctx, req, frameHandler, nil)
    if err != nil {
        log.Fatalf("合成失败: %v", err)
    }

    // 保存为WAV文件
    if err := saveWAVFile("output.wav", allAudioData, 22050); err != nil {
        log.Fatalf("保存文件失败: %v", err)
    }

    log.Printf("音频已保存到 output.wav")
}
```

## API 参考

### TTSClient

#### 方法

- `NewTTSClient(url string) *TTSClient`: 创建新的TTS客户端
- `Connect() error`: 连接到WebSocket服务器
- `Close() error`: 关闭连接
- `SynthesizeStream(ctx context.Context, req *SynthesisRequest, frameHandler AudioFrameHandler, responseHandler SynthesisHandler) error`: 流式合成音频
- `DecodeAudioFrame(frame *AudioFrame) ([]float32, error)`: 解码音频帧数据（默认32位）
- `DecodeAudioFrameWithFormat(frame *AudioFrame, bitDepth int) ([]float32, error)`: 根据位深度解码音频帧数据

### 数据结构

#### SynthesisRequest
```go
type SynthesisRequest struct {
    Text            string `json:"text"`                    // 要合成的文本
    FrameSize       int    `json:"frame_size,omitempty"`    // 音频帧大小
    Speaker         string `json:"speaker,omitempty"`       // 说话人
    SampleRate      int    `json:"sample_rate,omitempty"`   // 采样率，默认22050
    BitDepth        int    `json:"bit_depth,omitempty"`     // 位深度，默认32
    FrameDurationMs int    `json:"frame_duration_ms,omitempty"` // 帧时长（毫秒），可选
}
```

#### AudioFrame
```go
type AudioFrame struct {
    Type        string  `json:"type"`         // 消息类型
    FrameID     int     `json:"frame_id"`     // 帧ID
    Data        string  `json:"data"`         // base64编码的PCM数据
    TimestampMs float64 `json:"timestamp_ms"` // 时间戳
    IsLast      bool    `json:"is_last"`      // 是否为最后一帧
}
```

#### SynthesisResponse
```go
type SynthesisResponse struct {
    Type            string  `json:"type"`             // 消息类型
    Text            string  `json:"text"`             // 文本内容
    FrameSize       int     `json:"frame_size"`       // 帧大小
    Speaker         string  `json:"speaker"`          // 说话人
    SampleRate      int     `json:"sample_rate"`      // 采样率
    BitDepth        int     `json:"bit_depth"`        // 位深度
    FrameDurationMs int     `json:"frame_duration_ms"` // 帧时长
    AudioLength     int     `json:"audio_length"`     // 音频长度
    DurationMs      float64 `json:"duration_ms"`      // 持续时间
    TotalFrames     int     `json:"total_frames"`     // 总帧数
    TotalDurationMs float64 `json:"total_duration_ms"` // 总持续时间
    Error           string  `json:"error"`            // 错误信息
}
```

## 消息类型

- `start`: 开始合成
- `synthesized`: 音频合成完成
- `audio_frame`: 音频帧数据
- `complete`: 合成完成
- `error`: 错误信息

## 运行示例

```bash
# 进入示例目录
cd example

# 运行示例程序
go run main.go
```

## 注意事项

1. **音频格式**: 
   - 默认：32位浮点PCM格式，采样率22050Hz，单声道
   - 支持：16位整数PCM格式，采样率16000Hz，单声道
2. **帧大小**: 建议帧大小为2048采样点，或使用帧时长（如20ms）
3. **位深度**: 支持16位和32位，16位数据范围[-32768, 32767]，32位为浮点数
4. **超时设置**: 建议设置合理的超时时间，避免长时间等待
5. **错误处理**: 请妥善处理各种错误情况，包括网络错误和服务器错误
6. **资源清理**: 使用完毕后请调用`Close()`方法清理资源

## 依赖

- Go 1.21+
- github.com/gorilla/websocket v1.5.1 