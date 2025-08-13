package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"
	"tts-client/tts"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("用法: go run test_decode.go <文本>")
		fmt.Println("示例: go run test_decode.go \"Hello, this is a test.\"")
		os.Exit(1)
	}

	text := os.Args[1]

	// 创建TTS客户端
	client := tts.NewTTSClient("ws://192.168.1.167:8421/ws/synthesize")
	defer client.Close()

	// 测试不同的音频参数
	testCases := []struct {
		name   string
		params tts.SynthesisRequest
	}{
		{
			name: "16000Hz, 16bit, 20ms",
			params: tts.SynthesisRequest{
				Text:            text,
				SampleRate:      16000,
				BitDepth:        16,
				FrameDurationMs: 20,
				Speaker:         "default",
			},
		},
	}

	for _, testCase := range testCases {
		fmt.Printf("\n=== 测试: %s ===\n", testCase.name)

		var totalFrames int
		var totalSamples int
		var maxValue float32
		var minValue float32
		var hasNonZero bool

		// 音频帧处理器
		frameHandler := func(frame *tts.AudioFrame) error {
			// 根据测试用例选择解码方式
			var audioData []float32
			var err error

			if testCase.params.BitDepth == 16 {
				audioData, err = client.DecodeAudioFrameWithFormat(frame, 16)
			} else {
				audioData, err = client.DecodeAudioFrameWithFormat(frame, 32)
			}

			if err != nil {
				return fmt.Errorf("解码音频帧失败: %w", err)
			}

			totalFrames++
			totalSamples += len(audioData)

			// 统计音频数据
			for _, sample := range audioData {
				if sample > maxValue {
					maxValue = sample
				}
				if sample < minValue {
					minValue = sample
				}
				if sample != 0 {
					hasNonZero = true
				}
			}

			fmt.Printf("帧 %d: %d 采样点, 时间戳: %.0fms\n",
				frame.FrameID, len(audioData), frame.TimestampMs)

			return nil
		}

		// 合成事件处理器
		responseHandler := func(response *tts.SynthesisResponse) error {
			switch response.Type {
			case "start":
				fmt.Printf("开始合成: %s\n", response.Text)
				fmt.Printf("音频参数: 采样率=%dHz, 位深度=%dbit, 帧时长=%dms\n",
					response.SampleRate, response.BitDepth, response.FrameDurationMs)
			case "synthesized":
				fmt.Printf("音频合成完成: %d 采样点, %.0fms\n",
					response.AudioLength, response.DurationMs)
			case "complete":
				fmt.Printf("合成完成: 共%d帧, 总时长%.0fms\n",
					response.TotalFrames, response.TotalDurationMs)
			case "error":
				fmt.Printf("合成错误: %s\n", response.Error)
			}
			return nil
		}

		// 创建上下文（带超时）
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		// 开始流式合成
		fmt.Println("开始流式音频合成...")
		startTime := time.Now()

		err := client.SynthesizeStream(ctx, &testCase.params, frameHandler, responseHandler)
		if err != nil {
			log.Printf("合成失败: %v", err)
			continue
		}

		duration := time.Since(startTime)
		fmt.Printf("处理完成！总耗时: %v, 总帧数: %d, 总采样点: %d\n",
			duration, totalFrames, totalSamples)

		// 输出音频统计信息
		fmt.Printf("音频统计: 最大值=%.6f, 最小值=%.6f, 包含非零值=%v\n",
			maxValue, minValue, hasNonZero)

		if !hasNonZero {
			fmt.Println("⚠️  警告: 所有音频样本都是零值，可能存在解码问题！")
		} else if maxValue > 1.0 || minValue < -1.0 {
			fmt.Println("⚠️  警告: 音频值超出[-1.0, 1.0]范围，可能存在解码问题！")
		} else {
			fmt.Println("✅ 音频数据看起来正常")
		}
	}
}
