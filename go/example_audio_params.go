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
		fmt.Println("用法: go run example_audio_params.go <文本>")
		fmt.Println("示例: go run example_audio_params.go \"Hello, this is a test with custom audio parameters.\"")
		os.Exit(1)
	}

	text := os.Args[1]

	// 创建TTS客户端
	client := tts.NewTTSClient("ws://localhost:8421/ws/synthesize")
	defer client.Close()

	// 创建合成请求 - 使用16000采样率、16bit、20ms帧时长
	req := &tts.SynthesisRequest{
		Text:            text,
		SampleRate:      16000, // 16kHz采样率
		BitDepth:        16,    // 16位深度
		FrameDurationMs: 20,    // 20ms帧时长
		Speaker:         "default",
	}

	var totalFrames int
	var totalSamples int
	var currentBitDepth int

	// 音频帧处理器
	frameHandler := func(frame *tts.AudioFrame) error {
		// 解码音频数据（使用16位深度）
		audioData, err := client.DecodeAudioFrameWithFormat(frame, 16)
		if err != nil {
			return fmt.Errorf("解码音频帧失败: %w", err)
		}

		totalFrames++
		totalSamples += len(audioData)

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
			currentBitDepth = response.BitDepth
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
	fmt.Println("开始流式音频合成（16000Hz, 16bit, 20ms帧）...")
	startTime := time.Now()

	err := client.SynthesizeStream(ctx, req, frameHandler, responseHandler)
	if err != nil {
		log.Fatalf("合成失败: %v", err)
	}

	duration := time.Since(startTime)
	fmt.Printf("处理完成！总耗时: %v, 总帧数: %d, 总采样点: %d\n",
		duration, totalFrames, totalSamples)
}

// 保存音频到WAV文件的示例（支持16位）
func saveAudioToWAV16bit() {
	client := tts.NewTTSClient("ws://localhost:8421/ws/synthesize")
	defer client.Close()

	req := &tts.SynthesisRequest{
		Text:            "This audio will be saved as 16-bit WAV file.",
		SampleRate:      16000,
		BitDepth:        16,
		FrameDurationMs: 20,
		Speaker:         "default",
	}

	var allAudioData []float32

	frameHandler := func(frame *tts.AudioFrame) error {
		audioData, err := client.DecodeAudioFrameWithFormat(frame, 16)
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

	// 保存为16位WAV文件
	if err := saveWAVFile16bit("output_16bit.wav", allAudioData, 16000); err != nil {
		log.Fatalf("保存文件失败: %v", err)
	}

	log.Printf("16位音频已保存到 output_16bit.wav, 共 %d 采样点", len(allAudioData))
}

// 保存16位WAV文件
func saveWAVFile16bit(filename string, audioData []float32, sampleRate int) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// WAV文件头
	header := make([]byte, 44)

	// RIFF头
	copy(header[0:4], []byte("RIFF"))
	// 文件大小（减去8字节的RIFF头）
	fileSize := uint32(len(audioData)*2 + 36) // 16位 = 2字节/样本
	header[4] = byte(fileSize)
	header[5] = byte(fileSize >> 8)
	header[6] = byte(fileSize >> 16)
	header[7] = byte(fileSize >> 24)

	// WAVE标识
	copy(header[8:12], []byte("WAVE"))

	// fmt子块
	copy(header[12:16], []byte("fmt "))
	// fmt子块大小
	header[16] = 16
	header[17] = 0
	header[18] = 0
	header[19] = 0

	// 音频格式（PCM = 1）
	header[20] = 1
	header[21] = 0

	// 声道数（单声道 = 1）
	header[22] = 1
	header[23] = 0

	// 采样率
	header[24] = byte(sampleRate)
	header[25] = byte(sampleRate >> 8)
	header[26] = byte(sampleRate >> 16)
	header[27] = byte(sampleRate >> 24)

	// 字节率
	byteRate := sampleRate * 2 // 采样率 * 声道数 * 每样本字节数
	header[28] = byte(byteRate)
	header[29] = byte(byteRate >> 8)
	header[30] = byte(byteRate >> 16)
	header[31] = byte(byteRate >> 24)

	// 块对齐
	header[32] = 2 // 声道数 * 每样本字节数
	header[33] = 0

	// 每样本位数
	header[34] = 16
	header[35] = 0

	// data子块
	copy(header[36:40], []byte("data"))

	// 数据大小
	dataSize := uint32(len(audioData) * 2)
	header[40] = byte(dataSize)
	header[41] = byte(dataSize >> 8)
	header[42] = byte(dataSize >> 16)
	header[43] = byte(dataSize >> 24)

	// 写入文件头
	if _, err := file.Write(header); err != nil {
		return err
	}

	// 写入音频数据（转换为16位整数）
	for _, sample := range audioData {
		// 将float32转换为int16，范围[-32768, 32767]
		intSample := int16(sample * 32767.0)
		if intSample > 32767 {
			intSample = 32767
		} else if intSample < -32768 {
			intSample = -32768
		}

		// 小端序写入int16
		bytes := []byte{
			byte(intSample),
			byte(intSample >> 8),
		}
		if _, err := file.Write(bytes); err != nil {
			return err
		}
	}

	return nil
}
