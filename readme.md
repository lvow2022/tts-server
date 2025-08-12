# TTS Server - 高并发中文语音合成服务

基于 FastAPI + FastSpeech2 + HiFiGAN 的高并发中文语音合成服务，专为压测场景设计。

## 配置

### 1. 模型配置

在 `app/config.py` 中修改模型路径：

```python
MODEL_NAME: str = "tts_models/zh-CN/baker/tacotron2-DDC"  # 如需更换模型，修改此行
VOCODER_NAME: str = "vocoder_models/universal/hifigan_v2"  # HiFiGAN声码器
```

### 2. 服务配置

```python
HOST: str = "0.0.0.0"      # 服务地址
PORT: int = 8421           # 服务端口
WORKERS: int = 1           # worker线程数
DEVICE: str = "auto"       # 推理设备 (auto/cpu/cuda)
```

## 环境安装和启动

### 方式一：Conda 环境（推荐）

#### 1. 创建 Conda 环境

```bash
# 创建新的 conda 环境
conda create -n tts-server python=3.10

# 激活环境
conda activate tts-server
```

#### 2. 安装其他依赖

```bash
# 安装项目依赖
pip install -r requirements.txt
```

#### 3. 启动服务

```bash
# 启动 TTS 服务（首次运行会自动下载模型）
python run.py
```

### 方式二：Python venv 环境

#### 1. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate     # Windows
```

#### 2. 安装依赖

```bash
# 安装精简后的依赖
pip install -r requirements.txt
```

#### 3. 启动服务

```bash
# 启动 TTS 服务（首次运行会自动下载模型）
python run.py
```

## 访问测试页面

打开浏览器访问：http://localhost:8421

### WebSocket流式接口测试

访问：http://localhost:8421/static/websocket_test.html

支持实时音频流式播放，体验更流畅的语音合成效果。

## 压测

```bash
# 基本压测
python stress_test.py

# 自定义参数压测
python stress_test.py --url http://localhost:8421 --concurrent 4 --total 20

# 指定语言测试
python stress_test.py --lang en  # 只测试英文
python stress_test.py --lang zh  # 只测试中文
python stress_test.py --lang both  # 中英文混合测试
```

## API接口

### REST API

#### 语音合成接口
```
POST /synthesize
Content-Type: application/json

{
    "text": "要合成的文本",
    "speaker": "default",
    "timeout": 30.0
}
```

### WebSocket API

#### 流式语音合成接口
```
WebSocket: ws://localhost:8421/ws/synthesize
```

**客户端请求:**
```json
{
    "text": "要合成的文本",
    "frame_size": 2048,
    "speaker": "default"
}
```

**服务端响应序列:**
```json
// 1. 开始消息
{
    "type": "start",
    "text": "要合成的文本",
    "frame_size": 2048,
    "speaker": "default"
}

// 2. 合成完成消息
{
    "type": "synthesized",
    "audio_length": 48576,
    "duration_ms": 2200
}

// 3. 音频帧消息（多个）
{
    "type": "audio_frame",
    "frame_id": 1,
    "data": "base64_encoded_audio_data",
    "timestamp_ms": 0,
    "is_last": false
}

// 4. 完成消息
{
    "type": "complete",
    "total_frames": 24,
    "total_duration_ms": 2200
}
```

## 设备支持

### 自动设备检测

程序会自动检测并使用最佳可用设备：

1. **Apple Silicon GPU** (MPS) - macOS + M1/M2/M3
2. **NVIDIA GPU** (CUDA) - Windows/Linux + NVIDIA GPU  
3. **CPU** - 所有平台

### 性能对比

| 设备类型 | 响应时间 | QPS | 适用场景 |
|----------|----------|-----|----------|
| **CPU** | 1.4 秒 | 2.8 | 通用，稳定 |
| **MPS (M3 Pro)** | 1.1 秒 | 3.2 | macOS 优化 |
| **CUDA (RTX 4090)** | 0.3 秒 | 13.3 | 高性能 |

## 依赖说明

精简后的依赖包（9个必需包）：
- `fastapi` - Web框架
- `uvicorn` - ASGI服务器  
- `TTS` - 核心TTS功能
- `torch` - PyTorch深度学习框架
- `torchaudio` - 音频处理
- `numpy` - 数值计算
- `soundfile` - 音频文件处理
- `pydantic` - 数据验证
- `psutil` - 系统监控

