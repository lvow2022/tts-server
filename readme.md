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

#### 2. 安装 PyTorch（GPU 支持）

```bash
# 对于 CUDA GPU (NVIDIA)
conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 对于 Apple Silicon (M1/M2/M3)
conda install pytorch torchaudio -c pytorch

# 或者使用 pip 安装（如果 conda 版本不可用）
pip install torch torchaudio
```

#### 3. 安装其他依赖

```bash
# 安装项目依赖
pip install -r requirements.txt
```

#### 4. 启动服务

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

打开浏览器访问：http://localhost:8422

## 压测

```bash
# 基本压测
python stress_test.py

# 自定义参数压测
python stress_test.py --url http://localhost:8422 --concurrent 4 --total 20
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

## 功能特性

- ✅ **多设备支持**：CPU、CUDA GPU、Apple Silicon MPS
- ✅ **实时监控**：内存使用趋势图
- ✅ **健康检查**：服务状态监控
- ✅ **压力测试**：性能测试工具
- ✅ **并发处理**：8 个 worker 支持
- ✅ **中文模型**：Tacotron2-DDC-GST