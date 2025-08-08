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

## Mac 环境安装和启动

### 1. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
source venv/bin/activate
```

### 2. 安装依赖

```bash
# 安装精简后的依赖
pip install -r requirements.txt
```

### 3. 启动服务

```bash
# 启动TTS服务（首次运行会自动下载模型）
python run.py
```

### 4. 访问测试页面

打开浏览器访问：http://localhost:8421

## 压测

```bash
python stress_test.py
```

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