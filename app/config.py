import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    """TTS服务配置"""
    
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8422  # 修改端口避免冲突
    WORKERS: int = 8  # 增加 worker 数量，充分利用 M3 Pro 性能
    
    # TTS模型配置 - 使用中文模型
    MODEL_NAME: str = "tts_models/zh-CN/baker/tacotron2-DDC-GST"  # 中文 Tacotron2-GST 模型
    VOCODER_NAME: str = "vocoder_models/universal/hifigan_v2"  # HiFiGAN声码器
    DEVICE: str = "auto"  # auto, cpu, cuda
    
    # 音频配置
    SAMPLE_RATE: int = 22050
    AUDIO_FORMAT: str = "wav"  # wav, mp3
    
    # 性能配置
    REQUEST_TIMEOUT: int = 30  # 请求超时时间（秒）
    MAX_TEXT_LENGTH: int = 500  # 最大文本长度
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/tts_server.log"
    
    class Config:
        env_file = ".env"

# 全局配置实例
settings = Settings()

def get_device():
    """获取推理设备"""
    if settings.DEVICE == "auto":
        import torch
        # 优先使用 MPS (Apple Silicon GPU)
        if torch.backends.mps.is_available():
            return "mps"
        # 然后尝试 CUDA
        elif torch.cuda.is_available():
            return "cuda"
        # 最后使用 CPU
        else:
            return "cpu"
    return settings.DEVICE 