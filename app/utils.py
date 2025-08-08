import logging
import os
import time
import base64
import io
import numpy as np
import soundfile as sf
from typing import Dict, Any, Optional

def setup_logging(log_file: str, log_level: str = "INFO"):
    """设置日志配置"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.handlers.clear()  # 清除现有处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

def audio_to_base64(audio: np.ndarray, sample_rate: int, format: str = "wav") -> str:
    """将音频数据转换为base64编码"""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format=format)
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return audio_base64

def base64_to_audio(audio_base64: str) -> np.ndarray:
    """将base64编码转换为音频数据"""
    audio_bytes = base64.b64decode(audio_base64)
    buffer = io.BytesIO(audio_bytes)
    audio, sample_rate = sf.read(buffer)
    return audio

def validate_text(text: str, max_length: int = 500) -> bool:
    """验证文本输入"""
    if not text or not text.strip():
        return False
    if len(text) > max_length:
        return False
    return True

def format_response(success: bool, data: Any = None, error: str = None) -> Dict[str, Any]:
    """格式化响应数据"""
    response = {
        "success": success,
        "timestamp": time.time()
    }
    
    if success and data is not None:
        response["data"] = data
    elif not success and error:
        response["error"] = error
    
    return response

def get_memory_usage() -> Dict[str, float]:
    """获取内存使用情况"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    
    # 获取系统总内存
    system_memory = psutil.virtual_memory()
    
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,  # RSS内存（MB）
        "vms_mb": memory_info.vms / 1024 / 1024,  # VMS内存（MB）
        "percent": process.memory_percent(),  # 进程内存使用百分比
        "system_total_mb": system_memory.total / 1024 / 1024,  # 系统总内存（MB）
        "system_available_mb": system_memory.available / 1024 / 1024,  # 系统可用内存（MB）
        "system_used_mb": system_memory.used / 1024 / 1024,  # 系统已用内存（MB）
        "system_percent": system_memory.percent  # 系统内存使用百分比
    }

def get_cpu_usage() -> Dict[str, float]:
    """获取CPU使用情况"""
    import psutil
    process = psutil.Process()
    
    # 获取进程CPU使用率（不阻塞）
    try:
        process_percent = process.cpu_percent()
    except:
        process_percent = 0.0
    
    # 获取系统CPU使用率（不阻塞）
    try:
        system_percent = psutil.cpu_percent()
    except:
        system_percent = 0.0
    
    # 获取CPU核心数
    try:
        cpu_count = psutil.cpu_count()
    except:
        cpu_count = 0
    
    # 获取CPU频率 - 修复频率获取问题
    cpu_freq = None
    try:
        freq_info = psutil.cpu_freq()
        if freq_info and freq_info.current > 1000:  # 确保频率合理（>1GHz）
            cpu_freq = freq_info.current
        else:
            # 如果频率不合理，尝试获取其他信息
            cpu_freq = None
    except:
        cpu_freq = None
    
    return {
        "process_percent": round(process_percent, 2),
        "system_percent": round(system_percent, 2),
        "cpu_count": cpu_count,
        "cpu_freq": cpu_freq
    }

def get_gpu_info() -> Dict[str, Any]:
    """获取GPU信息"""
    try:
        import torch
        
        # 检查 MPS (Apple Silicon GPU)
        if torch.backends.mps.is_available():
            # 对于MPS，我们只能获取基本信息
            return {
                "available": True,
                "device_type": "mps",
                "device_count": 1,
                "name": "Apple Silicon GPU",
                "memory_total": None,  # MPS不提供内存信息
                "memory_used": None,
                "memory_free": None,
                "memory_reserved": None,
                "temperature": None,
                "utilization": None,
                "devices": [
                    {
                        "id": 0,
                        "name": "Apple Silicon GPU",
                        "device_type": "mps",
                        "memory_info": "MPS memory info not available",
                        "message": "Using Apple Silicon GPU acceleration"
                    }
                ]
            }
        
        # 检查 CUDA
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            
            if gpu_count == 0:
                return {
                    "available": False,
                    "message": "No GPU devices found"
                }
            
            # 获取当前设备
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            
            # 获取GPU内存信息
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024 / 1024  # MB
            memory_reserved = torch.cuda.memory_reserved(current_device) / 1024 / 1024    # MB
            memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024 / 1024  # MB
            memory_free = memory_total - memory_allocated
            
            # 尝试获取GPU利用率（需要nvidia-ml-py）
            utilization = None
            temperature = None
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(current_device)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except ImportError:
                # pynvml不可用，跳过
                pass
            except Exception:
                # 其他错误，跳过
                pass
            
            gpu_info = {
                "available": True,
                "device_type": "cuda",
                "device_count": gpu_count,
                "name": gpu_name,
                "memory_total": round(memory_total, 2),
                "memory_used": round(memory_allocated, 2),
                "memory_free": round(memory_free, 2),
                "memory_reserved": round(memory_reserved, 2),
                "temperature": temperature,
                "utilization": utilization,
                "devices": []
            }
            
            # 添加所有设备信息
            for i in range(gpu_count):
                device_name = torch.cuda.get_device_name(i)
                device_memory_allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                device_memory_reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
                device_memory_total = torch.cuda.get_device_properties(i).total_memory / 1024 / 1024
                
                device_info = {
                    "id": i,
                    "name": device_name,
                    "device_type": "cuda",
                    "memory_allocated_mb": round(device_memory_allocated, 2),
                    "memory_reserved_mb": round(device_memory_reserved, 2),
                    "memory_total_mb": round(device_memory_total, 2),
                    "memory_usage_percent": round((device_memory_allocated / device_memory_total) * 100, 2) if device_memory_total > 0 else 0
                }
                
                gpu_info["devices"].append(device_info)
            
            return gpu_info
        
        # 没有可用的 GPU
        return {
            "available": False,
            "message": "No GPU acceleration available"
        }
        
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        } 