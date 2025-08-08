import threading
import time
import logging
from typing import Dict, Any, Optional
import numpy as np
import torch
from TTS.api import TTS

from .config import settings, get_device
from .utils import audio_to_base64, validate_text, format_response

logger = logging.getLogger(__name__)

class TTSEngine:
    """TTS推理引擎 - 每个worker独立加载模型"""
    
    def __init__(self):
        self._local = threading.local()
        self.device = get_device()
        self.sample_rate = settings.SAMPLE_RATE
        self.audio_format = settings.AUDIO_FORMAT
        self.max_text_length = settings.MAX_TEXT_LENGTH
        
        logger.info(f"TTS Engine initialized on device: {self.device}")
    
    def _get_model(self):
        """获取当前线程的模型实例"""
        if not hasattr(self._local, 'model'):
            try:
                logger.info(f"Loading TTS model for thread {threading.current_thread().name}")
                logger.info(f"Model path: {settings.MODEL_NAME}")
                start_time = time.time()
                
                # 修复 PyTorch 2.6 兼容性问题
                import torch.serialization
                from TTS.utils.radam import RAdam
                import collections
                torch.serialization.add_safe_globals([
                    RAdam,
                    collections.defaultdict,
                    collections.OrderedDict,
                    collections.Counter,
                    dict,
                    list,
                    tuple,
                    set
                ])
                
                # 加载模型 - 适配 TTS 0.22.0 版本
                if self.device == "mps":
                    # MPS 设备需要特殊处理
                    self._local.model = TTS(
                        model_name=settings.MODEL_NAME,
                        progress_bar=False,
                        gpu=False  # MPS 设备不使用 gpu 参数
                    )
                    # 手动移动到 MPS 设备
                    self._local.model.to("mps")
                else:
                    self._local.model = TTS(
                        model_name=settings.MODEL_NAME,
                        progress_bar=False,
                        gpu=(self.device == "cuda")
                    )
                
                load_time = time.time() - start_time
                logger.info(f"Model loaded successfully in {load_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise e
        
        return self._local.model
    
    def synthesize(self, text: str, speaker: str = "default") -> Dict[str, Any]:
        """合成语音"""
        try:
            # 验证输入
            if not validate_text(text, self.max_text_length):
                return format_response(
                    success=False, 
                    error=f"Invalid text: length must be <= {self.max_text_length}"
                )
            
            # 获取模型并推理
            model = self._get_model()
            start_time = time.time()
            
            # 执行TTS推理 - 对于单说话人模型，不传入 speaker 参数
            audio = model.tts(text)
            
            inference_time = time.time() - start_time
            logger.info(f"TTS inference completed in {inference_time:.3f}s")
            
            # 转换为base64
            audio_base64 = audio_to_base64(audio, self.sample_rate, self.audio_format)
            
            return format_response(
                success=True,
                data={
                    "audio": audio_base64,
                    "sample_rate": self.sample_rate,
                    "format": self.audio_format,
                    "text": text,
                    "speaker": speaker,
                    "inference_time": inference_time
                }
            )
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return format_response(
                success=False,
                error=f"Synthesis failed: {str(e)}"
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        try:
            model = self._get_model()
            
            return {
                "model_name": settings.MODEL_NAME,
                "device": self.device,
                "sample_rate": self.sample_rate,
                "audio_format": self.audio_format,
                "max_text_length": self.max_text_length,
                "model_loaded": True
            }
        except Exception as e:
            return {
                "model_loaded": False,
                "error": str(e)
            }

class TTSEngineManager:
    """TTS引擎管理器"""
    
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or settings.WORKERS
        self.engines = []
        self.start_time = time.time()
        
        logger.info(f"Initializing TTS Engine Manager with {self.num_workers} workers")
        logger.info(f"Using model: {settings.MODEL_NAME}")
        
        # 预加载所有worker的模型
        self._preload_models()
    
    def _preload_models(self):
        """预加载所有worker的模型"""
        logger.info("Preloading models for all workers...")
        
        for i in range(self.num_workers):
            try:
                engine = TTSEngine()
                # 预热模型
                test_result = engine.synthesize("你好世界，这是一个文本转语音的测试。", "default")
                if test_result["success"]:
                    self.engines.append(engine)
                    logger.info(f"Worker {i} model loaded successfully")
                else:
                    logger.error(f"Worker {i} model test failed: {test_result.get('error')}")
                    
            except Exception as e:
                logger.error(f"Failed to load model for worker {i}: {e}")
        
        logger.info(f"Successfully loaded {len(self.engines)} models")
    
    def get_engine(self, worker_id: int = None) -> TTSEngine:
        """获取可用的引擎"""
        if not self.engines:
            raise RuntimeError("No TTS engines available")
        
        if worker_id is None:
            # 简单的轮询分配
            worker_id = int(time.time() * 1000) % len(self.engines)
        
        return self.engines[worker_id % len(self.engines)]
    
    def synthesize(self, text: str, speaker: str = "default") -> Dict[str, Any]:
        """使用任意可用引擎合成语音"""
        return self.get_engine().synthesize(text, speaker)
    
    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        from .utils import get_memory_usage, get_gpu_info
        
        memory_info = get_memory_usage()
        gpu_info = get_gpu_info()
        
        return {
            "uptime": time.time() - self.start_time,
            "num_workers": len(self.engines),
            "total_workers": self.num_workers,
            "memory_usage": memory_info,
            "device": get_device(),
            "model_name": settings.MODEL_NAME,
            "gpu_info": gpu_info
        } 