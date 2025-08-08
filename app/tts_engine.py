import threading
import time
import logging
import asyncio
from typing import Dict, Any, Optional, List
import numpy as np
import torch
from TTS.api import TTS
from collections import deque

from .config import settings, get_device
from .utils import audio_to_base64, validate_text, format_response

logger = logging.getLogger(__name__)

class TTSEngine:
    """TTS推理引擎 - 每个worker独立加载模型"""
    
    def __init__(self, engine_id: int = 0):
        self.engine_id = engine_id
        self.device = get_device()
        self.sample_rate = settings.SAMPLE_RATE
        self.audio_format = settings.AUDIO_FORMAT
        self.max_text_length = settings.MAX_TEXT_LENGTH
        
        # 状态管理
        self._busy = False
        self._lock = threading.RLock()  # 可重入锁
        self._current_task = None
        self._start_time = None
        
        # 在初始化时直接加载模型，而不是使用threading.local()
        self.model = self._load_model()
        
        logger.info(f"TTS Engine {engine_id} initialized on device: {self.device}")
    
    @property
    def busy(self) -> bool:
        """获取引擎是否忙碌"""
        with self._lock:
            return self._busy
    
    @property
    def available(self) -> bool:
        """获取引擎是否可用"""
        return not self.busy
    
    def _set_busy(self, busy: bool, task_info: str = None):
        """设置引擎忙碌状态"""
        with self._lock:
            self._busy = busy
            self._current_task = task_info if busy else None
            self._start_time = time.time() if busy else None
    
    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        with self._lock:
            status = {
                "engine_id": self.engine_id,
                "busy": self._busy,
                "available": not self._busy,
                "current_task": self._current_task,
                "device": self.device,
                "model_loaded": self.model is not None
            }
            
            if self._busy and self._start_time:
                status["task_duration"] = time.time() - self._start_time
            
            return status
    
    def _load_model(self):
        """加载TTS模型"""
        try:
            logger.info(f"Loading TTS model for engine {self.engine_id} (thread {threading.current_thread().name})")
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
                model = TTS(
                    model_name=settings.MODEL_NAME,
                    progress_bar=False,
                    gpu=False  # MPS 设备不使用 gpu 参数
                )
                # 手动移动到 MPS 设备
                model.to("mps")
            else:
                model = TTS(
                    model_name=settings.MODEL_NAME,
                    progress_bar=False,
                    gpu=(self.device == "cuda")
                )
            
            load_time = time.time() - start_time
            logger.info(f"Engine {self.engine_id} model loaded successfully in {load_time:.2f}s")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model for engine {self.engine_id}: {e}")
            raise e
    
    def synthesize(self, text: str, speaker: str = "default") -> Dict[str, Any]:
        """合成语音 - 线程安全版本"""
        # 检查是否已经忙碌
        if self.busy:
            return format_response(
                success=False,
                error=f"Engine {self.engine_id} is busy with task: {self._current_task}"
            )
        
        # 设置忙碌状态
        task_info = f"synthesize: {text[:20]}{'...' if len(text) > 20 else ''}"
        self._set_busy(True, task_info)
        
        try:
            # 验证输入
            if not validate_text(text, self.max_text_length):
                return format_response(
                    success=False, 
                    error=f"Invalid text: length must be <= {self.max_text_length}"
                )
            
            # 直接使用已加载的模型进行推理
            start_time = time.time()
            
            # 执行TTS推理 - 对于单说话人模型，不传入 speaker 参数
            audio = self.model.tts(text)
            
            inference_time = time.time() - start_time
            logger.info(f"Engine {self.engine_id} TTS inference completed in {inference_time:.3f}s")
            
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
                    "inference_time": inference_time,
                    "engine_id": self.engine_id
                }
            )
            
        except Exception as e:
            logger.error(f"Engine {self.engine_id} TTS synthesis failed: {e}")
            return format_response(
                success=False,
                error=f"Synthesis failed: {str(e)}"
            )
        finally:
            # 无论成功还是失败，都要释放忙碌状态
            self._set_busy(False)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        try:
            return {
                "engine_id": self.engine_id,
                "model_name": settings.MODEL_NAME,
                "device": self.device,
                "sample_rate": self.sample_rate,
                "audio_format": self.audio_format,
                "max_text_length": self.max_text_length,
                "model_loaded": True,
                "busy": self.busy,
                "available": self.available
            }
        except Exception as e:
            return {
                "engine_id": self.engine_id,
                "model_loaded": False,
                "error": str(e),
                "busy": self.busy,
                "available": self.available
            }

class TTSEngineManager:
    """TTS引擎管理器 - 支持智能分配和排队机制"""
    
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or settings.WORKERS
        self.engines = []
        self.start_time = time.time()
        
        # 排队机制
        self.request_queue = deque()
        self.queue_lock = threading.RLock()
        self.max_queue_size = 100  # 最大队列长度
        self.queue_timeout = 30.0  # 队列等待超时时间（秒）
        
        # 统计信息
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.queue_full_count = 0
        self.timeout_count = 0
        
        logger.info(f"Initializing TTS Engine Manager with {self.num_workers} workers")
        logger.info(f"Using model: {settings.MODEL_NAME}")
        
        # 预加载所有worker的模型
        self._preload_models()
    
    def _preload_models(self):
        """预加载所有worker的模型"""
        logger.info("Preloading models for all workers...")
        
        for i in range(self.num_workers):
            try:
                engine = TTSEngine(i)
                # 模型已经在初始化时加载，这里只需要验证模型是否正常工作
                test_result = engine.synthesize("你好世界，这是一个文本转语音的测试。", "default")
                if test_result["success"]:
                    self.engines.append(engine)
                    logger.info(f"Worker {i} model loaded and tested successfully")
                else:
                    logger.error(f"Worker {i} model test failed: {test_result.get('error')}")
                    
            except Exception as e:
                logger.error(f"Failed to load model for worker {i}: {e}")
        
        logger.info(f"Successfully loaded {len(self.engines)} models")
    
    def get_available_engine(self) -> Optional[TTSEngine]:
        """获取可用的引擎 - 优先选择空闲的引擎"""
        if not self.engines:
            return None
        
        # 首先尝试找到空闲的引擎
        for engine in self.engines:
            if engine.available:
                return engine
        
        # 如果没有空闲引擎，返回None（需要排队）
        return None
    
    def get_engine_by_id(self, engine_id: int) -> Optional[TTSEngine]:
        """根据ID获取引擎"""
        if 0 <= engine_id < len(self.engines):
            return self.engines[engine_id]
        return None
    
    def add_to_queue(self, text: str, speaker: str = "default") -> Dict[str, Any]:
        """将请求添加到队列"""
        with self.queue_lock:
            if len(self.request_queue) >= self.max_queue_size:
                self.queue_full_count += 1
                return format_response(
                    success=False,
                    error=f"Queue is full (max size: {self.max_queue_size}). Please try again later."
                )
            
            # 创建队列项
            queue_item = {
                "text": text,
                "speaker": speaker,
                "timestamp": time.time(),
                "id": self.total_requests
            }
            
            self.request_queue.append(queue_item)
            self.total_requests += 1
            
            queue_position = len(self.request_queue)
            logger.info(f"Request {queue_item['id']} added to queue at position {queue_position}")
            
            return format_response(
                success=True,
                data={
                    "queued": True,
                    "queue_position": queue_position,
                    "estimated_wait_time": queue_position * 2.0,  # 估算等待时间
                    "request_id": queue_item["id"]
                }
            )
    
    def process_queue(self) -> Optional[Dict[str, Any]]:
        """处理队列中的请求"""
        with self.queue_lock:
            if not self.request_queue:
                return None
            
            # 检查队列中的第一个请求是否超时
            first_item = self.request_queue[0]
            if time.time() - first_item["timestamp"] > self.queue_timeout:
                # 超时，移除请求
                self.request_queue.popleft()
                self.timeout_count += 1
                logger.warning(f"Request {first_item['id']} timed out in queue")
                return format_response(
                    success=False,
                    error="Request timed out in queue",
                    data={"request_id": first_item["id"]}
                )
            
            # 尝试获取可用引擎
            engine = self.get_available_engine()
            if engine is None:
                return None  # 没有可用引擎，继续等待
            
            # 有可用引擎，处理队列中的第一个请求
            queue_item = self.request_queue.popleft()
            logger.info(f"Processing queued request {queue_item['id']} with engine {engine.engine_id}")
            
            # 执行合成
            result = engine.synthesize(queue_item["text"], queue_item["speaker"])
            
            # 添加请求ID到结果中
            if result["success"] and "data" in result:
                result["data"]["request_id"] = queue_item["id"]
            elif not result["success"]:
                result["data"] = {"request_id": queue_item["id"]}
            
            # 更新统计信息
            if result["success"]:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            
            return result
    
    def synthesize(self, text: str, speaker: str = "default", timeout: float = None) -> Dict[str, Any]:
        """使用智能分配策略合成语音"""
        if timeout is None:
            timeout = self.queue_timeout
        
        start_time = time.time()
        
        # 首先尝试获取可用引擎
        engine = self.get_available_engine()
        if engine is not None:
            # 有可用引擎，直接处理
            logger.info(f"Direct processing with engine {engine.engine_id}")
            result = engine.synthesize(text, speaker)
            
            # 更新统计信息
            if result["success"]:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            
            return result
        
        # 没有可用引擎，需要排队
        logger.info("No available engines, adding request to queue")
        queue_result = self.add_to_queue(text, speaker)
        
        if not queue_result["success"]:
            return queue_result
        
        request_id = queue_result["data"]["request_id"]
        
        # 等待处理
        wait_start = time.time()
        while time.time() - wait_start < timeout:
            # 尝试处理队列
            result = self.process_queue()
            if result is not None:
                # 检查是否是我们的请求
                result_request_id = result.get("data", {}).get("request_id")
                if result_request_id == request_id:
                    return result
            
            # 短暂等待后重试
            time.sleep(0.1)
        
        # 超时，从队列中移除请求
        with self.queue_lock:
            # 查找并移除超时的请求
            for i, item in enumerate(self.request_queue):
                if item["id"] == request_id:
                    self.request_queue.remove(item)
                    break
        
        self.timeout_count += 1
        return format_response(
            success=False,
            error=f"Request timed out after {timeout:.1f}s",
            data={"request_id": request_id}
        )
    
    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        from .utils import get_memory_usage, get_gpu_info, get_cpu_usage
        
        memory_info = get_memory_usage()
        gpu_info = get_gpu_info()
        cpu_info = get_cpu_usage()
        
        # 获取所有引擎的状态
        engine_statuses = []
        available_engines = 0
        busy_engines = 0
        
        for engine in self.engines:
            status = engine.get_status()
            engine_statuses.append(status)
            if status["available"]:
                available_engines += 1
            else:
                busy_engines += 1
        
        # 队列状态
        with self.queue_lock:
            queue_size = len(self.request_queue)
            queue_status = {
                "size": queue_size,
                "max_size": self.max_queue_size,
                "timeout": self.queue_timeout
            }
        
        return {
            "uptime": time.time() - self.start_time,
            "num_workers": len(self.engines),
            "total_workers": self.num_workers,
            "available_engines": available_engines,
            "busy_engines": busy_engines,
            "memory_usage": memory_info,
            "cpu_usage": cpu_info,
            "device": get_device(),
            "model_name": settings.MODEL_NAME,
            "gpu_info": gpu_info,
            "queue": queue_status,
            "statistics": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "queue_full_count": self.queue_full_count,
                "timeout_count": self.timeout_count
            },
            "engine_statuses": engine_statuses
        } 