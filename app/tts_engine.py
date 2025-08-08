import threading
import time
import logging
import asyncio
from typing import Dict, Any, Optional, List
import numpy as np
import torch
from TTS.api import TTS
from collections import deque
import queue
from concurrent.futures import Future

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
        
        # 在初始化时直接加载模型
        self.model = self._load_model()
        
        logger.info(f"TTS Engine {engine_id} initialized on device: {self.device}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            "engine_id": self.engine_id,
            "device": self.device,
            "model_loaded": self.model is not None
        }
    
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
            elif self.device == "cuda":
                # CUDA 设备需要特殊处理
                import torch
                # 确保CUDA可用
                if not torch.cuda.is_available():
                    logger.warning(f"CUDA not available for engine {self.engine_id}, falling back to CPU")
                    model = TTS(
                        model_name=settings.MODEL_NAME,
                        progress_bar=False,
                        gpu=False
                    )
                else:
                    # 为每个worker分配不同的GPU（如果有多个GPU）
                    gpu_id = self.engine_id % torch.cuda.device_count()
                    torch.cuda.set_device(gpu_id)
                    logger.info(f"Engine {self.engine_id} using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                    
                    # 加载模型
                    model = TTS(
                        model_name=settings.MODEL_NAME,
                        progress_bar=False,
                        gpu=True
                    )
            else:
                # CPU 设备
                model = TTS(
                    model_name=settings.MODEL_NAME,
                    progress_bar=False,
                    gpu=False
                )
            
            load_time = time.time() - start_time
            logger.info(f"Engine {self.engine_id} model loaded successfully in {load_time:.2f}s on device: {self.device}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model for engine {self.engine_id}: {e}")
            raise e
    
    def synthesize(self, text: str, speaker: str = "default") -> Dict[str, Any]:
        """合成语音 - 简化版本，无状态管理"""
        try:
            # 验证输入
            if not validate_text(text, self.max_text_length):
                return format_response(
                    success=False, 
                    error=f"Invalid text: length must be <= {self.max_text_length}"
                )
            
            # 直接使用已加载的模型进行推理
            start_time = time.time()
            
            # 执行TTS推理
            with torch.no_grad():  # 禁用梯度计算以提高性能
                audio = self.model.tts(text)
            
            inference_time = time.time() - start_time
            logger.info(f"Engine {self.engine_id} TTS inference completed in {inference_time:.3f}s on {self.device}")
            
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
                "model_loaded": True
            }
        except Exception as e:
            return {
                "engine_id": self.engine_id,
                "model_loaded": False,
                "error": str(e)
            }
    


class TTSEngineManager:
    """TTS引擎管理器 - 使用生产者-消费者模式"""
    
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or settings.WORKERS
        self.engines = []
        self.start_time = time.time()
        
        # 请求队列 - 使用线程安全的Queue
        self.request_queue = queue.Queue(maxsize=100)
        self.max_queue_size = 100
        
        # 结果存储 - 使用字典存储Future对象
        self.results = {}
        self.results_lock = threading.RLock()
        
        # 统计信息
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.queue_full_count = 0
        self.timeout_count = 0
        
        # 线程控制
        self.worker_threads = []
        self.running = True
        
        logger.info(f"Initializing TTS Engine Manager with {self.num_workers} workers")
        logger.info(f"Using model: {settings.MODEL_NAME}")
        
        # 预加载所有worker的模型
        self._preload_models()
        
        # 启动worker线程
        self._start_worker_threads()
    
    def _preload_models(self):
        """预加载所有worker的模型"""
        logger.info("Preloading models for all workers...")
        
        for i in range(self.num_workers):
            try:
                engine = TTSEngine(i)
                # 模型已经在初始化时加载，直接添加到引擎列表
                self.engines.append(engine)
                logger.info(f"Worker {i} model loaded successfully")
                    
            except Exception as e:
                logger.error(f"Failed to load model for worker {i}: {e}")
        
        logger.info(f"Successfully loaded {len(self.engines)} models")
    
    def _start_worker_threads(self):
        """启动worker线程"""
        logger.info("Starting worker threads...")
        
        for i, engine in enumerate(self.engines):
            thread = threading.Thread(
                target=self._worker_loop,
                args=(engine,),
                name=f"TTSWorker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
            logger.info(f"Started worker thread {i} for engine {engine.engine_id}")
    
    def _worker_loop(self, engine: TTSEngine):
        """Worker线程的主循环"""
        logger.info(f"Worker thread {engine.engine_id} started")
        
        while self.running:
            try:
                # 从队列中获取请求，超时时间为1秒
                try:
                    request = self.request_queue.get(timeout=1.0)
                except queue.Empty:
                    # 队列为空，继续循环
                    continue
                
                # 处理请求
                logger.info(f"Worker {engine.engine_id} processing request {request['id']}")
                result = engine.synthesize(request["text"], request["speaker"])
                
                # 添加请求ID到结果中
                if result["success"] and "data" in result:
                    result["data"]["request_id"] = request["id"]
                elif not result["success"]:
                    result["data"] = {"request_id": request["id"]}
                
                # 更新统计信息
                if result["success"]:
                    self.successful_requests += 1
                else:
                    self.failed_requests += 1
                
                # 设置结果
                with self.results_lock:
                    if request["id"] in self.results:
                        future = self.results[request["id"]]
                        future.set_result(result)
                        del self.results[request["id"]]
                
                # 标记任务完成
                self.request_queue.task_done()
                
                logger.info(f"Worker {engine.engine_id} completed request {request['id']}")
                
            except Exception as e:
                logger.error(f"Worker {engine.engine_id} error: {e}")
                # 如果队列中有任务，标记为完成
                try:
                    self.request_queue.task_done()
                except:
                    pass
        
        logger.info(f"Worker thread {engine.engine_id} stopped")
    
    def synthesize(self, text: str, speaker: str = "default", timeout: float = None) -> Dict[str, Any]:
        """添加请求到队列并等待结果"""
        if timeout is None:
            timeout = 30.0
        
        # 创建请求
        request_id = self.total_requests
        request = {
            "text": text,
            "speaker": speaker,
            "timestamp": time.time(),
            "id": request_id
        }
        
        self.total_requests += 1
        
        # 创建Future对象
        future = Future()
        with self.results_lock:
            self.results[request_id] = future
        
        try:
            # 尝试将请求添加到队列
            self.request_queue.put(request, timeout=timeout)
            logger.info(f"Request {request_id} added to queue")
            
            # 等待结果
            result = future.result(timeout=timeout)
            return result
            
        except queue.Full:
            # 队列已满
            self.queue_full_count += 1
            with self.results_lock:
                if request_id in self.results:
                    del self.results[request_id]
            return format_response(
                success=False,
                error=f"Queue is full (max size: {self.max_queue_size}). Please try again later.",
                data={"request_id": request_id}
            )
        except Exception as e:
            # 其他错误
            with self.results_lock:
                if request_id in self.results:
                    del self.results[request_id]
            self.timeout_count += 1
            return format_response(
                success=False,
                error=f"Request failed: {str(e)}",
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
        
        for engine in self.engines:
            status = engine.get_status()
            engine_statuses.append(status)
        
        # 队列状态
        queue_size = self.request_queue.qsize()
        queue_status = {
            "size": queue_size,
            "max_size": self.max_queue_size
        }
        
        return {
            "uptime": time.time() - self.start_time,
            "num_workers": len(self.engines),
            "total_workers": self.num_workers,
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
    
    def shutdown(self):
        """关闭服务"""
        logger.info("Shutting down TTS Engine Manager...")
        self.running = False
        
        # 等待所有worker线程结束
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        logger.info("TTS Engine Manager shutdown complete") 