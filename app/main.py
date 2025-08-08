import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import settings
from .utils import setup_logging
from .tts_engine import TTSEngineManager

# 设置日志
logger = setup_logging(settings.LOG_FILE, settings.LOG_LEVEL)

# 创建FastAPI应用
app = FastAPI(title="TTS Server", version="1.0.0")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 全局变量
tts_manager = None
executor = None
start_time = None

# 请求模型
class SynthesisRequest(BaseModel):
    text: str
    speaker: str = "default"

class SynthesisResponse(BaseModel):
    success: bool
    data: Dict[str, Any] = None
    error: str = None
    timestamp: float

@app.on_event("startup")
async def startup_event():
    """服务启动事件"""
    global tts_manager, executor, start_time
    
    start_time = time.time()
    logger.info("Starting TTS Server...")
    
    try:
        # 初始化TTS引擎管理器
        tts_manager = TTSEngineManager(settings.WORKERS)
        executor = ThreadPoolExecutor(max_workers=settings.WORKERS)
        logger.info(f"TTS Server started successfully with {settings.WORKERS} workers")
        
    except Exception as e:
        logger.error(f"Failed to start TTS Server: {e}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭事件"""
    global executor
    logger.info("Shutting down TTS Server...")
    if executor:
        executor.shutdown(wait=True)

@app.get("/")
async def root():
    """返回前端页面"""
    return FileResponse("static/index.html")

@app.post("/synthesize", response_model=SynthesisResponse)
async def synthesize(request: SynthesisRequest):
    """语音合成接口"""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            tts_manager.synthesize,
            request.text,
            request.speaker
        )
        return SynthesisResponse(**result)
        
    except Exception as e:
        logger.error(f"Synthesis request failed: {e}")
        return SynthesisResponse(
            success=False,
            error=f"Internal server error: {str(e)}",
            timestamp=time.time()
        )

@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        if tts_manager is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": "TTS manager not initialized", "timestamp": time.time()}
            )
        
        status = tts_manager.get_status()
        
        if status["num_workers"] == 0:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": "No TTS workers available", "timestamp": time.time()}
            )
        
        return {
            "status": "healthy",
            "uptime": time.time() - start_time,
            "workers": status["num_workers"],
            "total_workers": status["total_workers"],
            "memory_usage_mb": status["memory_usage"]["rss_mb"],
            "device": status["device"],
            "gpu_info": status["gpu_info"],
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e), "timestamp": time.time()}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    ) 