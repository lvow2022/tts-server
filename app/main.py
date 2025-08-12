import asyncio
import time
import logging
import base64
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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

def split_audio_to_frames(audio: np.ndarray, frame_size: int = 2048, sample_rate: int = 22050):
    """将完整音频分割成帧并模拟流式发送"""
    frames = []
    frame_duration_ms = (frame_size / sample_rate) * 1000  # 每帧时长(ms)
    
    for i in range(0, len(audio), frame_size):
        frame = audio[i:i + frame_size]
        
        # 跳过空帧
        if len(frame) == 0:
            continue
            
        frames.append({
            "frame_id": len(frames) + 1,
            "data": frame,
            "timestamp_ms": len(frames) * frame_duration_ms,
            "is_last": i + frame_size >= len(audio)
        })
    
    return frames

def split_audio_bytes_to_frames(audio_bytes: bytes, frame_size: int = 2048, sample_rate: int = 22050):
    """将音频bytes数据分割成帧"""
    frames = []
    frame_duration_ms = (frame_size / sample_rate) * 1000  # 每帧时长(ms)
    bytes_per_frame = frame_size * 4  # 4字节/float32
    
    for i in range(0, len(audio_bytes), bytes_per_frame):
        frame_bytes = audio_bytes[i:i + bytes_per_frame]
        
        # 跳过空帧
        if len(frame_bytes) == 0:
            continue
            
        frames.append({
            "frame_id": len(frames) + 1,
            "data": frame_bytes,
            "timestamp_ms": len(frames) * frame_duration_ms,
            "is_last": i + bytes_per_frame >= len(audio_bytes)
        })
    
    return frames

async def synthesize_audio_async(text: str, speaker: str = "default", timeout: float = 30.0):
    """异步执行TTS合成"""
    loop = asyncio.get_event_loop()
    
    result = await loop.run_in_executor(
        executor,
        tts_manager.synthesize,
        text,
        speaker,
        timeout
    )
    
    return result

# 请求模型
class SynthesisRequest(BaseModel):
    text: str
    speaker: str = "default"
    timeout: float = 30.0  # 请求超时时间

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
    global executor, tts_manager
    logger.info("Shutting down TTS Server...")
    
    # 关闭TTS引擎管理器
    if tts_manager:
        tts_manager.shutdown()
    
    # 关闭线程池
    if executor:
        executor.shutdown(wait=True)

@app.get("/")
async def root():
    """返回前端页面"""
    return FileResponse("static/index.html")

@app.post("/synthesize", response_model=SynthesisResponse)
async def synthesize(request: SynthesisRequest):
    """语音合成接口 - 支持智能分配和排队"""
    try:
        loop = asyncio.get_event_loop()
        
        # 使用新的智能分配策略
        result = await loop.run_in_executor(
            executor,
            tts_manager.synthesize,
            request.text,
            request.speaker,
            request.timeout
        )
        
        # 添加时间戳
        result["timestamp"] = time.time()
        
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
    """健康检查接口 - 包含详细状态信息"""
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
        
        # 检查服务健康状态
        is_healthy = (
            status["num_workers"] > 0 and 
            status["queue"]["size"] < status["queue"]["max_size"]
        )
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "uptime": time.time() - start_time,
            "workers": {
                "total": status["total_workers"],
                "available": status["num_workers"],
                "busy": 0  # 在生产者-消费者模式下，所有worker都是可用的
            },
            "queue": status["queue"],
            "statistics": status["statistics"],
            "memory_usage": status["memory_usage"],
            "cpu_usage": status["cpu_usage"],
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

@app.get("/status")
async def get_detailed_status():
    """获取详细状态信息"""
    try:
        if tts_manager is None:
            return JSONResponse(
                status_code=503,
                content={"error": "TTS manager not initialized", "timestamp": time.time()}
            )
        
        status = tts_manager.get_status()
        status["timestamp"] = time.time()
        
        return status
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "timestamp": time.time()}
        )

@app.get("/engines")
async def get_engine_status():
    """获取所有引擎的详细状态"""
    try:
        if tts_manager is None:
            return JSONResponse(
                status_code=503,
                content={"error": "TTS manager not initialized", "timestamp": time.time()}
            )
        
        status = tts_manager.get_status()
        
        return {
            "engines": status["engine_statuses"],
            "summary": {
                "total": status["total_workers"],
                "available": status["available_engines"],
                "busy": status["busy_engines"]
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Engine status check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "timestamp": time.time()}
        )

@app.websocket("/ws/synthesize")
async def websocket_synthesize(websocket: WebSocket):
    """WebSocket流式语音合成接口"""
    await websocket.accept()
    
    try:
        # 1. 接收请求
        data = await websocket.receive_json()
        text = data.get("text")
        frame_size = data.get("frame_size", 2048)
        speaker = data.get("speaker", "default")
        
        if not text:
            await websocket.send_json({
                "type": "error",
                "error": "Text is required"
            })
            return
        
        # 2. 发送开始消息
        await websocket.send_json({
            "type": "start",
            "text": text,
            "frame_size": frame_size,
            "speaker": speaker
        })
        
        # 3. 执行TTS合成（等待完整音频）
        result = await synthesize_audio_async(text, speaker)
        
        if result["success"]:
            # 获取音频数据 - 直接使用PCM数据
            audio_bytes = result["data"]["audio_pcm"]
            
            # 4. 发送合成完成消息
            total_samples = len(audio_bytes) // 4  # 4字节/float32
            await websocket.send_json({
                "type": "synthesized",
                "audio_length": total_samples,
                "duration_ms": total_samples / 22050 * 1000
            })
            
            # 5. 分帧并快速发送
            audio_frames = split_audio_bytes_to_frames(audio_bytes, frame_size)
            
            logger.info(f"音频总长度: {total_samples} 采样点, 分帧数: {len(audio_frames)}")
            
            for frame in audio_frames:
                logger.debug(f"发送帧 {frame['frame_id']}: {len(frame['data'])} 字节")
                
                # 发送帧信息
                frame_info = {
                    "type": "audio_frame",
                    "frame_id": frame["frame_id"],
                    "data_length": len(frame["data"]),
                    "timestamp_ms": frame["timestamp_ms"],
                    "is_last": frame["is_last"]
                }
                await websocket.send_json(frame_info)
                
                # 发送PCM数据
                await websocket.send_bytes(frame["data"])
                
                # 模拟实时发送间隔（可选）
                await asyncio.sleep(0.01)  # 10ms间隔
            
            # 6. 发送完成消息
            await websocket.send_json({
                "type": "complete",
                "total_frames": len(audio_frames),
                "total_duration_ms": len(audio_array) / 22050 * 1000
            })
        else:
            await websocket.send_json({
                "type": "error",
                "error": result["error"]
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket synthesis error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    ) 