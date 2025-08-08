#!/usr/bin/env python3
"""
TTS Server 启动脚本
"""

import uvicorn
from app.config import settings

if __name__ == "__main__":
    print("🚀 启动 TTS Server...")
    print(f"📍 服务地址: http://{settings.HOST}:{settings.PORT}")
    print(f"🔧 Worker数量: {settings.WORKERS}")
    print(f"🎯 设备: {settings.DEVICE}")
    print("=" * 50)
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    ) 