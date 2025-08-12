#!/usr/bin/env python3
"""
测试音频保存功能
"""

import requests
import json
import time

def test_audio_save():
    """测试音频保存功能"""
    
    # 测试不同的音频参数
    test_cases = [
        {
            "name": "默认参数",
            "params": {
                "text": "Hello, this is a test with default parameters.",
                "frame_size": 2048,
                "speaker": "default"
            }
        },
        {
            "name": "16000Hz 16bit 20ms",
            "params": {
                "text": "Hello, this is a test with 16000Hz 16bit 20ms parameters.",
                "sample_rate": 16000,
                "bit_depth": 16,
                "frame_duration_ms": 20,
                "speaker": "default"
            }
        },
        {
            "name": "22050Hz 32bit 默认帧",
            "params": {
                "text": "Hello, this is a test with 22050Hz 32bit parameters.",
                "sample_rate": 22050,
                "bit_depth": 32,
                "frame_size": 2048,
                "speaker": "default"
            }
        }
    ]
    
    base_url = "http://localhost:8421"
    
    for test_case in test_cases:
        print(f"\n=== 测试: {test_case['name']} ===")
        
        try:
            # 发送WebSocket请求（这里用HTTP请求模拟，实际应该用WebSocket）
            # 由于WebSocket需要特殊处理，这里只是打印参数
            print(f"请求参数: {json.dumps(test_case['params'], indent=2)}")
            
            # 检查debug_audio目录
            import os
            if os.path.exists("debug_audio"):
                files = os.listdir("debug_audio")
                print(f"debug_audio目录中的文件: {files}")
            else:
                print("debug_audio目录不存在")
                
        except Exception as e:
            print(f"测试失败: {e}")
    
    print("\n=== 测试完成 ===")
    print("请检查debug_audio目录中的音频文件，确认音频质量是否正确。")

if __name__ == "__main__":
    test_audio_save() 