#!/usr/bin/env python3
"""
TTS服务压测脚本
用于验证高并发场景下的性能和稳定性
"""

import asyncio
import aiohttp
import time
import json
import statistics
from typing import List, Dict, Any
import argparse

class TTSStressTest:
    def __init__(self, base_url: str = "http://localhost:8421"):
        self.base_url = base_url
        self.results = []
        
    async def single_request(self, session: aiohttp.ClientSession, text: str, request_id: int) -> Dict[str, Any]:
        """发送单个请求"""
        start_time = time.time()
        
        try:
            payload = {
                "text": text,
                "speaker": "default"
            }
            
            async with session.post(
                f"{self.base_url}/synthesize",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # 转换为毫秒
                
                if response.status == 200:
                    result_data = await response.json()
                    return {
                        "request_id": request_id,
                        "status": "success",
                        "response_time": response_time,
                        "http_status": response.status,
                        "success": result_data.get("success", False),
                        "error": result_data.get("error") if not result_data.get("success") else None
                    }
                else:
                    return {
                        "request_id": request_id,
                        "status": "http_error",
                        "response_time": response_time,
                        "http_status": response.status,
                        "success": False,
                        "error": f"HTTP {response.status}"
                    }
                    
        except asyncio.TimeoutError:
            return {
                "request_id": request_id,
                "status": "timeout",
                "response_time": 30000,  # 30秒超时
                "success": False,
                "error": "Request timeout"
            }
        except Exception as e:
            return {
                "request_id": request_id,
                "status": "exception",
                "response_time": (time.time() - start_time) * 1000,
                "success": False,
                "error": str(e)
            }
    

    
    async def run_stress_test(self, 
                            concurrent_requests: int = 50,
                            total_requests: int = 200,
                            test_texts: List[str] = None) -> Dict[str, Any]:
        """运行压测"""
        
        print(f"开始压测: {concurrent_requests} 并发, {total_requests} 总请求")
        
        # 准备测试文本 - 支持中英文
        if test_texts is None:
            test_texts = [
                # 英文测试文本
                "Hello, this is a test.",
                "The weather is nice today.",
                "Artificial intelligence technology is developing rapidly.",
                "Text to speech technology is becoming more mature.",
                "This is a high concurrency stress test.",
                # 中文测试文本
                "你好，这是一个测试。",
                "今天天气很好。",
                "人工智能技术发展迅速。",
                "语音合成技术越来越成熟。",
                "这是一个高并发压测。"
            ]
        
        # 创建会话
        connector = aiohttp.TCPConnector(limit=concurrent_requests * 2)
        timeout = aiohttp.ClientTimeout(total=30)
        
        # 创建信号量控制并发数
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            
            start_time = time.time()
            tasks = []
            
            async def controlled_request(text: str, request_id: int):
                """受控的请求函数"""
                async with semaphore:  # 使用信号量控制并发
                    return await self.single_request(session, text, request_id)
            
            # 创建请求任务
            for i in range(total_requests):
                text = test_texts[i % len(test_texts)]
                task = asyncio.create_task(
                    controlled_request(text, i + 1)
                )
                tasks.append(task)
            
            # 执行所有请求
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 分析结果
            return self.analyze_results(results, total_time, concurrent_requests)
    
    def analyze_results(self, results: List[Dict], total_time: float, concurrent_requests: int) -> Dict[str, Any]:
        """分析压测结果"""
        
        # 过滤有效结果
        valid_results = [r for r in results if isinstance(r, dict)]
        
        # 统计成功/失败
        successful_requests = [r for r in valid_results if r.get("success")]
        failed_requests = [r for r in valid_results if not r.get("success")]
        
        # 响应时间统计
        response_times = [r.get("response_time", 0) for r in valid_results]
        
        # 计算统计指标
        total_requests = len(valid_results)
        success_count = len(successful_requests)
        failure_count = len(failed_requests)
        success_rate = (success_count / total_requests * 100) if total_requests > 0 else 0
        
        # 响应时间统计
        if response_times:
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max_response_time
        else:
            avg_response_time = median_response_time = min_response_time = max_response_time = p95_response_time = 0
        
        # QPS计算
        qps = total_requests / total_time if total_time > 0 else 0
        
        return {
            "summary": {
                "total_requests": total_requests,
                "successful_requests": success_count,
                "failed_requests": failure_count,
                "success_rate": f"{success_rate:.2f}%",
                "total_time": f"{total_time:.2f}s",
                "qps": f"{qps:.2f}",
                "concurrent_requests": concurrent_requests
            },
            "response_times": {
                "average_ms": f"{avg_response_time:.2f}",
                "median_ms": f"{median_response_time:.2f}",
                "min_ms": f"{min_response_time:.2f}",
                "max_ms": f"{max_response_time:.2f}",
                "p95_ms": f"{p95_response_time:.2f}"
            },
            "errors": [r.get("error") for r in failed_requests if r.get("error")],
            "raw_results": valid_results
        }
    
    def print_results(self, results: Dict[str, Any]):
        """打印压测结果"""
        print("\n" + "="*50)
        print("压测结果")
        print("="*50)
        
        if "error" in results:
            print(f"❌ 压测失败: {results['error']}")
            return
        
        summary = results["summary"]
        response_times = results["response_times"]
        
        print(f"📊 总体统计:")
        print(f"   总请求数: {summary['total_requests']}")
        print(f"   成功请求: {summary['successful_requests']}")
        print(f"   失败请求: {summary['failed_requests']}")
        print(f"   成功率: {summary['success_rate']}")
        print(f"   总耗时: {summary['total_time']}")
        print(f"   QPS: {summary['qps']}")
        print(f"   并发数: {summary['concurrent_requests']}")
        print(f"   测试语言: {args.lang if 'args' in locals() else 'both'}")
        
        print(f"\n⏱️  响应时间:")
        print(f"   平均: {response_times['average_ms']}ms")
        print(f"   中位数: {response_times['median_ms']}ms")
        print(f"   最小值: {response_times['min_ms']}ms")
        print(f"   最大值: {response_times['max_ms']}ms")
        print(f"   95分位: {response_times['p95_ms']}ms")
        
        if results["errors"]:
            print(f"\n❌ 错误信息:")
            error_counts = {}
            for error in results["errors"]:
                error_counts[error] = error_counts.get(error, 0) + 1
            
            for error, count in error_counts.items():
                print(f"   {error}: {count}次")

async def main():
    parser = argparse.ArgumentParser(description="TTS服务压测工具")
    parser.add_argument("--url", default="http://localhost:8421", help="TTS服务地址")
    parser.add_argument("--concurrent", type=int, default=50, help="并发请求数")
    parser.add_argument("--total", type=int, default=200, help="总请求数")
    parser.add_argument("--text", help="测试文本，用逗号分隔多个文本")
    parser.add_argument("--lang", choices=["en", "zh", "both"], default="both", help="测试语言: en(英文), zh(中文), both(中英文)")
    
    args = parser.parse_args()
    
    # 准备测试文本
    test_texts = None
    if args.text:
        test_texts = [text.strip() for text in args.text.split(",")]
    elif args.lang != "both":
        # 根据语言选择预设文本
        if args.lang == "en":
            test_texts = [
                "Hello, this is a test.",
                "The weather is nice today.",
                "Artificial intelligence technology is developing rapidly.",
                "Text to speech technology is becoming more mature.",
                "This is a high concurrency stress test."
            ]
        elif args.lang == "zh":
            test_texts = [
                "你好，这是一个测试。",
                "今天天气很好。",
                "人工智能技术发展迅速。",
                "语音合成技术越来越成熟。",
                "这是一个高并发压测。"
            ]
    
    # 创建压测实例
    stress_test = TTSStressTest(args.url)
    
    # 运行压测
    results = await stress_test.run_stress_test(
        concurrent_requests=args.concurrent,
        total_requests=args.total,
        test_texts=test_texts
    )
    
    # 打印结果
    stress_test.print_results(results)

if __name__ == "__main__":
    asyncio.run(main()) 