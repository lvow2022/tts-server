#!/usr/bin/env python3
"""
测试CPU、内存和GPU统计数据是否正确
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils import get_memory_usage, get_cpu_usage, get_gpu_info
import time

def test_memory_stats():
    """测试内存统计数据"""
    print("🔍 测试内存统计数据...")
    
    memory_info = get_memory_usage()
    print(f"进程RSS内存: {memory_info['rss_mb']:.2f} MB")
    print(f"进程VMS内存: {memory_info['vms_mb']:.2f} MB")
    print(f"进程内存使用率: {memory_info['percent']:.2f}%")
    print(f"系统总内存: {memory_info['system_total_mb']:.2f} MB")
    print(f"系统已用内存: {memory_info['system_used_mb']:.2f} MB")
    print(f"系统可用内存: {memory_info['system_available_mb']:.2f} MB")
    print(f"系统内存使用率: {memory_info['system_percent']:.2f}%")
    
    # 验证数据合理性
    assert memory_info['rss_mb'] > 0, "RSS内存应该大于0"
    assert memory_info['vms_mb'] > 0, "VMS内存应该大于0"
    assert 0 <= memory_info['percent'] <= 100, "进程内存使用率应该在0-100%之间"
    assert memory_info['system_total_mb'] > 0, "系统总内存应该大于0"
    assert 0 <= memory_info['system_percent'] <= 100, "系统内存使用率应该在0-100%之间"
    
    print("✅ 内存统计数据验证通过")

def test_cpu_stats():
    """测试CPU统计数据"""
    print("\n🔍 测试CPU统计数据...")
    
    cpu_info = get_cpu_usage()
    print(f"进程CPU使用率: {cpu_info['process_percent']:.2f}%")
    print(f"系统CPU使用率: {cpu_info['system_percent']:.2f}%")
    print(f"CPU核心数: {cpu_info['cpu_count']}")
    if cpu_info['cpu_freq']:
        print(f"CPU频率: {cpu_info['cpu_freq']:.1f} MHz")
    else:
        print("CPU频率: 不可用")
    
    # 验证数据合理性
    assert 0 <= cpu_info['process_percent'] <= 100, "进程CPU使用率应该在0-100%之间"
    assert 0 <= cpu_info['system_percent'] <= 100, "系统CPU使用率应该在0-100%之间"
    assert cpu_info['cpu_count'] > 0, "CPU核心数应该大于0"
    
    print("✅ CPU统计数据验证通过")

def test_gpu_stats():
    """测试GPU统计数据"""
    print("\n🔍 测试GPU统计数据...")
    
    gpu_info = get_gpu_info()
    print(f"GPU可用: {gpu_info['available']}")
    
    if gpu_info['available']:
        print(f"设备类型: {gpu_info['device_type']}")
        print(f"设备数量: {gpu_info['device_count']}")
        print(f"GPU名称: {gpu_info['name']}")
        
        if gpu_info['memory_total']:
            print(f"总显存: {gpu_info['memory_total']:.2f} MB")
            print(f"已用显存: {gpu_info['memory_used']:.2f} MB")
            print(f"可用显存: {gpu_info['memory_free']:.2f} MB")
            print(f"保留显存: {gpu_info['memory_reserved']:.2f} MB")
            
            # 验证显存数据合理性
            assert gpu_info['memory_total'] > 0, "总显存应该大于0"
            assert gpu_info['memory_used'] >= 0, "已用显存应该大于等于0"
            assert gpu_info['memory_free'] >= 0, "可用显存应该大于等于0"
            assert gpu_info['memory_reserved'] >= 0, "保留显存应该大于等于0"
            assert gpu_info['memory_used'] + gpu_info['memory_free'] <= gpu_info['memory_total'], "已用+可用显存不应该超过总显存"
        
        if gpu_info['temperature']:
            print(f"GPU温度: {gpu_info['temperature']}°C")
            assert 0 <= gpu_info['temperature'] <= 150, "GPU温度应该在合理范围内"
        
        if gpu_info['utilization']:
            print(f"GPU利用率: {gpu_info['utilization']}%")
            assert 0 <= gpu_info['utilization'] <= 100, "GPU利用率应该在0-100%之间"
        
        # 检查设备列表
        if gpu_info['devices']:
            print(f"设备列表:")
            for device in gpu_info['devices']:
                print(f"  - 设备 {device['id']}: {device['name']}")
                if 'memory_allocated_mb' in device:
                    print(f"    显存分配: {device['memory_allocated_mb']:.2f} MB")
                    print(f"    显存保留: {device['memory_reserved_mb']:.2f} MB")
                    print(f"    总显存: {device['memory_total_mb']:.2f} MB")
                    print(f"    使用率: {device['memory_usage_percent']:.2f}%")
    else:
        print(f"GPU不可用: {gpu_info.get('message', '未知原因')}")
    
    print("✅ GPU统计数据验证通过")

def test_continuous_monitoring():
    """测试连续监控"""
    print("\n🔍 测试连续监控（5秒）...")
    
    start_time = time.time()
    while time.time() - start_time < 5:
        memory_info = get_memory_usage()
        cpu_info = get_cpu_usage()
        gpu_info = get_gpu_info()
        
        print(f"\r⏱️  监控中... 内存: {memory_info['rss_mb']:.1f}MB, CPU: {cpu_info['process_percent']:.1f}%", end="")
        
        time.sleep(1)
    
    print("\n✅ 连续监控测试完成")

if __name__ == "__main__":
    print("🚀 开始测试CPU、内存和GPU统计数据...")
    print("=" * 50)
    
    try:
        test_memory_stats()
        test_cpu_stats()
        test_gpu_stats()
        test_continuous_monitoring()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！统计数据正确。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 