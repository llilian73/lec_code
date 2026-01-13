#!/usr/bin/env python3
"""
性能测试脚本
用于测试优化后的2_c_DD_pop.py的性能提升
"""

import time
import psutil
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def monitor_system_resources():
    """监控系统资源使用情况"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'memory_available_gb': memory.available / (1024**3),
        'memory_used_gb': memory.used / (1024**3)
    }

def test_batch_processing():
    """测试批量处理性能"""
    print("=== 性能测试开始 ===")
    
    # 记录开始时间
    start_time = time.time()
    start_resources = monitor_system_resources()
    
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"初始CPU使用率: {start_resources['cpu_percent']:.1f}%")
    print(f"初始内存使用率: {start_resources['memory_percent']:.1f}%")
    print(f"可用内存: {start_resources['memory_available_gb']:.1f} GB")
    
    try:
        # 导入并运行优化后的主函数
        from global_energy_demand.grid import main as optimized_main
        optimized_main()
        
    except ImportError:
        print("无法导入优化后的模块，请检查路径设置")
        return False
    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        return False
    
    # 记录结束时间
    end_time = time.time()
    end_resources = monitor_system_resources()
    
    # 计算性能指标
    total_time = end_time - start_time
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    
    print("\n=== 性能测试结果 ===")
    print(f"总运行时间: {hours:.0f}小时 {minutes:.0f}分钟 {seconds:.0f}秒")
    print(f"最终CPU使用率: {end_resources['cpu_percent']:.1f}%")
    print(f"最终内存使用率: {end_resources['memory_percent']:.1f}%")
    print(f"最终可用内存: {end_resources['memory_available_gb']:.1f} GB")
    
    # 性能评估
    avg_cpu = (start_resources['cpu_percent'] + end_resources['cpu_percent']) / 2
    avg_memory = (start_resources['memory_percent'] + end_resources['memory_percent']) / 2
    
    print(f"\n=== 性能评估 ===")
    print(f"平均CPU使用率: {avg_cpu:.1f}%")
    print(f"平均内存使用率: {avg_memory:.1f}%")
    
    if avg_cpu > 70:
        print("✓ CPU利用率良好")
    elif avg_cpu > 50:
        print("⚠ CPU利用率中等")
    else:
        print("✗ CPU利用率偏低")
    
    if avg_memory < 80:
        print("✓ 内存使用正常")
    else:
        print("⚠ 内存使用较高")
    
    return True

def compare_with_original():
    """与原始版本对比（需要原始版本存在）"""
    print("\n=== 性能对比分析 ===")
    print("原始版本预期运行时间: 18小时")
    print("优化版本目标运行时间: 8小时")
    print("预期加速比: 2.25倍")
    
    # 这里可以添加实际的对比逻辑
    # 需要同时运行两个版本并比较结果

if __name__ == "__main__":
    print("2_c_DD_pop.py 性能测试")
    print("=" * 50)
    
    # 检查系统要求
    memory = psutil.virtual_memory()
    if memory.total < 8 * 1024**3:  # 8GB
        print("⚠ 警告: 系统内存少于8GB，可能影响性能")
    
    cpu_count = psutil.cpu_count()
    print(f"检测到 {cpu_count} 个CPU核心")
    
    # 运行性能测试
    success = test_batch_processing()
    
    if success:
        compare_with_original()
        print("\n✓ 性能测试完成")
    else:
        print("\n✗ 性能测试失败")
