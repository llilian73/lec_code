"""
加载 energy_hourly1年.py 中使用的输入 parquet 数据，并输出内存占用大小
"""

import pandas as pd
import os
import sys

# 配置路径（与 energy_hourly1年.py 保持一致）
BASE_PATH = "/home/linbor/WORK/lishiying"
HEAT_WAVE_BASE_PATH = os.path.join(BASE_PATH, "heat_wave")

# 模型配置
MODELS = ["BCC-CSM2-MR"]

# SSP配置
SSP_PATHS = ["SSP126"]

# 年份配置
TARGET_YEARS = [2030, 2031, 2032, 2033, 2034]

# Case配置
CASES = ['ref'] + [f'case{i}' for i in range(1, 21)]


def get_memory_size_mb(df):
    """获取DataFrame的内存占用（MB）"""
    return df.memory_usage(deep=True).sum() / (1024 * 1024)


def load_and_check_memory(model_name, ssp_path, year, case_name):
    """加载单个parquet文件并返回内存占用
    
    Returns:
        tuple: (success, memory_mb, row_count) 或 (False, 0, 0) 如果文件不存在
    """
    parquet_path = os.path.join(
        HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", str(year),
        "point", case_name, f"{case_name}_hourly.parquet"
    )
    
    if not os.path.exists(parquet_path):
        return False, 0, 0
    
    try:
        # 读取必要的列（与 energy_hourly1年.py 保持一致）
        columns_needed = ['lat', 'lon', 'cooling_demand', 'date', 'time']
        
        try:
            # 尝试只读取必要列
            df = pd.read_parquet(
                parquet_path,
                columns=columns_needed,
                engine='pyarrow'
            )
        except Exception:
            # 如果指定列失败，读取全部列
            df = pd.read_parquet(parquet_path, engine='pyarrow')
            # 只保留需要的列
            needed_cols = ['lat', 'lon', 'cooling_demand']
            if 'datetime' in df.columns:
                needed_cols.append('datetime')
            elif 'date' in df.columns and 'time' in df.columns:
                needed_cols.extend(['date', 'time'])
            df = df[needed_cols]
        
        memory_mb = get_memory_size_mb(df)
        row_count = len(df)
        
        return True, memory_mb, row_count
    
    except Exception as e:
        print(f"  错误: 加载 {parquet_path} 时出错: {e}")
        return False, 0, 0


def main():
    """主函数：遍历所有配置的文件并统计内存占用"""
    print("=== 开始统计 parquet 文件内存占用 ===\n")
    
    total_memory_mb = 0
    total_files = 0
    successful_files = 0
    failed_files = 0
    
    # 按模型、SSP、年份、工况遍历
    for model_name in MODELS:
        for ssp_path in SSP_PATHS:
            for year in TARGET_YEARS:
                print(f"\n处理 {model_name} - {ssp_path} - {year} 年:")
                print("-" * 60)
                
                year_total_memory = 0
                year_files = 0
                
                for case_name in CASES:
                    total_files += 1
                    success, memory_mb, row_count = load_and_check_memory(
                        model_name, ssp_path, year, case_name
                    )
                    
                    if success:
                        successful_files += 1
                        year_files += 1
                        year_total_memory += memory_mb
                        total_memory_mb += memory_mb
                        print(f"  {case_name:10s}: {memory_mb:10.2f} MB ({row_count:>12,} 行)")
                    else:
                        failed_files += 1
                        print(f"  {case_name:10s}: 文件不存在或加载失败")
                
                print(f"\n  {year} 年小计: {year_files} 个文件, 总内存: {year_total_memory:.2f} MB")
    
    # 输出总结
    print("\n" + "=" * 60)
    print("=== 内存占用统计总结 ===")
    print(f"总文件数: {total_files}")
    print(f"成功加载: {successful_files}")
    print(f"失败/不存在: {failed_files}")
    print(f"总内存占用: {total_memory_mb:.2f} MB ({total_memory_mb / 1024:.2f} GB)")
    print("=" * 60)


if __name__ == "__main__":
    main()

