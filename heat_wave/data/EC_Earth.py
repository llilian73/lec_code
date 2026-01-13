"""
EC-Earth3历史数据整合工具

功能：
将EC-Earth3模型的一年一个nc文件（1981-2010年，共30个文件）整合成一个nc文件。

输入：
- 路径：/home/linbor/WORK/lishiying/GCM_input_processed/EC-Earth3/historical
- 文件格式：tasmax_day_EC-Earth3-HR_historical_r1i1p1f1_gr_19810101-19811231_interpolated_1deg.nc

输出：
- 文件：tasmax_day_EC-Earth3-HR_historical_r1i1p1f1_gr_19810101-20101231_interpolated_1deg.nc
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import gc
import logging
from datetime import datetime
import time
import cftime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置路径
BASE_PATH = "/home/linbor/WORK/lishiying"
INPUT_DIR = os.path.join(BASE_PATH, "GCM_input_processed/EC-Earth3/historical")
OUTPUT_DIR = INPUT_DIR  # 输出到同一目录

# 年份范围
START_YEAR = 1981
END_YEAR = 2010

# 并行进程数
NUM_PROCESSES = 30


def find_historical_files(input_dir):
    """查找所有历史数据文件"""
    all_files = [
        f for f in os.listdir(input_dir) 
        if f.endswith(".nc") and "tasmax" in f and "_interpolated_1deg" in f
    ]
    
    historical_files = []
    for year in range(START_YEAR, END_YEAR + 1):
        # 查找包含该年份的文件
        pattern = f"{year}0101-{year}1231"
        found = False
        for f in all_files:
            if pattern in f:
                historical_files.append(os.path.join(input_dir, f))
                found = True
                break
        if not found:
            logger.warning(f"未找到 {year} 年的历史数据文件")
    
    return sorted(historical_files)  # 按文件名排序，确保时间顺序


def load_single_file(file_path):
    """加载单个文件（用于并行处理）"""
    try:
        # EC-Earth3一年一个文件，文件较小，不需要chunks
        # 使用decode_times=False保持原始时间编码，避免转换为datetime64
        ds = xr.open_dataset(file_path, decode_times=False)
        # 加载数据到内存
        ds_loaded = ds.load()
        ds.close()
        return ds_loaded
    except Exception as e:
        logger.error(f"加载文件 {os.path.basename(file_path)} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def load_files_parallel(file_paths, num_processes):
    """并行加载多个文件"""
    logger.info(f"使用 {num_processes} 个进程并行加载 {len(file_paths)} 个文件...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        datasets = list(tqdm(
            pool.imap(load_single_file, file_paths),
            total=len(file_paths),
            desc="加载文件"
        ))
    
    # 过滤掉None（加载失败的文件）
    valid_datasets = [ds for ds in datasets if ds is not None]
    
    if len(valid_datasets) != len(file_paths):
        logger.warning(f"成功加载 {len(valid_datasets)}/{len(file_paths)} 个文件")
    
    return valid_datasets


def merge_datasets(datasets):
    """合并多个数据集"""
    logger.info(f"开始合并 {len(datasets)} 个数据集...")
    
    if len(datasets) == 0:
        raise ValueError("没有有效的数据集可以合并")
    
    # 按时间排序
    logger.info("按时间排序数据集...")
    # 使用cftime对象进行比较，避免转换为datetime64
    def get_first_time(ds):
        try:
            return ds.time.values[0]
        except:
            # 如果无法直接获取，尝试其他方法
            return pd.Timestamp(ds.time.values[0])
    
    datasets_sorted = sorted(datasets, key=get_first_time)
    
    # 合并数据集
    logger.info("合并数据集...")
    combined_ds = xr.concat(datasets_sorted, dim='time')
    combined_ds = combined_ds.sortby('time')
    
    # 检查是否有重复的时间点
    logger.info("检查重复时间点...")
    time_values = combined_ds.time.values
    unique_times, unique_indices = np.unique(time_values, return_index=True)
    
    if len(unique_times) < len(time_values):
        logger.warning(f"发现 {len(time_values) - len(unique_times)} 个重复时间点，将移除重复项")
        combined_ds = combined_ds.isel(time=unique_indices)
    
    # 确保时间坐标使用cftime编码（如果原始数据使用cftime）
    # 检查第一个数据集的时间编码
    if len(datasets) > 0:
        first_ds = datasets[0]
        if 'time' in first_ds.coords:
            # 如果原始数据的时间坐标有编码信息，保留它
            if hasattr(first_ds.time, 'encoding') and first_ds.time.encoding:
                combined_ds.time.encoding = first_ds.time.encoding.copy()
    
    return combined_ds


def verify_merged_data(combined_ds):
    """验证合并后的数据"""
    logger.info("验证合并后的数据...")
    
    # 处理cftime时间对象
    try:
        time_index = combined_ds.time.to_pandas()
    except Exception:
        ds_decoded = xr.decode_cf(combined_ds, decode_times=True)
        time_index = pd.to_datetime(ds_decoded.time.values)
    
    if not isinstance(time_index, pd.DatetimeIndex):
        time_index = pd.DatetimeIndex(time_index)
    
    # 验证年份
    actual_years = sorted(time_index.year.unique())
    expected_years = list(range(START_YEAR, END_YEAR + 1))
    missing_years = set(expected_years) - set(actual_years)
    
    if missing_years:
        logger.error(f"❌ 合并后的数据缺少年份: {sorted(missing_years)}")
        logger.error(f"   实际包含年份: {actual_years}")
        return False
    
    logger.info(f"✓ 数据验证通过，包含完整年份: {actual_years}")
    logger.info(f"   时间范围: {time_index[0]} 到 {time_index[-1]}")
    logger.info(f"   总时间点数: {len(time_index)}")
    
    return True


def save_merged_data(combined_ds, output_path):
    """保存合并后的数据（参考filter_year.py的方法）"""
    logger.info(f"保存合并后的数据到: {output_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 编码设置（参考filter_year.py的方法）
    encoding = {}
    for var in combined_ds.data_vars:
        encoding[var] = {
            'zlib': True,
            'complevel': 4,
            'shuffle': True
        }
    
    # 对坐标变量也进行压缩（除了time）
    for coord in combined_ds.coords:
        if coord != 'time':  # time坐标通常不压缩
            encoding[coord] = {
                'zlib': True,
                'complevel': 4
            }
    
    # 保存文件（参考filter_year.py的方法，直接保存，xarray会自动处理时间编码）
    # 由于使用了decode_times=False加载，时间坐标已经是数值类型，可以直接保存
    start_time = time.time()
    combined_ds.to_netcdf(
        output_path,
        encoding=encoding,
        format='NETCDF4'
    )
    elapsed_time = time.time() - start_time
    
    logger.info(f"✓ 文件保存完成，耗时 {elapsed_time:.2f} 秒")
    logger.info(f"   文件大小: {os.path.getsize(output_path) / 1024**3:.2f} GB")


def main():
    """主函数"""
    try:
        logger.info("="*80)
        logger.info("EC-Earth3历史数据整合工具")
        logger.info("="*80)
        
        # 检查输入目录
        if not os.path.exists(INPUT_DIR):
            raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR}")
        
        # 查找所有历史数据文件
        logger.info(f"查找历史数据文件: {INPUT_DIR}")
        historical_files = find_historical_files(INPUT_DIR)
        
        if len(historical_files) == 0:
            raise ValueError("未找到任何历史数据文件")
        
        logger.info(f"找到 {len(historical_files)} 个历史数据文件")
        logger.info(f"年份范围: {START_YEAR}-{END_YEAR}")
        
        # 检查输出文件是否已存在
        output_filename = f"tasmax_day_EC-Earth3-HR_historical_r1i1p1f1_gr_{START_YEAR}0101-{END_YEAR}1231_interpolated_1deg.nc"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        if os.path.exists(output_path):
            logger.warning(f"输出文件已存在: {output_path}，将自动覆盖")
            os.remove(output_path)
        
        # 并行加载文件
        start_time = time.time()
        datasets = load_files_parallel(historical_files, NUM_PROCESSES)
        load_time = time.time() - start_time
        logger.info(f"文件加载完成，耗时 {load_time:.2f} 秒")
        
        if len(datasets) == 0:
            raise ValueError("没有成功加载任何文件")
        
        # 合并数据集
        merge_start_time = time.time()
        combined_ds = merge_datasets(datasets)
        merge_time = time.time() - merge_start_time
        logger.info(f"数据集合并完成，耗时 {merge_time:.2f} 秒")
        
        # 验证数据
        if not verify_merged_data(combined_ds):
            raise ValueError("数据验证失败")
        
        # 保存合并后的数据
        save_merged_data(combined_ds, output_path)
        
        # 清理内存
        del datasets, combined_ds
        gc.collect()
        
        logger.info("="*80)
        logger.info("✓ 所有操作完成")
        logger.info("="*80)
        logger.info(f"输出文件: {output_path}")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        gc.collect()


if __name__ == "__main__":
    main()

