"""
气候数据提取工具

功能概述：
本工具用于从NetCDF格式的气候数据文件中提取指定坐标点的日最高温度数据，并转换为CSV格式保存。

输入数据：
1. 历史数据文件：
   - 文件路径：Z:\local_environment_creation\heat_wave\GCM_input_filter\tasmax_day_BCC-CSM2-MR_historical_r1i1p1f1_gn_19810101-20101231_interpolated_1deg.nc
   - 变量：tasmax（日最高温度）
   - 时间范围：1981-2010年

2. 未来数据文件：
   - 文件路径：Z:\local_environment_creation\heat_wave\GCM_input_filter\tasmax_day_BCC-CSM2-MR_ssp126_r1i1p1f1_gn_20300101-20341231_interpolated_1deg.nc
   - 变量：tasmax（日最高温度）
   - 时间范围：2030-2034年

主要功能：
1. 数据提取：
   - 从NetCDF文件中提取指定坐标点（lat=35.0, lon=102.0）的数据
   - 处理历史数据和未来数据

2. 数据格式转换：
   - 从NetCDF格式转换为CSV格式
   - 文件名格式：point_lat{lat}_lon{lon}.csv

输出结果：
1. 网格点CSV文件：
   - 目录：Z:\local_environment_creation\heat_wave\GCM_input_filter\
   - 文件名格式：point_lat{lat}_lon{lon}.csv
   - 包含列：time（时间）、tasmax（日最高温度）

输出格式：
- 文件格式：CSV（UTF-8编码）
- 时间格式：NetCDF时间格式
- 温度单位：原始单位（通常是K或°C）
- 坐标精度：三位小数
- 文件命名：point_lat{lat}_lon{lon}.csv
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import gc

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据文件路径
historical_file = r"Z:\local_environment_creation\heat_wave\GCM_input_filter\tasmax_day_BCC-CSM2-MR_historical_r1i1p1f1_gn_19810101-20101231_interpolated_1deg.nc"
future_file = r"Z:\local_environment_creation\heat_wave\GCM_input_filter\tasmax_day_BCC-CSM2-MR_ssp126_r1i1p1f1_gn_20300101-20341231_interpolated_1deg.nc"
output_dir = r"Z:\local_environment_creation\heat_wave\GCM_input_filter"

# 目标坐标点
target_lat = 35.0
target_lon = 102.0


def find_nearest_coord(array, value):
    """找到数组中与目标值最接近的索引"""
    idx = np.abs(array - value).argmin()
    return idx, array[idx]


def extract_point_data(nc_file, lat, lon):
    """
    从NetCDF文件中提取指定坐标点的数据
    
    参数:
        nc_file: NetCDF文件路径
        lat: 目标纬度
        lon: 目标经度
    
    返回:
        DataFrame包含time和tasmax列
    """
    try:
        logger.info(f"正在读取文件: {nc_file}")
        ds = xr.open_dataset(nc_file)
        
        try:
            # 找到最接近的经纬度索引
            lat_idx, nearest_lat = find_nearest_coord(ds.lat.values, lat)
            lon_idx, nearest_lon = find_nearest_coord(ds.lon.values, lon)
            
            logger.info(f"目标坐标: ({lat}, {lon}), 实际使用坐标: ({nearest_lat}, {nearest_lon})")
            
            # 提取该点的所有时间序列数据
            tasmax_data = ds.tasmax.isel(lat=lat_idx, lon=lon_idx)
            
            # 转换为DataFrame
            df = tasmax_data.to_dataframe().reset_index()
            
            # 只保留time和tasmax列
            df = df[['time', 'tasmax']]
            
            logger.info(f"成功提取 {len(df)} 条数据记录")
            return df
            
        finally:
            ds.close()
            gc.collect()
            
    except Exception as e:
        logger.error(f"读取文件 {nc_file} 时出错: {str(e)}")
        raise


def check_paths(historical_file, future_file, output_dir):
    """检查所有路径是否存在"""
    if not os.path.exists(historical_file):
        raise FileNotFoundError(f"历史数据文件不存在: {historical_file}")
    
    if not os.path.exists(future_file):
        raise FileNotFoundError(f"未来数据文件不存在: {future_file}")

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")


def process_data():
    """处理历史数据和未来数据"""
    logger.info("开始处理数据...")
    
    # 处理历史数据
    logger.info("处理历史数据（1981-2010年）...")
    hist_df = extract_point_data(historical_file, target_lat, target_lon)
    
    # 处理未来数据
    logger.info("处理未来数据（2030-2034年）...")
    future_df = extract_point_data(future_file, target_lat, target_lon)
    
    # 合并数据
    logger.info("合并历史数据和未来数据...")
    combined_df = pd.concat([hist_df, future_df], ignore_index=True)
    
    # 生成文件名并保存
    filename = f"point_lat{target_lat:.3f}_lon{target_lon:.3f}.csv"
    output_path = os.path.join(output_dir, filename)
    
    # 保存为CSV文件
    combined_df.to_csv(output_path, index=False)
    logger.info(f"数据已保存至: {output_path}")
    logger.info(f"总计 {len(combined_df)} 条数据记录")
    
    return output_path


def main():
    try:
        print("\n=== 开始处理 ===")
        # 检查路径
        check_paths(historical_file, future_file, output_dir)

        # 处理数据
        output_path = process_data()

        print(f"\n=== 处理完成 ===")
        print(f"输出文件: {output_path}")

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise
    finally:
        # 清理内存
        gc.collect()


if __name__ == "__main__":
    main()
