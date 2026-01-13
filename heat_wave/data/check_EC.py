"""
检查EC-Earth3模型future_56_60文件夹下所有NC文件包含的年份

功能：
遍历指定目录下的所有文件夹，检查每个.nc文件包含的数据年份，并打印结果。
"""

import os
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 目标目录
BASE_DIR = r"Z:\local_environment_creation\heat_wave\GCM_input_filter\EC-Earth3\future_56_60"


def convert_cftime_to_datetime(time_values):
    """将cftime对象转换为pandas DatetimeIndex"""
    try:
        # 方法1：尝试使用to_pandas或pd.to_datetime
        if hasattr(time_values, 'to_pandas'):
            time_index = time_values.to_pandas()
        else:
            time_index = pd.to_datetime(time_values)
        
        if not isinstance(time_index, pd.DatetimeIndex):
            time_index = pd.DatetimeIndex(time_index)
        return time_index
    except Exception as e:
        # 方法2：如果方法1失败，手动处理cftime对象
        try:
            import cftime
            if len(time_values) > 0 and isinstance(time_values[0], cftime.datetime):
                # 手动转换cftime对象为pandas Timestamp
                time_index = pd.to_datetime([pd.Timestamp(t.year, t.month, t.day) for t in time_values])
            else:
                time_index = pd.to_datetime(time_values)
            
            if not isinstance(time_index, pd.DatetimeIndex):
                time_index = pd.DatetimeIndex(time_index)
            return time_index
        except Exception as e2:
            # 方法3：最后尝试使用to_pandas（如果time_values是DataArray）
            try:
                if hasattr(time_values, 'to_pandas'):
                    time_index = time_values.to_pandas()
                    if not isinstance(time_index, pd.DatetimeIndex):
                        time_index = pd.DatetimeIndex(time_index)
                    return time_index
                else:
                    raise ValueError(f"无法转换时间索引: {e}, {e2}")
            except Exception:
                raise ValueError(f"无法转换时间索引: {e}, {e2}")


def get_file_years(nc_file_path):
    """获取NC文件包含的年份"""
    try:
        with xr.open_dataset(nc_file_path) as ds:
            # 处理时间坐标
            try:
                # 方法1：尝试使用decode_cf
                ds_decoded = xr.decode_cf(ds, decode_times=True)
                time_values = ds_decoded.time.values
                
                if hasattr(time_values, 'to_pandas'):
                    time_index = time_values.to_pandas()
                else:
                    time_index = pd.to_datetime(time_values)
                
                if not isinstance(time_index, pd.DatetimeIndex):
                    time_index = pd.DatetimeIndex(time_index)
                    
            except Exception as e:
                # 方法2：如果decode_cf失败，手动处理cftime对象
                try:
                    import cftime
                    time_values = ds.time.values
                    
                    if len(time_values) > 0 and isinstance(time_values[0], cftime.datetime):
                        time_index = pd.to_datetime([pd.Timestamp(t.year, t.month, t.day) for t in time_values])
                    else:
                        time_index = pd.to_datetime(time_values)
                    
                    if not isinstance(time_index, pd.DatetimeIndex):
                        time_index = pd.DatetimeIndex(time_index)
                        
                except Exception as e2:
                    # 方法3：最后尝试使用to_pandas
                    try:
                        time_index = ds.time.to_pandas()
                        if not isinstance(time_index, pd.DatetimeIndex):
                            time_index = pd.DatetimeIndex(time_index)
                    except Exception:
                        raise ValueError(f"无法转换时间索引: {e}, {e2}")
            
            # 提取年份
            years = sorted(time_index.year.unique())
            start_date = time_index.min()
            end_date = time_index.max()
            
            return {
                'years': years,
                'year_range': f"{min(years)}-{max(years)}",
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'total_days': len(time_index)
            }
            
    except Exception as e:
        logger.error(f"读取文件失败 {nc_file_path}: {e}")
        return None


def check_all_files(base_dir):
    """检查目录下所有NC文件的年份"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        logger.error(f"目录不存在: {base_dir}")
        return
    
    logger.info(f"开始检查目录: {base_dir}")
    logger.info("=" * 80)
    
    # 统计信息
    total_files = 0
    total_folders = 0
    files_by_folder = {}
    
    # 遍历所有子文件夹
    for folder_path in sorted(base_path.iterdir()):
        if not folder_path.is_dir():
            continue
        
        total_folders += 1
        folder_name = folder_path.name
        logger.info(f"\n文件夹: {folder_name}")
        logger.info("-" * 80)
        
        # 查找该文件夹下的所有.nc文件
        nc_files = list(folder_path.glob("*.nc"))
        
        if len(nc_files) == 0:
            logger.info(f"  未找到.nc文件")
            continue
        
        files_by_folder[folder_name] = []
        
        # 检查每个文件
        for nc_file in sorted(nc_files):
            total_files += 1
            file_name = nc_file.name
            relative_path = nc_file.relative_to(base_path)
            
            logger.info(f"\n  文件: {file_name}")
            logger.info(f"  路径: {relative_path}")
            
            # 获取年份信息
            year_info = get_file_years(nc_file)
            
            if year_info:
                years = year_info['years']
                year_range = year_info['year_range']
                start_date = year_info['start_date']
                end_date = year_info['end_date']
                total_days = year_info['total_days']
                
                logger.info(f"  包含年份: {year_range} ({len(years)}个年份)")
                logger.info(f"  年份列表: {years}")
                logger.info(f"  日期范围: {start_date} 至 {end_date}")
                logger.info(f"  总天数: {total_days}")
                
                files_by_folder[folder_name].append({
                    'file': file_name,
                    'years': years,
                    'year_range': year_range,
                    'start_date': start_date,
                    'end_date': end_date,
                    'total_days': total_days
                })
            else:
                logger.warning(f"  无法读取年份信息")
                files_by_folder[folder_name].append({
                    'file': file_name,
                    'years': None,
                    'error': True
                })
    
    # 打印汇总信息
    logger.info("\n" + "=" * 80)
    logger.info("汇总信息")
    logger.info("=" * 80)
    logger.info(f"总文件夹数: {total_folders}")
    logger.info(f"总文件数: {total_files}")
    logger.info(f"\n各文件夹文件统计:")
    
    for folder_name, files in files_by_folder.items():
        logger.info(f"\n  {folder_name}: {len(files)} 个文件")
        for file_info in files:
            if file_info.get('error'):
                logger.info(f"    - {file_info['file']}: 读取失败")
            else:
                logger.info(f"    - {file_info['file']}: {file_info['year_range']} ({file_info['total_days']} 天)")


def main():
    """主函数"""
    try:
        check_all_files(BASE_DIR)
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

