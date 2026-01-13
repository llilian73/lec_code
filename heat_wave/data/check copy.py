"""
检查插值后的NetCDF文件格式

功能：
检查插值后的NetCDF文件的经纬度范围和空间分辨率
"""

import xarray as xr
import numpy as np
import pandas as pd
import os

# 要检查的文件路径
FILE_PATH = r"Z:\local_environment_creation\heat_wave\GCM_input_filter\tasmax_day_BCC-CSM2-MR_ssp126_r1i1p1f1_gn_20300101-20341231_interpolated_1deg.nc"


def check_file_format(file_path):
    """检查NetCDF文件的格式信息"""
    print("="*80)
    print("NetCDF文件格式检查")
    print("="*80)
    print(f"文件路径: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    # 获取文件大小
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"文件大小: {file_size_mb:.2f} MB")
    
    try:
        # 打开NetCDF文件
        ds = xr.open_dataset(file_path)
        
        print(f"\n数据变量: {list(ds.data_vars.keys())}")
        print(f"坐标变量: {list(ds.coords.keys())}")
        print(f"维度: {dict(ds.sizes)}")
        
        # 获取经纬度变量名
        lat_var = None
        lon_var = None
        
        for name in ['lat', 'latitude', 'Lat', 'Latitude']:
            if name in ds.coords or name in ds.sizes:
                lat_var = name
                break
        
        for name in ['lon', 'longitude', 'Lon', 'Longitude']:
            if name in ds.coords or name in ds.sizes:
                lon_var = name
                break
        
        if lat_var is None or lon_var is None:
            print("❌ 无法找到经纬度坐标")
            print(f"可用的坐标: {list(ds.coords.keys())}")
            print(f"可用的维度: {list(ds.sizes.keys())}")
            ds.close()
            return
        
        # 获取经纬度数据
        lats = ds[lat_var].values
        lons = ds[lon_var].values
        
        # 计算经纬度范围
        lat_min = float(np.min(lats))
        lat_max = float(np.max(lats))
        lon_min = float(np.min(lons))
        lon_max = float(np.max(lons))
        
        # 计算网格数量
        lat_count = len(lats) if len(lats.shape) == 1 else lats.shape[0]
        lon_count = len(lons) if len(lons.shape) == 1 else lons.shape[-1]
        
        # 计算空间分辨率
        if len(lats.shape) == 1 and len(lats) > 1:
            lat_res = float(np.abs(np.diff(lats)).mean())
        else:
            lat_res = "N/A (2D grid)"
        
        if len(lons.shape) == 1 and len(lons) > 1:
            lon_res = float(np.abs(np.diff(lons)).mean())
        else:
            lon_res = "N/A (2D grid)"
        
        # 输出结果
        print("\n" + "="*80)
        print("空间信息")
        print("="*80)
        print(f"纬度变量: {lat_var}")
        print(f"  范围: {lat_min:.4f}° 到 {lat_max:.4f}°")
        print(f"  网格数: {lat_count}")
        print(f"  分辨率: {lat_res:.4f}°" if isinstance(lat_res, (int, float)) else f"  分辨率: {lat_res}")
        
        print(f"\n经度变量: {lon_var}")
        print(f"  范围: {lon_min:.4f}° 到 {lon_max:.4f}°")
        print(f"  网格数: {lon_count}")
        print(f"  分辨率: {lon_res:.4f}°" if isinstance(lon_res, (int, float)) else f"  分辨率: {lon_res}")
        
        print(f"\n空间分辨率: {lat_res}° × {lon_res}°" if isinstance(lat_res, (int, float)) and isinstance(lon_res, (int, float)) else f"空间分辨率: {lat_res} × {lon_res}")
        print(f"总网格点数: {lat_count} × {lon_count} = {lat_count * lon_count}")
        
        # 获取时间信息（如果有）
        time_var = None
        for name in ['time', 'Time', 'TIME']:
            if name in ds.coords or name in ds.sizes:
                time_var = name
                break
        
        if time_var:
            times = ds[time_var].values
            if len(times) > 0:
                print(f"\n时间信息:")
                print(f"  时间变量: {time_var}")
                print(f"  时间点数: {len(times)}")
                print(f"  起始时间(原始): {times[0]}")
                print(f"  结束时间(原始): {times[-1]}")
                
                # 提取年份信息和详细时间戳
                time_index = None
                try:
                    # 尝试转换为pandas DatetimeIndex
                    try:
                        time_index = ds[time_var].to_pandas()
                    except Exception:
                        # 如果失败，使用decode_cf
                        ds_decoded = xr.decode_cf(ds, decode_times=True)
                        time_index = pd.to_datetime(ds_decoded[time_var].values)
                    
                    # 确保是DatetimeIndex
                    if not isinstance(time_index, pd.DatetimeIndex):
                        time_index = pd.DatetimeIndex(time_index)
                    
                    # 打印前10个时间戳的详细信息
                    print(f"\n  前10个时间戳详细信息:")
                    print(f"  {'序号':<6} {'日期时间':<25} {'年份':<8} {'月份':<8} {'日期':<8} {'小时':<8} {'分钟':<8} {'秒':<8}")
                    print(f"  {'-'*6} {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
                    for i in range(min(10, len(time_index))):
                        ts = time_index[i]
                        print(f"  {i+1:<6} {str(ts):<25} {ts.year:<8} {ts.month:<8} {ts.day:<8} {ts.hour:<8} {ts.minute:<8} {ts.second:<8}")
                    
                    # 统计小时分布
                    hours = time_index.hour.unique()
                    print(f"\n  时间戳中的小时值: {sorted(hours)}")
                    if len(hours) == 1:
                        print(f"  ✓ 所有时间戳都是 {hours[0]} 点")
                    else:
                        print(f"  ⚠️  时间戳包含多个小时值")
                    
                    # 提取唯一年份
                    years = sorted(time_index.year.unique())
                    print(f"\n  包含的年份: {years}")
                    print(f"  年份数量: {len(years)} 年")
                    print(f"  年份范围: {min(years)} 年到 {max(years)} 年")
                    
                except Exception as e:
                    # 如果转换失败，尝试手动提取年份和时间
                    try:
                        import cftime
                        if len(times) > 0 and isinstance(times[0], cftime.datetime):
                            # 打印前10个时间戳
                            print(f"\n  前10个时间戳详细信息 (cftime格式):")
                            print(f"  {'序号':<6} {'日期时间':<30} {'年份':<8} {'月份':<8} {'日期':<8} {'小时':<8} {'分钟':<8} {'秒':<8}")
                            print(f"  {'-'*6} {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
                            for i in range(min(10, len(times))):
                                t = times[i]
                                print(f"  {i+1:<6} {str(t):<30} {t.year:<8} {t.month:<8} {t.day:<8} {t.hour:<8} {t.minute:<8} {t.second:<8}")
                            
                            # 统计小时分布
                            hours = sorted(set([t.hour for t in times]))
                            print(f"\n  时间戳中的小时值: {hours}")
                            if len(hours) == 1:
                                print(f"  ✓ 所有时间戳都是 {hours[0]} 点")
                            else:
                                print(f"  ⚠️  时间戳包含多个小时值")
                            
                            years = sorted(set([t.year for t in times]))
                            print(f"\n  包含的年份: {years}")
                            print(f"  年份数量: {len(years)} 年")
                            print(f"  年份范围: {min(years)} 年到 {max(years)} 年")
                        else:
                            print(f"\n  ⚠️  无法提取时间详细信息: {e}")
                            print(f"  时间类型: {type(times[0])}")
                    except Exception as e2:
                        print(f"\n  ⚠️  无法提取时间信息: {e}, {e2}")
                        print(f"  时间类型: {type(times[0]) if len(times) > 0 else 'N/A'}")
        
        ds.close()
        
        print("\n" + "="*80)
        print("检查完成")
        print("="*80)
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_file_format(FILE_PATH)

