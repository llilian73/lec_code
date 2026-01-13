"""
GCM数据线性插值工具

功能：
将不同空间分辨率的GCM数据通过线性插值统一到1°×1°分辨率
"""

import xarray as xr
import numpy as np
import os
import pandas as pd

# 输入目录路径
INPUT_DIR = r"Z:\local_environment_creation\heat_wave\GCM_input_filter\BCC-CSM2-MR"

# 目标分辨率
TARGET_RESOLUTION = 1.0  # 1度


def print_spatial_resolution(ds, title="数据空间分辨率信息"):
    """打印数据的空间分辨率信息"""
    print("\n" + "="*80)
    print(title)
    print("="*80)
    
    # 获取经纬度坐标
    lat_var = None
    lon_var = None
    
    # 常见的经纬度变量名
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
        return None, None
    
    lats = ds[lat_var].values
    lons = ds[lon_var].values
    
    # 计算分辨率
    if len(lats.shape) == 1:
        lat_res = np.abs(np.diff(lats)).mean() if len(lats) > 1 else 0
        lat_min = float(lats.min())
        lat_max = float(lats.max())
        lat_count = len(lats)
    else:
        # 2D网格
        lat_res = "N/A (2D grid)"
        lat_min = float(lats.min())
        lat_max = float(lats.max())
        lat_count = lats.shape[0] if len(lats.shape) > 0 else 0
    
    if len(lons.shape) == 1:
        lon_res = np.abs(np.diff(lons)).mean() if len(lons) > 1 else 0
        lon_min = float(lons.min())
        lon_max = float(lons.max())
        lon_count = len(lons)
    else:
        lon_res = "N/A (2D grid)"
        lon_min = float(lons.min())
        lon_max = float(lons.max())
        lon_count = lons.shape[1] if len(lons.shape) > 1 else 0
    
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
    
    return lat_var, lon_var


def create_target_grid(resolution=1.0, lon_range_180=True):
    """创建目标网格（1°×1°）
    
    固定范围：
    - 纬度：-90° 到 90°（步长1°，共181个点）
    - 经度：如果lon_range_180=True，则为-180° 到 180°（步长1°，共361个点）
            如果lon_range_180=False，则为0° 到 360°（步长1°，共361个点）
    """
    # 创建固定的目标经纬度数组
    # 纬度：-90, -89, -88, ..., 88, 89, 90
    target_lats = np.arange(-90, 90 + resolution, resolution)
    
    # 经度：根据lon_range_180参数决定
    if lon_range_180:
        # 经度：-180, -179, -178, ..., 178, 179, 180
        target_lons = np.arange(-180, 180 + resolution, resolution)
    else:
        # 经度：0, 1, 2, ..., 358, 359, 360
        target_lons = np.arange(0, 360 + resolution, resolution)
    
    return target_lats, target_lons


def check_lon_range(ds, lon_var):
    """检查经度范围（所有模型应该都是0~360）"""
    lons = ds[lon_var].values
    
    if len(lons.shape) == 1:
        lon_min = float(lons.min())
        lon_max = float(lons.max())
    else:
        lon_min = float(lons.min())
        lon_max = float(lons.max())
    
    print(f"  经度范围: {lon_min:.4f}° 到 {lon_max:.4f}°")
    
    # 验证经度范围（应该都是0~360）
    if lon_min < 0:
        print(f"  ⚠️  警告: 检测到经度范围包含负值，可能需要转换")
    elif lon_min >= 0 and lon_max <= 360:
        print(f"  ✓ 经度范围符合0~360标准")
    else:
        print(f"  ⚠️  警告: 经度范围超出0~360")
    
    return lon_min, lon_max


def interpolate_to_1degree(ds, lat_var, lon_var, lon_range_180=True):
    """将数据插值到1°×1°分辨率（固定范围：纬度-90~90，经度-180~180或0~360）"""
    print("正在进行线性插值到 1°×1° 分辨率...")
    
    # 创建固定的目标网格（根据lon_range_180参数决定经度范围）
    target_lats, target_lons = create_target_grid(TARGET_RESOLUTION, lon_range_180=lon_range_180)
    
    # 使用xarray的interp方法进行插值
    try:
        # 创建目标坐标的DataArray
        target_lat_da = xr.DataArray(target_lats, dims=[lat_var], coords={lat_var: target_lats})
        target_lon_da = xr.DataArray(target_lons, dims=[lon_var], coords={lon_var: target_lons})
        
        # 执行插值
        interpolated_ds = ds.interp(
            {lat_var: target_lat_da, lon_var: target_lon_da},
            method='linear',
            kwargs={'fill_value': np.nan}  # 超出范围的值设为NaN
        )
        
        print("✓ 插值完成")
        
        return interpolated_ds
        
    except Exception as e:
        print(f"❌ 插值失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_lon_comparison(ds_original, ds_converted, lon_var, lat_var, target_lon_360=358.0):
    """打印358°和-2°的数据对比"""
    print("\n" + "="*80)
    print(f"经度转换验证：{target_lon_360}° vs {target_lon_360-360}°")
    print("="*80)
    
    try:
        # 获取数据变量（排除坐标变量和边界变量）
        data_vars = [
            var for var in ds_original.data_vars 
            if var not in [lat_var, lon_var, 'time'] 
            and not var.endswith('_bnds')
            and not var.startswith('lat_bnds')
            and not var.startswith('lon_bnds')
            and not var.startswith('time_bnds')
        ]
        
        if len(data_vars) == 0:
            print("❌ 未找到数据变量")
            return
        
        var_name = data_vars[0]
        time_var = 'time' if 'time' in ds_original.sizes else None
        
        if time_var is None:
            print("❌ 未找到时间维度")
            return
        
        # 获取第一个时间点
        first_time = ds_original[time_var].values[0]
        
        # 在原始数据中找到358°附近的点
        lons_original = ds_original[lon_var].values
        idx_358 = np.argmin(np.abs(lons_original - target_lon_360))
        lon_358_actual = float(lons_original[idx_358])
        
        # 获取中间纬度（避免边界效应）
        lats = ds_original[lat_var].values
        lat_idx = len(lats) // 2
        lat_center = float(lats[lat_idx])
        
        print(f"\n原始数据（0~360坐标系）:")
        print(f"  经度: {lon_358_actual:.4f}° (索引 {idx_358})")
        print(f"  纬度: {lat_center:.4f}° (索引 {lat_idx})")
        print(f"  时间: {first_time}")
        
        # 提取原始数据中358°附近的数据
        if time_var in ds_original[var_name].dims:
            original_data = ds_original[var_name].isel({time_var: 0, lon_var: idx_358, lat_var: lat_idx})
        else:
            original_data = ds_original[var_name].isel({lon_var: idx_358, lat_var: lat_idx})
        
        original_value = float(original_data.values)
        print(f"  数据值: {original_value:.6f}")
        
        # 在转换后的数据中找到-2°附近的点
        lons_converted = ds_converted[lon_var].values
        target_lon_180 = target_lon_360 - 360  # -2°
        idx_minus2 = np.argmin(np.abs(lons_converted - target_lon_180))
        lon_minus2_actual = float(lons_converted[idx_minus2])
        
        print(f"\n转换后数据（-180~180坐标系）:")
        print(f"  经度: {lon_minus2_actual:.4f}° (索引 {idx_minus2}, 对应原始 {target_lon_360}°)")
        print(f"  纬度: {lat_center:.4f}° (索引 {lat_idx})")
        print(f"  时间: {first_time}")
        
        # 提取转换后数据中-2°附近的数据
        if time_var in ds_converted[var_name].dims:
            converted_data = ds_converted[var_name].isel({time_var: 0, lon_var: idx_minus2, lat_var: lat_idx})
        else:
            converted_data = ds_converted[var_name].isel({lon_var: idx_minus2, lat_var: lat_idx})
        
        converted_value = float(converted_data.values)
        print(f"  数据值: {converted_value:.6f}")
        
        # 比较数据是否一致
        print(f"\n数据一致性检查:")
        if np.isclose(original_value, converted_value, rtol=1e-10):
            print(f"  ✓ 数据一致！原始值 {original_value:.6f} = 转换值 {converted_value:.6f}")
        else:
            diff = abs(original_value - converted_value)
            print(f"  ⚠️  数据不一致！差值: {diff:.10f}")
            print(f"     原始值: {original_value:.10f}")
            print(f"     转换值: {converted_value:.10f}")
        
    except Exception as e:
        print(f"❌ 打印对比数据失败: {e}")
        import traceback
        traceback.print_exc()


def convert_lon_to_180_180(ds, lon_var, lat_var=None):
    """将经度从0~360转换为-180~180（只做转换，不做插值）"""
    print("正在将经度从0~360转换为-180~180...")
    
    try:
        # 获取当前经度值
        lons = ds[lon_var].values
        
        # 检查经度范围
        lon_min = float(lons.min())
        lon_max = float(lons.max())
        
        # 如果经度已经是-180~180范围，不需要转换
        if lon_min >= -180 and lon_max <= 180:
            print("  经度范围已经是-180~180，无需转换")
            return ds
        
        # 方法：将>180的经度减去360，重新排序（不插值）
        if len(lons.shape) == 1:
            # 找到180度对应的索引（用于roll）
            idx_180 = np.argmin(np.abs(lons - 180))
            
            # 使用roll将数据重新排列：[0~180]在前，[180~360]在后 -> [180~360]在前，[0~180]在后
            ds_rolled = ds.roll({lon_var: -idx_180}, roll_coords=True)
            
            # 更新经度坐标：将>180的经度减去360
            lons_rolled = ds_rolled[lon_var].values
            lons_converted = lons_rolled.copy()
            lons_converted[lons_converted > 180] = lons_converted[lons_converted > 180] - 360
            
            # 重新分配坐标并按经度排序（不插值，保持原始数据）
            ds_new = ds_rolled.assign_coords({lon_var: lons_converted}).sortby(lon_var)
        else:
            # 如果是2D网格，直接转换坐标（假设经度是1D的）
            lons_1d = lons.flatten()
            lons_converted = lons_1d.copy()
            lons_converted[lons_converted > 180] = lons_converted[lons_converted > 180] - 360
            ds_new = ds.assign_coords({lon_var: lons_converted}).sortby(lon_var)
        
        print("✓ 经度转换完成")
        
        return ds_new
        
    except Exception as e:
        print(f"❌ 经度转换失败: {e}")
        import traceback
        traceback.print_exc()
        return ds


def print_sample_data(ds, lat_var, lon_var, n_samples=5):
    """打印数据样本（前几行）"""
    print("\n" + "="*80)
    print(f"插值后的数据样本（前{n_samples}个时间点）")
    print("="*80)
    
    # 获取数据变量（排除坐标变量和边界变量）
    # 边界变量通常以 '_bnds' 结尾，应该被排除
    data_vars = [
        var for var in ds.data_vars 
        if var not in [lat_var, lon_var, 'time'] 
        and not var.endswith('_bnds')
        and not var.startswith('lat_bnds')
        and not var.startswith('lon_bnds')
        and not var.startswith('time_bnds')
    ]
    
    if len(data_vars) == 0:
        print("❌ 未找到数据变量（可能都是边界变量）")
        print(f"所有变量: {list(ds.data_vars.keys())}")
        return
    
    # 获取时间维度
    time_var = 'time' if 'time' in ds.sizes else None
    if time_var is None:
        print("❌ 未找到时间维度")
        return
    
    # 获取前几个时间点
    n_times = min(n_samples, len(ds[time_var]))
    
    print(f"\n数据变量: {data_vars}")
    print(f"时间维度: {len(ds[time_var])} 个时间点")
    print(f"空间维度: {len(ds[lat_var])} × {len(ds[lon_var])}")
    
    # 选择第一个数据变量和第一个时间点进行展示
    var_name = data_vars[0]
    var_data = ds[var_name]
    
    # 检查数据变量的实际维度
    print(f"\n变量 '{var_name}' 的维度: {var_data.dims}")
    print(f"变量 '{var_name}' 的坐标: {list(var_data.coords.keys())}")
    
    # 确定数据变量使用的维度名称（可能是lat_var/lon_var，也可能是其他名称）
    # 检查数据变量的维度中是否包含lat_var和lon_var
    var_dims = list(var_data.dims)
    
    # 找到对应的维度名称
    data_lat_dim = None
    data_lon_dim = None
    
    # 首先尝试使用lat_var和lon_var
    if lat_var in var_dims:
        data_lat_dim = lat_var
    else:
        # 尝试查找包含'lat'的维度
        for dim in var_dims:
            if 'lat' in dim.lower():
                data_lat_dim = dim
                break
    
    if lon_var in var_dims:
        data_lon_dim = lon_var
    else:
        # 尝试查找包含'lon'的维度
        for dim in var_dims:
            if 'lon' in dim.lower():
                data_lon_dim = dim
                break
    
    if data_lat_dim is None or data_lon_dim is None:
        print(f"❌ 无法确定数据变量的空间维度")
        print(f"  数据变量的维度: {var_dims}")
        print(f"  坐标变量的名称: lat={lat_var}, lon={lon_var}")
        # 仍然尝试显示统计信息（只对数值数据）
        try:
            first_time_data = var_data.isel({time_var: 0}) if time_var in var_dims else var_data
            # 确保获取的是数值，而不是时间对象
            min_val = first_time_data.min().values
            max_val = first_time_data.max().values
            mean_val = first_time_data.mean().values
            std_val = first_time_data.std().values
            
            # 如果是数组，取第一个元素或使用item()
            if isinstance(min_val, np.ndarray):
                min_val = min_val.item() if min_val.size == 1 else float(min_val.flat[0])
            if isinstance(max_val, np.ndarray):
                max_val = max_val.item() if max_val.size == 1 else float(max_val.flat[0])
            if isinstance(mean_val, np.ndarray):
                mean_val = mean_val.item() if mean_val.size == 1 else float(mean_val.flat[0])
            if isinstance(std_val, np.ndarray):
                std_val = std_val.item() if std_val.size == 1 else float(std_val.flat[0])
            
            # 检查是否为数值类型
            if isinstance(min_val, (int, float, np.number)):
                print(f"\n数据统计信息（所有数据）:")
                print(f"  最小值: {float(min_val):.6f}")
                print(f"  最大值: {float(max_val):.6f}")
                print(f"  平均值: {float(mean_val):.6f}")
                print(f"  标准差: {float(std_val):.6f}")
            else:
                print(f"\n⚠️  无法显示统计信息：数据类型不是数值（可能是时间或其他类型）")
                print(f"  最小值类型: {type(min_val)}, 值: {min_val}")
        except Exception as e:
            print(f"\n⚠️  无法计算统计信息: {e}")
            import traceback
            traceback.print_exc()
        return
    
    print(f"\n变量 '{var_name}' 的数据样本:")
    print(f"时间点: {ds[time_var].values[:n_times]}")
    
    # 显示第一个时间点的数据（选择部分经纬度点）
    if time_var in var_dims:
        first_time_data = var_data.isel({time_var: 0})
    else:
        first_time_data = var_data
    
    # 选择中间的经纬度点进行展示（避免边界效应）
    lat_size = len(ds[data_lat_dim]) if data_lat_dim in ds.sizes else len(ds[lat_var])
    lon_size = len(ds[data_lon_dim]) if data_lon_dim in ds.sizes else len(ds[lon_var])
    
    lat_idx_start = lat_size // 4
    lat_idx_end = lat_idx_start + 3
    lon_idx_start = lon_size // 4
    lon_idx_end = lon_idx_start + 3
    
    print(f"\n第一个时间点的数据（纬度索引 {lat_idx_start}-{lat_idx_end}, 经度索引 {lon_idx_start}-{lon_idx_end}）:")
    try:
        sample_data = first_time_data.isel({
            data_lat_dim: slice(lat_idx_start, lat_idx_end),
            data_lon_dim: slice(lon_idx_start, lon_idx_end)
        })
        print(sample_data)
    except Exception as e:
        print(f"❌ 无法选择数据样本: {e}")
        print(f"  尝试的维度: {data_lat_dim}, {data_lon_dim}")
        print(f"  数据变量的实际维度: {var_dims}")
    
    # 显示统计信息（确保获取的是数值）
    print(f"\n数据统计信息:")
    try:
        # 使用.item()确保获取标量值，如果失败则尝试转换为float
        min_val = first_time_data.min().values
        max_val = first_time_data.max().values
        mean_val = first_time_data.mean().values
        std_val = first_time_data.std().values
        
        # 如果是数组，取第一个元素或使用item()
        if isinstance(min_val, np.ndarray):
            min_val = min_val.item() if min_val.size == 1 else float(min_val.flat[0])
        if isinstance(max_val, np.ndarray):
            max_val = max_val.item() if max_val.size == 1 else float(max_val.flat[0])
        if isinstance(mean_val, np.ndarray):
            mean_val = mean_val.item() if mean_val.size == 1 else float(mean_val.flat[0])
        if isinstance(std_val, np.ndarray):
            std_val = std_val.item() if std_val.size == 1 else float(std_val.flat[0])
        
        # 确保是数值类型
        if isinstance(min_val, (int, float, np.number)):
            print(f"  最小值: {float(min_val):.6f}")
            print(f"  最大值: {float(max_val):.6f}")
            print(f"  平均值: {float(mean_val):.6f}")
            print(f"  标准差: {float(std_val):.6f}")
        else:
            print(f"  ⚠️  数据类型不是数值（可能是时间或其他类型）")
            print(f"  最小值类型: {type(min_val)}, 值: {min_val}")
    except Exception as e:
        print(f"  ⚠️  无法计算统计信息: {e}")
        import traceback
        traceback.print_exc()


def save_interpolated_data(interpolated_ds, input_file_path):
    """保存插值后的数据到NetCDF文件（保存到原始文件路径）"""
    try:
        # 从输入文件路径提取文件名和目录
        input_dir = os.path.dirname(input_file_path)
        input_filename = os.path.basename(input_file_path)
        
        # 输出目录就是输入目录（保存到原始文件路径）
        output_dir = input_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名（在原文件名基础上添加_interpolated_1deg标识）
        name_parts = input_filename.rsplit('.', 1)
        if len(name_parts) == 2:
            output_filename = f"{name_parts[0]}_interpolated_1deg.{name_parts[1]}"
        else:
            output_filename = f"{input_filename}_interpolated_1deg.nc"
        
        output_path = os.path.join(output_dir, output_filename)
        
        # 设置编码（压缩）
        encoding = {}
        for var in interpolated_ds.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 4,
                'shuffle': True
            }
        
        # 保存数据
        interpolated_ds.to_netcdf(
            output_path,
            encoding=encoding,
            format='NETCDF4'
        )
        
        # 显示文件大小
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ 已保存: {output_filename} ({file_size_mb:.2f} MB)")
        
    except Exception as e:
        print(f"❌ 保存数据失败: {e}")
        import traceback
        traceback.print_exc()


def process_single_file(file_path):
    """处理单个NetCDF文件"""
    try:
        print(f"\n处理文件: {os.path.basename(file_path)}")
        
        # 读取NetCDF文件
        ds = xr.open_dataset(file_path)
        
        # 获取经纬度变量名
        lat_var, lon_var = None, None
        for name in ['lat', 'latitude', 'Lat', 'Latitude']:
            if name in ds.coords or name in ds.sizes:
                lat_var = name
                break
        for name in ['lon', 'longitude', 'Lon', 'Longitude']:
            if name in ds.coords or name in ds.sizes:
                lon_var = name
                break
        
        if lat_var is None or lon_var is None:
            print(f"  ⚠️  无法确定经纬度变量，跳过")
            ds.close()
            return False
        
        # 第一步：先将经度从0~360转换为-180~180（只转换坐标，不插值）
        converted_ds = convert_lon_to_180_180(ds, lon_var)
        
        # 关闭原始数据集
        ds.close()
        
        # 第二步：在转换后的数据上进行插值（-180~180经度范围）
        final_ds = interpolate_to_1degree(converted_ds, lat_var, lon_var, lon_range_180=True)
        
        # 关闭转换后的数据集（已插值为final_ds）
        converted_ds.close()
        
        if final_ds is None:
            print(f"  ❌ 插值失败，跳过")
            return False
        
        # 保存插值后的数据
        save_interpolated_data(final_ds, file_path)
        
        # 关闭数据集
        final_ds.close()
        
        return True
        
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("="*80)
    print("GCM数据线性插值工具")
    print("="*80)
    print(f"输入目录: {INPUT_DIR}")
    print(f"目标分辨率: {TARGET_RESOLUTION}° × {TARGET_RESOLUTION}°")
    print("="*80)
    
    # 检查目录是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 目录不存在: {INPUT_DIR}")
        return
    
    # 查找所有.nc文件（递归搜索）
    nc_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith('.nc'):
                nc_files.append(os.path.join(root, file))
    
    if len(nc_files) == 0:
        print(f"❌ 未找到任何.nc文件")
        return
    
    print(f"\n找到 {len(nc_files)} 个NetCDF文件")
    
    # 处理每个文件
    success_count = 0
    fail_count = 0
    
    for i, file_path in enumerate(nc_files, 1):
        print(f"\n[{i}/{len(nc_files)}]")
        if process_single_file(file_path):
            success_count += 1
        else:
            fail_count += 1
    
    # 打印总结
    print("\n" + "="*80)
    print("处理完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print("="*80)


if __name__ == "__main__":
    main()

