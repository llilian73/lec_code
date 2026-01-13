"""
GCM数据线性插值工具

功能：
将不同空间分辨率的GCM数据通过线性插值统一到1°×1°分辨率
"""

import xarray as xr
import numpy as np
import os
import multiprocessing
import argparse
from tqdm import tqdm

# 输入基础目录路径（Linux路径）
INPUT_BASE_DIR = "/home/linbor/WORK/lishiying/GCM_input_filter"

# 输出基础目录路径（Linux路径）
OUTPUT_BASE_DIR = "/home/linbor/WORK/lishiying/GCM_input_processed"

# 支持的模型列表
MODELS = [
    "ACCESS-ESM1-5",
    "BCC-CSM2-MR",
    "CanESM5",
    "EC-Earth3",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0"
]

# 目标分辨率
TARGET_RESOLUTION = 1.0  # 1度

# 并行处理参数
MAX_WORKERS = 56  # 最大并行进程数（根据服务器核心数设置）


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


def interpolate_to_1degree(ds, lat_var, lon_var, lon_range_180=True):
    """将数据插值到1°×1°分辨率（固定范围：纬度-90~90，经度-180~180或0~360）"""
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
        
        return interpolated_ds
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def convert_lon_to_180_180(ds, lon_var, lat_var=None):
    """将经度从0~360转换为-180~180（只做转换，不做插值）"""
    try:
        # 获取当前经度值
        lons = ds[lon_var].values
        
        # 检查经度范围
        lon_min = float(lons.min())
        lon_max = float(lons.max())
        
        # 如果经度已经是-180~180范围，不需要转换
        if lon_min >= -180 and lon_max <= 180:
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
        
        return ds_new
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ds


def save_interpolated_data(interpolated_ds, input_file_path):
    """保存插值后的数据到NetCDF文件（保存到GCM_input_processed对应路径）"""
    try:
        # 从输入文件路径提取文件名和相对路径
        input_filename = os.path.basename(input_file_path)
        input_dir = os.path.dirname(input_file_path)
        
        # 计算相对于INPUT_BASE_DIR的相对路径
        # 例如：/home/.../GCM_input_filter/ACCESS-ESM1-5/future/SSP126/file.nc
        # 相对路径：ACCESS-ESM1-5/future/SSP126
        try:
            rel_path = os.path.relpath(input_dir, INPUT_BASE_DIR)
        except ValueError:
            # 如果无法计算相对路径，尝试从路径中提取
            path_parts = input_dir.split(os.sep)
            if 'GCM_input_filter' in path_parts:
                idx = path_parts.index('GCM_input_filter')
                rel_path = os.path.join(*path_parts[idx+1:])
            else:
                raise ValueError(f"无法从输入路径提取相对路径: {input_file_path}")
        
        # 构建输出目录路径
        # OUTPUT_BASE_DIR + 相对路径
        output_dir = os.path.join(OUTPUT_BASE_DIR, rel_path)
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
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise


def process_single_file(file_path):
    """处理单个NetCDF文件"""
    try:
        file_name = os.path.basename(file_path)
        
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
            ds.close()
            return (False, file_name, "无法确定经纬度变量")
        
        # 第一步：先将经度从0~360转换为-180~180（只转换坐标，不插值）
        converted_ds = convert_lon_to_180_180(ds, lon_var)
        
        # 关闭原始数据集
        ds.close()
        
        # 第二步：在转换后的数据上进行插值（-180~180经度范围）
        final_ds = interpolate_to_1degree(converted_ds, lat_var, lon_var, lon_range_180=True)
        
        # 关闭转换后的数据集（已插值为final_ds）
        converted_ds.close()
        
        if final_ds is None:
            return (False, file_name, "插值失败")
        
        # 保存插值后的数据
        save_interpolated_data(final_ds, file_path)
        
        # 关闭数据集
        final_ds.close()
        
        return (True, file_name, "成功")
        
    except Exception as e:
        error_msg = str(e)
        import traceback
        traceback.print_exc()
        return (False, os.path.basename(file_path) if file_path else "未知文件", error_msg)


def process_model_files(model_files, model_name):
    """处理单个模型的所有文件"""
    if len(model_files) == 0:
        return 0, 0, []
    
    success_count = 0
    fail_count = 0
    failed_files = []
    
    # 确定实际使用的进程数（不超过文件数和最大进程数）
    num_workers = min(MAX_WORKERS, len(model_files), multiprocessing.cpu_count())
    
    if num_workers > 1:
        # 使用多进程并行处理
        with multiprocessing.Pool(processes=num_workers) as pool:
            # 使用tqdm显示进度
            results = list(tqdm(
                pool.imap(process_single_file, model_files),
                total=len(model_files),
                desc=f"处理 {model_name}"
            ))
        
        # 统计结果
        for success, file_name, message in results:
            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_files.append((file_name, message))
    else:
        # 单进程处理（用于调试）
        for file_path in tqdm(model_files, desc=f"处理 {model_name}"):
            success, file_name, message = process_single_file(file_path)
            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_files.append((file_name, message))
    
    return success_count, fail_count, failed_files


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='GCM数据线性插值工具 - 将不同空间分辨率的GCM数据通过线性插值统一到1°×1°分辨率'
    )
    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='要处理的文件夹名称（例如: future, historical, future_56_60等）'
    )
    args = parser.parse_args()
    
    folder_name = args.folder
    
    print("="*80)
    print("GCM数据线性插值工具 (Linux并行版本 - 逐个模型处理)")
    print("="*80)
    print(f"输入基础目录: {INPUT_BASE_DIR}")
    print(f"输出基础目录: {OUTPUT_BASE_DIR}")
    print(f"要处理的文件夹: {folder_name}")
    print(f"支持的模型: {', '.join(MODELS)}")
    print(f"目标分辨率: {TARGET_RESOLUTION}° × {TARGET_RESOLUTION}°")
    print(f"最大并行进程数: {MAX_WORKERS}")
    print("="*80)
    
    # 检查输入基础目录是否存在
    if not os.path.exists(INPUT_BASE_DIR):
        print(f"❌ 输入基础目录不存在: {INPUT_BASE_DIR}")
        return
    
    # 创建输出基础目录
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # 收集每个模型的文件列表（不立即处理，避免内存占用）
    model_files_dict = {}
    
    print("\n扫描文件...")
    # 遍历所有模型，收集文件列表
    for model in MODELS:
        # 构建指定文件夹的路径：INPUT_BASE_DIR/{模型名}/{文件夹名}
        model_folder_dir = os.path.join(INPUT_BASE_DIR, model, folder_name)
        
        if not os.path.exists(model_folder_dir):
            print(f"⚠️  文件夹不存在，跳过: {model_folder_dir}")
            continue
        
        # 查找该模型指定文件夹下的所有.nc文件（递归搜索）
        model_files = []
        for root, dirs, files in os.walk(model_folder_dir):
            for file in files:
                if file.endswith('.nc'):
                    file_path = os.path.join(root, file)
                    model_files.append(file_path)
        
        if len(model_files) > 0:
            model_files_dict[model] = model_files
            print(f"✓ {model}/{folder_name}: 找到 {len(model_files)} 个文件")
        else:
            print(f"⚠️  {model}/{folder_name}: 未找到任何.nc文件")
    
    if len(model_files_dict) == 0:
        print(f"❌ 未找到任何.nc文件")
        return
    
    total_files = sum(len(files) for files in model_files_dict.values())
    print(f"\n总共找到 {total_files} 个NetCDF文件")
    print("\n各模型文件统计:")
    for model, files in model_files_dict.items():
        print(f"  {model}: {len(files)} 个文件")
    
    # 逐个模型处理（避免内存不足）
    print("\n" + "="*80)
    print("开始逐个模型处理（避免内存溢出）")
    print("="*80)
    
    total_success = 0
    total_fail = 0
    all_failed_files = []
    
    for model_idx, (model, model_files) in enumerate(model_files_dict.items(), 1):
        print(f"\n[{model_idx}/{len(model_files_dict)}] 处理模型: {model}")
        print(f"  文件数: {len(model_files)}")
        
        # 处理该模型的所有文件
        success_count, fail_count, failed_files = process_model_files(model_files, model)
        
        total_success += success_count
        total_fail += fail_count
        all_failed_files.extend(failed_files)
        
        print(f"  ✓ {model} 处理完成: 成功 {success_count}, 失败 {fail_count}")
        
        # 处理完一个模型后，强制垃圾回收释放内存
        import gc
        gc.collect()
    
    # 打印总结
    print("\n" + "="*80)
    print("所有模型处理完成!")
    print(f"总成功: {total_success} 个文件")
    print(f"总失败: {total_fail} 个文件")
    
    if all_failed_files:
        print("\n失败的文件:")
        for file_name, message in all_failed_files[:10]:  # 只显示前10个
            print(f"  - {file_name}: {message}")
        if len(all_failed_files) > 10:
            print(f"  ... 还有 {len(all_failed_files) - 10} 个失败文件")
    
    print("="*80)


if __name__ == "__main__":
    main()

