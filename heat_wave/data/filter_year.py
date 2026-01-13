"""
根据文件名解析CMIP6数据并提取指定年份的数据

功能：
1. 解析CMIP6文件名格式，提取模型名称、变量名、情景、时间范围
2. 处理历史数据：提取1981-2010年数据并整合
3. 处理未来数据：提取2030-2034年数据，每个变量一个文件
"""

import xarray as xr
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
from datetime import datetime
import cftime
import multiprocessing
from functools import partial
from tqdm import tqdm
import gc

# ============================================================================
# 模型选择配置 - 修改这里选择要处理的模型
# ============================================================================
MODEL_NAME = "ACCESS-ESM1-5"  # 可选: "BCC-CSM2-MR", "ACCESS-ESM1-5", "CanESM5", "EC-Earth3", "MPI-ESM1-2-HR", "MRI-ESM2-0"

# 已处理的模型列表（这些模型将被跳过）
PROCESSED_MODELS = ["BCC-CSM2-MR"]  # 如果模型已处理完成，添加到这个列表

# ============================================================================
# 配置路径
# ============================================================================
HISTORICAL_INPUT_DIR = r"Z:\local_environment_creation\heat_wave\GCM_input\historical"
FUTURE_INPUT_DIR = r"Z:\local_environment_creation\heat_wave\GCM_input"  # 未来数据可能在不同子目录
EC_EARTH_HISTORICAL_DIR = r"Z:\CMIP6\tasmax"  # EC-Earth3历史数据特殊路径
OUTPUT_BASE_DIR = r"Z:\local_environment_creation\heat_wave\GCM_input_filter"

# 时间范围配置
HISTORICAL_YEARS = (1981, 2010)  # 历史数据提取年份
FUTURE_YEARS = (2030, 2034)  # 未来数据提取年份

# 未来数据变量列表
FUTURE_VARIABLES = ['tas', 'tasmax', 'vas', 'uas', 'huss', 'rsds']

# 未来数据情景列表
FUTURE_SCENARIOS = ['ssp126', 'ssp245']

# 性能优化配置
USE_PARALLEL = True  # 是否使用并行处理
NUM_WORKERS = None  # None表示自动检测CPU核心数，或指定具体数量
CHUNKS = {'time': 1000}  # 只对时间维度分块（空间维度不筛选，不需要chunking）
COMPRESSION_LEVEL = 4  # NetCDF压缩级别 (1-9, 4是平衡速度和大小)


def parse_filename(filename):
    """
    解析CMIP6文件名
    
    格式：{变量名}_day_{模型名}_{情景}_{r1i1p1f1}_{网格类型}_{时间范围}.nc
    
    返回：
        dict: {
            'variable': 变量名,
            'model': 模型名,
            'scenario': 情景,
            'time_range': (start_date, end_date),
            'full_path': 完整路径
        }
    """
    # 去掉路径，只保留文件名
    basename = os.path.basename(filename)
    
    # 去掉.nc后缀
    if basename.endswith('.nc'):
        basename = basename[:-3]
    
    # 按下划线分割
    parts = basename.split('_')
    
    if len(parts) < 7:
        raise ValueError(f"文件名格式不正确: {filename}")
    
    # 解析各部分
    variable = parts[0]  # 变量名
    resolution = parts[1]  # day/mon/yr等
    model = parts[2]  # 模型名
    scenario = parts[3]  # 情景
    realization = parts[4]  # r1i1p1f1
    grid = parts[5]  # gn/gr
    time_range_str = parts[6]  # 时间范围，如19750101-19991231
    
    # 解析时间范围
    if '-' in time_range_str:
        start_str, end_str = time_range_str.split('-')
        start_date = datetime.strptime(start_str, '%Y%m%d')
        end_date = datetime.strptime(end_str, '%Y%m%d')
    else:
        raise ValueError(f"时间范围格式不正确: {time_range_str}")
    
    return {
        'variable': variable,
        'resolution': resolution,
        'model': model,
        'scenario': scenario,
        'realization': realization,
        'grid': grid,
        'time_range': (start_date, end_date),
        'start_date': start_date,
        'end_date': end_date,
        'filename': basename
    }


def find_historical_files(input_dir, model_name):
    """查找历史数据文件（根据不同模型使用不同策略）"""
    files = []
    
    # EC-Earth3使用特殊路径
    if model_name == "EC-Earth3":
        search_dir = EC_EARTH_HISTORICAL_DIR
    else:
        search_dir = input_dir
    
    if not os.path.exists(search_dir):
        print(f"警告: 目录不存在: {search_dir}")
        return files
    
    # 根据不同模型使用不同的查找策略
    if model_name == "ACCESS-ESM1-5":
        # 查找指定的两个文件
        target_files = [
            "tasmax_day_ACCESS-ESM1-5_historical_r1i1p1f1_gn_19500101-19991231.nc",
            "tasmax_day_ACCESS-ESM1-5_historical_r1i1p1f1_gn_20000101-20141231.nc"
        ]
        for target_file in target_files:
            filepath = os.path.join(search_dir, target_file)
            if os.path.exists(filepath):
                try:
                    file_info = parse_filename(filepath)
                    files.append({'path': filepath, 'info': file_info})
                except Exception as e:
                    print(f"警告: 无法解析文件 {target_file}: {e}")
    
    elif model_name == "CanESM5":
        # 查找指定的一个文件
        target_file = "tasmax_day_CanESM5_historical_r1i1p1f1_gn_18500101-20141231.nc"
        filepath = os.path.join(search_dir, target_file)
        if os.path.exists(filepath):
            try:
                file_info = parse_filename(filepath)
                files.append({'path': filepath, 'info': file_info})
            except Exception as e:
                print(f"警告: 无法解析文件 {target_file}: {e}")
    
    elif model_name == "EC-Earth3":
        # EC-Earth3是一年一个文件，需要查找所有相关年份的文件
        # 查找所有包含模型名和historical的文件
        for filename in os.listdir(search_dir):
            if filename.endswith('.nc') and 'EC-Earth3' in filename and 'historical' in filename:
                filepath = os.path.join(search_dir, filename)
                try:
                    file_info = parse_filename(filepath)
                    if file_info['model'] in ['EC-Earth3', 'EC-Earth3-HR'] and file_info['scenario'] == 'historical':
                        files.append({'path': filepath, 'info': file_info})
                except Exception as e:
                    continue  # 跳过无法解析的文件
    
    elif model_name == "MPI-ESM1-2-HR":
        # 查找所有包含模型名和historical的文件
        for filename in os.listdir(search_dir):
            if filename.endswith('.nc') and 'MPI-ESM1-2-HR' in filename and 'historical' in filename:
                filepath = os.path.join(search_dir, filename)
                try:
                    file_info = parse_filename(filepath)
                    if file_info['model'] == 'MPI-ESM1-2-HR' and file_info['scenario'] == 'historical':
                        files.append({'path': filepath, 'info': file_info})
                except Exception as e:
                    continue
    
    elif model_name == "MRI-ESM2-0":
        # 查找指定的两个文件
        target_files = [
            "tasmax_day_MRI-ESM2-0_historical_r1i1p1f1_gn_19500101-19991231.nc",
            "tasmax_day_MRI-ESM2-0_historical_r1i1p1f1_gn_20000101-20141231.nc"
        ]
        for target_file in target_files:
            filepath = os.path.join(search_dir, target_file)
            if os.path.exists(filepath):
                try:
                    file_info = parse_filename(filepath)
                    files.append({'path': filepath, 'info': file_info})
                except Exception as e:
                    print(f"警告: 无法解析文件 {target_file}: {e}")
    
    else:
        # 默认策略：BCC-CSM2-MR或其他模型
        for filename in os.listdir(search_dir):
            if filename.endswith('.nc') and model_name in filename and 'historical' in filename:
                filepath = os.path.join(search_dir, filename)
                try:
                    file_info = parse_filename(filepath)
                    if file_info['model'] == model_name and file_info['scenario'] == 'historical':
                        files.append({'path': filepath, 'info': file_info})
                except Exception as e:
                    print(f"警告: 无法解析文件 {filename}: {e}")
    
    return files


def find_future_files(input_base_dir, model_name, variables, scenarios, start_year, end_year):
    """查找未来数据文件（根据不同模型使用不同策略）"""
    files = []
    
    # 根据不同模型使用不同的查找策略
    if model_name == "ACCESS-ESM1-5":
        # 查找20150101-20641231的文件
        search_dirs = [
            os.path.join(input_base_dir, model_name),
            input_base_dir
        ]
        target_time_range = "20150101-20641231"
        
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            for root, dirs, filenames in os.walk(search_dir):
                for filename in filenames:
                    if filename.endswith('.nc') and target_time_range in filename:
                        filepath = os.path.join(root, filename)
                        try:
                            file_info = parse_filename(filepath)
                            if (file_info['model'] == model_name and 
                                file_info['variable'] in variables and
                                file_info['scenario'] in scenarios):
                                files.append({'path': filepath, 'info': file_info})
                        except:
                            continue
    
    elif model_name == "MPI-ESM1-2-HR":
        # 直接选20300101-20341231的文件（5年一个文件）
        search_dirs = [
            os.path.join(input_base_dir, model_name),
            input_base_dir
        ]
        target_time_range = "20300101-20341231"
        
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            for root, dirs, filenames in os.walk(search_dir):
                for filename in filenames:
                    if filename.endswith('.nc') and target_time_range in filename:
                        filepath = os.path.join(root, filename)
                        try:
                            file_info = parse_filename(filepath)
                            if (file_info['model'] == model_name and 
                                file_info['variable'] in variables and
                                file_info['scenario'] in scenarios):
                                files.append({'path': filepath, 'info': file_info})
                        except:
                            continue
    
    elif model_name == "MRI-ESM2-0":
        # 查找20150101-20641231的文件
        search_dirs = [
            os.path.join(input_base_dir, model_name),
            input_base_dir
        ]
        target_time_range = "20150101-20641231"
        
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            for root, dirs, filenames in os.walk(search_dir):
                for filename in filenames:
                    if filename.endswith('.nc') and target_time_range in filename:
                        filepath = os.path.join(root, filename)
                        try:
                            file_info = parse_filename(filepath)
                            if (file_info['model'] == model_name and 
                                file_info['variable'] in variables and
                                file_info['scenario'] in scenarios):
                                files.append({'path': filepath, 'info': file_info})
                        except:
                            continue
    
    elif model_name == "EC-Earth3":
        # EC-Earth3在对应模型文件夹下，一年一个文件，需要查找所有相关年份的文件
        search_dir = os.path.join(input_base_dir, "EC-Earth3")
        if not os.path.exists(search_dir):
            search_dir = input_base_dir
        
        # 计算需要查找的年份范围
        target_years = set(range(start_year, end_year + 1))
        
        for root, dirs, filenames in os.walk(search_dir):
            for filename in filenames:
                if not filename.endswith('.nc'):
                    continue
                if 'EC-Earth3' not in filename:
                    continue
                
                filepath = os.path.join(root, filename)
                try:
                    file_info = parse_filename(filepath)
                    # 检查文件时间范围是否与目标年份有重叠
                    file_start_year = file_info['start_date'].year
                    file_end_year = file_info['end_date'].year
                    file_years = set(range(file_start_year, file_end_year + 1))
                    
                    if (file_info['model'] in ['EC-Earth3', 'EC-Earth3-HR'] and 
                        file_info['variable'] in variables and
                        file_info['scenario'] in scenarios and
                        target_years & file_years):  # 有重叠
                        files.append({'path': filepath, 'info': file_info})
                except:
                    continue
    
    else:
        # 默认策略：CanESM5、BCC-CSM2-MR等
        # 可能的未来数据目录
        possible_dirs = [
            input_base_dir,
            os.path.join(input_base_dir, 'future'),
            os.path.join(input_base_dir, model_name),
        ]
        
        # 为每个模型创建可能的目录
        for scenario in scenarios:
            possible_dirs.append(os.path.join(input_base_dir, model_name, scenario))
            possible_dirs.append(os.path.join(input_base_dir, scenario, model_name))
        
        # 递归搜索
        for root_dir in possible_dirs:
            if not os.path.exists(root_dir):
                continue
            
            for root, dirs, filenames in os.walk(root_dir):
                for filename in filenames:
                    if not filename.endswith('.nc'):
                        continue
                    
                    filepath = os.path.join(root, filename)
                    try:
                        file_info = parse_filename(filepath)
                        
                        if (file_info['model'] == model_name and 
                            file_info['variable'] in variables and
                            file_info['scenario'] in scenarios):
                            files.append({'path': filepath, 'info': file_info})
                    except:
                        continue
    
    return files


def process_single_file(args):
    """处理单个文件（用于并行处理）"""
    file_info, start_year, end_year, model_name = args
    filepath = file_info['path']
    
    try:
        # 尝试使用netcdf4引擎打开文件，使用chunks优化
        try:
            ds = xr.open_dataset(filepath, engine='netcdf4', chunks=CHUNKS)
        except:
            try:
                ds = xr.open_dataset(filepath, engine='h5netcdf', chunks=CHUNKS)
            except:
                ds = xr.open_dataset(filepath, chunks=CHUNKS)
        
        # 对于MPI-ESM1-2-HR，如果文件时间范围正好是目标年份，可以直接使用
        if model_name == "MPI-ESM1-2-HR":
            file_start_year = file_info['info']['start_date'].year
            file_end_year = file_info['info']['end_date'].year
            if file_start_year == start_year and file_end_year == end_year:
                # 文件时间范围正好匹配，直接使用，不需要筛选
                if hasattr(ds, 'chunks') and ds.chunks:
                    filtered_ds = ds.load()
                else:
                    filtered_ds = ds
                # 注意：这里不关闭ds，因为filtered_ds可能引用它
                if len(filtered_ds.time) > 0:
                    return {
                        'success': True,
                        'data': filtered_ds,
                        'filename': os.path.basename(filepath),
                        'time_range': (filtered_ds.time.values[0], filtered_ds.time.values[-1]),
                        'time_count': len(filtered_ds.time)
                    }
        
        # 筛选时间范围
        filtered_ds = filter_time_range(ds, start_year, end_year)
        
        # 如果使用chunks，需要先加载数据
        if hasattr(filtered_ds, 'chunks') and filtered_ds.chunks:
            filtered_ds = filtered_ds.load()
        
        # 关闭原始数据集
        ds.close()
        
        if len(filtered_ds.time) > 0:
            return {
                'success': True,
                'data': filtered_ds,
                'filename': os.path.basename(filepath),
                'time_range': (filtered_ds.time.values[0], filtered_ds.time.values[-1]),
                'time_count': len(filtered_ds.time)
            }
        else:
            filtered_ds.close()
            return {
                'success': False,
                'filename': os.path.basename(filepath),
                'message': f'该文件在 {start_year}-{end_year} 范围内没有数据'
            }
    except Exception as e:
        return {
            'success': False,
            'filename': os.path.basename(filepath),
            'message': f'处理文件失败: {str(e)}'
        }


def check_missing_years(time_values, start_year, end_year):
    """检查缺少年份"""
    # 提取所有年份
    years = set()
    for t in time_values:
        if hasattr(t, 'year'):
            years.add(t.year)
        elif isinstance(t, pd.Timestamp):
            years.add(t.year)
        else:
            try:
                years.add(pd.Timestamp(t).year)
            except:
                pass
    
    # 计算期望的年份
    expected_years = set(range(start_year, end_year + 1))
    missing_years = sorted(expected_years - years)
    
    return missing_years, years


def filter_time_range(ds, start_year, end_year):
    """筛选指定年份范围的数据（处理CMIP6的cftime日历系统）"""
    # 确保时间维度存在
    if 'time' not in ds.dims and 'time' not in ds.coords:
        raise ValueError("数据集中没有找到时间维度")
    
    # 获取时间坐标
    time_coord = ds.time
    time_values = time_coord.values
    
    # 方法1：尝试使用年份索引筛选（最可靠的方法）
    try:
        # 获取时间值的年份
        if len(time_values) == 0:
            raise ValueError("时间坐标为空")
        
        first_time = time_values[0]
        
        # 判断时间类型并提取年份
        if hasattr(first_time, 'year'):
            # cftime对象
            years = np.array([t.year for t in time_values])
        elif isinstance(first_time, pd.Timestamp):
            # pandas Timestamp
            years = np.array([t.year for t in time_values])
        else:
            # 尝试转换为pandas时间
            try:
                time_pd = pd.to_datetime(time_values)
                years = np.array([t.year for t in time_pd])
            except:
                # 如果都失败，尝试从字符串解析
                years = np.array([int(str(t)[:4]) for t in time_values])
        
        # 创建年份掩码
        mask = (years >= start_year) & (years <= end_year)
        
        if not np.any(mask):
            # 如果没有匹配的数据，返回空数据集
            return ds.isel(time=[])
        
        # 使用掩码筛选
        filtered_ds = ds.isel(time=mask)
        return filtered_ds
        
    except Exception as e1:
        # 方法2：如果方法1失败，尝试使用时间切片
        try:
            # 获取日历类型
            calendar = 'standard'
            if hasattr(time_values[0], 'calendar'):
                calendar = time_values[0].calendar
            elif hasattr(time_coord, 'calendar'):
                calendar = time_coord.calendar
            
            # 根据日历类型创建对应的日期对象
            if calendar == 'noleap' or calendar == '365_day':
                start_date = cftime.DatetimeNoLeap(start_year, 1, 1)
                end_date = cftime.DatetimeNoLeap(end_year, 12, 31)
            elif calendar == '360_day':
                start_date = cftime.Datetime360Day(start_year, 1, 1)
                end_date = cftime.Datetime360Day(end_year, 12, 31)
            elif calendar == 'gregorian' or calendar == 'standard' or calendar == 'proleptic_gregorian':
                start_date = cftime.DatetimeGregorian(start_year, 1, 1)
                end_date = cftime.DatetimeGregorian(end_year, 12, 31)
            else:
                # 默认使用标准日历
                start_date = cftime.DatetimeGregorian(start_year, 1, 1)
                end_date = cftime.DatetimeGregorian(end_year, 12, 31)
            
            filtered_ds = ds.sel(time=slice(start_date, end_date))
            return filtered_ds
            
        except Exception as e2:
            # 方法3：使用pandas Timestamp（最后尝试）
            try:
                start_date = pd.Timestamp(f'{start_year}-01-01')
                end_date = pd.Timestamp(f'{end_year}-12-31 23:59:59')
                filtered_ds = ds.sel(time=slice(start_date, end_date))
                return filtered_ds
            except Exception as e3:
                raise ValueError(f"无法筛选时间范围: {e1}, {e2}, {e3}")


def process_historical_data(model_name, input_dir, output_dir, start_year, end_year):
    """处理历史数据：提取指定年份并整合"""
    print(f"\n处理历史数据: {model_name} ({start_year}-{end_year})")
    
    # 查找文件
    files = find_historical_files(input_dir, model_name)
    
    if len(files) == 0:
        print(f"❌ 未找到 {model_name} 的历史数据文件")
        return
    
    # print(f"找到 {len(files)} 个历史数据文件:")
    # for f in files:
    #     print(f"  - {os.path.basename(f['path'])}")
    #     print(f"    时间范围: {f['info']['start_date'].strftime('%Y-%m-%d')} 到 {f['info']['end_date'].strftime('%Y-%m-%d')}")
    
    # 筛选出包含目标年份的文件
    target_start = datetime(start_year, 1, 1)
    target_end = datetime(end_year, 12, 31)
    
    relevant_files = []
    for f in files:
        file_start = f['info']['start_date']
        file_end = f['info']['end_date']
        
        # 检查文件时间范围是否与目标年份有重叠
        if not (file_end < target_start or file_start > target_end):
            relevant_files.append(f)
    
    if len(relevant_files) == 0:
        print(f"❌ 未找到包含 {start_year}-{end_year} 年数据的文件")
        return
    
    print(f"处理文件: {', '.join([os.path.basename(f['path']) for f in relevant_files])}")
    
    # 读取并筛选数据（并行或串行）
    datasets = []
    
    if USE_PARALLEL and len(relevant_files) > 1:
        # 并行处理
        num_workers = NUM_WORKERS if NUM_WORKERS else min(multiprocessing.cpu_count(), len(relevant_files))
        # print(f"使用并行处理 ({num_workers} 个进程)...")
        
        # 准备参数（添加模型名称）
        args_list = [(f, start_year, end_year, model_name) for f in relevant_files]
        
        # 并行处理
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, args_list),
                total=len(args_list),
                desc="处理文件",
                disable=True  # 禁用tqdm显示
            ))
        
        # 收集成功的结果
        for result in results:
            if result['success']:
                # print(f"  ✓ {result['filename']}: {result['time_count']} 个时间点")
                datasets.append(result['data'])
            else:
                print(f"  ✗ {result['filename']}: {result['message']}")
    else:
        # 串行处理（原始方法，但使用chunks优化）
        for f in relevant_files:
            # print(f"\n处理文件: {os.path.basename(f['path'])}")
            try:
                # 尝试使用netcdf4引擎打开文件，使用chunks优化
                try:
                    ds = xr.open_dataset(f['path'], engine='netcdf4', chunks=CHUNKS)
                except:
                    try:
                        ds = xr.open_dataset(f['path'], engine='h5netcdf', chunks=CHUNKS)
                    except:
                        ds = xr.open_dataset(f['path'], chunks=CHUNKS)
                
                # print(f"  原始时间范围: {ds.time.values[0]} 到 {ds.time.values[-1]}")
                
                # 筛选时间范围
                filtered_ds = filter_time_range(ds, start_year, end_year)
                
                # 如果使用chunks，需要先加载数据
                if hasattr(filtered_ds, 'chunks') and filtered_ds.chunks:
                    filtered_ds = filtered_ds.load()
                
                if len(filtered_ds.time) > 0:
                    # print(f"  筛选后时间范围: {filtered_ds.time.values[0]} 到 {filtered_ds.time.values[-1]}")
                    # print(f"  筛选后数据点数: {len(filtered_ds.time)}")
                    datasets.append(filtered_ds)
                else:
                    print(f"  ⚠️  警告: {os.path.basename(f['path'])} 在 {start_year}-{end_year} 范围内没有数据")
                    ds.close()
            except Exception as e:
                print(f"  ❌ 错误: {os.path.basename(f['path'])} 处理失败: {e}")
                continue
    
    if len(datasets) == 0:
        print("没有有效的数据可以合并")
        return
    
    # 合并数据集（使用分批合并策略，避免内存溢出）
    # print(f"\n合并 {len(datasets)} 个数据集...")
    try:
        # 如果只有一个数据集，直接使用（避免不必要的合并操作）
        if len(datasets) == 1:
            combined_ds = datasets[0]
        else:
            # 对于大量数据集，使用分批合并策略
            # 特别是EC-Earth3这种一年一个文件的，需要分批处理
            batch_size = 5  # 每批合并5个数据集
            
            if len(datasets) > batch_size:
                # print(f"使用分批合并策略（每批 {batch_size} 个数据集）...")
                
                # 先按时间排序数据集（根据第一个时间点）
                def get_first_time(ds):
                    try:
                        return pd.Timestamp(ds.time.values[0])
                    except:
                        # 如果是cftime对象
                        if hasattr(ds.time.values[0], 'year'):
                            return datetime(ds.time.values[0].year, 1, 1)
                        return datetime(2000, 1, 1)
                
                datasets_sorted = sorted(datasets, key=get_first_time)
                
                # 分批合并
                current_combined = None
                for i in range(0, len(datasets_sorted), batch_size):
                    batch = datasets_sorted[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    total_batches = (len(datasets_sorted) + batch_size - 1) // batch_size
                    
                    # print(f"  合并批次 {batch_num}/{total_batches} ({len(batch)} 个数据集)...")
                    
                    # 合并当前批次
                    if len(batch) == 1:
                        batch_combined = batch[0]
                    else:
                        try:
                            batch_combined = xr.combine_by_coords(batch, combine_attrs='drop_conflicts')
                            if batch_combined is None or len(batch_combined.time) == 0:
                                batch_combined = xr.concat(batch, dim='time')
                                batch_combined = batch_combined.sortby('time')
                        except:
                            batch_combined = xr.concat(batch, dim='time')
                            batch_combined = batch_combined.sortby('time')
                    
                    # 释放批次数据集的内存
                    for ds in batch:
                        try:
                            ds.close()
                        except:
                            pass
                    del batch
                    gc.collect()
                    
                    # 与之前合并的结果合并
                    if current_combined is None:
                        current_combined = batch_combined
                    else:
                        try:
                            current_combined = xr.combine_by_coords(
                                [current_combined, batch_combined], 
                                combine_attrs='drop_conflicts'
                            )
                            if current_combined is None or len(current_combined.time) == 0:
                                current_combined = xr.concat([current_combined, batch_combined], dim='time')
                                current_combined = current_combined.sortby('time')
                        except:
                            current_combined = xr.concat([current_combined, batch_combined], dim='time')
                            current_combined = current_combined.sortby('time')
                        
                        # 释放批次合并结果的内存
                        try:
                            batch_combined.close()
                        except:
                            pass
                        del batch_combined
                        gc.collect()
                
                combined_ds = current_combined
                
                # 去除重复的时间点（使用更节省内存的方法）
                # print("  去除重复时间点...")
                try:
                    # 先获取唯一的时间索引
                    time_values = combined_ds.time.values
                    if hasattr(time_values[0], 'year'):
                        # cftime对象，转换为字符串进行比较
                        time_strs = [str(t) for t in time_values]
                        _, unique_indices = np.unique(time_strs, return_index=True)
                    else:
                        # pandas Timestamp或其他
                        _, unique_indices = np.unique(time_values, return_index=True)
                    
                    # 如果所有时间都是唯一的，跳过去重
                    if len(unique_indices) < len(time_values):
                        # print(f"  发现 {len(time_values) - len(unique_indices)} 个重复时间点")
                        combined_ds = combined_ds.isel(time=unique_indices)
                except Exception as e:
                    # print(f"  警告: 去重失败，尝试使用drop_duplicates: {e}")
                    # 如果上面的方法失败，尝试使用drop_duplicates（可能内存占用较大）
                    try:
                        combined_ds = combined_ds.drop_duplicates(dim='time')
                    except Exception as e2:
                        # print(f"  警告: drop_duplicates也失败: {e2}，跳过去重")
                        pass
            else:
                # 数据集数量较少，直接合并
                try:
                    combined_ds = xr.combine_by_coords(datasets, combine_attrs='drop_conflicts')
                    if combined_ds is None or len(combined_ds.time) == 0:
                        combined_ds = xr.concat(datasets, dim='time')
                        combined_ds = combined_ds.sortby('time')
                except:
                    combined_ds = xr.concat(datasets, dim='time')
                    combined_ds = combined_ds.sortby('time')
                
                # 去除重复的时间点（使用节省内存的方法）
                try:
                    time_values = combined_ds.time.values
                    if hasattr(time_values[0], 'year'):
                        time_strs = [str(t) for t in time_values]
                        _, unique_indices = np.unique(time_strs, return_index=True)
                    else:
                        _, unique_indices = np.unique(time_values, return_index=True)
                    
                    if len(unique_indices) < len(time_values):
                        combined_ds = combined_ds.isel(time=unique_indices)
                except:
                    try:
                        combined_ds = combined_ds.drop_duplicates(dim='time')
                    except:
                        pass  # 如果去重失败，继续处理
        
        # 检查缺少年份
        time_values = combined_ds.time.values
        missing_years, existing_years = check_missing_years(time_values, start_year, end_year)
        
        if missing_years:
            print(f"⚠️  缺少年份: {missing_years}")
            print(f"   已有年份: {sorted(existing_years)}")
        else:
            print(f"✓ 所有年份完整 ({start_year}-{end_year})")
        
        # print(f"合并后时间范围: {combined_ds.time.values[0]} 到 {combined_ds.time.values[-1]}")
        # print(f"合并后数据点数: {len(combined_ds.time)}")
        
        # 创建输出目录
        output_path = os.path.join(output_dir, model_name, 'historical')
        os.makedirs(output_path, exist_ok=True)
        
        # 生成输出文件名（从第一个文件的信息中获取变量名）
        variable = relevant_files[0]['info']['variable']
        
        output_filename = f"{variable}_day_{model_name}_historical_r1i1p1f1_gn_{start_year}0101-{end_year}1231.nc"
        output_filepath = os.path.join(output_path, output_filename)
        
        # 保存文件（使用压缩和优化选项）
        print(f"保存到: {output_filepath}")
        encoding = {}
        for var in combined_ds.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': COMPRESSION_LEVEL,
                'shuffle': True
            }
        # 对坐标变量也进行压缩
        for coord in combined_ds.coords:
            if coord != 'time':  # time坐标通常不压缩
                encoding[coord] = {
                    'zlib': True,
                    'complevel': COMPRESSION_LEVEL
                }
        
        combined_ds.to_netcdf(
            output_filepath,
            encoding=encoding,
            format='NETCDF4'
        )
        print(f"✓ 历史数据已保存\n")
        
        # 释放内存
        del combined_ds
        for ds in datasets:
            try:
                ds.close()
            except:
                pass
        datasets.clear()
        gc.collect()
        
    except Exception as e:
        print(f"错误: 合并数据失败: {e}")
        import traceback
        traceback.print_exc()
        # 关闭所有数据集
        for ds in datasets:
            try:
                ds.close()
            except:
                pass


def process_future_data(model_name, input_base_dir, output_dir, variables, scenarios, start_year, end_year):
    """处理未来数据：提取指定年份，每个变量一个文件"""
    print(f"\n处理未来数据: {model_name} ({start_year}-{end_year})")
    
    # 查找文件（传入年份范围用于特殊模型的查找）
    files = find_future_files(input_base_dir, model_name, variables, scenarios, start_year, end_year)
    
    if len(files) == 0:
        print(f"❌ 未找到 {model_name} 的未来数据文件")
        # print(f"搜索目录: {input_base_dir}")
        return
    
    # print(f"找到 {len(files)} 个未来数据文件")
    
    # 按变量和情景分组
    file_groups = {}
    for f in files:
        var = f['info']['variable']
        scenario = f['info']['scenario']
        key = (var, scenario)
        
        if key not in file_groups:
            file_groups[key] = []
        file_groups[key].append(f)
    
    # print(f"\n按变量和情景分组:")
    # for (var, scenario), group_files in file_groups.items():
    #     print(f"  {var} - {scenario}: {len(group_files)} 个文件")
    
    # 处理每个变量和情景的组合
    for (variable, scenario), group_files in file_groups.items():
        print(f"\n处理: {variable} - {scenario.upper()}")
        
        # 筛选出包含目标年份的文件
        target_start = datetime(start_year, 1, 1)
        target_end = datetime(end_year, 12, 31)
        
        relevant_files = []
        for f in group_files:
            file_start = f['info']['start_date']
            file_end = f['info']['end_date']
            
            if not (file_end < target_start or file_start > target_end):
                relevant_files.append(f)
        
        if len(relevant_files) == 0:
            print(f"  ❌ 未找到包含 {start_year}-{end_year} 年数据的文件")
            continue
        
        print(f"  处理文件: {', '.join([os.path.basename(f['path']) for f in relevant_files])}")
        
        # 读取并筛选数据（并行或串行）
        datasets = []
        
        if USE_PARALLEL and len(relevant_files) > 1:
            # 并行处理
            num_workers = NUM_WORKERS if NUM_WORKERS else min(multiprocessing.cpu_count(), len(relevant_files))
            # print(f"使用并行处理 ({num_workers} 个进程)...")
            
            # 准备参数（添加模型名称）
            args_list = [(f, start_year, end_year, model_name) for f in relevant_files]
            
            # 并行处理
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_single_file, args_list),
                    total=len(args_list),
                    desc=f"处理 {variable}-{scenario}",
                    disable=True  # 禁用tqdm显示
                ))
            
            # 收集成功的结果
            for result in results:
                if result['success']:
                    # print(f"  ✓ {result['filename']}: {result['time_count']} 个时间点")
                    datasets.append(result['data'])
                else:
                    print(f"  ✗ {result['filename']}: {result['message']}")
        else:
            # 串行处理（使用chunks优化）
            for f in relevant_files:
                # print(f"\n处理文件: {os.path.basename(f['path'])}")
                try:
                    # 尝试使用netcdf4引擎打开文件，使用chunks优化
                    try:
                        ds = xr.open_dataset(f['path'], engine='netcdf4', chunks=CHUNKS)
                    except:
                        try:
                            ds = xr.open_dataset(f['path'], engine='h5netcdf', chunks=CHUNKS)
                        except:
                            ds = xr.open_dataset(f['path'], chunks=CHUNKS)
                    
                    # print(f"  原始时间范围: {ds.time.values[0]} 到 {ds.time.values[-1]}")
                    
                    # 对于MPI-ESM1-2-HR，如果文件时间范围正好是目标年份，可以直接使用
                    if model_name == "MPI-ESM1-2-HR":
                        file_start_year = f['info']['start_date'].year
                        file_end_year = f['info']['end_date'].year
                        if file_start_year == start_year and file_end_year == end_year:
                            # 文件时间范围正好匹配，直接使用，不需要筛选
                            # print(f"  文件时间范围正好匹配目标年份，直接使用（无需筛选）")
                            if hasattr(ds, 'chunks') and ds.chunks:
                                filtered_ds = ds.load()
                            else:
                                filtered_ds = ds
                            datasets.append(filtered_ds)
                            continue
                    
                    # 筛选时间范围
                    filtered_ds = filter_time_range(ds, start_year, end_year)
                    
                    # 如果使用chunks，需要先加载数据
                    if hasattr(filtered_ds, 'chunks') and filtered_ds.chunks:
                        filtered_ds = filtered_ds.load()
                    
                    if len(filtered_ds.time) > 0:
                        # print(f"  筛选后时间范围: {filtered_ds.time.values[0]} 到 {filtered_ds.time.values[-1]}")
                        # print(f"  筛选后数据点数: {len(filtered_ds.time)}")
                        datasets.append(filtered_ds)
                    else:
                        print(f"  ⚠️  警告: {os.path.basename(f['path'])} 在 {start_year}-{end_year} 范围内没有数据")
                        ds.close()
                except Exception as e:
                    print(f"  ❌ 错误: {os.path.basename(f['path'])} 处理失败: {e}")
                    continue
        
        if len(datasets) == 0:
            print(f"没有有效的数据可以合并")
            continue
        
        # 合并数据集（使用分批合并策略，避免内存溢出）
        # print(f"\n合并 {len(datasets)} 个数据集...")
        try:
            # 如果只有一个数据集，直接使用（避免不必要的合并操作）
            if len(datasets) == 1:
                combined_ds = datasets[0]
            else:
                # 对于大量数据集，使用分批合并策略
                batch_size = 5  # 每批合并5个数据集
                
                if len(datasets) > batch_size:
                    print(f"使用分批合并策略（每批 {batch_size} 个数据集）...")
                    
                    # 先按时间排序数据集
                    def get_first_time(ds):
                        try:
                            return pd.Timestamp(ds.time.values[0])
                        except:
                            if hasattr(ds.time.values[0], 'year'):
                                return datetime(ds.time.values[0].year, 1, 1)
                            return datetime(2000, 1, 1)
                    
                    datasets_sorted = sorted(datasets, key=get_first_time)
                    
                    # 分批合并
                    current_combined = None
                    for i in range(0, len(datasets_sorted), batch_size):
                        batch = datasets_sorted[i:i + batch_size]
                        batch_num = i // batch_size + 1
                        total_batches = (len(datasets_sorted) + batch_size - 1) // batch_size
                        
                        # print(f"  合并批次 {batch_num}/{total_batches} ({len(batch)} 个数据集)...")
                        
                        # 合并当前批次
                        if len(batch) == 1:
                            batch_combined = batch[0]
                        else:
                            try:
                                batch_combined = xr.combine_by_coords(batch, combine_attrs='drop_conflicts')
                                if batch_combined is None or len(batch_combined.time) == 0:
                                    batch_combined = xr.concat(batch, dim='time')
                                    batch_combined = batch_combined.sortby('time')
                            except:
                                batch_combined = xr.concat(batch, dim='time')
                                batch_combined = batch_combined.sortby('time')
                        
                        # 释放批次数据集的内存
                        for ds in batch:
                            try:
                                ds.close()
                            except:
                                pass
                        del batch
                        gc.collect()
                        
                        # 与之前合并的结果合并
                        if current_combined is None:
                            current_combined = batch_combined
                        else:
                            try:
                                current_combined = xr.combine_by_coords(
                                    [current_combined, batch_combined], 
                                    combine_attrs='drop_conflicts'
                                )
                                if current_combined is None or len(current_combined.time) == 0:
                                    current_combined = xr.concat([current_combined, batch_combined], dim='time')
                                    current_combined = current_combined.sortby('time')
                            except:
                                current_combined = xr.concat([current_combined, batch_combined], dim='time')
                                current_combined = current_combined.sortby('time')
                            
                            try:
                                batch_combined.close()
                            except:
                                pass
                            del batch_combined
                            gc.collect()
                    
                    combined_ds = current_combined
                    
                    # 去除重复的时间点（使用节省内存的方法）
                    # print("  去除重复时间点...")
                    try:
                        time_values = combined_ds.time.values
                        if hasattr(time_values[0], 'year'):
                            time_strs = [str(t) for t in time_values]
                            _, unique_indices = np.unique(time_strs, return_index=True)
                        else:
                            _, unique_indices = np.unique(time_values, return_index=True)
                        
                        if len(unique_indices) < len(time_values):
                            # print(f"  发现 {len(time_values) - len(unique_indices)} 个重复时间点")
                            combined_ds = combined_ds.isel(time=unique_indices)
                    except Exception as e:
                        # print(f"  警告: 去重失败，尝试使用drop_duplicates: {e}")
                        try:
                            combined_ds = combined_ds.drop_duplicates(dim='time')
                        except Exception as e2:
                            # print(f"  警告: drop_duplicates也失败: {e2}，跳过去重")
                            pass
                else:
                    # 数据集数量较少，直接合并
                    try:
                        combined_ds = xr.combine_by_coords(datasets, combine_attrs='drop_conflicts')
                        if combined_ds is None or len(combined_ds.time) == 0:
                            combined_ds = xr.concat(datasets, dim='time')
                            combined_ds = combined_ds.sortby('time')
                    except:
                        combined_ds = xr.concat(datasets, dim='time')
                        combined_ds = combined_ds.sortby('time')
                    
                    # 去除重复的时间点（使用节省内存的方法）
                    try:
                        time_values = combined_ds.time.values
                        if hasattr(time_values[0], 'year'):
                            time_strs = [str(t) for t in time_values]
                            _, unique_indices = np.unique(time_strs, return_index=True)
                        else:
                            _, unique_indices = np.unique(time_values, return_index=True)
                        
                        if len(unique_indices) < len(time_values):
                            combined_ds = combined_ds.isel(time=unique_indices)
                    except:
                        try:
                            combined_ds = combined_ds.drop_duplicates(dim='time')
                        except:
                            pass  # 如果去重失败，继续处理
            
            # 检查缺少年份
            time_values = combined_ds.time.values
            missing_years, existing_years = check_missing_years(time_values, start_year, end_year)
            
            if missing_years:
                print(f"  ⚠️  缺少年份: {missing_years}")
                print(f"     已有年份: {sorted(existing_years)}")
            else:
                print(f"  ✓ 所有年份完整 ({start_year}-{end_year})")
            
            # print(f"合并后时间范围: {combined_ds.time.values[0]} 到 {combined_ds.time.values[-1]}")
            # print(f"合并后数据点数: {len(combined_ds.time)}")
            
            # 创建输出目录（按情景分类：SSP126和SSP245分别保存到不同子目录）
            scenario_upper = scenario.upper()  # ssp126 -> SSP126, ssp245 -> SSP245
            output_path = os.path.join(output_dir, model_name, 'future', scenario_upper)
            os.makedirs(output_path, exist_ok=True)
            
            # 生成输出文件名
            output_filename = f"{variable}_day_{model_name}_{scenario}_r1i1p1f1_gn_{start_year}0101-{end_year}1231.nc"
            output_filepath = os.path.join(output_path, output_filename)
            
            # 保存文件（使用压缩和优化选项）
            print(f"  保存到: {output_filepath}")
            encoding = {}
            for var in combined_ds.data_vars:
                encoding[var] = {
                    'zlib': True,
                    'complevel': COMPRESSION_LEVEL,
                    'shuffle': True
                }
            # 对坐标变量也进行压缩
            for coord in combined_ds.coords:
                if coord != 'time':  # time坐标通常不压缩
                    encoding[coord] = {
                        'zlib': True,
                        'complevel': COMPRESSION_LEVEL
                    }
            
            combined_ds.to_netcdf(
                output_filepath,
                encoding=encoding,
                format='NETCDF4'
            )
            print(f"  ✓ {variable} - {scenario.upper()} 数据已保存\n")
            
            # 释放内存
            del combined_ds
            for ds in datasets:
                try:
                    ds.close()
                except:
                    pass
            datasets.clear()
            gc.collect()
            
        except Exception as e:
            print(f"错误: 合并数据失败: {e}")
            import traceback
            traceback.print_exc()
            # 关闭所有数据集
            for ds in datasets:
                try:
                    ds.close()
                except:
                    pass


def main():
    """主函数"""
    print(f"CMIP6数据年份筛选工具 - 模型: {MODEL_NAME}")
    print(f"历史数据: {HISTORICAL_YEARS[0]}-{HISTORICAL_YEARS[1]}, 未来数据: {FUTURE_YEARS[0]}-{FUTURE_YEARS[1]}")
    
    # 检查是否已处理
    if MODEL_NAME in PROCESSED_MODELS:
        print(f"⚠️  {MODEL_NAME} 已在已处理模型列表中，跳过处理")
        return
    
    # 处理历史数据
    process_historical_data(
        model_name=MODEL_NAME,
        input_dir=HISTORICAL_INPUT_DIR,
        output_dir=OUTPUT_BASE_DIR,
        start_year=HISTORICAL_YEARS[0],
        end_year=HISTORICAL_YEARS[1]
    )
    
    # 处理未来数据
    process_future_data(
        model_name=MODEL_NAME,
        input_base_dir=FUTURE_INPUT_DIR,
        output_dir=OUTPUT_BASE_DIR,
        variables=FUTURE_VARIABLES,
        scenarios=FUTURE_SCENARIOS,
        start_year=FUTURE_YEARS[0],
        end_year=FUTURE_YEARS[1]
    )
    
    print("\n" + "="*80)
    print("处理完成!")
    print("="*80)


if __name__ == "__main__":
    main()

