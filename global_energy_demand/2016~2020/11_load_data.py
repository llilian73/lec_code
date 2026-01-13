"""
全球气候数据加载和预处理工具

功能概述：
本工具用于加载和处理全球气候数据，包括人口数据和气象数据。从MERRA2气候数据集中提取2019年的气象变量，并与人口数据结合，为后续的建筑能耗分析提供数据基础。

输入数据：
1. 人口数据：
   - 文件路径：gpw-v4-population-count-adjusted-to-2015-unwpp-country-totals-rev11_2020_30_sec_aligned_to_MERRA2.tif
   - 格式：GeoTIFF栅格文件
   - 分辨率：30秒（约1km）
   - 时间：2020年人口数据

2. 气候数据（MERRA2）：
   - SLV数据：M2T1NXSLV.5.12.4（地表变量）
     * T2M：2米温度 (K)
     * U2M：2米U风分量
     * V2M：2米V风分量
     * QV2M：2米比湿
   - RAD数据：M2T1NXRAD.5.12.4（辐射变量）
     * SWGDN：向下短波辐射
   - 时间范围：2019年全年（每日数据）
   - 格式：NetCDF4文件

3. 数据目录：
   - SLV数据目录：M2T1NXSLV
   - RAD数据目录：M2T1NXRAD
   - 输出目录：global_grid_result/data

主要功能：
1. 人口数据处理：
   - 加载GeoTIFF格式的人口数据
   - 提取有效人口点（人口数>0）
   - 生成经纬度网格坐标
   - 导出人口点数据到CSV格式

2. 气候数据文件管理：
   - 生成2019年全年的气候数据文件路径
   - 检查文件完整性
   - 记录缺失的数据文件
   - 验证SLV和RAD文件的配对

3. 气象数据提取：
   - 为每个有效人口点提取气象数据
   - 使用最近邻插值方法匹配网格点
   - 提取时间序列的气象变量
   - 处理多进程并行计算

4. 数据优化处理：
   - 预计算网格索引，避免重复计算
   - 分批处理数据，控制内存使用
   - 多进程并行处理，提高计算效率
   - 内存使用监控和垃圾回收

输出结果：
1. 人口数据文件：
   - population_points.csv：包含所有有效人口点的经纬度和人口数

2. 气象数据文件：
   - point_lat{lat}_lon{lon}_climate.csv：每个有效人口点的气象数据
   - 包含时间序列的T2M、U2M、V2M、QV2M、SWGDN数据

3. 日志文件：
   - data_loading.log：详细的处理日志
   - failed_climate_points.csv：处理失败的点信息（如果有）

4. 输出目录结构：
   - global_grid_result/data/：所有输出文件

数据流程：
1. 人口数据加载：
   - 读取GeoTIFF文件
   - 提取有效人口点
   - 生成经纬度坐标
   - 保存到CSV文件

2. 气候数据准备：
   - 扫描2019年全年的数据文件
   - 验证文件完整性
   - 生成文件路径列表

3. 气象数据提取：
   - 预计算网格索引
   - 分批处理数据文件
   - 并行提取各点的气象数据
   - 保存到CSV文件

计算特点：
- 高分辨率：30秒分辨率的人口数据
- 长时间序列：2019年全年的每日气象数据
- 并行处理：多进程并行计算，提高效率
- 内存优化：分批处理，控制内存使用
- 错误处理：完善的异常处理和日志记录

技术特色：
- 网格索引预计算：避免重复的网格匹配计算
- 分批处理策略：将大量数据分批处理，避免内存溢出
- 多进程并行：充分利用多核CPU资源
- 内存监控：实时监控内存使用情况
- 进度跟踪：使用tqdm显示处理进度

性能优化：
- 使用xarray进行高效的NetCDF文件读取
- 向量化操作进行网格索引计算
- 多进程并行处理文件和数据点
- 分批处理控制内存使用
- 及时进行垃圾回收
"""

import pandas as pd
import numpy as np
import xarray as xr
import os
import sys
from pathlib import Path
import rasterio
from datetime import datetime
import multiprocessing
from tqdm import tqdm
import logging
import gc
from functools import partial
import psutil

# 设置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('data_loading.log', encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])

# 配置参数
BASE_WEATHER_DIR = r"Z:\local_environment_creation\energy_consumption_gird\weather"
POPULATION_FILE = r"Z:\local_environment_creation\Population\gpw-v4-population-count-adjusted-to-2015-unwpp-country-totals-rev11_2020_30_sec_tif\gpw_v4_population_count_adjusted_to_2015_unwpp_country_totals_rev11_2020_30_sec_aligned_to_MERRA2.tif"
OUTPUT_BASE_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\data"
YEARS = [2016, 2017, 2018, 2019, 2020]

# 确保输出目录存在
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)


def log_memory_usage():
    """记录内存使用情况"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    logging.info(f"内存使用: {memory_info.rss / 1024 / 1024:.1f} MB ({memory_percent:.1f}%)")


def load_population_data():
    """加载人口数据并导出到CSV"""
    logging.info("开始加载人口数据...")
    log_memory_usage()

    with rasterio.open(POPULATION_FILE) as src:
        population_data = src.read(1)  # 读取第一个波段
        transform = src.transform
        crs = src.crs

    # 创建经纬度网格
    height, width = population_data.shape
    logging.info(f"人口数据形状: {height} x {width}")

    # 创建网格索引
    rows, cols = np.meshgrid(range(height), range(width), indexing='ij')

    # 使用rasterio.transform.xy获取每个像素中心的经纬度坐标
    lons, lats = rasterio.transform.xy(transform, rows.flatten(), cols.flatten(), offset='center')
    lons = np.array(lons).reshape(height, width)
    lats = np.array(lats).reshape(height, width)

    logging.info(f"人口数据加载完成，形状: {population_data.shape}")
    logging.info(f"经纬度网格形状: {lons.shape}")
    logging.info(f"纬度范围: {lats.min():.4f} 到 {lats.max():.4f}")
    logging.info(f"经度范围: {lons.min():.4f} 到 {lons.max():.4f}")

    # 筛选有效人口点（人口数不为0的点）
    valid_mask = population_data > 0
    valid_lats = lats[valid_mask]
    valid_lons = lons[valid_mask]
    valid_populations = population_data[valid_mask]

    # 创建DataFrame，确保都是3位小数格式
    population_df = pd.DataFrame({
        'lat': [f"{lat:.3f}" for lat in valid_lats],
        'lon': [f"{lon:.3f}" for lon in valid_lons],
        'population': valid_populations
    })

    logging.info(f"找到 {len(population_df)} 个有效人口点")

    # 保存到CSV
    population_csv_path = os.path.join(OUTPUT_BASE_DIR, 'population_points.csv')
    population_df.to_csv(population_csv_path, index=False, encoding='utf-8-sig')
    logging.info(f"人口数据已保存至: {population_csv_path}")
    log_memory_usage()

    return population_df


def generate_climate_filepaths(year):
    """生成指定年份的气候数据文件路径"""
    logging.info(f"生成{year}年气候数据文件路径...")

    # 构建年份路径
    year_dir = os.path.join(BASE_WEATHER_DIR, str(year))
    slv_dir = os.path.join(year_dir, 'M2T1NXSLV')
    rad_dir = os.path.join(year_dir, 'M2T1NXRAD')

    slv_files = []
    rad_files = []
    missing_dates = []

    for dt in pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D'):
        date_str = dt.strftime('%Y%m%d')
        slv_fname = f"M2T1NXSLV.5.12.4%3AMERRA2_400.tavg1_2d_slv_Nx.{date_str}.nc4.dap.nc4@dap4.ce=%2FQV2M;%2FT2M;%2FU2M;%2FV2M;%2Ftime;%2Flat;%2Flon"
        rad_fname = f"M2T1NXRAD.5.12.4%3AMERRA2_400.tavg1_2d_rad_Nx.{date_str}.nc4.dap.nc4@dap4.ce=%2FSWGDN;%2Ftime;%2Flat;%2Flon"

        slv_path = os.path.join(slv_dir, slv_fname)
        rad_path = os.path.join(rad_dir, rad_fname)

        if os.path.exists(slv_path) and os.path.exists(rad_path):
            slv_files.append(slv_path)
            rad_files.append(rad_path)
        else:
            missing_dates.append(date_str)
            if not os.path.exists(slv_path):
                logging.warning(f"SLV文件不存在: {slv_path}")
            if not os.path.exists(rad_path):
                logging.warning(f"RAD文件不存在: {rad_path}")

    logging.info(f"找到 {len(slv_files)} 个{year}年气候数据文件对")
    if missing_dates:
        logging.warning(f"{year}年缺失的数据日期: {missing_dates}")
        logging.warning(f"{year}年总共缺失 {len(missing_dates)} 天的数据")
    else:
        logging.info(f"{year}年所有气候数据文件都存在")

    return slv_files, rad_files


def extract_climate_data_for_point(point_data, slv_files, rad_files, output_dir):
    """为单个点提取气象数据"""
    try:
        lat, lon, population = point_data['lat'], point_data['lon'], point_data['population']
        
        # 如果lat和lon是字符串，转换为浮点数
        if isinstance(lat, str):
            lat = float(lat)
        if isinstance(lon, str):
            lon = float(lon)

        # 创建结果字典
        climate_data = {
            'time': [],
            'T2M': [],  # 2米温度 (K)
            'U2M': [],  # 2米U风分量
            'V2M': [],  # 2米V风分量
            'QV2M': [],  # 2米比湿
            'SWGDN': []  # 向下短波辐射
        }

        # 逐个加载文件并提取数据
        for slv_file, rad_file in zip(slv_files, rad_files):
            try:
                # 加载SLV数据
                slv_ds = xr.open_dataset(slv_file, engine='netcdf4')

                # 找到最近的网格点
                lat_idx = np.abs(slv_ds.lat.values - lat).argmin()
                lon_idx = np.abs(slv_ds.lon.values - lon).argmin()

                # 提取该点的SLV数据
                point_slv = slv_ds.isel(lat=lat_idx, lon=lon_idx)

                # 加载RAD数据
                rad_ds = xr.open_dataset(rad_file, engine='netcdf4')

                # 提取该点的RAD数据
                point_rad = rad_ds.isel(lat=lat_idx, lon=lon_idx)

                # 收集数据
                for t_idx in range(len(point_slv.time)):
                    climate_data['time'].append(point_slv.time.values[t_idx])
                    climate_data['T2M'].append(float(point_slv.T2M.values[t_idx]))
                    climate_data['U2M'].append(float(point_slv.U2M.values[t_idx]))
                    climate_data['V2M'].append(float(point_slv.V2M.values[t_idx]))
                    climate_data['QV2M'].append(float(point_slv.QV2M.values[t_idx]))
                    climate_data['SWGDN'].append(float(point_rad.SWGDN.values[t_idx]))

                # 关闭数据集
                slv_ds.close()
                rad_ds.close()

            except Exception as e:
                logging.warning(f"处理文件时出错 (lat={lat}, lon={lon}): {e}")
                continue

        # 创建DataFrame
        df = pd.DataFrame(climate_data)

        # 生成文件名
        filename = f"point_lat{lat:.3f}_lon{lon:.3f}_climate.csv"
        output_path = os.path.join(output_dir, filename)

        # 保存到CSV
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        return f"成功处理坐标点: ({lat}, {lon})"

    except Exception as e:
        return f"处理点 ({lat}, {lon}) 失败: {str(e)}"


def precompute_grid_indices(population_df, sample_slv_file):
    """预计算所有点的网格索引，避免重复计算"""
    logging.info("预计算网格索引...")

    # 加载一个样本文件获取网格信息
    with xr.open_dataset(sample_slv_file, engine='netcdf4') as sample_ds:
        lats = sample_ds.lat.values
        lons = sample_ds.lon.values

    # 预计算所有点的网格索引
    grid_indices = {}
    for _, point in population_df.iterrows():
        lat, lon = point['lat'], point['lon']
        
        # 如果lat和lon是字符串，转换为浮点数
        if isinstance(lat, str):
            lat = float(lat)
        if isinstance(lon, str):
            lon = float(lon)
            
        point_key = f"{lat:.3f}_{lon:.3f}"

        # 使用向量化操作找到最近的网格点
        lat_idx = np.abs(lats - lat).argmin()
        lon_idx = np.abs(lons - lon).argmin()

        grid_indices[point_key] = (lat_idx, lon_idx)

    logging.info(f"预计算完成，共 {len(grid_indices)} 个点的索引")
    return grid_indices, lats, lons


def process_single_file(args):
    """处理单个文件的函数 - 需要在模块级别定义以便pickle"""
    file_pair, population_df, grid_indices = args
    slv_file, rad_file = file_pair
    file_data = {}

    try:
        # 加载SLV数据
        slv_ds = xr.open_dataset(slv_file, engine='netcdf4')
        rad_ds = xr.open_dataset(rad_file, engine='netcdf4')

        # 为每个点提取数据
        for _, point in population_df.iterrows():
            lat, lon = point['lat'], point['lon']
            
            # 如果lat和lon是字符串，转换为浮点数
            if isinstance(lat, str):
                lat = float(lat)
            if isinstance(lon, str):
                lon = float(lon)
                
            point_key = f"{lat:.3f}_{lon:.3f}"

            # 使用预计算的索引
            lat_idx, lon_idx = grid_indices[point_key]

            # 提取该点的数据
            point_slv = slv_ds.isel(lat=lat_idx, lon=lon_idx)
            point_rad = rad_ds.isel(lat=lat_idx, lon=lon_idx)

            # 收集数据
            file_data[point_key] = {
                'time': point_slv.time.values.tolist(),
                'T2M': point_slv.T2M.values.tolist(),
                'U2M': point_slv.U2M.values.tolist(),
                'V2M': point_slv.V2M.values.tolist(),
                'QV2M': point_slv.QV2M.values.tolist(),
                'SWGDN': point_rad.SWGDN.values.tolist()
            }

        # 关闭数据集
        slv_ds.close()
        rad_ds.close()

        return file_data

    except Exception as e:
        logging.warning(f"处理文件时出错: {e}")
        return {}


def load_climate_batch_optimized(slv_files_batch, rad_files_batch, population_df, grid_indices):
    """优化版本：加载一批气候数据文件并提取所有点的数据"""
    logging.info(f"开始加载 {len(slv_files_batch)} 个气候数据文件...")
    log_memory_usage()

    # 创建存储所有点数据的字典
    all_points_data = {}

    # 初始化每个点的数据结构
    for _, point in population_df.iterrows():
        lat, lon = point['lat'], point['lon']
        
        # 如果lat和lon是字符串，转换为浮点数
        if isinstance(lat, str):
            lat = float(lat)
        if isinstance(lon, str):
            lon = float(lon)
            
        point_key = f"{lat:.3f}_{lon:.3f}"
        all_points_data[point_key] = {
            'lat': lat,
            'lon': lon,
            'time': [],
            'T2M': [],
            'U2M': [],
            'V2M': [],
            'QV2M': [],
            'SWGDN': []
        }

    # 使用多进程并行处理文件
    num_cores = multiprocessing.cpu_count()
    num_processes = min(num_cores * 2, 24)  # 减少进程数，降低内存压力

    logging.info(f"使用 {num_processes} 个进程并行处理文件...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        file_pairs = list(zip(slv_files_batch, rad_files_batch))

        # 准备参数
        args_list = [(file_pair, population_df, grid_indices) for file_pair in file_pairs]

        # 并行处理所有文件
        results = list(tqdm(
            pool.imap(process_single_file, args_list),
            total=len(args_list),
            desc="并行加载气候文件"
        ))

    # 合并所有文件的数据
    logging.info("合并文件数据...")
    for file_data in results:
        for point_key, data in file_data.items():
            all_points_data[point_key]['time'].extend(data['time'])
            all_points_data[point_key]['T2M'].extend(data['T2M'])
            all_points_data[point_key]['U2M'].extend(data['U2M'])
            all_points_data[point_key]['V2M'].extend(data['V2M'])
            all_points_data[point_key]['QV2M'].extend(data['QV2M'])
            all_points_data[point_key]['SWGDN'].extend(data['SWGDN'])

    # 清理中间结果
    del results
    del file_pairs
    del args_list
    gc.collect()

    logging.info(f"完成加载 {len(slv_files_batch)} 个气候数据文件")
    log_memory_usage()

    return all_points_data


def load_climate_batch(slv_files_batch, rad_files_batch, population_df, grid_indices):
    """加载一批气候数据文件并提取所有点的数据"""
    # 使用优化版本加载数据
    return load_climate_batch_optimized(slv_files_batch, rad_files_batch, population_df, grid_indices)


def write_point_data_batch(points_data_batch, output_dir):
    """并行写入一批点的数据到CSV文件"""
    try:
        results = []

        for point_key, point_data in points_data_batch.items():
            try:
                lat, lon = point_data['lat'], point_data['lon']

                # 创建DataFrame
                df = pd.DataFrame({
                    'time': point_data['time'],
                    'T2M': point_data['T2M'],
                    'U2M': point_data['U2M'],
                    'V2M': point_data['V2M'],
                    'QV2M': point_data['QV2M'],
                    'SWGDN': point_data['SWGDN']
                })

                # 生成文件名
                filename = f"point_lat{lat:.3f}_lon{lon:.3f}_climate.csv"
                output_path = os.path.join(output_dir, filename)

                # 使用二进制模式写入CSV（更快的写入速度）
                df.to_csv(output_path, index=False, encoding='utf-8-sig', mode='w')

                results.append(f"成功处理坐标点: ({lat}, {lon})")

            except Exception as e:
                results.append(f"处理点 ({lat}, {lon}) 失败: {str(e)}")

        return results

    except Exception as e:
        return [f"批处理失败: {str(e)}"]


def append_point_data_batch(points_data_batch, output_dir):
    """并行追加一批点的数据到现有CSV文件"""
    try:
        results = []

        for point_key, point_data in points_data_batch.items():
            try:
                lat, lon = point_data['lat'], point_data['lon']

                # 创建DataFrame
                df = pd.DataFrame({
                    'time': point_data['time'],
                    'T2M': point_data['T2M'],
                    'U2M': point_data['U2M'],
                    'V2M': point_data['V2M'],
                    'QV2M': point_data['QV2M'],
                    'SWGDN': point_data['SWGDN']
                })

                # 生成文件名
                filename = f"point_lat{lat:.3f}_lon{lon:.3f}_climate.csv"
                output_path = os.path.join(output_dir, filename)

                # 检查文件是否存在，决定是否写入表头
                file_exists = os.path.exists(output_path)

                # 追加写入CSV（不包含表头）
                df.to_csv(output_path, index=False, encoding='utf-8-sig',
                          mode='a', header=not file_exists)

                results.append(f"成功追加坐标点: ({lat}, {lon})")

            except Exception as e:
                results.append(f"追加点 ({lat}, {lon}) 失败: {str(e)}")

        return results

    except Exception as e:
        return [f"批处理失败: {str(e)}"]


def process_climate_batch(batch_data, slv_files, rad_files, output_dir):
    """处理一批点的气象数据"""
    try:
        batch_results = []

        for _, point_data in batch_data.iterrows():
            result = extract_climate_data_for_point(point_data, slv_files, rad_files, output_dir)
            batch_results.append(result)

        return batch_results
    except Exception as e:
        return [f"批处理失败: {str(e)}"]


def extract_climate_data_for_all_points(population_df, slv_files, rad_files, year, output_dir):
    """为所有有效人口点提取气象数据 - 超优化版本"""
    logging.info(f"开始为所有有效人口点提取{year}年气象数据...")
    log_memory_usage()

    # 配置参数 - 进一步减少内存使用
    BATCH_SIZE_DAYS = 25  # 减少到25天，避免内存溢出
    num_cores = multiprocessing.cpu_count()
    num_processes = min(num_cores * 2, 32)  # 减少进程数，降低内存压力

    logging.info(f"使用 {num_processes} 个进程进行并行处理")
    logging.info(f"每次加载 {BATCH_SIZE_DAYS} 天的数据（减少内存使用）")

    # 将文件分批
    total_files = len(slv_files)
    file_batches = []
    for i in range(0, total_files, BATCH_SIZE_DAYS):
        end_idx = min(i + BATCH_SIZE_DAYS, total_files)
        slv_batch = slv_files[i:end_idx]
        rad_batch = rad_files[i:end_idx]
        file_batches.append((slv_batch, rad_batch))

    logging.info(f"将 {total_files} 个文件分为 {len(file_batches)} 批处理")

    # 预计算网格索引（只需要计算一次）
    logging.info("=== 预计算网格索引（一次性计算） ===")
    grid_indices, _, _ = precompute_grid_indices(population_df, slv_files[0])

    # 分批处理
    for batch_idx, (slv_batch, rad_batch) in enumerate(file_batches):
        logging.info(f"=== 处理第 {batch_idx + 1}/{len(file_batches)} 批数据 ===")

        # 加载这一批的气候数据
        all_points_data = load_climate_batch(slv_batch, rad_batch, population_df, grid_indices)

        # 将点数据分批进行并行写入
        total_points = len(all_points_data)
        point_batch_size = max(1, min(100, total_points // (num_processes * 4)))  # 减少批次大小，降低内存压力
        point_batches = []

        # 将点数据分成批次
        point_keys = list(all_points_data.keys())
        for i in range(0, total_points, point_batch_size):
            end_idx = min(i + point_batch_size, total_points)
            batch_keys = point_keys[i:end_idx]
            batch_data = {key: all_points_data[key] for key in batch_keys}
            point_batches.append(batch_data)

        logging.info(f"将 {total_points} 个点分为 {len(point_batches)} 批进行写入")

        # 并行写入数据
        results = []
        failed_points = []

        with multiprocessing.Pool(processes=num_processes) as pool:
            # 第一批数据直接写入，后续批次追加写入
            if batch_idx == 0:
                write_func = partial(write_point_data_batch, output_dir=output_dir)
            else:
                write_func = partial(append_point_data_batch, output_dir=output_dir)

            chunksize = max(1, len(point_batches) // (num_processes * 4))  # 进一步减少chunksize

            with tqdm(total=len(point_batches), desc=f"写入第{batch_idx + 1}批数据") as pbar:
                for batch_results in pool.imap_unordered(write_func, point_batches, chunksize=chunksize):
                    if isinstance(batch_results, list):
                        for result in batch_results:
                            if "失败" in result:
                                failed_points.append(result)
                            else:
                                results.append(result)
                    pbar.update(1)

        # 输出进度
        successful_points = len([r for r in results if "成功" in r])
        logging.info(f"第 {batch_idx + 1} 批处理完成:")
        logging.info(f"- 成功处理: {successful_points} 个点")
        logging.info(f"- 失败点数: {len(failed_points)} 个点")

        # 清理内存
        del all_points_data
        del point_batches
        del results
        del failed_points
        gc.collect()
        log_memory_usage()

        # 保存失败点信息（如果有的话）
        try:
            if 'failed_points' in locals() and failed_points:
                failed_points_path = os.path.join(output_dir, f'failed_climate_points_{year}.csv')
                failed_df = pd.DataFrame({'error_message': failed_points})
                failed_df.to_csv(failed_points_path, index=False, encoding='utf-8-sig')
                logging.info(f"失败点信息已保存至: {failed_points_path}")
            else:
                logging.info("没有失败的点，无需保存失败点信息")
        except Exception as e:
            logging.warning(f"保存失败点信息时出错: {e}")

    logging.info(f"{year}年所有气候数据处理完成！")
    log_memory_usage()


def main():
    """主函数"""
    logging.info("开始数据加载和预处理...")

    try:
        # 第一步：加载人口数据并导出到CSV
        logging.info("=== 第一步：处理人口数据 ===")
        population_df = load_population_data()

        # 第二步：处理每个年份的数据
        for year in YEARS:
            logging.info(f"=== 处理{year}年数据 ===")
            
            # 创建年份输出目录
            year_output_dir = os.path.join(OUTPUT_BASE_DIR, str(year))
            os.makedirs(year_output_dir, exist_ok=True)
            
            # 生成气候数据文件路径
            logging.info(f"=== 生成{year}年气候数据文件路径 ===")
            slv_files, rad_files = generate_climate_filepaths(year)
            
            if len(slv_files) == 0:
                logging.warning(f"{year}年没有找到有效的气候数据文件，跳过")
                continue
            
            # 为所有有效人口点提取气象数据
            logging.info(f"=== 提取{year}年气象数据 ===")
            extract_climate_data_for_all_points(population_df, slv_files, rad_files, year, year_output_dir)

        logging.info("所有年份数据加载和预处理完成！")
        logging.info(f"所有数据已保存至: {OUTPUT_BASE_DIR}")

    except Exception as e:
        error_msg = f"数据加载过程中出现错误: {str(e)}"
        logging.error(error_msg)
        raise


if __name__ == "__main__":
    main()
