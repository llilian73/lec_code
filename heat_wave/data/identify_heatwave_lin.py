"""
热浪识别工具 - 直接从NetCDF文件处理

功能概述：
本工具用于识别全球各地的热浪事件，直接从NetCDF文件读取数据，无需中间CSV转换。
使用历史数据（1981-2010）计算阈值，识别未来年份（2030-2034）的热浪事件。

输入数据：
1. 历史数据：
   - 路径：{model_path}/historical/
   - 文件：tasmax_day_{model}_historical_r1i1p1f1_gn_19810101-20101231_interpolated_1deg.nc
   - 时间范围：1981-2010年
   - 来源：linear_interpolation_lin.py的输出文件

2. 未来数据：
   - 路径：{model_path}/future/{ssp}/
   - 文件：tasmax_day_{model}_{ssp}_r1i1p1f1_gn_20300101-20341231_interpolated_1deg.nc
   - 时间范围：2030-2034年
   - SSP路径：SSP126, SSP245
   - 来源：linear_interpolation_lin.py的输出文件

3. 陆地边界数据：
   - 文件路径：/home/linbor/WORK/lishiying/shapefiles/world_border2.shp
   - 经度范围：-180~180度
   - 不包含南极洲

主要功能：
1. 陆地点识别：
   - 从NetCDF文件读取经纬度网格
   - 与shp文件进行空间匹配
   - 筛选出陆地点

2. 阈值计算：
   - 使用历史数据（1981-2010）计算每日阈值
   - 使用31天滑动窗口和90%分位数

3. 热浪识别：
   - 识别连续3天以上超过阈值的事件
   - 处理2030-2034年每年数据

4. 并行处理：
   - 分批处理陆地点
   - 多进程并行计算

输出结果：
1. CSV格式的热浪事件：
   - 路径：/home/linbor/WORK/lishiying/heat_wave/{model}/{ssp}/{year}_all_heat_wave.csv
   - 包含列：lat, lon, number, start_day, date, Duration
   - 每年一个文件

技术参数：
- 经度范围：-180~180度（无需转换）
- 阈值窗口：31天
- 分位数：90%
- 最小热浪持续时间：3天
- 并行进程数：CPU核心数×2，最大32个
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import multiprocessing
from multiprocessing import shared_memory
import gc
import logging
import warnings
from datetime import datetime
import time
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置路径（Linux路径）
BASE_PATH = "/home/linbor/WORK/lishiying"
SHAPEFILE_PATH = os.path.join(BASE_PATH, "shapefiles/world_border2.shp")

# 模型配置
MODELS = [
    "ACCESS-ESM1-5",
    "BCC-CSM2-MR",
    "CanESM5",
    "EC-Earth3",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0"
]

# 已处理的模型（跳过处理）
PROCESSED_MODELS = ["BCC-CSM2-MR", "ACCESS-ESM1-5","EC-Earth3","MPI-ESM1-2-HR","MRI-ESM2-0"]

# SSP配置
SSP_PATHS = ["SSP126", "SSP245"]
TARGET_YEARS = [2030, 2031, 2032, 2033, 2034]
HISTORICAL_START_YEAR = 1981
HISTORICAL_END_YEAR = 2010


def identify_land_points(nc_file, world_data):
    """
    识别NetCDF文件中的陆地点（使用批量空间连接，更高效且健壮）
    
    参数:
        nc_file: NetCDF文件路径
        world_data: GeoDataFrame，包含陆地边界数据
    
    返回:
        land_points: 陆地点列表，格式为[(lat, lon), ...]
    """
    logger.info(f"正在识别陆地点: {nc_file}")
    land_points = []
    
    try:
        with xr.open_dataset(nc_file) as ds:
            lats = ds.lat.values
            lons = ds.lon.values
            
            # 创建所有点的GeoDataFrame（批量处理）
            points_data = []
            for lat in lats:
                for lon in lons:
                    points_data.append({
                        'lat': lat,
                        'lon': lon,
                        'geometry': Point(lon, lat)
                    })
            
            points_gdf = gpd.GeoDataFrame(points_data, crs="EPSG:4326")
            
            # 确保坐标系匹配
            if world_data.crs is None:
                world_data.set_crs(epsg=4326, inplace=True)
            elif world_data.crs != points_gdf.crs:
                world_data = world_data.to_crs(points_gdf.crs)
            
            logger.info(f"开始批量空间连接，共 {len(points_gdf)} 个点")
            
            # 使用空间连接（sjoin）批量匹配，更高效且避免拓扑错误
            # 使用within连接，找到包含每个点的陆地
            try:
                joined_points = gpd.sjoin(
                    points_gdf,
                    world_data[['geometry']],
                    how="left",
                    predicate="within"
                )
                
                # 提取匹配到陆地的点
                land_mask = joined_points['index_right'].notna()
                land_points_gdf = points_gdf[land_mask]
                
                # 转换为列表格式
                land_points = [(row['lat'], row['lon']) for _, row in land_points_gdf.iterrows()]
                
            except Exception as sjoin_error:
                logger.warning(f"批量空间连接失败，回退到逐个点检查: {str(sjoin_error)}")
                # 回退到逐个点检查，但使用更健壮的方法
                total_points = len(points_gdf)
                with tqdm(total=total_points, desc="识别陆地点（逐个检查）") as pbar:
                    for _, point_row in points_gdf.iterrows():
                        pbar.update(1)
                        point_geom = point_row['geometry']
                        lat = point_row['lat']
                        lon = point_row['lon']
                        
                        # 使用within方法，对拓扑错误更健壮
                        is_land = False
                        try:
                            for _, world_row in world_data.iterrows():
                                try:
                                    # 尝试使用within方法（更健壮）
                                    if world_row['geometry'].contains(point_geom):
                                        is_land = True
                                        break
                                except Exception:
                                    # 如果contains失败，尝试intersects
                                    try:
                                        if world_row['geometry'].intersects(point_geom):
                                            is_land = True
                                            break
                                    except Exception:
                                        # 跳过无效的几何图形
                                        continue
                        except Exception as e:
                            # 跳过有问题的点
                            logger.debug(f"跳过点 ({lat}, {lon}): {str(e)}")
                            continue
                        
                        if is_land:
                            land_points.append((lat, lon))
        
        logger.info(f"识别完成，共找到 {len(land_points)} 个陆地点")
        return land_points
    
    except Exception as e:
        logger.error(f"识别陆地点时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def day_of_year_to_date(day_of_year, year):
    """将一年中的第几天转换为日期字符串"""
    date = datetime(year, 1, 1) + pd.Timedelta(days=int(day_of_year) - 1)
    return f"{date.month}/{date.day}"


def calculate_threshold(historical_data, win_size=31, quantile=0.90):
    """
    计算参考期的阈值
    
    参数:
        historical_data: xarray DataArray，包含历史温度数据和时间坐标
        win_size: 滑动窗口大小（天）
        quantile: 分位数
    
    返回:
        threshold: xarray DataArray，包含每个dayofyear的阈值
    """
    # 计算每个dayofyear的阈值
    threshold_list = []
    for day in range(1, 367):
        plusminus = win_size // 2
        valid_days = (np.arange(day - plusminus - 1, day + plusminus) % 366) + 1
        window = historical_data.time.dt.dayofyear.isin(valid_days)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')
            threshold = historical_data.where(window, drop=True).quantile(quantile, dim='time', skipna=False)
        threshold_list.append(threshold)
    
    threshold = xr.concat(threshold_list, dim='dayofyear')
    threshold = threshold.assign_coords(dayofyear=range(1, 367))
    return threshold


def identify_heatwaves(temperature_data, threshold, time_coords, n_days=3):
    """
    识别热浪事件
    
    参数:
        temperature_data: numpy数组，温度数据
        threshold: xarray DataArray，阈值数据
        time_coords: pandas DatetimeIndex，时间坐标
        n_days: 最小热浪持续时间（天）
    
    返回:
        heatwave_events: 热浪事件列表，每个事件包含start_day和duration
    """
    # 将温度数据转换为DataArray
    da = xr.DataArray(
        temperature_data,
        coords={'time': time_coords},
        dims=['time']
    )
    
    # 计算每天是否超过阈值
    is_hot = da.groupby('time.dayofyear') > threshold
    
    # 识别连续的热浪事件
    heatwave_events = []
    current_event = None
    
    for i in range(len(da)):
        if is_hot[i]:
            if current_event is None:
                current_event = {'start': i, 'count': 1}
            else:
                current_event['count'] += 1
        else:
            if current_event is not None and current_event['count'] >= n_days:
                start_day = int(da.time.dt.dayofyear[i - current_event['count']].values)
                heatwave_events.append({
                    'start_day': start_day,
                    'duration': current_event['count']
                })
            current_event = None
    
    # 检查最后一个事件
    if current_event is not None and current_event['count'] >= n_days:
        start_day = int(da.time.dt.dayofyear[len(da) - current_event['count']].values)
        heatwave_events.append({
            'start_day': start_day,
            'duration': current_event['count']
        })
    
    return heatwave_events


# 全局变量用于存储数据（多进程使用）
_global_historical_file = None
_global_future_files = None
_global_historical_shm = None
_global_historical_shape = None
_global_historical_dtype = None
_global_historical_time_index = None
_global_historical_lat_index = None
_global_historical_lon_index = None
_global_future_file_cache = {}  # 缓存已打开的文件
_global_future_data_cache = {}  # 缓存已加载的年份数据


def init_worker(historical_file, future_files, shm_name, data_shape, data_dtype, 
                time_index, lat_index, lon_index):
    """初始化工作进程，从共享内存读取历史数据（不加载未来数据，按需加载）"""
    global _global_historical_file, _global_future_files
    global _global_historical_shm, _global_historical_shape, _global_historical_dtype
    global _global_historical_time_index, _global_historical_lat_index, _global_historical_lon_index
    global _global_future_file_cache, _global_future_data_cache
    
    _global_historical_file = historical_file
    _global_future_files = future_files
    
    # 从共享内存读取历史数据
    try:
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        _global_historical_shm = existing_shm
        _global_historical_shape = data_shape
        _global_historical_dtype = data_dtype
        _global_historical_time_index = time_index
        _global_historical_lat_index = lat_index
        _global_historical_lon_index = lon_index
    except Exception as e:
        logger.error(f"工作进程：无法访问共享内存: {str(e)}")
        raise
    
    # 不在这里加载未来数据，而是按需加载
    _global_future_file_cache = {}
    _global_future_data_cache = {}
    
    logger.info("工作进程：初始化完成（未来数据将按需加载）")


def load_year_data(year):
    """按需加载指定年份的数据（带缓存）"""
    global _global_future_files, _global_future_file_cache, _global_future_data_cache
    
    # 如果已经加载过，直接返回
    if year in _global_future_data_cache:
        return _global_future_data_cache[year]
    
    # 获取该年份对应的文件路径
    if year not in _global_future_files:
        logger.warning(f"未找到 {year} 年的文件路径")
        return None
    
    file_path = _global_future_files[year]
    
    try:
        # 如果文件还没有打开，先打开（使用缓存，使用chunks延迟加载）
        if file_path not in _global_future_file_cache:
            # 未来数据是5年一个文件，使用chunks延迟加载以提高性能
            # 指定time维度的chunk大小（366天，覆盖闰年），避免auto rechunking对object dtype的错误
            _global_future_file_cache[file_path] = xr.open_dataset(file_path, chunks={'time': 366})
        
        ds = _global_future_file_cache[file_path]
        
        # 处理cftime时间对象，转换为pandas DatetimeIndex
        # 使用decode_cf来正确处理cftime对象（特别是使用chunks时）
        try:
            # 先尝试decode_cf（这会处理cftime对象）
            # 注意：对于chunked数据，需要先load时间坐标
            time_coords = ds.time.load()  # 加载时间坐标到内存
            ds_decoded = xr.decode_cf(ds, decode_times=True)
            time_values = ds_decoded.time.values
            
            # 转换为pandas DatetimeIndex
            if hasattr(time_values, 'to_pandas'):
                time_index = time_values.to_pandas()
            else:
                time_index = pd.to_datetime(time_values)
            
            # 确保是DatetimeIndex
            if not isinstance(time_index, pd.DatetimeIndex):
                time_index = pd.DatetimeIndex(time_index)
                
        except Exception as e:
            # 如果decode_cf失败，尝试手动处理cftime对象
            try:
                import cftime
                time_values = ds.time.values
                
                # 检查是否是cftime对象
                if len(time_values) > 0 and isinstance(time_values[0], cftime.datetime):
                    # 将cftime对象转换为pandas Timestamp
                    time_index = pd.to_datetime([pd.Timestamp(t.year, t.month, t.day) for t in time_values])
                else:
                    # 尝试直接转换
                    time_index = pd.to_datetime(time_values)
                
                # 确保是DatetimeIndex
                if not isinstance(time_index, pd.DatetimeIndex):
                    time_index = pd.DatetimeIndex(time_index)
                    
            except Exception as e2:
                # 最后尝试：使用to_pandas
                try:
                    time_index = ds.time.to_pandas()
                    if not isinstance(time_index, pd.DatetimeIndex):
                        time_index = pd.DatetimeIndex(time_index)
                except Exception:
                    logger.error(f"无法转换时间索引: {e}, {e2}")
                    raise ValueError(f"无法转换时间索引: {e}, {e2}")
        
        # 提取该年份的数据
        year_mask = time_index.year == year
        if year_mask.sum() == 0:
            logger.warning(f"文件中不包含 {year} 年的数据")
            return None
        
        # 只加载该年份的数据
        year_data = ds['tasmax'].isel(time=year_mask).load()
        
        # 缓存该年份的数据
        _global_future_data_cache[year] = {
            'time_index': time_index[year_mask],
            'data': year_data.values,
            'lat_index': ds.lat.values,
            'lon_index': ds.lon.values
        }
        
        return _global_future_data_cache[year]
    
    except Exception as e:
        logger.error(f"加载 {year} 年数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def get_nearest_point(lat, lon, lat_index, lon_index):
    """获取最近的网格点索引"""
    lat_idx = np.abs(lat_index - lat).argmin()
    lon_idx = np.abs(lon_index - lon).argmin()
    return lat_idx, lon_idx


def process_single_point(point_info):
    """
    处理单个网格点的热浪识别（按年加载数据）
    
    参数:
        point_info: 字典，包含lat, lon
    
    返回:
        results: 列表，包含所有年份的热浪事件结果
    """
    global _global_historical_shm, _global_historical_shape, _global_historical_dtype
    global _global_historical_time_index, _global_historical_lat_index, _global_historical_lon_index
    global _global_future_files
    
    lat = point_info['lat']
    lon = point_info['lon']
    
    try:
        # 从共享内存读取历史数据
        hist_array = np.ndarray(_global_historical_shape, dtype=_global_historical_dtype, 
                               buffer=_global_historical_shm.buf)
        
        lat_idx, lon_idx = get_nearest_point(lat, lon, _global_historical_lat_index, 
                                            _global_historical_lon_index)
        historical_temps = hist_array[:, lat_idx, lon_idx]
        historical_times = _global_historical_time_index
        
        # 转换为DataArray用于计算阈值
        historical_da = xr.DataArray(
            historical_temps,
            coords={'time': historical_times},
            dims=['time']
        )
        
        # 计算阈值
        threshold = calculate_threshold(historical_da)
        
        # 处理每个目标年份（按年加载）
        all_results = []
        for year in sorted(_global_future_files.keys()):  # 按年份排序，确保顺序
            try:
                # 按需加载该年份的数据
                future_data = load_year_data(year)
                if future_data is None:
                    logger.debug(f"点 ({lat}, {lon}) 的 {year} 年数据加载失败，跳过")
                    continue
                
                # 获取未来数据
                lat_idx, lon_idx = get_nearest_point(lat, lon, future_data['lat_index'], future_data['lon_index'])
                future_temps = future_data['data'][:, lat_idx, lon_idx]
                future_times = future_data['time_index']
                
                # 识别热浪事件
                heatwave_events = identify_heatwaves(future_temps, threshold, future_times)
                
                # 生成结果
                for i, event in enumerate(heatwave_events, 1):
                    start_day = event['start_day']
                    duration = event['duration']
                    start_date = day_of_year_to_date(start_day, year)
                    all_results.append({
                        'year': year,
                        'lat': lat,
                        'lon': lon,
                        'number': i,
                        'start_day': start_day,
                        'date': start_date,
                        'Duration': duration
                    })
            except Exception as year_error:
                logger.error(f"处理点 ({lat}, {lon}) 的 {year} 年数据时出错: {str(year_error)}")
                # 继续处理下一个年份
                continue
        
        return all_results
    
    except Exception as e:
        logger.error(f"处理点 ({lat}, {lon}) 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def process_batch(batch_points_info):
    """处理一批网格点"""
    batch_results = []
    try:
        for point_info in batch_points_info:
            try:
                results = process_single_point(point_info)
                if results:
                    batch_results.extend(results)
            except Exception as e:
                logger.error(f"处理批次中的点时出错: {str(e)}")
                continue
    except Exception as e:
        logger.error(f"处理批次时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    return batch_results


def find_historical_files(historical_dir, model_name):
    """
    查找历史数据文件
    
    参数:
        historical_dir: 历史数据目录
        model_name: 模型名称
    
    返回:
        historical_files: 历史数据文件列表
    """
    all_files = [
        f for f in os.listdir(historical_dir) 
        if f.endswith(".nc") and "tasmax" in f and "_interpolated_1deg" in f
    ]
    
    if model_name == "EC-Earth3":
        # EC-Earth3: 一年一个文件，需要找到所有1981-2010年的文件
        historical_files = []
        for year in range(HISTORICAL_START_YEAR, HISTORICAL_END_YEAR + 1):
            # 查找包含该年份的文件
            pattern = f"{year}0101-{year}1231"
            found = False
            for f in all_files:
                if pattern in f:
                    historical_files.append(os.path.join(historical_dir, f))
                    found = True
                    break
            if not found:
                logger.warning(f"未找到 {year} 年的历史数据文件")
        return historical_files
    else:
        # 其他模型：只有一个文件
        if len(all_files) == 0:
            return []
        elif len(all_files) == 1:
            return [os.path.join(historical_dir, all_files[0])]
        else:
            # 多个文件，选择包含1981-2010年的文件
            # 通常文件名会包含日期范围
            selected_files = []
            for f in all_files:
                match = re.search(r'(\d{8})-(\d{8})', f)
                if match:
                    start_date = match.group(1)
                    end_date = match.group(2)
                    start_year = int(start_date[:4])
                    end_year = int(end_date[:4])
                    if start_year <= HISTORICAL_START_YEAR and end_year >= HISTORICAL_END_YEAR:
                        selected_files.append(os.path.join(historical_dir, f))
            
            if selected_files:
                return selected_files
            else:
                # 如果找不到匹配的文件，返回第一个
                logger.warning(f"未找到明确包含1981-2010年的文件，使用第一个文件: {all_files[0]}")
                return [os.path.join(historical_dir, all_files[0])]


def verify_historical_data(historical_files, model_name):
    """
    验证历史数据是否包含完整的1981-2010年数据
    
    参数:
        historical_files: 历史数据文件列表
        model_name: 模型名称
    
    返回:
        (is_valid, time_index, data, lat_index, lon_index): 验证结果和数据
    """
    if not historical_files:
        return False, None, None, None, None
    
    try:
        if model_name == "EC-Earth3":
            # EC-Earth3: 需要合并多个文件（一年一个文件，不需要chunks）
            datasets = []
            for file_path in historical_files:
                # EC-Earth3一年一个文件，文件较小，不需要chunks
                ds = xr.open_dataset(file_path)
                datasets.append(ds)
            
            # 合并所有数据集
            combined_ds = xr.concat(datasets, dim='time')
            combined_ds = combined_ds.sortby('time')
            
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
            expected_years = list(range(HISTORICAL_START_YEAR, HISTORICAL_END_YEAR + 1))
            missing_years = set(expected_years) - set(actual_years)
            
            if missing_years:
                logger.error(f"❌ 历史数据缺少年份: {sorted(missing_years)}")
                logger.error(f"   实际包含年份: {actual_years}")
                for ds in datasets:
                    ds.close()
                return False, None, None, None, None
            
            logger.info(f"✓ 历史数据验证通过，包含完整年份: {actual_years}")
            
            # 加载数据
            data = combined_ds['tasmax'].load().values
            lat_index = combined_ds.lat.values
            lon_index = combined_ds.lon.values
            
            # 关闭所有数据集
            for ds in datasets:
                ds.close()
            combined_ds.close()
            
            return True, time_index, data, lat_index, lon_index
            
        else:
            # 其他模型：只有一个文件（30年数据，需要chunks）
            file_path = historical_files[0]
            # 其他模型是30年一个文件，使用chunks延迟加载以提高性能
            with xr.open_dataset(file_path, chunks={'time': 366}) as ds:
                # 处理cftime时间对象（使用更健壮的转换逻辑）
                try:
                    # 先尝试decode_cf
                    ds_decoded = xr.decode_cf(ds, decode_times=True)
                    time_values = ds_decoded.time.values
                    
                    if hasattr(time_values, 'to_pandas'):
                        time_index = time_values.to_pandas()
                    else:
                        time_index = pd.to_datetime(time_values)
                    
                except Exception as e:
                    # 如果decode_cf失败，手动处理cftime对象
                    try:
                        import cftime
                        time_values = ds.time.values
                        
                        if len(time_values) > 0 and isinstance(time_values[0], cftime.datetime):
                            time_index = pd.to_datetime([pd.Timestamp(t.year, t.month, t.day) for t in time_values])
                        else:
                            time_index = pd.to_datetime(time_values)
                    except Exception as e2:
                        # 最后尝试：使用to_pandas
                        try:
                            time_index = ds.time.to_pandas()
                        except Exception:
                            raise ValueError(f"无法转换时间索引: {e}, {e2}")
                
                # 确保是DatetimeIndex
                if not isinstance(time_index, pd.DatetimeIndex):
                    time_index = pd.DatetimeIndex(time_index)
                
                # 验证年份
                actual_years = sorted(time_index.year.unique())
                expected_years = list(range(HISTORICAL_START_YEAR, HISTORICAL_END_YEAR + 1))
                missing_years = set(expected_years) - set(actual_years)
                
                if missing_years:
                    logger.error(f"❌ 历史数据缺少年份: {sorted(missing_years)}")
                    logger.error(f"   实际包含年份: {actual_years}")
                    return False, None, None, None, None
                
                logger.info(f"✓ 历史数据验证通过，包含完整年份: {actual_years}")
                
                # 加载数据
                data = ds['tasmax'].load().values
                lat_index = ds.lat.values
                lon_index = ds.lon.values
                
                return True, time_index, data, lat_index, lon_index
                
    except Exception as e:
        logger.error(f"验证历史数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None, None, None


def process_single_model(model_name, world_data):
    """处理单个模型"""
    logger.info(f"\n{'='*80}")
    logger.info(f"开始处理模型: {model_name}")
    logger.info(f"{'='*80}")
    
    # 构建模型路径
    model_base_path = os.path.join(BASE_PATH, "GCM_input_processed", model_name)
    output_base_path = os.path.join(BASE_PATH, "heat_wave", model_name)
    
    # 检查路径
    if not os.path.exists(model_base_path):
        logger.warning(f"模型数据路径不存在: {model_base_path}，跳过")
        return
    
    # 查找历史数据文件
    historical_dir = os.path.join(model_base_path, "historical")
    if not os.path.exists(historical_dir):
        logger.warning(f"历史数据目录不存在: {historical_dir}，跳过")
        return
    
    historical_files = find_historical_files(historical_dir, model_name)
    if not historical_files:
        logger.warning(f"未找到历史数据文件，跳过 {model_name}")
        return
    
    logger.info(f"找到 {len(historical_files)} 个历史数据文件")
    if model_name == "EC-Earth3":
        logger.info(f"  EC-Earth3使用多个文件（一年一个）")
    else:
        logger.info(f"  使用文件: {os.path.basename(historical_files[0])}")
    
    # 验证历史数据
    logger.info("验证历史数据是否包含完整的1981-2010年数据...")
    is_valid, time_index, data, lat_index, lon_index = verify_historical_data(historical_files, model_name)
    
    if not is_valid:
        logger.error(f"历史数据验证失败，跳过模型 {model_name}")
        return
    
    # 使用第一个文件识别陆地点（所有文件的空间网格应该相同）
    land_points = identify_land_points(historical_files[0], world_data)
    if not land_points:
        logger.warning(f"未找到任何陆地点，跳过模型 {model_name}")
        return
    
    logger.info(f"共找到 {len(land_points)} 个陆地点需要处理")
    
    # 处理每个SSP路径
    for ssp_path in SSP_PATHS:
        logger.info(f"\n=== 开始处理 {model_name} - {ssp_path} ===")
        
        # 查找未来数据文件
        future_dir = os.path.join(model_base_path, "future", ssp_path)
        if not os.path.exists(future_dir):
            logger.warning(f"未来数据目录不存在: {future_dir}，跳过")
            continue
        
        # 构建未来数据文件字典 {year: file_path}
        future_files = {}
        all_nc_files = [
            f for f in os.listdir(future_dir) 
            if f.endswith(".nc") and "tasmax" in f and "_interpolated_1deg" in f
        ]
        
        # 查找未来数据文件并验证是否都包含完整的2030-2034年数据
        tasmax_files = [
            f for f in all_nc_files 
            if "tasmax" in f
        ]
        
        if not tasmax_files:
            logger.warning(f"未找到tasmax文件，跳过 {ssp_path}")
            continue
        
        # 验证所有文件是否都包含完整的2030-2034年数据
        logger.info(f"验证 {len(tasmax_files)} 个tasmax文件是否包含完整的2030-2034年数据...")
        verified_files = {}  # {file_path: actual_years}
        missing_years_files = []  # 记录缺少年份的文件
        
        for nc_file in tasmax_files:
            file_path = os.path.join(future_dir, nc_file)
            try:
                # 打开文件验证年份
                with xr.open_dataset(file_path) as ds:
                    # 处理cftime时间对象，转换为pandas DatetimeIndex
                    try:
                        ds_decoded = xr.decode_cf(ds, decode_times=True)
                        time_values = ds_decoded.time.values
                        
                        if hasattr(time_values, 'to_pandas'):
                            time_index_future = time_values.to_pandas()
                        else:
                            time_index_future = pd.to_datetime(time_values)
                        
                        if not isinstance(time_index_future, pd.DatetimeIndex):
                            time_index_future = pd.DatetimeIndex(time_index_future)
                            
                    except Exception as e:
                        try:
                            import cftime
                            time_values = ds.time.values
                            
                            if len(time_values) > 0 and isinstance(time_values[0], cftime.datetime):
                                time_index_future = pd.to_datetime([pd.Timestamp(t.year, t.month, t.day) for t in time_values])
                            else:
                                time_index_future = pd.to_datetime(time_values)
                            
                            if not isinstance(time_index_future, pd.DatetimeIndex):
                                time_index_future = pd.DatetimeIndex(time_index_future)
                                
                        except Exception as e2:
                            try:
                                time_index_future = ds.time.to_pandas()
                                if not isinstance(time_index_future, pd.DatetimeIndex):
                                    time_index_future = pd.DatetimeIndex(time_index_future)
                            except Exception:
                                raise ValueError(f"无法转换时间索引: {e}, {e2}")
                    
                    # 提取实际包含的年份
                    actual_years = sorted(time_index_future.year.unique())
                    verified_files[file_path] = actual_years
                    
                    # 检查是否包含所有目标年份
                    missing_years = set(TARGET_YEARS) - set(actual_years)
                    if missing_years:
                        missing_years_files.append({
                            'file': nc_file,
                            'actual_years': actual_years,
                            'missing_years': sorted(missing_years)
                        })
                        logger.error(f"❌ 文件 {nc_file} 缺少年份: {sorted(missing_years)}")
                        logger.error(f"   实际包含年份: {actual_years}")
                    else:
                        logger.info(f"✓ 文件 {nc_file} 包含完整年份: {actual_years}")
                    
            except Exception as e:
                logger.error(f"❌ 验证文件 {nc_file} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                raise ValueError(f"无法验证文件 {nc_file}，程序停止")
        
        # 如果有文件缺少年份，停止运行
        if missing_years_files:
            logger.error("\n" + "="*80)
            logger.error("❌ 数据验证失败：部分文件缺少年份数据")
            logger.error("="*80)
            for file_info in missing_years_files:
                logger.error(f"文件: {file_info['file']}")
                logger.error(f"  缺少年份: {file_info['missing_years']}")
                logger.error(f"  实际包含年份: {file_info['actual_years']}")
            logger.error("="*80)
            logger.error("程序停止运行，请检查数据文件完整性")
            raise ValueError(f"数据验证失败：{len(missing_years_files)} 个文件缺少年份数据")
        
        # 所有文件验证通过，为每个目标年份分配文件
        logger.info(f"✓ 所有文件验证通过，都包含完整的2030-2034年数据")
        
        # 由于所有文件都包含完整年份，可以任意选择一个文件（通常只有一个文件）
        if len(verified_files) == 1:
            file_path = list(verified_files.keys())[0]
            for year in TARGET_YEARS:
                future_files[year] = file_path
            logger.info(f"所有年份使用同一个文件: {os.path.basename(file_path)}")
        else:
            for year in TARGET_YEARS:
                found = False
                for file_path, actual_years in verified_files.items():
                    if year in actual_years:
                        future_files[year] = file_path
                        found = True
                        break
                
                if not found:
                    raise ValueError(f"逻辑错误：验证通过但未找到包含 {year} 年数据的文件")
        
        if not future_files:
            logger.warning(f"未找到任何未来数据文件，跳过 {ssp_path}")
            continue
        
        logger.info(f"找到 {len(future_files)} 个年份的数据文件")
        
        # 创建输出目录
        output_dir = os.path.join(output_base_path, ssp_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 在主进程中加载历史数据到共享内存
        logger.info("正在加载历史数据到共享内存...")
        data_shape = data.shape
        data_dtype = data.dtype
        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        shm_array = np.ndarray(data_shape, dtype=data_dtype, buffer=shm.buf)
        shm_array[:] = data[:]  # 复制数据到共享内存
        
        logger.info(f"历史数据已加载到共享内存: {data_shape}, {data.nbytes / 1024**3:.2f} GB")
        
        # 准备批次数据
        num_cores = multiprocessing.cpu_count()
        num_processes = min(31, num_cores)
        logger.info(f"使用 {num_processes} 个进程进行并行处理（减少进程数以避免内存溢出）")
        
        batch_size = max(500, min(2000, len(land_points) // (num_processes * 2)))
        batches = []
        for i in range(0, len(land_points), batch_size):
            batch_points = land_points[i:i + batch_size]
            batch_info = [{'lat': lat, 'lon': lon} for lat, lon in batch_points]
            batches.append(batch_info)
        
        logger.info(f"数据分配情况:")
        logger.info(f"- 总陆地点数: {len(land_points)}")
        logger.info(f"- 批次大小: {batch_size}")
        logger.info(f"- 总批次数: {len(batches)}")
        
        # 初始化结果字典，按年份存储
        results_by_year = {year: [] for year in future_files.keys()}
        
        # 并行处理
        with multiprocessing.Pool(
            processes=num_processes,
            initializer=init_worker,
            initargs=(historical_files[0], future_files, shm.name, data_shape, data_dtype,
                     time_index, lat_index, lon_index)
        ) as pool:
            chunksize = max(1, len(batches) // (num_processes * 2))
            for batch_results in tqdm(
                pool.imap_unordered(process_batch, batches, chunksize=chunksize),
                total=len(batches),
                desc=f"处理批次 ({model_name} - {ssp_path})"
            ):
                for result in batch_results:
                    year = result['year']
                    results_by_year[year].append(result)
        
        # 保存结果到文件
        logger.info(f"\n保存结果统计:")
        for year in sorted(results_by_year.keys()):
            results = results_by_year[year]
            if results:
                output_file = os.path.join(output_dir, f"{year}_all_heat_wave.csv")
                df = pd.DataFrame(results)
                df = df[['lat', 'lon', 'number', 'start_day', 'date', 'Duration']]
                df.to_csv(output_file, index=False)
                logger.info(f"{year} 年结果已保存: {output_file}，共 {len(results)} 个热浪事件")
            else:
                logger.warning(f"{year} 年未找到热浪事件（可能数据加载失败或确实没有热浪）")
        
        # 清理共享内存
        try:
            shm.close()
            shm.unlink()
            logger.info("共享内存已清理")
        except Exception as e:
            logger.warning(f"清理共享内存时出错: {str(e)}")
        
        # 清理内存
        gc.collect()
    
    logger.info(f"\n✓ 模型 {model_name} 处理完成")


def main():
    """主函数"""
    try:
        logger.info("=== 开始热浪识别处理（多模型版本）===")
        logger.info(f"支持的模型: {', '.join(MODELS)}")
        logger.info(f"已处理的模型（将跳过）: {', '.join(PROCESSED_MODELS)}")
        
        # 检查Shapefile路径
        if not os.path.exists(SHAPEFILE_PATH):
            raise FileNotFoundError(f"Shapefile路径不存在: {SHAPEFILE_PATH}")
        
        # 读取陆地边界数据（所有模型共享）
        logger.info("正在读取陆地边界数据...")
        world_data = gpd.read_file(SHAPEFILE_PATH)
        logger.info(f"成功加载 {len(world_data)} 个陆地边界")
        
        # 处理每个模型
        models_to_process = [m for m in MODELS if m not in PROCESSED_MODELS]
        logger.info(f"\n需要处理的模型: {', '.join(models_to_process)}")
        
        for model_name in MODELS:
            if model_name in PROCESSED_MODELS:
                logger.info(f"\n跳过已处理的模型: {model_name}")
                continue
            
            try:
                process_single_model(model_name, world_data)
            except Exception as e:
                logger.error(f"处理模型 {model_name} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                logger.warning(f"继续处理下一个模型...")
                continue
        
        logger.info("\n=== 所有模型处理完成 ===")
    
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise
    finally:
        gc.collect()


if __name__ == "__main__":
    main()

