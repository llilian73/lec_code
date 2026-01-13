"""
按国家计算热浪期间的逐时能耗

功能：
1. 根据shp文件划分国家，获得点到国家的映射
2. 对于每个国家，根据热浪文件获取每一天发生热浪的点
3. 读取逐时能耗parquet文件，将每一天发生热浪的点的能耗按小时累加
4. 输出每个国家的逐时能耗CSV

输入数据：
1. 热浪文件：/home/linbor/WORK/lishiying/heat_wave/{模型名}/{SSP}/{年份}_all_heat_wave.csv
2. 逐时能耗：/home/linbor/WORK/lishiying/heat_wave/{模型名}/{SSP}/energy/{年份}/point/{case名}/{case名}_hourly.parquet
3. 国家边界：/home/linbor/WORK/lishiying/shapefiles/world_border2.shp
4. 国家信息：/home/linbor/WORK/lishiying/shapefiles/all_countries_info.csv

输出数据：
/home/linbor/WORK/lishiying/heat_wave/{模型名}/{SSP}/energy/hourly_energy/{大洲}/{国家代码}/{国家代码}_{年份}_hourly_energy.csv
"""

import pandas as pd
import geopandas as gpd
import os
import numpy as np
import logging
from shapely.geometry import Point
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from collections import defaultdict
import time
from multiprocessing import Pool, Manager
from functools import partial
from multiprocessing import shared_memory
import struct
import uuid

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置路径
BASE_PATH = "/home/linbor/WORK/lishiying"
HEAT_WAVE_BASE_PATH = os.path.join(BASE_PATH, "heat_wave")
SHAPEFILE_PATH = os.path.join(BASE_PATH, "shapefiles", "world_border2.shp")
COUNTRIES_INFO_CSV = os.path.join(BASE_PATH, "shapefiles", "all_countries_info.csv")

# 模型配置
MODELS = ["BCC-CSM2-MR"]  # 可以修改为其他模型

# SSP配置
SSP_PATHS = ["SSP126", "SSP245"]  # 可以修改为其他SSP

# 年份配置
TARGET_YEARS = [2031]

# Case配置
CASES = ['ref'] + [f'case{i}' for i in range(1, 21)]

# 并行处理配置
NUM_PROCESSES_COUNTRY = 56  # 计算国家逐时能耗的进程数（一个进程计算一个国家）


def read_csv_with_encoding(file_path, keep_default_na=True):
    """读取CSV文件，尝试多种编码"""
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, keep_default_na=keep_default_na)
            logger.debug(f"成功使用编码 '{encoding}' 读取文件: {os.path.basename(file_path)}")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    raise ValueError(f"无法使用常见编码读取文件: {file_path}")


def create_point_to_country_mapping(heat_wave_df, shp_path, country_to_continent):
    """创建点到国家的映射
    
    Parameters:
    -----------
    heat_wave_df : pd.DataFrame
        热浪文件DataFrame（已读取）
    shp_path : str
        国家边界shp文件路径
    country_to_continent : dict
        国家代码到大洲的映射（已读取）
    
    Returns:
    --------
    dict: {(lat, lon): (country_code, continent)}
    """
    start_time = time.time()
    logger.info("创建点到国家的映射...")
    
    # 从热浪文件获取所有点
    unique_points = heat_wave_df[['lat', 'lon']].drop_duplicates()
    logger.info(f"共 {len(unique_points)} 个唯一坐标点")
    
    # 读取国家边界
    countries_gdf = gpd.read_file(shp_path)
    logger.info(f"读取了 {len(countries_gdf)} 个国家/地区")
    
    # 将点转换为GeoDataFrame
    geometry = [Point(xy) for xy in zip(unique_points['lon'], unique_points['lat'])]
    points_gdf = gpd.GeoDataFrame(unique_points, geometry=geometry, crs="EPSG:4326")
    
    # 确保坐标系匹配
    if points_gdf.crs != countries_gdf.crs:
        points_gdf = points_gdf.to_crs(countries_gdf.crs)
    
    # 空间连接
    joined_gdf = gpd.sjoin(points_gdf, countries_gdf, how="left", predicate="within")
    
    # 创建映射字典
    point_to_country = {}
    for idx, row in joined_gdf.iterrows():
        lat = row['lat']
        lon = row['lon']
        country_code = row.get('GID_0', None)
        
        if pd.notna(country_code):
            # 处理HKG和MAC，合并到CHN
            if country_code in ['HKG', 'MAC']:
                country_code = 'CHN'
            
            continent = country_to_continent.get(country_code, 'Unknown')
            point_to_country[(lat, lon)] = (country_code, continent)
        else:
            point_to_country[(lat, lon)] = (None, None)
    
    # 统计
    mapped_count = sum(1 for v in point_to_country.values() if v[0] is not None)
    elapsed_time = time.time() - start_time
    logger.info(f"成功映射 {mapped_count}/{len(point_to_country)} 个点到国家，耗时 {elapsed_time:.2f} 秒")
    
    return point_to_country


def dataframe_to_shared_memory(df, case_name):
    """将DataFrame转换为真正的共享内存格式
    
    Parameters:
    -----------
    df : pd.DataFrame
        要转换的DataFrame
    case_name : str
        工况名称，用于生成共享内存名称
    
    Returns:
    --------
    dict: 包含共享内存信息的字典
    """
    shared_info = {}
    shm_refs = []  # 保存引用，防止被垃圾回收
    
    # 处理数值列：使用真正的共享内存
    for col in df.columns:
        if col in ['point_key', 'datetime']:
            # point_key和datetime需要特殊处理，先跳过
            continue
        
        if df[col].dtype != 'object':
            # 数值类型：使用共享内存
            arr = df[col].values
            # 生成共享内存名称（由于是串行处理，每个工况的名称是唯一的）
            # 共享内存名称限制为31个字符，使用简短的名称
            shm_name = f"{case_name[:10]}_{col[:15]}"  # 限制长度
            try:
                shm = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=shm_name)
            except FileExistsError:
                # 如果名称已存在（理论上不应该发生，因为串行处理），添加随机后缀
                shm_name = f"{case_name[:8]}_{col[:12]}_{uuid.uuid4().hex[:6]}"
                shm = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=shm_name)
            shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            shared_arr[:] = arr[:]
            
            shared_info[col] = {
                'type': 'shm',
                'name': shm.name,
                'shape': arr.shape,
                'dtype': str(arr.dtype)
            }
            shm_refs.append(shm)
    
    # 处理point_key列：转换为可序列化格式
    if 'point_key' in df.columns:
        # 将point_key转换为(lat, lon)的列表
        point_keys = df['point_key'].tolist()
        shared_info['point_key'] = {
            'type': 'list',
            'data': point_keys
        }
    
    # 处理datetime列：转换为时间戳（int64，秒级，节省内存）
    if 'datetime' in df.columns:
        # 转换为pandas Timestamp的int64表示（秒级，而不是纳秒级，节省内存）
        datetime_series = pd.to_datetime(df['datetime'])
        # 转换为秒级时间戳（除以1e9）
        datetime_s = (datetime_series.astype('int64') // 1_000_000_000).astype('int64')
        
        # 生成共享内存名称（由于是串行处理，每个工况的名称是唯一的）
        shm_name = f"{case_name[:15]}_dt"  # 限制长度，dt表示datetime
        try:
            shm = shared_memory.SharedMemory(create=True, size=datetime_s.nbytes, name=shm_name)
        except FileExistsError:
            # 如果名称已存在（理论上不应该发生），添加随机后缀
            shm_name = f"{case_name[:10]}_dt_{uuid.uuid4().hex[:6]}"
            shm = shared_memory.SharedMemory(create=True, size=datetime_s.nbytes, name=shm_name)
        shared_arr = np.ndarray(datetime_s.shape, dtype=datetime_s.dtype, buffer=shm.buf)
        shared_arr[:] = datetime_s[:]
        
        shared_info['datetime'] = {
            'type': 'shm',
            'name': shm.name,
            'shape': datetime_s.shape,
            'dtype': str(datetime_s.dtype),
            'unit': 'seconds'  # 标记为秒级时间戳
        }
        shm_refs.append(shm)
    
    # 保存其他信息
    shared_info['_columns'] = df.columns.tolist()
    shared_info['_nrows'] = len(df)
    shared_info['_shm_refs'] = shm_refs  # 保存引用
    
    return shared_info


def shared_memory_to_dataframe(shared_info, point_filter=None):
    """从共享内存重建DataFrame，并一次性预生成date列
    
    Parameters:
    -----------
    shared_info : dict
        包含共享内存信息的字典
    point_filter : set, optional
        如果提供，只提取这些点的数据（set of (lat, lon) tuples）
        如果不提供，提取全部数据
    
    Returns:
    --------
    tuple: (pd.DataFrame, list) - DataFrame和共享内存引用列表（需要保持引用直到使用完成）
    """
    columns = shared_info.get('_columns', [])
    nrows = shared_info.get('_nrows', 0)
    
    # 先恢复point_key列，用于建立行索引映射
    if 'point_key' in shared_info:
        point_keys = shared_info['point_key']['data']
        point_keys_list = [tuple(x) if isinstance(x, list) else x for x in point_keys]
    else:
        point_keys_list = None
    
    # 如果提供了point_filter，建立行索引映射
    row_indices = None
    if point_filter is not None and point_keys_list is not None:
        # 建立point_key到行索引的映射
        point_filter_set = set(point_filter)
        # 找出所有匹配的行索引
        row_indices = [i for i, pk in enumerate(point_keys_list) if pk in point_filter_set]
        row_indices = np.array(row_indices, dtype=np.int64)
    else:
        # 没有过滤，使用全部行
        row_indices = None
    
    # 重建DataFrame
    data = {}
    shm_refs = []  # 保存共享内存引用，防止被垃圾回收
    
    # 从共享内存恢复数值列
    for col in columns:
        if col in ['point_key', 'datetime']:
            continue
        
        if col in shared_info and shared_info[col]['type'] == 'shm':
            shm_info = shared_info[col]
            existing_shm = shared_memory.SharedMemory(name=shm_info['name'])
            arr = np.ndarray(shm_info['shape'], dtype=shm_info['dtype'], buffer=existing_shm.buf)
            
            # 如果提供了row_indices，只提取这些行的数据
            if row_indices is not None:
                # 只复制需要的行
                data[col] = arr[row_indices].copy()
            else:
                # 复制全部数据
                data[col] = arr.copy()
            shm_refs.append(existing_shm)  # 保存引用
    
    # 恢复point_key列（如果过滤了，只保留匹配的点）
    if 'point_key' in shared_info:
        if row_indices is not None:
            data['point_key'] = [point_keys_list[i] for i in row_indices]
        else:
            data['point_key'] = point_keys_list
    
    # 恢复datetime列，并一次性预生成date列
    if 'datetime' in shared_info and shared_info['datetime']['type'] == 'shm':
        shm_info = shared_info['datetime']
        existing_shm = shared_memory.SharedMemory(name=shm_info['name'])
        datetime_s = np.ndarray(shm_info['shape'], dtype=shm_info['dtype'], buffer=existing_shm.buf)
        
        # 如果提供了row_indices，只提取这些行的数据
        if row_indices is not None:
            datetime_s_filtered = datetime_s[row_indices].copy()
        else:
            datetime_s_filtered = datetime_s.copy()
        
        # 从秒级时间戳恢复为datetime（转换为纳秒级时间戳，然后转换为datetime）
        datetime_ns = datetime_s_filtered.astype('int64') * 1_000_000_000
        datetime_series = pd.to_datetime(datetime_ns)
        
        # 一次性预生成date列（避免循环中重复调用dt.date）
        # 注意：pd.to_datetime() 对 numpy 数组返回 DatetimeIndex，需要转换为 Series 才能使用 .dt 访问器
        data['datetime'] = datetime_series
        data['date'] = pd.Series(datetime_series).dt.date
        shm_refs.append(existing_shm)  # 保存引用
    
    df = pd.DataFrame(data)
    
    return df, shm_refs


def load_single_parquet(args):
    """并行读取单个parquet文件
    
    Parameters:
    -----------
    args : tuple
        (case_name, parquet_path, model_name, ssp_path, year)
    
    Returns:
    --------
    tuple: (case_name, df_hourly) 或 (case_name, None) 如果失败
    """
    case_name, parquet_path, model_name, ssp_path, year = args
    
    try:
        if not os.path.exists(parquet_path):
            logger.warning(f"文件不存在: {parquet_path}")
            return (case_name, None)
        
        # 读取parquet文件（使用pyarrow引擎，更快）
        df_hourly = pd.read_parquet(parquet_path, engine='pyarrow')
        
        # 检查DataFrame是否为空
        if df_hourly.empty:
            logger.warning(f"文件为空: {parquet_path}")
            return (case_name, None)
        
        # 调试日志：确认读取后的列类型
        logger.debug(f"  {case_name} 列类型: date={df_hourly['date'].dtype if 'date' in df_hourly.columns else 'N/A'}, "
                   f"time={df_hourly['time'].dtype if 'time' in df_hourly.columns else 'N/A'}")
        
        # 合并date和time列创建datetime
        if 'date' in df_hourly.columns and 'time' in df_hourly.columns:
            # 统一处理date列：无论什么类型，都用pd.to_datetime()转换
            date_series = None
            date_str = None
            try:
                # 先转换为datetime类型（处理date32类型）
                date_series = pd.to_datetime(df_hourly['date'])
                # 提取日期部分（YYYY-MM-DD格式）
                date_str = date_series.dt.strftime('%Y-%m-%d')
            except Exception as e:
                logger.error(f"处理date列时出错: {e}")
                # 如果转换失败，尝试直接使用
                try:
                    date_str = df_hourly['date'].astype(str).str[:10]  # 取前10个字符（YYYY-MM-DD）
                    date_series = pd.to_datetime(date_str)
                except Exception as e2:
                    logger.error(f"备用date转换也失败: {e2}")
                    return (case_name, None)
            
            # 统一处理time列：转换为字符串时指定格式
            time_str = None
            try:
                # 如果time是time类型，转换为字符串格式 HH:MM:SS
                if hasattr(df_hourly['time'].iloc[0], 'strftime'):
                    # time对象，使用strftime格式化
                    time_str = df_hourly['time'].apply(lambda x: x.strftime('%H:%M:%S') if pd.notna(x) else '00:00:00')
                elif isinstance(df_hourly['time'].iloc[0], str):
                    # 已经是字符串
                    time_str = df_hourly['time']
                else:
                    # 其他类型，先转换为字符串，然后尝试解析为时间格式
                    time_str = df_hourly['time'].astype(str)
                    # 如果格式不是HH:MM:SS，尝试转换
                    if len(time_str) > 0 and ':' not in str(time_str.iloc[0]):
                        time_str = pd.to_datetime(time_str, format='%H:%M:%S', errors='coerce').dt.strftime('%H:%M:%S')
            except Exception as e:
                logger.error(f"处理time列时出错: {e}")
                # 如果转换失败，尝试直接使用
                time_str = df_hourly['time'].astype(str)
            
            # 合并时使用pandas的日期时间操作，而不是字符串拼接
            try:
                # 方法1：使用字符串拼接（如果格式正确）
                datetime_str = date_str + ' ' + time_str
                df_hourly['datetime'] = pd.to_datetime(datetime_str, errors='coerce')
                
                # 检查是否有无效的datetime
                if df_hourly['datetime'].isna().any():
                    logger.warning(f"  {case_name}: 部分datetime转换失败，尝试备用方法")
                    # 备用方法：直接组合date和time
                    if date_series is not None:
                        df_hourly['datetime'] = date_series + pd.to_timedelta(time_str)
                    else:
                        raise ValueError("date_series未定义")
            except Exception as e:
                logger.error(f"合并datetime时出错: {e}")
                # 备用方法：直接组合date和time
                try:
                    if date_series is None:
                        date_series = pd.to_datetime(df_hourly['date'])
                    # 将time转换为timedelta
                    time_delta = pd.to_timedelta(time_str)
                    df_hourly['datetime'] = date_series + time_delta
                except Exception as e2:
                    logger.error(f"备用方法也失败: {e2}")
                    return (case_name, None)
        elif 'datetime' in df_hourly.columns:
            # 如果已经有datetime列，直接使用
            df_hourly['datetime'] = pd.to_datetime(df_hourly['datetime'], errors='coerce')
            if df_hourly['datetime'].isna().any():
                logger.warning(f"  {case_name}: 部分datetime列值无效")
        else:
            logger.warning(f"无法找到日期时间列: {parquet_path}")
            return (case_name, None)
        
        # 创建point_key列
        df_hourly['point_key'] = list(zip(df_hourly['lat'], df_hourly['lon']))
        
        logger.info(f"  ✓ {case_name}: {len(df_hourly)} 条记录")
        return (case_name, df_hourly)
    
    except Exception as e:
        logger.error(f"读取 {parquet_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return (case_name, None)


def get_heat_wave_points_by_date(heat_wave_file, year):
    """获取每一天发生热浪的点
    
    Parameters:
    -----------
    heat_wave_file : str
        热浪文件路径
    year : int
        年份
    
    Returns:
    --------
    dict: {date: set of (lat, lon) tuples}
    """
    start_time = time.time()
    logger.info(f"读取热浪文件: {heat_wave_file}")
    heat_wave_df = pd.read_csv(heat_wave_file)
    
    # 解析日期
    def parse_date(date_str, year):
        """解析日期：将 "M/D" 格式转换为该年份的datetime"""
        month, day = map(int, date_str.split('/'))
        return datetime(year, month, day)
    
    heat_wave_df['parsed_date'] = heat_wave_df['date'].apply(lambda x: parse_date(x, year))
    
    # 按日期分组，获取每天发生热浪的点
    daily_points = defaultdict(set)
    
    for _, row in heat_wave_df.iterrows():
        start_date = row['parsed_date']
        duration = int(row['Duration'])
        lat = row['lat']
        lon = row['lon']
        
        # 获取该热浪事件覆盖的所有日期
        for i in range(duration):
            current_date = start_date + timedelta(days=i)
            daily_points[current_date].add((lat, lon))
    
    elapsed_time = time.time() - start_time
    logger.info(f"共 {len(daily_points)} 天有热浪事件，耗时 {elapsed_time:.2f} 秒")
    return daily_points


def process_single_year(model_name, ssp_path, year):
    """处理单个年份的逐时能耗
    
    Parameters:
    -----------
    model_name : str
        模型名称
    ssp_path : str
        SSP路径
    year : int
        年份
    """
    year_start_time = time.time()
    logger.info(f"\n{'='*80}")
    logger.info(f"处理 {model_name} - {ssp_path} - {year} 年")
    logger.info(f"{'='*80}")
    
    # 文件路径
    heat_wave_file = os.path.join(
        HEAT_WAVE_BASE_PATH, model_name, ssp_path, f"{year}_all_heat_wave.csv"
    )
    
    if not os.path.exists(heat_wave_file):
        logger.warning(f"热浪文件不存在: {heat_wave_file}")
        return
    
    # ========== 读取共用数据（只读取一次，所有工况共用）==========
    logger.info("读取共用数据（热浪文件、国家信息）...")
    t0 = time.time()
    
    # 读取热浪文件（只读取一次）
    heat_wave_df = pd.read_csv(heat_wave_file)
    
    # 读取国家信息（只读取一次）
    countries_info_df = read_csv_with_encoding(COUNTRIES_INFO_CSV)
    country_to_continent = dict(zip(
        countries_info_df['Country_Code_3'],
        countries_info_df['continent']
    ))
    
    logger.info(f"[时间统计] 读取共用数据总耗时: {time.time() - t0:.2f} 秒")
    
    # ========== 第一步：建立点-国家的映射 ==========
    t0 = time.time()
    point_to_country = create_point_to_country_mapping(
        heat_wave_df, SHAPEFILE_PATH, country_to_continent
    )
    logger.info(f"[时间统计] 点-国家映射总耗时: {time.time() - t0:.2f} 秒")
    
    # ========== 第二步：建立点-热浪日期的映射 ==========
    t0 = time.time()
    # 格式：{(lat, lon): set of dates}
    point_to_heat_wave_dates = defaultdict(set)
    def parse_date(date_str, year):
        """解析日期：将 "M/D" 格式转换为该年份的datetime"""
        month, day = map(int, date_str.split('/'))
        return datetime(year, month, day)
    
    heat_wave_df['parsed_date'] = heat_wave_df['date'].apply(lambda x: parse_date(x, year))
    
    for _, row in heat_wave_df.iterrows():
        start_date = row['parsed_date']
        duration = int(row['Duration'])
        lat = row['lat']
        lon = row['lon']
        point_key = (lat, lon)
        
        # 获取该热浪事件覆盖的所有日期
        for i in range(duration):
            current_date = start_date + timedelta(days=i)
            point_to_heat_wave_dates[point_key].add(current_date)
    
    logger.info(f"[时间统计] 建立点-热浪日期映射总耗时: {time.time() - t0:.2f} 秒")
    logger.info(f"  共 {len(point_to_heat_wave_dates)} 个点有热浪事件")
    
    # ========== 第三步：建立国家-热浪日期-点的映射（转换为共享内存格式）==========
    t0 = time.time()
    # 格式：{country_code: {date: list of (lat, lon) tuples}}
    # 注意：Manager的dict不能存储set，所以使用list
    country_heat_wave_info_dict = {}
    
    for (lat, lon), (country_code, continent) in point_to_country.items():
        if country_code is None:
            continue
        
        point_key = (lat, lon)
        # 获取该点的所有热浪日期
        heat_wave_dates = point_to_heat_wave_dates.get(point_key, set())
        
        # 将该点添加到对应国家的对应日期
        if country_code not in country_heat_wave_info_dict:
            country_heat_wave_info_dict[country_code] = {}
        
        for date in heat_wave_dates:
            # 将date转换为字符串，因为Manager的dict的key必须是可序列化的
            date_str = date.isoformat() if isinstance(date, datetime) else str(date)
            if date_str not in country_heat_wave_info_dict[country_code]:
                country_heat_wave_info_dict[country_code][date_str] = []
            country_heat_wave_info_dict[country_code][date_str].append(point_key)
    
    logger.info(f"[时间统计] 建立国家-热浪日期-点映射总耗时: {time.time() - t0:.2f} 秒")
    logger.info(f"  共 {len(country_heat_wave_info_dict)} 个国家有热浪事件")
    
    # ========== 第四步：创建共享内存管理器 ==========
    manager = Manager()
    
    # 共享字典：存储国家-热浪日期-点映射
    shared_country_heat_wave_info = manager.dict()
    for country_code, dates_dict in country_heat_wave_info_dict.items():
        shared_dates_dict = manager.dict()
        for date_str, points_list in dates_dict.items():
            shared_dates_dict[date_str] = points_list
        shared_country_heat_wave_info[country_code] = shared_dates_dict
    
    # ========== 第五步：按工况串行读取，并行计算国家逐时能耗，立即输出 ==========
    total_start_time = time.time()
    
    # 按工况串行处理
    for case_name in CASES:
        case_start_time = time.time()
        logger.info(f"\n{'='*80}")
        logger.info(f"处理工况: {case_name}")
        logger.info(f"{'='*80}")
        
        # 读取当前工况的parquet文件
        parquet_path = os.path.join(
            HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", str(year),
            "point", case_name, f"{case_name}_hourly.parquet"
        )
        
        if not os.path.exists(parquet_path):
            logger.warning(f"文件不存在: {parquet_path}，跳过该工况")
            continue
        
        # 读取并处理parquet文件
        df_case = load_single_parquet((case_name, parquet_path, model_name, ssp_path, year))[1]
        if df_case is None:
            logger.warning(f"读取 {case_name} 失败，跳过该工况")
            continue
        
        logger.info(f"  {case_name}: 读取了 {len(df_case)} 条记录")
        
        # 将DataFrame转换为真正的共享内存格式
        shared_case_data = dataframe_to_shared_memory(df_case, case_name)
        logger.info(f"  {case_name}: 已转换为共享内存格式")
        
        # 释放原始DataFrame
        del df_case
        import gc
        gc.collect()
        
        # 准备所有国家的参数（使用共享内存）
        country_args = []
        for country_code in shared_country_heat_wave_info.keys():
            continent = country_to_continent.get(country_code, 'Unknown')
            country_args.append((
                country_code,
                shared_country_heat_wave_info[country_code],  # 共享内存中的映射
                continent,
                case_name,  # 当前工况
                shared_case_data,   # 共享内存中的工况数据
                year,
                model_name,
                ssp_path
            ))
        
        # 并行处理所有国家（在当前工况下），每个进程直接输出CSV
        logger.info(f"  并行处理 {len(country_args)} 个国家的 {case_name} 逐时能耗（{NUM_PROCESSES_COUNTRY}个进程）...")
        
        with Pool(processes=min(NUM_PROCESSES_COUNTRY, len(country_args))) as pool:
            pool.map(process_single_country_single_case, country_args)
        
        # 释放共享内存
        if '_shm_refs' in shared_case_data:
            for shm in shared_case_data['_shm_refs']:
                try:
                    shm.close()
                    shm.unlink()
                except:
                    pass
        del shared_case_data
        gc.collect()
        
        case_elapsed = time.time() - case_start_time
        logger.info(f"  工况 {case_name} 处理完成，耗时: {case_elapsed:.2f} 秒")
    
    logger.info(f"\n[时间统计] 所有工况处理总耗时: {time.time() - total_start_time:.2f} 秒")


def process_single_country_single_case(args):
    """并行处理单个国家在单个工况下的逐时能耗计算，直接输出CSV
    
    Parameters:
    -----------
    args : tuple
        (country_code, shared_country_heat_wave_dates_dict, continent, case_name, shared_case_data,
         year, model_name, ssp_path)
    """
    (country_code, shared_country_heat_wave_dates_dict, continent, case_name, shared_case_data,
     year, model_name, ssp_path) = args
    
    try:
        # 计时开始
        total_start_time = time.time()
        
        # 先收集该国家所有热浪日期和对应的点
        country_heat_wave_dates = {}  # {date: set of points}
        country_all_points = set()  # 收集该国家的所有点（用于过滤）
        
        for date_str, heat_wave_points_list in shared_country_heat_wave_dates_dict.items():
            # 将date_str转换回date对象
            try:
                if isinstance(date_str, str):
                    if 'T' in date_str or ' ' in date_str:
                        date = pd.to_datetime(date_str).date()
                    else:
                        date = pd.to_datetime(date_str).date()
                else:
                    date = date_str
            except Exception as e:
                logger.warning(f"无法解析日期: {date_str}, 错误: {e}")
                continue
            
            heat_wave_points_set = set(heat_wave_points_list)
            country_heat_wave_dates[date] = heat_wave_points_set
            # 收集所有点
            country_all_points.update(heat_wave_points_set)
        
        # 从共享内存重建DataFrame，只复制该国家的数据
        # 优化：只复制该国家相关的数据，而不是全部2700万行
        read_start_time = time.time()
        df_case, shm_refs = shared_memory_to_dataframe(shared_case_data, point_filter=country_all_points)
        read_elapsed = time.time() - read_start_time
        
        # 优化计算逻辑：使用向量化操作替代嵌套循环
        calc_start_time = time.time()
        
        # 步骤1：预先过滤出该国家的所有点（已经在shared_memory_to_dataframe中完成）
        # df_case 已经只包含该国家的点
        
        # 步骤2：创建热浪日期和点的映射，用于快速查找
        # 构建一个集合，包含所有(date, point_key)的组合
        heat_wave_date_point_set = set()
        for date, points_set in country_heat_wave_dates.items():
            for point in points_set:
                heat_wave_date_point_set.add((date, point))
        
        # 步骤3：使用向量化操作创建mask，筛选出热浪日期+热浪点的组合
        # 方法：先筛选出热浪日期的数据，然后筛选出热浪点
        # 这样可以减少需要检查的数据量
        heat_wave_dates_set = set(country_heat_wave_dates.keys())
        mask_date = df_case['date'].isin(heat_wave_dates_set)
        df_date_filtered = df_case[mask_date].copy()
        
        if len(df_date_filtered) > 0:
            # 创建(date, point_key)的元组Series用于快速查找
            df_date_filtered['date_point'] = list(zip(df_date_filtered['date'], df_date_filtered['point_key']))
            mask_point = df_date_filtered['date_point'].isin(heat_wave_date_point_set)
            df_filtered = df_date_filtered[mask_point].copy()
            
            # 删除临时列
            del df_date_filtered['date_point']
            del df_date_filtered
        else:
            df_filtered = pd.DataFrame()
        
        # 释放原始DataFrame
        del df_case
        
        # 步骤4：一次性按datetime分组并累加（向量化操作，替代嵌套循环）
        if len(df_filtered) > 0:
            # 使用groupby().sum()进行向量化累加，比循环累加快得多
            result_series = df_filtered.groupby('datetime')['total_demand'].sum()
            # 转换为字典，key为datetime的isoformat字符串
            country_case_energy = {dt.isoformat(): val for dt, val in result_series.items()}
        else:
            country_case_energy = {}
        
        # 释放中间DataFrame
        del df_filtered
        
        calc_elapsed = time.time() - calc_start_time
        
        # 直接输出CSV，不写入共享内存
        if country_case_energy:
            save_start_time = time.time()
            
            # 创建时间序列（该年份的所有小时）
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31, 23, 0, 0)
            time_range = pd.date_range(start=start_date, end=end_date, freq='h')  # 使用'h'替代'H'，避免FutureWarning
            
            # 创建结果DataFrame
            result_data = []
            for dt in time_range:
                dt_str = dt.isoformat()
                row_data = {'time': dt}
                # 当前工况的能耗
                energy = country_case_energy.get(dt_str, 0.0)
                row_data[case_name] = energy
                result_data.append(row_data)
            
            result_df = pd.DataFrame(result_data)
            
            # 保存CSV（文件名包含case名称）
            output_dir = os.path.join(
                HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", "hourly_energy",
                continent, country_code
            )
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, f"{country_code}_{year}_{case_name}_hourly_energy.csv")
            result_df.to_csv(output_file, index=False)
            
            save_elapsed = time.time() - save_start_time
            total_elapsed = time.time() - total_start_time
            
            # 精简日志：每个国家只打印一次成功信息，包含所有步骤的耗时
            logger.info(f"  ✓ {country_code} - {case_name}: 已保存 ({len(country_case_energy)} 个时间点) | "
                       f"读取: {read_elapsed:.2f}s, 计算: {calc_elapsed:.2f}s, 保存: {save_elapsed:.2f}s, 总计: {total_elapsed:.2f}s")
        
        # 释放共享内存引用（df_case已在前面删除）
        # 关闭共享内存引用（不unlink，因为主进程会统一unlink）
        for shm in shm_refs:
            try:
                shm.close()
            except:
                pass
        del shm_refs
        
        logger.debug(f"  国家 {country_code} - 工况 {case_name}: 处理完成，共 {len(country_case_energy)} 个时间点")
    
    except Exception as e:
        logger.error(f"处理国家 {country_code} - 工况 {case_name} 时出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    try:
        logger.info("=== 开始按国家计算热浪期间的逐时能耗 ===")
        logger.info(f"处理的模型: {', '.join(MODELS)}")
        logger.info(f"SSP路径: {', '.join(SSP_PATHS)}")
        logger.info(f"目标年份: {TARGET_YEARS}")
        
        for model_name in MODELS:
            for ssp_path in SSP_PATHS:
                # 处理所有年份
                for year in TARGET_YEARS:
                    try:
                        process_single_year(model_name, ssp_path, year)
                    except Exception as e:
                        logger.error(f"处理 {model_name} - {ssp_path} - {year} 年时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
        
        logger.info("\n=== 所有处理完成 ===")
    
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

