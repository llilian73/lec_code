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
from operator import itemgetter
import gc

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
SSP_PATHS = ["SSP126"]  # 可以修改为其他SSP

# 年份配置（支持同时处理多年份）
TARGET_YEARS = [2030, 2031, 2032, 2033, 2034]  # 5年数据

# 批次大小：一次处理1年（21个工况）
BATCH_YEARS = 1

# Case批次大小：一次读取11个工况
CASE_BATCH_SIZE = 11

# Case配置
CASES = ['ref'] + [f'case{i}' for i in range(1, 21)]

# 并行处理配置
NUM_PROCESSES_COUNTRY = 20  # 计算国家逐时能耗的进程数（一个进程计算一个国家）


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
    
    # 处理point_key列：转换为可序列化格式，并建立point_key到行索引的映射（优化查找速度）
    if 'point_key' in df.columns:
        # 将point_key转换为(lat, lon)的列表
        point_keys = df['point_key'].tolist()
        shared_info['point_key'] = {
            'type': 'list',
            'data': point_keys
        }
        
        # 建立point_key到行索引的映射（用于快速查找，避免遍历2700万行）
        # 格式：{point_key: [row_indices]}，一个点可能有多行（不同时间）
        point_key_to_indices = defaultdict(list)
        for i, pk in enumerate(point_keys):
            point_key_to_indices[pk].append(i)
        
        # 转换为普通字典（defaultdict不能序列化）
        shared_info['_point_key_to_indices'] = dict(point_key_to_indices)
    
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
    
    # 如果提供了point_filter，使用预建立的映射快速查找行索引（优化：避免遍历2700万行）
    row_indices = None
    if point_filter is not None:
        # 优先使用预建立的映射（如果存在）
        if '_point_key_to_indices' in shared_info:
            # 使用预建立的映射，O(m)复杂度，其中m是该国家的点数
            point_key_to_indices = shared_info['_point_key_to_indices']
            point_filter_set = set(point_filter)
            row_indices_list = []
            for point in point_filter_set:
                if point in point_key_to_indices:
                    row_indices_list.extend(point_key_to_indices[point])
            row_indices = np.array(row_indices_list, dtype=np.int64) if row_indices_list else None
        else:
            # 备用方案：如果没有预建立映射，使用原来的方法（向后兼容）
            # 注意：这个方法较慢，O(n)复杂度，应该尽量避免使用
            if 'point_key' in shared_info:
                point_keys = shared_info['point_key']['data']
                point_keys_list = [tuple(x) if isinstance(x, list) else x for x in point_keys]
                point_filter_set = set(point_filter)
                # 找出所有匹配的行索引（较慢，O(n)复杂度，遍历2700万行）
                row_indices_list = [i for i, pk in enumerate(point_keys_list) if pk in point_filter_set]
                row_indices = np.array(row_indices_list, dtype=np.int64) if row_indices_list else None
            else:
                row_indices = None
    else:
        # 没有过滤，使用全部行
        row_indices = None
    
    # 恢复point_key列（如果需要）
    if 'point_key' in shared_info and row_indices is not None:
        point_keys = shared_info['point_key']['data']
        point_keys_list = [tuple(x) if isinstance(x, list) else x for x in point_keys]
    elif 'point_key' in shared_info:
        point_keys = shared_info['point_key']['data']
        point_keys_list = [tuple(x) if isinstance(x, list) else x for x in point_keys]
    else:
        point_keys_list = None
    
    # 重建DataFrame
    data = {}
    shm_refs = []  # 保存共享内存引用，防止被垃圾回收
    
    # 从共享内存恢复数值列（优化：只复制计算时需要的列）
    # 计算时需要的列：cooling_demand（累加）, date（由datetime生成）, point_key（过滤）, datetime（分组）
    # 不需要的列：heating_demand, total_demand, lat, lon 等
    required_columns = {'cooling_demand'}  # 只复制计算时真正需要的列
    
    for col in columns:
        if col in ['point_key', 'datetime']:
            continue
        
        # 只复制需要的列（延迟复制策略）
        if col not in required_columns:
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
            # 优化：对于大量索引，分批处理或使用numpy索引
            # 如果索引数量较少，使用itemgetter；否则使用列表推导式（已优化）
            if len(row_indices) < 10000:
                # 小批量：使用itemgetter（更快）
                try:
                    getter = itemgetter(*row_indices)
                    data['point_key'] = list(getter(point_keys_list))
                except:
                    # 如果失败，回退到列表推导式
                    data['point_key'] = [point_keys_list[i] for i in row_indices]
            else:
                # 大批量：直接使用列表推导式（避免itemgetter参数过多）
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
        
        # 优化：从秒级时间戳恢复为datetime（使用更高效的方法）
        # 方法：直接使用pd.to_datetime的unit参数，避免先转换为纳秒
        datetime_series = pd.to_datetime(datetime_s_filtered, unit='s')
        
        # 一次性预生成date列（避免循环中重复调用dt.date）
        # 优化：直接使用DatetimeIndex的date属性，比转换为Series再取date快
        datetime_index = pd.DatetimeIndex(datetime_series) if not isinstance(datetime_series, pd.DatetimeIndex) else datetime_series
        data['datetime'] = datetime_series
        data['date'] = datetime_index.date
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
        
        # 方案2：只读取必要的列，避免读入heating_demand等无用列
        # 需要的列：lat, lon, cooling_demand, date, time（或已有datetime）
        columns_needed = ['lat', 'lon', 'cooling_demand', 'date', 'time']
        
        try:
            # 尝试只读取必要列
            df_hourly = pd.read_parquet(
                parquet_path,
                columns=columns_needed,  # 关键：只读必要列
                engine='pyarrow'
            )
        except Exception as e:
            # 如果指定列失败（可能列名不同），回退到全读
            logger.warning(f"指定列读取失败，回退全读: {e}")
            df_hourly = pd.read_parquet(parquet_path, engine='pyarrow')
            # 方案1：避免drop，直接选需要的列（使用reindex避免copy）
            needed_cols = ['lat', 'lon', 'cooling_demand']
            if 'datetime' in df_hourly.columns:
                needed_cols.append('datetime')
            elif 'date' in df_hourly.columns and 'time' in df_hourly.columns:
                needed_cols.extend(['date', 'time'])
            df_hourly = df_hourly.reindex(columns=needed_cols)
        
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
        
        # 创建point_key列（必需，用于后续过滤和映射）
        df_hourly['point_key'] = list(zip(df_hourly['lat'], df_hourly['lon']))
        
        # 修复pickle序列化问题：将datetime列转换为字符串，避免跨进程传递时的序列化错误
        # 方案1：避免drop，直接选需要的列（必须包含point_key，使用reindex避免copy）
        if 'datetime' in df_hourly.columns:
            df_hourly['datetime_str'] = df_hourly['datetime'].astype(str)
            # 只保留必要列：lat, lon, cooling_demand, point_key, datetime_str（使用reindex避免copy）
            needed_cols = ['lat', 'lon', 'cooling_demand', 'point_key', 'datetime_str']
            df_hourly = df_hourly.reindex(columns=needed_cols)
        else:
            # 如果没有datetime，保留date和time（如果存在）
            needed_cols = ['lat', 'lon', 'cooling_demand', 'point_key']
            if 'date' in df_hourly.columns:
                needed_cols.append('date')
            if 'time' in df_hourly.columns:
                needed_cols.append('time')
            df_hourly = df_hourly.reindex(columns=needed_cols)
        
        # 方案3：释放内存 + 强制GC（在子进程中）
        gc.collect()  # 使用全局的 gc 模块
        
        logger.info(f"  ✓ {case_name}: {len(df_hourly)} 条记录")
        gc.collect()  # 在 return 前再加一次，确保彻底清理
        return (case_name, df_hourly)
    
    except Exception as e:
        logger.error(f"读取 {parquet_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        gc.collect()  # 出错也回收
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
    
    # ========== 第五步：按批次处理工况（一次加载11个工况），并行计算国家逐时能耗，立即输出 ==========
    total_start_time = time.time()
    
    # 批次大小：一次加载11个工况（第一次11个，第二次10个）
    # 将工况分成批次：第一次11个（ref + case1-10），第二次10个（case11-20）
    case_batches = [CASES[i:i+CASE_BATCH_SIZE] for i in range(0, len(CASES), CASE_BATCH_SIZE)]
    
    # 按批次处理
    for batch_idx, case_batch in enumerate(case_batches):
        batch_start_time = time.time()
        logger.debug(f"处理批次 {batch_idx + 1}/{len(case_batches)}: {', '.join(case_batch)}")
        
        batch_shared_data = None  # 初始化，用于finally清理
        shm_refs_list = []  # 记录所有创建的共享内存，用于finally清理（即使batch_shared_data未完全初始化）
        
        try:
            # 优化1：并行读取当前批次的所有parquet文件
            parquet_read_start = time.time()
            parquet_args = []
            for case_name in case_batch:
                parquet_path = os.path.join(
                    HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", str(year),
                    "point", case_name, f"{case_name}_hourly.parquet"
                )
                if os.path.exists(parquet_path):
                    parquet_args.append((case_name, parquet_path, model_name, ssp_path, year))
            
            # 并行读取parquet文件（一次读取11个工况）
            batch_case_data = {}  # {case_name: df}
            if parquet_args:
                # 并行读取进程数：限制为28，防 memory burst
                read_processes = min(28, os.cpu_count())
                with Pool(processes=read_processes) as read_pool:
                    read_results = read_pool.map(load_single_parquet, parquet_args)
                
                for case_name, df_case in read_results:
                    if df_case is not None:
                        # 恢复datetime列（从字符串转换回datetime，避免pickle序列化问题）
                        # 方案1：避免drop，直接选需要的列（必须包含point_key）
                        if 'datetime_str' in df_case.columns:
                            df_case['datetime'] = pd.to_datetime(df_case['datetime_str'])
                            # 只保留必要列：lat, lon, cooling_demand, point_key, datetime
                            needed_cols = ['lat', 'lon', 'cooling_demand', 'point_key', 'datetime']
                            df_case = df_case[needed_cols].copy()
                        batch_case_data[case_name] = df_case
                        logger.info(f"  {case_name}: 读取了 {len(df_case)} 条记录")
                
                # 立即释放read_results中的DataFrame（避免内存累积）
                del read_results
                gc.collect()
            
            parquet_read_elapsed = time.time() - parquet_read_start
            logger.info(f"  批次 {batch_idx + 1}: 并行读取完成，成功加载 {len(batch_case_data)} 个工况 (耗时: {parquet_read_elapsed:.2f}s)")
            
            if not batch_case_data:
                logger.warning(f"批次 {batch_idx + 1} 没有有效的工况，跳过")
                continue
            
            # 优化2：共享point_key和datetime数组（跨工况）
            # 使用第一个工况的point_key和datetime作为共享数据
            first_case = list(batch_case_data.keys())[0]
            first_df = batch_case_data[first_case]
            
            # 提取共享的point_key和datetime
            shared_point_key = first_df['point_key'].tolist()
            datetime_series = pd.to_datetime(first_df['datetime'])
            datetime_s = (datetime_series.astype('int64') // 1_000_000_000).astype('int64')
            
            # 建立point_key到行索引的映射
            point_key_to_indices = defaultdict(list)
            for i, pk in enumerate(shared_point_key):
                point_key_to_indices[pk].append(i)
            shared_point_key_to_indices = dict(point_key_to_indices)
            
            # 将datetime存入共享内存
            shm_name_dt = f"batch{batch_idx}_dt"
            try:
                shm_dt = shared_memory.SharedMemory(create=True, size=datetime_s.nbytes, name=shm_name_dt)
            except FileExistsError:
                shm_name_dt = f"batch{batch_idx}_dt_{uuid.uuid4().hex[:6]}"
                shm_dt = shared_memory.SharedMemory(create=True, size=datetime_s.nbytes, name=shm_name_dt)
            shm_refs_list.append(shm_dt)  # 记录用于finally清理
            shared_dt_arr = np.ndarray(datetime_s.shape, dtype=datetime_s.dtype, buffer=shm_dt.buf)
            shared_dt_arr[:] = datetime_s[:]
            
            # 转换为共享内存格式（只存储cooling_demand列）
            shm_convert_start = time.time()
            batch_shared_data = {
                '_point_key': shared_point_key,
                '_point_key_to_indices': shared_point_key_to_indices,
                '_datetime_sec': {
                    'type': 'shm',
                    'name': shm_dt.name,
                    'shape': datetime_s.shape,
                    'dtype': str(datetime_s.dtype),
                    'unit': 'seconds'
                },
                '_case_data': {},  # {case_name: cooling_demand共享内存信息}
                '_shm_refs': [shm_dt]  # 初始化共享内存引用列表
            }
            
            # 为每个工况只存储cooling_demand列
            for case_name, df_case in batch_case_data.items():
                if 'cooling_demand' in df_case.columns:
                    cooling_arr = df_case['cooling_demand'].values
                    shm_name = f"batch{batch_idx}_{case_name[:10]}_cd"
                    try:
                        shm = shared_memory.SharedMemory(create=True, size=cooling_arr.nbytes, name=shm_name)
                    except FileExistsError:
                        shm_name = f"batch{batch_idx}_{case_name[:8]}_cd_{uuid.uuid4().hex[:6]}"
                        shm = shared_memory.SharedMemory(create=True, size=cooling_arr.nbytes, name=shm_name)
                    shm_refs_list.append(shm)  # 记录用于finally清理
                    shared_arr = np.ndarray(cooling_arr.shape, dtype=cooling_arr.dtype, buffer=shm.buf)
                    shared_arr[:] = cooling_arr[:]
                    
                    batch_shared_data['_case_data'][case_name] = {
                        'type': 'shm',
                        'name': shm.name,
                        'shape': cooling_arr.shape,
                        'dtype': str(cooling_arr.dtype)
                    }
                    batch_shared_data['_shm_refs'].append(shm)
            
            shm_convert_elapsed = time.time() - shm_convert_start
            logger.info(f"  批次 {batch_idx + 1}: 转换为共享内存格式完成 (耗时: {shm_convert_elapsed:.2f}s)")
            
            valid_cases = list(batch_case_data.keys())
            
            # 释放原始DataFrame
            del batch_case_data
            gc.collect()  # 使用全局的 gc 模块
            
            # 准备所有国家的参数（使用共享内存，包含多个工况）
            prep_start = time.time()
            country_args = []
            for country_code in shared_country_heat_wave_info.keys():
                continent = country_to_continent.get(country_code, 'Unknown')
                country_args.append((
                    country_code,
                    shared_country_heat_wave_info[country_code],  # 共享内存中的映射
                    continent,
                    valid_cases,  # 当前批次的工况列表
                    batch_shared_data,   # 共享内存中的工况数据字典
                    year,
                    model_name,
                    ssp_path
                ))
            prep_elapsed = time.time() - prep_start
            
            # 并行处理所有国家（在当前批次的所有工况下），每个进程直接输出CSV
            logger.info(f"  并行处理 {len(country_args)} 个国家的批次 {batch_idx + 1} 逐时能耗（{NUM_PROCESSES_COUNTRY}个进程）...")
            
            pool_start = time.time()
            with Pool(processes=min(NUM_PROCESSES_COUNTRY, len(country_args))) as pool:
                pool.map(process_single_country_batch_cases, country_args)
            pool_elapsed = time.time() - pool_start
            logger.info(f"  所有国家已处理完成 (耗时: {pool_elapsed:.2f}s)")
        
        finally:
            # 释放共享内存（确保即使中途退出（如OOM）也能执行cleanup）
            cleanup_start = time.time()
            # 优先使用 batch_shared_data['_shm_refs']（如果已完全初始化）
            if batch_shared_data is not None and '_shm_refs' in batch_shared_data:
                for shm in batch_shared_data['_shm_refs']:
                    try:
                        shm.close()
                        shm.unlink()  # 必须加这一行！主进程负责 unlink
                    except Exception as e:
                        logger.debug(f"清理 shm {shm.name} 时出错: {e}")
            else:
                # 如果 batch_shared_data 未完全初始化，使用 shm_refs_list
                for shm in shm_refs_list:
                    try:
                        shm.close()
                        shm.unlink()  # 必须加这一行！主进程负责 unlink
                    except Exception as e:
                        logger.debug(f"清理 shm {shm.name} 时出错: {e}")
            if batch_shared_data is not None:
                del batch_shared_data
            gc.collect()
            cleanup_elapsed = time.time() - cleanup_start
            logger.info(f"  共享内存释放完成 (耗时: {cleanup_elapsed:.2f}s)")
        
        batch_elapsed = time.time() - batch_start_time
        logger.debug(f"批次 {batch_idx + 1} 处理完成，耗时: {batch_elapsed:.2f} 秒")
    
    logger.info(f"所有工况处理完成，总耗时: {time.time() - total_start_time:.2f} 秒")


def process_multiple_years_batch(batch_items):
    """处理一个批次（2年，可能跨SSP）的逐时能耗（方案B：分批读取，并行处理）
    
    Parameters:
    -----------
    batch_items : list
        批次项列表，每个项是 (model_name, ssp_path, year) 的元组
        例如：[(model_name, 'SSP126', 2030), (model_name, 'SSP126', 2031)]
    """
    if not batch_items:
        return
    
    # 提取批次信息
    model_name = batch_items[0][0]  # 所有项应该属于同一个模型
    years_info = [(item[1], item[2]) for item in batch_items]  # [(ssp_path, year), ...]
    
    logger.debug(f"处理批次: {model_name} - {len(batch_items)} 个年份")
    
    # 按SSP分组处理（因为不同SSP的数据路径不同）
    ssp_years_dict = {}  # {ssp_path: [year1, year2, ...]}
    for ssp_path, year in years_info:
        if ssp_path not in ssp_years_dict:
            ssp_years_dict[ssp_path] = []
        ssp_years_dict[ssp_path].append(year)
    
    # 为每个SSP调用process_multiple_years
    for ssp_path, years in ssp_years_dict.items():
        try:
            process_multiple_years(model_name, ssp_path, years)
        except Exception as e:
            logger.error(f"处理批次 {model_name} - {ssp_path} - {years} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue


def process_multiple_years(model_name, ssp_path, years):
    """处理多个年份的逐时能耗（方案B：分批读取，并行处理）
    
    Parameters:
    -----------
    model_name : str
        模型名称
    ssp_path : str
        SSP路径
    years : list
        年份列表
    """
    total_start_time = time.time()
    logger.debug(f"处理 {model_name} - {ssp_path} - {len(years)} 个年份: {years}")
    
    # ========== 第一步：读取共用数据（所有年份共用）==========
    logger.info("读取共用数据（国家信息）...")
    t0 = time.time()
    
    # 读取国家信息（只读取一次）
    countries_info_df = read_csv_with_encoding(COUNTRIES_INFO_CSV)
    country_to_continent = dict(zip(
        countries_info_df['Country_Code_3'],
        countries_info_df['continent']
    ))
    # 立即释放DataFrame，只保留字典
    del countries_info_df
    logger.info(f"[时间统计] 读取共用数据总耗时: {time.time() - t0:.2f} 秒")
    
    # ========== 第二步：为每个年份建立点-国家和热浪映射 ==========
    years_data = {}  # {year: {point_to_country, point_to_heat_wave_dates, country_heat_wave_info_dict}}
    
    for year in years:
        logger.info(f"\n处理 {year} 年的热浪数据...")
        year_start = time.time()
        
        # 文件路径
        heat_wave_file = os.path.join(
            HEAT_WAVE_BASE_PATH, model_name, ssp_path, f"{year}_all_heat_wave.csv"
        )
        
        if not os.path.exists(heat_wave_file):
            logger.warning(f"热浪文件不存在: {heat_wave_file}，跳过该年份")
            continue
        
        # 读取热浪文件
        heat_wave_df = pd.read_csv(heat_wave_file)
        
        # 建立点-国家的映射
        t0 = time.time()
        point_to_country = create_point_to_country_mapping(
            heat_wave_df, SHAPEFILE_PATH, country_to_continent
        )
        logger.info(f"[时间统计] {year}年点-国家映射总耗时: {time.time() - t0:.2f} 秒")
        
        # 建立点-热浪日期的映射
        t0 = time.time()
        point_to_heat_wave_dates = defaultdict(set)
        def parse_date(date_str, year):
            month, day = map(int, date_str.split('/'))
            return datetime(year, month, day)
        
        heat_wave_df['parsed_date'] = heat_wave_df['date'].apply(lambda x: parse_date(x, year))
        
        for _, row in heat_wave_df.iterrows():
            start_date = row['parsed_date']
            duration = int(row['Duration'])
            lat = row['lat']
            lon = row['lon']
            point_key = (lat, lon)
            
            for i in range(duration):
                current_date = start_date + timedelta(days=i)
                point_to_heat_wave_dates[point_key].add(current_date)
        
        logger.info(f"[时间统计] {year}年建立点-热浪日期映射总耗时: {time.time() - t0:.2f} 秒")
        
        # 建立国家-热浪日期-点的映射
        t0 = time.time()
        country_heat_wave_info_dict = {}
        
        for (lat, lon), (country_code, continent) in point_to_country.items():
            if country_code is None:
                continue
            
            point_key = (lat, lon)
            heat_wave_dates = point_to_heat_wave_dates.get(point_key, set())
            
            if country_code not in country_heat_wave_info_dict:
                country_heat_wave_info_dict[country_code] = {}
            
            for date in heat_wave_dates:
                date_str = date.isoformat() if isinstance(date, datetime) else str(date)
                if date_str not in country_heat_wave_info_dict[country_code]:
                    country_heat_wave_info_dict[country_code][date_str] = []
                country_heat_wave_info_dict[country_code][date_str].append(point_key)
        
        logger.info(f"[时间统计] {year}年建立国家-热浪日期-点映射总耗时: {time.time() - t0:.2f} 秒")
        
        years_data[year] = {
            'point_to_country': point_to_country,
            'point_to_heat_wave_dates': point_to_heat_wave_dates,
            'country_heat_wave_info_dict': country_heat_wave_info_dict
        }
        
        logger.info(f"{year}年数据准备完成，耗时: {time.time() - year_start:.2f} 秒")
    
    if not years_data:
        logger.warning("没有有效的年份数据，退出")
        return
    
    # ========== 第三步：创建共享内存管理器 ==========
    manager = Manager()
    
    # 为每个年份创建共享字典
    shared_years_data = {}
    for year, year_info in years_data.items():
        shared_country_heat_wave_info = manager.dict()
        for country_code, dates_dict in year_info['country_heat_wave_info_dict'].items():
            shared_dates_dict = manager.dict()
            for date_str, points_list in dates_dict.items():
                shared_dates_dict[date_str] = points_list
            shared_country_heat_wave_info[country_code] = shared_dates_dict
        shared_years_data[year] = shared_country_heat_wave_info
    
    # ========== 第四步：分批读取parquet文件（每批11个工况，一年2批）==========
    logger.info("开始分批读取parquet文件...")
    
    # 存储所有年份的共享数据
    all_years_shared_data = {}  # {year: batch_shared_data}
    
    # 共享的point_key（所有年份共用，因为全球网格点相同）
    shared_point_key = None
    shared_point_key_to_indices = None
    
    # 按年份分批读取
    for year_idx, year in enumerate(years):
        if year not in years_data:
            continue
        
        logger.info(f"\n处理 {year} 年的parquet文件（批次 {year_idx + 1}/{len(years)}）...")
        year_read_start = time.time()
        
        # 读取该年份的所有21个工况（分2批处理：第一批11个，第二批10个）
        # 注意：process_multiple_years目前仍读取所有21个工况，但实际处理时会按批次
        # 如果需要，可以在这里也改为分批读取
        parquet_args = []
        for case_name in CASES:
            parquet_path = os.path.join(
                HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", str(year),
                "point", case_name, f"{case_name}_hourly.parquet"
            )
            if os.path.exists(parquet_path):
                parquet_args.append((case_name, parquet_path, model_name, ssp_path, year))
        
        # 并行读取parquet文件（注意：这里仍读取所有21个，但实际处理时会分批）
        batch_case_data = {}
        if parquet_args:
            read_processes = min(len(parquet_args), 7, os.cpu_count())
            with Pool(processes=read_processes) as read_pool:
                read_results = read_pool.map(load_single_parquet, parquet_args)
            
            for case_name, df_case in read_results:
                if df_case is not None:
                    # 方案1：避免drop，直接选需要的列（必须包含point_key）
                    if 'datetime_str' in df_case.columns:
                        df_case['datetime'] = pd.to_datetime(df_case['datetime_str'])
                        # 只保留必要列：lat, lon, cooling_demand, point_key, datetime
                        needed_cols = ['lat', 'lon', 'cooling_demand', 'point_key', 'datetime']
                        df_case = df_case[needed_cols].copy()
                    batch_case_data[case_name] = df_case
        
        if not batch_case_data:
            logger.warning(f"{year}年没有有效的工况数据，跳过")
            continue
        
        logger.info(f"  {year}年: 成功加载 {len(batch_case_data)} 个工况")
        
        # 提取共享的point_key和datetime（使用第一个工况）
        first_case = list(batch_case_data.keys())[0]
        first_df = batch_case_data[first_case]
        
        # 提取共享的point_key（所有年份共享，因为全球网格点相同）
        if shared_point_key is None:
            # 第一年提取point_key，后续年份复用
            shared_point_key = first_df['point_key'].tolist()
            point_key_to_indices = defaultdict(list)
            for i, pk in enumerate(shared_point_key):
                point_key_to_indices[pk].append(i)
            shared_point_key_to_indices = dict(point_key_to_indices)
            logger.info(f"  {year}年: 提取共享point_key，共 {len(shared_point_key)} 个点")
        else:
            # 后续年份复用第一年的point_key
            logger.debug(f"  {year}年: 复用共享point_key")
        
        # 提取该年份的datetime（每年独立）
        datetime_series = pd.to_datetime(first_df['datetime'])
        datetime_s = (datetime_series.astype('int64') // 1_000_000_000).astype('int64')
        
        # 将datetime存入共享内存
        shm_name_dt = f"year{year}_dt"
        try:
            shm_dt = shared_memory.SharedMemory(create=True, size=datetime_s.nbytes, name=shm_name_dt)
        except FileExistsError:
            shm_name_dt = f"year{year}_dt_{uuid.uuid4().hex[:6]}"
            shm_dt = shared_memory.SharedMemory(create=True, size=datetime_s.nbytes, name=shm_name_dt)
        shared_dt_arr = np.ndarray(datetime_s.shape, dtype=datetime_s.dtype, buffer=shm_dt.buf)
        shared_dt_arr[:] = datetime_s[:]
        
        # 转换为共享内存格式（只存储cooling_demand列）
        batch_shared_data = {
            '_point_key': shared_point_key,  # 共享（第一年提取）
            '_point_key_to_indices': shared_point_key_to_indices,  # 共享
            '_datetime_sec': {
                'type': 'shm',
                'name': shm_dt.name,
                'shape': datetime_s.shape,
                'dtype': str(datetime_s.dtype),
                'unit': 'seconds'
            },
            '_case_data': {},
            '_shm_refs': [shm_dt]
        }
        
        # 为每个工况只存储cooling_demand列
        for case_name, df_case in batch_case_data.items():
            if 'cooling_demand' in df_case.columns:
                cooling_arr = df_case['cooling_demand'].values
                shm_name = f"year{year}_{case_name[:10]}_cd"
                try:
                    shm = shared_memory.SharedMemory(create=True, size=cooling_arr.nbytes, name=shm_name)
                except FileExistsError:
                    shm_name = f"year{year}_{case_name[:8]}_cd_{uuid.uuid4().hex[:6]}"
                    shm = shared_memory.SharedMemory(create=True, size=cooling_arr.nbytes, name=shm_name)
                shared_arr = np.ndarray(cooling_arr.shape, dtype=cooling_arr.dtype, buffer=shm.buf)
                shared_arr[:] = cooling_arr[:]
                
                batch_shared_data['_case_data'][case_name] = {
                    'type': 'shm',
                    'name': shm.name,
                    'shape': cooling_arr.shape,
                    'dtype': str(cooling_arr.dtype)
                }
                batch_shared_data['_shm_refs'].append(shm)
        
        all_years_shared_data[year] = batch_shared_data
        
        # 释放原始DataFrame
        del batch_case_data
        gc.collect()
        
        logger.info(f"  {year}年: 转换为共享内存格式完成，耗时: {time.time() - year_read_start:.2f} 秒")
    
    logger.info(f"\n所有年份数据加载完成，共 {len(all_years_shared_data)} 个年份")
    
    # ========== 第五步：并行处理所有国家（所有年份）==========
    logger.info("开始并行处理所有国家的逐时能耗...")
    
    # 收集所有国家的列表（所有年份的并集）
    all_countries = set()
    for year_data in shared_years_data.values():
        all_countries.update(year_data.keys())
    
    # 准备所有国家的参数
    country_args = []
    for country_code in all_countries:
        continent = country_to_continent.get(country_code, 'Unknown')
        # 为每个国家收集所有年份的热浪数据
        country_years_data = {}
        for year in years:
            if year in shared_years_data and country_code in shared_years_data[year]:
                country_years_data[year] = shared_years_data[year][country_code]
        
        if country_years_data:
            country_args.append((
                country_code,
                country_years_data,  # {year: shared_country_heat_wave_dates_dict}
                continent,
                years,  # 年份列表
                all_years_shared_data,  # {year: batch_shared_data}
                model_name,
                ssp_path
            ))
    
    # 并行处理所有国家
    logger.info(f"并行处理 {len(country_args)} 个国家的逐时能耗（{NUM_PROCESSES_COUNTRY}个进程）...")
    
    pool_start = time.time()
    with Pool(processes=min(NUM_PROCESSES_COUNTRY, len(country_args))) as pool:
        pool.map(process_single_country_multiple_years, country_args)
    pool_elapsed = time.time() - pool_start
    logger.info(f"所有国家已处理完成 (耗时: {pool_elapsed:.2f}s)")
    
    # ========== 第六步：彻底释放所有内存，确保不影响下一次计算 ==========
    cleanup_start = time.time()
    logger.info("开始释放内存...")
    
    # 1. 释放所有共享内存
    logger.info("释放共享内存...")
    for year, batch_shared_data in all_years_shared_data.items():
        if '_shm_refs' in batch_shared_data:
            for shm in batch_shared_data['_shm_refs']:
                try:
                    shm.close()
                    shm.unlink()
                except Exception as e:
                    logger.debug(f"释放共享内存 {shm.name} 时出错: {e}")
                    pass
    
    # 2. 删除所有大对象
    logger.info("删除大对象...")
    del all_years_shared_data
    del shared_years_data
    del years_data
    del country_to_continent
    del country_args
    # 释放共享的point_key（如果存在）
    if 'shared_point_key' in locals():
        del shared_point_key
    if 'shared_point_key_to_indices' in locals():
        del shared_point_key_to_indices
    
    # 3. 关闭Manager（释放Manager创建的共享字典）
    try:
        manager.shutdown()
    except:
        pass
    del manager
    
    # 4. 强制垃圾回收（多次调用确保彻底清理）
    logger.info("执行垃圾回收...")
    for i in range(3):
        collected = gc.collect()
    
    cleanup_elapsed = time.time() - cleanup_start
    logger.info(f"内存释放完成 (耗时: {cleanup_elapsed:.2f}s)")
    
    total_elapsed = time.time() - total_start_time
    logger.info(f"所有年份处理完成，总耗时: {total_elapsed:.2f} 秒")


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
        prep_start_time = time.time()
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
        prep_elapsed = time.time() - prep_start_time
        
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
            result_series = df_filtered.groupby('datetime')['cooling_demand'].sum()
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
            
            # 不打印每个国家的保存信息，统一在最后打印
        
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


def process_single_country_batch_cases(args):
    """并行处理单个国家在多个工况（批次）下的逐时能耗计算，输出合并的CSV
    优化版本：使用共享的point_key和datetime，只读取cooling_demand列
    
    Parameters:
    -----------
    args : tuple
        (country_code, shared_country_heat_wave_dates_dict, continent, case_names, batch_shared_data,
         year, model_name, ssp_path)
    """
    (country_code, shared_country_heat_wave_dates_dict, continent, case_names, batch_shared_data,
     year, model_name, ssp_path) = args
    
    existing_shm_dt = None
    existing_shm_list = []  # 记录所有打开的共享内存，用于finally清理
    
    try:
        # 计时开始
        total_start_time = time.time()
        
        # Step 1: 收集该国家所有热浪点 + 日期
        prep_start_time = time.time()
        all_points = set()
        date_point_set = set()
        
        for date_str, points in shared_country_heat_wave_dates_dict.items():
            try:
                if isinstance(date_str, str):
                    date = pd.to_datetime(date_str).date()
                else:
                    date = date_str
            except Exception as e:
                logger.warning(f"无法解析日期: {date_str}, 错误: {e}")
                continue
            
            for pt in points:
                all_points.add(pt)
                date_point_set.add((date, pt))
        prep_elapsed = time.time() - prep_start_time
        
        # Step 2: 使用共享的point_key和datetime进行过滤（只做一次）
        read_start_time = time.time()
        
        # 获取共享的point_key和datetime
        global_point_key = batch_shared_data['_point_key']  # list of (lat, lon)
        point_key_to_indices = batch_shared_data['_point_key_to_indices']
        
        # 从共享内存读取datetime
        dt_info = batch_shared_data['_datetime_sec']
        existing_shm_dt = shared_memory.SharedMemory(name=dt_info['name'])
        existing_shm_list.append(existing_shm_dt)  # 记录用于finally清理
        global_datetime_sec = np.ndarray(dt_info['shape'], dtype=dt_info['dtype'], buffer=existing_shm_dt.buf)
        
        # 转换为date（一次性转换，避免重复）
        global_datetime_series = pd.to_datetime(global_datetime_sec, unit='s')
        global_date_series = pd.Series(global_datetime_series).dt.date
        global_date_list = global_date_series.tolist()  # 转换为列表，便于索引访问
        
        # 使用预建立的映射快速查找行索引
        row_indices_list = []
        for point in all_points:
            if point in point_key_to_indices:
                row_indices_list.extend(point_key_to_indices[point])
        
        if not row_indices_list:
            # 无匹配的点，直接返回
            read_elapsed = time.time() - read_start_time
            logger.info(f"  ✓ {country_code} - 批次: 无热浪数据 | 准备: {prep_elapsed:.2f}s, 读取: {read_elapsed:.2f}s, 总计: {time.time() - total_start_time:.2f}s")
            existing_shm_dt.close()
            return
        
        candidate_indices = np.array(row_indices_list, dtype=np.int64)
        
        # 进一步筛选：只保留热浪日期+热浪点的组合
        candidate_dates = [global_date_list[i] for i in candidate_indices]
        candidate_points = [global_point_key[i] for i in candidate_indices]
        
        final_indices_list = [
            candidate_indices[i] for i in range(len(candidate_indices))
            if (candidate_dates[i], candidate_points[i]) in date_point_set
        ]
        
        if not final_indices_list:
            read_elapsed = time.time() - read_start_time
            logger.info(f"  ✓ {country_code} - 批次: 无热浪数据 | 准备: {prep_elapsed:.2f}s, 读取: {read_elapsed:.2f}s, 总计: {time.time() - total_start_time:.2f}s")
            existing_shm_dt.close()
            return
        
        final_indices = np.array(final_indices_list, dtype=np.int64)
        final_datetime_sec = global_datetime_sec[final_indices]
        
        # 获取唯一的时间点，用于分组
        unique_times, inverse = np.unique(final_datetime_sec, return_inverse=True)
        
        read_elapsed = time.time() - read_start_time
        
        # Step 3: 对每个工况，从共享内存读取cooling_demand并累加
        calc_start_time = time.time()
        batch_results = {}  # {case_name: {datetime_sec: energy_sum}}
        
        for case_name in case_names:
            if case_name not in batch_shared_data.get('_case_data', {}):
                continue
            
            # 从共享内存读取cooling_demand
            case_info = batch_shared_data['_case_data'][case_name]
            existing_shm = shared_memory.SharedMemory(name=case_info['name'])
            existing_shm_list.append(existing_shm)  # 记录用于finally清理
            demand_arr = np.ndarray(case_info['shape'], dtype=case_info['dtype'], buffer=existing_shm.buf)
            
            # 提取该国家的cooling_demand
            case_demands = demand_arr[final_indices]
            
            # 向量化groupby sum（使用np.add.at）
            case_energy = np.zeros(len(unique_times))
            np.add.at(case_energy, inverse, case_demands)
            
            # 转换为字典，key为datetime的isoformat字符串
            unique_datetime_series = pd.to_datetime(unique_times, unit='s')
            batch_results[case_name] = {
                dt.isoformat(): float(energy) 
                for dt, energy in zip(unique_datetime_series, case_energy)
            }
            
            existing_shm.close()
        
        existing_shm_dt.close()
        calc_elapsed = time.time() - calc_start_time
        
        # Step 4: 合并所有工况的结果到一个CSV
        if batch_results:
            save_start_time = time.time()
            
            # 创建时间序列（该年份的所有小时）
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31, 23, 0, 0)
            time_range = pd.date_range(start=start_date, end=end_date, freq='h')
            
            # 创建结果DataFrame，包含所有工况的列
            result_data = []
            for dt in time_range:
                dt_str = dt.isoformat()
                row_data = {'time': dt}
                # 添加所有工况的能耗
                for case_name in case_names:
                    if case_name in batch_results:
                        energy = batch_results[case_name].get(dt_str, 0.0)
                        row_data[case_name] = energy
                result_data.append(row_data)
            
            result_df = pd.DataFrame(result_data)
            
            # 保存CSV（文件名包含批次信息）
            output_dir = os.path.join(
                HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", "hourly_energy",
                continent, country_code
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # 文件名：根据批次生成文件名
            # 第一次批次（11个）：ref + case1-10 -> batch1
            # 第二次批次（10个）：case11-20 -> batch2
            if len(case_names) == 11:
                if case_names[0] == 'ref' and case_names[-1] == 'case10':
                    batch_name = "batch1"  # 第一批：ref + case1-10
                else:
                    batch_name = f"batch_{case_names[0]}_{case_names[-1]}"
            elif len(case_names) == 10:
                if case_names[0] == 'case11' and case_names[-1] == 'case20':
                    batch_name = "batch2"  # 第二批：case11-20
                else:
                    batch_name = f"batch_{case_names[0]}_{case_names[-1]}"
            else:
                batch_name = f"batch_{case_names[0]}_{case_names[-1]}"
            output_file = os.path.join(output_dir, f"{country_code}_{year}_{batch_name}_hourly_energy.csv")
            result_df.to_csv(output_file, index=False)
            
            save_elapsed = time.time() - save_start_time
            total_elapsed = time.time() - total_start_time
            
            # 计算总时间点数
            total_time_points = sum(len(energy_dict) for energy_dict in batch_results.values())
            
            # 打印每个国家的已保存信息（所有工况处理完成后）
            logger.info(f"  ✓ {country_code}: 已保存")
    
    except Exception as e:
        logger.error(f"处理国家 {country_code} - 批次({case_names}) 时出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 释放本进程打开的shm（即使出错也要释放）
        # ▲ 不要 unlink! 主进程统一管理
        for shm in existing_shm_list:
            try:
                if shm is not None:
                    shm.close()
            except:
                pass
        if existing_shm_dt is not None:
            try:
                existing_shm_dt.close()
            except:
                pass


def process_single_country_multiple_years(args):
    """并行处理单个国家在多个年份下的逐时能耗计算，输出按年份分别保存CSV
    
    Parameters:
    -----------
    args : tuple
        (country_code, country_years_data, continent, years, all_years_shared_data,
         model_name, ssp_path)
        country_years_data: {year: shared_country_heat_wave_dates_dict}
        all_years_shared_data: {year: batch_shared_data}
    """
    (country_code, country_years_data, continent, years, all_years_shared_data,
     model_name, ssp_path) = args
    
    existing_shm_dt = None
    existing_shm_list = []  # 记录所有打开的共享内存，用于finally清理
    
    try:
        total_start_time = time.time()
        
        # 对每个年份分别处理
        for year in years:
            if year not in country_years_data or year not in all_years_shared_data:
                continue
            
            year_start_time = time.time()
            
            # 获取该年份的数据
            shared_country_heat_wave_dates_dict = country_years_data[year]
            batch_shared_data = all_years_shared_data[year]
            case_names = list(batch_shared_data.get('_case_data', {}).keys())
            
            if not case_names:
                continue
            
            # Step 1: 收集该国家该年份所有热浪点 + 日期
            prep_start_time = time.time()
            all_points = set()
            date_point_set = set()
            
            for date_str, points in shared_country_heat_wave_dates_dict.items():
                try:
                    if isinstance(date_str, str):
                        date = pd.to_datetime(date_str).date()
                    else:
                        date = date_str
                except Exception as e:
                    logger.warning(f"无法解析日期: {date_str}, 错误: {e}")
                    continue
                
                for pt in points:
                    all_points.add(pt)
                    date_point_set.add((date, pt))
            prep_elapsed = time.time() - prep_start_time
            
            # Step 2: 使用共享的point_key和datetime进行过滤
            read_start_time = time.time()
            
            global_point_key = batch_shared_data['_point_key']
            point_key_to_indices = batch_shared_data['_point_key_to_indices']
            
            dt_info = batch_shared_data['_datetime_sec']
            existing_shm_dt = shared_memory.SharedMemory(name=dt_info['name'])
            existing_shm_list.append(existing_shm_dt)  # 记录用于finally清理
            global_datetime_sec = np.ndarray(dt_info['shape'], dtype=dt_info['dtype'], buffer=existing_shm_dt.buf)
            
            global_datetime_series = pd.to_datetime(global_datetime_sec, unit='s')
            global_date_series = pd.Series(global_datetime_series).dt.date
            global_date_list = global_date_series.tolist()
            
            row_indices_list = []
            for point in all_points:
                if point in point_key_to_indices:
                    row_indices_list.extend(point_key_to_indices[point])
            
            if not row_indices_list:
                existing_shm_dt.close()
                continue
            
            candidate_indices = np.array(row_indices_list, dtype=np.int64)
            candidate_dates = [global_date_list[i] for i in candidate_indices]
            candidate_points = [global_point_key[i] for i in candidate_indices]
            
            final_indices_list = [
                candidate_indices[i] for i in range(len(candidate_indices))
                if (candidate_dates[i], candidate_points[i]) in date_point_set
            ]
            
            if not final_indices_list:
                existing_shm_dt.close()
                continue
            
            final_indices = np.array(final_indices_list, dtype=np.int64)
            final_datetime_sec = global_datetime_sec[final_indices]
            unique_times, inverse = np.unique(final_datetime_sec, return_inverse=True)
            
            read_elapsed = time.time() - read_start_time
            
            # Step 3: 对每个工况，从共享内存读取cooling_demand并累加
            calc_start_time = time.time()
            batch_results = {}
            
            for case_name in case_names:
                if case_name not in batch_shared_data.get('_case_data', {}):
                    continue
                
                case_info = batch_shared_data['_case_data'][case_name]
                existing_shm = shared_memory.SharedMemory(name=case_info['name'])
                existing_shm_list.append(existing_shm)  # 记录用于finally清理
                demand_arr = np.ndarray(case_info['shape'], dtype=case_info['dtype'], buffer=existing_shm.buf)
                
                case_demands = demand_arr[final_indices]
                case_energy = np.zeros(len(unique_times))
                np.add.at(case_energy, inverse, case_demands)
                
                unique_datetime_series = pd.to_datetime(unique_times, unit='s')
                batch_results[case_name] = {
                    dt.isoformat(): float(energy) 
                    for dt, energy in zip(unique_datetime_series, case_energy)
                }
                
                existing_shm.close()
            
            existing_shm_dt.close()
            calc_elapsed = time.time() - calc_start_time
            
            # Step 4: 保存该年份的CSV
            if batch_results:
                save_start_time = time.time()
                
                start_date = datetime(year, 1, 1)
                end_date = datetime(year, 12, 31, 23, 0, 0)
                time_range = pd.date_range(start=start_date, end=end_date, freq='h')
                
                result_data = []
                for dt in time_range:
                    dt_str = dt.isoformat()
                    row_data = {'time': dt}
                    for case_name in case_names:
                        if case_name in batch_results:
                            energy = batch_results[case_name].get(dt_str, 0.0)
                            row_data[case_name] = energy
                    result_data.append(row_data)
                
                result_df = pd.DataFrame(result_data)
                
                output_dir = os.path.join(
                    HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", "hourly_energy",
                    continent, country_code
                )
                os.makedirs(output_dir, exist_ok=True)
                
                # 文件名：根据批次生成文件名
                if len(case_names) == 11:
                    if case_names[0] == 'ref' and case_names[-1] == 'case10':
                        batch_name = "batch1"  # 第一批：ref + case1-10
                    else:
                        batch_name = f"batch_{case_names[0]}_{case_names[-1]}"
                elif len(case_names) == 10:
                    if case_names[0] == 'case11' and case_names[-1] == 'case20':
                        batch_name = "batch2"  # 第二批：case11-20
                    else:
                        batch_name = f"batch_{case_names[0]}_{case_names[-1]}"
                else:
                    batch_name = f"batch_{case_names[0]}_{case_names[-1]}"
                output_file = os.path.join(output_dir, f"{country_code}_{year}_{batch_name}_hourly_energy.csv")
                result_df.to_csv(output_file, index=False)
                
                save_elapsed = time.time() - save_start_time
                year_elapsed = time.time() - year_start_time
                total_time_points = sum(len(energy_dict) for energy_dict in batch_results.values())
                
                # 打印每个国家的已保存信息（所有工况处理完成后）
                logger.info(f"  ✓ {country_code} - {year}: 已保存")
        
        total_elapsed = time.time() - total_start_time
        logger.debug(f"  国家 {country_code}: 所有年份处理完成，总耗时: {total_elapsed:.2f}s")
    
    except Exception as e:
        logger.error(f"处理国家 {country_code} - 多年份({years}) 时出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 释放本进程打开的shm（即使出错也要释放）
        # ▲ 不要 unlink! 主进程统一管理
        for shm in existing_shm_list:
            try:
                if shm is not None:
                    shm.close()
            except:
                pass
        if existing_shm_dt is not None:
            try:
                existing_shm_dt.close()
            except:
                pass


def generate_batches():
    """生成批次列表，每批2年（可能跨SSP）
    
    返回:
    --------
    list: 批次列表，每个批次是 [(model_name, ssp_path, year), ...] 的列表
    """
    batches = []
    
    for model_name in MODELS:
        # 收集所有需要处理的项：(model_name, ssp_path, year)
        all_items = []
        for ssp_path in SSP_PATHS:
            for year in TARGET_YEARS:
                all_items.append((model_name, ssp_path, year))
        
        # 按批次大小分组（每批2年）
        for i in range(0, len(all_items), BATCH_YEARS):
            batch = all_items[i:i+BATCH_YEARS]
            batches.append(batch)
    
    return batches


def main():
    """主函数"""
    try:
        logger.info("=== 开始按国家计算热浪期间的逐时能耗 ===")
        logger.info(f"处理的模型: {', '.join(MODELS)}")
        logger.info(f"SSP路径: {', '.join(SSP_PATHS)}")
        logger.info(f"目标年份: {TARGET_YEARS}")
        logger.info(f"处理模式: 一次处理1年（{len(CASES)} 个工况），按顺序处理")
        
        # 按顺序处理：先处理一个SSP的所有年份，再处理下一个SSP
        for model_name in MODELS:
            for ssp_path in SSP_PATHS:
                logger.info(f"\n开始处理 {model_name} - {ssp_path} 的所有年份 ({len(TARGET_YEARS)} 年)")
                
                for year in TARGET_YEARS:
                    logger.info(f"处理 {model_name} - {ssp_path} - {year} 年...")
                    
                    try:
                        # 使用process_single_year处理单年数据
                        process_single_year(model_name, ssp_path, year)
                        
                        # 每个年份处理完后，确保内存清理
                        for i in range(2):
                            collected = gc.collect()
                        
                    except Exception as e:
                        logger.error(f"处理 {model_name} - {ssp_path} - {year} 年时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        # 即使出错也要清理内存
                        for i in range(2):
                            gc.collect()
                        continue
                
                # 一个SSP的所有年份处理完成后，执行最终清理
                logger.info(f"{model_name} - {ssp_path} 的所有年份处理完成")
                for i in range(2):
                    gc.collect()
        
        logger.info("\n=== 所有处理完成 ===")
    
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

