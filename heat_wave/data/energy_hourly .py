"""
按国家聚合逐时能耗（简化版）

功能：
1. 根据shp文件划分国家，获得点到国家的映射
2. 加载ref和case6-10的parquet数据
3. 按国家聚合：将属于这个国家的点的能耗数据按时间都加起来
4. 输出每个国家的逐时能耗CSV

输入数据：
1. 逐时能耗：/home/linbor/WORK/lishiying/heat_wave/{模型名}/{SSP}/energy/{年份}/point/{case名}/{case名}_hourly.parquet
2. 国家边界：/home/linbor/WORK/lishiying/shapefiles/world_border2.shp
3. 国家信息：/home/linbor/WORK/lishiying/shapefiles/all_countries_info.csv

输出数据：
/home/linbor/WORK/lishiying/heat_wave/{模型名}/{SSP}/energy/hourly_energy_v2/{大洲}/{国家代码}/{国家代码}_{年份}_hourly_energy.csv
"""

import pandas as pd
import geopandas as gpd
import os
import numpy as np
import logging
from shapely.geometry import Point
import pyarrow.parquet as pq
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import time
from multiprocessing import Pool, Manager
from functools import partial
from multiprocessing import shared_memory
import struct
import uuid
from operator import itemgetter
import gc
import pickle

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
P2C_CACHE_PATH = os.path.join(BASE_PATH, "cache", "point_to_country.pkl")

# 模型配置
MODELS = ["BCC-CSM2-MR"]  # 可以修改为其他模型

# SSP配置
SSP_PATHS = ["SSP126", "SSP245"]  # 所有SSP

# 年份配置
TARGET_YEARS = [2030, 2031, 2032, 2033, 2034]  # 所有年份

# Case配置：只处理ref和case6-10
CASES = ['ref'] + [f'case{i}' for i in range(6, 11)]  # ['ref', 'case6', 'case7', 'case8', 'case9', 'case10']

# 测试国家：None表示处理所有国家
TEST_COUNTRY = None

# 并行处理配置
NUM_PROCESSES_COUNTRY = 6  # 计算国家逐时能耗的进程数


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


def create_point_to_country_mapping_from_parquet(parquet_path, shp_path, country_to_continent, use_cache=True):
    """从parquet文件创建点到国家的映射（支持缓存）
    
    Parameters:
    -----------
    parquet_path : str
        第一个parquet文件路径（用于获取所有点的坐标）
    shp_path : str
        国家边界shp文件路径
    country_to_continent : dict
        国家代码到大洲的映射（已读取）
    use_cache : bool
        是否使用缓存
    
    Returns:
    --------
    dict: {(lat, lon): (country_code, continent)}
    """
    # 检查缓存
    if use_cache and os.path.exists(P2C_CACHE_PATH):
        logger.info(f"加载缓存的 point_to_country 映射: {P2C_CACHE_PATH}")
        try:
            with open(P2C_CACHE_PATH, 'rb') as f:
                point_to_country = pickle.load(f)
            logger.info(f"成功加载缓存，共 {len(point_to_country)} 个点")
            return point_to_country
        except Exception as e:
            logger.warning(f"加载缓存失败，将重新计算: {e}")
    
    start_time = time.time()
    logger.info("创建点到国家的映射...")
    
    # 从parquet文件获取所有点（只读取lat和lon列）
    logger.info(f"从parquet文件读取点坐标: {parquet_path}")
    try:
        df_sample = pd.read_parquet(parquet_path, columns=['lat', 'lon'], engine='pyarrow')
        unique_points = df_sample[['lat', 'lon']].drop_duplicates()
        logger.info(f"共 {len(unique_points)} 个唯一坐标点")
    except Exception as e:
        logger.error(f"读取parquet文件失败: {e}")
        raise
    
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
    
    # 保存缓存
    if use_cache:
        cache_dir = os.path.dirname(P2C_CACHE_PATH)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        try:
            with open(P2C_CACHE_PATH, 'wb') as f:
                pickle.dump(point_to_country, f)
            logger.info(f"点-国家映射缓存已保存至 {P2C_CACHE_PATH}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    return point_to_country


def pack_dataframe_to_single_shm(df, case_name):
    """将整个DataFrame打包为单块共享内存 + 元数据索引
    
    支持：所有数值列 + 'datetime' (as int64 seconds)
    注意：不存储 point_key，因为子进程通过 point_to_rows 映射直接获取行索引
    
    Parameters:
    -----------
    df : pd.DataFrame
        要转换的DataFrame，必须包含 'cooling_demand', 'datetime'
    case_name : str
        工况名称，用于生成共享内存名称
    
    Returns:
    --------
    tuple: (shm_name, meta, shm_ref)
        shm_name: 共享内存名称
        meta: 元数据字典（可序列化）
        shm_ref: SharedMemory对象引用（主进程保留）
    """
    # 1. 预处理：确保必要的列存在
    if 'cooling_demand' not in df.columns:
        raise ValueError("DataFrame必须包含 'cooling_demand' 列")
    if 'datetime' not in df.columns:
        raise ValueError("DataFrame必须包含 'datetime' 列")
    
    # 2. datetime → int64 seconds since epoch (UTC)
    datetime_series = pd.to_datetime(df['datetime'])
    datetime_seconds = (datetime_series.astype('int64') // 1_000_000_000).astype(np.int64)
    
    # 3. 构建结构化数组的dtype
    dtypes = [
        ('cooling_demand', '<f8'),  # float64
        ('datetime_seconds', '<i8')  # int64
    ]
    
    # 4. 添加其他数值列（如果有）
    other_numeric_cols = []
    for col in df.columns:
        if col not in ['lat', 'lon', 'point_key', 'datetime', 'cooling_demand']:
            if np.issubdtype(df[col].dtype, np.number):
                dtypes.append((col, df[col].dtype))
                other_numeric_cols.append(col)
    
    # 5. 创建结构化数组
    arr = np.empty(len(df), dtype=dtypes)
    arr['cooling_demand'] = df['cooling_demand'].values
    arr['datetime_seconds'] = datetime_seconds
    
    for col in other_numeric_cols:
        arr[col] = df[col].values
    
    # 6. 创建单块共享内存
    total_bytes = arr.nbytes
    shm_name = f"case_{case_name}_{uuid.uuid4().hex[:8]}"
    try:
        shm = shared_memory.SharedMemory(create=True, size=total_bytes, name=shm_name)
    except FileExistsError:
        # 如果名称冲突，添加更多随机字符
        shm_name = f"case_{case_name}_{uuid.uuid4().hex[:12]}"
        shm = shared_memory.SharedMemory(create=True, size=total_bytes, name=shm_name)
    
    shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shared_arr[:] = arr[:]
    
    # 7. 元数据（可序列化）
    meta = {
        'shape': arr.shape,
        'dtype': arr.dtype.descr,  # 结构化dtype的描述符（可序列化）
        'columns': ['cooling_demand', 'datetime_seconds'] + other_numeric_cols,
        'case_name': case_name,
        'nrows': len(df)
    }
    
    return shm_name, meta, shm


def read_rows_from_shm(shm_name, meta, row_indices=None):
    """从单块共享内存中读取指定行的数据
    
    Parameters:
    -----------
    shm_name : str
        共享内存名称
    meta : dict
        元数据字典，包含 shape, dtype, columns 等
    row_indices : np.ndarray, optional
        要读取的行索引。如果为None，读取全部行
    
    Returns:
    --------
    tuple: (shm_ref, arr_view)
        shm_ref: SharedMemory对象引用（需要保持引用）
        arr_view: numpy数组视图（如果row_indices不为None，是筛选后的数组）
    """
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(meta['shape'], dtype=np.dtype(meta['dtype']), buffer=existing_shm.buf)
    
    if row_indices is not None:
        # 只返回需要的行（这是视图，不复制）
        arr_view = arr[row_indices]
    else:
        arr_view = arr
    
    return existing_shm, arr_view


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
        
        # 只读取必要的列
        columns_needed = ['lat', 'lon', 'cooling_demand', 'date', 'time']
        
        try:
            df_hourly = pd.read_parquet(
                parquet_path,
                columns=columns_needed,
                engine='pyarrow'
            )
        except Exception as e:
            logger.warning(f"指定列读取失败，回退全读: {e}")
            df_hourly = pd.read_parquet(parquet_path, engine='pyarrow')
            needed_cols = ['lat', 'lon', 'cooling_demand']
            if 'datetime' in df_hourly.columns:
                needed_cols.append('datetime')
            elif 'date' in df_hourly.columns and 'time' in df_hourly.columns:
                needed_cols.extend(['date', 'time'])
            df_hourly = df_hourly.reindex(columns=needed_cols)
        
        if df_hourly.empty:
            logger.warning(f"文件为空: {parquet_path}")
            return (case_name, None)
        
        # 合并date和time列创建datetime
        if 'date' in df_hourly.columns and 'time' in df_hourly.columns:
            try:
                date_series = pd.to_datetime(df_hourly['date'])
                date_str = date_series.dt.strftime('%Y-%m-%d')
            except Exception as e:
                logger.error(f"处理date列时出错: {e}")
                try:
                    date_str = df_hourly['date'].astype(str).str[:10]
                    date_series = pd.to_datetime(date_str)
                except Exception as e2:
                    logger.error(f"备用date转换也失败: {e2}")
                    return (case_name, None)
            
            try:
                if hasattr(df_hourly['time'].iloc[0], 'strftime'):
                    time_str = df_hourly['time'].apply(lambda x: x.strftime('%H:%M:%S') if pd.notna(x) else '00:00:00')
                elif isinstance(df_hourly['time'].iloc[0], str):
                    time_str = df_hourly['time']
                else:
                    time_str = df_hourly['time'].astype(str)
            except Exception as e:
                logger.error(f"处理time列时出错: {e}")
                time_str = df_hourly['time'].astype(str)
            
            try:
                datetime_str = date_str + ' ' + time_str
                df_hourly['datetime'] = pd.to_datetime(datetime_str, errors='coerce')
                
                if df_hourly['datetime'].isna().any():
                    if date_series is not None:
                        df_hourly['datetime'] = date_series + pd.to_timedelta(time_str)
            except Exception as e:
                logger.error(f"合并datetime时出错: {e}")
                try:
                    if date_series is None:
                        date_series = pd.to_datetime(df_hourly['date'])
                    time_delta = pd.to_timedelta(time_str)
                    df_hourly['datetime'] = date_series + time_delta
                except Exception as e2:
                    logger.error(f"备用方法也失败: {e2}")
                    return (case_name, None)
        elif 'datetime' in df_hourly.columns:
            df_hourly['datetime'] = pd.to_datetime(df_hourly['datetime'], errors='coerce')
            if df_hourly['datetime'].isna().any():
                logger.warning(f"  {case_name}: 部分datetime列值无效")
        else:
            logger.warning(f"无法找到日期时间列: {parquet_path}")
            return (case_name, None)
        
        # 创建point_key列
        df_hourly['point_key'] = list(zip(df_hourly['lat'], df_hourly['lon']))
        
        # 只保留必要列
        needed_cols = ['lat', 'lon', 'cooling_demand', 'point_key', 'datetime']
        df_hourly = df_hourly[needed_cols].copy()
        
        logger.info(f"  ✓ {case_name}: {len(df_hourly)} 条记录")
        return (case_name, df_hourly)
    
    except Exception as e:
        logger.error(f"读取 {parquet_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return (case_name, None)


def process_single_country_v2(args):
    """处理单个国家的逐时能耗聚合（优化版：使用单块共享内存和预建映射）
    
    Parameters:
    -----------
    args : tuple
        (country_code, continent, point_to_rows, country_to_points, shm_metas, year, model_name, ssp_path)
        point_to_rows: {(lat, lon): [(case_name, row_idx), ...]}
        country_to_points: {country_code: set of (lat, lon)}
        shm_metas: {case_name: (shm_name, meta)}
    """
    (country_code, continent, point_to_rows, country_to_points, shm_metas, year, model_name, ssp_path) = args
    
    shm_refs_all = []  # 记录所有打开的共享内存，用于finally清理
    
    try:
        logger.info(f"处理国家 {country_code}...")
        start_time = time.time()
        
        # 获取该国家的所有点
        points = country_to_points.get(country_code, set())
        if not points:
            logger.warning(f"国家 {country_code} 没有匹配的点")
            return
        
        logger.info(f"  国家 {country_code} 有 {len(points)} 个点")
        
        # 收集该国所有 (case, row_idx) 对
        rows_needed = []
        for pk in points:
            rows_needed.extend(point_to_rows.get(pk, []))
        
        if not rows_needed:
            logger.warning(f"国家 {country_code} 没有匹配的数据行")
            return
        
        # 按 case 分组
        case_rows = defaultdict(list)
        for case, idx in rows_needed:
            case_rows[case].append(idx)
        
        # 从单块共享内存中直接读取需要的行，无需重建全量DataFrame
        country_results = {}  # {case_name: {datetime_iso: energy_sum}}
        
        for case_name, indices in case_rows.items():
            if case_name not in shm_metas:
                continue
            
            shm_name, meta = shm_metas[case_name]
            
            # 优化：对索引排序以提升内存局部性（累加操作不依赖顺序）
            indices_array = np.array(indices, dtype=np.int64)
            if len(indices_array) > 5000:
                # 大批量数据：排序索引以提升缓存效率
                sort_order = np.argsort(indices_array)
                indices_sorted = indices_array[sort_order]
            else:
                # 小批量数据：直接使用（排序开销可能大于收益）
                indices_sorted = indices_array
            
            # 打开共享内存并读取指定行
            shm_ref, arr_view = read_rows_from_shm(shm_name, meta, row_indices=indices_sorted)
            shm_refs_all.append(shm_ref)
            
            # 提取数据
            dt_sec = arr_view['datetime_seconds']
            energy = arr_view['cooling_demand']
            
            # 按时间分组累加
            # 使用字典累加（比groupby快）
            case_energy_by_time = defaultdict(float)
            for t_sec, e in zip(dt_sec, energy):
                # 使用 UTC 时区（气象模型标准）
                # 使用 timezone.utc 替代已弃用的 utcfromtimestamp
                dt = datetime.fromtimestamp(int(t_sec), tz=timezone.utc).replace(tzinfo=None)
                dt_iso = dt.isoformat()
                case_energy_by_time[dt_iso] += float(e)
            
            country_results[case_name] = dict(case_energy_by_time)
            logger.info(f"  国家 {country_code} - 工况 {case_name}: {len(country_results[case_name])} 个时间点")
        
        # 保存CSV
        if country_results:
            save_start_time = time.time()
            
            # 创建时间序列（该年份的所有小时）
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31, 23, 0, 0)
            time_range = pd.date_range(start=start_date, end=end_date, freq='h')
            
            # 创建结果DataFrame
            result_data = []
            for dt in time_range:
                dt_str = dt.isoformat()
                row_data = {'time': dt}
                # 添加所有工况的能耗
                for case_name in shm_metas.keys():
                    if case_name in country_results:
                        energy = country_results[case_name].get(dt_str, 0.0)
                        row_data[case_name] = energy
                result_data.append(row_data)
            
            result_df = pd.DataFrame(result_data)
            
            # 保存CSV
            output_dir = os.path.join(
                HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", "hourly_energy_v2",
                continent, country_code
            )
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, f"{country_code}_{year}_hourly_energy.csv")
            result_df.to_csv(output_file, index=False)
            
            save_elapsed = time.time() - save_start_time
            total_elapsed = time.time() - start_time
            logger.info(f"  ✓ {country_code}: 已保存，总耗时 {total_elapsed:.2f}s (保存耗时 {save_elapsed:.2f}s)")
    
    except Exception as e:
        logger.error(f"处理国家 {country_code} 时出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭共享内存引用（不unlink，主进程负责unlink）
        for shm in shm_refs_all:
            try:
                if shm is not None:
                    shm.close()
            except:
                pass
        
        # 子进程内存回收
        gc.collect()


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
    logger.info(f"\n{'='*60}")
    logger.info(f"处理 {model_name} - {ssp_path} - {year} 年")
    logger.info(f"{'='*60}")
    
    # ========== 第一步：读取国家信息 ==========
    logger.info("读取国家信息...")
    t0 = time.time()
    countries_info_df = read_csv_with_encoding(COUNTRIES_INFO_CSV)
    country_to_continent = dict(zip(
        countries_info_df['Country_Code_3'],
        countries_info_df['continent']
    ))
    logger.info(f"读取国家信息完成，耗时: {time.time() - t0:.2f} 秒")
    
    # ========== 第二步：建立点-国家的映射 ==========
    # 使用第一个parquet文件获取所有点
    first_case = CASES[0]
    first_parquet_path = os.path.join(
        HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", str(year),
        "point", first_case, f"{first_case}_hourly.parquet"
    )
    
    if not os.path.exists(first_parquet_path):
        logger.error(f"第一个parquet文件不存在: {first_parquet_path}")
        return
    
    t0 = time.time()
    point_to_country = create_point_to_country_mapping_from_parquet(
        first_parquet_path, SHAPEFILE_PATH, country_to_continent
    )
    logger.info(f"建立点-国家映射完成，耗时: {time.time() - t0:.2f} 秒")
    
    # ========== 第三步：加载所有工况的parquet数据 ==========
    logger.info(f"加载工况数据: {', '.join(CASES)}...")
    t0 = time.time()
    
    parquet_args = []
    for case_name in CASES:
        parquet_path = os.path.join(
            HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", str(year),
            "point", case_name, f"{case_name}_hourly.parquet"
        )
        if os.path.exists(parquet_path):
            parquet_args.append((case_name, parquet_path, model_name, ssp_path, year))
    
    # 并行读取parquet文件
    all_cases_data = {}  # {case_name: df}
    if parquet_args:
        read_processes = min(len(parquet_args), 7, os.cpu_count())
        with Pool(processes=read_processes) as read_pool:
            read_results = read_pool.map(load_single_parquet, parquet_args)
        
        for case_name, df_case in read_results:
            if df_case is not None:
                all_cases_data[case_name] = df_case
                logger.info(f"  {case_name}: 读取了 {len(df_case)} 条记录")
    
    if not all_cases_data:
        logger.error("没有有效的工况数据")
        return
    
    logger.info(f"加载所有工况数据完成，耗时: {time.time() - t0:.2f} 秒")
    
    # ========== 构建 point_key → [row_indices] 映射 ==========
    logger.info("构建 point_key → [row_indices] 映射...")
    t0 = time.time()
    point_to_rows = defaultdict(list)  # {(lat, lon): [(case_name, row_idx), ...]}
    
    for case_name, df_case in all_cases_data.items():
        for idx, pk in enumerate(df_case['point_key']):
            point_to_rows[pk].append((case_name, idx))
    
    logger.info(f"构建映射完成，共 {len(point_to_rows)} 个唯一点，耗时: {time.time() - t0:.2f} 秒")
    
    # ========== 转换为单块共享内存格式 ==========
    logger.info("转换为单块共享内存格式...")
    t0 = time.time()
    shm_metas = {}  # {case_name: (shm_name, meta)}
    all_shm_refs = []  # 记录所有共享内存引用，用于清理
    
    for case_name, df_case in all_cases_data.items():
        shm_name, meta, shm_ref = pack_dataframe_to_single_shm(df_case, case_name)
        shm_metas[case_name] = (shm_name, meta)
        all_shm_refs.append(shm_ref)
        logger.info(f"  {case_name}: 已转换为单块共享内存 ({shm_name})")
    
    # 释放原始DataFrame
    del all_cases_data
    gc.collect()
    
    logger.info(f"转换为共享内存格式完成，耗时: {time.time() - t0:.2f} 秒")
    
    # ========== 构建国家 → 点集映射 ==========
    logger.info("构建国家 → 点集映射...")
    country_to_points = defaultdict(set)  # {country_code: set of (lat, lon)}
    country_to_continent_map = {}  # {country_code: continent} 用于记录国家对应的大洲
    for (lat, lon), (country_code, continent) in point_to_country.items():
        if country_code is not None:
            if TEST_COUNTRY is None or country_code == TEST_COUNTRY:
                country_to_points[country_code].add((lat, lon))
                if country_code not in country_to_continent_map:
                    country_to_continent_map[country_code] = continent
    
    # ========== 第四步：按国家聚合 ==========
    logger.info("开始按国家聚合...")
    
    # 收集所有国家
    all_countries = set()
    for country_code, points in country_to_points.items():
        continent = country_to_continent_map.get(country_code)
        if continent:
            all_countries.add((country_code, continent))
    
    logger.info(f"需要处理 {len(all_countries)} 个国家")
    
    # 准备参数
    country_args = []
    for country_code, continent in all_countries:
        country_args.append((
            country_code,
            continent,
            point_to_rows,  # point → [(case, row_idx)]
            country_to_points,  # country → set of points
            shm_metas,  # {case_name: (shm_name, meta)}
            year,
            model_name,
            ssp_path
        ))
    
    # 并行处理所有国家
    if country_args:
        with Pool(processes=min(NUM_PROCESSES_COUNTRY, len(country_args))) as pool:
            pool.map(process_single_country_v2, country_args)
    
    # 释放共享内存
    logger.info("释放共享内存...")
    for shm in all_shm_refs:
        try:
            shm.close()
            shm.unlink()  # 主进程负责unlink
        except Exception as e:
            logger.debug(f"释放共享内存 {shm.name} 时出错: {e}")
    
    # 释放内存
    del shm_metas
    del point_to_rows
    del country_to_points
    del point_to_country
    gc.collect()
    
    logger.info(f"\n{year}年处理完成，总耗时: {time.time() - year_start_time:.2f} 秒")


def main():
    """主函数"""
    try:
        logger.info("=== 开始按国家聚合逐时能耗（简化版）===")
        logger.info(f"处理的模型: {', '.join(MODELS)}")
        logger.info(f"SSP路径: {', '.join(SSP_PATHS)}")
        logger.info(f"目标年份: {TARGET_YEARS}")
        logger.info(f"处理的工况: {', '.join(CASES)}")
        logger.info(f"并行进程数: {NUM_PROCESSES_COUNTRY}")
        logger.info(f"处理模式: 所有国家")
        
        # 按顺序处理
        for model_name in MODELS:
            for ssp_path in SSP_PATHS:
                logger.info(f"\n开始处理 {model_name} - {ssp_path}")
                
                for year in TARGET_YEARS:
                    logger.info(f"处理 {model_name} - {ssp_path} - {year} 年...")
                    
                    try:
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
                
                logger.info(f"{model_name} - {ssp_path} 的所有年份处理完成")
        
        logger.info("\n=== 所有处理完成 ===")
    
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
