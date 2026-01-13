"""
按国家聚合逐时能耗（流式聚合版）

功能：
1. 根据shp文件划分国家，获得点到国家的映射
2. 按30°×30°分块读取parquet数据（流式处理）
3. 边读边聚合：将属于这个国家的点的能耗数据按时间都加起来
4. 当某个国家的所有点都处理完时，立即输出CSV

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
import pyarrow.dataset as ds
import pyarrow.compute as pc
from datetime import datetime, timedelta
from collections import defaultdict
import time
from multiprocessing import Pool, Manager
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
SSP_PATHS = ["SSP126", "SSP245"]  # 所有SSP

# 年份配置
TARGET_YEARS = [2030, 2031]  # 所有年份

# Case配置：只处理ref和case6-10
CASES = ['ref'] + [f'case{i}' for i in range(6, 11)]  # ['ref', 'case6', 'case7', 'case8', 'case9', 'case10']

# 测试国家：None表示处理所有国家
TEST_COUNTRY = None

# 分块配置
CHUNK_SIZE_LON = 30  # 经度分块大小（度）
CHUNK_SIZE_LAT = 30  # 纬度分块大小（度）

# 并行处理配置
NUM_PROCESSES_COUNTRY = 10  # 计算国家逐时能耗的进程数
NUM_PROCESSES_CHUNK = 10  # 并行处理块的进程数


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


def create_point_to_country_mapping_from_parquet(parquet_path, shp_path, country_to_continent):
    """从parquet文件创建点到国家的映射
    
    Parameters:
    -----------
    parquet_path : str
        第一个parquet文件路径（用于获取所有点的坐标）
    shp_path : str
        国家边界shp文件路径
    country_to_continent : dict
        国家代码到大洲的映射（已读取）
    
    Returns:
    --------
    dict: {(lat, lon): (country_code, continent)}
    dict: {country_code: set of (lat, lon)} - 每个国家的所有点
    """
    start_time = time.time()
    logger.info("创建点到国家的映射...")
    
    # 从parquet文件获取所有点（只读取lat和lon列）
    logger.info(f"从parquet文件读取点坐标: {parquet_path}")
    try:
        df_sample = pd.read_parquet(parquet_path, columns=['lat', 'lon'], engine='pyarrow')
        unique_points = df_sample[['lat', 'lon']].drop_duplicates()
        
        # 四舍五入到5位小数，避免浮点精度问题（0.00001° ≈ 1.11米，足够精确）
        unique_points['lat'] = np.round(unique_points['lat'], 5)
        unique_points['lon'] = np.round(unique_points['lon'], 5)
        
        # 再次去重，确保精度归一化后的点唯一
        unique_points = unique_points.drop_duplicates()
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
    
    # 空间连接（使用intersects包含边界点，避免遗漏）
    joined_gdf = gpd.sjoin(points_gdf, countries_gdf, how="left", predicate="intersects")
    
    # 创建映射字典和每个国家的点集合
    point_to_country = {}
    country_points = defaultdict(set)  # {country_code: set of (lat, lon)}
    
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
            country_points[country_code].add((lat, lon))
        else:
            point_to_country[(lat, lon)] = (None, None)
    
    # 统计
    mapped_count = sum(1 for v in point_to_country.values() if v[0] is not None)
    elapsed_time = time.time() - start_time
    logger.info(f"成功映射 {mapped_count}/{len(point_to_country)} 个点到国家，耗时 {elapsed_time:.2f} 秒")
    
    return point_to_country, dict(country_points)


def load_chunk_data(args):
    """读取指定经纬度范围内的数据块（用于并行处理，使用pyarrow.dataset优化）
    
    Parameters:
    -----------
    args : tuple
        (parquet_path, lon_min, lon_max, lat_min, lat_max, case_name)
    
    Returns:
    --------
    tuple: (case_name, pd.DataFrame) 或 (case_name, None)
    """
    parquet_path, lon_min, lon_max, lat_min, lat_max, case_name = args
    try:
        if not os.path.exists(parquet_path):
            return (case_name, None)
        
        # 使用pyarrow.dataset进行过滤，避免读取整个文件
        try:
            dataset = ds.dataset(parquet_path, format='parquet')
            
            # 构建过滤条件（使用pyarrow.dataset的表达式）
            # 注意：如果dataset方式不支持复杂过滤，会回退到传统方式
            filter_condition = (
                (ds.field('lon') >= lon_min) & (ds.field('lon') < lon_max) &
                (ds.field('lat') >= lat_min) & (ds.field('lat') < lat_max)
            )
            
            # 只读取需要的列
            columns_needed = ['lat', 'lon', 'cooling_demand', 'date', 'time', 'datetime']
            available_columns = dataset.schema.names
            columns_to_read = [col for col in columns_needed if col in available_columns]
            
            # 读取过滤后的数据
            table = dataset.to_table(filter=filter_condition, columns=columns_to_read)
            df_chunk = table.to_pandas()
            
        except Exception as e:
            # 如果dataset方式失败，回退到传统方式
            try:
                df = pd.read_parquet(parquet_path, engine='pyarrow')
                mask = (
                    (df['lon'] >= lon_min) & (df['lon'] < lon_max) &
                    (df['lat'] >= lat_min) & (df['lat'] < lat_max)
                )
                df_chunk = df[mask].copy()
            except Exception as e2:
                logger.warning(f"读取文件失败 {parquet_path}: {e2}")
                return (case_name, None)
        
        if df_chunk.empty:
            return (case_name, None)
        
        # 四舍五入到5位小数，避免浮点精度问题（与映射时的精度保持一致）
        df_chunk['lat'] = np.round(df_chunk['lat'], 5)
        df_chunk['lon'] = np.round(df_chunk['lon'], 5)
        
        # 处理datetime列
        if 'date' in df_chunk.columns and 'time' in df_chunk.columns:
            try:
                date_series = pd.to_datetime(df_chunk['date'])
                date_str = date_series.dt.strftime('%Y-%m-%d')
            except Exception as e:
                try:
                    date_str = df_chunk['date'].astype(str).str[:10]
                    date_series = pd.to_datetime(date_str)
                except Exception as e2:
                    logger.error(f"处理date列时出错: {e2}")
                    return (case_name, None)
            
            try:
                if len(df_chunk) > 0:
                    if hasattr(df_chunk['time'].iloc[0], 'strftime'):
                        time_str = df_chunk['time'].apply(lambda x: x.strftime('%H:%M:%S') if pd.notna(x) else '00:00:00')
                    elif isinstance(df_chunk['time'].iloc[0], str):
                        time_str = df_chunk['time']
                    else:
                        time_str = df_chunk['time'].astype(str)
                else:
                    time_str = df_chunk['time'].astype(str)
            except Exception as e:
                time_str = df_chunk['time'].astype(str)
            
            try:
                datetime_str = date_str + ' ' + time_str
                df_chunk['datetime'] = pd.to_datetime(datetime_str, errors='coerce')
                
                if df_chunk['datetime'].isna().any():
                    if date_series is not None:
                        df_chunk['datetime'] = date_series + pd.to_timedelta(time_str)
            except Exception as e:
                try:
                    if date_series is None:
                        date_series = pd.to_datetime(df_chunk['date'])
                    time_delta = pd.to_timedelta(time_str)
                    df_chunk['datetime'] = date_series + time_delta
                except Exception as e2:
                    logger.error(f"合并datetime时出错: {e2}")
                    return (case_name, None)
        elif 'datetime' in df_chunk.columns:
            df_chunk['datetime'] = pd.to_datetime(df_chunk['datetime'], errors='coerce')
        else:
            logger.warning(f"无法找到日期时间列: {parquet_path}")
            return (case_name, None)
        
        # 创建point_key列
        df_chunk['point_key'] = list(zip(df_chunk['lat'], df_chunk['lon']))
        
        # 只保留必要列
        needed_cols = ['lat', 'lon', 'cooling_demand', 'point_key', 'datetime']
        df_chunk = df_chunk[[col for col in needed_cols if col in df_chunk.columns]].copy()
        
        return (case_name, df_chunk)
    
    except Exception as e:
        logger.error(f"读取数据块时出错: {e}")
        return (case_name, None)


def aggregate_chunk_to_countries(df_chunk, case_name, point_to_country, country_energy_data, country_processed_points):
    """将数据块聚合到各个国家
    
    Parameters:
    -----------
    df_chunk : pd.DataFrame
        数据块
    case_name : str
        工况名称
    point_to_country : dict
        点到国家的映射 {(lat, lon): (country_code, continent)}
    country_energy_data : dict
        每个国家的能耗数据 {country_code: {case_name: {datetime_str: energy_sum}}}
    country_processed_points : dict
        每个国家已处理的点 {country_code: set of (lat, lon)}
    
    Returns:
    --------
    set: 在这个块中有数据的国家代码集合
    """
    countries_in_chunk = set()
    
    if df_chunk is None or df_chunk.empty:
        return countries_in_chunk
    
    # 按点分组，找到对应的国家
    for point_key in df_chunk['point_key'].unique():
        if point_key not in point_to_country:
            continue
        
        country_code, continent = point_to_country[point_key]
        if country_code is None:
            continue
        
        countries_in_chunk.add(country_code)
        
        # 筛选该点的数据
        point_mask = df_chunk['point_key'] == point_key
        df_point = df_chunk[point_mask].copy()
        
        # 按datetime分组并累加cooling_demand
        grouped = df_point.groupby('datetime')['cooling_demand'].sum()
        
        # 累加到国家数据中
        if country_code not in country_energy_data:
            country_energy_data[country_code] = defaultdict(lambda: defaultdict(float))
        
        for dt, energy in grouped.items():
            dt_str = dt.isoformat()
            country_energy_data[country_code][case_name][dt_str] += float(energy)
        
        # 记录已处理的点（每个case都要记录，但只需要记录一次）
        if country_code not in country_processed_points:
            country_processed_points[country_code] = set()
        country_processed_points[country_code].add(point_key)
    
    return countries_in_chunk


def process_single_chunk(args):
    """处理单个块的数据（用于并行处理）
    
    Parameters:
    -----------
    args : tuple
        (chunk_id, lon_start, lon_end, lat_start, lat_end, model_name, ssp_path, year, point_to_country)
    
    Returns:
    --------
    tuple: (chunk_id, chunk_result, countries_in_chunk)
        chunk_result: {country_code: {case_name: {datetime_str: energy_sum}}}
        countries_in_chunk: set of country_code
        processed_points: {country_code: set of (lat, lon)}
    """
    chunk_id, lon_start, lon_end, lat_start, lat_end, model_name, ssp_path, year, point_to_country = args
    
    chunk_result = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    countries_in_chunk = set()
    processed_points = defaultdict(set)
    
    try:
        # 并行读取该空间块下的所有工况数据
        chunk_args = []
        for case_name in CASES:
            parquet_path = os.path.join(
                HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", str(year),
                "point", case_name, f"{case_name}_hourly.parquet"
            )
            chunk_args.append((parquet_path, lon_start, lon_end, lat_start, lat_end, case_name))
        
        # 读取所有工况的数据（在进程内顺序读取，因为已经是并行处理块了）
        chunk_results = []
        for chunk_arg in chunk_args:
            result = load_chunk_data(chunk_arg)
            chunk_results.append(result)
        
        # 对每个工况的数据进行聚合
        for case_name, df_chunk in chunk_results:
            if df_chunk is not None and not df_chunk.empty:
                # 按点分组，找到对应的国家
                for point_key in df_chunk['point_key'].unique():
                    if point_key not in point_to_country:
                        continue
                    
                    country_code, continent = point_to_country[point_key]
                    if country_code is None:
                        continue
                    
                    countries_in_chunk.add(country_code)
                    
                    # 筛选该点的数据
                    point_mask = df_chunk['point_key'] == point_key
                    df_point = df_chunk[point_mask].copy()
                    
                    # 按datetime分组并累加cooling_demand
                    grouped = df_point.groupby('datetime')['cooling_demand'].sum()
                    
                    # 累加到结果中
                    for dt, energy in grouped.items():
                        dt_str = dt.isoformat()
                        chunk_result[country_code][case_name][dt_str] += float(energy)
                    
                    # 记录已处理的点
                    processed_points[country_code].add(point_key)
        
        return (chunk_id, dict(chunk_result), countries_in_chunk, dict(processed_points))
    
    except Exception as e:
        logger.error(f"处理块 {chunk_id} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return (chunk_id, {}, set(), {})


def save_country_csv(country_code, continent, country_energy_data, country_points, country_processed_points, 
                     year, model_name, ssp_path, all_cases, clear_data=True):
    """保存国家的CSV文件，并可选择清空内存中的数据
    
    Parameters:
    -----------
    country_code : str
        国家代码
    continent : str
        大洲
    country_energy_data : dict
        该国家的能耗数据 {case_name: {datetime_str: energy_sum}}
    country_points : dict
        每个国家的所有点 {country_code: set of (lat, lon)}
    country_processed_points : dict
        每个国家已处理的点 {country_code: set of (lat, lon)}
    year : int
        年份
    model_name : str
        模型名称
    ssp_path : str
        SSP路径
    all_cases : list
        所有工况列表
    clear_data : bool
        是否在保存后清空数据（默认True）
    
    Returns:
    --------
    bool: 是否成功保存
    """
    # 检查是否所有点都已处理
    if country_code not in country_points or country_code not in country_processed_points:
        return False
    
    all_points = country_points[country_code]
    processed_points = country_processed_points[country_code]
    
    if len(processed_points) < len(all_points):
        return False  # 还有未处理的点
    
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
        for case_name in all_cases:
            if country_code in country_energy_data and case_name in country_energy_data[country_code]:
                energy = country_energy_data[country_code][case_name].get(dt_str, 0.0)
            else:
                energy = 0.0
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
    
    logger.info(f"{country_code}: 已保存")
    
    # 清空已保存国家的数据，释放内存
    if clear_data:
        if country_code in country_energy_data:
            del country_energy_data[country_code]
        if country_code in country_processed_points:
            del country_processed_points[country_code]
    
    return True


def process_single_year_streaming(model_name, ssp_path, year):
    """使用流式聚合处理单个年份的逐时能耗
    
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
    logger.info(f"处理 {model_name} - {ssp_path} - {year} 年（流式聚合）")
    logger.info(f"{'='*60}")
    
    # ========== 第一步：读取国家信息 ==========
    countries_info_df = read_csv_with_encoding(COUNTRIES_INFO_CSV)
    country_to_continent = dict(zip(
        countries_info_df['Country_Code_3'],
        countries_info_df['continent']
    ))
    
    # ========== 第二步：建立点-国家的映射 ==========
    first_case = CASES[0]
    first_parquet_path = os.path.join(
        HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", str(year),
        "point", first_case, f"{first_case}_hourly.parquet"
    )
    
    if not os.path.exists(first_parquet_path):
        logger.error(f"第一个parquet文件不存在: {first_parquet_path}")
        return
    
    point_to_country, country_points = create_point_to_country_mapping_from_parquet(
        first_parquet_path, SHAPEFILE_PATH, country_to_continent
    )
    
    # 过滤测试国家
    if TEST_COUNTRY is not None:
        country_points = {k: v for k, v in country_points.items() if k == TEST_COUNTRY}
    
    # ========== 第三步：初始化聚合数据结构 ==========
    # 每个国家的能耗数据：{country_code: {case_name: {datetime_str: energy_sum}}}
    country_energy_data = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    
    # 每个国家已处理的点：{country_code: set of (lat, lon)}
    country_processed_points = defaultdict(set)
    
    # 已完成的国家的集合
    completed_countries = set()
    
    # ========== 第四步：按块读取和聚合 ==========
    # 经度范围：-180到180，每30°一块
    # 纬度范围：-90到90，每30°一块
    lon_chunks = list(range(-180, 180, CHUNK_SIZE_LON))
    lat_chunks = list(range(-90, 90, CHUNK_SIZE_LAT))
    
    total_chunks = len(lon_chunks) * len(lat_chunks)
    
    logger.info(f"开始按块读取数据，共 {total_chunks} 块，并行处理 {NUM_PROCESSES_CHUNK} 个块")
    
    # 准备所有块的参数
    chunk_args_list = []
    chunk_id = 0
    for lon_start in lon_chunks:
        lon_end = lon_start + CHUNK_SIZE_LON
        for lat_start in lat_chunks:
            lat_end = lat_start + CHUNK_SIZE_LAT
            chunk_id += 1
            chunk_args_list.append((
                chunk_id, lon_start, lon_end, lat_start, lat_end,
                model_name, ssp_path, year, point_to_country
            ))
    
    # 使用进程池并行处理多个块
    processed_chunks = 0
    with Pool(processes=NUM_PROCESSES_CHUNK) as chunk_pool:
        # 使用map_async以便可以流式处理结果
        results = chunk_pool.map_async(process_single_chunk, chunk_args_list)
        
        # 等待所有结果完成
        chunk_results = results.get()
        
        # 合并所有块的结果
        for chunk_id, chunk_result, countries_in_chunk, processed_points in chunk_results:
            processed_chunks += 1
            
            # 打印这个块中有数据的国家
            if countries_in_chunk:
                lon_start = chunk_args_list[chunk_id - 1][1]
                lon_end = chunk_args_list[chunk_id - 1][2]
                lat_start = chunk_args_list[chunk_id - 1][3]
                lat_end = chunk_args_list[chunk_id - 1][4]
                countries_str = ', '.join(sorted(countries_in_chunk))
                logger.info(f"块 {chunk_id}/{total_chunks} ({lon_start}°-{lon_end}°E, {lat_start}°-{lat_end}°N): 正在聚合 {len(countries_in_chunk)} 个国家: {countries_str}")
            
            # 合并该块的结果到总数据中
            for country_code, case_data in chunk_result.items():
                if country_code not in country_energy_data:
                    country_energy_data[country_code] = defaultdict(lambda: defaultdict(float))
                
                for case_name, datetime_data in case_data.items():
                    for dt_str, energy in datetime_data.items():
                        country_energy_data[country_code][case_name][dt_str] += energy
            
            # 合并已处理的点
            for country_code, points in processed_points.items():
                country_processed_points[country_code].update(points)
            
            # 检查哪些国家已完成，立即保存并清空数据（避免内存爆炸）
            for country_code in list(country_points.keys()):
                if country_code in completed_countries:
                    continue
                
                continent = country_to_continent.get(country_code, 'Unknown')
                if save_country_csv(
                    country_code, continent, 
                    country_energy_data, country_points, country_processed_points,
                    year, model_name, ssp_path, CASES, clear_data=True
                ):
                    completed_countries.add(country_code)
            
            # 每处理10块输出一次进度
            if processed_chunks % 10 == 0:
                logger.info(f"已处理 {processed_chunks}/{total_chunks} 块，完成 {len(completed_countries)}/{len(country_points)} 个国家")
            
            # 定期释放内存
            if processed_chunks % 20 == 0:
                gc.collect()
    
    # 处理剩余未完成的国家
    for country_code in country_points.keys():
        if country_code not in completed_countries:
            continent = country_to_continent.get(country_code, 'Unknown')
            save_country_csv(
                country_code, continent,
                country_energy_data, country_points, country_processed_points,
                year, model_name, ssp_path, CASES
            )
    
    logger.info(f"\n{year}年处理完成，总耗时: {time.time() - year_start_time:.2f} 秒")


def main():
    """主函数"""
    try:
        logger.info("=== 开始按国家聚合逐时能耗（流式聚合版）===")
        logger.info(f"处理的模型: {', '.join(MODELS)}")
        logger.info(f"SSP路径: {', '.join(SSP_PATHS)}")
        logger.info(f"目标年份: {TARGET_YEARS}")
        logger.info(f"处理的工况: {', '.join(CASES)}")
        logger.info(f"分块大小: {CHUNK_SIZE_LON}°×{CHUNK_SIZE_LAT}°")
        logger.info(f"处理模式: 所有国家" if TEST_COUNTRY is None else f"测试国家: {TEST_COUNTRY}")
        
        # 按顺序处理
        for model_name in MODELS:
            for ssp_path in SSP_PATHS:
                logger.info(f"\n开始处理 {model_name} - {ssp_path}")
                
                for year in TARGET_YEARS:
                    logger.info(f"处理 {model_name} - {ssp_path} - {year} 年...")
                    
                    try:
                        process_single_year_streaming(model_name, ssp_path, year)
                        
                        # 每个年份处理完后，确保内存清理
                        for i in range(2):
                            gc.collect()
                        
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
