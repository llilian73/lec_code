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
SSP_PATHS = ["SSP126", "SSP245"]  # 所有SSP

# 年份配置
TARGET_YEARS = [2030, 2031]  # 所有年份

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
    """
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
    
    return point_to_country


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


def process_single_country(args):
    """处理单个国家的逐时能耗聚合
    
    Parameters:
    -----------
    args : tuple
        (country_code, continent, point_to_country, all_cases_data, year, model_name, ssp_path)
    """
    (country_code, continent, point_to_country, all_cases_data, year, model_name, ssp_path) = args
    
    try:
        # 获取该国家的所有点
        country_points = set()
        for (lat, lon), (code, _) in point_to_country.items():
            if code == country_code:
                country_points.add((lat, lon))
        
        if not country_points:
            logger.warning(f"国家 {country_code} 没有匹配的点")
            return
        
        # 对每个工况，提取该国家的数据并聚合
        country_results = {}  # {case_name: {datetime: energy_sum}}
        
        for case_name, df_case in all_cases_data.items():
            if df_case is None:
                continue
        
            # 筛选该国家的点
            mask = df_case['point_key'].isin(country_points)
            df_country = df_case[mask].copy()
            
            if len(df_country) == 0:
                country_results[case_name] = {}
                continue
            
            # 按datetime分组并累加cooling_demand
            grouped = df_country.groupby('datetime')['cooling_demand'].sum()
            
            # 转换为字典
            country_results[case_name] = {
                dt.isoformat(): float(energy) 
                for dt, energy in grouped.items()
            }
        
        # 保存CSV
        if country_results:
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
                for case_name in all_cases_data.keys():
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
            
            logger.info(f"{country_code}: 已保存")
    
    except Exception as e:
        logger.error(f"处理国家 {country_code} 时出错: {e}")
        import traceback
        traceback.print_exc()


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
    countries_info_df = read_csv_with_encoding(COUNTRIES_INFO_CSV)
    country_to_continent = dict(zip(
        countries_info_df['Country_Code_3'],
        countries_info_df['continent']
    ))
    
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
    
    point_to_country = create_point_to_country_mapping_from_parquet(
        first_parquet_path, SHAPEFILE_PATH, country_to_continent
    )
    
    # ========== 第三步：加载所有工况的parquet数据 ==========
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
    
    # ========== 第四步：按国家聚合 ==========
    logger.info("开始按国家聚合...")
    
    # 收集所有国家（如果指定了测试国家，只处理该国家）
    all_countries = set()
    for (lat, lon), (country_code, continent) in point_to_country.items():
        if country_code is not None:
            if TEST_COUNTRY is None or country_code == TEST_COUNTRY:
                all_countries.add((country_code, continent))
    
    logger.info(f"需要处理 {len(all_countries)} 个国家")
    
    # 准备参数
    country_args = []
    for country_code, continent in all_countries:
        country_args.append((
            country_code,
            continent,
            point_to_country,
            all_cases_data,
            year,
            model_name,
            ssp_path
        ))
    
    # 并行处理所有国家
    if country_args:
        with Pool(processes=min(NUM_PROCESSES_COUNTRY, len(country_args))) as pool:
            pool.map(process_single_country, country_args)
    
    # 释放内存
    del all_cases_data
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
