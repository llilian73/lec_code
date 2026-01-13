"""
国家能耗功率计算工具（带缓存优化版）

优化说明：
1. 为每个网格点增加缓存机制，缓存最大制冷/制热功率；
2. 再次运行时会跳过已计算的点，大幅提升速度；
3. 调整并行参数和日志控制，提高CPU利用率；
4. 所有缓存文件存储在 Z:\local_environment_creation\energy_consumption_gird\cache\
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from tqdm import tqdm
import multiprocessing
from functools import partial
import time


# ======================== 基本路径配置 ========================
GRID_RESULT_BASE_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\result_half"
POINT_COUNTRY_MAPPING_FILE = r"Z:\local_environment_creation\energy_consumption_gird\result\point_country_mapping.csv"
PARAMETERS_FILE = r"Z:\local_environment_creation\energy_consumption_gird\parameters.csv"
OUTPUT_DIR = r"Z:\local_environment_creation\cost"
CACHE_DIR = r"Z:\local_environment_creation\energy_consumption_gird\cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "country_energy_power.csv")
YEARS = [2016, 2017, 2018, 2019, 2020]


# ======================== 日志系统 ========================
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler('country_cost.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()


# ======================== 功能函数 ========================
def load_parameters():
    """加载功率系数参数（只读取一次）"""
    try:
        params_df = pd.read_csv(PARAMETERS_FILE)
        # 直接使用ISO代码作为键
        params_dict = {}
        for _, row in params_df.iterrows():
            iso_code = row['region']
            params_dict[iso_code] = {
                'heating_power': row['heating power'],
                'cooling_power': row['Cooling power']
            }
        return params_dict
    except Exception as e:
        logger.error(f"加载参数文件出错: {str(e)}")
        return {}

def load_point_country_mapping():
    """加载点-国家映射数据（只读取一次）"""
    mapping_df = pd.read_csv(POINT_COUNTRY_MAPPING_FILE)
    mapping_df = mapping_df[mapping_df['Country_Code'] != 'Unknown']
    return mapping_df


# ======================== 核心计算函数 ========================
def load_point_energy_data(lat, lon, year):
    """加载单个点在指定年份的能耗数据"""
    base_filename = f"point_lat{lat:.3f}_lon{lon:.3f}"
    cooling_path = os.path.join(GRID_RESULT_BASE_DIR, str(year), f"{base_filename}_cooling.csv")
    heating_path = os.path.join(GRID_RESULT_BASE_DIR, str(year), f"{base_filename}_heating.csv")

    cooling_data = None
    heating_data = None
    try:
        if os.path.exists(cooling_path):
            cooling_data = pd.read_csv(cooling_path, usecols=['ref'])
        if os.path.exists(heating_path):
            heating_data = pd.read_csv(heating_path, usecols=['ref'])
    except Exception:
        pass

    return cooling_data, heating_data


def find_max_power_for_point(lat, lon):
    """找到指定点在5年内的最大功率（带缓存）"""
    cache_file = os.path.join(CACHE_DIR, f"point_lat{lat:.3f}_lon{lon:.3f}_max.csv")

    # 若已存在缓存文件，直接读取
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            return df.loc[0, "max_cooling"], df.loc[0, "max_heating"]
        except Exception:
            pass

    max_cooling_power = 0.0
    max_heating_power = 0.0

    for year in YEARS:
        cooling_data, heating_data = load_point_energy_data(lat, lon, year)
        if cooling_data is not None and 'ref' in cooling_data.columns:
            max_cooling_power = max(max_cooling_power, cooling_data['ref'].max())
        if heating_data is not None and 'ref' in heating_data.columns:
            max_heating_power = max(max_heating_power, heating_data['ref'].max())

    # 写入缓存
    pd.DataFrame([{
        "max_cooling": max_cooling_power,
        "max_heating": max_heating_power
    }]).to_csv(cache_file, index=False)

    return max_cooling_power, max_heating_power


def calculate_max_power_with_coefficients(cooling_power, heating_power, country_code, params_dict):
    """应用功率系数计算最大功率"""
    # 默认功率系数
    default_heating_power = 27.9
    default_cooling_power = 48.5
    
    # 直接使用ISO代码查找功率系数
    if country_code in params_dict:
        heating_coefficient = params_dict[country_code]['heating_power']
        cooling_coefficient = params_dict[country_code]['cooling_power']
    else:
        heating_coefficient = default_heating_power
        cooling_coefficient = default_cooling_power
    
    # 应用功率系数（从GW转换为W）
    actual_cooling_power = cooling_power * cooling_coefficient * 1e9
    actual_heating_power = heating_power * heating_coefficient * 1e9
    
    return max(actual_cooling_power, actual_heating_power)


def process_point_batch(point_batch, params_dict):
    """处理一批网格点（带缓存）"""
    # 子进程只显示错误日志
    if multiprocessing.current_process().name != 'MainProcess':
        logger.setLevel(logging.ERROR)

    batch_results = []
    for _, row in point_batch.iterrows():
        try:
            lat = row['lat']
            lon = row['lon']
            country_code = row['Country_Code']
            continent = row['Continent']
            country_name = row['Country_Name']

            if country_code == 'CN-TW':
                country_code = 'CN'

            max_cooling_power, max_heating_power = find_max_power_for_point(lat, lon)
            max_power = calculate_max_power_with_coefficients(
                max_cooling_power, max_heating_power, country_code, params_dict
            )

            batch_results.append({
                'Continent': continent,
                'Country_Code': country_code,
                'Country_Name': country_name,
                'lat': lat,
                'lon': lon,
                'Pmax': max_power
            })
        except Exception as e:
            logger.error(f"处理网格点失败 (lat={lat:.3f}, lon={lon:.3f}): {e}")
            continue

    return batch_results


# ======================== 主程序 ========================
def main():
    logger.info("开始计算国家能耗功率（带缓存优化版）...")

    try:
        # 1. 一次性加载所有基础数据到内存
        logger.info("=== 第一步：加载基础数据到内存 ===")
        params_dict = load_parameters()
        mapping_df = load_point_country_mapping()
        
        logger.info(f"加载功率系数参数: {len(params_dict)} 个国家")
        logger.info(f"加载点-国家映射: {len(mapping_df)} 个点")

        # 2. 配置并行处理参数
        num_cores = multiprocessing.cpu_count()
        num_processes = max(1, num_cores - 2)
        batch_size = 200

        batches = [mapping_df.iloc[i:i + batch_size] for i in range(0, len(mapping_df), batch_size)]
        logger.info(f"CPU核心数: {num_cores}")
        logger.info(f"使用进程数: {num_processes}")
        logger.info(f"每批处理点数: {batch_size} (共 {len(batches)} 批)")

        # 3. 并行处理所有批次
        logger.info("=== 第二步：并行处理网格点 ===")
        all_results = []
        with multiprocessing.Pool(processes=num_processes) as pool:
            process_func = partial(process_point_batch, params_dict=params_dict)
            chunksize = max(1, len(batches) // (num_processes * 4))

            with tqdm(total=len(batches), desc="处理网格点批次") as pbar:
                for batch_results in pool.imap_unordered(process_func, batches, chunksize=chunksize):
                    all_results.extend(batch_results)
                    pbar.update(1)

        # 4. 保存结果
        logger.info("=== 第三步：保存结果 ===")
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

        logger.info(f"结果已保存至: {OUTPUT_FILE}")
        logger.info(f"共处理 {len(all_results)} 个点")

        # 5. 输出统计信息
        logger.info("=== 统计信息 ===")
        continent_stats = results_df.groupby('Continent').size()
        for continent, count in continent_stats.items():
            logger.info(f"{continent}: {count} 个点")

        logger.info(f"功率范围: {results_df['Pmax'].min():.2f} W - {results_df['Pmax'].max():.2f} W")
        logger.info(f"平均功率: {results_df['Pmax'].mean():.2f} W")
        logger.info("✅ 国家能耗功率计算完成（带缓存优化）")

    except Exception as e:
        logger.error(f"主程序执行出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()
