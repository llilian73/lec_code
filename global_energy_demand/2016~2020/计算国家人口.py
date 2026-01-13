"""
计算国家人口统计

功能：
根据 point_country_mapping.csv 和 population_points.csv 聚合计算每个国家的总人口数。
使用并行处理和分批处理策略，高效处理4万多个点的数据。

输入数据：
1. point_country_mapping.csv - 点与国家的映射关系
2. population_points.csv - 每个点的人口数据

输出数据：
country_population_2020.csv - 每个国家的总人口统计
"""

import pandas as pd
import numpy as np
import os
import logging
import multiprocessing
from tqdm import tqdm
from functools import partial

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('country_population.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 配置参数
POINT_COUNTRY_MAPPING_FILE = r"Z:\local_environment_creation\energy_consumption_gird\result\point_country_mapping.csv"
POPULATION_FILE = r"Z:\local_environment_creation\energy_consumption_gird\result\data\population_points.csv"
OUTPUT_FILE = r"Z:\local_environment_creation\energy_consumption_gird\result\country_population_2020.csv"

# 批处理参数
BATCH_SIZE = 5000  # 每批处理5000个点


def load_point_country_mapping():
    """加载点-国家映射数据"""
    logging.info("开始加载点-国家映射数据...")
    
    if not os.path.exists(POINT_COUNTRY_MAPPING_FILE):
        raise FileNotFoundError(f"点-国家映射文件不存在: {POINT_COUNTRY_MAPPING_FILE}")
    
    # 重要：keep_default_na=False 防止 'NA' 被识别为缺失值
    mapping_df = pd.read_csv(POINT_COUNTRY_MAPPING_FILE, keep_default_na=False, na_values=[''])
    logging.info(f"加载点-国家映射数据完成，共 {len(mapping_df)} 个点")
    
    # 过滤掉Unknown地区
    original_count = len(mapping_df)
    mapping_df = mapping_df[mapping_df['Country_Code_2'].notna() & (mapping_df['Country_Code_2'] != 'Unknown')]
    filtered_count = len(mapping_df)
    logging.info(f"过滤掉Unknown地区后，剩余 {filtered_count} 个点（过滤了 {original_count - filtered_count} 个点）")
    
    # 验证 NA (Namibia) 是否被正确加载
    na_count = len(mapping_df[mapping_df['Country_Code_2'] == 'NA'])
    if na_count > 0:
        logging.info(f"✓ Namibia (NA) 数据已正确加载: {na_count} 个点")
    else:
        logging.warning("⚠ 未找到 Namibia (NA) 的数据，可能被识别为缺失值")
    
    # 统计涉及的国家数量
    country_counts = mapping_df['Country_Code_2'].value_counts()
    logging.info(f"涉及 {len(country_counts)} 个国家/地区")
    logging.info("前10个国家/地区的点数:")
    for country, count in country_counts.head(10).items():
        logging.info(f"  {country}: {count} 个点")
    
    return mapping_df


def load_population_data():
    """加载人口数据"""
    logging.info("开始加载人口数据...")
    
    if not os.path.exists(POPULATION_FILE):
        raise FileNotFoundError(f"人口数据文件不存在: {POPULATION_FILE}")
    
    # keep_default_na=False 防止某些值被错误识别为缺失值
    population_df = pd.read_csv(POPULATION_FILE, keep_default_na=False, na_values=[''])
    logging.info(f"加载人口数据完成，共 {len(population_df)} 个点")
    
    return population_df


def process_batch(batch_df, point_to_country_dict):
    """
    处理一批人口数据
    
    Args:
        batch_df: 人口数据批次DataFrame
        point_to_country_dict: 点到国家的映射字典
    
    Returns:
        该批次的国家人口统计字典
    """
    batch_results = {}
    
    for _, row in batch_df.iterrows():
        lat = row['lat']
        lon = row['lon']
        population = row['population']
        
        # 查找该点对应的国家
        country_info = point_to_country_dict.get((lat, lon), None)
        
        if country_info is not None:
            # 使用Country_Code_2作为键（二字母代码）
            country_code_2 = country_info['country_code_2']
            
            if country_code_2 not in batch_results:
                batch_results[country_code_2] = {
                    'continent': country_info['continent'],
                    'country_code_2': country_info['country_code_2'],
                    'country_code_3': country_info['country_code_3'],
                    'country_name': country_info['country_name'],
                    'population': 0.0
                }
            
            batch_results[country_code_2]['population'] += population
    
    return batch_results


def calculate_country_population_parallel(population_df, mapping_df):
    """
    使用并行处理计算每个国家的总人口
    
    Args:
        population_df: 人口数据DataFrame
        mapping_df: 点-国家映射DataFrame
    
    Returns:
        国家人口统计DataFrame
    """
    logging.info("开始计算国家人口统计...")
    
    # 创建点到国家的映射字典
    logging.info("创建点到国家的映射字典...")
    point_to_country = {}
    for _, row in mapping_df.iterrows():
        lat = float(row['lat'])
        lon = float(row['lon'])
        point_to_country[(lat, lon)] = {
            'continent': row['Continent'],
            'country_code_2': row['Country_Code_2'],
            'country_code_3': row['Country_Code_3'],
            'country_name': row['Country_Name']
        }
    
    logging.info(f"创建了 {len(point_to_country)} 个点的国家映射")
    
    # 分批处理人口数据
    total_points = len(population_df)
    batches = [population_df.iloc[i:i + BATCH_SIZE] for i in range(0, total_points, BATCH_SIZE)]
    
    logging.info(f"将 {total_points} 个点分为 {len(batches)} 批进行处理（每批 {BATCH_SIZE} 个点）")
    
    # 配置并行处理参数
    num_cores = multiprocessing.cpu_count()
    num_processes = min(num_cores, 8)  # 最多使用8个进程
    
    logging.info(f"CPU核心数: {num_cores}")
    logging.info(f"使用进程数: {num_processes}")
    
    # 并行处理
    all_results = {}
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        process_func = partial(process_batch, point_to_country_dict=point_to_country)
        
        chunksize = max(1, len(batches) // (num_processes * 4))
        logging.info(f"chunksize: {chunksize}")
        
        with tqdm(total=len(batches), desc="处理人口数据批次") as pbar:
            for batch_results in pool.imap_unordered(process_func, batches, chunksize=chunksize):
                # 合并批次结果到总结果中
                for country_code_2, data in batch_results.items():
                    if country_code_2 not in all_results:
                        all_results[country_code_2] = {
                            'continent': data['continent'],
                            'country_code_2': data['country_code_2'],
                            'country_code_3': data['country_code_3'],
                            'country_name': data['country_name'],
                            'population': 0.0
                        }
                    all_results[country_code_2]['population'] += data['population']
                
                pbar.update(1)
    
    logging.info(f"人口统计完成，共统计 {len(all_results)} 个国家/地区")
    
    # 转换为DataFrame
    results_list = []
    for country_code_2, data in all_results.items():
        results_list.append({
            'Continent': data['continent'],
            'Country_Code_2': data['country_code_2'],
            'Country_Code_3': data['country_code_3'],
            'Country_Name': data['country_name'],
            'Population_2020': data['population']
        })
    
    results_df = pd.DataFrame(results_list)
    
    # 验证数据
    logging.info(f"转换为DataFrame，共 {len(results_df)} 行")
    
    # 检查是否有重复的国家代码
    duplicates = results_df[results_df.duplicated(subset=['Country_Code_2'], keep=False)]
    if len(duplicates) > 0:
        logging.warning(f"发现 {len(duplicates)} 个重复的国家代码:")
        logging.warning(duplicates[['Continent', 'Country_Code_2', 'Country_Code_3', 'Country_Name']].to_string())
        # 如果有重复，再次聚合
        logging.info("对重复的国家代码进行聚合...")
        results_df = results_df.groupby(['Continent', 'Country_Code_2', 'Country_Code_3', 'Country_Name'], as_index=False).agg({
            'Population_2020': 'sum'
        })
        logging.info(f"聚合后共 {len(results_df)} 个唯一国家")
    
    # 按大洲和国家代码排序
    results_df = results_df.sort_values(['Continent', 'Country_Code_2'])
    
    # 重置索引
    results_df = results_df.reset_index(drop=True)
    
    return results_df


def save_results(results_df, output_file):
    """保存结果到CSV文件"""
    logging.info(f"保存结果到: {output_file}")
    
    # 确保列顺序正确：Continent,Country_Code_2,Country_Code_3,Country_Name,Population_2020
    results_df = results_df[['Continent', 'Country_Code_2', 'Country_Code_3', 'Country_Name', 'Population_2020']]
    
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    logging.info(f"结果已保存，共 {len(results_df)} 个国家/地区")
    
    # 显示统计信息
    logging.info("\n=== 统计信息 ===")
    logging.info(f"总人口: {results_df['Population_2020'].sum():,.0f}")
    
    # 按大洲统计
    continent_stats = results_df.groupby('Continent').agg({
        'Country_Code_2': 'count',
        'Population_2020': 'sum'
    }).rename(columns={'Country_Code_2': 'country_count'})
    
    logging.info("\n各大洲统计:")
    for continent, row in continent_stats.iterrows():
        logging.info(f"  {continent}: {row['country_count']} 个国家, 人口 {row['Population_2020']:,.0f}")
    
    # 显示人口最多的前10个国家
    logging.info("\n人口最多的前10个国家:")
    top_10 = results_df.nlargest(10, 'Population_2020')
    for _, row in top_10.iterrows():
        logging.info(f"  {row['Country_Code_2']} ({row['Country_Name']}): {row['Population_2020']:,.0f}")


def main():
    """主函数"""
    logging.info("=" * 60)
    logging.info("开始计算国家人口统计")
    logging.info("=" * 60)
    
    try:
        # 1. 加载数据
        logging.info("\n=== 第一步：加载数据 ===")
        mapping_df = load_point_country_mapping()
        population_df = load_population_data()
        
        # 2. 计算国家人口（并行处理）
        logging.info("\n=== 第二步：计算国家人口（并行处理）===")
        results_df = calculate_country_population_parallel(population_df, mapping_df)
        
        # 3. 保存结果
        logging.info("\n=== 第三步：保存结果 ===")
        save_results(results_df, OUTPUT_FILE)
        
        logging.info("\n" + "=" * 60)
        logging.info("国家人口统计计算完成！")
        logging.info("=" * 60)
        
    except Exception as e:
        error_msg = f"程序执行出错: {str(e)}"
        logging.error(error_msg)
        raise


if __name__ == "__main__":
    main()

