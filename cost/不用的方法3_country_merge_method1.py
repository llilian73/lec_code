"""
国家成本聚合工具

功能概述：
本工具用于将网格点级别的成本数据聚合到国家级别。
基于2_country_cost.py的输出结果，按国家汇总成本数据，并计算人均成本。

输入数据：
1. 网格点成本数据：
   - 文件：Z:\local_environment_creation\cost\country_energy_power_with_cost.csv
   - 包含每个网格点的成本信息

2. 点-国家映射数据：
   - 文件：Z:\local_environment_creation\energy_consumption_gird\result\point_country_mapping.csv
   - 包含每个网格点对应的国家信息

3. 国家人口数据：
   - 文件：Z:\local_environment_creation\energy_consumption_gird\result\country_population_2020.csv
   - 包含2020年各国人口数据

输出结果：
- 文件：Z:\local_environment_creation\cost\country_cost.csv
- 格式：Continent,Country_Code,Country_Name,cost,per_cost
- cost：国家总成本（美元）
- per_cost：人均成本（美元/人）
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from tqdm import tqdm

# 设置日志记录
def setup_logging():
    """设置日志记录"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件处理器
    file_handler = logging.FileHandler('3_country_merge.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 初始化日志
logger = setup_logging()

# 配置参数
INPUT_FILE = r"Z:\local_environment_creation\cost\country_energy_power_with_cost.csv"
POINT_COUNTRY_MAPPING_FILE = r"Z:\local_environment_creation\energy_consumption_gird\result\point_country_mapping.csv"
COUNTRY_POPULATION_FILE = r"Z:\local_environment_creation\energy_consumption_gird\result\country_population_2020.csv"
OUTPUT_FILE = r"Z:\local_environment_creation\cost\country_cost.csv"

def load_cost_data():
    """加载网格点成本数据"""
    logger.info("开始加载网格点成本数据...")
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"成本数据文件不存在: {INPUT_FILE}")
    
    cost_df = pd.read_csv(INPUT_FILE)
    logger.info(f"加载成本数据完成，共 {len(cost_df)} 个点")
    
    return cost_df

def load_point_country_mapping():
    """加载点-国家映射数据"""
    logger.info("开始加载点-国家映射数据...")
    
    if not os.path.exists(POINT_COUNTRY_MAPPING_FILE):
        raise FileNotFoundError(f"点-国家映射文件不存在: {POINT_COUNTRY_MAPPING_FILE}")
    
    mapping_df = pd.read_csv(POINT_COUNTRY_MAPPING_FILE)
    logger.info(f"加载点-国家映射数据完成，共 {len(mapping_df)} 个点")
    
    # 过滤掉Unknown地区
    original_count = len(mapping_df)
    mapping_df = mapping_df[mapping_df['Country_Code'] != 'Unknown']
    filtered_count = len(mapping_df)
    logger.info(f"过滤掉Unknown地区后，剩余 {filtered_count} 个点（过滤了 {original_count - filtered_count} 个点）")
    
    return mapping_df

def load_country_population():
    """加载国家人口数据"""
    logger.info("开始加载国家人口数据...")
    
    if not os.path.exists(COUNTRY_POPULATION_FILE):
        raise FileNotFoundError(f"国家人口文件不存在: {COUNTRY_POPULATION_FILE}")
    
    population_df = pd.read_csv(COUNTRY_POPULATION_FILE)
    logger.info(f"加载国家人口数据完成，共 {len(population_df)} 个国家")
    
    # 创建国家代码到人口的映射字典
    country_to_population = {}
    for _, row in population_df.iterrows():
        country_code = row['Country_Code']
        population = row['Population_2020']
        country_to_population[country_code] = population
    
    logger.info(f"创建了 {len(country_to_population)} 个国家的人口映射")
    
    return country_to_population

def aggregate_by_country(cost_df, mapping_df, country_population):
    """按国家聚合成本数据"""
    logger.info("开始按国家聚合成本数据...")
    
    # 创建坐标到国家的映射字典
    coord_to_country = {}
    for _, row in mapping_df.iterrows():
        lat = row['lat']
        lon = row['lon']
        country_code = row['Country_Code']
        continent = row['Continent']
        country_name = row['Country_Name']
        
        # 处理中国特殊情况
        if country_code == 'CN-TW':
            country_code = 'CN'
        
        coord_to_country[(lat, lon)] = {
            'country_code': country_code,
            'continent': continent,
            'country_name': country_name
        }
    
    logger.info(f"创建了 {len(coord_to_country)} 个点的国家映射")
    
    # 按国家聚合成本数据
    country_costs = {}
    
    with tqdm(total=len(cost_df), desc="聚合成本数据") as pbar:
        for _, row in cost_df.iterrows():
            lat = row['lat']
            lon = row['lon']
            cost = row['cost']
            
            # 获取国家信息
            country_info = coord_to_country.get((lat, lon), None)
            
            if country_info is None:
                # 如果找不到国家映射，跳过
                pbar.update(1)
                continue
            
            country_code = country_info['country_code']
            continent = country_info['continent']
            country_name = country_info['country_name']
            
            # 初始化国家成本数据
            if country_code not in country_costs:
                country_costs[country_code] = {
                    'continent': continent,
                    'country_name': country_name,
                    'total_cost': 0.0,
                    'point_count': 0
                }
            
            # 累加成本
            country_costs[country_code]['total_cost'] += cost
            country_costs[country_code]['point_count'] += 1
            
            pbar.update(1)
    
    logger.info(f"成功聚合了 {len(country_costs)} 个国家的成本数据")
    
    # 计算人均成本
    results = []
    for country_code, data in country_costs.items():
        continent = data['continent']
        country_name = data['country_name']
        total_cost = data['total_cost']
        point_count = data['point_count']
        
        # 获取人口数据
        population = country_population.get(country_code, 0)
        
        # 计算人均成本
        if population > 0:
            per_cost = total_cost / population
        else:
            per_cost = 0
            logger.warning(f"国家 {country_name} ({country_code}) 没有人口数据")
        
        results.append({
            'Continent': continent,
            'Country_Code': country_code,
            'Country_Name': country_name,
            'cost': total_cost,
            'per_cost': per_cost,
            'population': population,
            'point_count': point_count
        })
    
    return results

def save_results(results):
    """保存结果"""
    logger.info("开始保存结果...")
    
    results_df = pd.DataFrame(results)
    
    # 只保留需要的列
    output_df = results_df[['Continent', 'Country_Code', 'Country_Name', 'cost', 'per_cost']]
    
    # 按大洲和国家代码排序
    output_df = output_df.sort_values(['Continent', 'Country_Code'])
    
    output_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    logger.info(f"结果已保存至: {OUTPUT_FILE}")
    logger.info(f"共处理了 {len(results)} 个国家")
    
    # 统计信息
    logger.info("=== 统计信息 ===")
    logger.info(f"总成本范围: {results_df['cost'].min():.2f} 美元 - {results_df['cost'].max():.2f} 美元")
    logger.info(f"平均总成本: {results_df['cost'].mean():.2f} 美元")
    logger.info(f"人均成本范围: {results_df['per_cost'].min():.2f} 美元/人 - {results_df['per_cost'].max():.2f} 美元/人")
    logger.info(f"平均人均成本: {results_df['per_cost'].mean():.2f} 美元/人")
    
    # 按大洲统计
    logger.info("按大洲统计:")
    continent_stats = results_df.groupby('Continent').agg({
        'cost': ['sum', 'mean', 'count'],
        'per_cost': 'mean'
    })
    
    for continent in continent_stats.index:
        total_cost = continent_stats.loc[continent, ('cost', 'sum')]
        avg_cost = continent_stats.loc[continent, ('cost', 'mean')]
        country_count = continent_stats.loc[continent, ('cost', 'count')]
        avg_per_cost = continent_stats.loc[continent, ('per_cost', 'mean')]
        
        logger.info(f"  {continent}: 总成本 {total_cost:.2f} 美元, 平均成本 {avg_cost:.2f} 美元, "
                   f"平均人均成本 {avg_per_cost:.2f} 美元/人, 国家数 {country_count}")
    
    # 显示成本最高的前10个国家
    logger.info("成本最高的前10个国家:")
    top_countries = results_df.nlargest(10, 'cost')
    for _, row in top_countries.iterrows():
        logger.info(f"  {row['Country_Name']} ({row['Country_Code']}): {row['cost']:.2f} 美元")

def main():
    """主函数"""
    logger.info("开始国家成本聚合...")
    
    try:
        # 1. 加载数据
        logger.info("=== 第一步：加载数据 ===")
        
        cost_df = load_cost_data()
        mapping_df = load_point_country_mapping()
        country_population = load_country_population()
        
        # 2. 按国家聚合成本数据
        logger.info("=== 第二步：按国家聚合成本数据 ===")
        
        results = aggregate_by_country(cost_df, mapping_df, country_population)
        
        # 3. 保存结果
        logger.info("=== 第三步：保存结果 ===")
        
        save_results(results)
        
        logger.info("国家成本聚合完成！")
        
    except Exception as e:
        error_msg = f"主程序执行出错: {str(e)}"
        logger.error(error_msg)
        raise

if __name__ == "__main__":
    main()
