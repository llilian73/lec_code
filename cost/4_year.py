"""
综合成本分析工具

功能概述：
本工具用于计算综合人均年均成本，包括设备成本、维护成本和节省的电费。

输入数据：
1. 国家成本数据：
   - 文件：Z:\local_environment_creation\cost\country_cost.csv
   - 来源：3_country_merge.py的输出

2. 电力节省数据：
   - 文件：Z:\local_environment_creation\energy_consumption_gird\result\result\figure_maps_and_data\per_capita\data\average\case9_summary_average.csv
   - 包含各国电力节省数据

3. 电价数据：
   - 文件：Z:\local_environment_creation\cost\global-electricity-per-kwh-pricing-2021.csv
   - 包含各国电价信息

计算参数：
- 设备寿命：10年
- 维护成本比例：5%（占设备成本的5%）

输出结果：
- 文件：Z:\local_environment_creation\cost\country_yearly_per_cost_USD.csv
- 列：Continent,Country_Code,Country_Name,equipment_cost,maintenance_cost,save_elec_cost,total_cost
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
    file_handler = logging.FileHandler('4_year.log', encoding='utf-8')
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
EQUIPMENT_LIFETIME = 10  # 设备寿命（年）
MAINTENANCE_RATIO = 0.05  # 维护成本比例（占设备成本的5%）

# 文件路径
COUNTRY_COST_FILE = r"Z:\local_environment_creation\cost\country_cost.csv"
ELECTRICITY_SAVINGS_FILE = r"Z:\local_environment_creation\energy_consumption_gird\result\result\figure_maps_and_data\per_capita\data\average\case9_summary_average.csv"
ELECTRICITY_PRICING_FILE = r"Z:\local_environment_creation\cost\global-electricity-per-kwh-pricing-2021.csv"
OUTPUT_FILE = r"Z:\local_environment_creation\cost\country_yearly_per_cost_USD.csv"

def load_country_cost_data():
    """加载国家成本数据"""
    logger.info("开始加载国家成本数据...")
    
    if not os.path.exists(COUNTRY_COST_FILE):
        raise FileNotFoundError(f"国家成本文件不存在: {COUNTRY_COST_FILE}")
    
    cost_df = pd.read_csv(COUNTRY_COST_FILE)
    logger.info(f"加载国家成本数据完成，共 {len(cost_df)} 个国家")
    
    return cost_df

def load_electricity_savings_data():
    """加载电力节省数据"""
    logger.info("开始加载电力节省数据...")
    
    if not os.path.exists(ELECTRICITY_SAVINGS_FILE):
        raise FileNotFoundError(f"电力节省文件不存在: {ELECTRICITY_SAVINGS_FILE}")
    
    savings_df = pd.read_csv(ELECTRICITY_SAVINGS_FILE)
    logger.info(f"加载电力节省数据完成，共 {len(savings_df)} 个国家")
    
    # 创建国家到节省电力的映射
    country_to_savings = {}
    for _, row in savings_df.iterrows():
        country = row['country']
        total_difference = row['total_difference']  # kWh/person
        country_to_savings[country] = total_difference
    
    logger.info(f"创建了 {len(country_to_savings)} 个国家的电力节省映射")
    
    return country_to_savings

def load_electricity_pricing_data():
    """加载电价数据"""
    logger.info("开始加载电价数据...")
    
    if not os.path.exists(ELECTRICITY_PRICING_FILE):
        raise FileNotFoundError(f"电价文件不存在: {ELECTRICITY_PRICING_FILE}")
    
    pricing_df = pd.read_csv(ELECTRICITY_PRICING_FILE)
    logger.info(f"加载电价数据完成，共 {len(pricing_df)} 个国家")
    
    # 创建国家到电价的映射（优先使用Country code，其次使用Country name）
    country_to_pricing = {}
    for _, row in pricing_df.iterrows():
        country_code = row['Country code']
        country_name = row['Country name']
        price = row['Average price of 1KW/h (USD)']  # 美元/kWh
        
        # 优先使用Country code
        country_to_pricing[country_code] = price
        # 同时使用Country name作为备选
        country_to_pricing[country_name] = price
    
    logger.info(f"创建了 {len(country_to_pricing)} 个国家的电价映射")
    
    return country_to_pricing

def calculate_comprehensive_cost(cost_df, savings_data, pricing_data):
    """计算综合成本"""
    logger.info("开始计算综合成本...")
    
    results = []
    
    with tqdm(total=len(cost_df), desc="计算综合成本") as pbar:
        for _, row in cost_df.iterrows():
            continent = row['Continent']
            country_code = row['Country_Code']
            country_name = row['Country_Name']
            per_cost = row['per_cost']  # 人均成本（美元）
            
            # 1. 计算人均年均设备成本
            equipment_cost = per_cost / EQUIPMENT_LIFETIME
            
            # 2. 计算人均年均维护成本
            maintenance_cost = equipment_cost * MAINTENANCE_RATIO
            
            # 3. 计算节省的电费
            save_elec_cost = 0.0
            
            # 获取电力节省数据
            electricity_savings = savings_data.get(country_code, 0)
            if electricity_savings == 0:
                # 如果Country code匹配不上，尝试Country name
                electricity_savings = savings_data.get(country_name, 0)
            
            if electricity_savings > 0:
                # 获取电价数据
                electricity_price = pricing_data.get(country_code, 0)
                if electricity_price == 0:
                    # 如果Country code匹配不上，尝试Country name
                    electricity_price = pricing_data.get(country_name, 0)
                
                if electricity_price > 0:
                    # 节省的电费 = 节省的电力(kWh/person) × 电价(美元/kWh)
                    save_elec_cost = electricity_savings * electricity_price
                else:
                    logger.warning(f"未找到国家 {country_name} ({country_code}) 的电价数据")
            else:
                logger.warning(f"未找到国家 {country_name} ({country_code}) 的电力节省数据")
            
            # 4. 计算综合人均年均成本
            total_cost = equipment_cost + maintenance_cost - save_elec_cost
            
            results.append({
                'Continent': continent,
                'Country_Code': country_code,
                'Country_Name': country_name,
                'equipment_cost': equipment_cost,
                'maintenance_cost': maintenance_cost,
                'save_elec_cost': save_elec_cost,
                'total_cost': total_cost
            })
            
            pbar.update(1)
    
    logger.info(f"完成综合成本计算，共处理 {len(results)} 个国家")
    
    return results

def save_results(results):
    """保存结果"""
    logger.info("开始保存结果...")
    
    results_df = pd.DataFrame(results)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_FILE)
    os.makedirs(output_dir, exist_ok=True)
    
    # 按大洲和国家代码排序
    results_df = results_df.sort_values(['Continent', 'Country_Code'])
    
    # 确保列的顺序符合要求：Continent,Country_Code,Country_Name,equipment_cost,maintenance_cost,save_elec_cost,total_cost
    column_order = ['Continent', 'Country_Code', 'Country_Name', 'equipment_cost', 'maintenance_cost', 'save_elec_cost', 'total_cost']
    # 只保留存在的列
    existing_columns = [col for col in column_order if col in results_df.columns]
    results_df = results_df[existing_columns]
    
    results_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    # 验证列顺序
    logger.info(f"输出CSV文件的列顺序: {list(results_df.columns)}")
    
    logger.info(f"结果已保存至: {OUTPUT_FILE}")
    logger.info(f"共处理了 {len(results)} 个国家")
    
    # 统计信息
    logger.info("=== 统计信息 ===")
    logger.info(f"设备成本范围: {results_df['equipment_cost'].min():.4f} 美元 - {results_df['equipment_cost'].max():.4f} 美元")
    logger.info(f"平均设备成本: {results_df['equipment_cost'].mean():.4f} 美元")
    logger.info(f"维护成本范围: {results_df['maintenance_cost'].min():.4f} 美元 - {results_df['maintenance_cost'].max():.4f} 美元")
    logger.info(f"平均维护成本: {results_df['maintenance_cost'].mean():.4f} 美元")
    logger.info(f"节省电费范围: {results_df['save_elec_cost'].min():.4f} 美元 - {results_df['save_elec_cost'].max():.4f} 美元")
    logger.info(f"平均节省电费: {results_df['save_elec_cost'].mean():.4f} 美元")
    logger.info(f"综合成本范围: {results_df['total_cost'].min():.4f} 美元 - {results_df['total_cost'].max():.4f} 美元")
    logger.info(f"平均综合成本: {results_df['total_cost'].mean():.4f} 美元")
    
    # 按大洲统计
    logger.info("按大洲统计:")
    continent_stats = results_df.groupby('Continent').agg({
        'equipment_cost': ['sum', 'mean', 'count'],
        'maintenance_cost': 'mean',
        'save_elec_cost': 'mean',
        'total_cost': 'mean'
    })
    
    for continent in continent_stats.index:
        total_equipment = continent_stats.loc[continent, ('equipment_cost', 'sum')]
        avg_equipment = continent_stats.loc[continent, ('equipment_cost', 'mean')]
        country_count = continent_stats.loc[continent, ('equipment_cost', 'count')]
        avg_maintenance = continent_stats.loc[continent, ('maintenance_cost', 'mean')]
        avg_savings = continent_stats.loc[continent, ('save_elec_cost', 'mean')]
        avg_total = continent_stats.loc[continent, ('total_cost', 'mean')]
        
        logger.info(f"  {continent}: 总设备成本 {total_equipment:.2f} 美元, 平均设备成本 {avg_equipment:.4f} 美元, "
                   f"平均维护成本 {avg_maintenance:.4f} 美元, 平均节省电费 {avg_savings:.4f} 美元, "
                   f"平均综合成本 {avg_total:.4f} 美元, 国家数 {country_count}")
    
    # 显示综合成本最低的前10个国家
    logger.info("综合成本最低的前10个国家:")
    top_countries = results_df.nsmallest(10, 'total_cost')
    for _, row in top_countries.iterrows():
        logger.info(f"  {row['Country_Name']} ({row['Country_Code']}): {row['total_cost']:.4f} 美元")
    
    # 显示综合成本最高的前10个国家
    logger.info("综合成本最高的前10个国家:")
    top_countries = results_df.nlargest(10, 'total_cost')
    for _, row in top_countries.iterrows():
        logger.info(f"  {row['Country_Name']} ({row['Country_Code']}): {row['total_cost']:.4f} 美元")

def main():
    """主函数"""
    logger.info("开始综合成本分析...")
    
    try:
        # 1. 加载数据
        logger.info("=== 第一步：加载数据 ===")
        
        cost_df = load_country_cost_data()
        savings_data = load_electricity_savings_data()
        pricing_data = load_electricity_pricing_data()
        
        # 2. 计算综合成本
        logger.info("=== 第二步：计算综合成本 ===")
        
        results = calculate_comprehensive_cost(cost_df, savings_data, pricing_data)
        
        # 3. 保存结果
        logger.info("=== 第三步：保存结果 ===")
        
        save_results(results)
        
        logger.info("综合成本分析完成！")
        
    except Exception as e:
        error_msg = f"主程序执行出错: {str(e)}"
        logger.error(error_msg)
        raise

if __name__ == "__main__":
    main()