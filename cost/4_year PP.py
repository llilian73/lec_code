"""
综合成本分析和投资回收期计算工具

功能概述：
本工具用于计算综合人均成本，包括设备成本、维护成本、节省的电费，并计算投资回收期。

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
- 维护成本比例：5%（占设备成本的5%）

输出结果：
- 文件：Z:\local_environment_creation\cost\country_yearly_per_cost_USD_PP.csv
- 列：Continent,Country_Code,Country_Name,equipment_cost,maintenance_cost,total_cost,save_elec_cost,PP
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
MAINTENANCE_RATIO = 0.05  # 维护成本比例（占设备成本的5%）

# 文件路径
COUNTRY_COST_FILE = r"Z:\local_environment_creation\cost\method2\country_cost.csv"
ELECTRICITY_SAVINGS_FILE = r"Z:\local_environment_creation\energy_consumption_gird\result\result\figure_maps_and_data\per_capita\data\average\case9_summary_average.csv"
ELECTRICITY_PRICING_FILE = r"Z:\local_environment_creation\cost\global-electricity-per-kwh-pricing-2021.csv"
OUTPUT_FILE = r"Z:\local_environment_creation\cost\method2\country_yearly_per_cost_USD_PP.csv"

def read_csv_with_encoding(file_path, keep_default_na=True):
    """读取CSV文件，尝试多种编码"""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk', 'gb2312', 'utf-8-sig']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, keep_default_na=keep_default_na)
            logger.info(f"成功使用编码 '{encoding}' 读取文件: {os.path.basename(file_path)}")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    raise ValueError(f"无法使用常见编码读取文件: {file_path}")

def load_country_cost_data():
    """加载国家成本数据"""
    logger.info("开始加载国家成本数据...")
    
    if not os.path.exists(COUNTRY_COST_FILE):
        raise FileNotFoundError(f"国家成本文件不存在: {COUNTRY_COST_FILE}")
    
    cost_df = read_csv_with_encoding(COUNTRY_COST_FILE, keep_default_na=False)
    logger.info(f"加载国家成本数据完成，共 {len(cost_df)} 个国家")
    
    # 在数据处理之前，将Namibia的Country_Code修正为'NA'
    if 'Country_Name' in cost_df.columns:
        namibia_mask = cost_df['Country_Name'].str.strip().str.lower() == 'namibia'
        namibia_count = namibia_mask.sum()
        if namibia_count > 0:
            cost_df.loc[namibia_mask, 'Country_Code'] = 'NA'
            logger.info(f"修正了 {namibia_count} 条Namibia记录的Country_Code为'NA'")
    
    return cost_df

def load_electricity_savings_data():
    """加载电力节省数据"""
    logger.info("开始加载电力节省数据...")
    
    if not os.path.exists(ELECTRICITY_SAVINGS_FILE):
        raise FileNotFoundError(f"电力节省文件不存在: {ELECTRICITY_SAVINGS_FILE}")
    
    savings_df = read_csv_with_encoding(ELECTRICITY_SAVINGS_FILE, keep_default_na=False)
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
    
    pricing_df = read_csv_with_encoding(ELECTRICITY_PRICING_FILE, keep_default_na=False)
    logger.info(f"加载电价数据完成，共 {len(pricing_df)} 个国家")
    
    # 创建国家到电价的映射（优先使用Country code，其次使用Country name）
    country_to_pricing = {}
    prices_list = []  # 收集所有有效的电价用于计算平均值
    
    for _, row in pricing_df.iterrows():
        # 处理Country code，确保'NA'不被当作缺失值
        country_code_raw = row['Country code']
        if country_code_raw == '' or (isinstance(country_code_raw, float) and pd.isna(country_code_raw)):
            country_code = ''
        else:
            country_code = str(country_code_raw).strip()
        
        country_name = row['Country name']
        price = row['Average price of 1KW/h (USD)']  # 美元/kWh
        
        # 只收集有效的电价（非空且大于0）
        if pd.notna(price) and price > 0:
            prices_list.append(price)
            # 优先使用Country code（如果有效）
            if country_code:
                country_to_pricing[country_code] = price
            # 同时使用Country name作为备选
            if country_name and pd.notna(country_name):
                country_to_pricing[country_name] = price
    
    # 计算平均电价
    if len(prices_list) > 0:
        average_price = np.mean(prices_list)
        logger.info(f"创建了 {len(country_to_pricing)} 个国家的电价映射")
        logger.info(f"平均电价: {average_price:.6f} 美元/kWh")
    else:
        average_price = 0.0
        logger.warning("未找到有效的电价数据，无法计算平均值")
    
    return country_to_pricing, average_price

def calculate_comprehensive_cost(cost_df, savings_data, pricing_data, average_price):
    """计算综合成本和投资回收期"""
    logger.info("开始计算综合成本和投资回收期...")
    
    results = []
    use_average_count = 0  # 统计使用平均电价的国家数量
    
    with tqdm(total=len(cost_df), desc="计算综合成本") as pbar:
        for _, row in cost_df.iterrows():
            continent = row['Continent']
            
            # 处理Country_Code，确保'NA'不被当作缺失值
            country_code_raw = row['Country_Code']
            if country_code_raw == '' or (isinstance(country_code_raw, float) and pd.isna(country_code_raw)):
                country_code = ''
            else:
                country_code = str(country_code_raw).strip()
            
            country_name = row['Country_Name']
            per_cost = row['per_cost']  # 人均成本（美元）
            
            # 1. 设备人均成本（直接从CSV获取，不除以寿命）
            equipment_cost = per_cost
            
            # 2. 计算人均维护成本
            maintenance_cost = equipment_cost * MAINTENANCE_RATIO
            
            # 3. 计算总成本 = 设备成本 + 维护成本
            total_cost = equipment_cost + maintenance_cost
            
            # 4. 计算节省的电费
            save_elec_cost = 0.0
            
            # 获取电力节省数据
            electricity_savings = savings_data.get(country_code, 0)
            if electricity_savings == 0:
                # 如果Country code匹配不上，尝试Country name
                electricity_savings = savings_data.get(country_name, 0)
            
            if electricity_savings > 0:
                # 获取电价数据
                electricity_price = pricing_data.get(country_code, None)
                if electricity_price is None or electricity_price <= 0:
                    # 如果Country code匹配不上，尝试Country name
                    electricity_price = pricing_data.get(country_name, None)
                
                # 如果仍然找不到电价或电价无效，使用平均电价
                if electricity_price is None or electricity_price <= 0:
                    if average_price > 0:
                        electricity_price = average_price
                        use_average_count += 1
                        logger.debug(f"国家 {country_name} ({country_code}) 使用平均电价: {average_price:.6f} 美元/kWh")
                    else:
                        logger.warning(f"国家 {country_name} ({country_code}) 无法找到电价数据，且平均电价为0，节省电费将设为0")
                        electricity_price = 0
                
                # 节省的电费 = 节省的电力(kWh/person) × 电价(美元/kWh)
                save_elec_cost = electricity_savings * electricity_price
            else:
                logger.warning(f"未找到国家 {country_name} ({country_code}) 的电力节省数据")
            
            # 5. 计算投资回收期 PP = total_cost / save_elec_cost
            if save_elec_cost > 0:
                PP = total_cost / save_elec_cost  # 单位：年
            else:
                PP = np.nan  # 如果年节省电费为0或负数，无法计算回收期
            
            results.append({
                'Continent': continent,
                'Country_Code': country_code,
                'Country_Name': country_name,
                'equipment_cost': equipment_cost,
                'maintenance_cost': maintenance_cost,
                'total_cost': total_cost,
                'save_elec_cost': save_elec_cost,
                'PP': PP
            })
            
            pbar.update(1)
    
    logger.info(f"完成综合成本计算，共处理 {len(results)} 个国家")
    logger.info(f"使用平均电价的国家数: {use_average_count}")
    
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
    
    # 确保列的顺序符合要求：Continent,Country_Code,Country_Name,equipment_cost,maintenance_cost,total_cost,save_elec_cost,PP
    column_order = ['Continent', 'Country_Code', 'Country_Name', 'equipment_cost', 'maintenance_cost', 'total_cost', 'save_elec_cost', 'PP']
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
    logger.info(f"总成本范围: {results_df['total_cost'].min():.4f} 美元 - {results_df['total_cost'].max():.4f} 美元")
    logger.info(f"平均总成本: {results_df['total_cost'].mean():.4f} 美元")
    logger.info(f"节省电费范围: {results_df['save_elec_cost'].min():.4f} 美元 - {results_df['save_elec_cost'].max():.4f} 美元")
    logger.info(f"平均节省电费: {results_df['save_elec_cost'].mean():.4f} 美元")
    
    # 投资回收期统计（排除NaN值）
    valid_pp = results_df['PP'].dropna()
    if len(valid_pp) > 0:
        logger.info(f"投资回收期范围: {valid_pp.min():.2f} 年 - {valid_pp.max():.2f} 年")
        logger.info(f"平均投资回收期: {valid_pp.mean():.2f} 年")
        logger.info(f"中位数投资回收期: {valid_pp.median():.2f} 年")
        logger.info(f"无法计算投资回收期的国家数: {len(results_df) - len(valid_pp)}")
    else:
        logger.info("没有有效的投资回收期数据")
    
    # 按大洲统计
    logger.info("按大洲统计:")
    continent_stats = results_df.groupby('Continent').agg({
        'equipment_cost': ['sum', 'mean', 'count'],
        'maintenance_cost': 'mean',
        'total_cost': 'mean',
        'save_elec_cost': 'mean',
        'PP': 'mean'
    })
    
    for continent in continent_stats.index:
        total_equipment = continent_stats.loc[continent, ('equipment_cost', 'sum')]
        avg_equipment = continent_stats.loc[continent, ('equipment_cost', 'mean')]
        country_count = continent_stats.loc[continent, ('equipment_cost', 'count')]
        avg_maintenance = continent_stats.loc[continent, ('maintenance_cost', 'mean')]
        avg_total = continent_stats.loc[continent, ('total_cost', 'mean')]
        avg_savings = continent_stats.loc[continent, ('save_elec_cost', 'mean')]
        avg_pp = continent_stats.loc[continent, ('PP', 'mean')]
        
        logger.info(f"  {continent}: 总设备成本 {total_equipment:.2f} 美元, 平均设备成本 {avg_equipment:.4f} 美元, "
                   f"平均维护成本 {avg_maintenance:.4f} 美元, 平均总成本 {avg_total:.4f} 美元, "
                   f"平均节省电费 {avg_savings:.4f} 美元, 平均投资回收期 {avg_pp:.2f} 年, 国家数 {country_count}")
    
    # 显示投资回收期最短的前10个国家
    valid_pp_df = results_df.dropna(subset=['PP'])
    if len(valid_pp_df) > 0:
        logger.info("投资回收期最短的前10个国家:")
        top_countries = valid_pp_df.nsmallest(10, 'PP')
        for _, row in top_countries.iterrows():
            logger.info(f"  {row['Country_Name']} ({row['Country_Code']}): {row['PP']:.2f} 年 (总成本: {row['total_cost']:.4f} 美元, 节省电费: {row['save_elec_cost']:.4f} 美元/年)")
        
        # 显示投资回收期最长的前10个国家
        logger.info("投资回收期最长的前10个国家:")
        top_countries = valid_pp_df.nlargest(10, 'PP')
        for _, row in top_countries.iterrows():
            logger.info(f"  {row['Country_Name']} ({row['Country_Code']}): {row['PP']:.2f} 年 (总成本: {row['total_cost']:.4f} 美元, 节省电费: {row['save_elec_cost']:.4f} 美元/年)")

def main():
    """主函数"""
    logger.info("开始综合成本分析...")
    
    try:
        # 1. 加载数据
        logger.info("=== 第一步：加载数据 ===")
        
        cost_df = load_country_cost_data()
        savings_data = load_electricity_savings_data()
        pricing_data, average_price = load_electricity_pricing_data()
        
        # 2. 计算综合成本
        logger.info("=== 第二步：计算综合成本 ===")
        
        results = calculate_comprehensive_cost(cost_df, savings_data, pricing_data, average_price)
        
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