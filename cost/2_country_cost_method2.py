"""
换热器成本计算工具

功能概述：
本工具用于计算换热器成本，包括铝片成本和风扇成本。
基于country_cost.py的输出结果，计算换热器面积、铝片体积和成本。

输入数据：
1. 功率数据：
   - 文件：Z:\local_environment_creation\cost\country_energy_power.csv
   - 包含每个网格点的最大功率Pmax

2. 人口数据：
   - 文件：Z:\local_environment_creation\energy_consumption_gird\result\data\population_points.csv
   - 包含每个网格点的人口数

计算参数：
- COP（性能系数）：3
- alpha系数：6.25%
- 平均温差：11.3°C
- 换热系数：11.1 W/m²K
- 铝片厚度：0.1mm
- 铝价格：2619美元/吨
- 风扇价格：14美元/个
- 外机功率：11.6 kW
- 外机价格：2730美元/台
- 换热器成本系数：2.0（铝片成本的倍数）

输出结果：
- 文件：Z:\local_environment_creation\cost\country_energy_power_with_cost.csv
- 新增列：load, HE_area(m2), HE_cost, fan_cost, outdoor_unit_cost, cost
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
import math
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
    file_handler = logging.FileHandler('2_country_cost.log', encoding='utf-8')
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
INPUT_FILE = r"Z:\local_environment_creation\cost\country_energy_power.csv"
POPULATION_FILE = r"Z:\local_environment_creation\energy_consumption_gird\result\data\population_points.csv"
ILO_EARNINGS_FILE = r"Z:\local_environment_creation\cost\Earnings\ILO_Average_hourly_earnings_processed.csv"
OUTPUT_DIR = r"Z:\local_environment_creation\cost\method2"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "country_energy_power_with_cost.csv")

# 计算参数
COP = 3.0  # 性能系数
ALPHA = 0.0625  # alpha系数：6.25%
DELTA_T = 11.3  # 平均温差 (°C)
HEAT_TRANSFER_COEFF = 11.1  # 换热系数 (W/m²K)
ALUMINUM_THICKNESS = 0.1e-3  # 铝片厚度 (m) - 0.1mm转换为米
ALUMINUM_PRICE = 2619  # 铝价格 (美元/吨)
FAN_PRICE = 14  # 风扇价格 (美元/个)
OUTDOOR_UNIT_PRICE_PER_KW = 133.168  # 外机价格 (美元/kW)
HEAT_EXCHANGER_COST_RATIO = 2.0  # 换热器成本系数（铝片成本的倍数）
INSTALLATION_TIME_PER_PERSON = 2.0  # 安装时长 (小时/人)

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

def load_power_data():
    """加载功率数据"""
    logger.info("开始加载功率数据...")
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"功率数据文件不存在: {INPUT_FILE}")
    
    power_df = read_csv_with_encoding(INPUT_FILE, keep_default_na=False)
    logger.info(f"加载功率数据完成，共 {len(power_df)} 个点")
    
    return power_df

def load_population_data():
    """加载人口数据"""
    logger.info("开始加载人口数据...")
    
    if not os.path.exists(POPULATION_FILE):
        raise FileNotFoundError(f"人口数据文件不存在: {POPULATION_FILE}")
    
    population_df = pd.read_csv(POPULATION_FILE)
    logger.info(f"加载人口数据完成，共 {len(population_df)} 个点")
    
    # 创建坐标到人口的映射字典
    # 将坐标四舍五入到2位小数，因为population_points.csv中lat为3位小数但第三位都是0
    coord_to_population = {}
    for _, row in population_df.iterrows():
        lat = round(float(row['lat']), 2)
        lon = round(float(row['lon']), 2)
        population = row['population']
        coord_to_population[(lat, lon)] = population
    
    logger.info(f"创建了 {len(coord_to_population)} 个点的人口映射")
    
    return coord_to_population

def load_ilo_earnings_data():
    """加载ILO时薪数据"""
    logger.info("开始加载ILO时薪数据...")
    
    if not os.path.exists(ILO_EARNINGS_FILE):
        raise FileNotFoundError(f"ILO时薪数据文件不存在: {ILO_EARNINGS_FILE}")
    
    earnings_df = pd.read_csv(ILO_EARNINGS_FILE, keep_default_na=False)
    logger.info(f"加载ILO时薪数据完成，共 {len(earnings_df)} 个国家")
    
    # 创建Country_Code到时薪的映射字典
    country_to_earnings = {}
    for _, row in earnings_df.iterrows():
        # 处理Country_Code，确保'NA'不被当作缺失值
        country_code_raw = row['Country_Code']
        if country_code_raw == '' or (isinstance(country_code_raw, float) and pd.isna(country_code_raw)):
            country_code = ''
        else:
            country_code = str(country_code_raw).strip().upper()
        
        obs_value = row['obs_value']
        
        if country_code:
            # 尝试转换为数值
            try:
                earnings_value = float(obs_value) if pd.notna(obs_value) and str(obs_value).strip() != '' else None
                if earnings_value is not None:
                    country_to_earnings[country_code] = earnings_value
            except (ValueError, TypeError):
                pass
    
    logger.info(f"创建了 {len(country_to_earnings)} 个国家的时薪映射")
    
    return country_to_earnings

def calculate_heat_exchanger_area(pmax):
    """计算换热器面积"""
    # load = Pmax * COP * alpha
    load = pmax * COP * ALPHA
    
    # HE_area = load / (HEAT_TRANSFER_COEFF * DELTA_T)
    # 换热器面积 = 热负荷 / (换热系数 * 温差)
    he_area = load / (HEAT_TRANSFER_COEFF * DELTA_T)
    
    return load, he_area

def calculate_aluminum_cost(he_area):
    """计算铝片成本"""
    # 铝片体积 = 面积 * 厚度
    aluminum_volume = he_area * ALUMINUM_THICKNESS  # m³
    
    # 铝密度约为2700 kg/m³
    aluminum_density = 2700  # kg/m³
    aluminum_mass = aluminum_volume * aluminum_density  # kg
    
    # 转换为吨
    aluminum_mass_ton = aluminum_mass / 1000  # 吨
    
    # 成本 = 质量(吨) * 价格(美元/吨)
    he_cost = aluminum_mass_ton * ALUMINUM_PRICE
    
    return he_cost

def calculate_fan_cost(population):
    """计算风扇成本"""
    # 风扇成本 = 人数 * 风扇价格(美元/个)
    fan_cost = population * FAN_PRICE
    return fan_cost

def calculate_outdoor_unit_cost(load):
    """计算外机成本"""
    # load单位为W，需要转换为kW
    load_kw = load / 1000.0  # 转换为kW
    
    # 外机成本 = load_kw * 价格(美元/kW)
    outdoor_unit_cost = load_kw * OUTDOOR_UNIT_PRICE_PER_KW
    
    return outdoor_unit_cost

def calculate_installation_cost(population, hourly_earnings):
    """计算安装成本"""
    # 安装成本 = 安装时长(小时/人) × 人数 × 工人时薪(美元/小时)
    # 如果时薪数据缺失，返回0
    if hourly_earnings is None or pd.isna(hourly_earnings):
        return 0.0
    
    installation_cost = INSTALLATION_TIME_PER_PERSON * population * hourly_earnings
    return installation_cost

def process_cost_calculation():
    """处理成本计算"""
    logger.info("开始成本计算...")
    
    # 加载数据
    power_df = load_power_data()
    coord_to_population = load_population_data()
    country_to_earnings = load_ilo_earnings_data()
    
    # 初始化结果列表
    results = []
    
    # 统计缺失时薪数据的国家
    missing_earnings_count = 0
    
    # 使用进度条显示处理进度
    with tqdm(total=len(power_df), desc="计算成本") as pbar:
        for _, row in power_df.iterrows():
            lat = row['lat']
            lon = row['lon']
            pmax = row['Pmax']
            # 处理Country_Code，确保'NA'不被当作缺失值
            country_code_raw = row['Country_Code']
            if country_code_raw == '' or (isinstance(country_code_raw, float) and pd.isna(country_code_raw)):
                country_code = ''
            else:
                country_code = str(country_code_raw).strip().upper()
            
            # 获取人口数据（将坐标四舍五入到2位小数进行匹配）
            lat_rounded = round(float(lat), 2)
            lon_rounded = round(float(lon), 2)
            population = coord_to_population.get((lat_rounded, lon_rounded), 0)
            
            # 获取工人时薪
            hourly_earnings = country_to_earnings.get(country_code, None)
            if hourly_earnings is None:
                missing_earnings_count += 1
            
            # 计算换热器面积
            load, he_area = calculate_heat_exchanger_area(pmax)
            
            # 计算铝片成本
            aluminum_cost = calculate_aluminum_cost(he_area)
            
            # 计算换热器成本（铝片成本的2倍）
            he_cost = aluminum_cost * HEAT_EXCHANGER_COST_RATIO
            
            # 计算风扇成本
            fan_cost = calculate_fan_cost(population)
            
            # 计算外机成本
            outdoor_unit_cost = calculate_outdoor_unit_cost(load)
            
            # 计算安装成本
            installation_cost = calculate_installation_cost(population, hourly_earnings)
            
            # 总成本 = 换热器成本 + 风机成本 + 外机成本 + 安装成本
            total_cost = he_cost + fan_cost + outdoor_unit_cost + installation_cost
            
            # 四舍五入到三位小数
            load = round(load, 3)
            he_area = round(he_area, 3)
            he_cost = round(he_cost, 3)
            fan_cost = round(fan_cost, 3)
            outdoor_unit_cost = round(outdoor_unit_cost, 3)
            installation_cost = round(installation_cost, 3)
            total_cost = round(total_cost, 3)
            
            # 添加到结果列表
            results.append({
                'Continent': row['Continent'],
                'Country_Code': row['Country_Code'],
                'Country_Name': row['Country_Name'],
                'lat': lat,
                'lon': lon,
                'Pmax': pmax,
                'load': load,
                'HE_area(m2)': he_area,
                'HE_cost': he_cost,
                'fan_cost': fan_cost,
                'outdoor_unit_cost': outdoor_unit_cost,
                'installation_cost': installation_cost,
                'cost': total_cost,
                'population': population
            })
            
            pbar.update(1)
    
    if missing_earnings_count > 0:
        logger.warning(f"有 {missing_earnings_count} 个点缺少时薪数据，安装成本为0")
    
    return results

def save_results(results):
    """保存结果"""
    logger.info("开始保存结果...")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    logger.info(f"结果已保存至: {OUTPUT_FILE}")
    logger.info(f"共处理了 {len(results)} 个网格点")
    
    # 统计信息
    logger.info("=== 统计信息 ===")
    logger.info(f"总成本范围: {results_df['cost'].min():.2f} 美元 - {results_df['cost'].max():.2f} 美元")
    logger.info(f"平均成本: {results_df['cost'].mean():.2f} 美元")
    logger.info(f"换热器面积范围: {results_df['HE_area(m2)'].min():.2f} m² - {results_df['HE_area(m2)'].max():.2f} m²")
    logger.info(f"平均换热器面积: {results_df['HE_area(m2)'].mean():.2f} m²")
    logger.info(f"安装成本范围: {results_df['installation_cost'].min():.2f} 美元 - {results_df['installation_cost'].max():.2f} 美元")
    logger.info(f"平均安装成本: {results_df['installation_cost'].mean():.2f} 美元")
    
    # 按大洲统计
    logger.info("按大洲统计总成本:")
    continent_stats = results_df.groupby('Continent')['cost'].agg(['sum', 'mean', 'count'])
    for continent, stats in continent_stats.iterrows():
        logger.info(f"  {continent}: 总成本 {stats['sum']:.2f} 美元, 平均成本 {stats['mean']:.2f} 美元, 点数 {stats['count']}")

def main():
    """主函数"""
    logger.info("开始换热器成本计算...")
    
    try:
        # 处理成本计算
        results = process_cost_calculation()
        
        # 保存结果
        save_results(results)
        
        logger.info("换热器成本计算完成！")
        
    except Exception as e:
        error_msg = f"主程序执行出错: {str(e)}"
        logger.error(error_msg)
        raise

if __name__ == "__main__":
    main()
