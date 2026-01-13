"""
国家能耗功率计算工具

功能概述：
本工具用于计算每个网格点在2016-2020年期间的最大能耗功率。通过分析每个经纬度点在不同年份的ref工况数据，
找到cooling和heating功率的最大值，应用相应的功率系数，并输出每个点的最大功率值。

输入数据：
1. 网格点能耗数据：
   - 目录：Z:\local_environment_creation\energy_consumption_gird\result\result_half\
   - 年份文件夹：2016, 2017, 2018, 2019, 2020
   - 文件格式：point_lat{lat}_lon{lon}_cooling.csv, point_lat{lat}_lon{lon}_heating.csv

2. 点-国家映射数据：
   - 文件：Z:\local_environment_creation\energy_consumption_gird\result\point_country_mapping.csv
   - 包含每个经纬度点对应的国家信息

3. 功率系数参数：
   - 文件：Z:\local_environment_creation\energy_consumption_gird\parameters.csv
   - 包含各国的供暖和制冷功率系数

输出结果：
- 文件：Z:\local_environment_creation\cost\country_energy_power.csv
- 格式：Continent,Country_Code,Country_Name,lat,lon,Pmax
- 单位：W（从GW转换）
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
import pycountry
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
    file_handler = logging.FileHandler('country_cost.log', encoding='utf-8')
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
GRID_RESULT_BASE_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\result_half"
POINT_COUNTRY_MAPPING_FILE = r"Z:\local_environment_creation\energy_consumption_gird\result\point_country_mapping.csv"
PARAMETERS_FILE = r"Z:\local_environment_creation\energy_consumption_gird\parameters.csv"
OUTPUT_DIR = r"Z:\local_environment_creation\cost"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "country_energy_power.csv")
YEARS = [2016, 2017, 2018, 2019, 2020]

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_country_name_from_iso(iso_code):
    """将ISO二字母代码转换为国家全称"""
    # 特殊映射 - 处理一些特殊情况
    special_mappings = {
        'XK': 'Kosovo',  # 科索沃
        'TW': 'Taiwan',  # 台湾
        'HK': 'Hong Kong',  # 香港
        'MO': 'Macau',  # 澳门
        'GU': 'Guam',  # 关岛
        'AS': 'American Samoa',  # 美属萨摩亚
        'MP': 'Northern Mariana Islands',  # 北马里亚纳群岛
        'VA': 'Vatican City',  # 梵蒂冈
        'FR': 'France',  # 法国
        'GL': 'Greenland'  # 格陵兰
    }

    # 首先检查特殊映射
    if iso_code in special_mappings:
        return special_mappings[iso_code]

    try:
        # 使用pycountry库进行转换
        country = pycountry.countries.get(alpha_2=iso_code)
        if country:
            return country.name
        else:
            logger.warning(f"未找到ISO代码 {iso_code} 对应的国家")
            return iso_code
    except Exception as e:
        logger.warning(f"转换ISO代码 {iso_code} 时出错: {e}")
        return iso_code


def load_parameters():
    """加载功率系数参数"""
    try:
        params_df = pd.read_csv(PARAMETERS_FILE)
        logger.info(f"加载参数文件，包含 {len(params_df)} 个国家/地区")

        # 将ISO代码转换为国家全称
        params_df['country_name'] = params_df['region'].apply(get_country_name_from_iso)

        params_dict = {}
        for _, row in params_df.iterrows():
            params_dict[row['country_name']] = {
                'heating_power': row['heating power'],
                'cooling_power': row['Cooling power']
            }

        logger.info(f"成功加载 {len(params_dict)} 个国家的功率系数参数")
        return params_dict
    except Exception as e:
        logger.error(f"加载参数文件出错: {str(e)}")
        return {}


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


def load_point_energy_data(lat, lon, year, grid_result_dir):
    """加载单个点在指定年份的能耗数据"""
    try:
        base_filename = f"point_lat{lat:.3f}_lon{lon:.3f}"

        # 加载制冷能耗数据
        cooling_path = os.path.join(grid_result_dir, str(year), f"{base_filename}_cooling.csv")
        heating_path = os.path.join(grid_result_dir, str(year), f"{base_filename}_heating.csv")

        cooling_data = None
        heating_data = None

        # 尝试读取cooling文件
        if os.path.exists(cooling_path):
            try:
                cooling_data = pd.read_csv(cooling_path, engine='python')
            except Exception:
                try:
                    cooling_data = pd.read_csv(cooling_path, engine='c')
                except Exception:
                    pass

        # 尝试读取heating文件
        if os.path.exists(heating_path):
            try:
                heating_data = pd.read_csv(heating_path, engine='python')
            except Exception:
                try:
                    heating_data = pd.read_csv(heating_path, engine='c')
                except Exception:
                    pass

        return cooling_data, heating_data

    except Exception as e:
        logger.error(f"加载点能耗数据失败 (lat={lat:.3f}, lon={lon:.3f}, year={year}): {e}")
        return None, None


def find_max_power_for_point(lat, lon, params_dict):
    """找到指定点在5年内的最大功率"""
    max_cooling_power = 0.0
    max_heating_power = 0.0

    # 遍历5年数据
    for year in YEARS:
        cooling_data, heating_data = load_point_energy_data(lat, lon, year, GRID_RESULT_BASE_DIR)

        if cooling_data is not None and 'ref' in cooling_data.columns:
            cooling_max = cooling_data['ref'].max()
            if cooling_max > max_cooling_power:
                max_cooling_power = cooling_max

        if heating_data is not None and 'ref' in heating_data.columns:
            heating_max = heating_data['ref'].max()
            if heating_max > max_heating_power:
                max_heating_power = heating_max

    return max_cooling_power, max_heating_power


def calculate_max_power_with_coefficients(cooling_power, heating_power, country_code, params_dict):
    """应用功率系数计算最大功率"""
    # 获取国家全称
    country_name = get_country_name_from_iso(country_code)

    # 默认功率系数
    default_heating_power = 27.9
    default_cooling_power = 48.5

    # 获取功率系数
    if country_name in params_dict:
        heating_coefficient = params_dict[country_name]['heating_power']
        cooling_coefficient = params_dict[country_name]['cooling_power']
    else:
        heating_coefficient = default_heating_power
        cooling_coefficient = default_cooling_power

    # 应用功率系数（从GW转换为W）
    actual_cooling_power = cooling_power * cooling_coefficient * 1e9  # GW * coefficient * 1e9 = W
    actual_heating_power = heating_power * heating_coefficient * 1e9  # GW * coefficient * 1e9 = W

    # 返回最大值
    return max(actual_cooling_power, actual_heating_power)


def debug_single_point(lat, lon, params_dict):
    """测试单个点的功率计算"""
    logger.info(f"=== 测试单个点: lat={lat}, lon={lon} ===")

    # 找到该点在5年内的最大功率
    max_cooling_power, max_heating_power = find_max_power_for_point(lat, lon, params_dict)
    logger.info(f"原始最大功率 - Cooling: {max_cooling_power:.6f} GW, Heating: {max_heating_power:.6f} GW")

    # 从点-国家映射中找到对应的国家信息
    mapping_df = load_point_country_mapping()
    point_info = mapping_df[(mapping_df['lat'] == lat) & (mapping_df['lon'] == lon)]

    if len(point_info) == 0:
        logger.error(f"未找到点 ({lat}, {lon}) 的国家映射信息")
        return None

    country_info = point_info.iloc[0]
    country_code = country_info['Country_Code']
    continent = country_info['Continent']
    country_name = country_info['Country_Name']

    logger.info(f"国家信息 - 代码: {country_code}, 名称: {country_name}, 大洲: {continent}")

    # 处理中国特殊情况
    if country_code == 'CN-TW':
        country_code = 'CN'

    # 应用功率系数计算实际最大功率
    max_power = calculate_max_power_with_coefficients(
        max_cooling_power, max_heating_power, country_code, params_dict
    )

    logger.info(f"应用功率系数后的最大功率: {max_power:.2f} W")

    # 显示详细的功率系数信息
    country_name_for_coeff = get_country_name_from_iso(country_code)
    if country_name_for_coeff in params_dict:
        heating_coeff = params_dict[country_name_for_coeff]['heating_power']
        cooling_coeff = params_dict[country_name_for_coeff]['cooling_power']
        logger.info(f"使用的功率系数 - 制热: {heating_coeff}, 制冷: {cooling_coeff}")
    else:
        logger.info("使用默认功率系数 - 制热: 27.9, 制冷: 48.5")

    return {
        'Continent': continent,
        'Country_Code': country_code,
        'Country_Name': country_name,
        'lat': lat,
        'lon': lon,
        'Pmax': max_power
    }


def main():
    """主函数"""
    logger.info("开始计算国家能耗功率...")

    try:
        # 1. 加载基础数据
        logger.info("=== 第一步：加载基础数据 ===")

        logger.info("加载功率系数参数...")
        params_dict = load_parameters()

        logger.info("加载点-国家映射数据...")
        mapping_df = load_point_country_mapping()

        # 2. 处理每个点
        logger.info("=== 第二步：计算每个点的最大功率 ===")

        results = []

        # 使用进度条显示处理进度
        with tqdm(total=len(mapping_df), desc="处理网格点") as pbar:
            for _, row in mapping_df.iterrows():
                lat = row['lat']
                lon = row['lon']
                country_code = row['Country_Code']
                continent = row['Continent']
                country_name = row['Country_Name']

                # 处理中国特殊情况
                if country_code == 'CN-TW':
                    country_code = 'CN'

                # 找到该点在5年内的最大功率
                max_cooling_power, max_heating_power = find_max_power_for_point(lat, lon, params_dict)

                # 应用功率系数计算实际最大功率
                max_power = calculate_max_power_with_coefficients(
                    max_cooling_power, max_heating_power, country_code, params_dict
                )

                # 添加到结果列表
                results.append({
                    'Continent': continent,
                    'Country_Code': country_code,
                    'Country_Name': country_name,
                    'lat': lat,
                    'lon': lon,
                    'Pmax': max_power
                })

                pbar.update(1)

        logger.info(f"处理完成，共处理 {len(results)} 个网格点")

        # 3. 保存结果
        logger.info("=== 第三步：保存结果 ===")

        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

        logger.info(f"结果已保存至: {OUTPUT_FILE}")
        logger.info(f"共处理了 {len(results)} 个网格点")

        # 统计信息
        logger.info("=== 统计信息 ===")
        logger.info(f"按大洲统计:")
        continent_stats = results_df.groupby('Continent').size()
        for continent, count in continent_stats.items():
            logger.info(f"  {continent}: {count} 个点")

        logger.info(f"功率范围: {results_df['Pmax'].min():.2f} W - {results_df['Pmax'].max():.2f} W")
        logger.info(f"平均功率: {results_df['Pmax'].mean():.2f} W")

        logger.info("国家能耗功率计算完成！")

    except Exception as e:
        error_msg = f"主程序执行出错: {str(e)}"
        logger.error(error_msg)
        raise


if __name__ == "__main__":
    main()
