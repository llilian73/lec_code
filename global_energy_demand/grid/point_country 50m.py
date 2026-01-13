"""
网格点国家归属确定工具

功能概述：
本工具用于确定每个经纬度点属于哪个国家，通过空间分析将网格点与国家边界进行匹配，
输出包含每个点的国家归属信息的CSV文件。

输入数据：
1. 网格点坐标数据：
   - 从网格点结果文件中提取坐标信息
   - 目录：energy_consumption_gird/result/result_half/

2. 国家边界数据：
   - 文件：ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp
   - 包含全球各国的地理边界和属性信息

输出结果：
- CSV文件：point_country_mapping.csv
- 包含列：Continent, Country_Code, Country_Name, lat, lon
- 其中Country_Code为二字母ISO代码

特殊处理：
- 保留3_country copy.py中对国家的特殊处理逻辑
- 处理特殊国家映射：FR, NO, US, AU, GL, XK, SO等
- 特殊国家代码映射：
  * France → FR
  * Norway → NO
  * United States of America → US
  * Australia → AU
  * Greenland → GL
  * Kosovo → XK
  * Somaliland → SO (归入索马里)
- 特殊大洲映射：
  * Taiwan (TW) → Asia
  * Hong Kong (HK) → Asia  
  * Macau (MO) → Asia
  * Kosovo (XK) → Europe
  * Western Sahara (EH) → Africa
  * Timor-Leste (TL) → Asia
  * Vatican City (VA) → Europe
  * US Minor Outlying Islands (UM) → Oceania
  * Sint Maarten (SX) → North America
  * Pitcairn (PN) → Oceania
- 中国特殊情况：CN-TW → CN
- 自动过滤南极洲及相关地区
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import geopandas as gpd
import logging
from datetime import datetime
import pycountry
import pycountry_convert
from shapely.geometry import Point
import multiprocessing
from tqdm import tqdm
from functools import partial

# 将项目的根目录加入到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('point_country_mapping.log', encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])

# 配置参数
GRID_RESULT_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\result_half_test_2019"
SHAPEFILE_PATH = r"Z:\local_environment_creation\shapefiles\ne_50m_admin_0_countries\ne_50m_admin_0_countries.shp"
OUTPUT_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 特殊国家映射：shapefile中的NAME -> 标准ISO代码
# 这些国家在shapefile中的ISO_A2字段可能缺失或不正确，需要通过NAME匹配
SPECIAL_COUNTRY_NAME_TO_ISO = {
    'France': 'FR',
    'Norway': 'NO',
    'United States of America': 'US',
    'Australia': 'AU',
    'Greenland': 'GL',
    'Kosovo': 'XK',  # 科索沃
    'Somaliland': 'SO'  # 索马里兰，归入索马里
}


def get_country_iso_from_shapefile_row(shapefile_row):
    """从shapefile行获取国家ISO代码
    
    主要基于Country_Name进行映射，确保所有中国地区都使用CN代码
    
    特殊处理：
    - 所有中国地区（包括台湾、香港、澳门）-> CN
    - Somaliland -> SO (Somalia)
    - 其他特殊国家映射
    """
    name = shapefile_row.get('NAME', None)
    iso_a2 = shapefile_row.get('ISO_A2', None)
    
    # 中国特殊情况：所有中国地区都使用CN
    china_names = ['China', 'Taiwan', 'Hong Kong', 'Macau', 'Macao']
    if name in china_names:
        logging.debug(f"中国地区 {name} -> CN")
        return 'CN'
    
    # 特殊处理：Somaliland 归为 Somalia
    if name == 'Somaliland':
        logging.debug(f"Somaliland -> SO (Somalia)")
        return 'SO'
    
    # 其他特殊国家映射
    if name and name in SPECIAL_COUNTRY_NAME_TO_ISO:
        mapped_iso = SPECIAL_COUNTRY_NAME_TO_ISO[name]
        logging.debug(f"通过NAME映射: {name} -> {mapped_iso}")
        return mapped_iso
    
    # 如果ISO_A2有效且不是-99，使用ISO_A2
    if iso_a2 and iso_a2 != '-99' and pd.notna(iso_a2):
        return iso_a2
    
    # 如果都失败，记录警告
    logging.warning(f"无法为国家获取有效的ISO代码: NAME={name}, ISO_A2={iso_a2}")
    return iso_a2


def get_country_name_from_iso(iso_code):
    """将ISO二字母代码转换为国家全称"""
    # 特殊映射 - 处理一些特殊情况
    special_mappings = {
        'XK': 'Kosovo',  # 科索沃
        'CN': 'China',  # 中国（包括台湾、香港、澳门）
        'GU': 'Guam',  # 关岛
        'AS': 'American Samoa',  # 美属萨摩亚
        'MP': 'Northern Mariana Islands',  # 北马里亚纳群岛
        'VA': 'Vatican City',  # 梵蒂冈
        'FR': 'France',  # 法国
        'GL': 'Greenland',  # 格陵兰
        'SO': 'Somalia'  # 索马里（包括索马里兰地区）
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
            logging.warning(f"未找到ISO代码 {iso_code} 对应的国家")
            return iso_code
    except Exception as e:
        logging.warning(f"转换ISO代码 {iso_code} 时出错: {e}")
        return iso_code


def get_country_continent_mapping():
    """获取国家与大洲的映射关系"""
    mapping = {}
    
    # 首先定义所有特殊情况的映射（包括pycountry_convert不支持的）
    special_cases = {
        'China': 'Asia',  # 中国（包括台湾、香港、澳门）
        'Kosovo': 'Europe',  # 科索沃
        'Western Sahara': 'Africa',  # 西撒哈拉
        'Timor-Leste': 'Asia',  # 东帝汶
        'Holy See (Vatican City State)': 'Europe',  # 梵蒂冈
        'United States Minor Outlying Islands': 'Oceania',  # 美属小岛屿
        'Sint Maarten (Dutch part)': 'North America',  # 荷属圣马丁
        'Pitcairn': 'Oceania',  # 皮特凯恩群岛
        'Antarctica': 'Antarctica',  # 南极洲（虽然会被过滤掉）
        'French Southern Territories': 'Antarctica',  # 法属南部领地
    }
    
    # 添加特殊映射
    mapping.update(special_cases)
    
    # 然后处理pycountry中的标准国家
    for country in pycountry.countries:
        try:
            # 获取ISO 3166-1 alpha-2代码
            alpha2 = country.alpha_2
            
            # 跳过已经在特殊映射中处理的国家
            if country.name in special_cases:
                continue
                
            # 跳过一些特殊代码
            if alpha2 in ['XK', 'TL', 'VA', 'UM', 'SX', 'PN', 'AQ', 'TF']:
                continue
                
            continent_code = pycountry_convert.country_alpha2_to_continent_code(alpha2)
            continent_name = pycountry_convert.convert_continent_code_to_continent_name(continent_code)
            mapping[country.name] = continent_name
        except Exception as e:
            logging.warning(f"跳过国家: {country.name}（{e}）")
            # 对于无法处理的国家，尝试手动映射
            if country.name not in mapping:
                mapping[country.name] = 'Unknown'

    return mapping


def load_country_shapefile():
    """加载国家边界数据"""
    logging.info("开始加载国家边界数据...")

    gdf = gpd.read_file(SHAPEFILE_PATH)

    # 确保坐标系一致
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')

    # 移除南极洲和相关的南极地区
    gdf = gdf[~gdf['CONTINENT'].isin(['Antarctica'])]
    # 也移除名称中包含Antarctica的国家
    gdf = gdf[~gdf['NAME'].str.contains('Antarctica', case=False, na=False)]
    logging.info(f"国家边界数据加载完成，包含 {len(gdf)} 个国家（已移除南极洲）")
    
    # 检查 Somaliland 是否存在
    somaliland_exists = 'Somaliland' in gdf['NAME'].values
    if somaliland_exists:
        somaliland_row = gdf[gdf['NAME'] == 'Somaliland'].iloc[0]
        logging.info(f"✓ 找到 Somaliland，ISO_A2={somaliland_row.get('ISO_A2', 'N/A')}，将归为 Somalia (SO)")
    else:
        logging.warning("⚠ Somaliland 在 shapefile 中不存在")
    
    # 检查 Somalia 是否存在
    somalia_exists = gdf['ISO_A2'].eq('SO').any() or gdf['NAME'].eq('Somalia').any()
    if somalia_exists:
        if 'SO' in gdf['ISO_A2'].values:
            logging.info(f"✓ 找到 Somalia (SO)")
        else:
            somalia_row = gdf[gdf['NAME'] == 'Somalia'].iloc[0]
            logging.info(f"✓ 找到 Somalia，ISO_A2={somalia_row.get('ISO_A2', 'N/A')}")
    else:
        logging.warning("⚠ Somalia 在 shapefile 中不存在")
    
    # 检查并记录特殊国家的存在性
    logging.info("检查特殊国家映射...")
    for name, iso in SPECIAL_COUNTRY_NAME_TO_ISO.items():
        if name in gdf['NAME'].values:
            row = gdf[gdf['NAME'] == name].iloc[0]
            original_iso = row.get('ISO_A2', 'N/A')
            logging.info(f"  找到特殊国家: {name} (shapefile中ISO_A2={original_iso}) -> 将使用: {iso}")
        else:
            logging.warning(f"  警告：特殊国家 {name} 在shapefile中未找到")

    return gdf


def load_grid_point_coords():
    """加载网格点坐标数据"""
    logging.info("开始加载网格点坐标数据...")

    if not os.path.exists(GRID_RESULT_DIR):
        raise FileNotFoundError(f"网格点结果目录不存在: {GRID_RESULT_DIR}")

    # 获取所有结果文件
    result_files = []
    for file in os.listdir(GRID_RESULT_DIR):
        if file.endswith('_cooling.csv') or file.endswith('_heating.csv'):
            result_files.append(file)

    logging.info(f"找到 {len(result_files)} 个结果文件")

    # 提取所有唯一的点坐标
    point_coords = set()
    greenland_points = []  # 专门记录格陵兰岛附近的点
    
    for file in result_files:
        # 从文件名提取坐标
        if '_cooling.csv' in file:
            coord_part = file.replace('_cooling.csv', '')
        elif '_heating.csv' in file:
            coord_part = file.replace('_heating.csv', '')
        else:
            continue

        # 解析坐标
        if 'point_lat' in coord_part and '_lon' in coord_part:
            try:
                lat_part = coord_part.split('_lat')[1].split('_lon')[0]
                lon_part = coord_part.split('_lon')[1]
                lat = float(lat_part)
                lon = float(lon_part)
                point_coords.add((lat, lon))
                
                # 检查是否是格陵兰岛附近的点
                # 格陵兰岛大致范围：59.8°N-83.6°N, 73.0°W-11.3°W
                if (59.8 <= lat <= 83.6) and (-73.0 <= lon <= -11.3):
                    greenland_points.append((lat, lon))
                    
            except:
                continue

    logging.info(f"找到 {len(point_coords)} 个唯一的网格点")
    # logging.info(f"格陵兰岛附近找到 {len(greenland_points)} 个点")
    # if greenland_points:
    #     logging.info(f"格陵兰岛附近点的坐标示例: {greenland_points[:5]}")
    
    return list(point_coords)


def process_point_batch(point_batch, country_gdf):
    """处理一批网格点的国家归属"""
    batch_results = []
    greenland_debug_count = 0  # 格陵兰岛调试计数器
    somaliland_count = 0  # Somaliland 计数器

    for lat, lon in point_batch:
        try:
            # 找到该点对应的国家
            point = gpd.GeoDataFrame([{'geometry': Point(lon, lat)}], crs="EPSG:4326")

            # 空间查询找到包含该点的国家
            country_iso = None
            country_name = None
            matched_shapefile_name = None
            
            # 检查是否是格陵兰岛附近的点
            is_greenland_area = (59.8 <= lat <= 83.6) and (-73.0 <= lon <= -11.3)
            
            for _, country_row in country_gdf.iterrows():
                try:
                    if country_row.geometry.contains(point.geometry.iloc[0]):
                        # 记录 shapefile 中的原始国家名称
                        matched_shapefile_name = country_row.get('NAME', 'Unknown')
                        
                        # 使用辅助函数获取国家ISO代码
                        country_iso = get_country_iso_from_shapefile_row(country_row)
                        country_name = country_row.get('NAME', 'Unknown')
                        
                        # Somaliland 调试信息
                        if matched_shapefile_name == 'Somaliland':
                            somaliland_count += 1
                        
                        # 格陵兰岛调试信息
                        if is_greenland_area and greenland_debug_count < 5:
                            # logging.info(f"格陵兰岛区域点 ({lat:.3f}, {lon:.3f}) 匹配到国家: {country_name} (ISO: {country_iso})")
                            greenland_debug_count += 1
                        
                        break
                except:
                    continue

            if country_iso is None:
                # 格陵兰岛区域点没有匹配到国家的情况
                if is_greenland_area and greenland_debug_count < 5:
                    # logging.warning(f"格陵兰岛区域点 ({lat:.3f}, {lon:.3f}) 没有匹配到任何国家")
                    greenland_debug_count += 1
                # 对于没有匹配到国家的点，记录为Unknown
                batch_results.append({
                    'lat': lat,
                    'lon': lon,
                    'country_iso': 'Unknown',
                    'country_name': 'Unknown',
                    'continent': 'Unknown'
                })
                continue

            # 中国特殊情况已在get_country_iso_from_shapefile_row中处理
            # 所有中国地区（包括台湾、香港、澳门）都使用CN代码

            # 获取国家全称
            country_full_name = get_country_name_from_iso(country_iso)
            
            # 获取大洲信息
            continent_mapping = get_country_continent_mapping()
            continent = continent_mapping.get(country_full_name, 'Unknown')
            
            batch_results.append({
                'lat': lat,
                'lon': lon,
                'country_iso': country_iso,
                'country_name': country_full_name,
                'continent': continent
            })

        except Exception as e:
            logging.error(f"处理网格点失败 (lat={lat:.3f}, lon={lon:.3f}): {e}")
            # 对于处理失败的点，记录为Unknown
            batch_results.append({
                'lat': lat,
                'lon': lon,
                'country_iso': 'Unknown',
                'country_name': 'Unknown',
                'continent': 'Unknown'
            })
            continue

    # 返回结果和Somaliland计数
    return {'results': batch_results, 'somaliland_count': somaliland_count}


def process_all_points(point_coords, country_gdf):
    """处理所有网格点的国家归属"""
    logging.info("开始处理所有网格点的国家归属...")

    # 配置并行处理参数
    num_cores = multiprocessing.cpu_count()
    num_processes = min(num_cores, 12)  # 最多使用8个进程
    batch_size = 100  # 每批处理100个点
    batches = [point_coords[i:i + batch_size] for i in range(0, len(point_coords), batch_size)]

    logging.info(f"CPU核心数: {num_cores}")
    logging.info(f"使用进程数: {num_processes}")
    logging.info(f"每批处理点数: {batch_size}")
    logging.info(f"将 {len(point_coords)} 个点分为 {len(batches)} 批进行处理")

    all_results = []
    total_somaliland_count = 0

    # 并行处理
    with multiprocessing.Pool(processes=num_processes) as pool:
        process_func = partial(process_point_batch, country_gdf=country_gdf)

        chunksize = max(1, len(batches) // (num_processes * 4))
        logging.info(f"chunksize: {chunksize}")

        with tqdm(total=len(batches), desc="处理网格点批次") as pbar:
            for batch_data in pool.imap_unordered(process_func, batches, chunksize=chunksize):
                all_results.extend(batch_data['results'])
                total_somaliland_count += batch_data['somaliland_count']
                pbar.update(1)

    logging.info(f"网格点处理完成，共处理 {len(all_results)} 个点")
    logging.info(f"✓ 其中有 {total_somaliland_count} 个点匹配到 Somaliland，已归为 Somalia (SO)")
    return all_results


def save_results(all_results, output_dir):
    """保存结果到CSV文件"""
    logging.info("开始保存结果...")

    # 创建DataFrame
    df = pd.DataFrame(all_results)
    
    # 重新排列列的顺序
    df = df[['continent', 'country_iso', 'country_name', 'lat', 'lon']]
    
    # 重命名列以匹配要求的格式
    df.columns = ['Continent', 'Country_Code', 'Country_Name', 'lat', 'lon']
    
    # 按照Continent、Country_Code排序
    df = df.sort_values(['Continent', 'Country_Code'], ascending=[True, True])
    logging.info("结果已按照Continent、Country_Code排序")
    
    # 格式化经纬度，保留3位小数
    df['lat'] = df['lat'].apply(lambda x: f"{x:.3f}")
    df['lon'] = df['lon'].apply(lambda x: f"{x:.3f}")
    logging.info("经纬度已格式化为3位小数")
    
    # 保存到CSV文件
    output_file = os.path.join(output_dir, 'point_country_mapping.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    logging.info(f"结果已保存到: {output_file}")
    logging.info(f"共处理 {len(df)} 个点")
    
    # 统计信息
    country_stats = df['Country_Code'].value_counts()
    logging.info(f"涉及 {len(country_stats)} 个国家/地区")
    logging.info("前10个国家/地区的点数:")
    for country, count in country_stats.head(10).items():
        logging.info(f"  {country}: {count} 个点")
    
    # 特别检查 Somalia (SO) 的点数
    so_count = len(df[df['Country_Code'] == 'SO'])
    if so_count > 0:
        logging.info(f"✓ Somalia (SO) 总共有 {so_count} 个点")
    
    # 检查Unknown点
    unknown_count = len(df[df['Country_Code'] == 'Unknown'])
    if unknown_count > 0:
        logging.warning(f"有 {unknown_count} 个点未能匹配到国家")
    
    return df


def main():
    """主函数"""
    logging.info("开始网格点国家归属确定...")

    try:
        # 1. 加载数据
        logging.info("=== 第一步：加载数据 ===")

        logging.info("加载国家边界数据...")
        country_gdf = load_country_shapefile()

        logging.info("加载网格点坐标...")
        point_coords = load_grid_point_coords()

        # 2. 处理网格点国家归属
        logging.info("=== 第二步：处理网格点国家归属 ===")
        all_results = process_all_points(point_coords, country_gdf)

        # 3. 保存结果
        logging.info("=== 第三步：保存结果 ===")
        result_df = save_results(all_results, OUTPUT_DIR)

        logging.info("网格点国家归属确定完成！")

    except Exception as e:
        error_msg = f"主程序执行出错: {str(e)}"
        logging.error(error_msg)
        raise


if __name__ == "__main__":
    main()
