"""
全球国家级别能耗聚合计算工具

功能概述：
本工具用于将网格点的能耗数据聚合到国家级别，计算每个国家的总能耗和人均能耗。通过空间分析和人口权重，将高分辨率的网格点数据转换为国家尺度的能耗统计，为全球建筑能耗分析提供国家级别的数据支持。

输入数据：
1. 网格点能耗数据：
   - 目录：energy_consumption_gird/result/result_half/
   - 文件格式：point_lat{lat}_lon{lon}_cooling.csv, point_lat{lat}_lon{lon}_heating.csv
   - 包含21种工况的逐时能耗数据（ref + case1-case20）

2. 人口数据：
   - 文件：energy_consumption_gird/result/data/population_points.csv
   - 包含所有有效人口点的经纬度和人口数

3. 国家边界数据：
   - 文件：ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp
   - 包含全球各国的地理边界和属性信息
   - 特殊国家通过NAME字段匹配：FR, NO, US, AU, GL

4. 功率系数参数：
   - 文件：parameters.csv
   - 包含各国的供暖和制冷功率系数

主要功能：
1. 数据加载和预处理：
   - 加载网格点能耗数据、人口数据、国家边界数据
   - 处理国家代码转换和特殊地区映射
   - 验证数据完整性和格式一致性

2. 空间聚合分析：
   - 使用空间连接将网格点匹配到对应国家
   - 按国家聚合人口数据和能耗数据
   - 处理跨边界和特殊地区的空间关系

3. 能耗计算和转换：
   - 汇总各网格点的能耗数据到国家级别
   - 应用功率系数进行单位转换（GW→TWh）
   - 计算总能耗、供暖能耗、制冷能耗

4. 统计分析和汇总：
   - 计算各工况相对于参考工况的差值和节能率（差值 = ref - case，正值表示节能）
   - 生成人均能耗数据（kWh/person）
   - 按大洲组织结果数据

5. 并行处理优化：
   - 多进程并行处理网格点数据
   - 分批处理策略，控制内存使用
   - 进度跟踪和错误处理

输出结果：
1. 国家级别能耗数据：
   - 按大洲分类的目录结构
   - 每个国家使用ISO二字母代码作为文件夹名（如：AL, FR, US）
   - 每个国家包含summary和summary_p两个子目录

2. 总能耗汇总文件：
   - {country_iso}_2019_summary_results.csv
   - 包含总能耗、供暖能耗、制冷能耗（TWh）
   - 差值和节能率数据（差值 = ref - case，正值表示节能）

3. 人均能耗汇总文件：
   - {country_iso}_2019_summary_p_results.csv
   - 包含人均总能耗、供暖能耗、制冷能耗（kWh/person）
   - 人均差值和节能率数据（差值 = ref - case，正值表示节能）

4. 日志文件：
   - country_aggregation.log：详细的计算日志

数据流程：
1. 数据加载阶段：
   - 加载功率系数参数
   - 加载人口数据和网格点坐标
   - 加载国家边界数据

2. 空间分析阶段：
   - 创建人口点的GeoDataFrame
   - 与国家边界进行空间连接
   - 按国家聚合人口数据

3. 能耗聚合阶段：
   - 并行处理网格点能耗数据
   - 将网格点数据匹配到对应国家
   - 汇总各国家的能耗数据

4. 功率系数应用：
   - 应用各国的功率系数
   - 进行单位转换（GW→TWh）
   - 处理缺失参数的国家

5. 结果保存阶段：
   - 计算差值和节能率
   - 生成人均能耗数据
   - 按大洲保存结果文件

计算特点：
- 空间精度：基于高分辨率网格点数据
- 国家覆盖：包含全球所有主要国家
- 多工况分析：支持21种不同的节能案例
- 并行处理：多进程并行计算，提高效率
- 数据完整性：完善的错误处理和日志记录

技术参数：
- 默认供暖功率：27.9 W/°C
- 默认制冷功率：48.5 W/°C
- 空间参考系统：EPSG:4326（WGS84）
- 并行进程数：最多8个进程
- 批处理大小：每批80个网格点

特殊处理：
- 中国台湾地区：CN-TW合并到CN
- 特殊国家代码：XK（科索沃）、TW（台湾）等
- 特殊大洲映射：Western Sahara（EH）→ Africa，Timor-Leste（TL）→ Asia
- 跨边界处理：使用空间包含关系
- 缺失数据处理：使用默认功率系数

性能优化：
- 空间索引优化：使用GeoDataFrame的空间索引
- 并行处理：多进程并行处理网格点
- 内存管理：分批处理，控制内存使用
- 进度跟踪：实时显示处理进度

数据质量保证：
- 空间数据验证：确保坐标系一致性
- 数据完整性检查：验证必需字段存在
- 异常值处理：处理缺失和异常数据
- 结果验证：检查聚合结果的合理性

输出格式：
- 文件格式：CSV（UTF-8编码）
- 能耗单位：TWh（总能耗）、kWh/person（人均能耗）
- 坐标系统：WGS84（EPSG:4326）
- 时间范围：2019年全年数据
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
import time
import psutil

# 将项目的根目录加入到 sys.path
# 当前文件在 global_energy_demand/grid/ 下，需要往上三层到达项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 设置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('country_aggregation.log', encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])

# 配置参数
GRID_RESULT_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\result_half"
POPULATION_FILE = r"Z:\local_environment_creation\energy_consumption_gird\result\data\population_points.csv"
SHAPEFILE_PATH = r"Z:\local_environment_creation\shapefiles\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp"
PARAMETERS_FILE = r"Z:\local_environment_creation\energy_consumption_gird\parameters.csv"
PROCESSED_COUNTRIES_FILE = r"Z:\local_environment_creation\energy_consumption\2016-2020result\processed_countries.csv"
OUTPUT_BASE_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\result"

# 性能优化配置
USE_OPTIMIZED_SPATIAL_JOIN = False  # 设置为True使用优化的批量空间连接（参考country_energy_cooling.py）

# 确保输出目录存在
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# 特殊国家映射：shapefile中的NAME -> 标准ISO代码
# 这些国家在shapefile中的ISO_A2字段可能缺失或不正确，需要通过NAME匹配
SPECIAL_COUNTRY_NAME_TO_ISO = {
    'France': 'FR',
    'Norway': 'NO',
    'United States of America': 'US',
    'Australia': 'AU',
    'Greenland': 'GL'
}


def get_country_iso_from_shapefile_row(shapefile_row):
    """从shapefile行获取国家ISO代码
    
    优先使用ISO_A2字段，如果为空或为-99，则尝试通过NAME字段映射
    """
    iso_a2 = shapefile_row.get('ISO_A2', None)
    name = shapefile_row.get('NAME', None)
    
    # 如果ISO_A2有效，直接使用
    if iso_a2 and iso_a2 != '-99' and pd.notna(iso_a2):
        return iso_a2
    
    # 否则，尝试通过NAME映射
    if name and name in SPECIAL_COUNTRY_NAME_TO_ISO:
        mapped_iso = SPECIAL_COUNTRY_NAME_TO_ISO[name]
        logging.debug(f"通过NAME映射: {name} -> {mapped_iso}")
        return mapped_iso
    
    # 如果都失败，返回原始ISO_A2（可能是None或-99）
    logging.warning(f"无法为国家获取有效的ISO代码: NAME={name}, ISO_A2={iso_a2}")
    return iso_a2


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
            logging.warning(f"未找到ISO代码 {iso_code} 对应的国家")
            return iso_code
    except Exception as e:
        logging.warning(f"转换ISO代码 {iso_code} 时出错: {e}")
        return iso_code


def get_iso_from_country_name(country_name):
    """将国家全称转换为ISO二字母代码"""
    # 特殊映射 - 处理一些特殊情况
    special_mappings = {
        'Taiwan': 'TW',
        'Hong Kong': 'HK',
        'Macau': 'MO',
        'Kosovo': 'XK',
        'Western Sahara': 'EH',
        'Timor-Leste': 'TL'
    }

    # 首先检查特殊映射
    if country_name in special_mappings:
        return special_mappings[country_name]

    try:
        # 使用pycountry库进行转换
        for country_obj in pycountry.countries:
            if country_obj.name == country_name:
                return country_obj.alpha_2
    except Exception as e:
        logging.warning(f"转换国家名称 {country_name} 时出错: {e}")
    
    # 如果都失败，返回None
    return None


def get_country_continent_mapping():
    """获取国家与大洲的映射关系"""
    mapping = {}
    for country in pycountry.countries:
        try:
            # 获取ISO 3166-1 alpha-2代码
            alpha2 = country.alpha_2
            if alpha2 in ['XK']:  # 特殊国家代码可忽略或单独处理
                continue
            continent_code = pycountry_convert.country_alpha2_to_continent_code(alpha2)
            continent_name = pycountry_convert.convert_continent_code_to_continent_name(continent_code)
            mapping[country.name] = continent_name
        except Exception as e:
            logging.warning(f"跳过国家: {country.name}（{e}）")

    # 添加特殊情况的处理
    special_cases = {
        'Taiwan': 'Asia',  # 台湾
        'Hong Kong': 'Asia',  # 香港
        'Macau': 'Asia',  # 澳门
        'Kosovo': 'Europe',  # 科索沃
        'Western Sahara': 'Africa',  # 西撒哈拉
        'Timor-Leste': 'Asia',  # 东帝汶
    }
    mapping.update(special_cases)
    return mapping


def load_processed_countries():
    """加载参考国家列表"""
    try:
        countries_df = pd.read_csv(PROCESSED_COUNTRIES_FILE)
        logging.info(f"加载参考国家列表，包含 {len(countries_df)} 个条目")
        
        # 去重，只保留唯一的国家代码
        unique_countries = countries_df.drop_duplicates(subset=['Country_Code'])
        logging.info(f"去重后包含 {len(unique_countries)} 个唯一国家")
        
        return unique_countries
    except Exception as e:
        logging.warning(f"加载参考国家列表失败: {str(e)}")
        return None


def load_parameters():
    """加载功率系数参数"""
    try:
        params_df = pd.read_csv(PARAMETERS_FILE)
        logging.info(f"加载参数文件，包含 {len(params_df)} 个国家/地区")

        # 将ISO代码转换为国家全称
        params_df['country_name'] = params_df['region'].apply(get_country_name_from_iso)

        # 显示转换结果
        # logging.info("ISO代码转换结果:")
        # for _, row in params_df.iterrows():
        #     logging.info(f"  {row['region']} -> {row['country_name']}")

        params_dict = {}
        for _, row in params_df.iterrows():
            params_dict[row['country_name']] = {
                'heating_power': row['heating power'],
                'cooling_power': row['Cooling power']
            }

        logging.info(f"成功加载 {len(params_dict)} 个国家的功率系数参数")
        return params_dict
    except Exception as e:
        logging.error(f"加载参数文件出错: {str(e)}")
        return {}


def load_population_data():
    """加载人口数据"""
    logging.info("开始加载人口数据...")

    if not os.path.exists(POPULATION_FILE):
        raise FileNotFoundError(f"人口数据文件不存在: {POPULATION_FILE}")

    population_df = pd.read_csv(POPULATION_FILE)
    logging.info(f"加载人口数据完成，共 {len(population_df)} 个点")
    
    # 检查格陵兰岛附近的人口点
    greenland_population = population_df[
        (population_df['lat'] >= 59.8) & (population_df['lat'] <= 83.6) &
        (population_df['lon'] >= -73.0) & (population_df['lon'] <= -11.3)
    ]
    
    logging.info(f"格陵兰岛附近找到 {len(greenland_population)} 个人口点")
    if len(greenland_population) > 0:
        logging.info(f"格陵兰岛附近人口点示例:")
        for _, row in greenland_population.head(3).iterrows():
            logging.info(f"  坐标: ({row['lat']:.3f}, {row['lon']:.3f}), 人口: {row['population']}")

    return population_df


def load_country_shapefile():
    """加载国家边界数据（参考country_energy_cooling.py的处理方法）"""
    logging.info("开始加载国家边界数据...")

    gdf = gpd.read_file(SHAPEFILE_PATH)
    logging.info(f"国家数量: {len(gdf)}")

    # 移除南极洲
    gdf = gdf[gdf['CONTINENT'] != 'Antarctica']
    logging.info(f"移除南极洲后国家数量: {len(gdf)}")

    if gdf.empty:
        raise ValueError("没有找到国家数据，请检查shapefile是否为空或路径是否正确。")

    # 确保坐标系一致
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    
    # 检查并记录特殊国家的存在性
    logging.info("检查特殊国家映射...")
    for name, iso in SPECIAL_COUNTRY_NAME_TO_ISO.items():
        if name in gdf['NAME'].values:
            row = gdf[gdf['NAME'] == name].iloc[0]
            original_iso = row.get('ISO_A2', 'N/A')
            logging.info(f"  找到特殊国家: {name} (shapefile中ISO_A2={original_iso}) -> 将使用: {iso}")
        else:
            logging.warning(f"  警告：特殊国家 {name} 在shapefile中未找到")

    logging.info(f"国家边界数据加载完成，包含 {len(gdf)} 个国家（已移除南极洲）")
    return gdf


def load_grid_point_results():
    """加载网格点结果数据"""
    logging.info("开始加载网格点结果数据...")

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
    logging.info(f"格陵兰岛附近找到 {len(greenland_points)} 个点")
    if greenland_points:
        logging.info(f"格陵兰岛附近点的坐标示例: {greenland_points[:5]}")
    
    return list(point_coords)


def load_point_energy_data(lat, lon):
    """加载单个点的能耗数据"""
    try:
        base_filename = f"point_lat{lat:.3f}_lon{lon:.3f}"

        # 加载制冷能耗数据
        cooling_path = os.path.join(GRID_RESULT_DIR, f"{base_filename}_cooling.csv")
        heating_path = os.path.join(GRID_RESULT_DIR, f"{base_filename}_heating.csv")

        cooling_data = None
        heating_data = None

        if os.path.exists(cooling_path):
            cooling_data = pd.read_csv(cooling_path)
        if os.path.exists(heating_path):
            heating_data = pd.read_csv(heating_path)

        return cooling_data, heating_data

    except Exception as e:
        logging.error(f"加载点数据失败 (lat={lat:.3f}, lon={lon:.3f}): {e}")
        return None, None


def process_point_batch_optimized(point_batch, country_gdf):
    """参考country_energy_cooling.py的批量空间聚合方法"""
    batch_results = {}
    cases = ['ref'] + [f'case{i}' for i in range(1, 21)]
    
    logging.debug(f"process_point_batch_optimized: 处理 {len(point_batch)} 个点")
    
    try:
        # 1. 批量创建所有点的几何对象
        points_data = []
        for lat, lon in point_batch:
            points_data.append({
                'lat': lat,
                'lon': lon,
                'geometry': Point(lon, lat)
            })
        
        # 2. 创建点的GeoDataFrame
        points_gdf = gpd.GeoDataFrame(points_data, crs="EPSG:4326")
        
        # 3. 确保几何图形有效（参考country_energy_cooling.py）
        points_gdf['geometry'] = points_gdf.geometry.buffer(0)
        points_gdf['geometry'] = points_gdf.geometry.apply(lambda geom: geom.make_valid() if not geom.is_valid else geom)
        
        # 4. 确保坐标系匹配
        if points_gdf.crs != country_gdf.crs:
            points_gdf = points_gdf.to_crs(country_gdf.crs)
            logging.debug(f"点数据CRS不匹配，已转换为 {country_gdf.crs}")
        
        # 5. 批量空间连接：使用inner连接，只保留有匹配的点
        joined_points = gpd.sjoin(points_gdf, country_gdf[['NAME', 'ISO_A2', 'geometry']], 
                                 how="inner", predicate="within")
        
        # 6. 为每个点添加正确的国家ISO代码
        def get_correct_iso_for_point(row):
            iso_a2 = row.get('ISO_A2', None)
            name = row.get('NAME', None)
            
            if iso_a2 and iso_a2 != '-99' and pd.notna(iso_a2):
                return iso_a2
            elif name and name in SPECIAL_COUNTRY_NAME_TO_ISO:
                return SPECIAL_COUNTRY_NAME_TO_ISO[name]
            return iso_a2
        
        joined_points['country_iso'] = joined_points.apply(get_correct_iso_for_point, axis=1)
        
        # 7. 处理每个点的能耗数据
        for _, point_row in joined_points.iterrows():
            lat, lon = point_row['lat'], point_row['lon']
            country_iso = point_row['country_iso']
            
            if pd.isna(country_iso):
                continue
                
            # 处理中国特殊情况
            if country_iso == 'CN-TW':
                country_iso = 'CN'
            
            # 加载该点的能耗数据
            cooling_data, heating_data = load_point_energy_data(lat, lon)
            if cooling_data is None or heating_data is None:
                continue
            
            # 初始化该国家的结果
            if country_iso not in batch_results:
                batch_results[country_iso] = {}
                for case in cases:
                    batch_results[country_iso][case] = {
                        'cooling_demand': 0.0,
                        'heating_demand': 0.0,
                        'total_demand': 0.0
                    }
            
            # 计算该点的总能耗
            for case in cases:
                if case in cooling_data.columns and case in heating_data.columns:
                    cooling_demand = cooling_data[case].sum()
                    heating_demand = heating_data[case].sum()
                    total_demand = cooling_demand + heating_demand
                    
                    batch_results[country_iso][case]['cooling_demand'] += cooling_demand
                    batch_results[country_iso][case]['heating_demand'] += heating_demand
                    batch_results[country_iso][case]['total_demand'] += total_demand
                    
    except Exception as e:
        logging.error(f"批量空间连接失败，回退到逐个处理: {e}")
        return process_point_batch_original(point_batch, country_gdf)
    
    return batch_results


def process_point_batch_original(point_batch, country_gdf):
    """原始的逐个遍历方法（作为备用）"""
    batch_results = {}
    cases = ['ref'] + [f'case{i}' for i in range(1, 21)]
    greenland_debug_count = 0  # 格陵兰岛调试计数器
    
    logging.debug(f"process_point_batch_original: 处理 {len(point_batch)} 个点")

    for lat, lon in point_batch:
        try:
            # 加载该点的能耗数据
            cooling_data, heating_data = load_point_energy_data(lat, lon)

            if cooling_data is None or heating_data is None:
                continue

            # 找到该点对应的国家
            point = gpd.GeoDataFrame([{'geometry': Point(lon, lat)}], crs="EPSG:4326")

            # 空间查询找到包含该点的国家
            country_iso = None
            country_name = None
            
            # 检查是否是格陵兰岛附近的点
            is_greenland_area = (59.8 <= lat <= 83.6) and (-73.0 <= lon <= -11.3)
            
            for _, country_row in country_gdf.iterrows():
                try:
                    if country_row.geometry.contains(point.geometry.iloc[0]):
                        # 使用辅助函数获取国家ISO代码
                        country_iso = get_country_iso_from_shapefile_row(country_row)
                        country_name = country_row.get('NAME', 'Unknown')
                        
                        # 格陵兰岛调试信息
                        if is_greenland_area and greenland_debug_count < 5:
                            logging.info(f"格陵兰岛区域点 ({lat:.3f}, {lon:.3f}) 匹配到国家: {country_name} (ISO: {country_iso})")
                            greenland_debug_count += 1
                        
                        break
                except:
                    continue

            if country_iso is None:
                # 格陵兰岛区域点没有匹配到国家的情况
                if is_greenland_area and greenland_debug_count < 5:
                    logging.warning(f"格陵兰岛区域点 ({lat:.3f}, {lon:.3f}) 没有匹配到任何国家")
                    greenland_debug_count += 1
                continue

            # 处理中国特殊情况
            if country_iso == 'CN-TW':
                country_iso = 'CN'

            # 初始化该国家的结果（如果还没有）
            if country_iso not in batch_results:
                batch_results[country_iso] = {}
                for case in cases:
                    batch_results[country_iso][case] = {
                        'cooling_demand': 0.0,
                        'heating_demand': 0.0,
                        'total_demand': 0.0
                    }

            # 计算该点的总能耗（所有工况）
            for case in cases:
                if case in cooling_data.columns and case in heating_data.columns:
                    cooling_demand = cooling_data[case].sum()
                    heating_demand = heating_data[case].sum()
                    total_demand = cooling_demand + heating_demand

                    batch_results[country_iso][case]['cooling_demand'] += cooling_demand
                    batch_results[country_iso][case]['heating_demand'] += heating_demand
                    batch_results[country_iso][case]['total_demand'] += total_demand

        except Exception as e:
            logging.error(f"处理网格点失败 (lat={lat:.3f}, lon={lon:.3f}): {e}")
            continue

    return batch_results


def calculate_national_energy(point_coords, population_df, country_gdf):
    """计算每个国家的总能耗"""
    logging.info("开始计算国家能耗...")

    # 创建人口数据的GeoDataFrame
    geometry = [Point(xy) for xy in zip(population_df['lon'], population_df['lat'])]
    population_gdf = gpd.GeoDataFrame(population_df, geometry=geometry, crs="EPSG:4326")

    # 空间连接，将人口点匹配到国家
    joined_population = gpd.sjoin(population_gdf, country_gdf[['NAME', 'ISO_A2', 'geometry']], how="inner",
                                  predicate="within")

    # 为每个点添加正确的国家ISO代码（处理特殊国家）
    def get_correct_iso(row):
        """根据NAME和ISO_A2获取正确的ISO代码"""
        iso_a2 = row.get('ISO_A2', None)
        name = row.get('NAME', None)
        
        # 如果ISO_A2有效，直接使用
        if iso_a2 and iso_a2 != '-99' and pd.notna(iso_a2):
            return iso_a2
        
        # 否则，尝试通过NAME映射
        if name and name in SPECIAL_COUNTRY_NAME_TO_ISO:
            return SPECIAL_COUNTRY_NAME_TO_ISO[name]
        
        return iso_a2
    
    joined_population['country_iso'] = joined_population.apply(get_correct_iso, axis=1)
    
    # 按国家聚合人口
    national_population = joined_population.groupby('country_iso')['population'].sum().reset_index()
    national_population.rename(columns={'country_iso': 'country', 'population': 'total_population'}, inplace=True)

    # 处理中国特殊情况
    cn_tw_population = 0
    if 'CN-TW' in national_population['country'].values:
        cn_tw_population = national_population[national_population['country'] == 'CN-TW']['total_population'].iloc[0]
        national_population = national_population[national_population['country'] != 'CN-TW']

    if 'CN' in national_population['country'].values:
        cn_idx = national_population[national_population['country'] == 'CN'].index[0]
        national_population.loc[cn_idx, 'total_population'] += cn_tw_population
    else:
        # 如果只有CN-TW没有CN，则创建CN记录
        national_population = pd.concat(
            [national_population, pd.DataFrame({'country': ['CN'], 'total_population': [cn_tw_population]})],
            ignore_index=True)

    logging.info(f"成功聚合 {len(national_population)} 个国家的人口数据")

    # 初始化国家能耗结果
    national_energy_results = {}
    cases = ['ref'] + [f'case{i}' for i in range(1, 21)]

    for country in national_population['country']:
        national_energy_results[country] = {}
        for case in cases:
            national_energy_results[country][case] = {
                'cooling_demand': 0.0,
                'heating_demand': 0.0,
                'total_demand': 0.0
            }

    # 并行处理网格点
    logging.info("开始并行处理网格点能耗数据...")

    # 配置并行处理参数
    num_cores = multiprocessing.cpu_count()
    num_processes = min(num_cores, 8)  # 增加最大进程数为8
    batch_size = 80  # 增加每批处理点数到80
    
    batches = [point_coords[i:i + batch_size] for i in range(0, len(point_coords), batch_size)]

    logging.info(f"CPU核心数: {num_cores}")
    logging.info(f"使用进程数: {num_processes}")
    logging.info(f"每批处理点数: {batch_size}")
    logging.info(f"将 {len(point_coords)} 个点分为 {len(batches)} 批进行处理")

    # 选择空间聚合方法
    if USE_OPTIMIZED_SPATIAL_JOIN:
        process_func = partial(process_point_batch_optimized, country_gdf=country_gdf)
        logging.info("🚀 使用优化的批量空间连接方法（参考country_energy_cooling.py）")
    else:
        process_func = partial(process_point_batch_original, country_gdf=country_gdf)
        logging.info("⚖️ 使用原始的逐个遍历方法")
    
    # 并行处理 - 添加内存优化
    with multiprocessing.Pool(processes=num_processes, maxtasksperchild=50) as pool:

        chunksize = max(1, len(batches) // (num_processes * 4))
        logging.info(f"chunksize: {chunksize}")

        # 性能监控
        start_time = time.time()
        processed_batches = 0
        
        with tqdm(total=len(batches), desc="处理网格点批次") as pbar:
            for batch_results in pool.imap_unordered(process_func, batches, chunksize=chunksize):
                # 合并批次结果到总结果中
                for country, cases_data in batch_results.items():
                    if country not in national_energy_results:
                        national_energy_results[country] = {}
                        for case in cases:
                            national_energy_results[country][case] = {
                                'cooling_demand': 0.0,
                                'heating_demand': 0.0,
                                'total_demand': 0.0
                            }

                    for case, data in cases_data.items():
                        national_energy_results[country][case]['cooling_demand'] += data['cooling_demand']
                        national_energy_results[country][case]['heating_demand'] += data['heating_demand']
                        national_energy_results[country][case]['total_demand'] += data['total_demand']

                processed_batches += 1
                pbar.update(1)
                
                # 每处理100个批次显示性能信息
                if processed_batches % 100 == 0:
                    elapsed_time = time.time() - start_time
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_percent = psutil.virtual_memory().percent
                    batches_per_sec = processed_batches / elapsed_time
                    
                    logging.info(f"性能监控 - 已处理: {processed_batches}/{len(batches)} 批次, "
                               f"CPU使用率: {cpu_percent:.1f}%, "
                               f"内存使用率: {memory_percent:.1f}%, "
                               f"处理速度: {batches_per_sec:.2f} 批次/秒")

    logging.info(f"网格点处理完成，共处理 {len(point_coords)} 个点")

    return national_energy_results, national_population


def apply_power_coefficients(national_energy_results, params_dict):
    """应用功率系数"""
    logging.info("开始应用功率系数...")

    default_heating_power = 27.9
    default_cooling_power = 48.5

    # 将ISO代码转换为国家全称
    iso_to_name = {}
    for country in pycountry.countries:
        iso_to_name[country.alpha_2] = country.name

    # 添加特殊映射
    special_mappings = {
        'XK': 'Kosovo',
        'TW': 'Taiwan',
        'HK': 'Hong Kong',
        'MO': 'Macau',
        'GU': 'Guam',
        'AS': 'American Samoa',
        'MP': 'Northern Mariana Islands',
        'VA': 'Vatican City'
    }
    iso_to_name.update(special_mappings)

    final_results = {}

    # 使用进度条显示功率系数应用进度
    with tqdm(total=len(national_energy_results), desc="应用功率系数") as pbar:
        for country_iso, cases in national_energy_results.items():
            # 获取国家全称
            country_name = iso_to_name.get(country_iso, country_iso)

            # 获取功率系数
            if country_name in params_dict:
                heating_power = params_dict[country_name]['heating_power']
                cooling_power = params_dict[country_name]['cooling_power']
                # logging.info(f"使用自定义功率系数: {country_name} - 制热: {heating_power}, 制冷: {cooling_power}")
            else:
                heating_power = default_heating_power
                cooling_power = default_cooling_power
                # logging.info(f"使用默认功率系数: {country_name} - 制热: {heating_power}, 制冷: {cooling_power}")

            final_results[country_name] = {}

            for case, data in cases.items():
                # 应用功率系数并转换单位（从GW到TWh）
                final_results[country_name][case] = {
                    'total_demand': (data['heating_demand'] * heating_power + data[
                        'cooling_demand'] * cooling_power) / 1e3,
                    'heating_demand': data['heating_demand'] * heating_power / 1e3,
                    'cooling_demand': data['cooling_demand'] * cooling_power / 1e3
                }

            pbar.update(1)

    logging.info(f"功率系数应用完成，处理了 {len(final_results)} 个国家")
    return final_results


def save_results(final_results, national_population, output_dir):
    """保存结果到文件"""
    logging.info("开始保存结果...")

    # 获取国家与大洲的映射关系
    continent_mapping = get_country_continent_mapping()
    logging.info(f"获取到 {len(continent_mapping)} 个国家的洲际映射关系")
    
    # 检查特殊国家的映射
    special_countries = ['Western Sahara', 'Timor-Leste']
    for country in special_countries:
        if country in continent_mapping:
            logging.info(f"特殊国家映射: {country} -> {continent_mapping[country]}")
        else:
            logging.warning(f"特殊国家 {country} 未找到映射")

    # 按大洲组织结果
    continents = {}
    for country in final_results.keys():
        continent = continent_mapping.get(country, 'Unknown')
        if continent not in continents:
            continents[continent] = []
        continents[continent].append(country)

    logging.info("按大洲分组结果:")
    for continent, countries in continents.items():
        logging.info(f"  {continent}: {len(countries)} 个国家")

    for continent, countries in continents.items():
        continent_dir = os.path.join(output_dir, continent)
        os.makedirs(continent_dir, exist_ok=True)

        # 创建summary目录
        summary_dir = os.path.join(continent_dir, 'summary')
        summary_p_dir = os.path.join(continent_dir, 'summary_p')
        os.makedirs(summary_dir, exist_ok=True)
        os.makedirs(summary_p_dir, exist_ok=True)

        # 处理该大洲的国家
        for country in countries:
            if country in final_results:
                country_data = final_results[country]

                # 获取国家ISO代码
                country_iso = get_iso_from_country_name(country)
                
                # 如果仍然没有找到ISO代码，使用国家名称作为备用
                if country_iso is None:
                    logging.warning(f"无法找到国家 {country} 的ISO代码，使用国家名称作为目录名")
                    country_iso = country.replace(' ', '_')  # 替换空格为下划线
                
                # 创建国家目录（使用ISO代码而不是国家全名）
                country_dir = os.path.join(continent_dir, country_iso)
                os.makedirs(country_dir, exist_ok=True)

                # 准备数据
                cases = ['ref'] + [f'case{i}' for i in range(1, 21)]
                total_demand = []
                heating_demand = []
                cooling_demand = []

                for case in cases:
                    if case in country_data:
                        data = country_data[case]
                        total_demand.append(data['total_demand'])
                        heating_demand.append(data['heating_demand'])
                        cooling_demand.append(data['cooling_demand'])
                    else:
                        total_demand.append(0)
                        heating_demand.append(0)
                        cooling_demand.append(0)

                population = 0
                if country_iso and country_iso in national_population['country'].values:
                    population = \
                    national_population[national_population['country'] == country_iso]['total_population'].iloc[0]

                # 计算差值和节能率
                ref_total = total_demand[0]
                ref_heating = heating_demand[0]
                ref_cooling = cooling_demand[0]

                total_demand_diff = []
                total_demand_reduction = []
                heating_demand_diff = []
                heating_demand_reduction = []
                cooling_demand_diff = []
                cooling_demand_reduction = []

                for i, case in enumerate(cases):
                    if i == 0:  # ref case
                        total_demand_diff.append(0)
                        total_demand_reduction.append(0)
                        heating_demand_diff.append(0)
                        heating_demand_reduction.append(0)
                        cooling_demand_diff.append(0)
                        cooling_demand_reduction.append(0)
                    else:  # case1-20
                        # 计算差值：ref - case（修正计算顺序）
                        total_diff = ref_total - total_demand[i]
                        heating_diff = ref_heating - heating_demand[i]
                        cooling_diff = ref_cooling - cooling_demand[i]

                        total_demand_diff.append(total_diff)
                        heating_demand_diff.append(heating_diff)
                        cooling_demand_diff.append(cooling_diff)

                        # 计算节能率
                        total_reduction = (ref_total - total_demand[i]) / ref_total * 100 if ref_total > 0 else 0
                        heating_reduction = (ref_heating - heating_demand[
                            i]) / ref_heating * 100 if ref_heating > 0 else 0
                        cooling_reduction = (ref_cooling - cooling_demand[
                            i]) / ref_cooling * 100 if ref_cooling > 0 else 0

                        total_demand_reduction.append(total_reduction)
                        heating_demand_reduction.append(heating_reduction)
                        cooling_demand_reduction.append(cooling_reduction)

                # 总能耗汇总
                summary_df = pd.DataFrame({
                    'total_demand_sum(TWh)': total_demand,
                    'total_demand_diff(TWh)': total_demand_diff,
                    'total_demand_reduction(%)': total_demand_reduction,
                    'heating_demand_sum(TWh)': heating_demand,
                    'heating_demand_diff(TWh)': heating_demand_diff,
                    'heating_demand_reduction(%)': heating_demand_reduction,
                    'cooling_demand_sum(TWh)': cooling_demand,
                    'cooling_demand_diff(TWh)': cooling_demand_diff,
                    'cooling_demand_reduction(%)': cooling_demand_reduction
                }, index=cases)

                # 人均能耗汇总
                if population > 0:
                    total_demand_p = [d * 1e9 / population for d in total_demand]  # TWh to kWh/person
                    heating_demand_p = [d * 1e9 / population for d in heating_demand]
                    cooling_demand_p = [d * 1e9 / population for d in cooling_demand]
                else:
                    total_demand_p = [0] * len(cases)
                    heating_demand_p = [0] * len(cases)
                    cooling_demand_p = [0] * len(cases)

                # 计算人均差值和节能率
                ref_total_p = total_demand_p[0]
                ref_heating_p = heating_demand_p[0]
                ref_cooling_p = cooling_demand_p[0]

                total_demand_diff_p = []
                total_demand_p_reduction = []
                heating_demand_diff_p = []
                heating_demand_p_reduction = []
                cooling_demand_diff_p = []
                cooling_demand_p_reduction = []

                for i, case in enumerate(cases):
                    if i == 0:  # ref case
                        total_demand_diff_p.append(0)
                        total_demand_p_reduction.append(0)
                        heating_demand_diff_p.append(0)
                        heating_demand_p_reduction.append(0)
                        cooling_demand_diff_p.append(0)
                        cooling_demand_p_reduction.append(0)
                    else:  # case1-20
                        # 计算差值：ref - case（修正计算顺序）
                        total_diff_p = ref_total_p - total_demand_p[i]
                        heating_diff_p = ref_heating_p - heating_demand_p[i]
                        cooling_diff_p = ref_cooling_p - cooling_demand_p[i]

                        total_demand_diff_p.append(total_diff_p)
                        heating_demand_diff_p.append(heating_diff_p)
                        cooling_demand_diff_p.append(cooling_diff_p)

                        # 计算节能率
                        total_reduction_p = (ref_total_p - total_demand_p[
                            i]) / ref_total_p * 100 if ref_total_p > 0 else 0
                        heating_reduction_p = (ref_heating_p - heating_demand_p[
                            i]) / ref_heating_p * 100 if ref_heating_p > 0 else 0
                        cooling_reduction_p = (ref_cooling_p - cooling_demand_p[
                            i]) / ref_cooling_p * 100 if ref_cooling_p > 0 else 0

                        total_demand_p_reduction.append(total_reduction_p)
                        heating_demand_p_reduction.append(heating_reduction_p)
                        cooling_demand_p_reduction.append(cooling_reduction_p)

                summary_p_df = pd.DataFrame({
                    'total_demand_sum_p(kWh/person)': total_demand_p,
                    'total_demand_diff_p(kWh/person)': total_demand_diff_p,
                    'total_demand_p_reduction(%)': total_demand_p_reduction,
                    'heating_demand_sum_p(kWh/person)': heating_demand_p,
                    'heating_demand_diff_p(kWh/person)': heating_demand_diff_p,
                    'heating_demand_p_reduction(%)': heating_demand_p_reduction,
                    'cooling_demand_sum_p(kWh/person)': cooling_demand_p,
                    'cooling_demand_diff_p(kWh/person)': cooling_demand_diff_p,
                    'cooling_demand_p_reduction(%)': cooling_demand_p_reduction
                }, index=cases)

                # 保存文件 - 使用ISO代码作为文件名
                if country_iso is not None:
                    summary_df.to_csv(os.path.join(summary_dir, f"{country_iso}_2019_summary_results.csv"))
                    summary_p_df.to_csv(os.path.join(summary_p_dir, f"{country_iso}_2019_summary_p_results.csv"))
                else:
                    logging.error(f"无法保存国家 {country} 的结果文件，因为ISO代码为None")

    logging.info("结果保存完成")


def check_missing_countries(final_results, processed_countries):
    """检查并记录缺失的国家"""
    logging.info("=== 检查缺失的国家 ===")
    
    if processed_countries is None:
        logging.warning("未加载参考国家列表，跳过检查")
        return
    
    # 获取参考国家代码列表，确保都是字符串类型
    reference_country_codes = set()
    for code in processed_countries['Country_Code'].unique():
        if pd.notna(code) and str(code).strip():
            reference_country_codes.add(str(code).strip())
    logging.info(f"参考列表包含 {len(reference_country_codes)} 个唯一国家代码")
    
    # 将国家全称转换为ISO代码
    processed_country_codes = set()
    for country_name in final_results.keys():
        # 尝试从国家名称获取ISO代码
        country_iso = None
        for country in pycountry.countries:
            if country.name == country_name:
                country_iso = country.alpha_2
                break
        
        # 特殊处理
        if country_name == 'Taiwan':
            country_iso = 'TW'
        elif country_name == 'Hong Kong':
            country_iso = 'HK'
        elif country_name == 'Macau':
            country_iso = 'MO'
        elif country_name == 'Kosovo':
            country_iso = 'XK'
        
        if country_iso:
            processed_country_codes.add(str(country_iso))  # 确保是字符串类型
    
    logging.info(f"实际处理了 {len(processed_country_codes)} 个国家")
    
    # 找出缺失的国家
    missing_countries = reference_country_codes - processed_country_codes
    
    if missing_countries:
        logging.warning(f"发现 {len(missing_countries)} 个缺失的国家:")
        missing_info = []
        # 过滤掉非字符串类型的代码，并转换为字符串进行排序
        valid_missing_codes = [str(code) for code in missing_countries if pd.notna(code) and str(code).strip()]
        for code in sorted(valid_missing_codes):
            # 从参考列表中获取国家名称
            country_info = processed_countries[processed_countries['Country_Code'] == code]
            if not country_info.empty:
                name = country_info.iloc[0]['Country_Name']
                continent = country_info.iloc[0]['Continent']
                logging.warning(f"  - {code}: {name} ({continent})")
                missing_info.append({'Code': code, 'Name': name, 'Continent': continent})
        
        # 保存缺失国家列表
        if missing_info:
            missing_df = pd.DataFrame(missing_info)
            missing_file = os.path.join(OUTPUT_BASE_DIR, 'missing_countries.csv')
            missing_df.to_csv(missing_file, index=False, encoding='utf-8-sig')
            logging.info(f"缺失国家列表已保存至: {missing_file}")
    else:
        logging.info("没有缺失的国家，所有参考国家都已处理")
    
    # 找出额外处理的国家（在结果中但不在参考列表中）
    extra_countries = processed_country_codes - reference_country_codes
    if extra_countries:
        logging.info(f"发现 {len(extra_countries)} 个额外处理的国家（不在参考列表中）:")
        # 过滤掉非字符串类型的代码，并转换为字符串进行排序
        valid_extra_codes = [str(code) for code in extra_countries if pd.notna(code) and str(code).strip()]
        for code in sorted(valid_extra_codes):
            logging.info(f"  - {code}")


def main():
    """主函数"""
    logging.info("开始国家级别能耗聚合计算...")

    try:
        # 1. 加载数据
        logging.info("=== 第一步：加载数据 ===")

        logging.info("加载参考国家列表...")
        processed_countries = load_processed_countries()

        logging.info("加载功率系数参数...")
        params_dict = load_parameters()

        logging.info("加载人口数据...")
        population_df = load_population_data()

        logging.info("加载国家边界数据...")
        country_gdf = load_country_shapefile()

        logging.info("加载网格点坐标...")
        point_coords = load_grid_point_results()

        # 2. 计算国家能耗
        logging.info("=== 第二步：计算国家能耗 ===")
        national_energy_results, national_population = calculate_national_energy(
            point_coords, population_df, country_gdf)

        # 3. 应用功率系数
        logging.info("=== 第三步：应用功率系数 ===")
        final_results = apply_power_coefficients(national_energy_results, params_dict)

        # 4. 保存结果
        logging.info("=== 第四步：保存结果 ===")
        save_results(final_results, national_population, OUTPUT_BASE_DIR)

        # 5. 检查缺失的国家
        logging.info("=== 第五步：检查缺失的国家 ===")
        check_missing_countries(final_results, processed_countries)

        logging.info("国家级别能耗聚合计算完成！")

    except Exception as e:
        error_msg = f"主程序执行出错: {str(e)}"
        logging.error(error_msg)
        raise


if __name__ == "__main__":
    main()
