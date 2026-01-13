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
   - 文件：world_border2.shp
   - 包含全球各国的地理边界和属性信息

3. 国家信息数据：
   - 文件：all_countries_info.csv
   - 包含Continent, Country_Code_2, Country_Code_3, Country_Name

输出结果：
- CSV文件：point_country_mapping.csv
- 包含列：Continent, Country_Code_2, Country_Code_3, Country_Name, lat, lon

特殊处理：
- Hong Kong和Macao的点都归为China
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import geopandas as gpd
import logging
from datetime import datetime
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
SHAPEFILE_PATH = r"Z:\local_environment_creation\shapefiles\全球国家边界\world_border2.shp"
COUNTRY_INFO_CSV = r"Z:\local_environment_creation\all_countries_info.csv"
OUTPUT_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 特殊处理：Hong Kong和Macao归为China
# 这些国家在shapefile中的名称可能不同，需要映射到China
SPECIAL_COUNTRY_TO_CHINA = {
    'Hong Kong': 'CHN',
    'Macao': 'CHN',
    'Macau': 'CHN'
}


def load_country_info_mapping():
    """从CSV文件加载国家信息映射"""
    logging.info("正在加载国家信息映射...")
    
    # 尝试不同的编码
    encodings = ['utf-8-sig', 'latin-1', 'cp1252', 'gbk']
    df = None
    
    for enc in encodings:
        try:
            # 注意：keep_default_na=False 确保"NA"字符串不被当作NaN处理（Namibia的代码）
            df = pd.read_csv(COUNTRY_INFO_CSV, encoding=enc, keep_default_na=False, na_values=[''])
            logging.info(f"成功使用编码 {enc} 读取国家信息文件")
            break
    except Exception as e:
            continue

    if df is None:
        raise FileNotFoundError(f"无法读取国家信息文件: {COUNTRY_INFO_CSV}")

    # 创建映射字典：Country_Code_3 -> (continent, Country_Code_2, Country_Name)
    mapping = {}
    for _, row in df.iterrows():
        code_3 = row['Country_Code_3']
        continent = row['continent']
        code_2 = row['Country_Code_2']
        name = row['Country_Name']
        
        # 注意：Namibia的代码是NA，不要当作无效值处理
        # 使用pd.isna()来检查，但要注意NA字符串是有效的
        # 由于使用了keep_default_na=False，NA字符串不会被当作NaN
        if code_3 and code_3 != '':
            mapping[str(code_3)] = {
                'continent': continent if continent and continent != '' else None,
                'Country_Code_2': code_2 if code_2 and code_2 != '' else None,
                'Country_Name': name if name and name != '' else None
            }
    
    logging.info(f"成功加载 {len(mapping)} 个国家信息映射")
    return mapping


def load_country_shapefile():
    """加载国家边界数据"""
    logging.info("开始加载国家边界数据...")

    gdf = gpd.read_file(SHAPEFILE_PATH)
    logging.info(f"成功读取shapefile，共{len(gdf)}个国家/地区")
    logging.info(f"可用的列名: {gdf.columns.tolist()}")

    # 确保坐标系一致
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')

    # 检查特殊国家（Hong Kong和Macao）
    logging.info("检查特殊国家映射...")
    for name in SPECIAL_COUNTRY_TO_CHINA.keys():
        if name in gdf['NAME_0'].values:
            row = gdf[gdf['NAME_0'] == name].iloc[0]
            code_3 = row.get('GID_0', 'N/A')
            logging.info(f"  找到特殊国家: {name} (GID_0={code_3}) -> 将归为 China (CHN)")
        else:
            # 尝试其他可能的名称变体
            name_variants = [name, name.replace('Macao', 'Macau'), name.replace('Macau', 'Macao')]
            found = False
            for variant in name_variants:
                if variant in gdf['NAME_0'].values:
                    row = gdf[gdf['NAME_0'] == variant].iloc[0]
                    code_3 = row.get('GID_0', 'N/A')
                    logging.info(f"  找到特殊国家: {variant} (GID_0={code_3}) -> 将归为 China (CHN)")
                    found = True
                    break
            if not found:
            logging.warning(f"  警告：特殊国家 {name} 在shapefile中未找到")

    logging.info(f"国家边界数据加载完成，包含 {len(gdf)} 个国家")
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
    lon_range = {'min': float('inf'), 'max': float('-inf')}
    lat_range = {'min': float('inf'), 'max': float('-inf')}
    
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
                # 提取纬度部分：point_lat{lat}_lon{lon} -> {lat}
                lat_part = coord_part.split('_lat')[1].split('_lon')[0]
                # 提取经度部分：point_lat{lat}_lon{lon} -> {lon}
                # 注意：经度可能是负数，所以需要处理负号
                lon_part = coord_part.split('_lon')[1]
                
                lat = float(lat_part)
                lon = float(lon_part)
                
                # 更新范围统计
                lon_range['min'] = min(lon_range['min'], lon)
                lon_range['max'] = max(lon_range['max'], lon)
                lat_range['min'] = min(lat_range['min'], lat)
                lat_range['max'] = max(lat_range['max'], lat)
                
                point_coords.add((lat, lon))
            except Exception as e:
                logging.warning(f"解析文件名失败: {file}, 错误: {e}")
                continue

    logging.info(f"找到 {len(point_coords)} 个唯一的网格点")
    if len(point_coords) > 0:
        logging.info(f"经度范围: {lon_range['min']:.6f}° ~ {lon_range['max']:.6f}°")
        logging.info(f"纬度范围: {lat_range['min']:.6f}° ~ {lat_range['max']:.6f}°")
        
        # 检查是否有负经度
        negative_lon_count = sum(1 for _, lon in point_coords if lon < 0)
        positive_lon_count = sum(1 for _, lon in point_coords if lon >= 0)
        logging.info(f"经度分布: 负经度点数={negative_lon_count}, 正经度点数={positive_lon_count}")
        
        # 显示一些示例坐标
        sample_coords = list(point_coords)[:5]
        logging.info(f"坐标示例（前5个）: {sample_coords}")
    else:
        logging.warning("未找到任何有效的网格点坐标！")
    
    return list(point_coords)


def process_point_batch(point_batch, country_gdf, country_info_mapping):
    """处理一批网格点的国家归属（使用批量空间连接）"""
    batch_results = []
    hk_macau_count = 0  # Hong Kong和Macao计数器

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
        
        # 3. 确保坐标系匹配
        if points_gdf.crs != country_gdf.crs:
            points_gdf = points_gdf.to_crs(country_gdf.crs)
        
        # 4. 批量空间连接：使用within连接，找到包含每个点的国家
        # 注意：如果多个国家包含同一个点（边界重叠），sjoin会返回多行
        # 使用how='left'保留所有点，即使没有匹配到国家
        joined_points = gpd.sjoin(
            points_gdf, 
            country_gdf[['GID_0', 'NAME_0', 'geometry']], 
            how="left", 
            predicate="within"
        )
        
        # 5. 处理每个点的结果
        # 如果sjoin返回重复的点（一个点匹配到多个国家），需要去重，每个点只保留第一个匹配
        processed_points = set()
        
        for idx, row in joined_points.iterrows():
            lat = row['lat']
            lon = row['lon']
            point_key = (lat, lon)
            
            # 如果这个点已经处理过，跳过（处理多重匹配的情况）
            if point_key in processed_points:
                    continue

            processed_points.add(point_key)
            
            # 检查是否匹配到国家
            if pd.notna(row.get('GID_0', None)):
                country_code_3 = row['GID_0']
                matched_shapefile_name = row.get('NAME_0', 'Unknown')
                country_name = matched_shapefile_name
                
                # 特殊处理：Hong Kong和Macao归为China
                if matched_shapefile_name in SPECIAL_COUNTRY_TO_CHINA:
                    country_code_3 = 'CHN'
                    country_name = 'China'
                    hk_macau_count += 1
                
                # 从CSV文件获取Continent和Country_Code_2
                country_info = country_info_mapping.get(str(country_code_3), None)
                
                if country_info:
                    continent = country_info.get('continent', None)
                    country_code_2 = country_info.get('Country_Code_2', None)
                else:
                    continent = None
                    country_code_2 = None
                    logging.warning(f"未找到Country_Code_3={country_code_3}的国家信息")
                
                batch_results.append({
                    'lat': lat,
                    'lon': lon,
                    'Continent': continent,
                    'Country_Code_2': country_code_2,
                    'Country_Code_3': country_code_3,
                    'Country_Name': country_name
                })
            else:
                # 没有匹配到国家
            batch_results.append({
                'lat': lat,
                'lon': lon,
                    'Continent': None,
                    'Country_Code_2': None,
                    'Country_Code_3': None,
                    'Country_Name': 'Unknown'
            })

        except Exception as e:
        logging.error(f"批量处理网格点失败: {e}")
        import traceback
        traceback.print_exc()
        # 如果批量处理失败，回退到逐个处理
        for lat, lon in point_batch:
            batch_results.append({
                'lat': lat,
                'lon': lon,
                'Continent': None,
                'Country_Code_2': None,
                'Country_Code_3': None,
                'Country_Name': 'Unknown'
            })

    # 返回结果和Hong Kong/Macau计数
    return {'results': batch_results, 'hk_macau_count': hk_macau_count}


def process_all_points(point_coords, country_gdf, country_info_mapping):
    """处理所有网格点的国家归属"""
    logging.info("开始处理所有网格点的国家归属...")

    # 配置并行处理参数
    num_cores = multiprocessing.cpu_count()
    num_processes = min(num_cores, 12)  # 最多使用12个进程
    batch_size = 100  # 每批处理100个点
    batches = [point_coords[i:i + batch_size] for i in range(0, len(point_coords), batch_size)]

    logging.info(f"CPU核心数: {num_cores}")
    logging.info(f"使用进程数: {num_processes}")
    logging.info(f"每批处理点数: {batch_size}")
    logging.info(f"将 {len(point_coords)} 个点分为 {len(batches)} 批进行处理")

    all_results = []
    total_hk_macau_count = 0

    # 并行处理
    with multiprocessing.Pool(processes=num_processes) as pool:
        process_func = partial(process_point_batch, country_gdf=country_gdf, country_info_mapping=country_info_mapping)

        chunksize = max(1, len(batches) // (num_processes * 4))
        logging.info(f"chunksize: {chunksize}")

        with tqdm(total=len(batches), desc="处理网格点批次") as pbar:
            for batch_data in pool.imap_unordered(process_func, batches, chunksize=chunksize):
                all_results.extend(batch_data['results'])
                total_hk_macau_count += batch_data['hk_macau_count']
                pbar.update(1)

    logging.info(f"网格点处理完成，共处理 {len(all_results)} 个点")
    logging.info(f"✓ 其中有 {total_hk_macau_count} 个点匹配到 Hong Kong/Macao，已归为 China (CHN)")
    return all_results


def save_results(all_results, output_dir):
    """保存结果到CSV文件"""
    logging.info("开始保存结果...")

    # 创建DataFrame
    df = pd.DataFrame(all_results)
    
    # 重新排列列的顺序：Continent, Country_Code_2, Country_Code_3, Country_Name, lat, lon
    df = df[['Continent', 'Country_Code_2', 'Country_Code_3', 'Country_Name', 'lat', 'lon']]
    
    # 按照Continent、Country_Code_3排序
    df = df.sort_values(['Continent', 'Country_Code_3'], ascending=[True, True], na_position='last')
    logging.info("结果已按照Continent、Country_Code_3排序")
    
    # 格式化经纬度，保留3位小数
    df['lat'] = df['lat'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else x)
    df['lon'] = df['lon'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else x)
    logging.info("经纬度已格式化为3位小数")
    
    # 保存到CSV文件
    output_file = os.path.join(output_dir, 'point_country_mapping.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    logging.info(f"结果已保存到: {output_file}")
    logging.info(f"共处理 {len(df)} 个点")
    
    # 统计信息
    country_stats = df['Country_Code_3'].value_counts()
    logging.info(f"涉及 {len(country_stats)} 个国家/地区")
    logging.info("前10个国家/地区的点数:")
    for country, count in country_stats.head(10).items():
        logging.info(f"  {country}: {count} 个点")
    
    # 特别检查 China (CHN) 的点数
    chn_count = len(df[df['Country_Code_3'] == 'CHN'])
    if chn_count > 0:
        logging.info(f"✓ China (CHN) 总共有 {chn_count} 个点")
    
    # 检查Unknown点
    unknown_count = len(df[df['Country_Name'] == 'Unknown'])
    if unknown_count > 0:
        logging.warning(f"有 {unknown_count} 个点未能匹配到国家")
    
    # 检查缺失数据
    missing_continent = df['Continent'].isna().sum()
    missing_code2 = df['Country_Code_2'].isna().sum()
    if missing_continent > 0:
        logging.warning(f"有 {missing_continent} 个点缺少Continent信息")
    if missing_code2 > 0:
        logging.warning(f"有 {missing_code2} 个点缺少Country_Code_2信息")
    
    return df


def main():
    """主函数"""
    logging.info("开始网格点国家归属确定...")

    try:
        # 1. 加载数据
        logging.info("=== 第一步：加载数据 ===")

        logging.info("加载国家边界数据...")
        country_gdf = load_country_shapefile()

        logging.info("加载国家信息映射...")
        country_info_mapping = load_country_info_mapping()

        logging.info("加载网格点坐标...")
        point_coords = load_grid_point_coords()

        # 2. 处理网格点国家归属
        logging.info("=== 第二步：处理网格点国家归属 ===")
        all_results = process_all_points(point_coords, country_gdf, country_info_mapping)

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
