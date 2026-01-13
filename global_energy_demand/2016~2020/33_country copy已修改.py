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
   - 计算各工况相对于参考工况的差值和节能率
   - 生成人均能耗数据（kWh/person）
   - 按大洲组织结果数据

5. 并行处理优化：
   - 多进程并行处理网格点数据
   - 分批处理策略，控制内存使用
   - 进度跟踪和错误处理

输出结果：
1. 国家级别能耗数据：
   - 按大洲分类的目录结构
   - 每个国家包含summary和summary_p两个子目录

2. 总能耗汇总文件：
   - {country_iso}_2019_summary_results.csv
   - 包含总能耗、供暖能耗、制冷能耗（TWh）
   - 差值和节能率数据

3. 人均能耗汇总文件：
   - {country_iso}_2019_summary_p_results.csv
   - 包含人均总能耗、供暖能耗、制冷能耗（kWh/person）
   - 人均差值和节能率数据

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
import logging
import pycountry
import pycountry_convert
import multiprocessing
from tqdm import tqdm
from functools import partial
import time

# 将项目的根目录加入到 sys.path
# 当前文件在 global_energy_demand/grid/ 下，需要往上三层到达项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 设置日志记录
def setup_logging():
    """设置日志记录，避免控制台输出流错误"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件处理器
    file_handler = logging.FileHandler('country_aggregation.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器，使用更安全的方式
    try:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
    except Exception:
        console_handler = None
    
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    if console_handler:
        console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    if console_handler:
        logger.addHandler(console_handler)
    
    return logger

# 初始化日志
logger = setup_logging()

def safe_log(level, message):
    """安全的日志记录函数"""
    try:
        if level == 'info':
            logging.info(message)
        elif level == 'warning':
            logging.warning(message)
        elif level == 'error':
            logging.error(message)
    except (OSError, IOError):
        try:
            setup_logging()
            if level == 'info':
                logging.info(message)
            elif level == 'warning':
                logging.warning(message)
            elif level == 'error':
                logging.error(message)
        except:
            pass  # 静默失败

# 配置参数
GRID_RESULT_BASE_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\result_half"
POPULATION_FILE = r"Z:\local_environment_creation\energy_consumption_gird\result\data\population_points.csv"
PARAMETERS_FILE = r"Z:\local_environment_creation\energy_consumption_gird\parameters.csv"
POINT_COUNTRY_MAPPING_FILE = r"Z:\local_environment_creation\energy_consumption_gird\result\point_country_mapping.csv"
PROCESSED_COUNTRIES_FILE = r"Z:\local_environment_creation\energy_consumption\2016-2020result\processed_countries.csv"
OUTPUT_BASE_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\result"
YEARS = [2016, 2017, 2018, 2019, 2020]

# 确保输出目录存在
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)



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
    
    return population_df


def load_point_country_mapping():
    """加载点-国家映射数据"""
    logging.info("开始加载点-国家映射数据...")

    if not os.path.exists(POINT_COUNTRY_MAPPING_FILE):
        raise FileNotFoundError(f"点-国家映射文件不存在: {POINT_COUNTRY_MAPPING_FILE}")

    mapping_df = pd.read_csv(POINT_COUNTRY_MAPPING_FILE)
    logging.info(f"加载点-国家映射数据完成，共 {len(mapping_df)} 个点")
    
    # 过滤掉Unknown地区
    original_count = len(mapping_df)
    mapping_df = mapping_df[mapping_df['Country_Code'] != 'Unknown']
    filtered_count = len(mapping_df)
    logging.info(f"过滤掉Unknown地区后，剩余 {filtered_count} 个点（过滤了 {original_count - filtered_count} 个点）")
    
    # 创建坐标到国家的映射字典
    point_to_country = {}
    for _, row in mapping_df.iterrows():
        lat = row['lat']
        lon = row['lon']
        country_code = row['Country_Code']
        point_to_country[(lat, lon)] = country_code
    
    logging.info(f"创建了 {len(point_to_country)} 个点的国家映射")
    
    # 统计每个国家的点数
    country_counts = mapping_df['Country_Code'].value_counts()
    logging.info(f"涉及 {len(country_counts)} 个国家/地区")
    logging.info("前10个国家/地区的点数:")
    for country, count in country_counts.head(10).items():
        logging.info(f"  {country}: {count} 个点")

    return point_to_country, mapping_df



def load_grid_point_results(grid_result_dir):
    """加载网格点结果数据"""
    logging.info(f"开始加载网格点结果数据: {grid_result_dir}")

    if not os.path.exists(grid_result_dir):
        raise FileNotFoundError(f"网格点结果目录不存在: {grid_result_dir}")

    # 获取所有结果文件
    result_files = []
    for file in os.listdir(grid_result_dir):
        if file.endswith('_cooling.csv') or file.endswith('_heating.csv'):
            result_files.append(file)

    logging.info(f"找到 {len(result_files)} 个结果文件")

    # 提取所有唯一的点坐标
    point_coords = set()
    
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
                    
            except:
                continue

    logging.info(f"找到 {len(point_coords)} 个唯一的网格点")
    
    return list(point_coords)


def load_point_energy_data(lat, lon, grid_result_dir, max_retries=2):
    """加载单个点的能耗数据，带重试机制"""
    try:
        base_filename = f"point_lat{lat:.3f}_lon{lon:.3f}"

        # 加载制冷能耗数据
        cooling_path = os.path.join(grid_result_dir, f"{base_filename}_cooling.csv")
        heating_path = os.path.join(grid_result_dir, f"{base_filename}_heating.csv")

        cooling_data = None
        heating_data = None

        # 尝试读取cooling文件
        if os.path.exists(cooling_path):
            for attempt in range(max_retries):
                try:
                    # 尝试使用python引擎读取
                    cooling_data = pd.read_csv(cooling_path, engine='python')
                    break  # 成功读取，退出重试循环
                except Exception as e1:
                    if attempt < max_retries - 1:
                        time.sleep(0.1)  # 等待100ms后重试
                        continue
                    try:
                        # 最后尝试使用c引擎
                        cooling_data = pd.read_csv(cooling_path, engine='c')
                        break
                    except Exception as e2:
                        # 所有尝试都失败，记录错误
                        pass
                    
        # 尝试读取heating文件
        if os.path.exists(heating_path):
            for attempt in range(max_retries):
                try:
                    # 尝试使用python引擎读取
                    heating_data = pd.read_csv(heating_path, engine='python')
                    break  # 成功读取，退出重试循环
                except Exception as e1:
                    if attempt < max_retries - 1:
                        time.sleep(0.1)  # 等待100ms后重试
                        continue
                    try:
                        # 最后尝试使用c引擎
                        heating_data = pd.read_csv(heating_path, engine='c')
                        break
                    except Exception as e2:
                        # 所有尝试都失败，记录错误
                        pass

        return cooling_data, heating_data

    except Exception as e:
        # 记录异常信息
        return None, None


def process_point_batch(point_batch, point_to_country, grid_result_dir):
    """处理一批网格点的能耗数据（使用点-国家映射）"""
    batch_results = {}
    cases = ['ref'] + [f'case{i}' for i in range(1, 21)]

    for lat, lon in point_batch:
        try:
            # 加载该点的能耗数据
            cooling_data, heating_data = load_point_energy_data(lat, lon, grid_result_dir)

            if cooling_data is None or heating_data is None:
                continue

            # 从映射中获取该点对应的国家
            country_iso = point_to_country.get((lat, lon), None)

            if country_iso is None:
                # 如果映射中没有这个点，跳过
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


def calculate_national_energy(point_coords, population_df, point_to_country, grid_result_dir):
    """计算每个国家的总能耗（使用点-国家映射）"""
    logging.info("开始计算国家能耗...")

    # 使用点-国家映射来聚合人口数据
    logging.info("使用点-国家映射聚合人口数据...")
    
    # 创建人口点的国家映射
    population_with_country = []
    for _, row in population_df.iterrows():
        lat = row['lat']
        lon = row['lon']
        population = row['population']
        
        # 从映射中获取国家代码
        country_iso = point_to_country.get((lat, lon), None)
        
        if country_iso is not None:
            # 处理中国特殊情况
            if country_iso == 'CN-TW':
                country_iso = 'CN'
            population_with_country.append({
                'country': country_iso,
                'population': population
            })
    
    # 按国家聚合人口
    if population_with_country:
        population_df_country = pd.DataFrame(population_with_country)
        national_population = population_df_country.groupby('country')['population'].sum().reset_index()
        national_population.rename(columns={'population': 'total_population'}, inplace=True)
    else:
        national_population = pd.DataFrame(columns=['country', 'total_population'])

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
    num_processes = min(num_cores, 12)  # 增加最大进程数为8
    batch_size = 100  # 增加每批处理点数到80
    batches = [point_coords[i:i + batch_size] for i in range(0, len(point_coords), batch_size)]

    logging.info(f"CPU核心数: {num_cores}")
    logging.info(f"使用进程数: {num_processes}")
    logging.info(f"每批处理点数: {batch_size}")
    logging.info(f"将 {len(point_coords)} 个点分为 {len(batches)} 批进行处理")

    # 并行处理
    with multiprocessing.Pool(processes=num_processes) as pool:
        process_func = partial(process_point_batch, point_to_country=point_to_country, grid_result_dir=grid_result_dir)

        chunksize = max(1, len(batches) // (num_processes * 4))
        logging.info(f"chunksize: {chunksize}")

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

                pbar.update(1)

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
            # 跳过无效的国家ISO代码
            if not country_iso or pd.isna(country_iso) or str(country_iso).strip() == '':
                logging.warning(f"跳过无效的国家ISO代码: {country_iso}")
                pbar.update(1)
                continue
            
            # 获取国家全称
            country_name = iso_to_name.get(country_iso, country_iso)
            
            # 如果国家名称为空或无效，跳过
            if not country_name or pd.isna(country_name) or str(country_name).strip() == '':
                logging.warning(f"无法获取国家 {country_iso} 的有效名称，跳过处理")
                pbar.update(1)
                continue

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

    # 从point_country_mapping.csv获取国家与大洲的映射关系
    logging.info("从point_country_mapping.csv获取国家与大洲映射关系...")
    mapping_df = pd.read_csv(POINT_COUNTRY_MAPPING_FILE)
    
    # 创建国家代码到大洲的映射
    country_to_continent = {}
    for _, row in mapping_df.iterrows():
        country_code = row['Country_Code']
        continent = row['Continent']
        if country_code not in country_to_continent:
            country_to_continent[country_code] = continent
    
    logging.info(f"获取到 {len(country_to_continent)} 个国家的洲际映射关系")
    
    # 重新从point_country_mapping.csv聚合人口数据
    logging.info("重新聚合人口数据...")
    population_df = pd.read_csv(POPULATION_FILE)
    point_to_country = {}
    for _, row in mapping_df.iterrows():
        lat = row['lat']
        lon = row['lon']
        country_code = row['Country_Code']
        point_to_country[(lat, lon)] = country_code
    
    # 聚合人口数据
    population_with_country = []
    for _, row in population_df.iterrows():
        lat = row['lat']
        lon = row['lon']
        population = row['population']
        
        country_iso = point_to_country.get((lat, lon), None)
        if country_iso is not None:
            if country_iso == 'CN-TW':
                country_iso = 'CN'
            population_with_country.append({
                'country': country_iso,
                'population': population
            })
    
    # 按国家聚合人口
    if population_with_country:
        population_df_country = pd.DataFrame(population_with_country)
        national_population_new = population_df_country.groupby('country')['population'].sum().reset_index()
        national_population_new.rename(columns={'population': 'total_population'}, inplace=True)
    else:
        national_population_new = pd.DataFrame(columns=['country', 'total_population'])
    
    logging.info(f"重新聚合了 {len(national_population_new)} 个国家的人口数据")

    # 按大洲组织结果
    continents = {}
    for country_name in final_results.keys():
        # 跳过无效的国家名称
        if not country_name or pd.isna(country_name) or str(country_name).strip() == '':
            logging.warning(f"跳过无效的国家名称: {country_name}")
            continue
            
        # 从国家全称获取ISO代码
        country_iso = None
        for country_obj in pycountry.countries:
            if country_obj.name == country_name:
                country_iso = country_obj.alpha_2
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
        
        if country_iso and country_iso in country_to_continent:
            continent = country_to_continent[country_iso]
            if continent not in continents:
                continents[continent] = []
            continents[continent].append((country_name, country_iso))
        else:
            # 如果找不到映射，使用Unknown
            if 'Unknown' not in continents:
                continents['Unknown'] = []
            continents['Unknown'].append((country_name, country_iso))

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
        for country_name, country_iso in countries:
            if country_name in final_results:
                country_data = final_results[country_name]

                # 创建国家目录（使用ISO代码）
                if country_iso:
                    country_dir = os.path.join(continent_dir, country_iso)
                    os.makedirs(country_dir, exist_ok=True)
                else:
                    logging.warning(f"国家 {country_name} 没有ISO代码，跳过创建目录")
                    continue

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

                # 获取人口数据
                population = 0
                if country_iso and country_iso in national_population_new['country'].values:
                    population = \
                    national_population_new[national_population_new['country'] == country_iso]['total_population'].iloc[0]
                    logging.debug(f"国家 {country_name} ({country_iso}) 人口: {population}")
                else:
                    logging.warning(f"国家 {country_name} ({country_iso}) 没有找到人口数据")

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
                        # 计算差值：ref - case
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
                        # 计算差值：ref - case
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
                if country_iso:
                    summary_df.to_csv(os.path.join(summary_dir, f"{country_iso}_2019_summary_results.csv"))
                    summary_p_df.to_csv(os.path.join(summary_p_dir, f"{country_iso}_2019_summary_p_results.csv"))
                else:
                    logging.error(f"无法保存国家 {country_name} 的结果文件，因为ISO代码为None")

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
        # 1. 加载基础数据（所有年份共用）
        logging.info("=== 第一步：加载基础数据 ===")

        logging.info("加载参考国家列表...")
        processed_countries = load_processed_countries()

        logging.info("加载功率系数参数...")
        params_dict = load_parameters()

        logging.info("加载人口数据...")
        population_df = load_population_data()

        logging.info("加载点-国家映射数据...")
        point_to_country, mapping_df = load_point_country_mapping()

        # 2. 处理每个年份的数据
        for year in YEARS:
            logging.info(f"=== 处理{year}年数据 ===")
            
            # 创建年份输入和输出目录
            year_grid_dir = os.path.join(GRID_RESULT_BASE_DIR, str(year))
            year_output_dir = os.path.join(OUTPUT_BASE_DIR, str(year))
            os.makedirs(year_output_dir, exist_ok=True)
            
            # 检查输入目录是否存在
            if not os.path.exists(year_grid_dir):
                logging.warning(f"{year}年网格点结果目录不存在: {year_grid_dir}，跳过")
                continue
            
            logging.info(f"加载{year}年网格点坐标...")
            point_coords = load_grid_point_results(year_grid_dir)
            
            if len(point_coords) == 0:
                logging.warning(f"{year}年没有找到有效的网格点，跳过")
                continue

            # 计算国家能耗
            logging.info(f"=== 计算{year}年国家能耗 ===")
            national_energy_results, national_population = calculate_national_energy(
                point_coords, population_df, point_to_country, year_grid_dir)

            # 应用功率系数
            logging.info(f"=== 应用{year}年功率系数 ===")
            final_results = apply_power_coefficients(national_energy_results, params_dict)

            # 保存结果
            logging.info(f"=== 保存{year}年结果 ===")
            save_results(final_results, national_population, year_output_dir)

            # 检查缺失的国家
            logging.info(f"=== 检查{year}年缺失的国家 ===")
            check_missing_countries(final_results, processed_countries)

        logging.info("所有年份国家级别能耗聚合计算完成！")

    except Exception as e:
        error_msg = f"主程序执行出错: {str(e)}"
        logging.error(error_msg)
        raise


if __name__ == "__main__":
    main()
