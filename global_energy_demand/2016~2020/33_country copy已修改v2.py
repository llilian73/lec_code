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
import multiprocessing
from tqdm import tqdm
from functools import partial
import time
import gc

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
GRID_RESULT_BASE_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\result_half_parquet"
POPULATION_FILE = r"Z:\local_environment_creation\energy_consumption_gird\result\data\population_points.csv"
PARAMETERS_FILE = r"Z:\local_environment_creation\energy_consumption_gird\parameters.csv"
POINT_COUNTRY_MAPPING_FILE = r"Z:\local_environment_creation\energy_consumption_gird\result\point_country_mapping.csv"
PROCESSED_COUNTRIES_FILE = r"Z:\local_environment_creation\all_countries_info.csv"
OUTPUT_BASE_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\result"
YEARS = [2016]
BLOCK_SIZE = 10  # 空间块大小（度）

# 确保输出目录存在
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)



def load_country_info():
    """从all_countries_info.csv加载国家信息"""
    try:
        # 尝试不同的编码
        encodings = ['utf-8-sig', 'latin-1', 'cp1252', 'gbk']
        df = None
        
        for enc in encodings:
            try:
                df = pd.read_csv(PROCESSED_COUNTRIES_FILE, encoding=enc, keep_default_na=False, na_values=[''])
                logging.info(f"成功使用编码 {enc} 读取国家信息文件")
                break
            except Exception as e:
                continue
        
        if df is None:
            raise FileNotFoundError(f"无法读取国家信息文件: {PROCESSED_COUNTRIES_FILE}")
        
        logging.info(f"加载国家信息，包含 {len(df)} 个国家/地区")
        
        # 创建映射字典：Country_Code_2 -> (Country_Name, Country_Code_3, continent)
        country_info_map = {}
        for _, row in df.iterrows():
            code_2 = row['Country_Code_2']
            code_3 = row['Country_Code_3']
            name = row['Country_Name']
            continent = row['continent']
            
            if code_2 and code_2 != '':
                country_info_map[str(code_2)] = {
                    'Country_Name': name if name and name != '' else None,
                    'Country_Code_3': code_3 if code_3 and code_3 != '' else None,
                    'continent': continent if continent and continent != '' else None
                }
        
        logging.info(f"成功加载 {len(country_info_map)} 个国家的信息映射")
        return country_info_map, df
    except Exception as e:
        logging.error(f"加载国家信息失败: {str(e)}")
        return {}, None


def load_processed_countries():
    """加载参考国家列表（从all_countries_info.csv）"""
    try:
        _, countries_df = load_country_info()
        if countries_df is None:
            return None
        
        logging.info(f"加载参考国家列表，包含 {len(countries_df)} 个条目")
        
        # 去重，只保留唯一的Country_Code_2
        unique_countries = countries_df.drop_duplicates(subset=['Country_Code_2'])
        logging.info(f"去重后包含 {len(unique_countries)} 个唯一国家")
        
        return unique_countries
    except Exception as e:
        logging.warning(f"加载参考国家列表失败: {str(e)}")
        return None


def load_parameters(country_info_map=None):
    """加载功率系数参数"""
    try:
        params_df = pd.read_csv(PARAMETERS_FILE, keep_default_na=False)
        logging.info(f"加载参数文件，包含 {len(params_df)} 个国家/地区")

        # 如果没有提供国家信息映射，则加载它
        if country_info_map is None:
            country_info_map, _ = load_country_info()

        params_dict = {}
        for _, row in params_df.iterrows():
            iso2_code = row['region']  # region列是二字母代码
            
            # 从country_info_map获取国家名称
            country_info = country_info_map.get(iso2_code, None)
            if country_info and country_info['Country_Name']:
                country_name = country_info['Country_Name']
            else:
                # 如果找不到，使用ISO代码作为名称
                country_name = iso2_code
                logging.warning(f"未找到ISO代码 {iso2_code} 对应的国家名称，使用ISO代码作为名称")

            params_dict[country_name] = {
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

    population_df = pd.read_csv(POPULATION_FILE, keep_default_na=False)
    logging.info(f"加载人口数据完成，共 {len(population_df)} 个点")
    
    return population_df


def load_point_country_mapping():
    """加载点-国家映射数据"""
    logging.info("开始加载点-国家映射数据...")

    if not os.path.exists(POINT_COUNTRY_MAPPING_FILE):
        raise FileNotFoundError(f"点-国家映射文件不存在: {POINT_COUNTRY_MAPPING_FILE}")

    mapping_df = pd.read_csv(POINT_COUNTRY_MAPPING_FILE, keep_default_na=False)
    logging.info(f"加载点-国家映射数据完成，共 {len(mapping_df)} 个点")
    
    # 过滤掉Unknown地区（使用Country_Name字段判断）
    original_count = len(mapping_df)
    # 过滤逻辑：如果Country_Name为'Unknown'则过滤掉
    mapping_df = mapping_df[
        (mapping_df['Country_Name'].astype(str) != 'Unknown') &
        (mapping_df['Country_Name'].notna())
    ]
    filtered_count = len(mapping_df)
    logging.info(f"过滤掉Unknown地区后，剩余 {filtered_count} 个点（过滤了 {original_count - filtered_count} 个点）")
    
    # 创建坐标到国家的映射字典（使用Country_Code_2）
    point_to_country = {}
    for _, row in mapping_df.iterrows():
        lat = float(row['lat'])
        lon = float(row['lon'])
        country_code = row['Country_Code_2']
        point_to_country[(lat, lon)] = country_code
    
    logging.info(f"创建了 {len(point_to_country)} 个点的国家映射")
    
    # 统计每个国家的点数
    country_counts = mapping_df['Country_Code_2'].value_counts()
    logging.info(f"涉及 {len(country_counts)} 个国家/地区")
    logging.info("前10个国家/地区的点数:")
    for country, count in country_counts.head(10).items():
        logging.info(f"  {country}: {count} 个点")

    return point_to_country, mapping_df



def get_spatial_block(lat, lon, block_size=10):
    """根据经纬度获取空间块标识"""
    # 计算空间块的下界（向下取整到10的倍数）
    lat_min = int(np.floor(lat / block_size) * block_size)
    lat_max = lat_min + block_size
    
    # 处理经度（包括负经度）
    if lon >= 0:
        lon_min = int(np.floor(lon / block_size) * block_size)
    else:
        # 对于负经度，向上取整到10的倍数
        lon_min = int(np.ceil(lon / block_size) * block_size) - block_size
    lon_max = lon_min + block_size
    
    return f"block_lat={lat_min}-{lat_max}_block_lon={lon_min}-{lon_max}"


def load_spatial_block_mapping(grid_result_dir):
    """从文件夹中读取所有parquet文件名，提取空间块标识并创建映射"""
    logging.info(f"开始加载空间块映射: {grid_result_dir}")
    
    if not os.path.exists(grid_result_dir):
        raise FileNotFoundError(f"网格点结果目录不存在: {grid_result_dir}")
    
    # 获取所有parquet文件
    parquet_files = [f for f in os.listdir(grid_result_dir) if f.endswith('.parquet')]
    logging.info(f"找到 {len(parquet_files)} 个parquet文件")
    
    # 创建空间块标识到文件路径的映射
    block_to_path = {}
    
    for parquet_file in parquet_files:
        # 从文件名提取空间块标识（去掉.parquet后缀）
        block_name = parquet_file.replace('.parquet', '')
        parquet_path = os.path.join(grid_result_dir, parquet_file)
        block_to_path[block_name] = parquet_path
    
    logging.info(f"成功创建 {len(block_to_path)} 个空间块的映射")
    
    return block_to_path


def group_points_by_spatial_block(point_to_country, block_to_path):
    """将点按空间块分组，返回 {block_name: [(lat, lon), ...]}"""
    logging.info("按空间块分组点坐标...")
    
    # 创建空间块到点的映射
    block_to_points = {}
    
    for (lat, lon), country_code in point_to_country.items():
        # 计算该点所属的空间块
        block_name = get_spatial_block(lat, lon, BLOCK_SIZE)
        
        # 检查该空间块是否存在对应的parquet文件
        if block_name in block_to_path:
            if block_name not in block_to_points:
                block_to_points[block_name] = []
            block_to_points[block_name].append((lat, lon))
    
    logging.info(f"共 {len(block_to_points)} 个空间块包含有效点")
    logging.info(f"总点数: {sum(len(points) for points in block_to_points.values())}")
    
    # 统计每个空间块的点数
    block_counts = {block: len(points) for block, points in block_to_points.items()}
    logging.info(f"每个空间块的点数统计（前10个）:")
    for block_name, count in sorted(block_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        logging.info(f"  {block_name}: {count} 个点")
    
    return block_to_points


def extract_point_data_from_block_df(block_df, lat, lon):
    """从已读取的空间块DataFrame中提取单个点的数据"""
    try:
        lat_rounded = round(float(lat), 3)
        lon_rounded = round(float(lon), 3)
        
        # 确保lat和lon列是数值类型
        if 'lat' not in block_df.columns or 'lon' not in block_df.columns:
            return None, None
        
        # 筛选出该点的数据（使用3位小数精度匹配）
        point_df = block_df[(block_df['lat'].round(3) == lat_rounded) & 
                           (block_df['lon'].round(3) == lon_rounded)].copy()
        
        if len(point_df) == 0:
            return None, None
        
        # 构建cooling和heating数据
        cases = ['ref'] + [f'case{i}' for i in range(1, 21)]
        
        # 提取时间索引（在处理前提取，避免重复访问）
        time_index = point_df['time'].values if 'time' in point_df.columns else None
        
        # 创建cooling DataFrame（只包含工况列，不包含time列，与原来的CSV格式保持一致）
        cooling_dict = {}
        for case in cases:
            cooling_col = f'{case}_cooling'
            if cooling_col in point_df.columns:
                cooling_dict[case] = pd.to_numeric(point_df[cooling_col], errors='coerce').fillna(0).values
        
        if not cooling_dict:
            del point_df
            return None, None
        
        cooling_data = pd.DataFrame(cooling_dict)
        # 设置索引为时间（如果time列存在），以便后续可以获取时间索引
        if time_index is not None:
            cooling_data.index = time_index
        
        # 创建heating DataFrame（只包含工况列，不包含time列，与原来的CSV格式保持一致）
        heating_dict = {}
        for case in cases:
            heating_col = f'{case}_heating'
            if heating_col in point_df.columns:
                heating_dict[case] = pd.to_numeric(point_df[heating_col], errors='coerce').fillna(0).values
        
        if not heating_dict:
            del point_df, cooling_data
            return None, None
        
        heating_data = pd.DataFrame(heating_dict)
        # 设置索引为时间（如果time列存在），以便后续可以获取时间索引
        if time_index is not None:
            heating_data.index = time_index
        
        # 释放point_df内存
        del point_df
        
        return cooling_data, heating_data
        
    except Exception as e:
        return None, None


def process_spatial_block_batch(args):
    """处理一批空间块（每个空间块只读取一次parquet文件）"""
    block_batch, point_to_country, block_to_path = args
    # block_batch 是一个列表，包含多个 (block_name, points) 元组
    
    batch_results = {}
    batch_hourly_results = {}  # 用于存储逐时数据
    cases = ['ref'] + [f'case{i}' for i in range(1, 21)]
    
    # 用于存储每个国家的时间索引（从第一个点获取）
    country_time_index = {}
    
    # 检查是否有pyarrow依赖
    try:
        import pyarrow
        engine = 'pyarrow'
    except ImportError:
        try:
            import fastparquet
            engine = 'fastparquet'
        except ImportError:
            return {}, {}
    
    # 处理批次中的每个空间块
    for block_name, points in block_batch:
        try:
            # 获取parquet文件路径
            if block_name not in block_to_path:
                continue
            
            parquet_path = block_to_path[block_name]
            
            if not os.path.exists(parquet_path):
                continue
            
            # 读取整个空间块的parquet文件（只读取一次）
            block_df = pd.read_parquet(parquet_path, engine=engine)
            
            # 确保lat和lon列是数值类型
            if 'lat' not in block_df.columns or 'lon' not in block_df.columns:
                del block_df
                continue
            
            block_df['lat'] = pd.to_numeric(block_df['lat'], errors='coerce')
            block_df['lon'] = pd.to_numeric(block_df['lon'], errors='coerce')
            
            # 处理该空间块内的所有点
            for lat, lon in points:
                try:
                    # 从已读取的DataFrame中提取该点的数据
                    cooling_data, heating_data = extract_point_data_from_block_df(block_df, lat, lon)
                    
                    if cooling_data is None or heating_data is None:
                        continue
                    
                    # 从映射中获取该点对应的国家（Country_Code_2，二字母代码）
                    country_iso2_code = point_to_country.get((lat, lon), None)
                    
                    if country_iso2_code is None:
                        # 释放中间数据
                        del cooling_data, heating_data
                        continue
                    
                    # 获取时间索引
                    if 'time' in cooling_data.columns:
                        time_index = cooling_data['time'].values
                    else:
                        time_index = cooling_data.index.values
                    
                    # 初始化该国家的结果（如果还没有）
                    if country_iso2_code not in batch_results:
                        batch_results[country_iso2_code] = {}
                        batch_hourly_results[country_iso2_code] = {}
                        country_time_index[country_iso2_code] = time_index
                        for case in cases:
                            batch_results[country_iso2_code][case] = {
                                'cooling_demand': 0.0,
                                'heating_demand': 0.0,
                                'total_demand': 0.0
                            }
                            # 初始化逐时数据
                            batch_hourly_results[country_iso2_code][case] = {
                                'cooling': np.zeros(len(time_index)),
                                'heating': np.zeros(len(time_index)),
                                'total': np.zeros(len(time_index))
                            }
                    
                    # 计算该点的总能耗和逐时能耗（所有工况）
                    for case in cases:
                        if case in cooling_data.columns and case in heating_data.columns:
                            # 全年总能耗
                            cooling_demand = cooling_data[case].sum()
                            heating_demand = heating_data[case].sum()
                            total_demand = cooling_demand + heating_demand
                            
                            batch_results[country_iso2_code][case]['cooling_demand'] += cooling_demand
                            batch_results[country_iso2_code][case]['heating_demand'] += heating_demand
                            batch_results[country_iso2_code][case]['total_demand'] += total_demand
                            
                            # 逐时能耗（原始数据，后续会应用功率系数）
                            cooling_hourly = cooling_data[case].values
                            heating_hourly = heating_data[case].values
                            total_hourly = cooling_hourly + heating_hourly
                            
                            batch_hourly_results[country_iso2_code][case]['cooling'] += cooling_hourly
                            batch_hourly_results[country_iso2_code][case]['heating'] += heating_hourly
                            batch_hourly_results[country_iso2_code][case]['total'] += total_hourly
                    
                    # 处理完该点后，立即释放中间数据
                    del cooling_data, heating_data
                
                except Exception as e:
                    logging.error(f"处理网格点失败 (lat={lat:.3f}, lon={lon:.3f}): {e}")
                    continue
            
            # 释放该空间块的内存
            del block_df
            # 强制垃圾回收，立即释放内存
            gc.collect()
            
        except Exception as e:
            logging.error(f"处理空间块失败 {block_name}: {e}")
            continue
    
    # 将numpy数组转换为Series（保留时间索引）
    for country_iso2_code in batch_hourly_results:
        time_idx = country_time_index[country_iso2_code]
        for case in cases:
            if case in batch_hourly_results[country_iso2_code]:
                batch_hourly_results[country_iso2_code][case]['cooling'] = pd.Series(
                    batch_hourly_results[country_iso2_code][case]['cooling'], index=time_idx)
                batch_hourly_results[country_iso2_code][case]['heating'] = pd.Series(
                    batch_hourly_results[country_iso2_code][case]['heating'], index=time_idx)
                batch_hourly_results[country_iso2_code][case]['total'] = pd.Series(
                    batch_hourly_results[country_iso2_code][case]['total'], index=time_idx)
    
    # 处理完批次后，强制垃圾回收释放内存
    gc.collect()
    
    return batch_results, batch_hourly_results


def calculate_national_population(population_df, point_to_country):
    """计算每个国家的人口总数（只计算一次，所有年份共用）"""
    logging.info("开始聚合国家人口数据...")
    
    # 创建人口点的国家映射
    population_with_country = []
    for _, row in population_df.iterrows():
        lat = float(row['lat'])
        lon = float(row['lon'])
        population = row['population']
        
        # 从映射中获取国家代码（二字母代码）
        country_iso2_code = point_to_country.get((lat, lon), None)
        
        if country_iso2_code is not None:
            population_with_country.append({
                'country': country_iso2_code,
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
    return national_population


def combine_blocks_into_batches(block_to_points, batch_point_limit=500):
    """将空间块组合成批次，每批最多batch_point_limit个点（减少批次大小以降低内存占用）"""
    logging.info(f"组合空间块成批次（每批最多 {batch_point_limit} 个点）...")
    
    block_batches = []  # 每个元素是一个批次，包含多个(block_name, points)的列表
    
    current_batch = []
    current_batch_points = 0
    
    # 按空间块名称排序，确保处理顺序一致
    sorted_blocks = sorted(block_to_points.items())
    
    for block_name, point_list in sorted_blocks:
        block_point_count = len(point_list)
        
        # 如果当前block单独就超过限制，单独成一批
        if block_point_count > batch_point_limit:
            # 先保存之前的批次
            if current_batch:
                block_batches.append(current_batch)
                current_batch = []
                current_batch_points = 0
            # 当前block单独成一批
            block_batches.append([(block_name, point_list)])
        # 如果加上当前block不超过限制，加入当前批次
        elif current_batch_points + block_point_count <= batch_point_limit:
            current_batch.append((block_name, point_list))
            current_batch_points += block_point_count
        # 如果加上当前block超过限制，开始新批次
        else:
            if current_batch:
                block_batches.append(current_batch)
            current_batch = [(block_name, point_list)]
            current_batch_points = block_point_count
    
    # 保存最后一个批次
    if current_batch:
        block_batches.append(current_batch)
    
    total_batches = len(block_batches)
    logging.info(f"共组合成 {total_batches} 个批次")
    for batch_idx, batch in enumerate(block_batches, 1):
        batch_points = sum(len(points) for _, points in batch)
        batch_blocks = [name for name, _ in batch]
        logging.info(f"  批次 {batch_idx}: {len(batch_blocks)} 个空间块, 共 {batch_points} 个点")
        if len(batch_blocks) <= 5:
            logging.info(f"    包含的空间块: {', '.join(batch_blocks)}")
        else:
            logging.info(f"    包含的空间块: {', '.join(batch_blocks[:5])}... (共{len(batch_blocks)}个)")
    
    return block_batches


def calculate_national_energy(point_to_country, grid_result_dir, block_to_path=None):
    """计算每个国家的总能耗和逐时能耗（按空间块分组处理，每个空间块只读取一次，多个空间块组合成批次）"""
    logging.info("开始计算国家能耗...")

    # 如果没有提供空间块映射，则创建它
    if block_to_path is None:
        block_to_path = load_spatial_block_mapping(grid_result_dir)

    # 按空间块分组点
    block_to_points = group_points_by_spatial_block(point_to_country, block_to_path)
    
    if len(block_to_points) == 0:
        logging.warning("没有找到有效的空间块，返回空结果")
        return {}, {}

    # 将空间块组合成批次（每批最多500个点，减少内存占用）
    block_batches = combine_blocks_into_batches(block_to_points, batch_point_limit=500)

    # 初始化国家能耗结果和逐时结果
    national_energy_results = {}
    national_hourly_results = {}
    cases = ['ref'] + [f'case{i}' for i in range(1, 21)]

    # 并行处理批次
    logging.info("开始并行处理空间块批次...")

    # 配置并行处理参数
    num_cores = multiprocessing.cpu_count()
    # 增加进程数以提高CPU利用率（使用更多核心）
    num_processes = min(num_cores, 12)  # 增加到12个进程

    logging.info(f"CPU核心数: {num_cores}")
    logging.info(f"使用进程数: {num_processes}")
    logging.info(f"共 {len(block_batches)} 个批次需要处理")

    # 准备参数列表
    batch_args = [(batch, point_to_country, block_to_path) for batch in block_batches]

    # 并行处理
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 减小chunksize，让任务更细粒度，提高负载均衡
        chunksize = max(1, len(batch_args) // (num_processes * 8))  # 从4改为8，减小chunksize
        logging.info(f"chunksize: {chunksize}")

        with tqdm(total=len(batch_args), desc="处理空间块批次") as pbar:
            for batch_results, batch_hourly_results in pool.imap_unordered(process_spatial_block_batch, batch_args, chunksize=chunksize):
                # 合并批次结果到总结果中
                for country, cases_data in batch_results.items():
                    if country not in national_energy_results:
                        national_energy_results[country] = {}
                        national_hourly_results[country] = {}
                        for case in cases:
                            national_energy_results[country][case] = {
                                'cooling_demand': 0.0,
                                'heating_demand': 0.0,
                                'total_demand': 0.0
                            }
                            # 初始化逐时数据（从第一个批次获取时间索引）
                            if country in batch_hourly_results and case in batch_hourly_results[country]:
                                time_index = batch_hourly_results[country][case]['cooling'].index
                                national_hourly_results[country][case] = {
                                    'cooling': pd.Series(0.0, index=time_index),
                                    'heating': pd.Series(0.0, index=time_index),
                                    'total': pd.Series(0.0, index=time_index)
                                }

                    for case, data in cases_data.items():
                        national_energy_results[country][case]['cooling_demand'] += data['cooling_demand']
                        national_energy_results[country][case]['heating_demand'] += data['heating_demand']
                        national_energy_results[country][case]['total_demand'] += data['total_demand']
                    
                    # 合并逐时数据
                    for case in cases:
                        if case in batch_hourly_results[country]:
                            national_hourly_results[country][case]['cooling'] += batch_hourly_results[country][case]['cooling']
                            national_hourly_results[country][case]['heating'] += batch_hourly_results[country][case]['heating']
                            national_hourly_results[country][case]['total'] += batch_hourly_results[country][case]['total']
                
                # 合并完批次结果后，立即释放批次数据的内存
                del batch_results, batch_hourly_results
                # 每处理几个批次后强制垃圾回收
                if pbar.n % 10 == 0:
                    gc.collect()

                pbar.update(1)

    total_points = sum(len(points) for points in block_to_points.values())
    logging.info(f"空间块批次处理完成，共处理 {total_points} 个点")

    return national_energy_results, national_hourly_results


def apply_power_coefficients(national_energy_results, params_dict, country_info_map=None):
    """应用功率系数到全年总能耗"""
    logging.info("开始应用功率系数...")

    default_heating_power = 27.9
    default_cooling_power = 48.5

    # 如果没有提供国家信息映射，则加载它
    if country_info_map is None:
        country_info_map, _ = load_country_info()

    final_results = {}

    # 使用进度条显示功率系数应用进度
    with tqdm(total=len(national_energy_results), desc="应用功率系数") as pbar:
        for country_iso2_code, cases in national_energy_results.items():
            # 跳过无效的国家ISO代码（注意：NA是Namibia的代码，不能当作无效值）
            # 使用str()转换后再检查，确保字符串"NA"不会被误判
            country_code_str = str(country_iso2_code) if country_iso2_code is not None else ''
            if pd.isna(country_iso2_code) or country_code_str.strip() == '':
                logging.warning(f"跳过无效的国家ISO代码: {country_iso2_code}")
                pbar.update(1)
                continue
            
            # 从country_info_map获取国家全称
            country_info = country_info_map.get(str(country_iso2_code), None)
            if country_info and country_info['Country_Name']:
                country_name = country_info['Country_Name']
            else:
                logging.warning(f"无法获取国家 {country_iso2_code} 的有效名称，跳过处理")
                pbar.update(1)
                continue

            # 获取功率系数
            if country_name in params_dict:
                heating_power = params_dict[country_name]['heating_power']
                cooling_power = params_dict[country_name]['cooling_power']
            else:
                heating_power = default_heating_power
                cooling_power = default_cooling_power

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


def apply_power_coefficients_to_hourly(national_hourly_results, params_dict, country_info_map=None):
    """应用功率系数到逐时能耗数据（GW单位）"""
    logging.info("开始对逐时能耗数据应用功率系数...")

    default_heating_power = 27.9
    default_cooling_power = 48.5

    # 如果没有提供国家信息映射，则加载它
    if country_info_map is None:
        country_info_map, _ = load_country_info()

    final_hourly_results = {}

    # 使用进度条显示功率系数应用进度
    with tqdm(total=len(national_hourly_results), desc="应用逐时功率系数") as pbar:
        for country_iso2_code, cases_data in national_hourly_results.items():
            # 跳过无效的国家ISO代码（注意：NA是Namibia的代码，不能当作无效值）
            # 使用str()转换后再检查，确保字符串"NA"不会被误判
            country_code_str = str(country_iso2_code) if country_iso2_code is not None else ''
            if pd.isna(country_iso2_code) or country_code_str.strip() == '':
                logging.warning(f"跳过无效的国家ISO代码: {country_iso2_code}")
                pbar.update(1)
                continue
            
            # 从country_info_map获取国家全称
            country_info = country_info_map.get(str(country_iso2_code), None)
            if country_info and country_info['Country_Name']:
                country_name = country_info['Country_Name']
            else:
                logging.warning(f"无法获取国家 {country_iso2_code} 的有效名称，跳过处理")
                pbar.update(1)
                continue

            # 获取功率系数
            if country_name in params_dict:
                heating_power = params_dict[country_name]['heating_power']
                cooling_power = params_dict[country_name]['cooling_power']
            else:
                heating_power = default_heating_power
                cooling_power = default_cooling_power

            # 应用功率系数到逐时数据（保持GW单位）
            final_hourly_results[country_iso2_code] = {}
            for case, hourly_data in cases_data.items():
                final_hourly_results[country_iso2_code][case] = {
                    'cooling': hourly_data['cooling'] * cooling_power,
                    'heating': hourly_data['heating'] * heating_power,
                    'total': hourly_data['heating'] * heating_power + hourly_data['cooling'] * cooling_power
                }

            pbar.update(1)

    logging.info(f"逐时功率系数应用完成，处理了 {len(final_hourly_results)} 个国家")
    return final_hourly_results


def save_results(final_results, national_population, output_dir, national_hourly_results, year, 
                  country_info_map=None, mapping_df=None):
    """保存结果到文件，包括全年总能耗和逐时能耗"""
    logging.info("开始保存结果...")

    # 如果没有提供国家信息映射，则加载它
    if country_info_map is None:
        country_info_map, _ = load_country_info()
    
    # 创建国家名称到Country_Code_2的映射（用于查找逐时数据）
    country_name_to_code = {}
    for code_2, info in country_info_map.items():
        if info['Country_Name']:
            country_name_to_code[info['Country_Name']] = code_2
    
    # 如果没有提供映射数据，则从point_country_mapping.csv加载
    if mapping_df is None:
        logging.info("从point_country_mapping.csv获取国家与大洲映射关系...")
        mapping_df = pd.read_csv(POINT_COUNTRY_MAPPING_FILE, keep_default_na=False)
    
    # 创建国家代码到大洲的映射（使用Country_Code_2）
    # 注意：NA是Namibia的代码，不能当作无效值
    country_to_continent = {}
    for _, row in mapping_df.iterrows():
        country_code = row['Country_Code_2']
        continent = row['Continent']
        # 检查是否为有效的国家代码（包括字符串"NA"）
        if country_code is not None and str(country_code).strip() != '' and country_code not in country_to_continent:
            country_to_continent[str(country_code)] = continent
    
    logging.info(f"获取到 {len(country_to_continent)} 个国家的洲际映射关系")
    
    # 使用传入的人口数据（已在年份循环外计算好，避免重复计算）
    national_population_new = national_population

    # 按大洲组织结果（从country_info_map获取国家代码）
    continents = {}
    for country_name in final_results.keys():
        # 跳过无效的国家名称（注意：这里检查的是国家名称，不是代码）
        country_name_str = str(country_name) if country_name is not None else ''
        if pd.isna(country_name) or country_name_str.strip() == '':
            logging.warning(f"跳过无效的国家名称: {country_name}")
            continue
        
        # 从country_info_map查找对应的Country_Code_2（二字母代码）
        country_iso2_code = None
        for code_2, info in country_info_map.items():
            if info['Country_Name'] == country_name:
                country_iso2_code = code_2
                break
        
        if country_iso2_code and country_iso2_code in country_to_continent:
            continent = country_to_continent[country_iso2_code]
            if continent not in continents:
                continents[continent] = []
            continents[continent].append((country_name, country_iso2_code))
        else:
            # 如果找不到映射，使用Unknown
            if 'Unknown' not in continents:
                continents['Unknown'] = []
            continents['Unknown'].append((country_name, country_iso2_code))

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
        for country_name, country_iso2_code in countries:
            if country_name in final_results:
                country_data = final_results[country_name]

                # 创建国家目录（使用二字母ISO代码）
                if country_iso2_code:
                    country_dir = os.path.join(continent_dir, country_iso2_code)
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
                if country_iso2_code and country_iso2_code in national_population_new['country'].values:
                    population = \
                    national_population_new[national_population_new['country'] == country_iso2_code]['total_population'].iloc[0]
                    logging.debug(f"国家 {country_name} ({country_iso2_code}) 人口: {population}")
                else:
                    logging.warning(f"国家 {country_name} ({country_iso2_code}) 没有找到人口数据")

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

                # 总能耗汇总（保留13位小数）
                summary_df = pd.DataFrame({
                    'total_demand_sum(TWh)': [f"{x:.13f}" for x in total_demand],
                    'total_demand_diff(TWh)': [f"{x:.13f}" for x in total_demand_diff],
                    'total_demand_reduction(%)': [f"{x:.13f}" for x in total_demand_reduction],
                    'heating_demand_sum(TWh)': [f"{x:.13f}" for x in heating_demand],
                    'heating_demand_diff(TWh)': [f"{x:.13f}" for x in heating_demand_diff],
                    'heating_demand_reduction(%)': [f"{x:.13f}" for x in heating_demand_reduction],
                    'cooling_demand_sum(TWh)': [f"{x:.13f}" for x in cooling_demand],
                    'cooling_demand_diff(TWh)': [f"{x:.13f}" for x in cooling_demand_diff],
                    'cooling_demand_reduction(%)': [f"{x:.13f}" for x in cooling_demand_reduction]
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
                    'total_demand_sum_p(kWh/person)': [f"{x:.4f}" for x in total_demand_p],
                    'total_demand_diff_p(kWh/person)': [f"{x:.4f}" for x in total_demand_diff_p],
                    'total_demand_p_reduction(%)': [f"{x:.4f}" for x in total_demand_p_reduction],
                    'heating_demand_sum_p(kWh/person)': [f"{x:.4f}" for x in heating_demand_p],
                    'heating_demand_diff_p(kWh/person)': [f"{x:.4f}" for x in heating_demand_diff_p],
                    'heating_demand_p_reduction(%)': [f"{x:.4f}" for x in heating_demand_p_reduction],
                    'cooling_demand_sum_p(kWh/person)': [f"{x:.4f}" for x in cooling_demand_p],
                    'cooling_demand_diff_p(kWh/person)': [f"{x:.4f}" for x in cooling_demand_diff_p],
                    'cooling_demand_p_reduction(%)': [f"{x:.4f}" for x in cooling_demand_p_reduction]
                }, index=cases)

                # 保存全年总能耗文件 - 使用二字母ISO代码作为文件名
                if country_iso2_code:
                    summary_df.to_csv(os.path.join(summary_dir, f"{country_iso2_code}_{year}_summary_results.csv"), index=True)
                    summary_p_df.to_csv(os.path.join(summary_p_dir, f"{country_iso2_code}_{year}_summary_p_results.csv"), index=True)
                    
                    # 保存逐时能耗文件（GW单位，保留10位小数，已应用功率系数）
                    # 注意：national_hourly_results 的键是 Country_Code_2（二字母代码）
                    # 从country_name_to_code获取对应的Country_Code_2
                    country_code_for_hourly = country_name_to_code.get(country_name, country_iso2_code)
                    if national_hourly_results and country_code_for_hourly in national_hourly_results:
                        save_hourly_energy(country_iso2_code, national_hourly_results[country_code_for_hourly], country_dir, year)
                else:
                    logging.error(f"无法保存国家 {country_name} 的结果文件，因为ISO代码为None")

    logging.info("结果保存完成")


def save_hourly_energy(country_iso2_code, country_hourly_data, country_dir, year):
    """保存国家的逐时能耗数据（GW单位，保留10位小数，已应用功率系数）"""
    cases = ['ref'] + [f'case{i}' for i in range(1, 21)]
    
    # 获取时间索引（从第一个工况获取）
    if not country_hourly_data or cases[0] not in country_hourly_data:
        logging.warning(f"国家 {country_iso2_code} 没有逐时数据，跳过保存")
        return
    
    time_index = country_hourly_data[cases[0]]['cooling'].index
    
    # 创建包含所有工况的DataFrame
    hourly_df = pd.DataFrame({'time': time_index})
    
    for case in cases:
        if case not in country_hourly_data:
            continue
        
        case_data = country_hourly_data[case]
        cooling_series = case_data['cooling']
        heating_series = case_data['heating']
        total_series = case_data['total']
        
        # 添加该工况的列（保留10位小数，已应用功率系数）
        hourly_df[f'{case}_heating_GW'] = [f"{x:.10f}" for x in heating_series.values]
        hourly_df[f'{case}_cooling_GW'] = [f"{x:.10f}" for x in cooling_series.values]
        hourly_df[f'{case}_total_GW'] = [f"{x:.10f}" for x in total_series.values]
    
    # 保存文件
    output_file = os.path.join(country_dir, f"{country_iso2_code}_{year}_hourly_energy.csv")
    hourly_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logging.debug(f"已保存 {country_iso2_code} {year}年逐时能耗数据到: {output_file}")


def check_missing_countries(final_results, processed_countries, country_info_map=None):
    """检查并记录缺失的国家"""
    logging.info("=== 检查缺失的国家 ===")
    
    if processed_countries is None:
        logging.warning("未加载参考国家列表，跳过检查")
        return
    
    # 如果没有提供国家信息映射，则加载它
    if country_info_map is None:
        country_info_map, _ = load_country_info()
    
    # 获取参考国家代码列表（使用Country_Code_2），确保都是字符串类型
    # 注意：NA是Namibia的代码，不能当作无效值
    reference_country_codes = set()
    for code in processed_countries['Country_Code_2'].unique():
        # 检查是否为有效的国家代码（包括字符串"NA"）
        if code is not None and str(code).strip():
            reference_country_codes.add(str(code).strip())
    logging.info(f"参考列表包含 {len(reference_country_codes)} 个唯一国家代码")
    
    # 从country_info_map获取处理的国家代码
    processed_country_codes = set()
    for country_name in final_results.keys():
        # 从country_info_map查找对应的Country_Code_2
        for code_2, info in country_info_map.items():
            if info['Country_Name'] == country_name:
                processed_country_codes.add(str(code_2))
                break
    
    logging.info(f"实际处理了 {len(processed_country_codes)} 个国家")
    
    # 找出缺失的国家
    missing_countries = reference_country_codes - processed_country_codes
    
    if missing_countries:
        logging.warning(f"发现 {len(missing_countries)} 个缺失的国家:")
        missing_info = []
        # 过滤掉非字符串类型的代码，并转换为字符串进行排序
        # 注意：NA是Namibia的代码，不能当作无效值
        valid_missing_codes = [str(code) for code in missing_countries if code is not None and str(code).strip()]
        for code in sorted(valid_missing_codes):
            # 从参考列表中获取国家名称
            country_info = processed_countries[processed_countries['Country_Code_2'] == code]
            if not country_info.empty:
                name = country_info.iloc[0]['Country_Name']
                continent = country_info.iloc[0]['continent']
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
        # 注意：NA是Namibia的代码，不能当作无效值
        valid_extra_codes = [str(code) for code in extra_countries if code is not None and str(code).strip()]
        for code in sorted(valid_extra_codes):
            logging.info(f"  - {code}")


def main():
    """主函数"""
    logging.info("开始国家级别能耗聚合计算...")

    try:
        # 1. 加载基础数据（所有年份共用）
        logging.info("=== 第一步：加载基础数据 ===")

        # 一次性加载所有通用数据（只读取一次，后续复用）
        logging.info("加载参考国家列表...")
        processed_countries = load_processed_countries()
        
        logging.info("加载国家信息映射...")
        country_info_map, countries_df = load_country_info()

        logging.info("加载功率系数参数...")
        params_dict = load_parameters(country_info_map)

        logging.info("加载人口数据...")
        population_df = load_population_data()

        logging.info("加载点-国家映射数据...")
        point_to_country, mapping_df = load_point_country_mapping()

        # 计算国家人口数据（只计算一次，所有年份共用）
        logging.info("=== 计算国家人口数据（所有年份共用） ===")
        national_population = calculate_national_population(population_df, point_to_country)

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
            
            # 加载空间块映射（一次性加载，后续复用）
            logging.info(f"加载{year}年空间块映射...")
            block_to_path = load_spatial_block_mapping(year_grid_dir)
            
            if len(block_to_path) == 0:
                logging.warning(f"{year}年没有找到有效的空间块文件，跳过")
                continue

            # 计算国家能耗（按空间块分组处理，每个空间块只读取一次）
            logging.info(f"=== 计算{year}年国家能耗 ===")
            national_energy_results, national_hourly_results = calculate_national_energy(
                point_to_country, year_grid_dir, block_to_path)

            # 应用功率系数到全年总能耗（传递country_info_map，避免重复读取）
            logging.info(f"=== 应用{year}年功率系数（全年总能耗） ===")
            final_results = apply_power_coefficients(national_energy_results, params_dict, country_info_map)

            # 应用功率系数到逐时能耗数据（传递country_info_map，避免重复读取）
            logging.info(f"=== 应用{year}年功率系数（逐时能耗） ===")
            final_hourly_results = apply_power_coefficients_to_hourly(national_hourly_results, params_dict, country_info_map)

            # 保存结果（包括逐时能耗，传递通用数据，避免重复读取）
            logging.info(f"=== 保存{year}年结果 ===")
            save_results(final_results, national_population, year_output_dir, final_hourly_results, year,
                        country_info_map, mapping_df)

            # 检查缺失的国家（传递country_info_map，避免重复读取）
            logging.info(f"=== 检查{year}年缺失的国家 ===")
            check_missing_countries(final_results, processed_countries, country_info_map)

        logging.info("所有年份国家级别能耗聚合计算完成！")

    except Exception as e:
        error_msg = f"主程序执行出错: {str(e)}"
        logging.error(error_msg)
        raise


if __name__ == "__main__":
    main()
