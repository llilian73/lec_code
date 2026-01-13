"""
全球网格点BAIT和能耗计算工具

功能概述：
本工具用于计算全球网格点的建筑适应性室内温度（BAIT）和建筑能耗需求。基于气候数据和人口数据，为每个有效人口点计算BAIT和多种节能案例的能耗需求，为后续的全球建筑能耗分析提供基础数据。

输入数据：
1. 人口数据：
   - 文件路径：energy_consumption_gird/result/data/population_points.csv
   - 包含所有有效人口点的经纬度和人口数
   - 格式：CSV文件，列包括lat、lon、population

2. 气候数据：
   - 文件路径：energy_consumption_gird/result/data/point_lat{lat}_lon{lon}_climate.csv
   - 包含每个网格点的气象时间序列数据
   - 变量：T2M（温度）、U2M/V2M（风速分量）、QV2M（湿度）、SWGDN（辐射）
   - 时间范围：2019年全年逐时数据

3. 计算参数：
   - 供暖阈值：14°C
   - 制冷阈值：20°C
   - BAIT平滑参数：0.50
   - 太阳辐射增益：0.012
   - 风寒效应：-0.20
   - 湿度不适感：0.050

主要功能：
1. 气候数据处理：
   - 加载每个网格点的气候数据
   - 数据单位转换（温度K→℃、湿度kg/kg→g/kg）
   - 计算风速和风向
   - 数据质量检查和清洗

2. BAIT计算：
   - 使用demand_ninja.core_p._bait函数
   - 基于温度、辐射、风速、湿度计算建筑适应性室内温度
   - 应用平滑处理和舒适度参数
   - 生成逐时BAIT时间序列

3. 能耗需求计算：
   - 计算21种工况的能耗需求（ref + case1-case20）
   - 包括供暖需求、制冷需求、总需求
   - 计算供暖度日数（HDD）和制冷度日数（CDD）
   - 考虑人口权重和功率系数

4. 并行处理：
   - 多进程并行处理所有网格点
   - 自动优化进程数和chunksize
   - 进度跟踪和错误处理
   - 内存管理和资源清理

输出结果：
1. BAIT数据文件：
   - point_lat{lat}_lon{lon}_BAIT.csv：逐时BAIT数据

2. 能耗需求文件：
   - point_lat{lat}_lon{lon}_cooling.csv：逐时制冷需求
   - point_lat{lat}_lon{lon}_heating.csv：逐时供暖需求
   - point_lat{lat}_lon{lon}_total.csv：逐时总需求

3. 日志文件：
   - grid_point_calculation.log：详细的计算日志
   - failed_energy_points.csv：计算失败的点信息

4. 输出目录：
   - energy_consumption_gird/result/result_half/：所有结果文件

数据流程：
1. 数据加载：
   - 读取人口数据文件
   - 为每个点加载对应的气候数据
   - 数据格式验证和质量检查

2. 数据预处理：
   - 单位转换和标准化
   - 缺失值处理和时间序列对齐
   - 数据插值和重采样

3. BAIT计算：
   - 创建天气数据DataFrame
   - 调用_bait函数计算建筑适应性温度
   - 应用平滑和舒适度参数

4. 能耗计算：
   - 计算日均BAIT
   - 为每个工况计算能耗需求
   - 生成逐时和汇总数据

5. 结果保存：
   - 保存BAIT和能耗数据到CSV文件
   - 记录处理状态和错误信息
   - 生成统计报告

计算特点：
- 高精度：基于逐时气象数据计算
- 多工况：支持21种不同的节能案例
- 并行处理：多进程并行计算，提高效率
- 错误容错：完善的异常处理和日志记录
- 内存优化：分批处理，控制内存使用

技术参数：
- BAIT平滑参数：0.50
- 太阳辐射增益系数：0.012
- 风寒效应系数：-0.20
- 湿度不适感系数：0.050
- 供暖阈值：14°C
- 制冷阈值：20°C
- 功率系数：1.0（不考虑功率变化）

性能优化：
- 多进程并行处理
- 预计算网格索引
- 内存使用监控
- 进度跟踪显示
- 错误恢复机制

数据质量保证：
- 输入数据验证
- 缺失值处理
- 异常值检测
- 时间序列对齐
- 结果一致性检查

输出格式：
- 时间格式：YYYY-MM-DD HH:MM:SS
- 能耗单位：GW（吉瓦）
- HDD/CDD单位：°C·day
- 坐标精度：三位小数
- 编码格式：UTF-8 with BOM
"""

import pandas as pd
import numpy as np
import xarray as xr
import os
import sys
from pathlib import Path
import multiprocessing
from tqdm import tqdm
import logging
from datetime import datetime
from collections import defaultdict
import gc

# 将项目的根目录加入到 sys.path
# 当前文件在 global_energy_demand/grid/ 下，需要往上三层到达项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from demand_ninja.core_p import _bait, demand_p as demand
from CandD.calculate import calculate_cases

# 设置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('grid_point_calculation.log', encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])

# 配置参数
DATA_BASE_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\data"
OUTPUT_BASE_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\result_half_parquet"
YEARS = [2016, 2017, 2018, 2019, 2020]
# 空间块大小（度）
BLOCK_SIZE = 10

# 确保输出目录存在
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)


def load_population_data():
    """加载人口数据"""
    logging.info("开始加载人口数据...")

    population_csv_path = os.path.join(DATA_BASE_DIR, 'population_points.csv')
    if not os.path.exists(population_csv_path):
        raise FileNotFoundError(f"人口数据文件不存在: {population_csv_path}")

    population_df = pd.read_csv(population_csv_path)
    logging.info(f"加载人口数据完成，共 {len(population_df)} 个点")

    return population_df


def load_climate_data_for_point(lat, lon, data_dir):
    """加载单个点的气候数据"""
    try:
        # 生成文件名 - 使用三位小数精度
        filename = f"point_lat{lat:.3f}_lon{lon:.3f}_climate.csv"
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            logging.error(f"文件不存在: {file_path}")
            return None

        # 读取CSV文件，使用python引擎避免tokenizing错误
        try:
            df = pd.read_csv(file_path, engine='python')
        except Exception as e:
            # 如果python引擎失败，尝试c引擎
            df = pd.read_csv(file_path, engine='c')

        # 检查数据完整性
        if len(df) == 0:
            logging.error("文件为空")
            return None

        # 检查必需的列是否存在
        required_columns = ['time', 'T2M', 'U2M', 'V2M', 'QV2M', 'SWGDN']
        for col in required_columns:
            if col not in df.columns:
                logging.error(f"缺少必需的列: {col}")
                return None

        # 删除包含NaN的行
        df = df.dropna(subset=required_columns)
        if len(df) == 0:
            logging.error("删除NaN后数据为空")
            return None

        # 转换时间戳为UTC时间
        df['time'] = pd.to_datetime(df['time'], unit='ns', utc=True)

        # 转换温度从K到℃
        df['T2M'] = df['T2M'] - 273.15

        # 计算风速，处理可能的NaN值
        df['wind_speed_2m'] = np.sqrt(df['U2M'] ** 2 + df['V2M'] ** 2)

        # 转换湿度单位 (kg/kg to g/kg)
        df['QV2M'] = df['QV2M'] * 1000

        # 再次检查是否有NaN值
        df = df.dropna()
        if len(df) == 0:
            logging.error("最终处理后数据为空")
            return None

        # 设置时间索引
        df.set_index('time', inplace=True)

        # 验证所有列的长度一致
        lengths = [len(df[col]) for col in ['T2M', 'SWGDN', 'wind_speed_2m', 'QV2M']]
        if len(set(lengths)) > 1:
            logging.error(f"列长度不一致: {lengths}")
            return None

        # 检查时间索引是否连续（静默处理）
        time_diff = df.index.to_series().diff().dropna()
        if len(time_diff) > 0:
            # 如果时间间隔不一致，重新采样到统一频率
            if time_diff.std() > pd.Timedelta(seconds=1):
                df = df.resample('1H').mean().interpolate(method='linear')

        return df

    except Exception as e:
        logging.error(f"加载气候数据失败 (lat={lat}, lon={lon}): {e}")
        return None


def calculate_bait_for_point(climate_df):
    """计算单个点的BAIT"""
    try:
        # 检查数据完整性
        required_columns = ['T2M', 'SWGDN', 'wind_speed_2m', 'QV2M']
        for col in required_columns:
            if col not in climate_df.columns:
                logging.error(f"缺少必需的列: {col}")
                return None

        # 检查数据长度一致性
        data_length = len(climate_df)
        if data_length == 0:
            logging.error("气候数据为空")
            return None

        # 验证所有列的长度一致
        lengths = [len(climate_df[col]) for col in required_columns]
        if len(set(lengths)) > 1:
            logging.error(f"列长度不一致: {lengths}")
            return None

        # 创建天气DataFrame，确保所有数组长度一致
        weather_df = pd.DataFrame({
            'temperature': climate_df['T2M'].values,
            'radiation_global_horizontal': climate_df['SWGDN'].values,
            'wind_speed_2m': climate_df['wind_speed_2m'].values,
            'humidity': climate_df['QV2M'].values
        }, index=climate_df.index)

        # 检查是否有NaN值
        if weather_df.isnull().any().any():
            weather_df = weather_df.interpolate(method='linear')
            # 如果插值后仍有NaN，删除这些行
            weather_df = weather_df.dropna()
            if len(weather_df) == 0:
                logging.error("插值后数据为空")
                return None

        # 使用固定的平滑参数，与3_C_global country state_province.py保持一致
        smoothing = 0.50

        # 移除调试信息，减少输出

        # 计算BAIT - 使用原始的_bait函数
        try:
            bait = _bait(
                weather=weather_df,
                smoothing=smoothing,
                solar_gains=0.012,
                wind_chill=-0.20,
                humidity_discomfort=0.050
            )

        except ValueError as e:
            if "All arrays must be of the same length" in str(e):
                logging.error(f"数组长度不匹配错误: {e}")
                logging.error(f"原始数据形状: {weather_df.shape}")
                logging.error(f"各列长度: {[len(weather_df[col]) for col in weather_df.columns]}")

                # 尝试修复数据
                weather_df_clean = weather_df.dropna()
                logging.info(f"清理后数据形状: {weather_df_clean.shape}")

                if len(weather_df_clean) > 0:
                    bait = _bait(
                        weather=weather_df_clean,
                        smoothing=smoothing,
                        solar_gains=0.014,
                        wind_chill=-0.12,
                        humidity_discomfort=0.036
                    )
                else:
                    logging.error("清理后数据为空")
                    return None
            else:
                logging.error(f"ValueError: {e}")
                return None
        except Exception as e:
            logging.error(f"BAIT计算失败: {e}")
            logging.error(f"错误类型: {type(e)}")
            return None

        return bait

    except Exception as e:
        logging.error(f"计算BAIT失败: {e}")
        return None


def calculate_energy_demand_for_point(bait, population):
    """计算单个点的能耗需求"""
    try:
        # 设置计算参数 - 与3_C_global country state_province.py保持一致
        base_params = {
            "heating_power": 1,  # 不考虑功率系数，设为1
            "cooling_power": 1,  # 不考虑功率系数，设为1
            "heating_threshold_people": 14,
            "cooling_threshold_people": 20,
            "base_power": 0,
            "population": population
        }

        # 获取计算工况
        cases = calculate_cases(base_params)

        # 计算日均BAIT
        daily_bait = bait.resample('D').mean()

        # 计算每个工况的能耗
        results = {}
        for case_name, params in cases.items():
            result = demand(
                daily_bait,
                heating_threshold_background=params["heating_threshold_background"],
                heating_threshold_people=params["heating_threshold_people"],
                cooling_threshold_background=params["cooling_threshold_background"],
                cooling_threshold_people=params["cooling_threshold_people"],
                p_ls=params["p_ls"],
                base_power=params["base_power"],
                heating_power=params["heating_power"],
                cooling_power=params["cooling_power"],
                population=params["population"],
                use_diurnal_profile=False  # 不使用日内负荷曲线
            )

            # 保存逐时能耗数据和HDD/CDD
            results[case_name] = {
                'heating_demand': result['heating_demand'],  # 逐时数据，单位：GW
                'cooling_demand': result['cooling_demand'],  # 逐时数据，单位：GW
                'total_demand': result['total_demand'],  # 逐时数据，单位：GW
                'hdd': result['hdd'],  # HDD数据，单位：°C·day
                'cdd': result['cdd']  # CDD数据，单位：°C·day
            }
        return results, daily_bait  # 返回日均BAIT以便后续使用

    except Exception as e:
        logging.error(f"计算能耗需求失败: {e}")
        return None, None


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


def process_single_point(args):
    """处理单个点的数据"""
    lat, lon, population, data_dir, output_dir = args

    try:
        # 1. 加载气候数据
        climate_df = load_climate_data_for_point(lat, lon, data_dir)
        if climate_df is None:
            logging.error(f"气候数据加载失败 (lat={lat:.3f}, lon={lon:.3f})")
            return None

        # 2. 计算BAIT
        bait = calculate_bait_for_point(climate_df)
        if bait is None:
            logging.error(f"BAIT计算失败 (lat={lat:.3f}, lon={lon:.3f})")
            return None

        # 3. 计算能耗
        energy_results, daily_bait = calculate_energy_demand_for_point(bait, population)
        if energy_results is None:
            logging.error(f"能耗计算失败 (lat={lat:.3f}, lon={lon:.3f})")
            return None

        # 4. 创建包含所有数据的DataFrame
        if energy_results:
            # 准备时间列
            time_col = bait.index.strftime('%Y-%m-%d %H:%M:%S')
            case_names = ['ref'] + [f'case{i}' for i in range(1, 21)]
            
            # 创建包含所有数据的DataFrame
            # 列：lat, lon, time, BAIT, ref_cooling, ref_heating, ref_total, case1_cooling, ...
            data_dict = {
                'lat': [lat] * len(time_col),
                'lon': [lon] * len(time_col),
                'time': time_col,
                'BAIT': bait.values
            }
            
            # 添加所有工况的cooling, heating, total数据
            for case_name in case_names:
                if case_name in energy_results:
                    data_dict[f'{case_name}_cooling'] = energy_results[case_name]['cooling_demand'].values
                    data_dict[f'{case_name}_heating'] = energy_results[case_name]['heating_demand'].values
                    data_dict[f'{case_name}_total'] = energy_results[case_name]['total_demand'].values
            
            result_df = pd.DataFrame(data_dict)
            
            result_data = {
                'lat': lat,
                'lon': lon,
                'population': population,
                'data': result_df
            }

            return result_data

        logging.error(f"未找到ref工况结果 (lat={lat:.3f}, lon={lon:.3f})")
        return None

    except Exception as e:
        logging.error(f"处理点失败 (lat={lat:.3f}, lon={lon:.3f}): {e}")
        return None


def process_single_point_for_parquet(point_data):
    """处理单个点的数据，返回结果DataFrame（不保存）"""
    try:
        lat, lon, population, data_dir = point_data['lat'], point_data['lon'], point_data['population'], point_data['data_dir']

        # 处理单个点
        result = process_single_point((lat, lon, population, data_dir, None))

        if result is not None and 'data' in result:
            return {
                'status': 'success',
                'lat': lat,
                'lon': lon,
                'population': population,
                'data': result['data']
            }
        else:
            return {
                'status': 'process_failed',
                'lat': lat,
                'lon': lon,
                'population': population,
                'data': None,
                'message': f"处理点 ({lat}, {lon}) 失败: 返回None"
            }

    except Exception as e:
        return {
            'status': 'exception',
            'lat': lat,
            'lon': lon,
            'population': population,
            'data': None,
            'message': f"处理点 ({lat}, {lon}) 异常: {str(e)}"
        }


def save_single_block(block_df, block_name, year, output_base_dir):
    """保存单个空间块数据为Parquet文件"""
    year_dir = os.path.join(output_base_dir, f'{year}')
    os.makedirs(year_dir, exist_ok=True)
    
    try:
        # 生成文件路径
        parquet_filename = f"{block_name}.parquet"
        parquet_path = os.path.join(year_dir, parquet_filename)
        
        if block_df is None or len(block_df) == 0:
            logging.warning(f"空间块 {block_name} 数据为空，跳过保存")
            return None
        
        # 保存为Parquet文件
        block_df.to_parquet(parquet_path, compression='snappy', index=False)
        logging.info(f"保存空间块 {block_name}: {len(block_df)} 行数据到 {parquet_path}")
        
        # 保存后立即释放内存
        del block_df
        
        # 强制垃圾回收
        gc.collect()
        
        return parquet_path
        
    except Exception as e:
        logging.error(f"保存空间块 {block_name} 失败: {str(e)}")
        return None


def main():
    """主函数"""
    logging.info("开始网格点BAIT和能耗计算...")

    try:
        # 1. 加载人口数据
        logging.info("=== 第一步：加载人口数据 ===")
        population_df = load_population_data()

        # 2. 配置并行处理参数
        total_points = len(population_df)
        num_cores = multiprocessing.cpu_count()
        # 计算合适的进程数：18小时 -> 8小时，需要约2.25倍加速
        # 考虑到并行开销，使用较少的进程数
        num_processes = min(num_cores *2, 16)  # 限制最大进程数为6

        logging.info(f"CPU核心数: {num_cores}")
        logging.info(f"使用进程数: {num_processes}")
        logging.info(f"总共需要处理 {total_points} 个点")
        logging.info(f"预期加速比: ~{num_processes:.1f}x")

        # 3. 处理每个年份的数据
        for year in YEARS:
            logging.info(f"=== 处理{year}年数据 ===")

            # 创建年份输入和输出目录
            year_data_dir = os.path.join(DATA_BASE_DIR, str(year))
            year_output_dir = os.path.join(OUTPUT_BASE_DIR, f'{year}')
            os.makedirs(year_output_dir, exist_ok=True)

            # 检查输入目录是否存在
            if not os.path.exists(year_data_dir):
                logging.warning(f"{year}年数据目录不存在: {year_data_dir}，跳过")
                continue

            # 第一步：确定所有点的block分布
            logging.info(f"=== 第一步：确定所有点的block分布 ===")
            block_points_dict = defaultdict(list)  # block_name -> list of point dicts
            
            for _, row in population_df.iterrows():
                lat = row['lat']
                lon = row['lon']
                block_name = get_spatial_block(lat, lon, BLOCK_SIZE)
                block_points_dict[block_name].append({
                    'lat': lat,
                    'lon': lon,
                    'population': row['population'],
                    'data_dir': year_data_dir,
                    'block_name': block_name  # 保存block名称，用于后续分组
                })
            
            total_blocks = len(block_points_dict)
            logging.info(f"共识别出 {total_blocks} 个空间块")
            
            # 统计每个block的点数
            block_counts = {block_name: len(points) for block_name, points in block_points_dict.items()}
            logging.info(f"每个block的点数统计（前10个）:")
            for block_name, count in sorted(block_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                logging.info(f"  {block_name}: {count} 个点")

            # 第二步：组合block成批次（每批最多1000个点）
            logging.info(f"=== 第二步：组合block成批次（每批最多1000个点）===")
            BATCH_POINT_LIMIT = 1000
            block_batches = []  # 每个元素是一个批次，包含多个(block_name, point_list)的列表
            
            current_batch = []
            current_batch_points = 0
            
            # 按block名称排序，确保处理顺序一致
            sorted_blocks = sorted(block_points_dict.items())
            
            for block_name, point_list in sorted_blocks:
                block_point_count = len(point_list)
                
                # 如果当前block单独就超过1000点，单独成一批
                if block_point_count > BATCH_POINT_LIMIT:
                    # 先保存之前的批次
                    if current_batch:
                        block_batches.append(current_batch)
                        current_batch = []
                        current_batch_points = 0
                    # 当前block单独成一批
                    block_batches.append([(block_name, point_list)])
                # 如果加上当前block不超过1000，加入当前批次
                elif current_batch_points + block_point_count <= BATCH_POINT_LIMIT:
                    current_batch.append((block_name, point_list))
                    current_batch_points += block_point_count
                # 如果加上当前block超过1000，开始新批次
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
                logging.info(f"  批次 {batch_idx}: {len(batch_blocks)} 个block, 共 {batch_points} 个点")
                logging.info(f"    包含的block: {', '.join(batch_blocks[:5])}{'...' if len(batch_blocks) > 5 else ''}")

            # 第三步：按批次处理
            logging.info(f"=== 第三步：按批次处理{year}年网格点 ===")
            failed_points = []
            total_saved_points = 0
            total_processed_points = 0

            # 配置并行处理参数
            chunksize = max(1, 10)
            logging.info(f"每个批次内的chunksize: {chunksize}")

            # 按批次处理
            for batch_idx, batch in enumerate(block_batches, 1):
                # 收集当前批次所有block的点
                batch_point_list = []
                # 创建坐标到block_name的映射（使用元组作为键）
                coord_to_block = {}  # (lat, lon) -> block_name
                
                for block_name, point_list in batch:
                    for point in point_list:
                        # 使用坐标作为键（保留3位小数精度）
                        lat_key = round(point['lat'], 3)
                        lon_key = round(point['lon'], 3)
                        coord_to_block[(lat_key, lon_key)] = block_name
                        batch_point_list.append(point)
                
                total_batch_points = len(batch_point_list)
                batch_block_names = [name for name, _ in batch]
                logging.info(f"--- 处理批次 {batch_idx}/{total_batches}，包含 {len(batch_block_names)} 个block，共 {total_batch_points} 个点 ---")
                
                # 并行处理当前批次的所有点
                batch_results = []  # 存储所有结果，包含block_name信息
                batch_failed = []
                
                with multiprocessing.Pool(processes=num_processes) as pool:
                    with tqdm(total=total_batch_points, desc=f"处理批次{batch_idx}") as pbar:
                        for result in pool.imap_unordered(process_single_point_for_parquet, batch_point_list, chunksize=chunksize):
                            if result['status'] == 'success' and result['data'] is not None:
                                # 通过坐标匹配找到block_name
                                lat_key = round(result['lat'], 3)
                                lon_key = round(result['lon'], 3)
                                block_name = coord_to_block.get((lat_key, lon_key), None)
                                
                                if block_name:
                                    batch_results.append({
                                        'block_name': block_name,
                                        'data': result['data']
                                    })
                                else:
                                    logging.warning(f"无法确定点 ({result['lat']}, {result['lon']}) 的block名称")
                                    batch_failed.append(f"处理点 ({result['lat']}, {result['lon']}) 失败: 无法确定block名称")
                                    # 释放无法匹配的数据
                                    if 'data' in result and result['data'] is not None:
                                        del result['data']
                            else:
                                batch_failed.append(result.get('message', f"处理点 ({result['lat']}, {result['lon']}) 失败"))
                                # 如果处理失败，确保释放 result 中的 data（如果存在）
                                if 'data' in result and result['data'] is not None:
                                    del result['data']
                            pbar.update(1)
                
                # 按block分组结果
                block_results_dict = defaultdict(list)  # block_name -> list of DataFrames
                for result_item in batch_results:
                    block_name = result_item['block_name']
                    block_results_dict[block_name].append(result_item['data'])
                
                # 释放batch_results
                batch_results.clear()
                
                # 保存每个block的数据
                for block_name, block_dataframes in block_results_dict.items():
                    if block_dataframes:
                        try:
                            # 记录成功处理的点数
                            success_count = len(block_dataframes)
                            
                            merged_block_df = pd.concat(block_dataframes, ignore_index=True)
                            logging.info(f"空间块 {block_name}: 成功处理 {success_count} 个点，合并后共 {len(merged_block_df)} 行数据")
                            
                            # 立即释放原始 DataFrame 列表的内存
                            for df in block_dataframes:
                                del df
                            block_dataframes.clear()
                            
                            # 保存当前block的数据
                            saved_path = save_single_block(merged_block_df, block_name, year, OUTPUT_BASE_DIR)
                            if saved_path:
                                total_saved_points += len(merged_block_df)
                                total_processed_points += success_count
                                logging.info(f"空间块 {block_name} 保存成功: {len(merged_block_df)} 行数据")
                            else:
                                logging.error(f"空间块 {block_name} 保存失败")
                            
                            # 释放合并后的DataFrame
                            del merged_block_df
                            
                        except Exception as e:
                            logging.error(f"合并空间块 {block_name} 数据失败: {str(e)}")
                            # 即使合并失败，也要释放内存
                            for df in block_dataframes:
                                del df
                            block_dataframes.clear()
                    else:
                        logging.warning(f"空间块 {block_name} 没有成功处理的数据")
                
                # 释放block_results_dict
                block_results_dict.clear()
                
                # 记录失败的点
                failed_points.extend(batch_failed)
                
                # 释放batch_point_list和coord_to_block的内存
                batch_point_list.clear()
                coord_to_block.clear()
                
                # 强制垃圾回收，释放内存
                gc.collect()
                
                logging.info(f"批次 {batch_idx} 处理完成，内存已释放")
            
            logging.info(f"{year}年总共保存了 {total_saved_points} 行数据")
            logging.info(f"{year}年并行处理完成，成功处理了 {total_processed_points} 个点")
            if failed_points:
                logging.warning(f"{year}年失败点数: {len(failed_points)}")

            # 保存失败点信息
            try:
                if failed_points:
                    failed_points_path = os.path.join(year_output_dir, f'failed_energy_points_{year}.csv')
                    failed_df = pd.DataFrame({'error_message': failed_points})
                    failed_df.to_csv(failed_points_path, index=False, encoding='utf-8-sig')
                    logging.info(f"{year}年失败点信息已保存至: {failed_points_path}")
                else:
                    logging.info(f"{year}年没有失败的点，无需保存失败点信息")
            except Exception as e:
                logging.warning(f"保存{year}年失败点信息时出错: {e}")
            
            # 清理block_points_dict的内存
            del block_points_dict
            gc.collect()

        # 输出统计信息
        logging.info("=== 所有年份计算完成 ===")
        logging.info(f"结果保存路径: {OUTPUT_BASE_DIR}")
        logging.info(f"文件格式: Parquet (compression=snappy)")
        logging.info(f"空间块大小: {BLOCK_SIZE}°×{BLOCK_SIZE}°")
        logging.info("能耗单位: GW (吉瓦)")
        logging.info("HDD单位: °C·day (摄氏度·天)")
        logging.info("CDD单位: °C·day (摄氏度·天)")

        logging.info("所有年份网格点BAIT和能耗计算完成！")

    except Exception as e:
        error_msg = f"主程序执行出错: {str(e)}"
        logging.error(error_msg)
        raise


if __name__ == "__main__":
    main()
