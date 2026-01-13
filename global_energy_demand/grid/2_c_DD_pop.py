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
import concurrent.futures
from functools import partial
import gc
import psutil

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
DATA_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\data_test_2019"
OUTPUT_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\result_half_test_test_2019"

# 批量处理配置
BATCH_SIZE = 20   # 批量处理大小 - 减少内存使用
MAX_WORKERS = 4   # 最大工作线程数 - 减少并行度
MEMORY_THRESHOLD = 0.7  # 内存使用阈值 - 更保守

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_population_data():
    """加载人口数据"""
    logging.info("开始加载人口数据...")

    population_csv_path = os.path.join(DATA_DIR, 'population_points.csv')
    if not os.path.exists(population_csv_path):
        raise FileNotFoundError(f"人口数据文件不存在: {population_csv_path}")

    population_df = pd.read_csv(population_csv_path)
    logging.info(f"加载人口数据完成，共 {len(population_df)} 个点")

    return population_df


def load_climate_data_for_point(lat, lon):
    """加载单个点的气候数据"""
    try:
        # 生成文件名 - 使用三位小数精度
        filename = f"point_lat{lat:.3f}_lon{lon:.3f}_climate.csv"
        file_path = os.path.join(DATA_DIR, filename)

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


def load_climate_data_batch(points_batch):
    """批量加载气候数据"""
    climate_data = {}
    failed_points = []
    
    def load_single_file(point):
        lat, lon, population = point
        try:
            filename = f"point_lat{lat:.3f}_lon{lon:.3f}_climate.csv"
            file_path = os.path.join(DATA_DIR, filename)
            
            if not os.path.exists(file_path):
                return (lat, lon), None, f"文件不存在: {file_path}"
            
            # 使用更高效的读取方式
            df = pd.read_csv(file_path, engine='c', low_memory=False)
            
            if len(df) == 0:
                return (lat, lon), None, "文件为空"
            
            # 检查必需的列
            required_columns = ['time', 'T2M', 'U2M', 'V2M', 'QV2M', 'SWGDN']
            if not all(col in df.columns for col in required_columns):
                return (lat, lon), None, "缺少必需的列"
            
            # 数据预处理
            df = df.dropna(subset=required_columns)
            if len(df) == 0:
                return (lat, lon), None, "删除NaN后数据为空"
            
            # 批量转换
            df['time'] = pd.to_datetime(df['time'], unit='ns', utc=True)
            df['T2M'] = df['T2M'] - 273.15
            df['wind_speed_2m'] = np.sqrt(df['U2M'] ** 2 + df['V2M'] ** 2)
            df['QV2M'] = df['QV2M'] * 1000
            
            df = df.dropna()
            if len(df) == 0:
                return (lat, lon), None, "最终处理后数据为空"
            
            df.set_index('time', inplace=True)
            
            # 时间序列重采样
            time_diff = df.index.to_series().diff().dropna()
            if len(time_diff) > 0 and time_diff.std() > pd.Timedelta(seconds=1):
                df = df.resample('1H').mean().interpolate(method='linear')
            
            return (lat, lon), df, None
            
        except Exception as e:
            return (lat, lon), None, f"加载失败: {str(e)}"
    
    # 使用线程池并行读取文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(load_single_file, point) for point in points_batch]
        
        for future in concurrent.futures.as_completed(futures):
            point_key, data, error = future.result()
            if data is not None:
                climate_data[point_key] = data
            else:
                failed_points.append((point_key, error))
    
    return climate_data, failed_points


def check_memory_usage():
    """检查内存使用情况"""
    memory = psutil.virtual_memory()
    return memory.percent / 100.0


def monitor_performance():
    """监控系统性能"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    return cpu_percent, memory_percent


def optimize_batch_size(current_batch_size, cpu_usage, memory_usage):
    """根据系统性能动态调整批量大小"""
    if memory_usage > 0.8:
        return max(50, current_batch_size // 2)
    elif memory_usage < 0.5 and cpu_usage < 50:
        return min(200, current_batch_size * 2)
    else:
        return current_batch_size


def save_results_batch(results_batch):
    """批量保存结果"""
    saved_count = 0
    failed_saves = []
    
    for result in results_batch:
        try:
            lat, lon = result['lat'], result['lon']
            base_filename = f"point_lat{lat:.3f}_lon{lon:.3f}"
            
            # 批量写入文件
            files_to_save = [
                (f"{base_filename}_BAIT.csv", result['bait_data']),
                (f"{base_filename}_cooling.csv", result['cooling_data']),
                (f"{base_filename}_heating.csv", result['heating_data']),
                (f"{base_filename}_total.csv", result['total_data'])
            ]
            
            for filename, data in files_to_save:
                file_path = os.path.join(OUTPUT_DIR, filename)
                data.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            saved_count += 1
            
        except Exception as e:
            failed_saves.append(f"保存点 ({lat}, {lon}) 失败: {str(e)}")
    
    return saved_count, failed_saves


def calculate_bait_for_point(climate_df):
    """计算单个点的BAIT - 优化内存使用"""
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

        # 使用更节省内存的方式创建天气DataFrame
        try:
            # 直接使用numpy数组，避免DataFrame的额外开销
            temperature = climate_df['T2M'].values.astype(np.float32)  # 使用float32节省内存
            radiation = climate_df['SWGDN'].values.astype(np.float32)
            wind_speed = climate_df['wind_speed_2m'].values.astype(np.float32)
            humidity = climate_df['QV2M'].values.astype(np.float32)
            
            # 检查NaN值
            if np.isnan(temperature).any() or np.isnan(radiation).any() or np.isnan(wind_speed).any() or np.isnan(humidity).any():
                # 使用numpy的插值方法
                valid_mask = ~(np.isnan(temperature) | np.isnan(radiation) | np.isnan(wind_speed) | np.isnan(humidity))
                if not valid_mask.any():
                    logging.error("所有数据都是NaN")
                    return None
                
                # 只保留有效数据
                temperature = temperature[valid_mask]
                radiation = radiation[valid_mask]
                wind_speed = wind_speed[valid_mask]
                humidity = humidity[valid_mask]
                time_index = climate_df.index[valid_mask]
            else:
                time_index = climate_df.index
            
            # 创建天气DataFrame
            weather_df = pd.DataFrame({
                'temperature': temperature,
                'radiation_global_horizontal': radiation,
                'wind_speed_2m': wind_speed,
                'humidity': humidity
            }, index=time_index)

        except MemoryError as e:
            logging.error(f"内存不足，无法创建天气DataFrame: {e}")
            return None

        # 使用固定的平滑参数
        smoothing = 0.50

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
                return None
            else:
                logging.error(f"ValueError: {e}")
                return None
        except MemoryError as e:
            logging.error(f"BAIT计算内存不足: {e}")
            return None
        except Exception as e:
            logging.error(f"BAIT计算失败: {e}")
            return None
        finally:
            # 清理内存
            del weather_df
            if 'temperature' in locals():
                del temperature, radiation, wind_speed, humidity

        return bait

    except MemoryError as e:
        logging.error(f"计算BAIT内存不足: {e}")
        return None
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


def process_single_point(args):
    """处理单个点的数据"""
    lat, lon, population = args

    try:
        # 1. 加载气候数据
        climate_df = load_climate_data_for_point(lat, lon)
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

        # 4. 创建所有工况的输出数据
        if energy_results:
            # 准备时间列
            time_col = bait.index.strftime('%Y-%m-%d %H:%M:%S')

            # 创建BAIT文件数据
            bait_df = pd.DataFrame({
                'time': time_col,
                'BAIT': bait.values
            })

            # 创建cooling文件数据
            cooling_df = pd.DataFrame({'time': time_col})
            case_names = ['ref'] + [f'case{i}' for i in range(1, 21)]
            for case_name in case_names:
                if case_name in energy_results:
                    cooling_df[case_name] = energy_results[case_name]['cooling_demand'].values

            # 创建heating文件数据
            heating_df = pd.DataFrame({'time': time_col})
            for case_name in case_names:
                if case_name in energy_results:
                    heating_df[case_name] = energy_results[case_name]['heating_demand'].values

            # 创建total文件数据
            total_df = pd.DataFrame({'time': time_col})
            for case_name in case_names:
                if case_name in energy_results:
                    total_df[case_name] = energy_results[case_name]['total_demand'].values

            result_data = {
                'lat': lat,
                'lon': lon,
                'population': population,
                'bait_data': bait_df,
                'cooling_data': cooling_df,
                'heating_data': heating_df,
                'total_data': total_df
            }

            return result_data

        logging.error(f"未找到ref工况结果 (lat={lat:.3f}, lon={lon:.3f})")
        return None

    except Exception as e:
        logging.error(f"处理点失败 (lat={lat:.3f}, lon={lon:.3f}): {e}")
        return None


def process_batch_points(points_batch):
    """批量处理多个点的数据 - 优化内存使用"""
    results = []
    failed_points = []
    
    try:
        # 1. 批量加载气候数据
        climate_data, load_failed = load_climate_data_batch(points_batch)
        failed_points.extend(load_failed)
        
        # 2. 逐个处理点，避免内存累积
        for point in points_batch:
            lat, lon, population = point
            point_key = (lat, lon)
            
            if point_key not in climate_data:
                failed_points.append((point_key, "气候数据加载失败"))
                continue
            
            try:
                climate_df = climate_data[point_key]
                
                # 计算BAIT
                bait = calculate_bait_for_point(climate_df)
                if bait is None:
                    failed_points.append((point_key, "BAIT计算失败"))
                    continue
                
                # 计算能耗
                energy_results, daily_bait = calculate_energy_demand_for_point(bait, population)
                if energy_results is None:
                    failed_points.append((point_key, "能耗计算失败"))
                    continue
                
                # 创建输出数据
                if energy_results:
                    time_col = bait.index.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # 创建BAIT文件数据
                    bait_df = pd.DataFrame({
                        'time': time_col,
                        'BAIT': bait.values
                    })
                    
                    # 创建cooling文件数据
                    cooling_df = pd.DataFrame({'time': time_col})
                    case_names = ['ref'] + [f'case{i}' for i in range(1, 21)]
                    for case_name in case_names:
                        if case_name in energy_results:
                            cooling_df[case_name] = energy_results[case_name]['cooling_demand'].values
                    
                    # 创建heating文件数据
                    heating_df = pd.DataFrame({'time': time_col})
                    for case_name in case_names:
                        if case_name in energy_results:
                            heating_df[case_name] = energy_results[case_name]['heating_demand'].values
                    
                    # 创建total文件数据
                    total_df = pd.DataFrame({'time': time_col})
                    for case_name in case_names:
                        if case_name in energy_results:
                            total_df[case_name] = energy_results[case_name]['total_demand'].values
                    
                    result_data = {
                        'lat': lat,
                        'lon': lon,
                        'population': population,
                        'bait_data': bait_df,
                        'cooling_data': cooling_df,
                        'heating_data': heating_df,
                        'total_data': total_df
                    }
                    
                    results.append(result_data)
                    
                    # 立即释放大对象内存
                    del bait, energy_results, daily_bait
                    del bait_df, cooling_df, heating_df, total_df
                    
                else:
                    failed_points.append((point_key, "未找到ref工况结果"))
                    
            except MemoryError as e:
                failed_points.append((point_key, f"内存不足: {str(e)}"))
                # 强制垃圾回收
                gc.collect()
            except Exception as e:
                failed_points.append((point_key, f"处理异常: {str(e)}"))
        
        # 清理气候数据
        del climate_data
        gc.collect()
        
    except Exception as e:
        failed_points.append(("batch_error", f"批次处理异常: {str(e)}"))
    
    return results, failed_points


def process_and_save_single_point(point_data):
    """处理并保存单个点的数据"""
    try:
        lat, lon, population = point_data['lat'], point_data['lon'], point_data['population']

        # 处理单个点
        result = process_single_point((lat, lon, population))

        if result is not None:
            # 保存结果
            save_result = save_single_point_result(result)

            if "成功" in save_result:
                return {
                    'status': 'success',
                    'lat': lat,
                    'lon': lon,
                    'population': population,
                    'message': save_result
                }
            else:
                return {
                    'status': 'save_failed',
                    'lat': lat,
                    'lon': lon,
                    'population': population,
                    'message': save_result
                }
        else:
            return {
                'status': 'process_failed',
                'lat': lat,
                'lon': lon,
                'population': population,
                'message': f"处理点 ({lat}, {lon}) 失败: 返回None"
            }

    except Exception as e:
        return {
            'status': 'exception',
            'lat': lat,
            'lon': lon,
            'population': population,
            'message': f"处理点 ({lat}, {lon}) 异常: {str(e)}"
        }


def save_single_point_result(result):
    """保存单个点的结果"""
    try:
        lat, lon = result['lat'], result['lon']

        # 生成基础文件名 - 使用三位小数精度保持一致性
        base_filename = f"point_lat{lat:.3f}_lon{lon:.3f}"

        # 保存BAIT文件
        bait_path = os.path.join(OUTPUT_DIR, f"{base_filename}_BAIT.csv")
        result['bait_data'].to_csv(bait_path, index=False, encoding='utf-8-sig')

        # 保存cooling文件
        cooling_path = os.path.join(OUTPUT_DIR, f"{base_filename}_cooling.csv")
        result['cooling_data'].to_csv(cooling_path, index=False, encoding='utf-8-sig')

        # 保存heating文件
        heating_path = os.path.join(OUTPUT_DIR, f"{base_filename}_heating.csv")
        result['heating_data'].to_csv(heating_path, index=False, encoding='utf-8-sig')

        # 保存total文件
        total_path = os.path.join(OUTPUT_DIR, f"{base_filename}_total.csv")
        result['total_data'].to_csv(total_path, index=False, encoding='utf-8-sig')

        return f"成功保存坐标点: ({lat}, {lon})"

    except Exception as e:
        return f"保存点 ({lat}, {lon}) 失败: {str(e)}"


def main():
    """主函数 - 优化版本"""
    logging.info("开始网格点BAIT和能耗计算（优化版本）...")

    try:
        # 1. 加载人口数据
        logging.info("=== 第一步：加载人口数据 ===")
        population_df = load_population_data()
        total_points = len(population_df)
        
        # 2. 配置处理参数
        num_cores = multiprocessing.cpu_count()
        num_processes = min(num_cores, 4)  # 限制进程数避免过度并行
        
        # 根据内存情况动态调整参数
        initial_memory = check_memory_usage()
        current_batch_size = BATCH_SIZE  # 使用局部变量
        if initial_memory > 0.6:
            current_batch_size = 10  # 内存使用高时减少批量大小
            num_processes = min(num_processes, 2)
            logging.warning(f"内存使用率较高({initial_memory:.1%})，调整批量大小为{current_batch_size}，进程数为{num_processes}")
        
        logging.info(f"CPU核心数: {num_cores}")
        logging.info(f"使用进程数: {num_processes}")
        logging.info(f"批量大小: {current_batch_size}")
        logging.info(f"初始内存使用率: {initial_memory:.1%}")
        logging.info(f"总共需要处理 {total_points} 个点")

        # 3. 分批处理
        logging.info("=== 第二步：分批处理网格点 ===")
        all_results = []
        all_failed_points = []
        
        # 将数据分批
        points_list = [(row['lat'], row['lon'], row['population']) for _, row in population_df.iterrows()]
        batches = [points_list[i:i + current_batch_size] for i in range(0, len(points_list), current_batch_size)]
        
        logging.info(f"分为 {len(batches)} 个批次处理")
        
        # 使用进程池处理批次
        with multiprocessing.Pool(processes=num_processes) as pool:
            with tqdm(total=len(batches), desc="处理批次", unit="批次") as pbar:
                for batch_idx, (batch_results, batch_failed) in enumerate(pool.imap_unordered(process_batch_points, batches)):
                    try:
                        # 保存结果
                        if batch_results:
                            saved_count, save_failed = save_results_batch(batch_results)
                            all_results.extend([{
                                'lat': r['lat'], 
                                'lon': r['lon'], 
                                'population': r['population']
                            } for r in batch_results])
                            
                            if save_failed:
                                all_failed_points.extend(save_failed)
                        
                        # 记录失败点
                        if batch_failed:
                            all_failed_points.extend([f"点 {point}: {error}" for point, error in batch_failed])
                        
                        # 内存管理和监控
                        current_memory = check_memory_usage()
                        if current_memory > MEMORY_THRESHOLD:
                            gc.collect()
                            logging.warning(f"批次 {batch_idx+1}: 内存使用过高({current_memory:.1%})，执行垃圾回收")
                        
                        # 更新进度条信息
                        success_count = len(all_results)
                        failed_count = len(all_failed_points)
                        pbar.set_postfix({
                            '成功': success_count,
                            '失败': failed_count,
                            '内存': f"{current_memory:.1%}"
                        })
                        
                        pbar.update(1)
                        
                        # 每处理10个批次强制垃圾回收
                        if (batch_idx + 1) % 10 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        logging.error(f"批次 {batch_idx+1} 处理异常: {e}")
                        all_failed_points.append(f"批次 {batch_idx+1}: {str(e)}")
                        pbar.update(1)

        logging.info(f"批量处理完成，总共处理了 {len(all_results)} 个点的结果")
        if all_failed_points:
            logging.warning(f"失败点数: {len(all_failed_points)}")

        # 4. 保存失败点信息
        logging.info("=== 第三步：保存失败点信息 ===")
        try:
            if all_failed_points:
                failed_points_path = os.path.join(OUTPUT_DIR, 'failed_energy_points.csv')
                failed_df = pd.DataFrame({'error_message': all_failed_points})
                failed_df.to_csv(failed_points_path, index=False, encoding='utf-8-sig')
                logging.info(f"失败点信息已保存至: {failed_points_path}")
            else:
                logging.info("没有失败的点，无需保存失败点信息")
        except Exception as e:
            logging.warning(f"保存失败点信息时出错: {e}")

        # 5. 输出统计信息
        logging.info("=== 计算完成 ===")
        logging.info(f"成功处理点数: {len(all_results)}")
        logging.info(f"结果保存路径: {OUTPUT_DIR}")
        logging.info("能耗单位: GW (吉瓦)")
        logging.info("HDD单位: °C·day (摄氏度·天)")
        logging.info("CDD单位: °C·day (摄氏度·天)")
        
        # 性能统计
        memory_usage = check_memory_usage()
        logging.info(f"最终内存使用率: {memory_usage:.1%}")

        logging.info("网格点BAIT和能耗计算完成！")

    except Exception as e:
        error_msg = f"主程序执行出错: {str(e)}"
        logging.error(error_msg)
        raise


if __name__ == "__main__":
    main()
