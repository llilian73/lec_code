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


def save_spatial_blocks(block_data_dict, year, output_base_dir, append_mode=False):
    """保存空间块数据为Parquet文件，支持追加模式"""
    import shutil
    import time
    
    year_dir = os.path.join(output_base_dir, f'{year}')
    os.makedirs(year_dir, exist_ok=True)
    
    saved_files = []
    for block_name, block_df in block_data_dict.items():
        if block_df is None or len(block_df) == 0:
            continue
        
        try:
            # 生成文件路径
            parquet_filename = f"{block_name}.parquet"
            parquet_path = os.path.join(year_dir, parquet_filename)
            
            if append_mode and os.path.exists(parquet_path):
                # 追加模式：读取现有文件，合并数据，然后保存
                backup_path = None
                temp_path = None
                existing_df = None
                combined_df = None
                # 保存 block_df 的引用，以便在异常处理中使用
                block_df_backup = block_df
                try:
                    # 读取现有文件
                    existing_df = pd.read_parquet(parquet_path)
                    # 合并数据
                    combined_df = pd.concat([existing_df, block_df], ignore_index=True)
                    
                    # 立即释放 existing_df 和 block_df 的内存
                    del existing_df
                    existing_df = None
                    del block_df
                    block_df = None
                    
                    # 先保存到临时文件，确保数据完整性
                    temp_path = parquet_path + f".tmp_{int(time.time())}"
                    combined_df.to_parquet(temp_path, compression='snappy', index=False)
                    
                    # 保存后立即释放 combined_df 的内存
                    del combined_df
                    combined_df = None
                    
                    # 备份原文件
                    backup_path = parquet_path + ".bak"
                    shutil.copy2(parquet_path, backup_path)
                    
                    # 用临时文件替换原文件
                    shutil.move(temp_path, parquet_path)
                    
                    # 删除备份文件（保存成功）
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    
                    # 强制垃圾回收
                    gc.collect()
                    
                    logging.info(f"追加保存空间块 {block_name}: 新增数据已保存")
                    
                except Exception as e:
                    # 追加失败时的处理
                    error_msg = str(e)
                    logging.error(f"追加空间块 {block_name} 失败: {error_msg}")
                    
                    # 释放可能存在的 DataFrame 内存
                    if existing_df is not None:
                        del existing_df
                        existing_df = None
                    if combined_df is not None:
                        del combined_df
                        combined_df = None
                    
                    # 如果临时文件存在，删除它
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                    
                    # 如果备份文件存在，恢复原文件
                    if backup_path and os.path.exists(backup_path):
                        try:
                            if os.path.exists(parquet_path):
                                os.remove(parquet_path)
                            shutil.move(backup_path, parquet_path)
                            logging.warning(f"已恢复空间块 {block_name} 的原始文件，新数据未保存")
                        except Exception as restore_error:
                            logging.error(f"恢复空间块 {block_name} 的原始文件失败: {str(restore_error)}")
                    
                    # 将新数据保存到单独的临时文件，避免数据丢失
                    # 使用备份的 block_df 引用
                    if block_df_backup is not None:
                        try:
                            timestamp = int(time.time())
                            temp_save_path = os.path.join(year_dir, f"{block_name}_failed_append_{timestamp}.parquet")
                            block_df_backup.to_parquet(temp_save_path, compression='snappy', index=False)
                            logging.warning(f"追加失败的数据已保存到临时文件: {temp_save_path}，需要手动合并")
                            # 保存后释放内存
                            del block_df_backup
                            block_df_backup = None
                        except Exception as save_error:
                            logging.error(f"保存追加失败的数据到临时文件也失败: {str(save_error)}")
                            # 即使保存失败也要释放内存
                            if block_df_backup is not None:
                                del block_df_backup
                                block_df_backup = None
                    
                    # 强制垃圾回收
                    gc.collect()
                    
                    # 不将失败的文件添加到saved_files
                    continue
            else:
                # 新建或覆盖保存（非追加模式）
                block_df.to_parquet(parquet_path, compression='snappy', index=False)
                logging.info(f"保存空间块 {block_name}: {len(block_df)} 行数据到 {parquet_path}")
                # 保存后立即释放内存
                del block_df
            
            saved_files.append(parquet_path)
            
            # 强制垃圾回收（每个文件保存后）
            gc.collect()
            
        except Exception as e:
            logging.error(f"保存空间块 {block_name} 失败: {str(e)}")
    
    return saved_files


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

            # 并行处理
            logging.info(f"=== 并行处理{year}年网格点 ===")
            failed_points = []
            
            # 按空间块组织数据
            block_data_dict = defaultdict(list)
            
            # 定期保存参数
            SAVE_INTERVAL = 100  # 每处理1000个点保存一次
            processed_count = 0
            total_saved_points = 0

            # 将DataFrame转换为字典列表，便于并行处理
            point_list = []
            for _, row in population_df.iterrows():
                point_list.append({
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'population': row['population'],
                    'data_dir': year_data_dir
                })

            with multiprocessing.Pool(processes=num_processes) as pool:
                # 计算合适的chunksize
                chunksize = max(1, total_points // (num_processes * 40))
                logging.info(f"chunksize: {chunksize}")

                with tqdm(total=total_points, desc=f"处理{year}年网格点") as pbar:
                    for result in pool.imap_unordered(process_single_point_for_parquet, point_list, chunksize=chunksize):
                        processed_count += 1
                        
                        if result['status'] == 'success' and result['data'] is not None:
                            # 获取空间块标识
                            block_name = get_spatial_block(result['lat'], result['lon'], BLOCK_SIZE)
                            # 将数据添加到对应的空间块
                            block_data_dict[block_name].append(result['data'])
                            # 注意：这里不删除 result['data']，因为还需要在合并时使用
                        else:
                            failed_points.append(result.get('message', f"处理点 ({result['lat']}, {result['lon']}) 失败"))
                            # 如果处理失败，确保释放 result 中的 data（如果存在）
                            if 'data' in result and result['data'] is not None:
                                del result['data']

                        pbar.update(1)
                        
                        # 定期保存数据，释放内存
                        if processed_count % SAVE_INTERVAL == 0:
                            logging.info(f"定期保存：已处理 {processed_count}/{total_points} 个点，开始保存数据...")
                            
                            # 合并当前收集的数据
                            temp_merged_data = {}
                            for block_name, dataframes in list(block_data_dict.items()):
                                if len(dataframes) > 0:
                                    try:
                                        merged_df = pd.concat(dataframes, ignore_index=True)
                                        temp_merged_data[block_name] = merged_df
                                        # 立即释放原始 DataFrame 列表中的每个 DataFrame
                                        for df in dataframes:
                                            del df
                                        dataframes.clear()
                                        # 删除整个列表引用
                                        del block_data_dict[block_name]
                                    except Exception as e:
                                        logging.error(f"合并空间块 {block_name} 数据失败: {str(e)}")
                                        # 即使合并失败，也要释放内存
                                        for df in dataframes:
                                            del df
                                        dataframes.clear()
                                        del block_data_dict[block_name]
                            
                            # 保存数据（追加模式）
                            if temp_merged_data:
                                saved_files = save_spatial_blocks(temp_merged_data, year, OUTPUT_BASE_DIR, append_mode=True)
                                saved_count = sum(len(df) for df in temp_merged_data.values())
                                total_saved_points += saved_count
                                logging.info(f"定期保存完成：保存了 {len(saved_files)} 个空间块，共 {saved_count} 行数据")
                                
                                # 显式删除所有合并后的 DataFrame，释放内存
                                for block_name in list(temp_merged_data.keys()):
                                    del temp_merged_data[block_name]
                                temp_merged_data.clear()
                            
                            # 清空已保存的数据，释放内存
                            block_data_dict.clear()
                            
                            # 强制垃圾回收，释放内存
                            gc.collect()
                            
                            logging.info(f"内存已释放，继续处理剩余 {total_points - processed_count} 个点...")

            # 保存剩余的数据
            logging.info(f"{year}年数据处理完成，开始保存剩余数据...")
            remaining_merged_data = {}
            for block_name, dataframes in list(block_data_dict.items()):
                if len(dataframes) > 0:
                    try:
                        merged_df = pd.concat(dataframes, ignore_index=True)
                        remaining_merged_data[block_name] = merged_df
                        # 立即释放原始 DataFrame 列表中的每个 DataFrame
                        for df in dataframes:
                            del df
                        dataframes.clear()
                        # 删除整个列表引用
                        del block_data_dict[block_name]
                        logging.info(f"空间块 {block_name}: 合并了数据，共 {len(merged_df)} 行")
                    except Exception as e:
                        logging.error(f"合并空间块 {block_name} 数据失败: {str(e)}")
                        failed_points.append(f"合并空间块 {block_name} 失败: {str(e)}")
                        # 即使合并失败，也要释放内存
                        for df in dataframes:
                            del df
                        dataframes.clear()
                        del block_data_dict[block_name]

            # 保存剩余的空间块数据为Parquet文件（追加模式）
            if remaining_merged_data:
                saved_files = save_spatial_blocks(remaining_merged_data, year, OUTPUT_BASE_DIR, append_mode=True)
                remaining_saved_count = sum(len(df) for df in remaining_merged_data.values())
                total_saved_points += remaining_saved_count
                logging.info(f"{year}年成功保存剩余 {len(saved_files)} 个Parquet文件，共 {remaining_saved_count} 行数据")
                
                # 显式删除所有合并后的 DataFrame，释放内存
                for block_name in list(remaining_merged_data.keys()):
                    del remaining_merged_data[block_name]
                remaining_merged_data.clear()
                
                # 强制垃圾回收
                gc.collect()
            
            logging.info(f"{year}年总共保存了 {total_saved_points} 行数据")
            
            # 计算成功处理的点数（每个点有8760行数据，假设是全年数据）
            # 如果数据行数不是8760，则使用实际保存的行数除以平均每个点的行数
            success_count = processed_count - len(failed_points)
            logging.info(f"{year}年并行处理完成，成功处理了 {success_count} 个点")
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
