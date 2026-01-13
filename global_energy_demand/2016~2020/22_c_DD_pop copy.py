"""
全球网格点BAIT和能耗计算工具 - 失败点重新处理版本

功能概述：
本工具专门用于重新计算之前处理失败的网格点的建筑适应性室内温度（BAIT）和建筑能耗需求。
从失败点CSV文件中提取坐标信息，重新进行BAIT和能耗计算，并保存到对应的输出目录。

主要功能：
1. 从失败点CSV文件中提取坐标信息
2. 重新计算失败点的BAIT和能耗需求
3. 保存结果到对应的年份目录
4. 记录仍然失败的点信息

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
import signal
import atexit

# 将项目的根目录加入到 sys.path
# 当前文件在 global_energy_demand/grid/ 下，需要往上三层到达项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from demand_ninja.core_p import _bait, demand_p as demand
from CandD.calculate import calculate_cases

# 设置日志记录
def setup_logging():
    """设置日志记录，避免控制台输出流错误"""
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件处理器
    file_handler = logging.FileHandler('grid_point_calculation.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器，使用更安全的方式
    try:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
    except Exception:
        # 如果控制台输出有问题，只使用文件日志
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

def safe_log(level, message, *args, **kwargs):
    """安全的日志记录函数，避免输出流错误"""
    try:
        if level == 'info':
            logger.info(message, *args, **kwargs)
        elif level == 'warning':
            logger.warning(message, *args, **kwargs)
        elif level == 'error':
            logger.error(message, *args, **kwargs)
        elif level == 'debug':
            logger.debug(message, *args, **kwargs)
    except (OSError, IOError) as e:
        # 如果日志输出失败，尝试重新设置日志
        try:
            setup_logging()
            if level == 'info':
                logger.info(message, *args, **kwargs)
            elif level == 'warning':
                logger.warning(message, *args, **kwargs)
            elif level == 'error':
                logger.error(message, *args, **kwargs)
            elif level == 'debug':
                logger.debug(message, *args, **kwargs)
        except:
            # 如果仍然失败，至少打印到stderr
            print(f"日志记录失败: {message}", file=sys.stderr)

def log_progress(current, total, prefix="进度", suffix="", decimals=1, bar_length=50):
    """显示进度条，减少频繁的日志输出"""
    if current == 0:
        return
    
    # 每1000个点或每10%进度才输出一次
    if current % 1000 != 0 and current % max(1, total // 10) != 0:
        return
    
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    message = f'\r{prefix} |{bar}| {percent}% {suffix} ({current}/{total})'
    safe_log("info", message)

def extract_coordinates_from_error(error_message):
    """从错误信息中提取坐标"""
    import re
    try:
        # 匹配格式: "保存点 (54.25, 15.312) 失败: [Errno 22] Invalid argument"
        # 支持正坐标和负坐标，使用 -? 来匹配可选的负号
        pattern = r'保存点 \((-?\d+\.?\d*), (-?\d+\.?\d*)\) 失败'
        match = re.search(pattern, error_message)
        if match:
            lat = float(match.group(1))
            lon = float(match.group(2))
            return lat, lon
        return None, None
    except Exception as e:
        safe_log("error", f"提取坐标失败: {error_message}, 错误: {e}")
        return None, None


def load_failed_points_from_csv(failed_points_path):
    """从失败点CSV文件中加载失败点的坐标"""
    try:
        if not os.path.exists(failed_points_path):
            safe_log("warning", f"失败点文件不存在: {failed_points_path}")
            return []
        
        failed_df = pd.read_csv(failed_points_path)
        failed_points = []
        
        for _, row in failed_df.iterrows():
            error_message = row['error_message']
            lat, lon = extract_coordinates_from_error(error_message)
            if lat is not None and lon is not None:
                failed_points.append({
                    'lat': lat,
                    'lon': lon,
                    'error_message': error_message
                })
            else:
                safe_log("warning", f"无法提取坐标: {error_message}")
        
        safe_log("info", f"从 {failed_points_path} 加载了 {len(failed_points)} 个失败点")
        return failed_points
        
    except Exception as e:
        safe_log("error", f"加载失败点文件失败: {failed_points_path}, 错误: {e}")
        return []

def get_population_for_point(lat, lon, population_df):
    """根据坐标获取人口数据"""
    try:
        # 查找匹配的坐标点（考虑浮点数精度）
        tolerance = 0.001  # 0.001度的容差
        for _, row in population_df.iterrows():
            if (abs(row['lat'] - lat) < tolerance and 
                abs(row['lon'] - lon) < tolerance):
                return row['population']
        
        safe_log("warning", f"未找到坐标 ({lat}, {lon}) 对应的人口数据")
        return None
        
    except Exception as e:
        safe_log("error", f"获取人口数据失败 ({lat}, {lon}): {e}")
        return None

# 配置参数
DATA_BASE_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\data"
OUTPUT_BASE_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\result_half"
# 只处理有失败点的年份
FAILED_YEARS = [2017, 2018]

# 确保输出目录存在
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# 全局变量用于跟踪程序状态
program_running = True

def cleanup_and_exit(signum=None, frame=None):
    """清理资源并退出程序"""
    global program_running
    program_running = False
    safe_log("info", "收到退出信号，正在清理资源...")
    safe_log("info", "程序已安全退出")
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)
atexit.register(cleanup_and_exit)


def load_population_data():
    """加载人口数据"""
    safe_log("info", "开始加载人口数据...")

    population_csv_path = os.path.join(DATA_BASE_DIR, 'population_points.csv')
    if not os.path.exists(population_csv_path):
        raise FileNotFoundError(f"人口数据文件不存在: {population_csv_path}")

    population_df = pd.read_csv(population_csv_path)
    safe_log("info", f"加载人口数据完成，共 {len(population_df)} 个点")

    return population_df


def load_climate_data_for_point(lat, lon, data_dir):
    """加载单个点的气候数据"""
    try:
        # 生成文件名 - 使用三位小数精度
        filename = f"point_lat{lat:.3f}_lon{lon:.3f}_climate.csv"
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            safe_log("error", f"文件不存在: {file_path}")
            return None

        # 读取CSV文件，使用python引擎避免tokenizing错误
        try:
            df = pd.read_csv(file_path, engine='python')
        except Exception as e:
            # 如果python引擎失败，尝试c引擎
            df = pd.read_csv(file_path, engine='c')

        # 检查数据完整性
        if len(df) == 0:
            safe_log("error", "文件为空")
            return None

        # 检查必需的列是否存在
        required_columns = ['time', 'T2M', 'U2M', 'V2M', 'QV2M', 'SWGDN']
        for col in required_columns:
            if col not in df.columns:
                safe_log("error", f"缺少必需的列: {col}")
                return None

        # 删除包含NaN的行
        df = df.dropna(subset=required_columns)
        if len(df) == 0:
            safe_log("error", "删除NaN后数据为空")
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
            safe_log("error", "最终处理后数据为空")
            return None

        # 设置时间索引
        df.set_index('time', inplace=True)

        # 验证所有列的长度一致
        lengths = [len(df[col]) for col in ['T2M', 'SWGDN', 'wind_speed_2m', 'QV2M']]
        if len(set(lengths)) > 1:
            safe_log("error", f"列长度不一致: {lengths}")
            return None

        # 检查时间索引是否连续（静默处理）
        time_diff = df.index.to_series().diff().dropna()
        if len(time_diff) > 0:
            # 如果时间间隔不一致，重新采样到统一频率
            if time_diff.std() > pd.Timedelta(seconds=1):
                df = df.resample('1H').mean().interpolate(method='linear')

        return df

    except Exception as e:
        safe_log("error", f"加载气候数据失败 (lat={lat}, lon={lon}): {e}")
        return None


def calculate_bait_for_point(climate_df):
    """计算单个点的BAIT"""
    try:
        # 检查数据完整性
        required_columns = ['T2M', 'SWGDN', 'wind_speed_2m', 'QV2M']
        for col in required_columns:
            if col not in climate_df.columns:
                safe_log("error", f"缺少必需的列: {col}")
                return None

        # 检查数据长度一致性
        data_length = len(climate_df)
        if data_length == 0:
            safe_log("error", "气候数据为空")
            return None

        # 验证所有列的长度一致
        lengths = [len(climate_df[col]) for col in required_columns]
        if len(set(lengths)) > 1:
            safe_log("error", f"列长度不一致: {lengths}")
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
                safe_log("error", "插值后数据为空")
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
                safe_log("error", f"数组长度不匹配错误: {e}")
                safe_log("error", f"原始数据形状: {weather_df.shape}")
                safe_log("error", f"各列长度: {[len(weather_df[col]) for col in weather_df.columns]}")

                # 尝试修复数据
                weather_df_clean = weather_df.dropna()
                safe_log("info", f"清理后数据形状: {weather_df_clean.shape}")

                if len(weather_df_clean) > 0:
                    bait = _bait(
                        weather=weather_df_clean,
                        smoothing=smoothing,
                        solar_gains=0.014,
                        wind_chill=-0.12,
                        humidity_discomfort=0.036
                    )
                else:
                    safe_log("error", "清理后数据为空")
                    return None
            else:
                safe_log("error", f"ValueError: {e}")
                return None
        except Exception as e:
            safe_log("error", f"BAIT计算失败: {e}")
            safe_log("error", f"错误类型: {type(e)}")
            return None

        return bait

    except Exception as e:
        safe_log("error", f"计算BAIT失败: {e}")
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
        safe_log("error", f"计算能耗需求失败: {e}")
        return None, None


def process_single_point(args):
    """处理单个点的数据"""
    lat, lon, population, data_dir, output_dir = args

    try:
        # 1. 加载气候数据
        climate_df = load_climate_data_for_point(lat, lon, data_dir)
        if climate_df is None:
            safe_log("error", f"气候数据加载失败 (lat={lat:.3f}, lon={lon:.3f})")
            return None

        # 2. 计算BAIT
        bait = calculate_bait_for_point(climate_df)
        if bait is None:
            safe_log("error", f"BAIT计算失败 (lat={lat:.3f}, lon={lon:.3f})")
            return None

        # 3. 计算能耗
        energy_results, daily_bait = calculate_energy_demand_for_point(bait, population)
        if energy_results is None:
            safe_log("error", f"能耗计算失败 (lat={lat:.3f}, lon={lon:.3f})")
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

        safe_log("error", f"未找到ref工况结果 (lat={lat:.3f}, lon={lon:.3f})")
        return None

    except Exception as e:
        safe_log("error", f"处理点失败 (lat={lat:.3f}, lon={lon:.3f}): {e}")
        return None


def process_and_save_single_point(point_data):
    """处理并保存单个点的数据"""
    try:
        lat, lon, population, data_dir, output_dir = point_data['lat'], point_data['lon'], point_data['population'], \
                                                     point_data['data_dir'], point_data['output_dir']

        # 处理单个点
        result = process_single_point((lat, lon, population, data_dir, output_dir))

        if result is not None:
            # 保存结果
            save_result = save_single_point_result(result, output_dir)

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
        # 记录详细的异常信息
        import traceback
        error_details = traceback.format_exc()
        safe_log("error", f"处理点 ({lat}, {lon}) 发生异常: {str(e)}")
        safe_log("error", f"异常详情: {error_details}")
        
        return {
            'status': 'exception',
            'lat': lat,
            'lon': lon,
            'population': population,
            'message': f"处理点 ({lat}, {lon}) 异常: {str(e)}"
        }


def save_single_point_result(result, output_dir):
    """保存单个点的结果"""
    try:
        lat, lon = result['lat'], result['lon']

        # 生成基础文件名 - 使用三位小数精度保持一致性
        base_filename = f"point_lat{lat:.3f}_lon{lon:.3f}"

        # 保存BAIT文件
        bait_path = os.path.join(output_dir, f"{base_filename}_BAIT.csv")
        result['bait_data'].to_csv(bait_path, index=False, encoding='utf-8-sig')

        # 保存cooling文件
        cooling_path = os.path.join(output_dir, f"{base_filename}_cooling.csv")
        result['cooling_data'].to_csv(cooling_path, index=False, encoding='utf-8-sig')

        # 保存heating文件
        heating_path = os.path.join(output_dir, f"{base_filename}_heating.csv")
        result['heating_data'].to_csv(heating_path, index=False, encoding='utf-8-sig')

        # 保存total文件
        total_path = os.path.join(output_dir, f"{base_filename}_total.csv")
        result['total_data'].to_csv(total_path, index=False, encoding='utf-8-sig')

        return f"成功保存坐标点: ({lat}, {lon})"

    except Exception as e:
        return f"保存点 ({lat}, {lon}) 失败: {str(e)}"


def main():
    """主函数 - 专门处理失败的点"""
    safe_log("info", "开始重新计算失败点的BAIT和能耗...")

    try:
        # 1. 加载人口数据
        safe_log("info", "=== 第一步：加载人口数据 ===")
        population_df = load_population_data()

        # 2. 配置并行处理参数
        num_cores = multiprocessing.cpu_count()
        num_processes = min(num_cores * 2, 16)  # 限制最大进程数

        safe_log("info", f"CPU核心数: {num_cores}")
        safe_log("info", f"使用进程数: {num_processes}")

        # 3. 定义需要处理的年份和对应的失败点文件
        failed_years = FAILED_YEARS  # 只处理有失败点的年份
        
        total_failed_points = 0
        all_failed_points = []

        # 4. 收集所有失败点
        for year in failed_years:
            failed_points_path = os.path.join(OUTPUT_BASE_DIR, str(year), f'failed_energy_points_{year}.csv')
            failed_points = load_failed_points_from_csv(failed_points_path)
            
            if failed_points:
                # 为每个失败点添加年份信息
                for point in failed_points:
                    point['year'] = year
                    all_failed_points.append(point)
                total_failed_points += len(failed_points)
                safe_log("info", f"{year}年有 {len(failed_points)} 个失败点")

        safe_log("info", f"总共需要重新处理 {total_failed_points} 个失败点")

        if total_failed_points == 0:
            safe_log("info", "没有找到需要重新处理的失败点")
            return

        # 5. 处理每个年份的失败点
        for year in failed_years:
            # 检查程序是否仍在运行
            if not program_running:
                safe_log("info", "程序收到退出信号，停止处理")
                break

            # 获取该年份的失败点
            year_failed_points = [point for point in all_failed_points if point['year'] == year]
            
            if not year_failed_points:
                safe_log("info", f"{year}年没有失败点需要处理，跳过")
                continue

            safe_log("info", f"=== 处理{year}年失败点 ===")

            # 创建年份输入和输出目录
            year_data_dir = os.path.join(DATA_BASE_DIR, str(year))
            year_output_dir = os.path.join(OUTPUT_BASE_DIR, str(year))
            os.makedirs(year_output_dir, exist_ok=True)

            # 检查输入目录是否存在
            if not os.path.exists(year_data_dir):
                safe_log("warning", f"{year}年数据目录不存在: {year_data_dir}，跳过")
                continue

            # 准备处理点列表
            point_list = []
            for failed_point in year_failed_points:
                lat, lon = failed_point['lat'], failed_point['lon']
                population = get_population_for_point(lat, lon, population_df)
                
                if population is not None:
                    point_list.append({
                        'lat': lat,
                        'lon': lon,
                        'population': population,
                        'data_dir': year_data_dir,
                        'output_dir': year_output_dir
                    })
                else:
                    safe_log("warning", f"跳过点 ({lat}, {lon})，未找到人口数据")

            if not point_list:
                safe_log("warning", f"{year}年没有有效的失败点需要处理")
                continue

            safe_log("info", f"开始处理{year}年的 {len(point_list)} 个失败点")

            # 并行处理失败点
            all_results = []
            still_failed_points = []

            with multiprocessing.Pool(processes=num_processes) as pool:
                # 计算合适的chunksize
                chunksize = max(1, len(point_list) // (num_processes * 4))
                safe_log("info", f"chunksize: {chunksize}")

                with tqdm(total=len(point_list), desc=f"重新处理{year}年失败点") as pbar:
                    processed_count = 0
                    for result in pool.imap_unordered(process_and_save_single_point, point_list, chunksize=chunksize):
                        processed_count += 1
                        
                        if result['status'] == 'success':
                            all_results.append({
                                'lat': result['lat'],
                                'lon': result['lon'],
                                'population': result['population']
                            })
                        else:
                            still_failed_points.append(result['message'])

                        pbar.update(1)
                        
                        # 每处理100个点报告一次进度
                        if processed_count % 100 == 0 or processed_count % max(1, len(point_list) // 10) == 0:
                            log_progress(processed_count, len(point_list), f"重新处理{year}年", f"成功:{len(all_results)} 仍失败:{len(still_failed_points)}")

            safe_log("info", f"{year}年重新处理完成，成功处理了 {len(all_results)} 个点")
            if still_failed_points:
                safe_log("warning", f"{year}年仍有 {len(still_failed_points)} 个点处理失败")

            # 保存仍然失败的点信息
            try:
                if still_failed_points:
                    still_failed_path = os.path.join(year_output_dir, f'still_failed_energy_points_{year}.csv')
                    still_failed_df = pd.DataFrame({'error_message': still_failed_points})
                    still_failed_df.to_csv(still_failed_path, index=False, encoding='utf-8-sig')
                    safe_log("info", f"{year}年仍然失败的点信息已保存至: {still_failed_path}")
                else:
                    safe_log("info", f"{year}年所有失败点都已成功重新处理")
            except Exception as e:
                safe_log("warning", f"保存{year}年仍然失败的点信息时出错: {e}")

        # 输出统计信息
        safe_log("info", "=== 所有失败点重新处理完成 ===")
        safe_log("info", f"结果保存路径: {OUTPUT_BASE_DIR}")
        safe_log("info", "能耗单位: GW (吉瓦)")
        safe_log("info", "HDD单位: °C·day (摄氏度·天)")
        safe_log("info", "CDD单位: °C·day (摄氏度·天)")

        safe_log("info", "所有失败点的BAIT和能耗重新计算完成！")

    except Exception as e:
        error_msg = f"主程序执行出错: {str(e)}"
        safe_log("error", error_msg)
        raise


if __name__ == "__main__":
    main()
