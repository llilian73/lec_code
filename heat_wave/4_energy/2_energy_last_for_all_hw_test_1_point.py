"""
单点热浪期间能耗计算工具

功能：
读取热浪事件CSV文件，计算单个点（lat=35.0, lon=102.0）在2030年热浪期间的能耗。

输入数据：
1. 热浪事件CSV文件：Z:\local_environment_creation\heat_wave\output\output_2030\2030_heat_wave_lat35.000_lon102.000.csv
2. 气候数据：Z:\local_environment_creation\heat_wave\GCM_input_filter\
   文件名格式：{变量名}_day_BCC-CSM2-MR_ssp126_r1i1p1f1_gn_20300101-20341231_interpolated_1deg.nc
3. 人口数据：需要从人口CSV文件读取（如果存在）

输出数据：
- CSV文件，包含每个case的能耗结果
- 仅计算2030年的结果
"""

import pandas as pd
import numpy as np
import xarray as xr
import os
import sys
from datetime import datetime, timedelta
import logging
import cftime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from demand_ninja.core_p import _bait, demand_p as demand
from CandD.calculate import calculate_cases

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置路径
HEAT_WAVE_CSV = r"Z:\local_environment_creation\heat_wave\output\output_2030\2030_heat_wave_lat35.000_lon102.000.csv"
CLIMATE_BASE_PATH = r"Z:\local_environment_creation\heat_wave\GCM_input_filter"
OUTPUT_DIR = r"Z:\local_environment_creation\heat_wave\output\output_2030"

# 目标点坐标
TARGET_LAT = 35.0
TARGET_LON = 102.0
TARGET_YEAR = 2030

# 气候变量
CLIMATE_VARIABLES = ['tas', 'rsds', 'huss', 'uas', 'vas']

# 模型和SSP信息（用于构建文件名）
MODEL_NAME = "BCC-CSM2-MR"
SSP_PATH = "ssp126"

# 人口数据（如果CSV文件不存在，使用默认值）
DEFAULT_POPULATION = 100000  # 默认人口数


def convert_cftime_to_datetime(time_values):
    """
    将cftime对象转换为pandas DatetimeIndex（健壮方法）
    """
    try:
        # 方法1：尝试使用to_pandas或pd.to_datetime
        if hasattr(time_values, 'to_pandas'):
            time_index = time_values.to_pandas()
        else:
            time_index = pd.to_datetime(time_values)
        
        if not isinstance(time_index, pd.DatetimeIndex):
            time_index = pd.DatetimeIndex(time_index)
        return time_index
    except Exception as e:
        # 方法2：如果方法1失败，手动处理cftime对象
        try:
            import cftime
            if len(time_values) > 0 and isinstance(time_values[0], cftime.datetime):
                # 手动转换cftime对象为pandas Timestamp
                time_index = pd.to_datetime([pd.Timestamp(t.year, t.month, t.day) for t in time_values])
            else:
                time_index = pd.to_datetime(time_values)
            
            if not isinstance(time_index, pd.DatetimeIndex):
                time_index = pd.DatetimeIndex(time_index)
            return time_index
        except Exception as e2:
            # 方法3：最后尝试使用to_pandas（如果time_values是DataArray）
            try:
                if hasattr(time_values, 'to_pandas'):
                    time_index = time_values.to_pandas()
                    if not isinstance(time_index, pd.DatetimeIndex):
                        time_index = pd.DatetimeIndex(time_index)
                    return time_index
                else:
                    raise ValueError(f"无法转换时间索引: {e}, {e2}")
            except Exception:
                raise ValueError(f"无法转换时间索引: {e}, {e2}")


def load_climate_data_for_period(lat, lon, start_date, end_date):
    """
    从NetCDF文件加载指定时间段的气候数据
    
    参数:
        lat: 纬度
        lon: 经度
        start_date: 开始日期 (datetime)
        end_date: 结束日期 (datetime)
    
    返回:
        weather_df: DataFrame，包含温度、辐射、风速、湿度
    """
    # 构建文件路径
    file_pattern = "{var}_day_{model}_{ssp}_r1i1p1f1_gn_20300101-20341231_interpolated_1deg.nc"
    
    climate_data = {}
    
    # 加载每个变量的数据
    for var_name in CLIMATE_VARIABLES:
        file_path = os.path.join(CLIMATE_BASE_PATH, file_pattern.format(
            var=var_name, model=MODEL_NAME, ssp=SSP_PATH
        ))
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"气候数据文件不存在: {file_path}")
        
        logger.info(f"加载变量 {var_name} 从文件: {os.path.basename(file_path)}")
        
        with xr.open_dataset(file_path) as ds:
            # 找到最近的网格点
            lat_idx = np.abs(ds.lat.values - lat).argmin()
            lon_idx = np.abs(ds.lon.values - lon).argmin()
            
            # 打印实际提取的坐标（只在第一个变量时打印一次）
            if var_name == 'tas':
                actual_lat = ds.lat.values[lat_idx]
                actual_lon = ds.lon.values[lon_idx]
                logger.info(f"[TEST_1_POINT.PY] 目标坐标: lat={lat:.3f}, lon={lon:.3f}")
                logger.info(f"[TEST_1_POINT.PY] 实际提取的坐标: lat={actual_lat:.6f}, lon={actual_lon:.6f}, lat_idx={lat_idx}, lon_idx={lon_idx}")
            
            # 提取该点的数据
            point_data = ds[var_name].isel(lat=lat_idx, lon=lon_idx)
            
            # 处理时间坐标
            time_index = convert_cftime_to_datetime(point_data.time.values)
            
            # 筛选2030年的数据
            year_mask = time_index.year == TARGET_YEAR
            if not np.any(year_mask):
                raise ValueError(f"未找到 {TARGET_YEAR} 年的数据")
            
            point_data_2030 = point_data.isel(time=np.where(year_mask)[0])
            time_index_2030 = time_index[year_mask]
            
            # 筛选指定时间段的数据
            time_mask = (time_index_2030 >= start_date) & (time_index_2030 <= end_date)
            if not np.any(time_mask):
                raise ValueError(f"未找到 {start_date} 到 {end_date} 之间的数据")
            
            climate_data[var_name] = {
                'values': point_data_2030.isel(time=np.where(time_mask)[0]).values,
                'time': time_index_2030[time_mask]
            }
    
    # 创建DataFrame
    if len(climate_data['tas']['time']) == 0:
        raise ValueError(f"未找到 {start_date} 到 {end_date} 之间的数据")
    
    # 提取原始数据（转换前）
    tas_raw = climate_data['tas']['values']
    rsds_raw = climate_data['rsds']['values']
    uas_raw = climate_data['uas']['values']
    vas_raw = climate_data['vas']['values']
    huss_raw = climate_data['huss']['values']
    
    # 打印原始气候数据值
    logger.info(f"[TEST_1_POINT.PY] 提取的原始气候数据值（前5天）:")
    num_days = min(5, len(tas_raw))
    for i in range(num_days):
        logger.info(f"[TEST_1_POINT.PY]   第{i+1}天: tas={tas_raw[i]:.6f}K ({tas_raw[i]-273.15:.2f}°C), "
                   f"rsds={rsds_raw[i]:.6f}, uas={uas_raw[i]:.6f}, vas={vas_raw[i]:.6f}, "
                   f"huss={huss_raw[i]:.9f} ({huss_raw[i]*1000:.6f}g/kg)")
    logger.info(f"[TEST_1_POINT.PY] 总天数: {len(tas_raw)}")
    
    weather_df = pd.DataFrame({
        'temperature': tas_raw - 273.15,  # 转换为摄氏度
        'radiation_global_horizontal': rsds_raw,
        'wind_speed_2m': np.sqrt(
            uas_raw ** 2 +
            vas_raw ** 2
        ),
        'humidity': huss_raw * 1000  # 转换为g/kg
    }, index=climate_data['tas']['time'])
    
    return weather_df


def calculate_energy_demand(weather_df, population):
    """
    计算能耗需求
    
    参数:
        weather_df: 天气数据DataFrame
        population: 人口数
    
    返回:
        results: 字典，包含每个case的能耗结果
    """
    # Calculate BAIT
    bait = _bait(
        weather=weather_df,
        smoothing=0.73,
        solar_gains=0.014,
        wind_chill=-0.12,
        humidity_discomfort=0.036
    )

    # Get calculation cases
    base_params = {
        "heating_power": 27.93,
        "cooling_power": 48.55,
        "heating_threshold_people": 14,
        "cooling_threshold_people": 20,
        "base_power": 0,
        "population": population
    }

    cases = calculate_cases(base_params)
    
    # 打印case1的BAIT（在处理case1时）
    if 'case1' in cases:
        logger.info(f"[TEST_1_POINT.PY] case1 BAIT值 (前5天):")
        logger.info(f"[TEST_1_POINT.PY] {bait.head()}")
        logger.info(f"[TEST_1_POINT.PY] case1 BAIT统计: min={bait.min():.2f}, max={bait.max():.2f}, mean={bait.mean():.2f}")

    # Calculate results for each case
    results = {}
    for case_name, params in cases.items():
        # 确保bait的时间索引是每日数据
        daily_bait = bait.copy()
        daily_bait.index = pd.date_range(
            start=daily_bait.index[0],
            end=daily_bait.index[-1],
            freq='D'
        )

        results[case_name] = demand(
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
            use_diurnal_profile=True,
            raw=True
        )

    return results


def parse_heatwave_date(date_str, year):
    """
    解析热浪日期字符串（格式：month/day）为datetime对象
    
    参数:
        date_str: 日期字符串，格式为 "month/day"
        year: 年份
    
    返回:
        datetime对象
    """
    month, day = map(int, date_str.split('/'))
    return datetime(year, month, day)


def load_population(lat, lon, population_file=None):
    """
    加载人口数据
    
    参数:
        lat: 纬度
        lon: 经度
        population_file: 人口CSV文件路径（可选）
    
    返回:
        人口数
    """
    if population_file and os.path.exists(population_file):
        try:
            pop_df = pd.read_csv(population_file)
            # 查找最近的点
            pop_df['dist'] = np.sqrt(
                (pop_df['lat'] - lat) ** 2 + (pop_df['lon'] - lon) ** 2
            )
            nearest = pop_df.loc[pop_df['dist'].idxmin()]
            return nearest['population']
        except Exception as e:
            logger.warning(f"无法从人口文件读取数据: {e}，使用默认值")
    
    logger.info(f"使用默认人口数: {DEFAULT_POPULATION}")
    return DEFAULT_POPULATION


def main():
    """主函数"""
    logger.info("=== 开始处理单点热浪能耗计算 ===")
    
    # 检查输入文件是否存在
    if not os.path.exists(HEAT_WAVE_CSV):
        raise FileNotFoundError(f"热浪CSV文件不存在: {HEAT_WAVE_CSV}")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 读取热浪数据
    logger.info(f"读取热浪数据: {HEAT_WAVE_CSV}")
    heatwave_df = pd.read_csv(HEAT_WAVE_CSV)
    
    # 只保留持续时间>=3天的热浪
    heatwave_df = heatwave_df[heatwave_df['Duration'] >= 3]
    
    if len(heatwave_df) == 0:
        logger.warning("未找到符合条件的热浪事件（持续时间>=3天）")
        return
    
    logger.info(f"找到 {len(heatwave_df)} 个热浪事件")
    
    # 加载人口数据
    population = load_population(TARGET_LAT, TARGET_LON)
    logger.info(f"使用人口数: {population}")
    
    # 初始化总能耗结果（动态创建所有case，支持20个case：ref + case1-20）
    # 先获取所有case名称
    base_params = {
        "heating_power": 27.93,
        "cooling_power": 48.55,
        "heating_threshold_people": 14,
        "cooling_threshold_people": 20,
        "base_power": 0,
        "population": population
    }
    cases = calculate_cases(base_params)
    
    # 初始化所有case的结果字典
    total_results = {}
    for case_name in cases.keys():
        total_results[case_name] = {
            'total_demand': 0,
            'heating_demand': 0,
            'cooling_demand': 0
        }
    
    logger.info(f"初始化了 {len(total_results)} 个case: {list(total_results.keys())}")
    
    # 处理每个热浪事件
    for idx, row in heatwave_df.iterrows():
        try:
            # 解析热浪日期
            start_date = parse_heatwave_date(row['date'], TARGET_YEAR)
            duration = int(row['Duration'])
            end_date = start_date + timedelta(days=duration - 1)
            
            logger.info(f"\n处理热浪事件 {row['number']}: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} (持续 {duration} 天)")
            
            # 加载气候数据
            weather_df = load_climate_data_for_period(TARGET_LAT, TARGET_LON, start_date, end_date)
            logger.info(f"加载了 {len(weather_df)} 天的气候数据")
            
            # 计算能耗
            results = calculate_energy_demand(weather_df, population)
            
            # 累加结果（动态处理所有case）
            for case_name, case_result in results.items():
                # 如果case不存在，则创建它（以防万一）
                if case_name not in total_results:
                    total_results[case_name] = {
                        'total_demand': 0,
                        'heating_demand': 0,
                        'cooling_demand': 0
                    }
                total_results[case_name]['total_demand'] += case_result['total_demand'].sum()
                total_results[case_name]['heating_demand'] += case_result['heating_demand'].sum()
                total_results[case_name]['cooling_demand'] += case_result['cooling_demand'].sum()
            
            logger.info(f"热浪事件 {row['number']} 处理完成")
            
        except Exception as e:
            logger.error(f"处理热浪事件 {row['number']} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果
    logger.info("\n保存结果...")
    for case_name, case_result in total_results.items():
        # 创建case文件夹
        case_dir = os.path.join(OUTPUT_DIR, case_name)
        os.makedirs(case_dir, exist_ok=True)
        
        # 准备CSV数据
        csv_data = {
            'lat': [TARGET_LAT],
            'lon': [TARGET_LON],
            'total_demand': [case_result['total_demand']],
            'heating_demand': [case_result['heating_demand']],
            'cooling_demand': [case_result['cooling_demand']]
        }
        csv_df = pd.DataFrame(csv_data)
        csv_file = os.path.join(case_dir, f"{case_name}_lat{TARGET_LAT}_lon{TARGET_LON}.csv")
        
        csv_df.to_csv(csv_file, index=False)
        logger.info(f"保存 {case_name} 结果到: {csv_file}")
    
    logger.info("\n=== 处理完成 ===")
    logger.info(f"总热浪事件数: {len(heatwave_df)}")
    logger.info(f"总能耗结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
