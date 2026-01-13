import pandas as pd
import numpy as np
import xarray as xr
import os
from pathlib import Path
import sys
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import gc
import logging
from multiprocessing import shared_memory

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from demand_ninja.core_p import _bait, demand_p as demand
from CandD.calculate import calculate_cases
# os.environ['PROJ_LIB'] = r'C:\Program Files\QGIS 3.36.0\apps\Python39\Lib\site-packages\rasterio\proj_data'
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 设置单个点的经纬度
TARGET_LAT = 35.000
TARGET_LON = 102.000

CHINA_BOUNDS = {
    'lat_min': -90.0,
    'lat_max': 90.0,
    'lon_min': 0.0,
    'lon_max': 180.0
}

# 设置气候数据路径
climate_base_path = r"Z:\local_environment_creation\heat_wave\GCM_input_filter"

# 设置输入点文件路径（CSV）
point_file = r"Z:\local_environment_creation\heat_wave\output\output_2030\2030_heat_wave_lat35.000_lon102.000.csv"

# 设置输出路径
output_dir = r"Z:\local_environment_creation\heat_wave\output\energy_output_lat35.000_lon102.000"

# 设置人口数量（如果文件中有则从文件读取，否则使用默认值）
DEFAULT_POPULATION = 100000  # 默认人口数量，可根据实际情况修改

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)


class SharedClimateData:
    def __init__(self):
        self.shm = None
        self.data = {}
        self.time_index = None
        self.lat_index = None
        self.lon_index = None
        self.load_all_data()

    def load_all_data(self):
        """预加载所有气候数据到共享内存，仅加载2030年的数据"""
        logger.info("Loading climate data...")
        start_time = time.time()

        # 定义文件路径（新格式）
        file_pattern = "{var}_day_BCC-CSM2-MR_ssp126_r1i1p1f1_gn_20300101-20341231_interpolated_1deg.nc"
        self.tas_file = os.path.join(climate_base_path, file_pattern.format(var="tas"))
        self.rsds_file = os.path.join(climate_base_path, file_pattern.format(var="rsds"))
        self.huss_file = os.path.join(climate_base_path, file_pattern.format(var="huss"))
        self.uas_file = os.path.join(climate_base_path, file_pattern.format(var="uas"))
        self.vas_file = os.path.join(climate_base_path, file_pattern.format(var="vas"))
        
        # 加载第一个文件来获取时间和坐标信息
        with xr.open_dataset(self.tas_file) as ds:
            # 使用xarray的索引来选择2030年的数据
            ds_2030 = ds.sel(time=slice('2030-01-01', '2030-12-31'))
            # 将时间索引转换为pandas DatetimeIndex
            # 处理cftime对象：先解码，然后转换
            import cftime
            time_values = ds_2030.time.values
            # 如果是cftime对象，转换为pandas时间戳
            if len(time_values) > 0 and isinstance(time_values[0], cftime.datetime):
                # cftime对象转换为pandas时间戳
                time_str_list = [f"{t.year}-{t.month:02d}-{t.day:02d}" for t in time_values]
                self.time_index = pd.to_datetime(time_str_list)
            else:
                # 尝试直接转换
                try:
                    self.time_index = pd.to_datetime(time_values)
                except (TypeError, ValueError):
                    # 如果失败，尝试先解码
                    try:
                        ds_decoded = xr.decode_cf(ds_2030)
                        self.time_index = pd.to_datetime(ds_decoded.time.values)
                    except:
                        # 最后尝试：使用字符串转换
                        time_str_list = [str(t)[:10] for t in time_values]  # 只取日期部分
                        self.time_index = pd.to_datetime(time_str_list)
            
            self.lat_index = ds.lat.values
            self.lon_index = ds.lon.values

            # 筛选区域的数据
            lat_mask = (self.lat_index >= CHINA_BOUNDS['lat_min']) & (self.lat_index <= CHINA_BOUNDS['lat_max'])
            lon_mask = (self.lon_index >= CHINA_BOUNDS['lon_min']) & (self.lon_index <= CHINA_BOUNDS['lon_max'])

            self.lat_index = self.lat_index[lat_mask]
            self.lon_index = self.lon_index[lon_mask]

        # 加载所有变量的数据到共享内存（仅2030年）
        for var_name, file_path in [
            ('tas', self.tas_file),
            ('rsds', self.rsds_file),
            ('huss', self.huss_file),
            ('uas', self.uas_file),
            ('vas', self.vas_file)
        ]:
            with xr.open_dataset(file_path) as ds:
                # 先选择2030年的时间范围
                ds_2030 = ds.sel(time=slice('2030-01-01', '2030-12-31'))
                # 筛选区域数据
                data = ds_2030[var_name].sel(
                    lat=slice(CHINA_BOUNDS['lat_min'], CHINA_BOUNDS['lat_max']),
                    lon=slice(CHINA_BOUNDS['lon_min'], CHINA_BOUNDS['lon_max'])
                )
                
                # 创建共享内存
                shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
                # 将数据复制到共享内存
                shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
                shared_array[:] = data.values[:]
                self.data[var_name] = (shm, shared_array)

        logger.info(f"Climate data loaded in {time.time() - start_time:.2f} seconds")
        logger.info(f"Loaded {len(self.time_index)} days of 2030 data")

    def get_nearest_point(self, lat, lon):
        """获取最近的网格点索引"""
        lat_idx = np.abs(self.lat_index - lat).argmin()
        lon_idx = np.abs(self.lon_index - lon).argmin()
        return lat_idx, lon_idx

    def get_data(self, lat, lon, start_date, end_date):
        """获取指定位置和时间范围的数据"""
        lat_idx, lon_idx = self.get_nearest_point(lat, lon)
        
        # 打印实际提取的坐标
        actual_lat = self.lat_index[lat_idx]
        actual_lon = self.lon_index[lon_idx]
        logger.info(f"[COPY.PY] 目标坐标: lat={lat:.3f}, lon={lon:.3f}")
        logger.info(f"[COPY.PY] 实际提取的坐标: lat={actual_lat:.6f}, lon={actual_lon:.6f}, lat_idx={lat_idx}, lon_idx={lon_idx}")
        
        # 确保时间比较时类型一致，并规范化為日期（去掉时间部分）
        if isinstance(start_date, np.datetime64):
            start_date = pd.Timestamp(start_date)
        if isinstance(end_date, np.datetime64):
            end_date = pd.Timestamp(end_date)
        
        # 规范化為日期，去掉时间部分，确保包含第一天的数据
        start_date = pd.Timestamp(start_date.date())
        end_date = pd.Timestamp(end_date.date())
        
        # 调试：打印时间筛选信息
        logger.info(f"[COPY.PY] 时间筛选: start_date={start_date}, end_date={end_date}")
        logger.info(f"[COPY.PY] time_index范围: {self.time_index[0]} 到 {self.time_index[-1]}")
        
        time_mask = (self.time_index >= start_date) & (self.time_index <= end_date)
        
        # 调试：打印匹配的天数
        logger.info(f"[COPY.PY] 时间mask匹配的天数: {np.sum(time_mask)}, 期望天数: {(end_date - start_date).days + 1}")

        # 提取原始数据（转换前）
        tas_raw = self.data['tas'][1][time_mask, lat_idx, lon_idx]
        rsds_raw = self.data['rsds'][1][time_mask, lat_idx, lon_idx]
        uas_raw = self.data['uas'][1][time_mask, lat_idx, lon_idx]
        vas_raw = self.data['vas'][1][time_mask, lat_idx, lon_idx]
        huss_raw = self.data['huss'][1][time_mask, lat_idx, lon_idx]
        
        # 打印原始气候数据值
        logger.info(f"[COPY.PY] 提取的原始气候数据值（前5天）:")
        num_days = min(5, len(tas_raw))
        for i in range(num_days):
            logger.info(f"[COPY.PY]   第{i+1}天: tas={tas_raw[i]:.6f}K ({tas_raw[i]-273.15:.2f}°C), "
                       f"rsds={rsds_raw[i]:.6f}, uas={uas_raw[i]:.6f}, vas={vas_raw[i]:.6f}, "
                       f"huss={huss_raw[i]:.9f} ({huss_raw[i]*1000:.6f}g/kg)")
        logger.info(f"[COPY.PY] 总天数: {len(tas_raw)}")
        
        return {
            'temperature': tas_raw - 273.15,  # 转换为摄氏度
            'radiation_global_horizontal': rsds_raw,
            'wind_speed_2m': np.sqrt(
                uas_raw ** 2 +
                vas_raw ** 2
            ),
            'humidity': huss_raw * 1000  # 转换为g/kg
        }

    def cleanup(self):
        """清理共享内存"""
        for var_name, (shm, _) in self.data.items():
            shm.close()
            shm.unlink()


def load_climate_data(point_data, start_date, duration, shared_data):
    """Load climate data for a specific point and time period"""
    # Calculate end date
    duration = int(duration)
    end_date = start_date + timedelta(days=duration - 1)

    # Convert datetime to numpy datetime64
    start_np = np.datetime64(start_date.strftime('%Y-%m-%dT12:00:00'))
    end_np = np.datetime64(end_date.strftime('%Y-%m-%dT12:00:00'))

    # 使用共享数据获取数据
    data = shared_data.get_data(point_data['lat'], point_data['lon'], start_np, end_np)

    # 创建DataFrame
    index = pd.date_range(start=start_date, periods=len(data['temperature']), freq='D')
    weather_df = pd.DataFrame(data, index=index)

    return weather_df


def calculate_energy_demand(point_data, weather_df, population):
    """Calculate energy demand for a point"""
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
        logger.info(f"[COPY.PY] case1 BAIT值 (前5天):")
        logger.info(f"[COPY.PY] {bait.head()}")
        logger.info(f"[COPY.PY] case1 BAIT统计: min={bait.min():.2f}, max={bait.max():.2f}, mean={bait.mean():.2f}")

    # Calculate results for each case
    results = {}
    for case_name, params in cases.items():
        # 确保bait的时间索引是每日数据，且设为0点（只保留日期部分）
        # 因为NC文件的数据实际是12点的，而demand_p函数会将0点转为12点
        # 所以这里需要确保索引是0点的，这样demand_p的+12小时操作才能正确
        daily_bait = bait.copy()
        start_date = pd.Timestamp(daily_bait.index[0].date())
        end_date = pd.Timestamp(daily_bait.index[-1].date())
        daily_bait.index = pd.date_range(
            start=start_date,
            end=end_date,
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
            raw=True  # 添加raw=True以获取更多调试信息
        )
        # # 将能耗单位从GW转换为kW (GW * 1e3 = MW)
        # results[case_name]['total_demand'] *= 1e3
        # results[case_name]['heating_demand'] *= 1e3
        # results[case_name]['cooling_demand'] *= 1e3

        # DEBUG: Save daily energy demand to CSV
        # lat = point_data['lat']
        # lon = point_data['lon']
        # csv_filename = f"energy_demand_lat{lat}_lon{lon}_{case_name}.csv"
        # csv_path = os.path.join(debug_output_dir, csv_filename)
        # results[case_name].to_csv(csv_path)

    return results


def calculate_demand_sums(results, point_data):
    """Calculate total sums for each demand type"""
    sums = {}
    for case_name, result in results.items():
        sums[case_name] = {
            'lat': point_data['lat'],
            'lon': point_data['lon'],
            'total_demand': result['total_demand'].sum(),
            'heating_demand': result['heating_demand'].sum(),
            'cooling_demand': result['cooling_demand'].sum()
        }
    return sums


def save_to_csv(demand_sums, output_dir):
    """Save demand sums to CSV files"""
    for case_name, sums in demand_sums.items():
        # 创建case文件夹
        case_dir = os.path.join(output_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)

        csv_data = {
            'lat': [sums['lat']],
            'lon': [sums['lon']],
            'total_demand': [sums['total_demand']],
            'heating_demand': [sums['heating_demand']],
            'cooling_demand': [sums['cooling_demand']]
        }
        csv_df = pd.DataFrame(csv_data)
        csv_file = os.path.join(case_dir, f"{case_name}.csv")

        # 如果文件已存在，追加数据；否则创建新文件
        if os.path.exists(csv_file):
            csv_df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            csv_df.to_csv(csv_file, index=False)








def main():
    # 检查输入文件是否存在
    if not os.path.exists(point_file):
        raise FileNotFoundError(f"Point file not found at: {point_file}")
    
    logger.info(f"Processing single point: lat={TARGET_LAT}, lon={TARGET_LON}")
    
    # 读取热浪数据
    points_df = pd.read_csv(point_file)
    points_df = points_df[points_df['Duration'] >= 3]  # 只保留持续时间>=3天的热浪
    
    # 筛选指定的点（如果CSV中有lat/lon列，否则假设所有行都是该点的数据）
    if 'lat' in points_df.columns and 'lon' in points_df.columns:
        points_df = points_df[
            (points_df['lat'] == TARGET_LAT) & 
            (points_df['lon'] == TARGET_LON)
        ]
    
    if len(points_df) == 0:
        logger.warning(f"No heat wave data found for point ({TARGET_LAT}, {TARGET_LON})")
        return
    
    logger.info(f"Found {len(points_df)} heat wave events")
    
    # 创建点数据
    point_data = {
        'lat': TARGET_LAT,
        'lon': TARGET_LON,
        'heat_waves': points_df.to_dict('records')  # 将该点的所有热浪事件作为列表存储
    }
    
    # 创建共享气候数据
    shared_data = SharedClimateData()
    
    # 使用默认人口数据
    population = DEFAULT_POPULATION
    logger.info(f"Using population: {population}")
    
    # 处理单个点
    lat = point_data['lat']
    lon = point_data['lon']
    
    try:
        # 获取该点的所有热浪事件
        heat_waves = point_data['heat_waves']
        
        # 初始化总能耗结果 - 动态初始化所有case（ref + case1-case20）
        # 先获取所有case名称
        from CandD.calculate import calculate_cases
        base_params = {
            "heating_power": 27.93,
            "cooling_power": 48.55,
            "heating_threshold_people": 14,
            "cooling_threshold_people": 20,
            "base_power": 0,
            "population": population
        }
        all_cases = calculate_cases(base_params)
        
        # 初始化所有case的结果字典
        total_results = {}
        for case_name in all_cases.keys():
            total_results[case_name] = {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0}
        
        # 计算每个热浪事件的能耗并累加
        logger.info(f"Processing {len(heat_waves)} heat wave events...")
        for idx, heat_wave in enumerate(heat_waves, 1):
            # Parse start date from the 'date' column (format: month/day)
            month, day = map(int, heat_wave['date'].split('/'))
            start_date = datetime(2030, month, day)
            duration = int(heat_wave['Duration'])
            
            logger.info(f"Processing heat wave {idx}/{len(heat_waves)}: {heat_wave['date']}, duration={duration} days")
            
            # Load climate data for this heat wave
            weather_df = load_climate_data(point_data, start_date, duration, shared_data)
            
            # Calculate energy demand for this heat wave
            results = calculate_energy_demand(point_data, weather_df, population)
            
            # Add the results to the total
            for case_name, case_result in results.items():
                total_results[case_name]['total_demand'] += case_result['total_demand'].sum()
                total_results[case_name]['heating_demand'] += case_result['heating_demand'].sum()
                total_results[case_name]['cooling_demand'] += case_result['cooling_demand'].sum()
        
        # 为每个case添加lat和lon信息，以便保存
        for case_name in total_results.keys():
            total_results[case_name]['lat'] = lat
            total_results[case_name]['lon'] = lon
        
        # Save the total results
        save_to_csv(total_results, output_dir)
        
        logger.info("\nProcessing completed successfully!")
        logger.info(f"Total results for lat={lat}, lon={lon}:")
        for case_name, result in total_results.items():
            logger.info(f"  {case_name}: total={result['total_demand']:.2f}, "
                       f"heating={result['heating_demand']:.2f}, cooling={result['cooling_demand']:.2f}")
        
    except Exception as e:
        logger.error(f"Error processing point ({lat}, {lon}): {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        # 清理内存
        logger.info("Cleaning up resources...")
        shared_data.cleanup()
        gc.collect()
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
