import pandas as pd
import numpy as np
import xarray as xr
import os
from pathlib import Path
import sys
from datetime import datetime, timedelta
import rasterio
from rasterio.transform import from_origin
from tqdm import tqdm
import time
import multiprocessing
import gc
import logging
from multiprocessing import shared_memory
import numpy as np

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

# 定义中国区域范围（稍微扩大一点以确保边界点也能被覆盖）
CHINA_BOUNDS = {
    'lat_min': -90.0, # 稍微扩大范围
    'lat_max': 90.0,
    'lon_min': 180.0,  # 修改为180.0
    'lon_max': 360.0  # 修改为360.0
}

# 设置气候数据路径
climate_base_path = r"Y:\CMIP6\future\SSP126_2030"  # 修改为你的气候数据路径

# 设置输入点文件路径（CSV）
point_file = r"C:\Users\thuarchdog\PycharmProjects\PythonProject\output_2030\2030_all_heat_wave_west.csv"
population_file = r"C:\Users\thuarchdog\PycharmProjects\PythonProject\population\West_population.csv"

# 设置输出路径
output_dir = r"C:\Users\thuarchdog\Desktop\result\WEST_output_all_heat_wave_SSP1"
# debug_output_dir = r"C:\Users\localhost\PycharmProjects\pythonProject\extreme weather v2\China\debug_output"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
# os.makedirs(debug_output_dir, exist_ok=True)


class SharedClimateData:
    def __init__(self):
        self.shm = None
        self.data = {}
        self.time_index = None
        self.lat_index = None
        self.lon_index = None
        self.load_all_data()

    def load_all_data(self):
        """预加载所有气候数据到共享内存"""
        logger.info("Loading climate data...")
        start_time = time.time()

        # 定义文件路径
        self.tas_file = os.path.join(climate_base_path, "tas_day_CNRM-CM6-1-HR_ssp126_r1i1p1f2_gr_20300101-20301231.nc")
        self.rsds_file = os.path.join(climate_base_path,
                                      "rsds_day_CNRM-CM6-1-HR_ssp126_r1i1p1f2_gr_20300101-20301231.nc")
        self.huss_file = os.path.join(climate_base_path,
                                      "huss_day_CNRM-CM6-1-HR_ssp126_r1i1p1f2_gr_20300101-20301231.nc")
        self.uas_file = os.path.join(climate_base_path, "uas_day_CNRM-CM6-1-HR_ssp126_r1i1p1f2_gr_20300101-20301231.nc")
        self.vas_file = os.path.join(climate_base_path, "vas_day_CNRM-CM6-1-HR_ssp126_r1i1p1f2_gr_20300101-20301231.nc")
        # 加载第一个文件来获取时间和坐标信息
        with xr.open_dataset(self.tas_file) as ds:
            self.time_index = pd.to_datetime(ds.time.values)
            self.lat_index = ds.lat.values
            self.lon_index = ds.lon.values

            # 筛选中国区域的数据
            lat_mask = (self.lat_index >= CHINA_BOUNDS['lat_min']) & (self.lat_index <= CHINA_BOUNDS['lat_max'])
            lon_mask = (self.lon_index >= CHINA_BOUNDS['lon_min']) & (self.lon_index <= CHINA_BOUNDS['lon_max'])

            self.lat_index = self.lat_index[lat_mask]
            self.lon_index = self.lon_index[lon_mask]

        # 加载所有变量的数据到共享内存
        for var_name, file_path in [
            ('tas', self.tas_file),
            ('rsds', self.rsds_file),
            ('huss', self.huss_file),
            ('uas', self.uas_file),
            ('vas', self.vas_file)
        ]:
            with xr.open_dataset(file_path) as ds:
                # 只加载180~360经度的数据
                data = ds[var_name].sel(
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

    def get_nearest_point(self, lat, lon):
        """获取最近的网格点索引"""
        lat_idx = np.abs(self.lat_index - lat).argmin()
        lon_idx = np.abs(self.lon_index - lon).argmin()
        return lat_idx, lon_idx

    def get_data(self, lat, lon, start_date, end_date):
        """获取指定位置和时间范围的数据"""
        lat_idx, lon_idx = self.get_nearest_point(lat, lon)
        time_mask = (self.time_index >= start_date) & (self.time_index <= end_date)

        return {
            'temperature': self.data['tas'][1][time_mask, lat_idx, lon_idx] - 273.15,  # 转换为摄氏度
            'radiation_global_horizontal': self.data['rsds'][1][time_mask, lat_idx, lon_idx],
            'wind_speed_2m': np.sqrt(
                self.data['uas'][1][time_mask, lat_idx, lon_idx] ** 2 +
                self.data['vas'][1][time_mask, lat_idx, lon_idx] ** 2
            ),
            'humidity': self.data['huss'][1][time_mask, lat_idx, lon_idx] * 1000  # 转换为g/kg
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


def save_results_to_tif(all_points_results, output_dir):
    """Save all points results as GeoTIFF files"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有点的经纬度范围
    lats = [point['lat'] for point in all_points_results]
    lons = [point['lon'] for point in all_points_results]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # 创建网格
    lat_grid = np.arange(min_lat - 0.1, max_lat + 0.1, 0.2)
    lon_grid = np.arange(min_lon - 0.1, max_lon + 0.1, 0.2)

    # 为每个case创建文件夹并保存结果
    case_names = ['ref', 'case1', 'case2', 'case3', 'case4', 'case5', 'case6', 'case7', 'case8', 'case9']
    for case_name in case_names:
        case_dir = os.path.join(output_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)

        # 创建空数组
        total_demand_grid = np.zeros((len(lat_grid), len(lon_grid)))
        heating_demand_grid = np.zeros((len(lat_grid), len(lon_grid)))
        cooling_demand_grid = np.zeros((len(lat_grid), len(lon_grid)))

        # 填充数据
        for point_result in all_points_results:
            if point_result['case_name'] == case_name:
                lat_idx = np.abs(lat_grid - point_result['lat']).argmin()
                lon_idx = np.abs(lon_grid - point_result['lon']).argmin()
                total_demand_grid[lat_idx, lon_idx] = point_result['total_demand']
                heating_demand_grid[lat_idx, lon_idx] = point_result['heating_demand']
                cooling_demand_grid[lat_idx, lon_idx] = point_result['cooling_demand']

        # 定义transform
        transform = from_origin(min_lon - 0.1, max_lat + 0.1, 0.2, 0.2)

        # 保存total_demand
        with rasterio.open(
                os.path.join(case_dir, "total_demand.tif"),
                'w',
                driver='GTiff',
                height=len(lat_grid),
                width=len(lon_grid),
                count=1,
                dtype=np.float32,
                crs='EPSG:4326',
                transform=transform,
        ) as dst:
            dst.write(total_demand_grid, 1)

        # 保存heating_demand
        with rasterio.open(
                os.path.join(case_dir, "heating_demand.tif"),
                'w',
                driver='GTiff',
                height=len(lat_grid),
                width=len(lon_grid),
                count=1,
                dtype=np.float32,
                crs='EPSG:4326',
                transform=transform,
        ) as dst:
            dst.write(heating_demand_grid, 1)

        # 保存cooling_demand
        with rasterio.open(
                os.path.join(case_dir, "cooling_demand.tif"),
                'w',
                driver='GTiff',
                height=len(lat_grid),
                width=len(lon_grid),
                count=1,
                dtype=np.float32,
                crs='EPSG:4326',
                transform=transform,
        ) as dst:
            dst.write(cooling_demand_grid, 1)


def process_point(point_data, shared_data, population_df, output_dir):
    """处理单个点的数据，计算所有热浪的能耗总和"""
    lat = point_data['lat']
    lon = point_data['lon']

    try:
        # Get population for this point
        pop_data = population_df[
            (population_df['lat'] == lat) &
            (population_df['lon'] == lon)
            ]

        if len(pop_data) == 0:
            logger.warning(f"No population data found for point ({lat}, {lon})")
            return None

        population = pop_data['population'].iloc[0]

        # 获取该点的所有热浪事件
        heat_waves = point_data['heat_waves']  # 这是一个包含所有热浪事件的列表

        # 初始化总能耗结果
        total_results = {
            'ref': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case1': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case2': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case3': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case4': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case5': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case6': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case7': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case8': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case9': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0}
        }

        # 计算每个热浪事件的能耗并累加
        for heat_wave in heat_waves:
            # Parse start date from the 'date' column (format: month/day)
            month, day = map(int, heat_wave['date'].split('/'))
            start_date = datetime(2030, month, day)
            duration = int(heat_wave['Duration'])

            # Load climate data for this heat wave
            weather_df = load_climate_data(point_data, start_date, duration, shared_data)

            # Calculate energy demand for this heat wave
            results = calculate_energy_demand(point_data, weather_df, population)

            # Add the results to the total
            for case_name, case_result in results.items():
                total_results[case_name]['total_demand'] += case_result['total_demand'].sum()
                total_results[case_name]['heating_demand'] += case_result['heating_demand'].sum()
                total_results[case_name]['cooling_demand'] += case_result['cooling_demand'].sum()

        # Save the total results
        save_to_csv(total_results, output_dir)

        return total_results

    except Exception as e:
        logger.error(f"Error processing point ({lat}, {lon}): {str(e)}")
        return None


def process_batch(batch_data):
    """处理一批数据点"""
    try:
        batch_results = []
        shared_data = batch_data['shared_data']
        population_df = batch_data['population_df']
        output_dir = batch_data['output_dir']

        for point_data in batch_data['points']:
            try:
                # 获取该点的所有热浪事件
                heat_waves = point_data['heat_waves']
                lat = point_data['lat']
                lon = point_data['lon']

                # 获取人口数据
                pop_data = population_df[
                    (population_df['lat'] == lat) &
                    (population_df['lon'] == lon)
                ]

                if len(pop_data) == 0:
                    logger.warning(f"No population data found for point ({lat}, {lon})")
                    continue

                population = pop_data['population'].iloc[0]

                # 初始化总能耗结果
                total_results = {
                    'ref': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
                    'case1': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
                    'case2': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
                    'case3': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
                    'case4': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
                    'case5': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
                    'case6': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
                    'case7': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
                    'case8': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
                    'case9': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0}
                }

                # 计算每个热浪事件的能耗并累加
                for heat_wave in heat_waves:
                    # Parse start date from the 'date' column (format: month/day)
                    month, day = map(int, heat_wave['date'].split('/'))
                    start_date = datetime(2030, month, day)
                    duration = int(heat_wave['Duration'])

                    # Load climate data for this heat wave
                    weather_df = load_climate_data(point_data, start_date, duration, shared_data)

                    # Calculate energy demand for this heat wave
                    results = calculate_energy_demand(point_data, weather_df, population)

                    # Add the results to the total
                    for case_name, case_result in results.items():
                        total_results[case_name]['total_demand'] += case_result['total_demand'].sum()
                        total_results[case_name]['heating_demand'] += case_result['heating_demand'].sum()
                        total_results[case_name]['cooling_demand'] += case_result['cooling_demand'].sum()

                # 保存每个case的结果到CSV
                for case_name, case_result in total_results.items():
                    # 创建case文件夹
                    case_dir = os.path.join(output_dir, case_name)
                    os.makedirs(case_dir, exist_ok=True)

                    # 准备CSV数据
                    csv_data = {
                        'lat': [lat],
                        'lon': [lon],
                        'total_demand': [case_result['total_demand']],
                        'heating_demand': [case_result['heating_demand']],
                        'cooling_demand': [case_result['cooling_demand']]
                    }
                    csv_df = pd.DataFrame(csv_data)
                    csv_file = os.path.join(case_dir, f"{case_name}.csv")

                    # 如果文件已存在，追加数据；否则创建新文件
                    if os.path.exists(csv_file):
                        csv_df.to_csv(csv_file, mode='a', header=False, index=False)
                    else:
                        csv_df.to_csv(csv_file, index=False)

                    # 将结果添加到批次结果中
                    batch_results.append({
                        'lat': lat,
                        'lon': lon,
                        'case_name': case_name,
                        'total_demand': case_result['total_demand'],
                        'heating_demand': case_result['heating_demand'],
                        'cooling_demand': case_result['cooling_demand']
                    })

            except Exception as e:
                logger.error(f"Error processing point ({lat}, {lon}): {str(e)}")
                continue

        return batch_results
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        return []


def main():
    # 检查输入文件是否存在
    if not os.path.exists(point_file):
        raise FileNotFoundError(f"Point file not found at: {point_file}")
    if not os.path.exists(population_file):
        raise FileNotFoundError(f"Population file not found at: {population_file}")

    # 读取热浪数据并按经纬度分组
    points_df = pd.read_csv(point_file)
    points_df = points_df[points_df['Duration'] >= 3]  # 只保留持续时间>=3天的热浪

    # 将数据按经纬度分组，每个组包含该点的所有热浪事件
    grouped_points = []
    for (lat, lon), group in points_df.groupby(['lat', 'lon']):
        point_data = {
            'lat': lat,
            'lon': lon,
            'heat_waves': group.to_dict('records')  # 将该点的所有热浪事件作为列表存储
        }
        grouped_points.append(point_data)

    # 获取CPU核心数并设置进程数
    num_cores = multiprocessing.cpu_count()
    num_processes = min(num_cores * 2, 32)
    logger.info(f"Using {num_processes} processes for parallel processing")

    # 准备批次数据
    total_points = len(grouped_points)
    batch_size = max(1, min(100, total_points // (num_processes * 8)))
    batches = [grouped_points[i:i + batch_size] for i in range(0, total_points, batch_size)]

    # 创建共享气候数据
    shared_data = SharedClimateData()

    # 为每个批次添加共享数据
    batch_data = [{
        'points': batch,
        'population_df': pd.read_csv(population_file),
        'output_dir': output_dir,
        'shared_data': shared_data
    } for batch in batches]

    logger.info(f"Data distribution:")
    logger.info(f"- Total points: {total_points}")
    logger.info(f"- Batch size: {batch_size}")
    logger.info(f"- Total batches: {len(batches)}")

    # 创建进程池进行并行处理
    results = []
    failed_points = []

    with multiprocessing.Pool(processes=num_processes) as pool:
        chunksize = max(1, len(batches) // (num_processes * 4))

        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            for batch_results in pool.imap_unordered(process_batch, batch_data, chunksize=chunksize):
                if batch_results:
                    results.extend(batch_results)
                pbar.update(1)

    # 清理内存
    logger.info("Parallel processing completed, cleaning up resources...")
    shared_data.cleanup()
    del grouped_points, batch_data
    gc.collect()

    # 输出处理结果统计
    successful_points = len(results)
    logger.info(f"\nProcessing completed:")
    logger.info(f"- Total points processed: {total_points}")
    logger.info(f"- Successfully processed: {successful_points}")
    logger.info(f"- Failed points: {len(failed_points)}")

    # 保存结果到tif文件
    logger.info("\nSaving results to GeoTIFF files...")
    save_results_to_tif(results, output_dir)
    logger.info("Done!")


if __name__ == "__main__":
    main()
