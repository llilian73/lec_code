"""
热浪期间能耗计算工具

功能：
读取identify_heatwave_lin.py输出的热浪事件CSV文件，计算各个经纬度点热浪期间的能耗。
支持多模型、多SSP、多年份处理。

输入数据：
1. 热浪事件CSV文件：/home/linbor/WORK/lishiying/heat_wave/{模型名}/{SSP路径}/{年份}_all_heat_wave.csv
2. 未来气候数据：/home/linbor/WORK/lishiying/GCM_input_processed/{模型名}/future/{SSP路径}/
3. 人口数据：/home/linbor/WORK/lishiying/population/SSP1_population.csv (SSP126) 或 SSP2_population.csv (SSP245)

输出数据：
- 路径：/home/linbor/WORK/lishiying/heat_wave/{模型名}/{SSP路径}/energy/{年份}/point/{case_name}/
- 每个case文件夹包含两个CSV文件：
  1. {case_name}.csv：总能耗（列：lat, lon, total_demand, heating_demand, cooling_demand）
  2. ref_hourly.csv 或 case{number}_hourly.csv：逐小时能耗（列：lat, lon, number, date, time, total_demand, heating_demand, cooling_demand）
"""

import pandas as pd
import numpy as np
import xarray as xr
import os
import sys
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import multiprocessing
import gc
import logging
from multiprocessing import shared_memory
import re
import cftime
import pyarrow as pa
import pyarrow.parquet as pq
from functools import partial

# === 新增：全局进程私有临时目录 → 由 init_worker 设置 ===
_worker_hourly_dir = None

# === 新增：Parquet Schema（类型安全 + 高效）===
HOURLY_SCHEMA = pa.schema([
    ('lat', pa.float64()),
    ('lon', pa.float64()),
    ('number', pa.int32()),
    ('date', pa.date32()),   # 仅日期，非 datetime
    ('time', pa.time32('ms')),  # 仅时间，毫秒精度
    ('total_demand', pa.float64()),
    ('heating_demand', pa.float64()),
    ('cooling_demand', pa.float64())
])
# Add project root to path
# 支持Linux服务器路径：确保能找到 /home/linbor/WORK/lishiying/CandD 和 /home/linbor/WORK/lishiying/demand_ninja
# 代码文件路径：/home/linbor/WORK/lishiying/energy_last_for_all_hw.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# 当前文件所在目录即为项目根目录（因为文件直接在 /home/linbor/WORK/lishiying/ 下）
project_root = current_dir

# 确保项目根目录在 sys.path 中（优先级最高）
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 备选方案：如果当前目录下没有 CandD 或 demand_ninja，尝试向上查找
if not (os.path.exists(os.path.join(project_root, "CandD")) or 
        os.path.exists(os.path.join(project_root, "demand_ninja"))):
    # 尝试向上查找（适用于文件在子目录中的情况）
    parent_dir = os.path.dirname(project_root)
    if os.path.exists(os.path.join(parent_dir, "CandD")) or \
       os.path.exists(os.path.join(parent_dir, "demand_ninja")):
        sys.path.insert(0, parent_dir)

from demand_ninja.core_p import _bait, demand_p as demand
from CandD.calculate import calculate_cases

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置路径（Linux路径）
BASE_PATH = "/home/linbor/WORK/lishiying"
HEAT_WAVE_BASE_PATH = os.path.join(BASE_PATH, "heat_wave")
CLIMATE_BASE_PATH = os.path.join(BASE_PATH, "GCM_input_processed")
POPULATION_BASE_PATH = os.path.join(BASE_PATH, "population")

# 模型配置（可以通过修改此列表来选择要处理的模型）
# 默认只处理 BCC-CSM2-MR 模型
MODELS = [
    # "BCC-CSM2-MR",
    # "ACCESS-ESM1-5",
    # "CanESM5",
    # "EC-Earth3",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0"
]

# SSP配置
SSP_PATHS = ["SSP126", "SSP245"]
TARGET_YEARS = [2030, 2031, 2032, 2033, 2034]

# 气候变量
CLIMATE_VARIABLES = ['tas', 'rsds', 'huss', 'uas', 'vas']

# 并行处理参数
NUM_PROCESSES = 31


def convert_cftime_to_datetime(time_values):
    """
    将cftime对象转换为pandas DatetimeIndex（使用与identify_heatwave_所有模型历史版本.py相同的健壮方法）
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


def find_climate_file(future_dir, variable, model_name, ssp_path):
    """查找气候数据文件（修复：精确匹配变量名前缀，避免 tas 匹配 tasmax）"""
    # 筛选基础条件：.nc + 插值标记
    all_files = [
        f for f in os.listdir(future_dir)
        if f.endswith(".nc") and "_interpolated_1deg" in f
    ]
    
    matching_files = []
    for f in all_files:
        # ✅ 精确匹配：变量名必须以 "{variable}_day_" 开头（如 "tas_day_"）
        #    允许前缀有路径，但文件名部分必须满足该模式
        if not f.startswith(f"{variable}_day_"):
            continue
        
        # 检查模型名（支持 BCC-CSM2-MR 或 BCC_CSM2_MR）
        model_ok = (model_name.replace("-", "_") in f) or (model_name in f)
        # 检查 SSP（不区分大小写）
        ssp_ok = ssp_path.lower() in f.lower()
        
        if model_ok and ssp_ok:
            matching_files.append(f)
    
    # 排序：确保确定性（如 tas_day_ 在 tasmax_day_ 之前——虽然已不会被误收）
    matching_files.sort()
    
    if len(matching_files) == 0:
        logger.error(f"❌ 未找到变量 '{variable}' 的文件（需精确匹配 ^{variable}_day_）")
        logger.info(f"目录 {future_dir} 中所有.nc文件: {sorted([f for f in os.listdir(future_dir) if f.endswith('.nc')])}")
        return None
    elif len(matching_files) == 1:
        return os.path.join(future_dir, matching_files[0])
    else:
        # 多个匹配文件：优先选含 20300101-20341231 的
        for f in matching_files:
            if "20300101" in f and "20341231" in f:
                return os.path.join(future_dir, f)
        # 仍无明确优先：返回字典序第一个（因已 sort）
        logger.warning(f"找到多个匹配文件，使用第一个: {matching_files[0]}")
        return os.path.join(future_dir, matching_files[0])


class SharedClimateData:
    """共享气候数据类（按年份加载）"""
    def __init__(self, future_dir, model_name, ssp_path, year):
        self.future_dir = future_dir
        self.model_name = model_name
        self.ssp_path = ssp_path
        self.year = year
        self.data = {}
        self.time_index = None
        self.lat_index = None
        self.lon_index = None
        self.file_cache = {}  # 缓存已打开的文件
        self.load_year_data()
    
    def load_year_data(self):
        """加载指定年份的气候数据到共享内存"""
        # logger.info(f"正在加载 {self.model_name} {self.ssp_path} {self.year} 年的气候数据...")
        start_time = time.time()
        
        # 查找所有变量的文件
        var_files = {}
        for var_name in CLIMATE_VARIABLES:
            file_path = find_climate_file(self.future_dir, var_name, self.model_name, self.ssp_path)
            if file_path is None:
                raise FileNotFoundError(f"未找到 {var_name} 变量文件: {self.future_dir}")
            var_files[var_name] = file_path
        
        # 打开第一个文件获取时间和坐标信息
        # 使用明确的chunks大小来加速加载，避免 'auto' 导致的 object dtype 错误
        first_file = var_files['tas']
        chunks_dict = None  # 用于后续文件
        
        if first_file not in self.file_cache:
            # 先打开获取维度信息
            ds_temp = xr.open_dataset(first_file)
            # 获取维度信息
            dims = ds_temp.dims
            # 构建chunks字典，只对数值维度使用chunks，时间维度用365天，空间维度不chunk
            chunks_dict = {}
            if 'time' in dims:
                chunks_dict['time'] = 366  # 每年365天为一个chunk
            if 'lat' in dims:
                chunks_dict['lat'] = -1  # 不chunk纬度维度
            if 'lon' in dims:
                chunks_dict['lon'] = -1  # 不chunk经度维度
            ds_temp.close()
            
            # 使用明确的chunks重新打开
            if chunks_dict:
                try:
                    self.file_cache[first_file] = xr.open_dataset(first_file, chunks=chunks_dict)
                except Exception:
                    # 如果chunks失败，回退到不使用chunks
                    logger.warning(f"使用chunks打开文件失败，回退到不使用chunks: {first_file}")
                    self.file_cache[first_file] = xr.open_dataset(first_file)
                    chunks_dict = None  # 如果失败，后续也不使用chunks
            else:
                self.file_cache[first_file] = xr.open_dataset(first_file)
        
        # 获取第一个文件的数据集和维度信息（用于后续文件）
        ds = self.file_cache[first_file]
        if chunks_dict is None:
            # 如果chunks_dict还是None，尝试从已打开的数据集获取维度信息
            dims = ds.dims
            chunks_dict = {}
            if 'time' in dims:
                chunks_dict['time'] = 366
            if 'lat' in dims:
                chunks_dict['lat'] = -1
            if 'lon' in dims:
                chunks_dict['lon'] = -1
            # 如果仍然没有chunks配置，设为None表示不使用chunks
            if not chunks_dict:
                chunks_dict = None
        
        # 处理cftime时间对象（使用与identify_heatwave_所有模型历史版本.py相同的健壮方法）
        try:
            # 方法1：尝试使用decode_cf
            ds_decoded = xr.decode_cf(ds, decode_times=True)
            time_values = ds_decoded.time.values
            
            if hasattr(time_values, 'to_pandas'):
                self.time_index = time_values.to_pandas()
            else:
                self.time_index = pd.to_datetime(time_values)
            
            if not isinstance(self.time_index, pd.DatetimeIndex):
                self.time_index = pd.DatetimeIndex(self.time_index)
                
        except Exception as e:
            # 方法2：如果decode_cf失败，手动处理cftime对象
            try:
                import cftime
                time_values = ds.time.values
                
                if len(time_values) > 0 and isinstance(time_values[0], cftime.datetime):
                    # 手动转换cftime对象为pandas Timestamp
                    self.time_index = pd.to_datetime([pd.Timestamp(t.year, t.month, t.day) for t in time_values])
                else:
                    self.time_index = pd.to_datetime(time_values)
                
                if not isinstance(self.time_index, pd.DatetimeIndex):
                    self.time_index = pd.DatetimeIndex(self.time_index)
                    
            except Exception as e2:
                # 方法3：最后尝试使用to_pandas
                try:
                    self.time_index = ds.time.to_pandas()
                    if not isinstance(self.time_index, pd.DatetimeIndex):
                        self.time_index = pd.DatetimeIndex(self.time_index)
                except Exception:
                    raise ValueError(f"无法转换时间索引: {e}, {e2}")
        
        self.lat_index = ds.lat.values
        self.lon_index = ds.lon.values
        
        # 提取该年份的数据
        year_mask = self.time_index.year == self.year
        if year_mask.sum() == 0:
            raise ValueError(f"文件中不包含 {self.year} 年的数据")
        
        # 保存该年份的时间索引（用于后续的时间筛选）
        self.time_index_year = self.time_index[year_mask]
        
        # 加载所有变量的数据到共享内存
        # 使用明确的chunks大小来加速加载，避免 'auto' 导致的 object dtype 错误
        # 使用与第一个文件相同的chunks配置
        for var_name, file_path in var_files.items():
            if file_path not in self.file_cache:
                # 使用明确的chunks打开
                if chunks_dict:
                    try:
                        self.file_cache[file_path] = xr.open_dataset(file_path, chunks=chunks_dict)
                    except Exception:
                        # 如果chunks失败，回退到不使用chunks
                        logger.warning(f"使用chunks打开文件失败，回退到不使用chunks: {file_path}")
                        self.file_cache[file_path] = xr.open_dataset(file_path)
                else:
                    self.file_cache[file_path] = xr.open_dataset(file_path)
            
            ds_var = self.file_cache[file_path]
            
            # 提取该年份的数据
            year_data = ds_var[var_name].isel(time=year_mask).load()
            
            # 创建共享内存
            shm = shared_memory.SharedMemory(create=True, size=year_data.nbytes)
            shared_array = np.ndarray(year_data.shape, dtype=year_data.dtype, buffer=shm.buf)
            shared_array[:] = year_data.values[:]
            
            self.data[var_name] = (shm, shared_array)
        
        # logger.info(f"气候数据加载完成，耗时 {time.time() - start_time:.2f} 秒")
    
    def get_nearest_point(self, lat, lon):
        """获取最近的网格点索引"""
        lat_idx = np.abs(self.lat_index - lat).argmin()
        lon_idx = np.abs(self.lon_index - lon).argmin()
        return lat_idx, lon_idx
    
    def get_data(self, lat, lon, start_date, end_date):
        """获取指定位置和时间范围的数据"""
        lat_idx, lon_idx = self.get_nearest_point(lat, lon)
        
        # 确保时间比较时类型一致，并规范化為日期（去掉时间部分）
        if isinstance(start_date, datetime):
            start_date = pd.Timestamp(start_date.date())
        elif isinstance(start_date, np.datetime64):
            start_date = pd.Timestamp(start_date)
            start_date = pd.Timestamp(start_date.date())
        else:
            start_date = pd.Timestamp(start_date)
            if hasattr(start_date, 'date'):
                start_date = pd.Timestamp(start_date.date())
        
        if isinstance(end_date, datetime):
            end_date = pd.Timestamp(end_date.date())
        elif isinstance(end_date, np.datetime64):
            end_date = pd.Timestamp(end_date)
            end_date = pd.Timestamp(end_date.date())
        else:
            end_date = pd.Timestamp(end_date)
            if hasattr(end_date, 'date'):
                end_date = pd.Timestamp(end_date.date())
        
        # 确保time_index_year也是日期精度（使用该年份的时间索引）
        time_index_norm = pd.to_datetime([pd.Timestamp(t).date() for t in self.time_index_year])
        
        # 使用该年份的时间索引来创建mask（与self.data的维度匹配）
        time_mask = (time_index_norm >= start_date) & (time_index_norm <= end_date)
        
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
        """清理共享内存和文件"""
        for var_name, (shm, _) in self.data.items():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        
        for ds in self.file_cache.values():
            try:
                ds.close()
            except Exception:
                pass


def load_climate_data(point_data, start_date, duration, shared_data):
    """加载指定点和时间范围的气候数据"""
    func_start = time.perf_counter()
    lat = point_data['lat']
    lon = point_data['lon']
    
    # logger.info(f"[load_climate_data] 开始加载气候数据: lat={lat}, lon={lon}, start_date={start_date}, duration={duration}")
    
    try:
        # 步骤1: 计算结束日期
        duration = int(duration)
        end_date = start_date + timedelta(days=duration - 1)
        # 步骤2: 规范化日期（去掉时间部分）

        start_date_norm = pd.Timestamp(start_date.date())
        end_date_norm = pd.Timestamp(end_date.date())
           # 步骤3: 获取数据（这里可能会卡住）
        data = shared_data.get_data(lat, lon, start_date_norm, end_date_norm)

       
        
        # 步骤4: 创建DataFrame
        
        index = pd.date_range(start=start_date_norm, periods=len(data['temperature']), freq='D')
        weather_df = pd.DataFrame(data, index=index)    
        return weather_df
        
    except Exception as e:
        total_duration = time.perf_counter() - func_start
        logger.error(f"[load_climate_data] 函数出错, 耗时 {total_duration:.4f}s, 错误: {str(e)}", exc_info=True)
        raise


def calculate_energy_demand(point_data, weather_df, population):
    """计算能耗"""
    # 计算BAIT
    bait = _bait(
        weather=weather_df,
        smoothing=0.73,
        solar_gains=0.014,
        wind_chill=-0.12,
        humidity_discomfort=0.036
    )
    
    # 获取计算cases
    base_params = {
        "heating_power": 27.93,
        "cooling_power": 48.55,
        "heating_threshold_people": 14,
        "cooling_threshold_people": 20,
        "base_power": 0,
        "population": population
    }
    
    cases = calculate_cases(base_params)
    
    # 计算每个case的结果
    results = {}
    for case_name, params in cases.items():
        daily_bait = bait.copy()
        # 将时间索引设为0点（只保留日期部分）
        # 因为NC文件的数据实际是12点的，而demand_p函数会将0点转为12点
        # 所以这里需要确保索引是0点的，这样demand_p的+12小时操作才能正确
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
            raw=True
        )
    
    return results


def process_point(point_data, shared_data, population_df, model_name, ssp_path, year):
    """处理单个点（流式写入逐时 Parquet，内存恒定）"""
    import os
    global _worker_hourly_dir
    # === 新增：防御性检查 ===
    if _worker_hourly_dir is None:
        logger.warning(f"[进程 {os.getpid()}] hourly 目录未初始化，跳过逐时输出")
        save_hourly = False
    else:
        save_hourly = True
        
    lat = point_data['lat']
    lon = point_data['lon']
    point_key = f"({lat:.2f},{lon:.2f})"
    
    start_time = time.perf_counter()
    try:
        # 1. 获取人口
        pop_data = population_df[(population_df['lat'] == lat) & (population_df['lon'] == lon)]
        if len(pop_data) == 0:
            logger.warning(f"未找到人口数据: {point_key}")
            return None
        population = pop_data['population'].iloc[0]

        # 2. 初始化总能耗
        total_results = {f'case{i}': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0}
                         for i in range(1, 21)}
        total_results['ref'] = {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0}

        # 3. 处理每个热浪事件
        heat_waves = point_data['heat_waves']
        for idx, heat_wave in enumerate(heat_waves, 1):
            try:
                # 解析日期 & 加载数据
                month, day = map(int, heat_wave['date'].split('/'))
                start_date = datetime(shared_data.year, month, day)
                weather_df = load_climate_data(point_data, start_date, int(heat_wave['Duration']), shared_data)

                # 计算能耗
                results = calculate_energy_demand(point_data, weather_df, population)

                # === 流式写入逐时数据 ===
                if save_hourly:
                    for case_name, hourly_df in results.items():
                        # 构建标准化 DataFrame
                        df = hourly_df[['total_demand', 'heating_demand', 'cooling_demand']].copy()
                        df['lat'] = lat
                        df['lon'] = lon
                        df['number'] = idx
                        # 日期：提取日期部分（转为datetime后PyArrow会自动转换为date32）
                        df['date'] = pd.to_datetime(df.index.date)
                        # 时间：df.index.time 已经返回time对象数组，直接使用即可
                        df['time'] = df.index.time
                        df = df.reset_index(drop=True)
                        df = df[['lat', 'lon', 'number', 'date', 'time', 'total_demand', 'heating_demand', 'cooling_demand']]

                        # 转为 Arrow Table（带 Schema）
                        table = pa.Table.from_pandas(df, schema=HOURLY_SCHEMA, preserve_index=False)

                        # 写入该进程私有 parquet（追加模式：多个热浪事件 → 同一文件）
                        parquet_path = os.path.join(_worker_hourly_dir, f"{case_name}.parquet")
                        if os.path.exists(parquet_path):
                            # 读旧 + 合并（小规模安全）
                            old_table = pq.read_table(parquet_path)
                            new_table = pa.concat_tables([old_table, table])
                            pq.write_table(new_table, parquet_path, compression='snappy')
                        else:
                            pq.write_table(table, parquet_path, compression='snappy')

                # 累加总能耗
                for case_name, case_result in results.items():
                    total_results[case_name]['total_demand'] += case_result['total_demand'].sum()
                    total_results[case_name]['heating_demand'] += case_result['heating_demand'].sum()
                    total_results[case_name]['cooling_demand'] += case_result['cooling_demand'].sum()

            except Exception as e:
                logger.error(f"热浪 #{idx} 处理失败 ({point_key}): {e}", exc_info=True)
                continue

        duration = time.perf_counter() - start_time
        logger.debug(f"✓ 点 {point_key} 完成，{len(heat_waves)} 热浪，{duration:.2f}s")
        return {'lat': lat, 'lon': lon, 'results': total_results}

    except Exception as e:
        logger.error(f"❌ 点 {point_key} 全局失败: {e}", exc_info=True)
        return None

# 全局变量用于多进程
_global_shared_data = None
_global_population_df = None


def init_worker(shared_data_dict, population_df_dict):
    """初始化工作进程（+ 创建私有 hourly 输出目录）"""
    import os, tempfile
    global _global_shared_data, _global_population_df, _worker_hourly_dir
    
    worker_id = os.getpid()
    logger.info(f"[进程 {worker_id}] 开始初始化工作进程...")

    try:
        # 1. 重建共享数据
        _global_shared_data = SharedClimateDataFromDict(shared_data_dict)
        logger.info(f"[进程 {worker_id}] 共享内存连接成功")

        # 2. 加载人口数据
        _global_population_df = pd.DataFrame(population_df_dict)
        logger.info(f"[进程 {worker_id}] 人口数据加载成功，共 {len(_global_population_df)} 条记录")

        # 3. 创建私有 hourly 输出目录（避免多进程竞争）
        model = shared_data_dict.get('model_name', 'unknown')
        ssp = shared_data_dict.get('ssp_path', 'unknown')
        year = shared_data_dict['year']
        base_temp = "/tmp" if os.path.exists("/tmp") else tempfile.gettempdir()
        _worker_hourly_dir = os.path.join(
            base_temp,
            f"energy_hourly_{model}_{ssp}_{year}_pid{worker_id}"
        )
        os.makedirs(_worker_hourly_dir, exist_ok=True)
        logger.info(f"[进程 {worker_id}] 私有 hourly 目录: {_worker_hourly_dir}")

    except Exception as e:
        logger.error(f"[进程 {worker_id}] 初始化失败: {e}", exc_info=True)
        raise


class SharedClimateDataFromDict:
    """从字典重建共享气候数据（用于多进程）"""
    def __init__(self, data_dict):
        self.data = {}
        self.time_index = pd.DatetimeIndex(data_dict['time_index'])
        self._shm_refs = []
        self.lat_index = data_dict['lat_index']
        self.lon_index = data_dict['lon_index']
        self.year = data_dict['year']  # 添加年份属性
        
        # 提取该年份的时间索引（与self.data的维度匹配）
        # 由于SharedClimateData已经筛选了年份，time_index应该已经是该年份的数据
        # 但为了安全，我们再次筛选
        year_mask = self.time_index.year == self.year
        self.time_index_year = self.time_index[year_mask]
        
        # 从共享内存名称重建数组
        for var_name, shm_info in data_dict['data'].items():
            shm_name = shm_info['shm_name']
            shape = shm_info['shape']
            dtype = shm_info['dtype']
            
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            self._shm_refs.append(existing_shm)
            array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
            self.data[var_name] = array
    
    def get_nearest_point(self, lat, lon):
        """获取最近的网格点索引"""
        lat_idx = np.abs(self.lat_index - lat).argmin()
        lon_idx = np.abs(self.lon_index - lon).argmin()
        return lat_idx, lon_idx
    
    def get_data(self, lat, lon, start_date, end_date):
        """获取指定位置和时间范围的数据"""
        func_start = time.perf_counter()
        
        try:
            # 步骤1: 获取最近点索引
            lat_idx, lon_idx = self.get_nearest_point(lat, lon)
            
            # 步骤2: 规范化start_date
            if isinstance(start_date, datetime):
                start_date = pd.Timestamp(start_date.date())
            elif isinstance(start_date, np.datetime64):
                start_date = pd.Timestamp(start_date)
                start_date = pd.Timestamp(start_date.date())
            else:
                start_date = pd.Timestamp(start_date)
                if hasattr(start_date, 'date'):
                    start_date = pd.Timestamp(start_date.date())
            
            # 步骤3: 规范化end_date
            if isinstance(end_date, datetime):
                end_date = pd.Timestamp(end_date.date())
            elif isinstance(end_date, np.datetime64):
                end_date = pd.Timestamp(end_date)
                end_date = pd.Timestamp(end_date.date())
            else:
                end_date = pd.Timestamp(end_date)
                if hasattr(end_date, 'date'):
                    end_date = pd.Timestamp(end_date.date())
            
            # 步骤4: 规范化time_index_year
            time_index_norm = pd.to_datetime([pd.Timestamp(t).date() for t in self.time_index_year])
            
            # 步骤5: 创建时间mask
            time_mask = (time_index_norm >= start_date) & (time_index_norm <= end_date)
            mask_sum = time_mask.sum()
            
            # 步骤6: 提取数据（可能耗时最长）
    
 

            temperature = self.data['tas'][time_mask, lat_idx, lon_idx] - 273.15

            radiation = self.data['rsds'][time_mask, lat_idx, lon_idx]

 
            uas_data = self.data['uas'][time_mask, lat_idx, lon_idx]
            vas_data = self.data['vas'][time_mask, lat_idx, lon_idx]
            wind_speed = np.sqrt(uas_data ** 2 + vas_data ** 2)

            humidity = self.data['huss'][time_mask, lat_idx, lon_idx] * 1000
 
            result = {
                'temperature': temperature,
                'radiation_global_horizontal': radiation,
                'wind_speed_2m': wind_speed,
                'humidity': humidity
            }
            
            total_duration = time.perf_counter() - func_start
            return result
            
        except Exception as e:
            total_duration = time.perf_counter() - func_start
            logger.error(f"[SharedClimateDataFromDict.get_data] 函数出错, 耗时 {total_duration:.4f}s, 错误: {str(e)}", exc_info=True)
            raise


def process_batch(batch_points, model_name, ssp_path, year):
    """处理一批（适配流式 hourly）"""
    import os
    global _global_shared_data, _global_population_df
    worker_id = os.getpid()
    batch_results = []

    for point_data in batch_points:
        res = process_point(point_data, _global_shared_data, _global_population_df, model_name, ssp_path, year)
        if res:
            batch_results.append(res)
    return batch_results


def process_single_year(model_name, ssp_path, year):
    """处理单个年份（含 hourly Parquet 合并）"""
    logger.info(f"  [初始化] {year} 年数据准备中...")
    
    # === 文件路径检查（同原逻辑）===
    heat_wave_file = os.path.join(HEAT_WAVE_BASE_PATH, model_name, ssp_path, f"{year}_all_heat_wave.csv")
    if not os.path.exists(heat_wave_file):
        logger.warning(f"  [跳过] 热浪文件不存在: {heat_wave_file}")
        return
    
    output_dir = os.path.join(HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", str(year), "point")
    os.makedirs(output_dir, exist_ok=True)
    
    future_dir = os.path.join(CLIMATE_BASE_PATH, model_name, "future", ssp_path)
    if not os.path.exists(future_dir):
        logger.warning(f"  [跳过] 气候数据目录不存在: {future_dir}")
        return
    
    # === 读取数据（同原逻辑）===
    points_df = pd.read_csv(heat_wave_file)
    points_df = points_df[points_df['Duration'] >= 3]
    if len(points_df) == 0:
        logger.warning(f"  [跳过] 无有效热浪事件")
        return
    
    grouped_points = []
    for (lat, lon), group in points_df.groupby(['lat', 'lon']):
        grouped_points.append({
            'lat': lat, 'lon': lon,
            'heat_waves': group.to_dict('records')
        })
    logger.info(f"  [统计] 共 {len(grouped_points)} 个点需要处理")
    
    # === 人口数据 ===
    pop_map = {"SSP126": "SSP1_population.csv", "SSP245": "SSP2_population.csv"}
    pop_file = os.path.join(POPULATION_BASE_PATH, pop_map[ssp_path])
    population_df = pd.read_csv(pop_file)
    
    # === 共享数据 ===
    logger.info(f"  [加载] {year} 年气候数据到共享内存...")
    shared_data = SharedClimateData(future_dir, model_name, ssp_path, year)
    logger.info(f"  [完成] 气候数据加载完成")
    
    # === 构建 shared_data_dict（+ 传 model/ssp 供 hourly 目录使用）===
    shared_data_dict = {
        'time_index': shared_data.time_index.values,
        'lat_index': shared_data.lat_index,
        'lon_index': shared_data.lon_index,
        'year': year,
        'model_name': model_name,   # 新增
        'ssp_path': ssp_path,       # 新增
        'data': {}
    }
    for var, (shm, arr) in shared_data.data.items():
        shared_data_dict['data'][var] = {'shm_name': shm.name, 'shape': arr.shape, 'dtype': arr.dtype}
    
    # === 分批 ===
    total = len(grouped_points)
    batch_size = max(80, min(200, total // (NUM_PROCESSES * 4)))
    batches = [grouped_points[i:i+batch_size] for i in range(0, total, batch_size)]
    logger.info(f"  [并行] 使用 {NUM_PROCESSES} 进程，批次大小={batch_size}，批次数={len(batches)}")
    
    # === 并行处理 ===
    parallel_start = time.perf_counter()
    results = []
    try:
        with multiprocessing.Pool(
            processes=NUM_PROCESSES,
            initializer=init_worker,
            initargs=(shared_data_dict, population_df.to_dict('list'))
        ) as pool:
            # 使用 partial 固定 model/ssp/year
            batch_func = partial(process_batch, 
                               model_name=model_name, 
                               ssp_path=ssp_path, 
                               year=year)
            
            with tqdm(total=len(batches), desc=f"  [{ssp_path}-{year}]") as pbar:
                for batch_res in pool.imap_unordered(batch_func, batches, chunksize=max(1, len(batches)//(NUM_PROCESSES*2))):
                    if batch_res:
                        results.extend(batch_res)
                    pbar.update(1)
                    
        logger.info(f"  [✓] 并行完成，耗时 {time.perf_counter() - parallel_start:.1f}s，{len(results)} 点")
    except Exception as e:
        logger.error(f"并行失败: {e}", exc_info=True)
        shared_data.cleanup()
        raise
    
    # === 保存总能耗 CSV（同原逻辑）===
    case_data = {}
    for res in results:
        if not res: continue
        lat, lon = res['lat'], res['lon']
        for case, vals in res['results'].items():
            case_data.setdefault(case, []).append({
                'lat': lat, 'lon': lon,
                'total_demand': vals['total_demand'],
                'heating_demand': vals['heating_demand'],
                'cooling_demand': vals['cooling_demand']
            })
    
    for case, data_list in case_data.items():
        df = pd.DataFrame(data_list)
        case_dir = os.path.join(output_dir, case)
        os.makedirs(case_dir, exist_ok=True)
        df.to_csv(os.path.join(case_dir, f"{case}.csv"), index=False)
    logger.info(f"  [✓] 总能耗 CSV 写入完成：{len(case_data)} cases")
    
    # === 【关键】合并 hourly Parquet ===
    logger.info(f"  [(hourly)] 合并各进程临时 Parquet...")
    temp_dirs = [os.path.join("/tmp", d) for d in os.listdir("/tmp")
                 if d.startswith(f"energy_hourly_{model_name}_{ssp_path}_{year}_pid")]
    
    if not temp_dirs:
        logger.warning("    ❗ 未找到任何 hourly 临时目录（可能无数据或路径变更）")
    else:
        for case in [f'case{i}' for i in range(1, 21)] + ['ref']:
            tables = []
            # 收集所有进程的该 case parquet
            for temp_dir in temp_dirs:
                parquet = os.path.join(temp_dir, f"{case}.parquet")
                if os.path.exists(parquet):
                    try:
                        tables.append(pq.read_table(parquet))
                    except Exception as e:
                        logger.error(f"读取 {parquet} 失败: {e}")
            if tables:
                # 合并 → 写入正式路径
                merged = pa.concat_tables(tables)
                num_rows = len(merged)  # 在删除前保存行数
                case_dir = os.path.join(output_dir, case)
                os.makedirs(case_dir, exist_ok=True)
                final_path = os.path.join(case_dir, f"{case}_hourly.parquet")
                pq.write_table(merged, final_path, compression='snappy')
                size_mb = os.path.getsize(final_path) / 1024 / 1024
                del tables, merged
                gc.collect()
                logger.info(f"    ✓ {case}_hourly.parquet ({num_rows} 行, {size_mb:.1f} MB)")
            else:
                logger.debug(f"    ⚪ {case} 无 hourly 数据")
        
        # 清理临时目录
        for d in temp_dirs:
            try:
                import shutil
                shutil.rmtree(d, ignore_errors=True)
            except:
                pass
        logger.info(f"  [✓] hourly 临时目录清理完成")
    
    # === 清理 ===
    shared_data.cleanup()
    del results, case_data
    gc.collect()
    logger.info(f"  [✓✓✓] {model_name}-{ssp_path}-{year} 全部完成\n")

def main():
    """主函数"""
    try:
        logger.info("=== 开始热浪期间能耗计算 ===")
        logger.info(f"支持的模型: {', '.join(MODELS)}")
        logger.info(f"SSP路径: {', '.join(SSP_PATHS)}")
        logger.info(f"目标年份: {TARGET_YEARS}")
        
        # 检查人口文件目录
        if not os.path.exists(POPULATION_BASE_PATH):
            raise FileNotFoundError(f"人口文件目录不存在: {POPULATION_BASE_PATH}")
        
        # 检查所有SSP对应的人口文件是否存在
        ssp_to_population = {
            "SSP126": "SSP1_population.csv",
            "SSP245": "SSP2_population.csv"
        }
        for ssp_path in SSP_PATHS:
            if ssp_path in ssp_to_population:
                pop_file = os.path.join(POPULATION_BASE_PATH, ssp_to_population[ssp_path])
                if not os.path.exists(pop_file):
                    logger.warning(f"人口文件不存在: {pop_file}，将在处理时跳过")
        
        # 计算总任务数
        total_tasks = len(MODELS) * len(SSP_PATHS) * len(TARGET_YEARS)
        current_task = 0
        
        # 处理每个模型
        for model_name in MODELS:
            logger.info(f"\n{'='*80}")
            logger.info(f"处理模型: {model_name}")
            logger.info(f"{'='*80}")
            
            # 处理每个SSP路径
            for ssp_path in SSP_PATHS:
                logger.info(f"\n>>> 当前发展路径: {ssp_path}")
                
                # 处理每个年份
                for year in TARGET_YEARS:
                    current_task += 1
                    logger.info(f"\n[{current_task}/{total_tasks}] 正在处理: {model_name} - {ssp_path} - {year} 年")
                    try:
                        process_single_year(model_name, ssp_path, year)
                    except Exception as e:
                        logger.error(f"处理 {model_name} - {ssp_path} - {year} 年时出错: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        logger.warning("继续处理下一个年份...")
                        continue
        
        logger.info("\n=== 所有处理完成 ===")
    
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise
    finally:
        gc.collect()


if __name__ == "__main__":
    main()

