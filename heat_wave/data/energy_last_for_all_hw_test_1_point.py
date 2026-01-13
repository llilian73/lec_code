"""
热浪期间能耗计算工具（单点测试版本）

功能：
读取指定点的热浪事件CSV文件，计算该点热浪期间的能耗。

输入数据：
1. 热浪事件CSV文件：Z:\local_environment_creation\heat_wave\output\output_2030\2030_heat_wave_lat35.000_lon102.000.csv
2. 未来气候数据：Z:\local_environment_creation\heat_wave\GCM_input_filter\
3. 人口数据：需要提供人口数据或使用默认值

输出数据：
- 路径：当前目录下的output文件夹
- 每个case一个CSV文件：{case_name}.csv
- 包含列：lat, lon, total_demand, heating_demand, cooling_demand
"""

import pandas as pd
import numpy as np
import xarray as xr
import os
import sys
from datetime import datetime, timedelta
import time
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

# 配置路径（Windows路径）
HEAT_WAVE_FILE = r"Z:\local_environment_creation\heat_wave\output\output_2030\2030_heat_wave_lat35.000_lon102.000.csv"
CLIMATE_BASE_PATH = r"Z:\local_environment_creation\heat_wave\GCM_input_filter"
OUTPUT_DIR = "Z:\local_environment_creation\heat_wave\GCM_input_filter\output"

# 目标点坐标
TARGET_LAT = 35.000
TARGET_LON = 102.000

# 模型配置
MODEL_NAME = "BCC-CSM2-MR"
SSP_PATH = "ssp126"
YEAR = 2030

# 气候变量
CLIMATE_VARIABLES = ['tas', 'rsds', 'huss', 'uas', 'vas']

# 默认人口（如果找不到人口数据，使用此值）
DEFAULT_POPULATION = 100000


def find_climate_file(climate_dir, variable):
    """查找气候数据文件"""
    # 文件名格式：{变量名}_day_BCC-CSM2-MR_ssp126_r1i1p1f1_gn_20300101-20341231_interpolated_1deg.nc
    filename = f"{variable}_day_{MODEL_NAME}_{SSP_PATH}_r1i1p1f1_gn_20300101-20341231_interpolated_1deg.nc"
    file_path = os.path.join(climate_dir, filename)
    
    if os.path.exists(file_path):
        return file_path
    else:
        # 尝试查找类似的文件
        all_files = [f for f in os.listdir(climate_dir) if f.endswith(".nc") and variable in f]
        matching_files = [f for f in all_files if MODEL_NAME.replace("-", "_") in f or MODEL_NAME in f]
        
        if len(matching_files) == 0:
            raise FileNotFoundError(f"未找到 {variable} 变量文件: {climate_dir}")
        elif len(matching_files) == 1:
            return os.path.join(climate_dir, matching_files[0])
        else:
            # 选择包含20300101-20341231的文件
            for f in matching_files:
                if "20300101" in f and "20341231" in f:
                    return os.path.join(climate_dir, f)
            logger.warning(f"找到多个匹配文件，使用第一个: {matching_files[0]}")
            return os.path.join(climate_dir, matching_files[0])


class ClimateData:
    """气候数据类（按年份加载）"""
    def __init__(self, climate_dir, year):
        self.climate_dir = climate_dir
        self.year = year
        self.data = {}
        self.time_index = None
        self.lat_index = None
        self.lon_index = None
        self.file_cache = {}  # 缓存已打开的文件
        self.load_year_data()
    
    def load_year_data(self):
        """加载指定年份的气候数据"""
        logger.info(f"正在加载 {MODEL_NAME} {SSP_PATH} {self.year} 年的气候数据...")
        start_time = time.time()
        
        # 查找所有变量的文件
        var_files = {}
        for var_name in CLIMATE_VARIABLES:
            file_path = find_climate_file(self.climate_dir, var_name)
            var_files[var_name] = file_path
        
        # 打开第一个文件获取时间和坐标信息
        first_file = var_files['tas']
        if first_file not in self.file_cache:
            self.file_cache[first_file] = xr.open_dataset(first_file)
        
        ds = self.file_cache[first_file]
        
        # 处理cftime时间对象
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
        
        # 加载所有变量的数据
        for var_name, file_path in var_files.items():
            if file_path not in self.file_cache:
                self.file_cache[file_path] = xr.open_dataset(file_path)
            
            ds_var = self.file_cache[file_path]
            
            # 提取该年份的数据
            year_data = ds_var[var_name].isel(time=year_mask).load()
            self.data[var_name] = year_data.values
        
        logger.info(f"气候数据加载完成，耗时 {time.time() - start_time:.2f} 秒")
    
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
        
        # 确保time_index_year也是日期精度
        time_index_year = pd.to_datetime([pd.Timestamp(t).date() for t in self.time_index_year])
        
        # 使用该年份的时间索引来创建mask（与self.data的维度匹配）
        time_mask = (time_index_year >= start_date) & (time_index_year <= end_date)
        
        # 调试：打印时间筛选信息
        logger.info(f"[DATA_TEST.PY] 时间筛选: start_date={start_date}, end_date={end_date}")
        logger.info(f"[DATA_TEST.PY] time_index_year范围: {time_index_year[0]} 到 {time_index_year[-1]}")
        logger.info(f"[DATA_TEST.PY] 时间mask匹配的天数: {np.sum(time_mask)}, 期望天数: {(end_date - start_date).days + 1}")
        
        return {
            'temperature': self.data['tas'][time_mask, lat_idx, lon_idx] - 273.15,  # 转换为摄氏度
            'radiation_global_horizontal': self.data['rsds'][time_mask, lat_idx, lon_idx],
            'wind_speed_2m': np.sqrt(
                self.data['uas'][time_mask, lat_idx, lon_idx] ** 2 +
                self.data['vas'][time_mask, lat_idx, lon_idx] ** 2
            ),
            'humidity': self.data['huss'][time_mask, lat_idx, lon_idx] * 1000  # 转换为g/kg
        }
    
    def cleanup(self):
        """清理文件"""
        for ds in self.file_cache.values():
            try:
                ds.close()
            except Exception:
                pass


def load_climate_data(lat, lon, start_date, duration, climate_data):
    """加载指定点和时间范围的气候数据"""
    duration = int(duration)
    end_date = start_date + timedelta(days=duration - 1)
    
    # 规范化日期（去掉时间部分），确保与get_data中的处理一致
    start_date_norm = pd.Timestamp(start_date.date())
    end_date_norm = pd.Timestamp(end_date.date())
    
    # 获取数据
    data = climate_data.get_data(lat, lon, start_date_norm, end_date_norm)
    
    # 获取实际的时间索引（规范化為日期精度）
    time_index_year_norm = pd.to_datetime([pd.Timestamp(t).date() for t in climate_data.time_index_year])
    time_mask = (time_index_year_norm >= start_date_norm) & (time_index_year_norm <= end_date_norm)
    actual_time_index = time_index_year_norm[time_mask]
    
    # 使用实际时间戳作为索引（与文件1保持一致）
    weather_df = pd.DataFrame(data, index=actual_time_index)
    
    return weather_df


def calculate_energy_demand(weather_df, population):
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


def process_point(lat, lon, heat_waves, climate_data, population, output_dir, year):
    """处理单个点的所有热浪事件，计算总能耗"""
    try:
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
            'case9': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case10': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case11': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case12': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case13': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case14': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case15': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case16': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case17': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case18': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case19': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0},
            'case20': {'total_demand': 0, 'heating_demand': 0, 'cooling_demand': 0}
        }
        
        # 计算每个热浪事件的能耗并累加
        for heat_wave in heat_waves:
            # 解析开始日期（格式：month/day）
            month, day = map(int, heat_wave['date'].split('/'))
            start_date = datetime(year, month, day)
            duration = int(heat_wave['Duration'])
            
            # 加载气候数据
            weather_df = load_climate_data(lat, lon, start_date, duration, climate_data)
            
            # 计算能耗
            results = calculate_energy_demand(weather_df, population)
            
            # 累加结果
            for case_name, case_result in results.items():
                total_results[case_name]['total_demand'] += case_result['total_demand'].sum()
                total_results[case_name]['heating_demand'] += case_result['heating_demand'].sum()
                total_results[case_name]['cooling_demand'] += case_result['cooling_demand'].sum()
        
        # 保存结果到CSV
        for case_name, case_result in total_results.items():
            case_dir = os.path.join(output_dir, case_name)
            os.makedirs(case_dir, exist_ok=True)
            
            csv_data = {
                'lat': [lat],
                'lon': [lon],
                'total_demand': [case_result['total_demand']],
                'heating_demand': [case_result['heating_demand']],
                'cooling_demand': [case_result['cooling_demand']]
            }
            csv_df = pd.DataFrame(csv_data)
            csv_file = os.path.join(case_dir, f"{case_name}.csv")
            
            # 追加模式
            if os.path.exists(csv_file):
                csv_df.to_csv(csv_file, mode='a', header=False, index=False)
            else:
                csv_df.to_csv(csv_file, index=False)
        
        return total_results
    
    except Exception as e:
        logger.error(f"处理点 ({lat}, {lon}) 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None




def main():
    """主函数"""
    try:
        logger.info("=== 开始热浪期间能耗计算（单点测试） ===")
        logger.info(f"模型: {MODEL_NAME}")
        logger.info(f"SSP路径: {SSP_PATH}")
        logger.info(f"目标年份: {YEAR}")
        logger.info(f"目标点: lat={TARGET_LAT}, lon={TARGET_LON}")
        
        # 检查热浪文件
        if not os.path.exists(HEAT_WAVE_FILE):
            raise FileNotFoundError(f"热浪文件不存在: {HEAT_WAVE_FILE}")
        
        # 检查气候数据目录
        if not os.path.exists(CLIMATE_BASE_PATH):
            raise FileNotFoundError(f"气候数据目录不存在: {CLIMATE_BASE_PATH}")
        
        # 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 读取热浪数据
        logger.info(f"读取热浪数据: {HEAT_WAVE_FILE}")
        heat_wave_df = pd.read_csv(HEAT_WAVE_FILE)
        heat_wave_df = heat_wave_df[heat_wave_df['Duration'] >= 3]  # 只保留持续时间>=3天的热浪
        
        if len(heat_wave_df) == 0:
            logger.warning("没有有效的热浪事件")
            return
        
        # 过滤目标点的数据
        point_heat_waves = heat_wave_df[
            (heat_wave_df['lat'] == TARGET_LAT) & 
            (heat_wave_df['lon'] == TARGET_LON)
        ]
        
        if len(point_heat_waves) == 0:
            logger.warning(f"未找到点 ({TARGET_LAT}, {TARGET_LON}) 的热浪事件")
            return
        
        logger.info(f"找到 {len(point_heat_waves)} 个热浪事件")
        
        # 转换为字典列表
        heat_waves = point_heat_waves.to_dict('records')
        
        # 加载气候数据
        climate_data = ClimateData(CLIMATE_BASE_PATH, YEAR)
        
        # 获取人口数据（使用默认值或从文件读取）
        population = DEFAULT_POPULATION
        logger.info(f"使用人口数据: {population}")
        
        # 处理该点
        logger.info(f"开始处理点 ({TARGET_LAT}, {TARGET_LON})")
        result = process_point(
            TARGET_LAT, 
            TARGET_LON, 
            heat_waves, 
            climate_data, 
            population, 
            OUTPUT_DIR, 
            YEAR
        )
        
        if result:
            logger.info("处理完成！")
            logger.info("结果摘要:")
            for case_name, case_result in result.items():
                logger.info(f"  {case_name}: total={case_result['total_demand']:.2f}, "
                          f"heating={case_result['heating_demand']:.2f}, "
                          f"cooling={case_result['cooling_demand']:.2f}")
        else:
            logger.error("处理失败")
        
        # 清理资源
        climate_data.cleanup()
        
        logger.info("\n=== 处理完成 ===")
    
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

