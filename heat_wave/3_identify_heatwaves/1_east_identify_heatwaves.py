import pandas as pd
import xarray as xr
import glob
import os
import re
import numpy as np
from datetime import datetime
from tqdm import tqdm
import multiprocessing
import gc
import logging
from multiprocessing import shared_memory
import time
import warnings


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SharedClimateData:
    def __init__(self, target_year_str):
        self.shm = None
        self.data = None
        self.time_index = None
        self.lat_index = None
        self.lon_index = None
        self.target_year_str = target_year_str
        self.load_data()

    def load_data(self):
        """预加载气候数据到共享内存"""
        logger.info("Loading climate data...")
        start_time = time.time()

        # 定义文件路径 未来数据
        climate_base_path = r"Y:\CMIP6\future"
        self.tasmax_file = os.path.join(climate_base_path,
                                        f"tasmax_day_EC-Earth3-HR_ssp245_r1i1p1f1_gr_{self.target_year_str}0101-{self.target_year_str}1231.nc")

        # 加载数据
        with xr.open_dataset(self.tasmax_file) as ds:
            self.time_index = pd.to_datetime(ds.time.values)
            self.lat_index = ds.lat.values
            self.lon_index = ds.lon.values

            # 创建共享内存
            data = ds['tasmax'].values
            shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
            shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
            shared_array[:] = data[:]
            self.data = (shm, shared_array)

        logger.info(f"Climate data loaded in {time.time() - start_time:.2f} seconds")

    def get_nearest_point(self, lat, lon):
        """获取最近的网格点索引"""
        lat_idx = np.abs(self.lat_index - lat).argmin()
        lon_idx = np.abs(self.lon_index - lon).argmin()
        return lat_idx, lon_idx

    def get_data(self, lat, lon):
        """获取指定位置的数据"""
        lat_idx, lon_idx = self.get_nearest_point(lat, lon)
        return self.data[1][:, lat_idx, lon_idx]

    def cleanup(self):
        """清理共享内存"""
        if self.data is not None:
            self.data[0].close()
            self.data[0].unlink()


def extract_coords(filename):
    lat_match = re.search(r'lat([-\d.]+)_', filename)
    lon_match = re.search(r'lon([-\d.]+)\.', filename)
    if lat_match and lon_match:
        return float(lat_match.group(1)), float(lon_match.group(1))
    else:
        raise ValueError(f"无法从文件名提取经纬度: {filename}")


def day_of_year_to_date(day_of_year, year):
    date = datetime(year, 1, 1) + pd.Timedelta(days=int(day_of_year) - 1)
    return f"{date.month}/{date.day}"


def calculate_threshold(ref_data, win_size=31, quantile=0.90):
    """计算参考期的阈值"""
    # 将数据转换为DataArray，确保使用正确的变量名
    ref_data = xr.DataArray(
        ref_data['tasmax'].values,
        coords={'time': ref_data['time']},
        dims=['time']
    )

    # 计算每个dayofyear的阈值
    threshold_list = []
    for day in range(1, 367):
        plusminus = win_size // 2
        valid_days = (np.arange(day - plusminus - 1, day + plusminus) % 366) + 1
        window = ref_data.time.dt.dayofyear.isin(valid_days)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN slice encountered')
            threshold = ref_data.where(window, drop=True).quantile(quantile, dim='time', skipna=False)
        threshold_list.append(threshold)

    threshold = xr.concat(threshold_list, dim='dayofyear')
    threshold = threshold.assign_coords(dayofyear=range(1, 367))
    return threshold


def identify_heatwaves(temperature_data, threshold, n_days=3):
    """识别热浪事件"""
    # 将温度数据转换为DataArray
    da = xr.DataArray(
        temperature_data,
        coords={'time': pd.date_range(start='2030-01-01', periods=len(temperature_data), freq='D')},
        dims=['time']
    )

    # 计算每天是否超过阈值
    is_hot = da.groupby('time.dayofyear') > threshold

    # 识别连续的热浪事件
    heatwave_events = []
    current_event = None

    for i in range(len(da)):
        if is_hot[i]:
            if current_event is None:
                current_event = {'start': i, 'count': 1}
            else:
                current_event['count'] += 1
        else:
            if current_event is not None and current_event['count'] >= n_days:
                start_day = int(da.time.dt.dayofyear[i - current_event['count']].values)  # 转换为整数
                heatwave_events.append({
                    'start_day': start_day,
                    'duration': current_event['count']
                })
            current_event = None

    # 检查最后一个事件
    if current_event is not None and current_event['count'] >= n_days:
        start_day = int(da.time.dt.dayofyear[len(da) - current_event['count']].values)  # 转换为整数
        heatwave_events.append({
            'start_day': start_day,
            'duration': current_event['count']
        })

    return heatwave_events


def process_single_file(file_info):
    file_path = file_info['file_path']
    shared_data = file_info['shared_data']
    target_year_str = file_info['target_year_str']
    target_year_int = int(target_year_str)

    try:
        # 从CSV文件名中提取经纬度
        lat, lon = extract_coords(file_path)

        # 读取参考期数据（从CSV文件）
        ref_data = pd.read_csv(file_path)
        ref_data['time'] = pd.to_datetime(ref_data['time'])

        # 打印参考期数据统计信息
        # logger.info(f"参考期数据统计 - 文件: {file_path}")
        # logger.info(f"参考期温度范围: {ref_data['tasmax'].min():.2f} - {ref_data['tasmax'].max():.2f}")

        # 计算阈值
        threshold = calculate_threshold(ref_data)
        # logger.info(f"阈值范围: {threshold.min().values:.2f} - {threshold.max().values:.2f}")

        # 使用共享数据获取目标年份的tasmax数据
        target_data = shared_data.get_data(lat, lon)
        # logger.info(f"目标年份温度范围: {target_data.min():.2f} - {target_data.max():.2f}")

        # 识别所有热浪事件
        heatwave_events = identify_heatwaves(target_data, threshold)

        if not heatwave_events:
            # logger.info(f"未找到热浪事件 - 文件: {file_path}")
            return None  # 没有热浪

        # 为每个热浪事件生成结果
        results = []
        for i, event in enumerate(heatwave_events, 1):
            start_day = event['start_day']
            duration = event['duration']
            start_date = day_of_year_to_date(start_day, year=target_year_int)
            results.append((lat, lon, i, start_day, start_date, duration))

        # logger.info(f"找到 {len(results)} 个热浪事件 - 文件: {file_path}")
        return results

    except Exception as e:
        logger.error(f"处理文件 {file_path} (目标年份: {target_year_str}) 时出错: {str(e)}")
        return None


def process_batch(batch_files_info):
    """处理一批文件"""
    batch_results = []
    for file_info in batch_files_info:
        results = process_single_file(file_info)
        if results is not None:
            batch_results.extend(results)
            # ogger.info(f"批次处理: 从文件 {file_info['file_path']} 添加了 {len(results)} 个结果")
    return batch_results


def main():
    # 定义要处理的目标年份
    target_years = ['2030']  # 可以添加更多年份

    csv_files = glob.glob(
        r"Y:\CMIP6\Energy consumption in heat wave code\Historical tasmax\east\point_lat*_lon*.csv")  # 作为参考期数据来源
    total_files = len(csv_files)
    logger.info(f"找到 {total_files} 个文件（点位）需要处理")

    # 获取CPU核心数并设置进程数
    num_cores = multiprocessing.cpu_count()
    num_processes = min(num_cores * 2, 32)  # 使用更多进程，但限制最大值为32
    logger.info(f"使用 {num_processes} 个进程进行并行处理")

    all_combined_results = []

    for target_year_str in target_years:
        logger.info(f"\n--- 开始处理目标年份: {target_year_str} ---")
        # 获取当前代码文件所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, f"output_{target_year_str}")  # 创建输出路径于脚本同目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f"{target_year_str}_all_heat_wave_east.csv")
        if not os.path.exists(output_file):
            with open(output_file, 'w') as f:
                f.write("lat,lon,number,start_day,date,Duration\n")

        # 创建共享气候数据
        shared_data = SharedClimateData(target_year_str)

        # 准备批次数据
        file_infos = [{
            'file_path': f,
            'shared_data': shared_data,
            'target_year_str': target_year_str
        } for f in csv_files]

        # 优化批次大小：根据总文件数和进程数动态调整
        batch_size = max(1, min(200, total_files // (num_processes * 4)))  # 增加批次大小
        batches = [file_infos[i:i + batch_size] for i in range(0, total_files, batch_size)]

        logger.info(f"数据分配情况 ({target_year_str}):")
        logger.info(f"- 总文件数: {total_files}")
        logger.info(f"- 批次大小: {batch_size}")
        logger.info(f"- 总批次数: {len(batches)}")

        # 创建进程池进行并行处理
        all_results_for_year = []
        with multiprocessing.Pool(processes=num_processes) as pool:
            # 使用tqdm显示进度，增加chunksize以提高效率
            chunksize = max(1, len(batches) // (num_processes * 4))
            for batch_results in tqdm(
                    pool.imap_unordered(process_batch, batches, chunksize=chunksize),
                    total=len(batches),
                    desc=f"处理批次 ({target_year_str})"
            ):
                if batch_results:  # 只处理非空结果
                    all_results_for_year.extend(batch_results)
                    # 每处理完一批就保存结果，避免内存占用过大
                    if len(all_results_for_year) >= batch_size * 2:
                        with open(output_file, 'a') as f:
                            for result in all_results_for_year:
                                lat, lon, number, start_day, start_date, duration = result
                                f.write(f"{lat},{lon},{number},{start_day},{start_date},{duration}\n")
                        all_results_for_year = []  # 清空结果列表

        # 保存剩余的结果
        if all_results_for_year:
            with open(output_file, 'a') as f:
                for result in all_results_for_year:
                    lat, lon, number, start_day, start_date, duration = result
                    f.write(f"{lat},{lon},{number},{start_day},{start_date},{duration}\n")

        logger.info(f"\n{target_year_str} 处理完成，找到 {len(all_results_for_year)} 个有效结果")
        logger.info(f"{target_year_str} 结果已保存至: {output_file}")
        all_combined_results.extend(all_results_for_year)

        # 清理共享内存
        shared_data.cleanup()

    # 清理内存
    gc.collect()
    logger.info("所有目标年份处理完成。")


if __name__ == "__main__":
    main()
