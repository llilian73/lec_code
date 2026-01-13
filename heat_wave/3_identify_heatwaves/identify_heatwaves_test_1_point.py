import pandas as pd
import xarray as xr
import os
import re
import numpy as np
from datetime import datetime
import gc
import logging
import warnings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 输入文件路径
input_csv_file = r"Z:\local_environment_creation\heat_wave\GCM_input_filter\point_lat35.000_lon102.000.csv"
target_year = 2030


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
    # 转换为numpy数组以便索引
    is_hot_array = is_hot.values

    # 识别连续的热浪事件
    heatwave_events = []
    current_event = None

    for i in range(len(da)):
        if is_hot_array[i]:
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


def process_single_point(csv_file, target_year):
    """
    处理单个坐标点的热浪识别
    
    参数:
        csv_file: 数据CSV文件路径（包含参考期和目标年份数据）
        target_year: 目标年份
    
    返回:
        热浪事件列表
    """
    try:
        # 从CSV文件名中提取经纬度
        lat, lon = extract_coords(csv_file)
        logger.info(f"处理坐标点: ({lat}, {lon})")

        # 读取所有数据（从CSV文件）
        logger.info(f"读取数据文件: {csv_file}")
        all_data = pd.read_csv(csv_file)
        all_data['time'] = pd.to_datetime(all_data['time'])
        
        # 筛选参考期数据（1981-2010年）
        ref_data = all_data[(all_data['time'].dt.year >= 1981) & (all_data['time'].dt.year <= 2010)]
        logger.info(f"参考期数据统计 - 温度范围: {ref_data['tasmax'].min():.2f} - {ref_data['tasmax'].max():.2f}")

        # 计算阈值
        logger.info("计算参考期阈值...")
        threshold = calculate_threshold(ref_data)
        logger.info(f"阈值范围: {threshold.min().values:.2f} - {threshold.max().values:.2f}")

        # 从CSV文件中筛选目标年份的数据
        target_data_df = all_data[all_data['time'].dt.year == target_year]
        if len(target_data_df) == 0:
            raise ValueError(f"CSV文件中未找到 {target_year} 年的数据")
        
        target_data = target_data_df['tasmax'].values
        logger.info(f"目标年份数据统计 - 温度范围: {target_data.min():.2f} - {target_data.max():.2f}")
        logger.info(f"成功提取 {target_year} 年 {len(target_data)} 天的数据")

        # 识别所有热浪事件
        logger.info("识别热浪事件...")
        heatwave_events = identify_heatwaves(target_data, threshold)

        if not heatwave_events:
            logger.info(f"未找到热浪事件")
            return []

        # 为每个热浪事件生成结果
        results = []
        for i, event in enumerate(heatwave_events, 1):
            start_day = event['start_day']
            duration = event['duration']
            start_date = day_of_year_to_date(start_day, year=target_year)
            results.append({
                'lat': lat,
                'lon': lon,
                'number': i,
                'start_day': start_day,
                'date': start_date,
                'Duration': duration
            })

        logger.info(f"找到 {len(results)} 个热浪事件")
        return results

    except Exception as e:
        logger.error(f"处理文件时出错: {str(e)}")
        raise


def main():
    try:
        logger.info("=== 开始处理热浪识别 ===")
        
        # 检查输入文件是否存在
        if not os.path.exists(input_csv_file):
            raise FileNotFoundError(f"输入CSV文件不存在: {input_csv_file}")
        
        # 获取当前代码文件所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, f"output_{target_year}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file = os.path.join(output_dir, f"{target_year}_heat_wave_lat35.000_lon102.000.csv")
        
        # 处理单个坐标点
        logger.info(f"\n--- 开始处理目标年份: {target_year} ---")
        results = process_single_point(input_csv_file, target_year)
        
        # 保存结果
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_csv(output_file, index=False)
            logger.info(f"\n处理完成，找到 {len(results)} 个热浪事件")
            logger.info(f"结果已保存至: {output_file}")
        else:
            logger.info(f"\n处理完成，未找到热浪事件")
            # 创建空文件
            df_empty = pd.DataFrame(columns=['lat', 'lon', 'number', 'start_day', 'date', 'Duration'])
            df_empty.to_csv(output_file, index=False)
            logger.info(f"空结果已保存至: {output_file}")
        
        logger.info("=== 处理完成 ===")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise
    finally:
        # 清理内存
        gc.collect()


if __name__ == "__main__":
    main()
