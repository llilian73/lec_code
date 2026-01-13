import pandas as pd
import xarray as xr
import glob
import os
from HWMID import Md
# from HWMID_V2 import Md_dask,Md
import re
import numpy as np
from datetime import datetime
from tqdm import tqdm


def process_point_data():
    # 如果输出目录不存在则创建
    output_dir = "output_2030"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建或打开输出CSV文件
    output_file = os.path.join(output_dir, "2030_heat_wave_west.csv")
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            f.write("lat,lon,HWMID,start_day,date,Duration\n")

    # 获取所有CSV文件
    csv_files = glob.glob("output_v10/point_lat*_lon*.csv")
    print(f"找到 {len(csv_files)} 个文件需要处理")

    def extract_coords(filename):
        """
        从文件名中提取经纬度信息
        参数:
            filename: 文件名
        返回:
            lat, lon: 纬度和经度值
        """
        lat_match = re.search(r'lat([-\d.]+)_', filename)
        lon_match = re.search(r'lon([-\d.]+)\.', filename)
        if lat_match and lon_match:
            return float(lat_match.group(1)), float(lon_match.group(1))
        else:
            raise ValueError(f"无法从文件名提取经纬度: {filename}")

    def day_of_year_to_date(day_of_year, year=2030):
        """
        将一年中的第几天转换为具体日期
        参数:
            day_of_year: 一年中的第几天
            year: 年份，默认为2030
        返回:
            str: 格式为 'M/D' 的日期字符串
        """
        date = datetime(year, 1, 1) + pd.Timedelta(days=int(day_of_year) - 1)
        return f"{date.month}/{date.day}"

    def process_single_file(file):
        """
        处理单个文件的函数
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(file)
            df['time'] = pd.to_datetime(df['time'])

            # 获取经纬度坐标
            lat, lon = extract_coords(file)

            # 创建DataArray对象
            da = xr.DataArray(
                data=df['tasmax'].values,
                dims=['time'],
                coords={
                    'time': df['time'],
                    'lat': lat,
                    'lon': lon
                }
            )

            # 使用1981-2010年作为参考期计算2030年的Md值
            # Test
            daily_magnitudes = Md(
                da,
                quantile=0.90,
                win_size=31,
                n_days=3,
                ref_period=slice('1981', '2010')
            )

            # 只选择2030年的数据
            magnitudes_2030 = daily_magnitudes.sel(time='2030')

            # 计算热浪事件分组
            eps = 1e-5  # 用于表示零值的阈值
            heatwave_groups = (magnitudes_2030 < eps).cumsum('time')

            # 计算每个热浪事件的累积强度
            grouped = magnitudes_2030.groupby(heatwave_groups)
            heatwave_magnitude = grouped.apply(lambda x: x.cumsum('time'))

            # 找出最大HWMID值及其对应的热浪事件
            max_hwmid = heatwave_magnitude.max().item()

            # 找出具有最大强度的热浪组
            max_group_mask = heatwave_magnitude == max_hwmid
            max_group_idx = heatwave_groups.where(max_group_mask, drop=True).values[0]

            # 获取该热浪事件的所有数据
            heatwave_event = magnitudes_2030.where(heatwave_groups == max_group_idx, drop=True)

            # 计算热浪开始日期和持续时间
            start_day = heatwave_event.time.dt.dayofyear.values[0]
            duration = len(heatwave_event.time)

            # 将start_day转换为具体日期
            start_date = day_of_year_to_date(start_day)

            return lat, lon, max_hwmid, start_day, start_date, duration

        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
            return None

    # 使用tqdm创建进度条，直接处理所有文件
    for file in tqdm(csv_files, desc="处理文件"):
        result = process_single_file(file)
        if result is not None:
            lat, lon, max_hwmid, start_day, start_date, duration = result
            # 将结果写入文件
            with open(output_file, 'a') as f:
                f.write(f"{lat},{lon},{max_hwmid:.2f},{start_day},{start_date},{duration}\n")
            tqdm.write(f"已完成: 纬度={lat}, 经度={lon}, HWMID={max_hwmid:.2f}, 日期={start_date}, 持续={duration}天")


if __name__ == "__main__":
    process_point_data()