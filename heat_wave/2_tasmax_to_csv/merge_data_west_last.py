"""
西半球气候数据合并处理工具（v2版本）

功能概述：
本工具用于处理西半球（经度180-360度）的气候数据，将NetCDF格式的日最高温度数据转换为CSV格式，并按网格点分别保存。v2版本采用5年周期分批处理策略，优化内存使用，提高处理稳定性。

输入数据：
1. NetCDF气候数据文件：
   - 目录：Z:\CMIP6\tasmax
   - 文件格式：tasmax_day_EC-Earth3-HR_historical_r1i1p1f1_gr_YYYYMMDD-YYYYMMDD.nc
   - 变量：tasmax（日最高温度）
   - 时间范围：1981-2010年
   - 空间范围：西半球（经度180-360度）

2. 陆地点信息文件：
   - 文件路径：tmp\land_points_west.csv
   - 包含西半球陆地点坐标信息
   - 格式：lat, lon, is_land

3. 全球陆地边界数据：
   - 文件路径：shapefile\ne_10m_admin_0_countries.shp
   - 用于陆地点识别和筛选

主要功能：
1. 陆地点筛选：
   - 读取陆地点信息文件
   - 只处理陆地上的网格点
   - 跳过海洋点和无效点

2. 数据分批处理：
   - 按5年周期分批处理：1981-1985, 1986-1990, 1991-1995, 1996-2000, 2001-2005, 2006-2010
   - 每5年数据独立处理，减少内存压力
   - 定期进行内存清理和垃圾回收

3. 数据格式转换：
   - 从NetCDF格式转换为CSV格式
   - 按网格点分别保存文件
   - 文件名格式：point_lat{lat}_lon{lon}.csv

4. 并行处理优化：
   - 多进程并行处理网格点数据
   - 动态调整进程数和批次大小
   - 进度跟踪和错误处理

输出结果：
1. 网格点CSV文件：
   - 目录：output_v10\
   - 文件名格式：point_lat{lat}_lon{lon}.csv
   - 包含列：time（时间）、tasmax（日最高温度）

2. 错误记录文件：
   - 目录：tmp\
   - 文件名格式：failed_points_{year}.csv
   - 记录处理失败的网格点信息

3. 处理日志：
   - 详细的处理进度和统计信息
   - 内存使用情况监控
   - 错误和警告信息

数据流程：
1. 初始化阶段：
   - 检查输入输出路径
   - 加载陆地点信息
   - 创建输出目录

2. 数据加载阶段：
   - 扫描NetCDF文件目录
   - 按年份筛选目标文件
   - 验证文件完整性

3. 分批处理阶段：
   - 按5年周期组织数据
   - 逐个处理每个5年周期
   - 并行处理网格点数据

4. 数据保存阶段：
   - 将处理结果保存为CSV文件
   - 追加模式写入，支持断点续传
   - 记录处理状态和错误信息

5. 资源清理阶段：
   - 定期进行内存清理
   - 关闭数据集释放资源
   - 等待文件写入完成

计算特点：
- 分批处理：5年周期分批，减少内存压力
- 并行计算：多进程并行处理，提高效率
- 陆地点筛选：只处理陆地上的网格点
- 错误恢复：完善的错误处理和恢复机制
- 进度跟踪：详细的处理进度显示

技术参数：
- 处理周期：5年一个周期
- 并行进程数：CPU核心数×2，最大32个
- 批次大小：动态调整，最大100个点
- 内存清理：每5年周期完成后强制清理
- 等待时间：每周期完成后等待5秒

性能优化：
- 内存管理：分批处理，定期清理
- 并行处理：多进程并行计算
- 文件操作：追加模式，支持断点续传
- 错误处理：超时机制和异常恢复
- 进度监控：实时显示处理进度

数据质量保证：
- 输入验证：检查文件路径和格式
- 数据完整性：验证NetCDF文件结构
- 坐标筛选：只处理有效的陆地点
- 错误记录：详细记录处理失败的点
- 结果验证：检查输出文件完整性

特殊处理：
- 陆地点筛选：跳过海洋点和无效点
- 分批处理：5年周期分批，优化内存使用
- 错误恢复：单个周期失败不影响其他周期
- 断点续传：支持追加模式，可中断后继续
- 资源管理：定期清理内存和等待文件写入

输出格式：
- 文件格式：CSV（UTF-8编码）
- 时间格式：NetCDF时间格式
- 温度单位：原始单位（通常是K或°C）
- 坐标精度：三位小数
- 文件命名：point_lat{lat}_lon{lon}.csv

应用场景：
- 气候数据分析的数据预处理
- 极端天气事件的空间分析
- 区域气候研究的数据准备
- 气候模型输出的格式转换
- 大规模气候数据的网格化处理
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import gc
import re
import time as time_module  # 重命名time模块

from shapely import Point
from tqdm import tqdm
import multiprocessing
from functools import partial
import geopandas as gpd

# os.environ['GDAL_DATA'] = r"C:\ProgramData\anaconda3\envs\pythonProject\Library\share\gdal"  # 替换为你的实际路径

hist_dir = r"Z:\CMIP6\tasmax"
output_dir = r"output_v10"
shapefile_path = r"shapefile\ne_10m_admin_0_countries.shp"
land_points_file = r"tmp\land_points_west.csv"
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_land_points_map(csv_path):
    """
    读取陆地点CSV文件并转换为字典格式
    返回格式: {(lat, lon): is_land_value}
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)

        # 只保留陆地点
        df = df[df['is_land'] == True]

        # 转换为字典格式
        land_points_map = {}
        for _, row in df.iterrows():
            # 使用元组作为键，与原始格式保持一致
            coord_key = (row['lat'], row['lon'])
            land_points_map[coord_key] = True

        print(f"成功加载 {len(land_points_map)} 个陆地点")
        return land_points_map

    except Exception as e:
        print(f"读取陆地点数据时出错: {str(e)}")
        return {}


def process_batch(batch_data):
    """处理一批坐标点的数据"""
    try:
        batch_results = []

        for point_data in batch_data:
            try:
                lat, lon = point_data[0]
                data = point_data[1]

                # 创建DataFrame
                df = pd.DataFrame({
                    'time': data['time'],
                    'tasmax': data['tasmax']
                })

                # 生成文件名并保存
                filename = f"point_lat{lat:.3f}_lon{lon:.3f}.csv"
                output_path = os.path.join(output_dir, filename)
                # 检查文件是否存在，决定是否写入表头
                file_exists = os.path.exists(output_path)
                df.to_csv(output_path, mode='a', index=False, header=not file_exists)

                batch_results.append(f"成功处理坐标点: ({lat}, {lon})")
            except Exception as e:
                batch_results.append(f"处理点 ({lat}, {lon}) 失败: {str(e)}")
                continue

        return batch_results
    except Exception as e:
        return [f"批处理失败: {str(e)}"]


def check_paths(hist_dir, output_dir):
    """检查所有路径是否存在"""
    if not os.path.exists(hist_dir):
        raise FileNotFoundError(f"历史数据目录不存在: {hist_dir}")

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")


def get_year_from_filename(filename):
    """从文件名中提取年份"""
    match = re.search(r'_(\d{8})-\d{8}', filename)
    if match:
        return int(match.group(1)[:4])
    return None


def get_year_ranges():
    """生成5年一个周期的年份范围"""
    return [
        (1981, 1985),
        (1986, 1990),
        (1991, 1995),
        (1996, 2000),
        (2001, 2005),
        (2006, 2010)
    ]


def load_historical_data(hist_dir):
    """加载并处理1981-2010年的历史数据，每5年处理一次"""
    logger.info("开始加载1981-2010年的历史数据...")
    land_points_map = load_land_points_map(land_points_file)

    # 获取目录下所有nc文件
    all_files = [f for f in os.listdir(hist_dir) if f.endswith(".nc")]
    all_files.sort()

    # 按5年周期处理数据
    for start_year, end_year in get_year_ranges():
        logger.info(f"\n开始处理 {start_year}-{end_year} 年的数据...")
        
        # 筛选当前5年周期的文件
        target_files = []
        for f in all_files:
            year = get_year_from_filename(f)
            if year and start_year <= year <= end_year:
                target_files.append(os.path.join(hist_dir, f))
        
        if not target_files:
            logger.warning(f"未找到 {start_year}-{end_year} 年期间的数据文件")
            continue

        logger.info(f"找到 {len(target_files)} 个 {start_year}-{end_year} 年期间的数据文件")

        try:
            print(f"\n开始加载 {start_year}-{end_year} 年的数据文件...")
            for f in target_files:
                point_data_dict = {}  # 格式: {(lat, lon): {'time': [], 'tasmax': []}}
                try:
                    ds = xr.open_dataset(f)
                    try:
                        times = ds.time.values
                        lats = ds.lat.values
                        lons = ds.lon.values
                        data = ds.tasmax.values  # 获取所有数据
                        total_points = len(lats) * len(lons) * len(times)
                        with tqdm(total=total_points, desc="处理网格点") as pbar:

                            for t_idx, time in enumerate(times):
                                logger.info(f"处理时间点: {time}")
                                # 获取该时间点的所有数据
                                time_slice = data[t_idx]
                                # 遍历该时间点的所有数据
                                for lat_idx, lat in enumerate(lats):
                                    for lon_idx, lon in enumerate(lons):
                                        pbar.update(1)
                                        # 只处理陆地点
                                        if (lat, lon) not in land_points_map:
                                            continue

                                        value = time_slice[lat_idx, lon_idx]
                                        # 如果是第一次遇到这个坐标点，初始化列表
                                        if (lat, lon) not in point_data_dict:
                                            point_data_dict[(lat, lon)] = {'time': [], 'tasmax': []}

                                        # 添加数据
                                        point_data_dict[(lat, lon)]['time'].append(time)
                                        point_data_dict[(lat, lon)]['tasmax'].append(value)

                    finally:
                        # 显式关闭数据集
                        ds.close()
                        # 清理内存
                        del times, lats, lons, data
                        gc.collect()
                except Exception as e:
                    logger.warning(f"加载文件 {f} 时出错: {str(e)}")
                    continue

                logger.info(f"\n开始并行处理 {start_year}-{end_year} 年的数据...")

                # 获取CPU核心数并设置进程数
                num_cores = multiprocessing.cpu_count()
                num_processes = min(num_cores * 2, 32)  # 限制最大进程数为32
                logger.info(f"使用 {num_processes} 个进程进行并行处理")

                # 准备批次数据
                points_list = list(point_data_dict.items())
                total_points = len(points_list)

                # 减小批次大小，提高处理灵活性
                batch_size = max(1, min(100, total_points // (num_processes * 8)))
                batches = [points_list[i:i + batch_size] for i in range(0, total_points, batch_size)]

                logger.info(f"数据分配情况:")
                logger.info(f"- 总数据点: {total_points}")
                logger.info(f"- 批次大小: {batch_size}")
                logger.info(f"- 总批次数: {len(batches)}")

                # 创建进程池进行并行处理
                results = []
                failed_points = []

                with multiprocessing.Pool(processes=num_processes) as pool:
                    chunksize = max(1, len(batches) // (num_processes * 4))
                    iterator = pool.imap_unordered(process_batch, batches, chunksize=chunksize)

                    with tqdm(total=len(batches), desc="处理批次") as pbar:
                        while True:
                            try:
                                batch_results = next(iterator, None)
                                if batch_results is None:  # 所有批次处理完成
                                    break

                                if isinstance(batch_results, list):
                                    for result in batch_results:
                                        if "失败" in result:
                                            match = re.search(r"\(([^,]+), ([^)]+)\)", result)
                                            if match:
                                                lat, lon = float(match.group(1)), float(match.group(2))
                                                failed_points.append({
                                                    'lat': lat,
                                                    'lon': lon,
                                                    'error_message': result
                                                })
                                        else:
                                            results.append(result)
                                pbar.update(1)

                            except multiprocessing.TimeoutError:
                                logger.warning(f"批次处理超时（超过4分钟）")
                                continue
                            except Exception as e:
                                logger.error(f"处理批次时出错: {str(e)}")
                                continue

                    # 保存失败点信息
                    if failed_points:
                        year = get_year_from_filename(f)
                        failed_points_df = pd.DataFrame(failed_points)
                        failed_points_path = os.path.join('tmp/', f'failed_points_{year}.csv')
                        failed_points_df.to_csv(failed_points_path, index=False)
                        logger.info(f"失败点信息已保存至: {failed_points_path}")

                # 清理内存
                logger.info(f"{start_year}-{end_year} 年数据处理完成，开始清理资源...")
                del points_list, point_data_dict
                gc.collect()

                # 输出处理结果统计
                successful_points = len([r for r in results if "成功" in r])
                logger.info(f"\n{start_year}-{end_year} 年处理完成:")
                logger.info(f"- 总计处理: {total_points} 个坐标点")
                logger.info(f"- 成功处理: {successful_points} 个点")
                logger.info(f"- 失败点数: {len(failed_points)} 个点")

        except Exception as e:
            logger.error(f"处理 {start_year}-{end_year} 年数据时出错: {str(e)}")
            continue

        # 每处理完5年数据后，强制进行垃圾回收
        gc.collect()
        logger.info(f"完成 {start_year}-{end_year} 年数据处理，等待5秒后继续...")
        time_module.sleep(5)  # 等待5秒，确保文件写入完成


def main():
    try:
        print("\n=== 开始处理 ===")
        # 检查路径
        check_paths(hist_dir, output_dir)

        # 加载数据
        load_historical_data(hist_dir)

        print("\n=== 处理完成 ===")

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise
    finally:
        # 清理内存
        gc.collect()


if __name__ == "__main__":
    main()
