"""
CSV转Parquet格式转换工具

功能概述：
本工具用于将result_half目录下的所有CSV文件转换为Parquet格式，以提高读取速度并节省存储空间。

输入数据：
- 目录：Z:\local_environment_creation\energy_consumption_gird\result\result_half\
- 年份文件夹：2016, 2017, 2018, 2019, 2020
- 文件格式：point_lat{lat}_lon{lon}_cooling.csv, point_lat{lat}_lon{lon}_heating.csv

输出数据：
- 目录：Z:\local_environment_creation\energy_consumption_gird\result\result_half_parquet\
- 年份文件夹：2016, 2017, 2018, 2019, 2020
- 文件格式：point_lat{lat}_lon{lon}_cooling.parquet, point_lat{lat}_lon{lon}_heating.parquet
"""

import pandas as pd
import os
from pathlib import Path
import logging
from tqdm import tqdm
import time

# 设置日志记录
def setup_logging():
    """设置日志记录"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件处理器
    file_handler = logging.FileHandler('convert_csv_to_parquet.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 初始化日志
logger = setup_logging()

# 配置参数
SOURCE_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\result_half"
TARGET_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\result_half_parquet"
YEARS = [2016, 2017, 2018, 2019, 2020]

def count_csv_files():
    """统计需要转换的CSV文件数量"""
    total_count = 0
    for year in YEARS:
        year_dir = os.path.join(SOURCE_DIR, str(year))
        if os.path.exists(year_dir):
            csv_files = list(Path(year_dir).glob('*.csv'))
            total_count += len(csv_files)
            logger.info(f"{year}年: {len(csv_files)} 个CSV文件")
    
    return total_count

def convert_csv_to_parquet():
    """将CSV文件转换为Parquet格式"""
    logger.info("开始CSV到Parquet格式转换...")
    
    # 统计文件数量
    logger.info("=== 统计文件数量 ===")
    total_files = count_csv_files()
    logger.info(f"总共需要转换 {total_files} 个CSV文件")
    
    # 创建输出目录
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    converted_count = 0
    skipped_count = 0
    error_count = 0
    
    # 处理每个年份
    for year in YEARS:
        logger.info(f"\n=== 处理{year}年数据 ===")
        
        source_year_dir = os.path.join(SOURCE_DIR, str(year))
        target_year_dir = os.path.join(TARGET_DIR, str(year))
        
        # 创建年份输出目录
        os.makedirs(target_year_dir, exist_ok=True)
        
        if not os.path.exists(source_year_dir):
            logger.warning(f"{year}年源目录不存在: {source_year_dir}")
            continue
        
        # 获取所有CSV文件
        csv_files = list(Path(source_year_dir).glob('*.csv'))
        logger.info(f"{year}年共有 {len(csv_files)} 个CSV文件")
        
        # 处理每个CSV文件
        with tqdm(total=len(csv_files), desc=f"转换{year}年文件") as pbar:
            for csv_file in csv_files:
                try:
                    # 生成目标文件路径
                    parquet_filename = csv_file.stem + '.parquet'
                    parquet_path = os.path.join(target_year_dir, parquet_filename)
                    
                    # 检查文件是否已存在
                    if os.path.exists(parquet_path):
                        skipped_count += 1
                        pbar.update(1)
                        continue
                    
                    # 读取CSV文件
                    try:
                        df = pd.read_csv(csv_file, engine='python')
                    except Exception:
                        try:
                            df = pd.read_csv(csv_file, engine='c')
                        except Exception as e:
                            logger.error(f"读取文件失败: {csv_file}, 错误: {e}")
                            error_count += 1
                            pbar.update(1)
                            continue
                    
                    # 转换为Parquet格式
                    df.to_parquet(parquet_path, compression='snappy')
                    
                    converted_count += 1
                    
                    # 每100个文件输出一次进度
                    if converted_count % 100 == 0:
                        logger.info(f"已转换 {converted_count} 个文件...")
                    
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"转换文件失败: {csv_file}, 错误: {e}")
                    error_count += 1
                    pbar.update(1)
                    continue
        
        logger.info(f"{year}年处理完成")
    
    # 输出统计信息
    logger.info("\n=== 转换完成 ===")
    logger.info(f"成功转换: {converted_count} 个文件")
    logger.info(f"跳过（已存在）: {skipped_count} 个文件")
    logger.info(f"错误: {error_count} 个文件")
    logger.info(f"总计: {converted_count + skipped_count + error_count} 个文件")
    
    # 计算文件大小统计
    logger.info("\n=== 文件大小对比 ===")
    if converted_count > 0:
        # 随机选择一个文件进行大小对比
        for year in YEARS:
            source_year_dir = os.path.join(SOURCE_DIR, str(year))
            target_year_dir = os.path.join(TARGET_DIR, str(year))
            
            if os.path.exists(source_year_dir) and os.path.exists(target_year_dir):
                csv_files = list(Path(source_year_dir).glob('*.csv'))
                if csv_files:
                    csv_file = csv_files[0]
                    parquet_file = Path(target_year_dir) / (csv_file.stem + '.parquet')
                    
                    if parquet_file.exists():
                        csv_size = os.path.getsize(csv_file)
                        parquet_size = os.path.getsize(parquet_file)
                        compression_ratio = (1 - parquet_size / csv_size) * 100
                        
                        logger.info(f"示例文件: {csv_file.name}")
                        logger.info(f"  CSV大小: {csv_size / 1024 / 1024:.2f} MB")
                        logger.info(f"  Parquet大小: {parquet_size / 1024 / 1024:.2f} MB")
                        logger.info(f"  压缩率: {compression_ratio:.2f}%")
                        break

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("CSV到Parquet格式转换工具")
    logger.info("=" * 60)
    
    try:
        # 检查源目录是否存在
        if not os.path.exists(SOURCE_DIR):
            raise FileNotFoundError(f"源目录不存在: {SOURCE_DIR}")
        
        logger.info(f"源目录: {SOURCE_DIR}")
        logger.info(f"目标目录: {TARGET_DIR}")
        
        # 执行转换
        start_time = time.time()
        convert_csv_to_parquet()
        end_time = time.time()
        
        logger.info(f"\n总耗时: {end_time - start_time:.2f} 秒")
        logger.info("转换完成！")
        
    except Exception as e:
        error_msg = f"主程序执行出错: {str(e)}"
        logger.error(error_msg)
        raise

if __name__ == "__main__":
    main()
