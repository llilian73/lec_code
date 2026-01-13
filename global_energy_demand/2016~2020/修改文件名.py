"""
批量修改CSV文件名 - 将2019年份改为对应的年份

功能：
遍历 Z:\local_environment_creation\energy_consumption_gird\result\result 目录下的
2016、2017、2018、2019、2020 五个年份文件夹，将每个大洲的 summary 和 summary_p
文件夹中的 CSV 文件名从 {country_iso}_2019_summary_results.csv 改为对应年份。
"""

import os
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rename_files.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 配置参数
BASE_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\result"
YEARS = [2016, 2017, 2018, 2019, 2020]
CONTINENTS = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
SUMMARY_DIRS = ['summary', 'summary_p']


def rename_files_in_directory(directory, year, dry_run=True):
    """
    重命名目录中的CSV文件
    
    Args:
        directory: 要处理的目录路径
        year: 目标年份
        dry_run: 如果为True，只显示将要重命名的文件，不实际重命名
    
    Returns:
        重命名的文件数量
    """
    if not os.path.exists(directory):
        logging.warning(f"目录不存在: {directory}")
        return 0
    
    renamed_count = 0
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # 检查文件名是否包含 _2019_
            if '_2019_' in filename:
                # 构造新文件名，将2019替换为对应年份
                new_filename = filename.replace('_2019_', f'_{year}_')
                
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                
                if dry_run:
                    logging.info(f"[预览] {filename} -> {new_filename}")
                else:
                    try:
                        os.rename(old_path, new_path)
                        logging.info(f"[已重命名] {filename} -> {new_filename}")
                        renamed_count += 1
                    except Exception as e:
                        logging.error(f"重命名失败 {filename}: {e}")
            # 如果不包含2019，检查是否已经是正确的年份
            elif f'_{year}_' in filename:
                logging.debug(f"[跳过] {filename} - 已经是正确的年份")
            else:
                logging.debug(f"[跳过] {filename} - 文件名格式不匹配")
    
    return renamed_count


def main(dry_run=True):
    """
    主函数
    
    Args:
        dry_run: 如果为True，只预览不实际修改
    """
    if dry_run:
        logging.info("=" * 60)
        logging.info("运行模式: 预览模式 (dry_run=True)")
        logging.info("将显示要重命名的文件，但不会实际修改")
        logging.info("=" * 60)
    else:
        logging.info("=" * 60)
        logging.info("运行模式: 执行模式 (dry_run=False)")
        logging.info("将实际重命名文件")
        logging.info("=" * 60)
    
    total_renamed = 0
    
    # 遍历每个年份
    for year in YEARS:
        year_dir = os.path.join(BASE_DIR, str(year))
        
        if not os.path.exists(year_dir):
            logging.warning(f"年份目录不存在: {year_dir}")
            continue
        
        logging.info(f"\n{'='*60}")
        logging.info(f"处理 {year} 年数据")
        logging.info(f"{'='*60}")
        
        year_total = 0
        
        # 遍历每个大洲
        for continent in CONTINENTS:
            continent_dir = os.path.join(year_dir, continent)
            
            if not os.path.exists(continent_dir):
                logging.debug(f"大洲目录不存在: {continent_dir}")
                continue
            
            logging.info(f"\n处理大洲: {continent}")
            
            # 遍历 summary 和 summary_p 目录
            for summary_dir in SUMMARY_DIRS:
                summary_path = os.path.join(continent_dir, summary_dir)
                
                if not os.path.exists(summary_path):
                    logging.debug(f"  {summary_dir} 目录不存在: {summary_path}")
                    continue
                
                logging.info(f"  处理目录: {summary_dir}")
                
                # 重命名该目录中的文件
                count = rename_files_in_directory(summary_path, year, dry_run)
                year_total += count
        
        logging.info(f"\n{year} 年共处理 {year_total} 个文件")
        total_renamed += year_total
    
    logging.info(f"\n{'='*60}")
    logging.info(f"总计处理 {total_renamed} 个文件")
    logging.info(f"{'='*60}")
    
    if dry_run:
        logging.info("\n这只是预览！要实际执行重命名，请设置 dry_run=False")
        logging.info("建议先检查上面的预览结果，确认无误后再执行实际重命名")


if __name__ == "__main__":
    # 第一步：预览模式，查看将要修改的文件
    # logging.info("第一步：预览模式")
    # main(dry_run=True)
    
    # 如果预览结果正确，取消下面的注释来执行实际重命名
    logging.info("\n\n第二步：执行实际重命名")
    main(dry_run=False)
    
    logging.info("\n完成！请查看 rename_files.log 了解详细信息")

