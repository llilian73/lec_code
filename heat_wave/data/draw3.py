"""
绘制峰值能耗时刻的节能率热图

功能：
1. 读取各个模型的逐时能耗CSV文件
2. 找到每个国家ref工况的最高能耗时刻
3. 按大洲选择top国家（亚洲、欧洲、非洲、美洲各5个，大洋洲2个）
4. 计算case6-10相对于ref的节能率
5. 对所有模型和年份求平均
6. 绘制热图

输入数据：
/home/linbor/WORK/lishiying/heat_wave/{模型名}/{SSP路径}/energy/hourly_energy_v2/{大洲}/{国家代码}/{国家代码}_{年份}_hourly_energy.csv
文件格式：每个文件包含time列以及ref、case6、case7、case8、case9、case10列

输出数据：
CSV: /home/linbor/WORK/lishiying/heat_wave/figure/peak_energy/{SSP}_peak_energy_reduction.csv
PNG: /home/linbor/WORK/lishiying/heat_wave/figure/peak_energy/{SSP}_peak_energy_reduction.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import logging
import re
from collections import defaultdict
import matplotlib.colors as mcolors
from multiprocessing import Pool
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置路径
BASE_PATH = "/home/linbor/WORK/lishiying"
HEAT_WAVE_BASE_PATH = os.path.join(BASE_PATH, "heat_wave")
FIGURE_PATH = os.path.join(HEAT_WAVE_BASE_PATH, "figure", "peak_energy")
COUNTRIES_INFO_CSV = os.path.join(BASE_PATH, "shapefiles", "all_countries_info.csv")

# 模型配置
MODELS = [
    "ACCESS-ESM1-5",
    "BCC-CSM2-MR",
    "CanESM5",
    "EC-Earth3",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0"
]

# SSP配置
SSP_PATHS = ["SSP126", "SSP245"]

# 年份配置
TARGET_YEARS = [2030, 2031, 2032, 2033, 2034]

# Case配置
CASE_NAMES = ['case6', 'case7', 'case8', 'case9', 'case10']
CASE_LABELS = ['1/2', '1/4', '1/8', '1/16', '1/32']  # 对应的局部空间比例

# 大洲配置
CONTINENT_TOP_COUNTS = {
    'Asia': 5,
    'Europe': 5,
    'Africa': 5,
    'Americas': 5,  # 包括北美洲和南美洲
    'Oceania': 2
}

# 大洲映射（将North America和South America合并为Americas）
CONTINENT_MAPPING = {
    'North America': 'Americas',
    'South America': 'Americas'
}

# 并行处理配置
NUM_PROCESSES = 50  # 并行读取数据的进程数


def read_csv_with_encoding(file_path, keep_default_na=True):
    """读取CSV文件，尝试多种编码"""
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, keep_default_na=keep_default_na)
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    raise ValueError(f"无法使用常见编码读取文件: {file_path}")


def parse_filename(filename):
    """解析文件名，提取国家代码和年份
    
    新格式：AGO_2030_hourly_energy.csv
    
    Returns:
    --------
    dict: {
        'country_code': str,
        'year': int
    } 或 None
    """
    # 提取国家代码（3个字母）
    country_match = re.match(r'^([A-Z]{3})_', filename)
    if not country_match:
        return None
    
    country_code = country_match.group(1)
    
    # 提取年份
    year_match = re.search(r'_(\d{4})_hourly_energy\.csv$', filename)
    if not year_match:
        return None
    
    year = int(year_match.group(1))
    
    return {
        'country_code': country_code,
        'year': year
    }


def load_country_data(model_name, ssp_path, country_code, continent, year, countries_info_df):
    """加载指定国家的逐时能耗数据（新格式：单个文件包含ref和case6-10）
    
    文件格式：AGO_2030_hourly_energy.csv
    路径：/energy/hourly_energy_v2/{大洲}/{国家代码}/
    
    Returns:
    --------
    pd.DataFrame: 包含time列和所需工况列的DataFrame，如果失败返回None
    """
    # 验证参数有效性
    if pd.isna(country_code) or pd.isna(continent):
        return None
    
    # 确保是字符串类型
    try:
        country_code = str(country_code).strip()
        continent = str(continent).strip()
    except (TypeError, AttributeError):
        return None
    
    # 跳过空字符串
    if not country_code or not continent:
        return None
    
    # 需要的工况
    required_cases = ['ref'] + CASE_NAMES
    
    # 构建文件路径（新路径：hourly_energy_v2）
    base_dir = os.path.join(
        HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", "hourly_energy_v2",
        continent, country_code
    )
    
    # 如果路径不存在，尝试从countries_info_df获取原始大洲名称
    if not os.path.exists(base_dir):
        country_row = countries_info_df[countries_info_df['Country_Code_3'] == country_code]
        if not country_row.empty:
            original_continent = country_row.iloc[0]['continent']
            # 检查原始大洲名称是否有效
            if pd.notna(original_continent):
                try:
                    original_continent = str(original_continent).strip()
                    if original_continent:
                        base_dir = os.path.join(
                            HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", "hourly_energy_v2",
                            original_continent, country_code
                        )
                except (TypeError, AttributeError):
                    pass
    
    if not os.path.exists(base_dir):
        return None
    
    # 构建文件名（新格式：{国家代码}_{年份}_hourly_energy.csv）
    filename = f"{country_code}_{year}_hourly_energy.csv"
    file_path = os.path.join(base_dir, filename)
    
    if not os.path.exists(file_path):
        return None
    
    try:
        # 读取CSV文件
        df = read_csv_with_encoding(file_path)
        
        # 确保time列存在
        if 'time' not in df.columns:
            return None
        
        # 转换time列为datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # 检查并提取所需的工况列
        available_cases = []
        for case in required_cases:
            if case in df.columns:
                available_cases.append(case)
        
        if not available_cases or 'ref' not in available_cases:
            return None
        
        # 构建结果DataFrame，只包含time和所需的工况列
        result_cols = ['time'] + available_cases
        result_df = df[result_cols].copy()
        
        # 确保所有需要的工况列都存在（如果不存在，填充NaN）
        for case in required_cases:
            if case not in result_df.columns:
                result_df[case] = np.nan
        
        return result_df
    
    except Exception as e:
        logger.warning(f"Failed to read file {file_path}: {e}")
        return None


def find_peak_hour(df):
    """找到ref工况的最高能耗时刻
    
    Returns:
    --------
    tuple: (peak_time, peak_energy) 或 (None, None) 如果失败
    """
    if 'ref' not in df.columns:
        return None, None
    
    peak_idx = df['ref'].idxmax()
    peak_time = df.loc[peak_idx, 'time']
    peak_energy = df.loc[peak_idx, 'ref']
    
    return peak_time, peak_energy


def calculate_reduction_rate(ref_energy, case_energy):
    """计算节能率（相对于ref）
    
    Returns:
    --------
    float: 节能率（%），如果ref_energy为0或无效，返回np.nan
    """
    if pd.isna(ref_energy) or pd.isna(case_energy) or ref_energy == 0:
        return np.nan
    
    reduction_rate = (ref_energy - case_energy) / ref_energy * 100
    return reduction_rate


def process_single_country(args):
    """处理单个国家的数据（用于多进程）
    
    Parameters:
    -----------
    args : tuple
        (model_name, ssp_path, year, country_code, continent, countries_info_df)
    
    Returns:
    --------
    tuple: (country_code, result_dict) 或 (country_code, None) 如果失败
    """
    model_name, ssp_path, year, country_code, continent, countries_info_df = args
    
    # 验证参数有效性
    if pd.isna(country_code) or pd.isna(continent):
        return (country_code, None)
    
    # 确保是字符串类型
    try:
        country_code = str(country_code).strip()
        continent = str(continent).strip()
    except (TypeError, AttributeError):
        return (country_code, None)
    
    # 跳过空字符串
    if not country_code or not continent:
        return (country_code, None)
    
    try:
        # 加载数据
        df = load_country_data(model_name, ssp_path, country_code, continent, year, countries_info_df)
        
        if df is None or df.empty:
            return (country_code, None)
        
        # 找到峰值时刻
        peak_time, peak_energy = find_peak_hour(df)
        
        if peak_time is None:
            return (country_code, None)
        
        # 找到峰值时刻对应的行
        peak_row = df[df['time'] == peak_time]
        if peak_row.empty:
            return (country_code, None)
        
        peak_row = peak_row.iloc[0]
        ref_energy = peak_row.get('ref', None)
        
        if pd.isna(ref_energy) or ref_energy == 0:
            return (country_code, None)
        
        # 计算各个case的节能率
        reduction_rates = {}
        for case_name in CASE_NAMES:
            if case_name in peak_row:
                case_energy = peak_row[case_name]
                reduction_rate = calculate_reduction_rate(ref_energy, case_energy)
                reduction_rates[case_name] = reduction_rate
        
        if reduction_rates:
            result = {
                'peak_time': peak_time,
                'peak_energy': peak_energy,
                'reduction_rates': reduction_rates,
                'continent': continent
            }
            return (country_code, result)
        else:
            return (country_code, None)
    
    except Exception as e:
        logger.warning(f"处理国家 {country_code} 时出错: {e}")
        return (country_code, None)


def process_single_model_year(model_name, ssp_path, year, countries_info_df):
    """处理单个模型单年的数据（使用多进程并行读取）
    
    Returns:
    --------
    dict: {
        country_code: {
            'peak_time': datetime,
            'peak_energy': float,
            'reduction_rates': {case_name: float}
        }
    }
    """
    results = {}
    
    # 获取所有国家信息（过滤掉NaN值）
    country_to_continent = {}
    for _, row in countries_info_df.iterrows():
        country_code = row['Country_Code_3']
        continent = row['continent']
        
        # 跳过NaN值
        if pd.isna(country_code) or pd.isna(continent):
            continue
        
        # 确保是字符串类型
        country_code = str(country_code).strip()
        continent = str(continent).strip()
        
        # 跳过空字符串
        if not country_code or not continent:
            continue
        
        # 应用大洲映射
        continent = CONTINENT_MAPPING.get(continent, continent)
        country_to_continent[country_code] = continent
    
    # 准备并行处理的参数
    country_args = []
    for country_code, continent in country_to_continent.items():
        country_args.append((
            model_name, ssp_path, year, country_code, continent, countries_info_df
        ))
    
    # 使用多进程并行处理
    if country_args:
        process_start_time = time.time()
        num_processes = min(NUM_PROCESSES, len(country_args))
        
        with Pool(processes=num_processes) as pool:
            country_results = pool.map(process_single_country, country_args)
        
        process_elapsed = time.time() - process_start_time
        logger.info(f"  Parallel processing of {len(country_args)} countries completed, elapsed: {process_elapsed:.2f}s")
        
        # 收集结果
        for country_code, result in country_results:
            if result is not None:
                results[country_code] = result
    
    return results


def select_top_countries_by_continent(all_results):
    """按大洲选择峰值能耗最高的国家
    
    Parameters:
    -----------
    all_results : dict
        {model: {year: {country_code: {...}}}}
    
    Returns:
    --------
    dict: {continent: [country_code, ...]}
    """
    # 收集所有国家的峰值能耗（跨所有模型和年份）
    country_peak_energies = defaultdict(list)
    country_continents = {}
    
    for model_results in all_results.values():
        for year_results in model_results.values():
            for country_code, country_data in year_results.items():
                continent = country_data.get('continent')
                peak_energy = country_data.get('peak_energy', 0)
                
                if continent and peak_energy > 0:
                    country_peak_energies[country_code].append(peak_energy)
                    country_continents[country_code] = continent
    
    # 计算每个国家的平均峰值能耗
    country_avg_peak = {}
    for country_code, energies in country_peak_energies.items():
        country_avg_peak[country_code] = np.mean(energies)
    
    # 按大洲分组并选择top国家
    continent_countries = defaultdict(list)
    for country_code, avg_peak in country_avg_peak.items():
        continent = country_continents.get(country_code)
        if continent:
            continent_countries[continent].append((country_code, avg_peak))
    
    # 对每个大洲排序并选择top N
    selected_countries = {}
    for continent, countries_list in continent_countries.items():
        # 按峰值能耗降序排序
        countries_list.sort(key=lambda x: x[1], reverse=True)
        
        # 选择top N
        top_count = CONTINENT_TOP_COUNTS.get(continent, 5)
        selected = [country_code for country_code, _ in countries_list[:top_count]]
        selected_countries[continent] = selected
    
    return selected_countries


def calculate_average_reduction_rates(all_results, selected_countries, countries_info_df):
    """计算平均节能率
    
    Parameters:
    -----------
    all_results : dict
        {model: {year: {country_code: {...}}}}
    selected_countries : dict
        {continent: [country_code, ...]}
    countries_info_df : pd.DataFrame
        国家信息DataFrame，包含Country_Code_3和Country_Code_2列
    
    Returns:
    --------
    pd.DataFrame: 包含Country_Code_2, Continent, case6, case7, case8, case9, case10列
    """
    # 创建三字母代码到二字母代码的映射
    code3_to_code2 = {}
    for _, row in countries_info_df.iterrows():
        code3 = row.get('Country_Code_3')
        code2 = row.get('Country_Code_2')
        if pd.notna(code3) and pd.notna(code2):
            code3_to_code2[str(code3).strip()] = str(code2).strip()
    
    # 收集所有数据
    country_data = defaultdict(lambda: defaultdict(list))  # {country_code: {case_name: [rates]}}
    country_continents = {}
    
    for model_results in all_results.values():
        for year_results in model_results.values():
            for country_code, country_info in year_results.items():
                # 只处理选中的国家
                continent = country_info.get('continent')
                if not continent or country_code not in selected_countries.get(continent, []):
                    continue
                
                country_continents[country_code] = continent
                reduction_rates = country_info.get('reduction_rates', {})
                
                for case_name, rate in reduction_rates.items():
                    if not pd.isna(rate):
                        country_data[country_code][case_name].append(rate)
    
    # 计算平均值
    result_rows = []
    for continent in sorted(selected_countries.keys()):
        for country_code in selected_countries[continent]:
            if country_code not in country_data:
                continue
            
            # 获取二字母代码
            code2 = code3_to_code2.get(country_code, country_code[:2] if len(country_code) >= 2 else country_code)
            
            row = {
                'Country_Code_2': code2,
                'Country_Code_3': country_code,  # 保留用于内部处理
                'Continent': continent
            }
            
            for case_name in CASE_NAMES:
                rates = country_data[country_code].get(case_name, [])
                if rates:
                    row[case_name] = np.mean(rates)
                else:
                    row[case_name] = np.nan
            
            result_rows.append(row)
    
    df = pd.DataFrame(result_rows)
    return df


def plot_heatmap(df, ssp_path, output_path):
    """绘制热图
    
    Parameters:
    -----------
    df : pd.DataFrame
        包含Country_Code_2, Continent, case6, case7, case8, case9, case10列
    ssp_path : str
        SSP路径
    output_path : str
        输出PNG路径
    """
    # 按大洲分组排序
    df_sorted = df.sort_values(['Continent', 'Country_Code_2']).reset_index(drop=True)
    
    # 准备数据矩阵
    case_cols = CASE_NAMES
    data_matrix = df_sorted[case_cols].values
    
    # 创建图形，缩小宽度使方格更窄
    fig, ax = plt.subplots(figsize=(10, max(10, len(df_sorted) * 0.3)))
    
    # 使用白色到#b71c2c的颜色映射
    colors = ['#FFFFFF', '#b71c2c']
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('red', colors, N=n_bins)
    
    # 绘制热图，使用更大的aspect使方格更窄
    im = ax.imshow(data_matrix, cmap=cmap, aspect=0.3, vmin=0, vmax=100)
    
    # 设置x轴标签（顶部），使用希腊字母alpha
    ax.set_xticks(np.arange(len(CASE_LABELS)))
    ax.set_xticklabels(CASE_LABELS)
    ax.set_xlabel(r'$\alpha$', fontsize=12)  # 使用LaTeX格式显示希腊字母alpha，不加粗
    ax.xaxis.set_label_position('top')  # 确保标签在顶部
    ax.xaxis.tick_top()  # 将刻度放在顶部
    ax.xaxis.set_ticks_position('top')  # 确保刻度只在顶部显示
    ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)  # 底部不显示刻度
    
    # 添加数值标注
    country_labels = df_sorted['Country_Code_2'].tolist()
    for i in range(len(country_labels)):
        for j in range(len(CASE_LABELS)):
            value = data_matrix[i, j]
            if not pd.isna(value):
                text = ax.text(j, i, f'{value:.1f}%',
                             ha="center", va="center", color="black", fontsize=8)
    
    # 添加颜色条，只变窄不变短
    cbar = plt.colorbar(im, ax=ax, aspect=40)  # aspect参数控制宽高比，更大的值让colorbar更窄
    cbar.set_label('Reduction Rate (%)', fontsize=10)
    
    # 设置y轴标签（国家二字母代码），不显示y轴标题
    ax.set_yticks(np.arange(len(country_labels)))
    ax.set_yticklabels(country_labels)
    
    # 添加大洲分组和左侧竖线
    continents = df_sorted['Continent'].tolist()
    continent_ranges = {}  # {continent: (start_idx, end_idx)}
    
    current_continent = None
    start_idx = 0
    
    for i, continent in enumerate(continents):
        if continent != current_continent:
            if current_continent is not None:
                continent_ranges[current_continent] = (start_idx, i - 1)
            current_continent = continent
            start_idx = i
    
    # 添加最后一个大洲
    if current_continent is not None:
        continent_ranges[current_continent] = (start_idx, len(continents) - 1)
    
    # 先调整布局
    plt.tight_layout()
    
    # 获取调整后的axes位置（figure坐标：0-1之间）
    pos = ax.get_position()
    
    # 计算左侧大洲标签区域的位置和大小
    # 在左侧留出空间用于大洲标签
    continent_area_left = max(0.01, pos.x0 - 0.1)  # 左侧额外空间增加
    continent_area_width = 0.08  # 大洲标签区域宽度增加
    
    # 创建用于大洲标签的axes
    ax_continent = fig.add_axes([continent_area_left, pos.y0, continent_area_width, pos.height])
    
    # 获取主axes的y轴范围（匹配imshow的y坐标）
    y_min, y_max = ax.get_ylim()  # 应该是(-0.5, n_rows-0.5)
    
    # 设置相同的y轴范围，确保对齐
    ax_continent.set_ylim(y_min, y_max)
    ax_continent.set_xlim(0, 1)
    ax_continent.axis('off')
    
    # 竖线位置（往左移，x值减小）
    line_x = 0.65  # 竖线在axes中的x位置（0-1之间，往左移）
    
    # 为每个大洲绘制竖线并添加标签
    # imshow中：行索引i对应y坐标i，其中i=0在顶部，i=n-1在底部
    # y轴范围通常是(-0.5, n_rows-0.5)
    for continent, (start_idx, end_idx) in continent_ranges.items():
        # 计算该大洲的y范围（使用行索引，匹配imshow的y坐标）
        # 行索引直接对应y坐标
        y_top = start_idx - 0.25    # 顶部行的y坐标（较小的索引，但y值也较小）
        y_bottom = end_idx + 0.25   # 底部行的y坐标（较大的索引，但y值也较大）
        
        # 绘制该大洲的竖线（从y_top到y_bottom）
        ax_continent.plot([line_x, line_x], [y_top, y_bottom], 
                         color='black', linewidth=1)
        
        # 计算大洲的中心位置
        center_y = (y_top + y_bottom) / 2
        
        # 添加大洲名称（垂直显示，在竖线左侧），不加粗
        label_x = line_x - 0.15  # 标签在竖线左边
        ax_continent.text(label_x, center_y, continent, 
                         rotation=90, ha='center', va='center', 
                         fontsize=10)
    
    # 调整主axes位置，确保不覆盖大洲标签区域
    new_pos = [max(continent_area_left + continent_area_width + 0.01, pos.x0), 
               pos.y0, pos.width, pos.height]
    ax.set_position(new_pos)
    
    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    
    logger.info(f"Heatmap saved to: {output_path}")


def main():
    """主函数"""
    try:
        logger.info("=== Starting to process peak energy reduction rate data ===")
        
        # 读取国家信息
        countries_info_df = read_csv_with_encoding(COUNTRIES_INFO_CSV)
        
        # 为每个SSP处理
        for ssp_path in SSP_PATHS:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing SSP path: {ssp_path}")
            logger.info(f"{'='*80}")
            
            # 收集所有模型和年份的数据
            all_results = {}  # {model: {year: {country_code: {...}}}}
            
            for model_name in MODELS:
                logger.info(f"\nProcessing model: {model_name}")
                model_results = {}
                
                for year in TARGET_YEARS:
                    logger.info(f"  Processing year: {year}")
                    try:
                        year_results = process_single_model_year(
                            model_name, ssp_path, year, countries_info_df
                        )
                        model_results[year] = year_results
                        logger.info(f"    Found data for {len(year_results)} countries")
                    except Exception as e:
                        logger.error(f"Error processing {model_name} - {year}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                if model_results:
                    all_results[model_name] = model_results
            
            if not all_results:
                logger.warning(f"{ssp_path}: No data found, skipping")
                continue
            
            # 选择top国家
            logger.info(f"\nSelecting top countries by peak energy for each continent...")
            selected_countries = select_top_countries_by_continent(all_results)
            
            for continent, countries in selected_countries.items():
                logger.info(f"  {continent}: {len(countries)} countries - {', '.join(countries)}")
            
            # 计算平均节能率
            logger.info(f"\nCalculating average reduction rates...")
            result_df = calculate_average_reduction_rates(all_results, selected_countries, countries_info_df)
            
            if result_df.empty:
                logger.warning(f"{ssp_path}: No valid data, skipping")
                continue
            
            # 保存CSV（只保存Country_Code_2，不保存Country_Code_3）
            csv_output_path = os.path.join(FIGURE_PATH, f"{ssp_path}_peak_energy_reduction.csv")
            os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
            # 创建输出DataFrame，排除Country_Code_3列
            output_df = result_df.drop(columns=['Country_Code_3'], errors='ignore')
            output_df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
            logger.info(f"CSV saved to: {csv_output_path}")
            
            # 绘制热图
            logger.info(f"\nPlotting heatmap...")
            png_output_path = os.path.join(FIGURE_PATH, f"{ssp_path}_peak_energy_reduction.png")
            plot_heatmap(result_df, ssp_path, png_output_path)
        
        logger.info("\n=== All processing completed ===")
    
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

