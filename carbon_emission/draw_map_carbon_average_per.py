"""
全球人均碳排放量平均地图绘制工具（2016-2020年）

功能概述：
本工具用于计算全球各个国家5年（2016-2020）的平均人均碳排放量，并绘制热图展示碳排放总量、减少量和减少率。

输入数据：
1. 各国碳排放汇总文件：
   - 路径：Z:\local_environment_creation\carbon_emission\2016-2020\result\{年份}\capita\{case}_summary.csv
   - 包含各国人均碳排放数据（kgCO2/person）
   - 涵盖ref、case1-case20共21种案例
   - 覆盖2016-2020年

输出结果：
1. 平均数据CSV文件：
   - {case_name}_summary_average.csv：各国5年平均人均碳排放数据
   - 格式：Country_Code_2, carbon_emission(kgCO2/person), carbon_emission_reduction(kgCO2/person), carbon_emission_reduction(%)

2. 热图PDF文件：
   - carbon_emission_map_{case_name}_average.pdf：碳排放总量热图（所有工况，包括ref）
   - carbon_emission_reduction_map_{case_name}_average.pdf：碳排放减少量热图（仅非ref工况）
   - carbon_emission_reduction_rate_map_{case_name}_average.pdf：碳排放减少率热图（仅非ref工况）
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import LineString

# 国家信息文件路径
COUNTRIES_INFO_FILE = r"Z:\local_environment_creation\all_countries_info.csv"

def load_country_code_mapping():
    """加载国家代码映射（二字母代码 -> 三字母代码）"""
    try:
        # 使用gbk编码读取文件，避免将 'NA' 识别为缺失值
        df = pd.read_csv(COUNTRIES_INFO_FILE, encoding='gbk', keep_default_na=False, na_values=[''])
        # 创建二字母代码到三字母代码的映射
        code2_to_code3 = {}
        for _, row in df.iterrows():
            code2 = str(row['Country_Code_2']).strip()
            code3 = str(row['Country_Code_3']).strip()
            if code2 and code3 and code2 != '' and code3 != '':
                code2_to_code3[code2] = code3
        print(f"成功加载 {len(code2_to_code3)} 个国家代码映射（使用编码: gbk）")
        return code2_to_code3
    except Exception as e:
        print(f"加载国家代码映射失败: {e}")
        return {}

# 在代码开始处加载国家代码映射（只加载一次）
CODE2_TO_CODE3 = load_country_code_mapping()

def create_custom_colormaps():
    """创建自定义颜色映射（参考draw_global per capita.py）"""
    # 碳排放量：从深绿色到深棕色的渐变色
    emission_colors = ['#5B9C4A', '#F0D264', '#8B4513']  # 深绿色到深棕色
    emission_cmap = LinearSegmentedColormap.from_list('emission', emission_colors)
    
    # 减少量：从白色到深绿色的渐变色
    reduction_colors = ['#FFFFFF', '#006400']  # 白色到深绿色
    reduction_cmap = LinearSegmentedColormap.from_list('reduction', reduction_colors)
    
    # 减少率：从白色到深绿色的渐变色（与减少量相同）
    reduction_rate_cmap = LinearSegmentedColormap.from_list('reduction_rate', reduction_colors)
    
    return {
        'emission': emission_cmap,
        'reduction': reduction_cmap,
        'reduction_rate': reduction_rate_cmap
    }

def add_tropic_lines(ax, world):
    """添加回归线"""
    minx, _, maxx, _ = world.total_bounds
    tropics = [
        (23.4368, 'Tropic of Cancer'),
        (-23.4368, 'Tropic of Capricorn')
    ]
    for lat, label in tropics:
        line = LineString([(minx, lat), (maxx, lat)])
        gpd.GeoSeries([line], crs=world.crs).plot(
            ax=ax,
            color='dimgray',
            linewidth=0.7,
            linestyle='--'
        )
        ax.text(
            minx + 2,
            lat + 1 if lat > 0 else lat - 1.5,
            label,
            fontsize=10,
            color='dimgray',
            ha='left',
            va='bottom' if lat > 0 else 'top'
        )

def calculate_average_data(data_dir, years, case_name):
    """计算每个国家5年平均值，返回dict
    Args:
        data_dir: 数据基础目录
        years: 年份列表
        case_name: case名称（如'ref', 'case1'等）
    """
    country_stats = {}
    
    for year in years:
        year_path = os.path.join(data_dir, str(year), 'capita', f"{case_name}_summary.csv")
        if not os.path.exists(year_path):
            continue
        
        # 关键修正：避免将 'NA' 识别为缺失，确保纳米比亚代码保留为字符串 'NA'
        df = pd.read_csv(year_path, keep_default_na=False)
        
        for _, row in df.iterrows():
            # 统一清洗国家代码：去空格并大写
            country = str(row['Country_Code_2']).strip().upper()
            
            if country not in country_stats:
                country_stats[country] = {
                    'carbon_emission(kgCO2/person)': [],
                    'carbon_emission_reduction(kgCO2/person)': [],
                    'carbon_emission_reduction(%)': []
                }
            
            # 处理碳排放量
            emission_value = pd.to_numeric(row.get('carbon_emission(kgCO2/person)', np.nan), errors='coerce')
            if pd.notna(emission_value) and emission_value >= 0:
                country_stats[country]['carbon_emission(kgCO2/person)'].append(emission_value)
            
            # 处理减少量（只对非ref工况）
            if case_name != 'ref':
                reduction_value = pd.to_numeric(row.get('carbon_emission_reduction(kgCO2/person)', np.nan), errors='coerce')
                if pd.notna(reduction_value) and reduction_value >= 0:
                    country_stats[country]['carbon_emission_reduction(kgCO2/person)'].append(reduction_value)
                
                reduction_rate_value = pd.to_numeric(row.get('carbon_emission_reduction(%)', np.nan), errors='coerce')
                if pd.notna(reduction_rate_value) and reduction_rate_value >= 0:
                    country_stats[country]['carbon_emission_reduction(%)'].append(reduction_rate_value)
    
    # 计算平均值
    avg_data = []
    for country, stats in country_stats.items():
        avg_row = {'Country_Code_2': country}
        for col, values in stats.items():
            avg_row[col] = np.nanmean(values) if values else np.nan
        avg_data.append(avg_row)
    
    return avg_data

def save_average_csv(avg_data, output_path):
    """保存平均数据到CSV"""
    df = pd.DataFrame(avg_data)
    df.to_csv(output_path, index=False)

def plot_carbon_maps(avg_data, shapefile_path, save_path, case_name, code2_to_code3):
    """绘制碳排放热图
    Args:
        avg_data: 平均数据列表（Country_Code_2列为二字母代码）
        shapefile_path: shapefile路径
        save_path: 保存路径
        case_name: case名称
        code2_to_code3: 二字母代码到三字母代码的映射
    """
    world = gpd.read_file(shapefile_path)
    
    # 准备平均数据 DataFrame，并统一清洗国家代码
    df_avg = pd.DataFrame(avg_data)
    if 'Country_Code_2' not in df_avg.columns:
        return
    df_avg['Country_Code_2'] = df_avg['Country_Code_2'].astype(str).str.strip().str.upper()
    # 按国家代码构造字典（二字母代码）
    country_data = {row['Country_Code_2']: row for _, row in df_avg.iterrows()}
    
    # 创建三字母代码的数据字典（用于绘图）
    country_data_code3 = {}
    for code2, row in country_data.items():
        code3 = code2_to_code3.get(code2, None)
        if code3:
            country_data_code3[code3] = row
    
    colormaps = create_custom_colormaps()
    
    # 1. 绘制碳排放总量热图（所有工况，包括ref工况）
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    world['value'] = np.nan
    
    # 使用三字母代码匹配
    for code3, row in country_data_code3.items():
        val = row.get('carbon_emission(kgCO2/person)', np.nan)
        if pd.notna(val):
            # 直接使用GID_0（三字母代码）匹配
            world.loc[world['GID_0'] == code3, 'value'] = val
    
    # 特殊处理：香港和澳门显示中国的数据
    if 'CN' in country_data:
        cn_row = country_data['CN']
        cn_val = cn_row.get('carbon_emission(kgCO2/person)', np.nan)
        if pd.notna(cn_val):
            # 获取中国的三字母代码
            chn_code3 = code2_to_code3.get('CN', 'CHN')
            if chn_code3:
                # 香港（HKG）显示中国的数据
                world.loc[world['GID_0'] == 'HKG', 'value'] = cn_val
                # 澳门（MAC）显示中国的数据
                world.loc[world['GID_0'] == 'MAC', 'value'] = cn_val
    
    # 绘制国家边界
    world.boundary.plot(ax=ax, linewidth=0.5, color='black')
    
    # 计算vmax（使用数据最大值）
    vmax = df_avg['carbon_emission(kgCO2/person)'].max() if not df_avg['carbon_emission(kgCO2/person)'].isna().all() else 1000
    
    world.plot(column='value',
               ax=ax,
               missing_kwds={'color': 'lightgrey'},
               legend=True,
               legend_kwds={'label': 'Carbon Emission (kgCO2/person)',
                           'orientation': 'horizontal',
                           'shrink': 0.6, 'pad': 0.02},
               cmap=colormaps['emission'],
               vmin=0,
               vmax=vmax)
    add_tropic_lines(ax, world)
    plt.title(f'Carbon Emission (kgCO2/person) - {case_name.capitalize()} - 5-Year Average', fontsize=16)
    ax.axis('off')
    output_file = os.path.join(save_path, f"carbon_emission_map_{case_name}_average.pdf")
    plt.savefig(output_file, format="pdf", bbox_inches='tight')
    print(f"已保存: {output_file}")
    plt.close()
    
    # 2. 绘制碳排放减少量热图（仅对非ref工况）
    if case_name != 'ref':
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        world['value'] = np.nan
        
        # 使用三字母代码匹配
        for code3, row in country_data_code3.items():
            val = row.get('carbon_emission_reduction(kgCO2/person)', np.nan)
            if pd.notna(val):
                # 直接使用GID_0（三字母代码）匹配
                world.loc[world['GID_0'] == code3, 'value'] = val
        
        # 特殊处理：香港和澳门显示中国的数据
        if 'CN' in country_data:
            cn_row = country_data['CN']
            cn_val = cn_row.get('carbon_emission_reduction(kgCO2/person)', np.nan)
            if pd.notna(cn_val):
                # 获取中国的三字母代码
                chn_code3 = code2_to_code3.get('CN', 'CHN')
                if chn_code3:
                    # 香港（HKG）显示中国的数据
                    world.loc[world['GID_0'] == 'HKG', 'value'] = cn_val
                    # 澳门（MAC）显示中国的数据
                    world.loc[world['GID_0'] == 'MAC', 'value'] = cn_val
        
        # 绘制国家边界
        world.boundary.plot(ax=ax, linewidth=0.5, color='black')
        
        # 计算vmax（使用数据最大值）
        vmax = df_avg['carbon_emission_reduction(kgCO2/person)'].max() if not df_avg['carbon_emission_reduction(kgCO2/person)'].isna().all() else 1000
        
        world.plot(column='value',
                   ax=ax,
                   missing_kwds={'color': 'lightgrey'},
                   legend=True,
                   legend_kwds={'label': 'Carbon Emission Reduction (kgCO2/person)',
                               'orientation': 'horizontal',
                               'shrink': 0.6, 'pad': 0.02},
                   cmap=colormaps['reduction'],
                   vmin=0,
                   vmax=vmax)
        add_tropic_lines(ax, world)
        plt.title(f'Carbon Emission Reduction (kgCO2/person) - {case_name.capitalize()} - 5-Year Average', fontsize=16)
        ax.axis('off')
        output_file = os.path.join(save_path, f"carbon_emission_reduction_map_{case_name}_average.pdf")
        plt.savefig(output_file, format="pdf", bbox_inches='tight')
        print(f"已保存: {output_file}")
        plt.close()
    
    # 3. 绘制碳排放减少率热图（仅对非ref工况）
    if case_name != 'ref':
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        world['value'] = np.nan
        
        # 使用三字母代码匹配
        for code3, row in country_data_code3.items():
            val = row.get('carbon_emission_reduction(%)', np.nan)
            if pd.notna(val):
                # 直接使用GID_0（三字母代码）匹配
                world.loc[world['GID_0'] == code3, 'value'] = val
        
        # 特殊处理：香港和澳门显示中国的数据
        if 'CN' in country_data:
            cn_row = country_data['CN']
            cn_val = cn_row.get('carbon_emission_reduction(%)', np.nan)
            if pd.notna(cn_val):
                # 获取中国的三字母代码
                chn_code3 = code2_to_code3.get('CN', 'CHN')
                if chn_code3:
                    # 香港（HKG）显示中国的数据
                    world.loc[world['GID_0'] == 'HKG', 'value'] = cn_val
                    # 澳门（MAC）显示中国的数据
                    world.loc[world['GID_0'] == 'MAC', 'value'] = cn_val
        
        # 绘制国家边界
        world.boundary.plot(ax=ax, linewidth=0.5, color='black')
        
        world.plot(column='value',
                   ax=ax,
                   missing_kwds={'color': 'lightgrey'},
                   legend=True,
                   legend_kwds={'label': 'Carbon Emission Reduction (%)',
                               'orientation': 'horizontal',
                               'shrink': 0.6, 'pad': 0.02},
                   cmap=colormaps['reduction_rate'],
                   vmin=0,
                   vmax=100)
        add_tropic_lines(ax, world)
        plt.title(f'Carbon Emission Reduction (%) - {case_name.capitalize()} - 5-Year Average', fontsize=16)
        ax.axis('off')
        output_file = os.path.join(save_path, f"carbon_emission_reduction_rate_map_{case_name}_average.pdf")
        plt.savefig(output_file, format="pdf", bbox_inches='tight')
        print(f"已保存: {output_file}")
        plt.close()

def process_single_case(args):
    """处理单个case的平均数据计算和地图绘制（用于并行处理）
    Args:
        args: 元组 (case_name, data_dir, years, shapefile_path, data_output_dir, figure_output_dir, code2_to_code3)
    """
    case_name, data_dir, years, shapefile_path, data_output_dir, figure_output_dir, code2_to_code3 = args
    
    try:
        print(f"正在处理 {case_name} …")
        avg_data = calculate_average_data(data_dir, years, case_name)
        csv_path = os.path.join(data_output_dir, f"{case_name}_summary_average.csv")
        save_average_csv(avg_data, csv_path)
        print(f"已保存: {csv_path}")
        plot_carbon_maps(avg_data, shapefile_path, figure_output_dir, case_name, code2_to_code3)
        return f"{case_name} 完成"
    except Exception as e:
        return f"{case_name} 处理失败: {str(e)}"

if __name__ == "__main__":
    # 输入数据目录
    data_dir = r"Z:\local_environment_creation\carbon_emission\2016-2020\result"
    
    # Shapefile路径
    shapefile_path = r"Z:\local_environment_creation\shapefiles\全球国家边界\world_border2.shp"
    
    # 输出目录
    data_output_dir = r"Z:\local_environment_creation\carbon_emission\2016-2020\figure_maps_and_data\per_capita\data"
    figure_output_dir = r"Z:\local_environment_creation\carbon_emission\2016-2020\figure_maps_and_data\per_capita\figure"
    
    os.makedirs(data_output_dir, exist_ok=True)
    os.makedirs(figure_output_dir, exist_ok=True)
    
    years = range(2016, 2021)
    
    # 准备所有case的参数列表（包括ref）
    case_names = ['ref'] + [f'case{i}' for i in range(1, 21)]
    batch_size = 10  # 每批处理10个case
    
    # 将case分批
    case_batches = [case_names[i:i + batch_size] for i in range(0, len(case_names), batch_size)]
    
    print(f"共 {len(case_names)} 个case，分为 {len(case_batches)} 批处理（每批 {batch_size} 个）")
    
    # 配置并行处理参数
    num_cores = multiprocessing.cpu_count()
    num_processes = min(num_cores, batch_size)  # 进程数不超过批次大小
    
    # 按批处理
    for batch_idx, case_batch in enumerate(case_batches, 1):
        print(f"\n处理第 {batch_idx}/{len(case_batches)} 批: {case_batch[0]}-{case_batch[-1]}")
        
        # 准备当前批次的参数
        batch_args = [
            (case_name, data_dir, years, shapefile_path, data_output_dir, figure_output_dir, CODE2_TO_CODE3)
            for case_name in case_batch
        ]
        
        # 并行处理当前批次
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(process_single_case, batch_args)
        
        # 打印结果
        for result in results:
            print(f"  {result}")
    
    print("\n所有case处理完成！")

