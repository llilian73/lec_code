"""
能耗分析地图绘制工具

功能概述：
本工具用于分析和可视化不同案例下的建筑能耗数据，生成世界地图形式的能耗差异和节能比例图表。

输入数据：
1. CSV文件：包含各地区能耗数据的summary结果文件（格式：{region}_summary_p_results.csv）
   - 文件包含ref（参考案例）和case1-case9（不同节能案例）的数据
   - 数据字段包括：total_demand_sum_p(kWh/person)、heating_demand_sum_p(kWh/person)、cooling_demand_sum_p(kWh/person)
   - 以及对应的reduction百分比数据

2. 地图文件：世界地图shapefile文件，用于地理可视化

主要功能：
1. 能耗差值分析（d_reduction_difference_map）：
   - 计算ref案例与指定case案例之间的能耗差值
   - 生成总能耗、供暖能耗、制冷能耗的差值世界地图
   - 差值 = ref能耗 - case能耗（正值表示节能效果）

2. 节能比例分析（d_reduction_map）：
   - 直接使用case案例中的节能百分比数据
   - 生成总能耗、供暖能耗、制冷能耗的节能比例世界地图
   - 显示各地区的节能效果百分比（0-100%）

输出结果：
1. PDF格式的世界地图文件：
   - {data_type}_demand_difference_map_case{case_num}.pdf（差值地图）
   - {data_type}_demand_reduction_map_case{case_num}.pdf（比例地图）
   - data_type包括：total、heating、cooling
   - case_num为1-9的案例编号

2. 自定义颜色映射：
   - 冷却需求：白色到深蓝色渐变
   - 供暖需求：白色到深红色渐变  
   - 总需求：白色到橙色渐变

使用场景：
- 建筑节能政策效果评估
- 不同气候区域节能潜力分析
- 国际建筑能耗对比研究
- 节能技术推广效果可视化
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def create_custom_colormaps():
    """创建自定义颜色映射"""
    # 冷却需求：浅蓝到深蓝
    cooling_colors = ['#FFFFFF', '#3182BD']
    cooling_cmap = LinearSegmentedColormap.from_list('cooling', cooling_colors)
    
    # 供暖需求：浅红到深红
    heating_colors = ['#FFFFFF', '#DE2D26']
    heating_cmap = LinearSegmentedColormap.from_list('heating', heating_colors)
    
    # 总需求：浅蓝过渡到深红
    total_colors = ['#FFFFFF', '#FF8C00', '#CC5500']
    total_cmap = LinearSegmentedColormap.from_list('total', total_colors)
    
    return {
        'cooling': cooling_cmap,
        'heating': heating_cmap,
        'total': total_cmap
    }

def load_demand_difference(base_paths, case_num, year):
    """加载所有地区的能耗差值数据（按年份）
    Args:
        base_paths: 基础路径列表
        case_num: 要计算的case编号（1-20）
        year: 年份（2016-2020）
    """
    difference_data = {
        'total': {},
        'heating': {},
        'cooling': {}
    }
    case_key = f"case{case_num}"
    for base_path in base_paths:
        # 按年份查找summary_p目录
        year_path = os.path.join(base_path)
        if not os.path.exists(year_path):
            continue
        files = [f for f in os.listdir(year_path) if f.endswith(f'_{year}_summary_results.csv')]
        for file in files:
            region = file.split('_')[0]
            if '.' in region:
                continue
            try:
                # 关键修正：避免将 'NA' 识别为缺失值
                df = pd.read_csv(os.path.join(year_path, file), index_col=0, keep_default_na=False, na_values=[''])
                if 'ref' not in df.index or case_key not in df.index:
                    continue
                ref_data = df.loc['ref']
                case_data = df.loc[case_key]
                difference_data['total'][region] = ref_data['total_demand_sum(TWh)'] - case_data['total_demand_sum(TWh)']
                difference_data['heating'][region] = ref_data['heating_demand_sum(TWh)'] - case_data['heating_demand_sum(TWh)']
                difference_data['cooling'][region] = ref_data['cooling_demand_sum(TWh)'] - case_data['cooling_demand_sum(TWh)']
            except Exception as e:
                print(f"处理{file}时出错: {e}")
                continue
    
    return difference_data

def d_reduction_difference_map(difference_data, shapefile_path, save_path, year, case_num):
    """绘制能耗差值世界地图（按年份）
    Args:
        difference_data: 已转换为三字母代码的差值数据字典 {data_type: {code3: value}}
        shapefile_path: 地图文件路径
        save_path: 保存路径
        year: 年份（2016-2020）
        case_num: case编号（1-20）
    """
    world = gpd.read_file(shapefile_path)
    colormaps = create_custom_colormaps()
    
    # 设置数据范围
    manual_vmin_vmax = {
        'total': (0, 900),
        'heating': (0, 500),
        'cooling': (0, 900)
    }
    for data_type, differences in difference_data.items():
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        world['difference'] = np.nan
        
        for code3, value in differences.items():
            # 直接使用GID_0（三字母代码）匹配
            world.loc[world['GID_0'] == code3, 'difference'] = value
        
        # 特殊处理：香港和澳门显示中国的数据
        if 'CHN' in differences:
            chn_value = differences['CHN']
            # 香港（HKG）显示中国的数据
            world.loc[world['GID_0'] == 'HKG', 'difference'] = chn_value
            # 澳门（MAC）显示中国的数据
            world.loc[world['GID_0'] == 'MAC', 'difference'] = chn_value
        
        # 绘制国家边界
        world.boundary.plot(ax=ax, linewidth=0.5, color='black')
        
        vmin = manual_vmin_vmax[data_type][0]
        vmax = manual_vmin_vmax[data_type][1]
        
        world.plot(column='difference', 
                  ax=ax,
                  missing_kwds={'color': 'lightgrey'},
                  legend=True,
                  legend_kwds={'label': 'Demand Difference (TWh)',
                              'orientation': 'horizontal',
                              'shrink': 0.6,'pad': 0.02},
                  cmap=colormaps[data_type],
                  vmin=vmin,
                  vmax=vmax)
        plt.title(f'{data_type.capitalize()} Demand Difference (TWh) - Case {case_num} - {year}', fontsize=16)
        ax.axis('off')
        
        output_file = os.path.join(save_path, f"{data_type}_demand_difference_map_case{case_num}.pdf")
        plt.savefig(output_file, format="pdf", bbox_inches='tight')
        plt.close()
    
    return difference_data

def load_reduction_data(base_paths, case_num, year):
    """加载所有地区的节能比例数据（按年份）
    Args:
        base_paths: 基础路径列表
        case_num: 要加载的case编号（1-20）
        year: 年份（2016-2020）
    """
    reduction_data = {
        'total': {},
        'heating': {},
        'cooling': {}
    }
    case_key = f"case{case_num}"
    for base_path in base_paths:
        year_path = os.path.join(base_path)
        if not os.path.exists(year_path):
            continue
        files = [f for f in os.listdir(year_path) if f.endswith(f'_{year}_summary_results.csv')]
        for file in files:
            region = file.split('_')[0]
            if '.' in region:
                continue
                
            try:
                # 关键修正：避免将 'NA' 识别为缺失值
                df = pd.read_csv(os.path.join(year_path, file), index_col=0, keep_default_na=False, na_values=[''])
                if case_key not in df.index:
                    continue
                case_data = df.loc[case_key]
                reduction_data['total'][region] = case_data['total_demand_reduction(%)']
                reduction_data['heating'][region] = case_data['heating_demand_reduction(%)']
                reduction_data['cooling'][region] = case_data['cooling_demand_reduction(%)']
            except Exception as e:
                print(f"处理{file}时出错: {e}")
                continue

    return reduction_data

def d_reduction_map(reduction_data, shapefile_path, save_path, year, case_num):
    """绘制节能比例世界地图（按年份）
    Args:
        reduction_data: 已转换为三字母代码的节能比例数据字典 {data_type: {code3: value}}
        shapefile_path: 地图文件路径
        save_path: 保存路径
        year: 年份（2016-2020）
        case_num: case编号（1-20）
    """
    world = gpd.read_file(shapefile_path)
    
    # 创建自定义颜色映射
    colormaps = create_custom_colormaps()
    
    for data_type, reductions in reduction_data.items():
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        
        world['reduction'] = np.nan
        
        for code3, value in reductions.items():
            # 直接使用GID_0（三字母代码）匹配
            world.loc[world['GID_0'] == code3, 'reduction'] = value
        
        # 特殊处理：香港和澳门显示中国的数据
        if 'CHN' in reductions:
            chn_value = reductions['CHN']
            # 香港（HKG）显示中国的数据
            world.loc[world['GID_0'] == 'HKG', 'reduction'] = chn_value
            # 澳门（MAC）显示中国的数据
            world.loc[world['GID_0'] == 'MAC', 'reduction'] = chn_value
        
        # 绘制国家边界
        world.boundary.plot(ax=ax, linewidth=0.5, color='black')
        
        world.plot(column='reduction', 
                  ax=ax,
                  missing_kwds={'color': 'lightgrey'},
                  legend=True,
                  legend_kwds={'label': 'Reduction (%)',
                              'orientation': 'horizontal',
                              'shrink': 0.6,'pad': 0.02},
                  cmap=colormaps[data_type],
                  vmin=0,
                  vmax=100)
        plt.title(f'{data_type.capitalize()} Demand Reduction (%) - Case {case_num} - {year}', fontsize=16)
        ax.axis('off')
        
        output_file = os.path.join(save_path, f"{data_type}_demand_reduction_map_case{case_num}.pdf")
        plt.savefig(output_file, format="pdf", bbox_inches='tight')
        plt.close()

    return reduction_data