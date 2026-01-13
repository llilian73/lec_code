import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def create_custom_colormaps():
    cooling_colors = ['#FFFFFF', '#3182BD']
    cooling_cmap = LinearSegmentedColormap.from_list('cooling', cooling_colors)
    heating_colors = ['#FFFFFF', '#DE2D26']
    heating_cmap = LinearSegmentedColormap.from_list('heating', heating_colors)
    total_colors = ['#FFFFFF', '#FF8C00', '#CC5500']
    total_cmap = LinearSegmentedColormap.from_list('total', total_colors)
    return {
        'cooling': cooling_cmap,
        'heating': heating_cmap,
        'total': total_cmap
    }

def calculate_average_data(data_dir, years, case_num):
    """计算每个国家5年平均值，返回dict"""
    case_key = f"case{case_num}_summary.csv"
    country_stats = {}
    for year in years:
        year_path = os.path.join(data_dir, str(year), case_key)
        if not os.path.exists(year_path):
            continue
        # 关键修正：避免将 'NA' 识别为缺失，确保纳米比亚代码保留为字符串 'NA'
        df = pd.read_csv(year_path, keep_default_na=False, dtype={'country': str})
        for _, row in df.iterrows():
            # 统一清洗国家代码：去空格并大写
            country = str(row['country']).strip().upper()
            if country not in country_stats:
                country_stats[country] = {
                    'total_difference': [],
                    'total_reduction': [],
                    'cooling_difference': [],
                    'cooling_reduction': [],
                    'heating_difference': [],
                    'heating_reduction': []
                }
            for col in country_stats[country]:
                # 将值安全转换为数值，无法转换的置为 NaN，避免字符串参与均值计算
                value = pd.to_numeric(row.get(col, np.nan), errors='coerce')
                country_stats[country][col].append(value)
    # 计算平均值
    avg_data = []
    for country, stats in country_stats.items():
        avg_row = {'country': country}
        for col, values in stats.items():
            avg_row[col] = np.nanmean(values) if values else np.nan
        avg_data.append(avg_row)
    return avg_data

def save_average_csv(avg_data, output_path):
    df = pd.DataFrame(avg_data)
    df.to_csv(output_path, index=False)

def plot_average_map(avg_data, shapefile_path, save_path, case_num):
    world = gpd.read_file(shapefile_path)
    world = world[world['CONTINENT'] != 'Antarctica']
    
    # 参考 draw_map.py：针对少数国家用 NAME 匹配，其余用 ISO_A2
    code_mapping = {
        'FR': 'France',
        'NO': 'Norway',
        'US': 'United States of America',
        'AU': 'Australia'
    }

    # 准备平均数据 DataFrame，并统一清洗国家代码
    df_avg = pd.DataFrame(avg_data)
    if 'country' not in df_avg.columns:
        return
    df_avg['country'] = df_avg['country'].astype(str).str.strip().str.upper()
    # 按国家代码构造字典
    country_data = {row['country']: row for _, row in df_avg.iterrows()}

    colormaps = create_custom_colormaps()
    manual_vmin_vmax = {
        'total_difference': (0, 1200),
        'cooling_difference': (0, 1200),
        'heating_difference': (0, 1200)
    }
    reduction_vmin_vmax = {
        'total_reduction': (0, 100),
        'cooling_reduction': (0, 100),
        'heating_reduction': (0, 100)
    }

    # 绘制 difference map（与 draw_map.py 相同策略：逐条赋值 + 特殊映射）
    data_types = [
        ('total_difference', 'total'),
        ('cooling_difference', 'cooling'),
        ('heating_difference', 'heating')
    ]
    for col, cmap_key in data_types:
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        world['value'] = np.nan
        # 逐条写入，兼容 FR/NO 等用 NAME 匹配
        for code, row in country_data.items():
            val = row.get(col, np.nan)
            if pd.notna(val):
                if code in code_mapping:
                    country_name = code_mapping[code]
                    world.loc[world['NAME'] == country_name, 'value'] = val
                else:
                    world.loc[world['ISO_A2'] == code, 'value'] = val
        world.boundary.plot(ax=ax, linewidth=0.5, color='black')
        vmin, vmax = manual_vmin_vmax[col]
        world.plot(column='value',
                   ax=ax,
                   missing_kwds={'color': 'lightgrey'},
                   legend=True,
                   legend_kwds={'label': f'{cmap_key.capitalize()} Demand Difference (kWh/person)',
                                'orientation': 'horizontal',
                                'shrink': 0.6, 'pad': 0.02},
                   cmap=colormaps[cmap_key],
                   vmin=vmin,
                   vmax=vmax)
        plt.title(f'{cmap_key.capitalize()} Demand Difference (kWh/person) - Case {case_num} - 5-Year Average', fontsize=16)
        ax.axis('off')
        output_file = os.path.join(save_path, f"{cmap_key}_demand_difference_map_case{case_num}_average.pdf")
        plt.savefig(output_file, format="pdf", bbox_inches='tight')
        print(f"已保存: {output_file}")
        plt.close()

    # 绘制 reduction map
    reduction_types = [
        ('total_reduction', 'total'),
        ('cooling_reduction', 'cooling'),
        ('heating_reduction', 'heating')
    ]
    for col, cmap_key in reduction_types:
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        world['value'] = np.nan
        for code, row in country_data.items():
            val = row.get(col, np.nan)
            if pd.notna(val):
                if code in code_mapping:
                    country_name = code_mapping[code]
                    world.loc[world['NAME'] == country_name, 'value'] = val
                else:
                    world.loc[world['ISO_A2'] == code, 'value'] = val
        world.boundary.plot(ax=ax, linewidth=0.5, color='black')
        vmin, vmax = reduction_vmin_vmax[col]
        world.plot(column='value',
                   ax=ax,
                   missing_kwds={'color': 'lightgrey'},
                   legend=True,
                   legend_kwds={'label': f'{cmap_key.capitalize()} Demand Reduction (%)',
                                'orientation': 'horizontal',
                                'shrink': 0.6, 'pad': 0.02},
                   cmap=colormaps[cmap_key],
                   vmin=vmin,
                   vmax=vmax)
        plt.title(f'{cmap_key.capitalize()} Demand Reduction (%) - Case {case_num} - 5-Year Average', fontsize=16)
        ax.axis('off')
        output_file = os.path.join(save_path, f"{cmap_key}_demand_reduction_map_case{case_num}_average.pdf")
        plt.savefig(output_file, format="pdf", bbox_inches='tight')
        print(f"已保存: {output_file}")
        plt.close()

if __name__ == "__main__":
    data_dir = r"Z:\local_environment_creation\energy_consumption\2016-2020result\figure_maps_and_data\per_capita\data"
    shapefile_path = r"Z:\local_environment_creation\shapefiles\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp"
    output_dir = os.path.join(data_dir, "average")
    os.makedirs(output_dir, exist_ok=True)
    years = range(2016, 2021)
    for case_num in range(1, 21):
        print(f"正在处理 case{case_num} …")
        avg_data = calculate_average_data(data_dir, years, case_num)
        csv_path = os.path.join(output_dir, f"case{case_num}_summary_average.csv")
        save_average_csv(avg_data, csv_path)
        print(f"已保存: {csv_path}")
        plot_average_map(avg_data, shapefile_path, output_dir, case_num)
        print(f"完成 case{case_num}\n")
