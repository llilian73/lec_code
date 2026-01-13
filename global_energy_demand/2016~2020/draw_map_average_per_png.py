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
    cooling_colors = ['#FFFFFF', '#3182BD']
    cooling_cmap = LinearSegmentedColormap.from_list('cooling', cooling_colors)
    heating_colors = ['#FFFFFF', '#DE2D26']
    heating_cmap = LinearSegmentedColormap.from_list('heating', heating_colors)
    total_colors = ['#FFFFFF', '#FF8C00', '#CC5500']  #['#FFFFFF', '#FF8C00', '#CC5500']
    total_cmap = LinearSegmentedColormap.from_list('total', total_colors)
    return {
        'cooling': cooling_cmap,
        'heating': heating_cmap,
        'total': total_cmap
    }

def add_tropic_lines(ax, world):
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
                if pd.isna(value):
                    continue
                # 对节能率、节能量为负值的年份进行排除
                if col.endswith('difference') or col.endswith('reduction'):
                    if value < 0:
                        continue
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

def plot_average_map(avg_data, shapefile_path, save_path, case_num, code2_to_code3):
    """绘制平均地图
    Args:
        avg_data: 平均数据列表（country列为二字母代码）
        shapefile_path: shapefile路径
        save_path: 保存路径
        case_num: case编号
        code2_to_code3: 二字母代码到三字母代码的映射
    """
    world = gpd.read_file(shapefile_path)

    # 准备平均数据 DataFrame，并统一清洗国家代码
    df_avg = pd.DataFrame(avg_data)
    if 'country' not in df_avg.columns:
        return
    df_avg['country'] = df_avg['country'].astype(str).str.strip().str.upper()
    # 按国家代码构造字典（二字母代码）
    country_data = {row['country']: row for _, row in df_avg.iterrows()}
    
    # 创建三字母代码的数据字典（用于绘图）
    country_data_code3 = {}
    for code2, row in country_data.items():
        code3 = code2_to_code3.get(code2, None)
        if code3:
            country_data_code3[code3] = row

    colormaps = create_custom_colormaps()
    manual_vmin_vmax = {
        'total_difference': (0, 1400),
        'cooling_difference': (0, 1400),
        'heating_difference': (0, 1000)
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
        
        # 使用三字母代码匹配
        for code3, row in country_data_code3.items():
            val = row.get(col, np.nan)
            if pd.notna(val):
                # 直接使用GID_0（三字母代码）匹配
                world.loc[world['GID_0'] == code3, 'value'] = val
        
        # 特殊处理：香港和澳门显示中国的数据
        if 'CN' in country_data:
            cn_row = country_data['CN']
            cn_val = cn_row.get(col, np.nan)
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
        add_tropic_lines(ax, world)
        plt.title(f'{cmap_key.capitalize()} Demand Difference (kWh/person) - Case {case_num} - 5-Year Average', fontsize=16)
        ax.axis('off')
        output_file = os.path.join(save_path, f"{cmap_key}_demand_difference_map_case{case_num}_average.png")
        plt.savefig(output_file, format="png", bbox_inches='tight', dpi=600)
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
        
        # 使用三字母代码匹配
        for code3, row in country_data_code3.items():
            val = row.get(col, np.nan)
            if pd.notna(val):
                # 直接使用GID_0（三字母代码）匹配
                world.loc[world['GID_0'] == code3, 'value'] = val
        
        # 特殊处理：香港和澳门显示中国的数据
        if 'CN' in country_data:
            cn_row = country_data['CN']
            cn_val = cn_row.get(col, np.nan)
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
        add_tropic_lines(ax, world)
        plt.title(f'{cmap_key.capitalize()} Demand Reduction (%) - Case {case_num} - 5-Year Average', fontsize=16)
        ax.axis('off')
        output_file = os.path.join(save_path, f"{cmap_key}_demand_reduction_map_case{case_num}_average.png")
        plt.savefig(output_file, format="png", bbox_inches='tight', dpi=600)
        print(f"已保存: {output_file}")
        plt.close()

def process_single_case(args):
    """处理单个case的平均数据计算和地图绘制（用于并行处理）
    Args:
        args: 元组 (case_num, data_dir, years, shapefile_path, output_dir, code2_to_code3)
    """
    case_num, data_dir, years, shapefile_path, output_dir, code2_to_code3 = args
    
    try:
        print(f"正在处理 case{case_num} …")
        avg_data = calculate_average_data(data_dir, years, case_num)
        csv_path = os.path.join(output_dir, f"case{case_num}_summary_average.csv")
        save_average_csv(avg_data, csv_path)
        print(f"已保存: {csv_path}")
        plot_average_map(avg_data, shapefile_path, output_dir, case_num, code2_to_code3)
        return f"Case {case_num} 完成"
    except Exception as e:
        return f"Case {case_num} 处理失败: {str(e)}"


if __name__ == "__main__":
    data_dir = r"Z:\local_environment_creation\energy_consumption_gird\result\result\figure_maps_and_data\per_capita\data"
    shapefile_path = r"Z:\local_environment_creation\shapefiles\全球国家边界\world_border2.shp"
    output_dir = os.path.join(data_dir, "average")
    os.makedirs(output_dir, exist_ok=True)
    years = range(2016, 2021)
    
    # 准备所有case的参数列表
    case_numbers = list(range(1, 21))  # case1到case20
    batch_size = 10  # 每批处理10个case
    
    # 将case分批
    case_batches = [case_numbers[i:i + batch_size] for i in range(0, len(case_numbers), batch_size)]
    
    print(f"共 {len(case_numbers)} 个case，分为 {len(case_batches)} 批处理（每批 {batch_size} 个）")
    
    # 配置并行处理参数
    num_cores = multiprocessing.cpu_count()
    num_processes = min(num_cores, batch_size)  # 进程数不超过批次大小
    
    # 按批处理
    for batch_idx, case_batch in enumerate(case_batches, 1):
        print(f"\n处理第 {batch_idx}/{len(case_batches)} 批: case {case_batch[0]}-{case_batch[-1]}")
        
        # 准备当前批次的参数
        batch_args = [
            (case_num, data_dir, years, shapefile_path, output_dir, CODE2_TO_CODE3)
            for case_num in case_batch
        ]
        
        # 并行处理当前批次
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(process_single_case, batch_args)
        
        # 打印结果
        for result in results:
            print(f"  {result}")
    
    print("\n所有case处理完成！")
