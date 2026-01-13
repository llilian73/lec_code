"""
国家成本地图绘制工具

功能概述：
本工具用于绘制国家人均成本地图热图，基于4_year.py的输出数据。

输入数据：
- 文件：Z:\local_environment_creation\cost\country_yearly_per_cost_USD_PP.csv
- 数据列：Continent,Country_Code,Country_Name,equipment_cost,maintenance_cost,save_elec_cost,total_cost

输出结果：
- PDF地图文件：Z:\local_environment_creation\cost\country_total_cost_map.pdf
- 使用total_cost数据绘制地图热图
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 配置文件路径
INPUT_FILE = r"Z:\local_environment_creation\cost\country_yearly_per_cost_USD_PP.csv"
SHAPEFILE_PATH = r"Z:\local_environment_creation\shapefiles\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp"
OUTPUT_DIR = r"Z:\local_environment_creation\cost"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "country_total_cost_map.pdf")

def create_cost_colormap():
    """创建成本数据的颜色映射"""
    colors = ['#FFFFFF', '#3182BD']
    cmap = LinearSegmentedColormap.from_list('cost', colors)
    return cmap

def load_cost_data():
    """加载成本数据"""
    print(f"读取成本数据文件: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"成本数据文件不存在: {INPUT_FILE}")
    
    df = pd.read_csv(INPUT_FILE)
    print(f"成功加载成本数据，共 {len(df)} 个国家")
    
    # 检查必要的列是否存在
    if 'Country_Code' not in df.columns:
        raise ValueError("CSV文件中缺少'Country_Code'列")
    if 'total_cost' not in df.columns:
        raise ValueError("CSV文件中缺少'total_cost'列")
    
    # 创建国家代码到total_cost的映射
    country_data = {}
    for _, row in df.iterrows():
        country_code = str(row['Country_Code']).strip().upper()
        total_cost = row['total_cost']
        if pd.notna(total_cost):
            country_data[country_code] = {
                'total_cost': total_cost,
                'Country_Name': row.get('Country_Name', ''),
                'equipment_cost': row.get('equipment_cost', np.nan),
                'maintenance_cost': row.get('maintenance_cost', np.nan),
                'save_elec_cost': row.get('save_elec_cost', np.nan)
            }
    
    print(f"创建了 {len(country_data)} 个国家的成本数据映射")
    return country_data

def plot_cost_map(country_data, shapefile_path, output_path):
    """绘制成本地图"""
    print(f"加载shapefile: {shapefile_path}")
    world = gpd.read_file(shapefile_path)
    world = world[world['CONTINENT'] != 'Antarctica']
    
    # 特殊国家映射（参考draw_map_average_per.py）
    code_mapping = {
        'FR': 'France',
        'NO': 'Norway',
        'US': 'United States of America',
        'AU': 'Australia'
    }
    
    # 创建颜色映射
    cmap = create_cost_colormap()
    
    # 初始化值列
    world['value'] = np.nan
    
    # 逐条写入数据，兼容特殊国家映射
    for code, data in country_data.items():
        val = data.get('total_cost', np.nan)
        if pd.notna(val):
            if code in code_mapping:
                # 使用NAME匹配特殊国家
                country_name = code_mapping[code]
                world.loc[world['NAME'] == country_name, 'value'] = val
            elif code == 'SOMALILAND' or code == 'SO':
                # Somaliland 通过 NAME 匹配
                if 'Somaliland' in world['NAME'].values:
                    world.loc[world['NAME'] == 'Somaliland', 'value'] = val
                # 如果有Somalia，也赋值
                if 'Somalia' in world['NAME'].values:
                    world.loc[world['NAME'] == 'Somalia', 'value'] = val
            else:
                # 使用ISO_A2匹配
                world.loc[world['ISO_A2'] == code, 'value'] = val
    
    # 计算数据的范围（排除NaN）
    valid_values = world['value'].dropna()
    if len(valid_values) > 0:
        vmin = valid_values.min()
        vmax = valid_values.max()
        print(f"成本数据范围: {vmin:.4f} - {vmax:.4f} 美元")
    else:
        vmin = 0
        vmax = 1
        print("警告：没有有效的成本数据")
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    
    # 绘制国家边界
    # 对于 Somalia 和 Somaliland，合并它们的几何体后再绘制外部边界
    somalia_somaliland = world[world['NAME'].isin(['Somaliland', 'Somalia'])]
    other_countries = world[~world['NAME'].isin(['Somaliland', 'Somalia'])]
    
    # 绘制其他国家的边界
    other_countries.boundary.plot(ax=ax, linewidth=0.5, color='black')
    
    # 合并 Somalia 和 Somaliland 的几何体，只绘制外部边界
    if len(somalia_somaliland) > 0:
        from shapely.ops import unary_union
        merged_geometry = unary_union(somalia_somaliland.geometry)
        # 创建一个临时 GeoDataFrame 来绘制合并后的边界
        merged_gdf = gpd.GeoDataFrame([{'geometry': merged_geometry}], crs=world.crs)
        merged_gdf.boundary.plot(ax=ax, linewidth=0.5, color='black')
    
    # 绘制地图
    world.plot(column='value',
               ax=ax,
               missing_kwds={'color': 'lightgrey'},
               legend=True,
               legend_kwds={'label': 'Total Annual Cost per Capita (USD)',
                          'orientation': 'horizontal',
                          'shrink': 0.6, 'pad': 0.02},
               cmap=cmap,
               vmin=vmin,
               vmax=vmax)
    
    # 设置标题
    plt.title('Total Annual Cost per Capita (USD) - Equipment + Maintenance - Electricity Savings', 
              fontsize=16, fontweight='bold', pad=20)
    
    ax.axis('off')
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为PDF
    plt.savefig(output_path, format="pdf", bbox_inches='tight', dpi=300)
    print(f"地图已保存至: {output_path}")
    plt.close()
    
    # 打印统计信息
    print("\n=== 统计信息 ===")
    print(f"有数据国家数: {len(valid_values)}")
    print(f"无数据国家数: {len(world) - len(valid_values)}")
    print(f"成本范围: {vmin:.4f} - {vmax:.4f} 美元")
    print(f"平均成本: {valid_values.mean():.4f} 美元")
    print(f"中位数成本: {valid_values.median():.4f} 美元")
    
    # 显示成本最低和最高的前5个国家
    if len(country_data) > 0:
        # 创建排序列表
        sorted_countries = sorted(country_data.items(), key=lambda x: x[1]['total_cost'])
        print("\n成本最低的前5个国家:")
        for i, (code, data) in enumerate(sorted_countries[:5]):
            print(f"  {i+1}. {data.get('Country_Name', code)} ({code}): {data['total_cost']:.4f} 美元")
        
        print("\n成本最高的前5个国家:")
        for i, (code, data) in enumerate(sorted_countries[-5:]):
            print(f"  {i+1}. {data.get('Country_Name', code)} ({code}): {data['total_cost']:.4f} 美元")

def main():
    """主函数"""
    print("开始绘制国家成本地图...")
    print("=" * 60)
    
    try:
        # 1. 加载成本数据
        country_data = load_cost_data()
        
        # 2. 检查shapefile是否存在
        if not os.path.exists(SHAPEFILE_PATH):
            raise FileNotFoundError(f"Shapefile不存在: {SHAPEFILE_PATH}")
        
        # 3. 绘制地图
        plot_cost_map(country_data, SHAPEFILE_PATH, OUTPUT_FILE)
        
        print("\n" + "=" * 60)
        print("成本地图绘制完成！")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
