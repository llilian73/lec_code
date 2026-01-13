"""
国家投资回收期（PP）地图绘制工具

功能概述：
本工具用于绘制国家投资回收期（PP）的地图热图，基于 country_yearly_per_cost_USD_PP.csv 的输出数据。

输入数据：
- 文件：Z:\local_environment_creation\cost\country_yearly_per_cost_USD_PP.csv
- 数据列：Continent,Country_Code,Country_Name,equipment_cost,maintenance_cost,total_cost,save_elec_cost,PP

输出结果：
- PNG地图文件：Z:\local_environment_creation\cost\method2\country_PP_map.png（600 DPI）
- 使用 PP（年）数据绘制地图热图
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 配置文件路径
INPUT_FILE = r"Z:\local_environment_creation\cost\method2\country_yearly_per_cost_USD_PP.csv"
SHAPEFILE_PATH = r"Z:\local_environment_creation\shapefiles\全球国家边界\world_border2.shp"
COUNTRIES_INFO_FILE = r"Z:\local_environment_creation\all_countries_info.csv"
OUTPUT_DIR = r"Z:\local_environment_creation\cost\method2"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "country_PP_map.png")

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

def read_csv_with_encoding(file_path, keep_default_na=True):
    """读取CSV文件，尝试多种编码"""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk', 'gb2312', 'utf-8-sig']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, keep_default_na=keep_default_na)
            print(f"成功使用编码 '{encoding}' 读取文件: {os.path.basename(file_path)}")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    raise ValueError(f"无法使用常见编码读取文件: {file_path}")

def create_cost_colormap():
    """创建PP数据的颜色映射（白->蓝）"""
    colors = ['#FFFFFF', '#3182BD']
    cmap = LinearSegmentedColormap.from_list('cost', colors)
    return cmap

def load_cost_data():
    """加载PP数据"""
    print(f"读取PP数据文件: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"成本数据文件不存在: {INPUT_FILE}")
    
    df = read_csv_with_encoding(INPUT_FILE, keep_default_na=False)
    print(f"成功加载PP数据，共 {len(df)} 个国家")
    
    # 检查必要的列是否存在
    if 'Country_Code' not in df.columns:
        raise ValueError("CSV文件中缺少'Country_Code'列")
    if 'PP' not in df.columns:
        raise ValueError("CSV文件中缺少'PP'列")
    
    # 创建国家代码到PP的映射
    country_data = {}
    for _, row in df.iterrows():
        # 处理Country_Code，确保'NA'不被当作缺失值
        country_code_raw = row['Country_Code']
        if country_code_raw == '' or (isinstance(country_code_raw, float) and pd.isna(country_code_raw)):
            country_code = ''
        else:
            country_code = str(country_code_raw).strip().upper()
        
        if not country_code:
            continue
        
        pp_val = row['PP']
        if pd.notna(pp_val):
            country_data[country_code] = {
                'PP': pp_val,
                'Country_Name': row.get('Country_Name', ''),
                'equipment_cost': row.get('equipment_cost', np.nan),
                'maintenance_cost': row.get('maintenance_cost', np.nan),
                'save_elec_cost': row.get('save_elec_cost', np.nan)
            }
    
    print(f"创建了 {len(country_data)} 个国家的PP数据映射")
    return country_data

def plot_cost_map(country_data, shapefile_path, output_path, code2_to_code3):
    """绘制PP地图
    Args:
        country_data: 国家数据字典（键为二字母代码）
        shapefile_path: shapefile路径
        output_path: 输出文件路径
        code2_to_code3: 二字母代码到三字母代码的映射
    """
    print(f"加载shapefile: {shapefile_path}")
    world = gpd.read_file(shapefile_path)
    
    # 创建颜色映射
    cmap = create_cost_colormap()
    
    # 初始化值列
    world['value'] = np.nan
    
    # 将二字母代码转换为三字母代码，并使用GID_0字段匹配
    for code2, data in country_data.items():
        val = data.get('PP', np.nan)
        if pd.notna(val):
            # 将二字母代码转换为三字母代码
            code3 = code2_to_code3.get(code2, None)
            if code3:
                # 使用GID_0（三字母代码）匹配
                world.loc[world['GID_0'] == code3, 'value'] = val
    
    # 计算数据的实际范围（排除NaN，用于统计信息）
    valid_values = world['value'].dropna()
    if len(valid_values) > 0:
        actual_min = valid_values.min()
        actual_max = valid_values.max()
        print(f"PP实际数据范围: {actual_min:.2f} - {actual_max:.2f} 年")
    else:
        actual_min = 0
        actual_max = 0
        print("警告：没有有效的PP数据")
    
    # 设置固定的颜色范围：0~10年
    vmin = 0
    vmax = 10
    print(f"热图颜色范围: {vmin:.0f} - {vmax:.0f} 年")
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    
    # 绘制国家边界
    world.boundary.plot(ax=ax, linewidth=0.5, color='black')
    
    # 绘制地图
    world.plot(column='value',
               ax=ax,
               missing_kwds={'color': 'lightgrey'},
               legend=True,
               legend_kwds={'label': 'Payback Period (years)',
                          'orientation': 'horizontal',
                          'shrink': 0.6, 'pad': 0.02},
               cmap=cmap,
               vmin=vmin,
               vmax=vmax)
    
    # 设置标题
    # plt.title('Payback Period (years)', fontsize=16, fontweight='bold', pad=20)
    
    ax.axis('off')
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为PNG
    plt.savefig(output_path, format="png", bbox_inches='tight', dpi=600)
    print(f"地图已保存至: {output_path}")
    plt.close()
    
    # 打印统计信息
    print("\n=== 统计信息 ===")
    print(f"有数据国家数: {len(valid_values)}")
    print(f"无数据国家数: {len(world) - len(valid_values)}")
    if len(valid_values) > 0:
        print(f"PP实际数据范围: {actual_min:.2f} - {actual_max:.2f} 年")
        print(f"热图颜色范围: {vmin:.0f} - {vmax:.0f} 年（固定）")
        print(f"平均PP: {valid_values.mean():.2f} 年")
        print(f"中位数PP: {valid_values.median():.2f} 年")
    
    # 显示PP最短和最长的前5个国家
    if len(country_data) > 0:
        # 创建排序列表（排除NaN值）
        valid_countries = [(code, data) for code, data in country_data.items() 
                          if pd.notna(data.get('PP', np.nan))]
        sorted_countries = sorted(valid_countries, key=lambda x: x[1]['PP'])
        
        if len(sorted_countries) > 0:
            print("\nPP最短的前5个国家:")
            for i, (code, data) in enumerate(sorted_countries[:5]):
                pp_val = data['PP']
                print(f"  {i+1}. {data.get('Country_Name', code)} ({code}): {pp_val:.2f} 年")
            
            print("\nPP最长的前5个国家:")
            for i, (code, data) in enumerate(sorted_countries[-5:]):
                pp_val = data['PP']
                print(f"  {i+1}. {data.get('Country_Name', code)} ({code}): {pp_val:.2f} 年")

def main():
    """主函数"""
    print("开始绘制国家PP地图...")
    print("=" * 60)
    
    try:
        # 1. 加载国家代码映射
        code2_to_code3 = load_country_code_mapping()
        
        # 2. 加载成本数据
        country_data = load_cost_data()
        
        # 3. 检查shapefile是否存在
        if not os.path.exists(SHAPEFILE_PATH):
            raise FileNotFoundError(f"Shapefile不存在: {SHAPEFILE_PATH}")
        
        # 4. 绘制地图
        plot_cost_map(country_data, SHAPEFILE_PATH, OUTPUT_FILE, code2_to_code3)
        
        print("\n" + "=" * 60)
        print("成本地图绘制完成！")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
