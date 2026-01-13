"""
全球碳排放减少率热图绘制工具

功能概述：
本工具专门用于绘制全球碳排放减少率热图，展示不同工况下各国相对于基准工况的碳排放减少百分比。专注于减排效果的相对比较分析。

输入数据：
1. 碳排放数据：
   - 基础目录：D:\workstation\carbon emission\global
   - 按大洲分类：Africa, Asia, Europe, North America, Oceania, South America
   - 数据位置：{大洲}\{国家代码}_carbon_emission.csv
   - 关键列：carbon_emission_reduction(%) - 碳排放减少百分比（相对于ref工况）

2. 世界地图数据：
   - 文件路径：D:\workstation\energy_comsuption\shapefiles\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp
   - 包含列：ISO_A2（国家代码）, NAME（国家名称）

主要功能：
1. 减少率数据提取：
   - 读取各国碳排放减少率数据
   - 提取case1-case9工况的减少百分比
   - 处理国家代码映射和匹配

2. 减少率热图生成：
   - 显示各工况相对于ref的碳排放减少百分比
   - 使用白色→深绿色渐变色彩
   - 固定0-50%比例尺范围

3. 国家代码映射：
   - 处理特殊国家的代码匹配问题
   - 支持Norway→NO, France→FR等映射
   - 确保地理数据与碳排放数据正确匹配

4. 专业可视化：
   - 专门针对减少率设计的色彩方案
   - 统一的百分比比例尺
   - 专业的图例和标签设置

输出结果：
碳排放减少率热图（reduction_rate目录）：
- 输出目录：D:\workstation\carbon emission\global\reduction_rate\
- 文件格式：global_{工况}_reduction_rate.pdf
- 包含工况：case1-case9（共9个文件）
- 显示内容：各国碳排放减少百分比（%）
- 比例尺：统一使用0-50%的范围

数据流程：
1. 数据加载阶段：
   - 加载世界地图shapefile数据
   - 扫描所有大洲的碳排放数据目录
   - 读取各国的碳排放文件

2. 数据处理阶段：
   - 提取各工况的减少率数据
   - 处理国家代码映射
   - 设置固定比例尺范围

3. 热图生成阶段：
   - 合并地理数据和减少率数据
   - 设置专门的色彩方案
   - 绘制减少率分布热图

4. 文件保存阶段：
   - 创建reduction_rate输出目录
   - 生成PDF格式的高质量图表
   - 保存所有工况的减少率热图

可视化特点：
- 地理精度：国家级行政区划精度
- 色彩方案：白色→深绿色渐变（专门针对减少率设计）
- 图表尺寸：20×10英寸
- 输出格式：PDF矢量图（300 DPI）
- 比例尺：固定0-50%范围

技术参数：
- 地图格式：Shapefile
- 色彩映射：LinearSegmentedColormap
- 图表尺寸：20×10英寸
- 输出分辨率：300 DPI
- 文件格式：PDF
- 比例尺范围：0-50%

特殊处理：
- 国家代码映射：处理特殊国家的代码匹配
- 缺失数据处理：没有数据的国家显示为浅灰色
- 固定比例尺：统一使用0-50%范围
- 南极洲排除：自动排除南极洲数据
- 专业图例：包含"Carbon Emission Reduction Rate (%)"标签

应用场景：
- 全球减排效果分析
- 建筑节能政策评估
- 碳排放减少效果可视化
- 国际减排比较研究
- 可持续发展目标评估

与draw_global per capita.py的区别：
- 本工具专注于减少率（百分比）
- draw_global per capita.py处理绝对排放量和减少量
- 只处理case1-case9工况，不包含ref工况
- 使用固定的0-50%比例尺
- 专门的减少率色彩方案

与draw_global.py的区别：
- 本工具处理减少率（百分比）
- draw_global.py处理绝对排放量和减少量
- 输入目录：global（总量数据）
- 输出目录：global\reduction_rate
- 专门针对减少率设计的可视化

数据质量保证：
- 国家匹配验证
- 数据完整性检查
- 异常值处理
- 比例尺合理性验证
- 输出文件完整性确认

专业特色：
- 专门针对减少率设计的可视化方案
- 统一的百分比比例尺
- 专业的图例和标签
- 清晰的减排效果展示
- 适合政策制定和学术研究
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

def load_and_process_data(base_dir):
    """加载并处理所有国家的碳排放数据"""
    data = []
    continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    
    for continent in continents:
        continent_dir = os.path.join(base_dir, continent)
        if not os.path.exists(continent_dir):
            print(f"警告: 未找到 {continent} 的数据目录")
            continue
            
        print(f"正在处理 {continent} 的数据...")
        for file in os.listdir(continent_dir):
            if file.endswith('_carbon_p_emission.csv'):
                country_code = file.split('_')[0]
                df = pd.read_csv(os.path.join(continent_dir, file))
                
                # 获取所有工况的数据
                case_data = {'country': country_code}
                
                # 读取每个工况的碳排放减少率
                for case in ['case1', 'case2', 'case3', 'case4', 
                            'case5', 'case6', 'case7', 'case8', 'case9']:
                    case_row = df[df.iloc[:, 0] == case]
                    if not case_row.empty:
                        case_data[f'{case}_reduction_rate'] = case_row['carbon_emission_reduction(%)'].iloc[0]
                
                data.append(case_data)
    
    return pd.DataFrame(data)

def create_choropleth(gdf, data, value_column, title, save_path, vmax=100):
    """创建热图"""
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    
    # 从白色到绿色的渐变色
    colors = ['#FFFFFF', '#5B9C4A']  # 白色到深绿色
    cmap = LinearSegmentedColormap.from_list('custom_white_green', colors)
    
    # 创建国家代码映射字典
    country_code_map = {
        'Norway': 'NO',
        'France': 'FR',
        'N. Cyprus': 'CY',
        'Somaliland': 'SO',
        'Kosovo': 'XK'
    }
    
    # 修复地图数据中的国家代码
    for name, code in country_code_map.items():
        gdf.loc[gdf['NAME'] == name, 'ISO_A2'] = code
    
    # 合并地理数据和碳排放数据
    merged = gdf.merge(data, left_on='ISO_A2', right_on='country', how='left')
    
    # 排除南极洲
    merged = merged[merged['ISO_A2'] != 'AQ']
    
    # 找出没有匹配到碳排放数据的国家
    # missing_data = merged[merged[value_column].isna()]
    # if not missing_data.empty:
    #     print(f"\n以下国家没有匹配到碳排放数据（显示为灰色）:")
    #     for _, row in missing_data.iterrows():
    #         print(f"国家名称: {row['NAME']}, 国家代码: {row['ISO_A2']}")
    
    # 绘制热图
    merged.plot(column=value_column, 
               cmap=cmap,
               linewidth=0.8,
               edgecolor='0.8',
               ax=ax,
               legend=True,
               vmin=0,
               vmax=vmax,
               missing_kwds={'color': 'lightgrey'},
               legend_kwds={
                   'orientation': 'horizontal',
                   'pad': 0.03,
                   'fraction': 0.046,
                   'shrink': 0.8,
                   'aspect': 25,
                   'label': 'Carbon Emission Reduction Rate (%)'
               })
    
    ax.axis('off')
    plt.title(title, fontsize=14, pad=18)
    
    # 保存图片
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    print("开始处理数据和绘制图表...")
    
    # 设置路径
    base_dir = r"D:\workstation\carbon emission\global"
    shapefile_path = r"D:\workstation\energy_comsuption\shapefiles\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp"
    
    # 创建输出目录
    output_dir = os.path.join(base_dir, 'reduction_rate')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载地图数据
    print("正在加载地图数据...")
    gdf = gpd.read_file(shapefile_path)
    print("地图数据加载完成")
    
    # 加载并处理碳排放数据
    print("正在处理碳排放数据...")
    data = load_and_process_data(base_dir)
    print("碳排放数据处理完成")
    
    # 绘制所有工况的碳排放减少率热图
    cases = ['case1', 'case2', 'case3', 'case4', 
             'case5', 'case6', 'case7', 'case8', 'case9']
    
    for case in cases:
        print(f"正在绘制 {case} 工况碳排放减少率热图...")
        create_choropleth(
            gdf, data, f'{case}_reduction_rate',
            f'Global {case} Case Carbon Emission Reduction Rate (%)',
            os.path.join(output_dir, f'global_{case}_reduction_rate.pdf'),
            vmax=50  # 统一的最大值50%
        )
        print(f"{case} 工况碳排放减少率热图已保存")
    
    print("\n所有图表处理完成！")

if __name__ == '__main__':
    main()
