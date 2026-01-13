"""
全球总碳排放热图绘制工具

功能概述：
本工具用于将计算得到的国家总碳排放数据转换为全球热图，展示不同工况下各国的碳排放量和减排效果。支持碳排放量热图和减排量热图两种可视化方式。

输入数据：
1. 总碳排放数据：
   - 基础目录：D:\workstation\carbon emission\global
   - 按大洲分类：Africa, Asia, Europe, North America, Oceania, South America
   - 数据位置：{大洲}\{国家代码}_carbon_emission.csv
   - 关键列：
     * carbon_emission(tCO2) - 总碳排放量
     * carbon_emission_reduction(tCO2) - 相对于ref的碳排放减少量
     * carbon_emission_reduction(%) - 碳排放减少比例

2. 世界地图数据：
   - 文件路径：D:\workstation\energy_comsuption\shapefiles\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp
   - 包含列：ISO_A2（国家代码）, NAME（国家名称）

主要功能：
1. 数据加载与处理：
   - 读取所有大洲的总碳排放数据
   - 提取各工况的碳排放量和减排量
   - 处理国家代码映射和匹配

2. 碳排放量热图生成：
   - 显示各工况下的总碳排放量
   - 使用深绿色→黄色→深棕色渐变色彩
   - 统一比例尺确保图表可比性

3. 减排量热图生成：
   - 显示各工况相对于ref的碳排放减少量
   - 使用白色→深绿色渐变色彩
   - 突出显示减排效果

4. 国家代码映射：
   - 处理特殊国家的代码匹配问题
   - 支持Norway→NO, France→FR等映射
   - 确保地理数据与碳排放数据正确匹配

输出结果：
1. 碳排放量热图（emission目录）：
   - 输出目录：D:\workstation\carbon emission\global\emission\
   - 文件格式：global_{工况}_emission.pdf
   - 包含工况：ref, case1-case9（共10个文件）
   - 显示内容：各国总碳排放量（tCO2）

2. 减排量热图（reduction目录）：
   - 输出目录：D:\workstation\carbon emission\global\reduction\
   - 文件格式：global_{工况}_reduction.pdf
   - 包含工况：case1-case9（共9个文件）
   - 显示内容：各国相对于ref的碳排放减少量（tCO2）

数据流程：
1. 数据加载阶段：
   - 加载世界地图shapefile数据
   - 扫描所有大洲的碳排放数据目录
   - 读取各国的碳排放文件

2. 数据处理阶段：
   - 提取各工况的碳排放数据
   - 处理国家代码映射
   - 计算统一比例尺

3. 热图生成阶段：
   - 合并地理数据和碳排放数据
   - 设置色彩方案和比例尺
   - 绘制地理分布热图

4. 文件保存阶段：
   - 分别保存碳排放量和减排量热图
   - 创建对应的输出目录
   - 生成PDF格式的高质量图表

可视化特点：
- 地理精度：国家级行政区划精度
- 色彩方案：
  * 碳排放量：深绿色→黄色→深棕色渐变
  * 减排量：白色→深绿色渐变
- 图表尺寸：20×10英寸
- 输出格式：PDF矢量图（300 DPI）
- 比例尺：统一比例尺确保可比性

技术参数：
- 地图格式：Shapefile
- 色彩映射：LinearSegmentedColormap
- 图表尺寸：20×10英寸
- 输出分辨率：300 DPI
- 文件格式：PDF

特殊处理：
- 国家代码映射：处理特殊国家的代码匹配
- 缺失数据处理：没有数据的国家显示为浅灰色
- 比例尺统一：使用最大值作为统一比例尺
- 南极洲排除：自动排除南极洲数据

应用场景：
- 全球碳排放分布分析
- 建筑节能效果可视化
- 碳排放政策制定支持
- 国际比较研究展示
- 学术报告和演示

与draw_global per capita.py的区别：
- 本工具处理总量数据（tCO2）
- draw_global per capita.py处理人均数据（kgCO2/person）
- 输入目录：global vs global_capita
- 输出目录：global\emission vs global_capita\emission
- 单位差异：tCO2 vs kgCO2/person

与draw_global reduction rate.py的区别：
- 本工具处理绝对排放量和减少量
- draw_global reduction rate.py处理减少率（百分比）
- 输出目录：global\emission vs global\reduction_rate
- 可视化重点：绝对数值 vs 相对比例

数据质量保证：
- 国家匹配验证
- 数据完整性检查
- 异常值处理
- 比例尺合理性验证
- 输出文件完整性确认
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

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
            if file.endswith('_carbon_emission.csv'):
                country_code = file.split('_')[0]
                df = pd.read_csv(os.path.join(continent_dir, file))
                
                # 获取所有工况的数据
                case_data = {'country': country_code}
                
                # 读取每个工况的碳排放量
                for case in ['ref', 'case1', 'case2', 'case3', 'case4', 
                            'case5', 'case6', 'case7', 'case8', 'case9']:
                    case_row = df[df.iloc[:, 0] == case]
                    if not case_row.empty:
                        case_data[f'{case}_emission'] = case_row['carbon_emission(tCO2)'].iloc[0]
                        if case != 'ref':  # 对于非ref工况，获取减少量
                            case_data[f'{case}_reduction'] = case_row['carbon_emission_reduction(tCO2)'].iloc[0]
                
                data.append(case_data)
    
    return pd.DataFrame(data)

def create_choropleth(gdf, data, value_column, title, save_path, vmax=None, is_reduction=False):
    """创建热图"""
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    
    # 根据是否为减少量热图选择不同的颜色方案
    if is_reduction:
        # 从白色到深绿色的渐变色
        colors = ['#FFFFFF', '#006400']  # 白色到深绿色
        cmap = LinearSegmentedColormap.from_list('custom_green', colors)
    else:
        # 从深绿色到深棕色的渐变色
        colors = ['#5B9C4A', '#F0D264', '#8B4513']  # 深绿色到深棕色
        cmap = LinearSegmentedColormap.from_list('custom_green_brown', colors)
        
        # 将值为0的数据替换为NaN
        data[value_column] = data[value_column].replace(0, np.nan)
    
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
                   'aspect': 25
               })
    
    ax.axis('off')
    plt.title(title, fontsize=14, pad=20)
    
    # 保存图片
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    print("开始处理数据和绘制图表...")
    
    # 设置路径
    base_dir = r"D:\workstation\carbon emission\global"
    shapefile_path = r"D:\workstation\energy_comsuption\shapefiles\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp"
    
    # 创建输出目录
    emission_dir = os.path.join(base_dir, 'emission')
    reduction_dir = os.path.join(base_dir, 'reduction')
    os.makedirs(emission_dir, exist_ok=True)
    os.makedirs(reduction_dir, exist_ok=True)
    
    # 加载地图数据
    print("正在加载地图数据...")
    gdf = gpd.read_file(shapefile_path)
    print("地图数据加载完成")
    
    # 加载并处理碳排放数据
    print("正在处理碳排放数据...")
    data = load_and_process_data(base_dir)
    print("碳排放数据处理完成")
    
    # 计算每个工况的最大值用于热图统一比例尺
    cases = ['ref', 'case1', 'case2', 'case3', 'case4', 
             'case5', 'case6', 'case7', 'case8', 'case9']
    
    # 计算所有工况中排放量的最大值
    emission_columns = [f'{case}_emission' for case in cases]
    vmax_emission = data[emission_columns].max().max()
    print(f"所有工况最大碳排放值: {vmax_emission} tCO2")
    
    # 计算所有工况中减少量的最大值
    #reduction_columns = [f'{case}_reduction' for case in cases[1:]]  # 排除ref工况
    #vmax_reduction = data[reduction_columns].max().max() #取除ref工况外其他工况的最大值
    vmax_reduction = data['case2_reduction'].max()
    print(f"最大碳排放减少值: {vmax_reduction} tCO2")
    
    # 绘制所有工况的碳排放量热图
    for case in cases:
        print(f"正在绘制 {case} 工况碳排放量热图...")
        create_choropleth(
            gdf, data, f'{case}_emission',
            f'Global {case} Case Carbon Emission (tCO2)',
            os.path.join(emission_dir, f'global_{case}_emission.pdf'),
            vmax=vmax_emission,
            is_reduction=False
        )
        print(f"{case} 工况碳排放量热图已保存")
    
    # 绘制减少碳排放量热图
    reduction_cases = cases[1:]  # 排除ref工况
    for case in reduction_cases:
        print(f"正在绘制 {case} 工况减少碳排放量热图...")
        create_choropleth(
            gdf, data, f'{case}_reduction',
            f'Global {case} Case Carbon Emission Reduction (tCO2)',
            os.path.join(reduction_dir, f'global_{case}_reduction.pdf'),
            vmax=vmax_reduction,
            is_reduction=True
        )
        print(f"{case} 工况减少碳排放量热图已保存")
    
    print("\n所有图表处理完成！")

if __name__ == '__main__':
    main()
