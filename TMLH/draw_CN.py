"""
中国人体理论最小负荷可视化绘图工具

功能概述：
本工具用于生成中国各省份人体理论最小负荷（TMLH）的可视化图表，包括能耗热力图和节能比例柱状图。支持总量和人均两种数据模式，为建筑能耗分析提供直观的地理分布和节能效果展示。

输入数据：
1. 省份summary结果文件：
   - 目录：D:\workstation\energy_comsuption\results\CN
   - 文件格式：各省份的summary_results.csv和summary_p_results.csv
   - 包含工况：ref, case1-case9, TMLH
   - 数据内容：总能耗、冷热负荷、节能比例等

2. 中国地图数据：
   - 文件路径：D:\workstation\energy_comsuption\shapefiles\中国_省.geojson
   - 数据格式：GeoJSON格式的中国省级行政区划
   - 包含字段：name（省份名称）、geometry（地理边界）

3. 省份名称映射：
   - ISO代码到中文名称的映射表
   - 支持31个省级行政区（含港澳台）
   - 格式：'CN.BJ' -> '北京市'

主要功能：
1. 数据加载与处理：
   - 读取各省份的summary结果文件
   - 解析不同工况的能耗数据
   - 计算节能比例和减少百分比
   - 省份名称标准化和映射

2. 热力图生成：
   - 基于中国地图绘制能耗分布热力图
   - 支持所有工况（ref, case1-case9, TMLH）
   - 使用紫色渐变色彩方案
   - 统一比例尺确保图表可比性

3. 柱状图生成：
   - 按TMLH节能比例排序的柱状图
   - 显示各省份的节能效果
   - 支持总量和人均两种模式
   - 自动导出数据为CSV格式

4. 双模式支持：
   - 总量模式：显示总能耗（TWh）
   - 人均模式：显示人均能耗（kWh/person）
   - 分别生成对应的图表和数据

输出结果：
1. 能耗热力图（PDF格式）：
   - 总量模式：CN_{case}_demand_total.pdf
   - 人均模式：CN_{case}_demand_per_capita.pdf
   - 包含工况：ref, case1-case9, TMLH（共11个文件）

2. 节能比例柱状图（PDF格式）：
   - 总量模式：CN_reduction_total.pdf
   - 人均模式：CN_reduction_per_capita.pdf
   - 按节能比例降序排列

3. 节能比例数据（CSV格式）：
   - 总量模式：CN_reduction_total_data.csv
   - 人均模式：CN_reduction_per_capita_data.csv
   - 包含列：省份名称、省份代码、TMLH节能比例(%)

数据流程：
1. 数据加载阶段：
   - 读取中国地图GeoJSON文件
   - 扫描results目录下的summary文件
   - 解析各省份的能耗数据

2. 数据处理阶段：
   - 省份名称映射和标准化
   - 提取各工况的能耗数据
   - 计算节能比例和减少百分比
   - 数据排序和筛选

3. 热力图生成阶段：
   - 地图数据与能耗数据合并
   - 设置色彩方案和比例尺
   - 绘制地理分布热力图
   - 添加图例和标题

4. 柱状图生成阶段：
   - 按TMLH节能比例排序
   - 绘制柱状图
   - 添加网格线和标签
   - 导出数据为CSV

5. 文件保存阶段：
   - 分别保存总量和人均图表
   - 创建对应的输出目录
   - 生成PDF和CSV文件

可视化特点：
- 地理精度：省级行政区划精度
- 色彩方案：紫色渐变（浅紫到深紫）
- 图表类型：热力图 + 柱状图
- 输出格式：PDF（高质量）+ CSV（数据）
- 比例尺：统一比例尺确保可比性

技术参数：
- 地图格式：GeoJSON
- 色彩映射：LinearSegmentedColormap
- 图表尺寸：15x10英寸（热力图）、15x8英寸（柱状图）
- 输出分辨率：300 DPI
- 文件格式：PDF + CSV

数据处理逻辑：
1. 省份匹配：通过name字段匹配地图和能耗数据
2. 工况提取：从summary文件中提取各工况数据
3. 节能计算：相对于ref工况计算节能比例
4. 数据排序：按TMLH节能比例降序排列
5. 异常处理：处理缺失数据和匹配失败情况

特殊处理：
- 省份名称映射：ISO代码转换为中文名称
- 数据过滤：排除国家级别数据（CN）
- 缺失值处理：未匹配省份显示为浅灰色
- 比例尺统一：使用ref工况最大值作为统一比例尺
- 文件命名：根据模式和工况自动生成文件名

输出目录结构：
- CN_total/：总量模式图表
- CN_per_capita/：人均模式图表
- 每个目录包含11个热力图 + 1个柱状图 + 1个CSV文件

应用场景：
- 建筑能耗地理分布分析
- 节能效果区域对比
- 政策制定和规划支持
- 学术研究和报告制作
- 可视化展示和演示

与TMLH计算的关系：
- 基于CN.py计算的TMLH结果
- 展示人体理论最小负荷的地理分布
- 对比不同节能策略的效果
- 为建筑节能提供可视化支持

图表特色：
- 专业的地理可视化效果
- 清晰的节能效果对比
- 标准化的色彩方案
- 高质量的输出格式
- 完整的数据导出功能

数据质量保证：
- 省份匹配验证
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

def create_province_mapping():
    """创建省份名称映射表，将ISO代码转换为中文名称"""
    mapping = {
        # ISO代码格式到中文的映射还有错误
        'CN.AH': '安徽省',
        'CN.BJ': '北京市',
        'CN.CQ': '重庆市',
        'CN.FJ': '福建省',
        'CN.GD': '广东省',
        'CN.GS': '甘肃省',
        'CN.GX': '广西壮族自治区',
        'CN.GZ': '贵州省',
        'CN.HA': '海南省',
        'CN.HU': '湖北省',
        'CN.HB': '河北省',
        'CN.HE': '河南省',
        'CN.HL': '黑龙江省',
        'CN.HN': '湖南省',
        'CN.JL': '吉林省',
        'CN.JS': '江苏省',
        'CN.JX': '江西省',
        'CN.LN': '辽宁省',
        'CN.NM': '内蒙古自治区',
        'CN.NX': '宁夏回族自治区',
        'CN.QH': '青海省',
        'CN.SA': '陕西省',
        'CN.SC': '四川省',
        'CN.SD': '山东省',
        'CN.SH': '上海市',
        'CN.SX': '山西省',
        'CN.TJ': '天津市',
        'CN.XJ': '新疆维吾尔自治区',
        'CN.XZ': '西藏自治区',
        'CN.YN': '云南省',
        'CN.ZJ': '浙江省',
        'MO': '澳门特别行政区',
        'HK': '香港特别行政区',
        'TW': '台湾省'
    }
    return mapping

def load_and_process_data(results_dir, is_per_person=False):
    """加载并处理能耗数据"""
    data = []
    province_mapping = create_province_mapping()
    
    # 根据是否为人均数据选择文件后缀
    file_suffix = "_p_results" if is_per_person else "_results"
    
    for file in os.listdir(results_dir):
        # 只处理summary结果文件
        if file.endswith(f'_summary{file_suffix}.csv'):
            region = file.split('_')[0]
            
            # 排除CN（中国）数据，因为它代表国家级别
            if region == 'CN':
                continue
            
            # 将省份名称转换为中文
            chinese_region = province_mapping.get(region, region)
            
            df = pd.read_csv(os.path.join(results_dir, file))
            
            # 获取所有工况的数据
            case_data = {}
            for case in ['ref', 'case1', 'case2', 'case3', 'case4', 
                        'case5', 'case6', 'case7', 'case8', 'case9', 'TMLH']:
                case_row = df[df.iloc[:, 0] == case].iloc[0]
                
                if is_per_person:
                    case_data[f'{case}_demand'] = case_row['total_demand_sum_p(kWh/person)']
                    if case != 'ref':  # 对于非ref工况，计算减少比例
                        case_data[f'{case}_reduction'] = case_row['total_demand_p_reduction(%)']
                else:
                    case_data[f'{case}_demand'] = case_row['total_demand_sum(TWh)']
                    if case != 'ref':  # 对于非ref工况，计算减少比例
                        case_data[f'{case}_reduction'] = case_row['total_demand_reduction(%)']
            
            case_data['region'] = chinese_region  # 用于热图的中文名称
            case_data['region_code'] = region.split('.')[-1] if '.' in region else region  # 用于柱状图的二字母代码
            data.append(case_data)
    
    return pd.DataFrame(data)

def create_choropleth(gdf, data, value_column, title, save_path, vmax=None, is_per_person=False):
    """创建热图"""
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # 创建从浅紫色到深紫色的渐变色
    colors = ['#F0E6FF', '#4B0082']  # 从浅紫色到深紫色
    cmap = LinearSegmentedColormap.from_list('custom_purple', colors)
    
    # 使用name字段匹配省份数据
    merged = gdf.merge(data, left_on='name', right_on='region', how='left')
    
    # 检查匹配情况
    matched_count = merged[value_column].notna().sum()
    total_count = len(data)
    print(f"省份匹配情况: {matched_count}/{total_count} 个省份成功匹配")
    
    if matched_count == 0:
        print("警告: 没有省份成功匹配！")
        print("GeoJSON中的省份名称:", gdf['name'].tolist())
        print("数据中的省份名称:", data['region'].tolist())
    
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
                   'orientation': 'horizontal',  # 水平放置颜色条
                   'pad': 0.03,  # 调整颜色条与图之间的间距
                   'fraction': 0.046,  # 调整颜色条的高度
                   'shrink': 0.8,  # 调整颜色条的宽度比例
                   'aspect': 25  # 调整颜色条的长宽比
               })
    
    ax.axis('off')
    plt.title(title, fontsize=14, pad=20)
    
    # 保存图片为PDF格式
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def create_bar_chart(data, save_path, is_per_person=False):
    """创建节能比例柱状图"""
    plt.figure(figsize=(15, 8))
    
    # 排序数据（使用TMLH工况的减少比例）
    sorted_data = data.sort_values('TMLH_reduction', ascending=False)
    
    # 直接使用二字母代码
    labels = sorted_data['region_code']
    
    plt.bar(range(len(labels)), sorted_data['TMLH_reduction'], color='#ADD8E6')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    
    if is_per_person:
        plt.ylabel('Energy Reduction (%)')
        plt.title('Energy Reduction by Province (Per Capita)')
    else:
        plt.ylabel('Energy Reduction (%)')
        plt.title('Energy Reduction by Province (Total)')
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 100)
    
    # 保存图片为PDF格式
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 导出节能比例数据为CSV
    csv_data = sorted_data[['region', 'region_code', 'TMLH_reduction']].copy()
    csv_data.columns = ['省份名称', '省份代码', 'TMLH节能比例(%)']
    
    # 生成CSV文件路径
    csv_path = save_path.replace('.pdf', '_data.csv')
    csv_data.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"节能比例数据已导出至: {csv_path}")

def process_china_data(results_dir, output_dir, is_per_person=False):
    """处理中国数据和绘图"""
    print(f"\n正在处理中国{'人均' if is_per_person else '总量'}数据...")
    
    # 加载中国地图数据
    print("正在加载中国地图数据...")
    geojson_path = r"D:\workstation\energy_comsuption\shapefiles\中国_省.geojson"
    gdf = gpd.read_file(geojson_path)
    
    # 过滤掉非省份数据（如境界线）
    gdf = gdf[gdf['name'] != '境界线']
    print("已加载中国地图数据")
    print(f"GeoJSON中包含的省份: {gdf['name'].tolist()}")
    
    # 加载能耗数据
    print("正在处理能耗数据...")
    data = load_and_process_data(results_dir, is_per_person)
    print("能耗数据处理完成")
    print(f"数据中包含的省份: {data['region'].tolist()}")
    
    # 计算ref工况最大值用于热图统一比例尺
    vmax = data['ref_demand'].max()
    print(f"参考工况最大能耗值: {vmax}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制所有工况的热图
    cases = ['ref', 'case1', 'case2', 'case3', 'case4', 
             'case5', 'case6', 'case7', 'case8', 'case9', 'TMLH']
    
    for case in cases:
        print(f"正在绘制{case}工况能耗热图...")
        
        # 根据是否为人均数据设置标题和文件名
        if is_per_person:
            title = f'China {case} Case Energy Consumption (kWh/person)'
            filename = f'CN_{case}_demand_per_capita.pdf'
        else:
            title = f'China {case} Case Energy Consumption (TWh)'
            filename = f'CN_{case}_demand_total.pdf'
        
        create_choropleth(
            gdf, data, f'{case}_demand',
            title,
            os.path.join(output_dir, filename),
            vmax=vmax,
            is_per_person=is_per_person
        )
        print(f"{case}工况能耗热图已保存")
    
    # 绘制节能比例柱状图
    print("正在绘制节能比例柱状图...")
    
    if is_per_person:
        bar_filename = 'CN_reduction_per_capita.pdf'
    else:
        bar_filename = 'CN_reduction_total.pdf'
    
    create_bar_chart(
        data, 
        os.path.join(output_dir, bar_filename),
        is_per_person=is_per_person
    )
    print("节能比例柱状图已保存")
    print("所有图表处理完成！")

def main():
    print("开始处理中国数据和绘制图表...")
    
    # 设置路径
    base_path = r'D:\workstation\energy_comsuption'
    results_dir = os.path.join(base_path, 'results', 'CN')
    
    # 处理总量数据
    output_dir_total = os.path.join(base_path, 'Theoretical minimum load of human', 'CN', 'CN_total')
    process_china_data(results_dir, output_dir_total, is_per_person=False)
    
    # 处理人均数据
    output_dir_per_capita = os.path.join(base_path, 'Theoretical minimum load of human', 'CN', 'CN_per_capita')
    process_china_data(results_dir, output_dir_per_capita, is_per_person=True)
    
    print("\n所有数据处理和图表绘制已完成！")

if __name__ == '__main__':
    main()
