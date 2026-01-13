"""
全球建筑能耗图表绘制工具

功能概述：
本工具用于为全球各地区生成建筑能耗分析图表，包括案例对比图表和逐时能耗图表。该工具优先使用校准后的数据，当校准数据不可用时自动使用原始数据。

输入数据：
1. 各国能耗汇总文件：
   - 校准数据：results/Calibrated regions/{region}_2019_summary_p_results.csv
   - 原始数据：results/global/{continent}/summary_p/{region}_2019_summary_p_results.csv
   - 包含各国人均能耗数据（kWh/person）
   - 涵盖ref、case1-case9共10种案例

2. 逐时能耗数据：
   - 校准数据：results/Calibrated regions/{region}_2019_heating_demand.csv, {region}_2019_cooling_demand.csv
   - 原始数据：results/global/{continent}/{region}/{region}_2019_heating_demand.csv, {region}_2019_cooling_demand.csv
   - 包含ref和case2的逐时供暖、制冷需求数据

3. 地区范围：
   - 主要处理：Asia, Europe, Oceania
   - 跳过州/省级数据（包含点号的地区代码）

主要功能：
1. 数据优先级处理：
   - 优先使用Calibrated regions中的校准数据
   - 当校准数据不可用时，自动使用global文件夹中的原始数据
   - 自动跳过州/省级数据，只处理国家级数据

2. 案例对比图表生成：
   - case1-4图表：显示ref和case1-case4的能耗对比
   - case5-9图表：显示ref和case5-case9的能耗对比
   - 包含总能耗、供暖能耗、制冷能耗的柱状图和折线图

3. 逐时能耗图表生成：
   - 绘制参考案例和case2的逐时供暖、制冷需求对比
   - 时间跨度：2019年全年
   - 显示能耗需求的时间变化趋势

4. 批量处理：
   - 自动发现所有可用的地区
   - 为每个地区生成完整的图表集
   - 包含错误处理和日志记录

输出结果：
1. 案例对比图表（SVG格式）：
   - {region}_demands_per_capita.svg：case1-4对比图
   - {region}_demands_per_capita_p_ls.svg：case5-9对比图
   - 包含柱状图（能耗值）和折线图（节能百分比）

2. 逐时能耗图表（SVG格式）：
   - {region}_heating_cooling_demand_comparison.svg：逐时对比图
   - 显示参考案例和case2的逐时需求变化

3. 输出目录结构：
   - 校准数据：Calibrated regions/figures/{region}/
   - 原始数据：global/{continent}/{region}/

数据流程：
1. 扫描可用地区：检查Calibrated regions和global文件夹
2. 数据优先级选择：优先使用校准数据，备用原始数据
3. 图表生成：
   - 读取summary数据，生成案例对比图表
   - 读取逐时数据，生成时间序列图表
4. 错误处理：记录处理过程中的错误和异常
5. 资源清理：确保所有图形资源正确释放

计算特点：
- 数据优先级：校准数据 > 原始数据
- 地区筛选：只处理国家级数据，跳过州/省级数据
- 多图表类型：案例对比图 + 逐时分析图
- 错误容错：包含完善的异常处理机制
- 资源管理：自动清理图形资源，避免内存泄漏

图表特色：
- 双Y轴设计：左侧显示能耗值，右侧显示节能百分比
- 颜色编码：蓝色表示能耗，红色表示节能效果
- 数据标签：在图表上直接显示数值
- 时间序列：逐时数据的时间轴格式化
- 响应式布局：适应不同数据范围

处理逻辑：
1. 文件查找：优先在Calibrated regions中查找，再查找global
2. 数据验证：确保数据格式正确，包含必要的列
3. 图表生成：调用CandD.draw模块的绘图函数
4. 错误处理：捕获并记录所有异常
5. 资源清理：确保matplotlib图形正确关闭
"""

import os
import pandas as pd
import numpy as np
import sys
# 将项目的根目录加入到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CandD.draw import d_hourly_demand, d_case1_4, d_case5_9
import matplotlib.pyplot as plt

def find_summary_file(region, base_paths):
    """查找指定地区的summary文件
    优先在Calibrated regions文件夹中查找，如果找不到则在global文件夹中查找
    """
    # 跳过包含点号的region（州/省级数据）
    if '.' in region:
        return None
        
    # 首先在Calibrated regions中查找
    calibrated_file = os.path.join(base_paths['calibrated'], f"{region}_2019_summary_p_results.csv")
    if os.path.exists(calibrated_file):
        return calibrated_file
    
    # 如果在Calibrated regions中未找到，则在global文件夹中查找
    continents = ['Asia', 'Europe', 'Oceania'] #['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    for continent in continents:
        global_file = os.path.join(base_paths['global'], continent, 'summary_p', f"{region}_2019_summary_p_results.csv")
        if os.path.exists(global_file):
            return global_file
    
    return None

def collect_all_regions(base_paths):
    """收集所有可用的地区代码"""
    regions = set()
    
    # 检查Calibrated regions
    if os.path.exists(base_paths['calibrated']):
        files = [f for f in os.listdir(base_paths['calibrated']) 
                if f.endswith('_summary_p_results.csv')]
        regions.update([f.split('_')[0] for f in files if '.' not in f.split('_')[0]])
    
    # 检查global下的各个大洲
    continents = ['Asia', 'Europe', 'Oceania']
    for continent in continents:
        continent_path = os.path.join(base_paths['global'], continent, 'summary_p')
        if os.path.exists(continent_path):
            files = [f for f in os.listdir(continent_path) 
                    if f.endswith('_summary_p_results.csv')]
            regions.update([f.split('_')[0] for f in files if '.' not in f.split('_')[0]])
    
    return sorted(list(regions))

def load_hourly_data(region, base_paths):
    """加载逐时数据"""
    def find_continent_for_region(region):
        """查找region所在的大洲"""
        continents = ['Asia', 'Europe', 'Oceania'] #['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
        for continent in continents:
            if os.path.exists(os.path.join(base_paths['global'], continent, region)):
                return continent
        return None

    try:
        # 首先尝试从Calibrated regions加载
        heating_file = os.path.join(base_paths['calibrated'], f"{region}_2019_heating_demand.csv")
        cooling_file = os.path.join(base_paths['calibrated'], f"{region}_2019_cooling_demand.csv")
        
        if os.path.exists(heating_file) and os.path.exists(cooling_file):
            base_path = base_paths['calibrated']
        else:
            # 如果在Calibrated regions中未找到，则在global中查找
            continent = find_continent_for_region(region)
            if continent:
                base_path = os.path.join(base_paths['global'], continent, region)
            else:
                return None, None, None, None
        
        # 读取数据
        heating_df = pd.read_csv(os.path.join(base_path, f"{region}_2019_heating_demand.csv"))
        cooling_df = pd.read_csv(os.path.join(base_path, f"{region}_2019_cooling_demand.csv"))
        
        # 设置时间索引
        heating_df.set_index(heating_df.columns[0], inplace=True)
        cooling_df.set_index(cooling_df.columns[0], inplace=True)
        
        # 转换索引为时间格式
        heating_df.index = pd.to_datetime(heating_df.index)
        cooling_df.index = pd.to_datetime(cooling_df.index)
        
        return heating_df['ref'], cooling_df['ref'], heating_df['case2'], cooling_df['case2']
        
    except Exception as e:
        print(f"加载 {region} 的逐时数据时出错: {e}")
        return None, None, None, None

def get_output_path(region, base_paths):
    """获取输出路径"""
    # 首先检查region是否在Calibrated regions中有数据
    if os.path.exists(os.path.join(base_paths['calibrated'], f"{region}_2019_summary_p_results.csv")):
        return os.path.join(base_paths['calibrated'], 'figures', region)
    
    # 否则在global中查找对应的大洲
    continents = ['Asia', 'Europe', 'Oceania'] #['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    for continent in continents:
        region_path = os.path.join(base_paths['global'], continent, region)
        if os.path.exists(region_path):
            return region_path
    
    return None

def process_region_data(region, base_paths):
    """处理单个地区的数据并生成图表"""
    try:
        # 获取输出路径
        output_path = get_output_path(region, base_paths)
        if not output_path:
            print(f"无法确定 {region} 的输出路径")
            return
            
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 读取summary数据
        summary_file = find_summary_file(region, base_paths)
        if summary_file:
            # 读取数据并确保索引正确
            summary_data = pd.read_csv(summary_file)
            summary_data.set_index(summary_data.columns[0], inplace=True)
            
            # 确保数据是正确的格式
            summary_data = summary_data.astype(float)
            
            # 绘制case1-4图表
            try:
                d_case1_4(region, {region: summary_data}, output_path)
                print(f"Generated case1-4 figure for {region}")
            except Exception as e:
                print(f"Error generating case1-4 figure for {region}: {e}")
                print(f"Data shape: {summary_data.shape}")
                print(f"Data columns: {summary_data.columns}")
            finally:
                plt.close('all')
            
            # 绘制case5-9图表
            try:
                d_case5_9(region, {region: summary_data}, output_path)
                print(f"Generated case5-9 figure for {region}")
            except Exception as e:
                print(f"Error generating case5-9 figure for {region}: {e}")
                print(f"Data shape: {summary_data.shape}")
                print(f"Data columns: {summary_data.columns}")
            finally:
                plt.close('all')
        
        # 读取和绘制逐时数据
        ref_heating, ref_cooling, case2_heating, case2_cooling = load_hourly_data(region, base_paths)
        if all(x is not None for x in [ref_heating, ref_cooling, case2_heating, case2_cooling]):
            try:
                d_hourly_demand(
                    region,
                    ref_heating,
                    ref_cooling,
                    case2_heating,
                    case2_cooling,
                    output_path
                )
                print(f"Generated hourly demand figure for {region}")
            except Exception as e:
                print(f"Error generating hourly demand figure for {region}: {e}")
                print(f"Data shapes: {ref_heating.shape}, {ref_cooling.shape}, {case2_heating.shape}, {case2_cooling.shape}")
            finally:
                plt.close('all')
        else:
            print(f"Hourly data not found for {region}")
            
    except Exception as e:
        print(f"处理 {region} 时发生错误: {e}")
    finally:
        plt.close('all')

def draw_all_figures():
    """绘制所有图表"""
    # 设置基础路径
    base_paths = {
        'calibrated': r"D:\workstation\energy_comsuption\results\Calibrated regions",
        'global': r"D:\workstation\energy_comsuption\results\global"
    }
    
    # 获取所有地区
    regions = collect_all_regions(base_paths)
    
    # 处理每个地区的数据
    for region in regions:
        print(f"Processing {region}...")
        process_region_data(region, base_paths)
        plt.close('all')  # 确保所有图形都被关闭

def main():
    print("开始绘制图表...")
    draw_all_figures()
    print("图表绘制完成！")

if __name__ == "__main__":
    main()
