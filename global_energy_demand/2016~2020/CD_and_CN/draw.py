"""
建筑能耗分析图表绘制工具

功能概述：
本工具用于生成建筑能耗分析的各种可视化图表，包括BAIT阈值分析、不同案例的能耗对比、以及逐时能耗需求分析。

输入数据：
1. BAIT数据：建筑适应性室内温度数据，包含各地区的时间序列温度数据
2. 能耗数据：各地区不同案例（ref, case1-case9）的能耗统计结果
   - 包含总能耗、供暖能耗、制冷能耗的人均数据
   - 包含对应的节能百分比数据
3. 阈值数据：各地区的供暖和制冷温度阈值
4. 逐时能耗数据：参考案例和case2的逐时供暖、制冷需求数据

主要功能：
1. BAIT阈值分析（d_BAIT）：
   - 绘制建筑适应性室内温度（BAIT）的时间序列曲线
   - 显示供暖和制冷的多个阈值线（不同线型区分）
   - 供暖阈值：基础阈值向下递减5个等级（红色）
   - 制冷阈值：基础阈值向上递增5个等级（蓝色）

2. 案例1-4能耗对比（d_case1_4）：
   - 生成三个子图：制冷、供暖、总能耗需求
   - 柱状图显示各案例的人均能耗（蓝色）
   - 折线图显示节能百分比（红色）
   - 包含ref和case1-case4的对比分析

3. 案例5-9能耗对比（d_case5_9）：
   - 类似案例1-4，但针对case5-case9
   - X轴标签显示p_ls百分比值（1.0%, 0.5%, 0.25%, 0.125%, 0.0625%, 0.03125%）
   - 分析不同渗透率对能耗的影响

4. 逐时能耗对比（d_hourly_demand）：
   - 绘制参考案例和case2的逐时供暖、制冷需求对比
   - 时间跨度：2019年全年
   - 显示能耗需求的时间变化趋势

输出结果：
1. PNG格式的BAIT阈值图：
   - {region}_BAIT_thresholds.png

2. SVG格式的能耗对比图：
   - {region}_demands_per_capita.svg（案例1-4）
   - {region}_demands_per_capita_p_ls.svg（案例5-9）
   - {region}_heating_cooling_demand_comparison.svg（逐时对比）

图表特点：
- 双Y轴设计：左侧显示能耗值，右侧显示节能百分比
- 颜色编码：蓝色表示能耗，红色表示节能效果
- 数据标签：在图表上直接显示数值
- 网格线：便于数据读取
- 响应式布局：适应不同数据范围

使用场景：
- 建筑节能技术效果评估
- 不同气候区域能耗特性分析
- 渗透率对建筑能耗的影响研究
- 建筑适应性温度阈值分析
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.dates as mdates

def d_BAIT(data, region, thresholds, save_path):
    """绘制BAIT和阈值线
    Args:
        data: BAIT数据，DataFrame格式
        region: 地区代码
        thresholds: 阈值数据字典
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制BAIT数据
    plt.plot(data.index, data[region], 'k-', label='BAIT', linewidth=1.5)
    
    # 获取基础阈值
    if region.startswith('US.'):
        base_heating = 9.7
        base_cooling = 18.8
    else:
        base_heating = thresholds[region]['heating']
        base_cooling = thresholds[region]['cooling']
    
    # 生成阈值线组
    heating_thresholds = [base_heating - i for i in range(5)]
    cooling_thresholds = [base_cooling + i for i in range(5)]
    
    # 定义线型
    linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]
    
    # 绘制供暖阈值线（红色）
    for i, (thresh, style) in enumerate(zip(heating_thresholds, linestyles)):
        plt.axhline(y=thresh, color='red', linestyle=style, alpha=0.6, 
                   label=f'Heating threshold {i+1}' if i == 0 else None)
    
    # 绘制制冷阈值线（蓝色）
    for i, (thresh, style) in enumerate(zip(cooling_thresholds, linestyles)):
        plt.axhline(y=thresh, color='blue', linestyle=style, alpha=0.6,
                   label=f'Cooling threshold {i+1}' if i == 0 else None)
    
    # 设置x轴格式
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y/%m/%d'))
    
    # 设置特定的x轴刻度点
    specific_dates = pd.to_datetime(['2019-01-01', '2019-04-01', '2019-07-01', '2019-10-01', '2020-01-01'])
    plt.xticks(specific_dates)
    plt.xlim(pd.Timestamp('2019-01-01'), pd.Timestamp('2020-01-01'))
    
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.title(f'BAIT and Threshold Lines - {region}')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_file = os.path.join(save_path, f"{region}_BAIT_thresholds.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def d_case1_4(region_name, data, output_dir):
    """绘制case1-4的需求图表"""
    case_labels = ['ref'] + [f"case{i}" for i in range(1, 5)]
    data = data[region_name]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 18))
    fig.suptitle(f'{region_name}', fontsize=16, y=0.95)
    plt.rcParams.update({'font.size': 16})
    
    # 设置颜色
    colors = ['lavender'] + ['skyblue'] * 4
    x = np.arange(len(case_labels))
    
    # 绘制三个子图
    for ax_idx, (ax, demand_type, reduction_type) in enumerate([
        (ax1, "cooling_demand_sum_p(kWh/person)", "cooling_demand_p_reduction(%)"),
        (ax2, "heating_demand_sum_p(kWh/person)", "heating_demand_p_reduction(%)"),
        (ax3, "total_demand_sum_p(kWh/person)", "total_demand_p_reduction(%)")
    ]):
        # 提取需求数据
        demand_sum = data[demand_type].loc[case_labels].values
        reduction = data[reduction_type].loc[case_labels[1:]].values
        
        # 绘制柱状图
        bars = ax.bar(x, demand_sum, color=colors)
        ax.set_ylabel(demand_type.split('_')[0].title() + " Demand (kWh/person)", color="blue", fontsize=16)
        ax.tick_params(axis="y", labelcolor="blue", labelsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(case_labels)
        
        # 绘制折线图
        ax_twin = ax.twinx()
        ax_twin.plot(x[1:], reduction, color="red", marker="o")
        ax_twin.set_ylabel("Reduction (%)", color="red", fontsize=16)
        ax_twin.tick_params(axis="y", labelcolor="red", labelsize=14)
        ax_twin.set_ylim(0, 100)
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', color="blue", fontsize=14)
        for i, value in enumerate(reduction):
            ax_twin.text(i + 1, value + 2,
                         f'{value:.1f}%', ha='center', va='bottom', color="red", fontsize=14)
    
    ax3.set_xlabel("Case", fontsize=16)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{region_name}_demands_per_capita.svg")
    plt.savefig(output_path, format="svg", bbox_inches='tight')
    plt.close()

def d_case5_9(region_name, data, output_dir):
    """绘制case5-9的需求图表"""
    case_labels = ['ref'] + [f"case{i}" for i in range(5, 10)]
    data = data[region_name]
    
    p_ls_values = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    p_ls_labels = [f"{p*100:.3g}%" for p in p_ls_values]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 18))
    fig.suptitle(f'{region_name}', fontsize=16, y=0.95)
    plt.rcParams.update({'font.size': 16})
    
    colors = ['lavender'] + ['skyblue'] * 5
    x = np.arange(len(case_labels))
    
    # 绘制三个子图
    for ax_idx, (ax, demand_type, reduction_type) in enumerate([
        (ax1, "cooling_demand_sum_p(kWh/person)", "cooling_demand_p_reduction(%)"),
        (ax2, "heating_demand_sum_p(kWh/person)", "heating_demand_p_reduction(%)"),
        (ax3, "total_demand_sum_p(kWh/person)", "total_demand_p_reduction(%)")
    ]):
        # 提取需求数据
        demand_sum = data[demand_type].loc[case_labels].values
        reduction = data[reduction_type].loc[case_labels[1:]].values
        
        # 绘制柱状图
        bars = ax.bar(x, demand_sum, color=colors)
        ax.set_ylabel(demand_type.split('_')[0].title() + " Demand (kWh/person)", color="blue", fontsize=16)
        ax.tick_params(axis="y", labelcolor="blue", labelsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(p_ls_labels, rotation=45)
        
        # 绘制折线图
        ax_twin = ax.twinx()
        ax_twin.plot(x[1:], reduction, color="red", marker="o")
        ax_twin.set_ylabel("Reduction (%)", color="red", fontsize=16)
        ax_twin.tick_params(axis="y", labelcolor="red", labelsize=14)
        ax_twin.set_ylim(0, 100)
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', color="blue", fontsize=14)
        for i, value in enumerate(reduction):
            ax_twin.text(i + 1, value + 2,
                         f'{value:.1f}%', ha='center', va='bottom', color="red", fontsize=14)
    
    ax3.set_xlabel("p_ls (%)", fontsize=16)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{region_name}_demands_per_capita_p_ls.svg")
    plt.savefig(output_path, format="svg", bbox_inches='tight')
    plt.close()

def d_hourly_demand(region_code, ref_heating, ref_cooling, case2_heating, case2_cooling, save_dir):
    """绘制逐时供暖和制冷需求对比图"""
    plt.figure(figsize=(14, 6))
    
    # 转换数据为numpy数组
    ref_heating_values = ref_heating.values
    ref_cooling_values = ref_cooling.values
    case2_heating_values = case2_heating.values
    case2_cooling_values = case2_cooling.values
    dates = mdates.date2num(ref_heating.index)
    
    plt.plot_date(dates, ref_heating_values, '-', color='red', 
                 label='Reference Heating', linewidth=2)
    plt.plot_date(dates, ref_cooling_values, '-', color='blue', 
                 label='Reference Cooling', linewidth=2)
    plt.plot_date(dates, case2_heating_values, '-', color='lightcoral', 
                 label='Case 2 Heating', linewidth=2)
    plt.plot_date(dates, case2_cooling_values, '-', color='lightskyblue', 
                 label='Case 2 Cooling', linewidth=2)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    
    plt.xticks(fontsize=12)
    
    start_date = pd.Timestamp('2019-01-01')
    end_date = pd.Timestamp('2020-01-01')
    ax.set_xlim([mdates.date2num(start_date), mdates.date2num(end_date)])
    
    y_max = max(ref_heating_values.max(), ref_cooling_values.max(), 
                case2_heating_values.max(), case2_cooling_values.max())
    ax.set_ylim([0, y_max * 1.1])
    plt.yticks(fontsize=12)
    
    plt.title(f'Hourly Heating and Cooling Demand Comparison - {region_code}', fontsize=14)
    plt.ylabel('Energy Demand (GW)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper center', ncol=4, framealpha=0.8, fontsize=12)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, f'{region_code}_heating_cooling_demand_comparison.svg'), 
                format="svg", bbox_inches='tight')
    plt.close()
