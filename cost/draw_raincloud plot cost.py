"""
云雨图（Raincloud Plot）绘制工具

功能概述：
本工具用于绘制total_cost的云雨图，按收入组分组显示数据分布。

输入数据：
- 文件：Z:\local_environment_creation\cost\method2\country_cost_proportion_to_GNI.csv
- 数据列：IncomeGroup, Country Name, Country Code_2, total_cost, GNI_per, propotion_to_GNI, PP

输出结果：
- PDF图片：Z:\local_environment_creation\cost\method2\raincloud_plot_total_cost.pdf
- CSV统计文件：Z:\local_environment_creation\cost\method2\a_total_cost_statistics.csv
  （包含每个收入组的最大值、最小值、中位数、平均值、标准差、四分位数、IQR、偏度、峰度等统计信息）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os
from matplotlib.patches import Rectangle
from scipy import stats

# 配置文件路径
INPUT_FILE = r"Z:\local_environment_creation\cost\method2\country_cost_proportion_to_GNI.csv"
OUTPUT_DIR = r"Z:\local_environment_creation\cost\method2"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raincloud_plot_total_cost.pdf")

def load_data():
    """加载数据"""
    print(f"读取数据文件: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"数据文件不存在: {INPUT_FILE}")
    
    df = pd.read_csv(INPUT_FILE)
    print(f"成功加载数据，共 {len(df)} 行")
    
    # 检查必要的列是否存在
    if 'IncomeGroup' not in df.columns:
        raise ValueError("CSV文件中缺少'IncomeGroup'列")
    if 'total_cost' not in df.columns:
        raise ValueError("CSV文件中缺少'total_cost'列")
    
    # 去除total_cost或IncomeGroup为空值的行
    original_count = len(df)
    df = df.dropna(subset=['total_cost', 'IncomeGroup'])
    filtered_count = len(df)
    print(f"过滤空值后，剩余 {filtered_count} 行（过滤了 {original_count - filtered_count} 行）")
    
    return df

def save_statistics_to_csv(df):
    """计算并保存每个收入组的统计信息到CSV文件"""
    print("计算统计信息并保存到CSV...")
    
    # 获取所有收入组（按指定顺序排列）
    available_groups = [g for g in df['IncomeGroup'].unique() if pd.notna(g)]
    income_order = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
    income_groups = [g for g in income_order if g in available_groups]
    
    # 计算每个收入组的详细统计信息
    stats_list = []
    for income_group in income_groups:
        data = df[df['IncomeGroup'] == income_group]['total_cost'].values
        data = data[~np.isnan(data)]  # 去除NaN值
        
        if len(data) == 0:
            continue
        
        # 计算各种统计量
        stats_dict = {
            'IncomeGroup': income_group,
            'Count': len(data),
            'Mean': np.mean(data),
            'Median': np.median(data),
            'Min': np.min(data),
            'Max': np.max(data),
            'Std': np.std(data),
            'Q1': np.percentile(data, 25),
            'Q3': np.percentile(data, 75),
            'IQR': np.percentile(data, 75) - np.percentile(data, 25),
            'Skewness': stats.skew(data) if len(data) > 2 else np.nan,
            'Kurtosis': stats.kurtosis(data) if len(data) > 2 else np.nan
        }
        stats_list.append(stats_dict)
    
    # 创建DataFrame
    stats_df = pd.DataFrame(stats_list)
    
    # 重新排列列的顺序，使其更易读
    column_order = ['IncomeGroup', 'Count', 'Mean', 'Median', 'Min', 'Max', 
                    'Std', 'Q1', 'Q3', 'IQR', 'Skewness', 'Kurtosis']
    stats_df = stats_df[column_order]
    
    # 保存到CSV文件
    output_csv_path = os.path.join(OUTPUT_DIR, "a_total_cost_statistics.csv")
    stats_df.to_csv(output_csv_path, index=False, float_format='%.4f')
    print(f"统计信息已保存至: {output_csv_path}")
    
    return stats_df

def plot_raincloud(df):
    """绘制云雨图：左边单边小提琴图，中间箱线图，右边散点图"""
    print("开始绘制云雨图...")
    
    # 保存统计信息到CSV
    save_statistics_to_csv(df)
    
    # 获取所有收入组（按指定顺序排列）
    available_groups = [g for g in df['IncomeGroup'].unique() if pd.notna(g)]
    income_order = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
    income_groups = [g for g in income_order if g in available_groups]
    n_groups = len(income_groups)
    
    # 设置颜色方案
    colors = ['#3182BD', '#DE2D26', '#31A354', '#756BB1', '#FEB24C', '#6BAED6', '#9E9AC8', '#FD8D3C']
    # 如果收入组数量超过颜色数量，重复使用
    if n_groups > len(colors):
        colors = colors * ((n_groups // len(colors)) + 1)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 设置背景和边框
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')
    
    # 为每个收入组绘制云雨图
    positions = []
    for i, income_group in enumerate(income_groups):
        data = df[df['IncomeGroup'] == income_group]['total_cost'].values
        data = data[~np.isnan(data)]  # 去除NaN值
        
        if len(data) == 0:
            continue
        
        # 计算位置（横轴位置）
        pos = i
        positions.append(pos)
        
        # 选择颜色
        color = colors[i % len(colors)]
        
        # 定义组件宽度和间隙（缩小间距）
        violin_width = 0.25  # 小提琴图宽度
        gap1 = 0.08  # 小提琴图和箱线图之间的间隙
        gap2 = 0.08  # 箱线图和散点图之间的间隙
        box_width = 0.08  # 箱线图宽度
        scatter_start_offset = 0.03  # 散点图起始偏移
        jitter_amount = 0.10  # 散点图抖动范围
        
        # 计算各个组件的位置
        violin_right_edge = pos - gap1  # 小提琴图右边缘
        box_position = pos  # 箱线图中心位置
        scatter_left_edge = pos + gap2  # 散点图左边缘
        
        # 1. 绘制左边单边小提琴图（half violin plot）
        if len(data) > 1:
            # 计算KDE
            kde = stats.gaussian_kde(data)
            y_kde = np.linspace(data.min(), data.max(), 200)
            density = kde(y_kde)
            
            # 归一化密度到合适的宽度（只绘制左侧）
            density_scaled = density / density.max() * violin_width
            
            # 绘制左侧单边小提琴图（向右延伸到gap1位置）
            violin_left_edge = violin_right_edge - density_scaled
            ax.fill_betweenx(y_kde, violin_left_edge, violin_right_edge, 
                           alpha=0.6, color=color, edgecolor='black', linewidth=0.5)
        
        # 2. 绘制中间箱线图
        bp = ax.boxplot([data], positions=[box_position], widths=box_width, 
                        patch_artist=True, showfliers=True,
                        medianprops=dict(color='black', linewidth=1.5),
                        boxprops=dict(linewidth=1.2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2))
        
        # 设置箱线图颜色（使用对应组的颜色）
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.9)
            patch.set_edgecolor('black')  # 黑色轮廓
        
        # 3. 绘制右边散点图（带抖动）
        # 如果数据点太多，可以抽样显示
        if len(data) > 150:
            # 随机抽样显示
            sample_indices = np.random.choice(len(data), 150, replace=False)
            data_to_plot = data[sample_indices]
        else:
            data_to_plot = data
        
        # 添加抖动（从scatter_left_edge开始向右）
        jitter = np.random.uniform(0, jitter_amount, len(data_to_plot))
        x_positions = scatter_left_edge + scatter_start_offset + jitter
        
        # 绘制散点图
        ax.scatter(x_positions, data_to_plot, color=color, alpha=0.5, s=15, 
                  edgecolors='black', linewidths=0.3, zorder=10)
    
    # 设置轴标签和标题
    ax.set_xticks(positions)
    ax.set_xticklabels(income_groups, rotation=0, ha='center')
    ax.set_ylabel('Total Cost (USD)', fontsize=16, labelpad=15)
    ax.set_xlabel('Income Group', fontsize=16, labelpad=15)
    # ax.set_title('Raincloud Plot of Total Cost by Income Group', 
    #              fontsize=14, pad=20)
    
    # 设置刻度标签字体大小
    ax.tick_params(axis='x', rotation=0, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_FILE, format='pdf', bbox_inches='tight', dpi=300)
    print(f"云雨图已保存至: {OUTPUT_FILE}")
    
    plt.close()

def plot_raincloud_simple(df):
    """绘制简化的云雨图（使用seaborn等库的简化版本）"""
    print("开始绘制云雨图（简化版）...")
    
    # 获取所有大洲
    continents = sorted(df['continent'].unique())
    
    # 设置颜色方案
    colors = ['#3182BD', '#DE2D26', '#31A354', '#756BB1', '#FEB24C', '#6BAED6', '#9E9AC8', '#FD8D3C']
    palette = {continent: colors[i % len(colors)] for i, continent in enumerate(continents)}
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制violin plot（密度云）
    parts = ax.violinplot([df[df['continent'] == c]['total_reduction'].dropna().values 
                          for c in continents],
                         positions=range(len(continents)),
                         widths=0.5, showmeans=False, showmedians=True)
    
    # 设置violin plot的颜色
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)
    
    # 绘制箱线图
    box_data = [df[df['continent'] == c]['total_reduction'].dropna().values 
                for c in continents]
    bp = ax.boxplot(box_data, positions=range(len(continents)), widths=0.15,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.5))
    
    # 设置箱线图颜色
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.8)
    
    # 绘制雨滴（散点图带抖动）
    for i, continent in enumerate(continents):
        data = df[df['continent'] == continent]['total_reduction'].dropna().values
        
        # 如果数据点太多，抽样显示
        if len(data) > 150:
            data = np.random.choice(data, 150, replace=False)
        
        # 添加抖动
        jitter = np.random.normal(i, 0.08, len(data))
        ax.scatter(jitter, data, color=colors[i % len(colors)], alpha=0.5, s=15, zorder=10)
    
    # 设置轴标签和标题
    ax.set_xticks(range(len(continents)))
    ax.set_xticklabels(continents, rotation=0, ha='right')
    ax.set_ylabel('Energy Reduction Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Continent', fontsize=12, fontweight='bold')
    ax.set_title('Raincloud Plot of Energy Reduction Rate by Continent', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_FILE, format='pdf', bbox_inches='tight', dpi=300)
    print(f"云雨图已保存至: {OUTPUT_FILE}")
    
    plt.close()

def print_statistics(df):
    """打印统计信息"""
    print("\n=== 统计信息 ===")
    print(f"总国家数: {len(df)}")
    print(f"收入组数: {df['IncomeGroup'].nunique()}")
    print("\n按收入组统计:")
    
    stats_df = df.groupby('IncomeGroup')['total_cost'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('std', 'std')
    ])
    
    print(stats_df.to_string())

def main():
    """主函数"""
    print("开始绘制云雨图...")
    print("=" * 60)
    
    try:
        # 1. 加载数据
        df = load_data()
        
        # 2. 打印统计信息
        print_statistics(df)
        
        # 3. 绘制云雨图（左边单边小提琴图，中间箱线图，右边散点图）
        plot_raincloud(df)
        
        print("\n" + "=" * 60)
        print("云雨图绘制完成！")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
