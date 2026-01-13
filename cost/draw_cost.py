"""
成本分布堆叠柱状图绘制工具

功能概述：
本工具用于根据total_cost数据按大洲绘制分组百分比堆叠柱状图。

输入数据：
- 文件：Z:\local_environment_creation\cost\country_yearly_per_cost_USD_PP.csv
- 数据列：Continent, Country_Code, Country_Name, equipment_cost, maintenance_cost, total_cost, save_elec_cost, PP

输出结果：
- PDF图片：Z:\local_environment_creation\cost\total_cost_distribution_chart.pdf
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 设置全局字体为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.style'] = 'normal'  # 非斜体


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


def load_cost_data(csv_path):
    """读取成本数据"""
    df = read_csv_with_encoding(csv_path, keep_default_na=False)
    print(f"成功加载数据，共 {len(df)} 行")
    
    # 检查必要的列是否存在
    required_cols = ['Continent', 'total_cost']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV文件中缺少必要的列: {missing_cols}")
    
    # 过滤掉Continent为Unknown的数据
    original_count = len(df)
    df = df[df['Continent'] != 'Unknown']
    df = df[df['Continent'].notna()]
    filtered_count = len(df)
    print(f"过滤Unknown大洲后，剩余 {filtered_count} 行（过滤了 {original_count - filtered_count} 行）")
    
    return df


def categorize_total_cost(value):
    """根据total_cost值分配到对应区间"""
    if pd.isna(value):
        return None
    if value < 10:
        return None  # 小于10的不在统计范围内，可以忽略或单独处理
    elif value < 20:
        return '10~20'
    elif value < 30:
        return '20~30'
    elif value < 40:
        return '30~40'
    elif value < 50:
        return '40~50'
    elif value < 60:
        return '50~60'
    else:
        return '>60'


def calculate_distribution_by_continent(df):
    """计算每个大洲的成本区间分布"""
    # 为每行数据分配区间
    df['range'] = df['total_cost'].apply(categorize_total_cost)
    
    # 过滤掉无法分类的数据（如小于10的）
    df = df[df['range'].notna()]
    
    # 按大洲和区间分组统计
    distribution_df = df.groupby(['Continent', 'range']).size().reset_index(name='count')
    
    # 计算每个大洲的总数
    continent_totals = df.groupby('Continent').size().reset_index(name='total')
    
    # 合并数据并计算百分比
    distribution_df = distribution_df.merge(continent_totals, on='Continent')
    distribution_df['rate'] = (distribution_df['count'] / distribution_df['total'] * 100).round(2)
    
    return distribution_df


def plot_stacked_percentage_chart(distribution_df, save_path):
    """绘制百分比堆叠柱状图"""
    # 定义颜色方案
    colors = {
        '10~20': '#D5D7D6',      # 浅灰
        '20~30': '#F1C3C1',      # 浅红
        '30~40': '#B7D5EC',      # 浅蓝
        '40~50': '#FEDFB1',      # 浅黄
        '50~60': '#B9C3DC',      # 浅紫
        '>60': '#E8D5C4'         # 浅棕
    }
    
    # 定义区间顺序
    range_order = ['10~20', '20~30', '30~40', '40~50', '50~60', '>60']
    
    # 定义大洲顺序
    continent_order = ['Asia', 'Europe', 'Africa', 'North America', 'South America', 'Oceania']
    
    # 创建数据透视表
    pivot_df = distribution_df.pivot(index='Continent', columns='range', values='rate').fillna(0)
    
    # 确保所有区间都存在
    for range_name in range_order:
        if range_name not in pivot_df.columns:
            pivot_df[range_name] = 0
    
    # 按指定顺序重新排列列
    pivot_df = pivot_df[range_order]
    
    # 获取所有大洲
    continents = pivot_df.index.unique()
    continents = [c for c in continents if pd.notna(c)]
    
    # 按指定顺序重新排列大洲
    continents = [c for c in continent_order if c in continents]
    # 添加不在预设顺序中的其他大洲
    other_continents = [c for c in continents if c not in continent_order]
    continents = [c for c in continent_order if c in continents] + sorted(other_continents)
    pivot_df = pivot_df.reindex(continents)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 设置背景
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')
    
    # 绘制堆叠柱状图
    bottom = np.zeros(len(continents))
    
    # 记录每个区间的位置信息，用于后续添加标签
    range_positions = {}  # {range_name: [bottom_heights]}
    
    for i, range_name in enumerate(range_order):
        values = pivot_df[range_name].values
        # 记录当前区间的bottom位置
        range_positions[range_name] = bottom.copy()
        ax.bar(continents, values, bottom=bottom, 
               color=colors[range_name], edgecolor='none', 
               label=range_name, alpha=0.8)
        bottom += values
    
    # 在每个洲的占比最大的区间上添加标签
    for idx, continent in enumerate(continents):
        # 找到该洲占比最大的区间
        continent_data = pivot_df.loc[continent]
        max_range = continent_data.idxmax()
        max_value = continent_data[max_range]
        
        # 如果最大值为0，跳过（无数据）
        if max_value == 0:
            continue
        
        # 计算标签位置（在该区间的中间位置）
        bottom_pos = range_positions[max_range][idx]
        height = max_value
        label_y = bottom_pos + height / 2
        
        # 格式化标签文本（四舍五入到整数，无小数位）
        label_text = f"{int(round(max_value))}%"
        
        # 添加标签
        ax.text(idx, label_y, label_text, 
                ha='center', va='center', 
                fontsize=12, fontweight='bold',
                color='black')
    
    # 设置坐标轴
    ax.set_ylabel('Proportion of countries (%)', fontsize=16, labelpad=15)
    ax.set_xlabel('Continent', fontsize=16, labelpad=15)
    
    # 设置Y轴范围
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    
    # 设置X轴标签旋转
    ax.tick_params(axis='x', rotation=0, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    # 添加图例（无边框）
    legend_elements = [mpatches.Patch(color=colors[range_name], label=range_name) 
                      for range_name in range_order]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
              fontsize=14, frameon=False, handleheight=2, handlelength=2)
    
    # 添加网格
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"已保存图片: {save_path}")


def main():
    """主函数"""
    # 输入文件
    input_file = r"Z:\local_environment_creation\cost\method2\country_yearly_per_cost_USD_PP.csv"
    
    # 输出文件
    output_file = r"Z:\local_environment_creation\cost\method2\total_cost_distribution_chart.pdf"
    
    print("开始处理成本分布数据...")
    print("=" * 60)
    
    try:
        # 1. 加载数据
        df = load_cost_data(input_file)
        
        # 2. 计算分布
        print("\n计算各大洲成本区间分布...")
        distribution_df = calculate_distribution_by_continent(df)
        print(f"分布数据：\n{distribution_df.head(20)}")
        
        # 3. 绘制图表
        print("\n绘制堆叠柱状图...")
        plot_stacked_percentage_chart(distribution_df, output_file)
        
        print("\n" + "=" * 60)
        print("处理完成！")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()

