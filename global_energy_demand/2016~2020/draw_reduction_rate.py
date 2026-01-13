import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 设置全局字体为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.style'] = 'normal'  # 非斜体


def load_reduction_rate_data(csv_path):
    """读取节能率分布数据"""
    df = pd.read_csv(csv_path)
    required_cols = {'continent', 'range', 'rate'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"文件 {csv_path} 缺少必要列: {', '.join(missing_cols)}")
    return df


def plot_stacked_percentage_chart(df, case_name, save_path):
    """绘制百分比堆叠柱状图"""
    # 全量颜色方案（10档）
    full_colors = {
        '0~10%': '#D5D7D6',
        '10~20%': '#F1C3C1',
        '20~30%': '#B7D5EC',
        '30~40%': '#FEDFB1',
        '40~50%': '#B9C3DC',
        '50~60%': '#E8D5C4',
        '60~70%': '#B2D3A4',
        '70~80%': '#F18072',
        '80~90%': '#BA7FB5',
        '90~100%': '#8CD0C3'
    }
    
    # 根据数据确定需要绘制的区间（来自C_reduction_rate输出）
    all_ranges = [
        '0~10%', '10~20%', '20~30%', '30~40%', '40~50%',
        '50~60%', '60~70%', '70~80%', '80~90%', '90~100%'
    ]
    distribution_df = df.copy()
    distribution_df = distribution_df[distribution_df['range'].isin(all_ranges)]
    distribution_df = distribution_df.dropna(subset=['continent', 'range'])
    distribution_df['rate'] = pd.to_numeric(distribution_df['rate'], errors='coerce').fillna(0)
    range_totals = distribution_df.groupby('range')['rate'].sum()
    available_ranges = [r for r in all_ranges if r in range_totals.index and range_totals[r] > 0]
    if not available_ranges:
        print(f"警告：{case_name} 中未找到有效区间，跳过绘制。")
        return
    distribution_df = distribution_df[distribution_df['range'].isin(available_ranges)]
    colors = {label: full_colors.get(label, '#CCCCCC') for label in available_ranges}
    
    # 定义大洲顺序（确保数据对应正确）
    continent_order = ['Asia', 'Europe', 'Africa', 'North America', 'South America', 'Oceania']
    
    # 获取所有大洲
    continents = distribution_df['continent'].unique()
    continents = [c for c in continents if pd.notna(c)]
    
    # 创建数据透视表
    pivot_df = distribution_df.pivot(index='continent', columns='range', values='rate').fillna(0)
    
    # 确保所有区间都存在
    for range_name in available_ranges:
        if range_name not in pivot_df.columns:
            pivot_df[range_name] = 0
    
    # 按指定顺序重新排列列
    pivot_df = pivot_df[available_ranges]
    
    # 按指定顺序重新排列大洲（确保数据对应正确）
    continents = [c for c in continent_order if c in continents]
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
    
    for i, range_name in enumerate(available_ranges):
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
    
    # # 调试：检查每个大洲的总和
    # print(f"调试信息 - {case_name}:")
    # for continent in continents:
    #     total = pivot_df.loc[continent].sum()
    #     print(f"  {continent}: {total:.2f}%")
    #     if abs(total - 100.0) > 0.01:  # 允许小的浮点误差
    #         print(f"    警告: {continent} 总和不为100%")
    
    # 设置坐标轴
    ax.set_ylabel('Proportion of countries (%)', fontsize=16, labelpad=15)
    ax.set_xlabel('Continent', fontsize=16, labelpad=15)
    
    # 设置Y轴范围
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    
    # 设置X轴标签旋转（调整为更平的角度）
    ax.tick_params(axis='x', rotation=0, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    # 添加图例（无边框）
    legend_elements = [mpatches.Patch(color=colors[range_name], label=range_name) 
                      for range_name in available_ranges]
    legend = ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
                       fontsize=14, frameon=False, handleheight=2, handlelength=2)
    
    # 添加网格
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 设置标题
    # ax.set_title(f'Energy Reduction Rate Distribution - {case_name}', fontsize=18, pad=20)
    
    # 调整布局
    plt.tight_layout()

    # 在图例上方添加说明文字
    fig = plt.gcf()
    fig.canvas.draw()
    legend_bbox = legend.get_window_extent()
    legend_bbox_fig = legend_bbox.transformed(fig.transFigure.inverted())
    legend_center_x = legend_bbox_fig.x0 + legend_bbox_fig.width / 2
    title_y = legend_bbox_fig.y1 + 0.02
    fig.text(legend_center_x, title_y, 'energy saving\nrate',
             ha='center', va='bottom', fontsize=12)
    
    # 保存图片
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"已保存图片: {save_path}")


def process_all_reduction_rate_files(input_dir, output_dir):
    """处理所有节能率分布文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有_reduction_rate_distribution.csv文件
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('_reduction_rate_distribution.csv')]
    
    if not csv_files:
        print(f"在 {input_dir} 中未找到节能率分布文件")
        return
    
    print(f"找到 {len(csv_files)} 个节能率分布文件")
    
    # 处理每个文件
    for csv_file in csv_files:
        print(f"\n正在处理: {csv_file}")
        
        # 读取数据
        csv_path = os.path.join(input_dir, csv_file)
        df = load_reduction_rate_data(csv_path)
        
        # 提取case名称
        case_name = csv_file.replace('_reduction_rate_distribution.csv', '')
        
        # 生成输出文件名
        output_filename = f"{case_name}_reduction_rate_chart.pdf"
        output_path = os.path.join(output_dir, output_filename)
        
        # 绘制图表
        plot_stacked_percentage_chart(df, case_name, output_path)
        
        print(f"完成处理: {csv_file}")


def main():
    # 输入目录（包含_reduction_rate_distribution.csv文件）
    input_dir = r"Z:\local_environment_creation\energy_consumption_gird\result\result\reduction_rate"
    
    # 输出目录（保存PDF图片）
    output_dir = r"Z:\local_environment_creation\energy_consumption_gird\result\result\reduction_rate\figure"
    
    print("开始处理节能率分布文件...")
    process_all_reduction_rate_files(input_dir, output_dir)
    print("\n所有文件处理完成！")


if __name__ == '__main__':
    main()
