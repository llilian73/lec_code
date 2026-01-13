import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 字体设置
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.style'] = 'normal'


def load_group_avg_csv(csv_path: str) -> pd.DataFrame:
    # 关键修正：避免将 'NA' 识别为缺失值
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[''])
    # 规范列名，确保存在
    expected_cols = [
        'group', 'index',
        'total_demand_sum(TWh)', 'total_demand_diff(TWh)', 'total_demand_reduction(%)',
        'heating_demand_sum(TWh)', 'heating_demand_diff(TWh)', 'heating_demand_reduction(%)',
        'cooling_demand_sum(TWh)', 'cooling_demand_diff(TWh)', 'cooling_demand_reduction(%)'
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f'CSV缺少必要列: {missing}')
    # 数值化
    for col in expected_cols:
        if col in ['group', 'index']:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['group'] = df['group'].astype(str)
    df['index'] = df['index'].astype(str)
    return df


def get_case_color(case_name: str) -> str:
    # 颜色方案与 draw_global_data_vertical.py 保持一致
    base_colors = {
        'ref': '#E5E5E5',  #'#E5E5E5'
        'case1': '#F1C3C1',#'#E89DA0'
        'case2': '#B7D5EC',#'#88CEE6'
        'case3': '#FEDFB1',#'#F6C8A8'
        'case4': '#B9C3DC',#'#B2D3A4'
        'case5': '#B2D3A4',#'#E6CECF'
    }

  
    if case_name == 'ref':
        return base_colors['ref']
    
    # 映射 case1-5, case6-10, case11-15, case16-20 分别对应颜色
    try:
        n = int(case_name.replace('case', ''))
        alpha_idx = (n - 1) % 5 + 1
        color_key = f'case{alpha_idx}'
        return base_colors.get(color_key, '#CCCCCC')
    except Exception:
        return '#CCCCCC'


def plot_vertical_bars(df: pd.DataFrame, save_pdf_path: str):
    # 选择组与工况
    groups = ['CN', 'IN', 'US', 'Europe', 'Others']
    cases = ['ref', 'case6', 'case7', 'case8', 'case9', 'case10']  # Texpansion=2℃
    n = len(groups)
    # 过滤并构造数据表
    df_sel = df[df['group'].isin(groups) & df['index'].isin(cases)].copy()

    # 按固定顺序排序
    df_sel['group_order'] = df_sel['group'].map({g: i for i, g in enumerate(groups)})
    case_order_map = {c: i for i, c in enumerate(cases)}
    df_sel['case_order'] = df_sel['index'].map(case_order_map)
    df_sel = df_sel.sort_values(['group_order', 'case_order'])

    # 画布
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')

    # 条形图参数
    num_groups = len(groups)
    bars_per_group = len(cases)
    bar_width = 0.15  # 增加柱子宽度：0.12 -> 0.15
    group_gap = 0.1

    # 计算每组的起始x
    group_width = bars_per_group * bar_width + group_gap
    x_starts = [i * group_width for i in range(num_groups)]

    # Y轴范围
    ymax = df_sel['total_demand_sum(TWh)'].max()
    if pd.isna(ymax):
        ymax = 1.0
    ax.set_ylim(0, ymax * 1.1)

    # 绘制每组 - 使用堆叠柱状图（供暖+制冷）
    for gi, g in enumerate(groups):
        x0 = x_starts[gi]
        for ci, case in enumerate(cases):
            row = df_sel[(df_sel['group'] == g) & (df_sel['index'] == case)]
            if row.empty:
                continue
            
            # 获取供暖、制冷、总需求数据
            heating_demand = float(row['heating_demand_sum(TWh)'].values[0]) if pd.notna(row['heating_demand_sum(TWh)'].values[0]) else 0.0
            cooling_demand = float(row['cooling_demand_sum(TWh)'].values[0]) if pd.notna(row['cooling_demand_sum(TWh)'].values[0]) else 0.0
            total_demand = float(row['total_demand_sum(TWh)'].values[0]) if pd.notna(row['total_demand_sum(TWh)'].values[0]) else 0.0
            
            # 获取节能率
            heating_reduction = row['heating_demand_reduction(%)'].values[0]
            cooling_reduction = row['cooling_demand_reduction(%)'].values[0]
            total_reduction = row['total_demand_reduction(%)'].values[0]
            
            x = x0 + ci * bar_width
            color = get_case_color(case)
            
            # 绘制供暖部分（底层，深色 alpha=0.9）
            if heating_demand > 0:
                ax.bar(x, heating_demand, bar_width, color=color, edgecolor='none', alpha=0.9)
            
            # 绘制制冷部分（顶层，浅色 alpha=0.6）
            if cooling_demand > 0:
                ax.bar(x, cooling_demand, bar_width, bottom=heating_demand, color=color, edgecolor='none', alpha=0.6)
            
            # 标注节能率（非ref工况）
            if case != 'ref':
                # # 供暖部分节能率（中间位置）
                # if heating_demand > 0 and pd.notna(heating_reduction):
                #     ax.text(x, heating_demand * 0.5, f"-{heating_reduction:.0f}%", 
                #            ha='center', va='center', fontsize=10, color='white', weight='bold')
                # # 制冷部分节能率（中间位置）
                # if cooling_demand > 0 and pd.notna(cooling_reduction):
                #     ax.text(x, heating_demand + cooling_demand * 0.5, f"-{cooling_reduction:.0f}%", 
                #            ha='center', va='center', fontsize=10, color='white', weight='bold')
                # 总需求节能率（柱子顶部）
                if pd.notna(total_reduction):
                    ax.text(x, total_demand + ymax * 0.01, f"-{total_reduction:.0f}%", 
                           ha='center', va='bottom', fontsize=10, weight='bold')

    # X轴：组名刻度与每组第三个柱子的右侧对齐（索引2的柱子右边缘）
    xticks = [i * group_width + 2 * bar_width + bar_width/2 for i in range(n)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(groups, fontsize=16)

    ax.set_xlabel('Group', fontsize=18, labelpad=12)
    ax.set_ylabel('Annual electricity demand(TWh)', fontsize=18, labelpad=12)

    # 构建图例：ref, α1..α5，并区分 Heating/Cooling
    legend_labels = ['ref', r'α1', r'α2', r'α3', r'α4', r'α5']
    legend_cases = ['ref', 'case6', 'case7', 'case8', 'case9', 'case10']
    
    # 创建自定义图例 - 在右上角绘制
    legend_x = 0.85
    legend_y = 0.95
    
    # 绘制图例内容
    for i, (label, case) in enumerate(zip(legend_labels, legend_cases)):
        row_y = legend_y - 0.05 - i * 0.04
        color = get_case_color(case)
        
        # 供暖色块（深色）
        heating_x = legend_x + 0.02
        heating_width = 0.03
        ax.add_patch(plt.Rectangle((heating_x, row_y-0.02), heating_width, 0.04, 
                                  transform=ax.transAxes, facecolor=color, edgecolor='none', 
                                  alpha=0.9, zorder=11))
        
        # 制冷色块（浅色）
        cooling_x = heating_x + heating_width + 0.01
        cooling_width = 0.03
        ax.add_patch(plt.Rectangle((cooling_x, row_y-0.02), cooling_width, 0.04, 
                                  transform=ax.transAxes, facecolor=color, edgecolor='none', 
                                  alpha=0.6, zorder=11))
        
        # 工况标签
        label_x = cooling_x + cooling_width + 0.02
        ax.text(label_x, row_y, label, transform=ax.transAxes, ha='left', va='center', 
                fontsize=10, zorder=12)
    
    # 添加Heating和Cooling标题
    ax.text(legend_x + 0.035, legend_y - 0.02, 'Heating', transform=ax.transAxes, 
            ha='center', va='center', fontsize=9, weight='bold', zorder=12)
    ax.text(legend_x + 0.075, legend_y - 0.02, 'Cooling', transform=ax.transAxes, 
            ha='center', va='center', fontsize=9, weight='bold', zorder=12)

    ax.grid(True, axis='y', alpha=0.25, linestyle='-', linewidth=0.6)
    ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    plt.savefig(save_pdf_path, format='pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    csv_path = r"Z:\local_environment_creation\energy_consumption_gird\result\result\group\group_summary_2016_2020_average.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = load_group_avg_csv(csv_path)
    out_pdf = os.path.join(os.path.dirname(csv_path), 'group_vertical_Texpansion2_total.pdf')
    plot_vertical_bars(df, out_pdf)
    print(f'已保存: {out_pdf}')


