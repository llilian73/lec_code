import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 字体设置
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.style'] = 'normal'


def load_group_avg_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
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
    # 颜色方案参考 draw_global_data_vertical.py
    base_colors = {
        'ref': '#E5E5E5',     # 浅灰
        'a1':  '#6C7F93',     # 灰蓝
        'a2':  '#7FA1D5',     # 浅蓝
        'a3':  '#E79B6B',     # 橙色
        'a4':  '#F2C757',     # 黄色
        'a5':  '#98C279',     # 浅绿
    }
    if case_name == 'ref':
        return base_colors['ref']
    # 映射 case6..10 -> α1..α5
    try:
        n = int(case_name.replace('case', ''))
        alpha_idx = (n - 1) % 5 + 1
        return base_colors[f'a{alpha_idx}']
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
    bar_width = 0.12
    group_gap = 0.25

    # 计算每组的起始x
    group_width = bars_per_group * bar_width + group_gap
    x_starts = [i * group_width for i in range(num_groups)]

    # Y轴范围
    ymax = df_sel['total_demand_sum(TWh)'].max()
    if pd.isna(ymax):
        ymax = 1.0
    ax.set_ylim(0, ymax * 1.25)

    # 绘制每组
    for gi, g in enumerate(groups):
        x0 = x_starts[gi]
        for ci, case in enumerate(cases):
            row = df_sel[(df_sel['group'] == g) & (df_sel['index'] == case)]
            if row.empty:
                continue
            total_demand = float(row['total_demand_sum(TWh)'].values[0]) if pd.notna(row['total_demand_sum(TWh)'].values[0]) else 0.0
            reduction = row['total_demand_reduction(%)'].values[0]
            x = x0 + ci * bar_width
            color = get_case_color(case)
            ax.bar(x, total_demand, bar_width, color=color, edgecolor='none')
            # 顶部标注节能率（非ref）
            if case != 'ref' and pd.notna(reduction):
                ax.text(x, total_demand + ymax * 0.01, f"-{reduction:.0f}%", ha='center', va='bottom', fontsize=10)

    # X轴：组名刻度与每组第三个柱子的右侧对齐（索引2的柱子右边缘）
    #xticks = [x0 + (2 + 1) * bar_width for x0 in x_starts]
    xticks = [i * group_width + 2 * bar_width + bar_width/2 for i in range(n)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(groups, fontsize=16)

    ax.set_xlabel('Group', fontsize=18, labelpad=12)
    ax.set_ylabel('Annual electricity demand(TWh)', fontsize=18, labelpad=12)

    # 构建图例：ref, α1..α5
    legend_labels = ['ref', r'α1', r'α2', r'α3', r'α4', r'α5']
    legend_cases = ['ref', 'case6', 'case7', 'case8', 'case9', 'case10']
    legend_patches = [plt.Rectangle((0, 0), 1, 1, facecolor=get_case_color(c), edgecolor='none') for c in legend_cases]
    ax.legend(legend_patches, legend_labels, loc='upper right', fontsize=12, ncol=1, frameon=False)

    ax.grid(True, axis='y', alpha=0.25, linestyle='-', linewidth=0.6)
    ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    plt.savefig(save_pdf_path, format='pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    csv_path = r"Z:\local_environment_creation\energy_consumption\2016-2020result\group\group_summary_2016_2020_average.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = load_group_avg_csv(csv_path)
    out_pdf = os.path.join(os.path.dirname(csv_path), 'group_vertical_Texpansion2_total.pdf')
    plot_vertical_bars(df, out_pdf)
    print(f'已保存: {out_pdf}')


