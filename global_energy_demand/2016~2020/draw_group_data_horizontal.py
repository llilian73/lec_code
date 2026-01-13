import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 全局字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.style'] = 'normal'


def load_group_avg_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
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
    colors = {
        'ref': '#E5E5E5',     # 浅灰
        'a1':  '#6C7F93',     # 灰蓝
        'a2':  '#7FA1D5',     # 浅蓝
        'a3':  '#E79B6B',     # 橙色
        'a4':  '#F2C757',     # 黄色
        'a5':  '#98C279',     # 浅绿
    }
    if case_name == 'ref':
        return colors['ref']
    try:
        n = int(case_name.replace('case', ''))
        alpha_idx = (n - 1) % 5 + 1
        return colors[f'a{alpha_idx}']
    except Exception:
        return '#CCCCCC'


def plot_horizontal(df: pd.DataFrame, save_pdf_path: str):
    groups = ['CN', 'IN', 'US', 'Europe', 'Others']
    cases = ['ref', 'case6', 'case7', 'case8', 'case9', 'case10']  # Texpansion=2℃

    df_sel = df[df['group'].isin(groups) & df['index'].isin(cases)].copy()
    # 排序
    df_sel['group_order'] = df_sel['group'].map({g: i for i, g in enumerate(groups)})
    df_sel['case_order'] = df_sel['index'].map({c: i for i, c in enumerate(cases)})
    df_sel = df_sel.sort_values(['group_order', 'case_order'])

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(3.0)
        spine.set_color('black')

    # y 位置（每组一行）
    y_positions = np.arange(len(groups)) + 1
    ax.set_yticks(y_positions)
    ax.set_yticklabels(groups, fontsize=16)
    ax.tick_params(axis='y', width=0, length=0)

    ax.set_xlabel('Annual electricity demand(TWh)', fontsize=18, labelpad=15)
    ax.set_ylabel('Group', fontsize=18, labelpad=15)

    # 计算 x 上限
    xmax = df_sel['total_demand_sum(TWh)'].max()
    xmax = xmax * 1.15 if pd.notna(xmax) else 1.0
    ax.set_xlim(0, xmax)

    bar_height = 0.5  # 修改柱子厚度的行

    # 每组一行：在同一 y 上按顺序叠画（ref 最先绘制，作为底层；后续 case 覆盖在上层）
    for gi, g in enumerate(groups):
        y = y_positions[gi]
        for case in cases:  # 顺序：ref -> case6 -> ... -> case10（顶层）
            row = df_sel[(df_sel['group'] == g) & (df_sel['index'] == case)]
            if row.empty:
                continue
            demand = row['total_demand_sum(TWh)'].values[0]
            red = row['total_demand_reduction(%)'].values[0]
            if not pd.notna(demand):
                continue
            color = get_case_color(case)
            ax.barh(y=y, width=demand, left=0.0, height=bar_height, color=color, edgecolor='none')
            # 仅标注 α1 与 α5 的节能率（对应 case6 与 case10）
            if case in ('case6', 'case10') and pd.notna(red):
                ax.text(demand, y + bar_height * 0.55, f"-{red:.0f}%", ha='center', va='top', fontsize=12)

    # 图例
    legend_labels = ['ref', r'α1', r'α2', r'α3', r'α4', r'α5']
    legend_cases = ['ref', 'case6', 'case7', 'case8', 'case9', 'case10']
    patches = [plt.Rectangle((0, 0), 1, 1, facecolor=get_case_color(c), edgecolor='none') for c in legend_cases]
    ax.legend(patches, legend_labels, loc='lower right', fontsize=12, frameon=False)

    # 翻转纵轴，使最上方为列表中的第一个组（CN）
    ax.invert_yaxis()

    ax.grid(False)
    ax.tick_params(axis='x', labelsize=14)

    plt.tight_layout()
    plt.savefig(save_pdf_path, format='pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    csv_path = r"Z:\local_environment_creation\energy_consumption\2016-2020result\group\group_summary_2016_2020_average.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = load_group_avg_csv(csv_path)
    out_pdf = os.path.join(os.path.dirname(csv_path), 'group_horizontal_Texpansion2_total.pdf')
    plot_horizontal(df, out_pdf)
    print(f'已保存: {out_pdf}')


