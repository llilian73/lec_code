import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fm

# 设置全局字体为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.style'] = 'normal'  # 非斜体


def load_and_average(csv_path):
    """读取一个CSV，计算2016-2020年每个case的 demand_sum 与 reduction_percentage 的平均值。
    返回 dict: {case_name: {"demand": float, "reduction": float}}
    """
    df = pd.read_csv(csv_path)
    # 仅保留2016-2020
    df = df[df['year'].between(2016, 2020)]

    # 确保数值列类型
    df['demand_sum'] = pd.to_numeric(df['demand_sum'], errors='coerce')
    df['reduction_percentage'] = pd.to_numeric(df['reduction_percentage'], errors='coerce')

    # 按 case 求均值
    grouped = df.groupby('index', as_index=False).agg({
        'demand_sum': 'mean',
        'reduction_percentage': 'mean'
    })

    results = {}
    for _, row in grouped.iterrows():
        key = str(row['index']).strip()
        results[key] = {
            'demand': float(row['demand_sum']) if pd.notna(row['demand_sum']) else np.nan,
            'reduction': float(row['reduction_percentage']) if pd.notna(row['reduction_percentage']) else np.nan
        }
    return results


def plot_global_bars(avg_by_case, title, save_pdf_path):
    """绘制示例样式图：四行分别对应 diff=1,2,3,4℃，每行堆叠显示 ref 与 α1-α5。
    avg_by_case: load_and_average 返回的 dict
    """
    # 颜色（近似示例配色）：α5, α4, α3, α2, α1, ref
    colors = {
        'a5': '#98C279',   # 近似浅绿
        'a4': '#F2C757',   # 近似黄
        'a3': '#E79B6B',   # 近似橙
        'a2': '#7FA1D5',   # 近似浅蓝
        'a1': '#6C7F93',   # 近似灰蓝
        'ref': '#E5E5E5'   # 浅灰
    }

    # α标签与数值映射（显示时用 α1..α5）
    alpha_labels = [('a5', r'α5'), ('a4', r'α4'), ('a3', r'α3'), ('a2', r'α2'), ('a1', r'α1')]

    # 每一行的 case 列表（堆叠顺序左->右），以示例为准 α5 ... α1 后跟 ref
    temp_rows = [
        {'diff': 1, 'cases': ['case5', 'case4', 'case3', 'case2', 'case1', 'ref']},
        {'diff': 2, 'cases': ['case10', 'case9', 'case8', 'case7', 'case6', 'ref']},
        {'diff': 3, 'cases': ['case15', 'case14', 'case13', 'case12', 'case11', 'ref']},
        {'diff': 4, 'cases': ['case20', 'case19', 'case18', 'case17', 'case16', 'ref']},
    ]

    # 将 case 名映射到颜色键
    def case_to_color_key(case):
        if case == 'ref':
            return 'ref'
        # caseX -> α 序号：X%5 映射 a1..a5（注意示例定义：case1->α1=0.03125, ... case5->α5=0.5）
        n = int(case.replace('case', ''))
        pos = (n - 1) % 5 + 1  # 1..5
        return f'a{pos}'

    # 画布
    fig, ax = plt.subplots(figsize=(14, 8))

    # 轴样式
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(3.0)
        spine.set_color('black')

    # Y 轴为 1..4
    y_positions = [1, 2, 3, 4]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(["1", "2", "3", "4"], fontsize=16)
    ax.tick_params(axis='y', width=0, length=0)

    # X 轴标签（非斜体）
    ax.set_xlabel('Annual electricity demand(TWh)', fontsize=18, labelpad=15)
    ax.set_ylabel(r'$T_{expansion}$', fontsize=18, rotation=0, labelpad=60)

    # 根据所有数据决定最大 X 范围（使用该 CSV 所有 case 的最大需求的 1.1 倍）
    all_demands = [v['demand'] for v in avg_by_case.values() if pd.notna(v['demand'])]
    xmax = max(all_demands) * 1.15 if all_demands else 1.0
    ax.set_xlim(0, xmax)

    # 在同一温度组内，6 个工况柱状完全重叠（同一 Y 位置），后绘制的覆盖前面的
    bar_height = 0.5  # 这一行是修改柱子宽度（高度）的代码

    for row_idx, row in enumerate(temp_rows):
        y = y_positions[row_idx]
        # 反转顺序绘制，实现覆盖效果：ref 先绘制（底层），α5 最后绘制（顶层）
        for case in reversed(row['cases']):
            if case not in avg_by_case:
                continue
            demand = avg_by_case[case]['demand']
            reduction = avg_by_case[case]['reduction']
            if not pd.notna(demand):
                continue
            ckey = case_to_color_key(case)
            color = colors.get(ckey, '#CCCCCC')
            # 绘制：全部从 0 开始（left=0），在同一 Y 位置
            ax.barh(y=y, width=demand, left=0.0, height=bar_height, color=color, edgecolor='none')
            # 仅标注 α1 与 α5 的节能率（非 ref）
            if case != 'ref' and pd.notna(reduction):
                try:
                    n = int(case.replace('case', ''))
                    alpha_idx = (n - 1) % 5 + 1  # 1..5 对应 α1..α5
                except Exception:
                    alpha_idx = None
                if alpha_idx in (1, 5):
                    txt = f"-{reduction:.0f}%"
                    ax.text(demand, y - bar_height * 0.55, txt, ha='center', va='top', fontsize=14)

    # 竖向图例（与 draw_group_data_horizontal.py 一致风格）
    legend_labels = ['ref', r'α1', r'α2', r'α3', r'α4', r'α5']
    # 选择一行的 case 映射颜色：使用 diff=2℃ 的映射即可
    legend_cases = ['ref', 'case6', 'case7', 'case8', 'case9', 'case10']
    def _legend_color_for_case(case_name: str) -> str:
        if case_name == 'ref':
            return colors['ref']
        try:
            n = int(case_name.replace('case', ''))
            a = (n - 1) % 5 + 1
            return colors[f'a{a}']
        except Exception:
            return '#CCCCCC'
    legend_patches = [Rectangle((0, 0), 1, 1, facecolor=_legend_color_for_case(c), edgecolor='none') for c in legend_cases]
    ax.legend(legend_patches, legend_labels, loc='lower right', fontsize=12, frameon=False)

    # 网格与刻度
    ax.grid(False)
    ax.tick_params(axis='x', labelsize=14)

    # 标题（可选）
    # ax.set_title(title, fontsize=18)

    plt.tight_layout()
    plt.savefig(save_pdf_path, format='pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    base_dir = r"Z:\local_environment_creation\energy_consumption\2016-2020result\global_total_energy_demand"
    sub_dir = os.path.join(base_dir, "horizontal")
    csv_files = [
        'global_per_capita_cooling_energy.csv',
        'global_per_capita_heating_energy.csv',
        'global_per_capita_total_energy.csv',
        'global_total_cooling_energy.csv',
        'global_total_heating_energy.csv',
        'global_total_total_energy.csv'
    ]

    for fname in csv_files:
        csv_path = os.path.join(base_dir, fname)
        if not os.path.exists(csv_path):
            print(f"未找到: {csv_path}")
            continue
        print(f"正在处理: {fname} …")
        avg_by_case = load_and_average(csv_path)
        # 固定顺序导出：ref, case1..case20
        export_order = ['ref'] + [f'case{i}' for i in range(1, 21)]
        # 导出平均后的数据到新的 CSV
        avg_rows = []
        for case_name in export_order:
            vals = avg_by_case.get(case_name, {'demand': np.nan, 'reduction': np.nan})
            avg_rows.append({
                'index': case_name,
                'demand_sum_avg': vals.get('demand', np.nan),
                'reduction_percentage_avg': vals.get('reduction', np.nan)
            })
        avg_df = pd.DataFrame(avg_rows)
        avg_csv_name = os.path.splitext(fname)[0] + '_average.csv'
        avg_csv_path = os.path.join(sub_dir, avg_csv_name)
        avg_df.to_csv(avg_csv_path, index=False)
        print(f"已保存: {avg_csv_path}")
        pdf_name = os.path.splitext(fname)[0] + '.pdf'
        save_pdf_path = os.path.join(sub_dir, pdf_name)
        plot_global_bars(avg_by_case, title=fname, save_pdf_path=save_pdf_path)
        print(f"已保存: {save_pdf_path}")