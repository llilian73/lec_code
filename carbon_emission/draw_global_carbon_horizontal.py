"""
全球总碳排放量水平柱状图绘制工具（2016-2020年）

功能概述：
本工具用于绘制全球总碳排放量的水平堆叠柱状图，展示不同节能案例（case1-case20）的碳排放量和减排效果。

输入数据：
- global_total_carbon_emission.csv：全球总碳排放数据（2016-2020年）
- 格式：year, index, carbon_emission(tCO2), carbon_emission_reduction(tCO2), carbon_emission_reduction(%)

输出结果：
- global_total_carbon_emission_average.csv：5年平均碳排放数据
- global_total_carbon_emission.pdf：水平堆叠柱状图
"""

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
    """读取CSV，计算2016-2020年每个case的碳排放量平均值，然后基于平均值计算减少量和减少率。
    返回 dict: {case_name: {"emission": float, "reduction": float}}
    """
    df = pd.read_csv(csv_path)
    # 仅保留2016-2020
    df = df[df['year'].between(2016, 2020)]

    # 确保数值列类型
    df['carbon_emission(tCO2)'] = pd.to_numeric(df['carbon_emission(tCO2)'], errors='coerce')
    df['carbon_emission_reduction(tCO2)'] = pd.to_numeric(df['carbon_emission_reduction(tCO2)'], errors='coerce')
    df['carbon_emission_reduction(%)'] = pd.to_numeric(df['carbon_emission_reduction(%)'], errors='coerce')

    # 按 case 求碳排放量平均值
    grouped = df.groupby('index', as_index=False).agg({
        'carbon_emission(tCO2)': 'mean'
    })

    results = {}
    # 获取ref的平均碳排放量
    ref_row = grouped[grouped['index'] == 'ref']
    ref_emission_avg = float(ref_row['carbon_emission(tCO2)'].iloc[0]) if len(ref_row) > 0 and pd.notna(ref_row['carbon_emission(tCO2)'].iloc[0]) else np.nan

    for _, row in grouped.iterrows():
        case_name = str(row['index']).strip()
        emission_avg = float(row['carbon_emission(tCO2)']) if pd.notna(row['carbon_emission(tCO2)']) else np.nan
        
        # 对于非ref的case，基于平均值计算减少量和减少率
        if case_name != 'ref' and pd.notna(emission_avg) and pd.notna(ref_emission_avg) and ref_emission_avg > 0:
            reduction = ref_emission_avg - emission_avg
            reduction_rate = (reduction / ref_emission_avg) * 100.0
        else:
            reduction = np.nan
            reduction_rate = np.nan

        results[case_name] = {
            'emission': emission_avg,
            'reduction': reduction_rate  # 存储减少率用于绘图标注
        }
    
    return results


def plot_global_bars(avg_by_case, title, save_pdf_path):
    """绘制水平堆叠柱状图：四行分别对应 diff=1,2,3,4℃，每行堆叠显示 ref 与 α1-α5。
    avg_by_case: load_and_average 返回的 dict
    """
    # 颜色（参考draw_global_data_vertical.py的配色方案）
    colors = {
        'a1': '#F1C3C1',   # 灰蓝/粉红（对应case1）
        'a2': '#B7D5EC',   # 浅蓝（对应case2）
        'a3': '#FEDFB1',   # 橙色（对应case3）
        'a4': '#B9C3DC',   # 黄色/淡紫（对应case4）
        'a5': '#B2D3A4',   # 浅绿（对应case5）
        'ref': '#E5E5E5'   # 浅灰
    }

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
        # caseX -> α 序号：X%5 映射 a1..a5
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
    ax.set_xlabel('Carbon emission (GtCO2)', fontsize=18, labelpad=15)
    ax.set_ylabel(r'$T_{expansion}(\degree C)$', fontsize=18, rotation=90, labelpad=20)

    # 转换因子：tCO2 转 GtCO2 (1 GtCO2 = 1e9 tCO2)
    conversion_factor = 1e9

    # 根据所有数据决定最大 X 范围（转换为 GtCO2，使用该 CSV 所有 case 的最大碳排放量的 1.15 倍）
    all_emissions = [v['emission'] / conversion_factor for v in avg_by_case.values() if pd.notna(v['emission'])]
    xmax = max(all_emissions) * 1.15 if all_emissions else 1.0
    ax.set_xlim(0, xmax)

    # 在同一温度组内，6 个工况柱状完全重叠（同一 Y 位置），后绘制的覆盖前面的
    bar_height = 0.5  # 这一行是修改柱子宽度（高度）的代码

    for row_idx, row in enumerate(temp_rows):
        y = y_positions[row_idx]
        # 反转顺序绘制，实现覆盖效果：ref 先绘制（底层），α5 最后绘制（顶层）
        for case in reversed(row['cases']):
            if case not in avg_by_case:
                continue
            emission = avg_by_case[case]['emission']
            reduction = avg_by_case[case]['reduction']
            if not pd.notna(emission):
                continue
            # 转换为 GtCO2
            emission_gt = emission / conversion_factor
            ckey = case_to_color_key(case)
            color = colors.get(ckey, '#CCCCCC')
            # 绘制：全部从 0 开始（left=0），在同一 Y 位置
            ax.barh(y=y, width=emission_gt, left=0.0, height=bar_height, color=color, edgecolor='none')
            # 仅标注 α1 与 α5 的减排率（非 ref）
            if case != 'ref' and pd.notna(reduction):
                try:
                    n = int(case.replace('case', ''))
                    alpha_idx = (n - 1) % 5 + 1  # 1..5 对应 α1..α5
                except Exception:
                    alpha_idx = None
                if alpha_idx in (1, 5):
                    txt = f"-{reduction:.0f}%"
                    ax.text(emission_gt, y - bar_height * 0.55, txt, ha='center', va='top', fontsize=14)

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
    # 输入文件路径
    input_file = r"Z:\local_environment_creation\carbon_emission\2016-2020\global_total_carbon_emission\global_total_carbon_emission.csv"
    
    # 输出目录
    output_dir = r"Z:\local_environment_creation\carbon_emission\2016-2020\global_total_carbon_emission"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_file):
        print(f"未找到输入文件: {input_file}")
    else:
        print(f"正在处理: global_total_carbon_emission.csv …")
        avg_by_case = load_and_average(input_file)
        
        # 固定顺序导出：ref, case1..case20
        export_order = ['ref'] + [f'case{i}' for i in range(1, 21)]
        
        # 导出平均后的数据到新的 CSV
        avg_rows = []
        for case_name in export_order:
            vals = avg_by_case.get(case_name, {'emission': np.nan, 'reduction': np.nan})
            emission_avg = vals.get('emission', np.nan)
            
            # 计算减少量（仅对非ref工况）
            if case_name != 'ref' and pd.notna(emission_avg):
                ref_emission = avg_by_case.get('ref', {}).get('emission', np.nan)
                if pd.notna(ref_emission) and ref_emission > 0:
                    reduction = ref_emission - emission_avg
                    reduction_rate = vals.get('reduction', np.nan)
                else:
                    reduction = np.nan
                    reduction_rate = np.nan
            else:
                reduction = np.nan
                reduction_rate = np.nan
            
            avg_rows.append({
                'index': case_name,
                'carbon_emission(tCO2)': emission_avg,
                'carbon_emission_reduction(tCO2)': reduction,
                'carbon_emission_reduction(%)': reduction_rate
            })
        
        avg_df = pd.DataFrame(avg_rows)
        avg_csv_name = 'global_total_carbon_emission_average.csv'
        avg_csv_path = os.path.join(output_dir, avg_csv_name)
        avg_df.to_csv(avg_csv_path, index=False)
        print(f"已保存: {avg_csv_path}")
        
        pdf_name = 'global_total_carbon_emission.pdf'
        save_pdf_path = os.path.join(output_dir, pdf_name)
        plot_global_bars(avg_by_case, title='global_total_carbon_emission', save_pdf_path=save_pdf_path)
        print(f"已保存: {save_pdf_path}")

