"""
分组碳排放量水平柱状图绘制工具（2016-2020年）

功能概述：
本工具用于绘制分组碳排放量的水平堆叠柱状图，展示不同组别（CN、IN、US、Europe、Others）在不同节能案例下的碳排放量和减排效果。

输入数据：
- group_summary_2016_2020_average.csv：分组碳排放数据（5年平均）
- 格式：group, index, carbon_emission(tCO2), carbon_emission_reduction(tCO2), carbon_emission_reduction(%)

输出结果：
- 4张PDF图，分别对应拓展温度1、2、3、4℃
- 每张图显示5个组别（纵轴），每行堆叠显示 ref 与 α1-α5
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


def load_group_data(csv_path):
    """读取分组CSV数据
    返回 dict: {group: {case: {"emission": float, "reduction": float}}}
    """
    df = pd.read_csv(csv_path, keep_default_na=False)
    
    # 确保数值列类型
    df['carbon_emission(tCO2)'] = pd.to_numeric(df['carbon_emission(tCO2)'], errors='coerce')
    df['carbon_emission_reduction(%)'] = pd.to_numeric(df['carbon_emission_reduction(%)'], errors='coerce')
    
    # 组织数据结构
    results = {}
    groups = ['CN', 'IN', 'US', 'Europe', 'Others']
    
    for group in groups:
        results[group] = {}
        group_data = df[df['group'] == group]
        
        for _, row in group_data.iterrows():
            case_name = str(row['index']).strip()
            emission = float(row['carbon_emission(tCO2)']) if pd.notna(row['carbon_emission(tCO2)']) else np.nan
            reduction_rate = float(row['carbon_emission_reduction(%)']) if pd.notna(row['carbon_emission_reduction(%)']) else np.nan
            
            results[group][case_name] = {
                'emission': emission,
                'reduction': reduction_rate
            }
    
    return results


def plot_group_bars(group_data, temp_diff, save_pdf_path):
    """绘制分组水平堆叠柱状图
    Args:
        group_data: load_group_data 返回的 dict，已筛选对应拓展温度的case
        temp_diff: 拓展温度（1, 2, 3, 4）
        save_pdf_path: 保存路径
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
    
    # 根据拓展温度确定对应的case列表
    temp_case_mapping = {
        1: ['case5', 'case4', 'case3', 'case2', 'case1', 'ref'],
        2: ['case10', 'case9', 'case8', 'case7', 'case6', 'ref'],
        3: ['case15', 'case14', 'case13', 'case12', 'case11', 'ref'],
        4: ['case20', 'case19', 'case18', 'case17', 'case16', 'ref']
    }
    
    cases_for_temp = temp_case_mapping[temp_diff]
    
    # 组别顺序（从上到下）
    groups = ['CN', 'IN', 'US', 'Europe', 'Others']
    
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
    
    # Y 轴为组别（5个组）
    y_positions = [1, 2, 3, 4, 5]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(groups, fontsize=16)
    ax.tick_params(axis='y', width=0, length=0)
    
    # X 轴标签（非斜体）
    ax.set_xlabel('Carbon emission (GtCO2)', fontsize=18, labelpad=15)
    ax.set_ylabel('Group', fontsize=18, rotation=90, labelpad=20)
    
    # 转换因子：tCO2 转 GtCO2 (1 GtCO2 = 1e9 tCO2)
    conversion_factor = 1e9
    
    # 根据所有数据决定最大 X 范围（转换为 GtCO2）
    all_emissions = []
    for group in groups:
        for case in cases_for_temp:
            if case in group_data.get(group, {}):
                emission = group_data[group][case].get('emission', np.nan)
                if pd.notna(emission):
                    all_emissions.append(emission / conversion_factor)
    xmax = max(all_emissions) * 1.05 if all_emissions else 1.0
    ax.set_xlim(0, xmax)
    
    # 在同一组内，6 个工况柱状完全重叠（同一 Y 位置），后绘制的覆盖前面的
    bar_height = 0.5
    
    # 需要特殊处理的组别（欧洲和美国）
    special_groups = ['Europe', 'US']
    
    for row_idx, group in enumerate(groups):
        y = y_positions[row_idx]
        group_cases = group_data.get(group, {})
        
        # 存储需要标注的case信息（仅对欧洲和美国）
        annotation_cases = {} if group in special_groups else None
        
        # 反转顺序绘制，实现覆盖效果：ref 先绘制（底层），α5 最后绘制（顶层）
        for case in reversed(cases_for_temp):
            if case not in group_cases:
                continue
            emission = group_cases[case].get('emission', np.nan)
            reduction = group_cases[case].get('reduction', np.nan)
            if not pd.notna(emission):
                continue
            # 转换为 GtCO2
            emission_gt = emission / conversion_factor
            ckey = case_to_color_key(case)
            color = colors.get(ckey, '#CCCCCC')
            # 绘制：全部从 0 开始（left=0），在同一 Y 位置
            ax.barh(y=y, width=emission_gt, left=0.0, height=bar_height, color=color, edgecolor='none')
            
            # 记录需要标注的case（仅对欧洲和美国）
            if group in special_groups and case != 'ref' and pd.notna(reduction):
                try:
                    n = int(case.replace('case', ''))
                    alpha_idx = (n - 1) % 5 + 1  # 1..5 对应 α1..α5
                except Exception:
                    alpha_idx = None
                if alpha_idx in (1, 5):
                    annotation_cases[alpha_idx] = {
                        'emission': emission_gt,  # 已转换为 GtCO2
                        'reduction': reduction,
                        'y': y
                    }
            # 对于其他组别，使用原来的标注方式
            elif group not in special_groups and case != 'ref' and pd.notna(reduction):
                try:
                    n = int(case.replace('case', ''))
                    alpha_idx = (n - 1) % 5 + 1  # 1..5 对应 α1..α5
                except Exception:
                    alpha_idx = None
                if alpha_idx in (1, 5):
                    txt = f"-{reduction:.0f}%"
                    ax.text(emission_gt, y - bar_height * 0.55, txt, ha='center', va='top', fontsize=14)
        
        # 对于欧洲和美国，绘制带连接线的标注
        if group in special_groups and annotation_cases:
            if 1 in annotation_cases:  # α1（左边的）
                alpha1_data = annotation_cases[1]
                emission_val = alpha1_data['emission']
                reduction_val = alpha1_data['reduction']
                y_pos = alpha1_data['y']
                
                # 计算文本位置（柱形左侧，向右偏移1%的xmax）
                text_x = emission_val + xmax * 0.01  # 向右偏移1%的xmax
                text_y = y_pos - bar_height * 0.55
                
                # 柱形下边缘的y位置
                y_bottom = y_pos - bar_height / 2
                
                # 绘制连接线：从柱形右边缘的下侧边缘到文本位置
                ax.plot([emission_val, text_x], [y_bottom, text_y], 
                       color='black', linewidth=1.0, linestyle='-', zorder=10)
                
                # 绘制文本（左对齐）
                txt = f"-{reduction_val:.0f}%"
                ax.text(text_x, text_y, txt, ha='left', va='top', fontsize=14, zorder=11)
            
            if 5 in annotation_cases:  # α5（右边的）
                alpha5_data = annotation_cases[5]
                emission_val = alpha5_data['emission']
                reduction_val = alpha5_data['reduction']
                y_pos = alpha5_data['y']
                
                # 计算文本位置（柱形右侧，向左偏移1%的xmax）
                text_x = emission_val - xmax * 0.01  # 向左偏移1%的xmax
                text_y = y_pos - bar_height * 0.55
                
                # 柱形下边缘的y位置
                y_bottom = y_pos - bar_height / 2
                
                # 绘制连接线：从柱形右边缘的下侧边缘到文本位置
                ax.plot([emission_val, text_x], [y_bottom, text_y], 
                       color='black', linewidth=1.0, linestyle='-', zorder=10)
                
                # 绘制文本（右对齐）
                txt = f"-{reduction_val:.0f}%"
                ax.text(text_x, text_y, txt, ha='right', va='top', fontsize=14, zorder=11)
    
    # 竖向图例
    legend_labels = ['ref', r'α1', r'α2', r'α3', r'α4', r'α5']
    # 选择一行的 case 映射颜色：使用 diff=2℃ 的映射即可（与 draw_global_carbon_horizontal.py 一致）
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
    
    # 标题
    # ax.set_title(f'$T_{{expansion}}$ = {temp_diff}°C', fontsize=18)
    
    plt.tight_layout()
    plt.savefig(save_pdf_path, format='pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # 输入文件路径
    input_file = r"Z:\local_environment_creation\carbon_emission\2016-2020\group\group_summary_2016_2020_average.csv"
    
    # 输出目录
    output_dir = r"Z:\local_environment_creation\carbon_emission\2016-2020\group"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"未找到输入文件: {input_file}")
    else:
        print(f"正在处理: group_summary_2016_2020_average.csv …")
        group_data = load_group_data(input_file)
        
        # 为每个拓展温度绘制一张图
        for temp_diff in [1, 2, 3, 4]:
            print(f"正在绘制拓展温度 {temp_diff}℃ 的图表...")
            pdf_name = f'group_carbon_emission_T_expansion_{temp_diff}.pdf'
            save_pdf_path = os.path.join(output_dir, pdf_name)
            plot_group_bars(group_data, temp_diff, save_pdf_path)
            print(f"已保存: {save_pdf_path}")
        
        print("\n所有图表绘制完成！")

