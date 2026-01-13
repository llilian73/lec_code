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


def load_combined_data(base_dir, file_prefix):
    """同时读取供暖、制冷、总需求数据并计算平均值"""
    # 读取三个文件
    heating_file = f'{file_prefix}_heating_energy.csv'
    cooling_file = f'{file_prefix}_cooling_energy.csv'
    total_file = f'{file_prefix}_total_energy.csv'
    
    heating_path = os.path.join(base_dir, heating_file)
    cooling_path = os.path.join(base_dir, cooling_file)
    total_path = os.path.join(base_dir, total_file)
    
    if not all(os.path.exists(p) for p in [heating_path, cooling_path, total_path]):
        print(f"未找到完整的文件组合: {file_prefix}")
        return None, None, None
    
    heating_data = load_and_average(heating_path)
    cooling_data = load_and_average(cooling_path)
    total_data = load_and_average(total_path)
    
    return heating_data, cooling_data, total_data

def plot_stacked_bars(heating_data, cooling_data, total_data, title, save_pdf_path):
    """绘制堆叠柱状图，使用真实的供暖和制冷数据"""
    # 参考原版draw_global_data.py的颜色方案，每个工况不同颜色
    case_colors = {
        'ref': '#E5E5E5',     # 浅灰
        'case1': '#6C7F93',   # 灰蓝
        'case2': '#7FA1D5',   # 浅蓝
        'case3': '#E79B6B',   # 橙色
        'case4': '#F2C757',   # 黄色
        'case5': '#98C279',   # 浅绿
    }
    
    # 生成所有case的颜色映射
    all_case_colors = {}
    for i in range(1, 21):
        case_name = f'case{i}'
        # case1-5, case6-10, case11-15, case16-20 分别对应 α1-α5
        alpha_idx = (i - 1) % 5 + 1
        if alpha_idx == 1:
            all_case_colors[case_name] = case_colors['case1']
        elif alpha_idx == 2:
            all_case_colors[case_name] = case_colors['case2']
        elif alpha_idx == 3:
            all_case_colors[case_name] = case_colors['case3']
        elif alpha_idx == 4:
            all_case_colors[case_name] = case_colors['case4']
        elif alpha_idx == 5:
            all_case_colors[case_name] = case_colors['case5']
    all_case_colors['ref'] = case_colors['ref']
    
    # 分组定义：每组包含ref, α1, α2, α3, α4, α5
    groups = [
        {'temp': 1, 'cases': ['ref', 'case1', 'case2', 'case3', 'case4', 'case5']},
        {'temp': 2, 'cases': ['ref', 'case6', 'case7', 'case8', 'case9', 'case10']},
        {'temp': 3, 'cases': ['ref', 'case11', 'case12', 'case13', 'case14', 'case15']},
        {'temp': 4, 'cases': ['ref', 'case16', 'case17', 'case18', 'case19', 'case20']},
    ]
    
    # 画布
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 轴样式
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')
    
    # 设置坐标轴标签
    ax.set_xlabel(r'$T_{expansion}$', fontsize=18, labelpad=15)
    ax.set_ylabel('Annual electricity demand(TWh)', fontsize=18, labelpad=15)
    
    # 计算柱子位置和宽度
    n_groups = len(groups)
    n_bars_per_group = 6  # ref + α1-α5
    bar_width = 0.12
    group_width = n_bars_per_group * bar_width + 0.1  # 组间距
    
    # 计算最大需求值
    all_demands = []
    for data in [heating_data, cooling_data, total_data]:
        if data:
            all_demands.extend([v['demand'] for v in data.values() if pd.notna(v['demand'])])
    ymax = max(all_demands) * 1.2 if all_demands else 1000
    ax.set_ylim(0, ymax)
    
    # 绘制分组柱状图
    for group_idx, group in enumerate(groups):
        x_center = group_idx * group_width
        
        for bar_idx, case in enumerate(group['cases']):
            # 获取供暖、制冷、总需求数据
            heating_demand = heating_data.get(case, {}).get('demand', 0) if heating_data else 0
            cooling_demand = cooling_data.get(case, {}).get('demand', 0) if cooling_data else 0
            total_demand = total_data.get(case, {}).get('demand', 0) if total_data else 0
            heating_reduction = heating_data.get(case, {}).get('reduction', 0) if heating_data else 0
            cooling_reduction = cooling_data.get(case, {}).get('reduction', 0) if cooling_data else 0
            total_reduction = total_data.get(case, {}).get('reduction', 0) if total_data else 0
            
            if not pd.notna(total_demand) or total_demand == 0:
                continue
                
            x_pos = x_center + bar_idx * bar_width
            
            # 获取该case的颜色
            case_color = all_case_colors.get(case, '#CCCCCC')
            
            # 绘制供暖部分（底层，深色）
            if pd.notna(heating_demand) and heating_demand > 0:
                # 供暖用深色（alpha=0.9），无边框
                ax.bar(x_pos, heating_demand, bar_width, 
                       color=case_color, edgecolor='none', 
                       alpha=0.9)
            
            # 绘制制冷部分（顶层，浅色）
            if pd.notna(cooling_demand) and cooling_demand > 0:
                # 制冷用浅色（alpha=0.6），无边框
                ax.bar(x_pos, cooling_demand, bar_width, 
                       bottom=heating_demand, color=case_color, edgecolor='none',
                       alpha=0.6)
            
            # 标注节能率（非ref工况）
            if case != 'ref':
                # 供暖部分节能率（中间位置）- 使用供暖的节能率
                if heating_demand > 0 and pd.notna(heating_reduction):
                    ax.text(x_pos, heating_demand * 0.5, f"-{heating_reduction:.0f}%", 
                           ha='center', va='center', fontsize=10, color='white', weight='bold')
                # 制冷部分节能率（中间位置）- 使用制冷的节能率
                if cooling_demand > 0 and pd.notna(cooling_reduction):
                    ax.text(x_pos, heating_demand + cooling_demand * 0.5, f"-{cooling_reduction:.0f}%", 
                           ha='center', va='center', fontsize=10, color='white', weight='bold')
                # 总需求节能率（柱子顶部）- 使用总需求的节能率
                if pd.notna(total_reduction):
                    ax.text(x_pos, total_demand + ymax * 0.01, f"-{total_reduction:.0f}%", 
                           ha='center', va='bottom', fontsize=12, weight='bold')
    
    # 设置X轴刻度 - 刻度位置在每组第三个柱子（α3）的右侧
    x_ticks = [i * group_width + 2 * bar_width + bar_width/2 for i in range(n_groups)]
    x_labels = ['1', '2', '3', '4']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=16)
    
    # 移除分组标签（不需要标注ref, α1, α2, α3, α4, α5）
    
    # 添加图例 - 显示工况颜色，供暖深色、制冷浅色
    legend_elements = []
    legend_labels = []
    
    # 为每个工况创建两个图例项：供暖（深色）和制冷（浅色）
    alpha_labels = ['ref', r'α1', r'α2', r'α3', r'α4', r'α5']
    alpha_colors = [case_colors['ref'], case_colors['case1'], case_colors['case2'], 
                   case_colors['case3'], case_colors['case4'], case_colors['case5']]
    
    # 创建自定义图例 - 6行格式，每行一个工况
    # 在图的右上角绘制图例
    legend_x = 0.85
    legend_y = 0.95
    legend_width = 0.2
    legend_height = 0.4
    
    # 取消图例背景
    
    # 绘制图例内容
    for i, (label, color) in enumerate(zip(alpha_labels, alpha_colors)):
        row_y = legend_y - 0.05 - i * 0.04  # 行间距缩小为原来的一半
        
        # 供暖色块（深色）
        heating_x = legend_x + 0.02
        heating_width = 0.03  # 方格宽度缩小为原来的一半
        ax.add_patch(plt.Rectangle((heating_x, row_y-0.02), heating_width, 0.04, 
                                  transform=ax.transAxes, facecolor=color, edgecolor='none', 
                                  alpha=0.9, zorder=11))
        
        # 制冷色块（浅色）
        cooling_x = heating_x + heating_width + 0.01
        cooling_width = 0.03  # 方格宽度缩小为原来的一半
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
    
    # 网格
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.tick_params(axis='y', labelsize=14)
    
    plt.tight_layout()
    plt.savefig(save_pdf_path, format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    base_dir = r"Z:\local_environment_creation\energy_consumption\2016-2020result\global_total_energy_demand"
    sub_dir = os.path.join(base_dir, "vertical")  # 在 base_dir 下创建一个子文件夹
    os.makedirs(sub_dir, exist_ok=True)          # 若不存在则自动创建
    # 处理global_total_energy_demand目录下的文件
    file_prefixes = [
        'global_per_capita',
        'global_total'
    ]
    
    for prefix in file_prefixes:
        print(f"正在处理: {prefix} …")
        
        # 同时读取供暖、制冷、总需求数据
        heating_data, cooling_data, total_data = load_combined_data(base_dir, prefix)
        
        if heating_data is None:
            continue
            
        # 导出平均值CSV
        export_order = ['ref'] + [f'case{i}' for i in range(1, 21)]
        
        # 导出总需求平均值
        if total_data:
            avg_rows = []
            for case_name in export_order:
                vals = total_data.get(case_name, {'demand': np.nan, 'reduction': np.nan})
                avg_rows.append({
                    'index': case_name,
                    'demand_sum_avg': vals.get('demand', np.nan),
                    'reduction_percentage_avg': vals.get('reduction', np.nan)
                })
            avg_df = pd.DataFrame(avg_rows)
            avg_csv_name = f'{prefix}_total_energy_average.csv'
            avg_csv_path = os.path.join(sub_dir, avg_csv_name)
            avg_df.to_csv(avg_csv_path, index=False)
            print(f"已保存: {avg_csv_path}")
        
        # 导出供暖平均值
        if heating_data:
            avg_rows = []
            for case_name in export_order:
                vals = heating_data.get(case_name, {'demand': np.nan, 'reduction': np.nan})
                avg_rows.append({
                    'index': case_name,
                    'demand_sum_avg': vals.get('demand', np.nan),
                    'reduction_percentage_avg': vals.get('reduction', np.nan)
                })
            avg_df = pd.DataFrame(avg_rows)
            avg_csv_name = f'{prefix}_heating_energy_average.csv'
            avg_csv_path = os.path.join(sub_dir, avg_csv_name)
            avg_df.to_csv(avg_csv_path, index=False)
            print(f"已保存: {avg_csv_path}")
        
        # 导出制冷平均值
        if cooling_data:
            avg_rows = []
            for case_name in export_order:
                vals = cooling_data.get(case_name, {'demand': np.nan, 'reduction': np.nan})
                avg_rows.append({
                    'index': case_name,
                    'demand_sum_avg': vals.get('demand', np.nan),
                    'reduction_percentage_avg': vals.get('reduction', np.nan)
                })
            avg_df = pd.DataFrame(avg_rows)
            avg_csv_name = f'{prefix}_cooling_energy_average.csv'
            avg_csv_path = os.path.join(sub_dir, avg_csv_name)
            avg_df.to_csv(avg_csv_path, index=False)
            print(f"已保存: {avg_csv_path}")
        
        # 绘制堆叠柱状图
        pdf_name = f'{prefix}_stacked_chart.pdf'
        save_pdf_path = os.path.join(sub_dir, pdf_name)
        plot_stacked_bars(heating_data, cooling_data, total_data, title=prefix, save_pdf_path=save_pdf_path)
        print(f"已保存: {save_pdf_path}")

