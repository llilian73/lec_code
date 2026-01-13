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

def plot_horizontal_stacked_bars(heating_data, cooling_data, total_data, temp_expansion, cases, save_pdf_path):
    """绘制水平堆叠柱状图，每个拓展温度范围一张图
    temp_expansion: 拓展温度范围 (1-4)
    cases: 该温度范围对应的case列表，包含ref和α1-α4（不包含α5）
    """
    # 参考原版draw_global_data.py的颜色方案，每个工况不同颜色
    case_colors = {
        'ref': '#E5E5E5',     # 浅灰
        'case1': '#F1C3C1',   # 灰蓝
        'case2': '#B7D5EC',   # 浅蓝
        'case3': '#FEDFB1',   # 橙色
        'case4': '#B9C3DC',   # 黄色
        'case5': '#B2D3A4',   # 浅绿
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
    
    # 工况标签：从上至下依次为ref, α1, α2, α3, α4
    case_labels = ['ref', r'α1', r'α2', r'α3', r'α4']
    
    # 画布
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 轴样式
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')
    
    # 设置坐标轴标签
    ax.set_xlabel('Annual electricity demand(TWh)', fontsize=18, labelpad=15)
    ax.set_ylabel('', fontsize=18, labelpad=15)
    
    # 计算柱子位置和宽度
    n_bars = len(cases)  # ref + α1-α4 = 5
    bar_height = 0.6
    bar_spacing = 0.8
    
    # 设置横轴范围为0到5500
    xmax = 5500
    ax.set_xlim(0, xmax)
    
    # 绘制水平堆叠柱状图
    for bar_idx, (case, label) in enumerate(zip(cases, case_labels)):
        # 获取供暖、制冷、总需求数据
        heating_demand = heating_data.get(case, {}).get('demand', 0) if heating_data else 0
        cooling_demand = cooling_data.get(case, {}).get('demand', 0) if cooling_data else 0
        total_demand = total_data.get(case, {}).get('demand', 0) if total_data else 0
        heating_reduction = heating_data.get(case, {}).get('reduction', 0) if heating_data else 0
        cooling_reduction = cooling_data.get(case, {}).get('reduction', 0) if cooling_data else 0
        total_reduction = total_data.get(case, {}).get('reduction', 0) if total_data else 0
        
        if not pd.notna(total_demand) or total_demand == 0:
            continue
        
        # Y轴位置（从上至下）
        y_pos = (n_bars - 1 - bar_idx) * bar_spacing
        
        # 获取该case的颜色
        case_color = all_case_colors.get(case, '#CCCCCC')
        
        # 绘制供暖部分（左侧，深色）
        if pd.notna(heating_demand) and heating_demand > 0:
            # 供暖用深色（alpha=0.9），无边框
            ax.barh(y_pos, heating_demand, bar_height, 
                   left=0, color=case_color, edgecolor='none', 
                   alpha=0.9)
        
        # 绘制制冷部分（右侧，浅色）
        if pd.notna(cooling_demand) and cooling_demand > 0:
            # 制冷用浅色（alpha=0.6），无边框
            ax.barh(y_pos, cooling_demand, bar_height, 
                   left=heating_demand, color=case_color, edgecolor='none',
                   alpha=0.6)
        
        # 标注节能率（非ref工况）
        if case != 'ref':
            # 供暖部分节能率（中间位置）- 使用供暖的节能率
            if heating_demand > 0 and pd.notna(heating_reduction):
                ax.text(heating_demand * 0.5, y_pos, f"-{heating_reduction:.0f}%", 
                       ha='center', va='center', fontsize=12, color='black', weight='bold')
            # 制冷部分节能率（中间位置）- 使用制冷的节能率
            if cooling_demand > 0 and pd.notna(cooling_reduction):
                ax.text(heating_demand + cooling_demand * 0.5, y_pos, f"-{cooling_reduction:.0f}%", 
                       ha='center', va='center', fontsize=12, color='black', weight='bold')
            # 总需求节能率（柱子右侧）- 使用总需求的节能率
            if pd.notna(total_reduction):
                ax.text(total_demand + xmax * 0.01, y_pos, f"-{total_reduction:.0f}%", 
                       ha='left', va='center', fontsize=12, weight='bold')
    
    # 设置Y轴刻度（从上到下：ref, α1, α2, α3, α4）
    y_ticks = [(n_bars - 1 - i) * bar_spacing for i in range(n_bars)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(case_labels, fontsize=16)  # case_labels已经是ref, α1, α2, α3, α4的顺序
    
    # 添加图例 - 显示每个工况的颜色，供暖深色、制冷浅色（放在右下角）
    legend_x = 0.8  # 右下角X位置
    legend_y_bottom = 0.08  # 底部起始位置
    legend_y_top = legend_y_bottom + 0.2  # 顶部位置（用于标题）
    
    # 为每个工况创建两个图例项：供暖（深色）和制冷（浅色）
    alpha_colors = []
    for case in cases:
        alpha_colors.append(all_case_colors.get(case, '#CCCCCC'))
    
    # 色块参数
    heating_x = legend_x + 0.02
    heating_width = 0.03  # 方格宽度
    spacing = 0.02  # heating和cooling之间的间距（增加间距）
    cooling_x = heating_x + heating_width + spacing
    cooling_width = 0.03  # 方格宽度
    
    # 添加Heating和Cooling标题（在顶部）
    heating_title_x = heating_x + heating_width / 2  # 第一个色块中心位置
    cooling_title_x = cooling_x + cooling_width / 2  # 第二个色块中心位置
    ax.text(heating_title_x, legend_y_top, 'Heating', transform=ax.transAxes, 
            ha='center', va='center', fontsize=12, weight='bold', zorder=12)
    ax.text(cooling_title_x, legend_y_top, 'Cooling', transform=ax.transAxes, 
            ha='center', va='center', fontsize=12, weight='bold', zorder=12)
    
    # 绘制图例内容（每个工况一行，不显示标签，从下往上）
    # 增加标题和色块之间的距离
    for i, color in enumerate(alpha_colors):
        row_y = legend_y_top - 0.06 - i * 0.04  # 从顶部往下排列
        
        # 供暖色块（深色）
        ax.add_patch(plt.Rectangle((heating_x, row_y-0.02), heating_width, 0.04, 
                                  transform=ax.transAxes, facecolor=color, edgecolor='none', 
                                  alpha=0.9, zorder=11))
        
        # 制冷色块（浅色）
        ax.add_patch(plt.Rectangle((cooling_x, row_y-0.02), cooling_width, 0.04, 
                                  transform=ax.transAxes, facecolor=color, edgecolor='none', 
                                  alpha=0.6, zorder=11))
    
    # 网格
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.tick_params(axis='x', labelsize=14)
    
    plt.tight_layout()
    plt.savefig(save_pdf_path, format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    base_dir = r"Z:\local_environment_creation\energy_consumption_gird\result\result\global_total_energy_demand"
    sub_dir = os.path.join(base_dir, "horizontal")  # 在 base_dir 下创建一个子文件夹
    os.makedirs(sub_dir, exist_ok=True)          # 若不存在则自动创建
    # 处理global_total_energy_demand目录下的文件
    file_prefixes = [
        'global_per_capita',
        'global_total'
    ]
    
    # 分组定义：每组包含ref, α1, α2, α3, α4（不包含α5）
    groups = [
        {'temp': 1, 'cases': ['ref', 'case1', 'case2', 'case3', 'case4']},
        {'temp': 2, 'cases': ['ref', 'case6', 'case7', 'case8', 'case9']},
        {'temp': 3, 'cases': ['ref', 'case11', 'case12', 'case13', 'case14']},
        {'temp': 4, 'cases': ['ref', 'case16', 'case17', 'case18', 'case19']},
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
        
        # 为每个拓展温度范围绘制一张水平堆叠柱状图
        for group in groups:
            temp_expansion = group['temp']
            cases = group['cases']
            pdf_name = f'{prefix}_temp{temp_expansion}_stacked_chart.pdf'
            save_pdf_path = os.path.join(sub_dir, pdf_name)
            plot_horizontal_stacked_bars(heating_data, cooling_data, total_data, 
                                        temp_expansion, cases, save_pdf_path)
            print(f"已保存: {save_pdf_path}")

