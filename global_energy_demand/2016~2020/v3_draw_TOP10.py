import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fm

# 设置全局字体为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.style'] = 'normal'  # 非斜体


def load_top10_data(csv_path):
    """读取TOP10汇总CSV文件，返回按国家分组的数据"""
    df = pd.read_csv(csv_path, keep_default_na=False, dtype={'group': str})
    df['group'] = df['group'].astype(str).str.strip().str.upper()
    
    # 按国家分组
    country_data = {}
    for country in df['group'].unique():
        country_df = df[df['group'] == country]
        country_data[country] = country_df.to_dict('records')
    
    return country_data


def get_case_for_temp(temp_expansion):
    """根据温度拓展范围返回对应的cases列表"""
    groups = {
        1: ['ref', 'case1', 'case2', 'case3', 'case4'],
        2: ['ref', 'case6', 'case7', 'case8', 'case9'],
        3: ['ref', 'case11', 'case12', 'case13', 'case14'],
        4: ['ref', 'case16', 'case17', 'case18', 'case19'],
    }
    return groups.get(temp_expansion, [])


def extract_case_data(country_records, case_name):
    """从国家记录中提取指定case的数据"""
    for record in country_records:
        if record.get('index') == case_name:
            return {
                'heating_demand': pd.to_numeric(record.get('heating_demand_sum(TWh)', np.nan), errors='coerce'),
                'cooling_demand': pd.to_numeric(record.get('cooling_demand_sum(TWh)', np.nan), errors='coerce'),
                'total_demand': pd.to_numeric(record.get('total_demand_sum(TWh)', np.nan), errors='coerce'),
                'heating_reduction': pd.to_numeric(record.get('heating_demand_reduction(%)', np.nan), errors='coerce'),
                'cooling_reduction': pd.to_numeric(record.get('cooling_demand_reduction(%)', np.nan), errors='coerce'),
                'total_reduction': pd.to_numeric(record.get('total_demand_reduction(%)', np.nan), errors='coerce'),
            }
    return {
        'heating_demand': np.nan,
        'cooling_demand': np.nan,
        'total_demand': np.nan,
        'heating_reduction': np.nan,
        'cooling_reduction': np.nan,
        'total_reduction': np.nan,
    }


def calculate_xmax(cases_data):
    """根据数据计算横轴最大值"""
    max_value = 0
    for case_data in cases_data:
        total = case_data.get('total_demand', 0)
        if pd.notna(total) and total > max_value:
            max_value = total
    # 增加10%的余量，并向上取整到合适的值
    if max_value > 0:
        xmax = max_value * 1.1
        # 根据数值大小，向上取整到合适的刻度
        if xmax < 100:
            xmax = np.ceil(xmax / 10) * 10
        elif xmax < 1000:
            xmax = np.ceil(xmax / 50) * 50
        else:
            xmax = np.ceil(xmax / 100) * 100
        return max(xmax, 100)  # 至少100
    return 1000  # 默认值


def sort_countries_by_total_demand(country_data):
    """按ref case的总能耗降序排序国家
    Args:
        country_data: 国家数据字典 {country: country_records}
    Returns:
        排序后的国家列表（按总能耗从高到低）
    """
    country_totals = []
    for country, country_records in country_data.items():
        ref_data = extract_case_data(country_records, 'ref')
        total_demand = ref_data.get('total_demand', 0)
        if pd.notna(total_demand):
            country_totals.append((country, total_demand))
        else:
            country_totals.append((country, 0))
    
    # 按总能耗降序排序
    country_totals.sort(key=lambda x: x[1], reverse=True)
    return [country for country, _ in country_totals]


def plot_combined_figure(country_data, sorted_countries, temp_expansion, cases, save_pdf_path):
    """绘制2×5布局的拼接图，包含10个国家的子图
    Args:
        country_data: 所有国家数据字典
        sorted_countries: 排序后的国家列表（总能耗从高到低）
        temp_expansion: 温度拓展范围 (1-4)
        cases: 该温度范围对应的case列表
        save_pdf_path: 保存路径
    """
    # 创建2×5的子图
    fig, axes = plt.subplots(2, 5, figsize=(35, 14))
    axes = axes.flatten()  # 展平为1D数组，方便索引
    
    # 计算所有国家的最大xmax（用于统一横轴范围）
    all_xmax = []
    for country in sorted_countries[:10]:  # 只处理前10个国家
        country_records = country_data.get(country, [])
        cases_data = []
        for case in cases:
            case_data = extract_case_data(country_records, case)
            cases_data.append(case_data)
        xmax = calculate_xmax(cases_data)
        all_xmax.append(xmax)
    unified_xmax = max(all_xmax) if all_xmax else 1000
    
    # 为每个国家绘制子图（从左到右，从上到下）
    for idx, country in enumerate(sorted_countries[:10]):
        ax = axes[idx]
        country_records = country_data.get(country, [])
        
        # 只在最后一个子图（右下角）显示图例
        show_legend = (idx == 9)
        plot_single_subplot(ax, country_records, country, cases, 
                           xmax=unified_xmax, show_legend=show_legend, font_scale=0.7)
    
    plt.tight_layout()
    plt.savefig(save_pdf_path, format='pdf', bbox_inches='tight')
    plt.close()


def get_case_colors():
    """获取case颜色映射"""
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
    return all_case_colors


def plot_single_subplot(ax, country_records, country, cases, xmax=None, show_legend=False, font_scale=1.0):
    """在给定的ax上绘制单个国家的水平堆叠柱状图
    Args:
        ax: matplotlib轴对象
        country_records: 该国家的所有记录列表
        country: 国家代码
        cases: 该温度范围对应的case列表，包含ref和α1-α4
        xmax: 横轴最大值（如果为None则自动计算）
        show_legend: 是否显示图例
        font_scale: 字体缩放因子（用于子图调整）
    """
    all_case_colors = get_case_colors()
    
    # 工况标签：从上至下依次为ref, α1, α2, α3, α4
    case_labels = ['ref', r'α1', r'α2', r'α3', r'α4']
    
    # 提取所有cases的数据
    cases_data = []
    for case in cases:
        case_data = extract_case_data(country_records, case)
        cases_data.append(case_data)
    
    # 计算横轴范围
    if xmax is None:
        xmax = calculate_xmax(cases_data)
    
    # 轴样式
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(2.0 * font_scale)
        spine.set_color('black')
    
    # 设置坐标轴标签
    ax.set_xlabel('Annual electricity demand(TWh)', fontsize=14 * font_scale, labelpad=10 * font_scale)
    ax.set_title(country, fontsize=14 * font_scale, weight='bold', pad=5 * font_scale)
    
    # 计算柱子位置和宽度
    n_bars = len(cases)  # ref + α1-α4 = 5
    bar_height = 0.6 * font_scale
    bar_spacing = 0.8 * font_scale
    
    # 设置横轴范围
    ax.set_xlim(0, xmax)
    
    # 绘制水平堆叠柱状图
    for bar_idx, (case, label) in enumerate(zip(cases, case_labels)):
        case_data = cases_data[bar_idx]
        
        # 获取供暖、制冷、总需求数据
        heating_demand = case_data.get('heating_demand', 0) if pd.notna(case_data.get('heating_demand', 0)) else 0
        cooling_demand = case_data.get('cooling_demand', 0) if pd.notna(case_data.get('cooling_demand', 0)) else 0
        total_demand = case_data.get('total_demand', 0) if pd.notna(case_data.get('total_demand', 0)) else 0
        heating_reduction = case_data.get('heating_reduction', 0) if pd.notna(case_data.get('heating_reduction', 0)) else 0
        cooling_reduction = case_data.get('cooling_reduction', 0) if pd.notna(case_data.get('cooling_reduction', 0)) else 0
        total_reduction = case_data.get('total_reduction', 0) if pd.notna(case_data.get('total_reduction', 0)) else 0
        
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
                       ha='center', va='center', fontsize=10 * font_scale, color='black', weight='bold')
            # 制冷部分节能率（中间位置）- 使用制冷的节能率
            if cooling_demand > 0 and pd.notna(cooling_reduction):
                ax.text(heating_demand + cooling_demand * 0.5, y_pos, f"-{cooling_reduction:.0f}%", 
                       ha='center', va='center', fontsize=10 * font_scale, color='black', weight='bold')
            # 总需求节能率（柱子右侧）- 使用总需求的节能率
            if pd.notna(total_reduction):
                ax.text(total_demand + xmax * 0.01, y_pos, f"-{total_reduction:.0f}%", 
                       ha='left', va='center', fontsize=10 * font_scale, weight='bold')
    
    # 设置Y轴刻度（从上到下：ref, α1, α2, α3, α4）
    y_ticks = [(n_bars - 1 - i) * bar_spacing for i in range(n_bars)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(case_labels, fontsize=12 * font_scale)  # case_labels已经是ref, α1, α2, α3, α4的顺序
    
    # 添加图例 - 只在最后一个子图显示
    if show_legend:
        legend_x = 0.75  # 右下角X位置
        legend_y_bottom = 0.08  # 底部起始位置
        legend_y_top = legend_y_bottom + 0.2  # 顶部位置（用于标题）
        
        # 为每个工况创建两个图例项：供暖（深色）和制冷（浅色）
        alpha_colors = []
        for case in cases:
            alpha_colors.append(all_case_colors.get(case, '#CCCCCC'))
        
        # 色块参数
        heating_x = legend_x + 0.02
        heating_width = 0.025  # 方格宽度
        spacing = 0.015  # heating和cooling之间的间距
        cooling_x = heating_x + heating_width + spacing
        cooling_width = 0.025  # 方格宽度
        
        # 添加Heating和Cooling标题（在顶部）
        heating_title_x = heating_x + heating_width / 2  # 第一个色块中心位置
        cooling_title_x = cooling_x + cooling_width / 2  # 第二个色块中心位置
        ax.text(heating_title_x, legend_y_top, 'Heating', transform=ax.transAxes, 
                ha='center', va='center', fontsize=10 * font_scale, weight='bold', zorder=12)
        ax.text(cooling_title_x, legend_y_top, 'Cooling', transform=ax.transAxes, 
                ha='center', va='center', fontsize=10 * font_scale, weight='bold', zorder=12)
        
        # 绘制图例内容（每个工况一行，不显示标签，从下往上）
        for i, color in enumerate(alpha_colors):
            row_y = legend_y_top - 0.06 - i * 0.035  # 从顶部往下排列
            
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
    ax.tick_params(axis='x', labelsize=11 * font_scale)


def plot_horizontal_stacked_bars(country_records, country, temp_expansion, cases, save_pdf_path):
    """绘制水平堆叠柱状图，每个拓展温度范围一张图（保留用于单独绘制）
    Args:
        country_records: 该国家的所有记录列表
        country: 国家代码
        temp_expansion: 拓展温度范围 (1-4)
        cases: 该温度范围对应的case列表，包含ref和α1-α4
        save_pdf_path: 保存路径
    """
    # 画布
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_single_subplot(ax, country_records, country, cases, show_legend=True, font_scale=1.0)
    plt.tight_layout()
    plt.savefig(save_pdf_path, format='pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # 输入文件路径
    input_csv = r"Z:\local_environment_creation\energy_consumption_gird\result\result\top10_horizontal\top10_summary_2016_2020_average.csv"
    
    # 输出目录
    base_root = r"Z:\local_environment_creation\energy_consumption_gird\result\result"
    output_dir = os.path.join(base_root, "top10_horizontal")
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    print("正在读取数据...")
    country_data = load_top10_data(input_csv)
    print(f"找到 {len(country_data)} 个国家")
    
    # 按总能耗排序国家（使用ref case的总需求）
    print("正在按总能耗排序国家...")
    sorted_countries = sort_countries_by_total_demand(country_data)
    print(f"排序后的国家顺序（前10个）: {sorted_countries[:10]}")
    
    # 温度拓展范围定义
    temp_expansions = [1, 2, 3, 4]
    
    # 为每个温度拓展范围生成一张2×5的拼接图
    for temp_expansion in temp_expansions:
        print(f"\n处理温度拓展范围 temp{temp_expansion}...")
        cases = get_case_for_temp(temp_expansion)
        
        # 生成输出文件名
        pdf_name = f'temp{temp_expansion}_combined_2x5.pdf'
        save_pdf_path = os.path.join(output_dir, pdf_name)
        
        # 绘制拼接图
        plot_combined_figure(country_data, sorted_countries, temp_expansion, cases, save_pdf_path)
        print(f"  已保存: {save_pdf_path}")
    
    print(f"\n所有图片处理完成！输出目录: {output_dir}")
