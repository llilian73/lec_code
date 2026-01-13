import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为Arial，并处理摄氏度符号
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.style'] = 'normal'  # 非斜体
# 设置字体回退，确保摄氏度符号能正确显示
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']


def load_20_percent_data(csv_path):
    """读取20%节能率数据"""
    df = pd.read_csv(csv_path)
    return df


def plot_20_percent_line_chart(df, save_path):
    """绘制20%节能率点线图"""
    # 定义温度和case的对应关系
    temp_cases = [
        {'temp': 1, 'cases': ['case1', 'case2', 'case3', 'case4', 'case5']},
        {'temp': 2, 'cases': ['case6', 'case7', 'case8', 'case9', 'case10']},
        {'temp': 3, 'cases': ['case11', 'case12', 'case13', 'case14', 'case15']},
        {'temp': 4, 'cases': ['case16', 'case17', 'case18', 'case19', 'case20']}
    ]
    
    # 定义颜色
    colors = ['#F1C3C1', '#B7D5EC', '#FEDFB1', '#B9C3DC']
    
    # 定义标记样式
    markers = ['o', '^', 's', 'D']  # 圆形、三角形、方形、菱形
    
    # 定义α标签
    alpha_labels = ['α1', 'α2', 'α3', 'α4', 'α5']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 设置背景
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')
    
    # 绘制每条温度线
    for i, temp_data in enumerate(temp_cases):
        temp = temp_data['temp']
        cases = temp_data['cases']
        
        # 获取该温度下所有case的数据
        temp_values = []
        for case in cases:
            case_data = df[df['case'] == case]
            if not case_data.empty:
                temp_values.append(case_data['>20%'].iloc[0])
            else:
                temp_values.append(0)  # 如果数据不存在，设为0
        
        # 绘制点线图
        ax.plot(alpha_labels, temp_values, 
                color=colors[i], marker=markers[i], 
                linewidth=2.5, markersize=8, 
                label=f'$T_{{expansion}}$={temp}°C',  # 使用°C替代℃，expansion为下标
                markerfacecolor=colors[i], 
                markeredgecolor='black',
                markeredgewidth=1)
    
    # 设置坐标轴
    ax.set_ylabel('Proportion of countries (%)', fontsize=16, labelpad=15)
    ax.set_xlabel('α', fontsize=16, labelpad=15)
    
    # 设置Y轴范围
    ax.set_ylim(40, 105)
    ax.set_yticks(np.arange(40, 101, 10))
    
    # 设置X轴标签
    ax.set_xticks(range(len(alpha_labels)))
    ax.set_xticklabels(alpha_labels, fontsize=14)
    
    # 设置左右边距（控制α1和α5与边框的距离）
    # 方法1：使用margins设置相对边距
    ax.margins(x=0.1)  # 设置左右边距为10%，可以调整这个值
    
    # 方法2：使用set_xlim精确控制X轴范围
    # ax.set_xlim(-0.5, 4.5)  # α1在0位置，α5在4位置，可以调整边距
    
    # 设置Y轴标签
    ax.tick_params(axis='y', labelsize=14)
    
    # 添加图例
    ax.legend(loc='lower right', fontsize=12, frameon=False, 
              fancybox=True, shadow=True)
    
    # 添加网格
    # ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 设置标题
    # ax.set_title('Proportion of Countries with ≥15% Energy Reduction', fontsize=18, pad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 方法3：使用subplots_adjust进一步调整边距
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # 可以调整这些值
    
    # 保存图片
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"已保存图片: {save_path}")


def main():
    # 输入文件路径
    input_file = r"Z:\local_environment_creation\energy_consumption_gird\result\result\reduction_rate\20%\20_percent_reduction_summary.csv"
    
    # 输出目录
    output_dir = r"Z:\local_environment_creation\energy_consumption_gird\result\result\reduction_rate\20%"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return
    
    # 读取数据
    print("正在读取数据...")
    df = load_20_percent_data(input_file)
    print(f"读取了 {len(df)} 条数据")
    
    # 生成输出文件名
    output_filename = "20_percent_line_chart.pdf"
    output_path = os.path.join(output_dir, output_filename)
    
    # 绘制图表
    print("正在绘制图表...")
    plot_20_percent_line_chart(df, output_path)
    
    print("图表绘制完成！")


if __name__ == '__main__':
    main()
