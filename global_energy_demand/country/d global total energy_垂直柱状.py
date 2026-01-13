"""
全球能耗垂直柱状图绘制工具

功能概述：
本工具用于绘制全球建筑能耗的垂直柱状图，展示不同节能案例（case1-case9）相对于参考案例的能耗减少效果。图表采用叠加柱状图的形式，同时显示原始需求和减少后的需求。

输入数据：
1. 全球能耗汇总文件：
   - global_total_total_energy.csv：全球总能耗数据
   - global_total_cooling_energy.csv：全球制冷能耗数据
   - global_total_heating_energy.csv：全球供暖能耗数据
   - 文件路径：results/global/
   - 数据格式：包含ref和case1-case9的能耗值（TWh）

2. 图表配置：
   - 能耗类型：总能耗、制冷能耗、供暖能耗
   - 案例范围：case1到case9
   - 颜色方案：橙色（总能耗）、蓝色（制冷）、珊瑚色（供暖）

主要功能：
1. 数据读取和处理：
   - 读取三种能耗类型的CSV文件
   - 提取ref案例的原始值和case1-case9的减少值
   - 计算各案例的节能百分比

2. 垂直柱状图绘制：
   - 创建叠加柱状图：灰色背景条（原始需求）+ 彩色条（减少后需求）
   - 显示三种能耗类型：总能耗、制冷能耗、供暖能耗
   - 添加百分比标注和箭头指示

3. 图表美化：
   - 设置合适的图表尺寸（6x5）
   - 配置颜色方案和标签
   - 设置Y轴范围和刻度间隔
   - 添加标题和轴标签

4. 批量输出：
   - 为每个case（case1-case9）生成单独的PDF文件
   - 文件命名格式：global_energy_demand_case{num}_vertical.pdf
   - 输出目录：results/global/global_energy_demand/

输出结果：
1. PDF格式的垂直柱状图：
   - 文件名：global_energy_demand_case{1-9}_vertical.pdf
   - 图表内容：三种能耗类型的对比柱状图
   - 包含原始需求、减少后需求、节能百分比

2. 图表元素：
   - 灰色背景条：显示ref案例的原始能耗
   - 彩色叠加条：显示各case的减少后能耗
   - 红色百分比标注：显示节能百分比
   - 黑色箭头：指示从原始值到减少值的减少量
"""

import matplotlib.pyplot as plt
import os
import pandas as pd

# 读取数据
base_path = r"D:\workstation\energy_comsuption\results\global"
energy_types = ['total', 'cooling', 'heating']

# 类别名称
categories = ['Overall\ndemand', 'Cooling\ndemand', 'Heating\ndemand']

# 颜色设置
colors = ['darkorange', 'steelblue', 'lightcoral']

# 确保输出目录存在
output_dir = r"D:\workstation\energy_comsuption\results\global\global_energy_demand"
os.makedirs(output_dir, exist_ok=True)

# 为每个case绘制图片
for case_num in range(1, 10):  # case1到case9
    case_name = f'case{case_num}'
    original_values = []
    reduced_values = []
    
    # 读取数据
    for energy_type in energy_types:
        file_path = os.path.join(base_path, f"global_total_{energy_type}_energy.csv")
        df = pd.read_csv(file_path, index_col=0)
        original_values.append(df.loc['ref', 'demand_sum'])
        reduced_values.append(df.loc[case_name, 'demand_sum'])
    
    # 计算减少比例
    reductions = [100 * (1 - r / o) for o, r in zip(original_values, reduced_values)]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 5))  # 宽度和高度均为5
    x_pos = list(range(len(categories)))
    
    # 背景灰条（原始值）
    ax.bar(x_pos, original_values, color='lightgray', width=0.6, label='Original demand')
    
    # 叠加彩色条（减少后的值）
    ax.bar(x_pos, reduced_values, color=colors, width=0.6, label='Reduced demand')
    
    # 文字和箭头
    for i, (orig, red, reduction) in enumerate(zip(original_values, reduced_values, reductions)):
        # 百分比标注
        ax.text(i, orig + 300, f"-{reduction:.1f}%", ha='center', va='bottom', fontsize=12, color='red', fontweight='bold')
        # 箭头指向原始值
        ax.annotate('', xy=(i, red + 100), xytext=(i, orig - 100),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # 设置x轴标签
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 12000)
    ax.set_yticks(range(0, 12001, 4000))  # 纵坐标标尺间隔4000
    ax.set_title("Annual electricity demand\nwith local environment creation (TWh)", fontsize=14)
    
    # # 去掉边框
    # for spine in ['top', 'right']:
    #     ax.spines[spine].set_visible(False)
    
    # 保存图片
    output_path = os.path.join(output_dir, f"global_energy_demand_{case_name}_vertical.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"图片已保存至: {output_path}")
    
    # 关闭图形，释放内存
    plt.close()
