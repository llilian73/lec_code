"""
全球建筑能耗地图绘制工具（非校准版本 - 调试版）

功能概述：
本工具用于绘制全球建筑能耗的世界地图，展示不同节能案例（case1-case9）的能耗差值和节能比例。该工具只使用global文件夹中的原始数据，不进行校准处理。包含详细的调试功能，特别用于检查特定国家（如ER-厄立特里亚）的数据完整性。

输入数据：
1. 各国能耗汇总文件：
   - 路径：results/global/{continent}/summary_p/{region}_2019_summary_p_results.csv
   - 包含各国人均能耗数据（kWh/person）
   - 涵盖ref、case1-case9共10种案例

2. 世界地图文件：
   - 路径：shapefiles/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp
   - 用于地理可视化

3. 地区分类：
   - 按大洲分类：Africa, Asia, Europe, North America, Oceania, South America
   - 每个大洲下包含多个国家的能耗数据

主要功能：
1. 数据收集和处理：
   - 自动发现所有可用的地区代码
   - 收集各地区的能耗汇总文件路径
   - 处理case1到case9的所有节能案例

2. 能耗差值地图绘制：
   - 计算ref案例与各case案例之间的能耗差值
   - 生成总能耗、供暖能耗、制冷能耗的差值世界地图
   - 差值 = ref能耗 - case能耗（正值表示节能效果）

3. 节能比例地图绘制：
   - 直接使用case案例中的节能百分比数据
   - 生成总能耗、供暖能耗、制冷能耗的节能比例世界地图
   - 显示各地区的节能效果百分比（0-100%）

4. CSV数据导出：
   - 将各案例的能耗差值和节能比例保存为CSV文件
   - 文件名格式：case{num}_summary.csv
   - 包含country、difference、reduction三列数据

5. 调试功能（特有）：
   - 检查特定国家（ER-厄立特里亚）数据的存在性
   - 提供详细的数据统计信息
   - 显示数据范围和排名信息
   - 输出调试信息到控制台

输出结果：
1. PDF格式的世界地图文件：
   - 差值地图：{data_type}_demand_difference_map_case{num}.pdf
   - 比例地图：{data_type}_demand_reduction_map_case{num}.pdf
   - data_type包括：total、heating、cooling

2. CSV格式的汇总数据：
   - case1_summary.csv 到 case9_summary.csv
   - 包含各国家的能耗差值和节能比例

3. 调试信息输出：
   - 控制台输出详细的调试信息
   - ER数据存在性检查结果
   - 数据统计和排名信息

4. 输出目录结构：
   - figure maps/difference/：能耗差值地图
   - figure maps/reduction/：节能比例地图
   - figure maps/：CSV汇总文件

计算特点：
- 原始数据：只使用global文件夹中的原始数据，不进行校准
- 全面覆盖：包含全球所有可用的国家数据
- 多案例对比：支持case1-case9的9种节能案例
- 双重分析：同时生成差值地图和比例地图
- 调试增强：包含详细的数据检查和统计功能

调试功能详解：
1. ER数据检查：专门检查厄立特里亚（ER）的数据是否存在
2. 数据统计：显示每个case包含的国家数量
3. 数据范围：显示能耗数据的最大值和最小值
4. 排名分析：显示ER数据在所有国家中的排名位置
5. 详细日志：输出每个处理步骤的详细信息

数据流程：
1. 扫描各大洲目录，收集所有可用的地区代码
2. 为每个case（case1-case9）调用地图绘制函数
3. 执行调试检查，输出详细的数据统计信息
4. 生成能耗差值地图和节能比例地图
5. 导出CSV格式的汇总数据
6. 保存所有结果到指定目录

适用场景：
- 开发和调试阶段：需要详细检查数据完整性
- 问题排查：当特定国家数据出现问题时
- 数据验证：确保所有国家数据都正确加载
- 研究分析：需要了解数据分布和排名情况
"""

import os
import sys
import glob
import pandas as pd
# 将项目的根目录加入到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CandD.draw_map import d_reduction_difference_map, d_reduction_map


## 绘制全球热图以及导出各个国家节能量和节能率的csv文件

def get_available_regions(base_paths):
    """获取所有可用的地区代码
    
    Args:
        base_paths: 包含所有可能路径的字典
    
    Returns:
        set: 所有可用的地区代码集合
    """
    regions = set()
    
    # 只从global文件夹获取地区代码
    continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    for continent in continents:
        summary_path = os.path.join(base_paths['global'], continent, 'summary_p')
        if os.path.exists(summary_path):
            pattern = os.path.join(summary_path, "*_2019_summary_p_results.csv")
            for file_path in glob.glob(pattern):
                region = os.path.basename(file_path).split('_')[0]
                regions.add(region)
    
    return regions

def get_file_path(region, base_paths):
    """获取指定地区的结果文件完整路径
    
    Args:
        region: 地区代码
        base_paths: 包含所有可能路径的字典
    
    Returns:
        str: 文件的完整路径，如果未找到返回None
    """
    # 只在global文件夹中查找
    continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    for continent in continents:
        file_path = os.path.join(base_paths['global'], continent, 'summary_p', f"{region}_2019_summary_p_results.csv")
        if os.path.exists(file_path):
            return file_path
    
    return None

def process_all_cases():
    """处理所有case的地图绘制"""
    # 设置基础路径
    base_paths = {
        'global': r"D:\workstation\energy_comsuption\results\global"
    }
    
    # 地图文件路径
    shapefile_path = r"D:\workstation\energy_comsuption\shapefiles\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp"
    
    # 创建输出目录
    output_base = r"D:\workstation\energy_comsuption\results\global\figure maps"
    difference_output = os.path.join(output_base, "difference")
    reduction_output = os.path.join(output_base, "reduction")
    os.makedirs(difference_output, exist_ok=True)
    os.makedirs(reduction_output, exist_ok=True)
    
    # 获取所有可用的地区代码
    available_regions = get_available_regions(base_paths)
    print(f"找到 {len(available_regions)} 个可用地区")
    
    # 收集所有global数据路径
    global_paths = []
    continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    
    # 处理所有国家
    for region in available_regions:
        for continent in continents:
            file_path = os.path.join(base_paths['global'], continent, 'summary_p', f"{region}_2019_summary_p_results.csv")
            if os.path.exists(file_path):
                path = os.path.join(base_paths['global'], continent, 'summary_p')
                if path not in global_paths:
                    global_paths.append(path)
                break
    
    print("\n使用的数据路径:")
    print("全球数据路径:", global_paths)
    
    # 处理每个case
    for case_num in range(1, 10):  # case1到case9
        print(f"\n处理 Case {case_num}")
        
        # 绘制能耗差值地图
        print(f"绘制 Case {case_num} 能耗差值地图...")
        difference_data = d_reduction_difference_map(
            global_paths,  # 传递global路径列表
            case_num,
            shapefile_path,
            difference_output
        )
        
        # 添加调试信息：检查ER数据
        print(f"\n=== Case {case_num} 调试信息 ===")
        for data_type, data in difference_data.items():
            if 'ER' in data:
                print(f"✓ {data_type}: ER数据存在，值为 {data['ER']:.2f}")
            else:
                print(f"✗ {data_type}: ER数据不存在")
        
        # 检查所有数据
        total_data = difference_data.get('total', {})
        print(f"Total数据统计: 共{len(total_data)}个国家")
        if 'ER' in total_data:
            er_value = total_data['ER']
            print(f"ER在total数据中的值: {er_value:.2f}")
            
            # 检查数据范围
            all_values = list(total_data.values())
            min_val = min(all_values)
            max_val = max(all_values)
            print(f"Total数据范围: {min_val:.2f} - {max_val:.2f}")
            print(f"ER值在数据中的排名: {len([v for v in all_values if v > er_value]) + 1}/{len(all_values)}")
        else:
            print("ER在total数据中不存在！")
        
        # 绘制节能比例地图
        print(f"绘制 Case {case_num} 节能比例地图...")
        reduction_data = d_reduction_map(
            global_paths,  # 传递global路径列表
            case_num,
            shapefile_path,
            reduction_output
        )

        # --- 新增代码：保存为CSV ---
        # 只处理 'total' 类型的数据
        total_difference = difference_data.get('total', {})
        total_reduction = reduction_data.get('total', {})
        
        # 合并数据
        combined_data = []
        all_countries = set(total_difference.keys()) | set(total_reduction.keys())
        
        for country in all_countries:
            combined_data.append({
                'country': country,
                'difference': total_difference.get(country),
                'reduction': total_reduction.get(country)
            })
            
        # 转换为DataFrame并保存
        if combined_data:
            df = pd.DataFrame(combined_data)
            csv_output_path = os.path.join(output_base, f"case{case_num}_summary.csv")
            df.to_csv(csv_output_path, index=False)
            print(f"Case {case_num} 的数据已保存到: {csv_output_path}")

def main():
    print("开始绘制世界地图...")
    process_all_cases()
    print("\n所有地图绘制完成！")
    print(f"能耗差值地图保存在: {os.path.join('results', 'global', 'figure maps', 'difference')}")
    print(f"节能比例地图保存在: {os.path.join('results', 'global', 'figure maps', 'reduction')}")

if __name__ == "__main__":
    main()
