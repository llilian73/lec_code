"""
全球建筑能耗TOP10国家数据收集工具

功能概述：
本工具用于收集总能耗TOP10国家的数据，从ref_summary_average.csv中选取总能耗最高的10个国家，
然后从case1-case20的汇总文件中收集对应国家的数据，整理并输出为CSV文件。

输入数据：
1. ref工况的平均值文件：
   - 路径：Z:\local_environment_creation\energy_consumption_gird\result\result\figure_maps_and_data\not_capita\data\average\ref_summary_average.csv
   - 包含各国5年平均值数据，包括total_energy字段（总能耗）

2. case1-case20的平均值文件：
   - 路径：Z:\local_environment_creation\energy_consumption_gird\result\result\figure_maps_and_data\not_capita\data\average\case{num}_summary_average.csv
   - 包含各国5年平均值数据

主要功能：
1. TOP10国家识别：
   - 读取ref_summary_average.csv文件
   - 根据total_energy字段，选取总能耗最高的10个国家
   - 获取这10个国家的国家代码列表

2. 数据收集：
   - 从ref_summary_average.csv中提取TOP10国家的数据
   - 从case1-case20的汇总文件中提取对应TOP10国家的数据
   - 收集每个国家的总能耗、供暖能耗、制冷能耗及其差值和节能率

3. 数据整理：
   - 将数据整理为统一的CSV格式
   - 包含以下列：group（国家代码）, index（工况名）,
                total_demand_sum(TWh), total_demand_diff(TWh), total_demand_reduction(%),
                heating_demand_sum(TWh), heating_demand_diff(TWh), heating_demand_reduction(%),
                cooling_demand_sum(TWh), cooling_demand_diff(TWh), cooling_demand_reduction(%)

4. CSV数据导出：
   - 将所有工况（ref、case1-case20）汇总到一个CSV文件
   - 文件名格式：top10_summary_2016_2020_average.csv
   - 包含所有工况的TOP10国家数据

输出结果：
1. CSV格式的TOP10国家汇总数据：
   - top10_summary_2016_2020_average.csv
   - 包含所有工况（ref、case1-case20）的能耗数据和节能率
   - 按国家代码和工况顺序排序

2. 输出目录结构：
   - group/：TOP10国家汇总文件（与C_group.py相同的输出路径）

计算特点：
- TOP10筛选：基于ref工况的总能耗选择TOP10国家
- 多工况处理：处理ref和case1-case20共21种工况
- 数据一致性：确保所有工况使用相同的TOP10国家列表
- 容错处理：对缺失数据和缺失值进行容错处理

数据流程：
1. 读取ref_summary_average.csv，根据total_energy字段选取TOP10国家
2. 读取ref_summary_average.csv，提取TOP10国家的数据并计算节能率
3. 遍历case1-case20，从对应的汇总文件中提取TOP10国家的数据并计算节能率
4. 将所有工况的数据汇总到一个CSV文件中，包含TOP10国家的完整数据
"""

import os
import pandas as pd
import numpy as np


def get_top10_countries(ref_file_path):
    """从ref_summary_average.csv中选取总能耗最高的10个国家
    Args:
        ref_file_path: ref_summary_average.csv文件路径
    Returns:
        list: TOP10国家的国家代码列表
    """
    try:
        df = pd.read_csv(ref_file_path, keep_default_na=False, dtype={'country': str})
        # 确保total_energy列存在
        if 'total_energy' not in df.columns:
            print(f"警告：{ref_file_path}中不存在total_energy列")
            return []
        
        # 将total_energy转换为数值类型
        df['total_energy_numeric'] = pd.to_numeric(df['total_energy'], errors='coerce')
        # 按total_energy降序排序，选取前10个
        df_sorted = df.sort_values('total_energy_numeric', ascending=False, na_position='last')
        top10_df = df_sorted.head(10)
        
        # 获取国家代码列表
        top10_countries = top10_df['country'].astype(str).str.strip().str.upper().tolist()
        print(f"TOP10国家（按总能耗排序）: {top10_countries}")
        return top10_countries
    except Exception as e:
        print(f"读取ref文件失败: {e}")
        return []


def load_case_data(case_file_path, top10_countries):
    """从case汇总文件中加载TOP10国家的数据
    Args:
        case_file_path: case汇总文件路径
        top10_countries: TOP10国家代码列表
    Returns:
        dict: {country: {字段: 值}}
    """
    country_data = {}
    try:
        df = pd.read_csv(case_file_path, keep_default_na=False, dtype={'country': str})
        df['country'] = df['country'].astype(str).str.strip().str.upper()
        
        for country in top10_countries:
            country_row = df[df['country'] == country]
            if not country_row.empty:
                row = country_row.iloc[0]
                country_data[country] = {
                    'total_energy': pd.to_numeric(row.get('total_energy', np.nan), errors='coerce'),
                    'heating_energy': pd.to_numeric(row.get('heating_energy', np.nan), errors='coerce'),
                    'cooling_energy': pd.to_numeric(row.get('cooling_energy', np.nan), errors='coerce'),
                    'total_difference': pd.to_numeric(row.get('total_difference', np.nan), errors='coerce'),
                    'heating_difference': pd.to_numeric(row.get('heating_difference', np.nan), errors='coerce'),
                    'cooling_difference': pd.to_numeric(row.get('cooling_difference', np.nan), errors='coerce'),
                    'total_reduction': pd.to_numeric(row.get('total_reduction', np.nan), errors='coerce'),
                    'heating_reduction': pd.to_numeric(row.get('heating_reduction', np.nan), errors='coerce'),
                    'cooling_reduction': pd.to_numeric(row.get('cooling_reduction', np.nan), errors='coerce'),
                }
    except Exception as e:
        print(f"读取case文件失败 {case_file_path}: {e}")
    
    return country_data


def compute_reductions_for_case(case_data, ref_data, case_name):
    """计算某个case的节能率数据（如果case_data中没有reduction数据，则基于ref计算）
    Args:
        case_data: case的数据字典 {country: {字段: 值}}
        ref_data: ref的数据字典 {country: {字段: 值}}
        case_name: case名称（'ref' 或 'case{num}'）
    Returns:
        list: 输出行列表
    """
    rows = []
    
    for country in case_data.keys():
        case_vals = case_data[country]
        ref_vals = ref_data.get(country, {})
        
        # 获取case的数据
        total_sum = case_vals.get('total_energy', np.nan)
        total_diff = case_vals.get('total_difference', np.nan)
        heat_sum = case_vals.get('heating_energy', np.nan)
        heat_diff = case_vals.get('heating_difference', np.nan)
        cool_sum = case_vals.get('cooling_energy', np.nan)
        cool_diff = case_vals.get('cooling_difference', np.nan)
        
        # 获取节能率（如果存在则使用，否则计算）
        total_red = case_vals.get('total_reduction', np.nan)
        heat_red = case_vals.get('heating_reduction', np.nan)
        cool_red = case_vals.get('cooling_reduction', np.nan)
        
        # 如果节能率不存在，基于ref计算
        if pd.isna(total_red) and case_name != 'ref':
            ref_total = ref_vals.get('total_energy', np.nan)
            if pd.notna(ref_total) and ref_total > 0 and pd.notna(total_sum):
                total_red = ((ref_total - total_sum) / ref_total * 100.0)
        
        if pd.isna(heat_red) and case_name != 'ref':
            ref_heat = ref_vals.get('heating_energy', np.nan)
            if pd.notna(ref_heat) and ref_heat > 0 and pd.notna(heat_sum):
                heat_red = ((ref_heat - heat_sum) / ref_heat * 100.0)
        
        if pd.isna(cool_red) and case_name != 'ref':
            ref_cool = ref_vals.get('cooling_energy', np.nan)
            if pd.notna(ref_cool) and ref_cool > 0 and pd.notna(cool_sum):
                cool_red = ((ref_cool - cool_sum) / ref_cool * 100.0)
        
        rows.append({
            'group': country,
            'index': case_name,
            'total_demand_sum(TWh)': total_sum,
            'total_demand_diff(TWh)': total_diff,
            'total_demand_reduction(%)': total_red,
            'heating_demand_sum(TWh)': heat_sum,
            'heating_demand_diff(TWh)': heat_diff,
            'heating_demand_reduction(%)': heat_red,
            'cooling_demand_sum(TWh)': cool_sum,
            'cooling_demand_diff(TWh)': cool_diff,
            'cooling_demand_reduction(%)': cool_red,
        })
    
    return rows


def main():
    # 输入和输出路径
    base_root = r"Z:\local_environment_creation\energy_consumption_gird\result\result"
    data_dir = os.path.join(base_root, r"figure_maps_and_data\not_capita\data\average")
    output_dir = os.path.join(base_root, 'group')  # 输出路径与C_group.py相同
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取ref文件，获取TOP10国家
    ref_file_path = os.path.join(data_dir, 'ref_summary_average.csv')
    if not os.path.exists(ref_file_path):
        print(f"错误：找不到ref文件 {ref_file_path}")
        return
    
    top10_countries = get_top10_countries(ref_file_path)
    if not top10_countries:
        print("错误：无法获取TOP10国家列表")
        return
    
    # 加载ref数据
    ref_data = load_case_data(ref_file_path, top10_countries)
    if not ref_data:
        print("错误：无法加载ref数据")
        return
    
    # 收集所有工况的数据
    all_rows = []
    
    # 处理ref工况
    print(f"\n处理 ref 工况...")
    ref_rows = compute_reductions_for_case(ref_data, ref_data, 'ref')
    if ref_rows:
        all_rows.extend(ref_rows)
        print(f"  ref工况: {len(ref_rows)} 条记录")
    
    # 处理case1-case20
    cases = [f'case{i}' for i in range(1, 21)]
    for case_name in cases:
        case_file_path = os.path.join(data_dir, f'{case_name}_summary_average.csv')
        if not os.path.exists(case_file_path):
            print(f"警告：找不到 {case_file_path}，跳过")
            continue
        
        print(f"处理 {case_name} 工况...")
        case_data = load_case_data(case_file_path, top10_countries)
        if not case_data:
            print(f"警告：{case_name} 无数据，跳过")
            continue
        
        case_rows = compute_reductions_for_case(case_data, ref_data, case_name)
        if case_rows:
            all_rows.extend(case_rows)
            print(f"  {case_name}工况: {len(case_rows)} 条记录")
    
    # 将所有工况汇总到一个CSV文件
    if all_rows:
        df_all = pd.DataFrame(all_rows, columns=[
            'group', 'index',
            'total_demand_sum(TWh)', 'total_demand_diff(TWh)', 'total_demand_reduction(%)',
            'heating_demand_sum(TWh)', 'heating_demand_diff(TWh)', 'heating_demand_reduction(%)',
            'cooling_demand_sum(TWh)', 'cooling_demand_diff(TWh)', 'cooling_demand_reduction(%)'
        ])
        
        # 排序：按国家代码排序，然后按工况顺序（ref, case1, case2, ..., case20）
        country_order = {country: i for i, country in enumerate(top10_countries)}
        case_order = {c: i for i, c in enumerate(['ref'] + [f'case{i}' for i in range(1, 21)])}
        df_all['country_order'] = df_all['group'].map(lambda x: country_order.get(x, 999))
        df_all['case_order'] = df_all['index'].map(lambda x: case_order.get(x, 999))
        df_all = df_all.sort_values(['country_order', 'case_order']).drop(['country_order', 'case_order'], axis=1)
        
        output_path = os.path.join(output_dir, 'top10_summary_2016_2020_average.csv')
        df_all.to_csv(output_path, index=False)
        print(f"\n所有工况汇总完成！已保存: {output_path}")
        print(f"共 {len(all_rows)} 条记录（{len(top10_countries)} 个国家 × {len(cases) + 1} 个工况）")
    else:
        print("\n错误：没有收集到任何数据")


if __name__ == '__main__':
    main()


