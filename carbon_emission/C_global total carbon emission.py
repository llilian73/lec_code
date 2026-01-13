"""
全球总碳排放量和人均碳排放量汇总计算工具（2016-2020年）

功能概述：
本工具用于汇总计算全球所有国家的碳排放数据（2016-2020年），包括总碳排放量和人均碳排放量，支持多种节能案例的对比分析。

输入数据：
1. 各国碳排放汇总文件：
   - 路径：Z:\local_environment_creation\carbon_emission\result\{年份}\notcapita\{case}_summary.csv
   - 包含各国总碳排放数据（tCO2）
   - 涵盖ref、case1-case20共21种案例
   - 覆盖2016-2020年

2. 人口数据文件：
   - country_population_2020.csv：全球各国人口数据
   - 用于计算全球总人口和人均碳排放量

输出结果：
1. 全球总碳排放数据（CSV格式）：
   - global_total_carbon_emission.csv：全球总碳排放汇总（含年份列）
   - 格式：year,index,carbon_emission(tCO2),carbon_emission_reduction(tCO2),carbon_emission_reduction(%)

2. 全球人均碳排放数据（CSV格式）：
   - global_per_capita_carbon_emission.csv：全球人均碳排放（含年份列）
   - 格式：year,index,carbon_emission(kgCO2/person),carbon_emission_reduction(kgCO2/person),carbon_emission_reduction(%)

计算公式：
- 全球总碳排放量 = 所有国家碳排放量之和（tCO2）
- 全球人均碳排放量 = 全球总碳排放量 / 全球总人口（kgCO2/person）
- 碳排放减少量 = ref碳排放量 - case碳排放量
- 碳排放减少比例 = (碳排放减少量 / ref碳排放量) × 100%
"""

import os
import pandas as pd
import numpy as np

def get_global_population():
    """获取全球总人口
    注意：country_population_2020.csv 格式为：
    Continent,Country_Code_2,Country_Code_3,Country_Name,Population_2020
    """
    population_file = r"Z:\local_environment_creation\energy_consumption_gird\result\country_population_2020.csv"
    # 使用 keep_default_na=False 避免 'NA'（纳米比亚）被识别为缺失值
    df = pd.read_csv(population_file, encoding="gbk", keep_default_na=False)
    return df['Population_2020'].sum()

def calculate_global_carbon_emission():
    """计算全球碳排放数据（2016-2020年，21个工况）"""
    # 输入基础路径
    input_base_dir = r"Z:\local_environment_creation\carbon_emission\2016-2020\result"
    
    # 处理年份范围：2016-2020
    years = range(2016, 2021)
    
    # 初始化结果数据结构 - 21个工况（ref + case1-20）
    cases = ['ref'] + [f'case{i}' for i in range(1, 21)]
    
    # 存储所有年份的数据
    all_years_data = {}
    
    for year in years:
        print(f"正在处理 {year} 年数据...")
        
        # 输入目录：{年份}\notcapita
        year_input_dir = os.path.join(input_base_dir, str(year), 'notcapita')
        
        if not os.path.exists(year_input_dir):
            print(f"警告: 未找到 {year} 年的输入目录: {year_input_dir}")
            continue
        
        # 初始化当年结果数据结构
        carbon_data = pd.DataFrame(0.0, index=cases, columns=['carbon_emission(tCO2)'])
        
        # 处理每个case的汇总文件
        for case in cases:
            summary_file = os.path.join(year_input_dir, f"{case}_summary.csv")
            
            if not os.path.exists(summary_file):
                print(f"警告: 未找到 {case} 的汇总文件: {summary_file}")
                continue
            
            try:
                # 读取汇总文件
                df = pd.read_csv(summary_file, keep_default_na=False)
                
                # 汇总所有国家的碳排放量
                if 'carbon_emission(tCO2)' in df.columns:
                    # 将值转换为数值类型
                    df['carbon_emission(tCO2)'] = pd.to_numeric(df['carbon_emission(tCO2)'], errors='coerce')
                    # 求和
                    total_emission = df['carbon_emission(tCO2)'].sum()
                    carbon_data.loc[case, 'carbon_emission(tCO2)'] = total_emission
                else:
                    print(f"警告: {case} 汇总文件中未找到 carbon_emission(tCO2) 列")
                    
            except Exception as e:
                print(f"处理 {case} 在 {year} 年数据时出错: {e}")
        
        # 计算减少量和减少比例
        ref_emission = carbon_data.loc['ref', 'carbon_emission(tCO2)']
        
        # 初始化减少量和减少比例列
        carbon_data['carbon_emission_reduction(tCO2)'] = 0.0
        carbon_data['carbon_emission_reduction(%)'] = 0.0
        
        for case in cases[1:]:  # 跳过ref
            case_emission = carbon_data.loc[case, 'carbon_emission(tCO2)']
            reduction = ref_emission - case_emission
            carbon_data.loc[case, 'carbon_emission_reduction(tCO2)'] = reduction
            
            if ref_emission > 0:
                reduction_percentage = (reduction / ref_emission) * 100
                carbon_data.loc[case, 'carbon_emission_reduction(%)'] = reduction_percentage
        
        all_years_data[year] = carbon_data
    
    return all_years_data

def calculate_global_per_capita_carbon_emission(all_years_data):
    """计算全球人均碳排放数据"""
    # 获取全球总人口
    global_population = get_global_population()
    print(f"全球总人口: {global_population:,.0f}")
    
    # 存储所有年份的人均数据
    all_years_per_capita_data = {}
    
    for year, carbon_data in all_years_data.items():
        # 创建人均数据DataFrame
        per_capita_data = pd.DataFrame(index=carbon_data.index)
        
        # 将总碳排放量转换为人均碳排放量（tCO2 -> kgCO2/person）
        # tCO2 × 1000 / 人口 = kgCO2/person
        per_capita_data['carbon_emission(kgCO2/person)'] = (
            carbon_data['carbon_emission(tCO2)'] * 1000 / global_population
        )
        
        # 计算人均碳排放减少量
        ref_per_capita = per_capita_data.loc['ref', 'carbon_emission(kgCO2/person)']
        per_capita_data['carbon_emission_reduction(kgCO2/person)'] = (
            ref_per_capita - per_capita_data['carbon_emission(kgCO2/person)']
        )
        
        # 计算减少比例（与总量相同）
        per_capita_data['carbon_emission_reduction(%)'] = carbon_data['carbon_emission_reduction(%)']
        
        all_years_per_capita_data[year] = per_capita_data
    
    return all_years_per_capita_data

def save_results(all_years_data, all_years_per_capita_data, output_dir):
    """保存结果到CSV文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    years = range(2016, 2021)
    
    # 保存总碳排放数据
    print("\n保存全球总碳排放数据...")
    total_data_list = []
    for year in years:
        if year in all_years_data:
            year_data = all_years_data[year].copy()
            year_data['year'] = year
            year_data = year_data.reset_index()
            year_data = year_data.rename(columns={'index': 'index'})
            total_data_list.append(year_data)
    
    if total_data_list:
        total_df = pd.concat(total_data_list, ignore_index=True)
        # 重新排列列顺序
        column_order = ['year', 'index', 'carbon_emission(tCO2)', 
                        'carbon_emission_reduction(tCO2)', 'carbon_emission_reduction(%)']
        total_df = total_df[column_order]
        
        output_file = os.path.join(output_dir, 'global_total_carbon_emission.csv')
        total_df.to_csv(output_file, index=False)
        print(f"已保存总碳排放数据至: {output_file}")
    
    # 保存人均碳排放数据
    print("\n保存全球人均碳排放数据...")
    per_capita_data_list = []
    for year in years:
        if year in all_years_per_capita_data:
            year_data = all_years_per_capita_data[year].copy()
            year_data['year'] = year
            year_data = year_data.reset_index()
            year_data = year_data.rename(columns={'index': 'index'})
            per_capita_data_list.append(year_data)
    
    if per_capita_data_list:
        per_capita_df = pd.concat(per_capita_data_list, ignore_index=True)
        # 重新排列列顺序
        column_order = ['year', 'index', 'carbon_emission(kgCO2/person)', 
                        'carbon_emission_reduction(kgCO2/person)', 'carbon_emission_reduction(%)']
        per_capita_df = per_capita_df[column_order]
        
        output_file = os.path.join(output_dir, 'global_per_capita_carbon_emission.csv')
        per_capita_df.to_csv(output_file, index=False)
        print(f"已保存人均碳排放数据至: {output_file}")

def main():
    print("="*60)
    print("开始计算全球总碳排放量...")
    print("="*60)
    
    # 计算全球总碳排放量
    all_years_data = calculate_global_carbon_emission()
    
    print("\n" + "="*60)
    print("开始计算全球人均碳排放量...")
    print("="*60)
    
    # 计算全球人均碳排放量
    all_years_per_capita_data = calculate_global_per_capita_carbon_emission(all_years_data)
    
    # 保存结果
    output_dir = r"Z:\local_environment_creation\carbon_emission\2016-2020\global_total_carbon_emission"
    save_results(all_years_data, all_years_per_capita_data, output_dir)
    
    # 打印汇总信息
    print("\n" + "="*60)
    print("总碳排放汇总:")
    print("="*60)
    for year in sorted(all_years_data.keys()):
        print(f"\n{year}年:")
        print(all_years_data[year])
    
    print("\n" + "="*60)
    print("人均碳排放汇总:")
    print("="*60)
    for year in sorted(all_years_per_capita_data.keys()):
        print(f"\n{year}年:")
        print(all_years_per_capita_data[year])
    
    print("\n" + "="*60)
    print("所有计算完成！")
    print("="*60)

if __name__ == "__main__":
    main()

