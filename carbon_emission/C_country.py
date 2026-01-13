"""
全球总碳排放量计算工具（2016-2020年）

功能概述：
本工具用于计算全球各个国家的总碳排放量，基于能耗数据和碳排放因子。将总能耗数据转换为碳排放数据，支持多种节能工况的碳排放效果分析。

输入数据：
1. 能耗数据：
   - 基础目录：Z:\local_environment_creation\energy_consumption_gird\result\result
   - 按年份组织：2016, 2017, 2018, 2019, 2020
   - 按大洲分类：Africa, Asia, Europe, North America, Oceania, South America
   - 数据位置：{年份}\{大洲}\summary\{国家代码}_{年份}_summary_results.csv
   - 关键列：total_demand_sum(TWh) - 总能耗
   - 工况：ref + case1-case20 共21种工况

2. 碳排放因子数据：
   - 文件路径：Z:\local_environment_creation\carbon_emission\2016-2020\carbon_emission_factors_processed.csv
   - 格式：continent,Country_Code_2,Country_Code_3,Country_Name,2016,2017,2018,2019,2020
   - 单位：gCO2eq/kWh

输出结果：
1. 国家级别碳排放数据：
   - 输出目录：Z:\local_environment_creation\carbon_emission\result\{年份}\{大洲}\
   - 文件格式：{国家代码}_{年份}_carbon_emission.csv
   - 包含列：
     * carbon_emission(tCO2) - 总碳排放量
     * carbon_emission_reduction(tCO2) - 相对于ref的碳排放减少量
     * carbon_emission_reduction(%) - 碳排放减少比例

2. 汇总数据：
   - 输出目录：Z:\local_environment_creation\carbon_emission\result\{年份}\notcapita\
   - 文件格式：{case_name}_summary.csv
   - 包含列：Country_Code_2, carbon_emission(tCO2), carbon_emission_reduction(tCO2), carbon_emission_reduction(%)

计算公式：
- 总碳排放量 = 总能耗(TWh) × 碳排放因子(gCO2/kWh) × 10^3
- 碳排放减少量 = ref碳排放量 - case碳排放量
- 碳排放减少比例 = (碳排放减少量 / ref碳排放量) × 100%
"""

import pandas as pd
import os

def load_emission_factors():
    """加载碳排放因子数据，按年份组织"""
    factor_file = r"Z:\local_environment_creation\carbon_emission\2016-2020\carbon_emission_factors_processed.csv"
    
    # 使用keep_default_na=False避免'NA'被识别为缺失值
    df = pd.read_csv(factor_file, keep_default_na=False)
    
    # 创建按年份组织的映射：{year: {country_code_2: factor}}
    emission_factors_by_year = {}
    years = ['2016', '2017', '2018', '2019', '2020']
    
    for year in years:
        if year not in df.columns:
            print(f"警告: 未找到{year}年的碳排放因子列")
            continue
        
        # 创建国家代码到碳排放因子的映射
        factors_dict = {}
        for _, row in df.iterrows():
            country_code_2 = str(row['Country_Code_2']).strip().upper()
            factor_value = row[year]
            
            # 检查是否为有效数值
            if pd.notna(factor_value) and str(factor_value).strip() != '':
                try:
                    factors_dict[country_code_2] = float(factor_value)
                except (ValueError, TypeError):
                    continue
        
        emission_factors_by_year[year] = factors_dict
        print(f"成功加载{year}年碳排放因子: {len(factors_dict)}个国家")
    
    return emission_factors_by_year

def process_country_data(country_code, energy_file, emission_factor, output_file):
    """处理单个国家的数据"""
    try:
        # 读取能耗数据
        df = pd.read_csv(energy_file, index_col=0)
        
        # 创建结果DataFrame
        results = pd.DataFrame(index=df.index)
        
        # 计算碳排放量（TWh * gCO2/kWh * 10^3 = tCO2）
        # TWh × 10^12 Wh × gCO2/kWh × 10^-9 = tCO2
        # 简化：TWh × gCO2/kWh × 10^3 = tCO2
        results['carbon_emission(tCO2)'] = df['total_demand_sum(TWh)'] * emission_factor * 1e3
        
        # 计算减少的碳排放量
        ref_emission = results.loc['ref', 'carbon_emission(tCO2)']
        results['carbon_emission_reduction(tCO2)'] = ref_emission - results['carbon_emission(tCO2)']
        
        # 计算减少比例
        results['carbon_emission_reduction(%)'] = (results['carbon_emission_reduction(tCO2)'] / ref_emission) * 100
        
        # 保存结果
        results.to_csv(output_file)
        return True, results, None
        
    except Exception as e:
        return False, None, str(e)

def process_year(year, energy_base_dir, output_base_dir, emission_factors_by_year, error_log):
    """处理单个年份的数据"""
    print(f"\n{'='*60}")
    print(f"开始处理 {year} 年的数据...")
    print(f"{'='*60}")
    
    # 获取该年份的碳排放因子
    emission_factors = emission_factors_by_year.get(year, {})
    if not emission_factors:
        print(f"警告: 未找到{year}年的碳排放因子数据，跳过该年份")
        return
    
    # 创建年份输出目录
    year_output_dir = os.path.join(output_base_dir, str(year))
    os.makedirs(year_output_dir, exist_ok=True)
    
    # 存储所有国家的结果（用于汇总）
    all_countries_results = {}
    cases = ['ref'] + [f'case{i}' for i in range(1, 21)]
    
    # 初始化汇总数据结构
    for case in cases:
        all_countries_results[case] = []
    
    # 处理每个大洲
    continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    
    for continent in continents:
        print(f"\n处理 {continent}...")
        
        # 创建大洲输出目录
        continent_output_dir = os.path.join(year_output_dir, continent)
        os.makedirs(continent_output_dir, exist_ok=True)
        
        # 能耗数据目录
        energy_dir = os.path.join(energy_base_dir, str(year), continent, 'summary')
        
        if not os.path.exists(energy_dir):
            print(f"  警告: 未找到 {continent} 的能耗数据目录: {energy_dir}")
            continue
        
        # 获取该洲下所有国家的能耗数据文件
        energy_files = [f for f in os.listdir(energy_dir) 
                       if f.endswith(f'_{year}_summary_results.csv')]
        
        print(f"  找到 {len(energy_files)} 个国家的数据文件")
        
        for file in energy_files:
            # 提取国家代码（二字母代码）
            country_code = file.split('_')[0].upper()
            
            # 获取该国家的碳排放因子
            factor = emission_factors.get(country_code)
            
            if factor is None:
                error_log.append({
                    'Year': year,
                    'Continent': continent,
                    'Country': country_code,
                    'Error': '未找到碳排放因子'
                })
                print(f"  警告: {country_code} 未找到碳排放因子")
                continue
            
            # 处理国家数据
            energy_file = os.path.join(energy_dir, file)
            output_file = os.path.join(continent_output_dir, f"{country_code}_{year}_carbon_emission.csv")
            
            success, results_df, error = process_country_data(
                country_code, energy_file, factor, output_file
            )
            
            if success:
                # 将结果添加到汇总数据中
                for case in cases:
                    if case in results_df.index:
                        all_countries_results[case].append({
                            'Country_Code_2': country_code,
                            'carbon_emission(tCO2)': results_df.loc[case, 'carbon_emission(tCO2)'],
                            'carbon_emission_reduction(tCO2)': results_df.loc[case, 'carbon_emission_reduction(tCO2)'],
                            'carbon_emission_reduction(%)': results_df.loc[case, 'carbon_emission_reduction(%)']
                        })
                print(f"  ✓ {country_code} 处理成功")
            else:
                error_log.append({
                    'Year': year,
                    'Continent': continent,
                    'Country': country_code,
                    'Error': error
                })
                print(f"  ✗ {country_code} 处理失败: {error}")
    
    # 保存汇总数据（按工况）
    print(f"\n保存 {year} 年的汇总数据...")
    # 创建notcapita子目录
    notcapita_output_dir = os.path.join(year_output_dir, 'notcapita')
    os.makedirs(notcapita_output_dir, exist_ok=True)
    
    for case in cases:
        if all_countries_results[case]:
            summary_df = pd.DataFrame(all_countries_results[case])
            # 按国家代码排序
            summary_df = summary_df.sort_values('Country_Code_2')
            summary_file = os.path.join(notcapita_output_dir, f"{case}_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"  已保存: notcapita/{case}_summary.csv ({len(summary_df)}个国家)")
    
    print(f"\n{year} 年数据处理完成！")

def main():
    # 设置路径
    energy_base_dir = r"Z:\local_environment_creation\energy_consumption_gird\result\result"
    output_base_dir = r"Z:\local_environment_creation\carbon_emission\2016-2020\result"
    
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 加载碳排放因子（按年份组织）
    print("正在加载碳排放因子数据...")
    emission_factors_by_year = load_emission_factors()
    
    # 存储错误信息
    error_log = []
    
    # 处理每个年份的数据
    years = [2016, 2017, 2018, 2019, 2020]
    for year in years:
        process_year(str(year), energy_base_dir, output_base_dir, 
                    emission_factors_by_year, error_log)
    
    # 保存错误日志
    if error_log:
        error_df = pd.DataFrame(error_log)
        error_file = os.path.join(output_base_dir, 'error_log.csv')
        error_df.to_csv(error_file, index=False)
        print(f"\n错误日志已保存至: {error_file}")
        print(f"共有 {len(error_log)} 个国家处理失败")
    else:
        print("\n所有国家处理成功！")
    
    print("\n所有年份数据处理完成！")

if __name__ == "__main__":
    main()
