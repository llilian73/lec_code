"""
脚本用途:
    - 从 Excel 源文件提取 2016–2020 年各国碳排放因子（gCO2eq/kWh）。
    - 使用 all_countries_info.csv 提供的国家代码映射（三字母代码到二字母代码），生成包含各年排放因子的数据集。
    - 记录缺失映射的国家条目。

输入:
    - Excel: Carbon emission factor_2016_2020.xls（按年份分工作表，包含 GID_0 三字母代码）。
    - CSV: all_countries_info.csv（含 Country_Code_2 和 Country_Code_3 的对应关系）。

输出:
    - carbon_emission_factors_processed.csv：按国家整理的 2016–2020 年排放因子（包含 Country_Code_2 和 Country_Code_3）。
    - missing_countries_carbon_emission.csv：在映射中缺失的国家项。
"""

import pandas as pd
import os

def process_carbon_emission_factors():
    """
    处理碳排放因子数据，从Excel文件提取并转换为所需的CSV格式。
    """
    
    # 文件路径
    excel_file = r"Z:\local_environment_creation\carbon_emission\Carbon emission factor_2016_2020.xls"
    countries_info_file = r"Z:\local_environment_creation\all_countries_info.csv"
    
    # 创建输出目录
    output_dir = r"Z:\local_environment_creation\carbon_emission\2016-2020"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "carbon_emission_factors_processed.csv")
    missing_countries_file = os.path.join(output_dir, "missing_countries_carbon_emission.csv")
    
    print("正在读取碳排放因子Excel文件...")
    
    # 读取Excel文件中的所有工作表
    years = ['2016', '2017', '2018', '2019', '2020']
    all_data = {}
    
    for year in years:
        try:
            df = pd.read_excel(excel_file, sheet_name=year)
            print(f"成功读取{year}年数据: {len(df)}行")
            all_data[year] = df
        except Exception as e:
            print(f"读取{year}年数据时出错: {e}")
            continue
    
    print("\n正在读取国家代码映射文件...")
    
    # 读取 all_countries_info.csv 以获取三字母代码到二字母代码的映射
    try:
        # 使用gbk编码读取文件，避免将 'NA' 识别为缺失值
        countries_info_df = pd.read_csv(countries_info_file, encoding='gbk', keep_default_na=False, na_values=[''])
        print(f"成功读取国家代码映射数据: {len(countries_info_df)}行")
        
        # 统一清洗国家代码：去除空格并大写
        countries_info_df['Country_Code_2'] = countries_info_df['Country_Code_2'].astype(str).str.strip().str.upper()
        countries_info_df['Country_Code_3'] = countries_info_df['Country_Code_3'].astype(str).str.strip().str.upper()
        countries_info_df['Country_Name'] = countries_info_df['Country_Name'].astype(str).str.strip()
        
        # 创建从三字母代码到二字母代码的映射
        code3_to_code2 = dict(zip(countries_info_df['Country_Code_3'], countries_info_df['Country_Code_2']))
        # 创建从三字母代码到国家名称的映射（用于输出）
        code3_to_name = dict(zip(countries_info_df['Country_Code_3'], countries_info_df['Country_Name']))
        print(f"为{len(code3_to_code2)}个国家创建了代码映射")
        
    except Exception as e:
        print(f"读取国家代码映射文件时出错: {e}")
        return
    
    print("\n正在处理数据...")
    
    # 处理每年的数据
    processed_data = {}
    missing_countries = []
    
    for year, df in all_data.items():
        print(f"正在处理{year}年数据...")
        
        for _, row in df.iterrows():
            # 使用 GID_0（三字母代码）来匹配
            gid_0 = str(row['GID_0']).strip().upper()
            continent = row['continent']
            carbon_intensity = row['Carbon intensity gCO2eq/kWh']
            
            # 检查三字母代码是否存在于我们的映射中
            if gid_0 in code3_to_code2:
                country_code_2 = code3_to_code2[gid_0]
                country_code_3 = gid_0
                # 获取国家名称（如果映射中有）
                country_name = code3_to_name.get(gid_0, row.get('NAME_0', ''))
                
                # 为每个国家创建唯一键（使用三字母代码）
                key = f"{continent}_{country_code_3}"
                
                if key not in processed_data:
                    processed_data[key] = {
                        'continent': continent,
                        'Country_Code_2': country_code_2,
                        'Country_Code_3': country_code_3,
                        'Country_Name': country_name,
                        '2016': None,
                        '2017': None,
                        '2018': None,
                        '2019': None,
                        '2020': None
                    }
                
                processed_data[key][year] = carbon_intensity
            else:
                # 在映射中未找到的国家
                country_name = row.get('NAME_0', '')
                missing_countries.append({
                    'continent': continent,
                    'GID_0': gid_0,
                    'Country_Name': country_name,
                    'Year': year,
                    'Carbon_intensity': carbon_intensity
                })
                print(f"缺失国家: {country_name} (GID_0: {gid_0}, {continent}) 在{year}年")
    
    print(f"\n已处理{len(processed_data)}个国家")
    print(f"发现{len(missing_countries)}个缺失的国家条目")
    
    # 转换为DataFrame
    result_df = pd.DataFrame.from_dict(processed_data, orient='index')
    result_df = result_df.reset_index(drop=True)
    
    # 重新排列列
    column_order = ['continent', 'Country_Code_2', 'Country_Code_3', 'Country_Name', '2016', '2017', '2018', '2019', '2020']
    result_df = result_df[column_order]
    
    print(f"\n正在保存处理后的数据到 {output_file}...")
    result_df.to_csv(output_file, index=False)
    print(f"成功保存{len(result_df)}个国家到 {output_file}")
    
    # 保存缺失的国家
    if missing_countries:
        print(f"\n正在保存缺失的国家到 {missing_countries_file}...")
        missing_df = pd.DataFrame(missing_countries)
        missing_df.to_csv(missing_countries_file, index=False)
        print(f"成功保存{len(missing_df)}个缺失的国家条目到 {missing_countries_file}")
        
        # 显示缺失国家的摘要
        print("\n缺失国家摘要:")
        missing_summary = missing_df.groupby(['continent', 'Country_Name']).size().reset_index(name='count')
        print(missing_summary)
    
    print("\n处理完成!")
    return result_df, missing_countries

if __name__ == "__main__":
    try:
        result_df, missing_countries = process_carbon_emission_factors()
        
        # 显示结果样本
        print("\n处理后的数据样本:")
        print(result_df.head(10))
        
        print(f"\n总共处理的国家数: {len(result_df)}")
        print(f"总共缺失的国家条目数: {len(missing_countries)}")
        
    except Exception as e:
        print(f"主程序执行时出错: {e}")
        import traceback
        traceback.print_exc()
