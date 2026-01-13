"""
ILO平均小时工资数据处理工具

功能概述：
本工具用于处理ILO平均小时工资数据，筛选特定条件的数据并与国家信息匹配。

输入数据：
- ILO数据：Z:\local_environment_creation\cost\Earnings\EAR_4HRL_SEX_OCU_CUR_NB_A-20251104T0843.csv
- 国家信息：Z:\local_environment_creation\cost\country_cost.csv

输出结果：
- CSV文件：Z:\local_environment_creation\cost\Earnings\ILO_Average_hourly_earnings_processed.csv
"""

import os
import pandas as pd
import numpy as np

# 配置文件路径
ILO_FILE = r"Z:\local_environment_creation\cost\Earnings\EAR_4HRL_SEX_OCU_CUR_NB_A-20251104T0843.csv"
COUNTRY_FILE = r"Z:\local_environment_creation\cost\country_cost.csv"
OUTPUT_DIR = r"Z:\local_environment_creation\cost\Earnings"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ILO_Average_hourly_earnings_processed.csv")


def read_csv_with_encoding(file_path):
    """读取CSV文件，尝试多种编码"""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk', 'gb2312', 'utf-8-sig']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"成功使用编码 '{encoding}' 读取文件: {os.path.basename(file_path)}")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    raise ValueError(f"无法使用常见编码读取文件: {file_path}")


def load_ilo_data():
    """加载ILO数据"""
    print(f"读取ILO数据文件: {ILO_FILE}")
    
    if not os.path.exists(ILO_FILE):
        raise FileNotFoundError(f"ILO数据文件不存在: {ILO_FILE}")
    
    df = read_csv_with_encoding(ILO_FILE)
    print(f"成功加载ILO数据，共 {len(df)} 行")
    
    return df


def load_country_data():
    """加载国家信息数据"""
    print(f"读取国家信息文件: {COUNTRY_FILE}")
    
    if not os.path.exists(COUNTRY_FILE):
        raise FileNotFoundError(f"国家信息文件不存在: {COUNTRY_FILE}")
    
    df = pd.read_csv(COUNTRY_FILE)
    print(f"成功加载国家信息，共 {len(df)} 个国家")
    
    # 只取前3列：Continent, Country_Code, Country_Name
    country_df = df[['Continent', 'Country_Code', 'Country_Name']].copy()
    
    return country_df


def filter_ilo_data(df):
    """筛选ILO数据"""
    print("\n开始筛选ILO数据...")
    
    original_count = len(df)
    
    # 筛选条件1：sex.label = "Total"
    df = df[df['sex.label'] == 'Total']
    print(f"筛选 sex.label='Total' 后，剩余 {len(df)} 行")
    
    # 筛选条件2：classif1.label 为指定值
    classif1_values = [
        'Occupation (ISCO-08): 7. Craft and related trades workers',
        'Occupation (ISCO-88): 7. Craft and related trades workers'
    ]
    df = df[df['classif1.label'].isin(classif1_values)]
    print(f"筛选 classif1.label 后，剩余 {len(df)} 行")
    
    # 筛选条件3：classif2.label = "Currency: U.S. dollars"
    df = df[df['classif2.label'] == 'Currency: U.S. dollars']
    print(f"筛选 classif2.label='Currency: U.S. dollars' 后，剩余 {len(df)} 行")
    
    # 筛选条件4：time列选择最接近2020年的年份
    print("\n处理年份数据，选择最接近2020年的数据...")
    
    # 确保time列是数值类型
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df[df['time'].notna()]
    
    # 计算与2020的距离
    df['time_diff'] = abs(df['time'] - 2020)
    
    # 按国家分组，选择最接近2020年的数据
    df = df.sort_values(['ref_area.label', 'time_diff', 'time'], ascending=[True, True, False])
    df = df.groupby('ref_area.label').first().reset_index()
    
    print(f"选择最接近2020年的数据后，剩余 {len(df)} 个国家")
    
    # 删除临时列
    df = df.drop(columns=['time_diff'])
    
    return df


def match_with_country_info(ilo_df, country_df):
    """将ILO数据与国家信息匹配"""
    print("\n开始匹配国家信息...")
    
    # 创建国家名称到国家信息的映射（标准化处理：去除空格、转小写）
    country_map = {}
    country_map_normalized = {}  # 用于模糊匹配
    
    for _, row in country_df.iterrows():
        country_name = str(row['Country_Name']).strip()
        country_code = str(row['Country_Code']).strip().upper()
        continent = str(row['Continent']).strip()
        
        # 使用国家名称作为key（精确匹配）
        country_map[country_name] = {
            'Continent': continent,
            'Country_Code': country_code,
            'Country_Name': country_name
        }
        
        # 创建标准化版本用于模糊匹配（去除空格、转小写）
        normalized_name = country_name.replace(' ', '').lower()
        if normalized_name not in country_map_normalized:
            country_map_normalized[normalized_name] = {
                'Continent': continent,
                'Country_Code': country_code,
                'Country_Name': country_name
            }
    
    # 匹配ILO数据
    matched_results = []
    matched_country_names = set()  # 记录已匹配的国家
    
    # 处理ILO数据中的国家
    for _, row in ilo_df.iterrows():
        ilo_country_name = str(row['ref_area.label']).strip()
        match_info = None
        
        # 首先尝试精确匹配
        if ilo_country_name in country_map:
            match_info = country_map[ilo_country_name]
        else:
            # 尝试标准化匹配
            normalized_ilo_name = ilo_country_name.replace(' ', '').lower()
            if normalized_ilo_name in country_map_normalized:
                match_info = country_map_normalized[normalized_ilo_name]
            else:
                print(f"警告：未找到匹配的国家信息: {ilo_country_name}")
        
        if match_info:
            matched_country_names.add(match_info['Country_Name'])
            matched_results.append({
                'Continent': match_info['Continent'],
                'Country_Code': match_info['Country_Code'],
                'Country_Name': match_info['Country_Name'],
                'source.label': row.get('source.label', ''),
                'sex.label': row.get('sex.label', ''),
                'classif1.label': row.get('classif1.label', ''),
                'classif2.label': row.get('classif2.label', ''),
                'time': row.get('time', ''),
                'obs_value': row.get('obs_value', ''),
                'obs_status.label': row.get('obs_status.label', ''),
                'note_indicator.label': row.get('note_indicator.label', ''),
                'note_source.label': row.get('note_source.label', '')
            })
    
    # 处理country_cost.csv中有但ILO数据中没有的国家
    country_cost_countries = set(country_df['Country_Name'].str.strip())
    missing_countries = country_cost_countries - matched_country_names
    
    if missing_countries:
        print(f"\n发现 {len(missing_countries)} 个国家在country_cost.csv中但不在ILO数据中，将添加空值记录")
        for country_name in missing_countries:
            country_info = country_df[country_df['Country_Name'].str.strip() == country_name].iloc[0]
            matched_results.append({
                'Continent': country_info['Continent'],
                'Country_Code': country_info['Country_Code'],
                'Country_Name': country_info['Country_Name'],
                'source.label': '',
                'sex.label': '',
                'classif1.label': '',
                'classif2.label': '',
                'time': '',
                'obs_value': '',
                'obs_status.label': '',
                'note_indicator.label': '',
                'note_source.label': ''
            })
    
    result_df = pd.DataFrame(matched_results)
    
    print(f"匹配完成，共 {len(result_df)} 个国家")
    print(f"  - 有ILO数据的国家: {len(result_df[result_df['obs_value'] != ''])}")
    print(f"  - 无ILO数据的国家: {len(result_df[result_df['obs_value'] == ''])}")
    
    return result_df


def save_results(df):
    """保存结果"""
    print(f"\n保存结果到: {OUTPUT_FILE}")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 按指定列顺序排列
    column_order = [
        'Continent', 'Country_Code', 'Country_Name', 'source.label',
        'sex.label', 'classif1.label', 'classif2.label', 'time',
        'obs_value', 'obs_status.label', 'note_indicator.label', 'note_source.label'
    ]
    
    # 确保所有列都存在
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    # 按Continent和Country_Name排序
    df = df.sort_values(['Continent', 'Country_Name'])
    
    # 保存为CSV
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print(f"结果已保存，共 {len(df)} 个国家")
    
    # 打印统计信息
    print("\n=== 统计信息 ===")
    print(f"有数据的国家数: {len(df[df['obs_value'] != ''])}")
    print(f"无数据的国家数: {len(df[df['obs_value'] == ''])}")
    
    # 按大洲统计
    print("\n按大洲统计（有数据的国家数）:")
    continent_stats = df[df['obs_value'] != ''].groupby('Continent').size()
    for continent, count in continent_stats.items():
        print(f"  {continent}: {count}")


def main():
    """主函数"""
    print("开始处理ILO平均小时工资数据...")
    print("=" * 60)
    
    try:
        # 1. 加载数据
        ilo_df = load_ilo_data()
        country_df = load_country_data()
        
        # 2. 筛选ILO数据
        filtered_ilo_df = filter_ilo_data(ilo_df)
        
        # 3. 匹配国家信息
        result_df = match_with_country_info(filtered_ilo_df, country_df)
        
        # 4. 保存结果
        save_results(result_df)
        
        print("\n" + "=" * 60)
        print("处理完成！")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()

