"""
ILO时薪数据按收入组处理和填充工具

功能概述：
本工具用于根据收入组对ILO时薪数据进行处理和填充缺失值。

输入数据：
- ILO数据：Z:\local_environment_creation\cost\Earnings\ILO_Average_hourly_earnings_processed.csv
- 元数据：Z:\local_environment_creation\cost\API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220\Metadata_Country_API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220.csv

输出结果：
- CSV文件：Z:\local_environment_creation\cost\Earnings\ILO_Average_hourly_earnings_processed.csv（更新后的文件）
"""

import os
import pandas as pd
import numpy as np

# 配置文件路径
ILO_FILE = r"Z:\local_environment_creation\cost\Earnings\ILO_Average_hourly_earnings_processed.csv"
METADATA_FILE = r"Z:\local_environment_creation\cost\API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220\Metadata_Country_API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220.csv"


def read_csv_with_encoding(file_path):
    """读取CSV文件，尝试多种编码"""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk', 'gb2312', 'utf-8-sig']
    
    for encoding in encodings:
        try:
            # 使用keep_default_na=False确保'NA'不被当作缺失值
            df = pd.read_csv(file_path, encoding=encoding, keep_default_na=False)
            print(f"成功使用编码 '{encoding}' 读取文件: {os.path.basename(file_path)}")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    raise ValueError(f"无法使用常见编码读取文件: {file_path}")


def load_metadata():
    """加载元数据文件"""
    print(f"读取元数据文件: {METADATA_FILE}")
    
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"元数据文件不存在: {METADATA_FILE}")
    
    df = read_csv_with_encoding(METADATA_FILE)
    print(f"成功加载元数据，共 {len(df)} 个国家")
    
    # 检查必要的列是否存在
    required_cols = ['Country Code_2', 'IncomeGroup']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"元数据文件中缺少必要的列: {missing_cols}")
    
    # 创建Country Code_2到IncomeGroup的映射
    metadata_map = {}
    for _, row in df.iterrows():
        # 处理Country Code_2，确保'NA'不被当作缺失值
        code_2_raw = row['Country Code_2']
        if code_2_raw == '' or (isinstance(code_2_raw, float) and pd.isna(code_2_raw)):
            code_2 = ''
        else:
            code_2 = str(code_2_raw).strip().upper()
        
        # 处理IncomeGroup
        income_group_raw = row['IncomeGroup']
        if income_group_raw == '' or (isinstance(income_group_raw, float) and pd.isna(income_group_raw)):
            income_group = ''
        else:
            income_group = str(income_group_raw).strip()
        
        if code_2 and income_group:
            metadata_map[code_2] = income_group
    
    print(f"创建了 {len(metadata_map)} 个国家的收入组映射")
    return metadata_map


def load_ilo_data():
    """加载ILO数据"""
    print(f"\n读取ILO数据文件: {ILO_FILE}")
    
    if not os.path.exists(ILO_FILE):
        raise FileNotFoundError(f"ILO数据文件不存在: {ILO_FILE}")
    
    df = read_csv_with_encoding(ILO_FILE)
    print(f"成功加载ILO数据，共 {len(df)} 个国家")
    
    # 检查必要的列是否存在
    if 'Country_Code' not in df.columns:
        raise ValueError("ILO数据文件中缺少'Country_Code'列")
    if 'obs_value' not in df.columns:
        raise ValueError("ILO数据文件中缺少'obs_value'列")
    
    # 在数据处理之前，将Namibia的Country_Code修正为'NA'
    if 'Country_Name' in df.columns:
        namibia_mask = df['Country_Name'].str.strip().str.lower() == 'namibia'
        namibia_count = namibia_mask.sum()
        if namibia_count > 0:
            df.loc[namibia_mask, 'Country_Code'] = 'NA'
            print(f"修正了 {namibia_count} 条Namibia记录的Country_Code为'NA'")
    
    return df


def process_ilo_data(df, metadata_map):
    """处理ILO数据：添加IncomeGroup，计算平均值，填充缺失值"""
    print("\n开始处理ILO数据...")
    
    # 添加IncomeGroup列（在第一列之后）
    df['IncomeGroup'] = ''
    
    # 匹配IncomeGroup
    matched_count = 0
    for idx, row in df.iterrows():
        # 处理Country_Code，确保'NA'不被当作缺失值
        country_code_raw = row['Country_Code']
        if country_code_raw == '' or (isinstance(country_code_raw, float) and pd.isna(country_code_raw)):
            country_code = ''
        else:
            country_code = str(country_code_raw).strip().upper()
        
        if country_code in metadata_map:
            df.at[idx, 'IncomeGroup'] = metadata_map[country_code]
            matched_count += 1
    
    print(f"成功匹配 {matched_count} 个国家的收入组")
    
    # 转换obs_value为数值类型
    df['obs_value_numeric'] = pd.to_numeric(df['obs_value'], errors='coerce')
    
    # 按IncomeGroup计算平均值（排除空值）
    income_group_averages = {}
    for income_group in df['IncomeGroup'].unique():
        if income_group and income_group != '':
            group_data = df[(df['IncomeGroup'] == income_group) & (df['obs_value_numeric'].notna())]
            if len(group_data) > 0:
                avg_value = group_data['obs_value_numeric'].mean()
                income_group_averages[income_group] = avg_value
                print(f"  {income_group}: 平均时薪 = {avg_value:.2f} (基于 {len(group_data)} 个国家)")
    
    # 填充缺失值
    filled_count = 0
    for idx, row in df.iterrows():
        # 如果obs_value为空或无法转换为数值
        if pd.isna(row['obs_value_numeric']) or row['obs_value'] == '':
            income_group = row['IncomeGroup']
            if income_group and income_group != '' and income_group in income_group_averages:
                avg_value = income_group_averages[income_group]
                df.at[idx, 'obs_value'] = str(avg_value)
                filled_count += 1
    
    print(f"\n填充了 {filled_count} 个国家的缺失时薪值")
    
    # 删除临时列
    df = df.drop(columns=['obs_value_numeric'])
    
    # 重新排列列顺序：将IncomeGroup放在第一列之后
    columns = df.columns.tolist()
    # 移除IncomeGroup
    columns = [col for col in columns if col != 'IncomeGroup']
    # 在第一列（Continent）之后插入IncomeGroup
    continent_idx = columns.index('Continent')
    columns.insert(continent_idx + 1, 'IncomeGroup')
    df = df[columns]
    
    return df


def save_results(df):
    """保存结果"""
    print(f"\n保存结果到: {ILO_FILE}")
    
    # 按Continent和Country_Name排序
    df = df.sort_values(['Continent', 'Country_Name'])
    
    # 保存为CSV（覆盖原文件）
    df.to_csv(ILO_FILE, index=False, encoding='utf-8-sig')
    
    print(f"结果已保存，共 {len(df)} 个国家")
    
    # 打印统计信息
    print("\n=== 统计信息 ===")
    print(f"总国家数: {len(df)}")
    
    # 按收入组统计
    print("\n按收入组统计:")
    income_group_stats = df.groupby('IncomeGroup').agg({
        'obs_value': lambda x: len([v for v in x if v != '' and pd.notna(v)])
    })
    income_group_stats.columns = ['有数据的国家数']
    income_group_stats['总国家数'] = df.groupby('IncomeGroup').size()
    
    for income_group in income_group_stats.index:
        if income_group and income_group != '':
            total = income_group_stats.loc[income_group, '总国家数']
            with_data = income_group_stats.loc[income_group, '有数据的国家数']
            print(f"  {income_group}: {with_data}/{total} 个国家有时薪数据")
    
    # 显示填充后的平均值
    print("\n填充后的平均时薪（按收入组）:")
    for income_group in df['IncomeGroup'].unique():
        if income_group and income_group != '':
            group_data = df[df['IncomeGroup'] == income_group].copy()
            # 转换obs_value为数值
            group_data['obs_value_numeric'] = pd.to_numeric(group_data['obs_value'], errors='coerce')
            valid_data = group_data[group_data['obs_value_numeric'].notna()]
            if len(valid_data) > 0:
                avg_value = valid_data['obs_value_numeric'].mean()
                print(f"  {income_group}: {avg_value:.2f} USD")


def main():
    """主函数"""
    print("开始处理ILO时薪数据...")
    print("=" * 60)
    
    try:
        # 1. 加载数据
        metadata_map = load_metadata()
        ilo_df = load_ilo_data()
        
        # 2. 处理数据
        processed_df = process_ilo_data(ilo_df, metadata_map)
        
        # 3. 保存结果
        save_results(processed_df)
        
        print("\n" + "=" * 60)
        print("处理完成！")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()

