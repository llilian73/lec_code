"""
人均成本占GNI比例计算工具

功能概述：
本工具用于计算国家人均成本占人均GNI的比例。

输入数据：
1. 人均成本数据：
   - 文件：Z:\local_environment_creation\cost\country_yearly_per_cost_USD_PP.csv
   - 来源：4_year.py的输出
   - 包含total_cost列（人均年度综合成本）
   - Country_Code为二字母代码

2. 人均GNI数据：
   - 文件：Z:\local_environment_creation\cost\API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220\API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220.csv
   - 包含2015列（人均GNI）
   - Country Code_2为二字母代码（用于匹配）

3. 经济收入分组数据：
   - 文件：Z:\local_environment_creation\cost\API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220\Metadata_Country_API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220.csv
   - 包含IncomeGroup列
   - Country Code_2为二字母代码（用于匹配）

输出结果：
- 文件：Z:\local_environment_creation\cost\country_cost_proportion_to_GNI.csv
- 列：IncomeGroup,Country Name,Country Code_2,total_cost,GNI_per,propotion_to_GNI,PP
"""

import pandas as pd
import numpy as np
import os
import sys
import logging

# 设置日志记录
def setup_logging():
    """设置日志记录"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件处理器
    file_handler = logging.FileHandler('cal_propotion_to_GNI.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 初始化日志
logger = setup_logging()

# 配置文件路径
COST_FILE = r"Z:\local_environment_creation\cost\method2\country_yearly_per_cost_USD_PP.csv"
GNI_FILE = r"Z:\local_environment_creation\cost\API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220\API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220.csv"
METADATA_FILE = r"Z:\local_environment_creation\cost\API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220\Metadata_Country_API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220.csv"
OUTPUT_FILE = r"Z:\local_environment_creation\cost\method2\country_cost_proportion_to_GNI.csv"

def read_csv_with_encoding(file_path, keep_default_na=True):
    """读取CSV文件，尝试多种编码"""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk', 'gb2312', 'utf-8-sig']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, keep_default_na=keep_default_na)
            logger.info(f"成功使用编码 '{encoding}' 读取文件: {os.path.basename(file_path)}")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    raise ValueError(f"无法使用常见编码读取文件: {file_path}")

def load_cost_data():
    """加载人均成本数据"""
    logger.info("开始加载人均成本数据...")
    
    if not os.path.exists(COST_FILE):
        raise FileNotFoundError(f"成本数据文件不存在: {COST_FILE}")
    
    df = read_csv_with_encoding(COST_FILE, keep_default_na=False)
    logger.info(f"加载成本数据完成，共 {len(df)} 个国家")
    
    # 检查必要的列是否存在
    required_cols = ['Country_Code', 'total_cost']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"成本文件中缺少必要的列: {missing_cols}")
    
    # 检查PP列是否存在（可选）
    has_pp = 'PP' in df.columns
    if has_pp:
        logger.info("检测到成本文件中包含'PP'列，将提取PP数据")
    else:
        logger.warning("成本文件中没有'PP'列，将不会提取PP数据")
    
    # 创建Country Code_2（即Country_Code）到成本的映射
    # 注意：成本文件的Country_Code就是二字母代码
    cost_dict = {}
    for _, row in df.iterrows():
        # 处理Country_Code，确保'NA'不被当作缺失值
        country_code_raw = row['Country_Code']
        if country_code_raw == '' or (isinstance(country_code_raw, float) and pd.isna(country_code_raw)):
            country_code_2 = ''
        else:
            country_code_2 = str(country_code_raw).strip().upper()
        
        if country_code_2 and pd.notna(row['total_cost']):
            cost_dict[country_code_2] = {
                'total_cost': row['total_cost'],
                'Country_Name': row.get('Country_Name', ''),
                'Continent': row.get('Continent', '')
            }
            # 提取PP列（如果存在）
            if has_pp:
                cost_dict[country_code_2]['PP'] = row.get('PP', np.nan)
    
    logger.info(f"创建了 {len(cost_dict)} 个国家的成本数据映射")
    return cost_dict

def load_gni_data():
    """加载人均GNI数据"""
    logger.info("开始加载人均GNI数据...")
    
    if not os.path.exists(GNI_FILE):
        raise FileNotFoundError(f"GNI数据文件不存在: {GNI_FILE}")
    
    df = read_csv_with_encoding(GNI_FILE, keep_default_na=False)
    logger.info(f"加载GNI数据完成，共 {len(df)} 个国家/地区")
    
    # 检查必要的列是否存在
    required_cols = ['Country Code_2', '2015']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"GNI文件中缺少必要的列: {missing_cols}")
    
    # 创建Country Code_2到GNI的映射
    gni_dict = {}
    for _, row in df.iterrows():
        # 处理Country Code_2，确保'NA'不被当作缺失值
        code_2_raw = row['Country Code_2']
        if code_2_raw == '' or (isinstance(code_2_raw, float) and pd.isna(code_2_raw)):
            code_2 = ''
        else:
            code_2 = str(code_2_raw).strip().upper()
        
        gni_value = row['2015']
        
        if code_2 and pd.notna(gni_value) and gni_value != '':
            try:
                gni_dict[code_2] = float(gni_value)
            except (ValueError, TypeError):
                logger.warning(f"无法转换GNI值: {code_2} = {gni_value}")
    
    logger.info(f"创建了 {len(gni_dict)} 个国家的GNI数据映射")
    return gni_dict

def load_metadata():
    """加载元数据（经济收入分组）"""
    logger.info("开始加载元数据...")
    
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"元数据文件不存在: {METADATA_FILE}")
    
    df = read_csv_with_encoding(METADATA_FILE, keep_default_na=False)
    logger.info(f"加载元数据完成，共 {len(df)} 个国家/地区")
    
    # 检查必要的列是否存在
    required_cols = ['Country Code_2', 'IncomeGroup', 'Country Name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"元数据文件中缺少必要的列: {missing_cols}")
    
    # 创建Country Code_2到元数据的映射
    metadata_dict = {}
    for _, row in df.iterrows():
        # 处理Country Code_2，确保'NA'不被当作缺失值
        code_2_raw = row['Country Code_2']
        if code_2_raw == '' or (isinstance(code_2_raw, float) and pd.isna(code_2_raw)):
            code_2 = ''
        else:
            code_2 = str(code_2_raw).strip().upper()
        
        if code_2:
            metadata_dict[code_2] = {
                'IncomeGroup': row.get('IncomeGroup', ''),
                'Country_Name': row.get('Country Name', ''),
                'Region': row.get('Region', '')
            }
    
    logger.info(f"创建了 {len(metadata_dict)} 个国家的元数据映射")
    return metadata_dict

def process_data(cost_dict, gni_dict, metadata_dict):
    """处理数据，计算比例"""
    logger.info("开始处理数据并计算比例...")
    
    results = []
    
    # 使用成本数据的所有国家作为基准
    matched_count = 0
    missing_gni = 0
    missing_metadata = 0
    
    for code_2, cost_data in cost_dict.items():
        gni_value = gni_dict.get(code_2, None)
        metadata = metadata_dict.get(code_2, None)
        
        # 跳过没有GNI数据的国家
        if gni_value is None:
            missing_gni += 1
            logger.debug(f"国家 {code_2} 缺少GNI数据")
            continue
        
        # 获取元数据（IncomeGroup和Country Name）
        if metadata:
            income_group = metadata.get('IncomeGroup', '')
            country_name = metadata.get('Country_Name', '')
        else:
            # 如果元数据中没有，使用成本数据中的Country Name
            income_group = ''
            country_name = cost_data.get('Country_Name', '')
            missing_metadata += 1
        
        # 计算比例
        if gni_value > 0:
            proportion = cost_data['total_cost'] / gni_value
        else:
            proportion = np.nan
            logger.warning(f"国家 {country_name} ({code_2}) 的GNI为0或负数，无法计算比例")
        
        result_dict = {
            'IncomeGroup': income_group,
            'Country Name': country_name if country_name else cost_data.get('Country_Name', ''),
            'Country Code_2': code_2,
            'total_cost': cost_data['total_cost'],
            'GNI_per': gni_value,
            'propotion_to_GNI': proportion
        }
        # 添加PP列（如果存在）
        if 'PP' in cost_data:
            result_dict['PP'] = cost_data['PP']
        
        results.append(result_dict)
        
        matched_count += 1
    
    logger.info(f"成功匹配: {matched_count} 个国家")
    logger.info(f"缺少GNI数据: {missing_gni} 个")
    logger.info(f"缺少元数据: {missing_metadata} 个")
    
    return results

def save_results(results):
    """保存结果"""
    logger.info("开始保存结果...")
    
    if len(results) == 0:
        logger.warning("没有数据可保存")
        return
    
    results_df = pd.DataFrame(results)
    
    # 确保列的顺序符合要求：IncomeGroup,Country Name,Country Code_2,total_cost,GNI_per,propotion_to_GNI,PP
    column_order = ['IncomeGroup', 'Country Name', 'Country Code_2', 'total_cost', 'GNI_per', 'propotion_to_GNI', 'PP']
    existing_columns = [col for col in column_order if col in results_df.columns]
    results_df = results_df[existing_columns]
    
    # 按IncomeGroup和Country Name排序
    results_df = results_df.sort_values(['IncomeGroup', 'Country Name'])
    
    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_FILE)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为CSV
    results_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    # 验证列顺序
    logger.info(f"输出CSV文件的列顺序: {list(results_df.columns)}")
    
    logger.info(f"结果已保存至: {OUTPUT_FILE}")
    logger.info(f"共保存 {len(results_df)} 个国家/地区的数据")
    
    # 统计信息
    logger.info("=== 统计信息 ===")
    logger.info(f"比例范围: {results_df['propotion_to_GNI'].min():.6f} - {results_df['propotion_to_GNI'].max():.6f}")
    logger.info(f"平均比例: {results_df['propotion_to_GNI'].mean():.6f}")
    logger.info(f"中位数比例: {results_df['propotion_to_GNI'].median():.6f}")
    
    # 按IncomeGroup统计
    if 'IncomeGroup' in results_df.columns:
        logger.info("\n按IncomeGroup统计:")
        income_group_stats = results_df.groupby('IncomeGroup').agg({
            'propotion_to_GNI': ['mean', 'median', 'count']
        })
        for group in income_group_stats.index:
            avg_prop = income_group_stats.loc[group, ('propotion_to_GNI', 'mean')]
            median_prop = income_group_stats.loc[group, ('propotion_to_GNI', 'median')]
            count = income_group_stats.loc[group, ('propotion_to_GNI', 'count')]
            logger.info(f"  {group}: 平均比例 {avg_prop:.6f}, 中位数比例 {median_prop:.6f}, 国家数 {count}")
    
    # 显示比例最高和最低的前10个国家
    logger.info("\n比例最高的前10个国家:")
    top_countries = results_df.nlargest(10, 'propotion_to_GNI')
    for _, row in top_countries.iterrows():
        logger.info(f"  {row['Country Name']} ({row['Country Code_2']}): {row['propotion_to_GNI']:.6f}")
    
    logger.info("\n比例最低的前10个国家:")
    bottom_countries = results_df.nsmallest(10, 'propotion_to_GNI')
    for _, row in bottom_countries.iterrows():
        logger.info(f"  {row['Country Name']} ({row['Country Code_2']}): {row['propotion_to_GNI']:.6f}")

def main():
    """主函数"""
    logger.info("开始计算人均成本占GNI比例...")
    
    try:
        # 1. 加载数据
        logger.info("=== 第一步：加载数据 ===")
        
        cost_dict = load_cost_data()
        gni_dict = load_gni_data()
        metadata_dict = load_metadata()
        
        # 2. 处理数据
        logger.info("=== 第二步：处理数据 ===")
        
        results = process_data(cost_dict, gni_dict, metadata_dict)
        
        # 3. 保存结果
        logger.info("=== 第三步：保存结果 ===")
        
        save_results(results)
        
        logger.info("计算完成！")
        
    except Exception as e:
        error_msg = f"主程序执行出错: {str(e)}"
        logger.error(error_msg)
        raise

if __name__ == "__main__":
    main()