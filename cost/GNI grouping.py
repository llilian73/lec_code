"""
GNI元数据分组处理工具

功能概述：
本工具用于处理GNI元数据CSV文件，添加Country Code_2列并重新排序。

输入数据：
1. 元数据文件：
   - 文件：Z:\local_environment_creation\cost\API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220\Metadata_Country_API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220.csv
   
2. 参考文件（用于匹配Country Code_2）：
   - 文件：Z:\local_environment_creation\cost\API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220\API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220.csv

处理过程：
- 读取元数据文件
- 将TableName列改名为Country Name
- 从参考文件中根据Country Code匹配Country Code_2
- 调整列顺序为：IncomeGroup,Country Name,Country Code,Country Code_2,Region
- 按照IncomeGroup排序，无Country Code_2的排在最后面
- 保存更新后的CSV文件
"""

import pandas as pd
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
    file_handler = logging.FileHandler('GNI_grouping.log', encoding='utf-8')
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
METADATA_FILE = r"Z:\local_environment_creation\cost\API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220\Metadata_Country_API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220.csv"
REFERENCE_FILE = r"Z:\local_environment_creation\cost\API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220\API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220.csv"

def load_reference_data():
    """加载参考数据，用于匹配Country Code_2"""
    logger.info("开始加载参考数据...")
    
    if not os.path.exists(REFERENCE_FILE):
        raise FileNotFoundError(f"参考文件不存在: {REFERENCE_FILE}")
    
    # 尝试多种编码格式
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk', 'gb2312', 'utf-8-sig']
    ref_df = None
    used_encoding = None
    
    for encoding in encodings:
        try:
            ref_df = pd.read_csv(REFERENCE_FILE, encoding=encoding)
            used_encoding = encoding
            logger.info(f"成功使用编码 '{encoding}' 读取参考文件")
            break
        except (UnicodeDecodeError, UnicodeError):
            logger.debug(f"编码 '{encoding}' 失败，尝试下一个...")
            continue
    
    if ref_df is None:
        raise ValueError(f"无法使用常见编码读取参考文件，请检查文件编码格式")
    
    logger.info(f"成功加载参考数据，共 {len(ref_df)} 行（使用编码: {used_encoding}）")
    
    # 检查必要的列是否存在
    required_cols = ['Country Code', 'Country Code_2']
    missing_cols = [col for col in required_cols if col not in ref_df.columns]
    if missing_cols:
        raise ValueError(f"参考文件中缺少必要的列: {missing_cols}")
    
    # 创建Country Code到Country Code_2的映射
    code_to_code2 = {}
    for _, row in ref_df.iterrows():
        country_code = row['Country Code']
        country_code_2 = row['Country Code_2']
        
        # 只添加非空的映射
        if pd.notna(country_code) and pd.notna(country_code_2) and country_code_2 != '':
            code_to_code2[country_code] = country_code_2
    
    logger.info(f"创建了 {len(code_to_code2)} 个Country Code到Country Code_2的映射")
    
    return code_to_code2

def process_metadata_file():
    """处理元数据文件"""
    logger.info("开始处理元数据文件...")
    
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"元数据文件不存在: {METADATA_FILE}")
    
    # 读取元数据文件（尝试多种编码）
    logger.info(f"读取文件: {METADATA_FILE}")
    
    # 尝试多种编码格式
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk', 'gb2312', 'utf-8-sig']
    df = None
    used_encoding = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(METADATA_FILE, encoding=encoding)
            used_encoding = encoding
            logger.info(f"成功使用编码 '{encoding}' 读取文件")
            break
        except (UnicodeDecodeError, UnicodeError):
            logger.debug(f"编码 '{encoding}' 失败，尝试下一个...")
            continue
    
    if df is None:
        raise ValueError(f"无法使用常见编码读取文件，请检查文件编码格式")
    
    logger.info(f"成功读取文件，共 {len(df)} 行，{len(df.columns)} 列（使用编码: {used_encoding}）")
    logger.info(f"原始列名: {list(df.columns)}")
    
    # 检查必要的列是否存在
    if 'TableName' not in df.columns:
        raise ValueError("元数据文件中缺少'TableName'列")
    
    if 'Country Code' not in df.columns:
        raise ValueError("元数据文件中缺少'Country Code'列")
    
    # 加载参考数据
    code_to_code2 = load_reference_data()
    
    # 将TableName列改名为Country Name
    if 'TableName' in df.columns:
        df = df.rename(columns={'TableName': 'Country Name'})
        logger.info("已将TableName列改名为Country Name")
    
    # 添加Country Code_2列
    logger.info("开始匹配Country Code_2...")
    country_codes_2 = []
    matched_count = 0
    unmatched_count = 0
    
    for _, row in df.iterrows():
        country_code = row['Country Code']
        
        if pd.notna(country_code) and country_code in code_to_code2:
            country_code_2 = code_to_code2[country_code]
            country_codes_2.append(country_code_2)
            matched_count += 1
        else:
            country_codes_2.append('')
            unmatched_count += 1
    
    df['Country Code_2'] = country_codes_2
    
    logger.info(f"成功匹配: {matched_count} 个国家")
    logger.info(f"未匹配: {unmatched_count} 个国家/地区")
    
    # 调整列顺序：IncomeGroup,Country Name,Country Code,Country Code_2,Region
    # 先检查所有必要的列是否存在
    required_cols = ['IncomeGroup', 'Country Name', 'Country Code', 'Country Code_2', 'Region']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"缺少以下列，将不会包含在输出中: {missing_cols}")
        available_cols = [col for col in required_cols if col in df.columns]
        # 获取其他列
        other_cols = [col for col in df.columns if col not in required_cols]
        df = df[available_cols + other_cols]
    else:
        # 获取其他列（如果有）
        other_cols = [col for col in df.columns if col not in required_cols]
        df = df[required_cols + other_cols]
    
    logger.info(f"调整后的列顺序: {list(df.columns)}")
    
    # 排序：按照IncomeGroup排序，无Country Code_2的排在最后面
    logger.info("开始排序...")
    
    # 先创建排序辅助列
    df['_has_code2'] = df['Country Code_2'].apply(lambda x: 1 if pd.notna(x) and x != '' else 0)
    
    # 对IncomeGroup进行排序，空值排在最后
    df['_income_group_order'] = df['IncomeGroup'].fillna('ZZZ_No_Group')
    
    # 排序：先按_has_code2降序（有code2的在前），再按IncomeGroup升序，最后按Country Code
    df = df.sort_values(
        by=['_has_code2', '_income_group_order', 'Country Code'],
        ascending=[False, True, True]
    )
    
    # 删除辅助列
    df = df.drop(columns=['_has_code2', '_income_group_order'])
    
    logger.info("排序完成")
    
    # 显示统计信息
    logger.info("=== 统计信息 ===")
    logger.info(f"总行数: {len(df)}")
    
    # 按IncomeGroup统计
    if 'IncomeGroup' in df.columns:
        income_group_stats = df.groupby('IncomeGroup').size()
        logger.info("按IncomeGroup统计:")
        for group, count in income_group_stats.items():
            logger.info(f"  {group}: {count} 个国家")
    
    # 统计有/无Country Code_2的数量
    has_code2_count = df['Country Code_2'].apply(lambda x: 1 if pd.notna(x) and x != '' else 0).sum()
    no_code2_count = len(df) - has_code2_count
    logger.info(f"有Country Code_2: {has_code2_count} 个国家")
    logger.info(f"无Country Code_2: {no_code2_count} 个国家/地区")
    
    # 显示前几行数据作为预览
    logger.info("\n数据预览（前10行）:")
    logger.info(f"\n{df.head(10).to_string()}")
    
    return df

def save_results(df):
    """保存结果"""
    logger.info("保存更新后的CSV文件...")
    
    df.to_csv(METADATA_FILE, index=False, encoding='utf-8-sig')
    logger.info(f"文件已保存至: {METADATA_FILE}")

def main():
    """主函数"""
    logger.info("开始处理GNI元数据文件...")
    
    try:
        # 处理元数据文件
        df = process_metadata_file()
        
        # 保存结果
        save_results(df)
        
        logger.info("处理完成！")
        
    except Exception as e:
        error_msg = f"主程序执行出错: {str(e)}"
        logger.error(error_msg)
        raise

if __name__ == "__main__":
    main()
