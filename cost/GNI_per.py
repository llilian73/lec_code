"""
GNI数据添加国家二字母代码工具

功能概述：
本工具用于在GNI数据CSV文件中添加国家二字母代码列。

输入数据：
- 文件：Z:\local_environment_creation\cost\API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220\API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220.csv

处理过程：
- 读取CSV文件
- 根据Country Name使用pycountry库获取ISO二字母代码
- 在Country Code列后插入Country Code_2列
- 保存更新后的CSV文件
"""

import pandas as pd
import pycountry
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
    file_handler = logging.FileHandler('GNI_per.log', encoding='utf-8')
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
INPUT_FILE = r"Z:\local_environment_creation\cost\API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220\API_NY.GNP.PCAP.KD_DS2_en_csv_v2_124220.csv"

def get_country_code_from_name(country_name):
    """根据国家名称获取ISO二字母代码"""
    # 特殊映射 - 处理一些特殊情况
    special_mappings = {
        'Aruba': 'AW',
        'Africa Eastern and Southern': 'AFE',  # 区域代码
        'Africa Western and Central': 'AFW',  # 区域代码
        'American Samoa': 'AS',
        'Antigua and Barbuda': 'AG',
        'Arab World': 'ARB',  # 区域代码
        'Bahamas, The': 'BS',
        'Brunei Darussalam': 'BN',
        'Cabo Verde': 'CV',
        'Caribbean small states': 'CSS',  # 区域代码
        'Central African Republic': 'CF',
        'Congo, Dem. Rep.': 'CD',
        'Congo, Rep.': 'CG',
        'Czech Republic': 'CZ',
        'Egypt, Arab Rep.': 'EG',
        'Faeroe Islands': 'FO',
        'Gambia, The': 'GM',
        'Hong Kong SAR, China': 'HK',
        'Iran, Islamic Rep.': 'IR',
        'Korea, Dem. People\'s Rep.': 'KP',
        'Korea, Rep.': 'KR',
        'Kyrgyz Republic': 'KG',
        'Lao PDR': 'LA',
        'Libya': 'LY',
        'Macao SAR, China': 'MO',
        'Micronesia, Fed. Sts.': 'FM',
        'Moldova': 'MD',
        'North Macedonia': 'MK',
        'Russian Federation': 'RU',
        'St. Kitts and Nevis': 'KN',
        'St. Lucia': 'LC',
        'St. Martin (French part)': 'MF',
        'St. Vincent and the Grenadines': 'VC',
        'Syrian Arab Republic': 'SY',
        'Trinidad and Tobago': 'TT',
        'Turks and Caicos Islands': 'TC',
        'United Kingdom': 'GB',
        'United States': 'US',
        'Venezuela, RB': 'VE',
        'Virgin Islands (U.S.)': 'VI',
        'West Bank and Gaza': 'PS',
        'Yemen, Rep.': 'YE',
        'East Asia & Pacific (excluding high income)': 'EAP',  # 区域代码
        'Europe & Central Asia (excluding high income)': 'ECA',  # 区域代码
        'Latin America & Caribbean (excluding high income)': 'LAC',  # 区域代码
        'Middle East & North Africa (excluding high income)': 'MNA',  # 区域代码
        'South Asia': 'SAS',  # 区域代码
        'Sub-Saharan Africa (excluding high income)': 'SSA',  # 区域代码
    }
    
    # 首先检查特殊映射
    if country_name in special_mappings:
        return special_mappings[country_name]
    
    try:
        # 使用pycountry库进行转换
        country = pycountry.countries.get(name=country_name)
        if country:
            return country.alpha_2
        else:
            # 尝试模糊匹配
            try:
                country = pycountry.countries.search_fuzzy(country_name)
                if country:
                    return country[0].alpha_2
            except LookupError:
                pass
            
            logger.warning(f"未找到国家 '{country_name}' 对应的ISO代码")
            return ''
    except Exception as e:
        logger.warning(f"转换国家名称 '{country_name}' 时出错: {e}")
        return ''

def process_gni_file():
    """处理GNI文件，添加国家二字母代码列"""
    logger.info("开始处理GNI文件...")
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"输入文件不存在: {INPUT_FILE}")
    
    # 读取CSV文件
    logger.info(f"读取文件: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    logger.info(f"成功读取文件，共 {len(df)} 行，{len(df.columns)} 列")
    
    # 检查必要的列是否存在
    if 'Country Name' not in df.columns:
        raise ValueError("CSV文件中缺少'Country Name'列")
    
    if 'Country Code' not in df.columns:
        raise ValueError("CSV文件中缺少'Country Code'列")
    
    # 创建Country Code_2列
    logger.info("开始生成Country Code_2列...")
    country_codes_2 = []
    
    for idx, row in df.iterrows():
        country_name = row['Country Name']
        country_code_2 = get_country_code_from_name(country_name)
        country_codes_2.append(country_code_2)
        
        if (idx + 1) % 100 == 0:
            logger.info(f"已处理 {idx + 1}/{len(df)} 个国家")
    
    # 找到Country Code列的位置
    country_code_idx = df.columns.get_loc('Country Code')
    
    # 在Country Code列后插入新列
    df.insert(country_code_idx + 1, 'Country Code_2', country_codes_2)
    
    logger.info(f"成功添加Country Code_2列")
    
    # 统计信息
    non_empty_codes = sum(1 for code in country_codes_2 if code)
    empty_codes = len(country_codes_2) - non_empty_codes
    
    logger.info(f"成功匹配: {non_empty_codes} 个国家")
    logger.info(f"未匹配: {empty_codes} 个国家/地区")
    
    # 显示未匹配的国家（前20个）
    if empty_codes > 0:
        logger.info("未匹配的国家/地区（前20个）:")
        unmatched_count = 0
        for idx, row in df.iterrows():
            if not row['Country Code_2']:
                logger.info(f"  {row['Country Name']}")
                unmatched_count += 1
                if unmatched_count >= 20:
                    break
    
    # 保存更新后的CSV文件
    logger.info("保存更新后的CSV文件...")
    df.to_csv(INPUT_FILE, index=False, encoding='utf-8-sig')
    logger.info(f"文件已保存至: {INPUT_FILE}")
    
    # 显示前几行数据作为预览
    logger.info("数据预览（前5行）:")
    preview_cols = ['Country Name', 'Country Code', 'Country Code_2']
    if len(df.columns) > 3:
        preview_cols.append(df.columns[3])  # 添加第一年数据列
    logger.info(f"\n{df[preview_cols].head().to_string()}")
    
    return df

def main():
    """主函数"""
    logger.info("开始处理GNI文件，添加国家二字母代码...")
    
    try:
        # 处理GNI文件
        df = process_gni_file()
        
        logger.info("处理完成！")
        
    except Exception as e:
        error_msg = f"主程序执行出错: {str(e)}"
        logger.error(error_msg)
        raise

if __name__ == "__main__":
    main()
