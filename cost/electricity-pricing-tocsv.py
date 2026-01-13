"""
全球电价数据转换工具

功能概述：
本工具用于从Excel文件中提取全球电价数据，并转换为CSV格式。

输入数据：
- 文件：Z:\local_environment_creation\cost\global-electricity-per-kwh-pricing-2021.xlsx
- 工作表：Global energy pricing league

输出结果：
- 文件：Z:\local_environment_creation\cost\global-electricity-per-kwh-pricing-2021.csv
- 列：rank, Country code, Country name, Continental region, Average price of 1KW/h (USD)
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
    file_handler = logging.FileHandler('electricity-pricing-tocsv.log', encoding='utf-8')
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

# 配置参数
INPUT_FILE = r"Z:\local_environment_creation\cost\global-electricity-per-kwh-pricing-2021.xlsx"
OUTPUT_FILE = r"Z:\local_environment_creation\cost\global-electricity-per-kwh-pricing-2021.csv"
SHEET_NAME = "Global energy pricing league"

def load_excel_data():
    """加载Excel数据"""
    logger.info("开始加载Excel数据...")
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Excel文件不存在: {INPUT_FILE}")
    
    try:
        # 读取Excel文件的第一个工作表
        df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
        logger.info(f"成功加载Excel数据，共 {len(df)} 行，{len(df.columns)} 列")
        
        # 显示列名
        logger.info(f"Excel文件列名: {list(df.columns)}")
        
        return df
    except Exception as e:
        logger.error(f"加载Excel文件出错: {str(e)}")
        raise

def extract_required_columns(df):
    """提取需要的列"""
    logger.info("开始提取需要的列...")
    
    # 定义需要的列名（可能在不同位置）
    required_columns = [
        'rank', 'Country code', 'Country name', 'Continental region', 
        'Average price of 1KW/h (USD)'
    ]
    
    # 查找实际存在的列名
    available_columns = list(df.columns)
    logger.info(f"可用的列名: {available_columns}")
    
    # 创建列名映射
    column_mapping = {}
    for req_col in required_columns:
        # 尝试精确匹配
        if req_col in available_columns:
            column_mapping[req_col] = req_col
        else:
            # 尝试模糊匹配
            for avail_col in available_columns:
                if req_col.lower() in avail_col.lower() or avail_col.lower() in req_col.lower():
                    column_mapping[req_col] = avail_col
                    logger.info(f"列名映射: '{req_col}' -> '{avail_col}'")
                    break
    
    # 检查是否找到了所有需要的列
    missing_columns = []
    for req_col in required_columns:
        if req_col not in column_mapping:
            missing_columns.append(req_col)
    
    if missing_columns:
        logger.warning(f"未找到以下列: {missing_columns}")
        logger.info("尝试使用前几列作为备选...")
        
        # 使用前几列作为备选
        if len(available_columns) >= 5:
            column_mapping = {
                'rank': available_columns[0],
                'Country code': available_columns[1],
                'Country name': available_columns[2],
                'Continental region': available_columns[3],
                'Average price of 1KW/h (USD)': available_columns[4]
            }
            logger.info(f"使用备选列名映射: {column_mapping}")
    
    # 提取数据
    try:
        extracted_df = df[list(column_mapping.values())].copy()
        
        # 重命名列为标准名称
        extracted_df.columns = required_columns
        
        # 清理数据
        extracted_df = extracted_df.dropna(how='all')  # 删除全空行
        
        logger.info(f"成功提取数据，共 {len(extracted_df)} 行")
        logger.info(f"提取的列: {list(extracted_df.columns)}")
        
        return extracted_df
        
    except Exception as e:
        logger.error(f"提取列数据出错: {str(e)}")
        logger.info("尝试手动指定列索引...")
        
        # 手动指定列索引（假设前5列就是需要的列）
        if len(df.columns) >= 5:
            extracted_df = df.iloc[:, :5].copy()
            extracted_df.columns = required_columns
            extracted_df = extracted_df.dropna(how='all')
            
            logger.info(f"使用手动列索引提取数据，共 {len(extracted_df)} 行")
            return extracted_df
        else:
            raise Exception("Excel文件列数不足，无法提取所需数据")

def save_to_csv(df):
    """保存为CSV文件"""
    logger.info("开始保存CSV文件...")
    
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(OUTPUT_FILE)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为CSV
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        
        logger.info(f"CSV文件已保存至: {OUTPUT_FILE}")
        logger.info(f"共保存 {len(df)} 行数据")
        
        # 显示前几行数据作为预览
        logger.info("数据预览:")
        logger.info(f"列名: {list(df.columns)}")
        logger.info("前5行数据:")
        for i, row in df.head().iterrows():
            logger.info(f"  行{i+1}: {dict(row)}")
        
        # 统计信息
        logger.info("=== 统计信息 ===")
        logger.info(f"总行数: {len(df)}")
        logger.info(f"总列数: {len(df.columns)}")
        
        # 检查是否有缺失值
        missing_info = df.isnull().sum()
        if missing_info.sum() > 0:
            logger.info("缺失值统计:")
            for col, missing_count in missing_info.items():
                if missing_count > 0:
                    logger.info(f"  {col}: {missing_count} 个缺失值")
        else:
            logger.info("没有缺失值")
        
    except Exception as e:
        logger.error(f"保存CSV文件出错: {str(e)}")
        raise

def main():
    """主函数"""
    logger.info("开始全球电价数据转换...")
    
    try:
        # 1. 加载Excel数据
        logger.info("=== 第一步：加载Excel数据 ===")
        df = load_excel_data()
        
        # 2. 提取需要的列
        logger.info("=== 第二步：提取需要的列 ===")
        extracted_df = extract_required_columns(df)
        
        # 3. 保存为CSV
        logger.info("=== 第三步：保存为CSV ===")
        save_to_csv(extracted_df)
        
        logger.info("全球电价数据转换完成！")
        
    except Exception as e:
        error_msg = f"主程序执行出错: {str(e)}"
        logger.error(error_msg)
        raise

if __name__ == "__main__":
    main()
