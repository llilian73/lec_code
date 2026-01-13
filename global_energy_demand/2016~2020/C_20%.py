import os
import pandas as pd
import numpy as np


def process_20_percent_reduction(input_dir, output_dir):
    """统计总节能率大于等于20%的国家占比"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有_with_continent.csv文件
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('_with_continent.csv')]
    
    if not csv_files:
        print(f"在 {input_dir} 中未找到_with_continent.csv文件")
        return
    
    print(f"找到 {len(csv_files)} 个_with_continent.csv文件")
    
    results = []
    
    # 处理每个文件
    for csv_file in csv_files:
        print(f"正在处理: {csv_file}")
        
        # 读取数据
        csv_path = os.path.join(input_dir, csv_file)
        df = pd.read_csv(csv_path)
        
        # 提取case名称（去掉_with_continent.csv后缀）
        case_name = csv_file.replace('_with_continent.csv', '')
        
        # 处理空值：将NaN值视为0
        df_clean = df.copy()
        df_clean['total_reduction'] = pd.to_numeric(df_clean['total_reduction'], errors='coerce').fillna(0)
        
        # 统计总国家数
        total_countries = len(df_clean)
        
        # 统计节能率大于等于20%的国家数
        countries_above_20 = len(df_clean[df_clean['total_reduction'] >= 20])
        
        # 计算百分比
        percentage_above_20 = (countries_above_20 / total_countries) * 100 if total_countries > 0 else 0
        
        results.append({
            'case': case_name,
            '>20%': percentage_above_20
        })
        
        print(f"  {case_name}: {countries_above_20}/{total_countries} = {percentage_above_20:.2f}%")
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(results)
    
    # 按照case1到case20排序
    def extract_case_number(case_name):
        """提取case名称中的数字用于排序"""
        if case_name.startswith('case'):
            try:
                return int(case_name[4:])  # 去掉'case'前缀，提取数字
            except ValueError:
                return 999  # 如果不是数字，放到最后
        return 999
    
    result_df['case_num'] = result_df['case'].apply(extract_case_number)
    result_df = result_df.sort_values('case_num').drop('case_num', axis=1)
    
    # 保存结果
    output_filename = "20_percent_reduction_summary.csv"
    output_path = os.path.join(output_dir, output_filename)
    result_df.to_csv(output_path, index=False)
    
    print(f"\n已保存结果到: {output_path}")
    print(f"处理了 {len(results)} 个case文件")


def main():
    # 输入目录（包含_with_continent.csv文件）
    input_dir = r"Z:\local_environment_creation\energy_consumption_gird\result\result\reduction_rate"
    
    # 输出目录（保存20%统计结果）
    output_dir = r"Z:\local_environment_creation\energy_consumption_gird\result\result\reduction_rate\20%"
    
    print("开始统计节能率大于等于20%的国家占比...")
    process_20_percent_reduction(input_dir, output_dir)
    print("\n统计完成！")


if __name__ == '__main__':
    main()
