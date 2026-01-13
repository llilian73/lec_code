import os
import pandas as pd
import numpy as np


def load_country_continent_mapping(mapping_file):
    """从CSV文件加载国家代码到大洲的映射"""
    df = pd.read_csv(mapping_file)
    return dict(zip(df['Country_Code'], df['Continent']))


def get_continent_from_country_code(country_code, country_continent_map):
    """根据国家二字母代码获取大洲名称"""
    return country_continent_map.get(country_code, 'Unknown')


def add_continent_column(csv_path, output_dir, mapping_file, case_name):
    """为CSV添加大洲列并排序"""
    df = pd.read_csv(csv_path)
    
    # 加载国家代码到大洲的映射
    country_continent_map = load_country_continent_mapping(mapping_file)
    
    # 添加大洲列
    df['continent'] = df['country'].apply(lambda x: get_continent_from_country_code(x, country_continent_map))
    
    # 重新排列列顺序
    columns = ['continent', 'country', 'total_difference', 'total_reduction', 
               'cooling_difference', 'cooling_reduction', 'heating_difference', 'heating_reduction']
    df = df[columns]
    
    # 按大洲排序
    continent_order = ['Asia', 'Europe', 'Africa', 'North America', 'South America', 'Oceania', 'Unknown']
    df['continent'] = pd.Categorical(df['continent'], categories=continent_order, ordered=True)
    df = df.sort_values('continent')
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{case_name}_with_continent.csv"
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)
    print(f"已保存带大洲列的文件: {output_path}")
    
    return df


def calculate_reduction_rate_distribution(df, output_dir, case_name):
    """计算各大洲节能率分布"""
    # 定义节能率区间（修正为5个区间）
    ranges = [
        ('0~15%', 0, 15),
        ('15~40%', 15, 40),
        ('40~60%', 40, 60),
        ('60~80%', 60, 80),
        ('80~100%', 80, 100)
    ]
    
    results = []
    
    for continent in df['continent'].unique():
        if pd.isna(continent) or continent == 'Unknown':
            continue
            
        continent_data = df[df['continent'] == continent]
        total_countries = len(continent_data)
        
        if total_countries == 0:
            continue
        
        # 处理空值：将NaN值视为0~15%区间
        continent_data = continent_data.copy()
        nan_mask = pd.isna(continent_data['total_reduction'])
        continent_data.loc[nan_mask, 'total_reduction'] = 0  # 将NaN设为0，归入0~15%区间
            
        for range_name, min_val, max_val in ranges:
            # 统计该区间内的国家数量
            if max_val == 100:
                count = len(continent_data[
                    (continent_data['total_reduction'] >= min_val) & 
                    (continent_data['total_reduction'] <= max_val)
                ])
            else:
                count = len(continent_data[
                    (continent_data['total_reduction'] >= min_val) & 
                    (continent_data['total_reduction'] < max_val)
                ])
            
            # 计算占比
            rate = (count / total_countries) * 100 if total_countries > 0 else 0
            
            results.append({
                'continent': continent,
                'range': range_name,
                'rate': rate
            })
    
    # 保存结果
    result_df = pd.DataFrame(results)
    output_filename = f"{case_name}_reduction_rate_distribution.csv"
    output_path = os.path.join(output_dir, output_filename)
    result_df.to_csv(output_path, index=False)
    print(f"已保存节能率分布统计: {output_path}")
    
    return result_df


def process_all_cases(input_dir, mapping_file, output_dir):
    """处理所有case文件"""
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"输入目录不存在: {input_dir}")
        return
    
    if not os.path.exists(mapping_file):
        print(f"国家大洲映射文件不存在: {mapping_file}")
        return
    
    # 获取所有case文件
    case_files = [f for f in os.listdir(input_dir) if f.startswith('case') and f.endswith('_summary_average.csv')]
    
    if not case_files:
        print(f"在 {input_dir} 中未找到case文件")
        return
    
    print(f"找到 {len(case_files)} 个case文件")
    
    # 处理每个case文件
    for case_file in case_files:
        print(f"\n正在处理: {case_file}")
        
        input_csv = os.path.join(input_dir, case_file)
        
        # 提取case名称（去掉_summary_average.csv后缀）
        case_name = case_file.replace('_summary_average.csv', '')
        
        # 添加大洲列
        df_with_continent = add_continent_column(input_csv, output_dir, mapping_file, case_name)
        
        # 计算节能率分布
        distribution_df = calculate_reduction_rate_distribution(df_with_continent, output_dir, case_name)
        
        print(f"完成处理: {case_file}")


def main():
    # 输入目录
    input_dir = r"Z:\local_environment_creation\energy_consumption\2016-2020result\figure_maps_and_data\per_capita\data\average"
    
    # 国家大洲映射文件
    mapping_file = r"Z:\local_environment_creation\energy_consumption\2016-2020result\processed_countries.csv"
    
    # 输出目录
    output_dir = r"Z:\local_environment_creation\energy_consumption\2016-2020result\reduction_rate"
    
    print("开始处理所有case文件...")
    process_all_cases(input_dir, mapping_file, output_dir)
    print("\n所有case文件处理完成！")


if __name__ == '__main__':
    main()
