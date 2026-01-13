"""
全球建筑能耗地图绘制工具（非校准版本，非人均数据）

功能概述：
本工具用于绘制全球建筑能耗的世界地图，展示不同节能案例（case1-case20）的能耗差值和节能比例。
该工具只使用原始数据，不进行校准处理，处理的是非人均数据（总能耗，单位：TWh）。

输入数据：
1. 各国能耗汇总文件：
   - 路径：results/{year}/{continent}/summary/{region}_{year}_summary_results.csv
   - 包含各国总能耗数据（TWh，非人均数据）
   - 涵盖ref、case1-case20共21种案例
   - 处理年份：2016-2020年

2. 世界地图文件：
   - 路径：shapefiles/全球国家边界/world_border2.shp
   - 用于地理可视化

3. 国家代码映射文件：
   - 路径：all_countries_info.csv
   - 用于将二字母国家代码转换为三字母代码（用于地图匹配）

4. 地区分类：
   - 按大洲分类：Africa, Asia, Europe, North America, Oceania, South America
   - 每个大洲下包含多个国家的能耗数据

主要功能：
1. 数据收集和处理：
   - 按年份（2016-2020）和大洲扫描所有可用的地区代码
   - 收集各地区的能耗汇总文件路径
   - 处理case1到case20的所有节能案例（共20个case）
   - 将二字母国家代码转换为三字母代码以匹配地图数据

2. 能耗差值地图绘制：
   - 计算ref案例与各case案例之间的能耗差值
   - 生成总能耗、供暖能耗、制冷能耗的差值世界地图
   - 差值 = ref能耗 - case能耗（正值表示节能效果）
   - 特殊处理：香港（HKG）和澳门（MAC）显示中国的数据

3. 节能比例地图绘制：
   - 直接使用case案例中的节能百分比数据
   - 生成总能耗、供暖能耗、制冷能耗的节能比例世界地图
   - 显示各地区的节能效果百分比（0-100%）
   - 特殊处理：香港（HKG）和澳门（MAC）显示中国的数据

4. CSV数据导出：
   - 将各案例的能耗差值和节能比例保存为CSV文件
   - 文件名格式：case{num}_summary.csv
   - 包含以下列：country, total_difference, total_reduction, 
                cooling_difference, cooling_reduction, 
                heating_difference, heating_reduction

5. 并行处理：
   - 使用多进程并行处理，提高处理效率
   - 每批处理5个case，避免内存占用过大
   - 按年份和批次顺序处理

输出结果：
1. 世界地图文件（由draw_map_nocapita模块生成）：
   - 差值地图：{data_type}_demand_difference_map_case{num}.{format}
   - 比例地图：{data_type}_demand_reduction_map_case{num}.{format}
   - data_type包括：total、heating、cooling
   - 按年份分别保存在不同目录

2. CSV格式的汇总数据：
   - case1_summary.csv 到 case20_summary.csv
   - 包含各国家的能耗差值和节能比例（6列数据）
   - 按年份分别保存在不同目录

3. 输出目录结构：
   - figure_maps_and_data/not_capita/{year}/：各年份的地图文件
   - figure_maps_and_data/not_capita/data/{year}/：各年份的CSV汇总文件

计算特点：
- 原始数据：只使用原始数据，不进行校准
- 非人均数据：处理的是总能耗（TWh），不是人均能耗
- 全面覆盖：包含全球所有可用的国家数据
- 多案例对比：支持case1-case20的20种节能案例
- 多年份处理：处理2016-2020年共5年的数据
- 双重分析：同时生成差值地图和比例地图
- 并行处理：使用多进程提高处理效率
- 国家代码转换：自动将二字母代码转换为三字母代码以匹配地图

数据流程：
1. 按年份（2016-2020）循环处理
2. 扫描各大洲目录，收集所有可用的地区代码和文件路径
3. 将case1-case20分批（每批5个）进行并行处理
4. 对每个case：
   a. 加载能耗差值和节能比例数据（二字母代码）
   b. 将数据转换为三字母代码用于绘图
   c. 特殊处理：为香港和澳门分配中国的数据
   d. 调用绘图函数生成差值地图和比例地图
   e. 汇总数据并导出CSV文件
5. 保存所有结果到按年份组织的目录结构中
"""

import os
import sys
import glob
import pandas as pd
import warnings
import multiprocessing
from functools import partial

# 抑制 GDAL 错误输出（QGIS 插件加载失败等警告）
# 必须在导入 geopandas 之前设置
os.environ['CPL_LOG'] = 'NUL'  # Windows 下重定向到空设备
os.environ['GDAL_SKIP'] = 'SOSI'  # 跳过 SOSI 驱动加载
os.environ['GDAL_ERROR_ON_MISSING_FILE'] = 'NO'

# 抑制 Python warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*GDAL.*')
warnings.filterwarnings('ignore', message='.*ogr_SOSI.*')

# 将项目的根目录加入到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)  # 项目根目录
sys.path.append(grandparent_dir)

# 在导入 geopandas 相关模块前，临时重定向 stderr 以抑制 GDAL 错误
import contextlib

@contextlib.contextmanager
def suppress_stderr():
    """临时抑制 stderr 输出"""
    with open(os.devnull, 'w', encoding='utf-8') as devnull:
        old_stderr = sys.stderr
        try:
            sys.stderr = devnull
            yield
        finally:
            sys.stderr = old_stderr

# 在导入时抑制 GDAL 错误（仅在导入时，不影响后续运行）
with suppress_stderr():
    from CandD.draw_map_nocapita import d_reduction_difference_map, d_reduction_map

# 国家信息文件路径
COUNTRIES_INFO_FILE = r"Z:\local_environment_creation\all_countries_info.csv"

def load_country_code_mapping():
    """加载国家代码映射（二字母代码 -> 三字母代码）"""
    try:
        # 使用gbk编码读取文件，避免将 'NA' 识别为缺失值
        df = pd.read_csv(COUNTRIES_INFO_FILE, encoding='gbk', keep_default_na=False, na_values=[''])
        # 创建二字母代码到三字母代码的映射
        code2_to_code3 = {}
        for _, row in df.iterrows():
            code2 = str(row['Country_Code_2']).strip()
            code3 = str(row['Country_Code_3']).strip()
            if code2 and code3 and code2 != '' and code3 != '':
                code2_to_code3[code2] = code3
        print(f"成功加载 {len(code2_to_code3)} 个国家代码映射（使用编码: gbk）")
        return code2_to_code3
    except Exception as e:
        print(f"加载国家代码映射失败: {e}")
        return {}

# 在代码开始处加载国家代码映射（只加载一次）
CODE2_TO_CODE3 = load_country_code_mapping()

def convert_code2_to_code3(data_dict, code2_to_code3):
    """将数据字典的键从二字母代码转换为三字母代码"""
    converted_dict = {}
    for code2, value in data_dict.items():
        code3 = code2_to_code3.get(code2, None)
        if code3:
            converted_dict[code3] = value
        else:
            # 如果找不到映射，保留原键（可能是其他格式）
            converted_dict[code2] = value
    return converted_dict


# 绘制全球热图以及导出各个国家节能量和节能率的csv文件

def process_single_case(args):
    """处理单个case的地图绘制和数据汇总（用于并行处理）
    Args:
        args: 元组 (case_num, global_paths, shapefile_path, year_map_output, 
              year_data_output, year, CODE2_TO_CODE3)
    """
    case_num, global_paths, shapefile_path, year_map_output, year_data_output, year, code2_to_code3 = args
    
    try:
        # 先加载原始数据（二字母代码）
        from CandD.draw_map_nocapita import load_demand_difference, load_reduction_data
        difference_data_raw = load_demand_difference(global_paths, case_num, year)
        reduction_data_raw = load_reduction_data(global_paths, case_num, year)
        
        # 为绘图准备数据：将二字母代码转换为三字母代码
        difference_data_for_plot = {}
        reduction_data_for_plot = {}
        for data_type in ['total', 'heating', 'cooling']:
            difference_data_for_plot[data_type] = convert_code2_to_code3(
                difference_data_raw.get(data_type, {}), code2_to_code3)
            reduction_data_for_plot[data_type] = convert_code2_to_code3(
                reduction_data_raw.get(data_type, {}), code2_to_code3)
        
        # 特殊处理：香港和澳门显示中国的数据（使用三字母代码）
        for data_type in ['total', 'heating', 'cooling']:
            if 'CHN' in difference_data_for_plot[data_type]:
                # 香港（HKG）显示中国的数值
                if 'HKG' not in difference_data_for_plot[data_type]:
                    difference_data_for_plot[data_type]['HKG'] = difference_data_for_plot[data_type]['CHN']
                # 澳门（MAC）显示中国的数值
                if 'MAC' not in difference_data_for_plot[data_type]:
                    difference_data_for_plot[data_type]['MAC'] = difference_data_for_plot[data_type]['CHN']
            if 'CHN' in reduction_data_for_plot[data_type]:
                # 香港（HKG）显示中国的数值
                if 'HKG' not in reduction_data_for_plot[data_type]:
                    reduction_data_for_plot[data_type]['HKG'] = reduction_data_for_plot[data_type]['CHN']
                # 澳门（MAC）显示中国的数值
                if 'MAC' not in reduction_data_for_plot[data_type]:
                    reduction_data_for_plot[data_type]['MAC'] = reduction_data_for_plot[data_type]['CHN']
        
        # 调用绘图函数（传入三字母代码的数据）
        d_reduction_difference_map(
            difference_data_for_plot, shapefile_path, year_map_output, year, case_num
        )
        d_reduction_map(
            reduction_data_for_plot, shapefile_path, year_map_output, year, case_num
        )

        # 汇总数据（使用原始的二字母代码）
        total_diff = difference_data_raw.get('total', {})
        total_red = reduction_data_raw.get('total', {})
        cooling_diff = difference_data_raw.get('cooling', {})
        cooling_red = reduction_data_raw.get('cooling', {})
        heating_diff = difference_data_raw.get('heating', {})
        heating_red = reduction_data_raw.get('heating', {})

        all_countries = set(total_diff.keys()) | set(total_red.keys()) | set(cooling_diff.keys()) | set(
            heating_diff.keys())
        combined_data = []
        for country in all_countries:
            combined_data.append({
                'country': country,
                'total_difference': total_diff.get(country),
                'total_reduction': total_red.get(country),
                'cooling_difference': cooling_diff.get(country),
                'cooling_reduction': cooling_red.get(country),
                'heating_difference': heating_diff.get(country),
                'heating_reduction': heating_red.get(country)
            })

        # 保存CSV
        if combined_data:
            df = pd.DataFrame(combined_data)
            csv_output_path = os.path.join(year_data_output, f"case{case_num}_summary.csv")
            df.to_csv(csv_output_path, index=False)
            return f"Case {case_num} 完成，汇总数据已保存到: {csv_output_path}"
        else:
            return f"Case {case_num} 完成，但无数据"
    except Exception as e:
        return f"Case {case_num} 处理失败: {str(e)}"


def process_all_cases():
    """按年份和case批量处理地图和汇总数据"""
    base_dir = r"Z:\local_environment_creation\energy_consumption_gird\result\result"
    shapefile_path = r"Z:\local_environment_creation\shapefiles\全球国家边界\world_border2.shp"
    output_base = r"Z:\local_environment_creation\energy_consumption_gird\result\result\figure_maps_and_data\not_capita"
    data_output_base = os.path.join(output_base, "data")
    years = range(2016, 2021)
    continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']

    for year in years:
        print(f"\n处理年份 {year}")
        year_map_output = os.path.join(output_base, str(year))
        year_data_output = os.path.join(data_output_base, str(year))
        os.makedirs(year_map_output, exist_ok=True)
        os.makedirs(year_data_output, exist_ok=True)

        # 收集所有大洲的 summary 路径
        global_paths = []
        for continent in continents:
            path = os.path.join(base_dir, str(year), continent, 'summary')
            if os.path.exists(path):
                global_paths.append(path)

        if not global_paths:
            print(f"  ⚠ {year}年没有数据，跳过")
            continue

        # 准备所有case的参数列表
        case_numbers = list(range(1, 21))  # case1到case20
        batch_size = 5  # 每批处理5个case
        
        # 将case分批
        case_batches = [case_numbers[i:i + batch_size] for i in range(0, len(case_numbers), batch_size)]
        
        print(f"  共 {len(case_numbers)} 个case，分为 {len(case_batches)} 批处理（每批 {batch_size} 个）")
        
        # 配置并行处理参数
        num_cores = multiprocessing.cpu_count()
        num_processes = min(num_cores, batch_size)  # 进程数不超过批次大小
        
        # 按批处理
        for batch_idx, case_batch in enumerate(case_batches, 1):
            print(f"  处理第 {batch_idx}/{len(case_batches)} 批: case {case_batch[0]}-{case_batch[-1]}")
            
            # 准备当前批次的参数
            batch_args = [
                (case_num, global_paths, shapefile_path, year_map_output, 
                 year_data_output, year, CODE2_TO_CODE3)
                for case_num in case_batch
            ]
            
            # 并行处理当前批次
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(process_single_case, batch_args)
            
            # 打印结果
            for result in results:
                print(f"    {result}")


def main():
    print("开始绘制世界地图...")
    process_all_cases()
    print("\n所有地图和数据汇总完成！")
    print("地图输出结构: figure_maps_and_data/not_capita/{year}/")
    print("汇总数据输出结构: figure_maps_and_data/not_capita/data/{year}/")


if __name__ == "__main__":
    main()
