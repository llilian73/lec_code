"""
全球总能耗和人均能耗汇总计算工具（2016-2020年）

功能概述：
本工具用于汇总计算全球所有国家的建筑能耗数据（2016-2020年），包括总能耗和人均能耗，支持多种节能案例的对比分析。该工具不进行校准，直接使用原始计算结果进行汇总。

输入数据：
1. 各国能耗汇总文件：
   - 路径：2016-2020result/{year}/{continent}/summary/{region}_{year}_summary_results.csv
   - 包含各国总能耗数据（TWh）
   - 涵盖ref、case1-case20共21种案例
   - 覆盖2016-2020年

2. 人口数据文件：
   - country_population_2020.csv：全球各国人口数据
   - 用于计算全球总人口和人均能耗

3. 地区分类：
   - 按大洲分类：Africa, Asia, Europe, North America, Oceania, South America
   - 每个大洲下包含多个国家的能耗数据

主要功能：
1. 全球能耗汇总：
   - 收集全球所有国家的能耗数据（2016-2020年）
   - 按总能耗、供暖能耗、制冷能耗分别汇总
   - 支持21种不同案例的对比分析（ref + case1-20）

2. 多年份数据处理：
   - 自动处理2016-2020年共5年的数据
   - 按年份组织输出结果
   - 在CSV中增加年份列便于分析

3. 人均能耗计算：
   - 基于全球总人口计算人均能耗
   - 将总能耗（TWh）转换为人均能耗（kWh/person）
   - 提供人均视角的能耗分析

4. 节能效果分析：
   - 计算各案例相对于ref案例的节能百分比
   - 分析不同节能技术的全球效果
   - 生成节能效果对比报告

输出结果：
1. 全球总能耗数据（CSV格式）：
   - global_total_total_energy.csv：全球总能耗汇总（含年份列）
   - global_total_cooling_energy.csv：全球制冷能耗汇总（含年份列）
   - global_total_heating_energy.csv：全球供暖能耗汇总（含年份列）

2. 全球人均能耗数据（CSV格式）：
   - global_per_capita_total_energy.csv：全球人均总能耗（含年份列）
   - global_per_capita_cooling_energy.csv：全球人均制冷能耗（含年份列）
   - global_per_capita_heating_energy.csv：全球人均供暖能耗（含年份列）

3. 数据内容：
   - year：年份（2016-2020）
   - index：工况名称（ref, case1-case20）
   - demand_sum：各案例的能耗总量
   - reduction_percentage：相对于ref案例的节能百分比
   - 涵盖ref、case1-case20共21种案例

计算特点：
- 直接汇总：不进行校准，直接使用原始计算结果
- 多年份支持：覆盖2016-2020年共5年数据
- 全面覆盖：包含全球所有可用的国家数据
- 多维度分析：总能耗和人均能耗双重分析
- 案例对比：支持21种不同节能案例的对比
- 结构化输出：CSV文件包含年份列，便于时间序列分析

与校准版本的区别：
- 不进行数据校准，直接使用原始计算结果
- 保持数据的原始性和真实性
- 适用于需要原始数据的分析场景
"""

import os
import pandas as pd
import numpy as np


## 计算全球总能耗和人均能耗

def get_global_population():
    """获取全球总人口"""
    population_file = r"Z:\local_environment_creation\Population\country_population_2020.csv"
    df = pd.read_csv(population_file, encoding="gbk")
    return df['Population_2020'].sum()


def find_summary_file(region, base_paths, year, is_per_capita=True):
    """查找指定地区和年份的summary文件
    在新的目录结构中查找数据

    Args:
        region: 地区代码
        base_paths: 包含所有可能路径的字典
        year: 年份（2016-2020）
        is_per_capita: 是否查找人均数据文件

    Returns:
        找到的文件路径，如果未找到返回None
    """
    # 确定文件名
    file_suffix = "_summary_p_results.csv" if is_per_capita else "_summary_results.csv"

    # 在新的目录结构中查找：2016-2020result/{year}/{continent}/summary/
    continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    folder_name = "summary_p" if is_per_capita else "summary"
    for continent in continents:
        global_file = os.path.join(base_paths['global'], str(year), continent, folder_name,
                                   f"{region}_{year}{file_suffix}")
        if os.path.exists(global_file):
            return global_file

    return None


def collect_all_regions(base_paths, year, is_per_capita=True):
    """收集指定年份所有可用的地区代码"""
    regions = set()
    file_suffix = "_summary_p_results.csv" if is_per_capita else "_summary_results.csv"
    folder_name = "summary_p" if is_per_capita else "summary"

    # 检查新目录结构：2016-2020result/{year}/{continent}/summary/
    continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    for continent in continents:
        continent_path = os.path.join(base_paths['global'], str(year), continent, folder_name)
        if os.path.exists(continent_path):
            files = [f for f in os.listdir(continent_path) if f.endswith(file_suffix)]
            regions.update([f.split('_')[0] for f in files])

    return sorted(list(regions))


def calculate_global_energy(is_per_capita=True):
    """计算全球能耗数据（2016-2020年，21个工况）

    Args:
        is_per_capita: 是否计算人均数据
    """
    # 设置基础路径
    base_paths = {
        'global': r"Z:\local_environment_creation\energy_consumption\2016-2020result"
    }

    # 处理年份范围：2016-2020
    years = range(2016, 2021)

    # 初始化结果数据结构 - 21个工况（ref + case1-20）
    cases = ['ref'] + [f'case{i}' for i in range(1, 21)]

    # 存储所有年份的数据
    all_years_data = {}

    for year in years:
        print(f"正在处理 {year} 年数据...")

        # 收集当年所有地区
        regions = collect_all_regions(base_paths, year, is_per_capita=False)  # 始终使用总能耗数据
        print(f"找到 {len(regions)} 个地区的数据")

        # 初始化当年结果数据结构
        energy_data = {
            'total': pd.DataFrame(0.0, index=cases, columns=['demand_sum']),
            'cooling': pd.DataFrame(0.0, index=cases, columns=['demand_sum']),
            'heating': pd.DataFrame(0.0, index=cases, columns=['demand_sum'])
        }

        # 处理每个地区的数据
        for region in regions:
            file_path = find_summary_file(region, base_paths, year, is_per_capita=False)  # 始终使用总能耗数据
            if file_path:
                try:
                    df = pd.read_csv(file_path, index_col=0)

                    # 使用总能耗列名
                    total_col = 'total_demand_sum(TWh)'
                    cooling_col = 'cooling_demand_sum(TWh)'
                    heating_col = 'heating_demand_sum(TWh)'

                    # 累加每个case的能耗
                    for case in cases:
                        if case in df.index:
                            energy_data['total'].loc[case, 'demand_sum'] += df.loc[case, total_col]
                            energy_data['cooling'].loc[case, 'demand_sum'] += df.loc[case, cooling_col]
                            energy_data['heating'].loc[case, 'demand_sum'] += df.loc[case, heating_col]
                except Exception as e:
                    print(f"处理 {region} 在 {year} 年数据时出错: {e}")

        # 计算减少比例
        for energy_type in energy_data:
            ref_value = energy_data[energy_type].loc['ref', 'demand_sum']
            reduction = pd.DataFrame(index=cases[1:], columns=['reduction_percentage'])

            for case in cases[1:]:
                case_value = energy_data[energy_type].loc[case, 'demand_sum']
                if ref_value > 0:
                    reduction.loc[case, 'reduction_percentage'] = ((ref_value - case_value) / ref_value) * 100
                else:
                    reduction.loc[case, 'reduction_percentage'] = 0

            # 将减少比例添加到结果中
            energy_data[energy_type]['reduction_percentage'] = pd.Series(dtype=float)
            energy_data[energy_type].loc[cases[1:], 'reduction_percentage'] = reduction['reduction_percentage'].astype(
                float)

        # 如果需要计算人均数据
        if is_per_capita:
            # 获取全球总人口
            global_population = get_global_population()

            # 将总能耗转换为人均能耗（TWh -> kWh/person）
            for energy_type in energy_data:
                energy_data[energy_type]['demand_sum'] = (energy_data[energy_type][
                                                              'demand_sum'] * 1e9) / global_population

        all_years_data[year] = energy_data

    # 合并所有年份的数据并保存
    output_dir = r"Z:\local_environment_creation\energy_consumption\2016-2020result\global_total_energy_demand"
    os.makedirs(output_dir, exist_ok=True)

    # 确定文件名前缀
    prefix = "global_per_capita_" if is_per_capita else "global_total_"

    for energy_type in ['total', 'cooling', 'heating']:
        # 合并所有年份的数据
        combined_data = []
        for year in years:
            year_data = all_years_data[year][energy_type].copy()
            year_data['year'] = year
            year_data = year_data.reset_index().set_index(['year', 'index'])
            combined_data.append(year_data)

        # 合并所有年份数据
        final_data = pd.concat(combined_data)

        # 保存结果
        output_file = os.path.join(output_dir, f"{prefix}{energy_type}_energy.csv")
        final_data.to_csv(output_file)
        print(f"已保存{energy_type}能耗数据至: {output_file}")

    return all_years_data


def main():
    print("开始计算全球人均能耗数据...")
    per_capita_energy_data = calculate_global_energy(is_per_capita=True)
    print("\n人均能耗计算完成！")

    print("\n开始计算全球总能耗数据...")
    total_energy_data = calculate_global_energy(is_per_capita=False)
    print("\n总能耗计算完成！")

    # 打印汇总信息
    print("\n人均能耗汇总:")
    for year in per_capita_energy_data:
        print(f"\n{year}年:")
        for energy_type in per_capita_energy_data[year]:
            print(f"  {energy_type.capitalize()}能耗:")
            print(f"  {per_capita_energy_data[year][energy_type]}")

    print("\n总能耗汇总:")
    for year in total_energy_data:
        print(f"\n{year}年:")
        for energy_type in total_energy_data[year]:
            print(f"  {energy_type.capitalize()}能耗:")
            print(f"  {total_energy_data[year][energy_type]}")


if __name__ == "__main__":
    main()
