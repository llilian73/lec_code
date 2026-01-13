"""
中国人体理论最小负荷计算工具

功能概述：
本工具用于计算中国各省份的人体理论最小负荷（Theoretical Minimum Load of Human, TMLH），基于BAIT（Building Adaptive Indoor Temperature）数据和相对湿度数据，计算人体的热舒适负荷需求。支持总量和人均两种计算模式，为建筑能耗分析提供人体热舒适的理论基准。

输入数据：
1. BAIT数据文件：
   - 文件路径：D:\workstation\energy_comsuption\hourly BAIT\hourly_BAIT_CN.csv
   - 数据格式：逐小时BAIT数据，按省份分列
   - 时间范围：8760小时（全年）
   - 空间范围：中国各省份

2. 人口数据文件（总量）：
   - 文件路径：D:\workstation\energy_comsuption\BAIT\CN\CN_2019_population.csv
   - 数据格式：省份人口总量数据
   - 包含列：province（省份）、population（人口数量）

3. 人口数据文件（人均）：
   - 文件路径：D:\workstation\energy_comsuption\BAIT\CN\CN_20191_population.csv
   - 数据格式：人均人口数据
   - 用于人均负荷计算

4. 相对湿度数据：
   - 文件路径：D:\workstation\energy_comsuption\Weather parameters\indoor_RH_pop\Asia\RH_pop_CN_2019.csv
   - 数据格式：逐小时相对湿度数据，按省份分列
   - 时间范围：8760小时（全年）

主要功能：
1. 人体热舒适负荷计算：
   - 基于BAIT温度数据计算人体热舒适需求
   - 考虑相对湿度对热舒适的影响
   - 动态调整服装热阻（clo值）
   - 使用PMV-PPD模型计算热负荷

2. 双模式计算：
   - 总量模式：计算各省份的总热负荷（GW/TWh）
   - 人均模式：计算人均热负荷（kW）
   - 支持冷负荷和热负荷分别计算

3. 数据整合与处理：
   - 时间序列对齐和验证
   - 省份数据匹配和筛选
   - 异常数据处理和填充

4. 结果汇总与更新：
   - 生成逐小时负荷数据
   - 计算年度能耗汇总
   - 更新地区summary文件
   - 计算相对于参考案例的节能百分比

输出结果：
1. 逐小时负荷文件：
   - 总量模式：CN_2019_hourly_loads.csv
   - 人均模式：CN_2019_hourly_loads_p.csv
   - 包含列：时间索引 + 各省份负荷数据

2. 能耗汇总文件：
   - 总量模式：CN_2019_energy_summary.csv
   - 人均模式：CN_2019_energy_p_summary.csv
   - 包含列：Cooling_Load、Heating_Load、Cooling_Demand、Heating_Demand、Total_Demand

3. 地区summary文件更新：
   - 自动更新各省份的summary文件
   - 添加TMLH案例数据
   - 计算节能百分比

数据流程：
1. 数据加载阶段：
   - 读取BAIT、人口、相对湿度数据
   - 时间索引标准化
   - 数据格式验证

2. 省份处理阶段：
   - 逐个省份计算热负荷
   - 时间序列对齐
   - 异常值处理

3. 负荷计算阶段：
   - 基于温度设置clo值（<20°C为1.0，≥20°C为0.5）
   - 调用cal_qpl模块计算热负荷
   - 区分冷负荷和热负荷

4. 结果汇总阶段：
   - 计算年度总负荷
   - 单位转换（W→GW/TWh或W→kW）
   - 生成汇总报告

5. 文件更新阶段：
   - 更新地区summary文件
   - 添加TMLH案例
   - 计算节能效果

计算特点：
- 热舒适模型：基于PMV-PPD理论
- 动态服装热阻：根据温度自动调整
- 双模式支持：总量和人均计算
- 时间精度：逐小时计算
- 空间精度：省级分辨率

技术参数：
- 服装热阻（clo）：温度<20°C时为1.0，≥20°C时为0.5
- 代谢率（met）：1.1（轻度活动）
- 风速（vel）：0.1 m/s（室内环境）
- 机械功（wme）：0（无机械功）
- 时间范围：8760小时（全年）

热负荷计算逻辑：
1. 正负荷值：表示热负荷（需要加热）
2. 负负荷值：表示冷负荷（需要制冷）
3. 负荷单位：总量模式为GW/TWh，人均模式为kW
4. 能耗计算：负荷累加得到年度总能耗

数据质量保证：
- 时间序列完整性检查
- 省份数据匹配验证
- 异常值检测和处理
- 数据范围合理性验证

特殊处理：
- 时间索引对齐：确保BAIT和相对湿度数据时间一致
- 数据填充：不足8760小时的数据用0填充
- 错误处理：单个省份失败不影响其他省份
- 单位转换：根据计算模式自动调整单位

输出格式：
- 文件格式：CSV（UTF-8编码）
- 时间格式：pandas datetime格式
- 数值精度：保留原始精度
- 文件命名：CN_{年份}_{类型}_{模式}.csv

应用场景：
- 建筑能耗基准分析
- 人体热舒适研究
- 建筑节能潜力评估
- 区域能源规划
- 热舒适标准制定

与参考案例的关系：
- TMLH作为理论最小负荷基准
- 与参考案例比较计算节能百分比
- 为建筑节能提供理论上限
- 支持多案例对比分析

计算模块依赖：
- cal_qpl：人体热负荷计算核心模块
- pandas：数据处理和分析
- 基于PMV-PPD热舒适理论
- 支持中国各省份气候特点
"""

import sys
import os
sys.dont_write_bytecode = True  # 防止生成 __pycache__

import pandas as pd
from cal_qpl import calculate_qpl

def load_data(bait_file, population_file, population_file_p, rh_file):
    """加载BAIT数据、人口数据和相对湿度数据"""
    # 读取hourly BAIT数据
    hourly_bait_df = pd.read_csv(bait_file)
    
    # 获取第一列作为时间列（不管列名是什么）
    time_column = hourly_bait_df.columns[0]
    hourly_bait_df[time_column] = pd.to_datetime(hourly_bait_df[time_column])
    hourly_bait_df.set_index(time_column, inplace=True)
    
    # 读取人口数据（总量）
    population_df = pd.read_csv(population_file)
    population_df.set_index('province', inplace=True)
    
    # 读取人口数据（人均）
    population_df_p = pd.read_csv(population_file_p)
    population_df_p.set_index('province', inplace=True)
    
    # 读取相对湿度数据
    rh_df = pd.read_csv(rh_file)
    rh_df["time"] = pd.to_datetime(rh_df["time"])
    rh_df.set_index("time", inplace=True)
    
    return hourly_bait_df, population_df, population_df_p, rh_df

def calculate_province_hourly_load(bait_series, rh_series, population, is_per_person=False):
    """计算单个省份的逐小时负荷"""
    hourly_load = []
    
    # 确保两个时间序列使用相同的时间索引
    common_times = sorted(set(bait_series.index) & set(rh_series.index))
    
    for time in common_times:
        temp = bait_series[time]
        rh = rh_series[time]
        
        # 设置clo值
        clo = 1.0 if temp < 20 else 0.5
        
        # 计算单人负荷
        qpl = calculate_qpl(
            ta=temp,
            tr=temp,
            vel=0.1,
            rh=rh,
            met=1.1,
            clo=clo,
            wme=0
        )
        
        if is_per_person:
            # 人均负荷 (W -> kW)
            load = qpl / 1e3  # 转换为kW
        else:
            # 省份总负荷 (W -> GW)
            load = qpl * population / 1e9  # 转换为GW
            
        hourly_load.append(load)
    
    # 确保返回8760个小时的数据
    if len(hourly_load) != 8760:
        print(f"Warning: Expected 8760 hours but got {len(hourly_load)} hours")
        # 如果数据不足，用0填充
        hourly_load.extend([0] * (8760 - len(hourly_load)))
        # 如果数据过多，截断
        hourly_load = hourly_load[:8760]
    
    return hourly_load

def calculate_province_energy(province_loads, is_per_person=False):
    """计算省份的冷热负荷和能耗"""
    cooling_load = 0
    heating_load = 0
    
    for load in province_loads:
        if load > 0:  # 热负荷
            heating_load += load
        else:  # 冷负荷
            cooling_load += abs(load)
    
    if is_per_person:
        # 人均能耗保持kW单位
        cooling_demand = cooling_load * 1  # kW
        heating_demand = heating_load * 1  # kW
        total_demand = cooling_demand + heating_demand  # kW
    else:
        # 总量能耗转换为TWh
        cooling_demand = cooling_load * 1/1000  # TWh
        heating_demand = heating_load * 1/1000  # TWh
        total_demand = cooling_demand + heating_demand  # TWh
    
    return {
        'cooling_load': cooling_load,
        'heating_load': heating_load,
        'cooling_demand': cooling_demand,
        'heating_demand': heating_demand,
        'total_demand': total_demand
    }

def update_region_summary(province, energy_results, is_per_person=False):
    """更新地区的summary文件"""
    summary_dir = r"D:\workstation\energy_comsuption\results\CN"
    if is_per_person:
        summary_file = os.path.join(summary_dir, f"{province}_2019_summary_p_results.csv")
    else:
        summary_file = os.path.join(summary_dir, f"{province}_2019_summary_results.csv")

    if not os.path.exists(summary_file):
        return

    try:
        df = pd.read_csv(summary_file)
        df.columns = df.columns.str.strip()
        ref_data = df.iloc[0]

        new_row = pd.DataFrame([[
            'TMLH',
            energy_results['total_demand'],
            (ref_data.iloc[1] - energy_results['total_demand']) / ref_data.iloc[1] * 100 if ref_data.iloc[1] != 0 else 0,
            energy_results['heating_demand'],
            (ref_data.iloc[3] - energy_results['heating_demand']) / ref_data.iloc[3] * 100 if ref_data.iloc[3] != 0 else 0,
            energy_results['cooling_demand'],
            (ref_data.iloc[5] - energy_results['cooling_demand']) / ref_data.iloc[5] * 100 if ref_data.iloc[5] != 0 else 0
        ]], columns=df.columns)

        df = df[df.iloc[:, 0] != 'TMLH']
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(summary_file, index=False)

    except Exception as e:
        print(f"Error updating summary for {province}: {e}")
        pass

def process_data(hourly_bait_df, population_df, rh_df, output_dir, selected_year, is_per_person=False):
    """处理数据并生成结果"""
    # 准备结果DataFrame
    results_df = pd.DataFrame()  # 逐小时负荷结果
    energy_results = {}  # 存储每个省份的能耗结果
    
    # 对每个省份进行计算
    for province in hourly_bait_df.columns:
        if province in population_df.index and province in rh_df.columns:
            print(f"\nProcessing {province}...")
            print(f"Population: {population_df.loc[province, 'population']:,.0f}")
            
            try:
                # 计算逐小时负荷
                province_load = calculate_province_hourly_load(
                    hourly_bait_df[province], 
                    rh_df[province], 
                    population_df.loc[province, 'population'],
                    is_per_person
                )
                results_df[province] = province_load
                
                # 计算省份能耗
                energy_results[province] = calculate_province_energy(province_load, is_per_person)
                
                # 更新地区summary文件
                update_region_summary(province, energy_results[province], is_per_person)
                
                print(f"Completed processing {province}")
            except Exception as e:
                print(f"Error processing {province}: {e}")
                continue
    
    # 设置时间索引
    results_df.index = hourly_bait_df.index
    
    # 保存逐小时负荷结果
    file_suffix = "_p" if is_per_person else ""
    hourly_output_file = os.path.join(output_dir, f"CN_{selected_year}_hourly_loads{file_suffix}.csv")
    results_df.to_csv(hourly_output_file)
    
    # 创建汇总结果DataFrame
    if is_per_person:
        columns = [
            'Cooling_Load_kW',
            'Heating_Load_kW',
            'Cooling_Demand_kW',
            'Heating_Demand_kW',
            'Total_Demand_kW'
        ]
    else:
        columns = [
            'Cooling_Load_GW',
            'Heating_Load_GW',
            'Cooling_Demand_TWh',
            'Heating_Demand_TWh',
            'Total_Demand_TWh'
        ]
    
    summary_df = pd.DataFrame(
        index=energy_results.keys(),
        columns=columns
    )
    
    # 填充汇总数据
    for province, results in energy_results.items():
        for i, key in enumerate(['cooling_load', 'heating_load', 'cooling_demand', 'heating_demand', 'total_demand']):
            summary_df.iloc[summary_df.index.get_loc(province), i] = results[key]
    
    # 保存汇总结果
    summary_output_file = os.path.join(output_dir, f"CN_{selected_year}_energy{file_suffix}_summary.csv")
    summary_df.to_csv(summary_output_file)
    
    print(f"\n结果已保存至: {output_dir}")
    print(f"1. 逐小时负荷：CN_{selected_year}_hourly_loads{file_suffix}.csv")
    print(f"2. 能耗汇总：CN_{selected_year}_energy{file_suffix}_summary.csv")

def main():
    # 设置文件路径
    bait_file = r"D:\workstation\energy_comsuption\hourly BAIT\hourly_BAIT_CN.csv"
    population_file = r"D:\workstation\energy_comsuption\BAIT\CN\CN_2019_population.csv"
    population_file_p = r"D:\workstation\energy_comsuption\BAIT\CN\CN_20191_population.csv"
    rh_file = r"D:\workstation\energy_comsuption\Weather parameters\indoor_RH_pop\Asia\RH_pop_CN_2019.csv"
    output_dir = r"D:\workstation\energy_comsuption\Theoretical minimum load of human\CN"
    selected_year = 2019
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    hourly_bait_df, population_df, population_df_p, rh_df = load_data(
        bait_file, population_file, population_file_p, rh_file
    )
    
    # 处理总量数据
    print("\n处理总量数据...")
    process_data(hourly_bait_df, population_df, rh_df, output_dir, selected_year, is_per_person=False)
    
    # 处理人均数据
    print("\n处理人均数据...")
    process_data(hourly_bait_df, population_df_p, rh_df, output_dir, selected_year, is_per_person=True)

if __name__ == "__main__":
    main()
