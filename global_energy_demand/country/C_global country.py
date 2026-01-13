"""
全球国家建筑能耗计算工具（2016-2020年）

功能概述：
本工具用于计算全球各个国家的建筑能耗数据（2016-2020年），包括总能耗和人均能耗，支持多种节能案例的对比分析。

输入数据：
1. BAIT数据文件：建筑适应性室内温度数据
   - 按大洲分类存储（Africa, Asia, Europe, North America, Oceania, South America）
   - 文件格式：demand_ninja_{CONTINENT}_BAIT.csv
   - 包含各地区的时间序列温度数据（2016-2020年）

2. 计算参数文件：parameters.csv
   - 包含各地区的供暖功率、供暖阈值、制冷功率、制冷阈值
   - 用于不同地区的个性化计算参数

3. 人口数据文件：
   - country_population_2020.csv：国家人口数据
   - global_province_population_2020_sorted.csv：省级人口数据

4. 需求计算模块：demand_ninja模块
   - 提供建筑能耗计算的核心算法

主要功能：
1. 全球能耗计算：
   - 按年份（2016-2020）、大洲和国家分别处理BAIT数据
   - 计算21种不同案例的能耗（ref + case1-case20）
   - 支持供暖、制冷、总能耗的计算

2. 多案例对比分析：
   - ref：参考案例（基准能耗）
   - case1-5：diff=1℃，不同渗透率案例（p_ls=[0.5, 0.25, 0.125, 0.0625, 0.03125]）
   - case6-10：diff=2℃，不同渗透率案例（p_ls=[0.5, 0.25, 0.125, 0.0625, 0.03125]）
   - case11-15：diff=3℃，不同渗透率案例（p_ls=[0.5, 0.25, 0.125, 0.0625, 0.03125]）
   - case16-20：diff=4℃，不同渗透率案例（p_ls=[0.5, 0.25, 0.125, 0.0625, 0.03125]）

3. 数据汇总统计：
   - 计算总能耗（TWh）和人均能耗（kWh/person）
   - 计算能耗差值和节能百分比
   - 生成详细的汇总报告

4. 错误处理和日志记录：
   - 记录国家代码错误
   - 记录缺失人口数据的国家
   - 保存已成功处理的国家列表

输出结果：
输出目录结构：
2016-2020result/
├── 2016/
│   ├── Africa/
│   │   ├── {country_code}/
│   │   │   ├── {country_code}_2016_cdd.csv
│   │   │   ├── {country_code}_2016_hdd.csv
│   │   │   ├── {country_code}_2016_cooling_demand.csv
│   │   │   ├── {country_code}_2016_heating_demand.csv
│   │   │   └── {country_code}_2016_total_demand.csv
│   │   ├── summary/
│   │   └── summary_p/
│   ├── Asia/
│   ├── Europe/
│   ├── North America/
│   ├── Oceania/
│   └── South America/
├── 2017/
├── 2018/
├── 2019/
├── 2020/
├── calculation_errors_code_errors.csv
├── calculation_errors_missing_population.csv
└── processed_countries.csv

1. 逐时能耗数据（CSV格式）：
   - {region}_{year}_cdd.csv：制冷度日数
   - {region}_{year}_hdd.csv：供暖度日数
   - {region}_{year}_cooling_demand.csv：制冷需求
   - {region}_{year}_heating_demand.csv：供暖需求
   - {region}_{year}_total_demand.csv：总需求

2. 汇总结果（CSV格式）：
   - {region}_{year}_summary_results.csv：总能耗汇总（TWh）
   - {region}_{year}_summary_p_results.csv：人均能耗汇总（kWh/person）

3. 错误记录文件：
   - calculation_errors_code_errors.csv：国家代码错误
   - calculation_errors_missing_population.csv：人口数据缺失
   - processed_countries.csv：已处理国家列表

4. 日志文件：
   - global_calculation.log：详细的计算过程日志

计算参数：
- 默认参数：供暖功率27.9W，制冷功率48.5W，供暖阈值20°C，制冷阈值26°C
- 个性化参数：根据parameters.csv文件中的地区特定参数
- 人口数据：使用2020年人口统计数据
"""

import pandas as pd
import os
import sys
from pathlib import Path
import numpy as np
import pycountry
import logging
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
# 将项目的根目录加入到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)  # 项目根目录
sys.path.append(grandparent_dir)

from CandD.calculate import calculate_cases  # 确保路径正确

# 计算全球各个国家总能耗和人均能耗

# 使用 core_p.py 中的 demand
from demand_ninja import demand  # 默认使用 core_p.py 中的 demand

# 设置日志记录
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler('global_calculation.log', encoding='utf-8'),
                       logging.StreamHandler(sys.stdout)  # 使用stdout确保正确处理Unicode字符
                   ])

# 设置tqdm与logging的兼容性
tqdm.pandas()

class ErrorRecord:
    def __init__(self):
        self.errors = []
        self.missing_population = []  # 新增：记录缺失人口数据的国家
    
    def add_error(self, continent, country_name, country_code, error_msg):
        if "无法找到国家代码" in error_msg:
            self.errors.append({
                'Continent': continent,
                'Country_Name': "Unknown",
                'Country_Code': country_code,
                'Error': error_msg
            })
        elif "未找到" in error_msg and "的人口数据" in error_msg:
            self.missing_population.append({
                'Continent': continent,
                'Country_Name': country_name,
                'Country_Code': country_code,
                'Error': "Missing population data"
            })
    
    def save_to_csv(self, output_path):
        # 保存国家代码错误
        if self.errors:
            df_errors = pd.DataFrame(self.errors)
            error_output = output_path.replace('.csv', '_code_errors.csv')
            df_errors.to_csv(error_output, index=False, encoding='utf-8')
            logging.info(f"国家代码错误记录已保存至: {error_output}")
        
        # 保存人口数据缺失错误
        if self.missing_population:
            df_missing = pd.DataFrame(self.missing_population)
            missing_output = output_path.replace('.csv', '_missing_population.csv')
            df_missing.to_csv(missing_output, index=False, encoding='utf-8')
            logging.info(f"人口数据缺失记录已保存至: {missing_output}")

class ProcessedCountries:
    def __init__(self):
        self.countries = []
    
    def add_country(self, continent, country_code, country_name):
        self.countries.append({
            'Continent': continent,
            'Country_Code': country_code,
            'Country_Name': country_name
        })
    
    def save_to_csv(self, output_path):
        if self.countries:
            df = pd.DataFrame(self.countries)
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            # 打印每个洲的国家数量
            continent_counts = df['Continent'].value_counts()
            print("\n各洲已计算国家数量：")
            for continent, count in continent_counts.items():
                print(f"{continent}: {count}个国家")

def get_country_name(alpha_2):
    """将二字母代码转换为国家全称"""
    # 特殊国家名称映射
    special_names = {
        'GB': 'United Kingdom',
        'US': 'United States',
        'XK': 'Kosovo',  # 添加Kosovo
        'CI': 'Coate d\'Ivoire'
    }
    
    try:
        # 首先检查特殊映射
        if alpha_2 in special_names:
            return special_names[alpha_2]
            
        # 如果不在特殊映射中，使用pycountry
        country = pycountry.countries.get(alpha_2=alpha_2)
        if country:
            return country.name
            
    except:
        return None
    return None

def load_parameters(params_file):
    """加载计算参数"""
    try:
        params_df = pd.read_csv(params_file)
        params_dict = dict(zip(params_df['region'], zip(
            params_df['heating power'],
            params_df['heating threshold'],
            params_df['Cooling power'],
            params_df['Cooling threshold']
        )))
        return params_dict
    except Exception as e:
        logging.error(f"加载参数文件出错: {str(e)}")
        return {}

def load_population_data():
    """加载人口数据"""
    country_pop_file = r"Z:\local_environment_creation\Population\country_population_2020.csv"
    province_pop_file = r"Z:\local_environment_creation\Population\global_province_population_2020.csv"
    
    try:
        # 加载国家人口数据
        country_pop = pd.read_csv(country_pop_file, encoding="gbk")
        # 使用 NAME_0 (国家全称) 作为键，Population_2020 作为值
        country_pop_dict = dict(zip(country_pop['NAME_0'], country_pop['Population_2020']))
        
        # 加载省级人口数据
        province_pop = pd.read_csv(province_pop_file, encoding="gbk")
        province_pop_dict = {}
        for _, row in province_pop.iterrows():
            # 使用 "国家全称_省份全称" 作为键
            key = f"{row['NAME_0']}_{row['NAME_1']}"
            province_pop_dict[key] = row['Population_2020']
        
        return country_pop_dict, province_pop_dict
        
    except Exception as e:
        logging.error(f"加载人口数据时出错: {str(e)}")
        return {}, {}

def calculate_summary(output_dir, region, selected_year, population):
    """计算并保存汇总结果"""
    # 读取数据文件
    total_demand_df = pd.read_csv(os.path.join(output_dir, f"{region}_{selected_year}_total_demand.csv"), index_col=0)
    heating_demand_df = pd.read_csv(os.path.join(output_dir, f"{region}_{selected_year}_heating_demand.csv"), index_col=0)
    cooling_demand_df = pd.read_csv(os.path.join(output_dir, f"{region}_{selected_year}_cooling_demand.csv"), index_col=0)

    # 计算全年总数（转换为TWh）
    total_sum = total_demand_df.sum() / 1000  # W -> TWh
    heating_demand_sum = heating_demand_df.sum() / 1000
    cooling_demand_sum = cooling_demand_df.sum() / 1000

    # 计算相对ref的差值（TWh）
    total_diff = total_sum["ref"] - total_sum
    heating_diff = heating_demand_sum["ref"] - heating_demand_sum
    cooling_diff = cooling_demand_sum["ref"] - cooling_demand_sum

    # 计算人均年度总数（kWh/person）
    total_sum_p = (total_sum * 1e9) / population
    heating_demand_sum_p = (heating_demand_sum * 1e9) / population
    cooling_demand_sum_p = (cooling_demand_sum * 1e9) / population

    # 计算人均差值（kWh/person）
    total_diff_p = total_sum_p["ref"] - total_sum_p
    heating_diff_p = heating_demand_sum_p["ref"] - heating_demand_sum_p
    cooling_diff_p = cooling_demand_sum_p["ref"] - cooling_demand_sum_p

    # 计算降低百分比
    def calculate_reduction(df_sum):
        ref_value = df_sum["ref"]
        return (ref_value - df_sum) / ref_value * 100

    # 计算各项降低百分比
    total_reduction = calculate_reduction(total_sum)
    heating_reduction = calculate_reduction(heating_demand_sum)
    cooling_reduction = calculate_reduction(cooling_demand_sum)
    total_reduction_p = calculate_reduction(total_sum_p)
    heating_reduction_p = calculate_reduction(heating_demand_sum_p)
    cooling_reduction_p = calculate_reduction(cooling_demand_sum_p)

    # 创建汇总DataFrame
    summary_df = pd.DataFrame({
        "total_demand_sum(TWh)": total_sum,
        "total_demand_diff(TWh)": total_diff,
        "total_demand_reduction(%)": total_reduction,
        "heating_demand_sum(TWh)": heating_demand_sum,
        "heating_demand_diff(TWh)": heating_diff,
        "heating_demand_reduction(%)": heating_reduction,
        "cooling_demand_sum(TWh)": cooling_demand_sum,
        "cooling_demand_diff(TWh)": cooling_diff,
        "cooling_demand_reduction(%)": cooling_reduction
    })

    summary_p_df = pd.DataFrame({
        "total_demand_sum_p(kWh/person)": total_sum_p,
        "total_demand_diff_p(kWh/person)": total_diff_p,
        "total_demand_p_reduction(%)": total_reduction_p,
        "heating_demand_sum_p(kWh/person)": heating_demand_sum_p,
        "heating_demand_diff_p(kWh/person)": heating_diff_p,
        "heating_demand_p_reduction(%)": heating_reduction_p,
        "cooling_demand_sum_p(kWh/person)": cooling_demand_sum_p,
        "cooling_demand_diff_p(kWh/person)": cooling_diff_p,
        "cooling_demand_p_reduction(%)": cooling_reduction_p
    })

    return summary_df, summary_p_df

def get_country_codes_from_bait(bait_file):
    """从BAIT文件中获取需要处理的国家代码"""
    try:
        # 获取文件名
        file_name = os.path.basename(bait_file)
        # 特殊文件列表
        special_files = [
            'demand_ninja_AFR_BAIT.csv',
            'demand_ninja_EUR_BAIT.csv',
            'demand_ninja_NAM_BAIT.csv',
            'demand_ninja_OCE_BAIT.csv',
            'demand_ninja_SAM_BAIT.csv'
        ]
        
        # 读取标题行
        df = pd.read_csv(bait_file, skiprows=4, nrows=0)
        columns = df.columns.tolist()
        
        if file_name in special_files:
            # 对于特殊文件，返回所有列（除了date）
            return [col for col in columns[1:] if '.' not in col]
        else:
            # 对于其他文件，只返回date后的第一列
            return [columns[1]] if len(columns) > 1 else []
            
    except Exception as e:
        logging.error(f"从BAIT文件获取国家代码时出错: {str(e)}")
        return []

def process_single_country(args):
    """处理单个国家的计算（用于并行处理）"""
    (bait_file, params_dict, country_pop_dict, continent, base_output_dir, year, 
     region_code, country_name, population, calc_params) = args
    
    try:
        # 获取国家全称
        if not country_name:
            return {"success": False, "error": f"无法找到国家代码 {region_code} 对应的国家全称", 
                   "continent": continent, "country_code": region_code, "country_name": None}
        
        # 创建按年份组织的输出目录
        year_output_dir = os.path.join(base_output_dir, str(year))
        country_dir = os.path.join(year_output_dir, continent, region_code)
        summary_dir = os.path.join(year_output_dir, continent, 'summary')
        summary_p_dir = os.path.join(year_output_dir, continent, 'summary_p')
        
        for dir_path in [country_dir, summary_dir, summary_p_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 设置基本参数
        base_params = {
            **calc_params,
            "base_power": 0,
            "population": population
        }

        # 使用calculate_cases函数计算所有工况
        cases = calculate_cases(base_params)

        # 加载BAIT数据
        df = pd.read_csv(bait_file, skiprows=4)
        df["date"] = pd.to_datetime(df["date"])
        # 筛选指定年份的数据
        df = df[df["date"].dt.year == year].copy()

        # 运行所有工况
        results = {}
        for case_name, params in cases.items():
            daily_bait = df.set_index("date")[region_code]
            results[case_name] = demand(
                daily_bait,
                heating_threshold_background=params["heating_threshold_background"],
                heating_threshold_people=params["heating_threshold_people"],
                cooling_threshold_background=params["cooling_threshold_background"],
                cooling_threshold_people=params["cooling_threshold_people"],
                p_ls=params["p_ls"],
                base_power=params["base_power"],
                heating_power=params["heating_power"],
                cooling_power=params["cooling_power"],
                population=params["population"],
                use_diurnal_profile=True
            )

        # 创建输出数据框
        hourly_index = pd.date_range(
            start=df["date"].min(),
            end=df["date"].max() + pd.Timedelta(days=1),
            freq='h',
            inclusive='left'
        )

        output_data = {
            "cdd": pd.DataFrame(index=hourly_index),
            "hdd": pd.DataFrame(index=hourly_index),
            "cooling_demand": pd.DataFrame(index=hourly_index),
            "heating_demand": pd.DataFrame(index=hourly_index),
            "total_demand": pd.DataFrame(index=hourly_index)
        }

        # 填充数据
        for case_name, result in results.items():
            output_data["cdd"][case_name] = result.cdd
            output_data["hdd"][case_name] = result.hdd
            output_data["cooling_demand"][case_name] = result.cooling_demand
            output_data["heating_demand"][case_name] = result.heating_demand
            output_data["total_demand"][case_name] = result.total_demand

        # 保存结果
        for data_type, df_out in output_data.items():
            filename = f"{region_code}_{year}_{data_type}.csv"
            df_out.to_csv(os.path.join(country_dir, filename))

        # 计算和保存汇总结果
        summary_df, summary_p_df = calculate_summary(country_dir, region_code, year, population)
        
        summary_df.to_csv(os.path.join(summary_dir, f"{region_code}_{year}_summary_results.csv"))
        summary_p_df.to_csv(os.path.join(summary_p_dir, f"{region_code}_{year}_summary_p_results.csv"))

        return {"success": True, "continent": continent, "country_code": region_code, "country_name": country_name}

    except Exception as e:
        return {"success": False, "error": str(e), "continent": continent, 
               "country_code": region_code, "country_name": country_name}

def collect_all_countries_info(bait_base_dir, continents, params_dict, country_pop_dict):
    """收集全球所有国家信息"""
    all_countries = []
    default_params = {
        "heating_power": 27.9,
        "cooling_power": 48.5,
        "heating_threshold_people": 20,
        "cooling_threshold_people": 26
    }
    
    for continent in continents:
        continent_dir = os.path.join(bait_base_dir, continent)
        if not os.path.exists(continent_dir):
            continue
            
        # 遍历大洲目录下的所有BAIT文件
        for file in os.listdir(continent_dir):
            if file.startswith("demand_ninja_") and file.endswith("_BAIT.csv"):
                bait_file = os.path.join(continent_dir, file)
                
                # 从BAIT文件中获取需要处理的国家代码
                country_codes = get_country_codes_from_bait(bait_file)
                if not country_codes:
                    continue
                
                for region_code in country_codes:
                    # 获取国家全称
                    country_name = get_country_name(region_code)
                    if not country_name:
                        continue
                    
                    # 获取计算参数
                    if region_code in params_dict:
                        hp, ht, cp, ct = params_dict[region_code]
                        calc_params = {
                            "heating_power": hp,
                            "cooling_power": cp,
                            "heating_threshold_people": ht,
                            "cooling_threshold_people": ct
                        }
                    else:
                        calc_params = default_params
                    
                    # 获取国家人口数据
                    population = country_pop_dict.get(country_name)
                    if not population:
                        continue
                    
                    # 添加到国家列表
                    country_info = {
                        "continent": continent,
                        "country_code": region_code,
                        "country_name": country_name,
                        "bait_file": bait_file,
                        "calc_params": calc_params,
                        "population": population
                    }
                    all_countries.append(country_info)
    
    return all_countries

def process_countries_batch_parallel(all_countries, year, base_output_dir, error_recorder, processed_countries, batch_size=20, max_workers=None):
    """分批并行处理所有国家"""
    if not all_countries:
        logging.warning("没有找到有效的国家信息")
        return
    
    # 准备任务参数
    tasks = []
    for country_info in all_countries:
        task_args = (
            country_info["bait_file"], 
            {},  # params_dict (不需要，因为已经包含在calc_params中)
            {},  # country_pop_dict (不需要，因为已经包含population)
            country_info["continent"], 
            base_output_dir, 
            year,
            country_info["country_code"], 
            country_info["country_name"], 
            country_info["population"], 
            country_info["calc_params"]
        )
        tasks.append(task_args)
    
    # 确定并行工作进程数
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)
    
    total_countries = len(tasks)
    total_batches = (total_countries + batch_size - 1) // batch_size
    
    print(f"\n开始分批并行处理 {total_countries} 个国家，分 {total_batches} 批，每批最多 {batch_size} 个国家，使用 {max_workers} 个进程")
    
    # 创建批次进度条
    batch_progress = tqdm(range(total_batches), desc=f"处理{year}年批次", unit="批", leave=True)
    
    # 分批处理
    for batch_idx in range(0, total_countries, batch_size):
        batch_tasks = tasks[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        
        # 更新进度条描述
        batch_progress.set_description(f"处理{year}年第{batch_num}/{total_batches}批")
        
        # print(f"\n正在处理第 {batch_num}/{total_batches} 批，包含 {len(batch_tasks)} 个国家")
        
        # 使用进程池并行处理当前批次
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(process_single_country, task): task for task in batch_tasks}
            
            # 处理完成的任务
            for future in as_completed(future_to_task):
                result = future.result()
                
                if result["success"]:
                    # 成功处理的国家
                    processed_countries.add_country(result["continent"], result["country_code"], result["country_name"])
                else:
                    # 处理失败的国家
                    error_recorder.add_error(result["continent"], result["country_name"], 
                                           result["country_code"], result["error"])
        
        # 更新进度条
        batch_progress.update(1)
        print(f"第 {batch_num}/{total_batches} 批处理完成")
    
    # 关闭批次进度条
    batch_progress.close()

def process_region_by_year_parallel(bait_file, params_dict, country_pop_dict, continent, base_output_dir, error_recorder, processed_countries, year, max_workers=None):
    """按年份并行处理单个BAIT文件中的所有国家"""
    try:
        # 从BAIT文件中获取需要处理的国家代码
        country_codes = get_country_codes_from_bait(bait_file)
        if not country_codes:
            logging.warning(f"在文件 {bait_file} 中未找到有效的国家代码")
            return

        # 准备并行任务参数
        tasks = []
        default_params = {
            "heating_power": 27.9,
            "cooling_power": 48.5,
            "heating_threshold_people": 20,
            "cooling_threshold_people": 26
        }

        for region_code in country_codes:
            # 获取国家全称
            country_name = get_country_name(region_code)
            if not country_name:
                error_recorder.add_error(continent, None, region_code, f"无法找到国家代码 {region_code} 对应的国家全称")
                continue

            # 获取计算参数
            if region_code in params_dict:
                hp, ht, cp, ct = params_dict[region_code]
                calc_params = {
                    "heating_power": hp,
                    "cooling_power": cp,
                    "heating_threshold_people": ht,
                    "cooling_threshold_people": ct
                }
            else:
                calc_params = default_params

            # 获取国家人口数据
            population = country_pop_dict.get(country_name)
            if not population:
                error_recorder.add_error(continent, country_name, region_code, f"未找到 {country_name} 的人口数据")
                continue

            # 添加到任务列表
            task_args = (bait_file, params_dict, country_pop_dict, continent, base_output_dir, year,
                        region_code, country_name, population, calc_params)
            tasks.append(task_args)

        if not tasks:
            logging.warning(f"在 {continent} 中没有有效的国家任务")
            return

        # 确定并行工作进程数
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(tasks), 8)  # 最多8个进程，避免过多进程导致内存不足

        logging.info(f"开始并行处理 {continent} - {year} 年，共 {len(tasks)} 个国家，使用 {max_workers} 个进程")

        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(process_single_country, task): task for task in tasks}
            
            # 处理完成的任务
            for future in as_completed(future_to_task):
                result = future.result()
                
                if result["success"]:
                    # 成功处理的国家
                    processed_countries.add_country(result["continent"], result["country_code"], result["country_name"])
                else:
                    # 处理失败的国家
                    error_recorder.add_error(result["continent"], result["country_name"], 
                                           result["country_code"], result["error"])

    except Exception as e:
        error_msg = f"并行处理BAIT文件时出错: {str(e)}"
        logging.error(error_msg)

def main():
    # 设置多进程启动方法（Windows兼容性）
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # 如果已经设置过，忽略错误
    
    # 基础路径设置
    bait_base_dir = r"Z:\local_environment_creation\energy_consumption\BAIT"
    output_base_dir = r"Z:\local_environment_creation\energy_consumption\2016-2020result"
    params_file = r"Z:\local_environment_creation\energy_consumption\2016-2020result\parameters.csv"
    error_output = os.path.join(output_base_dir, "calculation_errors.csv")

    # 创建错误记录器和已处理国家记录器
    error_recorder = ErrorRecord()
    processed_countries = ProcessedCountries()

    # 加载参数和人口数据
    params_dict = load_parameters(params_file)
    country_pop_dict, _ = load_population_data()  # 不需要省级人口数据

    # 处理年份范围：2016-2020
    years = range(2016, 2021)
    continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    
    # 收集全球所有国家信息
    print("正在收集全球所有国家信息...")
    all_countries = collect_all_countries_info(bait_base_dir, continents, params_dict, country_pop_dict)
    print(f"共收集到 {len(all_countries)} 个国家的信息\n")
    
    # 创建年份进度条
    year_progress = tqdm(years, desc="处理年份", unit="年")
    
    for year in year_progress:
        year_progress.set_description(f"处理 {year} 年")
        year_start_time = time.time()
        
        # 分批并行处理所有国家
        process_countries_batch_parallel(
            all_countries, year, output_base_dir, 
            error_recorder, processed_countries, 
            batch_size=20, max_workers=8
        )
        
        year_time = time.time() - year_start_time
        print(f"\n{year} 年所有国家计算完成，总耗时 {year_time:.2f} 秒\n")

    # 保存错误记录和已处理国家列表
    error_recorder.save_to_csv(error_output)
    processed_countries.save_to_csv(os.path.join(output_base_dir, "processed_countries.csv"))
    logging.info("全球2016-2020年计算完成")

if __name__ == "__main__":
    main()