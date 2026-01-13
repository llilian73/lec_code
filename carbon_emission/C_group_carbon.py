"""
分组碳排放量汇总计算工具（2016-2020年）

功能概述：
本工具用于按组（CN、IN、US、Europe、Others）汇总计算全球碳排放数据，计算5年平均值。

输入数据：
1. 各国碳排放汇总文件：
   - 路径：Z:\local_environment_creation\carbon_emission\2016-2020\result\{年份}\notcapita\{case}_summary.csv
   - 包含各国总碳排放数据（tCO2）
   - 涵盖ref、case1-case20共21种案例
   - 覆盖2016-2020年

分组规则：
- CN：中国
- IN：印度
- US：美国
- Europe：欧洲所有国家
- Others：其他所有国家

输出结果：
- group_summary_2016_2020_average.csv：各组5年平均碳排放数据
- 格式：group,index,carbon_emission(tCO2),carbon_emission_reduction(tCO2),carbon_emission_reduction(%)
"""

import os
import pandas as pd
import numpy as np

# 国家信息文件路径
COUNTRIES_INFO_FILE = r"Z:\local_environment_creation\all_countries_info.csv"

def load_country_continent_mapping():
    """加载国家代码到大洲的映射（从all_countries_info.csv）"""
    try:
        # 使用gbk编码读取文件，避免将 'NA' 识别为缺失值
        df = pd.read_csv(COUNTRIES_INFO_FILE, encoding='gbk', keep_default_na=False, na_values=[''])
        # 创建二字母代码到大洲的映射
        code2_to_continent = {}
        for _, row in df.iterrows():
            code2 = str(row['Country_Code_2']).strip().upper()
            continent = str(row['continent']).strip()
            if code2 and continent and code2 != '' and continent != '':
                code2_to_continent[code2] = continent
        print(f"成功加载 {len(code2_to_continent)} 个国家的大洲映射（使用编码: gbk）")
        return code2_to_continent
    except Exception as e:
        print(f"加载国家大洲映射失败: {e}")
        return {}

# 在代码开始处加载国家大洲映射（只加载一次）
CODE2_TO_CONTINENT = load_country_continent_mapping()


def list_case_summary_files(base_root: str, year: int):
    """列出指定年份下所有case的汇总文件路径，返回 (case_name, file_path) 列表。"""
    cases = ['ref'] + [f'case{i}' for i in range(1, 21)]
    results = []
    notcapita_dir = os.path.join(base_root, str(year), 'notcapita')
    if not os.path.isdir(notcapita_dir):
        return results
    
    for case in cases:
        file_path = os.path.join(notcapita_dir, f"{case}_summary.csv")
        if os.path.exists(file_path):
            results.append((case, file_path))
    return results


def determine_group(country_code: str, continent: str = None) -> str:
    """根据国家代码和大洲划分组：CN、IN、US、Europe、Others。
    Args:
        country_code: 国家二字母代码
        continent: 大洲名称（可选，用于判断Europe）
    """
    code = (country_code or '').strip().upper()
    if code in {'CN'}:
        return 'CN'
    if code in {'IN'}:
        return 'IN'
    if code in {'US'}:
        return 'US'
    if continent == 'Europe':
        return 'Europe'
    return 'Others'


def get_continent_from_country_code(country_code: str):
    """根据国家代码获取大洲（从all_countries_info.csv加载的映射）
    Args:
        country_code: 国家二字母代码
    Returns:
        大洲名称，如果未找到则返回None
    """
    code = (country_code or '').strip().upper()
    if not code:
        return None
    return CODE2_TO_CONTINENT.get(code, None)


def init_metric_dict(cases):
    """初始化指标字典"""
    metrics = ['carbon_emission(tCO2)']
    return {case: {m: 0.0 for m in metrics} for case in cases}


def empty_metric_lists(cases):
    """初始化指标列表字典"""
    metrics = ['carbon_emission(tCO2)']
    return {case: {m: [] for m in metrics} for case in cases}


def aggregate_group_averages(base_root: str):
    """遍历2016-2020年，按组聚合每个国家，得到5年平均值。"""
    years = list(range(2016, 2021))
    cases = ['ref'] + [f'case{i}' for i in range(1, 21)]
    groups = ['CN', 'IN', 'US', 'Europe', 'Others']
    # 每组保存每年的总和列表，最后取平均
    group_case_metric_lists = {g: empty_metric_lists(cases) for g in groups}

    for year in years:
        print(f"正在处理 {year} 年数据...")
        # 当年按组的聚合（先对国家累加，再写入列表）
        per_year_group_sums = {g: init_metric_dict(cases) for g in groups}

        # 读取每个case的汇总文件
        for case, file_path in list_case_summary_files(base_root, year):
            try:
                # 使用keep_default_na=False避免'NA'被识别为缺失值
                df = pd.read_csv(file_path, keep_default_na=False)
            except Exception as e:
                print(f"读取 {file_path} 失败: {e}")
                continue

            # 检查必需的列
            if 'Country_Code_2' not in df.columns or 'carbon_emission(tCO2)' not in df.columns:
                continue

            # 遍历每个国家
            for _, row in df.iterrows():
                country_code = str(row['Country_Code_2']).strip().upper()
                if not country_code or country_code == '':
                    continue
                
                # 确定分组
                continent = get_continent_from_country_code(country_code)
                group = determine_group(country_code, continent)
                
                # 读取碳排放量
                emission_value = pd.to_numeric(row.get('carbon_emission(tCO2)', np.nan), errors='coerce')
                if pd.notna(emission_value) and emission_value >= 0:
                    per_year_group_sums[group][case]['carbon_emission(tCO2)'] += float(emission_value)

        # 年度汇总写入列表
        for g in groups:
            for case in cases:
                for col, sum_val in per_year_group_sums[g][case].items():
                    group_case_metric_lists[g][case][col].append(sum_val)

    # 计算5年平均
    group_case_metric_avgs = {g: {case: {} for case in cases} for g in groups}
    for g in groups:
        for case in cases:
            for col, values in group_case_metric_lists[g][case].items():
                group_case_metric_avgs[g][case][col] = float(np.nanmean(values)) if len(values) > 0 else np.nan

    return group_case_metric_avgs, cases, groups


def compute_reductions(group_case_metric_avgs, cases, groups):
    """基于平均后的碳排放量，计算各组各工况的减排量和减排率(%)。"""
    rows = []
    for g in groups:
        # 参考值（ref工况）
        ref_emission = group_case_metric_avgs[g]['ref'].get('carbon_emission(tCO2)', np.nan)

        for case in cases:
            case_emission = group_case_metric_avgs[g][case].get('carbon_emission(tCO2)', np.nan)
            
            # 计算减排量（仅对非ref工况）
            if case != 'ref':
                reduction = ref_emission - case_emission if pd.notna(ref_emission) and pd.notna(case_emission) else np.nan
                # 计算减排率
                reduction_rate = ((reduction / ref_emission) * 100.0) if pd.notna(ref_emission) and ref_emission > 0 and pd.notna(reduction) else np.nan
            else:
                reduction = np.nan
                reduction_rate = np.nan

            rows.append({
                'group': g,
                'index': case,
                'carbon_emission(tCO2)': case_emission,
                'carbon_emission_reduction(tCO2)': reduction,
                'carbon_emission_reduction(%)': reduction_rate,
            })
    return rows


def main():
    base_root = r"Z:\local_environment_creation\carbon_emission\2016-2020\result"
    output_dir = r"Z:\local_environment_creation\carbon_emission\2016-2020\group"
    os.makedirs(output_dir, exist_ok=True)

    print("开始计算分组碳排放量...")
    group_case_metric_avgs, cases, groups = aggregate_group_averages(base_root)
    rows = compute_reductions(group_case_metric_avgs, cases, groups)

    # 排序输出：按组固定顺序，其次按 ref, case1..case20
    group_order = {g: i for i, g in enumerate(['CN', 'IN', 'US', 'Europe', 'Others'])}
    case_order = {c: i for i, c in enumerate(['ref'] + [f'case{i}' for i in range(1, 21)])}
    rows.sort(key=lambda r: (group_order.get(r['group'], 999), case_order.get(r['index'], 999)))

    df_out = pd.DataFrame(rows, columns=[
        'group', 'index',
        'carbon_emission(tCO2)',
        'carbon_emission_reduction(tCO2)',
        'carbon_emission_reduction(%)'
    ])

    out_path = os.path.join(output_dir, 'group_summary_2016_2020_average.csv')
    df_out.to_csv(out_path, index=False)
    print(f'分组5年平均结果已导出: {out_path}')
    print(f'共 {len(df_out)} 行数据')


if __name__ == '__main__':
    main()

