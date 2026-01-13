import os
import pandas as pd
import numpy as np


def list_country_files(base_root: str, year: int):
    """列出指定年份下所有大洲的 summary 文件路径，返回 (continent, file_path) 列表。"""
    continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    results = []
    for continent in continents:
        summary_dir = os.path.join(base_root, str(year), continent, 'summary')
        if not os.path.isdir(summary_dir):
            continue
        for fname in os.listdir(summary_dir):
            if fname.endswith('_summary_results.csv'):
                results.append((continent, os.path.join(summary_dir, fname)))
    return results


def determine_group(continent: str, region_code: str) -> str:
    """根据大洲与国家代码划分组：CN、IN、US、Europe、Others。"""
    code = (region_code or '').strip().upper()
    if code in {'CN'}:
        return 'CN'
    if code in {'IN'}:
        return 'IN'
    if code in {'US'}:
        return 'US'
    if continent == 'Europe':
        return 'Europe'
    return 'Others'


def init_metric_dict(cases):
    metrics = [
        'total_demand_sum(TWh)', 'total_demand_diff(TWh)',
        'heating_demand_sum(TWh)', 'heating_demand_diff(TWh)',
        'cooling_demand_sum(TWh)', 'cooling_demand_diff(TWh)'
    ]
    return {case: {m: 0.0 for m in metrics} for case in cases}


def empty_metric_lists(cases):
    metrics = [
        'total_demand_sum(TWh)', 'total_demand_diff(TWh)',
        'heating_demand_sum(TWh)', 'heating_demand_diff(TWh)',
        'cooling_demand_sum(TWh)', 'cooling_demand_diff(TWh)'
    ]
    return {case: {m: [] for m in metrics} for case in cases}


def aggregate_group_averages(base_root: str):
    """遍历2016-2020年，按组聚合每个国家，得到5年平均值。"""
    years = list(range(2016, 2021))
    cases = ['ref'] + [f'case{i}' for i in range(1, 21)]
    groups = ['CN', 'IN', 'US', 'Europe', 'Others']
    # 每组保存每年的总和列表，最后取平均
    group_case_metric_lists = {g: empty_metric_lists(cases) for g in groups}

    # 需要的列名，做容错（如果不存在则当作0）
    required_cols = {
        'total_demand_sum(TWh)', 'total_demand_diff(TWh)',
        'heating_demand_sum(TWh)', 'heating_demand_diff(TWh)',
        'cooling_demand_sum(TWh)', 'cooling_demand_diff(TWh)'
    }

    for year in years:
        # 当年按组的聚合（先对国家累加，再写入列表）
        per_year_group_sums = {g: init_metric_dict(cases) for g in groups}

        for continent, file_path in list_country_files(base_root, year):
            fname = os.path.basename(file_path)
            region_code = fname.split('_')[0]
            if '.' in region_code:
                continue
            group = determine_group(continent, region_code)

            try:
                df = pd.read_csv(file_path, index_col=0)
            except Exception:
                continue

            # 仅在存在时使用所需列
            available = [c for c in required_cols if c in df.columns]
            if not available:
                continue

            for case in cases:
                if case not in df.index:
                    continue
                for col in available:
                    val = pd.to_numeric(df.loc[case, col], errors='coerce')
                    if pd.notna(val):
                        per_year_group_sums[group][case][col] += float(val)

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
    """基于平均后的 sum 指标，计算各组各工况的节能率(%)."""
    # 输出结构为列表行
    rows = []
    for g in groups:
        # 参考值
        ref_total = group_case_metric_avgs[g]['ref'].get('total_demand_sum(TWh)', np.nan)
        ref_heat = group_case_metric_avgs[g]['ref'].get('heating_demand_sum(TWh)', np.nan)
        ref_cool = group_case_metric_avgs[g]['ref'].get('cooling_demand_sum(TWh)', np.nan)

        for case in cases:
            total_sum = group_case_metric_avgs[g][case].get('total_demand_sum(TWh)', np.nan)
            total_diff = group_case_metric_avgs[g][case].get('total_demand_diff(TWh)', np.nan)
            heat_sum = group_case_metric_avgs[g][case].get('heating_demand_sum(TWh)', np.nan)
            heat_diff = group_case_metric_avgs[g][case].get('heating_demand_diff(TWh)', np.nan)
            cool_sum = group_case_metric_avgs[g][case].get('cooling_demand_sum(TWh)', np.nan)
            cool_diff = group_case_metric_avgs[g][case].get('cooling_demand_diff(TWh)', np.nan)

            total_red = ((ref_total - total_sum) / ref_total * 100.0) if pd.notna(ref_total) and ref_total > 0 else np.nan
            heat_red = ((ref_heat - heat_sum) / ref_heat * 100.0) if pd.notna(ref_heat) and ref_heat > 0 else np.nan
            cool_red = ((ref_cool - cool_sum) / ref_cool * 100.0) if pd.notna(ref_cool) and ref_cool > 0 else np.nan

            rows.append({
                'group': g,
                'index': case,
                'total_demand_sum(TWh)': total_sum,
                'total_demand_diff(TWh)': total_diff,
                'total_demand_reduction(%)': total_red,
                'heating_demand_sum(TWh)': heat_sum,
                'heating_demand_diff(TWh)': heat_diff,
                'heating_demand_reduction(%)': heat_red,
                'cooling_demand_sum(TWh)': cool_sum,
                'cooling_demand_diff(TWh)': cool_diff,
                'cooling_demand_reduction(%)': cool_red,
            })
    return rows


def main():
    base_root = r"Z:\local_environment_creation\energy_consumption\2016-2020result"
    output_dir = os.path.join(base_root, 'group')
    os.makedirs(output_dir, exist_ok=True)

    group_case_metric_avgs, cases, groups = aggregate_group_averages(base_root)
    rows = compute_reductions(group_case_metric_avgs, cases, groups)

    # 排序输出：按组固定顺序，其次按 ref, case1..case20
    group_order = {g: i for i, g in enumerate(['CN', 'IN', 'US', 'Europe', 'Others'])}
    case_order = {c: i for i, c in enumerate(['ref'] + [f'case{i}' for i in range(1, 21)])}
    rows.sort(key=lambda r: (group_order.get(r['group'], 999), case_order.get(r['index'], 999)))

    df_out = pd.DataFrame(rows, columns=[
        'group', 'index',
        'total_demand_sum(TWh)', 'total_demand_diff(TWh)', 'total_demand_reduction(%)',
        'heating_demand_sum(TWh)', 'heating_demand_diff(TWh)', 'heating_demand_reduction(%)',
        'cooling_demand_sum(TWh)', 'cooling_demand_diff(TWh)', 'cooling_demand_reduction(%)'
    ])

    out_path = os.path.join(output_dir, 'group_summary_2016_2020_average.csv')
    df_out.to_csv(out_path, index=False)
    print(f'分组5年平均结果已导出: {out_path}')


if __name__ == '__main__':
    main()


