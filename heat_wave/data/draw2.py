"""
绘制case6的节能量和节能率箱线图（双y轴）

功能：
读取country_energy_cooling_多模型.py输出的国家能耗CSV文件，绘制case6的节能量和节能率箱线图。
横轴为global、CN、IN、US、Europe，左y轴为节能量，右y轴为节能率。
数据来自6个模型×5年=30个数据点。

输入数据：
1. 全球汇总：/home/linbor/WORK/lishiying/heat_wave/{模型名}/{SSP路径}/energy/{年份}/country/global_cooling_energy_summary.csv
2. 国家数据：/home/linbor/WORK/lishiying/heat_wave/{模型名}/{SSP路径}/energy/{年份}/country/case6/case6_national_cooling_energy.csv

输出数据：
图片：
/home/linbor/WORK/lishiying/heat_wave/figure/SSP126/case6_energy_reduction_boxplot.png
/home/linbor/WORK/lishiying/heat_wave/figure/SSP245/case6_energy_reduction_boxplot.png

统计数据CSV：
/home/linbor/WORK/lishiying/heat_wave/figure/SSP126/case6_group_boxplot_statistics.csv
/home/linbor/WORK/lishiying/heat_wave/figure/SSP245/case6_group_boxplot_statistics.csv

详细数据CSV：
/home/linbor/WORK/lishiying/heat_wave/figure/SSP126/case6_region_detail_data.csv
/home/linbor/WORK/lishiying/heat_wave/figure/SSP245/case6_region_detail_data.csv

详细数据CSV包含的列：
- Region: 区域名称（global, CN, IN, US, Europe）
- SSP: 发展路径（SSP126或SSP245）
- Model: 气候模型名称
- Year: 年份（2030-2034）
- Energy_Consumption_GWh: 能耗（GWh，来自ref场景）
- Energy_Reduction_GWh: 节能量（GWh）
- Reduction_Rate_Percent: 节能率（%）

统计信息CSV包含的统计信息（每个区域分别计算节能量和节能率）：
- Region: 区域名称
- Metric: 指标类型（Energy_Reduction_TWh 或 Reduction_Rate_Percent）
- Median: 中位数
- Mean: 平均数
- Q1: 第一四分位数（箱体下边界）
- Q3: 第三四分位数（箱体上边界）
- IQR: 四分位距（Q3 - Q1）
- Whisker_Lower: 下须线范围（线的下边界）
- Whisker_Upper: 上须线范围（线的上边界）
- Data_Min: 数据最小值
- Data_Max: 数据最大值
- Outlier_Count: 异常值数量
- Sample_Size: 样本数量
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置路径
BASE_PATH = "/home/linbor/WORK/lishiying"
HEAT_WAVE_BASE_PATH = os.path.join(BASE_PATH, "heat_wave")

# 模型配置（所有6个气候模型）
MODELS = [
    "ACCESS-ESM1-5",
    "BCC-CSM2-MR",
    "CanESM5",
    "EC-Earth3",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0"
]

# SSP配置
SSP_PATHS = ["SSP126", "SSP245"]

# 年份配置
TARGET_YEARS = [2030, 2031, 2032, 2033, 2034]

# Case配置
CASE_NAME = "case6"

# 目标国家/地区
TARGET_REGIONS = {
    'global': 'Global',
    'CN': 'China',
    'IN': 'India',
    'US': 'United States',
    'Europe': 'Europe'  # 需要汇总欧洲所有国家
}

# 不再使用预定义的欧洲国家列表，而是从CSV的continent列筛选


def read_csv_with_encoding(file_path, keep_default_na=True):
    """读取CSV文件，尝试多种编码"""
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, keep_default_na=keep_default_na)
            logger.debug(f"成功使用编码 '{encoding}' 读取文件: {os.path.basename(file_path)}")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    raise ValueError(f"无法使用常见编码读取文件: {file_path}")


def load_global_data(model_name, ssp_path):
    """加载全球汇总数据（6个模型×5年=30个数据点）"""
    global_energy_diff = []  # 节能量（GWh）
    global_reduction = []     # 节能率（%）
    
    for year in TARGET_YEARS:
        csv_path = os.path.join(
            HEAT_WAVE_BASE_PATH,
            model_name,
            ssp_path,
            "energy",
            str(year),
            "country",
            "global_cooling_energy_summary.csv"
        )
        
        if not os.path.exists(csv_path):
            logger.warning(f"文件不存在: {csv_path}")
            continue
        
        try:
            df = read_csv_with_encoding(csv_path, keep_default_na=False)
            
            # 检查是否有case列作为index
            if 'case' in df.columns:
                # case是列
                case6_row = df[df['case'] == CASE_NAME]
                if len(case6_row) == 0:
                    continue
                case6_row = case6_row.iloc[0]
            else:
                # case是index
                df = df.set_index('case')
                if CASE_NAME not in df.index:
                    continue
                case6_row = df.loc[CASE_NAME]
            
            # 提取节能量和节能率
            if 'cooling_demand_diff(GWh)' in case6_row:
                diff_val = pd.to_numeric(case6_row['cooling_demand_diff(GWh)'], errors='coerce')
                if pd.notna(diff_val) and diff_val != '':
                    global_energy_diff.append(diff_val)
            
            if 'cooling_demand_reduction(%)' in case6_row:
                reduction_val = pd.to_numeric(case6_row['cooling_demand_reduction(%)'], errors='coerce')
                if pd.notna(reduction_val) and reduction_val != '':
                    global_reduction.append(reduction_val)
        
        except Exception as e:
            logger.error(f"加载 {csv_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return global_energy_diff, global_reduction


def load_country_data(model_name, ssp_path, country_code):
    """加载指定国家的数据（6个模型×5年=30个数据点）"""
    energy_diff = []
    reduction = []
    
    for year in TARGET_YEARS:
        csv_path = os.path.join(
            HEAT_WAVE_BASE_PATH,
            model_name,
            ssp_path,
            "energy",
            str(year),
            "country",
            CASE_NAME,
            f"{CASE_NAME}_national_cooling_energy.csv"
        )
        
        if not os.path.exists(csv_path):
            logger.warning(f"文件不存在: {csv_path}")
            continue
        
        try:
            df = read_csv_with_encoding(csv_path)
            
            # 查找指定国家
            country_row = df[df['Country_Code_3'] == country_code]
            if len(country_row) > 0:
                row = country_row.iloc[0]
                
                # 提取节能量和节能率
                if 'cooling_demand_diff(GWh)' in row:
                    diff_val = pd.to_numeric(row['cooling_demand_diff(GWh)'], errors='coerce')
                    if pd.notna(diff_val):
                        energy_diff.append(diff_val)
                
                if 'cooling_demand_reduction(%)' in row:
                    reduction_val = pd.to_numeric(row['cooling_demand_reduction(%)'], errors='coerce')
                    if pd.notna(reduction_val):
                        reduction.append(reduction_val)
        
        except Exception as e:
            logger.error(f"加载 {csv_path} 时出错: {e}")
            continue
    
    return energy_diff, reduction


def load_europe_data(model_name, ssp_path):
    """加载欧洲所有国家的汇总数据（6个模型×5年=30个数据点）
    需要重新计算节能率：使用ref和case6的欧洲总能耗计算
    """
    energy_diff = []
    reduction = []
    
    for year in TARGET_YEARS:
        # 读取case6的欧洲数据
        case6_csv_path = os.path.join(
            HEAT_WAVE_BASE_PATH,
            model_name,
            ssp_path,
            "energy",
            str(year),
            "country",
            CASE_NAME,
            f"{CASE_NAME}_national_cooling_energy.csv"
        )
        
        # 读取ref的欧洲数据
        ref_csv_path = os.path.join(
            HEAT_WAVE_BASE_PATH,
            model_name,
            ssp_path,
            "energy",
            str(year),
            "country",
            "ref",
            "ref_national_cooling_energy.csv"
        )
        
        if not os.path.exists(case6_csv_path):
            logger.warning(f"文件不存在: {case6_csv_path}")
            continue
        
        if not os.path.exists(ref_csv_path):
            logger.warning(f"文件不存在: {ref_csv_path}")
            continue
        
        try:
            # 读取case6数据
            case6_df = read_csv_with_encoding(case6_csv_path)
            
            # 筛选欧洲国家（从continent列）
            case6_europe_df = case6_df[case6_df['continent'] == 'Europe']
            
            # 读取ref数据
            ref_df = read_csv_with_encoding(ref_csv_path)
            
            # 筛选欧洲国家（从continent列）
            ref_europe_df = ref_df[ref_df['continent'] == 'Europe']
            
            if len(case6_europe_df) > 0 and len(ref_europe_df) > 0:
                # 汇总欧洲所有国家的节能量（直接使用case6的diff列）
                diff_sum = pd.to_numeric(case6_europe_df['cooling_demand_diff(GWh)'], errors='coerce').sum()
                
                # 重新计算节能率：需要ref和case6的总能耗
                case6_total = pd.to_numeric(case6_europe_df['cooling_energy(GWh)'], errors='coerce').sum()
                ref_total = pd.to_numeric(ref_europe_df['cooling_energy(GWh)'], errors='coerce').sum()
                
                if pd.notna(diff_sum):
                    energy_diff.append(diff_sum)
                
                # 重新计算节能率
                if pd.notna(ref_total) and pd.notna(case6_total) and ref_total > 0:
                    reduction_val = (ref_total - case6_total) / ref_total * 100
                    reduction.append(reduction_val)
                elif pd.notna(diff_sum) and pd.notna(ref_total) and ref_total > 0:
                    # 如果case6_total不可用，使用diff_sum计算
                    reduction_val = diff_sum / ref_total * 100
                    reduction.append(reduction_val)
        
        except Exception as e:
            logger.error(f"加载 {case6_csv_path} 或 {ref_csv_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return energy_diff, reduction


def collect_detailed_data():
    """收集所有区域的详细数据（包含Region, SSP, Model, Year, 能耗, 节能量, 节能率）
    
    Returns:
    --------
    pd.DataFrame
        包含所有详细数据的DataFrame
    """
    detailed_data_list = []
    
    for ssp_path in SSP_PATHS:
        for model_name in MODELS:
            for year in TARGET_YEARS:
                # 1. 处理global数据
                global_csv_path = os.path.join(
                    HEAT_WAVE_BASE_PATH,
                    model_name,
                    ssp_path,
                    "energy",
                    str(year),
                    "country",
                    "global_cooling_energy_summary.csv"
                )
                
                if os.path.exists(global_csv_path):
                    try:
                        df = read_csv_with_encoding(global_csv_path, keep_default_na=False)
                        
                        # 获取case6和ref的数据
                        if 'case' in df.columns:
                            case6_row = df[df['case'] == CASE_NAME]
                            ref_row = df[df['case'] == 'ref']
                            if len(case6_row) > 0 and len(ref_row) > 0:
                                case6_row = case6_row.iloc[0]
                                ref_row = ref_row.iloc[0]
                            else:
                                continue
                        else:
                            df = df.set_index('case')
                            if CASE_NAME in df.index and 'ref' in df.index:
                                case6_row = df.loc[CASE_NAME]
                                ref_row = df.loc['ref']
                            else:
                                continue
                            
                            # 提取数据
                            energy_consumption = pd.to_numeric(ref_row.get('cooling_energy(GWh)', np.nan), errors='coerce')
                            energy_diff = pd.to_numeric(case6_row.get('cooling_demand_diff(GWh)', np.nan), errors='coerce')
                            reduction = pd.to_numeric(case6_row.get('cooling_demand_reduction(%)', np.nan), errors='coerce')
                            
                            if pd.notna(energy_diff) or pd.notna(reduction):
                                detailed_data_list.append({
                                    'Region': 'global',
                                    'SSP': ssp_path,
                                    'Model': model_name,
                                    'Year': year,
                                    'Energy_Consumption_GWh': round(energy_consumption, 4) if pd.notna(energy_consumption) else np.nan,
                                    'Energy_Reduction_GWh': round(energy_diff, 4) if pd.notna(energy_diff) else np.nan,
                                    'Reduction_Rate_Percent': round(reduction, 4) if pd.notna(reduction) else np.nan
                                })
                    except Exception as e:
                        logger.warning(f"加载global数据时出错 ({model_name}, {ssp_path}, {year}): {e}")
                
                # 2. 处理CN数据
                cn_csv_path = os.path.join(
                    HEAT_WAVE_BASE_PATH,
                    model_name,
                    ssp_path,
                    "energy",
                    str(year),
                    "country",
                    CASE_NAME,
                    f"{CASE_NAME}_national_cooling_energy.csv"
                )
                
                ref_csv_path = os.path.join(
                    HEAT_WAVE_BASE_PATH,
                    model_name,
                    ssp_path,
                    "energy",
                    str(year),
                    "country",
                    "ref",
                    "ref_national_cooling_energy.csv"
                )
                
                if os.path.exists(cn_csv_path) and os.path.exists(ref_csv_path):
                    try:
                        case6_df = read_csv_with_encoding(cn_csv_path)
                        ref_df = read_csv_with_encoding(ref_csv_path)
                        
                        # 查找中国
                        cn_case6 = case6_df[case6_df['Country_Code_3'] == 'CHN']
                        cn_ref = ref_df[ref_df['Country_Code_3'] == 'CHN']
                        
                        if len(cn_case6) > 0 and len(cn_ref) > 0:
                            cn_case6_row = cn_case6.iloc[0]
                            cn_ref_row = cn_ref.iloc[0]
                            
                            energy_consumption = pd.to_numeric(cn_ref_row.get('cooling_energy(GWh)', np.nan), errors='coerce')
                            energy_diff = pd.to_numeric(cn_case6_row.get('cooling_demand_diff(GWh)', np.nan), errors='coerce')
                            reduction = pd.to_numeric(cn_case6_row.get('cooling_demand_reduction(%)', np.nan), errors='coerce')
                            
                            if pd.notna(energy_diff) or pd.notna(reduction):
                                detailed_data_list.append({
                                    'Region': 'CN',
                                    'SSP': ssp_path,
                                    'Model': model_name,
                                    'Year': year,
                                    'Energy_Consumption_GWh': round(energy_consumption, 4) if pd.notna(energy_consumption) else np.nan,
                                    'Energy_Reduction_GWh': round(energy_diff, 4) if pd.notna(energy_diff) else np.nan,
                                    'Reduction_Rate_Percent': round(reduction, 4) if pd.notna(reduction) else np.nan
                                })
                    except Exception as e:
                        logger.warning(f"加载CN数据时出错 ({model_name}, {ssp_path}, {year}): {e}")
                
                # 3. 处理IN数据
                if os.path.exists(cn_csv_path) and os.path.exists(ref_csv_path):
                    try:
                        case6_df = read_csv_with_encoding(cn_csv_path)
                        ref_df = read_csv_with_encoding(ref_csv_path)
                        
                        # 查找印度
                        in_case6 = case6_df[case6_df['Country_Code_3'] == 'IND']
                        in_ref = ref_df[ref_df['Country_Code_3'] == 'IND']
                        
                        if len(in_case6) > 0 and len(in_ref) > 0:
                            in_case6_row = in_case6.iloc[0]
                            in_ref_row = in_ref.iloc[0]
                            
                            energy_consumption = pd.to_numeric(in_ref_row.get('cooling_energy(GWh)', np.nan), errors='coerce')
                            energy_diff = pd.to_numeric(in_case6_row.get('cooling_demand_diff(GWh)', np.nan), errors='coerce')
                            reduction = pd.to_numeric(in_case6_row.get('cooling_demand_reduction(%)', np.nan), errors='coerce')
                            
                            if pd.notna(energy_diff) or pd.notna(reduction):
                                detailed_data_list.append({
                                    'Region': 'IN',
                                    'SSP': ssp_path,
                                    'Model': model_name,
                                    'Year': year,
                                    'Energy_Consumption_GWh': round(energy_consumption, 4) if pd.notna(energy_consumption) else np.nan,
                                    'Energy_Reduction_GWh': round(energy_diff, 4) if pd.notna(energy_diff) else np.nan,
                                    'Reduction_Rate_Percent': round(reduction, 4) if pd.notna(reduction) else np.nan
                                })
                    except Exception as e:
                        logger.warning(f"加载IN数据时出错 ({model_name}, {ssp_path}, {year}): {e}")
                
                # 4. 处理US数据
                if os.path.exists(cn_csv_path) and os.path.exists(ref_csv_path):
                    try:
                        case6_df = read_csv_with_encoding(cn_csv_path)
                        ref_df = read_csv_with_encoding(ref_csv_path)
                        
                        # 查找美国
                        us_case6 = case6_df[case6_df['Country_Code_3'] == 'USA']
                        us_ref = ref_df[ref_df['Country_Code_3'] == 'USA']
                        
                        if len(us_case6) > 0 and len(us_ref) > 0:
                            us_case6_row = us_case6.iloc[0]
                            us_ref_row = us_ref.iloc[0]
                            
                            energy_consumption = pd.to_numeric(us_ref_row.get('cooling_energy(GWh)', np.nan), errors='coerce')
                            energy_diff = pd.to_numeric(us_case6_row.get('cooling_demand_diff(GWh)', np.nan), errors='coerce')
                            reduction = pd.to_numeric(us_case6_row.get('cooling_demand_reduction(%)', np.nan), errors='coerce')
                            
                            if pd.notna(energy_diff) or pd.notna(reduction):
                                detailed_data_list.append({
                                    'Region': 'US',
                                    'SSP': ssp_path,
                                    'Model': model_name,
                                    'Year': year,
                                    'Energy_Consumption_GWh': round(energy_consumption, 4) if pd.notna(energy_consumption) else np.nan,
                                    'Energy_Reduction_GWh': round(energy_diff, 4) if pd.notna(energy_diff) else np.nan,
                                    'Reduction_Rate_Percent': round(reduction, 4) if pd.notna(reduction) else np.nan
                                })
                    except Exception as e:
                        logger.warning(f"加载US数据时出错 ({model_name}, {ssp_path}, {year}): {e}")
                
                # 5. 处理Europe数据
                if os.path.exists(cn_csv_path) and os.path.exists(ref_csv_path):
                    try:
                        case6_df = read_csv_with_encoding(cn_csv_path)
                        ref_df = read_csv_with_encoding(ref_csv_path)
                        
                        # 筛选欧洲国家
                        case6_europe_df = case6_df[case6_df['continent'] == 'Europe']
                        ref_europe_df = ref_df[ref_df['continent'] == 'Europe']
                        
                        if len(case6_europe_df) > 0 and len(ref_europe_df) > 0:
                            # 汇总欧洲所有国家的数据
                            energy_consumption = pd.to_numeric(ref_europe_df['cooling_energy(GWh)'], errors='coerce').sum()
                            energy_diff = pd.to_numeric(case6_europe_df['cooling_demand_diff(GWh)'], errors='coerce').sum()
                            
                            # 重新计算节能率
                            case6_total = pd.to_numeric(case6_europe_df['cooling_energy(GWh)'], errors='coerce').sum()
                            ref_total = pd.to_numeric(ref_europe_df['cooling_energy(GWh)'], errors='coerce').sum()
                            
                            if pd.notna(ref_total) and pd.notna(case6_total) and ref_total > 0:
                                reduction = (ref_total - case6_total) / ref_total * 100
                            elif pd.notna(energy_diff) and pd.notna(ref_total) and ref_total > 0:
                                reduction = energy_diff / ref_total * 100
                            else:
                                reduction = np.nan
                            
                            if pd.notna(energy_diff) or pd.notna(reduction):
                                detailed_data_list.append({
                                    'Region': 'Europe',
                                    'SSP': ssp_path,
                                    'Model': model_name,
                                    'Year': year,
                                    'Energy_Consumption_GWh': round(energy_consumption, 4) if pd.notna(energy_consumption) else np.nan,
                                    'Energy_Reduction_GWh': round(energy_diff, 4) if pd.notna(energy_diff) else np.nan,
                                    'Reduction_Rate_Percent': round(reduction, 4) if pd.notna(reduction) else np.nan
                                })
                    except Exception as e:
                        logger.warning(f"加载Europe数据时出错 ({model_name}, {ssp_path}, {year}): {e}")
    
    df = pd.DataFrame(detailed_data_list)
    return df


def save_detailed_data_to_csv(detailed_df, ssp_path, output_path):
    """保存详细数据到CSV文件（按SSP路径分别保存）
    
    Parameters:
    -----------
    detailed_df : pd.DataFrame
        详细数据DataFrame
    ssp_path : str
        SSP路径（SSP126或SSP245）
    output_path : str
        输出CSV文件路径
    """
    # 筛选当前SSP的数据
    ssp_df = detailed_df[detailed_df['SSP'] == ssp_path].copy()
    
    if len(ssp_df) == 0:
        logger.warning(f"  {ssp_path}: 没有找到详细数据，跳过保存")
        return
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为CSV
    ssp_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"详细数据已保存至: {output_path} (共 {len(ssp_df)} 条记录)")


def calculate_boxplot_statistics(region_data_dict):
    """计算箱线图的统计数据
    
    Parameters:
    -----------
    region_data_dict : dict
        区域数据字典，格式：{
            'region_name': {
                'energy_diff': [数据列表],  # 单位：GWh
                'reduction': [数据列表]     # 单位：%
            }
        }
    
    Returns:
    --------
    pd.DataFrame
        包含每个区域、每个指标统计数据的DataFrame
    """
    statistics_list = []
    regions = ['global', 'CN', 'IN', 'US', 'Europe']
    
    for region in regions:
        if region not in region_data_dict:
            continue
        
        # 处理节能量数据（转换为TWh）
        energy_data = region_data_dict[region]['energy_diff']
        if len(energy_data) > 0:
            energy_data_array = np.array(energy_data) / 1000.0  # 转换为TWh
            
            # 计算基本统计量
            median = np.median(energy_data_array)
            mean = np.mean(energy_data_array)
            q1 = np.percentile(energy_data_array, 25)
            q3 = np.percentile(energy_data_array, 75)
            iqr = q3 - q1
            
            # 计算whisker范围（线的范围）
            whisker_lower = max(np.min(energy_data_array), q1 - 1.5 * iqr)
            whisker_upper = min(np.max(energy_data_array), q3 + 1.5 * iqr)
            
            # 实际数据的最小值和最大值
            data_min = np.min(energy_data_array)
            data_max = np.max(energy_data_array)
            
            # 异常值（超出whisker范围的数据点）
            outliers = energy_data_array[(energy_data_array < whisker_lower) | (energy_data_array > whisker_upper)]
            
            statistics_list.append({
                'Region': region,
                'Metric': 'Energy_Reduction_TWh',
                'Median': round(median, 4),
                'Mean': round(mean, 4),
                'Q1': round(q1, 4),
                'Q3': round(q3, 4),
                'IQR': round(iqr, 4),
                'Whisker_Lower': round(whisker_lower, 4),
                'Whisker_Upper': round(whisker_upper, 4),
                'Data_Min': round(data_min, 4),
                'Data_Max': round(data_max, 4),
                'Outlier_Count': len(outliers),
                'Sample_Size': len(energy_data_array)
            })
        
        # 处理节能率数据
        reduction_data = region_data_dict[region]['reduction']
        if len(reduction_data) > 0:
            reduction_data_array = np.array(reduction_data)
            
            # 计算基本统计量
            median = np.median(reduction_data_array)
            mean = np.mean(reduction_data_array)
            q1 = np.percentile(reduction_data_array, 25)
            q3 = np.percentile(reduction_data_array, 75)
            iqr = q3 - q1
            
            # 计算whisker范围（线的范围）
            whisker_lower = max(np.min(reduction_data_array), q1 - 1.5 * iqr)
            whisker_upper = min(np.max(reduction_data_array), q3 + 1.5 * iqr)
            
            # 实际数据的最小值和最大值
            data_min = np.min(reduction_data_array)
            data_max = np.max(reduction_data_array)
            
            # 异常值（超出whisker范围的数据点）
            outliers = reduction_data_array[(reduction_data_array < whisker_lower) | (reduction_data_array > whisker_upper)]
            
            statistics_list.append({
                'Region': region,
                'Metric': 'Reduction_Rate_Percent',
                'Median': round(median, 4),
                'Mean': round(mean, 4),
                'Q1': round(q1, 4),
                'Q3': round(q3, 4),
                'IQR': round(iqr, 4),
                'Whisker_Lower': round(whisker_lower, 4),
                'Whisker_Upper': round(whisker_upper, 4),
                'Data_Min': round(data_min, 4),
                'Data_Max': round(data_max, 4),
                'Outlier_Count': len(outliers),
                'Sample_Size': len(reduction_data_array)
            })
    
    df = pd.DataFrame(statistics_list)
    return df


def save_statistics_to_csv(statistics_df, output_path):
    """保存统计数据到CSV文件
    
    Parameters:
    -----------
    statistics_df : pd.DataFrame
        统计数据DataFrame
    output_path : str
        输出CSV文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为CSV
    statistics_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"统计数据已保存至: {output_path}")


def plot_boxplot_dual_axis(ssp_path, region_data_dict, output_path):
    """绘制双y轴箱线图
    
    Parameters:
    -----------
    ssp_path : str
        SSP路径（SSP126或SSP245）
    region_data_dict : dict
        区域数据字典，格式：{
            'region_name': {
                'energy_diff': [数据列表],
                'reduction': [数据列表]
            }
        }
    output_path : str
        输出图片路径
    """
    
    # 准备数据
    regions = ['global', 'CN', 'IN', 'US', 'Europe']
    energy_data = []
    reduction_data = []
    
    for region in regions:
        if region in region_data_dict:
            energy_data.append(region_data_dict[region]['energy_diff'])
            reduction_data.append(region_data_dict[region]['reduction'])
        else:
            energy_data.append([])
            reduction_data.append([])
    
    # 将节能量从GWh转换为TWh（除以1000）
    energy_data_twh = [[val / 1000.0 for val in data] if data else [] for data in energy_data]
    
    # 分离全球数据和其他组数据
    global_energy_twh = energy_data_twh[0]  # global
    other_energy_twh = energy_data_twh[1:]  # CN, IN, US, Europe
    
    # 创建图形和主坐标轴
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 创建第二个左y轴（用于其他组）
    # 使用twinx创建，然后将其移到左侧
    ax1_secondary = ax1.twinx()
    # 隐藏右侧spine，显示左侧spine
    ax1_secondary.spines['right'].set_visible(False)
    ax1_secondary.spines['left'].set_position(('outward', 40))
    ax1_secondary.spines['left'].set_visible(True)
    ax1_secondary.yaxis.set_ticks_position('left')
    ax1_secondary.yaxis.set_label_position('left')
    
    # 创建右y轴（用于节能率）
    ax2 = ax1.twinx()
    
    # 绘制其他组节能量箱线图（主左y轴ax1，范围0~200）
    bp1_other = ax1.boxplot(
        other_energy_twh,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=True,
        widths=0.35,
        medianprops=dict(color='black', linewidth=1.5),
        meanprops=dict(color='black', linewidth=1.5, linestyle='--'),
        positions=[2, 3, 4, 5]
    )
    
    # 绘制全球节能量箱线图（第二个左y轴ax1_secondary，范围0~800）
    bp1_global = ax1_secondary.boxplot(
        [global_energy_twh],
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=True,
        widths=0.35,
        medianprops=dict(color='black', linewidth=1.5),
        meanprops=dict(color='black', linewidth=1.5, linestyle='--'),
        positions=[1]
    )
    
    # 设置其他组节能量箱线图颜色
    for patch in bp1_other['boxes']:
        patch.set_facecolor('#89C9C8')
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    
    # 设置全球节能量箱线图颜色
    for patch in bp1_global['boxes']:
        patch.set_facecolor('#89C9C8')
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    
    # 绘制节能率箱线图（右y轴，位置稍微偏移）
    bp2 = ax2.boxplot(
        reduction_data,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=True,
        widths=0.35,
        medianprops=dict(color='black', linewidth=1.5),
        meanprops=dict(color='black', linewidth=1.5, linestyle='--'),
        positions=[1.4, 2.4, 3.4, 4.4, 5.4]  # 稍微偏移，避免重叠
    )
    
    # 设置节能率箱线图颜色
    for patch in bp2['boxes']:
        patch.set_facecolor('#F9BEBB')
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    
    # 设置主左y轴（其他组节能量，单位TWh）- 固定范围0~200，每50一个刻度
    ax1.tick_params(axis='y', labelcolor='#89C9C8')
    ax1.set_ylim(0, 200)
    # 设置刻度：0, 50, 100, 150, 200（间隔50，共4段）
    ax1.set_yticks(np.arange(0, 201, 50))
    
    # 设置第二个左y轴（全球节能量，单位TWh）- 固定范围0~800
    ax1_secondary.set_ylabel('Energy Reduction (TWh)', fontsize=12, color='#89C9C8')
    ax1_secondary.tick_params(axis='y', labelcolor='#89C9C8')
    ax1_secondary.set_ylim(0, 800)
    ax1_secondary.set_yticks(np.arange(0, 801, 200))
    
    # 设置右y轴（节能率）- 固定范围15~35，每隔5%一个刻度
    ax2.set_ylabel('Energy Reduction Rate (%)', fontsize=12, color='#F9BEBB')
    ax2.tick_params(axis='y', labelcolor='#F9BEBB')
    ax2.set_ylim(15, 35)
    # 设置刻度间隔为5：15, 20, 25, 30, 35
    ax2.set_yticks(np.arange(15, 36, 5))
    
    # 设置x轴
    ax1.set_xlabel('Region', fontsize=12)
    ax1.set_xticks([1.2, 2.2, 3.2, 4.2, 5.2])  # 两个箱线图的中间位置
    ax1.set_xticklabels(regions)
    
    # 在每个组之间绘制垂直虚线
    # 位置在每组之间：1.5, 2.5, 3.5, 4.5
    for x_pos in [1.7, 2.7, 3.7, 4.7]:
        ax1.axvline(x=x_pos, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
    
    # 设置标题
    ax1.set_title(f'Case6 Energy Reduction and Reduction Rate by Region ({ssp_path})', 
                 fontsize=14, fontweight='bold')
    
    # 添加网格
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 调整布局
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    logger.info(f"箱线图已保存至: {output_path}")
    
    return ax1.get_ylim(), ax2.get_ylim()  # 返回y轴范围，用于统一坐标轴


def main():
    """主函数"""
    try:
        logger.info("=== 开始绘制case6节能量和节能率箱线图（双y轴） ===")
        logger.info(f"处理的模型: {', '.join(MODELS)}")
        logger.info(f"SSP路径: {', '.join(SSP_PATHS)}")
        logger.info(f"目标年份: {TARGET_YEARS}")
        logger.info(f"Case: {CASE_NAME}")
        
        # 第零步：收集所有区域的详细数据
        logger.info(f"\n{'='*80}")
        logger.info("收集所有区域的详细数据...")
        logger.info(f"{'='*80}")
        detailed_df = collect_detailed_data()
        logger.info(f"共收集到 {len(detailed_df)} 条详细数据记录")
        
        # 第一步：收集所有SSP的数据，用于统一计算y轴范围
        all_ssp_data = {}
        for ssp_path in SSP_PATHS:
            logger.info(f"\n{'='*80}")
            logger.info(f"收集SSP路径数据: {ssp_path}")
            logger.info(f"{'='*80}")
            
            # 存储所有区域的数据
            region_data_dict = {}
            
            # 处理每个模型
            for model_name in MODELS:
                logger.info(f"\n加载 {model_name} 的数据...")
                
                # 1. 加载全球数据
                global_energy_diff, global_reduction = load_global_data(model_name, ssp_path)
                if 'global' not in region_data_dict:
                    region_data_dict['global'] = {'energy_diff': [], 'reduction': []}
                region_data_dict['global']['energy_diff'].extend(global_energy_diff)
                region_data_dict['global']['reduction'].extend(global_reduction)
                
                # 2. 加载CN数据
                cn_energy_diff, cn_reduction = load_country_data(model_name, ssp_path, 'CHN')
                if 'CN' not in region_data_dict:
                    region_data_dict['CN'] = {'energy_diff': [], 'reduction': []}
                region_data_dict['CN']['energy_diff'].extend(cn_energy_diff)
                region_data_dict['CN']['reduction'].extend(cn_reduction)
                
                # 3. 加载IN数据
                in_energy_diff, in_reduction = load_country_data(model_name, ssp_path, 'IND')
                if 'IN' not in region_data_dict:
                    region_data_dict['IN'] = {'energy_diff': [], 'reduction': []}
                region_data_dict['IN']['energy_diff'].extend(in_energy_diff)
                region_data_dict['IN']['reduction'].extend(in_reduction)
                
                # 4. 加载US数据
                us_energy_diff, us_reduction = load_country_data(model_name, ssp_path, 'USA')
                if 'US' not in region_data_dict:
                    region_data_dict['US'] = {'energy_diff': [], 'reduction': []}
                region_data_dict['US']['energy_diff'].extend(us_energy_diff)
                region_data_dict['US']['reduction'].extend(us_reduction)
                
                # 5. 加载Europe数据
                europe_energy_diff, europe_reduction = load_europe_data(model_name, ssp_path)
                if 'Europe' not in region_data_dict:
                    region_data_dict['Europe'] = {'energy_diff': [], 'reduction': []}
                region_data_dict['Europe']['energy_diff'].extend(europe_energy_diff)
                region_data_dict['Europe']['reduction'].extend(europe_reduction)
            
            all_ssp_data[ssp_path] = region_data_dict
        
        # 第二步：绘制每个SSP的箱线图（使用固定的y轴范围）
        for ssp_path in SSP_PATHS:
            logger.info(f"\n{'='*80}")
            logger.info(f"绘制SSP路径: {ssp_path}")
            logger.info(f"{'='*80}")
            
            region_data_dict = all_ssp_data[ssp_path]
            
            # 打印数据统计
            for region in ['global', 'CN', 'IN', 'US', 'Europe']:
                if region in region_data_dict:
                    logger.info(f"  {region}: 节能量数据点={len(region_data_dict[region]['energy_diff'])}, "
                              f"节能率数据点={len(region_data_dict[region]['reduction'])}")
            
            # 绘制箱线图
            if region_data_dict:
                output_path = os.path.join(
                    HEAT_WAVE_BASE_PATH,
                    "figure",
                    ssp_path,
                    f"{CASE_NAME}_group_boxplot.png"
                )
                plot_boxplot_dual_axis(ssp_path, region_data_dict, output_path)
                
                # 计算并保存统计数据
                logger.info(f"\n计算统计数据...")
                statistics_df = calculate_boxplot_statistics(region_data_dict)
                csv_output_path = os.path.join(
                    HEAT_WAVE_BASE_PATH,
                    "figure",
                    ssp_path,
                    f"{CASE_NAME}_group_boxplot_statistics.csv"
                )
                save_statistics_to_csv(statistics_df, csv_output_path)
                
                # 保存详细数据
                logger.info(f"\n保存详细数据...")
                detailed_csv_path = os.path.join(
                    HEAT_WAVE_BASE_PATH,
                    "figure",
                    ssp_path,
                    f"{CASE_NAME}_region_detail_data.csv"
                )
                save_detailed_data_to_csv(detailed_df, ssp_path, detailed_csv_path)
            else:
                logger.warning(f"  {ssp_path}: 没有找到任何数据，跳过绘图")
        
        logger.info("\n=== 所有处理完成 ===")
    
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

