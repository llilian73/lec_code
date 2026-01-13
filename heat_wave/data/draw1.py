"""
绘制case6的节能百分比箱线图

功能：
读取country_energy_cooling_多模型.py输出的国家能耗CSV文件，绘制case6的节能百分比箱线图。
横轴为6个模型（M1-M6），纵轴为节能百分比（cooling_demand_reduction(%)）。
数据来自5年所有国家的节能百分比。

输入数据：
/home/linbor/WORK/lishiying/heat_wave/{模型名}/{SSP路径}/energy/{年份}/country/case6/case6_national_cooling_energy.csv

输出数据：
图片：
/home/linbor/WORK/lishiying/heat_wave/figure/SSP126/case6_reduction_boxplot.png
/home/linbor/WORK/lishiying/heat_wave/figure/SSP245/case6_reduction_boxplot.png

统计数据CSV：
/home/linbor/WORK/lishiying/heat_wave/figure/SSP126/case6_reduction_boxplot_statistics.csv
/home/linbor/WORK/lishiying/heat_wave/figure/SSP245/case6_reduction_boxplot_statistics.csv

CSV包含的统计信息：
- Model: 模型名称（M1-M6为各模型，All_Models为所有模型合并）
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

注意：CSV中会包含每个模型的统计行（M1-M6），以及最后一行"All_Models"表示所有6个模型数据合并后的统计。
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

# 模型映射（模型名 -> 显示名称）
MODEL_DISPLAY_NAMES = {
    "ACCESS-ESM1-5": "M1",
    "BCC-CSM2-MR": "M2",
    "CanESM5": "M3",
    "EC-Earth3": "M4",
    "MPI-ESM1-2-HR": "M5",
    "MRI-ESM2-0": "M6"
}

# 模型颜色配置（按M1-M6顺序）
MODEL_COLORS = [
    '#F1C3C1',  # M1 - 浅粉色
    '#B7D5EC',  # M2 - 浅蓝色
    '#FEDFB1',  # M3 - 浅橙色/米色
    '#B9C3DC',  # M4 - 浅紫色/蓝灰色
    '#B2D3A4',  # M5 - 浅绿色
    '#BD88B0'   # M6 - 浅黄色/桃色
]

# SSP配置
SSP_PATHS = ["SSP126", "SSP245"]

# 年份配置
TARGET_YEARS = [2030, 2031, 2032, 2033, 2034]

# Case配置
CASE_NAME = "case6"


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


def load_case6_data(model_name, ssp_path):
    """加载指定模型和SSP路径下所有年份的case6数据"""
    all_reduction_data = []  # 存储所有年份所有国家的节能百分比
    
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
            
            # 提取节能百分比列
            if 'cooling_demand_reduction(%)' in df.columns:
                # 转换为数值类型，处理空字符串和无效值
                reduction_values = pd.to_numeric(df['cooling_demand_reduction(%)'], errors='coerce')
                # 移除NaN值
                reduction_values = reduction_values.dropna()
                # 添加到列表
                all_reduction_data.extend(reduction_values.tolist())
                logger.info(f"  {year}年: 加载了 {len(reduction_values)} 个国家的数据")
            else:
                logger.warning(f"  {year}年: CSV文件中没有 'cooling_demand_reduction(%)' 列")
        
        except Exception as e:
            logger.error(f"加载 {csv_path} 时出错: {e}")
            continue
    
    return all_reduction_data


def calculate_boxplot_statistics(model_data_dict):
    """计算箱线图的统计数据
    
    Parameters:
    -----------
    model_data_dict : dict
        模型数据字典，格式：{模型显示名: [数据列表]}
    
    Returns:
    --------
    pd.DataFrame
        包含每个模型统计数据和所有模型组合统计数据的DataFrame
    """
    statistics_list = []
    
    # 收集所有模型的数据用于合并统计
    all_models_data = []
    
    # 计算每个模型的统计
    for model_name, data in model_data_dict.items():
        if len(data) == 0:
            continue
        
        data_array = np.array(data)
        all_models_data.extend(data)  # 收集所有数据用于合并统计
        
        # 计算基本统计量
        median = np.median(data_array)
        mean = np.mean(data_array)
        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        
        # 计算whisker范围（线的范围）
        # 默认whisker范围是 Q1 - 1.5*IQR 到 Q3 + 1.5*IQR，但不超过实际数据范围
        whisker_lower = max(np.min(data_array), q1 - 1.5 * iqr)
        whisker_upper = min(np.max(data_array), q3 + 1.5 * iqr)
        
        # 实际数据的最小值和最大值
        data_min = np.min(data_array)
        data_max = np.max(data_array)
        
        # 异常值（超出whisker范围的数据点）
        outliers = data_array[(data_array < whisker_lower) | (data_array > whisker_upper)]
        
        statistics_list.append({
            'Model': model_name,
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
            'Sample_Size': len(data_array)
        })
    
    # 计算所有模型合并的统计
    if len(all_models_data) > 0:
        all_data_array = np.array(all_models_data)
        
        # 计算基本统计量
        median = np.median(all_data_array)
        mean = np.mean(all_data_array)
        q1 = np.percentile(all_data_array, 25)
        q3 = np.percentile(all_data_array, 75)
        iqr = q3 - q1
        
        # 计算whisker范围（线的范围）
        whisker_lower = max(np.min(all_data_array), q1 - 1.5 * iqr)
        whisker_upper = min(np.max(all_data_array), q3 + 1.5 * iqr)
        
        # 实际数据的最小值和最大值
        data_min = np.min(all_data_array)
        data_max = np.max(all_data_array)
        
        # 异常值（超出whisker范围的数据点）
        outliers = all_data_array[(all_data_array < whisker_lower) | (all_data_array > whisker_upper)]
        
        statistics_list.append({
            'Model': 'All_Models',
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
            'Sample_Size': len(all_data_array)
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


def plot_boxplot(ssp_path, model_data_dict, output_path):
    """绘制箱线图
    
    Parameters:
    -----------
    ssp_path : str
        SSP路径（SSP126或SSP245）
    model_data_dict : dict
        模型数据字典，格式：{模型显示名: [数据列表]}
    output_path : str
        输出图片路径
    """
    # 准备数据
    labels = list(model_data_dict.keys())
    data = list(model_data_dict.values())
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制箱线图
    bp = ax.boxplot(
        data,
        tick_labels=labels,  # 使用新的参数名（Matplotlib 3.9+）
        patch_artist=True,
        showmeans=True,  # 显示均值
        meanline=True,   # 均值用线表示
        showfliers=True,  # 显示异常值
        widths=0.6,  # 减小箱线图宽度，使间距更近
        medianprops=dict(color='black', linewidth=1.5),  # 中位数线为黑色
        meanprops=dict(color='black', linewidth=1.5, linestyle='--')  # 平均值线为黑色虚线
    )
    
    # 设置箱线图颜色（按M1-M6顺序使用对应颜色）
    # 根据标签确定模型索引（M1对应索引0，M2对应索引1，以此类推）
    for i, (label, patch) in enumerate(zip(labels, bp['boxes'])):
        # 从标签中提取模型编号（M1 -> 0, M2 -> 1, ...）
        try:
            model_num = int(label.replace('M', '')) - 1  # M1 -> 0, M2 -> 1, ...
            if 0 <= model_num < len(MODEL_COLORS):
                color = MODEL_COLORS[model_num]
            else:
                color = '#CCCCCC'  # 默认灰色
        except (ValueError, AttributeError):
            # 如果无法解析标签，使用索引
            color = MODEL_COLORS[i] if i < len(MODEL_COLORS) else '#CCCCCC'
        
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    
    # 设置标题和标签
    # ax.set_title(f'Case6 Cooling Energy Reduction Percentage by Model ({ssp_path})', 
    #              fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Cooling Demand Reduction (%)', fontsize=12)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 设置y轴范围（可选，根据数据调整）
    # ax.set_ylim([0, 100])
    
    # 调整布局
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    logger.info(f"箱线图已保存至: {output_path}")


def main():
    """主函数"""
    try:
        logger.info("=== 开始绘制case6节能百分比箱线图 ===")
        logger.info(f"处理的模型: {', '.join(MODELS)}")
        logger.info(f"SSP路径: {', '.join(SSP_PATHS)}")
        logger.info(f"目标年份: {TARGET_YEARS}")
        logger.info(f"Case: {CASE_NAME}")
        
        # 处理每个SSP路径
        for ssp_path in SSP_PATHS:
            logger.info(f"\n{'='*80}")
            logger.info(f"处理SSP路径: {ssp_path}")
            logger.info(f"{'='*80}")
            
            # 存储所有模型的数据
            model_data_dict = {}
            
            # 处理每个模型
            for model_name in MODELS:
                if model_name not in MODEL_DISPLAY_NAMES:
                    logger.warning(f"模型 {model_name} 没有对应的显示名称，跳过")
                    continue
                
                display_name = MODEL_DISPLAY_NAMES[model_name]
                logger.info(f"\n加载 {model_name} ({display_name}) 的数据...")
                
                # 加载该模型所有年份的case6数据
                reduction_data = load_case6_data(model_name, ssp_path)
                
                if len(reduction_data) > 0:
                    model_data_dict[display_name] = reduction_data
                    logger.info(f"  {model_name} ({display_name}): 共 {len(reduction_data)} 个数据点")
                else:
                    logger.warning(f"  {model_name} ({display_name}): 没有找到数据")
            
            # 如果有数据，绘制箱线图
            if model_data_dict:
                # 确保模型按M1-M6顺序排列
                sorted_models = sorted(model_data_dict.keys(), key=lambda x: int(x.replace('M', '')))
                sorted_data_dict = {k: model_data_dict[k] for k in sorted_models}
                
                # 输出路径：统一输出到heat_wave/figure/{SSP路径}/
                output_path = os.path.join(
                    HEAT_WAVE_BASE_PATH,
                    "figure",
                    ssp_path,
                    f"{CASE_NAME}_reduction_boxplot.png"
                )
                plot_boxplot(ssp_path, sorted_data_dict, output_path)
                
                # 计算并保存统计数据
                logger.info(f"\n计算统计数据...")
                statistics_df = calculate_boxplot_statistics(sorted_data_dict)
                csv_output_path = os.path.join(
                    HEAT_WAVE_BASE_PATH,
                    "figure",
                    ssp_path,
                    f"{CASE_NAME}_reduction_boxplot_statistics.csv"
                )
                save_statistics_to_csv(statistics_df, csv_output_path)
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

