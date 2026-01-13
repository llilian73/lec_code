"""
国家级别制冷能耗计算工具

功能：
读取energy_last_for_all_hw_last.py输出的点级能耗CSV文件，计算各个国家的制冷能耗。
支持多模型、多SSP、多年份处理。

输入数据：
1. 点级能耗CSV文件：/home/linbor/WORK/lishiying/{HEAT_WAVE_FOLDER}/{模型名}/{SSP路径}/energy/{年份}/point/{case_name}/{case_name}.csv
   （HEAT_WAVE_FOLDER根据FUTURE_FOLDER自动生成，例如：future_56_60 -> heat_wave_56_60）
2. 国家边界Shapefile：/home/linbor/WORK/lishiying/shapefiles/world_border2.shp
3. 国家信息CSV：/home/linbor/WORK/lishiying/shapefiles/all_countries_info.csv

输出数据：
- 路径：/home/linbor/WORK/lishiying/{HEAT_WAVE_FOLDER}/{模型名}/{SSP路径}/energy/{年份}/country/
- 每个年份文件夹下包含：
  1. global_cooling_energy_summary.csv：全球汇总
  2. {case_name}/：每个工况的文件夹
     - {case_name}_national_cooling_energy.csv：国家能耗数据
     - {case_name}_cooling_energy_heatmap.pdf：制冷能耗热图
     - {case_name}_cooling_demand_diff_heatmap.pdf：能耗差异热图
     - {case_name}_cooling_demand_reduction_heatmap.pdf：节能率热图
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import numpy as np
from shapely.geometry import Point
import logging
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 配置路径 ---
BASE_PATH = "/home/linbor/WORK/lishiying"
HEAT_WAVE_BASE_PATH = os.path.join(BASE_PATH, "heat_wave_56_60")
COUNTRIES_SHP_PATH = os.path.join(BASE_PATH, "shapefiles", "world_border2.shp")
COUNTRIES_INFO_CSV = os.path.join(BASE_PATH, "shapefiles", "all_countries_info.csv")

# 模型配置
MODELS = ["BCC-CSM2-MR", "CanESM5", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
# "EC-Earth3",
# SSP配置
SSP_PATHS = ["SSP126", "SSP245"]

# 年份配置（可根据需要修改，例如：[2051, 2052, 2053, 2054, 2055] 或 [2056, 2057, 2058, 2059, 2060]）
TARGET_YEARS = [2056, 2057, 2058, 2059, 2060]

# 工况配置（ref + case1-20）
cases = ['ref'] + [f'case{i}' for i in range(1, 21)]


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


def load_point_energy_data(csv_path):
    """加载点级能耗CSV文件"""
    if not os.path.exists(csv_path):
        logger.warning(f"文件不存在: {csv_path}")
        return None
    
    try:
        data = pd.read_csv(csv_path, usecols=['lat', 'lon', 'total_demand', 'heating_demand', 'cooling_demand'])
        
        # 确保数值列为float类型，无效值设为NaN
        data['total_demand'] = pd.to_numeric(data['total_demand'], errors='coerce')
        data['heating_demand'] = pd.to_numeric(data['heating_demand'], errors='coerce')
        data['cooling_demand'] = pd.to_numeric(data['cooling_demand'], errors='coerce')
        
        # 移除包含NaN的行
        data = data.dropna(subset=['cooling_demand'])
        
        logger.info(f"已加载 {len(data)} 个点的数据")
        return data
    except Exception as e:
        logger.error(f"加载文件失败 {csv_path}: {e}")
        return None


def calculate_national_energy(input_data, shp_path, countries_info_df, output_csv_path):
    """计算每个国家的总能耗并保存为CSV"""
    logger.info("正在处理国家能耗数据...")
    
    # --- 读取国家边界Shapefile ---
    countries = gpd.read_file(shp_path)
    logger.info(f"国家数量: {len(countries)}")
    
    if countries.empty:
        raise ValueError("没有找到国家数据，请检查shapefile是否为空或路径是否正确。")
    
    # 确保几何图形有效
    countries['geometry'] = countries.geometry.buffer(0)
    countries['geometry'] = countries.geometry.apply(lambda geom: geom.make_valid() if not geom.is_valid else geom)
    
    if countries.crs is None:
        logger.warning("Countries GeoDataFrame has no CRS defined. Setting to EPSG:4326.")
        countries = countries.set_crs("EPSG:4326", allow_override=True)
    
    # --- 将点数据转换为GeoDataFrame ---
    try:
        geometry = [Point(xy) for xy in zip(input_data['lon'], input_data['lat'])]
        gdf_points = gpd.GeoDataFrame(input_data, geometry=geometry, crs="EPSG:4326")
        logger.info(f"已读取 {len(gdf_points)} 个能耗点数据。")
    except Exception as e:
        logger.error(f"转换点数据时发生错误: {e}")
        raise
    
    # --- 空间连接并聚合能耗数据 ---
    logger.info("正在进行空间连接并聚合能耗数据...")
    try:
        # 确保坐标系匹配
        if gdf_points.crs != countries.crs:
            gdf_points = gdf_points.to_crs(countries.crs)
            logger.warning(f"点数据CRS不匹配，已转换为 {countries.crs}")
        
        joined_gdf = gpd.sjoin(gdf_points, countries, how="inner", predicate="within")
        
        # 按国家聚合制冷能耗（使用GID_0作为国家代码）
        national_energy = joined_gdf.groupby('GID_0')['cooling_demand'].sum().reset_index()
        national_energy.rename(columns={'GID_0': 'Country_Code_3', 'cooling_demand': 'cooling_energy(GWh)'}, inplace=True)
        
        # 合并HKG和MAC到中国（CHN）
        hkg_energy = 0
        mac_energy = 0
        if 'HKG' in national_energy['Country_Code_3'].values:
            hkg_energy = national_energy[national_energy['Country_Code_3'] == 'HKG']['cooling_energy(GWh)'].iloc[0]
            national_energy = national_energy[national_energy['Country_Code_3'] != 'HKG']
        
        if 'MAC' in national_energy['Country_Code_3'].values:
            mac_energy = national_energy[national_energy['Country_Code_3'] == 'MAC']['cooling_energy(GWh)'].iloc[0]
            national_energy = national_energy[national_energy['Country_Code_3'] != 'MAC']
        
        # 将HKG和MAC的能耗加到CHN
        if 'CHN' in national_energy['Country_Code_3'].values:
            chn_idx = national_energy[national_energy['Country_Code_3'] == 'CHN'].index[0]
            national_energy.loc[chn_idx, 'cooling_energy(GWh)'] += (hkg_energy + mac_energy)
        else:
            # 如果只有HKG/MAC没有CHN，则创建CHN记录
            if hkg_energy > 0 or mac_energy > 0:
                national_energy = pd.concat(
                    [national_energy, pd.DataFrame({'Country_Code_3': ['CHN'], 'cooling_energy(GWh)': [hkg_energy + mac_energy]})],
                    ignore_index=True)
        
        # 合并国家信息（大洲、国家名称等）
        national_energy = national_energy.merge(
            countries_info_df[['Country_Code_3', 'continent', 'Country_Name']],
            on='Country_Code_3',
            how='left'
        )
        
        # 重新排列列顺序
        national_energy = national_energy[['continent', 'Country_Code_3', 'Country_Name', 'cooling_energy(GWh)']]
        
        # 按大洲排序（缺失值放在最后）
        # 定义大洲排序顺序
        continent_order = ['Asia', 'Europe', 'Africa', 'North America', 'South America', 'Oceania']
        national_energy['continent_sort'] = national_energy['continent'].apply(
            lambda x: continent_order.index(x) if x in continent_order else len(continent_order)
        )
        national_energy = national_energy.sort_values(['continent_sort', 'Country_Name']).drop('continent_sort', axis=1)
        national_energy = national_energy.reset_index(drop=True)
        
        logger.info(f"已聚合 {len(national_energy)} 个国家的能耗数据。")
        
        # 保存结果到CSV
        national_energy.to_csv(output_csv_path, index=False)
        logger.info(f"结果已保存至: {output_csv_path}")
        
        return national_energy
        
    except Exception as e:
        logger.error(f"空间连接或数据聚合时发生错误: {e}")
        raise


def plot_heatmap(national_energy_df, shp_path, column_name, title, output_pdf_path, vmin=None, vmax=None, cmap='Oranges'):
    """绘制热图"""
    # 注意：在多进程环境中，减少日志输出以避免竞争
    # logger.debug(f"正在绘制 {column_name} 热图...")
    
    # 读取国家边界
    countries_gdf = gpd.read_file(shp_path)
    
    # 确保几何图形有效
    countries_gdf['geometry'] = countries_gdf.geometry.buffer(0)
    countries_gdf['geometry'] = countries_gdf.geometry.apply(lambda geom: geom.make_valid() if not geom.is_valid else geom)
    
    # 合并能耗数据到GeoDataFrame
    merged_gdf = countries_gdf.merge(
        national_energy_df,
        left_on='GID_0',
        right_on='Country_Code_3',
        how='left'
    )
    
    # 特殊处理CHN、HKG、MAC，确保它们使用相同的值
    chn_value = 0
    if 'CHN' in national_energy_df['Country_Code_3'].values:
        chn_row = national_energy_df[national_energy_df['Country_Code_3'] == 'CHN']
        if len(chn_row) > 0:
            chn_val = chn_row[column_name].iloc[0]
            # 处理可能的空字符串
            if chn_val != '' and pd.notna(chn_val):
                chn_value = float(chn_val)
    
    # 为CHN、HKG、MAC设置相同的值
    merged_gdf.loc[merged_gdf['GID_0'] == 'CHN', column_name] = chn_value
    merged_gdf.loc[merged_gdf['GID_0'] == 'HKG', column_name] = chn_value
    merged_gdf.loc[merged_gdf['GID_0'] == 'MAC', column_name] = chn_value
    
    # 将列转换为数值类型（处理空字符串）
    merged_gdf[column_name] = pd.to_numeric(merged_gdf[column_name], errors='coerce')
    
    # 填充NaN值为0
    merged_gdf[column_name] = merged_gdf[column_name].fillna(0)
    
    # 创建地图
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    
    # 确定颜色范围
    if vmin is None:
        vmin = merged_gdf[column_name].min()
    if vmax is None:
        vmax = merged_gdf[column_name].max()
    
    # 绘制热图
    merged_gdf.plot(
        column=column_name,
        cmap=cmap,
        linewidth=0.5,
        ax=ax,
        edgecolor='0.3',
        legend=True,
        legend_kwds={'label': title, 'orientation': "horizontal", 'shrink': 0.8, 'pad': 0.02},
        vmin=vmin,
        vmax=vmax
    )
    
    ax.set_title(title, fontsize=16)
    ax.set_axis_off()
    
    # 保存为PDF格式
    plt.savefig(output_pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    # 在多进程环境中，减少日志输出以避免竞争
    # logger.debug(f"热图已保存至: {output_pdf_path}")


def process_single_case(model_name, ssp_path, year, case, ref_national_energy=None):
    """处理单个case的数据（用于多进程并行）"""
    try:
        # 输入路径
        energy_base_path = os.path.join(HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", str(year), "point")
        
        # 输出路径
        output_base_dir = os.path.join(HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", str(year), "country")
        os.makedirs(output_base_dir, exist_ok=True)
        
        # 读取国家信息CSV（用于获取大洲信息）
        if not os.path.exists(COUNTRIES_INFO_CSV):
            raise FileNotFoundError(f"国家信息文件不存在: {COUNTRIES_INFO_CSV}")
        countries_info_df = read_csv_with_encoding(COUNTRIES_INFO_CSV)
        
        # 输入CSV路径
        csv_input_path = os.path.join(energy_base_path, case, f"{case}.csv")
        
        # 加载点级能耗数据
        point_data = load_point_energy_data(csv_input_path)
        
        if point_data is None or len(point_data) == 0:
            logger.warning(f"跳过 {case}，因为没有找到数据")
            return {
                'case': case,
                'year': year,
                'national_energy': None,
                'total_sum': None,
                'max_energy': None
            }
        
        # 创建工况输出目录
        case_output_dir = os.path.join(output_base_dir, case)
        os.makedirs(case_output_dir, exist_ok=True)
        
        # 输出CSV路径
        csv_output_path = os.path.join(case_output_dir, f"{case}_national_cooling_energy.csv")
        
        # 计算并保存国家能耗
        national_energy = calculate_national_energy(point_data, COUNTRIES_SHP_PATH, countries_info_df, csv_output_path)
        
        # 计算全球总能耗和最大能耗
        current_total_sum = national_energy['cooling_energy(GWh)'].sum()
        max_energy = float(national_energy['cooling_energy(GWh)'].max())
        
        # 如果是非ref工况，需要计算差异和节能率
        if case != 'ref' and ref_national_energy is not None:
            # 验证：确保ref数据是DataFrame类型
            if not isinstance(ref_national_energy, pd.DataFrame):
                logger.error(f"  [错误] {year}年 {case}工况：ref_national_energy不是DataFrame类型")
                raise TypeError(f"ref_national_energy必须是DataFrame，但得到{type(ref_national_energy)}")
            
            # 验证：检查ref数据是否包含必要的列
            if 'cooling_energy(GWh)' not in ref_national_energy.columns:
                logger.error(f"  [错误] {year}年 {case}工况：ref数据缺少'cooling_energy(GWh)'列")
                raise ValueError(f"ref数据缺少必要的列")
            
            logger.debug(f"  [验证] {year}年 {case}工况：使用对应年份的ref数据（{len(ref_national_energy)}个国家）")
            
            ref_national = ref_national_energy.set_index('Country_Code_3')
            current_national = national_energy.set_index('Country_Code_3')
            
            # 计算差异和节能率
            diff_list = []
            reduction_list = []
            
            for idx, row in national_energy.iterrows():
                country_code = row['Country_Code_3']
                current_value = row['cooling_energy(GWh)']
                
                if country_code in ref_national.index:
                    ref_value = ref_national.loc[country_code, 'cooling_energy(GWh)']
                    diff = ref_value - current_value
                    diff_list.append(diff)
                    
                    if ref_value > 0:
                        reduction = (diff / ref_value) * 100
                    else:
                        reduction = 0
                    reduction_list.append(reduction)
                else:
                    diff_list.append(0)
                    reduction_list.append(0)
            
            # 添加差异和节能率列
            national_energy = national_energy.reset_index(drop=True)
            national_energy['cooling_demand_diff(GWh)'] = diff_list
            national_energy['cooling_demand_reduction(%)'] = reduction_list
            
            # 重新排列列顺序
            national_energy = national_energy[['continent', 'Country_Code_3', 'Country_Name', 
                                               'cooling_energy(GWh)', 'cooling_demand_diff(GWh)', 
                                               'cooling_demand_reduction(%)']]
        else:
            # ref工况不需要差异和节能率
            national_energy['cooling_demand_diff(GWh)'] = ''
            national_energy['cooling_demand_reduction(%)'] = ''
        
        # 按大洲排序
        continent_order = ['Asia', 'Europe', 'Africa', 'North America', 'South America', 'Oceania']
        national_energy['continent_sort'] = national_energy['continent'].apply(
            lambda x: continent_order.index(x) if pd.notna(x) and x in continent_order else len(continent_order)
        )
        national_energy = national_energy.sort_values(['continent_sort', 'Country_Name']).drop('continent_sort', axis=1)
        national_energy = national_energy.reset_index(drop=True)
        
        # 保存CSV
        national_energy.to_csv(csv_output_path, index=False)
        
        return {
            'case': case,
            'year': year,
            'national_energy': national_energy,
            'total_sum': current_total_sum,
            'max_energy': max_energy
        }
    except Exception as e:
        logger.error(f"处理 {case} - {year} 年时出错: {e}", exc_info=True)
        return {
            'case': case,
            'year': year,
            'national_energy': None,
            'total_sum': None,
            'max_energy': None
        }


def process_single_year(model_name, ssp_path, year, num_processes=40):
    """处理单个年份的数据（使用多进程并行）"""
    logger.info(f"\n=== 处理 {model_name} - {ssp_path} - {year} 年 ===")
    
    # 输出路径
    output_base_dir = os.path.join(HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", str(year), "country")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 存储所有工况的结果
    all_case_results = {}
    global_summary = {}
    ref_max_energy = 0.0
    ref_total_sum = 0.0
    
    # 第一阶段：并行处理所有ref工况
    logger.info("第一阶段：并行处理ref工况...")
    ref_tasks = [(model_name, ssp_path, year, 'ref', None)]
    
    with ProcessPoolExecutor(max_workers=min(num_processes, len(ref_tasks))) as executor:
        ref_futures = {executor.submit(process_single_case, *task): task for task in ref_tasks}
        for future in as_completed(ref_futures):
            result = future.result()
            if result['national_energy'] is not None:
                all_case_results['ref'] = result['national_energy']
                global_summary['ref'] = {'cooling_demand_sum(GWh)': result['total_sum']}
                ref_max_energy = result['max_energy']
                ref_total_sum = result['total_sum']
                logger.info(f"Ref工况处理完成，全球总能耗: {ref_total_sum:.2f} GWh，最大能耗: {ref_max_energy}")
    
    if 'ref' not in all_case_results:
        logger.error("Ref工况处理失败，无法继续处理其他工况")
        return
    
    # 第二阶段：并行处理所有case1-20工况
    logger.info("第二阶段：并行处理case1-20工况...")
    case_tasks = []
    for case in [f'case{i}' for i in range(1, 21)]:
        case_tasks.append((model_name, ssp_path, year, case, all_case_results['ref']))
    
    with ProcessPoolExecutor(max_workers=min(num_processes, len(case_tasks))) as executor:
        case_futures = {executor.submit(process_single_case, *task): task for task in case_tasks}
        for future in as_completed(case_futures):
            result = future.result()
            if result['national_energy'] is not None:
                all_case_results[result['case']] = result['national_energy']
                global_summary[result['case']] = {'cooling_demand_sum(GWh)': result['total_sum']}
                logger.info(f"{result['case']}工况处理完成，全球总能耗: {result['total_sum']:.2f} GWh")
    
    # 计算每个工况相对于ref的能耗差异和节能率（全球汇总）
    if ref_total_sum > 0:
        for case in cases:
            if case == 'ref':
                global_summary['ref']['cooling_demand_diff(GWh)'] = ''
                global_summary['ref']['cooling_demand_reduction(%)'] = ''
            else:
                current_sum = global_summary.get(case, {}).get('cooling_demand_sum(GWh)')
                if current_sum is not None:
                    diff = ref_total_sum - current_sum
                    reduction_percent = (diff / ref_total_sum) * 100
                    global_summary[case]['cooling_demand_diff(GWh)'] = diff
                    global_summary[case]['cooling_demand_reduction(%)'] = reduction_percent
                else:
                    global_summary[case]['cooling_demand_diff(GWh)'] = None
                    global_summary[case]['cooling_demand_reduction(%)'] = None
    else:
        logger.warning("警告: ref工况总能耗为0，无法计算差异和节能率。")
        for case in cases:
            if case != 'ref':
                if case in global_summary:
                    global_summary[case]['cooling_demand_diff(GWh)'] = None
                    global_summary[case]['cooling_demand_reduction(%)'] = None
    
    # 整理全球汇总结果为DataFrame并保存
    summary_df = pd.DataFrame.from_dict(global_summary, orient='index')
    summary_df.index.name = 'case'
    summary_df = summary_df[['cooling_demand_sum(GWh)', 'cooling_demand_diff(GWh)', 'cooling_demand_reduction(%)']]
    
    # 保存全球汇总结果到CSV
    global_summary_csv_path = os.path.join(output_base_dir, 'global_cooling_energy_summary.csv')
    summary_df.to_csv(global_summary_csv_path, float_format='%.2f')
    logger.info(f"全球能耗汇总结果已保存至: {global_summary_csv_path}")
    
    # 绘制热图
    if ref_max_energy > 0 and 'ref' in all_case_results:
        # 计算case21（case20）的最大差值作为所有差值热图的最大值
        case21_diff_max = 0
        if 'case20' in all_case_results:
            case21_diff_values = pd.to_numeric(all_case_results['case20']['cooling_demand_diff(GWh)'], errors='coerce')
            case21_diff_max = abs(case21_diff_values).max() if len(case21_diff_values.dropna()) > 0 else 0
        
        for case_name, df_energy in all_case_results.items():
            logger.info(f"\n绘制 {case_name} 的热图...")
            case_output_dir = os.path.join(output_base_dir, case_name)
            
            # 1. 绘制制冷能耗热图（使用ref工况最高国家能耗作为最大值）
            cooling_energy_path = os.path.join(case_output_dir, f"{case_name}_cooling_energy_heatmap.pdf")
            plot_heatmap(
                df_energy,
                COUNTRIES_SHP_PATH,
                'cooling_energy(GWh)',
                f'Global Cooling Energy Consumption ({case_name})',
                cooling_energy_path,
                vmin=0,
                vmax=ref_max_energy,
                cmap='Oranges'
            )
            
            # 2. 绘制能耗差异热图（ref工况跳过，使用case21最大值）
            if case_name != 'ref':
                diff_path = os.path.join(case_output_dir, f"{case_name}_cooling_demand_diff_heatmap.pdf")
                plot_heatmap(
                    df_energy,
                    COUNTRIES_SHP_PATH,
                    'cooling_demand_diff(GWh)',
                    f'Cooling Energy Demand Difference from Ref ({case_name})',
                    diff_path,
                    vmin=0,
                    vmax=case21_diff_max if case21_diff_max > 0 else None,
                    cmap='Oranges'  # 使用与能耗热图相同的颜色方案
                )
                
                # 3. 绘制节能率热图（固定范围为0~100%，使用相同颜色方案）
                reduction_path = os.path.join(case_output_dir, f"{case_name}_cooling_demand_reduction_heatmap.pdf")
                plot_heatmap(
                    df_energy,
                    COUNTRIES_SHP_PATH,
                    'cooling_demand_reduction(%)',
                    f'Cooling Energy Demand Reduction Percentage ({case_name})',
                    reduction_path,
                    vmin=0,
                    vmax=100,
                    cmap='Oranges'  # 使用与能耗热图相同的颜色方案
                )
        
        logger.info(f"\n所有热图已保存到: {output_base_dir}")
    else:
        logger.warning("无法绘制热图：ref工况的最大能耗值为0或未找到ref工况数据。")
    
    logger.info(f"✓ {model_name} - {ssp_path} - {year} 年处理完成\n")


def process_all_years_parallel(model_name, ssp_path, years, num_processes=40):
    """并行处理多个年份的数据（跨年份并行）"""
    logger.info(f"\n{'='*80}")
    logger.info(f"并行处理 {model_name} - {ssp_path} - 年份: {years}")
    logger.info(f"{'='*80}")
    
    # 存储所有年份的ref数据
    ref_data_by_year = {}
    
    # ========== 第一阶段：并行处理所有年份的ref工况 ==========
    logger.info(f"\n第一阶段：并行处理所有年份的ref工况（共{len(years)}个年份）...")
    ref_tasks = [(model_name, ssp_path, year, 'ref', None) for year in years]
    
    with ProcessPoolExecutor(max_workers=min(num_processes, len(ref_tasks))) as executor:
        ref_futures = {executor.submit(process_single_case, *task): task for task in ref_tasks}
        for future in as_completed(ref_futures):
            result = future.result()
            if result['national_energy'] is not None:
                year = result['year']
                ref_data_by_year[year] = result['national_energy']
                logger.info(f"  ✓ {year}年 ref工况处理完成，全球总能耗: {result['total_sum']:.2f} GWh")
            else:
                year = result['year']
                logger.error(f"  ✗ {year}年 ref工况处理失败")
    
    if len(ref_data_by_year) == 0:
        logger.error("所有年份的ref工况处理失败，无法继续")
        return
    
    # ========== 第二阶段：并行处理所有年份的case1-20工况 ==========
    logger.info(f"\n第二阶段：并行处理所有年份的case1-20工况（共{len(years)}个年份 × 20个case = {len(years) * 20}个任务）...")
    case_tasks = []
    for year in years:
        if year in ref_data_by_year:
            ref_data = ref_data_by_year[year]
            logger.debug(f"  为{year}年准备case任务，使用该年份的ref数据（{len(ref_data)}个国家）")
            for case in [f'case{i}' for i in range(1, 21)]:
                # 确保每个case都使用对应年份的ref数据
                case_tasks.append((model_name, ssp_path, year, case, ref_data))
        else:
            logger.warning(f"  {year}年的ref数据不存在，跳过该年份的case处理")
    
    # 使用所有可用进程并行处理
    with ProcessPoolExecutor(max_workers=min(num_processes, len(case_tasks))) as executor:
        case_futures = {executor.submit(process_single_case, *task): task for task in case_tasks}
        completed = 0
        for future in as_completed(case_futures):
            result = future.result()
            completed += 1
            if result['national_energy'] is not None:
                logger.info(f"  ✓ [{completed}/{len(case_tasks)}] {result['year']}年 {result['case']}工况处理完成")
            else:
                logger.warning(f"  ✗ [{completed}/{len(case_tasks)}] {result['year']}年 {result['case']}工况处理失败")
    
    # ========== 第三阶段：生成汇总和并行绘制热图（参考计算阶段的并行逻辑）==========
    logger.info(f"\n第三阶段：生成汇总和并行绘制热图...")
    
    # 3.1 先为每个年份生成汇总（串行，因为需要读取数据）
    year_summary_data = {}  # {year: {'ref_max_energy': ..., 'case21_diff_max': ..., 'ref_total_sum': ...}}
    
    for year in years:
        if year not in ref_data_by_year:
            continue
        
        output_base_dir = os.path.join(HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", str(year), "country")
        all_case_results = {}
        global_summary = {}
        ref_max_energy = 0.0
        ref_total_sum = 0.0
        
        # 加载所有case的CSV文件
        for case in cases:
            csv_path = os.path.join(output_base_dir, case, f"{case}_national_cooling_energy.csv")
            if os.path.exists(csv_path):
                national_energy = pd.read_csv(csv_path)
                all_case_results[case] = national_energy
                current_total_sum = national_energy['cooling_energy(GWh)'].sum()
                global_summary[case] = {'cooling_demand_sum(GWh)': current_total_sum}
                
                if case == 'ref':
                    ref_max_energy = float(national_energy['cooling_energy(GWh)'].max())
                    ref_total_sum = current_total_sum
        
        # 计算每个工况相对于ref的能耗差异和节能率（全球汇总）
        if ref_total_sum > 0:
            for case in cases:
                if case == 'ref':
                    global_summary['ref']['cooling_demand_diff(GWh)'] = ''
                    global_summary['ref']['cooling_demand_reduction(%)'] = ''
                else:
                    current_sum = global_summary.get(case, {}).get('cooling_demand_sum(GWh)')
                    if current_sum is not None:
                        diff = ref_total_sum - current_sum
                        reduction_percent = (diff / ref_total_sum) * 100
                        global_summary[case]['cooling_demand_diff(GWh)'] = diff
                        global_summary[case]['cooling_demand_reduction(%)'] = reduction_percent
        
        # 保存全球汇总
        summary_df = pd.DataFrame.from_dict(global_summary, orient='index')
        summary_df.index.name = 'case'
        summary_df = summary_df[['cooling_demand_sum(GWh)', 'cooling_demand_diff(GWh)', 'cooling_demand_reduction(%)']]
        global_summary_csv_path = os.path.join(output_base_dir, 'global_cooling_energy_summary.csv')
        summary_df.to_csv(global_summary_csv_path, float_format='%.2f')
        logger.info(f"  ✓ {year}年全球能耗汇总结果已保存")
        
        # 计算case20的最大差值作为所有差值热图的最大值
        case21_diff_max = 0
        if 'case20' in all_case_results:
            case21_diff_values = pd.to_numeric(all_case_results['case20']['cooling_demand_diff(GWh)'], errors='coerce')
            case21_diff_max = abs(case21_diff_values).max() if len(case21_diff_values.dropna()) > 0 else 0
        
        year_summary_data[year] = {
            'ref_max_energy': ref_max_energy,
            'case21_diff_max': case21_diff_max,
            'ref_total_sum': ref_total_sum
        }
    
    # 3.2 第一阶段：并行绘制所有年份的ref工况热图
    logger.info(f"\n第一阶段：并行绘制所有年份的ref工况热图（共{len(years)}个年份）...")
    ref_plot_tasks = []
    for year in years:
        if year in year_summary_data:
            summary_data = year_summary_data[year]
            ref_plot_tasks.append((
                model_name, ssp_path, year, 'ref',
                summary_data['ref_max_energy'],
                summary_data['case21_diff_max']
            ))
    
    with ProcessPoolExecutor(max_workers=min(num_processes, len(ref_plot_tasks))) as executor:
        ref_plot_futures = {executor.submit(plot_single_case_heatmaps, *task): task for task in ref_plot_tasks}
        for future in as_completed(ref_plot_futures):
            result = future.result()
            if result['success']:
                logger.info(f"  ✓ {result['year']}年 ref工况热图绘制完成")
            else:
                logger.warning(f"  ✗ {result['year']}年 ref工况热图绘制失败")
    
    # 3.3 第二阶段：并行绘制所有年份的case1-20工况热图
    logger.info(f"\n第二阶段：并行绘制所有年份的case1-20工况热图（共{len(years)}个年份 × 20个case = {len(years) * 20}个任务）...")
    case_plot_tasks = []
    for year in years:
        if year in year_summary_data:
            summary_data = year_summary_data[year]
            for case in [f'case{i}' for i in range(1, 21)]:
                case_plot_tasks.append((
                    model_name, ssp_path, year, case,
                    summary_data['ref_max_energy'],
                    summary_data['case21_diff_max']
                ))
    
    # 使用所有可用进程并行处理
    with ProcessPoolExecutor(max_workers=min(num_processes, len(case_plot_tasks))) as executor:
        case_plot_futures = {executor.submit(plot_single_case_heatmaps, *task): task for task in case_plot_tasks}
        completed = 0
        for future in as_completed(case_plot_futures):
            result = future.result()
            completed += 1
            if result['success']:
                logger.info(f"  ✓ [{completed}/{len(case_plot_tasks)}] {result['year']}年 {result['case']}工况热图绘制完成")
            else:
                logger.warning(f"  ✗ [{completed}/{len(case_plot_tasks)}] {result['year']}年 {result['case']}工况热图绘制失败")


def plot_single_case_heatmaps(model_name, ssp_path, year, case_name, ref_max_energy, case21_diff_max):
    """绘制单个case的所有热图（用于多进程并行）"""
    try:
        output_base_dir = os.path.join(HEAT_WAVE_BASE_PATH, model_name, ssp_path, "energy", str(year), "country")
        csv_path = os.path.join(output_base_dir, case_name, f"{case_name}_national_cooling_energy.csv")
        
        if not os.path.exists(csv_path):
            logger.warning(f"  {year}年 {case_name}工况的CSV文件不存在，跳过绘图")
            return {'case': case_name, 'year': year, 'success': False}
        
        # 加载数据
        df_energy = pd.read_csv(csv_path)
        case_output_dir = os.path.join(output_base_dir, case_name)
        
        # 1. 绘制制冷能耗热图
        cooling_energy_path = os.path.join(case_output_dir, f"{case_name}_cooling_energy_heatmap.pdf")
        plot_heatmap(
            df_energy,
            COUNTRIES_SHP_PATH,
            'cooling_energy(GWh)',
            f'Global Cooling Energy Consumption ({case_name})',
            cooling_energy_path,
            vmin=0,
            vmax=ref_max_energy,
            cmap='Oranges'
        )
        
        # 2. 绘制能耗差异热图（ref工况跳过）
        if case_name != 'ref':
            diff_path = os.path.join(case_output_dir, f"{case_name}_cooling_demand_diff_heatmap.pdf")
            plot_heatmap(
                df_energy,
                COUNTRIES_SHP_PATH,
                'cooling_demand_diff(GWh)',
                f'Cooling Energy Demand Difference from Ref ({case_name})',
                diff_path,
                vmin=0,
                vmax=case21_diff_max if case21_diff_max > 0 else None,
                cmap='Oranges'
            )
            
            # 3. 绘制节能率热图
            reduction_path = os.path.join(case_output_dir, f"{case_name}_cooling_demand_reduction_heatmap.pdf")
            plot_heatmap(
                df_energy,
                COUNTRIES_SHP_PATH,
                'cooling_demand_reduction(%)',
                f'Cooling Energy Demand Reduction Percentage ({case_name})',
                reduction_path,
                vmin=0,
                vmax=100,
                cmap='Oranges'
            )
        
        return {'case': case_name, 'year': year, 'success': True}
    except Exception as e:
        logger.error(f"  绘制 {year}年 {case_name}工况热图时出错: {e}", exc_info=True)
        return {'case': case_name, 'year': year, 'success': False}


def main():
    """主函数（支持跨年份并行）"""
    try:
        logger.info("=== 开始国家级别制冷能耗计算 ===")
        logger.info(f"支持的模型: {', '.join(MODELS)}")
        logger.info(f"SSP路径: {', '.join(SSP_PATHS)}")
        logger.info(f"目标年份: {TARGET_YEARS}")
        logger.info(f"并行进程数: 40")
        
        # 检查必要的文件
        if not os.path.exists(COUNTRIES_SHP_PATH):
            raise FileNotFoundError(f"国家边界文件不存在: {COUNTRIES_SHP_PATH}")
        if not os.path.exists(COUNTRIES_INFO_CSV):
            raise FileNotFoundError(f"国家信息文件不存在: {COUNTRIES_INFO_CSV}")
        
        # 处理每个模型
        for model_name in MODELS:
            logger.info(f"\n{'='*80}")
            logger.info(f"处理模型: {model_name}")
            logger.info(f"{'='*80}")
            
            # 处理每个SSP路径
            for ssp_path in SSP_PATHS:
                logger.info(f"\n>>> 当前发展路径: {ssp_path}")
                
                # 跨年份并行处理（同时处理所有年份）
                process_all_years_parallel(model_name, ssp_path, TARGET_YEARS, num_processes=40)
        
        logger.info("\n=== 所有处理完成 ===")
    
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()