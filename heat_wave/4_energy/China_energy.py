import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
import os
import numpy as np
import tempfile  # 导入tempfile模块
from shapely.geometry import Point  # Add Point for creating geometries from CSV

# --- 配置路径 ---
base_tif_dir_ssp2 = r"C:\Users\thuarchdog\Desktop\result\EAST_output_all_heat_wave"
base_tif_dir_ssp1 = r"C:\Users\thuarchdog\Desktop\result\EAST_output_all_heat_wave_SSP1"
province_shp_path = r"C:\Users\thuarchdog\Desktop\result\shp\2. Province\province.shp"
plot_geojson_path = r"C:\Users\thuarchdog\Desktop\result\shp\中国_省.geojson"  # 新增用于绘图的GeoJSON文件路径
output_base_dir = r"C:\Users\thuarchdog\Desktop\result\China"

cases = [
    'case1', 'case2', 'case3', 'case4', 'case5',
    'case6', 'case7', 'case8', 'case9', 'ref'
]


def calculate_provincial_energy(input_csv_path, shp_path, output_csv_path):
    """计算每个省份的总能耗并保存为CSV (从点数据聚合)"""
    print(f"正在处理点数据文件: {input_csv_path}")

    # --- 读取省界Shapefile ---
    provinces = gpd.read_file(shp_path)
    print(f"原始省份数量: {len(provinces)}")

    if provinces.empty:
        raise ValueError("没有找到省份数据，请检查shapefile是否为空或路径是否正确。")

    # Ensure geometries are valid and get CRS
    provinces['geometry'] = provinces.geometry.buffer(0)
    provinces['geometry'] = provinces.geometry.apply(lambda geom: geom.make_valid() if not geom.is_valid else geom)

    print(f"Provinces CRS before spatial join: {provinces.crs}")
    if provinces.crs is None:
        print("Warning: Provinces GeoDataFrame has no CRS defined. Attempting to set to EPSG:4326.")
        provinces = provinces.set_crs("EPSG:4326", allow_override=True)
        print(f"Set CRS to EPSG:4326. New CRS: {provinces.crs}")

    # --- 读取点数据并转换为GeoDataFrame ---
    try:
        point_df = pd.read_csv(input_csv_path,
                               usecols=['lat', 'lon', 'total_demand', 'heating_demand', 'cooling_demand'])
        # 确保数值列为float类型
        point_df['total_demand'] = pd.to_numeric(point_df['total_demand'], errors='coerce')
        point_df['heating_demand'] = pd.to_numeric(point_df['heating_demand'], errors='coerce')
        point_df['cooling_demand'] = pd.to_numeric(point_df['cooling_demand'], errors='coerce')

        geometry = [Point(xy) for xy in zip(point_df['lon'], point_df['lat'])]
        gdf_points = gpd.GeoDataFrame(point_df, geometry=geometry, crs="EPSG:4326")
        print(f"已读取 {len(gdf_points)} 个人口点数据。")
    except Exception as e:
        print(f"读取点数据时发生错误: {e}")
        raise

    # --- 空间连接并聚合能耗数据 ---
    print("正在进行空间连接并聚合能耗数据...")
    try:
        if gdf_points.crs != provinces.crs:
            gdf_points = gdf_points.to_crs(provinces.crs)
            print(f"警告: 点数据CRS不匹配，已转换为 {provinces.crs}")

        joined_gdf = gpd.sjoin(gdf_points, provinces, how="inner", predicate="within")

        # Aggregate cooling_demand by province
        provincial_energy = joined_gdf.groupby('pr_name')['cooling_demand'].sum().reset_index()
        provincial_energy.rename(columns={'pr_name': 'province_name'}, inplace=True)
        print(f"已聚合 {len(provincial_energy)} 个省份的能耗数据。")

        merged_gdf = provinces.merge(
            provincial_energy,
            left_on='pr_name',
            right_on='province_name',
            how='left'
        )
        merged_gdf.rename(columns={'cooling_demand': 'cooling_energy'}, inplace=True)

        # Fill NaN values with 0 and ensure numeric type
        merged_gdf['cooling_energy'] = pd.to_numeric(merged_gdf['cooling_energy'].fillna(0), errors='coerce')

        # Save result to CSV
        result_df = merged_gdf[['pr_name', 'cooling_energy']].copy()
        result_df.rename(columns={'pr_name': 'province'}, inplace=True)
        result_df.to_csv(output_csv_path, index=False, float_format='%.2f')  # 使用float_format确保数值格式
        print(f"结果已保存至: {output_csv_path}")

    except Exception as e:
        print(f"空间连接或数据聚合时发生错误: {e}")
        raise


def process_ssp_data(base_tif_dir, output_dir, ssp_name):
    """处理单个SSP路径的数据"""
    print(f"\n=== 开始处理 {ssp_name} 数据 ===")

    all_case_results = {}
    ref_max_energy = 0.0

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    for case in cases:
        csv_input_path = os.path.join(base_tif_dir, case, f'{case}.csv')
        csv_output_path = os.path.join(output_dir, case, f'{case}_provincial_energy.csv')

        if not os.path.exists(csv_input_path):
            print(f"警告: 未找到文件 {csv_input_path}, 跳过此工况。")
            continue

        os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

        calculate_provincial_energy(csv_input_path, province_shp_path, csv_output_path)

        # 读取计算结果并确保数值类型
        df_current_case = pd.read_csv(csv_output_path)
        df_current_case['cooling_energy'] = pd.to_numeric(df_current_case['cooling_energy'], errors='coerce')
        all_case_results[case] = df_current_case

        if case == 'ref':
            ref_max_energy = float(df_current_case['cooling_energy'].max())

    # 计算并汇总全国总能耗
    summary_df = calculate_and_summarize_energy(all_case_results, output_dir, ssp_name)

    # 绘制热图
    if ref_max_energy > 0:
        plot_heatmap(all_case_results, province_shp_path, ref_max_energy, output_dir, ssp_name)
        print(f"\n所有热图已保存到: {output_dir}")
    else:
        print("无法绘制热图：ref工况的最大能耗值为0或未找到ref工况数据。")

    return summary_df


def calculate_and_summarize_energy(all_case_results, output_dir, ssp_name):
    """计算并汇总全国总能耗"""
    print(f"开始计算并汇总 {ssp_name} 全国总能耗...")

    all_cases_summary = {}
    ref_total_sum = 0

    # 计算每个工况的全国总能耗
    for case in cases:
        if case in all_case_results:
            current_total_sum = all_case_results[case]['cooling_energy'].sum()
            all_cases_summary[case] = {'cooling_demand_sum(GWh)': current_total_sum}

            if case == 'ref':
                ref_total_sum = current_total_sum
            print(f"工况 {case} 的全国总能耗: {current_total_sum:.2f} GWh")
        else:
            print(f"警告: 未找到工况 {case} 的数据，跳过此工况。")
            all_cases_summary[case] = {'cooling_demand_sum(GWh)': None, 'cooling_demand_diff(GWh)': None,
                                       'cooling_demand_reduction(%)': None}

    # 计算每个工况相对于ref的能耗差异和节能率
    if ref_total_sum == 0:  # 避免除以零
        print("警告: ref工况总能耗为0，无法计算差异和节能率。")
        for case in cases:
            if case != 'ref':
                all_cases_summary[case]['cooling_demand_diff(GWh)'] = None
                all_cases_summary[case]['cooling_demand_reduction(%)'] = None
    else:
        for case in cases:
            if case == 'ref':
                all_cases_summary['ref']['cooling_demand_diff(GWh)'] = ''  # ref工况不计算差异和节能率
                all_cases_summary['ref']['cooling_demand_reduction(%)'] = ''
            else:
                current_sum = all_cases_summary[case]['cooling_demand_sum(GWh)']
                if current_sum is not None:
                    diff = ref_total_sum - current_sum
                    reduction_percent = (diff / ref_total_sum) * 100
                    all_cases_summary[case]['cooling_demand_diff(GWh)'] = diff
                    all_cases_summary[case]['cooling_demand_reduction(%)'] = reduction_percent
                else:
                    all_cases_summary[case]['cooling_demand_diff(GWh)'] = None
                    all_cases_summary[case]['cooling_demand_reduction(%)'] = None

    # 整理结果为DataFrame
    summary_df = pd.DataFrame.from_dict(all_cases_summary, orient='index')
    summary_df.index.name = 'case'
    # 重新排列列的顺序以匹配期望格式
    summary_df = summary_df[['cooling_demand_sum(GWh)', 'cooling_demand_diff(GWh)', 'cooling_demand_reduction(%)']]

    # 保存结果到CSV
    output_csv_path = os.path.join(output_dir, f'CN_energy_{ssp_name}.csv')
    summary_df.to_csv(output_csv_path, float_format='%.2f')  # 格式化浮点数为两位小数
    print(f"汇总结果已保存至: {output_csv_path}")

    return summary_df


def plot_heatmap(case_results, shp_path, ref_max_energy, output_pdf_dir, ssp_name):
    """绘制每个工况的省份总能耗热图"""
    print(f"正在绘制 {ssp_name} 热图...")
    # 从新的GeoJSON文件读取用于绘图的省份数据
    provinces_gdf = gpd.read_file(plot_geojson_path)
    # 此处不再需要筛选，因为 中国_省.geojson 应该只包含中国省份

    for case_name, df_energy in case_results.items():
        print(f"绘制 {ssp_name} {case_name} 热图...")

        # 合并能耗数据到GeoDataFrame
        # 使用 'name' 列作为连接键
        merged_gdf = provinces_gdf.merge(
            df_energy,
            left_on='name',
            right_on='province',
            how='left'
        )

        # 填充NaN值为0，以便正确显示未匹配的省份（如果存在）
        merged_gdf['cooling_energy'] = merged_gdf['cooling_energy'].fillna(0)

        # 创建地图
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # 绘制热图
        merged_gdf.plot(
            column='cooling_energy',
            cmap='Oranges',
            linewidth=0.8,
            ax=ax,
            edgecolor='0.8',
            legend=True,
            legend_kwds={'label': "Cooling Energy Consumption (GW)", 'orientation': "horizontal", 'shrink': 0.7,
                         'pad': 0.0},
            vmin=0,
            vmax=15000  # 使用ref工况的最大值作为颜色条上限
        )

        ax.set_title(f'Cooling Energy Consumption in China ({ssp_name} {case_name})', fontsize=15)
        ax.set_axis_off()  # 移除坐标轴

        # 确保输出目录存在
        case_output_dir = os.path.join(output_pdf_dir, case_name)
        os.makedirs(case_output_dir, exist_ok=True)

        # 保存为PDF格式
        output_pdf_path = os.path.join(case_output_dir, f'{ssp_name}_{case_name}_cooling_energy_heatmap.pdf')
        plt.savefig(output_pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)  # 关闭当前图形，避免内存问题
        print(f"热图已保存至: {output_pdf_path}")


def main():
    # 处理SSP2数据
    output_dir_ssp2 = os.path.join(output_base_dir, 'SSP2')
    summary_ssp2 = process_ssp_data(base_tif_dir_ssp2, output_dir_ssp2, 'SSP2')

    # 处理SSP1数据
    output_dir_ssp1 = os.path.join(output_base_dir, 'SSP1')
    summary_ssp1 = process_ssp_data(base_tif_dir_ssp1, output_dir_ssp1, 'SSP1')

    print("\n所有SSP数据处理完成！")
    print(f"SSP2结果保存在: {output_dir_ssp2}")
    print(f"SSP1结果保存在: {output_dir_ssp1}")


if __name__ == "__main__":
    main()
