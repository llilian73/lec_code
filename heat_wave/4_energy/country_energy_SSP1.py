import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import numpy as np
from shapely.geometry import Point

# --- 配置路径 ---
east_dir = r"C:\Users\thuarchdog\Desktop\result\EAST_output_all_heat_wave_SSP1"
west_dir = r"C:\Users\thuarchdog\Desktop\result\WEST_output_all_heat_wave_SSP1"
countries_shp_path = r"Y:\CMIP6\Energy consumption in heat wave code\shapefile\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp"

cases = [
    'case1', 'case2', 'case3', 'case4', 'case5',
    'case6', 'case7', 'case8', 'case9', 'ref'
]


def load_and_merge_data(east_dir, west_dir, case):
    """加载并合并EAST和WEST文件夹的数据"""
    print(f"正在处理 {case} 的数据...")

    # 读取EAST数据
    east_csv_path = os.path.join(east_dir, case, f'{case}.csv')
    east_data = None
    if os.path.exists(east_csv_path):
        east_data = pd.read_csv(east_csv_path,
                                usecols=['lat', 'lon', 'total_demand', 'heating_demand', 'cooling_demand'])
        # 确保数值列为float类型，无效值设为NaN
        east_data['total_demand'] = pd.to_numeric(east_data['total_demand'], errors='coerce')
        east_data['heating_demand'] = pd.to_numeric(east_data['heating_demand'], errors='coerce')
        east_data['cooling_demand'] = pd.to_numeric(east_data['cooling_demand'], errors='coerce')
        # 移除包含NaN的行
        east_data = east_data.dropna(subset=['total_demand'])
        print(f"EAST数据: {len(east_data)} 个点")

    # 读取WEST数据
    west_csv_path = os.path.join(west_dir, case, f'{case}.csv')
    west_data = None
    if os.path.exists(west_csv_path):
        west_data = pd.read_csv(west_csv_path,
                                usecols=['lat', 'lon', 'total_demand', 'heating_demand', 'cooling_demand'])
        # 确保数值列为float类型，无效值设为NaN
        west_data['total_demand'] = pd.to_numeric(west_data['total_demand'], errors='coerce')
        west_data['heating_demand'] = pd.to_numeric(west_data['heating_demand'], errors='coerce')
        west_data['cooling_demand'] = pd.to_numeric(west_data['cooling_demand'], errors='coerce')
        # 移除包含NaN的行
        west_data = west_data.dropna(subset=['total_demand'])
        # 将经度从180~360转换为-180~0
        west_data['lon'] = west_data['lon'].apply(lambda x: x - 360 if x >= 180 else x)
        print(f"WEST数据: {len(west_data)} 个点")

    # 合并数据
    if east_data is not None and west_data is not None:
        merged_data = pd.concat([east_data, west_data], ignore_index=True)
    elif east_data is not None:
        merged_data = east_data
    elif west_data is not None:
        merged_data = west_data
    else:
        print(f"警告: {case} 在EAST和WEST文件夹中都没有找到数据")
        return None

    print(f"合并后数据: {len(merged_data)} 个点")
    return merged_data


def calculate_national_energy(input_data, shp_path, output_csv_path):
    """计算每个国家的总能耗并保存为CSV"""
    print(f"正在处理国家能耗数据...")

    # --- 读取国家边界Shapefile ---
    countries = gpd.read_file(shp_path)
    print(f"国家数量: {len(countries)}")

    # 移除南极洲
    countries = countries[countries['CONTINENT'] != 'Antarctica']
    print(f"移除南极洲后国家数量: {len(countries)}")

    if countries.empty:
        raise ValueError("没有找到国家数据，请检查shapefile是否为空或路径是否正确。")

    # 确保几何图形有效
    countries['geometry'] = countries.geometry.buffer(0)
    countries['geometry'] = countries.geometry.apply(lambda geom: geom.make_valid() if not geom.is_valid else geom)

    print(f"Countries CRS: {countries.crs}")
    if countries.crs is None:
        print("Warning: Countries GeoDataFrame has no CRS defined. Attempting to set to EPSG:4326.")
        countries = countries.set_crs("EPSG:4326", allow_override=True)
        print(f"Set CRS to EPSG:4326. New CRS: {countries.crs}")

    # --- 将点数据转换为GeoDataFrame ---
    try:
        geometry = [Point(xy) for xy in zip(input_data['lon'], input_data['lat'])]
        gdf_points = gpd.GeoDataFrame(input_data, geometry=geometry, crs="EPSG:4326")
        print(f"已读取 {len(gdf_points)} 个能耗点数据。")
    except Exception as e:
        print(f"转换点数据时发生错误: {e}")
        raise

    # --- 空间连接并聚合能耗数据 ---
    print("正在进行空间连接并聚合能耗数据...")
    try:
        # 确保坐标系匹配
        if gdf_points.crs != countries.crs:
            gdf_points = gdf_points.to_crs(countries.crs)
            print(f"警告: 点数据CRS不匹配，已转换为 {countries.crs}")

        joined_gdf = gpd.sjoin(gdf_points, countries, how="inner", predicate="within")

        # 按国家聚合总能耗
        national_energy = joined_gdf.groupby('ISO_A2')['total_demand'].sum().reset_index()
        national_energy.rename(columns={'ISO_A2': 'country', 'total_demand': 'total_energy'}, inplace=True)

        # 合并CN和CN-TW为CN
        cn_tw_energy = 0
        if 'CN-TW' in national_energy['country'].values:
            cn_tw_energy = national_energy[national_energy['country'] == 'CN-TW']['total_energy'].iloc[0]
            national_energy = national_energy[national_energy['country'] != 'CN-TW']

        if 'CN' in national_energy['country'].values:
            cn_idx = national_energy[national_energy['country'] == 'CN'].index[0]
            national_energy.loc[cn_idx, 'total_energy'] += cn_tw_energy
        else:
            # 如果只有CN-TW没有CN，则创建CN记录
            national_energy = pd.concat(
                [national_energy, pd.DataFrame({'country': ['CN'], 'total_energy': [cn_tw_energy]})], ignore_index=True)

        print(f"已聚合 {len(national_energy)} 个国家的能耗数据。")
        print(
            f"CN和CN-TW合并后的总能耗: {national_energy[national_energy['country'] == 'CN']['total_energy'].iloc[0] if 'CN' in national_energy['country'].values else 0}")

        # 保存结果到CSV
        national_energy.to_csv(output_csv_path, index=False)
        print(f"结果已保存至: {output_csv_path}")

        return national_energy

    except Exception as e:
        print(f"空间连接或数据聚合时发生错误: {e}")
        raise


def plot_global_heatmap(case_results, shp_path, ref_max_energy, output_pdf_dir):
    """绘制每个工况的全球总能耗热图"""
    print("正在绘制全球热图...")
    countries_gdf = gpd.read_file(shp_path)
    # 移除南极洲
    countries_gdf = countries_gdf[countries_gdf['CONTINENT'] != 'Antarctica']

    for case_name, df_energy in case_results.items():
        print(f"绘制 {case_name} 热图...")

        # 合并能耗数据到GeoDataFrame
        merged_gdf = countries_gdf.merge(
            df_energy,
            left_on='ISO_A2',
            right_on='country',
            how='left'
        )

        # 特殊处理CN和CN-TW，确保它们使用相同的颜色
        cn_energy = df_energy[df_energy['country'] == 'CN']['total_energy'].iloc[0] if 'CN' in df_energy[
            'country'].values else 0

        # 为CN和CN-TW设置相同的能耗值
        merged_gdf.loc[merged_gdf['ISO_A2'] == 'CN', 'total_energy'] = cn_energy
        merged_gdf.loc[merged_gdf['ISO_A2'] == 'CN-TW', 'total_energy'] = cn_energy

        # 填充NaN值为0
        merged_gdf['total_energy'] = merged_gdf['total_energy'].fillna(0)

        # 创建地图
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))

        # 绘制热图
        merged_gdf.plot(
            column='total_energy',
            cmap='Oranges',
            linewidth=0.5,
            ax=ax,
            edgecolor='0.3',
            legend=True,
            legend_kwds={'label': "Total Energy Consumption (GW)", 'orientation': "horizontal", 'shrink': 0.8,
                         'pad': 0.02},
            vmin=0,
            vmax=148843.97231048864  # 使用SSP2的ref工况的最大值作为颜色条上限
        )

        ax.set_title(f'Global Total Energy Consumption ({case_name})', fontsize=16)
        ax.set_axis_off()

        # 确保输出目录存在
        case_output_dir = os.path.join(output_pdf_dir, case_name)
        os.makedirs(case_output_dir, exist_ok=True)

        # 保存为PDF格式
        output_pdf_path = os.path.join(case_output_dir, f'{case_name}_global_total_energy_heatmap.pdf')
        plt.savefig(output_pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"热图已保存至: {output_pdf_path}")


def main():
    all_case_results = {}
    ref_max_energy = 0.0

    # 创建输出目录
    output_base_dir = r"C:\Users\thuarchdog\Desktop\result\global_energy_output_total_SSP1"
    os.makedirs(output_base_dir, exist_ok=True)

    # 用于存储全球总能耗汇总
    global_summary = {}

    for case in cases:
        print(f"\n=== 处理 {case} ===")

        # 加载并合并数据
        merged_data = load_and_merge_data(east_dir, west_dir, case)

        if merged_data is None:
            print(f"跳过 {case}，因为没有找到数据")
            global_summary[case] = {'total_demand_sum(GWh)': None, 'total_demand_diff(GWh)': None,
                                    'total_demand_reduction(%)': None}
            continue

        # 定义每个case的输出子目录
        case_output_dir = os.path.join(output_base_dir, case)
        os.makedirs(case_output_dir, exist_ok=True)
        # 定义输出CSV路径
        csv_output_path = os.path.join(case_output_dir, f'{case}_national_energy.csv')

        # 计算并保存国家能耗
        national_energy = calculate_national_energy(merged_data, countries_shp_path, csv_output_path)

        # 保存结果用于后续绘图
        all_case_results[case] = national_energy

        # 计算全球总能耗
        current_total_sum = national_energy['total_energy'].sum()
        global_summary[case] = {'total_demand_sum(GWh)': current_total_sum}
        print(f"工况 {case} 的全球总能耗: {current_total_sum:.2f} GWh")

        # 如果是ref工况，记录最大能耗值
        if case == 'ref':
            ref_max_energy = float(national_energy['total_energy'].max())
            ref_total_sum = current_total_sum
            print(f"Ref工况最大能耗值: {ref_max_energy}")

    # 计算每个工况相对于ref的能耗差异和节能率
    if ref_total_sum > 0:
        for case in cases:
            if case == 'ref':
                global_summary['ref']['total_demand_diff(GWh)'] = ''  # ref工况不计算差异和节能率
                global_summary['ref']['total_demand_reduction(%)'] = ''
            else:
                current_sum = global_summary[case]['total_demand_sum(GWh)']
                if current_sum is not None:
                    diff = ref_total_sum - current_sum
                    reduction_percent = (diff / ref_total_sum) * 100
                    global_summary[case]['total_demand_diff(GWh)'] = diff
                    global_summary[case]['total_demand_reduction(%)'] = reduction_percent
                else:
                    global_summary[case]['total_demand_diff(GWh)'] = None
                    global_summary[case]['total_demand_reduction(%)'] = None
    else:
        print("警告: ref工况总能耗为0，无法计算差异和节能率。")
        for case in cases:
            if case != 'ref':
                global_summary[case]['total_demand_diff(GWh)'] = None
                global_summary[case]['total_demand_reduction(%)'] = None

    # 整理全球汇总结果为DataFrame并保存
    summary_df = pd.DataFrame.from_dict(global_summary, orient='index')
    summary_df.index.name = 'case'
    summary_df = summary_df[['total_demand_sum(GWh)', 'total_demand_diff(GWh)', 'total_demand_reduction(%)']]

    # 保存全球汇总结果到CSV
    global_summary_csv_path = os.path.join(output_base_dir, 'global_energy_summary.csv')
    summary_df.to_csv(global_summary_csv_path, float_format='%.2f')
    print(f"全球能耗汇总结果已保存至: {global_summary_csv_path}")

    # 绘制热图
    if ref_max_energy > 0:
        plot_global_heatmap(all_case_results, countries_shp_path, ref_max_energy, output_base_dir)
        print(f"\n所有热图已保存到: {output_base_dir}")
    else:
        print("无法绘制热图：ref工况的最大能耗值为0或未找到ref工况数据。")

    print("全球能耗计算和汇总完成！")


if __name__ == "__main__":
    main()
