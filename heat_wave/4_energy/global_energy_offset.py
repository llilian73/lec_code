import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 配置路径 ---
ssp1_ref_path = r"D:\workstation\extreme weather\global\global_energy_output_SSP1\ref\ref_national_cooling_energy.csv"
ssp2_ref_path = r"D:\workstation\extreme weather\global\global_energy_output\ref\ref_national_cooling_energy.csv"
ssp2_50_path = r"D:\workstation\extreme weather\global\global_energy_output\case5\case5_national_cooling_energy.csv"
ssp2_3_125_path = r"D:\workstation\extreme weather\global\global_energy_output\case9\case9_national_cooling_energy.csv"
countries_shp_path = r"Y:\CMIP6\Energy consumption in heat wave code\shapefile\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp"


def load_energy_data(file_path, scenario_name):
    """加载能耗数据"""
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        print(f"{scenario_name} 数据加载成功: {len(data)} 个国家")
        return data
    else:
        print(f"警告: {scenario_name} 文件不存在: {file_path}")
        return None


def calculate_offset_percentage():
    """计算偏移百分比"""
    print("开始计算偏移百分比...")

    # 加载所有数据
    ssp1_ref = load_energy_data(ssp1_ref_path, "SSP1-ref")
    ssp2_ref = load_energy_data(ssp2_ref_path, "SSP2-ref")
    ssp2_50 = load_energy_data(ssp2_50_path, "SSP2-50")
    ssp2_3_125 = load_energy_data(ssp2_3_125_path, "SSP2-3.125")

    if ssp1_ref is None or ssp2_ref is None or ssp2_50 is None or ssp2_3_125 is None:
        print("错误: 无法加载所有必需的数据文件")
        return None

    # 合并所有数据到同一个DataFrame
    merged_data = ssp2_ref.merge(ssp1_ref, on='country', how='outer', suffixes=('_SSP2', '_SSP1'))
    merged_data = merged_data.merge(ssp2_50, on='country', how='outer', suffixes=('', '_SSP2_50'))
    merged_data = merged_data.merge(ssp2_3_125, on='country', how='outer', suffixes=('', '_SSP2_3_125'))

    # 重命名列以便计算
    merged_data = merged_data.rename(columns={
        'cooling_energy': 'cooling_energy_SSP2_50',
        'cooling_energy_SSP2_3_125': 'cooling_energy_SSP2_3_125'
    })

    # 填充NaN值为0
    energy_columns = ['cooling_energy_SSP2', 'cooling_energy_SSP1', 'cooling_energy_SSP2_50',
                      'cooling_energy_SSP2_3_125']
    for col in energy_columns:
        merged_data[col] = merged_data[col].fillna(0)

    # 计算差值
    merged_data['diff_SSP1_SSP2'] = merged_data['cooling_energy_SSP1'] - merged_data['cooling_energy_SSP2']
    merged_data['diff_SSP2_50_SSP2'] = merged_data['cooling_energy_SSP2_50'] - merged_data['cooling_energy_SSP2']
    merged_data['diff_SSP2_3_125_SSP2'] = merged_data['cooling_energy_SSP2_3_125'] - merged_data['cooling_energy_SSP2']

    # 计算偏移百分比
    # 避免除零错误
    merged_data['offset_percentage_50'] = np.where(
        merged_data['diff_SSP1_SSP2'] != 0,
        (merged_data['diff_SSP2_50_SSP2'] / merged_data['diff_SSP1_SSP2']) * 100,
        0
    )

    merged_data['offset_percentage_3_125'] = np.where(
        merged_data['diff_SSP1_SSP2'] != 0,
        (merged_data['diff_SSP2_3_125_SSP2'] / merged_data['diff_SSP1_SSP2']) * 100,
        0
    )

    # 处理无穷大和NaN值
    merged_data['offset_percentage_50'] = merged_data['offset_percentage_50'].replace([np.inf, -np.inf], 0)
    merged_data['offset_percentage_3_125'] = merged_data['offset_percentage_3_125'].replace([np.inf, -np.inf], 0)

    print(f"偏移百分比计算完成，共处理 {len(merged_data)} 个国家")
    return merged_data


def plot_offset_heatmap(data, output_dir):
    """绘制偏移百分比热力图"""
    print("开始绘制偏移百分比热力图...")

    # 读取国家边界数据
    countries_gdf = gpd.read_file(countries_shp_path)
    countries_gdf = countries_gdf[countries_gdf['CONTINENT'] != 'Antarctica']

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 绘制两个偏移百分比的热力图
    scenarios = [
        ('offset_percentage_50', 'SSP2-50 vs SSP2'),
        ('offset_percentage_3_125', 'SSP2-3.125 vs SSP2')
    ]

    for offset_col, scenario_name in scenarios:
        print(f"绘制 {scenario_name} 热力图...")

        # 合并数据到GeoDataFrame
        merged_gdf = countries_gdf.merge(
            data[['country', offset_col]],
            left_on='ISO_A2',
            right_on='country',
            how='left'
        )

        # 填充NaN值为0
        merged_gdf[offset_col] = merged_gdf[offset_col].fillna(0)

        # 创建地图
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))

        # 计算颜色范围
        vmin = merged_gdf[offset_col].min()
        vmax = merged_gdf[offset_col].max()

        # 使用对称的颜色范围
        abs_max = max(abs(vmin), abs(vmax))
        vmin = -abs_max
        vmax = abs_max

        # 绘制热图
        merged_gdf.plot(
            column=offset_col,
            cmap='RdBu_r',  # 红蓝对比色，红色表示正值，蓝色表示负值
            linewidth=0.5,
            ax=ax,
            edgecolor='0.3',
            legend=True,
            legend_kwds={'label': f"Offset Percentage (%)", 'orientation': "horizontal", 'shrink': 0.8, 'pad': 0.02},
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(f'Global Cooling Energy Offset Percentage: {scenario_name}', fontsize=16)
        ax.set_axis_off()

        # 保存为PDF格式
        output_pdf_path = os.path.join(output_dir,
                                       f'global_cooling_energy_offset_{scenario_name.replace(" ", "_").replace("-", "_")}.pdf')
        plt.savefig(output_pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"热图已保存至: {output_pdf_path}")


def save_offset_data(data, output_dir):
    """保存偏移百分比数据到CSV"""
    print("保存偏移百分比数据...")

    # 选择需要的列
    output_data = data[['country', 'cooling_energy_SSP2', 'cooling_energy_SSP1',
                        'cooling_energy_SSP2_50', 'cooling_energy_SSP2_3_125',
                        'diff_SSP1_SSP2', 'diff_SSP2_50_SSP2', 'diff_SSP2_3_125_SSP2',
                        'offset_percentage_50', 'offset_percentage_3_125']]

    # 保存到CSV
    output_csv_path = os.path.join(output_dir, 'global_cooling_energy_offset_percentage.csv')
    output_data.to_csv(output_csv_path, index=False, float_format='%.2f')
    print(f"偏移百分比数据已保存至: {output_csv_path}")

    # 打印统计信息
    print("\n偏移百分比统计信息:")
    print(
        f"SSP2-50 偏移百分比范围: {output_data['offset_percentage_50'].min():.2f}% 到 {output_data['offset_percentage_50'].max():.2f}%")
    print(
        f"SSP2-3.125 偏移百分比范围: {output_data['offset_percentage_3_125'].min():.2f}% 到 {output_data['offset_percentage_3_125'].max():.2f}%")

    return output_data


def main():
    print("开始计算全球制冷能耗偏移百分比...")

    # 创建输出目录
    output_dir = r"C:\Users\thuarchdog\Desktop\result\global_energy_offset_output"
    os.makedirs(output_dir, exist_ok=True)

    # 计算偏移百分比
    offset_data = calculate_offset_percentage()

    if offset_data is not None:
        # 保存数据
        save_offset_data(offset_data, output_dir)

        # 绘制热力图
        plot_offset_heatmap(offset_data, output_dir)

        print(f"\n所有结果已保存到: {output_dir}")
    else:
        print("计算失败，请检查数据文件路径")

    print("全球制冷能耗偏移百分比计算完成！")


if __name__ == "__main__":
    main()
