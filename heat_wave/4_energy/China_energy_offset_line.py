import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.collections import LineCollection
from shapely.geometry import LineString

# --- 配置路径 ---
# SSP1和SSP2的省级能耗数据路径
ssp1_ref_path = r"C:\Users\thuarchdog\Desktop\result\China\SSP1\ref\ref_provincial_energy.csv"
ssp2_ref_path = r"C:\Users\thuarchdog\Desktop\result\China\SSP2\ref\ref_provincial_energy.csv"
ssp2_50_path = r"C:\Users\thuarchdog\Desktop\result\China\SSP2\case5\case5_provincial_energy.csv"
ssp2_3_125_path = r"C:\Users\thuarchdog\Desktop\result\China\SSP2\case9\case9_provincial_energy.csv"
plot_geojson_path = r"C:\Users\thuarchdog\Desktop\result\shp\中国_省.geojson"


def load_energy_data(file_path, scenario_name):
    """加载能耗数据"""
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        print(f"{scenario_name} 数据加载成功: {len(data)} 个省份")
        return data
    else:
        print(f"警告: {scenario_name} 文件不存在: {file_path}")
        return None


def calculate_offset_percentage():
    """计算偏移百分比"""
    print("开始计算中国各省份偏移百分比...")

    # 加载所有数据
    ssp1_ref = load_energy_data(ssp1_ref_path, "SSP1-ref")
    ssp2_ref = load_energy_data(ssp2_ref_path, "SSP2-ref")
    ssp2_50 = load_energy_data(ssp2_50_path, "SSP2-50")
    ssp2_3_125 = load_energy_data(ssp2_3_125_path, "SSP2-3.125")

    if ssp1_ref is None or ssp2_ref is None or ssp2_50 is None or ssp2_3_125 is None:
        print("错误: 无法加载所有必需的数据文件")
        return None

    # 合并所有数据到同一个DataFrame
    merged_data = ssp2_ref.merge(ssp1_ref, on='province', how='outer', suffixes=('_SSP2', '_SSP1'))
    merged_data = merged_data.merge(ssp2_50, on='province', how='outer', suffixes=('', '_SSP2_50'))
    merged_data = merged_data.merge(ssp2_3_125, on='province', how='outer', suffixes=('', '_SSP2_3_125'))

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
    merged_data['diff_SSP1_SSP2'] = merged_data['cooling_energy_SSP2'] - merged_data['cooling_energy_SSP1']
    merged_data['diff_SSP2_50_SSP2'] = merged_data['cooling_energy_SSP2'] - merged_data['cooling_energy_SSP2_50']
    merged_data['diff_SSP2_3_125_SSP2'] = merged_data['cooling_energy_SSP2'] - merged_data['cooling_energy_SSP2_3_125']

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

    print(f"偏移百分比计算完成，共处理 {len(merged_data)} 个省份")
    return merged_data


def create_hatch_lines(geometry, hatch_type='diagonal', spacing=2, linewidth=0.5):
    """为几何形状创建填充线条"""
    if geometry.is_empty:
        return []

    # 获取几何形状的边界框
    bounds = geometry.bounds
    minx, miny, maxx, maxy = bounds

    lines = []

    if hatch_type == 'diagonal':
        # 创建对角线填充
        for i in range(0, int(maxx - minx + maxy - miny), spacing):
            # 从左上到右下
            line1 = LineString([(minx + i, maxy), (maxx, maxy - i)])
            # 从右上到左下
            line2 = LineString([(maxx - i, maxy), (minx, maxy - i)])

            # 与几何形状相交
            if line1.intersects(geometry):
                intersection1 = line1.intersection(geometry)
                if not intersection1.is_empty:
                    if intersection1.geom_type == 'LineString':
                        lines.append(intersection1)
                    elif intersection1.geom_type == 'MultiLineString':
                        lines.extend(list(intersection1.geoms))

            if line2.intersects(geometry):
                intersection2 = line2.intersection(geometry)
                if not intersection2.is_empty:
                    if intersection2.geom_type == 'LineString':
                        lines.append(intersection2)
                    elif intersection2.geom_type == 'MultiLineString':
                        lines.extend(list(intersection2.geoms))

    elif hatch_type == 'horizontal':
        # 创建水平线填充
        for y in np.arange(miny, maxy, spacing):
            line = LineString([(minx, y), (maxx, y)])
            if line.intersects(geometry):
                intersection = line.intersection(geometry)
                if not intersection.is_empty:
                    if intersection.geom_type == 'LineString':
                        lines.append(intersection)
                    elif intersection.geom_type == 'MultiLineString':
                        lines.extend(list(intersection.geoms))

    elif hatch_type == 'vertical':
        # 创建垂直线填充
        for x in np.arange(minx, maxx, spacing):
            line = LineString([(x, miny), (x, maxy)])
            if line.intersects(geometry):
                intersection = line.intersection(geometry)
                if not intersection.is_empty:
                    if intersection.geom_type == 'LineString':
                        lines.append(intersection)
                    elif intersection.geom_type == 'MultiLineString':
                        lines.extend(list(intersection.geoms))

    return lines


def plot_offset_heatmap(data, output_dir):
    """绘制偏移百分比热力图"""
    print("开始绘制中国各省份偏移百分比热力图...")

    # 读取中国省份边界数据
    provinces_gdf = gpd.read_file(plot_geojson_path)

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
        merged_gdf = provinces_gdf.merge(
            data[['province', offset_col]],
            left_on='name',
            right_on='province',
            how='left'
        )

        # 填充NaN值为0
        merged_gdf[offset_col] = merged_gdf[offset_col].fillna(0)

        # 创建地图
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # 设置固定的颜色范围为0~100%
        vmin = 0
        vmax = 100

        # 绘制基础热图（0~100%）
        merged_gdf.plot(
            column=offset_col,
            cmap='Blues',  # 蓝色系
            linewidth=0.8,
            ax=ax,
            edgecolor='0.8',
            legend=True,
            legend_kwds={'label': f"Offset Percentage (%)", 'orientation': "horizontal", 'shrink': 0.7, 'pad': 0.02},
            vmin=vmin,
            vmax=vmax
        )

        # 为超过100%的省份添加填充模式
        # 100%~200%用斜杠填充
        high_percent_1 = merged_gdf[merged_gdf[offset_col] > 100]
        high_percent_1 = high_percent_1[high_percent_1[offset_col] <= 200]
        if not high_percent_1.empty:
            all_lines = []
            for idx, row in high_percent_1.iterrows():
                lines = create_hatch_lines(row.geometry, 'diagonal', spacing=3, linewidth=0.3)
                # 将LineString对象转换为坐标数组
                for line in lines:
                    if hasattr(line, 'coords'):
                        coords = list(line.coords)
                        all_lines.append(coords)

            if all_lines:
                lc = LineCollection(all_lines, colors='white', linewidths=0.3, alpha=0.8)
                ax.add_collection(lc)

        # 200%~300%用水平线填充
        high_percent_2 = merged_gdf[merged_gdf[offset_col] > 200]
        high_percent_2 = high_percent_2[high_percent_2[offset_col] <= 300]
        if not high_percent_2.empty:
            all_lines = []
            for idx, row in high_percent_2.iterrows():
                lines = create_hatch_lines(row.geometry, 'horizontal', spacing=2, linewidth=0.3)
                # 将LineString对象转换为坐标数组
                for line in lines:
                    if hasattr(line, 'coords'):
                        coords = list(line.coords)
                        all_lines.append(coords)

            if all_lines:
                lc = LineCollection(all_lines, colors='white', linewidths=0.3, alpha=0.8)
                ax.add_collection(lc)

        # 300%~400%用垂直线填充
        high_percent_3 = merged_gdf[merged_gdf[offset_col] > 300]
        high_percent_3 = high_percent_3[high_percent_3[offset_col] <= 400]
        if not high_percent_3.empty:
            all_lines = []
            for idx, row in high_percent_3.iterrows():
                lines = create_hatch_lines(row.geometry, 'vertical', spacing=2, linewidth=0.3)
                # 将LineString对象转换为坐标数组
                for line in lines:
                    if hasattr(line, 'coords'):
                        coords = list(line.coords)
                        all_lines.append(coords)

            if all_lines:
                lc = LineCollection(all_lines, colors='white', linewidths=0.3, alpha=0.8)
                ax.add_collection(lc)

        ax.set_title(f'China Provincial Cooling Energy Offset Percentage: {scenario_name}', fontsize=15)
        ax.set_axis_off()

        # 保存为PDF格式
        output_pdf_path = os.path.join(output_dir,
                                       f'China_cooling_energy_offset_{scenario_name.replace(" ", "_").replace("-", "_")}.pdf')
        plt.savefig(output_pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"热图已保存至: {output_pdf_path}")


def save_offset_data(data, output_dir):
    """保存偏移百分比数据到CSV"""
    print("保存中国各省份偏移百分比数据...")

    # 选择需要的列
    output_data = data[['province', 'cooling_energy_SSP2', 'cooling_energy_SSP1',
                        'cooling_energy_SSP2_50', 'cooling_energy_SSP2_3_125',
                        'diff_SSP1_SSP2', 'diff_SSP2_50_SSP2', 'diff_SSP2_3_125_SSP2',
                        'offset_percentage_50', 'offset_percentage_3_125']]

    # 保存到CSV
    output_csv_path = os.path.join(output_dir, 'China_cooling_energy_offset_percentage.csv')
    output_data.to_csv(output_csv_path, index=False, float_format='%.2f')
    print(f"偏移百分比数据已保存至: {output_csv_path}")

    # 打印统计信息
    print("\n中国各省份偏移百分比统计信息:")
    print(
        f"SSP2-50 偏移百分比范围: {output_data['offset_percentage_50'].min():.2f}% 到 {output_data['offset_percentage_50'].max():.2f}%")
    print(
        f"SSP2-3.125 偏移百分比范围: {output_data['offset_percentage_3_125'].min():.2f}% 到 {output_data['offset_percentage_3_125'].max():.2f}%")

    return output_data


def main():
    print("开始计算中国各省份制冷能耗偏移百分比...")

    # 创建输出目录
    output_dir = r"C:\Users\thuarchdog\Desktop\result\China\offset_analysis"
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

    print("中国各省份制冷能耗偏移百分比计算完成！")


if __name__ == "__main__":
    main()
