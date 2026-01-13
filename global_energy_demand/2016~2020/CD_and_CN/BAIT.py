"""
根据指定经纬度读取 BAIT 数据并绘制阈值图。

绘制方式参考 `draw.py` 中的 BAIT 阈值图：
- BAIT 时间序列作为折线
- 供暖阈值（heating）基线 14°C，向下递减 1°C 共 5 条线
- 制冷阈值（cooling）基线 20°C，向上递增 1°C 共 5 条线

使用示例（命令行）：
    python BAIT.py --lat 30.750 --lon 121.562

输出：
- PDF 图表保存至 `OUTPUT_DIR`（见常量定义），文件名包含经纬度信息。
"""

import argparse
import os
from typing import Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\result_half\2020"
OUTPUT_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\CDandCN"

# 可根据需要在此处直接修改默认经纬度
DEFAULT_LAT = 0.250 #30.750
DEFAULT_LON = 17.812 #121.562

BASE_HEATING = 14.0
BASE_COOLING = 20.0


def format_coord(value: str, decimals: int = 3) -> str:
    """将经纬度格式化为固定小数位字符串，便于匹配文件名。"""
    return f"{float(value):.{decimals}f}"


def build_file_path(lat: str, lon: str, data_dir: str) -> Tuple[str, str, str]:
    """
    根据输入经纬度拼接文件路径，并返回格式化后的经纬度字符串。
    Returns:
        (file_path, lat_str, lon_str)
    """
    lat_str = format_coord(lat)
    lon_str = format_coord(lon)
    filename = f"point_lat{lat_str}_lon{lon_str}_BAIT.csv"
    file_path = os.path.join(data_dir, filename)
    return file_path, lat_str, lon_str


def load_bait_data(file_path: str) -> pd.DataFrame:
    """读取 BAIT CSV 数据并返回按时间索引的 DataFrame。"""
    df = pd.read_csv(file_path, parse_dates=["time"])
    df = df.set_index("time").sort_index()
    return df


def plot_bait(df: pd.DataFrame, lat_str: str, lon_str: str, output_dir: str) -> str:
    """绘制 BAIT 与阈值图，并返回输出文件路径。"""
    plt.figure(figsize=(14, 9))
    ax = plt.gca()

    # 按日求平均，绘制逐日温度曲线
    daily_bait = df["BAIT"].resample("D").mean().dropna()
    ax.plot(daily_bait.index, daily_bait, color="black", linewidth=1.5, label="Daily Avg BAIT")

    # 设置阈值线
    heating_thresholds = [BASE_HEATING - i for i in range(5)]
    cooling_thresholds = [BASE_COOLING + i for i in range(5)]
    linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]

    for i, (thresh, style) in enumerate(zip(heating_thresholds, linestyles)):
        label = "Heating thresholds" if i == 0 else None
        ax.axhline(y=thresh, color="red", linestyle=style, alpha=0.6, label=label)

    for i, (thresh, style) in enumerate(zip(cooling_thresholds, linestyles)):
        label = "Cooling thresholds" if i == 0 else None
        ax.axhline(y=thresh, color="blue", linestyle=style, alpha=0.6, label=label)

    # X 轴刻度：季度
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.xticks(rotation=30)

    # 图表元素
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Daily Average BAIT and Threshold Lines - Lat {lat_str}, Lon {lon_str}")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"BAIT_lat{lat_str}_lon{lon_str}.pdf"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, format="pdf", bbox_inches='tight')
    plt.close()
    return output_path


def main(lat: str, lon: str, data_dir: str = DATA_DIR, output_dir: str = OUTPUT_DIR) -> str:
    file_path, lat_str, lon_str = build_file_path(lat, lon, data_dir)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到数据文件：{file_path}")
    df = load_bait_data(file_path)
    return plot_bait(df, lat_str, lon_str, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制指定经纬度的 BAIT 阈值图")
    parser.add_argument("--lat", default=DEFAULT_LAT, help="纬度（可包含小数），默认值来自脚本顶部常量")
    parser.add_argument("--lon", default=DEFAULT_LON, help="经度（可包含小数），默认值来自脚本顶部常量")
    parser.add_argument("--data-dir", default=DATA_DIR, help="CSV 数据所在目录")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="图表输出目录")
    args = parser.parse_args()

    output = main(args.lat, args.lon, args.data_dir, args.output_dir)
    print(f"已生成：{output}")

