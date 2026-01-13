"""
读取 enenry.py 的输出结果，绘制 Democratic Republic of the Congo（CD）
与 China（CN）的逐时能耗对比图（ref vs case6），并补充人均能耗图。

绘图风格参考 draw.py 中的逐时能耗绘制方式，输出 PDF 文件，尺寸 14:9。
"""

import os
from typing import Dict, Iterable, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\CDandCN"
OUTPUT_DIR = DATA_DIR
POPULATION_FILE = r"Z:\local_environment_creation\energy_consumption_gird\result\country_population_2020.csv"

COUNTRY_INFO: Dict[str, str] = {
    "CD": "Democratic Republic of the Congo",
    "CN": "China",
}

CASE_REF = "ref"
CASE_TARGET = "case6"
TIME_COL = "time"


def load_country_data(country_code: str) -> pd.DataFrame:
    """加载指定国家的逐时能耗 CSV，并转换为数值与时间索引。"""
    file_path = os.path.join(DATA_DIR, f"{country_code}_hourly_energy.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到输入文件：{file_path}")

    df = pd.read_csv(file_path)
    if TIME_COL not in df.columns:
        raise KeyError(f"{file_path} 缺少 '{TIME_COL}' 列")

    numeric_cols = [col for col in df.columns if col != TIME_COL]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.set_index(TIME_COL).sort_index()

    required_cols = [
        f"{CASE_REF}_heating_TWh",
        f"{CASE_REF}_cooling_TWh",
        f"{CASE_REF}_total_TWh",
        f"{CASE_TARGET}_heating_TWh",
        f"{CASE_TARGET}_cooling_TWh",
        f"{CASE_TARGET}_total_TWh",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"{file_path} 缺少必要列: {', '.join(missing)}")

    return df


def load_population_data() -> Dict[str, float]:
    """读取人口数据，返回 {Country_Code: Population_2020}。"""
    df = pd.read_csv(POPULATION_FILE)
    required_cols = {"Country_Code", "Population_2020"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"人口文件缺少列: {', '.join(sorted(missing))}")
    populations = df.set_index("Country_Code")["Population_2020"].to_dict()
    return populations


def plot_curves(country_code: str,
                country_name: str,
                series_configs: Iterable[Tuple[pd.Series, str, str, str]],
                title: str,
                y_label: str,
                filename_suffix: str,
                threshold: float) -> None:
    """通用绘图函数，根据阈值决定是否绘制曲线。"""
    plt.figure(figsize=(14, 9))
    ax = plt.gca()

    plotted = False
    for series, label, color, linestyle in series_configs:
        if series.max() < threshold:
            continue
        ax.plot(series.index, series, color=color, linewidth=1.2, linestyle=linestyle, label=label)
        plotted = True

    ax.set_title(title, fontsize=16)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_xlabel("Time", fontsize=13)

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d"))
    plt.xticks(rotation=30)

    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.6)
    if plotted:
        ax.legend(loc="upper right", ncol=2, fontsize=11)

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, f"{country_code}_{filename_suffix}.pdf")
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"{country_name} 图像已保存：{output_file}")


def plot_hourly_comparison(country_code: str, country_name: str, df: pd.DataFrame, population: float) -> None:
    """绘制单个国家的总量与人均逐时能耗对比图。"""
    # 总能耗（TWh -> GWh）
    energy_configs = [
        (df[f"{CASE_REF}_heating_TWh"] * 1e3, "Ref Heating", "red", "-"),
        (df[f"{CASE_REF}_cooling_TWh"] * 1e3, "Ref Cooling", "blue", "-"),
        (df[f"{CASE_TARGET}_heating_TWh"] * 1e3, "Case 6 Heating", "lightcoral", "-"),
        (df[f"{CASE_TARGET}_cooling_TWh"] * 1e3, "Case 6 Cooling", "lightskyblue", "-"),
    ]
    plot_curves(
        country_code=country_code,
        country_name=country_name,
        series_configs=energy_configs,
        title=f"Hourly Energy Demand (Ref vs Case 6) - {country_name}",
        y_label="Energy Demand (GWh)",
        filename_suffix="hourly_energy_ref_case6",
        threshold=0.01,
    )

    # 人均能耗（TWh -> kWh/person）
    per_capita_configs = [
        (df[f"{CASE_REF}_heating_TWh"] * 1e9 / population, "Ref Heating", "red", "-"),
        (df[f"{CASE_REF}_cooling_TWh"] * 1e9 / population, "Ref Cooling", "blue", "-"),
        (df[f"{CASE_TARGET}_heating_TWh"] * 1e9 / population, "Case 6 Heating", "lightcoral", "-"),
        (df[f"{CASE_TARGET}_cooling_TWh"] * 1e9 / population, "Case 6 Cooling", "lightskyblue", "-"),
    ]
    plot_curves(
        country_code=country_code,
        country_name=country_name,
        series_configs=per_capita_configs,
        title=f"Hourly Per Capita Energy (Ref vs Case 6) - {country_name}",
        y_label="Per Capita Energy (kWh/capita)",
        filename_suffix="hourly_energy_per_capita_ref_case6",
        threshold=0.01,
    )


def main():
    populations = load_population_data()
    for code, name in COUNTRY_INFO.items():
        df = load_country_data(code)
        if code not in populations or pd.isna(populations[code]):
            raise KeyError(f"人口数据中缺少 {code}")
        plot_hourly_comparison(code, name, df, populations[code])


if __name__ == "__main__":
    main()

