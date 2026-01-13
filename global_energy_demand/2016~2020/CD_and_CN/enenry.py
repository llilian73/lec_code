"""
计算 Democratic Republic of the Congo（CD）和 China（CN）的逐时能耗。

数据来源与参数说明：
- 点位与国家映射：point_country_mapping.csv
- 网格点逐时结果：result_half/2020/point_lat{lat}_lon{lon}_{cooling|heating}.csv
- 功率系数：parameters.csv（与 3_country copy已修改.py 保持一致）

计算流程：
1. 找到两个国家对应的所有经纬度点；
2. 读取每个点的制冷/供暖逐时数据（包含 ref 与 case1-case20）；
3. 汇总至国家层面，并应用功率系数，将结果转换为 TWh；
4. 输出逐时能耗（总量、供暖、制冷），保留 11 位有效数字。

输出文件：
- Z:\local_environment_creation\energy_consumption_gird\result\CDandCN\CD_hourly_energy.csv
- Z:\local_environment_creation\energy_consumption_gird\result\CDandCN\CN_hourly_energy.csv
"""

import os
from typing import Dict, List, Tuple

import pandas as pd

# 路径与常量
DATA_YEAR = 2020
RESULT_BASE_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\result_half"
POINT_COUNTRY_MAPPING_FILE = r"Z:\local_environment_creation\energy_consumption_gird\result\point_country_mapping.csv"
PARAMETERS_FILE = r"Z:\local_environment_creation\energy_consumption_gird\parameters.csv"
OUTPUT_DIR = r"Z:\local_environment_creation\energy_consumption_gird\result\CDandCN"
COUNTRY_CODES = {
    "CD": "Democratic Republic of the Congo",
    "CN": "China",
}


def format_coord(value: float) -> str:
    """将坐标格式化为文件名使用的 3 位小数。"""
    return f"{float(value):.3f}"


def load_parameters() -> Dict[str, Dict[str, float]]:
    """读取功率系数，返回 {ISO: {heating_power, cooling_power}}。"""
    params_df = pd.read_csv(PARAMETERS_FILE)
    required_cols = {"region", "heating power", "Cooling power"}
    missing_cols = required_cols - set(params_df.columns)
    if missing_cols:
        raise KeyError(f"参数文件缺少列: {', '.join(sorted(missing_cols))}")
    params_df = params_df.set_index("region")
    return params_df[["heating power", "Cooling power"]].to_dict(orient="index")


def load_country_points() -> Dict[str, List[Tuple[float, float]]]:
    """返回每个目标国家对应的经纬度点列表。"""
    mapping_df = pd.read_csv(POINT_COUNTRY_MAPPING_FILE)
    required_cols = {"lat", "lon", "Country_Code"}
    missing_cols = required_cols - set(mapping_df.columns)
    if missing_cols:
        raise KeyError(f"点-国家映射文件缺少列: {', '.join(sorted(missing_cols))}")

    mapping_df["Country_Code"] = mapping_df["Country_Code"].replace({"CN-TW": "CN"})
    mapping_subset = mapping_df[mapping_df["Country_Code"].isin(COUNTRY_CODES.keys())]
    mapping_subset = mapping_subset[["Country_Code", "lat", "lon"]].drop_duplicates()

    country_points: Dict[str, List[Tuple[float, float]]] = {code: [] for code in COUNTRY_CODES}
    for _, row in mapping_subset.iterrows():
        country_points[row["Country_Code"]].append((row["lat"], row["lon"]))
    return country_points


def load_point_timeseries(lat: float, lon: float, data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取单个点的制冷和供暖逐时数据，返回 (cooling_df, heating_df)。"""
    lat_str = format_coord(lat)
    lon_str = format_coord(lon)
    base_name = f"point_lat{lat_str}_lon{lon_str}"
    cooling_path = os.path.join(data_dir, f"{base_name}_cooling.csv")
    heating_path = os.path.join(data_dir, f"{base_name}_heating.csv")

    if not os.path.exists(cooling_path):
        raise FileNotFoundError(f"未找到制冷文件: {cooling_path}")
    if not os.path.exists(heating_path):
        raise FileNotFoundError(f"未找到供暖文件: {heating_path}")

    cooling_df = pd.read_csv(cooling_path, parse_dates=["time"])
    heating_df = pd.read_csv(heating_path, parse_dates=["time"])

    # 去除潜在的索引列
    cooling_df = cooling_df.loc[:, ~cooling_df.columns.str.contains(r"^Unnamed")]
    heating_df = heating_df.loc[:, ~heating_df.columns.str.contains(r"^Unnamed")]

    cooling_df = cooling_df.set_index("time").sort_index()
    heating_df = heating_df.set_index("time").sort_index()

    return cooling_df, heating_df


def aggregate_country_hourly(country_code: str,
                             points: List[Tuple[float, float]],
                             params: Dict[str, Dict[str, float]],
                             data_dir: str) -> pd.DataFrame:
    """按照国家汇总逐时的制冷/供暖数据并应用功率系数。"""
    if not points:
        raise ValueError(f"{COUNTRY_CODES[country_code]} 未在映射文件中找到任何点。")

    default_heating_power = 27.9
    default_cooling_power = 48.5

    heating_power = params.get(country_code, {}).get("heating power", default_heating_power)
    cooling_power = params.get(country_code, {}).get("Cooling power", default_cooling_power)

    cooling_sum = None
    heating_sum = None
    cases: List[str] = []

    for lat, lon in points:
        cooling_df, heating_df = load_point_timeseries(lat, lon, data_dir)

        if cooling_sum is None:
            cases = [col for col in cooling_df.columns if col != "time"]
            cooling_sum = cooling_df[cases].copy()
            heating_sum = heating_df[cases].copy()
        else:
            cooling_sum = cooling_sum.add(cooling_df[cases], fill_value=0.0)
            heating_sum = heating_sum.add(heating_df[cases], fill_value=0.0)

    # 应用功率系数并转换为 TWh（与国家汇总脚本保持一致）
    cooling_energy = cooling_sum * cooling_power / 1e3
    heating_energy = heating_sum * heating_power / 1e3
    total_energy = heating_energy + cooling_energy

    # 整合输出
    output_frames = []
    for case in cases:
        case_df = pd.DataFrame({
            f"{case}_heating_TWh": heating_energy[case],
            f"{case}_cooling_TWh": cooling_energy[case],
            f"{case}_total_TWh": total_energy[case],
        })
        output_frames.append(case_df)

    country_df = pd.concat(output_frames, axis=1)
    country_df.index.name = "time"
    return country_df


def format_significant_digits(df: pd.DataFrame, digits: int = 11) -> pd.DataFrame:
    """将 DataFrame 数值转换为指定有效数字的字符串表示。"""
    formatted = df.applymap(lambda x: format(x, f".{digits}g"))
    formatted.insert(0, "time", df.index.astype(str))
    return formatted


def main():
    output_path = os.path.join(RESULT_BASE_DIR, str(DATA_YEAR))
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"数据目录不存在: {output_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    params = load_parameters()
    country_points = load_country_points()

    for country_code in COUNTRY_CODES:
        country_df = aggregate_country_hourly(
            country_code=country_code,
            points=country_points[country_code],
            params=params,
            data_dir=output_path,
        )
        formatted_df = format_significant_digits(country_df)
        output_file = os.path.join(OUTPUT_DIR, f"{country_code}_hourly_energy.csv")
        formatted_df.to_csv(output_file, index=False)
        print(f"{COUNTRY_CODES[country_code]} 逐时能耗已保存: {output_file}")


if __name__ == "__main__":
    main()

