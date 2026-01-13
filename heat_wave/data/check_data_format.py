"""
检查6个气候模型的数据格式

检查内容：
- 经纬度范围
- 网格大小（分辨率）
- 时间范围
- 数据变量
- 文件大小
"""

import xarray as xr
import numpy as np
import os
import pandas as pd
from pathlib import Path

# 定义文件路径
BASE_DIR = r"Z:\local_environment_creation\heat_wave\GCM_input\historical"
EC_EARTH_DIR = r"Z:\CMIP6\tasmax"

# 定义要检查的模型文件
MODEL_FILES = {
    "ACCESS-ESM1-5": os.path.join(BASE_DIR, "tasmax_day_ACCESS-ESM1-5_historical_r1i1p1f1_gn_19500101-19991231.nc"),
    "BCC-CSM2-MR": os.path.join(BASE_DIR, "tasmax_day_BCC-CSM2-MR_historical_r1i1p1f1_gn_19750101-19991231.nc"),
    "CanESM5": os.path.join(BASE_DIR, "tasmax_day_CanESM5_historical_r1i1p1f1_gn_18500101-20141231.nc"),
    "EC-Earth3": os.path.join(EC_EARTH_DIR, "tasmax_day_EC-Earth3-HR_historical_r1i1p1f1_gr_18510101-18511231.nc"),
    "MPI-ESM1-2-HR": os.path.join(BASE_DIR, "tasmax_day_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_19800101-19841231.nc"),
    "MRI-ESM2-0": r"Z:\local_environment_creation\heat_wave\GCM_input\MRI-ESM2-0\huss_day_MRI-ESM2-0_ssp126_r1i1p1f1_gn_20150101-20641231.nc"
}

# 输出文件路径
OUTPUT_DIR = r"Z:\local_environment_creation\heat_wave\GCM_input\historical"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "model_format_check.txt")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "model_format_summary.csv")


def find_mri_file():
    """查找MRI-ESM2-0文件（已直接指定路径，此函数保留用于备用查找）"""
    # 如果直接指定的路径不存在，尝试查找
    mri_path = MODEL_FILES.get("MRI-ESM2-0")
    if mri_path and os.path.exists(mri_path):
        return mri_path
    
    # 在BASE_DIR中查找MRI-ESM2-0文件
    if os.path.exists(BASE_DIR):
        for file in os.listdir(BASE_DIR):
            if "MRI-ESM2-0" in file and file.endswith(".nc"):
                return os.path.join(BASE_DIR, file)
    
    # 也在EC_EARTH_DIR中查找
    if os.path.exists(EC_EARTH_DIR):
        for file in os.listdir(EC_EARTH_DIR):
            if "MRI-ESM2-0" in file and file.endswith(".nc"):
                return os.path.join(EC_EARTH_DIR, file)
    
    return None


def check_file_format(model_name, file_path):
    """检查单个文件格式"""
    result = {
        "Model": model_name,
        "File_Path": file_path,
        "File_Exists": False,
        "File_Size_MB": None,
        "Lat_Min": None,
        "Lat_Max": None,
        "Lon_Min": None,
        "Lon_Max": None,
        "Lat_Count": None,
        "Lon_Count": None,
        "Lat_Resolution": None,
        "Lon_Resolution": None,
        "Time_Start": None,
        "Time_End": None,
        "Time_Count": None,
        "Variables": None,
        "Dimensions": None,
        "Error": None
    }
    
    if file_path is None or not os.path.exists(file_path):
        result["Error"] = "File not found"
        return result
    
    try:
        # 获取文件大小
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        result["File_Size_MB"] = f"{file_size:.2f}"
        result["File_Exists"] = True
        
        # 打开NetCDF文件
        ds = xr.open_dataset(file_path)
        
        # 获取维度信息
        result["Dimensions"] = str(dict(ds.dims))
        
        # 获取变量信息
        result["Variables"] = ", ".join(list(ds.variables.keys()))
        
        # 获取经纬度信息（尝试不同的变量名）
        lat_var = None
        lon_var = None
        
        # 常见的经纬度变量名
        lat_names = ['lat', 'latitude', 'Lat', 'Latitude']
        lon_names = ['lon', 'longitude', 'Lon', 'Longitude']
        
        for name in lat_names:
            if name in ds.variables or name in ds.coords:
                lat_var = name
                break
        
        for name in lon_names:
            if name in ds.variables or name in ds.coords:
                lon_var = name
                break
        
        if lat_var is None or lon_var is None:
            result["Error"] = f"Lat/Lon variables not found. Available: {list(ds.variables.keys())}"
            ds.close()
            return result
        
        # 获取经纬度数据
        lats = ds[lat_var].values
        lons = ds[lon_var].values
        
        # 计算经纬度范围
        result["Lat_Min"] = f"{float(np.min(lats)):.4f}"
        result["Lat_Max"] = f"{float(np.max(lats)):.4f}"
        result["Lon_Min"] = f"{float(np.min(lons)):.4f}"
        result["Lon_Max"] = f"{float(np.max(lons)):.4f}"
        
        # 计算网格数量
        result["Lat_Count"] = len(lats)
        result["Lon_Count"] = len(lons)
        
        # 计算分辨率（如果是一维数组）
        if len(lats.shape) == 1 and len(lats) > 1:
            lat_res = np.abs(np.diff(lats)).mean()
            result["Lat_Resolution"] = f"{lat_res:.4f}"
        else:
            result["Lat_Resolution"] = "N/A (2D grid)"
        
        if len(lons.shape) == 1 and len(lons) > 1:
            lon_res = np.abs(np.diff(lons)).mean()
            result["Lon_Resolution"] = f"{lon_res:.4f}"
        else:
            result["Lon_Resolution"] = "N/A (2D grid)"
        
        # 获取时间信息
        time_var = None
        time_names = ['time', 'Time', 'TIME']
        
        for name in time_names:
            if name in ds.variables or name in ds.coords:
                time_var = name
                break
        
        if time_var:
            times = ds[time_var].values
            if len(times) > 0:
                result["Time_Start"] = str(times[0])
                result["Time_End"] = str(times[-1])
                result["Time_Count"] = len(times)
            else:
                result["Time_Start"] = "N/A"
                result["Time_End"] = "N/A"
                result["Time_Count"] = 0
        else:
            result["Time_Start"] = "Time variable not found"
            result["Time_End"] = "Time variable not found"
            result["Time_Count"] = 0
        
        ds.close()
        
    except Exception as e:
        result["Error"] = str(e)
    
    return result


def format_output_text(results):
    """格式化输出文本"""
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("气候模型数据格式检查报告")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    for result in results:
        model_name = result["Model"]
        output_lines.append(f"\n{'=' * 80}")
        output_lines.append(f"模型: {model_name}")
        output_lines.append(f"{'=' * 80}")
        
        if not result["File_Exists"]:
            output_lines.append(f"❌ 文件不存在")
            if result["Error"]:
                output_lines.append(f"错误: {result['Error']}")
            continue
        
        output_lines.append(f"\n文件路径: {result['File_Path']}")
        output_lines.append(f"文件大小: {result['File_Size_MB']} MB")
        
        if result["Error"]:
            output_lines.append(f"❌ 错误: {result['Error']}")
            continue
        
        output_lines.append(f"\n📊 空间信息:")
        output_lines.append(f"  纬度范围: {result['Lat_Min']}° 到 {result['Lat_Max']}°")
        output_lines.append(f"  经度范围: {result['Lon_Min']}° 到 {result['Lon_Max']}°")
        output_lines.append(f"  纬度网格数: {result['Lat_Count']}")
        output_lines.append(f"  经度网格数: {result['Lon_Count']}")
        output_lines.append(f"  纬度分辨率: {result['Lat_Resolution']}°")
        output_lines.append(f"  经度分辨率: {result['Lon_Resolution']}°")
        output_lines.append(f"  总网格点数: {result['Lat_Count']} × {result['Lon_Count']} = {result['Lat_Count'] * result['Lon_Count']}")
        
        output_lines.append(f"\n⏰ 时间信息:")
        output_lines.append(f"  起始时间: {result['Time_Start']}")
        output_lines.append(f"  结束时间: {result['Time_End']}")
        output_lines.append(f"  时间步数: {result['Time_Count']}")
        
        output_lines.append(f"\n📦 数据信息:")
        output_lines.append(f"  维度: {result['Dimensions']}")
        output_lines.append(f"  变量: {result['Variables']}")
    
    output_lines.append(f"\n{'=' * 80}")
    output_lines.append("检查完成")
    output_lines.append(f"{'=' * 80}")
    
    return "\n".join(output_lines)


def main():
    """主函数"""
    print("开始检查气候模型数据格式...")
    
    # 检查MRI-ESM2-0文件是否存在
    mri_file = MODEL_FILES.get("MRI-ESM2-0")
    if mri_file and os.path.exists(mri_file):
        print(f"找到MRI-ESM2-0文件: {mri_file}")
    elif mri_file:
        print(f"⚠️  警告: MRI-ESM2-0文件不存在: {mri_file}")
        # 尝试查找其他MRI-ESM2-0文件
        found_file = find_mri_file()
        if found_file:
            MODEL_FILES["MRI-ESM2-0"] = found_file
            print(f"找到替代的MRI-ESM2-0文件: {found_file}")
    else:
        print("⚠️  警告: 未指定MRI-ESM2-0文件路径")
    
    # 检查所有文件
    results = []
    for model_name, file_path in MODEL_FILES.items():
        print(f"\n正在检查 {model_name}...")
        result = check_file_format(model_name, file_path)
        results.append(result)
        
        if result["File_Exists"] and not result["Error"]:
            print(f"  ✓ {model_name} 检查完成")
        else:
            print(f"  ✗ {model_name} 检查失败: {result.get('Error', 'File not found')}")
    
    # 生成输出文本
    output_text = format_output_text(results)
    
    # 打印到控制台
    print("\n" + output_text)
    
    # 保存到文件
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(output_text)
    print(f"\n✓ 结果已保存到: {OUTPUT_FILE}")
    
    # 保存CSV摘要
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✓ CSV摘要已保存到: {OUTPUT_CSV}")
    
    # 打印摘要统计
    print("\n" + "=" * 80)
    print("摘要统计:")
    print("=" * 80)
    successful = sum(1 for r in results if r["File_Exists"] and not r.get("Error"))
    print(f"成功检查: {successful}/{len(results)} 个模型")
    
    if successful > 0:
        print("\n各模型网格大小:")
        for result in results:
            if result["File_Exists"] and not result.get("Error"):
                print(f"  {result['Model']}: {result['Lat_Count']} × {result['Lon_Count']} = {result['Lat_Count'] * result['Lon_Count']:,} 个网格点")


if __name__ == "__main__":
    main()

