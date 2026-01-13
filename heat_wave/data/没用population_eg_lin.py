import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import numpy as np
import os

# ===== 文件路径 =====
nc_path = r"/home/linbor/WORK/lishiying/GCM_input_processed/BCC-CSM2-MR/future/SSP126/tas_day_BCC-CSM2-MR_ssp126_r1i1p1f1_gn_20300101-20341231_interpolated_1deg.nc"
pop_tif_path = r"/home/linbor/WORK/lishiying/population/SSP1_2030.tif"
output_path = r"/home/linbor/WORK/lishiying/population/SSP1_2030_aligned_to_CMIP6.tif"

# ===== 提取 NetCDF 网格信息 =====
ds = xr.open_dataset(nc_path)
lat = ds['lat'].values
lon = ds['lon'].values


# 重新排序经度和数据（如果需要）
sort_idx = np.argsort(lon)
lon = lon[sort_idx]

res_lat = abs(lat[1] - lat[0])
res_lon = abs(lon[1] - lon[0])
lat_min, lat_max = lat.min(), lat.max()
lon_min, lon_max = lon.min(), lon.max()

print(f"[NetCDF网格信息]")
print(f"纬度范围: {lat_min} ~ {lat_max}，分辨率: {res_lat}")
print(f"经度范围: {lon_min} ~ {lon_max}，分辨率: {res_lon}")

height = int((lat_max - lat_min) / res_lat)
width = int((lon_max - lon_min) / res_lon)

dst_transform = from_origin(lon_min, lat_max, res_lon, res_lat)
dst_shape = (height, width)

# ===== 打开人口 GeoTIFF 并重采样到 CMIP6 网格 =====
with rasterio.open(pop_tif_path) as src:
    src_data = src.read(1)
    dst_data = np.zeros(dst_shape, dtype=np.float32)

    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs='EPSG:4326',
        resampling=Resampling.sum  # 汇总人口
    )

    # ===== 写入重采样后的新 GeoTIFF =====
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=dst_data.shape[0],
        width=dst_data.shape[1],
        count=1,
        dtype=dst_data.dtype,
        crs="EPSG:4326",
        transform=dst_transform,
        nodata=0
    ) as dst:
        dst.write(dst_data, 1)

print(f"\n✅ 新人口栅格已保存至：{output_path}")
