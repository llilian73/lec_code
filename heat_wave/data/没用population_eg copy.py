import os
import sys
import warnings

# 修复Windows下的编码问题
if sys.platform == 'win32':
    # 禁用rasterio的警告和错误日志
    os.environ['CPL_ZIP_ENCODING'] = 'UTF-8'
    # 设置GDAL相关环境变量
    if 'GDAL_DATA' not in os.environ:
        # 如果GDAL_DATA未设置，尝试从常见位置查找
        possible_paths = [
            r"C:\Program Files\QGIS 3.36.0\share\gdal",
            r"C:\OSGeo4W\share\gdal",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                os.environ['GDAL_DATA'] = path
                break

# 捕获并忽略rasterio的编码警告
warnings.filterwarnings('ignore', category=UnicodeWarning)
warnings.filterwarnings('ignore', message='.*codec.*')

import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.crs import CRS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

# 禁用rasterio的日志以避免编码问题
import logging
rasterio_logger = logging.getLogger('rasterio')
rasterio_logger.setLevel(logging.CRITICAL)  # 设置为CRITICAL级别，只显示严重错误

# 禁用rasterio._env的日志
rasterio_env_logger = logging.getLogger('rasterio._env')
rasterio_env_logger.setLevel(logging.CRITICAL)

# ===== 文件路径 =====
nc_path = r"Z:\local_environment_creation\heat_wave\GCM_input_filter\tas_day_BCC-CSM2-MR_ssp126_r1i1p1f1_gn_20300101-20341231_interpolated_1deg.nc"
pop_tif_path = r"Z:\local_environment_creation\heat_wave\population\SSP1_2030.tif"
output_path = r"Z:\local_environment_creation\heat_wave\population\SSP1_2030_aligned_to_CMIP6.tif"

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

height = len(lat)
width = len(lon)

dst_transform = from_origin(lon_min, lat_max, res_lon, res_lat)
dst_shape = (height, width)

# ===== 打开人口 GeoTIFF 并重采样到 CMIP6 网格 =====
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with rasterio.open(pop_tif_path) as src:
        src_data = src.read(1)
        dst_data = np.zeros(dst_shape, dtype=np.float32)

        # 处理CRS：避免PROJ版本不兼容问题
        src_crs = src.crs
        if src_crs is None:
            # 如果没有CRS，假设是EPSG:4326
            try:
                src_crs = CRS.from_epsg(4326)
            except:
                src_crs = None
            if src_crs is None:
                print(f"  警告: 源文件没有CRS，将尝试使用默认CRS")
        
        # 目标CRS：使用CRS对象而不是字符串，避免PROJ版本问题
        try:
            dst_crs = CRS.from_epsg(4326)
        except Exception as e:
            # 如果EPSG:4326无法创建，尝试使用源CRS或None
            print(f"  警告: 无法创建EPSG:4326 CRS对象: {str(e)}")
            if src_crs is not None:
                # 如果源CRS和目标CRS相同，直接使用源CRS
                if '4326' in str(src_crs) or str(src_crs) == 'EPSG:4326':
                    dst_crs = src_crs
                    print(f"  使用源文件的CRS（EPSG:4326）")
                else:
                    # 尝试使用源CRS
                    dst_crs = src_crs
                    print(f"  使用源文件的CRS: {src_crs}")
            else:
                # 最后尝试：使用None，让rasterio自动处理
                dst_crs = None
                print(f"  使用None作为CRS，让rasterio自动处理")

        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,  # 使用处理后的CRS对象，而不是字符串
            resampling=Resampling.sum  # 汇总人口
        )

        # ===== 写入重采样后的新 GeoTIFF =====
        # 处理输出CRS
        try:
            output_crs = CRS.from_epsg(4326)
        except Exception as e:
            # 如果EPSG:4326无法创建，使用之前处理的目标CRS
            print(f"  警告: 无法创建输出EPSG:4326 CRS对象: {str(e)}")
            output_crs = dst_crs if dst_crs is not None else "EPSG:4326"
        
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=dst_data.shape[0],
            width=dst_data.shape[1],
            count=1,
            dtype=dst_data.dtype,
            crs=output_crs,
            transform=dst_transform,
            nodata=0
        ) as dst:
            dst.write(dst_data, 1)

print(f"\n✅ 新人口栅格已保存至：{output_path}")

# ===== 绘制重采样后的人口地图热图 =====
print(f"\n开始绘制人口地图热图...")

# 处理数据：将0和负值设为NaN以便在图中显示为透明
plot_data = dst_data.copy()
plot_data[plot_data <= 0] = np.nan

# 创建图形
fig, ax = plt.subplots(figsize=(16, 8))

# 创建经纬度网格用于绘图
lon_grid, lat_grid = np.meshgrid(lon, lat)

# 使用对数刻度显示人口（因为人口数据范围很大）
# 设置颜色映射
cmap = plt.cm.get_cmap('YlOrRd').copy()
cmap.set_bad('lightgray', 1.0)  # 将NaN值显示为浅灰色

# 计算有效数据的最大值（用于设置颜色范围）
valid_data = plot_data[~np.isnan(plot_data)]
if len(valid_data) > 0:
    vmax = valid_data.max()
    vmin = max(1, valid_data.min())  # 最小值至少为1（对数刻度需要）
else:
    vmax = 1
    vmin = 1

# 绘制热图
im = ax.pcolormesh(lon_grid, lat_grid, plot_data, 
                   cmap=cmap, 
                   norm=LogNorm(vmin=vmin, vmax=vmax),
                   shading='auto')

# 设置坐标轴
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xlabel('经度 (°)', fontsize=12)
ax.set_ylabel('纬度 (°)', fontsize=12)
ax.set_title('重采样后的人口分布热图', fontsize=14, fontweight='bold')

# 添加颜色条
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=40)
cbar.set_label('人口数 (人)', fontsize=12)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

# 设置坐标轴刻度
ax.set_xticks(np.arange(-180, 181, 30))
ax.set_yticks(np.arange(-90, 91, 30))

# 调整布局
plt.tight_layout()

# 保存图片
output_image_path = output_path.replace('.tif', '_heatmap.png')
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
print(f"✅ 人口地图热图已保存至：{output_image_path}")

# 打印统计信息
if len(valid_data) > 0:
    print(f"  数据统计:")
    print(f"    最小值: {valid_data.min():,.0f} 人")
    print(f"    最大值: {valid_data.max():,.0f} 人")
    print(f"    平均值: {valid_data.mean():,.0f} 人")
    print(f"    非零网格数: {len(valid_data)}")

# 可选：显示图片（如果在交互式环境中）
plt.show()

plt.close()
