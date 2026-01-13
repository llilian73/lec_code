"""
人口数据重采样和导出工具

功能：
1. 读取TIFF格式的人口数据文件
2. 重采样到1°×1°分辨率（整数经纬度网格），使用sum方法汇总人口
3. 导出为CSV格式
"""

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

import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# 禁用rasterio的日志以避免编码问题
import logging
rasterio_logger = logging.getLogger('rasterio')
rasterio_logger.setLevel(logging.CRITICAL)  # 设置为CRITICAL级别，只显示严重错误

# 禁用rasterio._env的日志
rasterio_env_logger = logging.getLogger('rasterio._env')
rasterio_env_logger.setLevel(logging.CRITICAL)

# 输入文件路径（Linux路径）
INPUT_DIR = "/home/linbor/WORK/lishiying/population"
INPUT_FILES = [
    os.path.join(INPUT_DIR, "SSP1_2030.tif"),
    os.path.join(INPUT_DIR, "SSP2_2030.tif")
]

# 输出目录（Linux路径）
OUTPUT_DIR = "/home/linbor/WORK/lishiying/population"

# 国家地图路径（Linux路径）
COUNTRY_SHAPEFILE = "/home/linbor/WORK/lishiying/shapefiles/world_border2.shp"

# NetCDF文件路径（用于提取目标网格信息）
NC_PATH = "/home/linbor/WORK/lishiying/GCM_input_processed/BCC-CSM2-MR/future/SSP126/tas_day_BCC-CSM2-MR_ssp126_r1i1p1f1_gn_20300101-20341231_interpolated_1deg.nc"


def load_china_boundary():
    """
    加载中国边界（使用GID_0列，值为CHN）
    
    返回:
        china_gdf: 中国边界的GeoDataFrame
    """
    try:
        # 读取shapefile
        world_gdf = gpd.read_file(COUNTRY_SHAPEFILE)
        
        # 使用GID_0列筛选中国（值为CHN）
        if 'GID_0' not in world_gdf.columns:
            raise ValueError("shapefile中未找到GID_0列")
        
        china_gdf = world_gdf[world_gdf['GID_0'] == 'CHN'].copy()
        
        if len(china_gdf) == 0:
            raise ValueError("无法在shapefile中找到中国边界（GID_0='CHN'）")
        
        print(f"  找到中国边界: {len(china_gdf)} 个多边形")
        
        # 确保使用正确的CRS
        if china_gdf.crs is None:
            china_gdf.set_crs('EPSG:4326', inplace=True)
        else:
            china_gdf = china_gdf.to_crs('EPSG:4326')
        
        # 如果有多条记录，合并为一个多边形
        if len(china_gdf) > 1:
            china_gdf = gpd.GeoDataFrame([1], geometry=[china_gdf.unary_union], crs='EPSG:4326')
            print(f"  合并为单个多边形")
        
        return china_gdf
        
    except Exception as e:
        print(f"警告: 无法加载中国边界: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def calculate_china_population(df, china_gdf):
    """
    计算中国范围内的总人口
    
    参数:
        df: 包含lat, lon, population的DataFrame
        china_gdf: 中国边界的GeoDataFrame
    
    返回:
        total_population: 中国总人口
    """
    if china_gdf is None:
        return None
    
    try:
        # 创建点的GeoDataFrame
        geometry = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
        points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        
        # 空间连接，筛选出中国范围内的点
        china_points = gpd.sjoin(points_gdf, china_gdf, how='inner', predicate='within')
        
        # 计算总人口
        total_population = china_points['population'].sum()
        
        return total_population
        
    except Exception as e:
        print(f"警告: 计算中国人口时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# 全局变量：缓存NetCDF网格信息，避免重复读取
_nc_grid_cache = None

def load_nc_grid():
    """加载NetCDF网格信息（缓存结果）"""
    global _nc_grid_cache
    if _nc_grid_cache is not None:
        return _nc_grid_cache
    
    with xr.open_dataset(NC_PATH) as ds:
        lat = ds['lat'].values
        lon = ds['lon'].values
    
    # 经度范围已经是-180~180，不需要转换
    # 先判断纬度是升序还是降序，确保 res_lat 是负值（纬度向下递减）
    if lat[1] > lat[0]:  # 升序
        lat_max, lat_min = lat[-1], lat[0]
        res_lat = lat[0] - lat[1]  # 负值
    else:  # 降序
        lat_max, lat_min = lat[0], lat[-1]
        res_lat = lat[1] - lat[0]  # 负值
    
    res_lon = abs(lon[1] - lon[0])
    lon_min, lon_max = lon.min(), lon.max()
    
    height = len(lat)
    width = len(lon)
    
    # from_origin(west, north, xres, yres) 要求 yres 是负值（纬度向下递减）
    dst_transform = from_origin(lon_min, lat_max, res_lon, res_lat)
    dst_shape = (height, width)
    
    # 预先创建经纬度网格（避免每次重复创建）
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    _nc_grid_cache = {
        'lat': lat,
        'lon': lon,
        'lat_grid': lat_grid,
        'lon_grid': lon_grid,
        'dst_transform': dst_transform,
        'dst_shape': dst_shape,
        'height': height,
        'width': width
    }
    
    return _nc_grid_cache


def resample_tif_to_grid(input_file, output_file, china_gdf=None):
    """
    将TIFF文件重采样到NetCDF网格并导出为CSV
    使用population_eg.py的方法：从NetCDF文件提取网格信息，然后重采样
    
    参数:
        input_file: 输入TIFF文件路径
        output_file: 输出CSV文件路径
        china_gdf: 中国边界GeoDataFrame（可选）
    """
    print(f"\n处理文件: {input_file}")
    
    import time
    start_time = time.time()
    
    # ===== 加载 NetCDF 网格信息（使用缓存） =====
    grid_info = load_nc_grid()
    lat = grid_info['lat']
    lon = grid_info['lon']
    lat_grid = grid_info['lat_grid']
    lon_grid = grid_info['lon_grid']
    dst_transform = grid_info['dst_transform']
    dst_shape = grid_info['dst_shape']
    height = grid_info['height']
    width = grid_info['width']
    
    # ===== 打开人口 GeoTIFF 并重采样到 NetCDF 网格 =====
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with rasterio.open(input_file) as src:
                src_data = src.read(1)
                nodata = src.nodata
                
                print(f"\n原始数据范围:")
                print(f"  经度: {src.bounds.left:.6f}° 到 {src.bounds.right:.6f}°")
                print(f"  纬度: {src.bounds.bottom:.6f}° 到 {src.bounds.top:.6f}°")
                print(f"  数据尺寸: {src.width} x {src.height}")
                print(f"  无效值: {nodata}")
                
                dst_data = np.zeros(dst_shape, dtype=np.float32)
                
                print(f"\n开始重采样（使用sum方法汇总人口）...")
                
                # 使用reproject进行重采样，使用sum方法汇总人口
                # 使用src_nodata参数处理无效值
                reproject(
                    source=src_data,
                    destination=dst_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=nodata,  # 指定源数据的nodata值
                    dst_transform=dst_transform,
                    dst_crs='EPSG:4326',
                    dst_nodata=np.nan,  # 目标数据的nodata值设为NaN
                    resampling=Resampling.sum  # 汇总人口
                )
                
                elapsed_total = time.time() - start_time
                print(f"重采样完成 (总耗时: {elapsed_total:.1f}秒)")
                
                # 补充：将nan转为0（避免后续DataFrame出现nan）
                dst_data = np.nan_to_num(dst_data, nan=0.0)
                
                # 处理可能的负值或异常值
                dst_data[dst_data < 0] = 0
                
                valid_dst_data = dst_data[dst_data > 0]
                if len(valid_dst_data) > 0:
                    print(f"  重采样后数据范围: {valid_dst_data.min():.2f} 到 {dst_data.max():.2f}")
                print(f"  非零数据网格数: {np.count_nonzero(dst_data)}")
                
                # 创建DataFrame（使用预计算的网格，避免重复计算）
                df = pd.DataFrame({
                    'lat': lat_grid.flatten(),
                    'lon': lon_grid.flatten(),
                    'population': dst_data.flatten()
                })
                
                # 按纬度和经度排序（使用更高效的方法）
                df = df.sort_values(['lat', 'lon'], kind='mergesort')
                
                # 计算中国总人口
                if china_gdf is not None:
                    print(f"\n计算中国总人口...")
                    print(f"  总网格点数: {len(df)}")
                    print(f"  非零人口网格数: {len(df[df['population'] > 0])}")
                    print(f"  全球总人口（所有网格）: {df['population'].sum():,.0f} 人")
                    
                    china_population = calculate_china_population(df, china_gdf)
                    if china_population is not None:
                        # 创建点的GeoDataFrame用于调试
                        geometry = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
                        points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
                        china_points = gpd.sjoin(points_gdf, china_gdf, how='inner', predicate='within')
                        
                        print(f"  中国范围内网格点数: {len(china_points)}")
                        print(f"  中国范围内非零人口网格数: {len(china_points[china_points['population'] > 0])}")
                        if len(china_points[china_points['population'] > 0]) > 0:
                            print(f"  中国范围内平均每网格人口: {china_points[china_points['population'] > 0]['population'].mean():,.0f} 人")
                            print(f"  中国范围内最大网格人口: {china_points['population'].max():,.0f} 人")
                        
                        print(f"  中国总人口: {china_population:,.0f} 人")
                    else:
                        print(f"  警告: 无法计算中国总人口")
                
                # 保存为CSV
                print(f"\n保存到: {output_file}")
                df.to_csv(output_file, index=False)
                print(f"✓ 成功保存 {len(df)} 行数据")
                
                return df
        except Exception as e:
            print(f"错误: 无法打开文件或处理数据: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """主函数"""
    print("=" * 80)
    print("人口数据重采样和导出工具")
    print("=" * 80)
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载中国边界
    print(f"\n加载中国边界...")
    china_gdf = load_china_boundary()
    if china_gdf is not None:
        print(f"✓ 成功加载中国边界")
    else:
        print(f"⚠ 警告: 无法加载中国边界，将跳过中国人口计算")
    
    # 准备文件处理任务
    tasks = []
    for input_file in INPUT_FILES:
        if not os.path.exists(input_file):
            print(f"警告: 文件不存在，跳过: {input_file}")
            continue
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(OUTPUT_DIR, f"{base_name}_1deg.csv")
        tasks.append((input_file, output_file))
    
    if not tasks:
        print("没有需要处理的文件")
        return
    
    print(f"\n处理配置:")
    print(f"  - 处理文件数: {len(tasks)}")
    print(f"  - 处理方式: 串行处理文件（一个处理完再处理下一个）")
    print(f"  - 重采样方法: sum（汇总人口）")
    print("=" * 80)
    
    # 串行处理文件
    for file_idx, (input_file, output_file) in enumerate(tasks, 1):
        print(f"\n{'='*80}")
        print(f"处理文件 {file_idx}/{len(tasks)}: {os.path.basename(input_file)}")
        print(f"{'='*80}")
        
        try:
            df = resample_tif_to_grid(input_file, output_file, china_gdf=china_gdf)
            print(f"\n✓ [{file_idx}/{len(tasks)}] 处理完成: {input_file}")
        except Exception as e:
            print(f"\n✗ [{file_idx}/{len(tasks)}] 处理失败: {input_file}")
            print(f"  错误: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("所有文件处理完成")
    print("=" * 80)


if __name__ == "__main__":
    main()

