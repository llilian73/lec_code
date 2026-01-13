"""
人口数据处理脚本

功能：
1. 读取TIFF格式的人口数据文件
2. 计算全球总人口
3. 计算中国总人口（使用shapefile筛选）
4. 计算美国总人口（使用shapefile筛选）
5. 导出为CSV格式（lat, lon, population）
"""

import os
import sys
import warnings

# 修复Windows下的编码问题
if sys.platform == 'win32':
    os.environ['CPL_ZIP_ENCODING'] = 'UTF-8'
    if 'GDAL_DATA' not in os.environ:
        possible_paths = [
            r"C:\Program Files\QGIS 3.36.0\share\gdal",
            r"C:\OSGeo4W\share\gdal",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                os.environ['GDAL_DATA'] = path
                break

warnings.filterwarnings('ignore', category=UnicodeWarning)
warnings.filterwarnings('ignore', message='.*codec.*')

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# 禁用rasterio的日志
import logging
rasterio_logger = logging.getLogger('rasterio')
rasterio_logger.setLevel(logging.CRITICAL)
rasterio_env_logger = logging.getLogger('rasterio._env')
rasterio_env_logger.setLevel(logging.CRITICAL)

# 输入文件路径
INPUT_FILE = r"Z:\local_environment_creation\heat_wave\population\SSP2.tif"

# 国家地图路径
COUNTRY_SHAPEFILE = r"Z:\local_environment_creation\shapefiles\全球国家边界\world_border2.shp"

# 输出CSV文件路径
OUTPUT_CSV = r"Z:\local_environment_creation\heat_wave\population\SSP2_population.csv"


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


def load_usa_boundary():
    """
    加载美国边界（使用GID_0列，值为USA）
    
    返回:
        usa_gdf: 美国边界的GeoDataFrame
    """
    try:
        # 读取shapefile
        world_gdf = gpd.read_file(COUNTRY_SHAPEFILE)
        
        # 使用GID_0列筛选美国（值为USA）
        if 'GID_0' not in world_gdf.columns:
            raise ValueError("shapefile中未找到GID_0列")
        
        usa_gdf = world_gdf[world_gdf['GID_0'] == 'USA'].copy()
        
        if len(usa_gdf) == 0:
            raise ValueError("无法在shapefile中找到美国边界（GID_0='USA'）")
        
        print(f"  找到美国边界: {len(usa_gdf)} 个多边形")
        
        # 确保使用正确的CRS
        if usa_gdf.crs is None:
            usa_gdf.set_crs('EPSG:4326', inplace=True)
        else:
            usa_gdf = usa_gdf.to_crs('EPSG:4326')
        
        # 如果有多条记录，合并为一个多边形
        if len(usa_gdf) > 1:
            usa_gdf = gpd.GeoDataFrame([1], geometry=[usa_gdf.unary_union], crs='EPSG:4326')
            print(f"  合并为单个多边形")
        
        return usa_gdf
        
    except Exception as e:
        print(f"警告: 无法加载美国边界: {str(e)}")
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
        
        # 确保CRS匹配
        if points_gdf.crs != china_gdf.crs:
            points_gdf = points_gdf.to_crs(china_gdf.crs)
        
        # 空间连接，筛选出中国范围内的点
        china_points = gpd.sjoin(points_gdf, china_gdf, how='inner', predicate='within')
        
        # 如果使用within没有结果，尝试使用intersects（包含边界上的点）
        if len(china_points) == 0:
            china_points = gpd.sjoin(points_gdf, china_gdf, how='inner', predicate='intersects')
        
        # 计算总人口
        total_population = china_points['population'].sum()
        
        return total_population
        
    except Exception as e:
        print(f"警告: 计算中国人口时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def calculate_usa_population(df, usa_gdf):
    """
    计算美国范围内的总人口
    
    参数:
        df: 包含lat, lon, population的DataFrame
        usa_gdf: 美国边界的GeoDataFrame
    
    返回:
        total_population: 美国总人口
    """
    if usa_gdf is None:
        return None
    
    try:
        # 创建点的GeoDataFrame
        geometry = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
        points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        
        # 确保CRS匹配
        if points_gdf.crs != usa_gdf.crs:
            points_gdf = points_gdf.to_crs(usa_gdf.crs)
        
        # 空间连接，筛选出美国范围内的点
        usa_points = gpd.sjoin(points_gdf, usa_gdf, how='inner', predicate='within')
        
        # 如果使用within没有结果，尝试使用intersects（包含边界上的点）
        if len(usa_points) == 0:
            usa_points = gpd.sjoin(points_gdf, usa_gdf, how='inner', predicate='intersects')
        
        # 计算总人口
        total_population = usa_points['population'].sum()
        
        return total_population
        
    except Exception as e:
        print(f"警告: 计算美国人口时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def process_population_data():
    """
    处理人口数据：读取TIFF，计算全球、中国和美国总人口，导出CSV
    """
    print("=" * 80)
    print("人口数据处理")
    print("=" * 80)
    
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 输入文件不存在: {INPUT_FILE}")
        return
    
    # 加载中国边界
    print(f"\n加载中国边界...")
    china_gdf = load_china_boundary()
    if china_gdf is not None:
        print(f"✓ 成功加载中国边界")
    else:
        print(f"⚠ 警告: 无法加载中国边界，将跳过中国人口计算")
    
    # 加载美国边界
    print(f"\n加载美国边界...")
    usa_gdf = load_usa_boundary()
    if usa_gdf is not None:
        print(f"✓ 成功加载美国边界")
    else:
        print(f"⚠ 警告: 无法加载美国边界，将跳过美国人口计算")
    
    # 读取人口数据
    print(f"\n读取人口数据文件: {INPUT_FILE}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with rasterio.open(INPUT_FILE) as src:
                # 读取数据
                data = src.read(1)
                nodata = src.nodata
                
                print(f"\n原始数据信息:")
                print(f"  数据尺寸: {src.width} x {src.height}")
                print(f"  经度范围: {src.bounds.left:.6f}° 到 {src.bounds.right:.6f}°")
                print(f"  纬度范围: {src.bounds.bottom:.6f}° 到 {src.bounds.top:.6f}°")
                print(f"  无效值: {nodata}")
                
                # 处理无效值
                if nodata is None or nodata > 1e30:
                    # 常见背景值（FLT_MAX）
                    nodata = 3.4028235e38
                    print(f"  修正后的无效值: {nodata} (FLT_MAX)")
                
                # 将无效值设为0
                if nodata is not None:
                    data = np.where(data == nodata, 0, data)
                
                # 处理负值
                data = np.where(data < 0, 0, data)
                
                # 获取每个像素的经纬度
                height, width = data.shape
                
                print(f"\n提取像素坐标和人口数据...")
                # 使用rasterio的xy方法获取每个像素的中心坐标
                lons = []
                lats = []
                for row in range(height):
                    for col in range(width):
                        lon, lat = src.xy(row, col)
                        lons.append(lon)
                        lats.append(lat)
                
                # 将数据展平（按行优先顺序，与上面的循环顺序一致）
                populations = data.flatten()
                
                # 创建DataFrame
                df = pd.DataFrame({
                    'lat': lats,
                    'lon': lons,
                    'population': populations
                })
                
                print(f"  总像素数: {len(df)}")
                print(f"  非零人口像素数: {len(df[df['population'] > 0])}")
                
                # 计算全球总人口
                global_population = df['population'].sum()
                print(f"\n全球总人口: {global_population:,.0f} 人")
                
                # 计算中国总人口
                if china_gdf is not None:
                    print(f"\n计算中国总人口...")
                    china_population = calculate_china_population(df, china_gdf)
                    if china_population is not None:
                        print(f"中国总人口: {china_population:,.0f} 人")
                        print(f"中国人口占全球比例: {china_population/global_population*100:.2f}%")
                    else:
                        print(f"警告: 无法计算中国总人口")
                
                # 计算美国总人口
                if usa_gdf is not None:
                    print(f"\n计算美国总人口...")
                    usa_population = calculate_usa_population(df, usa_gdf)
                    if usa_population is not None:
                        print(f"美国总人口: {usa_population:,.0f} 人")
                        print(f"美国人口占全球比例: {usa_population/global_population*100:.2f}%")
                    else:
                        print(f"警告: 无法计算美国总人口")
                
                # 保存为CSV
                print(f"\n保存CSV文件: {OUTPUT_CSV}")
                # 确保输出目录存在
                output_dir = os.path.dirname(OUTPUT_CSV)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                df.to_csv(OUTPUT_CSV, index=False)
                print(f"✓ 成功保存 {len(df)} 行数据")
                print(f"  CSV格式: lat,lon,population")
                
        except Exception as e:
            print(f"错误: 无法处理数据: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    process_population_data()

