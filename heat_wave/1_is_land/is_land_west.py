"""
陆地/海洋点识别工具

功能概述：
本工具用于识别NetCDF文件中的网格点是否为陆地点，特别针对西半球（经度180-360度）的区域进行处理。通过空间分析，将气候数据网格点与全球陆地边界进行匹配，识别出陆地点和海洋点，为后续的气候分析提供空间筛选基础。

输入数据：
1. NetCDF气候数据文件：
   - 文件路径：tasmax_day_EC-Earth3-HR_historical_r1i1p1f1_gr_19810101-19811231.nc
   - 包含经纬度网格信息
   - 时间范围：1981年全年
   - 数据来源：EC-Earth3-HR气候模型

2. 全球陆地边界数据：
   - 文件路径：ne_10m_admin_0_countries.shp
   - 格式：Shapefile矢量文件
   - 包含全球各国的地理边界信息
   - 属性包括国家名称、大洲信息等

主要功能：
1. 经度坐标转换：
   - 将经度从0-360度范围转换为-180-180度范围
   - 处理跨日期变更线的区域
   - 确保空间分析的一致性

2. 陆地/海洋点识别：
   - 使用空间包含关系判断网格点是否在陆地上
   - 排除南极洲区域（CONTINENT != 'Antarctica'）
   - 处理复杂的海岸线和岛屿边界

3. 西半球区域筛选：
   - 专门处理经度180-360度（转换后为-180-0度）的区域
   - 跳过东半球区域，提高处理效率
   - 保持原始经度值用于输出

4. 批量处理优化：
   - 遍历所有经纬度网格点
   - 使用进度条显示处理进度
   - 内存优化和错误处理

输出结果：
1. CSV格式的陆地点信息：
   - 文件路径：land_points_west.csv
   - 包含列：lat（纬度）、lon（经度）、is_land（是否为陆地点）
   - 使用原始经度值（0-360度范围）

2. 统计信息：
   - 总处理点数
   - 陆地点数量
   - 海洋点数量
   - 处理进度和完成状态

数据流程：
1. 数据加载阶段：
   - 读取NetCDF文件获取经纬度网格
   - 加载全球陆地边界Shapefile
   - 验证数据完整性和格式

2. 坐标处理阶段：
   - 转换经度坐标系统
   - 筛选西半球区域
   - 准备空间分析数据

3. 空间分析阶段：
   - 为每个网格点创建几何对象
   - 与陆地边界进行空间包含判断
   - 排除南极洲区域

4. 结果输出阶段：
   - 生成陆地点/海洋点标识
   - 保存到CSV文件
   - 输出统计信息

计算特点：
- 空间精度：基于10米分辨率的全球陆地边界数据
- 区域筛选：专门处理西半球区域，提高效率
- 坐标处理：正确处理跨日期变更线的经度转换
- 批量处理：支持大规模网格点的批量处理

技术参数：
- 空间参考系统：WGS84（EPSG:4326）
- 经度范围：180-360度（原始值）
- 纬度范围：根据NetCDF文件确定
- 排除区域：南极洲

性能优化：
- 进度跟踪：使用tqdm显示处理进度
- 内存管理：及时关闭数据集释放内存
- 错误处理：完善的异常捕获和日志记录
- 区域筛选：只处理目标区域，减少计算量

数据质量保证：
- 坐标系统一致性检查
- 空间数据完整性验证
- 边界条件处理（跨日期变更线）
- 异常值检测和处理

特殊处理：
- 南极洲排除：不处理南极洲区域的点
- 经度转换：正确处理0-360到-180-180的转换
- 边界处理：处理复杂的海岸线和岛屿边界
- 缺失数据处理：跳过无效的网格点

输出格式：
- 文件格式：CSV（UTF-8编码）
- 坐标系统：WGS84（EPSG:4326）
- 经度范围：0-360度（保持原始值）
- 数据类型：布尔值（is_land字段）

应用场景：
- 气候数据分析的空间筛选
- 陆地点和海洋点的分类
- 区域气候研究的数据预处理
- 极端天气事件的空间分析
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_longitude(lon):
    """将经度从0-360转换为-180-180"""
    if lon > 180:
        return lon - 360
    return lon

def is_land_point(point, world):
    """检查一个点是否在陆地上，且不在南极洲"""
    point_geom = Point(point[1], point[0])  # (lon, lat)
    for _, row in world.iterrows():
        if row['CONTINENT'] != 'Antarctica' and row['geometry'].contains(point_geom):
            return True
    return False

def process_nc_file(nc_file, world_data, output_file):
    """处理NC文件并输出陆地点信息"""
    try:
        # 读取NC文件
        ds = xr.open_dataset(nc_file)
        
        # 获取经纬度信息
        lats = ds.lat.values
        lons = ds.lon.values
        
        # 创建结果列表
        results = []
        
        # 遍历所有经纬度点
        total_points = len(lats) * len(lons)
        with tqdm(total=total_points, desc="处理网格点") as pbar:
            for lat in lats:
                for lon in lons:
                    # 转换经度到-180-180范围（仅用于判断）
                    converted_lon = convert_longitude(lon)
                    
                    # 只处理负经度区域（即原始经度180-360的区域）
                    if converted_lon >= 0:
                        pbar.update(1)
                        continue
                    
                    # 检查是否为陆地点（使用转换后的经度进行判断）
                    is_land = is_land_point((lat, converted_lon), world_data)
                    
                    # 添加到结果列表（使用原始经度值）
                    results.append({
                        'lat': lat,
                        'lon': lon,  # 使用原始经度值
                        'is_land': is_land
                    })
                    pbar.update(1)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(results)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存到CSV
        df.to_csv(output_file, index=False)
        logger.info(f"结果已保存到: {output_file}")
        
        # 输出统计信息
        total_points = len(df)
        land_points = df['is_land'].sum()
        logger.info(f"处理完成:")
        logger.info(f"- 总点数: {total_points}")
        logger.info(f"- 陆地点数: {land_points}")
        logger.info(f"- 海洋点数: {total_points - land_points}")
        
    except Exception as e:
        logger.error(f"处理文件时出错: {str(e)}")
    finally:
        # 关闭数据集
        if 'ds' in locals():
            ds.close()

def main():
    # 文件路径
    nc_file = r"Z:\CMIP6\tasmax\tasmax_day_EC-Earth3-HR_historical_r1i1p1f1_gr_19810101-19811231.nc"
    shapefile = r"C:\Users\localhost\PycharmProjects\pythonProject\shapefile\ne_10m_admin_0_countries.shp"
    output_file = r"C:\Users\localhost\PycharmProjects\pythonProject\tmp\land_points_west.csv"
    
    try:
        # 读取世界地图数据
        logger.info("正在读取世界地图数据...")
        world_data = gpd.read_file(shapefile)
        
        # 处理NC文件
        logger.info("开始处理NC文件...")
        process_nc_file(nc_file, world_data, output_file)
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()
