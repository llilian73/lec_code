import geopandas as gpd
import matplotlib.pyplot as plt

# 文件路径
shapefile_path = r"Z:\local_environment_creation\shapefiles\全球国家边界\world_border2.shp"

print("正在读取shapefile...")

try:
    # 读取shapefile
    world = gpd.read_file(shapefile_path)
    print(f"成功读取shapefile，共{len(world)}个国家/地区")
    print(f"可用的列名: {world.columns.tolist()}")
    
    # 确保坐标系正确
    if world.crs is None:
        world.set_crs(epsg=4326, inplace=True)
    elif world.crs != 'EPSG:4326':
        world = world.to_crs('EPSG:4326')
    
    # 获取经纬度范围
    bounds = world.total_bounds  # [minx, miny, maxx, maxy] 在EPSG:4326中对应 [最小经度, 最小纬度, 最大经度, 最大纬度]
    min_lon, min_lat, max_lon, max_lat = bounds
    
    print(f"\n经纬度范围:")
    print(f"  经度范围: {min_lon:.6f}° ~ {max_lon:.6f}°")
    print(f"  纬度范围: {min_lat:.6f}° ~ {max_lat:.6f}°")
    print(f"  经度跨度: {max_lon - min_lon:.6f}°")
    print(f"  纬度跨度: {max_lat - min_lat:.6f}°")
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    
    # 绘制地图（可以根据国家名称着色）
    world.plot(ax=ax, 
               column='NAME_0',  # 根据国家名称着色
               cmap='Set3',      # 使用颜色映射
               edgecolor='black',  # 边界颜色
               linewidth=0.3,    # 边界线宽
               legend=False)      # 不显示图例（因为国家太多）
    
    # 设置标题
    ax.set_title('World Map - Country Boundaries', fontsize=16, fontweight='bold', pad=20)
    
    # 移除坐标轴
    ax.axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 显示地图
    print("正在显示地图...")
    plt.show()
    
    print("地图显示完成！")
    
except Exception as e:
    print(f"处理过程中出错: {e}")
    import traceback
    traceback.print_exc()

