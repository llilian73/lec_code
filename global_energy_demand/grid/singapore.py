import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import os

def check_singapore_points():
    """
    检查指定文件中是否有落在新加坡内部的点
    """
    
    # 文件路径
    nc_file = r"Z:\local_environment_creation\energy_consumption_gird\weather\2020\M2T1NXSLV\M2T1NXSLV.5.12.4%3AMERRA2_400.tavg1_2d_slv_Nx.20200101.nc4.dap.nc4@dap4.ce=%2FQV2M;%2FT2M;%2FU2M;%2FV2M;%2Ftime;%2Flat;%2Flon"
    shapefile_path = r"Z:\local_environment_creation\shapefiles\ne_50m_admin_0_countries\ne_50m_admin_0_countries.shp"
    
    print("正在检查新加坡内部的点...")
    
    try:
        # 1. 加载国家边界数据
        print("加载国家边界数据...")
        world = gpd.read_file(shapefile_path)
        
        # 确保坐标系一致
        if world.crs != 'EPSG:4326':
            world = world.to_crs('EPSG:4326')
        
        # 筛选出新加坡
        singapore = world[world['NAME'] == 'Singapore']
        
        if len(singapore) == 0:
            print("错误：未找到新加坡")
            return
        
        print(f"找到新加坡，共{len(singapore)}个几何体")
        
        # 2. 加载NetCDF文件
        print("加载NetCDF文件...")
        if not os.path.exists(nc_file):
            print(f"错误：文件不存在 - {nc_file}")
            return
            
        ds = xr.open_dataset(nc_file)
        
        # 获取经纬度数据
        if 'lat' in ds.variables and 'lon' in ds.variables:
            lats = ds['lat'].values
            lons = ds['lon'].values
        elif 'latitude' in ds.variables and 'longitude' in ds.variables:
            lats = ds['latitude'].values
            lons = ds['longitude'].values
        else:
            print("错误：未找到经纬度变量")
            print(f"可用的变量: {list(ds.variables.keys())}")
            return
        
        print(f"数据维度: lat={lats.shape}, lon={lons.shape}")
        
        # 3. 获取新加坡的边界框，只检查附近的点
        singapore_bounds = singapore.total_bounds  # [minx, miny, maxx, maxy]
        min_lon, min_lat, max_lon, max_lat = singapore_bounds
        
        # 添加一些缓冲区域（约0.1度）
        buffer = 0.1
        min_lon -= buffer
        min_lat -= buffer
        max_lon += buffer
        max_lat += buffer
        
        print(f"新加坡边界框: 经度[{min_lon:.3f}, {max_lon:.3f}], 纬度[{min_lat:.3f}, {max_lat:.3f}]")
        
        # 创建网格点
        if len(lats.shape) == 1 and len(lons.shape) == 1:
            # 1D坐标 - 筛选新加坡附近的点
            lat_mask = (lats >= min_lat) & (lats <= max_lat)
            lon_mask = (lons >= min_lon) & (lons <= max_lon)
            
            # 找到符合条件的索引
            lat_indices = np.where(lat_mask)[0]
            lon_indices = np.where(lon_mask)[0]
            
            if len(lat_indices) == 0 or len(lon_indices) == 0:
                print("无")
                return
            
            # 创建筛选后的坐标网格
            filtered_lats = lats[lat_mask]
            filtered_lons = lons[lon_mask]
            lon_grid, lat_grid = np.meshgrid(filtered_lons, filtered_lats)
            
            # 创建索引网格
            lon_idx_grid, lat_idx_grid = np.meshgrid(lon_indices, lat_indices)
            
        else:
            # 2D坐标 - 筛选新加坡附近的点
            lat_mask = (lats >= min_lat) & (lats <= max_lat)
            lon_mask = (lons >= min_lon) & (lons <= max_lon)
            
            # 找到符合条件的索引
            lat_indices, lon_indices = np.where(lat_mask & lon_mask)
            
            if len(lat_indices) == 0:
                print("无")
                return
            
            # 创建筛选后的坐标
            filtered_lats = lats[lat_indices, lon_indices]
            filtered_lons = lons[lat_indices, lon_indices]
            
            # 重新组织为网格形式
            unique_lats = np.unique(filtered_lats)
            unique_lons = np.unique(filtered_lons)
            lon_grid, lat_grid = np.meshgrid(unique_lons, unique_lats)
            
            # 创建索引映射
            lat_idx_map = {lat: i for i, lat in enumerate(unique_lats)}
            lon_idx_map = {lon: j for j, lon in enumerate(unique_lons)}
        
        print(f"筛选后检查 {lon_grid.size} 个点（原总点数: {lats.size if hasattr(lats, 'size') else 'N/A'}）")
        
        singapore_points = []
        
        for i in range(lon_grid.shape[0]):
            for j in range(lon_grid.shape[1]):
                lat = lat_grid[i, j]
                lon = lon_grid[i, j]
                
                # 创建点几何体
                point = Point(lon, lat)
                
                # 检查点是否在新加坡内部
                for _, singapore_geom in singapore.iterrows():
                    if singapore_geom.geometry.contains(point):
                        # 计算原始索引
                        if len(lats.shape) == 1 and len(lons.shape) == 1:
                            orig_i = lat_idx_grid[i, j]
                            orig_j = lon_idx_grid[i, j]
                        else:
                            # 对于2D坐标，需要找到原始索引
                            orig_i, orig_j = lat_indices[i], lon_indices[j]
                        
                        singapore_points.append({
                            'lat': lat,
                            'lon': lon,
                            'i': orig_i,
                            'j': orig_j
                        })
                        break
        
        # 4. 输出结果
        if singapore_points:
            print(f"找到 {len(singapore_points)} 个落在新加坡内部的点:")
            print("纬度\t经度\t索引(i,j)")
            print("-" * 40)
            for point in singapore_points:
                print(f"{point['lat']:.6f}\t{point['lon']:.6f}\t({point['i']},{point['j']})")
            
            # 保存到CSV文件
            output_file = "singapore_points.csv"
            df = pd.DataFrame(singapore_points)
            df.to_csv(output_file, index=False)
            print(f"\n结果已保存到: {output_file}")
            
        else:
            print("无")
        
        # 关闭数据集
        ds.close()
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_singapore_points()
