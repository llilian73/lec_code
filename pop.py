"""
查看TIFF文件的经纬度范围
"""

import rasterio
from rasterio.transform import xy

# TIFF文件路径
tif_file = r"Z:\local_environment_creation\heat_wave\population\SSP1_2030.tif"

try:
    # 打开TIFF文件
    with rasterio.open(tif_file) as src:
        # 获取地理边界框（bounds）
        bounds = src.bounds
        
        # 获取坐标参考系统（CRS）
        crs = src.crs
        
        # 获取图像尺寸
        width = src.width
        height = src.height
        
        # 获取地理变换参数
        transform = src.transform
        
        print("=" * 80)
        print(f"文件: {tif_file}")
        print("=" * 80)
        print(f"\n坐标参考系统 (CRS): {crs}")
        print(f"\n图像尺寸: {width} x {height} 像素")
        print(f"\n地理变换参数:")
        print(f"  {transform}")
        
        print(f"\n经纬度范围:")
        print(f"  最小经度 (left):   {bounds.left:.6f}°")
        print(f"  最大经度 (right):  {bounds.right:.6f}°")
        print(f"  最小纬度 (bottom): {bounds.bottom:.6f}°")
        print(f"  最大纬度 (top):    {bounds.top:.6f}°")
        
        print(f"\n边界框 (bounds):")
        print(f"  ({bounds.left:.6f}, {bounds.bottom:.6f}, {bounds.right:.6f}, {bounds.top:.6f})")
        print(f"  (min_lon, min_lat, max_lon, max_lat)")
        
        # 计算经度和纬度的跨度
        lon_span = bounds.right - bounds.left
        lat_span = bounds.top - bounds.bottom
        
        print(f"\n跨度:")
        print(f"  经度跨度: {lon_span:.6f}°")
        print(f"  纬度跨度: {lat_span:.6f}°")
        
        # 获取像素分辨率
        if transform:
            pixel_width = abs(transform[0])  # 经度方向的分辨率
            pixel_height = abs(transform[4])  # 纬度方向的分辨率
            print(f"\n像素分辨率:")
            print(f"  经度方向: {pixel_width:.6f}° / 像素")
            print(f"  纬度方向: {pixel_height:.6f}° / 像素")
        
        # 获取数据类型和统计信息
        print(f"\n数据类型: {src.dtypes[0]}")
        print(f"波段数: {src.count}")
        
        # 读取第一个波段的数据（用于统计）
        data = src.read(1)
        valid_data = data[data != src.nodata] if src.nodata is not None else data
        
        if len(valid_data) > 0:
            print(f"\n数据统计 (第一个波段):")
            print(f"  最小值: {valid_data.min()}")
            print(f"  最大值: {valid_data.max()}")
            print(f"  平均值: {valid_data.mean():.2f}")
            print(f"  有效像素数: {len(valid_data)}")
            if src.nodata is not None:
                print(f"  无效值 (nodata): {src.nodata}")
        
        print("=" * 80)
        
except FileNotFoundError:
    print(f"错误: 文件不存在 - {tif_file}")
except Exception as e:
    print(f"错误: {str(e)}")
    import traceback
    traceback.print_exc()

