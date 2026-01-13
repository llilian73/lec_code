import geopandas as gpd
import pandas as pd
import os

def export_gadm_country_data():
    """
    从GADM GPKG文件中导出所有国家的信息到CSV文件
    """
    
    # 文件路径
    gpkg_file = r"Z:\local_environment_creation\shapefiles\gadm_410-gpkg\gadm_410.gpkg"
    output_dir = r"Z:\local_environment_creation"
    output_file = os.path.join(output_dir, "gadm_countries_info.csv")
    
    print("正在读取GADM GPKG文件...")
    
    try:
        # 首先检查GPKG文件中可用的图层
        import fiona
        print("检查GPKG文件中的可用图层...")
        layers = fiona.listlayers(gpkg_file)
        print(f"可用的图层: {layers}")
        
        # 尝试找到国家层级的图层
        country_layer = None
        possible_country_layers = ['ADM0', 'gadm_0', 'level0', 'countries', 'country']
        
        for layer_name in possible_country_layers:
            if layer_name in layers:
                country_layer = layer_name
                break
        
        if not country_layer:
            # 如果没有找到标准名称，使用第一个图层
            country_layer = layers[0]
            print(f"未找到标准国家图层，使用第一个图层: {country_layer}")
        else:
            print(f"找到国家图层: {country_layer}")
        
        # 读取GADM GPKG文件
        world = gpd.read_file(gpkg_file, layer=country_layer)
        print(f"成功读取GADM数据，共{len(world)}个国家/地区")
        
        # 查看可用的列名
        print("可用的列名:")
        print(world.columns.tolist())
        
        # 根据GADM的列名结构提取所需字段
        # GADM通常使用GID_0作为国家代码，NAME_0作为国家名称，CONTINENT作为大洲
        if 'GID_0' in world.columns and 'NAME_0' in world.columns:
            country_data = world[['GID_0', 'NAME_0']].copy()
            
            # 检查是否有CONTINENT列，如果没有则尝试其他可能的大洲列名
            continent_col = None
            possible_continent_cols = ['CONTINENT', 'continent', 'CONTINENT_', 'REGION_UN', 'REGION_WB']
            
            for col in possible_continent_cols:
                if col in world.columns:
                    continent_col = col
                    break
            
            if continent_col:
                country_data['continent'] = world[continent_col]
                print(f"使用大洲列: {continent_col}")
            else:
                # 如果没有大洲信息，设置为未知
                country_data['continent'] = 'Unknown'
                print("未找到大洲信息，设置为Unknown")
            
            # 重命名列以匹配要求的格式
            country_data = country_data.rename(columns={
                'GID_0': 'Country_Code',
                'NAME_0': 'Country_Name'
            })
            
        else:
            print("错误：未找到GID_0或NAME_0列")
            print("可用的列名:", world.columns.tolist())
            return None
        
        # 重新排列列顺序
        country_data = country_data[['continent', 'Country_Code', 'Country_Name']]
        
        print(f"总国家数量: {len(country_data)}")
        
        # 显示数据样本
        print("\n数据样本:")
        print(country_data.head(10))
        
        # 保存到CSV文件
        print(f"\n正在保存数据到 {output_file}...")
        country_data.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"成功保存{len(country_data)}个国家的信息到 {output_file}")
        
        return country_data
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    country_data = export_gadm_country_data()
    
    if country_data is not None:
        print(f"\n处理完成!")
        print(f"总共处理的国家数: {len(country_data)}")
        
        # 显示大洲分布
        if 'continent' in country_data.columns:
            print("\n大洲分布:")
            continent_counts = country_data['continent'].value_counts()
            print(continent_counts)
    else:
        print("处理失败，请检查文件路径和格式")
