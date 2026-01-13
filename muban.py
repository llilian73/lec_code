import geopandas as gpd
import pandas as pd
import os

def export_country_data():
    """
    从shapefile中导出所有国家的信息到CSV文件
    通过gadm_countries_info.csv获取continent信息
    通过global-electricity-per-kwh-pricing-2021.csv获取二字母国家代码
    """
    
    # 文件路径
    shapefile_path = r"Z:\local_environment_creation\shapefiles\全球国家边界\world_border2.shp"
    gadm_csv_path = r"Z:\local_environment_creation\gadm_countries_info.csv"
    electricity_csv_path = r"Z:\local_environment_creation\cost\global-electricity-per-kwh-pricing-2021.csv"
    output_dir = r"Z:\local_environment_creation"
    output_file = os.path.join(output_dir, "all_countries_info_1.csv")
    
    print("正在读取shapefile...")
    
    try:
        # 读取shapefile
        world = gpd.read_file(shapefile_path)
        print(f"成功读取shapefile，共{len(world)}个国家/地区")
        
        # 提取所需字段（根据实际shapefile的字段结构）
        # 新shapefile包含: GID_0 (国家代码), NAME_0 (国家名称), area, latitude
        country_data = world[['GID_0', 'NAME_0']].copy()
        
        # 重命名列
        country_data = country_data.rename(columns={
            'GID_0': 'Country_Code_3',
            'NAME_0': 'Country_Name'
        })
        
        # 读取gadm_countries_info.csv获取continent信息
        print("\n正在读取gadm_countries_info.csv...")
        gadm_df = pd.read_csv(gadm_csv_path, encoding='utf-8-sig')
        print(f"成功读取gadm_countries_info.csv，共{len(gadm_df)}条记录")
        
        # 创建gadm的映射字典（使用Country_Name作为键）
        gadm_continent_map = dict(zip(gadm_df['Country_Name'], gadm_df['continent']))
        
        # 读取global-electricity-per-kwh-pricing-2021.csv获取二字母国家代码
        print("\n正在读取global-electricity-per-kwh-pricing-2021.csv...")
        electricity_df = pd.read_csv(electricity_csv_path, encoding='latin-1')
        print(f"成功读取global-electricity-per-kwh-pricing-2021.csv，共{len(electricity_df)}条记录")
        
        # 创建electricity的映射字典（使用Country name作为键）
        electricity_code_map = dict(zip(electricity_df['Country name'], electricity_df['Country code']))
        
        # 根据Country_Name匹配continent
        country_data['continent'] = country_data['Country_Name'].map(gadm_continent_map)
        
        # 根据Country_Name匹配二字母国家代码
        country_data['Country_Code_2'] = country_data['Country_Name'].map(electricity_code_map)
        
        # 重新排列列顺序: continent, Country_Code_2, Country_Code_3, Country_Name
        country_data = country_data[['continent', 'Country_Code_2', 'Country_Code_3', 'Country_Name']]
        
        # 统计匹配情况
        matched_continent = country_data['continent'].notna().sum()
        matched_code2 = country_data['Country_Code_2'].notna().sum()
        print(f"\n匹配统计:")
        print(f"  成功匹配continent: {matched_continent}/{len(country_data)}")
        print(f"  成功匹配Country_Code_2: {matched_code2}/{len(country_data)}")
        
        # 显示未匹配的国家（用于调试）
        unmatched_continent = country_data[country_data['continent'].isna()]
        if len(unmatched_continent) > 0:
            print(f"\n未匹配到continent的国家 ({len(unmatched_continent)}个):")
            print(unmatched_continent[['Country_Name']].head(10))
        
        unmatched_code2 = country_data[country_data['Country_Code_2'].isna()]
        if len(unmatched_code2) > 0:
            print(f"\n未匹配到Country_Code_2的国家 ({len(unmatched_code2)}个):")
            print(unmatched_code2[['Country_Name']].head(10))
        
        print(f"\n总国家数量: {len(country_data)}")
        
        # 保存到CSV文件
        print(f"\n正在保存数据到 {output_file}...")
        country_data.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"成功保存{len(country_data)}个国家的信息到 {output_file}")
        
        # 显示数据样本
        print("\n数据样本:")
        print(country_data.head(10))
        
        return country_data
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    country_data = export_country_data()
    
    if country_data is not None:
        print(f"\n处理完成!")
        print(f"总共处理的国家数: {len(country_data)}")