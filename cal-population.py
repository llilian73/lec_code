import pandas as pd
import pycountry
import pycountry_convert

# 读取CSV文件
df = pd.read_csv(r"D:\workstation\energy_comsuption\Population\global_province_population_2020.csv", encoding="gbk")

# 创建获取国家代码与对应洲的函数
def get_country_continent_mapping():
    mapping = {}
    for country in pycountry.countries:
        try:
            # 获取ISO 3166-1 alpha-3代码（三字母代码）
            alpha3 = country.alpha_3
            # 通过alpha-2代码获取大洲信息
            alpha2 = country.alpha_2
            if alpha2 in ['XK']:  # 特殊国家代码可忽略或单独处理
                continue
            continent_code = pycountry_convert.country_alpha2_to_continent_code(alpha2)
            continent_name = pycountry_convert.convert_continent_code_to_continent_name(continent_code)
            mapping[alpha3] = continent_name
        except Exception as e:
            print(f"⚠️ 跳过国家: {country.name}（{e}）")
    
    # 添加特殊情况的处理
    special_cases = {
        'TWN': 'Asia',     # 台湾
        'HKG': 'Asia',     # 香港
        'MAC': 'Asia',     # 澳门
        'XKX': 'Europe',   # 科索沃
    }
    mapping.update(special_cases)
    return mapping

# 获取完整的国家与大洲映射
continent_mapping = get_country_continent_mapping()

# 添加大洲列
df['continent'] = df['GID_0'].map(continent_mapping)

# 检查是否有未映射的国家
unmapped_countries = df[df['continent'].isna()]['GID_0'].unique()
if len(unmapped_countries) > 0:
    print("警告：以下国家代码未能映射到大洲：")
    print(unmapped_countries)
    print("请检查这些国家代码并手动添加到映射中")

# 重新排序列，将continent放在最前面
df = df[['continent', 'GID_0', 'NAME_0', 'GID_1', 'NAME_1', 'Population_2020']]

# 按大洲和国家代码排序
df = df.sort_values(['continent', 'GID_0', 'GID_1'])

# 保存重新排序后的完整数据
df.to_csv(r"D:\workstation\energy_comsuption\Population\global_province_population_2020_sorted.csv", 
          index=False)

# 计算国家级人口总数
country_population = df.groupby(['continent', 'GID_0', 'NAME_0'])['Population_2020'].sum().reset_index()

# 按大洲和国家排序
country_population = country_population.sort_values(['continent', 'GID_0'])

# 保存国家级人口数据
country_population.to_csv(r"D:\workstation\energy_comsuption\Population\country_population_2020.csv", 
                         index=False)

print("处理完成！")
print(f"已生成两个文件：")
print("1. global_province_population_2020_sorted.csv - 包含按大洲和国家排序的省级数据")
print("2. country_population_2020.csv - 包含国家级人口总数")

# 显示前几行数据作为示例
print("\n省级数据示例：")
print(df.head())
print("\n国家级数据示例：")
print(country_population.head())

# 显示统计信息
print("\n数据统计：")
print(f"总共处理的国家数量: {len(country_population)}")
print("\n各大洲国家数量：")
print(country_population['continent'].value_counts())