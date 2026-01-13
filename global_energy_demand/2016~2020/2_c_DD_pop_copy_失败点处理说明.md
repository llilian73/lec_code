# 2_c_DD_pop copy.py 失败点重新处理工具说明

## 功能概述
本工具专门用于重新计算之前处理失败的网格点的BAIT和能耗需求。从失败点CSV文件中提取坐标信息，重新进行BAIT和能耗计算，并保存到对应的输出目录。

## 主要功能

### 1. 失败点坐标提取
- 从失败点CSV文件中读取错误信息
- 使用正则表达式提取坐标信息
- 支持格式：`"保存点 (54.25, 15.312) 失败: [Errno 22] Invalid argument"`

### 2. 人口数据匹配
- 根据提取的坐标在人口数据中查找对应的人口数
- 使用0.001度的容差进行坐标匹配
- 确保每个失败点都有对应的人口数据

### 3. 重新计算处理
- 重新加载气候数据
- 重新计算BAIT
- 重新计算能耗需求
- 保存所有结果文件

### 4. 结果保存
- 保存到对应的年份目录
- 生成新的失败点报告（如果仍有失败）
- 记录处理统计信息

## 处理的年份
- 2017年：`Z:\local_environment_creation\energy_consumption_gird\result\result_half\2017\failed_energy_points_2017.csv`
- 2018年：`Z:\local_environment_creation\energy_consumption_gird\result\result_half\2018\failed_energy_points_2018.csv`
- 2020年：`Z:\local_environment_creation\energy_consumption_gird\result\result_half\2020\failed_energy_points_2020.csv`

## 输出文件
对于每个成功处理的点，会生成以下文件：
- `point_lat{lat}_lon{lon}_BAIT.csv`：BAIT数据
- `point_lat{lat}_lon{lon}_cooling.csv`：制冷需求数据
- `point_lat{lat}_lon{lon}_heating.csv`：供暖需求数据
- `point_lat{lat}_lon{lon}_total.csv`：总需求数据

## 新增功能

### 1. 坐标提取函数
```python
def extract_coordinates_from_error(error_message):
    """从错误信息中提取坐标"""
```

### 2. 失败点加载函数
```python
def load_failed_points_from_csv(failed_points_path):
    """从失败点CSV文件中加载失败点的坐标"""
```

### 3. 人口数据匹配函数
```python
def get_population_for_point(lat, lon, population_df):
    """根据坐标获取人口数据"""
```

## 处理流程

1. **加载人口数据**：读取完整的人口数据文件
2. **收集失败点**：从各年份的失败点CSV文件中提取坐标
3. **匹配人口数据**：为每个失败点找到对应的人口数
4. **并行处理**：使用多进程重新计算失败点
5. **保存结果**：将结果保存到对应的年份目录
6. **生成报告**：记录处理统计和仍然失败的点

## 错误处理

- 如果某个失败点没有对应的人口数据，会跳过该点
- 如果重新处理仍然失败，会记录到新的失败点文件
- 所有错误都会记录到日志文件中

## 使用方法

1. 确保失败点CSV文件存在于指定路径
2. 确保人口数据文件存在
3. 确保对应年份的气候数据目录存在
4. 运行脚本：`python 2_c_DD_pop copy.py`

## 预期结果

- 成功处理的点会生成完整的BAIT和能耗数据文件
- 仍然失败的点会记录到 `still_failed_energy_points_{year}.csv` 文件中
- 详细的处理日志会记录在 `grid_point_calculation.log` 文件中

## 注意事项

1. 本工具只处理指定的年份（2017, 2018, 2020）
2. 坐标精度保持三位小数
3. 使用与原始计算相同的参数和算法
4. 支持优雅退出（Ctrl+C）
5. 具有完善的错误处理和日志记录
