# `energy_last_for_all_hw.py` 代码说明

## 一、功能概述

该代码用于计算**热浪事件期间**的建筑物能源消耗（制冷和制热需求）。它读取热浪识别结果，结合气候数据和人口数据，使用BAIT（Building Adaptive Indoor Temperature）模型计算不同情景（case）下的能耗。

## 二、输入数据

### 1. 热浪事件CSV文件
- **路径格式**: `/home/linbor/WORK/lishiying/heat_wave/{模型名}/{SSP路径}/{年份}_all_heat_wave.csv`
- **示例**: `/home/linbor/WORK/lishiying/heat_wave/BCC-CSM2-MR/SSP126/2030_all_heat_wave.csv`
- **必需列**:
  - `lat`: 纬度
  - `lon`: 经度
  - `date`: 热浪开始日期（格式：`month/day`，如 `7/15`）
  - `Duration`: 热浪持续时间（天数，>=3天）
- **说明**: 由 `identify_heatwave_lin.py` 生成，包含所有识别出的热浪事件

### 2. 未来气候数据（NetCDF格式）
- **路径格式**: `/home/linbor/WORK/lishiying/GCM_input_processed/{模型名}/future/{SSP路径}/`
- **文件命名**: `{变量名}_day_{模型名}_{SSP}_*_*_interpolated_1deg.nc`
- **示例**: `tas_day_BCC-CSM2-MR_ssp126_r1i1p1f1_gn_20300101-20341231_interpolated_1deg.nc`
- **必需变量**:
  - `tas`: 近地表气温（K，会转换为℃）
  - `rsds`: 地面短波辐射（W/m²）
  - `huss`: 比湿（kg/kg，会转换为g/kg）
  - `uas`: 10米高度东西向风速（m/s）
  - `vas`: 10米高度南北向风速（m/s）
- **时间范围**: 包含目标年份的数据（2030-2034年）
- **空间分辨率**: 1度网格（interpolated_1deg）

### 3. 人口数据（CSV格式）
- **路径格式**: `/home/linbor/WORK/lishiying/population/{SSP}_population.csv`
- **文件**:
  - `SSP126` → `SSP1_population.csv`
  - `SSP245` → `SSP2_population.csv`
- **必需列**:
  - `lat`: 纬度
  - `lon`: 经度
  - `population`: 人口数量
- **说明**: 不同SSP情景对应不同的人口数据文件

## 三、输出数据

### 输出路径
- **格式**: `/home/linbor/WORK/lishiying/heat_wave/{模型名}/{SSP路径}/energy/{年份}/{case_name}.csv`
- **示例**: `/home/linbor/WORK/lishiying/heat_wave/BCC-CSM2-MR/SSP126/energy/2030/ref.csv`

### 输出文件格式
每个case生成一个CSV文件，包含以下列：
- `lat`: 纬度
- `lon`: 经度
- `total_demand`: 总能耗（GW，该点所有热浪事件的总和）
- `heating_demand`: 制热能耗（GW）
- `cooling_demand`: 制冷能耗（GW）

### Case说明
代码计算21个case的能耗（ref + case1-case20）：
- **ref**: 基准情景（reference case）
- **case1-case20**: 不同的参数组合，通过 `CandD.calculate.calculate_cases()` 生成

每个case的参数差异主要在：
- `heating_threshold_background`: 制热背景阈值（℃）
- `heating_threshold_people`: 制热人员阈值（℃）
- `cooling_threshold_background`: 制冷背景阈值（℃）
- `cooling_threshold_people`: 制冷人员阈值（℃）
- `p_ls`: 负荷分担参数
- `heating_power`: 制热功率系数（W/(℃*人)）
- `cooling_power`: 制冷功率系数（W/(℃*人)）

## 四、处理逻辑

### 整体流程

```
主函数 main()
  ├─ 遍历所有模型（6个）
  │   ├─ 遍历所有SSP路径（SSP126, SSP245）
  │   │   ├─ 遍历所有目标年份（2030-2034）
  │   │   │   └─ process_single_year() 处理单年数据
```

### 核心处理步骤（`process_single_year`）

#### 1. 数据加载阶段
```python
# a) 读取热浪事件文件
points_df = pd.read_csv(heat_wave_file)
points_df = points_df[points_df['Duration'] >= 3]  # 过滤：只保留>=3天的热浪

# b) 按经纬度分组，合并同一地点的多个热浪事件
grouped_points = []
for (lat, lon), group in points_df.groupby(['lat', 'lon']):
    point_data = {
        'lat': lat,
        'lon': lon,
        'heat_waves': group.to_dict('records')  # 该点所有热浪事件列表
    }

# c) 读取人口数据（根据SSP选择对应文件）
population_df = pd.read_csv(population_file)

# d) 加载气候数据到共享内存
shared_data = SharedClimateData(future_dir, model_name, ssp_path, year)
```

#### 2. 气候数据加载（`SharedClimateData.load_year_data`）

**关键步骤**：
- 查找所有5个气候变量文件
- 打开第一个文件（通常用`tas`）获取时间和坐标信息
- **处理cftime对象**：NetCDF中的时间可能是`cftime.DatetimeNoLeap`等非标准格式，需要转换为`pandas.DatetimeIndex`
- **筛选年份数据**：从完整时间序列中提取目标年份的数据（如2030年）
- **加载到共享内存**：将各变量的年份数据加载到NumPy数组，并创建`shared_memory.SharedMemory`对象，以便多进程共享

**共享内存的优势**：
- 避免每个进程重复加载大型NetCDF文件
- 多个进程可以共享同一份数据，节省内存
- 提高并行处理效率

#### 3. 并行处理阶段

**数据分片**：
- 将所有点分成批次（`batch_size`）
- 使用`multiprocessing.Pool`创建进程池（默认31个进程）

**每个进程处理一个批次**（`process_batch`）：
```python
for point_data in batch_points:
    result = process_point(point_data, shared_data, population_df, output_dir)
```

#### 4. 单点处理（`process_point`）

对每个经纬度点，执行以下步骤：

**a) 获取人口数据**
```python
pop_data = population_df[(population_df['lat'] == lat) & (population_df['lon'] == lon)]
population = pop_data['population'].iloc[0]
```

**b) 遍历该点的所有热浪事件**

对每个热浪事件：
1. **解析热浪日期**：从`date`（如`"7/15"`）解析为完整日期（如`2030-07-15`）
2. **加载气候数据**（`load_climate_data`）：
   - 根据热浪开始日期和持续时间，计算时间范围
   - 调用`shared_data.get_data(lat, lon, start_date, end_date)`提取该时间段的气候数据
   - **时间对齐处理**：确保时间索引为00:00:00（日期精度），因为NC数据实际是12点的，后续`demand_p`会将0点转为12点
3. **计算BAIT**（`_bait`）：
   ```python
   bait = _bait(
       weather=weather_df,  # 包含temperature, radiation_global_horizontal, wind_speed_2m, humidity
       smoothing=0.73,
       solar_gains=0.014,
       wind_chill=-0.12,
       humidity_discomfort=0.036
   )
   ```
   - **BAIT计算原理**：
     - 基于气温、辐射、风速、湿度的综合指数
     - 考虑建筑物的"体感温度"（feels like）
     - 应用时间平滑（2天加权平均）
     - 混合原始温度和平滑温度（根据温度范围应用sigmoid混合）
4. **计算每个case的能耗**（`calculate_energy_demand`）：
   ```python
   for case_name, params in calculate_cases(base_params).items():
       daily_bait = bait.copy()
       # 确保时间索引为0点
       daily_bait.index = pd.date_range(start_date, end_date, freq='D')
       
       results[case_name] = demand_p(
           daily_bait,
           heating_threshold_background=params["heating_threshold_background"],
           heating_threshold_people=params["heating_threshold_people"],
           cooling_threshold_background=params["cooling_threshold_background"],
           cooling_threshold_people=params["cooling_threshold_people"],
           p_ls=params["p_ls"],
           base_power=params["base_power"],
           heating_power=params["heating_power"],
           cooling_power=params["cooling_power"],
           population=population,
           use_diurnal_profile=True
       )
   ```
   - **`demand_p`内部处理**：
     1. 将`daily_bait`的时间索引从0点转为12点（+12小时）
     2. 通过三次样条插值将日数据转为小时数据
     3. 计算HDD（Heating Degree Days）和CDD（Cooling Degree Days）
     4. 根据阈值计算制热/制冷需求（W），并转换为GW
     5. 应用日周期曲线（diurnal profile）调整小时需求

**c) 累加所有热浪事件的能耗**
```python
for case_name, case_result in results.items():
    total_results[case_name]['total_demand'] += case_result['total_demand'].sum()
    total_results[case_name]['heating_demand'] += case_result['heating_demand'].sum()
    total_results[case_name]['cooling_demand'] += case_result['cooling_demand'].sum()
```

**d) 保存结果**
- 为每个case创建单独的CSV文件
- 追加该点的结果（lat, lon, total_demand, heating_demand, cooling_demand）

### 关键技术细节

#### 1. 时间处理的一致性
- **问题**：NetCDF文件的时间可能是`cftime`对象，且不同模型使用的日历不同
- **解决**：`convert_cftime_to_datetime()`函数采用多种方法尝试转换
- **关键**：确保`start_date`、`end_date`和`time_index`都规范化为日期精度（00:00:00），避免因时间精度不匹配导致的第一天数据丢失

#### 2. 时间索引对齐（NC数据12点 vs demand_p期望0点）
- **问题**：从NC文件提取的数据实际是12:00:00的，但`demand_p`函数内部会将输入的时间索引+12小时（期望输入是0点的）
- **解决**：在`load_climate_data`和`calculate_energy_demand`中，确保传入`demand_p`的`daily_bait.index`为00:00:00
- **原理**：这样`demand_p`的+12小时操作才能正确得到12:00:00的时间点

#### 3. 空间索引查找
- **方法**：使用`np.abs(lat_index - lat).argmin()`找到最近的网格点
- **注意**：确保用于`argmin`的索引数组与实际数据数组的维度匹配

#### 4. 共享内存的传递
- **主进程**：创建`SharedMemory`对象，将数据加载到共享内存
- **子进程**：通过共享内存名称（`shm.name`）重建对同一块内存的访问
- **清理**：处理完成后，主进程调用`cleanup()`释放共享内存资源

## 五、代码配置

### 模型列表
```python
MODELS = [
    "ACCESS-ESM1-5",
    "BCC-CSM2-MR",
    "CanESM5",
    "EC-Earth3",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0"
]
```

### SSP路径
```python
SSP_PATHS = ["SSP126", "SSP245"]
```

### 目标年份
```python
TARGET_YEARS = [2030, 2031, 2032, 2033, 2034]
```

### 并行处理参数
```python
NUM_PROCESSES = 31  # 进程池大小
```

### BAIT参数
```python
smoothing = 0.73
solar_gains = 0.014
wind_chill = -0.12
humidity_discomfort = 0.036
```

### 能耗计算基础参数
```python
base_params = {
    "heating_power": 27.93,      # W/(℃*人)
    "cooling_power": 48.55,      # W/(℃*人)
    "heating_threshold_people": 14,   # ℃
    "cooling_threshold_people": 20,   # ℃
    "base_power": 0,
    "population": population  # 从人口数据读取
}
```

## 六、运行示例

```bash
python code/heat_wave/data/energy_last_for_all_hw.py
```

**输出日志示例**：
```
2025-12-06 20:00:00 - INFO - === 开始热浪期间能耗计算 ===
2025-12-06 20:00:00 - INFO - 支持的模型: ACCESS-ESM1-5, BCC-CSM2-MR, ...
2025-12-06 20:00:00 - INFO - 处理模型: BCC-CSM2-MR
2025-12-06 20:00:01 - INFO - 正在加载 BCC-CSM2-MR SSP126 2030 年的气候数据...
2025-12-06 20:00:15 - INFO - 气候数据加载完成，耗时 14.23 秒
2025-12-06 20:00:15 - INFO - 共 1250 个点需要处理
2025-12-06 20:00:15 - INFO - 数据分配: 总点数=1250, 批次大小=5, 批次数=250
2025-12-06 20:00:15 - INFO - 处理 BCC-CSM2-MR-SSP126-2030: 100%|████████| 250/250
2025-12-06 20:30:00 - INFO - ✓ BCC-CSM2-MR - SSP126 - 2030 年处理完成，成功处理 1250 个点
```

## 七、依赖库

- `pandas`: 数据处理和CSV读写
- `numpy`: 数值计算和数组操作
- `xarray`: NetCDF文件读取
- `multiprocessing`: 多进程并行处理
- `demand_ninja.core_p`: BAIT计算和能耗计算
- `CandD.calculate`: Case参数生成
- `cftime`: cftime对象处理（xarray依赖）

## 八、注意事项

1. **内存占用**：共享内存会占用较大内存（5个变量 × 年份数据大小），需要确保有足够内存
2. **文件路径**：所有路径都是Linux格式（`/home/linbor/...`），如果在Windows上运行需要修改
3. **热浪过滤**：只处理持续时间>=3天的热浪事件
4. **时间精度**：确保所有时间比较都使用日期精度（00:00:00），避免时间戳精度不匹配
5. **人口数据匹配**：如果某个点没有对应的人口数据，会跳过该点并记录警告
6. **错误处理**：单个点或单个年份的处理失败不会中断整个流程，会记录错误并继续处理

