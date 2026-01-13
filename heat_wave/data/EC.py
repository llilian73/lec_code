"""
EC-Earth3历史数据整合工具

功能：
整合EC-Earth3历史数据（一年一个文件）为单个NetCDF文件
"""

import xarray as xr
import numpy as np
import os
import pandas as pd
import re
import multiprocessing
from tqdm import tqdm
import gc

# ============================================================================
# EC-Earth3历史数据整合配置
# ============================================================================
EC_EARTH_HISTORICAL_INPUT_DIR = r"/home/linbor/WORK/lishiying/GCM_input_processed/EC-Earth3/historical"
EC_EARTH_HISTORICAL_OUTPUT_DIR = r"/home/linbor/WORK/lishiying/GCM_input_processed/EC-Earth3/historical"
EC_EARTH_NUM_WORKERS = 30  # 使用30个进程并行加载
COMPRESSION_LEVEL = 4  # NetCDF压缩级别 (1-9, 4是平衡速度和大小)


def check_missing_years(time_values, start_year, end_year):
    """检查缺少年份"""
    # 提取所有年份
    years = set()
    for t in time_values:
        if hasattr(t, 'year'):
            years.add(t.year)
        elif isinstance(t, pd.Timestamp):
            years.add(t.year)
        else:
            try:
                years.add(pd.Timestamp(t).year)
            except:
                pass
    
    # 计算期望的年份
    expected_years = set(range(start_year, end_year + 1))
    missing_years = sorted(expected_years - years)
    
    return missing_years, years


def load_single_year_file(filepath):
    """加载单个年份文件（用于并行处理）"""
    try:
        # 从文件名提取年份（更稳健的方法）
        filename = os.path.basename(filepath)
        # 查找文件名中的年份模式（4位数字，如1981）
        year_match = re.search(r'(\d{4})0101', filename)
        if year_match:
            year = int(year_match.group(1))
        else:
            # 如果找不到，尝试从文件名分割
            parts = filename.split('_')
            if len(parts) >= 7:
                year = int(parts[6][:4])
            else:
                raise ValueError(f"无法从文件名提取年份: {filename}")
        
        # 尝试使用netcdf4引擎打开文件
        try:
            ds = xr.open_dataset(filepath, engine='netcdf4')
        except:
            try:
                ds = xr.open_dataset(filepath, engine='h5netcdf')
            except:
                ds = xr.open_dataset(filepath)
        
        # 加载数据到内存
        ds = ds.load()
        
        return {
            'success': True,
            'data': ds,
            'filename': filename,
            'year': year
        }
    except Exception as e:
        return {
            'success': False,
            'filename': os.path.basename(filepath),
            'message': f'加载文件失败: {str(e)}'
        }


def integrate_ec_earth3_historical_data(input_dir, output_dir, start_year=1981, end_year=2010):
    """
    整合EC-Earth3历史数据（一年一个文件）
    
    参数:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        start_year: 起始年份
        end_year: 结束年份
    """
    print(f"\n整合EC-Earth3历史数据 ({start_year}-{end_year})")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    # 查找所有年份的文件
    file_list = []
    for year in range(start_year, end_year + 1):
        # 构建文件名模式
        filename_pattern = f"tasmax_day_EC-Earth3-HR_historical_r1i1p1f1_gr_{year}0101-{year}1231_interpolated_1deg.nc"
        filepath = os.path.join(input_dir, filename_pattern)
        
        if os.path.exists(filepath):
            file_list.append((filepath, year))
        else:
            print(f"⚠️  警告: 未找到 {year} 年的文件: {filename_pattern}")
    
    if len(file_list) == 0:
        print(f"❌ 未找到任何数据文件")
        return
    
    print(f"找到 {len(file_list)} 个文件（期望 {end_year - start_year + 1} 个）")
    
    # 按年份排序
    file_list.sort(key=lambda x: x[1])
    
    # 使用30个进程并行加载数据
    print(f"使用 {EC_EARTH_NUM_WORKERS} 个进程并行加载数据...")
    datasets = []
    
    with multiprocessing.Pool(processes=EC_EARTH_NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(load_single_year_file, [f[0] for f in file_list]),
            total=len(file_list),
            desc="加载文件"
        ))
    
    # 收集成功的结果，并按年份排序
    successful_results = []
    for result in results:
        if result['success']:
            successful_results.append(result)
        else:
            print(f"  ✗ {result['filename']}: {result['message']}")
    
    # 按年份排序
    successful_results.sort(key=lambda x: x['year'])
    
    if len(successful_results) == 0:
        print("❌ 没有成功加载的数据")
        return
    
    print(f"成功加载 {len(successful_results)} 个文件")
    
    # 提取数据集
    datasets = [r['data'] for r in successful_results]
    
    # 整合数据集
    print(f"整合 {len(datasets)} 个数据集...")
    try:
        # 使用分批合并策略，避免内存溢出
        batch_size = 10  # 每批合并10个数据集
        
        if len(datasets) > batch_size:
            print(f"使用分批合并策略（每批 {batch_size} 个数据集）...")
            
            # 分批合并
            current_combined = None
            for i in range(0, len(datasets), batch_size):
                batch = datasets[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(datasets) + batch_size - 1) // batch_size
                
                print(f"  合并批次 {batch_num}/{total_batches} ({len(batch)} 个数据集)...")
                
                # 合并当前批次
                if len(batch) == 1:
                    batch_combined = batch[0]
                else:
                    try:
                        batch_combined = xr.combine_by_coords(batch, combine_attrs='drop_conflicts')
                        if batch_combined is None or len(batch_combined.time) == 0:
                            batch_combined = xr.concat(batch, dim='time')
                            batch_combined = batch_combined.sortby('time')
                    except:
                        batch_combined = xr.concat(batch, dim='time')
                        batch_combined = batch_combined.sortby('time')
                
                # 释放批次数据集的内存
                for ds in batch:
                    try:
                        ds.close()
                    except:
                        pass
                del batch
                gc.collect()
                
                # 与之前合并的结果合并
                if current_combined is None:
                    current_combined = batch_combined
                else:
                    try:
                        current_combined = xr.combine_by_coords(
                            [current_combined, batch_combined], 
                            combine_attrs='drop_conflicts'
                        )
                        if current_combined is None or len(current_combined.time) == 0:
                            current_combined = xr.concat([current_combined, batch_combined], dim='time')
                            current_combined = current_combined.sortby('time')
                    except:
                        current_combined = xr.concat([current_combined, batch_combined], dim='time')
                        current_combined = current_combined.sortby('time')
                    
                    # 释放批次合并结果的内存
                    try:
                        batch_combined.close()
                    except:
                        pass
                    del batch_combined
                    gc.collect()
            
            combined_ds = current_combined
        else:
            # 数据集数量较少，直接合并
            try:
                combined_ds = xr.combine_by_coords(datasets, combine_attrs='drop_conflicts')
                if combined_ds is None or len(combined_ds.time) == 0:
                    combined_ds = xr.concat(datasets, dim='time')
                    combined_ds = combined_ds.sortby('time')
            except:
                combined_ds = xr.concat(datasets, dim='time')
                combined_ds = combined_ds.sortby('time')
            
            # 释放内存
            for ds in datasets:
                try:
                    ds.close()
                except:
                    pass
        
        # 去除重复的时间点
        print("去除重复时间点...")
        try:
            time_values = combined_ds.time.values
            if hasattr(time_values[0], 'year'):
                # cftime对象，转换为字符串进行比较
                time_strs = [str(t) for t in time_values]
                _, unique_indices = np.unique(time_strs, return_index=True)
            else:
                # pandas Timestamp或其他
                _, unique_indices = np.unique(time_values, return_index=True)
            
            if len(unique_indices) < len(time_values):
                print(f"  发现 {len(time_values) - len(unique_indices)} 个重复时间点")
                combined_ds = combined_ds.isel(time=unique_indices)
        except Exception as e:
            print(f"  警告: 去重失败: {e}")
            try:
                combined_ds = combined_ds.drop_duplicates(dim='time')
            except:
                pass
        
        # 检查缺少年份
        time_values = combined_ds.time.values
        missing_years, existing_years = check_missing_years(time_values, start_year, end_year)
        
        if missing_years:
            print(f"⚠️  缺少年份: {missing_years}")
            print(f"   已有年份: {sorted(existing_years)}")
        else:
            print(f"✓ 所有年份完整 ({start_year}-{end_year})")
        
        print(f"整合后时间范围: {combined_ds.time.values[0]} 到 {combined_ds.time.values[-1]}")
        print(f"整合后数据点数: {len(combined_ds.time)}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        output_filename = f"tasmax_day_EC-Earth3-HR_historical_r1i1p1f1_gr_{start_year}0101-{end_year}1231_interpolated_1deg.nc"
        output_filepath = os.path.join(output_dir, output_filename)
        
        # 保存文件（使用压缩和优化选项）
        print(f"保存到: {output_filepath}")
        encoding = {}
        for var in combined_ds.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': COMPRESSION_LEVEL,
                'shuffle': True
            }
        # 对坐标变量也进行压缩
        for coord in combined_ds.coords:
            if coord != 'time':  # time坐标通常不压缩
                encoding[coord] = {
                    'zlib': True,
                    'complevel': COMPRESSION_LEVEL
                }
        
        combined_ds.to_netcdf(
            output_filepath,
            encoding=encoding,
            format='NETCDF4'
        )
        print(f"✓ 数据已保存\n")
        
        # 释放内存
        del combined_ds
        datasets.clear()
        gc.collect()
        
    except Exception as e:
        print(f"❌ 错误: 整合数据失败: {e}")
        import traceback
        traceback.print_exc()
        # 关闭所有数据集
        for ds in datasets:
            try:
                ds.close()
            except:
                pass


def main():
    """主函数"""
    print("="*80)
    print("EC-Earth3历史数据整合工具")
    print("="*80)
    
    integrate_ec_earth3_historical_data(
        input_dir=EC_EARTH_HISTORICAL_INPUT_DIR,
        output_dir=EC_EARTH_HISTORICAL_OUTPUT_DIR,
        start_year=1981,
        end_year=2010
    )
    
    print("\n" + "="*80)
    print("处理完成!")
    print("="*80)


if __name__ == "__main__":
    main()
