import xarray as xr
import os
import glob
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_nc_file(input_file, output_dir):
    """处理单个NC文件，提取2030年数据并保存"""
    try:
        # 获取文件名
        file_name = os.path.basename(input_file)
        
        # 创建新的文件名（替换日期部分）
        new_file_name = file_name.replace('20150101-20391231', '20300101-20301231')
        new_file_name = new_file_name.replace('20150101-20641231', '20300101-20301231')
        
        # 构建输出文件路径
        output_file = os.path.join(output_dir, new_file_name)
        
        logger.info(f"处理文件: {file_name}")
        
        # 读取数据
        with xr.open_dataset(input_file) as ds:
            # 筛选2030年的数据
            ds_2030 = ds.sel(time=slice('2030-01-01', '2030-12-31'))
            
            # 保存到新文件
            ds_2030.to_netcdf(output_file)
            
        logger.info(f"已保存到: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"处理文件 {input_file} 时出错: {str(e)}")
        return False

def main():
    # 定义输入和输出路径
    input_dir = r"Z:\CMIP6\future\SSP1"
    output_dir = r"Z:\CMIP6\future\SSP126_2030"
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 要处理的文件列表
    file_patterns = [
        "huss_day_CNRM-CM6-1-HR_ssp126_r1i1p1f2_gr_*.nc",
        "rsds_day_CNRM-CM6-1-HR_ssp126_r1i1p1f2_gr_*.nc",
        "tas_day_CNRM-CM6-1-HR_ssp126_r1i1p1f2_gr_*.nc",
        "tasmax_day_CNRM-CM6-1-HR_ssp126_r1i1p1f2_gr_*.nc",
        "uas_day_CNRM-CM6-1-HR_ssp126_r1i1p1f2_gr_*.nc",
        "vas_day_CNRM-CM6-1-HR_ssp126_r1i1p1f2_gr_*.nc"
    ]
    
    # 获取所有匹配的文件
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(os.path.join(input_dir, pattern)))
    
    logger.info(f"找到 {len(all_files)} 个文件需要处理")
    
    # 处理每个文件
    success_count = 0
    for file_path in tqdm(all_files, desc="处理文件"):
        if process_nc_file(file_path, output_dir):
            success_count += 1
    
    logger.info(f"处理完成: 成功处理 {success_count}/{len(all_files)} 个文件")

if __name__ == "__main__":
    main()
