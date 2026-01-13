import pandas as pd
import os
import sys

def check_parquet_dependencies():
    """检查parquet读取所需的依赖"""
    try:
        import pyarrow
        return True, 'pyarrow'
    except ImportError:
        try:
            import fastparquet
            return True, 'fastparquet'
        except ImportError:
            return False, None

def install_pyarrow():
    """尝试安装pyarrow"""
    try:
        import subprocess
        print("正在尝试安装 pyarrow...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
        print("pyarrow 安装成功！")
        return True
    except Exception as e:
        print(f"自动安装失败: {e}")
        return False

def extract_point_data_from_parquet(parquet_path):
    """
    从parquet文件中读取任意一点的数据并输出为CSV
    
    参数:
        parquet_path: parquet文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(parquet_path):
        print(f"错误: 文件不存在 - {parquet_path}")
        return
    
    # 检查parquet依赖
    has_dependency, engine = check_parquet_dependencies()
    if not has_dependency:
        print("=" * 60)
        print("错误: 缺少读取parquet文件所需的依赖库")
        print("=" * 60)
        print("请安装以下任一库:")
        print("  方法1: pip install pyarrow")
        print("  方法2: pip install fastparquet")
        print("=" * 60)
        print("正在尝试自动安装 pyarrow...")
        if not install_pyarrow():
            print("\n自动安装失败，请手动执行以下命令安装:")
            print("  pip install pyarrow")
            return
        # 重新检查
        has_dependency, engine = check_parquet_dependencies()
        if not has_dependency:
            print("安装后仍无法导入，请重启Python环境后重试")
            return
    
    print(f"正在读取文件: {parquet_path}")
    print(f"使用引擎: {engine}")
    
    # 读取parquet文件
    try:
        # 明确指定引擎
        df = pd.read_parquet(parquet_path, engine=engine)
        print(f"成功读取文件，共 {len(df)} 行数据")
        print(f"列名: {list(df.columns)}")
        
        # 如果数据为空，提示并返回
        if len(df) == 0:
            print("警告: 文件中没有数据")
            return
        
        # 提取第一行数据（任意一点）
        point_data = df.iloc[0:1].copy()  # 使用iloc[0:1]保持DataFrame格式
        
        # 获取文件所在目录
        file_dir = os.path.dirname(parquet_path)
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(parquet_path))[0]
        output_filename = f"{base_name}_sample_point.csv"
        output_path = os.path.join(file_dir, output_filename)
        
        # 保存为CSV
        point_data.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"已保存数据到: {output_path}")
        print(f"提取的数据点信息:")
        print(point_data.to_string())
        
    except ImportError as e:
        print(f"错误: 无法导入parquet引擎 - {e}")
        print("请尝试安装: pip install pyarrow")
    except Exception as e:
        print(f"错误: 读取或处理文件时出错 - {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    # 指定parquet文件路径
    parquet_path = r"Z:\local_environment_creation\energy_consumption_gird\result\result_half_parquet\2016\block_lat=-10-0_block_lon=-100--90.parquet"
    
    # 提取数据
    extract_point_data_from_parquet(parquet_path)


if __name__ == '__main__':
    main()

