#from demand_ninja.core import demand
import os  # 导入操作系统模块

# 检查环境变量 USE_CORE 是否设置为 "NP"
if os.getenv("USE_CORE") == "NP":
    from demand_ninja.core_np import demand_np as demand  # 如果是 "NP"，导入 core_np.py 中的 demand
else:
    from demand_ninja.core_p import demand_p as demand  # 否则，默认导入 core_p.py 中的 demand
