def calculate_cases(base_params):
    """计算所有工况（20个case）的参数
    
    Args:
        base_params: 包含基本参数的字典，必须包含以下键：
            - heating_threshold_people
            - cooling_threshold_people
            - heating_power
            - cooling_power
            - base_power
            - population
            
    Returns:
        dict: 包含所有工况参数的字典
        
    工况说明：
    - ref: 参考案例
    - case1-5: diff=1℃，p_ls=[0.5, 0.25, 0.125, 0.0625, 0.03125]
    - case6-10: diff=2℃，p_ls=[0.5, 0.25, 0.125, 0.0625, 0.03125]
    - case11-15: diff=3℃，p_ls=[0.5, 0.25, 0.125, 0.0625, 0.03125]
    - case16-20: diff=4℃，p_ls=[0.5, 0.25, 0.125, 0.0625, 0.03125]
    """
    def create_case_params(base_params, diff_temp, p_ls):
        """创建单个工况的参数"""
        return {
            "heating_threshold_background": base_params["heating_threshold_people"] - diff_temp,
            "heating_threshold_people": base_params["heating_threshold_people"],
            "cooling_threshold_background": base_params["cooling_threshold_people"] + diff_temp,
            "cooling_threshold_people": base_params["cooling_threshold_people"],
            "p_ls": p_ls,
            "base_power": base_params["base_power"],
            "heating_power": base_params["heating_power"],
            "cooling_power": base_params["cooling_power"],
            "population": base_params["population"]
        }

    # 定义所有工况
    cases = {
        "ref": {
            "heating_threshold_background": base_params["heating_threshold_people"],
            "heating_threshold_people": base_params["heating_threshold_people"],
            "cooling_threshold_background": base_params["cooling_threshold_people"],
            "cooling_threshold_people": base_params["cooling_threshold_people"],
            "p_ls": 0,
            **base_params
        }
    }

    # Case 1-5: diff=1℃，p_ls=[0.5, 0.25, 0.125, 0.0625, 0.03125]
    for i, p_ls in enumerate([0.5, 0.25, 0.125, 0.0625, 0.03125], 1):
        cases[f"case{i}"] = create_case_params(base_params, 1, p_ls)

    # Case 6-10: diff=2℃，p_ls=[0.5, 0.25, 0.125, 0.0625, 0.03125]
    for i, p_ls in enumerate([0.5, 0.25, 0.125, 0.0625, 0.03125], 6):
        cases[f"case{i}"] = create_case_params(base_params, 2, p_ls)

    # Case 11-15: diff=3℃，p_ls=[0.5, 0.25, 0.125, 0.0625, 0.03125]
    for i, p_ls in enumerate([0.5, 0.25, 0.125, 0.0625, 0.03125], 11):
        cases[f"case{i}"] = create_case_params(base_params, 3, p_ls)

    # Case 16-20: diff=4℃，p_ls=[0.5, 0.25, 0.125, 0.0625, 0.03125]
    for i, p_ls in enumerate([0.5, 0.25, 0.125, 0.0625, 0.03125], 16):
        cases[f"case{i}"] = create_case_params(base_params, 4, p_ls)

    return cases