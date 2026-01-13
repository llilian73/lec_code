def calculate_cases(base_params):
    """计算所有工况（9个case）的参数
    
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

    # Case 1-4
    for i, diff in enumerate([1, 2, 3, 4], 1):
        cases[f"case{i}"] = create_case_params(base_params, diff, 0.03125)

    # Case 5-8
    #for i, p_ls in enumerate([0.5, 0.25, 0.125, 0.0625, 0.03125], 5):
    #    cases[f"case{i}"] = create_case_params(base_params, 2, p_ls)
    for i, diff in enumerate([1, 2, 3, 4], 5):
        cases[f"case{i}"] = create_case_params(base_params, diff, 0.5)

    return cases