import pandas as pd


def smooth_temperature(temperature: pd.Series, weights: list) -> pd.Series:
    """
    Smooth a temperature series over time with the given weighting for previous days.

    Params
    ------

    temperature : pd.Series
    weights : list
        The weights for smoothing. The first element is how much
        yesterday's temperature will be, the 2nd element is 2 days ago, etc.

    """
    assert isinstance(temperature, pd.Series)
    lag = temperature.copy()
    smooth = temperature.copy()

    # Run through each weight in turn going one time-step backwards each time
    for w in weights:
        # Create a time series of temperatures the day before
        lag = lag.shift(1, fill_value=lag.iloc[0])

        # Add on these lagged temperatures multiplied by this smoothing factor
        if w != 0:
            smooth = (smooth + (lag * w)).reindex()

    smooth = smooth.reindex().dropna()

    # Renormalise and return
    return smooth / (1 + sum(weights))


def get_hdd(temperature: list, threshold_background: float, threshold_people: float, p_ls: float) -> list:
    temp_series = pd.Series(temperature)
    HDD_1 = (threshold_background - temp_series).clip(lower=0)
    HDD_2 = (threshold_people - temp_series).clip(lower=0)
    return (HDD_1 * (1 - p_ls) + HDD_2 * p_ls).to_list()


def get_cdd(temperature: list, threshold_background: float, threshold_people: float, p_ls: float) -> list:
    temp_series = pd.Series(temperature)
    CDD_1 = (temp_series - threshold_background).clip(lower=0)
    CDD_2 = (temp_series - threshold_people).clip(lower=0)
    return (CDD_1 * (1 - p_ls) + CDD_2 * p_ls).to_list()
