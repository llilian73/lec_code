import os

import numpy as np
import pandas as pd

from demand_ninja.util import smooth_temperature, get_cdd, get_hdd

DIURNAL_PROFILES = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "diurnal_profiles.csv"), index_col=0
)


def _bait(
        weather: pd.DataFrame,
        smoothing: float,
        solar_gains: float,
        wind_chill: float,
        humidity_discomfort: float,
) -> pd.Series:
    # We compute 'setpoint' values for wind, sun and humidity
    # these are the 'average' values for the given temperature
    # and are used to decide if it is windier than average,
    # sunnier than aveage, etc.  this makes N correlate roughly
    # 1:1 with T, rather than is biased above or below it.
    T = weather["temperature"]
    setpoint_S = 100 + 7 * T  # W/m2
    setpoint_W = 4.5 - 0.025 * T  # m/s
    setpoint_H = (1.1 + 0.06 * T).rpow(np.e)  # g water per kg air
    setpoint_T = 16  # degrees - around which 'discomfort' is measured

    # Calculate the unsmoothed ninja temperature index
    # this is a 'feels like' index - how warm does it 'feel' to your building

    # Initialise it to temperature
    N = weather["temperature"].copy()

    # If it's sunny, it feels warmer
    N = N + (weather["radiation_global_horizontal"] - setpoint_S) * solar_gains

    # If it's windy, it feels colder
    N = N + (weather["wind_speed_2m"] - setpoint_W) * wind_chill

    # If it's humid, both hot and cold feel more extreme
    discomfort = N - setpoint_T
    N = (
            setpoint_T
            + discomfort
            + (
                    discomfort
                    # Convert humidity from g/kg to kg/kg
                    * ((weather["humidity"] / 1000) - setpoint_H)
                    * humidity_discomfort
            )
    )

    # Apply temporal smoothing to our temperatures over the last two days
    # we assume 2nd day smoothing is the square of the first day (i.e. compounded decay)
    N = smooth_temperature(N, weights=[smoothing, smoothing ** 2])

    # Blend the smoothed BAIT with raw temperatures to account for occupant
    # behaviour changing with the weather (i.e. people open windows when it's hot)

    # These are fixed parameters we don't expose the user to
    lower_blend = 15  # *C at which we start blending T into N
    upper_blend = 23  # *C at which we have fully blended T into N
    max_raw_var = 0.5  # maximum amount of T that gets blended into N

    # Transform this window to a sigmoid function, mapping lower & upper onto -5 and +5
    avg_blend = (lower_blend + upper_blend) / 2
    dif_blend = upper_blend - lower_blend
    blend = (weather["temperature"] - avg_blend) * 10 / dif_blend
    blend = max_raw_var / (1 + (-blend).rpow(np.e))

    # Apply the blend
    N = (weather["temperature"] * blend) + (N * (1 - blend))

    return N


def _energy_demand_from_bait_p(
        bait: pd.Series,
        heating_threshold_background: float,
        heating_threshold_people: float,
        cooling_threshold_background: float,
        cooling_threshold_people: float,
        p_ls: float,
        base_power: float,
        heating_power: float,
        cooling_power: float,
        population: float,
        use_diurnal_profile: bool,
) -> pd.DataFrame:
    """
    Convert temperatures into energy demand.
    
    Parameters:
        bait: Temperature time series
        heating_threshold_background: Background heating threshold temperature
        heating_threshold_people: People heating threshold temperature
        cooling_threshold_background: Background cooling threshold temperature
        cooling_threshold_people: People cooling threshold temperature
        p_ls: Load sharing parameter
        base_power: Base power demand
        heating_power: Heating power coefficient (W/(℃*p))
        cooling_power: Cooling power coefficient (W/(℃*p))
        population: Number of people
        use_diurnal_profile: Whether to apply diurnal profiles
    """
    output = pd.DataFrame(index=bait.index.copy())
    output["hdd"] = 0
    output["cdd"] = 0
    output["heating_demand"] = 0
    output["cooling_demand"] = 0
    output["total_demand"] = 0

    # Add demand for heating
    if heating_power > 0:
        output["hdd"] = get_hdd(bait, heating_threshold_background, 
                               heating_threshold_people, p_ls)
        # 将W转换为GW，所以除以1e9
        output["heating_demand"] = output["hdd"] * heating_power * population / 1e9

    # Add demand for cooling
    if cooling_power > 0:
        output["cdd"] = get_cdd(bait, cooling_threshold_background, 
                               cooling_threshold_people, p_ls)
        # 将W转换为GW，所以除以1e9
        output["cooling_demand"] = output["cdd"] * cooling_power * population / 1e9

    # Apply the diurnal profiles if wanted
    if use_diurnal_profile:
        hours = output.index.hour
        profiles = DIURNAL_PROFILES.loc[hours, :]
        profiles.index = output.index

        output["heating_demand"] = output["heating_demand"] * profiles.loc[:, "heating"]
        output["cooling_demand"] = output["cooling_demand"] * profiles.loc[:, "cooling"]

    # Sum total demand
    output["total_demand"] = (
            base_power + output["heating_demand"] + output["cooling_demand"]
    )

    return output


def demand_p(
        daily_bait: pd.Series,
        heating_threshold_background: float,
        heating_threshold_people: float,
        cooling_threshold_background: float,
        cooling_threshold_people: float,
        p_ls: float = 0.5,
        base_power: float = 0,
        heating_power: float = 0.3,
        cooling_power: float = 0.15,
        population: float = 1000000,
        use_diurnal_profile: bool = True,
        raw: bool = False,
) -> pd.DataFrame:

    # # 检查输入是否为每日数据
    # if not isinstance(daily_bait.index, pd.DatetimeIndex):
    #     raise ValueError("daily_bait 的索引必须是 DatetimeIndex。")
    # if daily_bait.index.freq != "D":
    #     raise ValueError("daily_bait 必须是每日数据（频率为 'D'）。")
    # 将每日 BAIT 插值为每小时数据

    # hourly_index = pd.date_range(
    #     start=daily_bait.index[0],
    #     end=daily_bait.index[-1] + pd.Timedelta("23H"),
    #     freq="1H",
    # )
    # hourly_bait = daily_bait.reindex(hourly_index).interpolate(
    #     method="cubicspline", limit_direction="both"
    # )
    # 更改 daily_bait 的时间索引
    daily_bait.index = pd.date_range(
        daily_bait.index[0] + pd.Timedelta("12h"),
        daily_bait.index[-1] + pd.Timedelta("12h"),
        freq="1D",
    )

    # 新建 hourly_bait，时间索引从开头日期的 0 点到结尾日期的 23 点，逐小时递增
    hourly_bait = pd.Series(
        index=pd.date_range(
            start=daily_bait.index[0].replace(hour=0, minute=0, second=0),
            end=daily_bait.index[-1].replace(hour=23, minute=59, second=59),
            freq="1h",
        ),
        dtype=float,
    )

    # 将 daily_bait 和 hourly_bait 对齐并进行样条插值
    hourly_bait = daily_bait.reindex(hourly_bait.index).interpolate(
        method="cubicspline", limit_direction="both"
    )

    # 计算能源需求
    result = _energy_demand_from_bait_p(
        hourly_bait,
        heating_threshold_background,
        heating_threshold_people,
        cooling_threshold_background,
        cooling_threshold_people,
        p_ls,
        base_power,
        heating_power,
        cooling_power,
        population,
        use_diurnal_profile,
    )

    # 修改返回的结果，包含hdd和cdd
    result = result.loc[:, ["total_demand", "heating_demand", "cooling_demand", "hdd", "cdd"]]

    if raw:
        result = pd.concat((result, hourly_bait.rename("bait")), axis=1)

    return result