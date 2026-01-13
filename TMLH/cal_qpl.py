import sys
import os
sys.dont_write_bytecode = True  # 防止生成 __pycache__
import math

def calculate_qpl(ta, tr, vel, rh, met, clo, wme):
    """
    Calculate Qpl for PMV in the range -0.5 < PMV < 0.5.
    ta: air temperature (°C)
    tr: mean radiant temperature (°C)
    vel: relative air speed (m/s)
    rh: relative humidity (%)
    met: metabolic rate (met)
    clo: clothing (clo)
    wme: external work, normally around 0 (met)
    Returns: Qpl
    """
    # Calculate water vapor pressure (Pa)
    pa = rh * 10 * math.exp(16.6536 - 4030.183 / (ta + 235))

    # Calculate clothing insulation (M2K/W)
    icl = 0.155 * clo

    # Convert metabolic rate to W/M2
    m = met * 58.15

    # Convert external work to W/M2
    w = wme * 58.15

    # Internal heat production in the human body
    mw = m - w

    # Calculate clothing area factor
    if icl <= 0.078:
        fcl = 1 + (1.29 * icl)
    else:
        fcl = 1.05 + (0.645 * icl)

    # Heat transfer coefficient by forced convection
    hcf = 12.1 * math.sqrt(vel)

    # Convert temperatures to Kelvin
    taa = ta + 273
    tra = tr + 273

    # Calculate clothing surface temperature (initial guess)
    tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)

    # Intermediate calculations for heat balance
    pl = icl * fcl
    p2 = pl * 3.96
    p3 = pl * 100
    p4 = pl * taa
    p5 = 308.7 - 0.028 * mw + p2 * math.pow(tra / 100, 4)

    # Initialize variables for iteration
    xn = tcla / 100
    xf = tcla / 50
    eps = 0.00015
    n = 0

    # Iterative calculation of clothing surface temperature
    while abs(xn - xf) > eps:
        xf = (xf + xn) / 2
        hcn = 2.38 * math.pow(abs(100.0 * xf - taa), 0.25)
        if hcf > hcn:
            hc = hcf
        else:
            hc = hcn
        xn = (p5 + p4 * hc - p2 * math.pow(xf, 4)) / (100 + p3 * hc)
        n += 1
        if n > 150:
            raise Exception('Max iterations exceeded')

    # Calculate clothing surface temperature in °C
    tcl = 100 * xn - 273

    # Heat loss difference through skin
    h11 = 3.05 * 0.001 * (5733 - (6.99 * mw) - pa)

    # Heat loss by sweating
    if mw > 58.15:
        h12 = 0.42 * (mw - 58.15)
    else:
        h12 = 0

    # Latent respiration heat loss
    h13 = 1.7 * 0.00001 * m * (5867 - pa)

    # Dry respiration heat loss
    h14 = 0.0014 * m * (34 - ta)

    # Heat loss by radiation
    h15 = 3.96 * fcl * (math.pow(xn, 4) - math.pow(tra / 100, 4))

    # Heat loss by convection
    h16 = fcl * hc * (tcl - ta)

    # Thermal sensation coefficient
    ts = 0.303 * math.exp(-0.036 * m) + 0.028

    # Initialize Qpl
    Qpl = 0

    # Calculate PMV with Qpl
    pmv = ts * (mw - h11 - h12 - h13 - h14 - h15 - h16 + Qpl)

    # Adjust Qpl based on PMV range
    if pmv < -0.5:
        Qpl = (-0.5 / ts) - (mw - h11 - h12 - h13 - h14 - h15 - h16)#Qql>0，加热
    elif pmv > 0.5:
        Qpl = (0.5 / ts) - (mw - h11 - h12 - h13 - h14 - h15 - h16)#Qql<0，冷却
    elif -0.5 < pmv < 0.5:
        Qpl = 0  # No adjustment needed

    # Return Qpl
    return Qpl


