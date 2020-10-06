#!/usr/bin/env python3

import numpy as np
import os
import pandas as pd

################################################################################

def mse(arr, val):
    return np.mean((arr-val)**2)

################################################################################

def percent_err(arr, val):
    return np.mean(100 * np.abs((arr-val)/arr))

################################################################################

def baseline(func, data, low=None, high=None, tolerance=1):
    """
    """
    if low is None:
        low_val  = np.min(data)
        low_mse  = func(data, low_val)
        high_val = np.max(data)
        high_mse = func(data, high_val)
        
        if low_mse < high_mse:
            low  = (low_val, low_mse)
            high = (high_val, high_mse)
        else:
            high = (low_val, low_mse)
            low  = (high_val, high_mse)

    # Compute the MSE for the Half-Way Point 
    half_val = (high[0] + low[0])/2
    half_mse = func(data, half_val)
    half     = (half_val, half_mse)


    if np.abs(high[0] - low[0]) < tolerance:
        return half if half_mse < low[1] else low
    elif half_mse < low[1]:
        return baseline(func, data, half, low, tolerance)
    else: 
        return baseline(func, data, low, half, tolerance)

################################################################################

if __name__ == "__main__":

    # Load Data
    # DATA_DIR  = '/home/u11/jkadowaki/UDG_RedshiftEstimator/data'
    DATA_DIR  = '/Users/jkadowaki/dataset'
    DATA_FILE = os.path.join(DATA_DIR, 'training.csv')
    data      = pd.read_csv(DATA_FILE)
    data      = data.sort_values(by=["cz"], ascending=True)['cz'].values

    # Compute Baselines
    mse_base      = baseline(mse, data)
    perc_err_base = baseline(percent_err, data)

    print(f"MSE:    \targmin: {mse_base[0]} \tMSE:{mse_base[1]}")
    print(f"%Error: \targmin: {perc_err_base[0]} \t%Error:{perc_err_base[1]}")
