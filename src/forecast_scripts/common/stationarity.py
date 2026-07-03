"""
Shared ADF stationarity testing and differencing helpers.

Extracted from ETS_ica_forecast.py and ETSX_ica_forecast.py, where both
carried the same logic (ETSX's copy was a cosmetic reformat, no behavior
change). reconstruct_from_differences is only used by the plain-ETS
forecaster (ETSX/SARIMAX handles differencing internally via its (p,d,q)
order instead), but lives here alongside the test it complements.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_test(series, variable_name):
    """
    Perform Augmented Dickey-Fuller test for stationarity.

    Returns:
        dict: adf_statistic, p_value, used_lag, n_obs, critical_values, is_stationary.
    """
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'variable': variable_name,
        'adf_statistic': result[0],
        'p_value': result[1],
        'used_lag': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05,
    }


def make_stationary(series, variable_name, max_diff=2, differencing_info=None):
    """
    Make a series stationary through differencing if needed.

    Args:
        series (pd.Series): Time series data.
        variable_name (str): Name of the variable (used for logging/info key).
        max_diff (int): Maximum number of differences to apply.
        differencing_info (dict, optional): if provided, records
            {variable_name: {...}} into this dict (mirrors the old
            self.differencing_info instance attribute pattern).

    Returns:
        tuple: (stationary_series, diff_order, adf_results)
    """
    print(f"\n  Testing stationarity for {variable_name}...")

    original_series = series.copy()
    current_series = series.copy()
    diff_order = 0
    adf_results = []

    for d in range(max_diff + 1):
        adf_result = adf_test(current_series, variable_name)
        adf_results.append(adf_result)

        print(f"    Diff order {d}: ADF={adf_result['adf_statistic']:.4f}, "
              f"p-value={adf_result['p_value']:.4f}, "
              f"Stationary={adf_result['is_stationary']}")

        if adf_result['is_stationary']:
            diff_order = d
            break

        if d < max_diff:
            current_series = current_series.diff().dropna()

    if differencing_info is not None:
        differencing_info[variable_name] = {
            'diff_order': diff_order,
            'adf_results': adf_results,
            'original_series': original_series,
            'stationary_series': current_series,
        }

    return current_series, diff_order, adf_results


def reconstruct_from_differences(differenced_forecast, original_series, diff_order):
    """
    Reconstruct original-scale forecast from differenced forecasts.

    Args:
        differenced_forecast (pd.Series or np.ndarray): Forecast in differenced form.
        original_series (pd.Series): Original series before differencing.
        diff_order (int): Number of differences applied.

    Returns:
        pd.Series or np.ndarray: Reconstructed forecast, same type as input.
    """
    if diff_order == 0:
        return differenced_forecast

    if isinstance(differenced_forecast, pd.Series):
        is_series = True
        forecast_index = differenced_forecast.index
        forecast_values = differenced_forecast.values
    else:
        is_series = False
        forecast_values = np.array(differenced_forecast)

    reconstructed = forecast_values.copy()
    last_value = original_series.iloc[-1]

    for _ in range(diff_order):
        reconstructed = np.cumsum(reconstructed) + last_value
        last_value = reconstructed[-1]

    if is_series:
        return pd.Series(reconstructed, index=forecast_index)
    return reconstructed
