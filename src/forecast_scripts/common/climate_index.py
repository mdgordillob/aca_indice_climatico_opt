"""
Shared Actuarial Climate Index (ICA) composite calculation.

Formula: ICA = (T90 - T10 + wind + precipitation + drought) / 5

Extracted from forecast_ica_monthly.py, ETS_ica_forecast.py, and
ETSX_ica_forecast.py, which each carried a near-identical copy. The wind
match condition here is the broader of the two variants found (matches on
'wind' OR on 'anomal'+'above'); ETSX_ica_forecast.py's narrower 'wind'-only
condition was a subset of this, so unifying does not change its behavior
for existing column names.

Based on the methodology in src/scripts/graficas.py's plot_ICA function.
"""

import pandas as pd


def calculate_climate_index(data):
    """
    Calculate the ICA composite from a DataFrame of anomaly columns.

    Args:
        data (pd.DataFrame): DataFrame with anomaly columns (temperature,
            wind, precipitation, drought - column names matched by substring).

    Returns:
        pd.Series: Climate index values, divided by 5 if at least 4 of the
            5 components were found, otherwise divided by however many were.
    """
    ica = pd.Series(0.0, index=data.index)
    component_count = 0

    t90_col = None
    t10_col = None
    for col in data.columns:
        if 't_90' in col.lower():
            t90_col = col
        elif 't_10' in col.lower():
            t10_col = col

    if t90_col is not None and t10_col is not None:
        ica += data[t90_col] - data[t10_col]
        component_count += 1

    for col in data.columns:
        if 'wind' in col.lower() or ('anomal' in col.lower() and 'above' in col.lower()):
            ica += data[col]
            component_count += 1
            break

    for col in data.columns:
        if 'precip' in col.lower() or 'lluvia' in col.lower():
            ica += data[col]
            component_count += 1
            break

    for col in data.columns:
        if 'drought' in col.lower() or 'sequia' in col.lower():
            ica += data[col]
            component_count += 1
            break

    divisor = 5 if component_count >= 4 else component_count
    return ica / divisor if divisor > 0 else ica
