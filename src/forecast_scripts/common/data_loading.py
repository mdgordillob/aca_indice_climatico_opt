"""
Shared monthly data loading for the ETS/ETSX/AutoTS monthly forecasters.

Extracted from forecast_ica_monthly.py and ETS_ica_forecast.py, where this
code was byte-identical. Reads the per-region monthly anomaly outputs from
data/processed/anomalias_<region>/ produced by src/scripts/calcular_anomalias_*.py:
NetCDF for temperature/wind, CSV for precipitation/drought.
"""

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

# Number of months from 1961-01 used as the synthetic index when the source
# files don't carry a usable time coordinate (matches the 1961-2024 baseline).
MONTHLY_SERIES_LENGTH = 768


def load_monthly_data(base_path):
    """
    Load monthly NetCDF files (temperature, wind) and CSV files (precipitation, drought)
    for a single region from base_path (typically data/processed/anomalias_<region>).

    Returns:
        dict: keys among {'temperature', 'wind', 'precipitation', 'drought'},
              values are xr.Dataset (temperature/wind) or pd.DataFrame (precipitation/drought).
    """
    print("\n[Loading monthly data...]")

    datasets = {}

    print("\n  Loading temperature files...")
    temp_files = sorted(glob.glob(os.path.join(base_path, "anomalies_temperature_*.nc")))
    if temp_files:
        try:
            temp_datasets = [xr.open_dataset(f) for f in temp_files]
            datasets['temperature'] = xr.concat(temp_datasets, dim='time')
            print(f"  [OK] Loaded {len(temp_files)} temperature files")
        except Exception as e:
            print(f"  [WARNING] Error loading temperature: {e}")
    else:
        print(f"  [WARNING] No temperature files found")

    print("\n  Loading wind files...")
    wind_files = sorted(glob.glob(os.path.join(base_path, "anomalies_wind_*.nc")))
    if wind_files:
        try:
            wind_datasets = [xr.open_dataset(f) for f in wind_files]
            datasets['wind'] = xr.concat(wind_datasets, dim='time')
            print(f"  [OK] Loaded {len(wind_files)} wind files")
        except Exception as e:
            print(f"  [WARNING] Error loading wind: {e}")
    else:
        print(f"  [WARNING] No wind files found")

    print("\n  Loading precipitation data...")
    precip_file = os.path.join(base_path, "anomalies_precipitation_combined.csv")
    if os.path.exists(precip_file):
        try:
            df_precip = pd.read_csv(precip_file)
            if 'time' in df_precip.columns:
                df_precip['time'] = pd.to_datetime(df_precip['time'])
            datasets['precipitation'] = df_precip
            print(f"  [OK] Loaded precipitation CSV")
        except Exception as e:
            print(f"  [WARNING] Error loading precipitation: {e}")
    else:
        print(f"  [WARNING] Precipitation file not found")

    print("\n  Loading drought data...")
    drought_file = os.path.join(base_path, "anomalies_drought_combined.csv")
    if os.path.exists(drought_file):
        try:
            df_drought = pd.read_csv(drought_file)
            if 'time' in df_drought.columns:
                df_drought['time'] = pd.to_datetime(df_drought['time'])
            datasets['drought'] = df_drought
            print(f"  [OK] Loaded drought CSV")
        except Exception as e:
            print(f"  [WARNING] Error loading drought: {e}")
    else:
        print(f"  [WARNING] Drought file not found")

    return datasets


def extract_regional_series(datasets):
    """
    Extract spatial mean time series from datasets and combine into a single
    wide DataFrame. Keeps only anomaly variables: T90, T10, and anomaly
    columns from the precipitation/drought CSVs.

    Args:
        datasets (dict): output of load_monthly_data() - xr.Dataset or pd.DataFrame values.

    Returns:
        pd.DataFrame or None: combined, deduplicated, NaN-dropped wide time series.
    """
    print("\n[Extracting regional time series...]")

    series_dict = {}
    time_index = None

    for var_name, ds in datasets.items():
        if isinstance(ds, xr.Dataset):
            try:
                if time_index is None:
                    if 'time' in ds.indexes:
                        time_index = ds.indexes['time'].to_index() if hasattr(ds.indexes['time'], 'to_index') else ds['time'].values
                    elif 'time' in ds.coords:
                        time_index = pd.date_range(start='1961-01-01', periods=len(ds['time']), freq='MS')
                        print(f"  Using generated monthly index (time not found in data)")
                    else:
                        print(f"  Warning: No time coordinate found in {var_name}")

                for data_var in ds.data_vars:
                    try:
                        if 't_90' in data_var.lower() or 't_10' in data_var.lower() or 'anomal' in data_var.lower():
                            spatial_mean = ds[data_var].mean(dim=['latitude', 'longitude'], skipna=True)

                            df_temp = spatial_mean.to_dataframe().reset_index()
                            df_temp = df_temp[['time', data_var]]
                            df_temp.columns = ['time', f'{var_name}_{data_var}']
                            df_temp.set_index('time', inplace=True)

                            series_dict[f'{var_name}_{data_var}'] = df_temp[f'{var_name}_{data_var}']
                            print(f"  [OK] {var_name}_{data_var}: {len(df_temp)} observations")
                    except Exception as e:
                        print(f"  [WARNING] Could not process {var_name}_{data_var}: {e}")
            except Exception as e:
                print(f"  [WARNING] Error processing {var_name} dataset: {e}")

        elif isinstance(ds, pd.DataFrame):
            try:
                df = ds.copy()

                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'], errors='coerce')
                    time_index = df['time']
                    df.set_index('time', inplace=True)
                elif 'Año' in df.columns and 'Mes' in df.columns:
                    df['time'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].astype(str).str.zfill(2) + '-01', format='%Y-%m-%d')
                    time_index = df['time']
                    df.set_index('time', inplace=True)

                anomaly_cols = [col for col in df.columns
                                 if 'Anomalia' in col or 'anomalies' in col]

                for col in anomaly_cols:
                    col_name = f'{var_name}_{col}'
                    series_dict[col_name] = df[col]
                    print(f"  [OK] {col_name}: {len(df)} observations")
            except Exception as e:
                print(f"  [WARNING] Error processing {var_name} DataFrame: {e}")

    if not series_dict:
        print("[FAILED] No data could be extracted!")
        return None

    proper_index = pd.date_range(start='1961-01-01', periods=MONTHLY_SERIES_LENGTH, freq='MS')
    for key in series_dict.keys():
        series_dict[key].index = proper_index

    data_wide = pd.concat(series_dict, axis=1)

    if not isinstance(data_wide.index, pd.DatetimeIndex):
        if data_wide.index.dtype in [np.int64, np.int32, np.float64, np.float32]:
            data_wide.index = pd.date_range(start='1961-01-01', periods=len(data_wide), freq='MS')
        else:
            data_wide.index = pd.to_datetime(data_wide.index, errors='coerce')

    data_wide = data_wide.sort_index()
    data_wide = data_wide[~data_wide.index.duplicated(keep='first')]
    data_wide = data_wide.dropna()

    print(f"\n[OK] Combined dataset shape: {data_wide.shape}")
    print(f"  Date range: {data_wide.index.min()} to {data_wide.index.max()}")
    print(f"  Index type: {type(data_wide.index)}")
    print(f"  Variables: {list(data_wide.columns)}")

    return data_wide
