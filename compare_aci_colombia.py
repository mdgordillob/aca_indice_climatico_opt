import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point

WORK_DIR = Path(__file__).resolve().parent
RAW_ERA5_DIR = WORK_DIR / 'data' / 'raw' / 'era5'
PROCESSED_DIR = WORK_DIR / 'data' / 'processed'
SHAPEFILE_PATH = WORK_DIR / 'data' / 'shapefiles' / 'colombia_4326.shp'
MASK_PATH = PROCESSED_DIR / 'colombia_era5_mask.nc'
ACI_TEMP_FILE = PROCESSED_DIR / 'era5_daily_combined_tmp.nc'
ACI_WIND_FILE = PROCESSED_DIR / 'era5_daily_combined_wind.nc'
ACI_PRECIP_FILE = PROCESSED_DIR / 'era5_daily_combined_rain.nc'
DAILY_BY_YEAR_TMP_DIR = PROCESSED_DIR / 'daily_by_year_tmp'
DAILY_BY_YEAR_RAIN_DIR = PROCESSED_DIR / 'daily_by_year_rain'
DAILY_BY_YEAR_WIND_DIR = PROCESSED_DIR / 'daily_by_year_wind'
ACI_OUTPUT_CSV = PROCESSED_DIR / 'aci_python_output_1961_1990.csv'
COMPARISON_PLOT = WORK_DIR / 'aci_vs_original_comparison.png'
OVERLAY_PLOT = WORK_DIR / 'aci_vs_original_overlay.png'
COMPONENTS_PLOT = WORK_DIR / 'aci_components_plot.png'
ORIGINAL_INDEX_CSV = WORK_DIR / 'articles' / 'graficas' / 'forecast_cundinamarca_bogota' / 'forecast_cundinamarca_bogota_combined.csv'

YEAR_START = 1961
YEAR_END = 1990
COUNTRY_ABBREV = 'COL'
STUDY_PERIOD = ('1961-01-01', '1990-12-31')
REFERENCE_PERIOD = ('1961-01-01', '1990-12-31')


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def make_colombia_mask(mask_path: Path, shapefile_path: Path, template_ds: xr.Dataset):
    if mask_path.exists():
        print(f'Skipping mask creation, exists: {mask_path}')
        return mask_path

    print('Creating Colombia mask...')
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is None or gdf.crs.to_string() != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')

    geometry = gdf.unary_union
    lat = template_ds.latitude.values
    lon = template_ds.longitude.values
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    points = [Point(x, y) for x, y in zip(lon_grid.ravel(), lat_grid.ravel())]
    mask_flat = np.array([geometry.covers(pt) for pt in points], dtype=np.int8)
    mask_data = mask_flat.reshape(lat_grid.shape)

    mask_ds = xr.Dataset(
        {'country': (('latitude', 'longitude'), mask_data)},
        coords={'latitude': lat, 'longitude': lon}
    )
    mask_ds.to_netcdf(mask_path)
    print(f'Mask saved to: {mask_path}')
    return mask_path


def merge_temperature(year_start: int, year_end: int, output_path: Path):
    if output_path.exists():
        print(f'Skipping temperature merge, exists: {output_path}')
        return output_path

    paths = [str(RAW_ERA5_DIR / f'era5_tmp_{year}.grib') for year in range(year_start, year_end + 1)]
    print(f'Merging temperature data for years {year_start}-{year_end} into {output_path}')
    ds = xr.open_mfdataset(paths, engine='cfgrib', concat_dim='time', combine='nested', backend_kwargs={'indexpath': ''})
    ds = ds.assign_coords(time=ds.indexes['time'] - pd.Timedelta(hours=5))
    ds.to_netcdf(output_path, encoding={'t2m': {'zlib': True, 'complevel': 4}})
    print(f'Temperature dataset saved: {output_path}')
    return output_path


def merge_wind(year_start: int, year_end: int, output_path: Path):
    if output_path.exists():
        print(f'Skipping wind merge, exists: {output_path}')
        return output_path

    paths = [str(RAW_ERA5_DIR / f'era5_wind_{year}.grib') for year in range(year_start, year_end + 1)]
    print(f'Merging wind data for years {year_start}-{year_end} into {output_path}')
    ds = xr.open_mfdataset(paths, engine='cfgrib', concat_dim='time', combine='nested', backend_kwargs={'indexpath': ''})
    ds = ds.assign_coords(time=ds.indexes['time'] - pd.Timedelta(hours=5))
    ds.to_netcdf(output_path, encoding={
        'u10': {'zlib': True, 'complevel': 4},
        'v10': {'zlib': True, 'complevel': 4}
    })
    print(f'Wind dataset saved: {output_path}')
    return output_path


def merge_precipitation(year_start: int, year_end: int, output_path: Path):
    if output_path.exists():
        print(f'Skipping precipitation merge, exists: {output_path}')
        return output_path

    paths = [str(RAW_ERA5_DIR / f'era5_rain_{year}.grib') for year in range(year_start, year_end + 1)]
    print(f'Merging precipitation data for years {year_start}-{year_end} into {output_path}')
    ds = xr.open_mfdataset(paths, engine='cfgrib', concat_dim='time', combine='nested', backend_kwargs={'indexpath': ''})
    ds = ds.assign_coords(valid_time=ds['valid_time'] - pd.Timedelta(hours=5))
    ds2 = ds.stack(valid_time_dim=('time', 'step')).reset_index('valid_time_dim')
    daily = ds2.groupby(ds2['valid_time'].dt.floor('D')).sum(dim='valid_time_dim')
    daily = daily.rename({'floor': 'time'})
    daily = daily.drop_vars([v for v in ['step', 'valid_time', 'number', 'surface'] if v in daily.data_vars or v in daily.coords], errors='ignore')
    daily.to_netcdf(output_path, encoding={'tp': {'zlib': True, 'complevel': 4}})
    print(f'Precipitation dataset saved: {output_path}')
    return output_path


def combine_yearly_netcdf(source_dir: Path, pattern: str, output_path: Path, rename_vars: dict = None):
    if output_path.exists():
        print(f'Skipping combine, exists: {output_path}')
        return output_path

    paths = sorted(source_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No files found in {source_dir} matching {pattern}')

    print(f'Combining yearly files from {source_dir} into {output_path}')
    ds = xr.open_mfdataset([str(path) for path in paths], concat_dim='time', combine='nested')
    if rename_vars:
        ds = ds.rename(rename_vars)
    ds = ds.sortby('time')

    time_index = ds.indexes['time']
    if time_index.has_duplicates:
        ds = ds.sel(time=~time_index.duplicated())

    encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}
    ds.to_netcdf(output_path, encoding=encoding)
    print(f'Combined dataset saved: {output_path}')
    return output_path


def prepare_input_files():
    if not ACI_TEMP_FILE.exists():
        if DAILY_BY_YEAR_TMP_DIR.exists():
            combine_yearly_netcdf(DAILY_BY_YEAR_TMP_DIR, 't2m_daily_*.nc', ACI_TEMP_FILE)
        else:
            merge_temperature(YEAR_START, YEAR_END, ACI_TEMP_FILE)

    if not ACI_WIND_FILE.exists():
        if DAILY_BY_YEAR_WIND_DIR.exists():
            combine_yearly_netcdf(DAILY_BY_YEAR_WIND_DIR, 'wind_speed_daily_*.nc', ACI_WIND_FILE)
        else:
            merge_wind(YEAR_START, YEAR_END, ACI_WIND_FILE)

    if not ACI_PRECIP_FILE.exists():
        if DAILY_BY_YEAR_RAIN_DIR.exists():
            combine_yearly_netcdf(DAILY_BY_YEAR_RAIN_DIR, 'tp_daily_*.nc', ACI_PRECIP_FILE)
        else:
            merge_precipitation(YEAR_START, YEAR_END, ACI_PRECIP_FILE)


def run_aci(temperature_file: Path, precipitation_file: Path, wind_file: Path, mask_file: Path):
    import sys
    aci_dir = Path(__file__).resolve().parent.parent / 'ACI-Python-main'
    if not aci_dir.exists():
        raise FileNotFoundError(f'ACI-Python-main repository not found at {aci_dir}')
    sys.path.insert(0, str(aci_dir))

    from aci.aci import ActuarialClimateIndex

    aci = ActuarialClimateIndex(
        temperature_data_path=str(temperature_file),
        precipitation_data_path=str(precipitation_file),
        wind_speed_data_path=str(wind_file),
        country_abbrev=None,
        mask_data_path=str(mask_file),
        study_period=STUDY_PERIOD,
        reference_period=REFERENCE_PERIOD,
    )

    print('Calculating ACI...')
    df = aci.calculate_aci()
    df.to_csv(ACI_OUTPUT_CSV, index=True)
    print(f'ACI output saved: {ACI_OUTPUT_CSV}')
    return df


def load_original_index(original_csv: Path):
    df = pd.read_csv(original_csv, index_col=0)
    df.index = pd.to_datetime(df.index, errors='coerce')
    df.index = df.index.to_period('M').to_timestamp('M')
    return df


def plot_comparison(aci_df: pd.DataFrame, original_df: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')

    joined = pd.DataFrame({
        'ACI_Python': aci_df['ACI'],
        'ACA_Original': original_df['Climate_Index']
    }).dropna()
    if joined.empty:
        raise ValueError('No overlapping monthly dates found for comparison.')

    rolling_window = 60
    joined['ACI_Python_5yr'] = joined['ACI_Python'].rolling(window=rolling_window, min_periods=1).mean()
    joined['ACA_Original_5yr'] = joined['ACA_Original'].rolling(window=rolling_window, min_periods=1).mean()

    plt.figure(figsize=(15, 6))
    plt.plot(joined.index, joined['ACI_Python_5yr'], color='#2874A6', linewidth=2, label='ACI-Python 5yr avg')
    plt.plot(joined.index, joined['ACA_Original_5yr'], color='#E67E22', linewidth=2, label='ACA original 5yr avg')
    plt.axvline(pd.Timestamp('1990-01-01'), color='gray', linestyle='dotted', linewidth=1)

    plt.title('Colombia ERA5 ACI comparison (5-year moving average)')
    plt.xlabel('Year')
    plt.ylabel('Standardized index')
    plt.legend()
    plt.grid(True, linestyle='dotted', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f'Comparison plot saved: {output_path}')


def plot_components(aci_df: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')

    component_columns = [col for col in ['t90', 't10', 'precipitation', 'drought', 'wind'] if col in aci_df.columns]
    if not component_columns:
        raise ValueError('No ACI component columns found in the DataFrame.')

    rolling_window = 60
    comp_df = aci_df[component_columns].rolling(window=rolling_window, min_periods=1).mean()

    plt.figure(figsize=(15, 6))
    color_map = {
        't90': '#6B8E8E',
        't10': '#A9C8C8',
        'precipitation': '#D9A441',
        'drought': '#C08056',
        'wind': '#5D6D7E'
    }
    for column in comp_df.columns:
        plt.plot(comp_df.index, comp_df[column], color=color_map.get(column, None), linewidth=2, label=f'{column} 5yr avg')

    plt.axvline(pd.Timestamp('1990-01-01'), color='gray', linestyle='dotted', linewidth=1)
    plt.title('ACI component 5-year moving average series for Colombia ERA5')
    plt.xlabel('Year')
    plt.ylabel('Standardized component value')
    plt.legend()
    plt.grid(True, linestyle='dotted', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f'Component series plot saved: {output_path}')


def plot_aci_overlay(aci_df: pd.DataFrame, original_df: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')

    joined = pd.DataFrame({
        'ACI_Python': aci_df['ACI'],
        'ACA_Original': original_df['Climate_Index']
    }).dropna()
    if joined.empty:
        raise ValueError('No overlapping monthly dates found for comparison.')

    joined['ACI_Python_5yr'] = joined['ACI_Python'].rolling(window=60, min_periods=1).mean()
    joined['ACA_Original_5yr'] = joined['ACA_Original'].rolling(window=60, min_periods=1).mean()

    plt.figure(figsize=(15, 6))
    plt.plot(joined.index, joined['ACI_Python'], color='#2874A6', alpha=0.25, linewidth=1, label='ACI-Python monthly')
    plt.plot(joined.index, joined['ACA_Original'], color='#E67E22', alpha=0.25, linewidth=1, label='ACA original monthly')
    plt.plot(joined.index, joined['ACI_Python_5yr'], color='#2874A6', linewidth=2.5, label='ACI-Python 5yr avg')
    plt.plot(joined.index, joined['ACA_Original_5yr'], color='#E67E22', linewidth=2.5, label='ACA original 5yr avg')
    plt.axvline(pd.Timestamp('1990-01-01'), color='gray', linestyle='dotted', linewidth=1)

    plt.title('ACI-Python vs ACA original Climate Index (monthly and 5-year avg)')
    plt.xlabel('Year')
    plt.ylabel('Standardized index')
    plt.legend()
    plt.grid(True, linestyle='dotted', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f'Overlay comparison plot saved: {output_path}')


def main():
    ensure_dir(PROCESSED_DIR)
    prepare_input_files()
    if not ACI_TEMP_FILE.exists():
        raise FileNotFoundError(f"Processed temperature file not found: {ACI_TEMP_FILE}")
    sample_temp = xr.open_dataset(str(ACI_TEMP_FILE))
    make_colombia_mask(MASK_PATH, SHAPEFILE_PATH, sample_temp)
    aci_df = run_aci(ACI_TEMP_FILE, ACI_PRECIP_FILE, ACI_WIND_FILE, MASK_PATH)
    original_df = load_original_index(ORIGINAL_INDEX_CSV)
    plot_comparison(aci_df, original_df, COMPARISON_PLOT)
    plot_aci_overlay(aci_df, original_df, OVERLAY_PLOT)
    plot_components(aci_df, COMPONENTS_PLOT)


if __name__ == '__main__':
    main()
