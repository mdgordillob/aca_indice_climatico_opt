import os
import xarray as xr
import pdb

# File setup
ruta_datos_grib = "../../data/raw/era5/"
file = 'era5_2m_temperature_81_82.grib'
archivo_comparar = os.path.join(ruta_datos_grib, file)

# Year, month, and variable configuration
year = 1981
month = 1
variable = 't2m'

# Load grid data for the specific year and month
grid_data = xr.open_dataset(archivo_comparar, engine='cfgrib')[variable].sel(time=f"{year}-{month:02}")

# Convert temperature to Celsius if necessary
if variable == 't2m':
    grid_data -= 273.15

# Resample to daily frequency and calculate max and min for each day
daily_max = grid_data.resample(time='1D').max(dim='time')
daily_min = grid_data.resample(time='1D').min(dim='time')

# Load precomputed percentiles
ruta_datos_percentiles = "../../data/processed"
datos_percentiles_archivo = "era5_temperatura_percentil.nc"
archivo_percentiles = os.path.join(ruta_datos_percentiles, datos_percentiles_archivo)
percentiles_data = xr.open_dataset(archivo_percentiles)

# Filter percentiles for the specific month
month_percentiles = percentiles_data.sel(month=month)

# Extract 10th and 90th percentiles for max and min temperatures
percentile_10_max = month_percentiles['percentiles_max'].sel(quantile=0.1)
percentile_90_max = month_percentiles['percentiles_max'].sel(quantile=0.9)
percentile_10_min = month_percentiles['percentiles_min'].sel(quantile=0.1)
percentile_90_min = month_percentiles['percentiles_min'].sel(quantile=0.9)

# Count occurrences for max temperature
count_above_90_max = (daily_max > percentile_90_max).sum(dim='time')
count_below_10_max = (daily_max < percentile_10_max).sum(dim='time')

# Count occurrences for min temperature
count_above_90_min = (daily_min > percentile_90_min).sum(dim='time')
count_below_10_min = (daily_min < percentile_10_min).sum(dim='time')

# Calculate the normalized anomalies
anomalies_above_max = (count_above_90_max - month_percentiles['mean_max']) / month_percentiles['std_dev_max']
anomalies_below_max = (count_below_10_max - month_percentiles['mean_max']) / month_percentiles['std_dev_max']
anomalies_above_min = (count_above_90_min - month_percentiles['mean_min']) / month_percentiles['std_dev_min']
anomalies_below_min = (count_below_10_min - month_percentiles['mean_min']) / month_percentiles['std_dev_min']

# Drop the 'quantile' coordinate from each DataArray
count_above_90_max = count_above_90_max.drop('quantile')
count_below_10_max = count_below_10_max.drop('quantile')
count_above_90_min = count_above_90_min.drop('quantile')
count_below_10_min = count_below_10_min.drop('quantile')
anomalies_above_max = anomalies_above_max.drop('quantile')
anomalies_below_max = anomalies_below_max.drop('quantile')
anomalies_above_min = anomalies_above_min.drop('quantile')
anomalies_below_min = anomalies_below_min.drop('quantile')


# Create a dataset for anomalies
pdb.set_trace()
anomalies = xr.Dataset({
    'count_above_90_max': count_above_90_max,
    'count_below_10_max': count_below_10_max,
    'count_above_90_min': count_above_90_min,
    'count_below_10_min': count_below_10_min,
    'anomalies_above_max': anomalies_above_max,
    'anomalies_below_max': anomalies_below_max,
    'anomalies_above_min': anomalies_above_min,
    'anomalies_below_min': anomalies_below_min
}, attrs={'description': 'Anomalies and counts of temperature extremes'})

# Save the anomalies dataset to a NetCDF file
anomalies.to_netcdf("../../data/processed/era5_temperatura_anomalias2.nc")