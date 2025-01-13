import os
import xarray as xr


    
def load_grid_data(file_path, year, month, variable):
    """Load grid data for a specific year, month, and variable."""
    grid_data = xr.open_dataset(file_path, engine='cfgrib')[variable].sel(time=f"{year}-{month:02}")
    if variable == 't2m':  # Convert temperature to Celsius
        grid_data -= 273.15
    return grid_data

def resample_to_daily(grid_data):
    """Resample hourly data to daily max and min."""
    daily_max = grid_data.resample(time='1D').max(dim='time')
    daily_min = grid_data.resample(time='1D').min(dim='time')
    return daily_max, daily_min

def load_percentiles(percentile_file, month):
    """Load precomputed percentiles for a specific month."""
    percentiles_data = xr.open_dataset(percentile_file)
    return percentiles_data.sel(month=month)

def compute_occurrences(daily_data, percentile_10, percentile_90):
    """Compute occurrences of values above the 90th percentile and below the 10th percentile."""
    count_above_90 = (daily_data > percentile_90).sum(dim='time')
    count_below_10 = (daily_data < percentile_10).sum(dim='time')
    return count_above_90, count_below_10

def calculate_anomalies(count_data, mean, std_dev):
    """Calculate normalized anomalies."""
    return (count_data - mean) / std_dev

def drop_unnecessary_coords(data_arrays, coord_name):
    """Drop an unnecessary coordinate from a list of DataArrays."""
    return [data_array.drop(coord_name) for data_array in data_arrays]

def create_anomalies_dataset(variables, attrs):
    """Create an xarray.Dataset for anomalies."""
    return xr.Dataset(variables, attrs=attrs)

def main():
    # File paths and configuration
    ruta_datos_grib = "../../data/raw/era5/"
    file = 'era5_2m_temperature_81_82.grib'
    archivo_comparar = os.path.join(ruta_datos_grib, file)
    ruta_datos_percentiles = "../../data/processed"
    datos_percentiles_archivo = "era5_temperatura_percentil.nc"
    archivo_percentiles = os.path.join(ruta_datos_percentiles, datos_percentiles_archivo)
    year, month, variable = 1981, 1, 't2m'

    # Load and preprocess grid data
    grid_data = load_grid_data(archivo_comparar, year, month, variable)
    daily_max, daily_min = resample_to_daily(grid_data)

    # Load percentiles
    month_percentiles = load_percentiles(archivo_percentiles, month)
    percentile_10_max = month_percentiles['percentiles_max'].sel(quantile=0.1)
    percentile_90_max = month_percentiles['percentiles_max'].sel(quantile=0.9)
    percentile_10_min = month_percentiles['percentiles_min'].sel(quantile=0.1)
    percentile_90_min = month_percentiles['percentiles_min'].sel(quantile=0.9)

    # Compute occurrences
    count_above_90_max, count_below_10_max = compute_occurrences(daily_max, percentile_10_max, percentile_90_max)
    count_above_90_min, count_below_10_min = compute_occurrences(daily_min, percentile_10_min, percentile_90_min)

    # Calculate anomalies
    anomalies_above_max = calculate_anomalies(count_above_90_max, month_percentiles['mean_max'], month_percentiles['std_dev_max'])
    anomalies_below_max = calculate_anomalies(count_below_10_max, month_percentiles['mean_max'], month_percentiles['std_dev_max'])
    anomalies_above_min = calculate_anomalies(count_above_90_min, month_percentiles['mean_min'], month_percentiles['std_dev_min'])
    anomalies_below_min = calculate_anomalies(count_below_10_min, month_percentiles['mean_min'], month_percentiles['std_dev_min'])

    # Drop unnecessary coordinates
    variables_to_drop = [count_above_90_max, count_below_10_max, count_above_90_min, count_below_10_min,
                         anomalies_above_max, anomalies_below_max, anomalies_above_min, anomalies_below_min]
    variables_dropped = drop_unnecessary_coords(variables_to_drop, 'quantile')

    # Create anomalies dataset
    anomalies = create_anomalies_dataset({
        'count_above_90_max': variables_dropped[0],
        'count_below_10_max': variables_dropped[1],
        'count_above_90_min': variables_dropped[2],
        'count_below_10_min': variables_dropped[3],
        'anomalies_above_max': variables_dropped[4],
        'anomalies_below_max': variables_dropped[5],
        'anomalies_above_min': variables_dropped[6],
        'anomalies_below_min': variables_dropped[7]
    }, attrs={'description': 'Anomalies and counts of temperature extremes'})

    # Save the dataset
    anomalies.to_netcdf("../../data/processed/era5_temperatura_anomalias2.nc")

    # Output results
    print(anomalies)

if __name__ == "__main__":
    main()
