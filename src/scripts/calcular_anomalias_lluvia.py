import xarray as xr
import pandas as pd
import os
import numpy as np
import rioxarray as rio
import geopandas as gpd
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

# Global cache for shapefiles and statistics
_shapefile_cache = {}
_estadisticas_cache = {}

def get_cached_shapefile(shapefile_path):
    """Load shapefile once and cache it."""
    if shapefile_path not in _shapefile_cache:
        _shapefile_cache[shapefile_path] = gpd.read_file(shapefile_path)
    return _shapefile_cache[shapefile_path]

def get_cached_estadisticas(estadisticas_file, shapefile_path):
    """Load and cache statistics for a specific file."""
    cache_key = (estadisticas_file, shapefile_path)
    if cache_key not in _estadisticas_cache:
        estadisticas_data = xr.open_dataset(estadisticas_file)
        estadisticas_data = estadisticas_data.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
        estadisticas_data = estadisticas_data.rio.write_crs("EPSG:4326")
        
        # Clip to shapefile
        shapefile = get_cached_shapefile(shapefile_path)
        estadisticas_data = estadisticas_data.rio.clip(shapefile.geometry, shapefile.crs)
        _estadisticas_cache[cache_key] = estadisticas_data
    
    return _estadisticas_cache[cache_key]

def load_grid_data(file_path, variable, shapefile):
    """
    Load grid data for a specific variable from a GRIB file.
    If a shapefile is provided, clips the data to the specified region.
    
    Parameters:
    - file_path (str): Path to the GRIB file containing the data.
    - variable (str): Name of the variable to extract from the file.
    - shapefile (GeoDataFrame): Shapefile that defines the region to clip.

    Returns:
    - grid_data (xarray.DataArray): Selected variable data, clipped if shapefile provided.
    """
    grid_data = xr.open_dataset(file_path, engine="cfgrib")[variable]
    grid_data = grid_data.rio.write_crs("EPSG:4326")
    grid_data = grid_data.rio.clip(shapefile.geometry, shapefile.crs, drop=True)
    
    return grid_data

def resample_to_daily_precipitation(grid_data):
    """
    Convert hourly precipitation data to daily sums.

    Parameters:
    - grid_data (xarray.DataArray): Precipitation data with hourly resolution.

    Returns:
    - daily_sum (xarray.DataArray): Precipitation data summed to daily level.
    """
    # Adjust to UTC-5 (Colombia timezone)
    grid_data['valid_time'] = grid_data['valid_time'] - pd.Timedelta(hours=5)

    # Flatten the data structure by combining 'time' and 'step' dimensions
    grid_data_plain = grid_data.stack(valid_time_dim=("time", "step")).reset_index("valid_time_dim")

    # Group by day and sum daily precipitation values
    daily_sum = grid_data_plain.groupby(grid_data_plain["valid_time"].dt.floor("D")).sum(dim='valid_time_dim')

    # Rename time dimension
    daily_sum = daily_sum.rename({"floor": "time"})

    return daily_sum

def alinear_data(data1, data2, op):
    """Align precipitation data with reference statistics."""
    meses = data1["time"].dt.month

    mean_tp_aligned = data2["mean_tp"].sel(month=meses).drop_vars("month")
    std_tp_aligned = data2["std_tp"].sel(month=meses).drop_vars("month")

    # Convert to Dataset if DataArray
    if isinstance(data1, xr.DataArray):
        data1 = data1.to_dataset(name="tp_daily_sum")

    data1 = data1.assign(mean_tp=mean_tp_aligned, std_tp=std_tp_aligned)
        
    if op == 1:
        return data1.drop_vars("month", errors="ignore")
    else:
        return data1

def anomalias(data):
    """
    Calculate precipitation anomalies.

    Parameters:
    - data (xarray.Dataset): Data with daily precipitation and statistics.

    Returns:
    - data (xarray.Dataset): Data with new "anomalias" variable.
    """
    return data.assign(anomalias=(data["tp_daily_sum"] - data["mean_tp"]) / data["std_tp"])

def calcular_5_dias_maximo(data):
    """
    Calculate maximum accumulated precipitation over 5 consecutive days within each month.

    Parameters:
    - data (xarray.DataArray): Daily precipitation data.

    Returns:
    - Rx5day (xarray.DataArray): Maximum accumulated precipitation over 5 consecutive days per month.
    """
    # Sum precipitation over 5-day rolling windows
    data_5_day_sum = data.rolling(time=5, min_periods=1).sum()

    # Add month column to verify 5 days belong to same month
    data_5_day_sum['month'] = data_5_day_sum['time'].dt.month

    # Calculate month difference between first and last day of 5-day window
    month_diff = data_5_day_sum['month'] - data_5_day_sum['month'].shift(time=4)

    # Create mask to filter only values where all 5 days are in same month
    mask = (month_diff == 0)

    # Apply mask to remove values that cross month boundaries
    data_5_day_sum = data_5_day_sum.where(mask, drop=True)

    # Select maximum accumulated 5-day precipitation per month
    Rx5day = data_5_day_sum.resample(time='1M').max()

    return Rx5day

def calcular_anomalias_lluvia(data, p):
    """
    Calculate precipitation anomalies based on maximum 5-day accumulated precipitation.

    Parameters:
    - data (xarray.DataArray): Daily precipitation data.
    - p (xarray.Dataset): Dataset with mean and standard deviation of maximum 5-day accumulated precipitation.

    Returns:
    - anomalias_lluvia (xarray.Dataset): Precipitation anomalies.
    """
    Rx5day = calcular_5_dias_maximo(data)
    Rx5day = alinear_data(Rx5day, p, 1)
    anomalias_lluvia = anomalias(Rx5day)
    
    return anomalias_lluvia

def count_most_frequent_with_condition(data):
    """
    Find maximum frequency of values in array and adjust based on condition:
    - If value `0` is present, return maximum frequency.
    - If `0` not present, subtract 1 from maximum frequency.

    Parameters:
    - data (numpy array): Input data.

    Returns:
    - int: Adjusted maximum frequency.
    """
    values, counts = np.unique(data, return_counts=True)
    return counts.max() - 1

def calcular_interpolacion(data, aux_year, ruta_salida):
    """Calculate interpolated dry days for drought analysis."""
    ruta = ruta_salida

    data = data.where(data < 0.001, other=1)
    data = data.where(data >= 0.001, other=0)

    # Calculate cumulative sum of dry days within each year
    data_cumsum = data.cumsum(dim='time')
    data_cumsum['time'] = data['time']

    # Apply function to count most frequent consecutive dry days
    count_most_frequent_da = xr.apply_ufunc(
        count_most_frequent_with_condition,
        data_cumsum.groupby('time.year'),
        input_core_dims=[['time']],
        output_core_dims=[[]],
        vectorize=True,
        dask="allowed",
    )

    count_most_frequent_da = count_most_frequent_da.squeeze("year")

    if aux_year == 1961:
        count_most_frequent_da = count_most_frequent_da.expand_dims(month=np.arange(1, 13))
        CDD = count_most_frequent_da * (count_most_frequent_da['month'] / 12)
    else:
        aux_year_prev = aux_year - 1
        data1 = count_most_frequent_da
        
        try:
            data2 = xr.open_dataarray(os.path.join(ruta, f'datos_maximos_{aux_year_prev}.nc'))
            data_aligned = data2.sel(latitude=data1.latitude, longitude=data1.longitude, method="nearest")
            
            if isinstance(data1, xr.DataArray):
                data1 = data1.to_dataset(name="tp")
            
            data1 = data1.assign(tp_daily_sum=data_aligned)
            data1 = data1.expand_dims(month=np.arange(1, 13))
            
            data1['new'] = xr.where(
                data1['month'] != 12,
                ((12 - data1['month']) / 12) * data1['tp_daily_sum'] + (data1['month'] / 12) * data1['tp'],
                data1['tp']
            )
            
            CDD = data1['new']
        except FileNotFoundError:
            # If previous year data doesn't exist, use current year data
            count_most_frequent_da = count_most_frequent_da.expand_dims(month=np.arange(1, 13))
            CDD = count_most_frequent_da * (count_most_frequent_da['month'] / 12)

    CDD.sel(month=12).to_netcdf(os.path.join(ruta, f'datos_maximos_{aux_year}.nc'))
    
    new_date = pd.to_datetime(dict(year=aux_year, month=CDD['month'].values, day=1))
    CDD = CDD.assign_coords(date=("month", new_date))
    CDD = CDD.rename("tp_daily_sum")
    CDD = CDD.assign_coords(month=CDD.coords["date"].values)
    CDD = CDD.rename({"month": "time"})

    return CDD

def calcular_anomalias_sequia(data, p, year, ruta_salida):
    """
    Calculate drought anomalies based on duration of consecutive dry days.

    Parameters:
    - data (xarray.Dataset): Daily precipitation data.
    - p (xarray.Dataset): Dataset with mean and standard deviation of consecutive dry days.
    - year (int): Year being processed.
    - ruta_salida (str): Output directory path.

    Returns:
    - anomalias_sequia (xarray.Dataset): Drought anomalies.
    """
    CDD = calcular_interpolacion(data, year, ruta_salida)
    CDD = alinear_data(CDD, p, 0)
    anomalias_sequia = anomalias(CDD)
    
    return anomalias_sequia

def procesar_anomalias(est1, est2, file, shapefile, year, ruta_salida):
    """Process anomalies for a single year."""
    try:
        # Load grid data
        archivo = load_grid_data(file, "tp", shapefile)

        # Resample to daily
        ds_resampled = resample_to_daily_precipitation(archivo)
        ds_resampled = ds_resampled.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))

        # Calculate anomalies
        anomalias_lluvia = calcular_anomalias_lluvia(ds_resampled, est1)
        anomalias_sequia = calcular_anomalias_sequia(ds_resampled, est2, year, ruta_salida)

        # Extract monthly means
        anomalias_lluvias_resultado = anomalias_lluvia['anomalias'].groupby("time.month").mean(dim=["latitude", "longitude"])
        anomalias_sequia_resultado = anomalias_sequia['anomalias'].groupby("time.month").mean(dim=["latitude", "longitude"])

        return anomalias_lluvias_resultado, anomalias_sequia_resultado
    
    except Exception as e:
        print(f"  Error processing year {year}: {e}")
        return None, None

def guardar_anomalias(df_lluvia, df_sequia, ruta_salida, salida_lluvia, salida_sequia):
    """Save anomalies to CSV files."""
    df_lluvia = pd.DataFrame(df_lluvia)
    df_sequia = pd.DataFrame(df_sequia)

    df_lluvia.to_csv(os.path.join(ruta_salida, salida_lluvia), index=False)
    df_sequia.to_csv(os.path.join(ruta_salida, salida_sequia), index=False)
    
    print(f"✓ Precipitation anomalies saved to {os.path.join(ruta_salida, salida_lluvia)}")
    print(f"✓ Drought anomalies saved to {os.path.join(ruta_salida, salida_sequia)}")

def process_year(args):
    """Process a single year - designed for multiprocessing."""
    year, ruta, ruta_grib, ruta_salida, shapefile_path, files = args
    
    year_anomalies_lluvia = []
    year_anomalies_sequia = []
    
    # Find file for this year
    archivo_lluvia = [file for file in files if str(year) in file and file.endswith(".grib")]
    
    if not archivo_lluvia:
        print(f"  ⚠️  No GRIB files found for {year}")
        return year_anomalies_lluvia, year_anomalies_sequia
    
    archivo_lluvia = archivo_lluvia[0]
    archivo_lluvia_full = os.path.join(ruta_grib, archivo_lluvia)
    
    print(f"Processing year {year} with file: {archivo_lluvia}")
    
    try:
        # Load statistics (cached)
        shapefile = get_cached_shapefile(shapefile_path)
        est1 = get_cached_estadisticas(os.path.join(ruta, "era5_lluvias_percentil.nc"), shapefile_path)
        est2 = get_cached_estadisticas(os.path.join(ruta, "era5_sequia_percentil.nc"), shapefile_path)
        
        # Process anomalies for the year
        anomalias_mensuales_lluvia, anomalias_mensuales_sequia = procesar_anomalias(
            est1, est2, archivo_lluvia_full, shapefile, year, ruta_salida
        )
        
        if anomalias_mensuales_lluvia is None:
            return year_anomalies_lluvia, year_anomalies_sequia
        
        # Extract monthly values
        for mes in range(1, 13):
            try:
                anomalia_lluvia_mes = anomalias_mensuales_lluvia.sel(
                    time=anomalias_mensuales_lluvia.time.dt.month == mes, method="nearest"
                ).item()
                year_anomalies_lluvia.append({"Año": year, "Mes": mes, "Anomalia_Lluvia": anomalia_lluvia_mes})
                
                anomalia_sequia_mes = anomalias_mensuales_sequia.sel(
                    time=anomalias_mensuales_sequia.time.dt.month == mes, method="nearest"
                ).item()
                year_anomalies_sequia.append({"Año": year, "Mes": mes, "Anomalia_Sequia": anomalia_sequia_mes})
            
            except (KeyError, ValueError) as e:
                print(f"  Warning: Could not extract month {mes} for year {year}")
                continue
        
        print(f"✓ Completed year {year}")
    
    except Exception as e:
        print(f"  Error loading year {year}: {e}")
    
    return year_anomalies_lluvia, year_anomalies_sequia

def procesar_anomalias_lluvia(
    shapefile_path, ruta, ruta_grib, ruta_salida,
    salida_lluvia="anomalies_precipitation_combined.csv",
    salida_sequia="anomalies_drought_combined.csv",
    use_multiprocessing=True, num_workers=None
):
    """Main function to process precipitation and drought anomalies."""
    print(f"Looking for files in: {ruta_grib}")
    
    if not os.path.exists(ruta_grib):
        raise FileNotFoundError(f"Directory not found: {ruta_grib}")
    
    # Create output directory if needed
    os.makedirs(ruta_salida, exist_ok=True)
    
    # List all files
    files = os.listdir(ruta_grib)
    
    if not files:
        print(f"⚠️  No files found in {ruta_grib}")
        return None, None
    
    print(f"Found {len(files)} files in directory")

    # Prepare years to process
    years_to_process = list(range(1961, 2025))
    
    df_lluvia = []
    df_sequia = []
    
    if use_multiprocessing and len(years_to_process) > 1:
        # Use multiprocessing for year-level parallelization
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)
        
        print(f"Using multiprocessing with {num_workers} workers")
        
        # Prepare arguments for each year
        process_args = [
            (year, ruta, ruta_grib, ruta_salida, shapefile_path, files)
            for year in years_to_process
        ]
        
        # Process years in parallel
        with Pool(num_workers) as pool:
            results = pool.map(process_year, process_args)
        
        # Collect results
        for year_lluvia, year_sequia in results:
            df_lluvia.extend(year_lluvia)
            df_sequia.extend(year_sequia)
    
    else:
        # Sequential processing
        for year in years_to_process:
            print(f"\nProcessing year {year}...")
            year_lluvia, year_sequia = process_year((year, ruta, ruta_grib, ruta_salida, shapefile_path, files))
            df_lluvia.extend(year_lluvia)
            df_sequia.extend(year_sequia)

    if not df_lluvia or not df_sequia:
        print("⚠️  No anomalies calculated!")
        return None, None
    
    # Save anomalies
    guardar_anomalias(df_lluvia, df_sequia, ruta_salida, salida_lluvia, salida_sequia)

    return df_lluvia, df_sequia


if __name__ == "__main__":
    # Get the script's directory and navigate to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    shapefile_path = os.path.join(project_root, "data", "shapefiles", "colombia_4326.shp")
    ruta = os.path.join(project_root, "data", "processed")
    ruta_grib = os.path.join(project_root, "data", "raw", "era5")
    ruta_salida = os.path.join(project_root, "data", "processed", "anomalias_colombia")
    salida_lluvia = "anomalies_precipitation_combined.csv"
    salida_sequia = "anomalies_drought_combined.csv"

    print("=" * 60)
    print("CALCULATING PRECIPITATION AND DROUGHT ANOMALIES")
    print("=" * 60)     
    print(f"Project root: {project_root}")
    print(f"Data directory: {ruta_grib}")
    print("=" * 60)

    df1, df2 = procesar_anomalias_lluvia(
        shapefile_path, ruta, ruta_grib, ruta_salida,
        salida_lluvia, salida_sequia,
        use_multiprocessing=True,
        num_workers=None  # None = auto-detect available cores
    )
    print('\n✓ Process completed')
