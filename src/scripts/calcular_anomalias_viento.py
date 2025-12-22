import os
import xarray as xr
import numpy as np
import pandas as pd
import rioxarray as rxr
import geopandas as gpd
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

from calcular_anomalias_temperatura import (
    get_cached_shapefile,
    get_cached_percentiles,
    load_percentiles,
    load_grid_data,
    load_annual_grid_data,
    get_monthly_data,
    calculate_anomalies,
    create_anomalies_dataset
)

# Wind power density constant (kg/m3)
AIR_DENSITY = 1.23

def compute_wind_power(wind_speed):
    """Calculate wind power density from wind speed."""
    return (AIR_DENSITY * (wind_speed**3)) / 2

def compute_occurrences(daily_data, percentile):
    """Compute binary occurrences of days exceeding the percentile threshold (preserves time dimension)."""
    count_above = (daily_data > percentile).astype(int)
    return count_above

def load_annual_grid_data_safe(file_path, year, variable, shapefile_path=None):
    """
    Safely load annual grid data with error handling for corrupted files.
    
    Returns:
        xarray.Dataset or None if file is corrupted/incomplete
    """
    try:
        # Load wind components with error suppression for corrupted messages
        grid_data = xr.open_dataset(
            file_path, 
            engine='cfgrib',
            backend_kwargs={'errors': 'warn'}
        )[['u10', 'v10']]
        
        # Filter to requested year
        grid_data = grid_data.sel(time=f"{year}")
        
        # Adjust time to UTC-5
        grid_data['time'] = grid_data.indexes['time'] - pd.Timedelta(hours=5)
        
        # Resample to daily mean
        grid_data = grid_data.resample(time="1D").mean(dim="time")
        
        # Calculate wind speed
        grid_data["wind_speed"] = (grid_data["u10"]**2 + grid_data["v10"]**2)**0.5
        grid_data = grid_data.drop_vars(["u10", "v10"])
        
        # Validate data completeness (at least 80% of year = 292 days)
        days_available = len(grid_data.time)
        days_expected = 365 if year % 4 != 0 else 366
        coverage = days_available / days_expected * 100
        
        if coverage < 80:
            print(f"    ⚠️  Incomplete year: only {days_available}/{days_expected} days ({coverage:.1f}%)")
            return None
        
        # Clip to shapefile if provided
        if shapefile_path:
            shape = get_cached_shapefile(shapefile_path)
            grid_data = grid_data.rio.write_crs("EPSG:4326", inplace=True)
            grid_data = grid_data.rio.clip(shape.geometry, shape.crs, drop=True)
        
        return grid_data
    
    except Exception as e:
        error_name = type(e).__name__
        error_msg = str(e)[:100]
        print(f"    ⚠️  Cannot load file ({error_name}: {error_msg})")
        return None

def calcular_anomalias_viento(archivo_percentiles, grid_data_monthly, year, month, salida_anomalias, shapefile_path=None):
    """Calculate wind anomalies for a specific month using pre-loaded grid data."""
    
    # Calculate wind power from pre-loaded monthly wind speed data
    wind_power = compute_wind_power(grid_data_monthly["wind_speed"])
    
    # Load percentiles from cache
    month_percentiles = get_cached_percentiles(archivo_percentiles, month)
    
    # Extract threshold (90th percentile of wind power)
    percentile_90_power = month_percentiles['threshold']
    
    # Clip percentiles if shapefile provided
    if shapefile_path:
        shape = get_cached_shapefile(shapefile_path)
        month_percentiles = month_percentiles.rio.write_crs("EPSG:4326", inplace=True)
        month_percentiles = month_percentiles.rio.clip(shape.geometry, shape.crs, drop=True)
        percentile_90_power = month_percentiles['threshold']
    
    # Compute daily occurrences above threshold (preserves time dimension)
    count_above = compute_occurrences(wind_power, percentile_90_power)
    
    # Calculate monthly average occurrences
    count_above_monthly = count_above.groupby(["time.month"]).mean(dim="time").sel(month=month)
    
    # Calculate anomalies
    anomalies_above = calculate_anomalies(
        count_above_monthly,
        month_percentiles['mean_exceeding'],
        month_percentiles['std_exceeding']
    )
    
    # Create anomalies dataset
    anomalies = create_anomalies_dataset({
        'count_above': count_above_monthly,
        'anomalies_above': anomalies_above
    }, attrs={'description': 'Anomalies of wind speed extremes'})
    
    # Save the dataset
    anomalies.to_netcdf(salida_anomalias)
    
    return anomalies.mean(dim=['latitude', 'longitude'], keep_attrs=True)

def process_year_wind(args):
    """Process a single year for wind anomalies - designed for multiprocessing."""
    year, archivo_percentiles, archivo_comparar_location, shapefile_path, output_netcdf, files = args
    
    year_anomalies = []
    
    # Find all files containing the year
    archivo_comparar = [file for file in files if str(year) in file]
    archivo_comparar = [file for file in archivo_comparar if file.endswith(".grib")]
    if not archivo_comparar:
        print(f"  ⚠️  No GRIB files found for {year}")
        return year_anomalies
    
    archivo_comparar = [file for file in archivo_comparar if "wind" in file]
    if not archivo_comparar:
        print(f"  ⚠️  No wind files found for {year}")
        return year_anomalies
    
    if len(archivo_comparar) != 1:
        print(f"  ⚠️  Expected 1 wind file for {year}, found {len(archivo_comparar)}")
        return year_anomalies
    
    archivo_comparar = archivo_comparar[0]
    archivo_comparar_full = os.path.join(archivo_comparar_location, archivo_comparar)
    
    print(f"Processing year {year} with file: {archivo_comparar}")
    
    # Load the entire year once with safe error handling
    annual_grid_data = load_annual_grid_data_safe(archivo_comparar_full, year, 'wind_speed', shapefile_path)
    
    if annual_grid_data is None:
        print(f"  ⚠️  Skipping year {year} (incomplete or corrupted data)")
        return year_anomalies
    
    # Process all months for this year
    for month in range(1, 13):
        try:
            # Extract monthly data from already-loaded annual data
            grid_data_monthly = get_monthly_data(annual_grid_data, year, month, 'wind_speed')
            
            # Skip if month has no data
            if len(grid_data_monthly.time) == 0:
                continue
            
            ds_month = calcular_anomalias_viento(
                archivo_percentiles=archivo_percentiles,
                grid_data_monthly=grid_data_monthly,
                year=year,
                month=month,
                salida_anomalias=os.path.join(output_netcdf, f"anomalies_wind_{year}_{month}.nc"),
                shapefile_path=shapefile_path
            )
            ds_month = ds_month.assign_coords(year=year)
            year_anomalies.append(ds_month)
        except Exception as e:
            print(f"  ⚠️  Error in {year}-{month:02d}: {type(e).__name__}")
            continue
    
    return year_anomalies

def procesar_anomalias_viento(archivo_percentiles, archivo_comparar_location, output_csv_path, shapefile_path, output_netcdf, use_multiprocessing=True, num_workers=None):
    """Process wind anomalies for all years and months."""
    print(f"Looking for files in: {archivo_comparar_location}")
    
    if not os.path.exists(archivo_comparar_location):
        raise FileNotFoundError(f"Directory not found: {archivo_comparar_location}")
    
    # List all files in the directory
    files = os.listdir(archivo_comparar_location)
    
    if not files:
        print(f"⚠️  No files found in {archivo_comparar_location}")
        return
    
    print(f"Found {len(files)} files in directory")

    # Create output directory if it doesn't exist
    os.makedirs(output_netcdf, exist_ok=True)

    # Prepare list of years to process
    years_to_process = list(range(1961, 2025))
    
    if use_multiprocessing and len(years_to_process) > 1:
        # Use multiprocessing for year-level parallelization
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)
        
        print(f"Using multiprocessing with {num_workers} workers")
        
        # Prepare arguments for each year
        process_args = [
            (year, archivo_percentiles, archivo_comparar_location, shapefile_path, output_netcdf, files)
            for year in years_to_process
        ]
        
        # Process years in parallel
        with Pool(num_workers) as pool:
            results = pool.map(process_year_wind, process_args)
        
        # Flatten results
        all_anomalies = [anomaly for year_result in results for anomaly in year_result]
    else:
        # Sequential processing
        all_anomalies = []
        for year in years_to_process:
            print(f"\nProcessing year {year}...")
            year_anomalies = process_year_wind((year, archivo_percentiles, archivo_comparar_location, shapefile_path, output_netcdf, files))
            all_anomalies.extend(year_anomalies)

    if not all_anomalies:
        print("⚠️  No anomalies calculated!")
        return
    
    # Combine all monthly datasets into one
    combined_anomalies = xr.concat(all_anomalies, dim='time')

    # Convert the xarray.Dataset to a pandas.DataFrame
    anomalies_df = combined_anomalies.to_dataframe().reset_index()

    # Save the DataFrame to a CSV file
    anomalies_df.to_csv(output_csv_path, index=False)
    print(f"\n✓ Anomalies saved to {output_csv_path}")

if __name__ == "__main__":
    # Get the script's directory and navigate to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    archivo_percentiles = os.path.join(project_root, "data", "processed", "era5_wind_percentil.nc")
    archivo_comparar_location = os.path.join(project_root, "data", "raw", "era5")
    output_csv_path = os.path.join(project_root, "data", "processed", "anomalias_colombia", "anomalies_wind_combined.csv")
    shapefile_path = os.path.join(project_root, "data", "shapefiles", "colombia_4326.shp")
    output_netcdf = os.path.join(project_root, "data", "processed", "anomalias_colombia")
    
    print("=" * 60)
    print("CALCULATING WIND ANOMALIES")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Data directory: {archivo_comparar_location}")
    print("=" * 60)

    # Run with multiprocessing enabled (set to False for single-threaded debugging)
    procesar_anomalias_viento(
        archivo_percentiles, 
        archivo_comparar_location, 
        output_csv_path, 
        shapefile_path, 
        output_netcdf,
        use_multiprocessing=True,
        num_workers=None  # None = auto-detect available cores
    )
