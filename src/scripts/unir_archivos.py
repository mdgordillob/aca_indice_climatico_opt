import xarray as xr
import pdb
import os
import pandas as pd

def load_grid_data(file_path, variable):
    """
    Load grid data for a specific variable.
    """
    grid_data = xr.open_dataset(file_path, engine="cfgrib")[variable]
    return grid_data

def resample_to_daily(grid_data):
    """
    Resample hourly data to daily max and min.
    """
    # For Colombia we have to convert the time from UTC to UTC-5
    grid_data['time'] = grid_data.indexes['time'] - pd.Timedelta(hours=5)

    daily_max = grid_data.resample(time="1D").max(dim="time")
    daily_min = grid_data.resample(time="1D").min(dim="time")
    return daily_max, daily_min

def save_yearly_data_combined(daily_max, daily_min, year, output_dir, variable):
    """
    Save daily max and min data for a year to a single NetCDF file.
    """
    os.makedirs(output_dir, exist_ok=True)
    combined = xr.Dataset({
        f"daily_max": daily_max,
        f"daily_min": daily_min
    })
    output_file = os.path.join(output_dir, f"{variable}_daily_{year}.nc")
    combined.to_netcdf(output_file)
    print(f"Saved yearly file: {output_file}")

def process_yearly_data(file_path, year, variable, output_dir):
    """
    Process a single year's data: load, resample, and save combined max and min.
    """
    grid_data = load_grid_data(file_path, variable)
    daily_max, daily_min = resample_to_daily(grid_data)
    save_yearly_data_combined(daily_max, daily_min, year, output_dir, variable)

def merge_yearly_files(output_dir, merged_file, variable):
    """
    Merge all yearly NetCDF files into a single dataset.

    :param output_dir: Directory containing the yearly NetCDF files.
    :param merged_file: Path for the final merged NetCDF file.
    :param variable: The variable name (e.g., "t2m").
    """
    # Get all yearly files in the directory
    yearly_files = sorted([os.path.join(output_dir, file) for file in os.listdir(output_dir)
                           if file.endswith(".nc")])

    # Open all files into a list of datasets
    datasets = [xr.open_dataset(file) for file in yearly_files]

    # Concatenate datasets along the "time" dimension
    merged_dataset = xr.concat(datasets, dim="time")

    # Save the merged dataset to a single NetCDF file
    merged_dataset.to_netcdf(merged_file)
    print(f"Merged dataset saved to: {merged_file}")

# Example Usage
if __name__ == "__main__":
    input_dir = "../../data/raw/era5"
    output_dir = "../../data/processed/daily_by_year"
    merged_file = "../../data/processed/era5_daily_combined_tmp.nc"
    variable = "t2m"

    ## Process each year
    #for year in range(1960, 1991):  # Adjust year range as needed
    #    file_path = os.path.join(input_dir, f"era5_tmp_{year}.grib")
    #    if os.path.exists(file_path):
    #        process_yearly_data(file_path, year, variable, output_dir)
    
    # Merge all yearly files into one
    merge_yearly_files(output_dir, merged_file, variable)


#def load_grid_data(file_path, year, month, variable):
#    """
#    Load grid data for a specific year, month, and variable.
#    """
#    grid_data = xr.open_dataset(file_path, engine="cfgrib")[variable].sel(time=f"{year}-{month:02}")
#    return grid_data
#
#def resample_to_daily(grid_data):
#    """
#    Resample hourly data to daily max and min.
#    """
#    daily_max = grid_data.resample(time="1D").max(dim="time")
#    daily_min = grid_data.resample(time="1D").min(dim="time")
#    return daily_max, daily_min
#
#def save_daily_data_combined(daily_max, daily_min, year, month, output_dir, variable):
#    """
#    Save daily max and min data to a single NetCDF file.
#    """
#    os.makedirs(output_dir, exist_ok=True)
#    combined = xr.Dataset({
#        f"{variable}_max": daily_max,
#        f"{variable}_min": daily_min
#    })
#    output_file = os.path.join(output_dir, f"{variable}_daily_{year}_{month:02}.nc")
#    combined.to_netcdf(output_file)
#    print(f"Saved combined file: {output_file}")
#
#def process_monthly_data(file_path, year, month, variable, output_dir):
#    """
#    Process a single month's data: load, resample, and save combined max and min.
#    """
#    grid_data = load_grid_data(file_path, year, month, variable)
#    daily_max, daily_min = resample_to_daily(grid_data)
#    save_daily_data_combined(daily_max, daily_min, year, month, output_dir, variable)
#
#def merge_combined_files(output_dir, merged_file):
#    """
#    Merge all intermediate combined NetCDF files into one.
#    """
#    combined_files = sorted([os.path.join(output_dir, file) for file in os.listdir(output_dir)
#                             if file.endswith(".nc")])
#    datasets = [xr.open_dataset(file) for file in combined_files]
#    merged_dataset = xr.concat(datasets, dim="time")
#    merged_dataset.to_netcdf(merged_file)
#    print(f"Merged file saved to: {merged_file}")
#
## Example Usage
#if __name__ == "__main__":
#    input_dir = "../../data/raw/era5"
#    output_dir = "../../data/processed/daily_combined"
#    merged_file = "../../data/processed/era5_daily_combined.nc"
#    variable = "t2m"
#
#    # Process each month (update with your file paths and years)
#    for year in range(1960, 1991):  # Adjust year range as needed
#        for month in range(1, 13):
#            file_path = os.path.join(input_dir, f"era5_tmp_{year}.grib")
#            if os.path.exists(file_path):
#                process_monthly_data(file_path, year, month, variable, output_dir)
#
#    # Merge all intermediate files into one
#    merge_combined_files(output_dir, merged_file)


#def unir_archivos_grib(lista_archivos, salida="era5_2m_temperature_union.nc"):
#    """
#    unir varios archivos GRIB en un único archivo netCDF.
#
#    Parámetros:
#        - lista_archivos (list): Lista de rutas a los archivos GRIB.
#        - salida (str): Ruta del archivo GRIB combinado.
#
#    Retorna:
#        str: Ruta del archivo combinado netcdf.
#    """
#    if not lista_archivos:
#        raise ValueError("La lista de archivos está vacía.")
#
#    try:
#        datasets = [xr.open_dataset(archivo, engine="cfgrib") for archivo in lista_archivos]
#        dataset_union = xr.merge(datasets)
#        dataset_union.to_netcdf(salida) 
#        return salida
# 
#    except Exception as e:
#        raise RuntimeError(f"Error al unir los archivos GRIB: {e}")
# 
#    finally:
#        for ds in datasets:
#            ds.close()
#
#if __name__ == "__main__":
#    #### tmp
#    ruta_datos = "../../data/raw/era5"
#    ruta_datos_salida = "../../data/raw/processed"
#    files = os.listdir(ruta_datos)
#
#    # Filter files that start with era5_2m_temperature
#    files = [file for file in files if file.startswith("era5_tmp_")]
#
#    # Files that ends with .grib
#    files = [file for file in files if file.endswith(".grib")]
#
#    # Sort files
#    files = sorted(files)
#
#    # Create a list with the full path of the files
#    archivos = [os.path.join(ruta_datos, file) for file in files]
#
#    archivo_union = unir_archivos_grib(archivos, salida= os.path.join(ruta_datos_salida, "era5_tmp_union_col.nc"))
#    
#    ########################## rain
#    ruta_datos = "../../data/raw/era5"
#    ruta_datos_salida = "../../data/raw/processed"
#    files = os.listdir(ruta_datos)
#
#    # Filter files that start with era5_2m_temperature
#    files = [file for file in files if file.startswith("era5_rain_")]
#
#    # Files that ends with .grib
#    files = [file for file in files if file.endswith(".grib")]
#
#    # Sort files
#    files = sorted(files)
#
#    # Create a list with the full path of the files
#    archivos = [os.path.join(ruta_datos, file) for file in files]
#
#    archivo_union = unir_archivos_grib(archivos, salida= os.path.join(ruta_datos_salida, "era5_rain_union_col.nc"))
#
#    ########################## wind
#    ruta_datos = "../../data/raw/era5"
#    ruta_datos_salida = "../../data/raw/processed"
#    files = os.listdir(ruta_datos)
#
#    # Filter files that start with era5_2m_temperature
#    files = [file for file in files if file.startswith("era5_wind_")]
#
#    # Files that ends with .grib
#    files = [file for file in files if file.endswith(".grib")]
#
#    # Sort files
#    files = sorted(files)
#
#    # Create a list with the full path of the files
#    archivos = [os.path.join(ruta_datos, file) for file in files]
#
#    archivo_union = unir_archivos_grib(archivos, salida= os.path.join(ruta_datos_salida, "era5_wind_union_col.nc"))
#    
#    # Print output
#    print('Proceso finalizado')
#    
