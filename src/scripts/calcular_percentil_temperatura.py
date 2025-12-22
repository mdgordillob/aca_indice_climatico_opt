import xarray as xr
import pandas as pd
import os
import pdb


# Function to count days where daily_max > 90th percentile
def count_days_above_90th(x):
    quantile_90 = x.quantile(0.9)
    return (x > quantile_90).sum(dim="time")

def calcular_percentiles(archivo_entrada, variable = 't2m'):    

    """
    Calcula los percentiles 10 y 90 de un archivo NetCDF por mes.

    Parámetros:
        archivo_entrada (str): Ruta del archivo NetCDF de entrada.

    Retorna:
        xr.Dataset: Dataset con los percentiles calculados.
    """
    print(f"Loading data from: {archivo_entrada}")
    
    if not os.path.exists(archivo_entrada):
        raise FileNotFoundError(f"Input file not found: {archivo_entrada}")
    
    dataset = xr.open_dataset(archivo_entrada)
    
    # Filter the data in between 1961 and 1990
    dataset_filtered = dataset.sel(time=slice('1961', '1990'))
    
    # Resample to daily frequency and calculate daily max and min
    daily_max = dataset_filtered['daily_max']
    daily_min = dataset_filtered['daily_min']

    # If the variable is temperature, convert from Kelvin to Celsius
    if variable == 't2m':
        daily_max -= 273.15
        daily_min -= 273.15
    
    # Combine daily max and min into a single dataset
    daily_data = xr.Dataset({
        'daily_max': daily_max,  
        'daily_min': daily_min   
    })
    
    # Calculate percentiles for each variable
    percentiles_max = daily_data['daily_max'].groupby("time.month").quantile([0.1, 0.9], dim="time")
    percentiles_min = daily_data['daily_min'].groupby("time.month").quantile([0.1, 0.9], dim="time")
    
    # Identify days exceeding the 90th percentile or below the 10th percentile
    
    # Maximum temperature
    ## Temperatures above the 90th percentile (These are from our interest)
    exceed_90_max = (daily_data['daily_max'].groupby("time.month") > percentiles_max.sel(quantile=0.9)).astype(int)
    ## Temperatures below the 10th percentile
    below_10_max = (daily_data['daily_max'].groupby("time.month") < percentiles_max.sel(quantile=0.1)).astype(int)
    
    # Minimum temperature
    ## Temperatures above the 90th percentile
    exceed_90_min = (daily_data['daily_min'].groupby("time.month") > percentiles_min.sel(quantile=0.9)).astype(int)
    ## Temperatures below the 10th percentile (These are from our interest)
    below_10_min = (daily_data['daily_min'].groupby("time.month") < percentiles_min.sel(quantile=0.1)).astype(int)

    # Promedio de los valores máximos y mínimos
    valores_max = (exceed_90_max + exceed_90_min)/2
    valores_min = (below_10_max + below_10_min)/2

    exceed_90_max_y_m = valores_max.groupby(["time.year", "time.month"]).mean(dim="time")
    mean_max = exceed_90_max_y_m.groupby("month").mean(dim="year")
    std_dev_max = exceed_90_max_y_m.groupby("month").std(dim="year")

    below_10_min_y_m = valores_min.groupby(["time.year", "time.month"]).mean(dim="time")
    mean_min = below_10_min_y_m.groupby("month").mean(dim="year")
    std_dev_min = below_10_min_y_m.groupby("month").std(dim="year")
    
    # Combine all statistics into a single dataset
    estadisticas = xr.Dataset({
        'percentiles_max': percentiles_max,
        'percentiles_min': percentiles_min,
        'mean_max': mean_max,
        'mean_min': mean_min,
        'std_dev_max': std_dev_max,
        'std_dev_min': std_dev_min
    })
    
    return estadisticas

def guardar_percentiles(estadisticas, archivo_salida, data_dir, guardar_csv=False):
    """
    Guarda los percentiles calculados en un archivo NetCDF y CSV.

    Parámetros:
        estadisticas (xr.Dataset): Dataset con los percentiles calculados.
        archivo_salida (str): Ruta del archivo NetCDF de salida.
        data_dir (str): Ruta del directorio de datos.
        guardar_csv (bool): Si guardar también en formato CSV.
    """
    print(f"Saving percentiles to: {archivo_salida}")
    estadisticas.to_netcdf(archivo_salida)
    
    if guardar_csv:
        try:
            subset = estadisticas.sel(latitude=5.9, longitude=-72.99, method='nearest')[['percentiles_max', 'percentiles_min', 'mean_max', 'mean_min', 'std_dev_max', 'std_dev_min']]
            df = subset.to_dataframe().reset_index()
            output_path = os.path.join(data_dir, "percentiles_temperatura.csv")
            df.to_csv(output_path, index=False)
            print(f"CSV saved to: {output_path}")
        except Exception as e:
            print(f"Warning: Could not save CSV: {e}")

def main():
    # Get the script's directory and navigate to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, 'data', 'processed')
    
    file = 'era5_daily_combined_tmp.nc'
    archivo_union = os.path.join(data_dir, file)
    archivo_salida = os.path.join(data_dir, "era5_temperatura_percentil.nc")

    try:
        print("=" * 60)
        print("CALCULATING TEMPERATURE PERCENTILES")
        print("=" * 60)
        print(f"Project root: {project_root}")
        print(f"Data directory: {data_dir}")
        
        estadisticas = calcular_percentiles(archivo_union)
        guardar_percentiles(estadisticas, archivo_salida, data_dir, guardar_csv=True)
        print(f"✓ Archivo de percentiles creado en: {archivo_salida}")
        print("=" * 60)
    except Exception as e:
        print(f"Error al calcular percentiles: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
