import xarray as xr
import pandas as pd
import os
import pdb

def calcular_percentiles(archivo_entrada, variable):#antes -> (archivo_entrada, variable="t2m"):  este cambio se hizo para ser llamada para cualquier "ocasion"
    """
    Calcula los percentiles 10 y 90 de un archivo NetCDF por mes.

    Parámetros:
        archivo_entrada (str): Ruta del archivo NetCDF de entrada.

    Retorna:
        xr.Dataset: Dataset con los percentiles calculados.
    """
    dataset = xr.open_dataset(archivo_entrada)
    
    # Convert time from UTC to UTC-5
    dataset['time'] = dataset.indexes['time'] - pd.Timedelta(hours=5)
    
    # Resample to daily frequency and calculate daily max and min
    daily_max = dataset.resample(time='1D').max()
    daily_min = dataset.resample(time='1D').min()

    # If the variable is temperature, convert from Kelvin to Celsius
    if variable == 't2m':
        daily_max -= 273.15
        daily_min -= 273.15
    
    # Combine daily max and min into a single dataset
    daily_data = xr.Dataset({
        'daily_max': daily_max[variable],  
        'daily_min': daily_min[variable]   
    })
    
    # Calculate percentiles, mean, and standard deviation for each variable
    percentiles_max = daily_data['daily_max'].groupby("time.month").map(
        lambda x: x.quantile([0.1, 0.9], dim="time")
    )
    percentiles_min = daily_data['daily_min'].groupby("time.month").map(
        lambda x: x.quantile([0.1, 0.9], dim="time")
    )
    # Calculate the number of days per month where daily_max > 90th percentile
    days_above_90_max = daily_data['daily_max'].groupby("time.month").map(
        lambda x: (x > x.quantile(0.9)).sum(dim="time")
    )
    
    # Calculate the number of days per month where daily_min < 10th percentile
    days_below_10_min = daily_data['daily_min'].groupby("time.month").map(
        lambda x: (x < x.quantile(0.1)).sum(dim="time")
    )
    
    # Calculate mean and standard deviation of the number of days per month
    mean_max =   days_above_90_max.mean(dim="month")
    std_dev_max = days_above_90_max.std(dim="month")
    mean_min =   days_below_10_min.mean(dim="month")
    std_dev_min = days_below_10_min.std(dim="month")    
    # mean_max = daily_data['daily_max'].groupby("time.month").mean(dim="time")
    # mean_min = daily_data['daily_min'].groupby("time.month").mean(dim="time")
    # std_dev_max = daily_data['daily_max'].groupby("time.month").std(dim="time")
    # std_dev_min = daily_data['daily_min'].groupby("time.month").std(dim="time")
    
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

def guardar_percentiles(estadisticas, archivo_salida):
    """
    Guarda los percentiles calculados en un archivo NetCDF y CSV.

    Parámetros:
        estadisticas (xr.Dataset): Dataset con los percentiles calculados.
        archivo_salida (str): Ruta del archivo NetCDF de salida.
    """
    estadisticas.to_netcdf(archivo_salida)
    
    subset = estadisticas.sel(latitude=4, longitude=-73)[['month', 'percentiles_max', 'percentiles_min', 'mean_max', 'mean_min', 'std_dev_max', 'std_dev_min']]
    df = subset.to_dataframe().reset_index()
    output_path = "../../data/processed/percentiles.csv"
    df.to_csv(output_path, index=False)

def main():
    ruta_datos = "../../data/processed"
    file = 'era5_2m_temperature_union.nc'
    archivo_union = os.path.join(ruta_datos, file)
    archivo_salida = os.path.join(ruta_datos, "era5_temperatura_percentil.nc")

    try:
        estadisticas = calcular_percentiles(archivo_union)
        guardar_percentiles(estadisticas, archivo_salida)
        print(f"Archivo de percentiles creado en: {archivo_salida}")
    except Exception as e:
        print(f"Error al calcular percentiles: {e}")

if __name__ == "__main__":
    main()