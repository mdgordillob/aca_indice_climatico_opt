import os
import numpy as np
import xarray as xr
import pandas as pd
import pdb


def calcular_percentiles_viento(archivo_entrada): 
    """
    Se calcula aquellos valores que están encima del percentil 90:
    *Std
    *Mean
    *WP_90: el percentil 90
    *WP_90_JK: promedio mensual de días que superan dicho percentil, tasa de mensual
    """
    print(f"Loading wind data from: {archivo_entrada}")
    
    if not os.path.exists(archivo_entrada):
        raise FileNotFoundError(f"Input file not found: {archivo_entrada}")
    
    ds = xr.open_dataset(archivo_entrada)
    ds = ds.sel(time=slice('1961', '1990'))
    ds = ds.assign_coords(month=ds["time"].dt.month) 
    p= 1.23  #constante de la densidad del aire (kg/m3)
    ds["wind_power"]= (p* (ds["wind_speed"]**3))/2 
    
    # Threshold
    # mean
    promedio_wind_power = ds['wind_power'].groupby("time.month").mean(dim="time")
    # std
    std_wind_power = ds['wind_power'].groupby("time.month").std(dim="time")
    # threshold
    threshold = promedio_wind_power + (1.28 * std_wind_power)

    percentiles_max = ds['wind_power'].groupby("time.month").quantile(0.9, dim="time")
    exceeding_values = ds['wind_power'].groupby("time.month") > threshold

    # Calcular la media y la desviación estándar por mes de los valores que exceden el umbral de 90 
    exceed_90_max_y_m = exceeding_values.groupby(["time.year", "time.month"]).mean(dim="time")
    mean_exceeding = exceed_90_max_y_m.groupby("month").mean(dim="year")
    std_exceeding = exceed_90_max_y_m.groupby("month").std(dim="year")

    estadisticas = xr.Dataset({
        'percentil_90': percentiles_max,
        'mean_exceeding': mean_exceeding,
        'std_exceeding': std_exceeding,
        'threshold': threshold
    })
    
    return estadisticas
     

def guardar_percentiles_viento(estadisticas, archivo_salida, data_dir, guardar_csv=False):
    """
    Guarda los percentiles calculados en un archivo NetCDF y CSV.

    Parámetros:
        estadisticas (xr.Dataset): Dataset con los percentiles calculados.
        archivo_salida (str): Ruta del archivo NetCDF de salida.
        data_dir (str): Ruta del directorio de datos.
        guardar_csv (bool): Si guardar también en formato CSV.
    """
    print(f"Saving wind percentiles to: {archivo_salida}")
    estadisticas.to_netcdf(archivo_salida)
    
    if guardar_csv:
        try:
            subset = estadisticas.sel(latitude=5.9, longitude=-72.99, method='nearest')[['percentil_90', 'mean_exceeding', 'std_exceeding']]
            df = subset.to_dataframe().reset_index()
            output_path = os.path.join(data_dir, "viento_percentiles.csv")
            df.to_csv(output_path, index=False)
            print(f"CSV saved to: {output_path}")
        except Exception as e:
            print(f"Warning: Could not save CSV: {e}")


def main():
    # Get the script's directory and navigate to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, 'data', 'processed')
    
    file = 'era5_daily_combined_wind.nc'
    archivo_union = os.path.join(data_dir, file)
    archivo_salida = os.path.join(data_dir, "era5_wind_percentil.nc")

    try:
        print("=" * 60)
        print("CALCULATING WIND PERCENTILES")
        print("=" * 60)
        print(f"Project root: {project_root}")
        print(f"Data directory: {data_dir}")
        
        estadisticas = calcular_percentiles_viento(archivo_union)
        guardar_percentiles_viento(estadisticas, archivo_salida, data_dir, guardar_csv=True)
        print(f"✓ Archivo de percentiles creado en: {archivo_salida}")
        print("=" * 60)
    except Exception as e:
        print(f"Error al calcular percentiles: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
