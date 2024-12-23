import xarray as xr
import pandas as pd
import os
import pdb

def calcular_percentiles(archivo_entrada, archivo_salida="era5_2m_temperature_percentil.nc"):
    """
    Calcula los percentiles 10 y 90 de un archivo NetCDF por mes.

    Parámetros:
        archivo_entrada (str): Ruta del archivo NetCDF de entrada.
        archivo_salida (str): Ruta del archivo Netcdf con los percentiles calculados.

    Retorna:
        str: Ruta del archivo con los percentiles calculados.
    """
    try:
        dataset = xr.open_dataset(archivo_entrada)
        
        # Convert time from UTC to UTC-5
        dataset['time'] = dataset.indexes['time'] - pd.Timedelta(hours=5)
        
        # Resample to daily frequency and calculate daily max and min
        daily_max = dataset.resample(time='1D').max()
        daily_min = dataset.resample(time='1D').min()
        
        # Convert temperature from Kelvin to Celsius
        daily_max = daily_max - 273.15
        daily_min = daily_min - 273.15
        
        # Combine daily max and min into a single dataset
        pdb.set_trace()
        daily_data = xr.Dataset({
            'daily_max': daily_max['t2m'],  # Replace 't2m' with the actual variable name
            'daily_min': daily_min['t2m']   # Replace 't2m' with the actual variable name
        })
        
        # Calculate percentiles for each cell
        percentiles = daily_data.groupby("time.month").map(
            lambda x: x.quantile([0.1, 0.9], dim="time")
        )
        
        percentiles.to_netcdf(archivo_salida)
        
        return archivo_salida
    except Exception as e:
        raise RuntimeError(f"Error al calcular percentiles: {e}")
    finally:
        dataset.close()

if __name__ == "__main__":
    ruta_datos = "../../data/raw/processed"
    file = 'era5_2m_temperature_union.nc'

    archivo_union = os.path.join(ruta_datos, file)

    archivo_percentil = calcular_percentiles(archivo_union, archivo_salida= os.path.join(ruta_datos, "era5_temperatura_percentil.nc"))
    print(f"Archivo de percentiles creado en: {archivo_percentil}")