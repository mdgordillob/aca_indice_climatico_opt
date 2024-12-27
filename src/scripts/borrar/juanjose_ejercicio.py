import os
from pathlib import Path
import xarray as xr

def unir_archivos_grib(lista_archivos, salida="era5_2m_temperature_union.grib"):
    """
    Función para unir varuis archivos GRIB en un único archivo.

    Parámetros:
        lista_archivos (list): Lista de rutas a los archivos GRIB.
        salida (str): Ruta del archivo GRIB combinado.

    Retorna:
        str: Ruta del archivo combinado.
    """

    #en caso de que no haya archivos
    if not lista_archivos:
        raise ValueError("La lista de archivos está vacía.")

    try:
        datasets = [xr.open_dataset(archivo, engine="cfgrib") for archivo in lista_archivos]
        dataset_union = xr.concat(datasets, dim="time")
        dataset_union.to_netcdf(salida)  
        return salida
    
    except Exception as e:
        raise RuntimeError(f"Error al unir los archivos GRIB: {e}")
    
    finally:
        for ds in datasets:
            ds.close()
            
######################################################################################################################

def calcular_percentiles(archivo_entrada, archivo_salida="era5_2m_temperature_percentil.grib"):
    """
    Calcula los percentiles 10 y 90 de un archivo GRIB por mes.

    Parámetros:
        archivo_entrada (str): Ruta del archivo GRIB de entrada.
        archivo_salida (str): Ruta del archivo GRIB con los percentiles calculados.

    Retorna:
        str: Ruta del archivo con los percentiles.
    """
    try:
        dataset = xr.open_dataset(archivo_entrada, engine="cfgrib")
        #agrupados por mes
        percentiles = dataset.groupby("time.month").reduce(
            lambda x: xr.concat([x.quantile(0.1), x.quantile(0.9)], dim="percentile")
        )
        percentiles.to_netcdf(archivo_salida)
        return archivo_salida
    
    except Exception as e:
        raise RuntimeError(f"Error al calcular percentiles: {e}")
    
    finally:
        dataset.close()

if __name__ == "__main__":
    
    ruta_datos = Path("../../data/raw/era5/temperatura/")
    archivos = [ruta_datos / "era5_2m_temperature_81_82.grib",
                ruta_datos / "era5_2m_temperature_83_84.grib"]

    archivo_union = unir_archivos_grib(archivos)
    print(f"Archivo unido creado en: {archivo_union}")

    archivo_percentil = calcular_percentiles(archivo_union)
    print(f"Archivo de percentiles creado en: {archivo_percentil}")