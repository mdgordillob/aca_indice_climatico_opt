import xarray as xr
import os
import pdb

def calcular_percentiles(archivo_entrada, archivo_salida="era5_2m_temperature_percentil.nc"):
    """
    calcula los percentiles 10 y 90 de un archivo NetCDF por mes.

    Parámetros:
        archivo_entrada (str): Ruta del archivo NetCDF de entrada.
        archivo_salida (str): Ruta del archivo Netcdf con los percentiles calculados.

    Retorna:
        str: Ruta del archivo con los percentiles calculados.
    """
    try:
        dataset = xr.open_dataset(archivo_entrada)
        percentiles = dataset.groupby("time.month").map(
            lambda x: x.quantile([0.1, 0.9], dim="time")
        )
        pdb.set_trace()
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

    archivo_percentil = calcular_percentiles(archivo_union, archivo_salida= os.path.join(ruta_datos, "era5_all_var_percentil.nc"))
    print(f"Archivo de percentiles creado en: {archivo_percentil}")