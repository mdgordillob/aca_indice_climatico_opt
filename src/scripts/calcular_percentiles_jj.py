import xarray as xr

def calcular_percentiles(archivo_entrada, archivo_salida="era5_2m_temperature_percentil.grib"):
    """
    calcula los percentiles 10 y 90 de un archivo GRIB por mes.

    Parámetros:
        archivo_entrada (str): Ruta del archivo GRIB de entrada.
        archivo_salida (str): Ruta del archivo GRIB con los percentiles calculados.

    Retorna:
        str: Ruta del archivo con los percentiles.
    """
    try:
        dataset = xr.open_dataset(archivo_entrada, engine="cfgrib")
        percentiles = dataset.groupby("time.month").reduce(
            lambda x: xr.concat([x.quantile(0.1), x.quantile(0.9)], dim="percentile")
        )
        percentiles.to_netcdf(archivo_salida)
        return archivo_salida
    except Exception as e:
        raise RuntimeError(f"Error al calcular percentiles: {e}")
    finally:
        dataset.close()

