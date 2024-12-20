import xarray as xr

def unir_archivos_grib(lista_archivos, salida="era5_2m_temperature_union.grib"):
    """
    unir varios archivos GRIB en un único archivo.

    Parámetros:
        - lista_archivos (list): Lista de rutas a los archivos GRIB.
        - salida (str): Ruta del archivo GRIB combinado.

    Retorna:
        str: Ruta del archivo combinado.
    """
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
