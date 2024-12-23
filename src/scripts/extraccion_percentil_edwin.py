from unir_archivos_jj import unir_archivos_grib
import os
import pdb

ruta_datos = "../../data/raw/era5"
ruta_datos_salida = "../../data/raw/processed"
files = os.listdir(ruta_datos)

# Filter files that start with era5_2m_temperature
files = [file for file in files if file.startswith("era5_2m_temperature_8")]

# Files that ends with .grib
files = [file for file in files if file.endswith(".grib")]

# Create a list with the full path of the files
archivos = [os.path.join(ruta_datos, file) for file in files]

archivo_union = unir_archivos_grib(archivos, salida= os.path.join(ruta_datos_salida, "era5_2m_temperature_union.nc"))