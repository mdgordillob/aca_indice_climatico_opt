# Función para extraer el percentil de la temperatura de la base de datos de ERA5
import xarray as xr
import pdb

# Path to the GRIB file
path = '../../data/raw/era5/temperatura/'
file = 'era5_2m_temperature_1961_1970.grib'

# Open the GRIB file using xarray and cfgrib
ds = xr.open_dataset(f'{path}{file}', engine='cfgrib')
pdb.set_trace()

# Calculate the 90th percentile
percentile_90 = ds.quantile(0.9, dim='time')

# Print the result
print(percentile_90)