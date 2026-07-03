import xarray as xr

for var, fname in [('rain', 'era5_rain_2025.grib'), ('wind', 'era5_wind_2025.grib')]:
    ds = xr.open_dataset(
        f'data/raw/era5/{fname}', engine='cfgrib',
        backend_kwargs={'indexpath': ''}
    )
    has_vt = 'valid_time' in ds
    print(f"{var}: vars={list(ds.data_vars)}, dims={dict(ds.sizes)}, valid_time={has_vt}")
    if has_vt:
        print(f"  valid_time shape: {ds['valid_time'].shape}")
        print(f"  valid_time[0,:3]: {ds['valid_time'].values[0,:3]}")
