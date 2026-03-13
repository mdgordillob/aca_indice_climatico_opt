import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 1961
F_LAND = rf"C:\Users\mdgor\data\raw\era5_land\era5land_rain_{YEAR}.grib"
F_ERA5 = rf"C:\Users\mdgor\aca\aca_indice_climatico_opt-main\data\raw\era5\era5_rain_{YEAR}.grib"

def standardize_coords(ds):
    if ds.longitude.max() > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    return ds.sortby(["latitude", "longitude"])

def clip_colombia(da):
    return da.sel(latitude=slice(-4.5, 13.0), longitude=slice(-82.0, -66.1))

def get_land(path):
    print("Processing ERA5-Land...")
    ds = xr.open_dataset(path, engine="cfgrib",
                         backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
    ds = standardize_coords(ds)
    var_name = "tp" if "tp" in ds else list(ds.data_vars)[0]
    data = ds[var_name]
    times = data.time.values
    dates = np.array([str(t)[:10] for t in times.astype("datetime64[D]")])
    daily_totals = []
    for date in np.unique(dates):
        day = data.isel(time=dates == date)
        daily_total = day.isel(time=-1) - day.isel(time=0)
        daily_total = xr.where(daily_total < 0, 0.0, daily_total)
        daily_totals.append(daily_total)
    mean_daily = xr.concat(daily_totals, dim="day").mean(dim="day", skipna=True)
    return clip_colombia(mean_daily).load()

def get_era5(path):
    print("Processing ERA5...")
    ds = xr.open_dataset(path, engine="cfgrib",
                         backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
    ds = standardize_coords(ds)
    var_name = "tp" if "tp" in ds else list(ds.data_vars)[0]
    data = ds[var_name]
    accum_12h = data.isel(step=-1)
    times = accum_12h.time.values
    dates = np.array([str(t)[:10] for t in times.astype("datetime64[D]")])
    daily_totals = []
    for date in np.unique(dates):
        day = accum_12h.isel(time=dates == date)
        daily_totals.append(day.sum(dim="time", skipna=True))
    mean_daily = xr.concat(daily_totals, dim="day").mean(dim="day", skipna=True)
    return clip_colombia(mean_daily * 1000).load()

da_land = get_land(F_LAND)
da_era5 = get_era5(F_ERA5)

print(f"\nAverage Daily Rainfall (mm/day) — {YEAR}:")
print(f"  ERA5-Land  mean: {float(da_land.mean()):.2f}  max: {float(da_land.max()):.2f}")
print(f"  ERA5       mean: {float(da_era5.mean()):.2f}  max: {float(da_era5.max()):.2f}")

# ── Two rows: full Colombia + zoomed Chocó/Bogotá ────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 16),
                         subplot_kw={'projection': ccrs.PlateCarree()},
                         layout="constrained")

extents = {
    "Colombia":      [-82.0, -66.1, -4.5, 13.0],
    "Chocó–Bogotá":  [-78.0, -72.0,  2.0,  8.0],
}

for col, (data, title) in enumerate(zip([da_land, da_era5], ["ERA5-Land (0.1°)", "ERA5 (0.25°)"])):
    for row, (zoom_label, extent) in enumerate(extents.items()):
        ax = axes[row, col]
        im = ax.pcolormesh(data.longitude, data.latitude, data,
                           cmap='YlGnBu', vmin=0,
                           transform=ccrs.PlateCarree(), shading='auto')
        ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.5)
        ax.add_feature(cfeature.RIVERS, alpha=0.3)
        ax.coastlines(resolution='10m')
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_title(f"{title}\n{zoom_label}", fontsize=13, fontweight='bold', pad=8)
        ax.plot(-74.07, 4.71, 'ro', markersize=6, transform=ccrs.PlateCarree())  # Bogotá
        # Chocó marker
        ax.plot(-76.8, 5.5, 'r^', markersize=6, transform=ccrs.PlateCarree())

plt.colorbar(im, ax=axes, orientation='horizontal',
             label='Average Daily Precipitation (mm/day)', pad=0.04, aspect=50)
plt.suptitle(f"Mean Daily Precipitation — {YEAR}", fontsize=15, fontweight='bold')
plt.show()