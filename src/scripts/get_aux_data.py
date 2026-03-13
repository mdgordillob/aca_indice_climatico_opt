# src/scripts/get_auxiliary_data.py
import os
import numpy as np
import xarray as xr
import rioxarray
import planetary_computer
import pystac_client
import stackstac
import pandas as pd
from rasterio.enums import Resampling

OUTPUT_DIR = r"C:\Users\mdgor\aca\aca_indice_climatico_opt-main\data\raw\auxiliary"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOMBIA_BBOX = [-83, -5, -66, 13]  # minx, miny, maxx, maxy

# ── 1. DEM ────────────────────────────────────────────────────────────────────
print("Fetching DEM tiles...")
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)

search = catalog.search(
    collections=["cop-dem-glo-30"],
    bbox=COLOMBIA_BBOX
)
items = list(search.items())
print(f"  Found {len(items)} tiles")

print("  Loading and mosaicking DEM (this may take a few minutes)...")
dem = (
    stackstac.stack(
        items,
        assets=["data"],
        bounds_latlon=COLOMBIA_BBOX,
        resolution=0.01,
        resampling=Resampling.bilinear,
        epsg=4326,
        dtype="float64",
        fill_value=np.nan,
        rescale=False,
    )
    .squeeze()
    .median(dim="time")
    .compute()
)
print(f"  DEM shape: {dem.shape}")
print(f"  Elevation range: {float(dem.min()):.0f} – {float(dem.max()):.0f} m")

# ── 2. Slope & Aspect (from DEM) ──────────────────────────────────────────────
print("Computing slope and aspect...")

elev = dem.values.astype(float)
res  = 0.01 * 111320  # approx metres per degree at equator

# Gradient in x (lon) and y (lat) directions
dy, dx = np.gradient(elev, res, res)

slope  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
aspect = np.degrees(np.arctan2(-dx, dy)) % 360

slope_da = xr.DataArray(
    slope.astype("float32"),
    coords={"y": dem.y, "x": dem.x},
    dims=["y", "x"]
)
aspect_da = xr.DataArray(
    aspect.astype("float32"),
    coords={"y": dem.y, "x": dem.x},
    dims=["y", "x"]
)

print(f"  Slope range:  {slope.min():.1f} – {slope.max():.1f} degrees")
print(f"  Aspect range: {aspect.min():.1f} – {aspect.max():.1f} degrees")

# ── 3. Save ───────────────────────────────────────────────────────────────────
print("Saving...")
dem.rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs("EPSG:4326").rio.to_raster(
    os.path.join(OUTPUT_DIR, "dem_colombia_001deg.tif")
)
slope_da.rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs("EPSG:4326").rio.to_raster(
    os.path.join(OUTPUT_DIR, "slope_colombia_001deg.tif")
)
aspect_da.rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs("EPSG:4326").rio.to_raster(
    os.path.join(OUTPUT_DIR, "aspect_colombia_001deg.tif")
)

print(f"\n✓ Saved DEM, slope, aspect to {OUTPUT_DIR}")