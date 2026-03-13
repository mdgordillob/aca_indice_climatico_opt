import xarray as xr
import dask
from dask.callbacks import Callback
import time
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm.auto import tqdm
import cfgrib

# ── Config ────────────────────────────────────────────────────────────────────
PAT        = "edh_pat_15b8e3b9a0f1fb8e6763c78757d5dd321b74f4f667f1406b5b490e984ecd120b7857183fe0106d703b156dce3fbefc26"
OUTPUT_DIR = "../../data/raw/era5_land/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tp_greece_sept_1993_2023.nc")   # full merged cache
YEARS      = list(range(1993, 2024))

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Real dask-chunk progress bar ──────────────────────────────────────────────
class TqdmDaskCallback(Callback):
    def __init__(self, **tqdm_kwargs):
        self._tqdm_kwargs = tqdm_kwargs
        self.pbar = None

    def _start(self, dsk):
        self.pbar = tqdm(total=len(dsk), **self._tqdm_kwargs)

    def _posttask(self, key, result, dsk, state, worker_id):
        self.pbar.update(1)

    def _finish(self, dsk, state, errored):
        if self.pbar:
            self.pbar.close()


# ── Download year by year, save each as .grib ─────────────────────────────────
def download_by_year(da, years, label="var", retries=5, wait=15):
    """
    Downloads one year at a time and immediately saves each year to:
        era5land_rain_{year}.grib
    If a .grib for that year already exists it is skipped (safe to resume).
    Returns the full xr.DataArray (all years concatenated).
    """
    chunks = []

    overall = tqdm(
        total=len(years),
        desc=f"Overall [{label}]",
        unit="yr",
        colour="green",
        position=0,
        bar_format="{l_bar}{bar}| {n}/{total} years [{elapsed}<{remaining}]",
    )

    for year in years:
        grib_path = os.path.join(OUTPUT_DIR, f"era5land_rain_{year}.grib")

        # ── Resume: skip years that are already on disk ────────────────────
        if os.path.exists(grib_path):
            overall.set_postfix(last=f"{year} (cached)")
            overall.update(1)
            try:
                cached = xr.open_dataarray(grib_path, engine="cfgrib")
                chunks.append(cached)
            except Exception as e:
                print(f"\n  Warning: could not re-open {grib_path}: {e}")
            continue

        year_data = da.sel(valid_time=str(year))

        for attempt in range(retries):
            try:
                cb = TqdmDaskCallback(
                    desc=f"  {year}   ",
                    unit="chunk",
                    colour="blue",
                    position=1,
                    leave=False,
                    bar_format="{l_bar}{bar}| {n}/{total} chunks [{elapsed}<{remaining}, {rate_fmt}]",
                )
                with cb:
                    with dask.config.set(scheduler="synchronous"):
                        result = year_data.compute()

                # ── Save year as .grib ─────────────────────────────────────
                try:
                    cfgrib.xarray_store.to_grib(
                        result.to_dataset(name="tp"),
                        grib_path,
                    )
                    print(f"\n  Saved: {grib_path}")
                except Exception as grib_err:
                    # Fallback: save as .nc if GRIB writing fails
                    nc_path = grib_path.replace(".grib", ".nc")
                    print(f"\n  GRIB write failed ({grib_err}). Saving as NetCDF: {nc_path}")
                    result.to_netcdf(nc_path)

                chunks.append(result)
                overall.set_postfix(last=f"{year} ✓", attempt=f"{attempt+1}/{retries}")
                overall.update(1)
                break

            except Exception as e:
                overall.set_postfix(last=f"{year} ✗", retry=f"{attempt+1}/{retries}")
                if attempt < retries - 1:
                    time.sleep(wait)
                else:
                    overall.close()
                    raise RuntimeError(
                        f"Year {year} failed after {retries} attempts: {e}"
                    )

    overall.close()
    print("\nConcatenating years...")
    return xr.concat(chunks, dim="valid_time")


# ── Map helper ────────────────────────────────────────────────────────────────
def make_map(data, projection, vmax, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": projection}, figsize=(10, 6)
        )
    ax.set_extent([19, 28, 34, 41], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
    im = ax.pcolormesh(
        data.longitude, data.latitude, data.values,
        transform=ccrs.PlateCarree(),
        cmap="Blues", vmin=0, vmax=vmax,
    )
    plt.colorbar(im, ax=ax, label="mm", shrink=0.7)
    ax.set_title(title, fontsize=12)
    return ax


# ── Open remote dataset ───────────────────────────────────────────────────────
print("Opening remote Zarr dataset...")
ds = xr.open_dataset(
    f"https://edh:{PAT}@data.earthdatahub.destine.eu/era5/reanalysis-era5-land-no-antartica-v0.zarr",
    chunks={},
    engine="zarr",
).astype("float32")

xr.set_options(keep_attrs=True)

# ── Preprocessing ─────────────────────────────────────────────────────────────
print("Preprocessing...")
tp = ds.tp * 1000
tp.attrs["units"] = "mm"

tp_greece           = tp.sel(latitude=slice(41, 34), longitude=slice(19, 28))
tp_greece_sept      = tp_greece[tp_greece.valid_time.dt.month.isin([9])]
tp_greece_sept_full = tp_greece_sept.sel(valid_time=slice("1993", "2023"))

total_mb = tp_greece_sept_full.nbytes / (1024 ** 2)
print(f"Estimated download: {total_mb:.1f} MiB across {len(YEARS)} years\n")

# ── Download / load from merged cache ────────────────────────────────────────
if os.path.exists(OUTPUT_FILE):
    print(f"Merged cache found — loading from '{OUTPUT_FILE}'")
    tp_computed = xr.open_dataarray(OUTPUT_FILE)
else:
    print("No merged cache — starting download (per-year .grib files)...")
    tp_computed = download_by_year(
        tp_greece_sept_full, YEARS, label="rain", retries=5, wait=15
    )
    print(f"\nSaving merged file to '{OUTPUT_FILE}'...")
    tp_computed.to_netcdf(OUTPUT_FILE)
    print("Saved. Future runs will skip all downloads.")

# ── Storm Daniel ──────────────────────────────────────────────────────────────
print("\nComputing Storm Daniel precipitation...")
tp_storm_daniel = (
    tp_computed
    .sel(valid_time=["2023-09-06", "2023-09-07"])
    .sum("valid_time")
)

# ── 30-year September mean ────────────────────────────────────────────────────
print("Computing 30-year September mean...")
tp_sept_mean = (
    tp_computed[tp_computed["valid_time"].dt.hour == 0]
    .sel(valid_time=slice("1993", "2022"))
    .sum("valid_time")
    / 30
)

# ── Plots ─────────────────────────────────────────────────────────────────────
plt.style.use("bmh")

print("\nPlotting Storm Daniel map...")
fig1, ax1 = plt.subplots(
    subplot_kw={"projection": ccrs.Miller()}, figsize=(10, 6)
)
make_map(
    tp_storm_daniel, ccrs.Miller(), 400,
    "Storm Daniel total precipitation, 6–7 September 2023",
    ax=ax1,
)
ax1.gridlines(draw_labels=True, alpha=0.2)
plt.tight_layout()
plt.savefig("storm_daniel.png", dpi=150)
plt.show()

print("Plotting comparison maps...")
fig2, axs = plt.subplots(
    1, 2, subplot_kw={"projection": ccrs.Miller()}, figsize=(16, 6)
)
make_map(
    tp_storm_daniel, ccrs.Miller(), 400,
    "Storm Daniel precipitation, 6–7 September 2023",
    ax=axs[0],
)
make_map(
    tp_sept_mean, ccrs.Miller(), 400,
    "Average September precipitation, 1993–2022",
    ax=axs[1],
)
for ax in axs:
    ax.gridlines(draw_labels=True, alpha=0.2)
plt.tight_layout()
plt.savefig("comparison.png", dpi=150)
plt.show()

print("\nDone.")