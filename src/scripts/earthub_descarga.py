    import xarray as xr
    import dask
    from dask.callbacks import Callback
    import time
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from tqdm.auto import tqdm
    import eccodes

    # ── Config ───────────────────────────────────────────────────────────────────
    PAT = "edh_pat_15b8e3b9a0f1fb8e6763c78757d5dd321b74f4f667f1406b5b490e984ecd120b7857183fe0106d703b156dce3fbefc26"
    OUTPUT_DIR = "../../data/raw/era5_land/"
    YEARS = list(range(1961, 2025))

    # Colombia bounding box
    LAT_MAX, LON_MIN, LAT_MIN, LON_MAX = 13.0, -83.0, -4.6, -66.1

    # ERA5 stores lon as 0–360, convert
    LON_MIN_360 = LON_MIN % 360  # -83.0  → 277.0
    LON_MAX_360 = LON_MAX % 360  # -66.1  → 293.9

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Variable metadata for GRIB2 encoding ─────────────────────────────────────
    GRIB_META = {
        "temperature":   {"slug": "tmp",      "paramId": 167, "typeOfLevel": "heightAboveGround", "level": 2},
        "precipitation": {"slug": "rain",     "paramId": 228, "typeOfLevel": "surface",            "level": 0},
        "wind_u10":      {"slug": "wind_u10", "paramId": 165, "typeOfLevel": "heightAboveGround", "level": 10},
        "wind_v10":      {"slug": "wind_v10", "paramId": 166, "typeOfLevel": "heightAboveGround", "level": 10},
    }


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


    # ── GRIB2 writer using eccodes ────────────────────────────────────────────────
    def save_year_grib(result, label, year):
        meta      = GRIB_META[label]
        slug      = meta["slug"]
        grib_path = os.path.join(OUTPUT_DIR, f"era5land_{slug}_{year}.grib")

        lats  = result.latitude.values.astype(float)
        lons  = result.longitude.values.astype(float)
        times = result.valid_time.values
        data  = result.values  # (time, lat, lon)

        n_lat = len(lats)
        n_lon = len(lons)
        dlat  = abs(float(lats[1] - lats[0]))
        dlon  = abs(float(lons[1] - lons[0]))

        with open(grib_path, "wb") as f:
            for i, t in enumerate(tqdm(
                times,
                desc=f"    writing {year}",
                unit="msg",
                leave=False,
                position=2,
            )):
                dt   = pd.Timestamp(t)
                flat = data[i].flatten().astype(np.float64)

                nan_mask    = np.isnan(flat)
                has_missing = nan_mask.any()
                if has_missing:
                    bitmap = (~nan_mask).astype(np.int32)
                    flat[nan_mask] = 0.0

                sample = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
                try:
                    eccodes.codes_set(sample, "edition",    2)
                    eccodes.codes_set(sample, "centre",     98)
                    eccodes.codes_set(sample, "subCentre",  0)

                    eccodes.codes_set(sample, "gridType",  "regular_ll")
                    eccodes.codes_set(sample, "Ni",         n_lon)
                    eccodes.codes_set(sample, "Nj",         n_lat)
                    eccodes.codes_set(sample, "latitudeOfFirstGridPointInDegrees",  float(lats[0]))
                    eccodes.codes_set(sample, "longitudeOfFirstGridPointInDegrees", float(lons[0]))
                    eccodes.codes_set(sample, "latitudeOfLastGridPointInDegrees",   float(lats[-1]))
                    eccodes.codes_set(sample, "longitudeOfLastGridPointInDegrees",  float(lons[-1]))
                    eccodes.codes_set(sample, "iDirectionIncrementInDegrees",       dlon)
                    eccodes.codes_set(sample, "jDirectionIncrementInDegrees",       dlat)
                    eccodes.codes_set(sample, "jScansPositively",
                                    0 if lats[0] > lats[-1] else 1)

                    if has_missing:
                        eccodes.codes_set(sample, "bitmapPresent", 1)
                        eccodes.codes_set_values(sample, bitmap.tolist())
                        eccodes.codes_set(sample, "bitmapPresent", 1)
                    else:
                        eccodes.codes_set(sample, "bitmapPresent", 0)

                    eccodes.codes_set(sample, "paramId",     meta["paramId"])
                    eccodes.codes_set(sample, "typeOfLevel", meta["typeOfLevel"])
                    eccodes.codes_set(sample, "level",       meta["level"])

                    eccodes.codes_set(sample, "dataDate",  int(dt.strftime("%Y%m%d")))
                    eccodes.codes_set(sample, "dataTime",  dt.hour * 100)
                    eccodes.codes_set(sample, "stepRange", "0")

                    try:
                        eccodes.codes_set(sample, "packingType", "grid_ccsds")
                    except eccodes.EncodingError:
                        eccodes.codes_set(sample, "packingType", "grid_simple")

                    eccodes.codes_set_values(sample, flat.tolist())
                    eccodes.codes_write(sample, f)

                finally:
                    eccodes.codes_release(sample)

        return grib_path


    # ── Check if year already cached ─────────────────────────────────────────────
    def year_cached(label, year):
        slug      = GRIB_META[label]["slug"]
        grib_path = os.path.join(OUTPUT_DIR, f"era5land_{slug}_{year}.grib")
        nc_path   = grib_path.replace(".grib", ".nc")
        return os.path.exists(grib_path) or os.path.exists(nc_path)


    # ── Download per-year GRIB files, no concatenation ───────────────────────────
    def download_all_years(da, label, years, retries=5, wait=15):
        overall = tqdm(
            total=len(years),
            desc=f"Overall [{label}]",
            unit="yr",
            colour="green",
            position=0,
            bar_format="{l_bar}{bar}| {n}/{total} years [{elapsed}<{remaining}]",
        )

        for year in years:

            # ── Skip if already on disk ────────────────────────────────────────
            if year_cached(label, year):
                overall.set_postfix(last=f"{year} (cached)")
                overall.update(1)
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

                    saved_path = save_year_grib(result, label, year)
                    print(f"\n  ✓ Saved: {saved_path}")
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


    # ── Open remote dataset ───────────────────────────────────────────────────────
    print("Opening remote Zarr dataset...")
    ds = xr.open_dataset(
        f"https://edh:{PAT}@data.earthdatahub.destine.eu/era5/reanalysis-era5-land-no-antartica-v0.zarr",
        chunks={},
        engine="zarr",
    ).astype("float32")

    xr.set_options(keep_attrs=True)

    # ── Crop to region (0–360 lon) ────────────────────────────────────────────────
    print("Cropping to region...")
    ds_region = ds.sel(
        latitude=slice(LAT_MAX, LAT_MIN),
        longitude=slice(LON_MIN_360, LON_MAX_360),
    )

    # Convert longitudes back to -180/180 for plotting
    ds_region = ds_region.assign_coords(
        longitude=(ds_region.longitude + 180) % 360 - 180
    )

    print(f"Grid size : {dict(ds_region.sizes)}")
    print(f"Latitude  : {float(ds_region.latitude.min()):.2f} → {float(ds_region.latitude.max()):.2f}")
    print(f"Longitude : {float(ds_region.longitude.min()):.2f} → {float(ds_region.longitude.max()):.2f}")

    # ── Variable definitions ──────────────────────────────────────────────────────
    t2m = ds_region["t2m"] - 273.15
    t2m.attrs["units"] = "°C"
    t2m.attrs["long_name"] = "2m Temperature"

    tp = ds_region["tp"] * 1000
    tp.attrs["units"] = "mm"
    tp.attrs["long_name"] = "Total Precipitation"

    u10 = ds_region["u10"]
    v10 = ds_region["v10"]

    # ── Download all variables (tmp already done, will be skipped) ────────────────
    # download_all_years(t2m, "temperature",   YEARS)  # ← already downloaded, uncomment if needed
    download_all_years(tp,  "precipitation", YEARS)
    download_all_years(u10, "wind_u10",      YEARS)
    download_all_years(v10, "wind_v10",      YEARS)

    print("\n✓ All done.")