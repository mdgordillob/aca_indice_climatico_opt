"""
Appends 2025-2026 daily anomaly data to existing .xlsx files.

2025/2026 GRIBs use a newer format: (time=daily, step=24_hourly_offsets, lat, lon)
at 0.1 degree resolution. This script stacks via valid_time, applies UTC-5, and
regrids to the 0.25 degree reference grid before computing daily stats.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

WORK_DIR  = Path(__file__).resolve().parent
GRIB_DIR  = WORK_DIR / "data" / "raw" / "era5"
DAILY_DIR = WORK_DIR / "data" / "processed" / "aci_daily"
SHP_DIR   = WORK_DIR / "data" / "shapefiles"
OUT_DIR   = WORK_DIR / "data" / "processed" / "daily_anomalies"

REF_START   = "1961-01-01"
REF_END     = "1990-12-31"
NEW_YEARS   = [2025, 2026]
DAY_HOURS   = list(range(6, 22))
NIGHT_HOURS = [0, 1, 2, 3, 4, 5, 22, 23]

REGIONS = {
    "colombia":               "colombia_4326",
    "cundinamarca_bogota":    "Cundinamarca_Bogota_4326",
    "antioquia":              "antioquia_4326",
    "valle_cauca":            "valle_cauca_4326",
    "san_andres_providencia": "san_andres_providencia",
    "medellin":               "medellin_4326",
    "cali":                   "cali_4326",
    "bogota":                 "bogota",
    "pacifico":               "pacifico_4326",
    "amazonas":               "amazonas_4326",
}

# Reference 0.25-degree grid (loaded once from existing daily file)
_REF = xr.open_dataset(str(DAILY_DIR / "tmp_1961.nc")).isel(time=0)
REF_LAT = _REF.latitude.values
REF_LON = _REF.longitude.values


# ---------------------------------------------------------------------------
# Helpers for new GRIB format (time=daily, step=hourly_offsets)
# ---------------------------------------------------------------------------

def stack_hourly(ds, varname):
    """
    Convert (time, step, lat, lon) with valid_time to flat (hourly, lat, lon)
    shifted to Colombia local time (UTC-5).
    """
    da  = ds[varname]
    vt  = ds["valid_time"].values                         # (time, step)
    flat_vt = vt.ravel() + np.timedelta64(-5, "h")       # UTC-5
    flat_d  = da.values.reshape(-1, da.shape[-2], da.shape[-1])
    return xr.DataArray(
        flat_d,
        dims=["time", "latitude", "longitude"],
        coords={"time": flat_vt,
                "latitude": da.latitude.values,
                "longitude": da.longitude.values},
    ).sortby("time")


def regrid(da):
    """Interpolate from 0.1-degree to reference 0.25-degree grid."""
    return da.interp(latitude=REF_LAT, longitude=REF_LON, method="linear")


def open_grib(path):
    return xr.open_dataset(str(path), engine="cfgrib",
                           backend_kwargs={"indexpath": ""})


# ---------------------------------------------------------------------------
# Phase 1 — preprocess new years to daily 0.25-degree NetCDF
# ---------------------------------------------------------------------------

def preprocess_temperature_year(year):
    out = DAILY_DIR / f"tmp_{year}.nc"
    if out.exists():
        print(f"  tmp_{year}.nc already exists — skipping"); return
    grib = list(GRIB_DIR.glob(f"era5_tmp_{year}.grib"))
    if not grib:
        print(f"  WARNING: no temperature GRIB for {year}"); return

    ds  = open_grib(grib[0])
    t2m = stack_hourly(ds, "t2m")
    t2m = regrid(t2m)

    t_day   = t2m.isel(time=t2m.time.dt.hour.isin(DAY_HOURS))
    t_night = t2m.isel(time=t2m.time.dt.hour.isin(NIGHT_HOURS))

    out_ds = xr.Dataset({
        "day_max":   t_day.resample(time="1D").max(),
        "day_min":   t_day.resample(time="1D").min(),
        "night_max": t_night.resample(time="1D").max(),
        "night_min": t_night.resample(time="1D").min(),
    })
    enc = {v: {"zlib": True, "complevel": 4} for v in out_ds.data_vars}
    out_ds.to_netcdf(out, encoding=enc)
    ds.close()
    print(f"  tmp_{year}.nc saved")


def preprocess_wind_year(year):
    out = DAILY_DIR / f"wind_{year}.nc"
    if out.exists():
        print(f"  wind_{year}.nc already exists — skipping"); return
    grib = list(GRIB_DIR.glob(f"era5_wind_{year}.grib"))
    if not grib:
        print(f"  WARNING: no wind GRIB for {year}"); return

    ds  = open_grib(grib[0])
    u10 = regrid(stack_hourly(ds, "u10"))
    v10 = regrid(stack_hourly(ds, "v10"))

    ws       = np.sqrt(u10 ** 2 + v10 ** 2)
    daily_wp = (0.5 * 1.23 * ws.resample(time="1D").mean() ** 3).rename("wind_power")
    daily_wp.to_dataset().to_netcdf(
        out, encoding={"wind_power": {"zlib": True, "complevel": 4}})
    ds.close()
    print(f"  wind_{year}.nc saved")


def preprocess_rain_year(year):
    out = DAILY_DIR / f"rain_{year}.nc"
    if out.exists():
        print(f"  rain_{year}.nc already exists — skipping"); return
    grib = list(GRIB_DIR.glob(f"era5_rain_{year}.grib"))
    if not grib:
        print(f"  WARNING: no rain GRIB for {year}"); return

    ds = open_grib(grib[0])
    tp = regrid(stack_hourly(ds, "tp"))
    tp_daily = tp.resample(time="1D").sum().rename("tp")
    tp_daily.to_dataset().to_netcdf(
        out, encoding={"tp": {"zlib": True, "complevel": 4}})
    ds.close()
    print(f"  rain_{year}.nc saved")


# ---------------------------------------------------------------------------
# Loaders (subset of years)
# ---------------------------------------------------------------------------

def load_years(pattern_tpl, years):
    files = sorted([str(DAILY_DIR / pattern_tpl.format(yr)) for yr in years
                    if (DAILY_DIR / pattern_tpl.format(yr)).exists()])
    if not files:
        raise FileNotFoundError(f"No files found for {pattern_tpl}")
    ds = xr.open_mfdataset(files, combine="nested", concat_dim="time")
    _, idx = np.unique(ds.time.values, return_index=True)
    return ds.isel(time=idx)


# ---------------------------------------------------------------------------
# DOY percentile threshold
# ---------------------------------------------------------------------------

def doy_percentile(da_ref, q, window=15):
    doy = da_ref.time.dt.dayofyear.values
    thresholds = []
    for d in range(1, 367):
        lo, hi = d - window, d + window
        if lo < 1:
            mask = (doy >= lo + 366) | (doy <= hi)
        elif hi > 366:
            mask = (doy >= lo) | (doy <= hi - 366)
        else:
            mask = (doy >= lo) & (doy <= hi)
        subset = da_ref.isel(time=mask)
        t = subset.quantile(q / 100.0, dim="time").drop_vars("quantile")
        thresholds.append(t.expand_dims(dayofyear=[d]))
    return xr.concat(thresholds, dim="dayofyear")


# ---------------------------------------------------------------------------
# Daily component computations
# ---------------------------------------------------------------------------

def t90_t10_daily(ds_tmp, p90_day, p90_night, p10_daymin, p10_nightmin):
    doy = ds_tmp.time.dt.dayofyear
    T90 = 0.5 * (
        xr.where(ds_tmp["day_max"]   > p90_day.sel(dayofyear=doy).drop_vars("dayofyear"),   1, 0)
      + xr.where(ds_tmp["night_max"] > p90_night.sel(dayofyear=doy).drop_vars("dayofyear"), 1, 0))
    T10 = 0.5 * (
        xr.where(ds_tmp["day_min"]   < p10_daymin.sel(dayofyear=doy).drop_vars("dayofyear"),   1, 0)
      + xr.where(ds_tmp["night_min"] < p10_nightmin.sel(dayofyear=doy).drop_vars("dayofyear"), 1, 0))
    return T90, T10


def wind_daily(ds_wind, wind_thresh):
    doy = ds_wind.time.dt.dayofyear
    return xr.where(
        ds_wind["wind_power"] > wind_thresh.sel(dayofyear=doy).drop_vars("dayofyear"), 1, 0
    ).astype(float)


def cdd_daily(tp):
    cdd_list = []
    for yr in np.unique(tp.time.dt.year.values):
        tp_yr = tp.sel(time=str(yr))
        if tp_yr.sizes["time"] == 0:
            continue
        is_dry = xr.where(tp_yr < 0.001, 1.0, 0.0)
        cs   = is_dry.cumsum(dim="time")
        last = cs.where(is_dry == 0).ffill(dim="time").fillna(0)
        cdd_list.append(cs - last)
    return xr.concat(cdd_list, dim="time")


# ---------------------------------------------------------------------------
# Region mask + spatial mean
# ---------------------------------------------------------------------------

def region_mask(da, shp_stem):
    import rioxarray  # noqa
    shp    = gpd.read_file(SHP_DIR / f"{shp_stem}.shp")
    sample = da.isel(time=0).drop_vars("time", errors="ignore")
    sample = sample.rio.write_crs("EPSG:4326")
    return sample.rio.clip(shp.geometry, shp.crs, drop=False, all_touched=True).notnull()


def to_series(da, mask, name):
    arr = da.where(mask).mean(dim=["latitude", "longitude"]).compute()
    return pd.Series(arr.values, index=pd.DatetimeIndex(arr.time.values), name=name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Phase 1: preprocessing 2025-2026 ===")
    for yr in NEW_YEARS:
        preprocess_temperature_year(yr)
        preprocess_wind_year(yr)
        preprocess_rain_year(yr)

    ready = [yr for yr in NEW_YEARS
             if (DAILY_DIR / f"tmp_{yr}.nc").exists()
             and (DAILY_DIR / f"wind_{yr}.nc").exists()
             and (DAILY_DIR / f"rain_{yr}.nc").exists()]
    if not ready:
        print("No new years fully ready — exiting."); return
    print(f"Ready years: {ready}")

    print("\n=== Phase 2: DOY thresholds (1961-1990) ===")
    ref_years   = list(range(1961, 1991))
    ds_ref_tmp  = load_years("tmp_{}.nc",  ref_years)
    ds_ref_wind = load_years("wind_{}.nc", ref_years)

    print("  T90 day-max...");    p90_day      = doy_percentile(ds_ref_tmp["day_max"].load(),   90)
    print("  T90 night-max...");  p90_night    = doy_percentile(ds_ref_tmp["night_max"].load(), 90)
    print("  T10 day-min...");    p10_daymin   = doy_percentile(ds_ref_tmp["day_min"].load(),   10)
    print("  T10 night-min...");  p10_nightmin = doy_percentile(ds_ref_tmp["night_min"].load(), 10)
    print("  Wind threshold...")
    ref_wp      = ds_ref_wind["wind_power"].load()
    wind_thresh = (ref_wp.groupby("time.dayofyear").mean().load()
                 + 1.28 * ref_wp.groupby("time.dayofyear").std().load())

    print("\n=== Phase 3: computing components for new years ===")
    ds_new_tmp  = load_years("tmp_{}.nc",  ready)
    ds_new_wind = load_years("wind_{}.nc", ready)
    ds_new_rain = load_years("rain_{}.nc", ready)
    tp_new = ds_new_rain["tp"]

    T90_sp, T10_sp = t90_t10_daily(ds_new_tmp, p90_day, p90_night, p10_daymin, p10_nightmin)
    wind_sp = wind_daily(ds_new_wind, wind_thresh)
    cdd_sp  = cdd_daily(tp_new)

    print("\n=== Phase 4: appending to .xlsx files ===")
    for region, shp_stem in REGIONS.items():
        shp_path = SHP_DIR / f"{shp_stem}.shp"
        xlsx     = OUT_DIR / f"anomalias_diarias_{region}.xlsx"
        if not shp_path.exists():
            print(f"  SKIP {region}: no shapefile"); continue
        if not xlsx.exists():
            print(f"  SKIP {region}: no existing xlsx"); continue

        print(f"  {region}...")
        mask = region_mask(ds_new_tmp["day_max"], shp_stem)
        if not mask.any():
            print(f"    empty mask — skipping"); continue

        new_df = pd.concat([
            to_series(T90_sp,  mask, "T90"),
            to_series(T10_sp,  mask, "T10"),
            to_series(wind_sp, mask, "viento"),
            to_series(tp_new,  mask, "lluvia_tp_m"),
            to_series(cdd_sp,  mask, "sequia_CDD"),
        ], axis=1, join="inner")

        existing = pd.read_excel(xlsx, index_col=0, parse_dates=True)
        new_only  = new_df[~new_df.index.isin(existing.index)]
        combined  = pd.concat([existing, new_only]).sort_index()
        combined.to_excel(xlsx)
        print(f"    +{len(new_only)} rows → {len(combined)} total")

    print("\nRebuilding combined workbook...")
    out_all = OUT_DIR / "anomalias_diarias_todas_las_regiones.xlsx"
    with pd.ExcelWriter(out_all, engine="openpyxl") as writer:
        for region in REGIONS:
            xlsx = OUT_DIR / f"anomalias_diarias_{region}.xlsx"
            if xlsx.exists():
                df = pd.read_excel(xlsx, index_col=0, parse_dates=True)
                df.to_excel(writer, sheet_name=region[:31])
    print(f"Saved: {out_all}")
    print("Done.")


if __name__ == "__main__":
    main()
