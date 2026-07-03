"""
Runs ACI-Python's algorithms on raw ERA5-Land GRIB data (1961-2024) for Colombia.
No sea level component. Reference period: 1961-1990.

Algorithm faithfully follows ACI-Python-main:
  - Temperature T90: fraction of days where daily max > P90(DOY reference)
                     day-hours (6-21) and night-hours (0-5, 22-23) averaged
  - Temperature T10: fraction of days where daily min < P10(DOY reference)
                     same day/night split, averaged
  - Wind:  wind_power = 0.5*1.23*sqrt(u10²+v10²)³  →  daily mean
           → fraction above DOY threshold (ref mean + 1.28*ref std) → standardise
  - Precipitation: daily tp sum → 5-day rolling max → monthly max → standardise
  - Drought: daily tp → CDD (< 0.001 m) → annual max → linear monthly interp → standardise
  ICA = (T90 - T10 + wind + precipitation + drought) / 5

Two-phase approach to stay within memory limits:
  Phase 1 - per-year preprocessing: converts raw GRIB → daily NetCDF intermediates
            stored in data/processed/aci_daily/  (skipped if file already exists)
  Phase 2 - component calculation: opens all daily files with open_mfdataset (lazy)
            computes reference statistics, then standardised monthly anomalies

Output: data/processed/aci_colombia_1961_2024.csv
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
WORK_DIR  = Path(__file__).resolve().parent
GRIB_DIR  = WORK_DIR / "data" / "raw" / "era5"
PROC_DIR  = WORK_DIR / "data" / "processed"
DAILY_DIR = PROC_DIR / "aci_daily"      # intermediate per-year daily files
MASK_PATH = PROC_DIR / "colombia_era5_mask.nc"
OUT_CSV   = PROC_DIR / "aci_colombia_1961_2024.csv"

REF_START  = "1961-01-01"
REF_END    = "1990-12-31"
YEARS      = list(range(1961, 2025))
UTC_OFFSET = pd.Timedelta(hours=-5)     # Colombia UTC-5

DAY_HOURS   = list(range(6, 22))
NIGHT_HOURS = [0, 1, 2, 3, 4, 5, 22, 23]


# ---------------------------------------------------------------------------
# Phase 1 helpers — build one yearly daily NetCDF per variable
# ---------------------------------------------------------------------------

def _grib(pattern: str) -> list[str]:
    return sorted(glob.glob(str(GRIB_DIR / pattern)))


def preprocess_temperature_year(year: int, out_path: Path):
    """Saves (day_max, day_min, night_max, night_min) daily grids for one year."""
    grib = _grib(f"era5_tmp_{year}.grib")
    if not grib:
        return False
    ds = xr.open_dataset(grib[0], engine="cfgrib",
                         backend_kwargs={"indexpath": ""})
    t2m = ds["t2m"]
    t2m["time"] = t2m.indexes["time"] + UTC_OFFSET

    t_day   = t2m.isel(time=t2m.time.dt.hour.isin(DAY_HOURS))
    t_night = t2m.isel(time=t2m.time.dt.hour.isin(NIGHT_HOURS))

    out = xr.Dataset({
        "day_max":   t_day.resample(time="1D").max(),
        "day_min":   t_day.resample(time="1D").min(),
        "night_max": t_night.resample(time="1D").max(),
        "night_min": t_night.resample(time="1D").min(),
    })
    enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    out.to_netcdf(out_path, encoding=enc)
    ds.close()
    return True


def preprocess_wind_year(year: int, out_path: Path):
    """Saves daily mean wind power for one year."""
    grib = _grib(f"era5_wind_{year}.grib")
    if not grib:
        return False
    ds = xr.open_dataset(grib[0], engine="cfgrib",
                         backend_kwargs={"indexpath": ""})
    u10 = ds["u10"]
    v10 = ds["v10"]
    u10["time"] = u10.indexes["time"] + UTC_OFFSET
    v10["time"] = v10.indexes["time"] + UTC_OFFSET

    ws = np.sqrt(u10 ** 2 + v10 ** 2)
    daily_mean_ws = ws.resample(time="1D").mean()          # daily mean wind speed
    daily_wp = (0.5 * 1.23 * daily_mean_ws ** 3).rename("wind_power")

    enc = {"wind_power": {"zlib": True, "complevel": 4}}
    daily_wp.to_dataset().to_netcdf(out_path, encoding=enc)
    ds.close()
    return True


def preprocess_rain_year(year: int, out_path: Path):
    """Saves daily total precipitation (m) for one year."""
    grib = _grib(f"era5_rain_{year}.grib")
    if not grib:
        return False
    ds = xr.open_dataset(grib[0], engine="cfgrib",
                         backend_kwargs={"indexpath": ""})
    tp   = ds["tp"]                   # (time, step, lat, lon)
    vt   = ds["valid_time"].values    # (time, step)

    # Flatten to (obs, lat, lon) indexed by valid_time
    flat_vt = vt.ravel() + np.timedelta64(-5, "h")   # UTC-5
    flat_tp = tp.values.reshape(-1, tp.shape[2], tp.shape[3])

    tp_da = xr.DataArray(
        flat_tp,
        dims=["time", "latitude", "longitude"],
        coords={"time": flat_vt,
                "latitude": ds.latitude.values,
                "longitude": ds.longitude.values},
    ).sortby("time")

    tp_daily = tp_da.resample(time="1D").sum().rename("tp")
    enc = {"tp": {"zlib": True, "complevel": 4}}
    tp_daily.to_dataset().to_netcdf(out_path, encoding=enc)
    ds.close()
    return True


def phase1_preprocess():
    """Ensures all per-year daily files exist.  Skips years that are cached."""
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        tp = DAILY_DIR / f"tmp_{year}.nc"
        wp = DAILY_DIR / f"wind_{year}.nc"
        rp = DAILY_DIR / f"rain_{year}.nc"

        if not tp.exists():
            print(f"  preprocessing temperature {year}...", end=" ", flush=True)
            ok = preprocess_temperature_year(year, tp)
            print("done" if ok else "MISSING GRIB")
        if not wp.exists():
            print(f"  preprocessing wind {year}...", end=" ", flush=True)
            ok = preprocess_wind_year(year, wp)
            print("done" if ok else "MISSING GRIB")
        if not rp.exists():
            print(f"  preprocessing rain {year}...", end=" ", flush=True)
            ok = preprocess_rain_year(year, rp)
            print("done" if ok else "MISSING GRIB")


# ---------------------------------------------------------------------------
# Phase 2 helpers — compute components from daily files
# ---------------------------------------------------------------------------

def load_daily(pattern: str) -> xr.Dataset:
    files = sorted(glob.glob(str(DAILY_DIR / pattern)))
    ds = xr.open_mfdataset(files, combine="nested", concat_dim="time")
    # Drop duplicate timestamps that arise from UTC-5 overlap at year boundaries
    _, idx = np.unique(ds.time.values, return_index=True)
    return ds.isel(time=idx)


def apply_mask(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    m = mask.interp_like(da.isel(time=0), method="nearest") >= 0.5
    return da.where(m)


def doy_percentile(da_ref: xr.DataArray, q: float, window: int = 15) -> xr.DataArray:
    """Bootstrap per-DOY percentile over ±window days from reference period."""
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


def standardise(da: xr.DataArray) -> xr.DataArray:
    ref = da.sel(time=slice(REF_START, REF_END))
    mean = ref.groupby("time.month").mean()
    std  = ref.groupby("time.month").std()
    std  = std.where(std > 0)   # avoid ÷0 for pixels with no variability
    mi   = da.time.dt.month
    return ((da - mean.sel(month=mi)) / std.sel(month=mi)).drop_vars("month")


# ---------------------------------------------------------------------------
# Component computations
# ---------------------------------------------------------------------------

def compute_temperature(mask_da):
    print("  Loading temperature daily files...")
    ds = load_daily("tmp_*.nc")

    day_max   = apply_mask(ds["day_max"],   mask_da)
    day_min   = apply_mask(ds["day_min"],   mask_da)
    night_max = apply_mask(ds["night_max"], mask_da)
    night_min = apply_mask(ds["night_min"], mask_da)

    # Reference period data for thresholds
    ref_dm  = day_max.sel(time=slice(REF_START, REF_END)).load()
    ref_nm  = night_max.sel(time=slice(REF_START, REF_END)).load()
    ref_dmin = day_min.sel(time=slice(REF_START, REF_END)).load()
    ref_nmin = night_min.sel(time=slice(REF_START, REF_END)).load()

    print("  Computing T90 DOY thresholds...")
    p90_day   = doy_percentile(ref_dm,   90)
    p90_night = doy_percentile(ref_nm,   90)
    print("  Computing T10 DOY thresholds...")
    p10_day   = doy_percentile(ref_dmin, 10)
    p10_night = doy_percentile(ref_nmin, 10)

    def exceedance_fraction(daily_da, thresh, above: bool):
        doy_idx = daily_da.time.dt.dayofyear
        t = thresh.sel(dayofyear=doy_idx).drop_vars("dayofyear")
        flag = xr.where(daily_da > t, 1, 0) if above else xr.where(daily_da < t, 1, 0)
        return (flag.resample(time="ME").sum() /
                flag.resample(time="ME").count())

    t90 = 0.5 * (exceedance_fraction(day_max,   p90_day,   True)
               + exceedance_fraction(night_max, p90_night, True))
    t10 = 0.5 * (exceedance_fraction(day_min,   p10_day,   False)
               + exceedance_fraction(night_min, p10_night, False))

    print("  Standardising temperature...")
    t90_std = standardise(t90).mean(dim=["latitude", "longitude"]).compute()
    t10_std = standardise(t10).mean(dim=["latitude", "longitude"]).compute()

    idx = pd.DatetimeIndex(t90_std.time.values).to_period("M").to_timestamp(how="S")
    return pd.DataFrame({"t90": t90_std.values, "t10": t10_std.values}, index=idx)


def compute_wind(mask_da):
    print("  Loading wind daily files...")
    ds = load_daily("wind_*.nc")
    wp = apply_mask(ds["wind_power"], mask_da)

    # DOY threshold: ref mean + 1.28*ref std (= P90 under normality)
    wp_ref   = wp.sel(time=slice(REF_START, REF_END))
    doy_mean = wp_ref.groupby("time.dayofyear").mean().load()
    doy_std  = wp_ref.groupby("time.dayofyear").std().load()
    thresh   = doy_mean + 1.28 * doy_std

    doy_idx = wp.time.dt.dayofyear
    t = thresh.sel(dayofyear=doy_idx).drop_vars("dayofyear")
    above = xr.where(wp > t, 1, 0)
    freq  = (above.resample(time="ME").sum() /
             above.resample(time="ME").count())

    print("  Standardising wind...")
    wind_std = standardise(freq).mean(dim=["latitude", "longitude"]).compute()
    idx = pd.DatetimeIndex(wind_std.time.values).to_period("M").to_timestamp(how="S")
    return pd.Series(wind_std.values, index=idx, name="wind")


def compute_precipitation(mask_da):
    print("  Loading precipitation daily files...")
    ds = load_daily("rain_*.nc")
    tp = apply_mask(ds["tp"], mask_da)
    # Trim to avoid UTC-5 bleed creating a spurious Dec-1960 bucket
    tp = tp.sel(time=slice("1961-01-01", "2024-12-31"))

    rolling5   = tp.rolling(time=5, min_periods=1).sum()
    monthly_max = rolling5.resample(time="ME").max()

    print("  Standardising precipitation...")
    prec_std = standardise(monthly_max).mean(dim=["latitude", "longitude"]).compute()
    idx = pd.DatetimeIndex(prec_std.time.values).to_period("M").to_timestamp(how="S")
    return pd.Series(prec_std.values, index=idx, name="precipitation")


def compute_drought(mask_da):
    print("  Loading precipitation daily files for drought...")
    ds = load_daily("rain_*.nc")
    tp = apply_mask(ds["tp"], mask_da)
    tp = tp.sel(time=slice("1961-01-01", "2024-12-31"))

    # CDD: max consecutive dry days per year, reset at Jan 1 (matches OPT approach)
    annual_cdd_list = []
    for yr in range(1961, 2025):
        tp_yr = tp.sel(time=str(yr))
        if tp_yr.sizes["time"] == 0:
            continue
        d = xr.where(tp_yr < 0.001, 1, 0)
        cs = d.cumsum(dim="time")
        run = cs - cs.where(d == 0).ffill(dim="time").fillna(0)
        cdd_yr = run.max(dim="time")
        cdd_yr = cdd_yr.expand_dims(
            time=[np.datetime64(f"{yr}-12-31", "ns")])
        annual_cdd_list.append(cdd_yr)

    annual_cdd = xr.concat(annual_cdd_list, dim="time").load()

    # Interpolate to monthly using xarray resample — backward-looking like OPT:
    # Dec of year i = CDD_i, then Jan–Nov of year i+1 blend toward CDD_{i+1}
    monthly_cdd = annual_cdd.resample(time="ME").interpolate("linear")

    print("  Standardising drought...")
    drgt_std = standardise(monthly_cdd).mean(dim=["latitude", "longitude"])
    idx = pd.DatetimeIndex(drgt_std.time.values).to_period("M").to_timestamp(how="S")
    return pd.Series(drgt_std.values, index=idx, name="drought")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if OUT_CSV.exists():
        print(f"Cached result found: {OUT_CSV}")
        print("Delete it to recompute.")
        return pd.read_csv(OUT_CSV, index_col=0, parse_dates=True)

    print("=== Phase 1: preprocessing raw GRIB -> daily NetCDF ===")
    phase1_preprocess()
    # Restrict to years with all 3 variables ready
    tmp_yrs  = {int(f.stem.split("_")[1]) for f in DAILY_DIR.glob("tmp_*.nc")}
    wind_yrs = {int(f.stem.split("_")[1]) for f in DAILY_DIR.glob("wind_*.nc")}
    rain_yrs = {int(f.stem.split("_")[1]) for f in DAILY_DIR.glob("rain_*.nc")}
    available = sorted(tmp_yrs & wind_yrs & rain_yrs)
    print(f"Available complete years: {available[0]}-{available[-1]} ({len(available)} years)")

    print("\n=== Phase 2: computing ACI components ===")
    mask_ds = xr.open_dataset(str(MASK_PATH))
    mask_da = mask_ds["country"].astype(float)

    df_temp = compute_temperature(mask_da)
    s_wind  = compute_wind(mask_da)
    s_prec  = compute_precipitation(mask_da)
    s_drgt  = compute_drought(mask_da)

    df = df_temp.join(s_wind).join(s_prec).join(s_drgt)
    df["ACI"] = (df["t90"] - df["t10"] + df["wind"] +
                 df["precipitation"] + df["drought"]) / 5

    df.to_csv(OUT_CSV)
    print(f"\nSaved: {OUT_CSV}")
    print(df.head())
    return df


if __name__ == "__main__":
    main()
