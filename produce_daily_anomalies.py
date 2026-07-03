"""
Produce daily ACI-CO component data (without standardization) for all regions.

Pipeline (follows run_aci_colombia.py algorithm but stays at daily resolution):
  T90   - daily: avg(is_above_p90(day_max), is_above_p90(night_max))        [0-1]
  T10   - daily: avg(is_below_p10(day_min), is_below_p10(night_min))        [0-1]
  wind  - daily: is_above_DOY_threshold(wind_power)                          [0-1]
  prec  - daily: raw precipitation total                                      [m]
  cdd   - daily: running consecutive dry days (reset each year)              [days]
Percentile thresholds use the 1961-1990 reference period with the same
DOY-bootstrap (±15 day window) as run_aci_colombia.py.

Outputs (data/processed/daily_anomalies/):
  anomalias_diarias_{region}.xlsx  — one sheet per region
  anomalias_diarias_colombia.xlsx  — full-country time series
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

# ---------------------------------------------------------------------------
WORK_DIR  = Path(__file__).resolve().parent
DAILY_DIR = WORK_DIR / "data" / "processed" / "aci_daily"
SHP_DIR   = WORK_DIR / "data" / "shapefiles"
OUT_DIR   = WORK_DIR / "data" / "processed" / "daily_anomalies"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_START = "1961-01-01"
REF_END   = "1990-12-31"

# All regions: name → shapefile stem (in SHP_DIR)
REGIONS = {
    "colombia":              "colombia_4326",
    "cundinamarca_bogota":   "Cundinamarca_Bogota_4326",
    "antioquia":             "antioquia_4326",
    "valle_cauca":           "valle_cauca_4326",
    "san_andres_providencia":"san_andres_providencia",
    "medellin":              "medellin_4326",
    "cali":                  "cali_4326",
    "bogota":                "bogota",
    "pacifico":              "pacifico_4326",
    "amazonas":              "amazonas_4326",
}

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_daily(pattern: str) -> xr.Dataset:
    files = sorted(glob.glob(str(DAILY_DIR / pattern)))
    if not files:
        raise FileNotFoundError(f"No files matched {DAILY_DIR / pattern}")
    ds = xr.open_mfdataset(files, combine="nested", concat_dim="time")
    _, idx = np.unique(ds.time.values, return_index=True)
    return ds.isel(time=idx)


def region_mask(da: xr.DataArray, shp_stem: str) -> xr.DataArray:
    """Return boolean mask on the da grid from the shapefile."""
    shp = gpd.read_file(SHP_DIR / f"{shp_stem}.shp")
    import rioxarray  # noqa: F401 – registers the .rio accessor
    sample = da.isel(time=0).drop_vars("time", errors="ignore")
    sample = sample.rio.write_crs("EPSG:4326")
    clipped = sample.rio.clip(shp.geometry, shp.crs, drop=False, all_touched=True)
    return clipped.notnull()


# ---------------------------------------------------------------------------
# DOY-bootstrap percentile (same as run_aci_colombia.py)
# ---------------------------------------------------------------------------

def doy_percentile(da_ref: xr.DataArray, q: float, window: int = 15) -> xr.DataArray:
    """Per-DOY percentile over ±window days of the reference period."""
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
# Daily component helpers (return full spatial DataArrays – mask applied later)
# ---------------------------------------------------------------------------

def compute_t90_t10_daily(ds_tmp: xr.Dataset,
                           p90_day, p90_night,
                           p10_daymin, p10_nightmin) -> tuple:
    """
    Returns (T90_daily, T10_daily) as DataArrays with dims (time, lat, lon).
    Values are 0 or 0.5 or 1 (average of two binary exceedance flags).
    """
    doy = ds_tmp.time.dt.dayofyear
    t_p90d = p90_day.sel(dayofyear=doy).drop_vars("dayofyear")
    t_p90n = p90_night.sel(dayofyear=doy).drop_vars("dayofyear")
    t_p10d = p10_daymin.sel(dayofyear=doy).drop_vars("dayofyear")
    t_p10n = p10_nightmin.sel(dayofyear=doy).drop_vars("dayofyear")

    above_dm = xr.where(ds_tmp["day_max"]   > t_p90d, 1, 0)
    above_nm = xr.where(ds_tmp["night_max"] > t_p90n, 1, 0)
    below_dm = xr.where(ds_tmp["day_min"]   < t_p10d, 1, 0)
    below_nm = xr.where(ds_tmp["night_min"] < t_p10n, 1, 0)

    T90 = 0.5 * (above_dm + above_nm)
    T10 = 0.5 * (below_dm + below_nm)
    return T90, T10


def compute_wind_daily(ds_wind: xr.Dataset,
                       wind_thresh) -> xr.DataArray:
    """
    Binary flag (0/1) per day: 1 if daily mean wind power > DOY threshold.
    wind_thresh has dim dayofyear (366, lat, lon).
    """
    doy = ds_wind.time.dt.dayofyear
    t_w = wind_thresh.sel(dayofyear=doy).drop_vars("dayofyear")
    return xr.where(ds_wind["wind_power"] > t_w, 1, 0).astype(float)


def compute_precip_daily(tp: xr.DataArray) -> xr.DataArray:
    """Raw daily precipitation total, in metres."""
    return tp


def compute_cdd_daily(tp: xr.DataArray) -> xr.DataArray:
    """
    Running consecutive dry days (CDD) reset each calendar year.
    A day is 'dry' when tp < 0.001 m.  The value at each day is the
    number of consecutive dry days ending on (and including) that day.
    """
    cdd_list = []
    years = np.unique(tp.time.dt.year.values)
    for yr in years:
        tp_yr = tp.sel(time=str(yr))
        if tp_yr.sizes["time"] == 0:
            continue
        is_dry = xr.where(tp_yr < 0.001, 1.0, 0.0)
        cs   = is_dry.cumsum(dim="time")
        last = cs.where(is_dry == 0).ffill(dim="time").fillna(0)
        cdd_list.append(cs - last)
    return xr.concat(cdd_list, dim="time")


# ---------------------------------------------------------------------------
# Region-level spatial averaging
# ---------------------------------------------------------------------------

def spatial_mean(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Area-weighted mean over masked grid cells (equal weights here)."""
    return da.where(mask).mean(dim=["latitude", "longitude"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_region(region_name: str, shp_stem: str,
                   ds_tmp, ds_wind, tp_full,
                   p90_day, p90_night, p10_daymin, p10_nightmin,
                   wind_thresh) -> pd.DataFrame:
    print(f"  Processing {region_name}...")

    # Build mask on the temperature grid (same lat/lon as all grids)
    mask = region_mask(ds_tmp["day_max"], shp_stem)
    if not mask.any():
        print(f"    WARNING: empty mask for {region_name} — skipping")
        return pd.DataFrame()

    # Daily components (spatial)
    T90_sp, T10_sp = compute_t90_t10_daily(
        ds_tmp, p90_day, p90_night, p10_daymin, p10_nightmin)
    wind_sp  = compute_wind_daily(ds_wind, wind_thresh)
    prec_sp  = compute_precip_daily(tp_full)
    cdd_sp   = compute_cdd_daily(tp_full)

    # Spatial mean over masked region → pandas Series, join on common dates
    print(f"    Computing spatial means...")

    def to_series(da, name):
        arr = spatial_mean(da, mask).compute()
        return pd.Series(arr.values,
                         index=pd.DatetimeIndex(arr.time.values),
                         name=name)

    s_T90  = to_series(T90_sp,  "T90")
    s_T10  = to_series(T10_sp,  "T10")
    s_wind = to_series(wind_sp, "viento")
    s_prec = to_series(prec_sp, "lluvia_tp_m")
    s_cdd  = to_series(cdd_sp,  "sequia_CDD")

    df = pd.concat([s_T90, s_T10, s_wind, s_prec, s_cdd], axis=1, join="inner")
    return df


def main():
    print("=== Loading daily intermediate files ===")
    ds_tmp  = load_daily("tmp_*.nc")
    ds_wind = load_daily("wind_*.nc")
    ds_rain = load_daily("rain_*.nc")

    tp_full = ds_rain["tp"].sel(time=slice("1961-01-01", "2024-12-31"))

    # Reference period for thresholds
    ref_dm   = ds_tmp["day_max"].sel(time=slice(REF_START, REF_END)).load()
    ref_nm   = ds_tmp["night_max"].sel(time=slice(REF_START, REF_END)).load()
    ref_dmin = ds_tmp["day_min"].sel(time=slice(REF_START, REF_END)).load()
    ref_nmin = ds_tmp["night_min"].sel(time=slice(REF_START, REF_END)).load()
    ref_wp   = ds_wind["wind_power"].sel(time=slice(REF_START, REF_END)).load()

    print("=== Computing DOY thresholds (1961–1990) ===")
    print("  T90 day-max..."); p90_day   = doy_percentile(ref_dm,   90)
    print("  T90 night-max..."); p90_night = doy_percentile(ref_nm,   90)
    print("  T10 day-min...");  p10_daymin  = doy_percentile(ref_dmin, 10)
    print("  T10 night-min..."); p10_nightmin = doy_percentile(ref_nmin, 10)
    print("  Wind DOY threshold (mean+1.28*std)...")
    doy_mean_wp = ref_wp.groupby("time.dayofyear").mean().load()
    doy_std_wp  = ref_wp.groupby("time.dayofyear").std().load()
    wind_thresh = doy_mean_wp + 1.28 * doy_std_wp

    print("\n=== Processing regions ===")
    all_dfs = {}
    for region_name, shp_stem in REGIONS.items():
        shp_path = SHP_DIR / f"{shp_stem}.shp"
        if not shp_path.exists():
            print(f"  SKIP {region_name}: shapefile not found ({shp_path})")
            continue
        df = compute_region(
            region_name, shp_stem,
            ds_tmp, ds_wind, tp_full,
            p90_day, p90_night, p10_daymin, p10_nightmin, wind_thresh,
        )
        if df.empty:
            continue
        all_dfs[region_name] = df

        # Per-region Excel
        out_xlsx = OUT_DIR / f"anomalias_diarias_{region_name}.xlsx"
        df.to_excel(out_xlsx)
        print(f"    Saved {out_xlsx.name}")

    # Country-level Excel (Colombia sheet already in all_dfs)
    country_df = all_dfs.get("colombia", pd.DataFrame())
    if not country_df.empty:
        out_co = OUT_DIR / "anomalias_diarias_colombia.xlsx"
        country_df.to_excel(out_co)
        print(f"\nSaved country file: {out_co}")

    # Combined Excel – all regions as separate sheets
    out_all = OUT_DIR / "anomalias_diarias_todas_las_regiones.xlsx"
    with pd.ExcelWriter(out_all, engine="openpyxl") as writer:
        for rname, df in all_dfs.items():
            sheet = rname[:31]          # Excel sheet names ≤ 31 chars
            df.to_excel(writer, sheet_name=sheet)
    print(f"Saved combined file (all regions): {out_all}")

    print("\nDone.")


if __name__ == "__main__":
    main()
