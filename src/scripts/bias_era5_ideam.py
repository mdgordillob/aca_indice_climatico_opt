import os
import xarray as xr
import cfgrib
import pyreadr
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.interpolate import RegularGridInterpolator

# ── Paths ─────────────────────────────────────────────────────────────────────
IDEAM_DIR     = r"C:\Users\mdgor\aca\aca_indice_climatico_opt-main\data\raw\ideam"
ERA5_LAND_DIR = r"C:\Users\mdgor\data\raw\era5_land"
ERA5_DIR      = r"C:\Users\mdgor\aca\aca_indice_climatico_opt-main\data\raw\era5"
OUTPUT_DIR    = r"C:\Users\mdgor\aca\aca_indice_climatico_opt-main\data\processed\era5_ideam_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
VAR_MAP = {
    "TSSM_CON": dict(slug="tmp",  grib_var="t2m", scale=1, offset=0, agg="mean", label="Temperature (°C)",    era5_dir="land"),
    "PTPM_CON": dict(slug="rain", grib_var="tp",  scale=1, offset=0, agg="sum",  label="Precipitation (mm)", era5_dir="era5"),
}

PERIODS = [
    dict(label="2010",      years=[2010]),
    dict(label="2020-2024", years=list(range(2020, 2025))),
]

# ── IDEAM ─────────────────────────────────────────────────────────────────────
def load_ideam_monthly(tag, period_label):
    path = os.path.join(IDEAM_DIR, f"ideam_{tag}_{period_label}.rds")
    if not os.path.exists(path):
        print(f"  ✗ Missing: {os.path.basename(path)}")
        return None
    df = pyreadr.read_r(path)[None]
    df["date"]      = pd.to_datetime(df["date"])
    df["month"]     = df["date"].dt.strftime("%Y-%m")
    df["longitude"] = df["longitude"].astype(float)
    df["latitude"]  = df["latitude"].astype(float)
    df["value"]     = df["value"].astype(float)
    monthly = (
        df.groupby(["station", "longitude", "latitude", "tag", "month"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "obs"})
    )
    return monthly

# ── ERA5 ──────────────────────────────────────────────────────────────────────
def load_era5_monthly(slug, grib_var, years, scale, offset, agg, era5_dir):
    monthly_arrays = []

    for yr in years:
        if era5_dir == "land":
            path = os.path.join(ERA5_LAND_DIR, f"era5land_{slug}_{yr}.grib")
        else:
            path = os.path.join(ERA5_DIR, f"era5_{slug}_{yr}.grib")

        if not os.path.exists(path):
            print(f"  ✗ Missing ERA5: {os.path.basename(path)}")
            continue
        print(f"  Loading {os.path.basename(path)}...", flush=True)

        if era5_dir == "land":
            ds = xr.open_dataset(
                path,
                engine="cfgrib",
                backend_kwargs={"indexpath": "", "errors": "ignore"},
                filter_by_keys={"edition": 2}
            )
            ds = ds.assign_coords(longitude=(ds.longitude - 360))
            ds = ds.sel(
                latitude=slice(13, -4.6),
                longitude=slice(-82.9, -66.1)
            )
            da = ds[grib_var] * scale + offset
            monthly_arrays.append(da.resample(time="ME").mean())

        else:
            datasets = cfgrib.open_datasets(
                path, backend_kwargs={"errors": "ignore"}
            )
            ds = datasets[0]
            da = ds[grib_var]

            step_vals = da.values                                          # (time, step, lat, lon)
            hourly    = np.diff(step_vals, axis=1,
                                prepend=np.zeros_like(step_vals[:, :1, :, :]))
            hourly    = np.clip(hourly, 0, None)
            run_total = hourly.sum(axis=1)                                 # (time, lat, lon)

            da_run = xr.DataArray(
                run_total,
                coords={"time": ds.time, "latitude": ds.latitude, "longitude": ds.longitude},
                dims=["time", "latitude", "longitude"]
            )
            monthly_arrays.append((da_run * 1000).resample(time="ME").sum())

    if not monthly_arrays:
        return None
    return xr.concat(monthly_arrays, dim="time")

# ── Extraction ────────────────────────────────────────────────────────────────
def extract_era5_at_stations(da, stations_df):
    lats        = stations_df["latitude"].values.astype(float)
    lons        = stations_df["longitude"].values.astype(float)
    station_ids = stations_df["station"].values

    da_lat_vals = da.latitude.values.astype(float)
    da_lon_vals = da.longitude.values.astype(float)

    ascending   = da_lat_vals[0] < da_lat_vals[-1]
    grid_lats   = da_lat_vals if ascending else da_lat_vals[::-1]
    grid_lons   = da_lon_vals

    lat_idx = np.clip(
        np.array([np.argmin(np.abs(da_lat_vals - lat)) for lat in lats]),
        0, len(da_lat_vals) - 1
    )
    lon_idx = np.clip(
        np.array([np.argmin(np.abs(da_lon_vals - lon)) for lon in lons]),
        0, len(da_lon_vals) - 1
    )

    records = []
    for t in da.time.values:
        slab         = da.sel(time=t).values.astype(float)
        nn_vals      = slab[lat_idx, lon_idx]
        slab_flipped = slab if ascending else slab[::-1, :]
        interp_fn    = RegularGridInterpolator(
            (grid_lats, grid_lons), slab_flipped,
            method="linear", bounds_error=False, fill_value=np.nan
        )
        bil_vals  = interp_fn(np.column_stack([lats, lons]))
        month_str = pd.Timestamp(t).strftime("%Y-%m")

        for i, sid in enumerate(station_ids):
            records.append({
                "station":  sid,
                "month":    month_str,
                "era5_nn":  float(nn_vals[i]),
                "era5_bil": float(bil_vals[i]),
            })

    return pd.DataFrame(records)

# ── Stats ─────────────────────────────────────────────────────────────────────
def compute_stats(df, era5_col):
    rows = []
    for keys, grp in df.groupby(["station", "longitude", "latitude", "tag", "period"]):
        valid = grp[["obs", era5_col]].dropna()
        n = len(valid)
        if n < 2:
            continue
        if valid["obs"].std() == 0 or valid[era5_col].std() == 0:
            r2 = np.nan
        else:
            r2 = pearsonr(valid["obs"], valid[era5_col])[0] ** 2
        bias = (valid[era5_col] - valid["obs"]).mean()
        rmse = np.sqrt(((valid[era5_col] - valid["obs"]) ** 2).mean())
        rows.append(dict(
            station=keys[0], longitude=keys[1], latitude=keys[2],
            tag=keys[3], period=keys[4], n=n, bias=bias, rmse=rmse, r2=r2
        ))
    return pd.DataFrame(rows)

# ── Main ──────────────────────────────────────────────────────────────────────
all_comparison = []

for tag, meta in VAR_MAP.items():
    print(f"\n{'='*60}\n{tag}\n{'='*60}")

    for period in PERIODS:
        print(f"\n── Period: {period['label']} ──")

        ideam_mon = load_ideam_monthly(tag, period["label"])
        if ideam_mon is None:
            continue
        print(f"  ✓ IDEAM: {len(ideam_mon)} station-months from {ideam_mon['station'].nunique()} stations")

        da = load_era5_monthly(
            meta["slug"], meta["grib_var"], period["years"],
            meta["scale"], meta["offset"], meta["agg"], meta["era5_dir"]
        )
        if da is None:
            print(f"  ✗ No ERA5 data, skipping")
            continue
        print(f"  ✓ ERA5 loaded: {len(da.time)} months")
        print(f"    sample value (center, t=0): {float(da.isel(time=0, latitude=len(da.latitude)//2, longitude=len(da.longitude)//2).values):.4f}")

        stations_unique = (
            ideam_mon[["station", "longitude", "latitude"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        print(f"  Extracting ERA5 at {len(stations_unique)} stations...")
        era5_extracted = extract_era5_at_stations(da, stations_unique)

        comparison = ideam_mon.merge(era5_extracted, on=["station", "month"], how="left")
        comparison["period"] = period["label"]
        print(f"  ✓ Joined: {len(comparison)} rows")
        print(f"    IDEAM obs range:   {comparison['obs'].min():.2f} – {comparison['obs'].max():.2f}")
        print(f"    ERA5 nn range:     {comparison['era5_nn'].min():.2f} – {comparison['era5_nn'].max():.2f}")
        print(f"    ERA5 bil range:    {comparison['era5_bil'].min():.2f} – {comparison['era5_bil'].max():.2f}")
        all_comparison.append(comparison)

# ── Save ──────────────────────────────────────────────────────────────────────
if all_comparison:
    full_df = pd.concat(all_comparison, ignore_index=True)
    full_df.to_csv(os.path.join(OUTPUT_DIR, "era5_ideam_comparison.csv"), index=False)

    for col in ["era5_nn", "era5_bil"]:
        stats = compute_stats(full_df, col)
        stats.to_csv(os.path.join(OUTPUT_DIR, f"stats_{col}.csv"), index=False)
        print(f"\n── Stats {col} ──")
        print(stats[["tag", "period", "bias", "rmse", "r2"]].groupby(["tag", "period"]).mean().round(3))

    print(f"\n✓ All saved to {OUTPUT_DIR}")
else:
    print("\n✗ No data processed")