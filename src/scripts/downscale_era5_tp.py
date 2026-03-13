# src/scripts/downscale_era5_tp.py
import os
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import pyreadr
import cfgrib
import warnings
from scipy.interpolate import RegularGridInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
AUX_DIR       = r"C:\Users\mdgor\aca\aca_indice_climatico_opt-main\data\raw\auxiliary"
IDEAM_DIR     = r"C:\Users\mdgor\aca\aca_indice_climatico_opt-main\data\raw\ideam"
ERA5_LAND_DIR = r"C:\Users\mdgor\data\raw\era5_land"
OUTPUT_DIR    = r"C:\Users\mdgor\aca\aca_indice_climatico_opt-main\data\processed\downscaled"
PLOT_DIR      = r"C:\Users\mdgor\aca\aca_indice_climatico_opt-main\data\processed\plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)

COLOMBIA_BBOX = dict(lat_min=-4.6, lat_max=13.0, lon_min=-82.9, lon_max=-66.1)
YEARS         = [2010, 2024]

# ── 1. Load auxiliary data ────────────────────────────────────────────────────
print("Loading auxiliary data...")
aux_files = {
    "dem":    "dem_colombia_001deg.tif",
    "slope":  "slope_colombia_001deg.tif",
    "aspect": "aspect_colombia_001deg.tif",
    "ndvi":   "ndvi_colombia_001deg.tif",
    "ndbi":   "ndbi_colombia_001deg.tif",
    "mndwi":  "mndwi_colombia_001deg.tif",
}
aux = {}
for name, fname in aux_files.items():
    da = rioxarray.open_rasterio(os.path.join(AUX_DIR, fname)).squeeze()
    aux[name] = da
ref_lats  = aux["dem"].y.values.astype(float)
ref_lons  = aux["dem"].x.values.astype(float)
aux_stack = np.stack([aux[k].values.astype(float) for k in aux_files], axis=-1)

dem_vals   = aux_stack[:, :, 0]
valid_mask = np.all(np.isfinite(aux_stack), axis=-1) & (dem_vals > 0)
print(f"  Fine grid: {len(ref_lats)} lats x {len(ref_lons)} lons")
print(f"  Valid land pixels: {valid_mask.sum():,}")

# ── 2. Load ERA5-Land precipitation (0.1°) ────────────────────────────────────
print("\nLoading ERA5-Land precipitation (0.1°)...")

def load_era5land_tp_monthly(years):
    arrays = []
    for yr in years:
        path = os.path.join(ERA5_LAND_DIR, f"era5land_rain_{yr}.grib")
        print(f"  Reading {os.path.basename(path)}...")
        ds    = cfgrib.open_datasets(path)[0]
        arr   = ds["tp"].values
        times = pd.to_datetime(ds.valid_time.values)
        lats  = ds.latitude.values.astype(float)
        lons  = ds.longitude.values.astype(float)

        if lons.max() > 180:
            lons = lons - 360

        lat_ok = (lats >= COLOMBIA_BBOX["lat_min"]) & (lats <= COLOMBIA_BBOX["lat_max"])
        lon_ok = (lons >= COLOMBIA_BBOX["lon_min"]) & (lons <= COLOMBIA_BBOX["lon_max"])
        arr    = arr[:, lat_ok, :][:, :, lon_ok]
        lats   = lats[lat_ok]
        lons   = lons[lon_ok]

        # Daily total = value at hour==00 of next day (full day accumulation)
        # assigned to previous day. Drop index 0 (Jan 1 00:00 = prior year carryover)
        mask_00     = times.hour == 0
        arr_00      = arr[mask_00][1:]
        times_00    = times[mask_00][1:]
        times_daily = (pd.DatetimeIndex(times_00) - pd.Timedelta(days=1)).normalize()
        arr_daily   = arr_00

        print(f"    {yr}: {arr_daily.shape[0]} daily values, "
              f"range {arr_daily.min():.1f}–{arr_daily.max():.1f} mm, "
              f"mean {arr_daily.mean():.2f} mm/day")

        da = xr.DataArray(
            arr_daily,
            coords={"time": times_daily, "latitude": lats, "longitude": lons},
            dims=["time", "latitude", "longitude"]
        )
        monthly = da.resample(time="ME").sum()
        arrays.append(monthly)

    return xr.concat(arrays, dim="time")

era5_tp = load_era5land_tp_monthly(YEARS)

stub_mask = np.array([float(era5_tp.isel(time=i).mean()) < 1.0
                      for i in range(len(era5_tp.time))])
era5_tp   = era5_tp.isel(time=~stub_mask)
months    = [str(t)[:7] for t in era5_tp.time.values]
era5_lats = era5_tp.latitude.values.astype(float)
era5_lons = era5_tp.longitude.values.astype(float)

print(f"  ERA5-Land tp shape: {era5_tp.shape} — {len(months)} months")
print(f"  Months: {months[0]} → {months[-1]}")
print(f"  ERA5-Land tp range: {float(era5_tp.min()):.1f} – {float(era5_tp.max()):.1f} mm")

# ── 3. Load IDEAM precipitation ───────────────────────────────────────────────
print("\nLoading IDEAM precipitation...")

def load_ideam_tp(years):
    dfs = []
    for yr in years:
        period = str(yr) if yr == 2010 else "2020-2024"
        path   = os.path.join(IDEAM_DIR, f"ideam_PTPM_CON_{period}.rds")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        df = pyreadr.read_r(path)[None]
        df["date"]      = pd.to_datetime(df["date"])
        df["year"]      = df["date"].dt.year
        df              = df[df["year"] == yr]
        df["month"]     = df["date"].dt.strftime("%Y-%m")
        df["longitude"] = df["longitude"].astype(float)
        df["latitude"]  = df["latitude"].astype(float)
        df["value"]     = df["value"].astype(float)
        dfs.append(df)
    all_df    = pd.concat(dfs, ignore_index=True)
    ideam_mon = (
        all_df.groupby(["station", "longitude", "latitude", "month"])["value"]
        .sum().reset_index().rename(columns={"value": "obs"})
    )
    return ideam_mon

ideam_mon = load_ideam_tp(YEARS)
print(f"  {len(ideam_mon)} station-months from {ideam_mon['station'].nunique()} stations")
print(f"  obs range: {ideam_mon['obs'].min():.1f} – {ideam_mon['obs'].max():.1f} mm")

# ── 4. Interpolate ERA5-Land to fine grid ─────────────────────────────────────
def interp_era5_to_fine(era5_slice, ref_lats, ref_lons):
    lats = era5_slice.latitude.values.astype(float)
    lons = era5_slice.longitude.values.astype(float)
    vals = era5_slice.values.astype(float)
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        vals = vals[::-1, :]
    interp_fn = RegularGridInterpolator(
        (lats, lons), vals, method="linear", bounds_error=False, fill_value=np.nan
    )
    grid_lons, grid_lats = np.meshgrid(ref_lons, ref_lats)
    pts = np.column_stack([grid_lats.ravel(), grid_lons.ravel()])
    return interp_fn(pts).reshape(len(ref_lats), len(ref_lons))

# ── 5. Build training data ────────────────────────────────────────────────────
print("\nBuilding training dataset...")

def get_aux_at_point(lat, lon):
    li = np.argmin(np.abs(ref_lats - lat))
    lo = np.argmin(np.abs(ref_lons - lon))
    return aux_stack[li, lo, :]

records = []
for _, row in ideam_mon.iterrows():
    if row["month"] not in months:
        continue
    era5_slice = era5_tp.sel(time=pd.Timestamp(row["month"] + "-01"), method="nearest")
    li       = np.argmin(np.abs(era5_lats - row["latitude"]))
    lo       = np.argmin(np.abs(era5_lons - row["longitude"]))
    era5_val = float(era5_slice.values[li, lo])
    aux_vals = get_aux_at_point(row["latitude"], row["longitude"])
    if np.all(np.isfinite(aux_vals)) and np.isfinite(era5_val) and np.isfinite(row["obs"]):
        records.append({
            "month":  row["month"],
            "obs":    row["obs"],
            "era5":   era5_val,
            "dem":    aux_vals[0],
            "slope":  aux_vals[1],
            "aspect": aux_vals[2],
            "ndvi":   aux_vals[3],
            "ndbi":   aux_vals[4],
            "mndwi":  aux_vals[5],
        })

train_df = pd.DataFrame(records).dropna()
p999     = train_df["obs"].quantile(0.999)
train_df = train_df[train_df["obs"] <= p999]
print(f"  Training samples: {len(train_df)} (obs clipped at {p999:.1f} mm)")
print(f"  obs range:   {train_df['obs'].min():.1f} – {train_df['obs'].max():.1f} mm")
print(f"  era5 range (at station locations): {train_df['era5'].min():.1f} – {train_df['era5'].max():.1f} mm")
print(f"  dem range (at station locations):  {train_df['dem'].min():.0f} – {train_df['dem'].max():.0f} m")
print(f"  mean residual (obs - era5): {(train_df['obs'] - train_df['era5']).mean():.1f} mm")

# ── 6. Train RF on multiplicative ratio (obs / era5) ─────────────────────────
print("\nTraining Random Forest on correction ratio (obs / ERA5-Land)...")
features = ["era5", "dem", "slope", "aspect", "ndvi", "ndbi", "mndwi"]

# Only train where era5 > 1mm to avoid division instability
train_rf  = train_df[train_df["era5"] > 1.0].copy()
train_rf["ratio"] = train_rf["obs"] / train_rf["era5"]

# Clip extreme ratios
ratio_p01 = train_rf["ratio"].quantile(0.01)
ratio_p99 = train_rf["ratio"].quantile(0.99)
train_rf  = train_rf[
    (train_rf["ratio"] >= ratio_p01) &
    (train_rf["ratio"] <= ratio_p99)
]

print(f"  RF training samples: {len(train_rf)}")
print(f"  Ratio range (p01–p99): {ratio_p01:.3f} – {ratio_p99:.3f}")
print(f"  Ratio mean: {train_rf['ratio'].mean():.3f}  (1.0 = perfect ERA5)")

X       = train_rf[features].values
y_ratio = train_rf["ratio"].values

rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf.fit(X, y_ratio)

ratio_pred   = rf.predict(X)
y_pred_final = train_rf["era5"].values * ratio_pred
print(f"  Training R² (obs vs ERA5*RF):  {r2_score(train_rf['obs'].values, y_pred_final):.3f}")
print(f"  Training R² (obs vs ERA5 raw): {r2_score(train_rf['obs'].values, train_rf['era5'].values):.3f}")
print(f"  Training RMSE: {np.sqrt(mean_squared_error(train_rf['obs'].values, y_pred_final)):.2f} mm")
print(f"  Mean ratio learned: {ratio_pred.mean():.3f}")
print("  Feature importances:")
for feat, imp in sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"    {feat:8s}: {imp:.3f}")

# ── 7. Predict on full fine grid ──────────────────────────────────────────────
print("\nDownscaling ERA5-Land precipitation to 0.01° for all months...")
aux_flat         = aux_stack[valid_mask]
downscaled_maps  = {}
era5_interp_maps = {}

for i, month in enumerate(months):
    era5_slice = era5_tp.isel(time=i)
    era5_fine  = interp_era5_to_fine(era5_slice, ref_lats, ref_lons)
    era5_interp_maps[month] = era5_fine

    era5_flat  = era5_fine[valid_mask].reshape(-1, 1)
    X_pred     = np.hstack([era5_flat, aux_flat])
    ratio_pred = rf.predict(X_pred)

    # Clip ratio to training range
    ratio_pred  = np.clip(ratio_pred, ratio_p01, ratio_p99)

    # Where ERA5 is near zero, skip RF and use ERA5 directly
    era5_vals_flat = era5_fine[valid_mask]
    y_pred_flat    = np.where(
        era5_vals_flat > 1.0,
        era5_vals_flat * ratio_pred,
        era5_vals_flat
    )
    y_pred_flat = np.clip(y_pred_flat, 0, None)

    result = np.full((len(ref_lats), len(ref_lons)), np.nan)
    result[valid_mask] = y_pred_flat
    downscaled_maps[month] = result

    print(f"  {month}: ERA5-Land mean={float(era5_slice.mean()):.1f}mm "
          f"→ downscaled mean={np.nanmean(result):.1f}mm  "
          f"ratio mean={ratio_pred.mean():.2f}")

# ── 8. Save downscaled NetCDF ─────────────────────────────────────────────────
print("\nSaving downscaled maps...")
times  = pd.to_datetime([m + "-01" for m in months])
data   = np.stack([downscaled_maps[m] for m in months], axis=0)
ds_out = xr.Dataset({
    "tp_downscaled": xr.DataArray(
        data.astype("float32"),
        coords={"time": times, "lat": ref_lats, "lon": ref_lons},
        dims=["time", "lat", "lon"],
        attrs={"units": "mm", "long_name": "Downscaled monthly precipitation"}
    )
})
ds_out.to_netcdf(os.path.join(OUTPUT_DIR, "era5land_downscaled_tp_2010_2024.nc"))
print(f"  Saved NetCDF to {OUTPUT_DIR}")

# ── 9. Colombia comparison maps (4 months) ────────────────────────────────────
print("\nGenerating Colombia comparison maps...")
plot_months = ["2010-01", "2010-04", "2010-07", "2010-10"]
month_names = {"2010-01": "January", "2010-04": "April",
               "2010-07": "July",    "2010-10": "October"}

all_vals = np.concatenate([
    era5_interp_maps[m][valid_mask] for m in plot_months if m in era5_interp_maps
])
vmin = 0
vmax = float(np.nanpercentile(all_vals, 95))
print(f"  Colorscale: 0 – {vmax:.0f} mm (95th pct)")

fig = plt.figure(figsize=(20, 24))
fig.suptitle("ERA5-Land Precipitation Downscaling — Colombia 2010\n"
             "Raw ERA5-Land (0.1°) | Bilinear Interp (0.01°) | RF Downscaled (0.01°) | Station scatter",
             fontsize=14, fontweight="bold", y=0.98)

for row_idx, month in enumerate(plot_months):
    era5_slice  = era5_tp.sel(time=pd.Timestamp(month + "-01"), method="nearest")
    era5_coarse = era5_slice.values.astype(float)
    era5_fine   = era5_interp_maps[month]
    ds_fine     = downscaled_maps[month]
    ideam_pts   = ideam_mon[ideam_mon["month"] == month]

    for col_idx, (data_grid, lats_g, lons_g, title) in enumerate([
        (era5_coarse, era5_slice.latitude.values, era5_slice.longitude.values, "Raw ERA5-Land (0.1°)"),
        (era5_fine,   ref_lats, ref_lons,          "Bilinear Interp (0.01°)"),
        (ds_fine,     ref_lats, ref_lons,          "RF Downscaled (0.01°)"),
    ]):
        ax = fig.add_subplot(4, 4, row_idx * 4 + col_idx + 1)
        im = ax.pcolormesh(lons_g, lats_g, data_grid,
                           cmap="YlGnBu", vmin=vmin, vmax=vmax)
        ax.scatter(ideam_pts["longitude"], ideam_pts["latitude"],
                   c=ideam_pts["obs"], cmap="YlGnBu", vmin=vmin, vmax=vmax,
                   s=15, edgecolors="k", linewidths=0.3, zorder=5)
        ax.set_title(f"{month_names[month]}\n{title}", fontsize=9)
        ax.set_xlim(-82.9, -66.1); ax.set_ylim(-4.6, 13)
        plt.colorbar(im, ax=ax, shrink=0.8, label="mm")

    ax4 = fig.add_subplot(4, 4, row_idx * 4 + 4)
    e5_at_stn, bi_at_stn, ds_at_stn = [], [], []
    for _, stn in ideam_pts.iterrows():
        li   = np.argmin(np.abs(ref_lats  - stn["latitude"]))
        lo   = np.argmin(np.abs(ref_lons  - stn["longitude"]))
        li_e = np.argmin(np.abs(era5_lats - stn["latitude"]))
        lo_e = np.argmin(np.abs(era5_lons - stn["longitude"]))
        e5_at_stn.append(float(era5_slice.values[li_e, lo_e]))
        bi_at_stn.append(float(era5_fine[li, lo]))
        ds_at_stn.append(float(ds_fine[li, lo]))

    obs_v = ideam_pts["obs"].values
    e5_v  = np.array(e5_at_stn)
    bi_v  = np.array(bi_at_stn)
    ds_v  = np.array(ds_at_stn)
    valid = np.isfinite(obs_v) & np.isfinite(e5_v) & np.isfinite(ds_v)
    if valid.sum() > 2:
        all_v = np.concatenate([obs_v[valid], e5_v[valid], bi_v[valid], ds_v[valid]])
        lims  = [0, all_v.max() + 5]
        ax4.plot(lims, lims, "k--", lw=1)
        for vals, label, color in [
            (e5_v, "ERA5-Land", "steelblue"),
            (bi_v, "Bilinear",  "darkorange"),
            (ds_v, "RF",        "tomato"),
        ]:
            v = valid & np.isfinite(vals)
            if v.sum() > 1:
                r2 = r2_score(obs_v[v], vals[v])
                ax4.scatter(obs_v[v], vals[v], s=10, alpha=0.5,
                            color=color, label=f"{label} R²={r2:.2f}")
        ax4.set_xlim(lims); ax4.set_ylim(lims)
        ax4.legend(fontsize=7)
    ax4.set_xlabel("IDEAM obs (mm)", fontsize=8)
    ax4.set_ylabel("ERA5 / RF (mm)", fontsize=8)
    ax4.set_title(f"{month_names[month]}\nStation scatter", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(PLOT_DIR, "downscaling_tp_comparison_2010.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved Colombia comparison plot")

# ── 10. Bogota monthly comparison — one row per month ────────────────────────
print("\nGenerating Bogota monthly precipitation plot...")
BOG_LAT, BOG_LON = 4.7, -74.1
BOG_DELTA = 2.0

bog_stations = ideam_mon[
    (ideam_mon["latitude"].between(BOG_LAT - 0.5, BOG_LAT + 0.5)) &
    (ideam_mon["longitude"].between(BOG_LON - 0.5, BOG_LON + 0.5))
]
bog_monthly = bog_stations.groupby("month")["obs"].mean().reset_index()
print(f"  {bog_stations['station'].nunique()} IDEAM stations near Bogota")

lat_mask      = (ref_lats  >= BOG_LAT - BOG_DELTA) & (ref_lats  <= BOG_LAT + BOG_DELTA)
lon_mask      = (ref_lons  >= BOG_LON - BOG_DELTA) & (ref_lons  <= BOG_LON + BOG_DELTA)
era5_lat_mask = (era5_lats >= BOG_LAT - BOG_DELTA) & (era5_lats <= BOG_LAT + BOG_DELTA)
era5_lon_mask = (era5_lons >= BOG_LON - BOG_DELTA) & (era5_lons <= BOG_LON + BOG_DELTA)
bog_lats      = ref_lats[lat_mask]
bog_lons      = ref_lons[lon_mask]
bog_era5_lats = era5_lats[era5_lat_mask]
bog_era5_lons = era5_lons[era5_lon_mask]

bog_vmax = float(np.nanpercentile(
    np.concatenate([downscaled_maps[m][np.ix_(lat_mask, lon_mask)].ravel()
                    for m in months if m in downscaled_maps]), 98
))
print(f"  Bogota colorscale: 0 – {bog_vmax:.0f} mm (98th pct)")

n_months = len(months)
fig4, axes4 = plt.subplots(n_months, 4, figsize=(20, n_months * 4.5))
fig4.suptitle("Bogotá Region Precipitation 2010 & 2024 — Monthly\n"
              "Raw ERA5-Land (0.1°) | Bilinear Interp (0.01°) | RF Downscaled (0.01°) | Station scatter",
              fontsize=14, fontweight="bold", y=0.995)

for i, month in enumerate(months):
    era5_slice  = era5_tp.isel(time=i)
    ideam_pts   = ideam_mon[
        (ideam_mon["month"] == month) &
        (ideam_mon["latitude"].between(BOG_LAT - BOG_DELTA, BOG_LAT + BOG_DELTA)) &
        (ideam_mon["longitude"].between(BOG_LON - BOG_DELTA, BOG_LON + BOG_DELTA))
    ]

    era5_crop   = era5_slice.values[np.ix_(era5_lat_mask, era5_lon_mask)]
    bil_crop    = era5_interp_maps[month][np.ix_(lat_mask, lon_mask)]
    rf_crop     = downscaled_maps[month][np.ix_(lat_mask, lon_mask)]
    month_label = pd.Timestamp(month + "-01").strftime("%b %Y")

    for col_idx, (grid, glats, glons, title) in enumerate([
        (era5_crop, bog_era5_lats, bog_era5_lons, "Raw ERA5-Land (0.1°)"),
        (bil_crop,  bog_lats,      bog_lons,       "Bilinear (0.01°)"),
        (rf_crop,   bog_lats,      bog_lons,       "RF Downscaled (0.01°)"),
    ]):
        ax = axes4[i, col_idx]
        im = ax.pcolormesh(glons, glats, grid, cmap="YlGnBu", vmin=0, vmax=bog_vmax)
        ax.scatter(ideam_pts["longitude"], ideam_pts["latitude"],
                   c=ideam_pts["obs"], cmap="YlGnBu", vmin=0, vmax=bog_vmax,
                   s=40, edgecolors="k", linewidths=0.5, zorder=5)
        ax.set_xlim(BOG_LON - BOG_DELTA, BOG_LON + BOG_DELTA)
        ax.set_ylim(BOG_LAT - BOG_DELTA, BOG_LAT + BOG_DELTA)
        ax.set_title(f"{month_label} — {title}", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8, label="mm")

    ax4 = axes4[i, 3]
    if len(ideam_pts) > 0:
        e5_at, bi_at, ds_at = [], [], []
        for _, stn in ideam_pts.iterrows():
            li   = np.argmin(np.abs(ref_lats  - stn["latitude"]))
            lo   = np.argmin(np.abs(ref_lons  - stn["longitude"]))
            li_e = np.argmin(np.abs(era5_lats - stn["latitude"]))
            lo_e = np.argmin(np.abs(era5_lons - stn["longitude"]))
            e5_at.append(float(era5_slice.values[li_e, lo_e]))
            bi_at.append(float(era5_interp_maps[month][li, lo]))
            ds_at.append(float(downscaled_maps[month][li, lo]))

        obs_v = ideam_pts["obs"].values
        e5_v  = np.array(e5_at)
        bi_v  = np.array(bi_at)
        ds_v  = np.array(ds_at)
        valid = np.isfinite(obs_v) & np.isfinite(e5_v) & np.isfinite(ds_v)
        if valid.sum() > 1:
            all_v = np.concatenate([obs_v[valid], e5_v[valid], bi_v[valid], ds_v[valid]])
            lims  = [0, all_v.max() + 5]
            ax4.plot(lims, lims, "k--", lw=1)
            for vals, label, color in [
                (e5_v, "ERA5-Land", "steelblue"),
                (bi_v, "Bilinear",  "darkorange"),
                (ds_v, "RF",        "tomato"),
            ]:
                v = valid & np.isfinite(vals)
                if v.sum() > 1:
                    r2 = r2_score(obs_v[v], vals[v])
                    ax4.scatter(obs_v[v], vals[v], s=30, color=color, alpha=0.8,
                                label=f"{label} R²={r2:.2f}")
            ax4.set_xlim(lims); ax4.set_ylim(lims)
            ax4.legend(fontsize=6)
    ax4.set_xlabel("IDEAM obs (mm)", fontsize=7)
    ax4.set_ylabel("Predicted (mm)", fontsize=7)
    ax4.set_title(f"{month_label} — Station scatter", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.995])
plt.savefig(os.path.join(PLOT_DIR, "bogota_tp_monthly_comparison.png"), dpi=120, bbox_inches="tight")
plt.close()
print("  Saved Bogota monthly precipitation plot")
print("\n✓ All done.")