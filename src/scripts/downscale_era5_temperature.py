# src/scripts/downscale_era5.py
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
AUX_DIR    = r"C:\Users\mdgor\aca\aca_indice_climatico_opt-main\data\raw\auxiliary"
IDEAM_DIR  = r"C:\Users\mdgor\aca\aca_indice_climatico_opt-main\data\raw\ideam"
ERA5_LAND  = r"C:\Users\mdgor\data\raw\era5_land"
OUTPUT_DIR = r"C:\Users\mdgor\aca\aca_indice_climatico_opt-main\data\processed\downscaled"
PLOT_DIR   = r"C:\Users\mdgor\aca\aca_indice_climatico_opt-main\data\processed\plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)

COLOMBIA_BBOX = dict(lat_min=-4.6, lat_max=13.0, lon_min=-82.9, lon_max=-66.1)

# ── 1. Load auxiliary data at 0.01° ──────────────────────────────────────────
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
    print(f"  {name}: {da.shape}")

ref_lats  = aux["dem"].y.values.astype(float)
ref_lons  = aux["dem"].x.values.astype(float)
aux_stack = np.stack([aux[k].values.astype(float) for k in aux_files], axis=-1)
valid_mask = np.all(np.isfinite(aux_stack), axis=-1)
print(f"  Fine grid: {len(ref_lats)} lats x {len(ref_lons)} lons")
print(f"  Valid land pixels: {valid_mask.sum():,}")

# ── 2. Load ERA5-Land temperature ─────────────────────────────────────────────
print("\nLoading ERA5 temperature...")
def load_era5_tmp_monthly(years):
    arrays = []
    for yr in years:
        path = os.path.join(ERA5_LAND, f"era5land_tmp_{yr}.grib")
        ds = xr.open_dataset(
            path, engine="cfgrib",
            backend_kwargs={"indexpath": "", "errors": "ignore"},
            filter_by_keys={"edition": 2}
        )
        ds = ds.assign_coords(longitude=(ds.longitude - 360))
        ds = ds.sel(
            latitude=slice(COLOMBIA_BBOX["lat_max"], COLOMBIA_BBOX["lat_min"]),
            longitude=slice(COLOMBIA_BBOX["lon_min"], COLOMBIA_BBOX["lon_max"])
        )
        da = ds["t2m"].resample(time="ME").mean()
        arrays.append(da)
    return xr.concat(arrays, dim="time")

era5_tmp  = load_era5_tmp_monthly([2010])
era5_lats = era5_tmp.latitude.values.astype(float)
era5_lons = era5_tmp.longitude.values.astype(float)
months    = [str(t)[:7] for t in era5_tmp.time.values]
print(f"  ERA5 shape: {era5_tmp.shape} — {len(months)} months")
print(f"  ERA5 temp range: {float(era5_tmp.min()):.1f} – {float(era5_tmp.max()):.1f} °C")

# ── 3. Load IDEAM station data ────────────────────────────────────────────────
print("\nLoading IDEAM station data...")
df = pyreadr.read_r(os.path.join(IDEAM_DIR, "ideam_TSSM_CON_2010.rds"))[None]
df["date"]      = pd.to_datetime(df["date"])
df["month"]     = df["date"].dt.strftime("%Y-%m")
df["longitude"] = df["longitude"].astype(float)
df["latitude"]  = df["latitude"].astype(float)
df["value"]     = df["value"].astype(float)
ideam_mon = (
    df.groupby(["station", "longitude", "latitude", "month"])["value"]
    .mean().reset_index().rename(columns={"value": "obs"})
)
print(f"  {len(ideam_mon)} station-months from {ideam_mon['station'].nunique()} stations")

# ── 4. Interpolate ERA5 to fine grid ─────────────────────────────────────────
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
    lat_idx = np.argmin(np.abs(ref_lats - lat))
    lon_idx = np.argmin(np.abs(ref_lons - lon))
    return aux_stack[lat_idx, lon_idx, :]

records = []
for _, row in ideam_mon.iterrows():
    era5_slice = era5_tmp.sel(
        time=pd.Timestamp(row["month"] + "-01"), method="nearest"
    )
    li = np.argmin(np.abs(era5_lats - row["latitude"]))
    lo = np.argmin(np.abs(era5_lons - row["longitude"]))
    era5_val  = float(era5_slice.values[li, lo])
    aux_vals  = get_aux_at_point(row["latitude"], row["longitude"])
    if np.all(np.isfinite(aux_vals)) and np.isfinite(era5_val):
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
print(f"  Training samples: {len(train_df)}")
print(f"  obs range:  {train_df['obs'].min():.1f} – {train_df['obs'].max():.1f} °C")
print(f"  era5 range: {train_df['era5'].min():.1f} – {train_df['era5'].max():.1f} °C")

# ── 6. Train Random Forest ────────────────────────────────────────────────────
print("\nTraining Random Forest...")
features = ["era5", "dem", "slope", "aspect", "ndvi", "ndbi", "mndwi"]
X = train_df[features].values
y = train_df["obs"].values

rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf.fit(X, y)

y_pred = rf.predict(X)
print(f"  Training R²:   {r2_score(y, y_pred):.3f}")
print(f"  Training RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.2f} °C")
print("  Feature importances:")
for feat, imp in sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"    {feat:8s}: {imp:.3f}")

# ── 7. Predict on full fine grid ──────────────────────────────────────────────
print("\nDownscaling ERA5 to 0.01° for all months...")
aux_flat         = aux_stack[valid_mask]
downscaled_maps  = {}
era5_interp_maps = {}

for i, month in enumerate(months):
    era5_slice = era5_tmp.isel(time=i)
    era5_fine  = interp_era5_to_fine(era5_slice, ref_lats, ref_lons)
    era5_interp_maps[month] = era5_fine

    era5_flat   = era5_fine[valid_mask].reshape(-1, 1)
    X_pred      = np.hstack([era5_flat, aux_flat])
    y_pred_flat = rf.predict(X_pred)

    result = np.full((len(ref_lats), len(ref_lons)), np.nan)
    result[valid_mask] = y_pred_flat
    downscaled_maps[month] = result

    print(f"  {month}: ERA5 mean={float(era5_slice.mean()):.1f}°C  "
          f"→ downscaled mean={np.nanmean(result):.1f}°C")

# ── 8. Save downscaled NetCDF ─────────────────────────────────────────────────
print("\nSaving downscaled maps...")
times  = pd.to_datetime([m + "-01" for m in months])
data   = np.stack([downscaled_maps[m] for m in months], axis=0)
ds_out = xr.Dataset({
    "t2m_downscaled": xr.DataArray(
        data.astype("float32"),
        coords={"time": times, "lat": ref_lats, "lon": ref_lons},
        dims=["time", "lat", "lon"],
        attrs={"units": "°C", "long_name": "Downscaled 2m temperature"}
    )
})
ds_out.to_netcdf(os.path.join(OUTPUT_DIR, "era5_downscaled_tmp_2010.nc"))
print(f"  Saved NetCDF to {OUTPUT_DIR}")

# ── 9. Colombia comparison maps ───────────────────────────────────────────────
print("\nGenerating Colombia comparison maps...")
plot_months  = ["2010-01", "2010-04", "2010-07", "2010-10"]
month_names  = {"2010-01": "January", "2010-04": "April",
                "2010-07": "July",    "2010-10": "October"}
vmin, vmax   = 5, 35

fig = plt.figure(figsize=(20, 24))
fig.suptitle("ERA5 Temperature Downscaling — Colombia 2010\n"
             "Raw ERA5 (0.1°) | Bilinear Interpolation | RF Downscaled (0.01°) | Station scatter",
             fontsize=14, fontweight="bold", y=0.98)

for row_idx, month in enumerate(plot_months):
    era5_slice  = era5_tmp.sel(time=pd.Timestamp(month + "-01"), method="nearest")
    era5_coarse = era5_slice.values.astype(float)
    era5_fine   = era5_interp_maps[month]
    ds_fine     = downscaled_maps[month]
    ideam_pts   = ideam_mon[ideam_mon["month"] == month]

    for col_idx, (data_grid, lats_g, lons_g, title) in enumerate([
        (era5_coarse, era5_slice.latitude.values, era5_slice.longitude.values, f"Raw ERA5 (0.1°)"),
        (era5_fine,   ref_lats, ref_lons,          f"Bilinear Interp (0.01°)"),
        (ds_fine,     ref_lats, ref_lons,          f"RF Downscaled (0.01°)"),
    ]):
        ax = fig.add_subplot(4, 4, row_idx * 4 + col_idx + 1)
        im = ax.pcolormesh(lons_g, lats_g, data_grid, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        ax.scatter(ideam_pts["longitude"], ideam_pts["latitude"],
                   c=ideam_pts["obs"], cmap="RdYlBu_r", vmin=vmin, vmax=vmax,
                   s=15, edgecolors="k", linewidths=0.3, zorder=5)
        ax.set_title(f"{month_names[month]}\n{title}", fontsize=9)
        ax.set_xlim(-82.9, -66.1); ax.set_ylim(-4.6, 13)
        plt.colorbar(im, ax=ax, shrink=0.8, label="°C")

    # col 4: scatter
    ax4 = fig.add_subplot(4, 4, row_idx * 4 + 4)
    e5_at_stn, ds_at_stn = [], []
    for _, stn in ideam_pts.iterrows():
        li = np.argmin(np.abs(ref_lats - stn["latitude"]))
        lo = np.argmin(np.abs(ref_lons - stn["longitude"]))
        e5_at_stn.append(era5_fine[li, lo])
        ds_at_stn.append(ds_fine[li, lo])

    obs_v = ideam_pts["obs"].values
    e5_v  = np.array(e5_at_stn)
    ds_v  = np.array(ds_at_stn)
    valid = np.isfinite(obs_v) & np.isfinite(e5_v) & np.isfinite(ds_v)
    if valid.sum() > 2:
        r2_e5 = r2_score(obs_v[valid], e5_v[valid])
        r2_ds = r2_score(obs_v[valid], ds_v[valid])
        lims  = [min(obs_v[valid].min(), e5_v[valid].min(), ds_v[valid].min()) - 1,
                 max(obs_v[valid].max(), e5_v[valid].max(), ds_v[valid].max()) + 1]
        ax4.plot(lims, lims, "k--", lw=1)
        ax4.scatter(obs_v[valid], e5_v[valid], alpha=0.5, s=10,
                    color="steelblue", label=f"ERA5 R²={r2_e5:.2f}")
        ax4.scatter(obs_v[valid], ds_v[valid], alpha=0.5, s=10,
                    color="tomato",   label=f"RF   R²={r2_ds:.2f}")
        ax4.set_xlim(lims); ax4.set_ylim(lims)
        ax4.legend(fontsize=7)
    ax4.set_xlabel("IDEAM obs (°C)", fontsize=8)
    ax4.set_ylabel("ERA5 / RF (°C)", fontsize=8)
    ax4.set_title(f"{month_names[month]}\nStation scatter", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(PLOT_DIR, "downscaling_comparison_2010.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved Colombia comparison plot")

# ── 10. Bogota comparison — one row per month ─────────────────────────────────
print("\nGenerating Bogota monthly comparison plot...")
BOG_LAT, BOG_LON = 4.7, -74.1
BOG_DELTA = 2.0  # degrees around Bogota to zoom into

bog_stations = ideam_mon[
    (ideam_mon["latitude"].between(BOG_LAT - 0.5, BOG_LAT + 0.5)) &
    (ideam_mon["longitude"].between(BOG_LON - 0.5, BOG_LON + 0.5))
]
bog_monthly = bog_stations.groupby("month")["obs"].mean().reset_index()
print(f"  {bog_stations['station'].nunique()} IDEAM stations near Bogota")

# Lat/lon index slices for zoomed Bogota region
lat_mask = (ref_lats >= BOG_LAT - BOG_DELTA) & (ref_lats <= BOG_LAT + BOG_DELTA)
lon_mask = (ref_lons >= BOG_LON - BOG_DELTA) & (ref_lons <= BOG_LON + BOG_DELTA)
bog_lats = ref_lats[lat_mask]
bog_lons = ref_lons[lon_mask]

# ERA5 coarse lat/lon mask for Bogota region
era5_lat_mask = (era5_lats >= BOG_LAT - BOG_DELTA) & (era5_lats <= BOG_LAT + BOG_DELTA)
era5_lon_mask = (era5_lons >= BOG_LON - BOG_DELTA) & (era5_lons <= BOG_LON + BOG_DELTA)
bog_era5_lats = era5_lats[era5_lat_mask]
bog_era5_lons = era5_lons[era5_lon_mask]

vmin_b, vmax_b = 8, 28

fig3, axes3 = plt.subplots(12, 4, figsize=(20, 54))
fig3.suptitle("Bogotá Region Temperature 2010 — Monthly\n"
              "Raw ERA5 (0.1°) | Bilinear Interp (0.01°) | RF Downscaled (0.01°) | Station scatter",
              fontsize=14, fontweight="bold", y=0.995)

for i, month in enumerate(months):
    era5_slice  = era5_tmp.isel(time=i)
    ideam_pts   = ideam_mon[
        (ideam_mon["month"] == month) &
        (ideam_mon["latitude"].between(BOG_LAT - BOG_DELTA, BOG_LAT + BOG_DELTA)) &
        (ideam_mon["longitude"].between(BOG_LON - BOG_DELTA, BOG_LON + BOG_DELTA))
    ]

    # Crop grids to Bogota region
    era5_crop = era5_slice.values[np.ix_(era5_lat_mask, era5_lon_mask)]
    bil_crop  = era5_interp_maps[month][np.ix_(lat_mask, lon_mask)]
    rf_crop   = downscaled_maps[month][np.ix_(lat_mask, lon_mask)]

    month_label = pd.Timestamp(month + "-01").strftime("%B")

    # col 1: Raw ERA5
    ax = axes3[i, 0]
    im = ax.pcolormesh(bog_era5_lons, bog_era5_lats, era5_crop,
                       cmap="RdYlBu_r", vmin=vmin_b, vmax=vmax_b)
    ax.scatter(ideam_pts["longitude"], ideam_pts["latitude"],
               c=ideam_pts["obs"], cmap="RdYlBu_r", vmin=vmin_b, vmax=vmax_b,
               s=40, edgecolors="k", linewidths=0.5, zorder=5)
    ax.set_xlim(BOG_LON - BOG_DELTA, BOG_LON + BOG_DELTA)
    ax.set_ylim(BOG_LAT - BOG_DELTA, BOG_LAT + BOG_DELTA)
    ax.set_title(f"{month_label} — Raw ERA5 (0.1°)", fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8, label="°C")

    # col 2: Bilinear
    ax = axes3[i, 1]
    im = ax.pcolormesh(bog_lons, bog_lats, bil_crop,
                       cmap="RdYlBu_r", vmin=vmin_b, vmax=vmax_b)
    ax.scatter(ideam_pts["longitude"], ideam_pts["latitude"],
               c=ideam_pts["obs"], cmap="RdYlBu_r", vmin=vmin_b, vmax=vmax_b,
               s=40, edgecolors="k", linewidths=0.5, zorder=5)
    ax.set_xlim(BOG_LON - BOG_DELTA, BOG_LON + BOG_DELTA)
    ax.set_ylim(BOG_LAT - BOG_DELTA, BOG_LAT + BOG_DELTA)
    ax.set_title(f"{month_label} — Bilinear Interp (0.01°)", fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8, label="°C")

    # col 3: RF Downscaled
    ax = axes3[i, 2]
    im = ax.pcolormesh(bog_lons, bog_lats, rf_crop,
                       cmap="RdYlBu_r", vmin=vmin_b, vmax=vmax_b)
    ax.scatter(ideam_pts["longitude"], ideam_pts["latitude"],
               c=ideam_pts["obs"], cmap="RdYlBu_r", vmin=vmin_b, vmax=vmax_b,
               s=40, edgecolors="k", linewidths=0.5, zorder=5)
    ax.set_xlim(BOG_LON - BOG_DELTA, BOG_LON + BOG_DELTA)
    ax.set_ylim(BOG_LAT - BOG_DELTA, BOG_LAT + BOG_DELTA)
    ax.set_title(f"{month_label} — RF Downscaled (0.01°)", fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8, label="°C")

    # col 4: Scatter
    ax = axes3[i, 3]
    if len(ideam_pts) > 0:
        e5_at_stn, ds_at_stn, bi_at_stn = [], [], []
        for _, stn in ideam_pts.iterrows():
            li = np.argmin(np.abs(ref_lats - stn["latitude"]))
            lo = np.argmin(np.abs(ref_lons - stn["longitude"]))
            li_e = np.argmin(np.abs(era5_lats - stn["latitude"]))
            lo_e = np.argmin(np.abs(era5_lons - stn["longitude"]))
            e5_at_stn.append(float(era5_slice.values[li_e, lo_e]))
            bi_at_stn.append(float(era5_interp_maps[month][li, lo]))
            ds_at_stn.append(float(downscaled_maps[month][li, lo]))

        obs_v = ideam_pts["obs"].values
        e5_v  = np.array(e5_at_stn)
        bi_v  = np.array(bi_at_stn)
        ds_v  = np.array(ds_at_stn)
        valid = np.isfinite(obs_v) & np.isfinite(e5_v) & np.isfinite(ds_v)

        if valid.sum() > 1:
            all_vals = np.concatenate([obs_v[valid], e5_v[valid], bi_v[valid], ds_v[valid]])
            lims = [all_vals.min() - 0.5, all_vals.max() + 0.5]
            ax.plot(lims, lims, "k--", lw=1)
            for vals, label, color in [
                (e5_v, "ERA5", "steelblue"),
                (bi_v, "Bilinear", "darkorange"),
                (ds_v, "RF", "tomato"),
            ]:
                v = np.isfinite(obs_v) & np.isfinite(vals)
                if v.sum() > 1:
                    r2 = r2_score(obs_v[v], vals[v])
                    ax.scatter(obs_v[v], vals[v], s=30, color=color, alpha=0.8,
                               label=f"{label} R²={r2:.2f}")
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.legend(fontsize=6)
        ax.set_xlabel("IDEAM obs (°C)", fontsize=7)
        ax.set_ylabel("Predicted (°C)", fontsize=7)
    ax.set_title(f"{month_label} — Station scatter", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.995])
bog_plot_path = os.path.join(PLOT_DIR, "bogota_monthly_comparison_2010.png")
plt.savefig(bog_plot_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"  Saved Bogota monthly plot: {bog_plot_path}")
print("\n✓ All done.")