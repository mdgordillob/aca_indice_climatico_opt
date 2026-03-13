"""
compare_land_only.py
──────────────────────────────────────────────────────────────────────────────
Compare EDH vs CDS ERA5-Land temperature, masking out sea points from BOTH.
A pixel is only included if it is valid (non-NaN) in ALL timesteps in BOTH sources.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

F_EDH = r"C:\Users\mdgor\data\raw\era5_land\era5land_tmp_1961.grib"
F_CDS = r"C:\Users\mdgor\data\raw\era5_land\era5_tmp_1961_cds.grib"
F_REP = r"C:\Users\mdgor\data\raw\era5_land\era5land_tmp_1961_repacked.grib"
OUT   = r"C:\Users\mdgor\data\raw\era5_land\comparison_land_only.png"

SEP = "─" * 70


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
def load_t2m(path, kelvin=False):
    ds = xr.open_dataset(path, engine="cfgrib")
    ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180).sortby("longitude")
    da = ds["t2m"].load()
    if kelvin:
        da = da - 273.15
    return da

print("Loading files...")
da_edh = load_t2m(F_EDH)
da_rep = load_t2m(F_REP)
da_cds = load_t2m(F_CDS, kelvin=True)


# ══════════════════════════════════════════════════════════════════════════════
# ALIGN GRIDS + TIMESTAMPS
# ══════════════════════════════════════════════════════════════════════════════
def align(da_ref, da_other):
    times_r = da_ref.valid_time.values
    times_o = da_other.valid_time.values.flatten()
    vals_o  = da_other.values.reshape(-1, da_other.shape[-2], da_other.shape[-1])

    _, i_r, i_o = np.intersect1d(times_r, times_o, return_indices=True)

    a = da_ref.values[i_r]
    b = vals_o[i_o]

    lats_r, lons_r = da_ref.latitude.values, da_ref.longitude.values
    lats_o, lons_o = da_other.latitude.values, da_other.longitude.values

    lat_idx = np.array([np.argmin(np.abs(lats_o - la)) for la in lats_r])
    lon_idx = np.array([np.argmin(np.abs(lons_o - lo)) for lo in lons_r])
    b = b[:, lat_idx, :][:, :, lon_idx]
    return a, b

print("Aligning grids...")
a_edh, b_cds = align(da_edh, da_cds)
a_rep, b_cds2 = align(da_rep, da_cds)


# ══════════════════════════════════════════════════════════════════════════════
# BUILD LAND MASK  — valid in EVERY timestep in BOTH sources
# ══════════════════════════════════════════════════════════════════════════════
# A pixel is "land" if it is non-NaN in all timesteps in both arrays
land_edh_cds = (~np.any(np.isnan(a_edh), axis=0)) & (~np.any(np.isnan(b_cds), axis=0))
land_rep_cds = (~np.any(np.isnan(a_rep), axis=0)) & (~np.any(np.isnan(b_cds2), axis=0))

print(f"\nLand pixels (EDH ∩ CDS)       : {land_edh_cds.sum():,} / {land_edh_cds.size:,}  ({land_edh_cds.mean()*100:.1f}%)")
print(f"Land pixels (repacked ∩ CDS)  : {land_rep_cds.sum():,} / {land_rep_cds.size:,}  ({land_rep_cds.mean()*100:.1f}%)")
print(f"EDH-only land pixels          : {(~np.any(np.isnan(a_edh),axis=0)).sum():,}")
print(f"CDS-only land pixels          : {(~np.any(np.isnan(b_cds),axis=0)).sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# STATS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def masked_stats(a, b, mask):
    """Stats over land pixels only (mask is 2D, applied across all timesteps)."""
    # Apply mask: (T, lat, lon) → keep only masked pixels
    a_land = a[:, mask]   # (T, N_land)
    b_land = b[:, mask]

    diff = a_land - b_land
    flat_diff = diff.flatten()

    return {
        "n_land_px":    int(mask.sum()),
        "n_total_vals": flat_diff.size,
        "mean_a":       np.nanmean(a_land),
        "mean_b":       np.nanmean(b_land),
        "mean_diff":    np.nanmean(diff),
        "std_diff":     np.nanstd(diff),
        "mae":          np.nanmean(np.abs(diff)),
        "max_diff":     np.nanmax(np.abs(diff)),
        "pct_001":      np.mean(np.abs(flat_diff) < 0.01) * 100,
        "pct_010":      np.mean(np.abs(flat_diff) < 0.10) * 100,
        "mean_diff_map": np.nanmean(diff, axis=0),   # per-pixel mean bias
        "std_diff_map":  np.nanstd(diff,  axis=0),   # per-pixel std
    }

print("\nComputing land-only statistics...")
s1 = masked_stats(a_edh, b_cds,  land_edh_cds)
s2 = masked_stats(a_rep, b_cds2, land_rep_cds)

# Also compute ALL-pixel stats for comparison
def all_stats(a, b):
    diff = (a - b).flatten()
    return {
        "mean_a":    np.nanmean(a),
        "mean_b":    np.nanmean(b),
        "mean_diff": np.nanmean(diff),
        "std_diff":  np.nanstd(diff),
        "mae":       np.nanmean(np.abs(diff)),
        "max_diff":  np.nanmax(np.abs(diff)),
        "pct_001":   np.nanmean(np.abs(diff) < 0.01) * 100,
        "pct_010":   np.nanmean(np.abs(diff) < 0.10) * 100,
    }

s_all = all_stats(a_edh, b_cds)


# ══════════════════════════════════════════════════════════════════════════════
# PRINT REPORT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  LAND-ONLY COMPARISON  (pixels valid in BOTH sources, all timesteps)")
print(SEP)
print(f"  {'Metric':<30} {'ALL pixels':>15} {'Land only (EDH∩CDS)':>20} {'Land (repacked∩CDS)':>20}")
print("  " + "─"*87)

metrics = [
    ("land pixels",        "n_land_px",    False),
    ("mean A  (°C)",       "mean_a",       True),
    ("mean B  (°C)",       "mean_b",       True),
    ("mean diff (°C)",     "mean_diff",    True),
    ("std diff  (°C)",     "std_diff",     True),
    ("MAE      (°C)",      "mae",          True),
    ("max |diff| (°C)",    "max_diff",     True),
    ("% within 0.01°C",   "pct_001",      True),
    ("% within 0.10°C",   "pct_010",      True),
]

for label, k, is_float in metrics:
    va = s_all.get(k, "—")
    v1 = s1[k]
    v2 = s2[k]
    if is_float:
        va_str = f"{va:>15.4f}" if isinstance(va, float) else f"{'—':>15}"
        print(f"  {label:<30} {va_str} {v1:>20.4f} {v2:>20.4f}")
    else:
        print(f"  {label:<30} {'—':>15} {str(v1):>20} {str(v2):>20}")

print(SEP)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════
lats = da_edh.latitude.values
lons = da_edh.longitude.values

# Expand land-only stats back to full grid for plotting
def to_grid(vals_1d, mask):
    grid = np.full(mask.shape, np.nan)
    grid[mask] = vals_1d
    return grid

bias_grid = to_grid(s1["mean_diff_map"], land_edh_cds)
std_grid  = to_grid(s1["std_diff_map"],  land_edh_cds)
mean_edh  = np.where(land_edh_cds, np.nanmean(a_edh, axis=0), np.nan)
mean_cds  = np.where(land_edh_cds, np.nanmean(b_cds, axis=0), np.nan)

fig = plt.figure(figsize=(16, 9), facecolor="#0e1117")
fig.suptitle("ERA5-Land 1961 · EDH vs CDS · 2m Temperature (land pixels only)",
             fontsize=14, color="white", weight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35,
                       left=0.06, right=0.97, top=0.92, bottom=0.06)
BG, TC = "#1a1f2e", "white"

def ax_style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TC, fontsize=9, pad=5)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.tick_params(colors=TC, labelsize=7)
    return ax

def cbar(fig, im, ax):
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.ax.yaxis.set_tick_params(color=TC, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TC)

# ── mean EDH (land only)
ax = ax_style(fig.add_subplot(gs[0,0]), "EDH mean T — land only (°C)")
im = ax.pcolormesh(lons, lats, mean_edh, cmap="plasma", shading="auto")
cbar(fig, im, ax)

# ── mean CDS (land only)
ax = ax_style(fig.add_subplot(gs[0,1]), "CDS mean T — land only (°C)")
im = ax.pcolormesh(lons, lats, mean_cds, cmap="plasma", shading="auto")
cbar(fig, im, ax)

# ── land mask comparison
mask_edh = (~np.any(np.isnan(a_edh), axis=0)).astype(float)
mask_cds = (~np.any(np.isnan(b_cds), axis=0)).astype(float)
mask_diff = mask_edh - mask_cds   # +1 = EDH only, -1 = CDS only, 0 = both
ax = ax_style(fig.add_subplot(gs[0,2]), "Land mask diff (EDH − CDS)")
im = ax.pcolormesh(lons, lats, mask_diff, cmap="RdBu", vmin=-1, vmax=1, shading="auto")
cbar(fig, im, ax)
ax.set_title("Land mask diff\n+1=EDH only  0=both  −1=CDS only", color=TC, fontsize=8)

# ── mean bias map (land only)
vmax = np.nanmax(np.abs(bias_grid))
ax = ax_style(fig.add_subplot(gs[1,0]), "Mean bias EDH−CDS, land only (°C)")
im = ax.pcolormesh(lons, lats, bias_grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
cbar(fig, im, ax)

# ── std diff map (land only)
ax = ax_style(fig.add_subplot(gs[1,1]), "Std of diff, land only (°C)")
im = ax.pcolormesh(lons, lats, std_grid, cmap="OrRd", shading="auto")
cbar(fig, im, ax)

# ── histogram land-only diffs
ax = ax_style(fig.add_subplot(gs[1,2]), "Distribution of EDH−CDS diff (land only)")
flat = (a_edh[:, land_edh_cds] - b_cds[:, land_edh_cds]).flatten()
ax.hist(flat, bins=120, color="#4fc3f7", alpha=0.85, edgecolor="none")
ax.axvline(0, color="white", lw=1.2, ls="--")
ax.set_xlabel("Difference (°C)", color=TC, fontsize=8)
ax.set_ylabel("Count", color=TC, fontsize=8)
txt = (f"μ = {s1['mean_diff']:+.4f} °C\nσ = {s1['std_diff']:.4f} °C\n"
       f"MAE = {s1['mae']:.4f} °C\nmax|Δ| = {s1['max_diff']:.4f} °C")
ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color=TC, bbox=dict(facecolor="#0e1117", alpha=0.6, edgecolor="none"))

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="#0e1117")
print(f"\nPlot saved → {OUT}")
plt.show()