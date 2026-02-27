import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr
from pathlib import Path

BASE   = r'C:\Users\mdgor\aca\aca_indice_climatico_opt-main'
RAW    = BASE + r'\data\raw\era5'
PROC   = BASE + r'\data\processed'

# ── Load and merge ────────────────────────────────────────────────────────
sst = pd.read_csv(PROC + r'\sst_index_colombia_pacific.csv', parse_dates=['time'])

oni_raw = pd.read_csv(BASE + r'\data\raw\oni.ascii.txt', sep=r'\s+', engine='python')
season_to_month = {
    'DJF':1,'JFM':2,'FMA':3,'MAM':4,'AMJ':5,'MJJ':6,
    'JJA':7,'JAS':8,'ASO':9,'SON':10,'OND':11,'NDJ':12
}
oni_raw['MON'] = oni_raw['SEAS'].map(season_to_month)
oni_raw['time'] = pd.to_datetime(oni_raw['YR'].astype(str) + '-' + oni_raw['MON'].astype(str) + '-01')
oni = oni_raw.rename(columns={'ANOM': 'oni'})[['time', 'oni']]

df = sst.merge(oni, on='time', how='inner')[['time', 'sst_index', 'sst_anomaly', 'oni']].dropna()
corr = df['sst_index'].corr(df['oni'])
print(f"Merged: {len(df)} rows  |  Correlation: {corr:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# SANITY CHECKS (console)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SANITY CHECKS")
print("="*60)

# 1. Raw SST units
print("\n[1] Raw SST units & range (1997 snapshot)")
sst_raw = None
try:
    f = sorted(Path(RAW).glob('era5_sst_1997.grib'))[0]
    ds = xr.open_dataset(f, engine='cfgrib')
    sst_raw = ds['sst'].isel(time=0)
    vmin, vmax, vmean = float(sst_raw.min()), float(sst_raw.max()), float(sst_raw.mean())
    units = ds['sst'].attrs.get('units', 'NOT SET')
    print(f"    Units : {units}")
    print(f"    Min   : {vmin:.2f}   Max: {vmax:.2f}   Mean: {vmean:.2f}")
    if vmean > 200:
        print("    WARNING: Values look like Kelvin!")
    else:
        print("    OK: Values look like Celsius")
except Exception as e:
    print(f"    Could not load GRIB: {e}")

# 2. Monthly anomaly mean per month (should be ~0)
print("\n[2] Monthly anomaly mean (should be ~0 for each month)")
monthly_stats = df.groupby(df['time'].dt.month)['sst_anomaly'].agg(['mean','std'])
monthly_stats.index = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
print(monthly_stats.round(4).to_string())

# 3. Known El Nino years
print("\n[3] El Nino years — sst_index should be positive in 1983, 1998, 2016")
el_nino = df[df['time'].dt.year.isin([1982,1983,1997,1998,2015,2016])]
print(el_nino[['time','sst_anomaly','sst_index','oni']].to_string(index=False))

# 4. NaN / land mask
print("\n[4] NaN (land) mask check")
if sst_raw is not None:
    total = sst_raw.size
    nans  = int(np.isnan(sst_raw).sum())
    valid = total - nans
    print(f"    Total: {total}  NaN(land): {nans} ({100*nans/total:.1f}%)  Ocean: {valid} ({100*valid/total:.1f}%)")
    if valid / total < 0.3:
        print("    WARNING: Less than 30% ocean — spatial mean may be unreliable")
    else:
        print("    OK: Ocean coverage looks reasonable")

print("\n" + "="*60 + "\n")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Main comparison dashboard
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10), facecolor='#0d1117')
fig.suptitle('Colombian Pacific SST Index vs ONI (Nino 3.4)',
             fontsize=16, color='white', fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#161b22')
ax1.plot(df['time'], df['oni'],       color='#58a6ff', lw=1.2, label='ONI (Nino 3.4)',        alpha=0.9)
ax1.plot(df['time'], df['sst_index'], color='#ff7b72', lw=1.2, label='Colombian Pacific SST', alpha=0.9)
ax1.axhline(0,    color='white',   lw=0.5, alpha=0.3)
ax1.axhline( 0.5, color='#58a6ff', lw=0.5, ls='--', alpha=0.3)
ax1.axhline(-0.5, color='#ff7b72', lw=0.5, ls='--', alpha=0.3)
ax1.fill_between(df['time'], df['oni'], 0, where=df['oni'] >= 0.5,  alpha=0.15, color='#58a6ff')
ax1.fill_between(df['time'], df['oni'], 0, where=df['oni'] <= -0.5, alpha=0.15, color='#ff7b72')
ax1.set_ylabel('Anomaly (C)', color='#8b949e')
ax1.tick_params(colors='#8b949e')
ax1.spines[:].set_color('#30363d')
ax1.legend(framealpha=0, labelcolor='white', fontsize=9)
ax1.set_title(f'Time Series  |  r = {corr:.3f}', color='#8b949e', fontsize=10, loc='right')

ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#161b22')
ax2.scatter(df['oni'], df['sst_index'],
            c=df['time'].astype(np.int64), cmap='plasma', alpha=0.5, s=12)
m, b = np.polyfit(df['oni'], df['sst_index'], 1)
x_line = np.linspace(df['oni'].min(), df['oni'].max(), 100)
ax2.plot(x_line, m * x_line + b, color='white', lw=1.5, alpha=0.7)
ax2.axhline(0, color='white', lw=0.4, alpha=0.3)
ax2.axvline(0, color='white', lw=0.4, alpha=0.3)
ax2.set_xlabel('ONI (C)', color='#8b949e')
ax2.set_ylabel('Colombian Pacific SST (C)', color='#8b949e')
ax2.tick_params(colors='#8b949e')
ax2.spines[:].set_color('#30363d')
ax2.set_title('Scatter (colored by year)', color='#8b949e', fontsize=10)

ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('#161b22')
roll_corr = df['sst_index'].rolling(60, center=True).corr(df['oni'])
ax3.plot(df['time'], roll_corr, color='#3fb950', lw=1.3)
ax3.axhline(corr, color='white', lw=0.8, ls='--', alpha=0.5, label=f'Mean r={corr:.2f}')
ax3.axhline(0, color='white', lw=0.4, alpha=0.3)
ax3.fill_between(df['time'], roll_corr, 0, where=roll_corr >= 0, alpha=0.15, color='#3fb950')
ax3.fill_between(df['time'], roll_corr, 0, where=roll_corr <  0, alpha=0.15, color='#ff7b72')
ax3.set_ylabel('Correlation', color='#8b949e')
ax3.tick_params(colors='#8b949e')
ax3.spines[:].set_color('#30363d')
ax3.legend(framealpha=0, labelcolor='white', fontsize=9)
ax3.set_title('5-year Rolling Correlation', color='#8b949e', fontsize=10)

ax4 = fig.add_subplot(gs[2, 0])
ax4.set_facecolor('#161b22')
ax4.hist(df['oni'],       bins=40, color='#58a6ff', alpha=0.6, label='ONI',                  density=True)
ax4.hist(df['sst_index'], bins=40, color='#ff7b72', alpha=0.6, label='Colombian Pacific SST', density=True)
ax4.axvline( 0.5, color='white', lw=0.8, ls='--', alpha=0.5)
ax4.axvline(-0.5, color='white', lw=0.8, ls='--', alpha=0.5)
ax4.set_xlabel('Anomaly (C)', color='#8b949e')
ax4.set_ylabel('Density',     color='#8b949e')
ax4.tick_params(colors='#8b949e')
ax4.spines[:].set_color('#30363d')
ax4.legend(framealpha=0, labelcolor='white', fontsize=9)
ax4.set_title('Distribution', color='#8b949e', fontsize=10)

ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor('#161b22')
diff = df['sst_index'] - df['oni']
ax5.fill_between(df['time'], diff, 0, where=diff >= 0, alpha=0.5, color='#ff7b72', label='SST > ONI')
ax5.fill_between(df['time'], diff, 0, where=diff <  0, alpha=0.5, color='#58a6ff', label='SST < ONI')
ax5.axhline(0, color='white', lw=0.5, alpha=0.5)
ax5.set_ylabel('SST - ONI (C)', color='#8b949e')
ax5.tick_params(colors='#8b949e')
ax5.spines[:].set_color('#30363d')
ax5.legend(framealpha=0, labelcolor='white', fontsize=9)
ax5.set_title('Difference (Colombian SST - ONI)', color='#8b949e', fontsize=10)

plt.savefig(PROC + r'\sst_oni_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("Saved: sst_oni_comparison.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Sanity check plots
# ══════════════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor='#0d1117')
fig2.suptitle('Sanity Checks — Colombian Pacific SST Construction',
              fontsize=14, color='white', fontweight='bold')

# A. Monthly anomaly mean (should be ~0)
ax = axes[0, 0]
ax.set_facecolor('#161b22')
monthly_mean = df.groupby(df['time'].dt.month)['sst_anomaly'].mean()
monthly_std  = df.groupby(df['time'].dt.month)['sst_anomaly'].std()
months = ['J','F','M','A','M','J','J','A','S','O','N','D']
ax.bar(range(1,13), monthly_mean, color='#58a6ff', alpha=0.7)
ax.errorbar(range(1,13), monthly_mean, yerr=monthly_std,
            fmt='none', color='white', alpha=0.5, capsize=3)
ax.axhline(0, color='white', lw=0.8, alpha=0.5)
ax.set_xticks(range(1,13)); ax.set_xticklabels(months)
ax.set_ylabel('Mean anomaly (C)', color='#8b949e')
ax.tick_params(colors='#8b949e'); ax.spines[:].set_color('#30363d')
ax.set_title('Monthly anomaly mean (should be ~0)', color='#8b949e', fontsize=10)

# B. El Nino years highlighted
ax = axes[0, 1]
ax.set_facecolor('#161b22')
ax.plot(df['time'], df['sst_index'], color='#8b949e', lw=0.8, alpha=0.4, label='All months')

# Use masked series so non-El Nino months are NaN (no connecting lines)
el_nino_mask = df['time'].dt.year.isin([1982, 1983, 1997, 1998, 2015, 2016])
el_nino_series = df['sst_index'].where(el_nino_mask)  # NaN outside El Nino years
ax.plot(df['time'], el_nino_series,
        color='#ff7b72', lw=1.8, label='El Nino years (1982-83, 1997-98, 2015-16)')

ax.axhline(0,   color='white',   lw=0.5, alpha=0.3)
ax.axhline(0.5, color='#ff7b72', lw=0.5, ls='--', alpha=0.4)
ax.set_ylabel('SST Index (C)', color='#8b949e')
ax.tick_params(colors='#8b949e'); ax.spines[:].set_color('#30363d')
ax.legend(framealpha=0, labelcolor='white', fontsize=8)
ax.set_title('El Nino years should show positive anomalies', color='#8b949e', fontsize=10)

# C. Domain map
ax = axes[1, 0]
ax.set_facecolor('#161b22')
if sst_raw is not None:
    try:
        vals = sst_raw.values.copy()
        if np.nanmean(vals) > 200:
            vals = vals - 273.15
        im = ax.pcolormesh(sst_raw.longitude, sst_raw.latitude, vals,
                           cmap='RdBu_r', vmin=20, vmax=32)
        plt.colorbar(im, ax=ax, label='SST (C)', shrink=0.8)
        ax.set_title('ERA5 SST domain snapshot (1997)', color='#8b949e', fontsize=10)
    except Exception as e:
        ax.text(0.5, 0.5, str(e), color='white', ha='center', va='center',
                transform=ax.transAxes, fontsize=8)
else:
    ax.text(0.5, 0.5, 'GRIB not loaded', color='white', ha='center', va='center',
            transform=ax.transAxes)
ax.tick_params(colors='#8b949e'); ax.spines[:].set_color('#30363d')
ax.set_xlabel('Longitude', color='#8b949e')
ax.set_ylabel('Latitude',  color='#8b949e')

# D. Correlation by decade
ax = axes[1, 1]
ax.set_facecolor('#161b22')
df['decade'] = (df['time'].dt.year // 10) * 10
decade_corr = df.groupby('decade').apply(
    lambda x: x['sst_index'].corr(x['oni']), include_groups=False
)
bar_colors = ['#3fb950' if v >= 0.4 else '#ff7b72' for v in decade_corr.values]
ax.bar([str(d)+'s' for d in decade_corr.index], decade_corr.values,
       color=bar_colors, alpha=0.8)
ax.axhline(corr, color='white', lw=0.8, ls='--', alpha=0.5, label=f'Overall r={corr:.2f}')
ax.axhline(0, color='white', lw=0.4, alpha=0.3)
ax.set_ylabel('Correlation with ONI', color='#8b949e')
ax.tick_params(colors='#8b949e', axis='y')
ax.tick_params(colors='#8b949e', axis='x', rotation=30)
ax.spines[:].set_color('#30363d')
ax.legend(framealpha=0, labelcolor='white', fontsize=9)
ax.set_title('Correlation by decade (red = weak, threshold 0.4)', color='#8b949e', fontsize=10)

plt.tight_layout()
plt.savefig(PROC + r'\sst_sanity_checks.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("Saved: sst_sanity_checks.png")

plt.show()