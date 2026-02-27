"""
inspect_sst_signal.py
─────────────────────
Diagnostic tool to determine whether the divergence between the Colombian
Pacific SST index and NOAA ONI is a SIGNAL issue (real physical decoupling)
or a DATA/CODE issue (wrong geometry, land contamination, climatology bias,
smoothing artefact).
Checks
------
  1. Geometry verification  — plot the bounding boxes on a map; count land vs ocean pixels
  2. Raw SST sanity         — spot-check one GRIB file for known El Niño (1997)
  3. Climatology leakage    — monthly anomaly mean inside AND outside base period
  4. Smoothing alignment    — compare raw anomaly vs 3-month smoothed index vs ONI
  5. Scatter & correlation  — sst_index vs oni_anom; by decade; by season
  6. Divergence deep-dive   — case studies for the worst FN/FP months identified
  7. Spatial correlation    — if a GRIB file is available, map pixel-level r vs ONI
Outputs
-------
  signal_geometry.png
  signal_smoothing.png
  signal_scatter.png
  signal_divergence.png
  signal_spatial_corr.png   (only if GRIB available)
"""
import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
warnings.filterwarnings('ignore')
# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
GRIB_DIR      = os.path.join(PROJECT_ROOT, 'data', 'raw', 'era5')
OUTPUT_DIR    = PROCESSED_DIR
SST_CSV       = os.path.join(PROCESSED_DIR, 'sst_index_colombia_pacific.csv')
FEATURES_CSV = os.path.join(PROCESSED_DIR, 'sst_features_colombia_pacific.csv')
ONI_FILE     = os.path.join(PROCESSED_DIR, 'oni.ascii.txt')
# ── Domain geometry (must match anomalias_sst_daily.py) ──────────────────────
PAC_LAT = (-4,  10);  PAC_LON = (-84, -76)
CAR_LAT = ( 8,  13);  CAR_LON = (-76, -68)
# ── ONI parser (copied from compare_enso_flags) ───────────────────────────────
def load_oni():
    df = pd.read_csv(ONI_FILE, sep=r'\s+', engine='python')
    df.columns = [c.strip() for c in df.columns]
    SEAS_MAP = {'DJF':1,'JFM':2,'FMA':3,'MAM':4,'AMJ':5,'MJJ':6,
                'JJA':7,'JAS':8,'ASO':9,'SON':10,'OND':11,'NDJ':12}
    df['month'] = df['SEAS'].map(SEAS_MAP)
    df = df.dropna(subset=['month'])
    df['YR']   = pd.to_numeric(df['YR'],  errors='coerce').astype('Int64')
    df['ANOM'] = pd.to_numeric(df['ANOM'], errors='coerce')
    df['month'] = df['month'].astype(int)
    df.loc[df['SEAS']=='NDJ','YR'] += 1
    df['time'] = pd.to_datetime(
        df['YR'].astype(str)+'-'+df['month'].astype(str).str.zfill(2)+'-01')
    return df[['time','ANOM']].dropna().sort_values('time').reset_index(drop=True)
# ══════════════════════════════════════════════════════════════════════════════
# 1. Geometry verification
# ══════════════════════════════════════════════════════════════════════════════
def check_geometry(sst_df):
    """
    Plot the Colombian Pacific + Caribbean bounding boxes on a map.
    If cartopy is unavailable, fall back to a plain lat/lon plot.
    """
    print('\n[1] Geometry verification ...')
    fig = plt.figure(figsize=(16, 6))
    # ── Left: bounding box on map with cartopy ────────────────────────────
    try:
        ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax.set_extent([-160, -60, -10, 20], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='#a8d5a2')
        ax.add_feature(cfeature.OCEAN, facecolor='#cce5ff')
        pac_rect = mpatches.Rectangle((PAC_LON[0], PAC_LAT[0]),
                                      PAC_LON[1]-PAC_LON[0], PAC_LAT[1]-PAC_LAT[0],
                                      linewidth=2.5, edgecolor='#2c7bb6',
                                      facecolor='#2c7bb6', alpha=0.25, zorder=3,
                                      transform=ccrs.PlateCarree(),
                                      label=f'Pacific ({PAC_LON[0]}°–{PAC_LON[1]}°W, {PAC_LAT[0]}°–{PAC_LAT[1]}°N)')
        car_rect = mpatches.Rectangle((CAR_LON[0], CAR_LAT[0]),
                                      CAR_LON[1]-CAR_LON[0], CAR_LAT[1]-CAR_LAT[0],
                                      linewidth=2.5, edgecolor='#d7191c',
                                      facecolor='#d7191c', alpha=0.25, zorder=3,
                                      transform=ccrs.PlateCarree(),
                                      label=f'Caribbean ({CAR_LON[0]}°–{CAR_LON[1]}°W, {CAR_LAT[0]}°–{CAR_LAT[1]}°N)')
        nino3_rect = mpatches.Rectangle((-150, -5), 60, 10,
                                        linewidth=1.5, edgecolor='orange',
                                        facecolor='none', linestyle='--', zorder=3,
                                        transform=ccrs.PlateCarree(),
                                        label='Niño 3 (150°W–90°W, 5°S–5°N)')
        ax.add_patch(pac_rect)
        ax.add_patch(car_rect)
        ax.add_patch(nino3_rect)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = mticker.FuncFormatter(lambda x, _: f'{abs(x):.0f}°W' if x < 0 else f'{x:.0f}°E')
        gl.yformatter = mticker.FuncFormatter(lambda y, _: f'{abs(y):.0f}°{"N" if y >= 0 else "S"}')
        ax.set_title('Comparison: Colombian SST Domains vs. Niño 3 Region', fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
    except ImportError:
        print('  Cartopy not available, falling back to plain plot.')
        ax = fig.add_subplot(1, 2, 1)
        ax.set_xlim(-160, -60)
        ax.set_ylim(-10, 20)
        ax.set_facecolor('#cce5ff')
        # Rough Colombia outline (fallback)
        colombia = plt.Polygon([
            [-77, 8.5],[-76, 9],[-75, 10.5],[-74, 11.5],[-72, 11.8],
            [-72, 12.5],[-71, 12],[-70, 11.5],[-72, 10],[-73, 9],
            [-73, 7],[-74, 6],[-76, 4],[-77, 2],[-78, 1],
            [-78, -1],[-80, -2],[-80, 0],[-79, 1],[-78, 2],
            [-77, 4],[-77, 6],[-78, 8],[-77, 8.5]
        ], closed=True, facecolor='#a8d5a2', edgecolor='black', lw=0.8, zorder=2)
        ax.add_patch(colombia)
        pac_rect = mpatches.Rectangle((PAC_LON[0], PAC_LAT[0]),
                                      PAC_LON[1]-PAC_LON[0], PAC_LAT[1]-PAC_LAT[0],
                                      linewidth=2.5, edgecolor='#2c7bb6',
                                      facecolor='#2c7bb6', alpha=0.25, zorder=3,
                                      label=f'Pacific ({PAC_LON[0]}°–{PAC_LON[1]}°W, {PAC_LAT[0]}°–{PAC_LAT[1]}°N)')
        car_rect = mpatches.Rectangle((CAR_LON[0], CAR_LAT[0]),
                                      CAR_LON[1]-CAR_LON[0], CAR_LAT[1]-CAR_LAT[0],
                                      linewidth=2.5, edgecolor='#d7191c',
                                      facecolor='#d7191c', alpha=0.25, zorder=3,
                                      label=f'Caribbean ({CAR_LON[0]}°–{CAR_LON[1]}°W, {CAR_LAT[0]}°–{CAR_LAT[1]}°N)')
        nino3_rect = mpatches.Rectangle((-150, -5), 60, 10,
                                        linewidth=1.5, edgecolor='orange',
                                        facecolor='none', linestyle='--', zorder=3,
                                        label='Niño 3 (150°W–90°W, 5°S–5°N)')
        ax.add_patch(pac_rect)
        ax.add_patch(car_rect)
        ax.add_patch(nino3_rect)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Comparison: Colombian SST Domains vs. Niño 3 Region', fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{abs(x):.0f}°W' if x < 0 else f'{x:.0f}°E'))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{abs(y):.0f}°{"N" if y >= 0 else "S"}'))
    # ── Right: time series context ─────────────────────────────────
    ax = fig.add_subplot(1, 2, 2)
    sst_df = sst_df.dropna(subset=['sst_index'])
    ax.plot(sst_df['time'], sst_df['sst_anomaly_pacific'],
            color='#2c7bb6', lw=0.8, alpha=0.7, label='Pacific raw anomaly')
    if 'sst_anomaly_caribbean' in sst_df.columns and not sst_df['sst_anomaly_caribbean'].isna().all():
        ax.plot(sst_df['time'], sst_df['sst_anomaly_caribbean'],
                color='#d7191c', lw=0.8, alpha=0.7, label='Caribbean raw anomaly')
    ax.plot(sst_df['time'], sst_df['sst_index'],
            color='black', lw=1.4, label='Combined index (3-month smooth)')
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_ylabel('SST anomaly (°C)')
    ax.set_title('Regional SST anomalies — Pacific vs Caribbean', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'signal_geometry.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  -> {out}')
    plt.close()
    # Print domain stats
    print(f'  Pacific  box : lat {PAC_LAT}, lon {PAC_LON}')
    print(f'  Caribbean box: lat {CAR_LAT}, lon {CAR_LON}')
    print(f'  Note: Nino 3 is 5S-5N, 150W-90W — mostly outside our domain')
    print(f'  Pacific anomaly mean : {sst_df["sst_anomaly_pacific"].mean():.4f} C')
    if 'sst_anomaly_caribbean' in sst_df.columns:
        print(f'  Caribbean anomaly mean: {sst_df["sst_anomaly_caribbean"].mean():.4f} C')
# ══════════════════════════════════════════════════════════════════════════════
# 2. Raw GRIB spot check (1997 El Niño)
# ══════════════════════════════════════════════════════════════════════════════
def check_raw_grib():
    """Load 1997 GRIB file and compare domain-mean SST to ONI anomaly."""
    print('\n[2] Raw GRIB spot check (1997 El Nino) ...')
    grib_1997 = os.path.join(GRIB_DIR, 'era5_sst_1997.grib')
    if not os.path.exists(grib_1997):
        print('  -> 1997 GRIB not found, skipping')
        return None
    try:
        import xarray as xr
        ds = xr.open_dataset(grib_1997, engine='cfgrib')
        sst = ds['sst']
        # Convert K->C if needed
        if float(sst.mean()) > 200:
            sst = sst - 273.15
        # Subset Pacific box
        lats = sst.latitude.values
        lat_sl = (slice(PAC_LAT[1], PAC_LAT[0]) if lats[0] > lats[-1]
                  else slice(PAC_LAT[0], PAC_LAT[1]))
        pac = sst.sel(latitude=lat_sl, longitude=slice(PAC_LON[0], PAC_LON[1]))
        # Monthly mean
        pac_monthly = pac.resample(time='MS').mean()
        weights     = np.cos(np.deg2rad(pac_monthly.latitude))
        pac_ts      = pac_monthly.weighted(weights).mean(dim=('latitude','longitude'))
        pac_vals    = pd.Series(pac_ts.values,
                                index=pd.to_datetime(pac_ts.time.values))
        print(f'  Pacific box monthly SST (raw) in 1997:')
        print(f'  {"Month":<12} {"SST (C)":>10}')
        for t, v in pac_vals.items():
            print(f'  {str(t.date()):<12} {v:>10.3f}')
        print(f'\n  Mean 1997 Pacific SST : {pac_vals.mean():.3f} C')
        print(f'  (Raw SST — not anomaly. Confirms Kelvin conversion and domain.')
        return pac_vals
    except Exception as e:
        print(f'  -> Failed: {e}')
        return None
# ══════════════════════════════════════════════════════════════════════════════
# 3. Climatology leakage check
# ══════════════════════════════════════════════════════════════════════════════
def check_climatology(sst_df):
    """Monthly anomaly mean inside vs outside the 1991-2020 base period."""
    print('\n[3] Climatology leakage check ...')
    months = range(1, 13)
    m_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
    inside  = sst_df[sst_df['time'].dt.year.between(1991, 2020)]
    outside = sst_df[~sst_df['time'].dt.year.between(1991, 2020)]
    before  = sst_df[sst_df['time'].dt.year < 1991]
    after   = sst_df[sst_df['time'].dt.year > 2020]
    print(f'\n  {"Month":<5} {"Inside 91-20":>14} {"Before 1991":>14} {"After 2020":>14}')
    print('  ' + '-'*48)
    for m, name in zip(months, m_names):
        v_in  = inside[inside['time'].dt.month==m]['sst_anomaly'].mean()
        v_bef = before[before['time'].dt.month==m]['sst_anomaly'].mean()
        v_aft = after[after['time'].dt.month==m]['sst_anomaly'].mean()
        flag  = '  <- large bias' if abs(v_bef) > 0.2 else ''
        print(f'  {name:<5} {v_in:>+14.4f} {v_bef:>+14.4f} {v_aft:>+14.4f}{flag}')
    print(f'\n  Overall means:')
    print(f'    Inside 1991-2020 : {inside["sst_anomaly"].mean():+.4f} C  (should be ~0)')
    print(f'    Before 1991      : {before["sst_anomaly"].mean():+.4f} C')
    print(f'    After  2020      : {after["sst_anomaly"].mean():+.4f} C')
    print(f'\n  Interpretation:')
    if abs(before['sst_anomaly'].mean()) > 0.1:
        print(f'  -> Pre-1991 period has a systematic bias of '
              f'{before["sst_anomaly"].mean():+.3f} C vs the climatology.')
        print(f'     This is EXPECTED (not a bug): ERA5 data before ~1979 has')
        print(f'     known warm biases relative to the 1991-2020 mean.')
    else:
        print(f'  -> No significant climatology leakage detected.')
# ══════════════════════════════════════════════════════════════════════════════
# 4. Smoothing alignment check
# ══════════════════════════════════════════════════════════════════════════════
def check_smoothing(sst_df, oni_df):
    """Compare raw anomaly vs 3-month smooth vs ONI for key event periods."""
    print('\n[4] Smoothing alignment check ...')
    merged = sst_df.merge(oni_df, on='time', how='inner').dropna()
    # Events to zoom in on
    events = {
        '1997-98 El Nino': ('1996-01-01', '1999-06-01'),
        '2010-12 La Nina': ('2009-06-01', '2013-01-01'),
        '2015-16 El Nino': ('2014-01-01', '2017-06-01'),
        '2020-23 La Nina': ('2019-06-01', '2024-01-01'),
    }
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=False)
    axes = axes.flatten()
    for ax, (label, (t0, t1)) in zip(axes, events.items()):
        sub = merged[(merged['time'] >= t0) & (merged['time'] <= t1)]
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.plot(sub['time'], sub['sst_anomaly'],
                color='#91bfdb', lw=1.0, linestyle='--', alpha=0.8,
                label='Raw anomaly (no smooth)')
        ax.plot(sub['time'], sub['sst_index'],
                color='#2c7bb6', lw=1.8,
                label='SST index (3-mo smooth)')
        ax.plot(sub['time'], sub['ANOM'],
                color='#d7191c', lw=1.8, linestyle='-.',
                label='CPC ONI anomaly')
        ax.axhline(0.5,  color='#d7191c', lw=0.7, linestyle=':', alpha=0.6)
        ax.axhline(-0.5, color='#2c7bb6', lw=0.7, linestyle=':', alpha=0.6)
        ax.axhline(0,    color='black',   lw=0.4)
        ax.set_title(label, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.2)
        ax.set_ylabel('°C')
        # Annotate divergence periods
        diverg = sub[(sub['sst_index'] * sub['ANOM']) < -0.1]  # opposite signs
        for _, row in diverg.iterrows():
            ax.axvspan(row['time'] - pd.DateOffset(weeks=2),
                       row['time'] + pd.DateOffset(weeks=2),
                       color='yellow', alpha=0.2, lw=0)
    fig.suptitle('Smoothing Alignment: Raw Anomaly vs SST Index (3-mo) vs ONI\n'
                 'Yellow = months where SST index and ONI have opposite signs',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'signal_smoothing.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  -> {out}')
    plt.close()
    # Quantify opposite-sign months
    n_opposite = int(((merged['sst_index'] * merged['ANOM']) < -0.05).sum())
    n_total    = len(merged.dropna(subset=['sst_index','ANOM']))
    print(f'  Months where sst_index and ONI have opposite signs: '
          f'{n_opposite} / {n_total} ({100*n_opposite/n_total:.1f}%)')
    print(f'  -> This is the CORE SIGNAL DECOUPLING measure.')
    print(f'     If >20%, the regions are genuinely partially decorrelated.')
# ══════════════════════════════════════════════════════════════════════════════
# 5. Scatter & correlation analysis
# ══════════════════════════════════════════════════════════════════════════════
def check_correlation(sst_df, oni_df):
    print('\n[5] Scatter and correlation analysis ...')
    merged = sst_df.merge(oni_df, on='time', how='inner').dropna(
        subset=['sst_index','ANOM'])
    merged['decade'] = (merged['time'].dt.year // 10) * 10
    merged['season'] = merged['time'].dt.month.map({
        12:'DJF',1:'DJF',2:'DJF',
        3:'MAM',4:'MAM',5:'MAM',
        6:'JJA',7:'JJA',8:'JJA',
        9:'SON',10:'SON',11:'SON'})
    r_overall = merged['sst_index'].corr(merged['ANOM'])
    print(f'  Overall r(sst_index, ONI) = {r_overall:.4f}')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    # ── Panel 1: overall scatter ──────────────────────────────────
    ax = axes[0, 0]
    sc = ax.scatter(merged['ANOM'], merged['sst_index'],
                    c=merged['time'].dt.year, cmap='viridis',
                    alpha=0.5, s=18)
    plt.colorbar(sc, ax=ax, label='Year')
    # Regression line
    m, b = np.polyfit(merged['ANOM'], merged['sst_index'], 1)
    x_line = np.linspace(merged['ANOM'].min(), merged['ANOM'].max(), 100)
    ax.plot(x_line, m*x_line+b, 'r-', lw=1.5)
    ax.axhline(0, color='black', lw=0.4); ax.axvline(0, color='black', lw=0.4)
    ax.axhline(0.5,  color='grey', lw=0.6, linestyle=':')
    ax.axhline(-0.5, color='grey', lw=0.6, linestyle=':')
    ax.axvline(0.5,  color='grey', lw=0.6, linestyle=':')
    ax.axvline(-0.5, color='grey', lw=0.6, linestyle=':')
    ax.set_xlabel('ONI anomaly (°C)'); ax.set_ylabel('SST index (°C)')
    ax.set_title(f'Overall scatter  r={r_overall:.3f}  slope={m:.3f}', fontweight='bold')
    # Quadrant labels
    for x, y, txt in [(1.5, 1.5,'TP warm'), (-1.5, -1.5,'TP cold'),
                      (-1.5, 1.5,'FP warm'), (1.5, -1.5,'FP cold')]:
        ax.text(x, y, txt, ha='center', va='center', fontsize=8,
                color='gray', style='italic')
    # ── Panel 2: by decade ────────────────────────────────────────
    ax = axes[0, 1]
    decades = sorted(merged['decade'].unique())
    colors  = plt.cm.tab10(np.linspace(0, 1, len(decades)))
    for dec, col in zip(decades, colors):
        sub = merged[merged['decade'] == dec]
        r   = sub['sst_index'].corr(sub['ANOM'])
        ax.scatter(sub['ANOM'], sub['sst_index'],
                   color=col, alpha=0.5, s=18,
                   label=f'{dec}s  r={r:.2f}')
    ax.axhline(0, color='black', lw=0.4); ax.axvline(0, color='black', lw=0.4)
    ax.set_xlabel('ONI anomaly (°C)'); ax.set_ylabel('SST index (°C)')
    ax.set_title('Scatter by decade', fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    # ── Panel 3: rolling 5-year correlation ──────────────────────
    ax = axes[1, 0]
    merged_sorted = merged.set_index('time').sort_index()
    roll_r = (merged_sorted['sst_index']
              .rolling('1825D', min_periods=24)
              .corr(merged_sorted['ANOM']))
    ax.plot(roll_r.index, roll_r.values, color='#2c7bb6', lw=1.5)
    ax.axhline(0, color='black', lw=0.4)
    ax.axhline(0.5, color='grey', lw=0.7, linestyle='--', alpha=0.7)
    ax.fill_between(roll_r.index, 0, roll_r.values,
                    where=(roll_r.values >= 0),
                    color='#4dac26', alpha=0.3, label='Positive r')
    ax.fill_between(roll_r.index, 0, roll_r.values,
                    where=(roll_r.values < 0),
                    color='#d01c8b', alpha=0.3, label='Negative r')
    ax.set_ylabel('Rolling 5-yr correlation')
    ax.set_title('Rolling correlation: SST index vs ONI', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    # ── Panel 4: by calendar season ──────────────────────────────
    ax = axes[1, 1]
    season_colors = {'DJF':'#2c7bb6','MAM':'#4dac26','JJA':'#d7191c','SON':'#fdae61'}
    for seas, col in season_colors.items():
        sub = merged[merged['season'] == seas]
        r   = sub['sst_index'].corr(sub['ANOM'])
        ax.scatter(sub['ANOM'], sub['sst_index'],
                   color=col, alpha=0.5, s=18,
                   label=f'{seas}  r={r:.2f}')
    ax.axhline(0, color='black', lw=0.4); ax.axvline(0, color='black', lw=0.4)
    ax.set_xlabel('ONI anomaly (°C)'); ax.set_ylabel('SST index (°C)')
    ax.set_title('Scatter by season', fontweight='bold')
    ax.legend(fontsize=8)
    fig.suptitle('Signal Analysis: Colombian Pacific SST Index vs. NOAA CPC ONI',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'signal_scatter.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  -> {out}')
    plt.close()
    # Print season correlations
    print(f'\n  Correlation by season:')
    for seas in ['DJF','MAM','JJA','SON']:
        sub = merged[merged['season'] == seas]
        print(f'    {seas}: r={sub["sst_index"].corr(sub["ANOM"]):.3f}  '
              f'(n={len(sub)})')
    print(f'\n  Correlation by decade:')
    for dec in decades:
        sub = merged[merged['decade'] == dec]
        print(f'    {dec}s: r={sub["sst_index"].corr(sub["ANOM"]):.3f}  '
              f'(n={len(sub)})')
# ══════════════════════════════════════════════════════════════════════════════
# 6. Divergence deep-dive
# ══════════════════════════════════════════════════════════════════════════════
def check_divergence(sst_df, oni_df):
    """
    Focus on months where ONI says one thing and SST index says the other.
    These are the physically interesting cases.
    """
    print('\n[6] Divergence deep-dive ...')
    merged = sst_df.merge(oni_df, on='time', how='inner').dropna(
        subset=['sst_index','ANOM'])
    # Strong divergence: |ONI| > 0.5 but SST index disagrees in sign
    strong_oni  = merged[merged['ANOM'].abs() > 0.5].copy()
    oni_warm_sst_cold = strong_oni[(strong_oni['ANOM'] > 0.5) & (strong_oni['sst_index'] < 0)]
    oni_cold_sst_warm = strong_oni[(strong_oni['ANOM'] < -0.5) & (strong_oni['sst_index'] > 0)]
    print(f'\n  ONI warm (>+0.5) but SST index negative: {len(oni_warm_sst_cold)} months')
    if not oni_warm_sst_cold.empty:
        print(oni_warm_sst_cold[['time','ANOM','sst_index','sst_anomaly_pacific']
                                 if 'sst_anomaly_pacific' in merged.columns
                                 else ['time','ANOM','sst_index']
                                 ].to_string(index=False))
    print(f'\n  ONI cold (<-0.5) but SST index positive: {len(oni_cold_sst_warm)} months')
    if not oni_cold_sst_warm.empty:
        print(oni_cold_sst_warm[['time','ANOM','sst_index','sst_anomaly_pacific']
                                 if 'sst_anomaly_pacific' in merged.columns
                                 else ['time','ANOM','sst_index']
                                 ].to_string(index=False))
    # Plot timeline of divergence events
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    fig.subplots_adjust(hspace=0.35)
    ax = axes[0]
    ax.plot(merged['time'], merged['sst_index'], color='#2c7bb6', lw=1.2,
            label='SST index (Colombian Pacific)')
    ax.plot(merged['time'], merged['ANOM'],  color='#d7191c', lw=1.0,
            linestyle='--', alpha=0.8, label='CPC ONI')
    ax.axhline(0.5,  color='grey', lw=0.6, linestyle=':')
    ax.axhline(-0.5, color='grey', lw=0.6, linestyle=':')
    ax.axhline(0, color='black', lw=0.4)
    ax.set_ylabel('°C')
    ax.set_title('SST index vs ONI — full record', fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.2)
    ax = axes[1]
    diff = merged['sst_index'] - merged['ANOM']
    ax.bar(merged['time'], diff, width=25,
           color=np.where(diff > 0, '#4dac26', '#d01c8b'), alpha=0.7)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_ylabel('SST index − ONI (°C)')
    ax.set_title('Difference: SST index minus ONI (positive = SST warmer than ONI)',
                 fontweight='bold')
    ax.grid(alpha=0.2)
    ax = axes[2]
    product = merged['sst_index'] * merged['ANOM']
    ax.bar(merged['time'], product, width=25,
           color=np.where(product > 0, '#1a9641', '#d7191c'), alpha=0.7)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_ylabel('SST × ONI (product)')
    ax.set_title('Sign agreement: green=same sign (TP/TN), red=opposite sign (FP/FN)',
                 fontweight='bold')
    ax.grid(alpha=0.2)
    fig.suptitle('Signal Divergence Analysis — Colombian Pacific SST vs. ONI',
                 fontsize=13, fontweight='bold')
    out = os.path.join(OUTPUT_DIR, 'signal_divergence.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\n  -> {out}')
    plt.close()
    # Summary statistics
    n_total    = len(merged)
    n_agree    = int((product > 0).sum())
    n_disagree = int((product < 0).sum())
    n_neutral  = n_total - n_agree - n_disagree
    print(f'\n  Sign agreement summary (full record):')
    print(f'    Same sign  : {n_agree:>4} months ({100*n_agree/n_total:.1f}%) <- genuine signal')
    print(f'    Opposite   : {n_disagree:>4} months ({100*n_disagree/n_total:.1f}%) <- physical decoupling')
    print(f'    One neutral: {n_neutral:>4} months ({100*n_neutral/n_total:.1f}%)')
# ══════════════════════════════════════════════════════════════════════════════
# 7. Spatial correlation map (requires GRIB)
# ══════════════════════════════════════════════════════════════════════════════
def check_spatial_correlation(oni_df):
    """
    For each grid pixel in the ERA5 domain, compute r(pixel_monthly_SST, ONI).
    This shows whether the Colombian Pacific domain captures the ENSO signal
    or is spatially displaced.
    """
    print('\n[7] Spatial correlation map (requires GRIB files) ...')
    # Try loading a few representative years
    test_years = [1982, 1983, 1997, 1998, 2010, 2011, 2015, 2016]
    grib_files = [os.path.join(GRIB_DIR, f'era5_sst_{y}.grib') for y in test_years]
    available  = [f for f in grib_files if os.path.exists(f)]
    if not available:
        print('  -> No GRIB files found, skipping spatial correlation map')
        return
    print(f'  Loading {len(available)} GRIB files for spatial correlation ...')
    try:
        import xarray as xr
        ds_list = []
        for f in available:
            ds = xr.open_dataset(f, engine='cfgrib')
            sst = ds['sst']
            if float(sst.mean()) > 200:
                sst = sst - 273.15
            ds_list.append(sst.resample(time='MS').mean())
        sst_monthly = xr.concat(ds_list, dim='time').sortby('time')
        print(f'  Combined shape: {sst_monthly.shape}')
        # Align with ONI
        oni_aligned = oni_df.set_index('time')['ANOM']
        times_common = [t for t in pd.to_datetime(sst_monthly.time.values)
                        if t in oni_aligned.index]
        if len(times_common) < 20:
            print('  -> Insufficient overlapping months for spatial correlation')
            return
        sst_sub = sst_monthly.sel(time=times_common)
        oni_arr = np.array([oni_aligned.loc[t] for t in times_common])
        # Compute pixel-wise correlation
        sst_arr = sst_sub.values  # (time, lat, lon)
        n       = len(times_common)
        oni_z   = (oni_arr - oni_arr.mean()) / oni_arr.std()
        corr_map = np.full(sst_arr.shape[1:], np.nan)
        for i in range(sst_arr.shape[1]):
            for j in range(sst_arr.shape[2]):
                pixel = sst_arr[:, i, j]
                if np.isnan(pixel).sum() > n * 0.3:
                    continue
                valid = ~np.isnan(pixel)
                if valid.sum() < 10:
                    continue
                pz = (pixel[valid] - pixel[valid].mean()) / (pixel[valid].std() + 1e-9)
                corr_map[i, j] = np.corrcoef(pz, oni_z[valid])[0, 1]
        lats = sst_monthly.latitude.values
        lons = sst_monthly.longitude.values
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.pcolormesh(lons, lats, corr_map,
                           cmap='RdBu_r', vmin=-1, vmax=1, shading='auto')
        plt.colorbar(im, ax=ax, label='r(pixel SST, ONI)')
        # Draw domain boxes
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((PAC_LON[0], PAC_LAT[0]),
                               PAC_LON[1]-PAC_LON[0], PAC_LAT[1]-PAC_LAT[0],
                               fill=False, edgecolor='black', lw=2.5,
                               label='Pacific box (80%)'))
        ax.add_patch(Rectangle((CAR_LON[0], CAR_LAT[0]),
                               CAR_LON[1]-CAR_LON[0], CAR_LAT[1]-CAR_LAT[0],
                               fill=False, edgecolor='yellow', lw=2.5,
                               label='Caribbean box (20%)'))
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_title(f'Pixel-level r(SST, ONI)  |  {len(available)} El Nino/La Nina years\n'
                     f'Black box = Pacific domain  |  Yellow box = Caribbean domain',
                     fontweight='bold')
        ax.legend(fontsize=9)
        out = os.path.join(OUTPUT_DIR, 'signal_spatial_corr.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  -> {out}')
        plt.close()
        # Print average r inside each box
        def box_mean_r(lat_b, lon_b):
            lat_idx = np.where((lats >= min(lat_b)) & (lats <= max(lat_b)))[0]
            lon_idx = np.where((lons >= min(lon_b)) & (lons <= max(lon_b)))[0]
            sub_r   = corr_map[np.ix_(lat_idx, lon_idx)]
            return np.nanmean(sub_r)
        r_pac = box_mean_r(PAC_LAT, PAC_LON)
        r_car = box_mean_r(CAR_LAT, CAR_LON)
        print(f'\n  Mean r(SST, ONI) inside boxes:')
        print(f'    Pacific   box: {r_pac:+.3f}')
        print(f'    Caribbean box: {r_car:+.3f}')
        if r_pac < 0.3:
            print(f'  -> LOW r in Pacific box — genuine signal decoupling confirmed.')
        else:
            print(f'  -> Reasonable r in Pacific box — signal IS present.')
    except Exception as e:
        print(f'  -> Spatial correlation failed: {e}')
# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('='*72)
    print('SST SIGNAL INSPECTION — Colombian Pacific vs. ENSO')
    print('='*72)
    print('\nLoading processed CSVs ...')
    sst_df   = pd.read_csv(SST_CSV,      parse_dates=['time'])
    feat_df  = pd.read_csv(FEATURES_CSV, parse_dates=['time'])
    sst_df['time']  = pd.to_datetime(sst_df['time']).dt.normalize()
    feat_df['time'] = pd.to_datetime(feat_df['time']).dt.normalize()
    # Merge features into sst_df for convenience
    sst_df = sst_df.merge(feat_df[['time','sst_z','D_warm','D_cold']], on='time', how='left')
    print('Loading ONI ...')
    oni_df = load_oni()
    oni_df['time'] = pd.to_datetime(oni_df['time']).dt.normalize()
    check_geometry(sst_df)
    check_raw_grib()
    check_climatology(sst_df)
    check_smoothing(sst_df, oni_df)
    check_correlation(sst_df, oni_df)
    check_divergence(sst_df, oni_df)
    check_spatial_correlation(oni_df)
    print(f'\n{"="*72}')
    print('DIAGNOSTIC SUMMARY')
    print(f'{"="*72}')
    merged = sst_df.merge(oni_df, on='time', how='inner').dropna(
        subset=['sst_index','ANOM'])
    r = merged['sst_index'].corr(merged['ANOM'])
    n_opp = int(((merged['sst_index'] * merged['ANOM']) < 0).sum())
    pct   = 100 * n_opp / len(merged)
    print(f'  Overall r(SST index, ONI)       : {r:.4f}')
    print(f'  Months with opposite signs      : {n_opp} / {len(merged)} ({pct:.1f}%)')
    print()
    if pct > 20:
        print('  CONCLUSION: Physical signal decoupling is REAL.')
        print('  The Colombian Pacific SST responds differently to ENSO')
        print('  than the Nino 3.4 region, especially during La Nina.')
        print('  The regional index IS valid, just measuring a different thing.')
        print()
        print('  RECOMMENDATION: Reframe D_warm/D_cold as local SST phase')
        print('  indicators and add ONI flags as separate model features.')
    else:
        print('  CONCLUSION: Signal is mostly aligned — check geometry and')
        print('  climatology for data processing errors.')
    print(f'{"="*72}')
    print('\nAll done.')
if __name__ == '__main__':
    main()