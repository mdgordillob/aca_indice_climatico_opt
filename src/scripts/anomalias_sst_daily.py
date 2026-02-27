import os
import xarray as xr
import pandas as pd
import numpy as np
import warnings
import urllib.request
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Climatology base period (1991–2020, NOAA/CPC standard) ───────────────────
CLIM_START = 1991
CLIM_END   = 2020

# ── Regional weights ──────────────────────────────────────────────────────────
W_PACIFIC   = 0.80
W_CARIBBEAN = 0.20

# ── Colombian coastal bounding boxes ─────────────────────────────────────────
PAC_LAT = (-4,  10)   # Colombian Pacific  (south, north)
PAC_LON = (-84, -76)  # (west, east)

CAR_LAT = ( 8,  13)   # Colombian Caribbean
CAR_LON = (-76, -68)

# ── ONI-consistent threshold convention ──────────────────────────────────────
ONI_THRESHOLD  = 0.5   # °C
ONI_MIN_CONSEC = 5     # consecutive overlapping 3-month seasons

# ── Niño 3 reference (IDEAM standard for Colombia) ───────────────────────────
# ERSSTv5, 1991–2020 base period — all Niño regions in one file
NINO3_URL  = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
NINO3_FILE = "ersst5.nino.mth.91-20.ascii"   # saved to output_dir

REGION_NAME = "Colombian Pacific (80%) + Caribbean (20%)"

# ── Known ENSO events for sanity checks (IDEAM/NOAA consensus) ───────────────
EL_NINO_EVENTS = {
    '1972-73': [1972, 1973],
    '1982-83': [1982, 1983],
    '1987':    [1987],
    '1997-98': [1997, 1998],
    '2009-10': [2009, 2010],
    '2015-16': [2015, 2016],
}

LA_NINA_EVENTS = {
    '1970-71': [1970, 1971],
    '1973-76': [1973, 1974, 1975, 1976],
    '1988-89': [1988, 1989],
    '1999-00': [1999, 2000],
    '2007-08': [2007, 2008],
    '2010-12': [2010, 2011, 2012],
    '2020-23': [2020, 2021, 2022, 2023],
}


# ══════════════════════════════════════════════════════════════════════════════
# Niño 3 reference download
# ══════════════════════════════════════════════════════════════════════════════

def download_nino3(output_dir):
    """
    Download the NOAA CPC ERSSTv5 Niño index file (1991–2020 base period).
    Contains monthly Niño 1+2, 3, 4, and 3.4 anomalies and total SST.

    Source: https://www.cpc.ncep.noaa.gov/data/indices/
    File  : ersst5.nino.mth.91-20.ascii

    Returns pd.DataFrame with columns: time, nino3_sst, nino3_anom
    Returns None on download failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    local_path = os.path.join(output_dir, NINO3_FILE)

    # Download if not already cached
    if not os.path.exists(local_path):
        print(f"   Downloading Niño 3 index from NOAA CPC...")
        try:
            urllib.request.urlretrieve(NINO3_URL, local_path)
            print(f"   ✓ Saved to {local_path}")
        except Exception as e:
            print(f"   ✗ Download failed: {e}")
            print(f"   → Manually download from:\n     {NINO3_URL}")
            return None
    else:
        print(f"   ✓ Using cached file: {local_path}")

    # Parse the fixed-width file
    # Format: YR MON NINO1+2_SST NINO1+2_ANOM NINO3_SST NINO3_ANOM
    #              NINO4_SST NINO4_ANOM NINO3.4_SST NINO3.4_ANOM
    try:
        df = pd.read_csv(
            local_path,
            sep=r'\s+',
            engine='python',
            header=0,
            names=['YR','MON',
                   'NINO12_SST','NINO12_ANOM',
                   'NINO3_SST', 'NINO3_ANOM',
                   'NINO4_SST', 'NINO4_ANOM',
                   'NINO34_SST','NINO34_ANOM'],
        )
        df = df[pd.to_numeric(df['YR'], errors='coerce').notna()].copy()
        df['YR']  = df['YR'].astype(int)
        df['MON'] = df['MON'].astype(int)
        df['time'] = pd.to_datetime(
            df['YR'].astype(str) + '-' + df['MON'].astype(str).str.zfill(2) + '-01'
        )

        # 3-month centred rolling mean on the raw Niño 3 anomaly
        df = df.sort_values('time').reset_index(drop=True)
        df['nino3_index'] = (
            pd.to_numeric(df['NINO3_ANOM'], errors='coerce')
            .rolling(window=3, center=True, min_periods=3)
            .mean()
        )

        nino3_df = df[['time', 'NINO3_SST', 'NINO3_ANOM', 'nino3_index']].rename(
            columns={'NINO3_SST': 'nino3_sst', 'NINO3_ANOM': 'nino3_anom'}
        )
        nino3_df[['nino3_sst','nino3_anom','nino3_index']] = (
            nino3_df[['nino3_sst','nino3_anom','nino3_index']]
            .apply(pd.to_numeric, errors='coerce')
        )

        valid = nino3_df.dropna(subset=['nino3_index'])
        print(f"   Niño 3 records: {len(valid)}  "
              f"({valid['time'].min().date()} → {valid['time'].max().date()})")
        return nino3_df

    except Exception as e:
        print(f"   ✗ Failed to parse Niño 3 file: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_daily_sst_files(grib_folder):
    """
    Load and combine all SST GRIB files, automatically resampling hourly
    data to daily means when detected.
    """
    sst_files = sorted(Path(grib_folder).glob('era5_sst_*.grib'))

    if not sst_files:
        raise FileNotFoundError(f"No SST GRIB files found in {grib_folder}")

    print(f"Found {len(sst_files)} SST GRIB files")

    sst_data_list = []
    for file_path in sst_files:
        print(f"  Loading {file_path.name}...", end="")
        try:
            ds = xr.open_dataset(file_path, engine='cfgrib')
            if 'sst' not in ds.data_vars:
                print(f" ✗ (no sst variable)")
                continue

            sst = ds['sst']

            # Detect temporal resolution from first few timestamps
            if len(sst.time) > 1:
                time_diff = pd.to_timedelta(
                    np.diff(sst.time.values[:5])
                ).median()
                is_hourly = time_diff < pd.Timedelta('2h')
            else:
                is_hourly = False

            if is_hourly:
                sst = sst.resample(time='1D').mean()
                print(f" ✓ hourly→daily ({len(sst.time)} days)")
            else:
                print(f" ✓ ({len(sst.time)} days)")

            sst_data_list.append(sst)

        except Exception as e:
            print(f" ✗ (Error: {e})")

    combined_sst = xr.concat(sst_data_list, dim='time')
    combined_sst = combined_sst.sortby('time')

    # ── K → °C conversion ─────────────────────────────────────────
    raw_mean = float(combined_sst.mean())
    if raw_mean > 200:
        print(f"\n⚠️  Kelvin detected (mean={raw_mean:.1f} K) — converting to °C")
        combined_sst = combined_sst - 273.15
        combined_sst.attrs['units'] = 'C'
    else:
        print(f"\n✓  Units look like °C (mean={raw_mean:.1f})")

    print(f"\nCombined SST shape : {combined_sst.shape}")
    print(f"Time range         : {combined_sst.time.values[0]} to {combined_sst.time.values[-1]}")
    print(f"Spatial domain     : lat [{float(combined_sst.latitude.min()):.2f}, "
          f"{float(combined_sst.latitude.max()):.2f}]  "
          f"lon [{float(combined_sst.longitude.min()):.2f}, "
          f"{float(combined_sst.longitude.max()):.2f}]")

    return combined_sst


# ══════════════════════════════════════════════════════════════════════════════
# Spatial helpers
# ══════════════════════════════════════════════════════════════════════════════

def subset_region(da, lat_bounds, lon_bounds):
    """Subset DataArray to a bounding box, handling ascending/descending lat."""
    lat_min, lat_max = min(lat_bounds), max(lat_bounds)
    lon_min, lon_max = min(lon_bounds), max(lon_bounds)

    lats = da.latitude.values
    lat_slice = slice(lat_max, lat_min) if lats[0] > lats[-1] else slice(lat_min, lat_max)

    subset = da.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))

    if subset.size == 0:
        raise ValueError(
            f"Subset is empty for lat={lat_bounds}, lon={lon_bounds}. "
            f"Check that your ERA5 files cover this domain."
        )
    return subset


def area_weighted_mean_ts(da):
    """Cosine-latitude area-weighted spatial mean → 1-D time series."""
    weights = np.cos(np.deg2rad(da.latitude))
    return da.weighted(weights).mean(dim=('latitude', 'longitude'))


# ══════════════════════════════════════════════════════════════════════════════
# Core processing
# ══════════════════════════════════════════════════════════════════════════════

def resample_to_monthly_sst(sst_daily):
    """Resample daily SST to monthly means."""
    return sst_daily.resample(time='MS').mean()


def calculate_climatology(sst_monthly, clim_start=CLIM_START, clim_end=CLIM_END):
    """
    Compute the monthly climatological mean over a fixed 30-year base period.
    Returns one mean value per calendar month (12 values) at each grid point.
    """
    base = sst_monthly.sel(
        time=sst_monthly.time.dt.year.isin(range(clim_start, clim_end + 1))
    )

    if len(base.time) == 0:
        raise ValueError(
            f"No data found in the climatology base period "
            f"{clim_start}–{clim_end}. Adjust CLIM_START / CLIM_END."
        )

    n_years = len(base.time) / 12
    print(f"  Climatology base: {clim_start}–{clim_end} "
          f"({n_years:.1f} years of monthly data)")

    return base.groupby('time.month').mean(dim='time')


def calculate_monthly_anomalies(sst_monthly, climatology):
    """Monthly SST anomaly:  A_t = SST_t − SST_clim_{m(t)}"""
    return sst_monthly.groupby('time.month') - climatology


# ══════════════════════════════════════════════════════════════════════════════
# Regional index — Colombian coast (80% Pacific / 20% Caribbean)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_sst_index(anomalies, pac_weight=W_PACIFIC, car_weight=W_CARIBBEAN):
    """
    Compute the Colombian coastal SST anomaly index:
      1. Split domain → Pacific [−84,−76] and Caribbean [−76,−68].
      2. Cosine-latitude area-weighted spatial mean per region.
      3. Combine: index = 0.80 * pac + 0.20 * car
      4. 3-month centred running mean (same smoothing as ONI).

    Falls back to Pacific-only if Caribbean pixels are unavailable.

    Returns pd.DataFrame with columns:
        time, sst_anomaly, sst_anomaly_pacific, sst_anomaly_caribbean,
        sst_index, season
    """
    pac_data = subset_region(anomalies, PAC_LAT, PAC_LON)
    print(f"   Pacific   grid: {pac_data.shape[-2]}×{pac_data.shape[-1]}  "
          f"lat [{float(pac_data.latitude.min()):.1f}, {float(pac_data.latitude.max()):.1f}]  "
          f"lon [{float(pac_data.longitude.min()):.1f}, {float(pac_data.longitude.max()):.1f}]")

    try:
        car_data = subset_region(anomalies, CAR_LAT, CAR_LON)
        print(f"   Caribbean grid: {car_data.shape[-2]}×{car_data.shape[-1]}  "
              f"lat [{float(car_data.latitude.min()):.1f}, {float(car_data.latitude.max()):.1f}]  "
              f"lon [{float(car_data.longitude.min()):.1f}, {float(car_data.longitude.max()):.1f}]")
        has_caribbean = True
    except ValueError as e:
        print(f"   ⚠️  Caribbean not available ({e}) — Pacific only (100%)")
        has_caribbean = False

    pac_ts = area_weighted_mean_ts(pac_data)

    if has_caribbean:
        car_ts     = area_weighted_mean_ts(car_data)
        combined   = pac_weight * pac_ts + car_weight * car_ts
        car_values = car_ts.values
        print(f"   Weights: Pacific {pac_weight*100:.0f}%  Caribbean {car_weight*100:.0f}%")
    else:
        combined   = pac_ts
        car_values = np.full(len(pac_ts), np.nan)

    # 3-month centred rolling mean
    anom_series  = pd.Series(combined.values, index=pd.to_datetime(combined.time.values))
    index_series = anom_series.rolling(window=3, center=True, min_periods=3).mean()

    # Season labels (DJF, JFM, …)
    month_abbr = ['Jan','Feb','Mar','Apr','May','Jun',
                  'Jul','Aug','Sep','Oct','Nov','Dec']

    def season_label(ts):
        prev_m = (ts.month - 2) % 12
        curr_m = (ts.month - 1) % 12
        next_m =  ts.month      % 12
        return month_abbr[prev_m][0] + month_abbr[curr_m][0] + month_abbr[next_m][0]

    sst_df = pd.DataFrame({
        'time':                  anom_series.index,
        'sst_anomaly':           anom_series.values,
        'sst_anomaly_pacific':   pac_ts.values,
        'sst_anomaly_caribbean': car_values,
        'sst_index':             index_series.values,
    })
    sst_df['season'] = sst_df['time'].apply(season_label)

    return sst_df


# ══════════════════════════════════════════════════════════════════════════════
# ONI-consistent phase indicators
# ══════════════════════════════════════════════════════════════════════════════

def consecutive_flag(condition_mask, min_run=ONI_MIN_CONSEC):
    """
    Return a binary array that is 1 only for months belonging to a run of
    at least `min_run` consecutive True values — identical to the NOAA ONI
    5-consecutive-overlapping-seasons rule.
    """
    flag = np.zeros(len(condition_mask), dtype=int)
    arr  = condition_mask.values
    i    = 0
    while i < len(arr):
        if arr[i]:
            j = i
            while j < len(arr) and arr[j]:
                j += 1
            if (j - i) >= min_run:
                flag[i:j] = 1
            i = j
        else:
            i += 1
    return flag


def build_sst_features(sst_df, lags=None, training_mask=None):
    """
    Construct modelling features from the regional SST index:

        sst_z        = (sst_index − mean) / std      standardised index
        D_warm       = 1 for El Niño months (≥5 consec seasons ≥ +0.5°C)
        D_cold       = 1 for La Niña months (≥5 consec seasons ≤ −0.5°C)
        sst_z_lag{l} for l in lags                    distributed lags

    Phase detection follows the official NOAA/CPC ONI methodology.
    Standardisation uses the training window only when training_mask is given.
    """
    if lags is None:
        lags = [0, 1, 2, 3]

    df = sst_df[['time', 'sst_index']].copy()

    # Standardisation
    if training_mask is not None:
        mu  = df.loc[training_mask, 'sst_index'].mean()
        sig = df.loc[training_mask, 'sst_index'].std(ddof=1)
    else:
        mu  = df['sst_index'].mean()
        sig = df['sst_index'].std(ddof=1)

    print(f"  SST index standardisation: mean={mu:.4f} °C, std={sig:.4f} °C")
    df['sst_z'] = (df['sst_index'] - mu) / sig

    # ONI-consistent phase indicators
    warm_mask = df['sst_index'] >= +ONI_THRESHOLD
    cold_mask = df['sst_index'] <= -ONI_THRESHOLD

    df['D_warm'] = consecutive_flag(warm_mask)
    df['D_cold'] = consecutive_flag(cold_mask)

    warm_events = int(pd.Series(df['D_warm']).diff().eq(1).sum())
    cold_events = int(pd.Series(df['D_cold']).diff().eq(1).sum())
    print(f"  El Niño events (≥{ONI_MIN_CONSEC} consec seasons ≥ +{ONI_THRESHOLD}°C): "
          f"{warm_events}  ({df['D_warm'].sum()} months flagged)")
    print(f"  La Niña events (≥{ONI_MIN_CONSEC} consec seasons ≤ -{ONI_THRESHOLD}°C): "
          f"{cold_events}  ({df['D_cold'].sum()} months flagged)")

    # Distributed lags
    for lag in lags:
        df[f'sst_z_lag{lag}'] = df['sst_z'].shift(lag)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Sanity checks
# ══════════════════════════════════════════════════════════════════════════════

def run_sanity_checks(sst_df, features_df, nino3_df=None):
    """
    Cross-validate the Colombian Pacific SST index against:
      1. Known El Niño events — peak index should be positive
      2. Known La Niña events — peak index should be negative
      3. Monthly anomaly bias inside the 1991–2020 base period (should be ≈0)
      4. Phase count plausibility
      5. Correlation with Niño 3 reference index (NOAA ERSSTv5, 1991–2020 base)
    """
    print("\n" + "=" * 70)
    print("SANITY CHECKS — Colombian Pacific SST Index")
    print("=" * 70)

    # ── 1. El Niño events ─────────────────────────────────────────
    print("\n[1] El Niño events — peak sst_index should be positive")
    print(f"    {'Event':<12} {'Peak index':>12}  {'Peak month':>12}  {'D_warm':>7}  OK?")
    print(f"    {'-'*58}")
    for name, years in EL_NINO_EVENTS.items():
        subset = sst_df[sst_df['time'].dt.year.isin(years)].dropna(subset=['sst_index'])
        if subset.empty:
            print(f"    {name:<12}  NO DATA")
            continue
        peak_row = subset.loc[subset['sst_index'].idxmax()]
        flag     = features_df.loc[features_df['time'] == peak_row['time'], 'D_warm']
        flag_val = int(flag.values[0]) if not flag.empty else -1
        status   = '✓' if peak_row['sst_index'] > 0 else '✗'
        print(f"    {name:<12}  {peak_row['sst_index']:>+.3f} °C   "
              f"{str(peak_row['time'].date()):>12}   {flag_val:>5}   {status}")

    # ── 2. La Niña events ─────────────────────────────────────────
    print(f"\n[2] La Niña events — peak sst_index should be negative")
    print(f"    {'Event':<12} {'Peak index':>12}  {'Peak month':>12}  {'D_cold':>7}  OK?")
    print(f"    {'-'*58}")
    for name, years in LA_NINA_EVENTS.items():
        subset = sst_df[sst_df['time'].dt.year.isin(years)].dropna(subset=['sst_index'])
        if subset.empty:
            print(f"    {name:<12}  NO DATA")
            continue
        peak_row = subset.loc[subset['sst_index'].idxmin()]
        flag     = features_df.loc[features_df['time'] == peak_row['time'], 'D_cold']
        flag_val = int(flag.values[0]) if not flag.empty else -1
        status   = '✓' if peak_row['sst_index'] < 0 else '✗'
        print(f"    {name:<12}  {peak_row['sst_index']:>+.3f} °C   "
              f"{str(peak_row['time'].date()):>12}   {flag_val:>5}   {status}")

    # ── 3. Monthly anomaly bias inside climatology period ─────────
    print(f"\n[3] Monthly anomaly mean inside {CLIM_START}–{CLIM_END} base period (target ≈ 0)")
    base    = sst_df[sst_df['time'].dt.year.between(CLIM_START, CLIM_END)]
    monthly = base.groupby(base['time'].dt.month)['sst_anomaly'].mean()
    m_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    for m, name in zip(range(1, 13), m_names):
        val  = monthly.get(m, float('nan'))
        flag = '✓' if abs(val) < 0.05 else '⚠️ '
        print(f"    {name}: {val:>+.4f} °C  {flag}")

    # ── 4. Phase count plausibility ───────────────────────────────
    print(f"\n[4] Phase counts over full record")
    warm_months = int(features_df['D_warm'].sum())
    cold_months = int(features_df['D_cold'].sum())
    total       = len(features_df.dropna(subset=['sst_index']))
    neutral     = total - warm_months - cold_months
    print(f"    El Niño months : {warm_months:>4}  ({100*warm_months/total:.1f}%)")
    print(f"    La Niña months : {cold_months:>4}  ({100*cold_months/total:.1f}%)")
    print(f"    Neutral months : {neutral:>4}  ({100*neutral/total:.1f}%)")
    print(f"    ℹ  La Niña dominance is physically expected for the Colombian Pacific")
    print(f"       (IDEAM documents stronger/longer cold responses on this coast)")

    # ── 5. Correlation with Niño 3 (ERSSTv5, 1991–2020 base) ─────
    print(f"\n[5] Correlation with Niño 3 reference (NOAA ERSSTv5, 1991–2020 base)")
    if nino3_df is not None:
        merged = sst_df[['time','sst_index']].merge(
            nino3_df[['time','nino3_index']], on='time', how='inner'
        ).dropna()
        if len(merged) > 10:
            r = merged['sst_index'].corr(merged['nino3_index'])
            flag = '✓' if 0.35 <= r <= 0.85 else '⚠️ '
            print(f"    r = {r:.4f}  (expected 0.35–0.85 for Colombian Pacific)  {flag}")
            if r < 0.35:
                print(f"    ⚠️  Low correlation — check domain or climatology period")
            elif r > 0.85:
                print(f"    ⚠️  Very high — index may be too similar to Niño 3")

            # Decade-by-decade correlation
            print(f"\n    Correlation by decade:")
            merged['decade'] = (merged['time'].dt.year // 10) * 10
            dec_corr = merged.groupby('decade').apply(
                lambda x: x['sst_index'].corr(x['nino3_index']), include_groups=False
            )
            for dec, rc in dec_corr.items():
                flag_d = '✓' if rc >= 0.3 else '⚠️ '
                print(f"      {dec}s: r = {rc:.3f}  {flag_d}")
        else:
            print(f"    ⚠️  Insufficient overlapping data for correlation")
    else:
        print(f"    ⚠️  Niño 3 reference not available — download failed or skipped")

    print("\n" + "=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_daily_sst(grib_folder, output_dir=None,
                      clim_start=CLIM_START, clim_end=CLIM_END):
    """
    Full pipeline:
      daily SST → monthly mean → 30-yr climatology (1991–2020)
      → monthly anomalies
      → Colombian coastal SST index (80% Pacific / 20% Caribbean, 3-month smooth)
      → Niño 3 reference download (NOAA ERSSTv5, IDEAM standard for Colombia)
      → modelling features (Z_t, D_warm/D_cold ONI-consistent, lags 0–3)
      → sanity checks

    Returns
    -------
    sst_df      : pd.DataFrame — monthly SST index
    features_df : pd.DataFrame — modelling features
    nino3_df    : pd.DataFrame — Niño 3 reference (or None)
    """
    print("=" * 80)
    print(f"COMPUTING REGIONAL SST ANOMALY INDEX  —  {REGION_NAME}")
    print(f"Climatology base period : {clim_start}–{clim_end}")
    print(f"ENSO reference          : Niño 3 (5°N–5°S, 150°W–90°W) — IDEAM standard")
    print(f"                          Source: NOAA CPC ERSSTv5, 1991–2020 base period")
    print("=" * 80)

    # 1. Load
    print("\n1. Loading SST GRIB files...")
    sst_daily = load_daily_sst_files(grib_folder)

    # 2. Daily → monthly
    print("\n2. Resampling to monthly means...")
    sst_monthly = resample_to_monthly_sst(sst_daily)
    print(f"   Monthly SST shape: {sst_monthly.shape}")

    # 3. Climatology
    print(f"\n3. Computing {clim_end - clim_start + 1}-year climatology...")
    climatology = calculate_climatology(sst_monthly, clim_start, clim_end)

    # 4. Anomalies
    print("\n4. Computing monthly SST anomalies...")
    anomalies = calculate_monthly_anomalies(sst_monthly, climatology)

    # 5. Colombian coastal index (80/20)
    print("\n5. Computing Colombian coastal SST index (80% Pacific / 20% Caribbean)...")
    sst_df = calculate_sst_index(anomalies)
    valid  = sst_df.dropna(subset=['sst_index'])
    print(f"   Valid records: {len(valid)}  "
          f"({valid['time'].min().date()} → {valid['time'].max().date()})")

    # 6. Download Niño 3 reference
    print("\n6. Downloading Niño 3 reference index (NOAA CPC ERSSTv5, 1991–2020 base)...")
    nino3_df = download_nino3(output_dir) if output_dir else download_nino3('.')

    # 7. Modelling features
    print(f"\n7. Building modelling features "
          f"(ONI rule: ≥{ONI_MIN_CONSEC} consec seasons beyond ±{ONI_THRESHOLD}°C)...")
    features_df = build_sst_features(sst_df, lags=[0, 1, 2, 3])

    # 8. Sanity checks
    print("\n8. Running sanity checks...")
    run_sanity_checks(sst_df, features_df, nino3_df)

    # ── Summary stats ─────────────────────────────────────────────
    print(f"\nSST index descriptive statistics:")
    print(sst_df['sst_index'].describe().to_string())
    print(f"\n── Anomaly bias ──────────────────────────────────────────────────────")
    print(f"   Combined anomaly mean : {sst_df['sst_anomaly'].mean():.4f} °C  (target ≈ 0)")
    print(f"   Pacific  anomaly mean : {sst_df['sst_anomaly_pacific'].mean():.4f} °C")
    if not sst_df['sst_anomaly_caribbean'].isna().all():
        print(f"   Caribbean anomaly mean: {sst_df['sst_anomaly_caribbean'].mean():.4f} °C")
    print(f"──────────────────────────────────────────────────────────────────────")

    # ── Save ──────────────────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        idx_path  = os.path.join(output_dir, "sst_index_colombia_pacific.csv")
        feat_path = os.path.join(output_dir, "sst_features_colombia_pacific.csv")

        sst_df.to_csv(idx_path, index=False)
        features_df.to_csv(feat_path, index=False)
        print(f"\n✓ SST index saved to    {idx_path}")
        print(f"✓ Features saved to     {feat_path}")

        if nino3_df is not None:
            nino3_path = os.path.join(output_dir, "nino3_reference.csv")
            nino3_df.to_csv(nino3_path, index=False)
            print(f"✓ Niño 3 ref saved to   {nino3_path}")

    return sst_df, features_df, nino3_df


if __name__ == "__main__":
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    grib_folder = os.path.join(project_root, "data", "raw", "era5")
    output_dir  = os.path.join(project_root, "data", "processed")

    print(f"Project root: {project_root}")

    sst_df, features_df, nino3_df = process_daily_sst(
        grib_folder,
        output_dir=output_dir,
        clim_start=CLIM_START,
        clim_end=CLIM_END,
    )

    print('\n✓ All processes completed')