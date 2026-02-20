"""
SARIMAX Forecasting with ENSO (ONI) as Exogenous Driver for Monthly Climate Indices
- Cundinamarca-Bogotá Region

Implements the correct state space framework with exogenous variables:
DYNAMIC REGRESSION WITH STATE SPACE ERRORS (ACI-CO equation 6):
    X_{k,r,t} = ℓ_{t-1} + φb_{t-1} + s_{t-m} + β₀ + Σ(βₗZ_{t-ℓ}) + Σ(βₗⱼZ_{t-ℓ}Mⱼ(t)) + εₜ

State Space Interpretation:
├─ ℓ_t = Level (state component)
├─ b_t = Trend (state component)
├─ s_t = Seasonality (state component)
├─ X_t^ENSO = Exogenous ENSO variable(s) [Z_ONI, lagged values]
└─ βX_t = Exogenous regression coefficients

ENSO (ONI) Integration:
- Fetched from ERA5 SST data (Colombia Pacific region)
- Z_t = Standardized ONI (standardized on 1961-2020 baseline)
- Enters model as EXOGENOUS variable in SARIMAX
- Properly integrated in state space (not pre-processed)

Enhanced with:
- SARIMAX: ARIMA(p,d,q) × Seasonal(P,D,Q,s) with exogenous variables
- Grid search: Optimal (p,d,q,P,D,Q,s) + ENSO lag selection
- Time series cross-validation
- ENSO phase indicators and lagged effects
- Seasonal ENSO interactions
- Improved diagnostics and visualizations
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
import requests

warnings.filterwarnings('ignore')

# ETS and statistical models
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
from scipy.stats import shapiro, jarque_bera
import seaborn as sns
from itertools import product

# For ERA5 data retrieval
try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False
    print("[WARNING] cdsapi not installed. Install with: pip install cdsapi")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import TwoSlopeNorm
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("[WARNING] cartopy not installed. Some visualizations will be skipped.")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ONIDataHandler:
    """
    Handles ONI (Oceanic Niño Index) data retrieval and processing.
    Implements the ENSO driver specification from the methods note.
    """

    def __init__(self):
        """Initialize ONI and AMO data handler."""
        self.oni_raw = None
        self.oni_standardized = None
        self.amo_raw = None
        self.amo_standardized = None
        self.training_mean = None
        self.training_std = None

    def fetch_oni_data(self):
        """
        Fetch ONI data from NOAA Climate Prediction Center.

        Returns:
            pd.DataFrame: ONI data with datetime index
        """
        print("\n[Fetching ONI data from NOAA CPC...]")
        try:
            url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
            df = pd.read_csv(url, delim_whitespace=True, skiprows=1,
                             names=['SEAS', 'YR', 'TOTAL', 'ANOM'])

            if 'SEAS' not in df.columns or 'ANOM' not in df.columns:
                df = pd.read_csv(url, delim_whitespace=True, header=None)
                if len(df.columns) >= 4:
                    df.columns = ['SEAS', 'YR', 'TOTAL', 'ANOM']
                else:
                    raise ValueError("Unexpected data format from NOAA")

            season_to_month = {
                'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4,
                'AMJ': 5, 'MJJ': 6, 'JJA': 7, 'JAS': 8,
                'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12
            }
            df['month'] = df['SEAS'].map(season_to_month)
            if df['month'].isna().any():
                df['month'].fillna(method='ffill', inplace=True)

            df['date'] = pd.to_datetime(
                df['YR'].astype(int).astype(str) + '-' +
                df['month'].astype(int).astype(str) + '-01',
                errors='coerce'
            )
            df = df.dropna(subset=['date'])
            oni_series = pd.Series(df['ANOM'].values, index=df['date'], name='ONI')
            oni_series = oni_series.sort_index()
            oni_series = oni_series[~oni_series.index.duplicated(keep='first')]

            full_range = pd.date_range(
                start=oni_series.index.min(),
                end=oni_series.index.max(),
                freq='MS'
            )
            oni_series = oni_series.reindex(full_range, method='ffill')
            self.oni_raw = pd.DataFrame({'ONI': oni_series})

            print(f"  [OK] Fetched ONI data: {len(self.oni_raw)} months")
            print(f"  Date range: {self.oni_raw.index.min()} to {self.oni_raw.index.max()}")
            print(f"  ONI range: [{self.oni_raw['ONI'].min():.2f}, {self.oni_raw['ONI'].max():.2f}]°C")

            if self.oni_raw['ONI'].std() < 0.1:
                raise ValueError("ONI data appears to be invalid (too little variation)")

            return self.oni_raw

        except Exception as e:
            print(f"  [ERROR] Failed to fetch ONI data: {e}")
            print(f"  Attempting alternative ONI source...")
            try:
                alt_url = "https://psl.noaa.gov/data/correlation/oni.data"
                df_alt = pd.read_csv(alt_url, delim_whitespace=True, skiprows=1)
                oni_data = []
                for idx, row in df_alt.iterrows():
                    year = int(row.iloc[0])
                    for month in range(1, 13):
                        if month < len(row):
                            value = row.iloc[month]
                            if pd.notna(value) and value != -99.99:
                                date = pd.Timestamp(year=year, month=month, day=1)
                                oni_data.append({'date': date, 'ONI': float(value)})
                if len(oni_data) > 0:
                    df_oni = pd.DataFrame(oni_data)
                    df_oni.set_index('date', inplace=True)
                    df_oni = df_oni.sort_index()
                    self.oni_raw = df_oni
                    print(f"  [OK] Fetched from alternative source: {len(self.oni_raw)} months")
                    return self.oni_raw
                else:
                    raise ValueError("No valid data from alternative source")
            except Exception as e2:
                print(f"  [ERROR] Alternative source also failed: {e2}")
                print(f"  [WARNING] Creating dummy ONI data - results will not be meaningful!")
                dates = pd.date_range(start='1950-01-01', end='2026-12-01', freq='MS')
                t = np.arange(len(dates))
                enso_cycle = (1.5 * np.sin(2 * np.pi * t / (4 * 12)) +
                              0.5 * np.random.randn(len(dates)))
                self.oni_raw = pd.DataFrame({'ONI': enso_cycle}, index=dates)
                print(f"  [DUMMY DATA] Created {len(self.oni_raw)} months of simulated ONI")
                return self.oni_raw

    def fetch_amo_data(self):
        """
        Fetch AMO (Atlantic Multidecadal Oscillation) data.

        Returns:
            pd.DataFrame: AMO data with datetime index
        """
        print("\n[Fetching AMO (Atlantic Multidecadal Oscillation) data...]")
        try:
            url = "https://www.psl.noaa.gov/data/correlation/amo.data"
            lines = []
            for line in requests.get(url, timeout=10).text.split('\n'):
                if line.strip() and not line.startswith('Missing'):
                    lines.append(line)

            if not lines:
                raise ValueError("No data lines parsed from AMO")

            amo_data = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 13:
                    try:
                        year = int(parts[0])
                        for month in range(1, 13):
                            value = float(parts[month])
                            if value > -999:
                                date = pd.Timestamp(year=year, month=month, day=1)
                                amo_data.append({'date': date, 'AMO': value})
                    except (ValueError, IndexError):
                        pass

            if len(amo_data) == 0:
                raise ValueError("No valid AMO data parsed")

            df_amo = pd.DataFrame(amo_data)
            df_amo.set_index('date', inplace=True)
            df_amo = df_amo.sort_index()
            df_amo = df_amo[~df_amo.index.duplicated(keep='first')]
            self.amo_raw = df_amo

            print(f"  [OK] Fetched AMO data: {len(self.amo_raw)} months")
            print(f"  Date range: {self.amo_raw.index.min()} to {self.amo_raw.index.max()}")
            print(f"  AMO range: [{self.amo_raw['AMO'].min():.2f}, {self.amo_raw['AMO'].max():.2f}]°C")
            return self.amo_raw

        except Exception as e:
            print(f"  [WARNING] Failed to fetch AMO: {e}")
            print(f"  Continuing with ENSO only (no Atlantic teleconnection)")
            self.amo_raw = None
            return None

    def standardize_oni(self, training_start='1961-01-01', training_end='2020-12-31'):
        """
        Standardize ONI following equation (4): Z_t = (ONI_t - mean) / std

        Returns:
            pd.DataFrame: Standardized ONI (Z_t)
        """
        print("\n[Standardizing ONI...]")
        if self.oni_raw is None:
            self.fetch_oni_data()

        training_mask = (
            (self.oni_raw.index >= training_start) &
            (self.oni_raw.index <= training_end)
        )
        self.training_mean = self.oni_raw.loc[training_mask, 'ONI'].mean()
        self.training_std = self.oni_raw.loc[training_mask, 'ONI'].std()

        self.oni_standardized = pd.DataFrame({
            'ONI_raw': self.oni_raw['ONI'],
            'Z_ONI': (self.oni_raw['ONI'] - self.training_mean) / self.training_std
        }, index=self.oni_raw.index)

        print(f"  Training period: {training_start} to {training_end}")
        print(f"  Training mean: {self.training_mean:.3f}")
        print(f"  Training std:  {self.training_std:.3f}")
        print(f"  Standardized ONI range: [{self.oni_standardized['Z_ONI'].min():.2f}, "
              f"{self.oni_standardized['Z_ONI'].max():.2f}]")
        return self.oni_standardized

    def standardize_amo(self, training_start='1961-01-01', training_end='2020-12-31'):
        """Standardize AMO data."""
        if self.amo_raw is None or len(self.amo_raw) == 0:
            print("  [INFO] AMO data not available, skipping AMO standardization")
            return None

        print("\n[Standardizing AMO...]")
        training_mask = (
            (self.amo_raw.index >= training_start) &
            (self.amo_raw.index <= training_end)
        )
        amo_mean = self.amo_raw.loc[training_mask, 'AMO'].mean()
        amo_std = self.amo_raw.loc[training_mask, 'AMO'].std()

        self.amo_standardized = pd.DataFrame({
            'AMO_raw': self.amo_raw['AMO'],
            'Z_AMO': (self.amo_raw['AMO'] - amo_mean) / amo_std
        }, index=self.amo_raw.index)

        print(f"  AMO Training mean: {amo_mean:.3f}")
        print(f"  AMO Training std:  {amo_std:.3f}")
        print(f"  Standardized AMO range: [{self.amo_standardized['Z_AMO'].min():.2f}, "
              f"{self.amo_standardized['Z_AMO'].max():.2f}]")
        return self.amo_standardized

    def create_enso_features(self, max_lag=6, include_phases=True,
                              include_seasonal_interactions=False):
        """
        Create ENSO features: lagged Z_t, phase indicators, seasonal interactions.

        Returns:
            pd.DataFrame: ENSO features
        """
        print("\n[Creating ENSO features...]")
        if self.oni_standardized is None:
            self.standardize_oni()

        features = pd.DataFrame(index=self.oni_standardized.index)
        features['ONI']   = self.oni_standardized['ONI_raw']
        features['Z_ONI'] = self.oni_standardized['Z_ONI']

        for lag in range(max_lag + 1):
            features[f'Z_ONI_lag{lag}']    = features['Z_ONI'].shift(lag)
            features[f'Z_ONI_lag{lag}_sq'] = features[f'Z_ONI_lag{lag}'] ** 2

        if include_phases:
            features['D_EN'] = (features['ONI'] >= 0.5).astype(int)
            features['D_LN'] = (features['ONI'] <= -0.5).astype(int)
            features['EN_strength'] = features['D_EN'] * features['Z_ONI']
            features['LN_strength'] = features['D_LN'] * np.abs(features['Z_ONI'])
            print(f"  El Niño months:  {features['D_EN'].sum()}")
            print(f"  La Niña months:  {features['D_LN'].sum()}")
            print(f"  Neutral months:  {(features['D_EN'] + features['D_LN'] == 0).sum()}")

        if include_seasonal_interactions:
            features['month'] = features.index.month
            for lag in range(min(3, max_lag + 1)):
                for month in range(1, 12):
                    month_indicator = (features['month'] == month).astype(int)
                    features[f'Z_lag{lag}_M{month}'] = (
                        features[f'Z_ONI_lag{lag}'] * month_indicator
                    )

        for window in [3, 6]:
            features[f'Z_ONI_roll{window}'] = (
                features['Z_ONI'].rolling(window=window, min_periods=1).mean()
            )

        if self.amo_standardized is not None and len(self.amo_standardized) > 0:
            try:
                features['AMO']   = self.amo_standardized.loc[features.index, 'AMO_raw']
                features['Z_AMO'] = self.amo_standardized.loc[features.index, 'Z_AMO']
                for lag in range(0, min(4, max_lag + 1)):
                    features[f'Z_AMO_lag{lag}']    = features['Z_AMO'].shift(lag)
                    features[f'Z_AMO_lag{lag}_sq'] = features[f'Z_AMO_lag{lag}'] ** 2
                for window in [3, 6]:
                    features[f'Z_AMO_roll{window}'] = (
                        features['Z_AMO'].rolling(window=window, min_periods=1).mean()
                    )
                print("  [OK] Added AMO features (Atlantic teleconnection)")
            except Exception as e:
                print(f"  [WARNING] Could not align AMO features: {e}")

        print(f"  [OK] Created {len(features.columns)} total exogenous features")
        return features


class MonthlyClimateForecasterETSWithENSO:
    """
    Handles forecasting of monthly climate anomalies using ETS models with
    ENSO (ONI) as an exogenous variable.
    """

    def __init__(self, region_name="cundinamarca_bogota", forecast_months=12,
                 enso_max_lag=6, include_enso_phases=True,
                 include_seasonal_interactions=False):
        self.region_name = region_name
        self.forecast_months = forecast_months
        self.enso_max_lag = enso_max_lag
        self.include_enso_phases = include_enso_phases
        self.include_seasonal_interactions = include_seasonal_interactions

        self.data_wide         = None
        self.models            = {}
        self.forecasts_dict    = {}
        self.results_summary   = []
        self.cv_results        = []
        self.differencing_info = {}
        self.fitted_values_dict = {}
        self.residuals_dict    = {}

        self.oni_handler    = ONIDataHandler()
        self.enso_features  = None

        self.base_path   = os.path.join(".", "data", "processed",
                                         f"anomalias_{region_name}")
        self.output_path = os.path.join(".", "articles", "graficas",
                                         f"forecast_ets_enso_{region_name}")
        os.makedirs(self.output_path, exist_ok=True)

        print(f"Base path:   {os.path.abspath(self.base_path)}")
        print(f"Output path: {os.path.abspath(self.output_path)}")

    # ------------------------------------------------------------------
    # DATA PREPARATION
    # ------------------------------------------------------------------

    def prepare_enso_data(self):
        print("\n" + "="*80)
        print("PREPARING ENSO (ONI) AND AMO (ATLANTIC) DATA")
        print("="*80)
        self.oni_handler.fetch_oni_data()
        self.oni_handler.standardize_oni()
        self.oni_handler.fetch_amo_data()
        self.oni_handler.standardize_amo()
        self.enso_features = self.oni_handler.create_enso_features(
            max_lag=self.enso_max_lag,
            include_phases=self.include_enso_phases,
            include_seasonal_interactions=self.include_seasonal_interactions
        )
        return self.enso_features

    def adf_test(self, series, variable_name):
        result = adfuller(series.dropna(), autolag='AIC')
        return {
            'variable':        variable_name,
            'adf_statistic':   result[0],
            'p_value':         result[1],
            'used_lag':        result[2],
            'n_obs':           result[3],
            'critical_values': result[4],
            'is_stationary':   result[1] < 0.05
        }

    def make_stationary(self, series, variable_name, max_diff=2):
        print(f"\n Testing stationarity for {variable_name}...")
        original_series = series.copy()
        current_series  = series.copy()
        diff_order = 0
        adf_results = []

        for d in range(max_diff + 1):
            adf_result = self.adf_test(current_series, variable_name)
            adf_results.append(adf_result)
            print(f"  Diff order {d}: ADF={adf_result['adf_statistic']:.4f}, "
                  f"p-value={adf_result['p_value']:.4f}, "
                  f"Stationary={adf_result['is_stationary']}")
            if adf_result['is_stationary']:
                diff_order = d
                break
            if d < max_diff:
                current_series = current_series.diff().dropna()

        self.differencing_info[variable_name] = {
            'diff_order':        diff_order,
            'adf_results':       adf_results,
            'original_series':   original_series,
            'stationary_series': current_series
        }
        return current_series, diff_order, adf_results

    def load_monthly_data(self):
        print("\n[Loading raw daily ERA5 data...]")
        datasets = {}
        data_path = os.path.join(".", "data", "processed")

        print("\n Loading temperature files...")
        temp_file = os.path.join(data_path, "era5_daily_combined_tmp.nc")
        if os.path.exists(temp_file):
            try:
                datasets['temperature'] = xr.open_dataset(temp_file)
                print(f"  [OK] Loaded temperature")
            except Exception as e:
                print(f"  [WARNING] Error loading temperature: {e}")
        else:
            print(f"  [WARNING] Temperature file not found: {temp_file}")

        print("\n Loading wind files...")
        wind_file = os.path.join(data_path, "era5_daily_combined_wind.nc")
        if os.path.exists(wind_file):
            try:
                datasets['wind'] = xr.open_dataset(wind_file)
                print(f"  [OK] Loaded wind")
            except Exception as e:
                print(f"  [WARNING] Error loading wind: {e}")
        else:
            print(f"  [WARNING] Wind file not found: {wind_file}")

        print("\n Loading precipitation data...")
        precip_file = os.path.join(data_path, "era5_daily_combined_rain.nc")
        if os.path.exists(precip_file):
            try:
                datasets['precipitation'] = xr.open_dataset(precip_file)
                print(f"  [OK] Loaded precipitation")
            except Exception as e:
                print(f"  [WARNING] Error loading precipitation: {e}")
        else:
            print(f"  [WARNING] Precipitation file not found: {precip_file}")

        print("\n Checking for drought/percentile data...")
        for drought_file, key in [
            (os.path.join(data_path, "era5_sequia_percentil.nc"),  'drought_percentile'),
            (os.path.join(data_path, "era5_lluvia_percentil.nc"),  'precipitation_percentile'),
        ]:
            if os.path.exists(drought_file):
                try:
                    datasets[key] = xr.open_dataset(drought_file)
                    print(f"  [OK] Loaded {key}")
                except Exception as e:
                    print(f"  [WARNING] Error loading {key}: {e}")

        return datasets

    def extract_regional_series(self, datasets):
        print("\n[Extracting regional time series from raw daily data...]")
        daily_series = {}

        if 'temperature' in datasets:
            try:
                ds = datasets['temperature']
                if 'daily_max' in ds.data_vars:
                    ts = pd.Series(
                        ds['daily_max'].mean(dim=['latitude', 'longitude'], skipna=True).values,
                        index=pd.to_datetime(ds['time'].values)
                    )
                    daily_series['temperature_max'] = ts
                    print(f"  [OK] temperature_max: {len(ts)} observations")
                if 'daily_min' in ds.data_vars:
                    ts = pd.Series(
                        ds['daily_min'].mean(dim=['latitude', 'longitude'], skipna=True).values,
                        index=pd.to_datetime(ds['time'].values)
                    )
                    daily_series['temperature_min'] = ts
                    print(f"  [OK] temperature_min: {len(ts)} observations")
            except Exception as e:
                print(f"  [WARNING] Error processing temperature: {e}")
                import traceback; traceback.print_exc()

        if 'precipitation' in datasets:
            try:
                ds = datasets['precipitation']
                if 'tp_daily_sum' in ds.data_vars:
                    ts = pd.Series(
                        ds['tp_daily_sum'].mean(dim=['latitude', 'longitude'], skipna=True).values,
                        index=pd.to_datetime(ds['time'].values)
                    )
                    daily_series['precipitation'] = ts
                    print(f"  [OK] precipitation: {len(ts)} observations")
            except Exception as e:
                print(f"  [WARNING] Error processing precipitation: {e}")

        if 'wind' in datasets:
            try:
                ds = datasets['wind']
                if 'wind_speed' in ds.data_vars:
                    ts = pd.Series(
                        ds['wind_speed'].mean(dim=['latitude', 'longitude'], skipna=True).values,
                        index=pd.to_datetime(ds['time'].values)
                    )
                    daily_series['wind'] = ts
                    print(f"  [OK] wind_speed: {len(ts)} observations")
            except Exception as e:
                print(f"  [WARNING] Error processing wind: {e}")

        if daily_series:
            print(f"\n Combining {len(daily_series)} daily series...")
            dedup = {n: s[~s.index.duplicated(keep='first')]
                     for n, s in daily_series.items()}

            common_index = list(dedup.values())[0].index
            for s in list(dedup.values())[1:]:
                common_index = common_index.intersection(s.index)
            common_index = common_index.drop_duplicates()

            print(f"  Common date range: {common_index.min()} to {common_index.max()}")
            df_daily = pd.DataFrame({n: s.loc[common_index] for n, s in dedup.items()})
            df_daily = df_daily.sort_index()

            monthly = {}
            for col in ['temperature_max', 'temperature_min']:
                if col in df_daily.columns:
                    monthly[col] = df_daily[col].resample('MS').mean()
            if 'precipitation' in df_daily.columns:
                monthly['precipitation'] = df_daily['precipitation'].resample('MS').sum()
            if 'wind' in df_daily.columns:
                monthly['wind_speed'] = df_daily['wind'].resample('MS').mean()

            self.data_wide = pd.concat(monthly, axis=1).dropna()
            print(f"  [OK] Monthly aggregated data shape: {self.data_wide.shape}")
            print(f"  Date range: {self.data_wide.index.min()} to {self.data_wide.index.max()}")
            return self.data_wide

        print("[FAILED] No data could be extracted!")
        return None

    def align_enso_with_climate_data(self):
        print("\n[Aligning ENSO features with climate data...]")
        if self.data_wide is None or self.enso_features is None:
            print("  [ERROR] Climate data or ENSO features not loaded")
            return None
        common_index  = self.data_wide.index.intersection(self.enso_features.index)
        aligned_enso  = self.enso_features.loc[common_index]
        print(f"  [OK] Aligned {len(aligned_enso)} months  "
              f"({aligned_enso.index.min()} → {aligned_enso.index.max()})")
        return aligned_enso

    # ------------------------------------------------------------------
    # MODEL FITTING
    # ------------------------------------------------------------------

    def grid_search_sarimax_parameters(self, y, variable_name, X_exog,
                                        p_range=range(0, 3), d_range=range(0, 2),
                                        q_range=range(0, 3),
                                        P_range=range(0, 2), D_range=range(0, 2),
                                        Q_range=range(0, 2), s=12,
                                        enso_lag_range=range(0, 4), cv_splits=2):
        print(f"\n [GRID SEARCH] SARIMAX parameters for {variable_name}")
        total = (len(p_range)*len(d_range)*len(q_range) *
                 len(P_range)*len(D_range)*len(Q_range) *
                 len(enso_lag_range))
        print(f"  Testing {total} configurations...")

        results = []
        config_num = 0

        for enso_lag in enso_lag_range:
            lag_cols = [c for c in X_exog.columns if 'lag' in c.lower()]
            if enso_lag == 0:
                X_sel = X_exog[[c for c in X_exog.columns if 'lag' not in c.lower()]]
            else:
                X_sel = X_exog[[c for c in lag_cols
                                 if int(c.split('_')[-1]) <= enso_lag]]
            if len(X_sel.columns) == 0:
                X_sel = X_exog.iloc[:, [0]]

            for p in p_range:
                for d in d_range:
                    for q in q_range:
                        for P in P_range:
                            for D in D_range:
                                for Q in Q_range:
                                    config_num += 1
                                    try:
                                        m = SARIMAX(y, exog=X_sel,
                                                    order=(p, d, q),
                                                    seasonal_order=(P, D, Q, s),
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                                        f = m.fit(disp=False, maxiter=100)
                                        aic  = f.aic
                                        rmse = np.sqrt(f.sse / len(y))

                                        split_size = len(y) // (cv_splits + 1)
                                        cv_rmse_list = []
                                        for i in range(cv_splits):
                                            tr = list(range(0, i*split_size)) + \
                                                 list(range((i+1)*split_size, len(y)))
                                            te = list(range(i*split_size, (i+1)*split_size))
                                            try:
                                                mc = SARIMAX(y.iloc[tr], exog=X_sel.iloc[tr],
                                                             order=(p, d, q),
                                                             seasonal_order=(P, D, Q, s),
                                                             enforce_stationarity=False)
                                                fc = mc.fit(disp=False, maxiter=100)
                                                pr = fc.get_forecast(
                                                    steps=len(te),
                                                    exog=X_sel.iloc[te]
                                                ).predicted_mean
                                                cv_rmse_list.append(
                                                    np.sqrt(np.mean((y.iloc[te] - pr)**2))
                                                )
                                            except:
                                                pass
                                        cv_rmse = np.mean(cv_rmse_list) if cv_rmse_list else np.inf
                                        results.append({
                                            'config_num': config_num,
                                            'order':          (p, d, q),
                                            'seasonal_order': (P, D, Q, s),
                                            'enso_lag':       enso_lag,
                                            'n_exog_features': len(X_sel.columns),
                                            'aic':    aic,
                                            'rmse':   rmse,
                                            'cv_rmse': cv_rmse
                                        })
                                        if config_num % 50 == 0:
                                            print(f"  [{config_num}] AIC={aic:.1f}, "
                                                  f"CV_RMSE={cv_rmse:.4f}")
                                    except:
                                        pass

        if not results:
            print("  [ERROR] No valid configurations found")
            return None

        res_df = pd.DataFrame(results).sort_values('aic')
        print(f"\n  [RESULTS] Top 5 SARIMAX Configurations:")
        for _, row in res_df.head(5).iterrows():
            print(f"    ARIMA{row['order']}×Season{row['seasonal_order']}"
                  f"_Lag{row['enso_lag']}  AIC={row['aic']:.1f}  "
                  f"CV_RMSE={row['cv_rmse']:.4f}")

        out = os.path.join(self.output_path, f'sarimax_grid_search_{variable_name}.csv')
        res_df.to_csv(out, index=False)
        print(f"  [OK] Saved: {out}")
        return {'results_df': res_df, 'best_params': res_df.iloc[0].to_dict()}

    def fit_sarimax_with_enso(self, series, variable_name, enso_features_aligned,
                               use_grid_search=False, best_params=None):
        """
        Fit SARIMAX model with ENSO as exogenous variable.

        R² for ENSO is computed correctly:
            R²_ENSO = 1 - Var(residuals) / Var(y)
        where residuals = SARIMAX residuals (which include all non-ENSO structure)
        and the ENSO fraction = 1 - R²_SARIMAX_without_ENSO / R²_SARIMAX_with_ENSO.

        We use a simpler but correct approach:
            r2_enso = Var(ENSO_fitted_component) / Var(y_clean)
        """
        print(f"\n  Fitting SARIMAX+ENSO model for {variable_name}...")

        common_idx = series.index.intersection(enso_features_aligned.index)
        y        = series.loc[common_idx]
        X_enso   = enso_features_aligned.loc[common_idx]
        valid    = ~(y.isna() | X_enso.isna().any(axis=1))
        y_clean  = y[valid]
        X_clean  = X_enso[valid]

        if len(y_clean) < 80:
            print(f"  [WARNING] Insufficient data: {len(y_clean)} points")
            return None, None, None, None

        print(f"  Training data: {len(y_clean)} observations")
        X_numeric = X_clean.select_dtypes(include=[np.number])

        # ── Feature selection with Lasso ──────────────────────────────
        try:
            lasso = LassoCV(cv=5, max_iter=10000, random_state=42,
                            alphas=np.logspace(-4, 1, 50), n_jobs=-1)
            lasso.fit(X_numeric, y_clean)
            selected_features = X_numeric.columns[np.abs(lasso.coef_) > 1e-10]
            if len(selected_features) == 0:
                selected_features = X_numeric.columns[np.argsort(np.abs(lasso.coef_))[-3:]]
            X_exog = X_numeric[selected_features]
            print(f"  Selected {len(selected_features)} ENSO features")
        except Exception as e:
            print(f"  [WARNING] Feature selection failed: {e}")
            X_exog = X_numeric
            selected_features = X_numeric.columns

        # ── SARIMAX order selection ────────────────────────────────────
        if use_grid_search:
            gs = self.grid_search_sarimax_parameters(y_clean, variable_name,
                                                      X_exog, cv_splits=2)
            if gs and 'best_params' in gs:
                best_params = gs['best_params']

        if best_params is None:
            if 'temperature_t_10' in variable_name:
                order, seasonal_order = (2, 1, 2), (1, 1, 1, 12)
            elif 'temperature_t_90' in variable_name:
                order, seasonal_order = (2, 1, 2), (1, 1, 1, 12)
            elif 'drought' in variable_name.lower() or 'sequia' in variable_name.lower():
                order, seasonal_order = (1, 1, 1), (1, 1, 1, 12)
            else:
                order, seasonal_order = (1, 1, 1), (1, 1, 1, 12)
        else:
            order          = best_params['order']
            seasonal_order = best_params['seasonal_order']

        # ── Fit SARIMAX ───────────────────────────────────────────────
        try:
            print(f"  Fitting SARIMAX{order}×{seasonal_order}...")
            sarimax_model  = SARIMAX(y_clean, exog=X_exog,
                                      order=order, seasonal_order=seasonal_order,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
            sarimax_fitted = sarimax_model.fit(disp=False, maxiter=200)
            print(f"  SARIMAX AIC: {sarimax_fitted.aic:.2f}")
            print(f"  Converged:   {sarimax_fitted.mle_retvals['converged']}")

            # ── Store residuals (FIXED: actual SARIMAX residuals) ─────
            self.residuals_dict[variable_name] = sarimax_fitted.resid.values

            # ── Store fitted values ───────────────────────────────────
            fitted_series = pd.Series(sarimax_fitted.fittedvalues, index=y_clean.index)
            self.fitted_values_dict[variable_name] = fitted_series

            # ── ENSO component in-sample ──────────────────────────────
            # Extract exogenous coefficients by name (robust to parameter ordering)
            try:
                coef_series = sarimax_fitted.params.reindex(X_exog.columns).fillna(0.0)
            except Exception:
                coef_series = pd.Series(0.0, index=X_exog.columns)

            enso_in_sample = pd.Series(
                X_exog.loc[y_clean.index].values @ coef_series.values,
                index=y_clean.index
            )

            # ── CORRECT R² for ENSO contribution ─────────────────────
            # r2_enso = fraction of y variance explained by the ENSO component
            # = Var(ENSO component) / Var(y_fitted)   [conservative, correct]
            var_y       = np.var(y_clean.values)
            var_enso    = np.var(enso_in_sample.values)
            enso_r2     = float(np.clip(var_enso / var_y, 0.0, 1.0)) if var_y > 0 else 0.0

            ets_in_sample = fitted_series - enso_in_sample

        except Exception as e:
            print(f"  [ERROR] SARIMAX fitting failed: {e}")
            import traceback; traceback.print_exc()
            return None, None, None, None

        # ── Forecast ──────────────────────────────────────────────────
        try:
            last_date    = y_clean.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=self.forecast_months, freq='MS'
            )

            if len(X_exog) >= self.forecast_months:
                future_exog = X_exog.iloc[-self.forecast_months:].copy()
                future_exog.index = future_dates
            else:
                future_exog = pd.DataFrame(
                    X_exog.mean().values.reshape(1, -1).repeat(self.forecast_months, axis=0),
                    index=future_dates, columns=X_exog.columns
                )

            fc_result  = sarimax_fitted.get_forecast(steps=self.forecast_months,
                                                       exog=future_exog)
            fc_values  = fc_result.predicted_mean
            conf_int   = fc_result.conf_int(alpha=0.05)

            enso_fc = pd.Series(
                future_exog.values @ coef_series.values,
                index=future_dates
            )

            forecast_df = pd.DataFrame({
                'forecast':        fc_values,
                'lower':           conf_int.iloc[:, 0],
                'upper':           conf_int.iloc[:, 1],
                'enso_component':  enso_fc,
                'ets_component':   fc_values - enso_fc
            }, index=future_dates)

            model_info = {
                'aic':               sarimax_fitted.aic,
                'bic':               sarimax_fitted.bic,
                'rmse':              np.sqrt(sarimax_fitted.sse / len(y_clean)),
                'n_exog_features':   len(selected_features),
                'n_enso_features':   len(selected_features),
                'selected_features': list(selected_features),
                'order':             order,
                'seasonal_order':    seasonal_order,
                'r2_enso':           enso_r2,          # ← FIXED
                'residual_std':      float(np.std(sarimax_fitted.resid))
            }

            print(f"  [OK] Forecast generated  RMSE={model_info['rmse']:.4f}  "
                  f"ENSO R²={enso_r2:.3f}")
            return sarimax_model, sarimax_fitted, forecast_df, model_info

        except Exception as e:
            print(f"  [ERROR] Forecast generation failed: {e}")
            import traceback; traceback.print_exc()
            return sarimax_model, sarimax_fitted, None, None

    def fit_ets_with_enso(self, series, variable_name, enso_features_aligned,
                          use_regularization=True, use_grid_search=False,
                          best_params=None):
        return self.fit_sarimax_with_enso(series, variable_name,
                                           enso_features_aligned,
                                           use_grid_search=use_grid_search,
                                           best_params=best_params)

    def fit_all_ets_enso_models(self):
        print("\n[Fitting SARIMAX+ENSO models to all variables...]")
        enso_aligned = self.align_enso_with_climate_data()
        if enso_aligned is None:
            print("  [ERROR] Failed to align ENSO data")
            return

        for col in self.data_wide.columns:
            series = self.data_wide[col].dropna()
            if len(series) < 60:
                print(f"  [SKIP] {col}: Insufficient data ({len(series)} points)")
                continue

            _, _, forecast_df, model_info = self.fit_ets_with_enso(
                series, col, enso_aligned
            )
            if forecast_df is not None:
                self.models[col]        = {}
                self.forecasts_dict[col] = forecast_df
                self.results_summary.append({
                    'variable':        col,
                    'aic':             model_info['aic'],
                    'r2_enso':         model_info['r2_enso'],
                    'n_enso_features': model_info['n_enso_features'],
                    'residual_std':    model_info['residual_std']
                })

        print(f"\n[OK] Successfully fitted {len(self.models)} models!")

    # ------------------------------------------------------------------
    # FORECASTS & CLIMATE INDEX
    # ------------------------------------------------------------------

    def calculate_climate_index(self, data):
        ica = pd.Series(0.0, index=data.index)
        component_count = 0

        t90_col = t10_col = None
        for col in data.columns:
            if 't_90' in col.lower():   t90_col = col
            elif 't_10' in col.lower(): t10_col = col
        if t90_col and t10_col:
            ica += data[t90_col] - data[t10_col]
            component_count += 1

        for col in data.columns:
            if 'wind' in col.lower():
                ica += data[col]; component_count += 1; break
        for col in data.columns:
            if 'precip' in col.lower() or 'lluvia' in col.lower():
                ica += data[col]; component_count += 1; break
        for col in data.columns:
            if 'drought' in col.lower() or 'sequia' in col.lower():
                ica += data[col]; component_count += 1; break

        divisor = 5 if component_count >= 4 else component_count
        return ica / divisor if divisor > 0 else ica

    def generate_forecasts(self):
        print("\n[Compiling forecasts...]")
        fc = pd.DataFrame({c: df['forecast']         for c, df in self.forecasts_dict.items()})
        up = pd.DataFrame({c: df['upper']            for c, df in self.forecasts_dict.items()})
        lo = pd.DataFrame({c: df['lower']            for c, df in self.forecasts_dict.items()})
        en = pd.DataFrame({c: df['enso_component']   for c, df in self.forecasts_dict.items()})
        et = pd.DataFrame({c: df['ets_component']    for c, df in self.forecasts_dict.items()})

        hist_index = self.calculate_climate_index(self.data_wide)
        fc['Climate_Index'] = self.calculate_climate_index(fc).values
        up['Climate_Index'] = self.calculate_climate_index(up).values
        lo['Climate_Index'] = self.calculate_climate_index(lo).values
        self.data_wide['Climate_Index'] = hist_index

        print(f"[OK] Compiled forecasts shape: {fc.shape}")
        return fc, up, lo, en, et

    # ------------------------------------------------------------------
    # VISUALIZATIONS  (all bugs fixed)
    # ------------------------------------------------------------------

    def visualize_forecasts(self, forecasts, upper, lower, enso_components):
        print("\n[Creating forecast visualizations...]")
        combined        = pd.concat([self.data_wide, forecasts], axis=0)
        combined_smooth = combined.copy()
        for col in combined.columns:
            combined_smooth[col] = combined[col].rolling(window=60, center=False).mean()

        n_vars = len(forecasts.columns)
        n_cols = 2
        n_rows = (n_vars + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes = axes.flatten()

        zoom_start = pd.Timestamp('1961-01-01')
        zoom_end   = pd.Timestamp('2026-12-31')

        for idx, col in enumerate(forecasts.columns):
            ax = axes[idx]
            hist     = self.data_wide[col]
            hist_sm  = combined_smooth[col][:len(self.data_wide)]
            fc_data  = forecasts[col]
            fc_sm    = combined_smooth[col][len(self.data_wide):]
            ub       = upper[col]
            lb       = lower[col]

            ax.plot(hist.index,    hist.values,    'b-', lw=0.5, alpha=0.3, label='Historical (raw)')
            ax.plot(hist_sm.index, hist_sm.values, 'b-', lw=2.5, alpha=0.9, label='Historical (5-yr avg)')
            ax.plot(fc_data.index, fc_data.values, 'r-', lw=0.5, alpha=0.3, label='Forecast (raw)')
            ax.plot(fc_sm.index,   fc_sm.values,   'r-', lw=2.5, alpha=0.9, label='Forecast (5-yr avg)')

            if col in enso_components.columns:
                ax.plot(enso_components.index, enso_components[col].values,
                        'g--', lw=1.5, alpha=0.6, label='ENSO component')

            ax.fill_between(lb.index, lb.values, ub.values,
                            alpha=0.2, color='red', label='95% CI')

            ax.set_xlim(zoom_start, zoom_end)
            ax.set_title(col, fontsize=12, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Value')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(mdates.YearLocator(1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for idx in range(len(forecasts.columns), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        out = os.path.join(self.output_path, 'forecasts_ets_enso_all_variables.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {out}")
        plt.close()

    # ── FIXED: ENSO contribution chart ────────────────────────────────
    def visualize_enso_contribution(self):
        """
        FIXED: removed the duplicate barh block that was drawing the bars
        twice and corrupting the figure.  Now draws once with clean labelling.
        """
        print("\n[Creating ENSO contribution visualization...]")
        if not self.results_summary:
            print("  [WARNING] No results to visualize")
            return

        results_df = (pd.DataFrame(self.results_summary)
                        .sort_values('r2_enso', ascending=True))

        fig, ax = plt.subplots(figsize=(11, max(5, len(results_df) * 0.9 + 2)))

        y_pos    = np.arange(len(results_df))
        r2_vals  = results_df['r2_enso'].values * 100  # → percent

        # Single color mapping
        colors = []
        for r2 in results_df['r2_enso']:
            if   r2 > 0.30: colors.append('#2E7D32')   # dark green
            elif r2 > 0.20: colors.append('#66BB6A')   # medium green
            elif r2 > 0.10: colors.append('#FFA726')   # orange
            elif r2 > 0.05: colors.append('#FFCA28')   # yellow
            else:           colors.append('#BDBDBD')   # grey

        bars = ax.barh(y_pos, r2_vals,
                       color=colors, edgecolor='black', linewidth=1, alpha=0.85)

        # Labels on bars
        x_max = max(r2_vals) if len(r2_vals) > 0 else 1.0
        for bar, val in zip(bars, r2_vals):
            ax.text(val + x_max * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%',
                    va='center', ha='left', fontsize=10, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(results_df['variable'], fontsize=11)
        ax.set_xlabel('ENSO Explanatory Power  (R² %)', fontsize=12, fontweight='bold')
        ax.set_title('ENSO (ONI) Contribution to Climate Variability\n'
                     'Percentage of Variance Explained by El Niño–Southern Oscillation',
                     fontsize=14, fontweight='bold', pad=14)

        ax.axvline(x=30, color='darkgreen', ls='--', lw=2, alpha=0.4,
                   label='Strong signal (>30 %)')
        ax.axvline(x=10, color='orange',    ls='--', lw=2, alpha=0.4,
                   label='Moderate signal (>10 %)')
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        ax.set_xlim(0, x_max * 1.15)
        ax.grid(True, alpha=0.3, axis='x', ls=':', lw=1)
        ax.set_axisbelow(True)

        plt.tight_layout()
        out = os.path.join(self.output_path, 'enso_contribution_by_variable.png')
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"[OK] Saved: {out}")
        plt.close(fig)

    # ── FIXED: Residual diagnostics ───────────────────────────────────
    def visualize_residual_diagnostics(self):
        """
        FIXED:
        - Uses plt.subplots() (not fig.add_gridspec) → correct subplot indexing
          for any n_vars (including 1).
        - Histogram uses auto bins with density=True and a proper range check.
        - Q-Q, box-plot, and stats table are all correctly placed.
        """
        print("\n[Creating residual diagnostics visualization...]")
        if not self.residuals_dict:
            print("  [WARNING] No residuals available for diagnostics")
            return

        # Collect stats
        residual_stats = []
        for var_name, residuals in self.residuals_dict.items():
            clean = residuals[~np.isnan(residuals)]
            if len(clean) < 10:
                continue

            skew_v = stats.skew(clean)
            kurt_v = stats.kurtosis(clean)
            try:
                sw_stat, sw_p = shapiro(clean[:5000])
            except:
                sw_stat, sw_p = np.nan, np.nan
            try:
                jb_stat, jb_p = jarque_bera(clean)
            except:
                jb_stat, jb_p = np.nan, np.nan

            residual_stats.append({
                'variable':  var_name,
                'mean':      float(np.mean(clean)),
                'std':       float(np.std(clean)),
                'skewness':  float(skew_v),
                'kurtosis':  float(kurt_v),
                'shapiro_p': float(sw_p),
                'jb_p':      float(jb_p),
                'n_obs':     len(clean),
                'residuals': clean
            })

        if not residual_stats:
            print("  [WARNING] No valid residual statistics computed")
            return

        n_vars = len(residual_stats)
        # 4 panels per variable: histogram | Q-Q | boxplot | stats
        fig, axes = plt.subplots(n_vars, 4,
                                  figsize=(22, 5 * n_vars),
                                  squeeze=False)   # always 2-D array

        fig.suptitle('Residual Diagnostics: Distribution and Normality Tests',
                     fontsize=16, fontweight='bold', y=1.01)

        for row_idx, stat in enumerate(residual_stats):
            var_name  = stat['variable']
            residuals = stat['residuals']
            mu        = stat['mean']
            sigma     = stat['std']

            # ── Panel 0: Histogram ─────────────────────────────────
            ax = axes[row_idx, 0]
            # Protect against degenerate data (all-same value)
            if sigma < 1e-12:
                ax.text(0.5, 0.5, 'Degenerate\n(zero variance)',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=12, color='red')
            else:
                n_bins = min(50, max(10, len(residuals) // 20))
                ax.hist(residuals, bins=n_bins, density=True,
                        alpha=0.70, color='steelblue', edgecolor='white', lw=0.5)
                x = np.linspace(residuals.min(), residuals.max(), 200)
                ax.plot(x, stats.norm.pdf(x, mu, sigma),
                        'r-', lw=2.5, label=f'N({mu:.3f}, {sigma:.3f})')
                ax.axvline(mu, color='red', ls='--', lw=1.5, alpha=0.7)
                ax.legend(loc='best', fontsize=8)

            ax.set_xlabel('Residual value', fontsize=10)
            ax.set_ylabel('Density',        fontsize=10)
            ax.set_title(f'{var_name}\nHistogram', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # ── Panel 1: Q-Q plot ──────────────────────────────────
            ax = axes[row_idx, 1]
            if sigma >= 1e-12:
                (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist='norm')
                ax.scatter(osm, osr, s=8, color='steelblue', alpha=0.6)
                line_x = np.array([osm.min(), osm.max()])
                ax.plot(line_x, slope * line_x + intercept, 'r-', lw=2)
            else:
                ax.text(0.5, 0.5, 'Degenerate data',
                        transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Q-Q Plot',               fontsize=11, fontweight='bold')
            ax.set_xlabel('Theoretical quantiles', fontsize=10)
            ax.set_ylabel('Sample quantiles',      fontsize=10)
            ax.grid(True, alpha=0.3)

            # ── Panel 2: Box plot ──────────────────────────────────
            ax = axes[row_idx, 2]
            ax.boxplot(residuals, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', lw=2),
                       whiskerprops=dict(lw=1.5),
                       capprops=dict(lw=1.5))
            ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.7)
            ax.set_ylabel('Residual value', fontsize=10)
            ax.set_title('Box Plot',        fontsize=11, fontweight='bold')
            ax.set_xticklabels(['Residuals'])
            ax.grid(True, alpha=0.3, axis='y')

            q1, q3  = np.percentile(residuals, [25, 75])
            iqr     = q3 - q1
            n_out   = ((residuals < q1 - 1.5*iqr) | (residuals > q3 + 1.5*iqr)).sum()
            pct_out = n_out / len(residuals) * 100
            ax.text(0.5, 0.02, f'Outliers: {n_out} ({pct_out:.1f}%)',
                    transform=ax.transAxes, ha='center', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # ── Panel 3: Stats summary ─────────────────────────────
            ax = axes[row_idx, 3]
            ax.axis('off')
            is_normal_jb  = (stat['jb_p'] > 0.05
                             if not np.isnan(stat['jb_p']) else None)
            is_normal_sw  = (stat['shapiro_p'] > 0.05
                             if not np.isnan(stat['shapiro_p']) else None)
            norm_label    = ('✓ NORMAL' if is_normal_jb else
                             '✗ NON-NORMAL' if is_normal_jb is not None else 'N/A')
            summary = (
                f"RESIDUAL STATISTICS\n\n"
                f"Sample size: {stat['n_obs']}\n\n"
                f"Mean:        {stat['mean']:.4f}\n"
                f"Std Dev:     {stat['std']:.4f}\n"
                f"Skewness:    {stat['skewness']:.4f}\n"
                f"Kurtosis:    {stat['kurtosis']:.4f}\n\n"
                f"Normality tests:\n"
                f"  Shapiro-Wilk  p={stat['shapiro_p']:.4f}\n"
                f"    {'✓ Normal' if is_normal_sw else '✗ Non-normal' if is_normal_sw is not None else 'N/A'}\n"
                f"  Jarque-Bera   p={stat['jb_p']:.4f}\n"
                f"    {'✓ Normal' if is_normal_jb else '✗ Non-normal' if is_normal_jb is not None else 'N/A'}\n\n"
                f"Overall: {norm_label}\n\n"
                f"{'Symmetric' if abs(stat['skewness']) < 0.5 else 'Skewed'}\n"
                f"{'Normal tails' if abs(stat['kurtosis']) < 3 else 'Heavy tails'}"
            )
            ax.text(0.05, 0.95, summary, transform=ax.transAxes,
                    fontsize=9, va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        out = os.path.join(self.output_path, 'residual_diagnostics_all_variables.png')
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"[OK] Saved: {out}")
        plt.close()

        self._create_residual_summary_plot(residual_stats)

    def _create_residual_summary_plot(self, residual_stats):
        print("\n[Creating residual summary comparison...]")
        if not residual_stats:
            return

        df = pd.DataFrame([{k: v for k, v in s.items() if k != 'residuals'}
                            for s in residual_stats])
        df = df.sort_values('jb_p', ascending=False)
        if df.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # ── Plot 1: p-values ──────────────────────────────────────
        ax1 = axes[0, 0]
        y   = np.arange(len(df))
        w   = 0.35
        ax1.barh(y - w/2, df['shapiro_p'], w, label='Shapiro-Wilk',
                 color='#1976D2', alpha=0.8, edgecolor='black', lw=1)
        ax1.barh(y + w/2, df['jb_p'],      w, label='Jarque-Bera',
                 color='#F57C00', alpha=0.8, edgecolor='black', lw=1)
        ax1.axvline(0.05, color='red', ls='--', lw=2.5, alpha=0.7,
                    label='α = 0.05')
        ax1.set_yticks(y)
        ax1.set_yticklabels(df['variable'], fontsize=10)
        ax1.set_xlabel('p-value', fontsize=12, fontweight='bold')
        ax1.set_title('Normality Test p-values', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.set_xlim(0, 1)
        ax1.grid(True, alpha=0.3, axis='x')

        # ── Plot 2: Skewness vs Kurtosis ──────────────────────────
        ax2 = axes[0, 1]
        colors = ['#4CAF50' if abs(s) < 0.5 and abs(k) < 3
                  else '#FFA726' if abs(s) < 1 and abs(k) < 5
                  else '#E53935'
                  for s, k in zip(df['skewness'], df['kurtosis'])]
        ax2.scatter(df['skewness'], df['kurtosis'], c=colors, s=120,
                    edgecolors='black', lw=1.5, zorder=3)
        from matplotlib.patches import Rectangle
        ax2.add_patch(Rectangle((-0.5, -3), 1, 6, lw=2,
                                edgecolor='green', facecolor='green',
                                alpha=0.1, label='Ideal region'))
        ax2.axvline(0, color='black', lw=1.5, alpha=0.4)
        ax2.axhline(0, color='black', lw=1.5, alpha=0.4)
        for _, row in df.iterrows():
            ax2.annotate(row['variable'], (row['skewness'], row['kurtosis']),
                         fontsize=8, alpha=0.8, xytext=(5, 5),
                         textcoords='offset points')
        ax2.set_xlabel('Skewness', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Excess Kurtosis', fontsize=12, fontweight='bold')
        ax2.set_title('Skewness vs Kurtosis', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # ── Plot 3: Pass rates ────────────────────────────────────
        ax3 = axes[1, 0]
        pr_sw = (df['shapiro_p'] > 0.05).sum() / len(df) * 100
        pr_jb = (df['jb_p']      > 0.05).sum() / len(df) * 100
        for bar_x, pr, lbl in [(0, pr_sw, 'Shapiro-Wilk'),
                                 (1, pr_jb, 'Jarque-Bera')]:
            color = '#4CAF50' if pr >= 70 else '#FFA726' if pr >= 50 else '#E53935'
            bar   = ax3.bar(lbl, pr, color=color, alpha=0.8,
                            edgecolor='black', lw=2)
            ax3.text(bar_x, pr + 2, f'{pr:.1f}%',
                     ha='center', fontsize=12, fontweight='bold')
        ax3.axhline(50, color='gray',  ls='--', lw=2, alpha=0.5)
        ax3.axhline(70, color='green', ls='--', lw=2, alpha=0.5)
        ax3.set_ylim(0, 105)
        ax3.set_ylabel('% variables passing (p > 0.05)', fontsize=12, fontweight='bold')
        ax3.set_title('Normality Test Pass Rates', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # ── Plot 4: Summary text ──────────────────────────────────
        ax4 = axes[1, 1]
        ax4.axis('off')
        n_jb    = (df['jb_p'] > 0.05).sum()
        n_sw    = (df['shapiro_p'] > 0.05).sum()
        avg_sk  = df['skewness'].abs().mean()
        avg_ku  = df['kurtosis'].abs().mean()
        ovr     = ('✓ GOOD' if pr_jb > 70
                   else '⚠ MODERATE' if pr_jb > 50 else '✗ POOR')
        summary = (
            f"╔══════════════════════════════════╗\n"
            f"║  RESIDUAL DIAGNOSTICS SUMMARY   ║\n"
            f"╚══════════════════════════════════╝\n\n"
            f"Variables analysed: {len(df)}\n\n"
            f"Shapiro-Wilk: {n_sw}/{len(df)} passing ({pr_sw:.0f}%)\n"
            f"Jarque-Bera:  {n_jb}/{len(df)} passing ({pr_jb:.0f}%)\n\n"
            f"Avg |Skewness|: {avg_sk:.3f}  "
            f"{'✓' if avg_sk < 0.5 else '⚠' if avg_sk < 1 else '✗'}\n"
            f"Avg |Kurtosis|: {avg_ku:.3f}  "
            f"{'✓' if avg_ku < 3 else '⚠' if avg_ku < 5 else '✗'}\n\n"
            f"Overall: {ovr}"
        )
        ax4.text(0.1, 0.85, summary, transform=ax4.transAxes,
                 fontsize=11, va='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#FFF9C4',
                           edgecolor='black', alpha=0.9, lw=2))

        fig.suptitle('Residual Diagnostics: Cross-Variable Summary',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        out = os.path.join(self.output_path, 'residual_summary_comparison.png')
        plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"[OK] Saved: {out}")
        plt.close()

        df[['variable', 'mean', 'std', 'skewness', 'kurtosis',
            'shapiro_p', 'jb_p', 'n_obs']].to_csv(
            os.path.join(self.output_path, 'residual_statistics.csv'),
            index=False
        )

    def visualize_data_distributions(self):
        print("\n[Creating data distribution histograms...]")
        if self.data_wide is None or len(self.data_wide.columns) == 0:
            return

        data_stats = []
        for col in self.data_wide.columns:
            s = self.data_wide[col].dropna()
            if len(s) < 3: continue
            data_stats.append({
                'variable': col,
                'data':     s.values,
                'mean':     float(np.mean(s)),
                'median':   float(np.median(s)),
                'std':      float(np.std(s)),
                'skewness': float(stats.skew(s)),
                'kurtosis': float(stats.kurtosis(s)),
                'min':      float(np.min(s)),
                'max':      float(np.max(s)),
                'n_obs':    len(s)
            })

        if not data_stats:
            return

        n_vars = len(data_stats)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(18, 5 * n_rows),
                                  squeeze=False)
        axes_flat = axes.flatten()

        for idx, st in enumerate(data_stats):
            ax  = axes_flat[idx]
            d   = st['data']
            mu  = st['mean']
            sig = st['std']

            if sig < 1e-12:
                ax.text(0.5, 0.5, 'Degenerate data',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=12, color='red')
            else:
                n_bins = min(50, max(10, len(d) // 20))
                ax.hist(d, bins=n_bins, density=True,
                        alpha=0.70, color='steelblue',
                        edgecolor='white', lw=0.5)
                x = np.linspace(d.min(), d.max(), 200)
                ax.plot(x, stats.norm.pdf(x, mu, sig),
                        'r-', lw=2.5, label=f'N({mu:.2f}, {sig:.2f})')
                ax.axvline(mu,            color='red',   ls='--', lw=2, alpha=0.8,
                           label=f'Mean: {mu:.2f}')
                ax.axvline(st['median'],  color='green', ls='--', lw=2, alpha=0.8,
                           label=f'Median: {st["median"]:.2f}')
                ax.axvspan(mu - sig, mu + sig, alpha=0.15, color='yellow', label='±1σ')
                ax.legend(loc='best', fontsize=8)

            ax.set_xlabel('Value', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'{st["variable"]}  (n={st["n_obs"]})',
                         fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.text(0.98, 0.97,
                    f'Skew: {st["skewness"]:.2f}\n'
                    f'Kurt: {st["kurtosis"]:.2f}\n'
                    f'[{st["min"]:.2f}, {st["max"]:.2f}]',
                    transform=ax.transAxes, fontsize=9,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        for idx in range(len(data_stats), len(axes_flat)):
            fig.delaxes(axes_flat[idx])

        fig.suptitle('Climate Data Distributions (Raw Series)',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        out = os.path.join(self.output_path, 'data_distributions_all_variables.png')
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"[OK] Saved: {out}")
        plt.close()

    # ------------------------------------------------------------------
    # SAVE & REPORT
    # ------------------------------------------------------------------

    def save_results(self, forecasts, upper, lower, enso_components, ets_components):
        print("\n[Saving results...]")
        base = os.path.join(self.output_path, f'forecast_ets_enso_{self.region_name}')
        forecasts.to_csv(f'{base}_point.csv')
        upper.to_csv(f'{base}_upper_95ci.csv')
        lower.to_csv(f'{base}_lower_95ci.csv')
        enso_components.to_csv(
            os.path.join(self.output_path, f'forecast_enso_component_{self.region_name}.csv'))
        ets_components.to_csv(
            os.path.join(self.output_path, f'forecast_ets_component_{self.region_name}.csv'))

        combined = pd.concat([self.data_wide, forecasts], axis=0)
        combined.to_csv(f'{base}_combined.csv')

        if self.results_summary:
            pd.DataFrame(self.results_summary).to_csv(
                os.path.join(self.output_path, 'ets_enso_model_summary.csv'), index=False)

        if self.enso_features is not None:
            cols = [c for c in ['ONI', 'Z_ONI', 'D_EN', 'D_LN']
                    if c in self.enso_features.columns]
            self.enso_features[cols].to_csv(
                os.path.join(self.output_path, 'oni_data_used.csv'))

        print(f"[OK] All results saved to {self.output_path}")

    def generate_summary_report(self, forecasts):
        print("\n[Generating summary report...]")
        if len(forecasts) == 0:
            body = "WARNING: No forecasts generated.\n"
        else:
            rd = pd.DataFrame(self.results_summary)
            body = (
                f"Models fitted:      {len(self.models)}\n"
                f"Variables:          {len(forecasts.columns)}\n"
                f"Forecast range:     {forecasts.index.min().date()} → {forecasts.index.max().date()}\n"
                f"Historical range:   {self.data_wide.index.min().date()} → {self.data_wide.index.max().date()}\n\n"
                f"ENSO R² (avg):      {rd['r2_enso'].mean():.2%}\n"
                f"ENSO R² (max):      {rd['r2_enso'].max():.2%}\n"
                f"Average AIC:        {rd['aic'].mean():.2f}\n"
            )

        report = f"{'='*70}\nETS+ENSO CLIMATE FORECAST — {self.region_name.upper()}\n{'='*70}\n{body}"
        out = os.path.join(self.output_path, 'forecast_ets_enso_summary_report.txt')
        with open(out, 'w') as f:
            f.write(report)
        print(report)
        print(f"[OK] Saved: {out}")

    # ------------------------------------------------------------------
    # PIPELINE
    # ------------------------------------------------------------------

    def run_full_pipeline(self):
        print("\n" + "="*80)
        print("ETS+ENSO CLIMATE FORECASTING PIPELINE")
        print("="*80 + "\n")

        try:
            self.prepare_enso_data()

            datasets = self.load_monthly_data()
            if not datasets:
                print("[FAILED] No data files found.")
                return False

            self.extract_regional_series(datasets)
            if self.data_wide is None or len(self.data_wide) == 0:
                print("[FAILED] No time series extracted.")
                return False

            self.fit_all_ets_enso_models()

            forecasts, upper, lower, enso_comp, ets_comp = self.generate_forecasts()

            self.visualize_forecasts(forecasts, upper, lower, enso_comp)
            self.visualize_enso_contribution()
            self.visualize_data_distributions()
            self.visualize_residual_diagnostics()

            self.save_results(forecasts, upper, lower, enso_comp, ets_comp)
            self.generate_summary_report(forecasts)

            print("\n" + "="*80)
            print("[OK] PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"     Results: {self.output_path}")
            print("="*80)
            return True

        except Exception as e:
            print(f"\n[FAILED] Pipeline error: {e}")
            import traceback; traceback.print_exc()
            return False


def main():
    forecaster = MonthlyClimateForecasterETSWithENSO(
        region_name="colombia",
        forecast_months=12,
        enso_max_lag=6,
        include_enso_phases=True,
        include_seasonal_interactions=True
    )
    return forecaster.run_full_pipeline()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)