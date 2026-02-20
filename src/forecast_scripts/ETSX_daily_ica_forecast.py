"""
Daily ETS+ENSO Climate Forecasting with Monthly Aggregation

Implements SARIMAX with ENSO (ONI) as exogenous driver applied to DAILY
ERA5 climate data, then aggregates forecasts to monthly.

FIXES over the broken two-stage Lasso→ETS approach:
  1. Uses a single SARIMAX(p,d,q) model with ENSO as exogenous variable
     instead of the broken Lasso→ETS-on-residuals pipeline.
     Reason: fitting ETS on Lasso residuals loses the level/trend of the
     original series, so ETS forecasts were always near zero.

  2. ENSO lags are now in MONTHS, not DAYS.
     ONI is a monthly index.  Shifting a daily series by N days does not
     equal shifting by N months.  The fix maps each day to its calendar
     month and then to the correctly lagged monthly ONI value.

  3. Future exogenous ENSO values are filled with the most recent observed
     monthly ONI value (last-known-value climatology), which is the correct
     scalar assignment not the broken Series-as-fill-value.

  4. aggregate_to_monthly() now produces clean column names without the
     double variable-name concatenation that previously occurred.

DAILY→MONTHLY AGGREGATION STRATEGY:
  1. Load ERA5 daily data (temperature, precipitation, wind)
  2. Prepare ENSO features (repeat monthly ONI across days, correct lags)
  3. Fit SARIMAX(1,1,1) + ENSO exog for each daily variable
  4. Generate daily forecasts via SARIMAX get_forecast()
  5. Aggregate to monthly (mean for temp/wind, sum for precip)
"""

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
import requests

warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from scipy import stats
from scipy.stats import shapiro, jarque_bera
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ──────────────────────────────────────────────────────────────────────────────
# ONI DATA HANDLER
# ──────────────────────────────────────────────────────────────────────────────

class ONIDataHandlerDaily:
    """
    ONI handler for daily forecasting.
    Repeats monthly ENSO values across daily observations using correct
    calendar-month alignment (no day-based shifting).
    """

    def __init__(self):
        self.oni_raw          = None   # pd.Series, monthly, MS index
        self.oni_standardized = None   # pd.Series, same
        self.training_mean    = None
        self.training_std     = None

    # ── fetch ──────────────────────────────────────────────────────────────

    def fetch_oni_data(self):
        print("\n[Fetching ONI data from NOAA CPC...]")
        try:
            url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
            df  = pd.read_csv(url, delim_whitespace=True, skiprows=1,
                              names=['SEAS', 'YR', 'TOTAL', 'ANOM'])

            season_to_month = {
                'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6,
                'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12
            }
            df['month'] = df['SEAS'].map(season_to_month)
            df['date']  = pd.to_datetime(
                df['YR'].astype(int).astype(str) + '-' +
                df['month'].astype(int).astype(str) + '-01'
            )
            s = pd.Series(df['ANOM'].values, index=df['date'], name='ONI')
            s = s.sort_index()[~s.index.duplicated(keep='first')]

            self.oni_raw = s
            print(f"  [OK] {len(s)} monthly ONI values  "
                  f"({s.index.min().date()} → {s.index.max().date()})")
            return self.oni_raw

        except Exception as e:
            print(f"  [ERROR] {e} — creating dummy ONI")
            dates = pd.date_range('1950-01-01', '2026-12-01', freq='MS')
            t     = np.arange(len(dates))
            self.oni_raw = pd.Series(
                1.5 * np.sin(2*np.pi*t/(4*12)) + 0.3*np.random.randn(len(dates)),
                index=dates, name='ONI'
            )
            return self.oni_raw

    # ── standardise ────────────────────────────────────────────────────────

    def standardize_oni(self, training_start=1961, training_end=2020):
        if self.oni_raw is None:
            self.fetch_oni_data()

        print(f"\n[Standardising ONI ({training_start}–{training_end})...]")
        mask = ((self.oni_raw.index.year >= training_start) &
                (self.oni_raw.index.year <= training_end))
        self.training_mean = float(self.oni_raw[mask].mean())
        self.training_std  = float(self.oni_raw[mask].std())
        self.oni_standardized = (
            (self.oni_raw - self.training_mean) / self.training_std
        )
        print(f"  μ={self.training_mean:.4f}  σ={self.training_std:.4f}")
        print(f"  Z range: [{self.oni_standardized.min():.2f}, "
              f"{self.oni_standardized.max():.2f}]")
        return self.oni_standardized

    # ── daily features ─────────────────────────────────────────────────────

    def create_daily_enso_features(self, date_index, max_lag=6):
        """
        Create daily ENSO feature matrix.

        FIX: Lags are in MONTHS, not days.
        For each day we find its calendar month, then look up the ONI value
        for that month minus `lag` months.  This is the correct interpretation
        of a monthly ENSO lag applied to daily data.

        Args:
            date_index : pd.DatetimeIndex  – daily dates to cover
            max_lag    : int               – number of monthly lags (0…max_lag)

        Returns:
            pd.DataFrame with columns Z_ONI_lag0 … Z_ONI_lag{max_lag}
        """
        print(f"\n[Creating daily ENSO features (monthly lags 0–{max_lag})...]")

        if self.oni_standardized is None:
            self.standardize_oni()

        features = pd.DataFrame(index=date_index)

        for lag in range(max_lag + 1):
            col_name = f'Z_ONI_lag{lag}'
            values   = np.full(len(date_index), np.nan)

            for i, date in enumerate(date_index):
                # Calendar month of this day, shifted back by `lag` months
                target_month = pd.Timestamp(date.year, date.month, 1) - \
                               pd.DateOffset(months=lag)
                if target_month in self.oni_standardized.index:
                    values[i] = self.oni_standardized[target_month]

            # Forward-fill any remaining NaNs (boundary months)
            col_series = pd.Series(values, index=date_index)
            col_series = col_series.ffill().bfill()
            features[col_name] = col_series.values

        print(f"  [OK] {len(features.columns)} features × {len(features)} days")
        return features

    def get_future_enso_features(self, future_dates, max_lag=6):
        """
        Build exogenous ENSO features for the forecast horizon.

        For future months where ONI data is not yet available, we use the
        last known standardised ONI value (persistence forecast).

        Returns:
            pd.DataFrame  same columns as create_daily_enso_features
        """
        if self.oni_standardized is None:
            self.standardize_oni()

        last_known_z = float(self.oni_standardized.iloc[-1])
        last_known_month = self.oni_standardized.index[-1]

        # Extend oni_standardized with persistence values for future months
        future_ms = pd.date_range(
            start=last_known_month + pd.DateOffset(months=1),
            end=pd.Timestamp(future_dates.max().year,
                             future_dates.max().month, 1),
            freq='MS'
        )
        if len(future_ms) > 0:
            extended = pd.concat([
                self.oni_standardized,
                pd.Series(last_known_z, index=future_ms)
            ])
        else:
            extended = self.oni_standardized.copy()

        # Temporarily swap standardized series so create_daily_enso_features
        # can look up future months
        original = self.oni_standardized
        self.oni_standardized = extended

        features = self.create_daily_enso_features(future_dates, max_lag=max_lag)

        self.oni_standardized = original  # restore
        return features


# ──────────────────────────────────────────────────────────────────────────────
# DAILY FORECASTER
# ──────────────────────────────────────────────────────────────────────────────

class DailyClimateForecasterETSWithENSO:
    """
    Daily climate forecaster using SARIMAX + ENSO exogenous variable.

    Core fix: replaces the broken two-stage Lasso→ETS-on-residuals pipeline
    with a single SARIMAX(1,1,1) model where ENSO enters as `exog`.
    This preserves the level and trend of the original series while
    correctly attributing ENSO-driven variance.
    """

    def __init__(self, region_name="colombia", forecast_days=180,
                 enso_max_lag=6):
        self.region_name   = region_name
        self.forecast_days = forecast_days
        self.enso_max_lag  = enso_max_lag

        self.daily_data      = {}
        self.enso_features   = None
        self.models          = {}
        self.forecasts_dict  = {}
        self.residuals_dict  = {}
        self.results_summary = []

        self.oni_handler = ONIDataHandlerDaily()

        self.output_path = os.path.join(
            ".", "articles", "graficas",
            f"forecast_daily_ets_enso_{region_name}"
        )
        os.makedirs(self.output_path, exist_ok=True)

        print(f"[Initialized] Daily SARIMAX+ENSO Forecaster")
        print(f"  Region: {region_name}  |  Forecast: {forecast_days} days  "
              f"|  ENSO max lag: {enso_max_lag} months")
        print(f"  Output: {os.path.abspath(self.output_path)}")

    # ── data loading ───────────────────────────────────────────────────────

    def load_daily_era5_data(self):
        print("\n" + "="*70)
        print("LOADING DAILY ERA5 DATA")
        print("="*70)

        datasets  = {}
        data_path = os.path.join(".", "data", "processed")

        for filename, key, shape_var in [
            ("era5_daily_combined_tmp.nc",  "temperature",   "daily_max"),
            ("era5_daily_combined_rain.nc", "precipitation", "tp_daily_sum"),
            ("era5_daily_combined_wind.nc", "wind",          "wind_speed"),
        ]:
            path = os.path.join(data_path, filename)
            if os.path.exists(path):
                try:
                    ds = xr.open_dataset(path)
                    datasets[key] = ds
                    shape = ds[shape_var].shape if shape_var in ds.data_vars else "?"
                    print(f"  [OK] {key}: {shape}")
                except Exception as e:
                    print(f"  [WARNING] {key}: {e}")
            else:
                print(f"  [WARNING] Not found: {path}")

        return datasets if datasets else None

    def extract_regional_daily_series(self, datasets):
        print("\n[Extracting regional daily time series...]")
        series_dict = {}

        if 'temperature' in datasets:
            ds = datasets['temperature']
            for var in ['daily_max', 'daily_min']:
                if var in ds.data_vars:
                    s = (ds[var]
                         .mean(dim=['latitude', 'longitude'], skipna=True)
                         .to_pandas())
                    s.index = pd.to_datetime(s.index)
                    s = s[~s.index.duplicated(keep='first')]
                    series_dict[f'temperature_{var}'] = s
                    print(f"  [OK] temperature_{var}: {len(s)} days")

        if 'precipitation' in datasets:
            ds = datasets['precipitation']
            if 'tp_daily_sum' in ds.data_vars:
                s = (ds['tp_daily_sum']
                     .mean(dim=['latitude', 'longitude'], skipna=True)
                     .to_pandas())
                s.index = pd.to_datetime(s.index)
                s = s[~s.index.duplicated(keep='first')]
                series_dict['precipitation_daily'] = s
                print(f"  [OK] precipitation_daily: {len(s)} days")

        if 'wind' in datasets:
            ds = datasets['wind']
            if 'wind_speed' in ds.data_vars:
                s = (ds['wind_speed']
                     .mean(dim=['latitude', 'longitude'], skipna=True)
                     .to_pandas())
                s.index = pd.to_datetime(s.index)
                s = s[~s.index.duplicated(keep='first')]
                series_dict['wind_speed_daily'] = s
                print(f"  [OK] wind_speed_daily: {len(s)} days")

        if not series_dict:
            print("  [ERROR] No series extracted")
            return None

        # ── align to common date range ─────────────────────────────
        min_date = max(s.index.min() for s in series_dict.values())
        max_date = min(s.index.max() for s in series_dict.values())
        common   = pd.date_range(min_date, max_date, freq='D')
        print(f"\n  Common range: {min_date.date()} → {max_date.date()} "
              f"({len(common)} days)")

        for k in list(series_dict):
            series_dict[k] = (series_dict[k]
                              .reindex(common)
                              .ffill()
                              .bfill())

        self.daily_data = series_dict
        print(f"\n[OK] {len(series_dict)} daily series loaded")
        return series_dict

    # ── ENSO preparation ───────────────────────────────────────────────────

    def prepare_enso_data_daily(self):
        print("\n" + "="*70)
        print("PREPARING DAILY ENSO DATA")
        print("="*70)

        self.oni_handler.fetch_oni_data()
        self.oni_handler.standardize_oni()

        # Use date index from the first loaded variable
        first_series = next(iter(self.daily_data.values()))
        self.enso_features = self.oni_handler.create_daily_enso_features(
            first_series.index, max_lag=self.enso_max_lag
        )
        return self.enso_features

    # ── SARIMAX model fitting ──────────────────────────────────────────────

    def _select_enso_features(self, X, y):
        """
        Use Lasso to select the most informative ENSO columns.
        Returns the reduced X and the list of selected column names.
        """
        try:
            lasso = LassoCV(cv=5, max_iter=10000, random_state=42,
                            alphas=np.logspace(-4, 1, 40))
            lasso.fit(X, y)
            mask = np.abs(lasso.coef_) > 1e-10
            selected = X.columns[mask].tolist()
            if not selected:
                # Fall back to top-3 by coefficient magnitude
                top3 = np.argsort(np.abs(lasso.coef_))[-3:]
                selected = X.columns[top3].tolist()
        except Exception:
            selected = X.columns.tolist()[:3]

        return X[selected], selected

    def fit_daily_sarimax_model(self, series, variable_name):
        """
        Fit SARIMAX(1,1,1) model with ENSO as exogenous variable.

        This is the CORRECT single-stage approach:
          - ARIMA (1,1,1) captures autocorrelation and trend in the raw series
          - ENSO columns enter as `exog` in the state space
          - get_forecast() correctly propagates both components forward

        No seasonal component is included because daily seasonality (365-day)
        requires far more data than is typically available and massively
        increases estimation cost.  Monthly seasonality is captured implicitly
        through the lagged monthly ONI features.
        """
        print(f"\n  [SARIMAX] {variable_name}")

        # ── align & clean ──────────────────────────────────────────
        common = series.index.intersection(self.enso_features.index)
        y = series.loc[common].copy()
        X = self.enso_features.loc[common].copy()

        valid = ~(y.isna() | X.isna().any(axis=1))
        y = y[valid]
        X = X[valid]

        if len(y) < 365:
            print(f"    [SKIP] Only {len(y)} valid observations (need ≥ 365)")
            return None

        print(f"    Observations: {len(y)}")

        # ── feature selection ──────────────────────────────────────
        X_sel, sel_cols = self._select_enso_features(X, y)
        print(f"    Selected ENSO features: {sel_cols}")

        # ── fit SARIMAX(1,1,1) + exog ──────────────────────────────
        try:
            model  = SARIMAX(y, exog=X_sel,
                             order=(1, 1, 1),
                             seasonal_order=(0, 0, 0, 0),
                             enforce_stationarity=False,
                             enforce_invertibility=False)
            fitted = model.fit(disp=False, maxiter=300)

            print(f"    AIC:       {fitted.aic:.2f}")
            print(f"    Converged: {fitted.mle_retvals['converged']}")

        except Exception as e:
            print(f"    [ERROR] SARIMAX fit failed: {e}")
            return None

        # ── ENSO R² (variance of ENSO component / variance of y) ──
        try:
            coefs       = fitted.params.reindex(X_sel.columns).fillna(0.0)
            enso_comp   = pd.Series(X_sel.values @ coefs.values, index=y.index)
            enso_r2     = float(np.clip(
                np.var(enso_comp.values) / np.var(y.values), 0.0, 1.0
            ))
        except Exception:
            enso_r2 = np.nan

        # ── residual diagnostics ───────────────────────────────────
        resid = fitted.resid.values
        self.residuals_dict[variable_name] = resid

        try:
            sw_p = float(shapiro(resid[:5000])[1])
        except Exception:
            sw_p = np.nan
        try:
            jb_p = float(jarque_bera(resid)[1])
        except Exception:
            jb_p = np.nan

        print(f"    ENSO R²:    {enso_r2:.4f}")
        print(f"    Resid std:  {np.std(resid):.4f}")
        print(f"    JB p-val:   {jb_p:.4f}")

        result = {
            'variable':    variable_name,
            'sarimax':     fitted,
            'X_sel':       X_sel,
            'sel_cols':    sel_cols,
            'enso_r2':     enso_r2,
            'aic':         fitted.aic,
            'n_obs':       len(y),
            'y_train':     y,
            'enso_coefs':  coefs if 'coefs' in dir() else None,
        }

        self.results_summary.append({
            'variable':  variable_name,
            'enso_r2':   enso_r2,
            'aic':       fitted.aic,
            'mean_res':  float(np.mean(resid)),
            'std_res':   float(np.std(resid)),
            'skewness':  float(stats.skew(resid)),
            'kurtosis':  float(stats.kurtosis(resid)),
            'shapiro_p': sw_p,
            'jb_p':      jb_p,
        })

        return result

    def fit_all_daily_models(self):
        print("\n" + "="*70)
        print("FITTING DAILY SARIMAX+ENSO MODELS")
        print("="*70)

        for var_name, series in self.daily_data.items():
            result = self.fit_daily_sarimax_model(series, var_name)
            if result is not None:
                self.models[var_name] = result

        print(f"\n[OK] Fitted {len(self.models)}/{len(self.daily_data)} models")
        return self.models

    # ── forecasting ────────────────────────────────────────────────────────

    def forecast_daily(self, steps=None):
        """
        Generate daily forecasts using SARIMAX get_forecast().

        FIX: Uses the correct SARIMAX forecast path.
          - get_forecast(steps, exog=future_exog) handles the joint
            autoregressive + exogenous propagation correctly.
          - Future ENSO is filled with persistence (last observed monthly ONI).
        """
        if steps is None:
            steps = self.forecast_days

        print(f"\n[Forecasting {steps} days ahead...]")

        last_date    = max(s.index.max() for s in self.daily_data.values())
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=steps, freq='D'
        )

        # Build future ENSO features with persistence fill
        future_enso = self.oni_handler.get_future_enso_features(
            future_dates, max_lag=self.enso_max_lag
        )

        forecasts_dict = {}

        for var_name, result in self.models.items():
            try:
                print(f"\n  Forecasting {var_name}...")

                fitted   = result['sarimax']
                sel_cols = result['sel_cols']

                # Select same columns used during training
                future_X = future_enso[sel_cols].copy()

                # Safety: fill any residual NaN with column mean from training
                for col in sel_cols:
                    if future_X[col].isna().any():
                        fill_val = float(result['X_sel'][col].mean())
                        future_X[col] = future_X[col].fillna(fill_val)

                # SARIMAX forecast
                fc_result  = fitted.get_forecast(steps=steps, exog=future_X)
                fc_mean    = fc_result.predicted_mean
                fc_ci      = fc_result.conf_int(alpha=0.05)

                # ENSO contribution to forecast
                try:
                    coefs      = fitted.params.reindex(sel_cols).fillna(0.0)
                    enso_fc    = pd.Series(
                        future_X.values @ coefs.values, index=future_dates
                    )
                except Exception:
                    enso_fc = pd.Series(0.0, index=future_dates)

                forecast_df = pd.DataFrame({
                    'point':          fc_mean.values,
                    'lower_95':       fc_ci.iloc[:, 0].values,
                    'upper_95':       fc_ci.iloc[:, 1].values,
                    'enso_component': enso_fc.values,
                    'ets_component':  fc_mean.values - enso_fc.values,
                }, index=future_dates)

                forecasts_dict[var_name] = forecast_df
                print(f"    [OK] Range: [{fc_mean.min():.3f}, {fc_mean.max():.3f}]")

            except Exception as e:
                print(f"    [ERROR] {var_name}: {e}")
                import traceback; traceback.print_exc()

        self.forecasts_dict = forecasts_dict
        print(f"\n[OK] Generated forecasts for {len(forecasts_dict)} variables")
        return forecasts_dict

    # ── monthly aggregation ────────────────────────────────────────────────

    def aggregate_to_monthly(self, forecast_dict):
        """
        Aggregate daily forecasts to monthly statistics.

        FIX: builds column names directly, avoiding the double-concatenation
        that pd.concat MultiIndex produced in the original code.

        Aggregation rules:
          - precipitation → monthly SUM
          - everything else → monthly MEAN
        """
        print("\n[Aggregating daily forecasts to monthly...]")

        monthly_frames = []

        for var_name, daily_df in forecast_dict.items():
            if daily_df is None or len(daily_df) == 0:
                continue

            agg_func = 'sum' if 'precipitation' in var_name.lower() else 'mean'

            monthly = daily_df.resample('MS').agg(agg_func)

            # Rename columns: prepend variable name cleanly
            monthly.columns = [f'{var_name}_{col}' for col in monthly.columns]
            monthly_frames.append(monthly)

            print(f"  {var_name}: {len(monthly)} months  (agg={agg_func})")

        if not monthly_frames:
            print("  [WARNING] No monthly data produced")
            return None

        monthly_combined = pd.concat(monthly_frames, axis=1)
        print(f"\n[OK] Monthly aggregate shape: {monthly_combined.shape}")
        return monthly_combined

    # ── visualisations ─────────────────────────────────────────────────────

    def visualize_forecasts(self, forecast_dict):
        """Plot historical daily series + daily forecast with CI band."""
        print("\n[Creating forecast visualisations...]")

        n_vars = len(forecast_dict)
        if n_vars == 0:
            return

        n_cols = 2
        n_rows = (n_vars + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(18, 4 * n_rows),
                                  squeeze=False)
        axes_flat = axes.flatten()

        for idx, (var_name, fc_df) in enumerate(forecast_dict.items()):
            ax = axes_flat[idx]

            # Historical (last 2 years for readability)
            hist = self.daily_data[var_name]
            hist_plot = hist[hist.index >= hist.index.max() - pd.DateOffset(years=2)]

            ax.plot(hist_plot.index, hist_plot.values,
                    color='steelblue', lw=0.8, alpha=0.7, label='Historical')
            ax.plot(fc_df.index, fc_df['point'].values,
                    color='crimson', lw=1.5, label='Forecast')
            ax.fill_between(fc_df.index,
                            fc_df['lower_95'].values,
                            fc_df['upper_95'].values,
                            color='crimson', alpha=0.15, label='95% CI')

            ax.axvline(x=hist.index.max(), color='black',
                       ls='--', lw=1, alpha=0.5)
            ax.set_title(var_name, fontsize=11, fontweight='bold')
            ax.set_xlabel('Date')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for idx in range(n_vars, len(axes_flat)):
            fig.delaxes(axes_flat[idx])

        plt.suptitle('Daily SARIMAX+ENSO Forecasts', fontsize=14, fontweight='bold')
        plt.tight_layout()
        out = os.path.join(self.output_path, 'forecast_daily_all_variables.png')
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"  [OK] {out}")
        plt.close()

    def visualize_enso_contribution(self):
        """Bar chart of ENSO R² for each variable."""
        print("\n[Creating ENSO contribution chart...]")
        if not self.results_summary:
            return

        df = (pd.DataFrame(self.results_summary)
               .sort_values('enso_r2', ascending=True))

        fig, ax = plt.subplots(figsize=(10, max(4, len(df)*0.9 + 1.5)))
        y_pos  = np.arange(len(df))
        r2_pct = df['enso_r2'].values * 100

        colors = ['#2E7D32' if v > 30 else '#66BB6A' if v > 20
                  else '#FFA726' if v > 10 else '#FFCA28' if v > 5
                  else '#BDBDBD' for v in r2_pct]

        bars = ax.barh(y_pos, r2_pct, color=colors,
                       edgecolor='black', lw=0.8, alpha=0.85)

        x_max = max(r2_pct) if len(r2_pct) > 0 else 1.0
        for bar, val in zip(bars, r2_pct):
            ax.text(val + x_max*0.02,
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', ha='left',
                    fontsize=9, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['variable'], fontsize=10)
        ax.set_xlabel('ENSO Explanatory Power (R² %)', fontsize=11, fontweight='bold')
        ax.set_title('ENSO (ONI) Contribution to Daily Climate Variability',
                     fontsize=13, fontweight='bold', pad=12)
        ax.axvline(30, color='darkgreen', ls='--', lw=1.5, alpha=0.4)
        ax.axvline(10, color='orange',    ls='--', lw=1.5, alpha=0.4)
        ax.set_xlim(0, x_max * 1.15)
        ax.grid(True, alpha=0.3, axis='x', ls=':', lw=1)
        ax.set_axisbelow(True)

        plt.tight_layout()
        out = os.path.join(self.output_path, 'enso_contribution_daily.png')
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  [OK] {out}")
        plt.close(fig)

    def visualize_residual_diagnostics(self):
        """4-panel residual diagnostics for each variable."""
        print("\n[Creating residual diagnostics...]")
        if not self.residuals_dict:
            return

        residual_stats = []
        for var_name, resid in self.residuals_dict.items():
            clean = resid[~np.isnan(resid)]
            if len(clean) < 10:
                continue
            try:
                sw_p = float(shapiro(clean[:5000])[1])
            except Exception:
                sw_p = np.nan
            try:
                jb_p = float(jarque_bera(clean)[1])
            except Exception:
                jb_p = np.nan
            residual_stats.append({
                'variable':  var_name,
                'mean':      float(np.mean(clean)),
                'std':       float(np.std(clean)),
                'skewness':  float(stats.skew(clean)),
                'kurtosis':  float(stats.kurtosis(clean)),
                'shapiro_p': sw_p,
                'jb_p':      jb_p,
                'n_obs':     len(clean),
                'residuals': clean,
            })

        if not residual_stats:
            return

        n_vars = len(residual_stats)
        fig, axes = plt.subplots(n_vars, 4,
                                  figsize=(22, 5*n_vars),
                                  squeeze=False)
        fig.suptitle('Residual Diagnostics — Daily SARIMAX+ENSO',
                     fontsize=15, fontweight='bold', y=1.01)

        for row, stat in enumerate(residual_stats):
            resid = stat['residuals']
            mu, sg = stat['mean'], stat['std']

            # histogram
            ax = axes[row, 0]
            if sg > 1e-12:
                n_bins = min(60, max(15, len(resid)//50))
                ax.hist(resid, bins=n_bins, density=True,
                        color='steelblue', alpha=0.7, edgecolor='white', lw=0.4)
                x = np.linspace(resid.min(), resid.max(), 200)
                ax.plot(x, stats.norm.pdf(x, mu, sg), 'r-', lw=2)
                ax.axvline(mu, color='red', ls='--', lw=1.5, alpha=0.7)
                ax.legend([f'N({mu:.3f}, {sg:.3f})'], fontsize=7)
            else:
                ax.text(0.5, 0.5, 'Degenerate', transform=ax.transAxes,
                        ha='center', va='center', color='red')
            ax.set_title(f'{stat["variable"]}\nHistogram',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('Residual', fontsize=9)
            ax.set_ylabel('Density',  fontsize=9)
            ax.grid(True, alpha=0.3)

            # Q-Q
            ax = axes[row, 1]
            if sg > 1e-12:
                (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist='norm')
                ax.scatter(osm, osr, s=4, color='steelblue', alpha=0.5)
                lx = np.array([osm.min(), osm.max()])
                ax.plot(lx, slope*lx + intercept, 'r-', lw=2)
            ax.set_title('Q-Q Plot', fontsize=10, fontweight='bold')
            ax.set_xlabel('Theoretical quantiles', fontsize=9)
            ax.set_ylabel('Sample quantiles',      fontsize=9)
            ax.grid(True, alpha=0.3)

            # box plot
            ax = axes[row, 2]
            ax.boxplot(resid, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', lw=2),
                       whiskerprops=dict(lw=1.5), capprops=dict(lw=1.5))
            ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.7)
            q1, q3 = np.percentile(resid, [25, 75])
            n_out  = ((resid < q1-1.5*(q3-q1)) | (resid > q3+1.5*(q3-q1))).sum()
            ax.text(0.5, 0.02, f'Outliers: {n_out} ({n_out/len(resid)*100:.1f}%)',
                    transform=ax.transAxes, ha='center', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_title('Box Plot', fontsize=10, fontweight='bold')
            ax.set_xticklabels(['Residuals'])
            ax.grid(True, alpha=0.3, axis='y')

            # stats summary
            ax = axes[row, 3]
            ax.axis('off')
            is_n_jb = stat['jb_p'] > 0.05 if not np.isnan(stat['jb_p']) else None
            is_n_sw = stat['shapiro_p'] > 0.05 if not np.isnan(stat['shapiro_p']) else None
            txt = (
                f"n = {stat['n_obs']}\n\n"
                f"Mean:     {stat['mean']:.4f}\n"
                f"Std:      {stat['std']:.4f}\n"
                f"Skewness: {stat['skewness']:.4f}\n"
                f"Kurtosis: {stat['kurtosis']:.4f}\n\n"
                f"Shapiro-Wilk  p={stat['shapiro_p']:.4f}\n"
                f"  {'✓ Normal' if is_n_sw else '✗ Non-normal' if is_n_sw is not None else 'N/A'}\n"
                f"Jarque-Bera   p={stat['jb_p']:.4f}\n"
                f"  {'✓ Normal' if is_n_jb else '✗ Non-normal' if is_n_jb is not None else 'N/A'}\n\n"
                f"Overall: {'✓ NORMAL' if is_n_jb else '✗ NON-NORMAL' if is_n_jb is not None else 'N/A'}"
            )
            ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                    fontsize=9, va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.35))

        plt.tight_layout()
        out = os.path.join(self.output_path, 'residual_diagnostics_daily.png')
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"  [OK] {out}")
        plt.close()

    def visualize_monthly_aggregates(self, monthly_df):
        """Bar chart of monthly aggregated forecast values."""
        print("\n[Creating monthly aggregate visualisation...]")
        if monthly_df is None or len(monthly_df) == 0:
            return

        # Only plot 'point' columns
        point_cols = [c for c in monthly_df.columns if c.endswith('_point')]
        if not point_cols:
            return

        n_cols = 2
        n_rows = (len(point_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(16, 4*n_rows),
                                  squeeze=False)
        axes_flat = axes.flatten()

        for idx, col in enumerate(point_cols):
            ax = axes_flat[idx]
            ax.bar(monthly_df.index, monthly_df[col].values,
                   color='steelblue', alpha=0.8, edgecolor='white', lw=0.5,
                   width=25)
            ax.set_title(col, fontsize=11, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Aggregated value')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')

        for idx in range(len(point_cols), len(axes_flat)):
            fig.delaxes(axes_flat[idx])

        plt.suptitle('Monthly Aggregated Forecasts', fontsize=14, fontweight='bold')
        plt.tight_layout()
        out = os.path.join(self.output_path, 'forecast_monthly_aggregated.png')
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"  [OK] {out}")
        plt.close()

    # ── save & report ──────────────────────────────────────────────────────

    def save_results(self, daily_forecasts, monthly_forecasts):
        print("\n[Saving results...]")

        for var_name, df in daily_forecasts.items():
            path = os.path.join(self.output_path, f'forecast_daily_{var_name}.csv')
            df.to_csv(path)
            print(f"  [OK] {path}")

        if monthly_forecasts is not None:
            path = os.path.join(self.output_path, 'forecast_monthly_aggregated.csv')
            monthly_forecasts.to_csv(path)
            print(f"  [OK] {path}")

        if self.results_summary:
            path = os.path.join(self.output_path, 'model_summary.csv')
            pd.DataFrame(self.results_summary).to_csv(path, index=False)
            print(f"  [OK] {path}")

    def generate_summary_report(self):
        print("\n[Generating summary report...]")
        lines = [
            "="*70,
            f"DAILY SARIMAX+ENSO FORECAST — {self.region_name.upper()}",
            "="*70,
            f"Forecast horizon:  {self.forecast_days} days",
            f"ENSO max lag:      {self.enso_max_lag} months",
            f"Models fitted:     {len(self.models)}",
            "",
            "ENSO R² by variable:",
        ]
        for s in self.results_summary:
            lines.append(f"  {s['variable']:35s}  R²={s['enso_r2']:.4f}  "
                         f"AIC={s['aic']:.1f}")
        lines += [
            "",
            "Residual normality (JB test):",
        ]
        for s in self.results_summary:
            ok = "✓ Normal" if s['jb_p'] > 0.05 else "✗ Non-normal"
            lines.append(f"  {s['variable']:35s}  JB p={s['jb_p']:.4f}  {ok}")
        lines += ["", f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}", "="*70]

        report = "\n".join(lines)
        path   = os.path.join(self.output_path, 'forecast_summary_report.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(report)
        print(f"\n[OK] Report saved: {path}")

    # ── pipeline ───────────────────────────────────────────────────────────

    def run_full_pipeline(self):
        print("\n" + "="*70)
        print("DAILY SARIMAX+ENSO CLIMATE FORECASTING PIPELINE")
        print("="*70 + "\n")

        try:
            # 1. Load
            datasets = self.load_daily_era5_data()
            if not datasets:
                print("[FAILED] No data loaded"); return False

            # 2. Extract
            self.extract_regional_daily_series(datasets)
            if not self.daily_data:
                print("[FAILED] No regional series"); return False

            # 3. ENSO
            self.prepare_enso_data_daily()
            if self.enso_features is None:
                print("[FAILED] ENSO features missing"); return False

            # 4. Fit
            self.fit_all_daily_models()
            if not self.models:
                print("[FAILED] No models fitted"); return False

            # 5. Forecast
            daily_fc = self.forecast_daily(steps=self.forecast_days)
            if not daily_fc:
                print("[FAILED] No forecasts"); return False

            # 6. Aggregate
            monthly_fc = self.aggregate_to_monthly(daily_fc)

            # 7. Visualise
            self.visualize_forecasts(daily_fc)
            self.visualize_enso_contribution()
            self.visualize_residual_diagnostics()
            self.visualize_monthly_aggregates(monthly_fc)

            # 8. Save
            self.save_results(daily_fc, monthly_fc)
            self.generate_summary_report()

            print("\n" + "="*70)
            print("[OK] PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"     Results: {self.output_path}")
            print("="*70)
            return True

        except Exception as e:
            print(f"\n[FAILED] {e}")
            import traceback; traceback.print_exc()
            return False


# ──────────────────────────────────────────────────────────────────────────────

def main():
    forecaster = DailyClimateForecasterETSWithENSO(
        region_name="colombia",
        forecast_days=180,   # 6 months of daily forecasts
        enso_max_lag=6
    )
    return forecaster.run_full_pipeline()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)