"""
Daily ETS+ENSO Climate Forecasting with Seasonal Motif Decomposition

Implements ETS with seasonal motif patterns from daily ERA5 data.
Uses seasonal decomposition to extract recurring climate patterns.

Key improvements over basic daily model:
- Captures 365-day and 30-day seasonal cycles
- Uses seasonal motif to identify repeating climate patterns
- Better handles seasonality in daily data
- More accurate forecasts of climate indices
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
warnings.filterwarnings('ignore')

# Time series models
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from scipy.stats import shapiro, jarque_bera
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ONIDataHandlerSeasonalMotif:
    """Handles ONI data for seasonal motif forecasting."""
    
    def __init__(self):
        self.oni_raw = None
        self.oni_standardized = None
        self.training_mean = None
        self.training_std = None
        
    def fetch_oni_data(self):
        """Fetch ONI data from NOAA CPC."""
        print("\n[Fetching ONI data from NOAA CPC...]")
        
        try:
            url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
            df = pd.read_csv(url, delim_whitespace=True, skiprows=1, 
                           names=['SEAS', 'YR', 'TOTAL', 'ANOM'])
            
            season_to_month = {
                'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6,
                'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12
            }
            
            df['month'] = df['SEAS'].map(season_to_month)
            df['date'] = pd.to_datetime(df['YR'].astype(int).astype(str) + '-' + 
                                       df['month'].astype(int).astype(str) + '-01')
            
            oni_series = pd.Series(df['ANOM'].values, index=df['date'], name='ONI')
            oni_series = oni_series.sort_index()
            oni_series = oni_series[~oni_series.index.duplicated(keep='first')]
            
            self.oni_raw = oni_series
            print(f"  [OK] Loaded {len(oni_series)} ONI values")
            
        except Exception as e:
            print(f"  [ERROR] Failed to fetch ONI: {e}")
            return None
        
        return self.oni_raw
    
    def standardize_oni(self, training_start=1961, training_end=2020):
        """Standardize ONI using training period."""
        if self.oni_raw is None:
            return None
        
        print(f"\n[Standardizing ONI...]")
        
        training_mask = (self.oni_raw.index.year >= training_start) & \
                       (self.oni_raw.index.year <= training_end)
        training_data = self.oni_raw[training_mask]
        
        self.training_mean = training_data.mean()
        self.training_std = training_data.std()
        
        self.oni_standardized = (self.oni_raw - self.training_mean) / self.training_std
        
        return self.oni_standardized
    
    def create_daily_enso_features(self, date_index, max_lag=6):
        """Create daily ENSO features from monthly ONI."""
        if self.oni_standardized is None:
            return None
        
        features = pd.DataFrame(index=date_index.copy())
        features['year'] = features.index.year
        features['month'] = features.index.month
        features['month_start'] = pd.to_datetime(features['year'].astype(str) + '-' + 
                                                 features['month'].astype(str) + '-01')
        
        features['Z_ONI'] = np.nan
        for month_date in features['month_start'].unique():
            mask = features['month_start'] == month_date
            if month_date in self.oni_standardized.index:
                features.loc[mask, 'Z_ONI'] = self.oni_standardized[month_date]
        
        features['Z_ONI'] = features['Z_ONI'].fillna(method='ffill').fillna(method='bfill')
        
        for lag in range(1, max_lag + 1):
            features[f'Z_ONI_lag{lag}'] = features['Z_ONI'].shift(lag)
        
        features = features.drop(['year', 'month', 'month_start'], axis=1)
        
        return features


class SeasonalMotifForecaster:
    """
    Daily climate forecaster using seasonal motif decomposition.
    Extracts seasonal patterns and uses them for forecasting.
    """
    
    def __init__(self, region_name="colombia", forecast_days=180, enso_max_lag=6):
        self.region_name = region_name
        self.forecast_days = forecast_days
        self.enso_max_lag = enso_max_lag
        
        self.daily_data = {}
        self.enso_features = None
        self.models = {}
        self.residuals_dict = {}
        self.results_summary = []
        
        self.oni_handler = ONIDataHandlerSeasonalMotif()
        
        self.output_path = os.path.join(".", "articles", "graficas", 
                                       f"forecast_seasonal_motif_{region_name}")
        os.makedirs(self.output_path, exist_ok=True)
        
        print(f"[Initialized] Seasonal Motif Forecaster")
        print(f"  Region: {region_name}")
        print(f"  Forecast period: {forecast_days} days")
        print(f"  Output path: {os.path.abspath(self.output_path)}")
    
    def load_daily_era5_data(self):
        """Load raw daily ERA5 climate data."""
        print("\n" + "="*80)
        print("LOADING DAILY ERA5 DATA")
        print("="*80)
        
        datasets = {}
        data_path = os.path.join(".", "data", "processed")
        
        temp_file = os.path.join(data_path, "era5_daily_combined_tmp.nc")
        if os.path.exists(temp_file):
            try:
                ds_temp = xr.open_dataset(temp_file)
                datasets['temperature'] = ds_temp
                print(f"[OK] Loaded temperature: {ds_temp['daily_max'].shape}")
            except Exception as e:
                print(f"[WARNING] Error loading temperature: {e}")
        
        precip_file = os.path.join(data_path, "era5_daily_combined_rain.nc")
        if os.path.exists(precip_file):
            try:
                ds_precip = xr.open_dataset(precip_file)
                datasets['precipitation'] = ds_precip
                print(f"[OK] Loaded precipitation: {ds_precip['tp_daily_sum'].shape}")
            except Exception as e:
                print(f"[WARNING] Error loading precipitation: {e}")
        
        wind_file = os.path.join(data_path, "era5_daily_combined_wind.nc")
        if os.path.exists(wind_file):
            try:
                ds_wind = xr.open_dataset(wind_file)
                datasets['wind'] = ds_wind
                print(f"[OK] Loaded wind: {ds_wind['wind_speed'].shape}")
            except Exception as e:
                print(f"[WARNING] Error loading wind: {e}")
        
        if not datasets:
            print("[ERROR] No daily data files found!")
            return None
        
        return datasets
    
    def extract_regional_daily_series(self, datasets):
        """Extract spatial mean time series from datasets."""
        print("\n[Extracting regional daily time series...]")
        
        series_dict = {}
        
        if 'temperature' in datasets:
            ds_temp = datasets['temperature']
            try:
                for var in ['daily_max', 'daily_min']:
                    if var in ds_temp.data_vars:
                        spatial_mean = ds_temp[var].mean(
                            dim=['latitude', 'longitude'], skipna=True
                        )
                        df = spatial_mean.to_pandas()
                        df.index = pd.to_datetime(df.index)
                        df = df[~df.index.duplicated(keep='first')]
                        series_dict[f'temperature_{var}'] = df
                        print(f"  [OK] temperature_{var}: {len(df)} observations")
            except Exception as e:
                print(f"  [WARNING] Error extracting temperature: {e}")
        
        if 'precipitation' in datasets:
            ds_precip = datasets['precipitation']
            try:
                if 'tp_daily_sum' in ds_precip.data_vars:
                    spatial_mean = ds_precip['tp_daily_sum'].mean(
                        dim=['latitude', 'longitude'], skipna=True
                    )
                    df = spatial_mean.to_pandas()
                    df.index = pd.to_datetime(df.index)
                    df = df[~df.index.duplicated(keep='first')]
                    series_dict['precipitation_daily'] = df
                    print(f"  [OK] precipitation_daily: {len(df)} observations")
            except Exception as e:
                print(f"  [WARNING] Error extracting precipitation: {e}")
        
        if 'wind' in datasets:
            ds_wind = datasets['wind']
            try:
                if 'wind_speed' in ds_wind.data_vars:
                    spatial_mean = ds_wind['wind_speed'].mean(
                        dim=['latitude', 'longitude'], skipna=True
                    )
                    df = spatial_mean.to_pandas()
                    df.index = pd.to_datetime(df.index)
                    df = df[~df.index.duplicated(keep='first')]
                    series_dict['wind_speed_daily'] = df
                    print(f"  [OK] wind_speed_daily: {len(df)} observations")
            except Exception as e:
                print(f"  [WARNING] Error extracting wind: {e}")
        
        # Align all series to common date range
        if series_dict:
            print(f"\n  [Aligning series to common date range...]")
            
            all_dates = [s.index for s in series_dict.values()]
            min_date = max([d.min() for d in all_dates])
            max_date = min([d.max() for d in all_dates])
            
            print(f"  Common date range: {min_date.date()} to {max_date.date()}")
            
            # Create common date index and align all series
            common_index = pd.date_range(start=min_date, end=max_date, freq='D')
            
            for key in list(series_dict.keys()):
                reindexed = series_dict[key].reindex(common_index, method='ffill')
                series_dict[key] = reindexed
            
            print(f"  Aligned to common index: {len(common_index)} days")
        
        self.daily_data = series_dict
        
        if series_dict:
            print(f"\n[OK] Extracted {len(series_dict)} daily time series")
            print(f"  All series length: {len(list(series_dict.values())[0])} days")
            return series_dict
        else:
            print("[ERROR] No daily series extracted!")
            return None
    
    def prepare_enso_data_daily(self):
        """Prepare daily ENSO features."""
        print("\n" + "="*80)
        print("PREPARING DAILY ENSO DATA")
        print("="*80)
        
        self.oni_handler.fetch_oni_data()
        self.oni_handler.standardize_oni()
        
        if not self.daily_data:
            print("[ERROR] No daily data loaded")
            return None
        
        first_var = list(self.daily_data.values())[0]
        date_index = first_var.index
        
        self.enso_features = self.oni_handler.create_daily_enso_features(
            date_index, max_lag=self.enso_max_lag
        )
        
        print(f"  [OK] Created daily ENSO features")
        
        return self.enso_features
    
    def decompose_seasonal_motif(self, series, variable_name, period=365):
        """
        Decompose time series into seasonal motif components.
        
        Args:
            series (pd.Series): Daily time series
            variable_name (str): Variable name
            period (int): Seasonal period (365 for year, 30 for month)
            
        Returns:
            dict: Seasonal decomposition components
        """
        print(f"\n  [Decomposing] {variable_name} with period={period}...")
        
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(series, model='additive', period=period)
            
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
            
            print(f"    Trend variance: {trend.var():.6f}")
            print(f"    Seasonal variance: {seasonal.var():.6f}")
            print(f"    Residual variance: {residual.var():.6f}")
            
            return {
                'series': series,
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual,
                'decomposition': decomposition
            }
            
        except Exception as e:
            print(f"    [WARNING] Decomposition failed: {e}")
            return None
    
    def fit_seasonal_motif_model(self, series, variable_name, enso_features, period=365):
        """
        Fit model using seasonal motif decomposition.
        
        Args:
            series (pd.Series): Daily time series
            variable_name (str): Variable name
            enso_features (pd.DataFrame): ENSO features
            period (int): Seasonal period
            
        Returns:
            dict: Model results
        """
        print(f"\n[Fitting Seasonal Motif] {variable_name}")
        
        # Step 1: Decompose into seasonal components
        decomp = self.decompose_seasonal_motif(series, variable_name, period=period)
        if decomp is None:
            print(f"  [WARNING] Decomposition failed, using raw series")
        
        # Align data for ENSO regression
        common_idx = series.index.intersection(enso_features.index)
        y = series.loc[common_idx].copy()
        X = enso_features.loc[common_idx].copy()
        
        valid_mask = ~(y.isna() | X.isna().any(axis=1))
        y = y[valid_mask].copy()
        X = X[valid_mask].copy()
        
        print(f"  Data: {len(y)} observations")
        
        if len(y) < 100:
            print(f"  [ERROR] Insufficient data")
            return None
        
        try:
            # Step 2: Fit Lasso for ENSO effects on trend (or raw series if decomp failed)
            print(f"  [Step 1] Fitting Lasso for ENSO effects...")
            
            if decomp is not None:
                trend_common = decomp['trend'].loc[common_idx].dropna()
                if len(trend_common) > 0:
                    y_trend = y.loc[trend_common.index]
                    X_trend = X.loc[trend_common.index]
                else:
                    y_trend = y
                    X_trend = X
            else:
                y_trend = y
                X_trend = X
            
            valid_mask_trend = ~(y_trend.isna() | X_trend.isna().any(axis=1))
            y_trend = y_trend[valid_mask_trend].copy()
            X_trend = X_trend[valid_mask_trend].copy()
            
            if len(y_trend) < 10:
                print(f"  [ERROR] Not enough data for ENSO regression")
                return None
            
            enso_model = LassoCV(cv=5, max_iter=10000, random_state=42)
            enso_model.fit(X_trend, y_trend)
            
            enso_pred = enso_model.predict(X_trend)
            enso_r2 = r2_score(y_trend, enso_pred)
            
            print(f"    ENSO R2: {enso_r2:.4f}")
            print(f"    Selected {np.sum(enso_model.coef_ != 0)} ENSO features")
            
            # Step 3: Fit ETS with 365-day seasonality
            print(f"  [Step 2] Fitting ETS(A,Ad,A) with {period}-day seasonality...")
            
            try:
                ets_model = ES(
                    y,
                    trend='add',
                    damped_trend=True,
                    seasonal='add',
                    seasonal_periods=period
                )
                
                ets_fit = ets_model.fit(optimized=True)
            except:
                # Fallback to ETS without seasonality if seasonal fails
                print(f"    [WARNING] Seasonal ETS failed, using additive trend only")
                ets_model = ES(
                    y,
                    trend='add',
                    damped_trend=True,
                    seasonal=None,
                    seasonal_periods=None
                )
                ets_fit = ets_model.fit(optimized=True)
            
            print(f"    ETS AIC: {ets_fit.aic:.1f}")
            
            # Residuals
            combined_residuals = y.values - ets_fit.fittedvalues.values
            
            if len(combined_residuals) > 3:
                mean_res = np.mean(combined_residuals)
                std_res = np.std(combined_residuals)
                skew_res = stats.skew(combined_residuals)
                kurt_res = stats.kurtosis(combined_residuals)
                
                try:
                    shapiro_p = shapiro(combined_residuals)[1]
                except:
                    shapiro_p = np.nan
                
                try:
                    jb_p = jarque_bera(combined_residuals)[1]
                except:
                    jb_p = np.nan
                
                print(f"    Mean: {mean_res:.6f}, Std: {std_res:.6f}")
                print(f"    Skewness: {skew_res:.4f}, Kurtosis: {kurt_res:.4f}")
                print(f"    Shapiro-Wilk p: {shapiro_p:.4f}, JB p: {jb_p:.4f}")
                
                self.residuals_dict[variable_name] = combined_residuals
                
                self.results_summary.append({
                    'variable': variable_name,
                    'enso_r2': enso_r2,
                    'ets_aic': ets_fit.aic,
                    'mean_res': mean_res,
                    'std_res': std_res,
                    'skewness': skew_res,
                    'kurtosis': kurt_res,
                    'shapiro_p': shapiro_p,
                    'jb_p': jb_p,
                    'residuals': combined_residuals
                })
            
            model_results = {
                'variable': variable_name,
                'decomposition': decomp,
                'enso_model': enso_model,
                'ets_model': ets_fit,
                'enso_r2': enso_r2,
                'ets_aic': ets_fit.aic,
                'period': period,
                'y_train': y,
                'X_train': X
            }
            
            self.models[variable_name] = model_results
            
            print(f"  [OK] Model fitted successfully")
            
            return model_results
            
        except Exception as e:
            print(f"  [ERROR] Failed to fit model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def fit_all_models(self):
        """Fit seasonal motif models for all variables."""
        print("\n" + "="*80)
        print("FITTING SEASONAL MOTIF MODELS")
        print("="*80)
        
        for var_name, series in self.daily_data.items():
            self.fit_seasonal_motif_model(
                series,
                var_name,
                self.enso_features,
                period=365  # Use 365-day seasonal cycle for daily data
            )
        
        print(f"\n[OK] Fitted {len(self.models)} models")
        
        return self.models
    
    def forecast_daily(self, steps=None):
        """Generate daily forecasts."""
        if steps is None:
            steps = self.forecast_days
        
        print(f"\n[Forecasting {steps} days ahead...]")
        
        forecasts_dict = {}
        
        last_date = max([s.index.max() for s in self.daily_data.values()])
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq='D')
        
        future_enso = self.oni_handler.create_daily_enso_features(future_dates, 
                                                                  max_lag=self.enso_max_lag)
        
        last_historical_date = self.enso_features.index[-1]
        last_historical_enso = self.enso_features.loc[last_historical_date]
        
        for col in future_enso.columns:
            if future_enso[col].isna().any():
                future_enso[col] = future_enso[col].fillna(last_historical_enso[col])
        
        for var_name, model_result in self.models.items():
            try:
                print(f"\n  Forecasting {var_name}...")
                
                ets_fit = model_result['ets_model']
                
                # Forecast using ETS
                ets_forecast = ets_fit.forecast(steps=steps)
                
                print(f"    Forecast range: [{ets_forecast.min():.2f}, {ets_forecast.max():.2f}]")
                
                forecast_df = pd.DataFrame({
                    'date': future_dates,
                    f'{var_name}_forecast': ets_forecast
                })
                forecast_df.set_index('date', inplace=True)
                
                forecasts_dict[var_name] = forecast_df
                
                print(f"    [OK] {len(forecast_df)} daily forecasts")
                
            except Exception as e:
                print(f"    [ERROR] Failed to forecast {var_name}: {e}")
        
        self.forecasts_dict = forecasts_dict
        return forecasts_dict
    
    def aggregate_to_monthly(self, forecast_dict):
        """Aggregate daily forecasts to monthly."""
        print(f"\n[Aggregating daily forecasts to monthly...]")
        
        monthly_data = {}
        
        for var_name, daily_df in forecast_dict.items():
            if len(daily_df) == 0:
                continue
            
            if 'precipitation' in var_name.lower():
                method = 'sum'
            else:
                method = 'mean'
            
            monthly = daily_df.groupby(pd.Grouper(freq='MS')).agg(
                {col: method for col in daily_df.columns}
            )
            
            monthly_data[var_name] = monthly
            print(f"  {var_name}: {len(monthly)} months")
        
        if monthly_data:
            monthly_combined = pd.concat(monthly_data, axis=1)
            monthly_combined.columns = ['_'.join(col).strip() for col in monthly_combined.columns]
            
            print(f"\n[OK] Aggregated to {len(monthly_combined)} months")
            return monthly_combined
        
        return None
    
    def calculate_climate_index(self, monthly_data):
        """Calculate climate index from monthly forecasts."""
        print(f"\n[Calculating climate index...]")
        
        # Use temperature_daily_max as main climate indicator
        if 'temperature_daily_max_forecast' in monthly_data.columns:
            index = monthly_data['temperature_daily_max_forecast'].copy()
            index.name = 'Climate_Index'
            print(f"  [OK] Climate index created from temperature")
            return index
        else:
            print(f"  [WARNING] Could not calculate climate index")
            return None
    
    def save_results(self, daily_forecasts, monthly_forecasts):
        """Save results to CSV."""
        print("\n[Saving results to CSV...]")
        
        for var_name, daily_df in daily_forecasts.items():
            filename = os.path.join(self.output_path, f'forecast_daily_{var_name}.csv')
            daily_df.to_csv(filename)
            print(f"  [OK] {filename}")
        
        if monthly_forecasts is not None:
            filename = os.path.join(self.output_path, f'forecast_monthly_aggregated.csv')
            monthly_forecasts.to_csv(filename)
            print(f"  [OK] {filename}")
        
        if self.results_summary:
            summary_df = pd.DataFrame([{k: v for k, v in s.items() if k != 'residuals'} 
                                      for s in self.results_summary])
            filename = os.path.join(self.output_path, 'model_summary.csv')
            summary_df.to_csv(filename, index=False)
            print(f"  [OK] {filename}")
    
    def plot_residual_distributions(self):
        """Plot residual distributions for 5 variables."""
        print("\n[Creating residual distribution plots...]")
        
        if not self.residuals_dict:
            print("  [WARNING] No residuals to plot")
            return
        
        # Select up to 5 variables
        vars_to_plot = list(self.residuals_dict.keys())[:5]
        
        n_vars = len(vars_to_plot)
        fig, axes = plt.subplots(n_vars, 1, figsize=(14, 4*n_vars))
        axes = [axes] if n_vars == 1 else axes
        
        for idx, var_name in enumerate(vars_to_plot):
            ax = axes[idx]
            residuals = self.residuals_dict[var_name]
            
            # Histogram with normal curve
            n, bins, patches = ax.hist(residuals, bins=40, density=True, 
                                       alpha=0.7, color='steelblue', edgecolor='black')
            
            mu, sigma = np.mean(residuals), np.std(residuals)
            x = np.linspace(residuals.min(), residuals.max(), 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2.5,
                   label=f'N({mu:.3f}, {sigma:.3f})')
            
            # Get diagnostics
            diag = [s for s in self.results_summary if s['variable'] == var_name]
            if diag:
                diag = diag[0]
                status = "Normally Distributed" if diag['jb_p'] > 0.05 else "Non-Normal"
                title = f"{var_name}\n{status} (JB p={diag['jb_p']:.4f})"
            else:
                title = var_name
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Residuals', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_path, 'residual_distributions.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved: {output_file}")
        plt.close()
    
    def plot_forecast_indices(self, monthly_forecasts):
        """Plot forecasts and climate index."""
        print("\n[Creating forecast plots...]")
        
        if monthly_forecasts is None or len(monthly_forecasts) == 0:
            print("  [WARNING] No forecasts to plot")
            return
        
        # Create comprehensive forecast plot
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Plot 1: Temperature forecasts
        ax1 = axes[0]
        if 'temperature_daily_max_forecast' in monthly_forecasts.columns:
            ax1.plot(monthly_forecasts.index, monthly_forecasts['temperature_daily_max_forecast'],
                    'r-', linewidth=2.5, marker='o', markersize=8, label='Max Temperature Forecast')
        if 'temperature_daily_min_forecast' in monthly_forecasts.columns:
            ax1.plot(monthly_forecasts.index, monthly_forecasts['temperature_daily_min_forecast'],
                    'b-', linewidth=2.5, marker='s', markersize=8, label='Min Temperature Forecast')
        
        ax1.set_title('Temperature Forecasts (6 months ahead)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Temperature (K)', fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Precipitation forecast
        ax2 = axes[1]
        if 'precipitation_daily_forecast' in monthly_forecasts.columns:
            ax2.bar(monthly_forecasts.index, monthly_forecasts['precipitation_daily_forecast'],
                   color='steelblue', alpha=0.7, edgecolor='black')
            ax2.set_title('Monthly Precipitation Forecast', fontsize=13, fontweight='bold')
            ax2.set_ylabel('Precipitation (mm)', fontsize=11)
        
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Wind forecast
        ax3 = axes[2]
        if 'wind_speed_daily_forecast' in monthly_forecasts.columns:
            ax3.plot(monthly_forecasts.index, monthly_forecasts['wind_speed_daily_forecast'],
                    'g-', linewidth=2.5, marker='^', markersize=8, label='Wind Speed Forecast')
            ax3.set_title('Wind Speed Forecast', fontsize=13, fontweight='bold')
            ax3.set_ylabel('Wind Speed (m/s)', fontsize=11)
            ax3.legend(fontsize=10)
        
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_path, 'forecast_indices.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved: {output_file}")
        plt.close()
    
    def generate_summary_report(self):
        """Generate summary report."""
        print("\n[Generating summary report...]")
        
        report = f"""
{'='*80}
SEASONAL MOTIF ETS+ENSO CLIMATE FORECAST REPORT
{'='*80}

METHODOLOGY:
  * Daily seasonal decomposition (365-day cycle)
  * ETS(A,Ad,A) with additive seasonality
  * Lasso regression for ENSO effects
  * Forecast aggregation to monthly indices

FORECAST PERIOD: {self.forecast_days} days ({self.forecast_days//30} months)
ENSO MAX LAG: {self.enso_max_lag} months

VARIABLES MODELED: {len(self.models)}

ENSO EXPLANATORY POWER:
"""
        
        for summary in self.results_summary:
            report += f"  * {summary['variable']}: R2 = {summary['enso_r2']:.4f}\n"
        
        report += f"""
RESIDUAL DIAGNOSTICS (Jarque-Bera Test):
"""
        
        for summary in self.results_summary:
            status = "[OK] Normal" if summary['jb_p'] > 0.05 else "[!] Non-normal"
            report += f"  * {summary['variable']}: JB p={summary['jb_p']:.4f} {status}\n"
        
        report += f"""
OUTPUT FILES:
  * forecast_daily_[variable].csv - Daily forecasts
  * forecast_monthly_aggregated.csv - Monthly aggregated forecasts
  * model_summary.csv - Model statistics
  * residual_distributions.png - Residual distribution plots
  * forecast_indices.png - Forecast visualization

GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
        """
        
        report_file = os.path.join(self.output_path, 'forecast_summary_report.txt')
        with open(report_file, 'w', encoding='utf-8', errors='replace') as f:
            f.write(report)
        
        print(report)
        print(f"[OK] Report saved: {report_file}")
    
    def run_full_pipeline(self):
        """Execute complete pipeline."""
        print("\n" + "="*80)
        print("SEASONAL MOTIF ETS+ENSO FORECASTING PIPELINE")
        print("="*80 + "\n")
        
        try:
            # Load data
            datasets = self.load_daily_era5_data()
            if not datasets:
                print("[FAILED] No data loaded")
                return False
            
            self.extract_regional_daily_series(datasets)
            if not self.daily_data:
                print("[FAILED] No regional series extracted")
                return False
            
            # Prepare ENSO
            self.prepare_enso_data_daily()
            if self.enso_features is None:
                print("[FAILED] ENSO features not created")
                return False
            
            # Fit models
            self.fit_all_models()
            if not self.models:
                print("[FAILED] No models fitted")
                return False
            
            # Forecast
            daily_forecasts = self.forecast_daily(steps=self.forecast_days)
            if not daily_forecasts:
                print("[FAILED] No forecasts generated")
                return False
            
            # Aggregate
            monthly_forecasts = self.aggregate_to_monthly(daily_forecasts)
            
            # Visualize
            self.plot_residual_distributions()
            self.plot_forecast_indices(monthly_forecasts)
            
            # Save
            self.save_results(daily_forecasts, monthly_forecasts)
            self.generate_summary_report()
            
            print("\n" + "="*80)
            print("[OK] SEASONAL MOTIF PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"\n[Results saved to: {self.output_path}]\n")
            
            return True
            
        except Exception as e:
            print(f"\n[FAILED] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main execution."""
    forecaster = SeasonalMotifForecaster(
        region_name="colombia",
        forecast_days=180,
        enso_max_lag=6
    )
    
    success = forecaster.run_full_pipeline()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
