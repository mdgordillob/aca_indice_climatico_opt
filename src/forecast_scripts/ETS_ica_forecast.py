"""
ETS (Error Trend Seasonality) Forecasting for Monthly Climate Indices - Cundinamarca-Bogotá Region

Enhanced with:
- Time series cross-validation
- Automated ETS model selection (additive, multiplicative, etc.)
- Stationarity testing with ADF and differencing
- Improved x-axis formatting

This script performs time series forecasting on monthly climate anomalies
(temperature, precipitation, drought, wind) for the Cundinamarca-Bogotá region
using statsmodels ETS models with proper validation and visualization.

Works with monthly NetCDF files (temperature, wind) and CSV files (precipitation, drought).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.data_loading import load_monthly_data, extract_regional_series
from common.climate_index import calculate_climate_index
from common.stationarity import adf_test, make_stationary, reconstruct_from_differences

# ETS and statistical models
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MonthlyClimateForecasterETS:
    """
    Handles forecasting of monthly climate anomalies using ETS models.
    """
    
    def __init__(self, region_name="cundinamarca_bogota", forecast_months=12):
        """
        Initialize the forecaster.
        
        Args:
            region_name (str): Region identifier
            forecast_months (int): Number of months to forecast
        """
        self.region_name = region_name
        self.forecast_months = forecast_months
        self.data_wide = None
        self.models = {}
        self.forecasts_dict = {}
        self.results_summary = []
        self.cv_results = []
        self.differencing_info = {}
        
        # Set up paths - relative to project root
        self.base_path = os.path.join(".", "data", "processed", f"anomalias_{region_name}")
        self.output_path = os.path.join(".", "articles", "graficas", f"forecast_ets_{region_name}")
        
        os.makedirs(self.output_path, exist_ok=True)
        print(f"Base path: {os.path.abspath(self.base_path)}")
        print(f"Output path: {os.path.abspath(self.output_path)}")
    
    def adf_test(self, series, variable_name):
        """
        Perform Augmented Dickey-Fuller test for stationarity.

        Args:
            series (pd.Series): Time series data
            variable_name (str): Name of the variable

        Returns:
            dict: ADF test results
        """
        return adf_test(series, variable_name)

    def make_stationary(self, series, variable_name, max_diff=2):
        """
        Make series stationary through differencing if needed.

        Args:
            series (pd.Series): Time series data
            variable_name (str): Name of the variable
            max_diff (int): Maximum number of differences to apply

        Returns:
            tuple: (stationary_series, diff_order, adf_results)
        """
        return make_stationary(series, variable_name, max_diff, differencing_info=self.differencing_info)

    def reconstruct_from_differences(self, differenced_forecast, original_series, diff_order):
        """
        Reconstruct original scale forecast from differenced forecasts.

        Args:
            differenced_forecast (pd.Series or np.ndarray): Forecast in differenced form
            original_series (pd.Series): Original series before differencing
            diff_order (int): Number of differences applied

        Returns:
            pd.Series or np.ndarray: Reconstructed forecast
        """
        return reconstruct_from_differences(differenced_forecast, original_series, diff_order)

    def load_monthly_data(self):
        """
        Load monthly NetCDF files (temperature, wind) and CSV files (precipitation, drought).

        Returns:
            dict: Dictionary containing processed DataFrames
        """
        return load_monthly_data(self.base_path)

    def extract_regional_series(self, datasets):
        """
        Extract spatial mean time series from datasets and combine into single DataFrame.
        Keep only anomaly variables: T90, T10, and anomaly columns from CSVs.
        
        Args:
            datasets (dict): Dictionary of xarray Datasets or DataFrames
            
        Returns:
            pd.DataFrame: Time series with all variables
        """
        self.data_wide = extract_regional_series(datasets)
        if self.data_wide is not None:
            print(f"\n[OK] Data loaded (already in anomaly form from source files)")
        return self.data_wide

    def time_series_cv(self, series, variable_name, n_splits=5, model_configs=None):
        """
        Perform time series cross-validation to find best ETS model.
        
        Args:
            series (pd.Series): Time series data
            variable_name (str): Name of the variable
            n_splits (int): Number of CV splits
            model_configs (list): List of ETS configurations to test
            
        Returns:
            dict: Best model configuration and CV results
        """
        print(f"\n  Running cross-validation for {variable_name}...")
        
        if model_configs is None:
            # Test different ETS configurations
            model_configs = [
                {'error': 'add', 'trend': 'add', 'seasonal': 'add', 'damped': True},
                {'error': 'add', 'trend': 'add', 'seasonal': 'add', 'damped': False},
                {'error': 'add', 'trend': 'add', 'seasonal': 'mul', 'damped': True},
                {'error': 'mul', 'trend': 'add', 'seasonal': 'add', 'damped': True},
                {'error': 'mul', 'trend': 'mul', 'seasonal': 'mul', 'damped': True},
                {'error': 'add', 'trend': 'add', 'seasonal': None, 'damped': True},
                {'error': 'add', 'trend': None, 'seasonal': 'add', 'damped': False},
            ]
        
        test_size = len(series) // (n_splits + 1)
        min_train_size = max(48, len(series) - n_splits * test_size)
        
        cv_results = []
        
        for config_idx, config in enumerate(model_configs):
            config_errors = []
            
            for split in range(n_splits):
                train_end = len(series) - (n_splits - split) * test_size
                train_data = series[:train_end]
                test_data = series[train_end:train_end + test_size]
                
                if len(train_data) < min_train_size or len(test_data) == 0:
                    continue
                
                try:
                    # Fit model
                    model = ETSModel(
                        train_data,
                        error=config['error'],
                        trend=config['trend'],
                        seasonal=config['seasonal'],
                        seasonal_periods=12 if config['seasonal'] else None,
                        damped_trend=config.get('damped', False)
                    )
                    fitted = model.fit(disp=False)
                    
                    # Forecast
                    forecast = fitted.forecast(steps=len(test_data))
                    
                    # Calculate error
                    mape = np.mean(np.abs((test_data.values - forecast.values) / (test_data.values + 1e-10))) * 100
                    rmse = np.sqrt(np.mean((test_data.values - forecast.values) ** 2))
                    
                    config_errors.append({'mape': mape, 'rmse': rmse})
                    
                except Exception as e:
                    continue
            
            if config_errors:
                avg_mape = np.mean([e['mape'] for e in config_errors])
                avg_rmse = np.mean([e['rmse'] for e in config_errors])
                
                cv_results.append({
                    'config': config,
                    'avg_mape': avg_mape,
                    'avg_rmse': avg_rmse,
                    'n_successful_splits': len(config_errors)
                })
        
        if cv_results:
            # Select best model by RMSE
            best_config = min(cv_results, key=lambda x: x['avg_rmse'])
            print(f"    Best config: {best_config['config']}")
            print(f"    Avg RMSE: {best_config['avg_rmse']:.4f}, Avg MAPE: {best_config['avg_mape']:.2f}%")
            
            return best_config, cv_results
        else:
            print(f"    [WARNING] All CV attempts failed, using default config")
            default_config = {
                'config': {'error': 'add', 'trend': 'add', 'seasonal': 'add', 'damped': True},
                'avg_mape': np.nan,
                'avg_rmse': np.nan,
                'n_successful_splits': 0
            }
            return default_config, []
    
    def fit_ets_model(self, series, variable_name, use_cv=True):
        """
        Fit an ETS model to a single time series with stationarity testing and CV.
        
        Args:
            series (pd.Series): Time series data
            variable_name (str): Name of the variable
            use_cv (bool): Whether to use cross-validation
            
        Returns:
            tuple: (fitted_model, forecast_df, model_info)
        """
        print(f"\n  Fitting ETS model for {variable_name}...")
        
        # Test for stationarity
        stationary_series, diff_order, adf_results = self.make_stationary(series, variable_name)
        
        # Use cross-validation to find best config
        if use_cv and len(stationary_series) >= 60:
            best_config, cv_results = self.time_series_cv(stationary_series, variable_name)
            config = best_config['config']
        else:
            config = {'error': 'add', 'trend': 'add', 'seasonal': 'add', 'damped': True}
            cv_results = []
        
        try:
            # Fit model on stationary data
            model = ETSModel(
                stationary_series,
                error=config['error'],
                trend=config['trend'],
                seasonal=config['seasonal'],
                seasonal_periods=12 if config['seasonal'] else None,
                damped_trend=config.get('damped', False)
            )
            
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.forecast(steps=self.forecast_months)
            
            # Reconstruct to original scale if differenced
            if diff_order > 0:
                forecast = self.reconstruct_from_differences(
                    forecast, 
                    self.differencing_info[variable_name]['original_series'],
                    diff_order
                )
            
            # Generate confidence intervals
            simulations = fitted_model.simulate(
                nsimulations=self.forecast_months,
                repetitions=1000,
                anchor='end'
            )
            
            # Reconstruct simulations if needed
            if diff_order > 0:
                reconstructed_sims = []
                original_last_val = self.differencing_info[variable_name]['original_series'].iloc[-1]
                
                for i in range(simulations.shape[1]):
                    sim_values = simulations[:, i]
                    # Integrate the differenced simulation
                    recon_sim = np.cumsum(sim_values) + original_last_val
                    reconstructed_sims.append(recon_sim)
                
                simulations = np.array(reconstructed_sims).T
            
            lower = np.percentile(simulations, 5, axis=1)
            upper = np.percentile(simulations, 95, axis=1)
            
            # Generate forecast index
            last_date = series.index[-1]
            forecast_index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=self.forecast_months,
                freq='MS'
            )
            
            forecast_df = pd.DataFrame({
                'forecast': forecast.values if hasattr(forecast, 'values') else forecast,
                'lower': lower,
                'upper': upper
            }, index=forecast_index)
            
            model_info = {
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'error': config['error'],
                'trend': config['trend'],
                'seasonal': config['seasonal'],
                'damped': config.get('damped', False),
                'diff_order': diff_order,
                'cv_results': cv_results
            }
            
            print(f"    [OK] AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}, Diff: {diff_order}")
            return fitted_model, forecast_df, model_info
            
        except Exception as e:
            print(f"    [ERROR] Model fitting failed: {e}")
            return None, None, None
    
    def fit_all_ets_models(self):
        """
        Fit ETS models to all variables in the dataset.
        """
        print("\n[Fitting ETS models to all variables with CV and stationarity testing...]")
        print("  This may take several minutes...\n")
        
        for col in self.data_wide.columns:
            series = self.data_wide[col].dropna()
            
            if len(series) < 24:
                print(f"  [SKIP] {col}: Insufficient data ({len(series)} points)")
                continue
            
            model, forecast_df, model_info = self.fit_ets_model(series, col, use_cv=True)
            
            if model is not None:
                self.models[col] = model
                self.forecasts_dict[col] = forecast_df
                self.results_summary.append({
                    'variable': col,
                    'aic': model_info['aic'],
                    'bic': model_info['bic'],
                    'error': model_info['error'],
                    'trend': model_info['trend'],
                    'seasonal': model_info['seasonal'],
                    'damped': model_info['damped'],
                    'diff_order': model_info['diff_order']
                })
                
                if model_info['cv_results']:
                    self.cv_results.extend([
                        {**r, 'variable': col} for r in model_info['cv_results']
                    ])
        
        print(f"\n[OK] Successfully fitted {len(self.models)} ETS models!")
    
    def calculate_climate_index(self, data):
        """
        Calculate Actuarial Climate Index (ICA) from anomaly variables.
        Formula: ICA = (T90 - T10 + W_std + P_std + D_std) / 5

        Args:
            data (pd.DataFrame): DataFrame with anomaly columns

        Returns:
            pd.Series: Climate index values
        """
        return calculate_climate_index(data)

    def generate_forecasts(self):
        """
        Generate forecasts by compiling individual ETS model outputs.
        """
        print("\n[Compiling forecasts...]")
        
        forecast_dict = {}
        upper_dict = {}
        lower_dict = {}
        
        for col, forecast_df in self.forecasts_dict.items():
            forecast_dict[col] = forecast_df['forecast']
            upper_dict[col] = forecast_df['upper']
            lower_dict[col] = forecast_df['lower']
        
        forecasts = pd.DataFrame(forecast_dict)
        upper = pd.DataFrame(upper_dict)
        lower = pd.DataFrame(lower_dict)
        
        # Calculate climate index for historical data
        hist_index = self.calculate_climate_index(self.data_wide)
        
        # Calculate climate index for forecasts
        forecast_index = self.calculate_climate_index(forecasts)
        upper_index = self.calculate_climate_index(upper)
        lower_index = self.calculate_climate_index(lower)
        
        # Add index as new columns
        forecasts['Climate_Index'] = forecast_index.values
        upper['Climate_Index'] = upper_index.values
        lower['Climate_Index'] = lower_index.values
        
        # Also add to data_wide for visualization
        self.data_wide['Climate_Index'] = hist_index
        
        print(f"[OK] Compiled forecasts shape: {forecasts.shape}")
        print(f"  Variables: {list(forecasts.columns)}")
        
        return forecasts, upper, lower
    
    def evaluate_model(self):
        """
        Create evaluation summary of all fitted models.
        """
        print("\n[Model Performance Evaluation:]")
        print("-" * 80)
        
        if not self.results_summary:
            print("No models to evaluate!")
            return
        
        results_df = pd.DataFrame(self.results_summary)
        results_df = results_df.sort_values('aic')
        
        print("\nETS Model Summary (sorted by AIC):")
        print("-" * 80)
        print(results_df.to_string(index=False))
        
        print(f"\nAverage AIC: {results_df['aic'].mean():.2f}")
        print(f"Average BIC: {results_df['bic'].mean():.2f}")
        
        # Print stationarity summary
        print("\nStationarity Analysis:")
        print("-" * 80)
        for var, info in self.differencing_info.items():
            print(f"{var}: Differencing order = {info['diff_order']}")
        
        return results_df
    
    def visualize_forecasts(self, forecasts, upper, lower):
        """
        Create comprehensive visualizations of forecasts with improved x-axis.
        
        Args:
            forecasts (pd.DataFrame): Point forecasts
            upper (pd.DataFrame): Upper confidence bounds
            lower (pd.DataFrame): Lower confidence bounds
        """
        print("\n[Creating visualizations...]")
        
        # Combine historical and forecast data
        combined = pd.concat([self.data_wide, forecasts], axis=0)
        
        # Apply 5-year smoothing for visualization
        combined_smooth = combined.copy()
        for col in combined.columns:
            combined_smooth[col] = combined[col].rolling(window=60, center=False).mean()
        
        n_vars = len(forecasts.columns)
        n_cols = 2
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        zoom_start = pd.Timestamp('1961-01-01')
        zoom_end = pd.Timestamp('2026-12-31')
        
        for idx, col in enumerate(forecasts.columns):
            ax = axes[idx]
            
            hist_data = self.data_wide[col]
            forecast_data = forecasts[col]
            hist_smooth_data = combined_smooth[col][:len(self.data_wide)]
            forecast_smooth_data = combined_smooth[col][len(self.data_wide):]
            upper_data = upper[col]
            lower_data = lower[col]
            
            # Plot
            ax.plot(hist_data.index, hist_data.values, 'b-', linewidth=0.5, alpha=0.3, label='Historical (raw)')
            ax.plot(hist_smooth_data.index, hist_smooth_data.values, 'b-', linewidth=2.5, alpha=0.9, label='Historical (5-yr avg)')
            ax.plot(forecast_data.index, forecast_data.values, 'r-', linewidth=0.5, alpha=0.3, label='ETS Forecast (raw)')
            ax.plot(forecast_smooth_data.index, forecast_smooth_data.values, 'r-', linewidth=2.5, alpha=0.9, label='ETS Forecast (5-yr avg)')
            ax.fill_between(lower_data.index, lower_data.values, upper_data.values, alpha=0.2, color='red', label='90% CI')
            
            ax.set_xlim(zoom_start, zoom_end)
            ax.set_title(f'{col}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Anomaly Value')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Improved x-axis: show every 5 years
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(mdates.YearLocator(1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        for idx in range(len(forecasts.columns), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        output_file = os.path.join(self.output_path, 'forecasts_ets_all_variables.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_file}")
        plt.close()
    
    def visualize_climate_index(self, forecasts, upper, lower):
        """
        Create a dedicated visualization for the Climate Index.
        """
        print("\n[Creating Climate Index visualization...]")
        
        if 'Climate_Index' not in forecasts.columns:
            print("[WARNING] Climate Index not found in forecasts")
            return
        
        hist_index = self.data_wide['Climate_Index']
        forecast_index = forecasts['Climate_Index']
        combined_index = pd.concat([hist_index, forecast_index], axis=0)
        combined_index_smooth = combined_index.rolling(window=60, center=False).mean()
        
        hist_smooth = combined_index_smooth[:len(hist_index)]
        forecast_smooth = combined_index_smooth[len(hist_index):]
        upper_index = upper['Climate_Index']
        lower_index = lower['Climate_Index']
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        zoom_start = pd.Timestamp('1961-01-01')
        zoom_end = pd.Timestamp('2026-12-31')
        
        ax.plot(hist_index.index, hist_index.values, 'b-', linewidth=0.3, alpha=0.2, label='Historical ICA (raw)')
        ax.plot(forecast_index.index, forecast_index.values, 'r-', linewidth=0.3, alpha=0.2, label='ETS Forecast ICA (raw)')
        ax.plot(hist_smooth.index, hist_smooth.values, 'b-', linewidth=3.0, alpha=0.9, label='Historical ICA (5-yr avg)')
        ax.plot(forecast_smooth.index, forecast_smooth.values, 'r-', linewidth=3.0, alpha=0.9, label='ETS Forecast ICA (5-yr avg)')
        ax.fill_between(lower_index.index, lower_index.values, upper_index.values, alpha=0.2, color='red', label='90% CI')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlim(zoom_start, zoom_end)
        ax.set_title('Actuarial Climate Index (ICA) - ETS Forecast with Cross-Validation', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Climate Index Value', fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Improved x-axis: every 5 years
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        output_file = os.path.join(self.output_path, 'forecast_ets_climate_index.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_file}")
        plt.close()
    
    def save_forecasts_to_csv(self, forecasts, upper, lower):
        """
        Save forecast results to CSV files.
        """
        print("\n[Saving forecasts to CSV...]")
        
        forecast_file = os.path.join(self.output_path, f'forecast_ets_{self.region_name}_point.csv')
        forecasts.to_csv(forecast_file)
        print(f"[OK] Saved: {forecast_file}")
        
        upper_file = os.path.join(self.output_path, f'forecast_ets_{self.region_name}_upper_90ci.csv')
        upper.to_csv(upper_file)
        print(f"[OK] Saved: {upper_file}")
        
        lower_file = os.path.join(self.output_path, f'forecast_ets_{self.region_name}_lower_90ci.csv')
        lower.to_csv(lower_file)
        print(f"[OK] Saved: {lower_file}")
        
        combined = pd.concat([self.data_wide, forecasts], axis=0)
        combined_file = os.path.join(self.output_path, f'forecast_ets_{self.region_name}_combined.csv')
        combined.to_csv(combined_file)
        print(f"[OK] Saved: {combined_file}")
        
        combined_smooth = combined.copy()
        for col in combined.columns:
            combined_smooth[col] = combined[col].rolling(window=60, center=False).mean()
        
        combined_smooth_file = os.path.join(self.output_path, f'forecast_ets_{self.region_name}_combined_5yr_avg.csv')
        combined_smooth.to_csv(combined_smooth_file)
        print(f"[OK] Saved: {combined_smooth_file}")
        
        if self.results_summary:
            results_df = pd.DataFrame(self.results_summary)
            results_file = os.path.join(self.output_path, 'ets_model_summary.csv')
            results_df.to_csv(results_file, index=False)
            print(f"[OK] Saved: {results_file}")
        
        if self.cv_results:
            cv_df = pd.DataFrame(self.cv_results)
            cv_file = os.path.join(self.output_path, 'ets_cv_results.csv')
            cv_df.to_csv(cv_file, index=False)
            print(f"[OK] Saved: {cv_file}")
        
        # Save stationarity test results
        stationarity_results = []
        for var, info in self.differencing_info.items():
            for adf_result in info['adf_results']:
                stationarity_results.append(adf_result)
        
        if stationarity_results:
            stationarity_df = pd.DataFrame(stationarity_results)
            stationarity_file = os.path.join(self.output_path, 'stationarity_tests.csv')
            stationarity_df.to_csv(stationarity_file, index=False)
            print(f"[OK] Saved: {stationarity_file}")
    
    def generate_summary_report(self, forecasts):
        """
        Generate a text summary of the forecasting results.
        """
        print("\n[Generating summary report...]")
        
        if len(forecasts) == 0:
            report = f"""
{'='*80}
ETS CLIMATE ANOMALY FORECAST REPORT - {self.region_name.upper()}
{'='*80}

WARNING: No forecasts were generated.

{'='*80}
            """
        else:
            forecast_start = forecasts.index.min()
            forecast_end = forecasts.index.max()
            
            forecast_start_str = forecast_start.date() if isinstance(forecast_start, pd.Timestamp) else str(forecast_start)
            forecast_end_str = forecast_end.date() if isinstance(forecast_end, pd.Timestamp) else str(forecast_end)
            
            avg_aic = pd.DataFrame(self.results_summary)['aic'].mean() if len(self.results_summary) > 0 else None
            avg_bic = pd.DataFrame(self.results_summary)['bic'].mean() if len(self.results_summary) > 0 else None
            
            # Format AIC/BIC strings
            aic_str = f"{avg_aic:.2f}" if avg_aic is not None else "N/A"
            bic_str = f"{avg_bic:.2f}" if avg_bic is not None else "N/A"
            
            # Count stationary vs non-stationary
            stationary_count = sum(1 for info in self.differencing_info.values() if info['diff_order'] == 0)
            differenced_count = len(self.differencing_info) - stationary_count
            
            report = f"""
{'='*80}
ENHANCED ETS CLIMATE ANOMALY FORECAST REPORT - {self.region_name.upper()}
{'='*80}

FORECAST CONFIGURATION:
  • Region: {self.region_name.replace('_', ' ').title()}
  • Forecast Period: {self.forecast_months} months
  • Forecast Date Range: {forecast_start_str} to {forecast_end_str}
  • Historical Data Range: {self.data_wide.index.min().date()} to {self.data_wide.index.max().date()}

MODEL ENHANCEMENTS:
  • Time Series Cross-Validation: Enabled (5 splits)
  • Stationarity Testing: ADF test with auto-differencing
  • Model Selection: Automated (additive, multiplicative, damped tested)
  • Confidence Interval: 90% (simulation-based)

STATIONARITY ANALYSIS:
  • Variables Already Stationary: {stationary_count}
  • Variables Requiring Differencing: {differenced_count}
  • Maximum Differencing Order: {max((info['diff_order'] for info in self.differencing_info.values()), default=0)}

MODEL INFORMATION:
  • Method: ETS (Error Trend Seasonality)
  • Models Fitted: {len(self.models)}
  • Total Variables: {len(forecasts.columns)}
  • Total Data Points (monthly): {len(self.data_wide)}

FORECAST VARIABLES:
{chr(10).join([f'  • {col}' for col in forecasts.columns])}

MODEL PERFORMANCE:
  • Average AIC: {aic_str}
  • Average BIC: {bic_str}
  • Cross-Validation Results: {len(self.cv_results)} configurations tested

OUTPUT FILES GENERATED:
  • forecast_ets_{self.region_name}_point.csv - Point forecasts
  • forecast_ets_{self.region_name}_upper_90ci.csv - Upper confidence bounds
  • forecast_ets_{self.region_name}_lower_90ci.csv - Lower confidence bounds
  • forecast_ets_{self.region_name}_combined.csv - Historical + Forecast (raw)
  • forecast_ets_{self.region_name}_combined_5yr_avg.csv - Historical + Forecast (smoothed)
  • ets_model_summary.csv - Model performance metrics
  • ets_cv_results.csv - Cross-validation results
  • stationarity_tests.csv - ADF test results
  • forecasts_ets_all_variables.png - All variables visualization
  • forecast_ets_climate_index.png - Climate Index visualization

RECOMMENDATIONS:
  1. Review cross-validation results to assess model reliability
  2. Check stationarity tests to understand data preprocessing
  3. Monitor confidence intervals for forecast uncertainty
  4. Validate forecasts against actual observations
  5. Update model quarterly with new data

GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
            """
        
        report_file = os.path.join(self.output_path, 'forecast_ets_summary_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"[OK] Saved: {report_file}")
    
    def run_full_pipeline(self):
        """
        Execute the complete enhanced ETS forecasting pipeline.
        """
        print("\n" + "="*80)
        print("ENHANCED ETS CLIMATE ANOMALY FORECASTING PIPELINE")
        print("With Cross-Validation, Stationarity Testing, and Model Selection")
        print("="*80 + "\n")
        
        try:
            datasets = self.load_monthly_data()
            
            if not datasets:
                print("[FAILED] No data files found. Cannot proceed.")
                return False
            
            self.extract_regional_series(datasets)
            
            if self.data_wide is None or len(self.data_wide) == 0:
                print("[FAILED] No time series data extracted. Cannot proceed.")
                return False
            
            self.fit_all_ets_models()
            forecasts, upper, lower = self.generate_forecasts()
            self.evaluate_model()
            self.visualize_forecasts(forecasts, upper, lower)
            self.visualize_climate_index(forecasts, upper, lower)
            self.save_forecasts_to_csv(forecasts, upper, lower)
            self.generate_summary_report(forecasts)
            
            print("\n" + "="*80)
            print("[OK] ENHANCED ETS PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"\n[Results saved to: {self.output_path}]\n")
            
            return True
            
        except Exception as e:
            print(f"\n[FAILED] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """
    Main execution function.
    """
    forecaster = MonthlyClimateForecasterETS(
        region_name="cundinamarca_bogota",
        forecast_months=12
    )
    
    success = forecaster.run_full_pipeline()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)