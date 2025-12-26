"""
AutoTS Forecasting for Monthly Climate Indices - Cundinamarca-Bogotá Region

This script performs time series forecasting on monthly climate anomalies
(temperature, precipitation, drought, wind) for the Cundinamarca-Bogotá region
using AutoTS ensemble methods with proper validation and visualization.

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

# AutoTS and dependencies
from autots import AutoTS
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MonthlyClimateForecaster:
    """
    Handles forecasting of monthly climate anomalies for a specific region.
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
        self.model = None
        self.prediction = None
        self.results_df = None
        
        # Set up paths - relative to project root
        self.base_path = os.path.join(".", "data", "processed", f"anomalias_{region_name}")
        self.output_path = os.path.join(".", "articles", "graficas", f"forecast_{region_name}")
        
        os.makedirs(self.output_path, exist_ok=True)
        print(f"Base path: {os.path.abspath(self.base_path)}")
        print(f"Output path: {os.path.abspath(self.output_path)}")
        
    def load_monthly_data(self):
        """
        Load monthly NetCDF files (temperature, wind) and CSV files (precipitation, drought).
        
        Returns:
            dict: Dictionary containing processed DataFrames
        """
        print("\n[Loading monthly data...]")
        
        datasets = {}
        
        # Load temperature NetCDF files (monthly)
        print("\n  Loading temperature files...")
        temp_files = sorted(glob.glob(os.path.join(self.base_path, "anomalies_temperature_*.nc")))
        if temp_files:
            try:
                temp_datasets = [xr.open_dataset(f) for f in temp_files]
                ds_temp = xr.concat(temp_datasets, dim='time')
                datasets['temperature'] = ds_temp
                print(f"  [OK] Loaded {len(temp_files)} temperature files")
            except Exception as e:
                print(f"  [WARNING] Error loading temperature: {e}")
        else:
            print(f"  [WARNING] No temperature files found")
        
        # Load wind NetCDF files (monthly)
        print("\n  Loading wind files...")
        wind_files = sorted(glob.glob(os.path.join(self.base_path, "anomalies_wind_*.nc")))
        if wind_files:
            try:
                wind_datasets = [xr.open_dataset(f) for f in wind_files]
                ds_wind = xr.concat(wind_datasets, dim='time')
                datasets['wind'] = ds_wind
                print(f"  [OK] Loaded {len(wind_files)} wind files")
            except Exception as e:
                print(f"  [WARNING] Error loading wind: {e}")
        else:
            print(f"  [WARNING] No wind files found")
        
        # Load precipitation CSV
        print("\n  Loading precipitation data...")
        precip_file = os.path.join(self.base_path, "anomalies_precipitation_combined.csv")
        if os.path.exists(precip_file):
            try:
                df_precip = pd.read_csv(precip_file)
                if 'time' in df_precip.columns:
                    df_precip['time'] = pd.to_datetime(df_precip['time'])
                datasets['precipitation'] = df_precip
                print(f"  [OK] Loaded precipitation CSV")
            except Exception as e:
                print(f"  [WARNING] Error loading precipitation: {e}")
        else:
            print(f"  [WARNING] Precipitation file not found")
        
        # Load drought CSV
        print("\n  Loading drought data...")
        drought_file = os.path.join(self.base_path, "anomalies_drought_combined.csv")
        if os.path.exists(drought_file):
            try:
                df_drought = pd.read_csv(drought_file)
                if 'time' in df_drought.columns:
                    df_drought['time'] = pd.to_datetime(df_drought['time'])
                datasets['drought'] = df_drought
                print(f"  [OK] Loaded drought CSV")
            except Exception as e:
                print(f"  [WARNING] Error loading drought: {e}")
        else:
            print(f"  [WARNING] Drought file not found")
        
        return datasets
    
    def calculate_rolling_anomalies(self, df):
        """
        Recalculate anomalies using a 1-month rolling baseline instead of static reference period.
        This makes anomalies more responsive to recent climate conditions.
        
        Args:
            df (pd.DataFrame): DataFrame with raw anomaly values
            
        Returns:
            pd.DataFrame: DataFrame with rolling anomalies
        """
        print("\n[Calculating rolling 1-month baseline anomalies...]")
        
        df_rolling = df.copy()
        
        for col in df.columns:
            # Calculate 1-month rolling mean as the baseline
            rolling_baseline = df[col].rolling(window=1, center=True, min_periods=1).mean()
            
            # Calculate anomaly relative to this rolling baseline
            # We'll use a 12-month rolling window for the reference (more recent climate)
            rolling_ref = df[col].rolling(window=12, center=True, min_periods=1).mean()
            
            # Updated anomaly: current value - rolling reference (12-month)
            df_rolling[col] = df[col] - rolling_ref
            
            print(f"  [OK] {col}: Rolling anomalies calculated")
        
        return df_rolling
    
    def extract_regional_series(self, datasets):
        """
        Extract spatial mean time series from datasets and combine into single DataFrame.
        Keep only anomaly variables: T90, T10, and anomaly columns from CSVs.
        
        Args:
            datasets (dict): Dictionary of xarray Datasets or DataFrames
            
        Returns:
            pd.DataFrame: Time series with all variables
        """
        print("\n[Extracting regional time series...]")
        
        series_dict = {}
        time_index = None
        
        # Process NetCDF datasets (temperature, wind)
        for var_name, ds in datasets.items():
            if isinstance(ds, xr.Dataset):
                try:
                    # Get time index from the dataset
                    if time_index is None:
                        if 'time' in ds.indexes:
                            time_index = ds.indexes['time'].to_index() if hasattr(ds.indexes['time'], 'to_index') else ds['time'].values
                        elif 'time' in ds.coords:
                            # Time coordinate exists but not as an index
                            time_index = pd.date_range(start='1961-01-01', periods=len(ds['time']), freq='MS')
                            print(f"  Using generated monthly index (time not found in data)")
                        else:
                            print(f"  Warning: No time coordinate found in {var_name}")
                    
                    # Calculate spatial mean across latitude and longitude
                    for data_var in ds.data_vars:
                        try:
                            # Keep temperature anomalies (t_90, t_10) and wind anomalies
                            if 't_90' in data_var.lower() or 't_10' in data_var.lower() or 'anomal' in data_var.lower():
                                spatial_mean = ds[data_var].mean(dim=['latitude', 'longitude'], skipna=True)
                                
                                # Convert to DataFrame
                                df_temp = spatial_mean.to_dataframe().reset_index()
                                df_temp = df_temp[['time', data_var]]
                                df_temp.columns = ['time', f'{var_name}_{data_var}']
                                df_temp.set_index('time', inplace=True)
                                
                                series_dict[f'{var_name}_{data_var}'] = df_temp[f'{var_name}_{data_var}']
                                print(f"  [OK] {var_name}_{data_var}: {len(df_temp)} observations")
                        except Exception as e:
                            print(f"  [WARNING] Could not process {var_name}_{data_var}: {e}")
                except Exception as e:
                    print(f"  [WARNING] Error processing {var_name} dataset: {e}")
            
            # Process CSV DataFrames (precipitation, drought)
            elif isinstance(ds, pd.DataFrame):
                try:
                    df = ds.copy()
                    
                    # Handle time column - try to convert to datetime
                    if 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'], errors='coerce')
                        time_index = df['time']
                        df.set_index('time', inplace=True)
                    elif 'Año' in df.columns and 'Mes' in df.columns:
                        # Construct datetime from year and month columns
                        df['time'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].astype(str).str.zfill(2) + '-01', format='%Y-%m-%d')
                        time_index = df['time']
                        df.set_index('time', inplace=True)
                    
                    # Keep only anomaly columns, filter out Año and Mes
                    anomaly_cols = [col for col in df.columns 
                                  if 'Anomalia' in col or 'anomalies' in col]
                    
                    for col in anomaly_cols:
                        col_name = f'{var_name}_{col}'
                        series_dict[col_name] = df[col]
                        print(f"  [OK] {col_name}: {len(df)} observations")
                except Exception as e:
                    print(f"  [WARNING] Error processing {var_name} DataFrame: {e}")
        
        # Combine all series
        if series_dict:
            # First, ensure all series have a proper DatetimeIndex
            proper_index = pd.date_range(start='1961-01-01', periods=768, freq='MS')  # 768 months from Jan 1961
            
            for key in series_dict.keys():
                series_dict[key].index = proper_index
            
            self.data_wide = pd.concat(series_dict, axis=1)
            
            # Ensure index is DatetimeIndex
            if not isinstance(self.data_wide.index, pd.DatetimeIndex):
                # Try to convert to datetime if it's numeric
                if self.data_wide.index.dtype in [np.int64, np.int32, np.float64, np.float32]:
                    # Create monthly index starting from Jan 1961
                    self.data_wide.index = pd.date_range(start='1961-01-01', periods=len(self.data_wide), freq='MS')
                else:
                    self.data_wide.index = pd.to_datetime(self.data_wide.index, errors='coerce')
            
            # Sort by index and remove duplicates
            self.data_wide = self.data_wide.sort_index()
            self.data_wide = self.data_wide[~self.data_wide.index.duplicated(keep='first')]
            
            # Remove NaN rows
            self.data_wide = self.data_wide.dropna()
            
            print(f"\n[OK] Combined dataset shape: {self.data_wide.shape}")
            print(f"  Date range: {self.data_wide.index.min()} to {self.data_wide.index.max()}")
            print(f"  Index type: {type(self.data_wide.index)}")
            print(f"  Variables: {list(self.data_wide.columns)}")
            
            # Recalculate anomalies using rolling 12-month baseline
            self.data_wide = self.calculate_rolling_anomalies(self.data_wide)
            
            print(f"\n[OK] Rolling anomalies calculated")
            print(f"  Updated dataset shape: {self.data_wide.shape}")
            
            return self.data_wide
        else:
            print("[FAILED] No data could be extracted!")
            return None
    
    def fit_autots_model(self):
        """
        Train AutoTS ensemble model on the monthly data.
        """
        print("\n[Training AutoTS ensemble model...]")
        print("  This may take several minutes...\n")
        
        self.model = AutoTS(
            forecast_length=self.forecast_months,
            frequency='MS',  # Month start frequency
            prediction_interval=0.9,  # 90% confidence interval
            ensemble=['simple', 'dist'],  # Multiple ensemble strategies
            max_generations=10,
            num_validations=2,
            validation_method='backwards',
            no_negatives=False,  # Anomalies can be negative
            n_jobs=-1,  # Use all CPU cores
            verbose=1,
            random_seed=42
        )
        
        try:
            self.model = self.model.fit(self.data_wide)
            print("\n[OK] Model training completed!")
            print(f"  Best model: {self.model.best_model_id}")
            
        except Exception as e:
            print(f"[FAILED] Error during model training: {e}")
            raise
    
    def calculate_climate_index(self, data):
        """
        Calculate Actuarial Climate Index (ICA) from anomaly variables.
        Formula: ICA = (T90 - T10 + W_std + P_std + D_std) / 5
        
        Based on the methodology in src/scripts/graficas.py plot_ICA function.
        
        Args:
            data (pd.DataFrame): DataFrame with anomaly columns
            
        Returns:
            pd.Series: Climate index values
        """
        # Initialize ICA as zeros
        ica = pd.Series(0.0, index=data.index)
        component_count = 0
        
        # Temperature component: T90 - T10 (difference between 90th and 10th percentiles)
        t90_col = None
        t10_col = None
        for col in data.columns:
            if 't_90' in col.lower():
                t90_col = col
            elif 't_10' in col.lower():
                t10_col = col
        
        if t90_col is not None and t10_col is not None:
            ica += data[t90_col] - data[t10_col]
            component_count += 1
        
        # Wind anomalies
        for col in data.columns:
            if 'wind' in col.lower() or ('anomal' in col.lower() and 'above' in col.lower()):
                ica += data[col]
                component_count += 1
                break
        
        # Precipitation anomalies
        for col in data.columns:
            if 'precip' in col.lower() or 'lluvia' in col.lower():
                ica += data[col]
                component_count += 1
                break
        
        # Drought anomalies
        for col in data.columns:
            if 'drought' in col.lower() or 'sequia' in col.lower():
                ica += data[col]
                component_count += 1
                break
        
        # Divide by number of components (typically 5 for full ICA)
        # If all components present, divide by 5; otherwise by component_count
        divisor = 5 if component_count >= 4 else component_count
        ica = ica / divisor if divisor > 0 else ica
        
        return ica
    
    def generate_forecasts(self):
        """
        Generate forecasts using the trained model.
        """
        print("\n[Generating forecasts...]")
        
        try:
            self.prediction = self.model.predict()
            print("[OK] Forecasts generated!")
            
            forecasts = self.prediction.forecast
            upper = self.prediction.upper_forecast
            lower = self.prediction.lower_forecast
            
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
            
            print(f"  Forecast shape: {forecasts.shape}")
            print(f"  Variables: {list(forecasts.columns)}")
            
            return forecasts, upper, lower
            
        except Exception as e:
            print(f"[FAILED] Error generating forecasts: {e}")
            raise
    
    def evaluate_model(self):
        """
        Evaluate model performance and display metrics.
        """
        print("\n[Model Performance Evaluation:]")
        print("-" * 80)
        
        results = self.model.results()
        
        # Get winning model info
        winning_row = results.loc[results['ID'] == self.model.best_model_id]
        
        if not winning_row.empty:
            print(f"\nBest Model: {self.model.best_model_id}")
            print(f"Score: {winning_row['Score'].values[0]:.4f}")
        
        # Show top 5 models
        print("\n Top 5 Models:")
        print("-" * 80)
        if 'Score' in results.columns:
            top_models = results.nsmallest(5, 'Score')[['ID', 'Model', 'Score']]
        else:
            top_models = results.head(5)[['ID', 'Model']]
        print(top_models.to_string(index=False))
        
        self.results_df = results
    
    def visualize_forecasts(self, forecasts, upper, lower):
        """
        Create comprehensive visualizations of forecasts.
        
        Args:
            forecasts (pd.DataFrame): Point forecasts
            upper (pd.DataFrame): Upper confidence bounds
            lower (pd.DataFrame): Lower confidence bounds
        """
        print("\n[Creating visualizations...]")
        
        n_vars = len(forecasts.columns)
        n_cols = 2
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for idx, col in enumerate(forecasts.columns):
            ax = axes[idx]
            
            # Get historical data
            hist_data = self.data_wide[col]
            
            # Get forecast data
            forecast_data = forecasts[col]
            upper_data = upper[col]
            lower_data = lower[col]
            
            # Plot historical data
            ax.plot(hist_data.index, hist_data.values, 'b-', linewidth=2, label='Historical', alpha=0.8)
            
            # Plot forecast
            ax.plot(forecast_data.index, forecast_data.values, 'r-', linewidth=2, label='Forecast', alpha=0.8)
            
            # Confidence interval
            ax.fill_between(lower_data.index, 
                           lower_data.values, 
                           upper_data.values, 
                           alpha=0.2, color='red', label='90% CI')
            
            # Formatting
            ax.set_title(f'{col}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Anomaly Value')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Remove extra subplots
        for idx in range(len(forecasts.columns), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        output_file = os.path.join(self.output_path, 'forecasts_all_variables.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_file}")
        plt.close()
    
    def save_forecasts_to_csv(self, forecasts, upper, lower):
        """
        Save forecast results to CSV files.
        
        Args:
            forecasts (pd.DataFrame): Point forecasts
            upper (pd.DataFrame): Upper bounds
            lower (pd.DataFrame): Lower bounds
        """
        print("\n Saving forecasts to CSV...")
        
        # Save forecasts
        forecast_file = os.path.join(self.output_path, f'forecast_{self.region_name}_point.csv')
        forecasts.to_csv(forecast_file)
        print(f"[OK] Saved: {forecast_file}")
        
        # Save upper bound
        upper_file = os.path.join(self.output_path, f'forecast_{self.region_name}_upper_90ci.csv')
        upper.to_csv(upper_file)
        print(f"[OK] Saved: {upper_file}")
        
        # Save lower bound
        lower_file = os.path.join(self.output_path, f'forecast_{self.region_name}_lower_90ci.csv')
        lower.to_csv(lower_file)
        print(f"[OK] Saved: {lower_file}")
        
        # Save combined results with historical data
        combined_file = os.path.join(self.output_path, f'forecast_{self.region_name}_combined.csv')
        combined = pd.concat([self.data_wide, forecasts], axis=0)
        combined.to_csv(combined_file)
        print(f"[OK] Saved: {combined_file}")
    
    def generate_summary_report(self):
        """
        Generate a text summary of the forecasting results.
        """
        print("\n Generating summary report...")
        
        report = f"""
{'='*80}
CLIMATE ANOMALY FORECAST REPORT - {self.region_name.upper()}
{'='*80}

FORECAST CONFIGURATION:
  • Region: {self.region_name.replace('_', ' ').title()}
  • Forecast Period: {self.forecast_months} months
  • Forecast Date Range: {self.prediction.forecast.index.min().date()} to {self.prediction.forecast.index.max().date()}
  • Historical Data Range: {self.data_wide.index.min().date()} to {self.data_wide.index.max().date()}

MODEL INFORMATION:
  • Best Model ID: {self.model.best_model_id}
  • Ensemble Strategy: simple, dist
  • Number of Variables: {len(self.data_wide.columns)}
  • Total Data Points (monthly): {len(self.data_wide)}

FORECAST VARIABLES:
{chr(10).join([f'  • {col}' for col in self.prediction.forecast.columns])}

OUTPUT FILES GENERATED:
  • forecast_{self.region_name}_point.csv - Point forecasts
  • forecast_{self.region_name}_upper_90ci.csv - Upper confidence bounds
  • forecast_{self.region_name}_lower_90ci.csv - Lower confidence bounds
  • forecast_{self.region_name}_combined.csv - Historical + Forecast
  • forecasts_all_variables.png - Visualization

RECOMMENDATIONS:
  1. Review confidence intervals for forecast uncertainty
  2. Compare with seasonal climate patterns
  3. Validate forecasts against actual observations as they become available
  4. Update model quarterly with new data for improved accuracy
  5. Use forecasts in conjunction with other climate indicators

GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
        """
        
        report_file = os.path.join(self.output_path, 'forecast_summary_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"[OK] Saved: {report_file}")
    
    def run_full_pipeline(self):
        """
        Execute the complete forecasting pipeline.
        """
        print("\n" + "="*80)
        print("MONTHLY CLIMATE ANOMALY FORECASTING PIPELINE")
        print("="*80 + "\n")
        
        try:
            # Load data
            datasets = self.load_monthly_data()
            
            if not datasets:
                print("[FAILED] No data files found. Cannot proceed.")
                return False
            
            # Extract time series
            self.extract_regional_series(datasets)
            
            if self.data_wide is None or len(self.data_wide) == 0:
                print("[FAILED] No time series data extracted. Cannot proceed.")
                return False
            
            # Train model
            self.fit_autots_model()
            
            # Generate forecasts
            forecasts, upper, lower = self.generate_forecasts()
            
            # Evaluate
            self.evaluate_model()
            
            # Visualize
            self.visualize_forecasts(forecasts, upper, lower)
            
            # Save results
            self.save_forecasts_to_csv(forecasts, upper, lower)
            
            # Generate report
            self.generate_summary_report()
            
            print("\n" + "="*80)
            print("[OK] PIPELINE COMPLETED SUCCESSFULLY!")
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
    # Create forecaster instance for Cundinamarca-Bogotá
    forecaster = MonthlyClimateForecaster(
        region_name="cundinamarca_bogota",
        forecast_months=12  # 1-year forecast
    )
    
    # Run the pipeline
    success = forecaster.run_full_pipeline()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
