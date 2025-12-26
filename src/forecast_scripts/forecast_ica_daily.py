"""
AutoTS Forecasting for Daily Climate Indices - Cundinamarca-Bogotá Region

This script performs time series forecasting on daily climate anomalies
(temperature, precipitation, drought, wind) for the Cundinamarca-Bogotá region
using AutoTS ensemble methods with proper validation and visualization.
"""

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rio
import geopandas as gpd
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


class DailyClimateForecaster:
    """
    Handles forecasting of daily climate anomalies for a specific region.
    """
    
    def __init__(self, region_name="cundinamarca_bogota", forecast_days=90):
        """
        Initialize the forecaster.
        
        Args:
            region_name (str): Region identifier
            forecast_days (int): Number of days to forecast
        """
        self.region_name = region_name
        self.forecast_days = forecast_days
        self.data_wide = None
        self.model = None
        self.prediction = None
        self.results_df = None
        
        # Set up paths
        self.base_path = os.path.join(".", "data", "processed", f"anomalias_{region_name}")
        self.output_path = os.path.join(".", "articles", "graficas", f"forecast_{region_name}")
        
        os.makedirs(self.output_path, exist_ok=True)
        
    def load_daily_netcdf_data(self):
        """
        Load daily NetCDF anomaly files for temperature, wind, precipitation, and drought.
        
        Returns:
            dict: Dictionary containing xarray Datasets
        """
        print("📂 Loading daily NetCDF data...")
        
        data_files = {
            'temperature': os.path.join(self.base_path, "anomalies_temperature_daily.nc"),
            'wind': os.path.join(self.base_path, "anomalies_wind_daily.nc"),
            'precipitation': os.path.join(self.base_path, "anomalies_precipitation_daily.nc"),
            'drought': os.path.join(self.base_path, "anomalies_drought_daily.nc")
        }
        
        datasets = {}
        for key, filepath in data_files.items():
            if os.path.exists(filepath):
                try:
                    datasets[key] = xr.open_dataset(filepath)
                    print(f"✓ Loaded {key}: {filepath}")
                except Exception as e:
                    print(f"⚠ Could not load {key}: {e}")
            else:
                print(f"⚠ File not found: {filepath}")
        
        return datasets
    
    def extract_regional_daily_series(self, datasets):
        """
        Extract spatial mean daily time series from each NetCDF dataset.
        
        Args:
            datasets (dict): Dictionary of xarray Datasets
            
        Returns:
            pd.DataFrame: Daily time series with all variables
        """
        print("📊 Extracting regional daily time series...")
        
        series_dict = {}
        
        for var_name, ds in datasets.items():
            try:
                # Get all data variables in the dataset
                for data_var in ds.data_vars:
                    # Calculate spatial mean across latitude and longitude
                    spatial_mean = ds[data_var].mean(dim=['latitude', 'longitude'], skipna=True)
                    
                    # Convert to DataFrame
                    df_temp = spatial_mean.to_dataframe().reset_index()
                    df_temp.columns = ['time', f'{var_name}_{data_var}']
                    df_temp.set_index('time', inplace=True)
                    
                    series_dict[f'{var_name}_{data_var}'] = df_temp[f'{var_name}_{data_var}']
                    print(f"  ✓ {var_name}_{data_var}: {len(df_temp)} days")
                    
            except Exception as e:
                print(f"  ⚠ Error processing {var_name}: {e}")
        
        # Combine all series
        self.data_wide = pd.concat(series_dict, axis=1)
        
        # Sort by index and remove duplicates
        self.data_wide = self.data_wide.sort_index()
        self.data_wide = self.data_wide[~self.data_wide.index.duplicated(keep='first')]
        
        print(f"\n✓ Combined dataset shape: {self.data_wide.shape}")
        print(f"  Date range: {self.data_wide.index.min()} to {self.data_wide.index.max()}")
        print(f"  Variables: {list(self.data_wide.columns)}")
        
        return self.data_wide
    
    def generate_daily_aggregations(self):
        """
        Create additional features: 7-day, 30-day moving averages for better forecasting.
        
        Returns:
            pd.DataFrame: Enhanced dataset with aggregations
        """
        print("\n📈 Generating aggregated features...")
        
        enhanced_data = self.data_wide.copy()
        
        for col in self.data_wide.columns:
            # 7-day rolling mean
            enhanced_data[f'{col}_7d_ma'] = enhanced_data[col].rolling(window=7, center=False).mean()
            
            # 30-day rolling mean
            enhanced_data[f'{col}_30d_ma'] = enhanced_data[col].rolling(window=30, center=False).mean()
            
            # 7-day rolling std
            enhanced_data[f'{col}_7d_std'] = enhanced_data[col].rolling(window=7, center=False).std()
        
        # Remove NaN rows from rolling calculations
        enhanced_data = enhanced_data.dropna()
        
        print(f"✓ Enhanced dataset with aggregations: {enhanced_data.shape}")
        
        self.data_wide = enhanced_data
        return enhanced_data
    
    def fit_autots_model(self):
        """
        Train AutoTS ensemble model on the daily data.
        """
        print("\n🤖 Training AutoTS ensemble model...")
        print("  This may take several minutes...\n")
        
        self.model = AutoTS(
            forecast_length=self.forecast_days,
            frequency='D',  # Daily frequency
            prediction_interval=0.9,  # 90% confidence interval
            ensemble=['simple', 'dist', 'horizontal-min'],  # Multiple ensemble strategies
            max_generations=15,  # Increase for better accuracy
            num_validations=3,  # More robust validation
            validation_method='backwards',  # Better for climate data
            no_negatives=False,  # Anomalies can be negative
            metric='smape',  # Better for anomaly data
            n_jobs=-1,  # Use all CPU cores
            verbose=1,
            random_seed=42,
            model_list='default',  # Use sensible defaults
            remove_outliers=None,  # Don't remove outliers in anomalies
            aggfunc='mean'
        )
        
        try:
            self.model = self.model.fit(self.data_wide)
            print("\n✓ Model training completed successfully!")
            print(f"  Best model: {self.model.best_model_id}")
            
        except Exception as e:
            print(f"✗ Error during model training: {e}")
            raise
    
    def generate_forecasts(self):
        """
        Generate forecasts using the trained model.
        """
        print("\n📮 Generating forecasts...")
        
        try:
            self.prediction = self.model.predict()
            print("✓ Forecasts generated successfully!")
            
            # Access forecast data
            forecasts = self.prediction.forecast
            upper = self.prediction.upper_forecast
            lower = self.prediction.lower_forecast
            
            print(f"  Forecast shape: {forecasts.shape}")
            print(f"  Variables: {list(forecasts.columns)}")
            
            return forecasts, upper, lower
            
        except Exception as e:
            print(f"✗ Error generating forecasts: {e}")
            raise
    
    def evaluate_model(self):
        """
        Evaluate model performance and display metrics.
        """
        print("\n📊 Model Performance Evaluation:")
        print("-" * 80)
        
        results = self.model.results()
        
        # Get winning model metrics
        winning_row = results.loc[results['ID'] == self.model.best_model_id].iloc[0:1]
        
        # Extract metric columns
        metric_cols = [col for col in results.columns 
                      if any(m in col.lower() for m in ['score', 'mae', 'rmse', 'smape', 'spl'])]
        
        if metric_cols:
            stats_display = winning_row[metric_cols].T
            stats_display.columns = ['Value']
            print(stats_display)
        
        # Show top 5 models
        print("\n🏆 Top 5 Models:")
        print("-" * 80)
        top_models = results.nsmallest(5, 'Score')[['ID', 'Model', 'Score']]
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
        print("\n📊 Creating visualizations...")
        
        # Determine number of variables to plot
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
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Highlight recent history
            if len(hist_data) > 180:
                recent_date = hist_data.index[-180]
                ax.axvline(recent_date, color='gray', linestyle='--', alpha=0.5)
        
        # Remove extra subplots
        for idx in range(len(forecasts.columns), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        output_file = os.path.join(self.output_path, 'forecasts_all_variables.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_model_performance(self):
        """
        Plot model optimization history.
        """
        print("\n📈 Plotting model optimization history...")
        
        try:
            fig, ax = plt.subplots(figsize=(14, 6))
            self.model.plot_generation_loss(ax=ax)
            plt.title('AutoTS Model Optimization History', fontsize=14, fontweight='bold')
            plt.xlabel('Generation')
            plt.ylabel('Loss (Lower is Better)')
            plt.grid(True, alpha=0.3)
            
            output_file = os.path.join(self.output_path, 'model_optimization_history.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
            plt.close()
            
        except Exception as e:
            print(f"⚠ Could not generate optimization plot: {e}")
    
    def save_forecasts_to_csv(self, forecasts, upper, lower):
        """
        Save forecast results to CSV files.
        
        Args:
            forecasts (pd.DataFrame): Point forecasts
            upper (pd.DataFrame): Upper bounds
            lower (pd.DataFrame): Lower bounds
        """
        print("\n💾 Saving forecasts to CSV...")
        
        # Save forecasts
        forecast_file = os.path.join(self.output_path, f'forecast_{self.region_name}_point.csv')
        forecasts.to_csv(forecast_file)
        print(f"✓ Saved: {forecast_file}")
        
        # Save upper bound
        upper_file = os.path.join(self.output_path, f'forecast_{self.region_name}_upper_90ci.csv')
        upper.to_csv(upper_file)
        print(f"✓ Saved: {upper_file}")
        
        # Save lower bound
        lower_file = os.path.join(self.output_path, f'forecast_{self.region_name}_lower_90ci.csv')
        lower.to_csv(lower_file)
        print(f"✓ Saved: {lower_file}")
        
        # Save combined results with historical data
        combined_file = os.path.join(self.output_path, f'forecast_{self.region_name}_combined.csv')
        combined = pd.concat([self.data_wide, forecasts], axis=0)
        combined.to_csv(combined_file)
        print(f"✓ Saved: {combined_file}")
    
    def generate_summary_report(self):
        """
        Generate a text summary of the forecasting results.
        """
        print("\n📄 Generating summary report...")
        
        report = f"""
{'='*80}
CLIMATE ANOMALY FORECAST REPORT - {self.region_name.upper()}
{'='*80}

FORECAST CONFIGURATION:
  • Region: {self.region_name.replace('_', ' ').title()}
  • Forecast Period: {self.forecast_days} days
  • Forecast Date Range: {self.prediction.forecast.index.min().date()} to {self.prediction.forecast.index.max().date()}
  • Historical Data Range: {self.data_wide.index.min().date()} to {self.data_wide.index.max().date()}

MODEL INFORMATION:
  • Best Model ID: {self.model.best_model_id}
  • Ensemble Strategy: simple, dist, horizontal-min
  • Number of Variables: {len(self.data_wide.columns)}
  • Total Data Points (daily): {len(self.data_wide)}

FORECAST VARIABLES:
{chr(10).join([f'  • {col}' for col in self.prediction.forecast.columns])}

PERFORMANCE METRICS:
  See model_evaluation.txt for detailed metrics

OUTPUT FILES GENERATED:
  • forecast_{self.region_name}_point.csv - Point forecasts
  • forecast_{self.region_name}_upper_90ci.csv - Upper confidence bounds
  • forecast_{self.region_name}_lower_90ci.csv - Lower confidence bounds
  • forecast_{self.region_name}_combined.csv - Historical + Forecast
  • forecasts_all_variables.png - Visualization
  • model_optimization_history.png - Training history
  • forecast_summary_report.txt - This report

RECOMMENDATIONS:
  1. Review confidence intervals for forecast uncertainty
  2. Compare with seasonal climate patterns
  3. Validate forecasts against actual observations as they become available
  4. Update model monthly with new data for improved accuracy
  5. Use forecasts in conjunction with other climate indicators

GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
        """
        
        report_file = os.path.join(self.output_path, 'forecast_summary_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"✓ Saved: {report_file}")
    
    def run_full_pipeline(self):
        """
        Execute the complete forecasting pipeline.
        """
        print("\n" + "="*80)
        print("🌍 DAILY CLIMATE ANOMALY FORECASTING PIPELINE")
        print("="*80 + "\n")
        
        try:
            # Load data
            datasets = self.load_daily_netcdf_data()
            
            if not datasets:
                print("✗ No data files found. Cannot proceed.")
                return False
            
            # Extract time series
            self.extract_regional_daily_series(datasets)
            
            # Generate aggregations
            self.generate_daily_aggregations()
            
            # Train model
            self.fit_autots_model()
            
            # Generate forecasts
            forecasts, upper, lower = self.generate_forecasts()
            
            # Evaluate
            self.evaluate_model()
            
            # Visualize
            self.visualize_forecasts(forecasts, upper, lower)
            self.plot_model_performance()
            
            # Save results
            self.save_forecasts_to_csv(forecasts, upper, lower)
            
            # Generate report
            self.generate_summary_report()
            
            print("\n" + "="*80)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"\n📁 Results saved to: {self.output_path}\n")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """
    Main execution function.
    """
    # Create forecaster instance for Cundinamarca-Bogotá
    forecaster = DailyClimateForecaster(
        region_name="cundinamarca_bogota",
        forecast_days=90  # 3-month forecast
    )
    
    # Run the pipeline
    success = forecaster.run_full_pipeline()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)