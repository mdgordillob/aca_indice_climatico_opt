#!/usr/bin/env python3
"""
Example: Using Grid Search + ERA5 ONI for Climate Forecasting
==============================================================

This script demonstrates how to use the new grid search and ERA5 ONI features
in the enhanced ETSX_ica_forecast.py module.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from forecast_scripts.ETSX_ica_forecast import (
    ONIDataHandler,
    MonthlyClimateForecasterETSWithENSO
)


def example_1_era5_oni_basic():
    """
    Example 1: Basic ERA5 ONI Data Fetching
    ========================================
    
    This is the simplest way to get ONI data from ERA5.
    - Automatically downloads SST data for Colombia Pacific
    - Creates climatology baseline
    - Applies 3-month smoothing
    - Generates comprehensive visualizations
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic ERA5 ONI Fetching")
    print("="*80)
    
    # Create ONI handler
    oni_handler = ONIDataHandler()
    
    # Fetch ONI from ERA5 (with automatic fallback to NOAA if ERA5 unavailable)
    # Note: First run downloads ~500MB, subsequent runs use cached file
    oni_data = oni_handler.fetch_oni_data(
        use_era5=True,
        cds_key=None  # Set your CDS API key here if needed
    )
    
    print(f"\nONI Data Shape: {oni_data.shape}")
    print(f"Date Range: {oni_data.index.min()} to {oni_data.index.max()}")
    print(f"\nONI Summary Statistics:")
    print(oni_data['ONI'].describe())
    
    # Standardize for use as a regressor
    oni_standardized = oni_handler.standardize_oni(
        training_start='1961-01-01',
        training_end='2020-12-31'
    )
    
    print(f"\nStandardized ONI (Z-scores):")
    print(f"  Mean: {oni_standardized['Z_ONI'].mean():.4f}")
    print(f"  Std: {oni_standardized['Z_ONI'].std():.4f}")
    
    return oni_handler


def example_2_grid_search_single_variable():
    """
    Example 2: Grid Search for a Single Variable
    =============================================
    
    Demonstrates how to find the optimal ETS configuration for one climate variable.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Grid Search for Single Variable")
    print("="*80)
    
    # Initialize forecaster
    forecaster = MonthlyClimateForecasterETSWithENSO(
        region_name="cundinamarca_bogota",
        forecast_months=12
    )
    
    # Prepare ENSO data
    print("\nPreparing ENSO data...")
    forecaster.prepare_enso_data()
    
    # Load climate data
    print("Loading climate data...")
    datasets = forecaster.load_monthly_data()
    forecaster.extract_regional_series(datasets)
    
    # Align ENSO with climate data
    enso_aligned = forecaster.align_enso_with_climate_data()
    
    # Get first variable
    if forecaster.data_wide is None or len(forecaster.data_wide.columns) == 0:
        print("ERROR: No climate data loaded!")
        return None
    
    variable_name = forecaster.data_wide.columns[0]
    series = forecaster.data_wide[variable_name].dropna()
    
    print(f"\nRunning grid search for: {variable_name}")
    print(f"Data points: {len(series)}")
    print("This will test ~960 configurations...\n")
    
    # Run grid search
    gs_results = forecaster.grid_search_ets_parameters(
        series=series,
        variable_name=variable_name,
        enso_features_aligned=enso_aligned,
        error_types=['add', 'mul'],
        trend_types=['add', 'mul', None],
        seasonal_types=['add', 'mul', None],
        seasonal_periods=[12, 6, 3],
        damped_trend_options=[True, False],
        enso_lag_range=range(0, 8),
        cv_splits=3
    )
    
    if gs_results is None:
        print("Grid search failed!")
        return None
    
    # Display best parameters
    best = gs_results['best_params']
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION FOUND:")
    print(f"{'='*80}")
    print(f"Error Type:      {best['error']} ({'Additive' if best['error']=='add' else 'Multiplicative'})")
    print(f"Trend:           {best['trend']} ({'None' if best['trend'] is None else best['trend'].capitalize()})")
    print(f"Seasonality:     {best['seasonal']} ({'None' if best['seasonal'] is None else best['seasonal'].capitalize()})")
    print(f"Seasonal Period: {best['seasonal_period'] if best['seasonal'] else 'N/A'} months")
    print(f"Damped Trend:    {best['damped_trend']}")
    print(f"ENSO Lag:        {best['enso_lag']} months")
    print(f"AIC:             {best['aic']:.2f} (lower is better)")
    print(f"CV RMSE:         {best['cv_rmse']:.4f}")
    print(f"{'='*80}\n")
    
    return forecaster, gs_results, series, enso_aligned


def example_3_grid_search_vs_fixed():
    """
    Example 3: Compare Grid Search Results with Fixed Configuration
    ===============================================================
    
    Shows how grid search can improve over a fixed "one-size-fits-all" config.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Grid Search vs Fixed Configuration")
    print("="*80)
    
    forecaster = MonthlyClimateForecasterETSWithENSO(region_name="cundinamarca_bogota")
    forecaster.prepare_enso_data()
    datasets = forecaster.load_monthly_data()
    forecaster.extract_regional_series(datasets)
    enso_aligned = forecaster.align_enso_with_climate_data()
    
    if forecaster.data_wide is None or len(forecaster.data_wide.columns) == 0:
        print("ERROR: No data loaded!")
        return None
    
    variable = forecaster.data_wide.columns[0]
    series = forecaster.data_wide[variable].dropna()
    
    print(f"\nVariable: {variable}")
    print(f"Data points: {len(series)}")
    
    # Approach 1: Fixed configuration (old default)
    print("\n" + "-"*60)
    print("APPROACH 1: Fixed Configuration (Default)")
    print("-"*60)
    print("Config: Error=Add, Trend=Add, Season=Add, Period=12, Damped=True")
    
    _, _, forecast_fixed, info_fixed = forecaster.fit_ets_with_enso(
        series, variable, enso_aligned,
        use_grid_search=False  # Use default config
    )
    
    print(f"Result: AIC = {info_fixed['aic']:.2f}, R²(ENSO) = {info_fixed['r2_enso']:.3f}")
    
    # Approach 2: Grid search for best config
    print("\n" + "-"*60)
    print("APPROACH 2: Grid Search (Optimized)")
    print("-"*60)
    
    gs_results = forecaster.grid_search_ets_parameters(
        series, variable, enso_aligned,
        error_types=['add', 'mul'],
        trend_types=['add', 'mul', None],
        seasonal_types=['add', 'mul', None],
        seasonal_periods=[12],  # Focus on monthly for speed
        damped_trend_options=[True, False],
        enso_lag_range=range(0, 6),
        cv_splits=2  # 2-fold to speed up
    )
    
    best_params = gs_results['best_params']
    _, _, forecast_gs, info_gs = forecaster.fit_ets_with_enso(
        series, variable, enso_aligned,
        best_params=best_params
    )
    
    print(f"Best Config: {best_params['error']}/{best_params['trend']}/{best_params['seasonal']}")
    print(f"Result: AIC = {info_gs['aic']:.2f}, R²(ENSO) = {info_gs['r2_enso']:.3f}")
    
    # Compare
    print("\n" + "-"*60)
    print("COMPARISON")
    print("-"*60)
    improvement_aic = ((info_fixed['aic'] - info_gs['aic']) / abs(info_fixed['aic'])) * 100
    improvement_r2 = ((info_gs['r2_enso'] - info_fixed['r2_enso']) / (info_fixed['r2_enso'] + 0.001)) * 100
    
    print(f"AIC Improvement:    {improvement_aic:+.1f}% {'✓ Better' if improvement_aic > 0 else '✗ Worse'}")
    print(f"R² ENSO Change:     {improvement_r2:+.1f}% {'✓ Better' if improvement_r2 > 0 else '✗ Worse'}")
    
    return forecaster


def example_4_quick_configuration_sweep():
    """
    Example 4: Quick Configuration Sweep (Reduced Search Space)
    ===========================================================
    
    When you need results fast, reduce the search space strategically.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Quick Configuration Sweep")
    print("="*80)
    
    forecaster = MonthlyClimateForecasterETSWithENSO(region_name="cundinamarca_bogota")
    forecaster.prepare_enso_data()
    datasets = forecaster.load_monthly_data()
    forecaster.extract_regional_series(datasets)
    enso_aligned = forecaster.align_enso_with_climate_data()
    
    if forecaster.data_wide is None:
        return None
    
    variable = forecaster.data_wide.columns[0]
    series = forecaster.data_wide[variable].dropna()
    
    # Quick search: only best practices for climate data
    print("\nTesting subset of configurations (climate best practices):")
    print("  - Error: Additive only (most common for anomalies)")
    print("  - Trend: Additive or None")
    print("  - Seasonal: Additive (monthly climate data)")
    print("  - Period: Monthly (12) only")
    print("  - Damped: True and False (test both)")
    print("  - ENSO Lag: 0-5 months (typical tropical lag)")
    print("  - CV: 2-fold (faster validation)")
    print()
    
    gs_results = forecaster.grid_search_ets_parameters(
        series, variable, enso_aligned,
        error_types=['add'],           # ← Only additive
        trend_types=['add', None],      # ← No multiplicative
        seasonal_types=['add'],         # ← Only additive
        seasonal_periods=[12],          # ← Monthly only
        damped_trend_options=[True, False],  # ← Test both
        enso_lag_range=range(0, 6),    # ← Lags 0-5
        cv_splits=2                     # ← 2-fold CV
    )
    
    total_configs = len(gs_results['results_df'])
    print(f"\nTotal configurations tested: {total_configs}")
    print(f"(vs. ~960 for full grid search)")
    print(f"Speed improvement: ~5x faster!")
    
    return forecaster, gs_results


def main():
    """Run all examples."""
    
    print("\n" + "="*80)
    print("GRID SEARCH + ERA5 ONI EXAMPLES")
    print("="*80)
    
    # Example 1: Basic ERA5 ONI
    try:
        oni_handler = example_1_era5_oni_basic()
        print("✓ Example 1 completed successfully")
    except Exception as e:
        print(f"✗ Example 1 failed: {e}")
        oni_handler = None
    
    # Example 2: Grid Search Single Variable
    try:
        result = example_2_grid_search_single_variable()
        if result:
            forecaster, gs_results, series, enso_aligned = result
            print("✓ Example 2 completed successfully")
        else:
            print("✗ Example 2: No data available")
    except Exception as e:
        print(f"✗ Example 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 3: Grid Search vs Fixed
    try:
        example_3_grid_search_vs_fixed()
        print("✓ Example 3 completed successfully")
    except Exception as e:
        print(f"✗ Example 3 failed: {e}")
    
    # Example 4: Quick Sweep
    try:
        example_4_quick_configuration_sweep()
        print("✓ Example 4 completed successfully")
    except Exception as e:
        print(f"✗ Example 4 failed: {e}")
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review output visualizations in: articles/graficas/")
    print("2. Check grid_search_results_*.csv for all tested configurations")
    print("3. Use best parameters found for production forecasting")
    print("4. See GRID_SEARCH_AND_ERA5_GUIDE.md for detailed documentation")
    print()


if __name__ == "__main__":
    main()
