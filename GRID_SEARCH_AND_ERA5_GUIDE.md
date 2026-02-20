# Enhanced ETS+ENSO Forecasting with Grid Search and ERA5 ONI Data

## Overview

Your forecast script now includes two major enhancements:

### 1. **Improved ONI Data Fetching from ERA5**
Constructs the Oceanic Niño Index (ONI) directly from ERA5 sea surface temperature data for the Colombia Pacific region, replacing the simple NOAA CPC method.

### 2. **ETS Hyperparameter Grid Search**
Automatically searches over all ETS model configurations and ENSO lag strategies to find the optimal combination for each climate variable.

---

## Feature 1: ERA5-Based ONI Construction

### What Changed

The `fetch_oni_data()` method now supports two approaches:

```python
# Method 1: ERA5-based (NEW - RECOMMENDED)
ONI = fetch_oni_data(use_era5=True, cds_key="your_cds_api_key")

# Method 2: NOAA CPC (Fallback)
ONI = fetch_oni_data(use_era5=False)  # Uses old method
```

### How to Use

1. **Get CDS API credentials** (if using ERA5):
   - Visit: https://cds.climate.copernicus.eu/
   - Register and get your API key
   - Found in your profile under "API key"

2. **Run with ERA5**:
   ```python
   # In your main script
   forecaster = MonthlyClimateForecasterETSWithENSO()
   
   oni_handler = ONIDataHandler()
   oni_data = oni_handler.fetch_oni_data(
       use_era5=True,
       cds_key="8e8b7254-adc9-4a09-9c04-a2b5ff2390b9"  # Your key here
   )
   ```

### ERA5 Method Advantages

✓ **Direct SST data** → More accurate for Colombia Pacific region  
✓ **Customizable domain** → Focus on coastal zone (1°-8°N, 77°-79°W)  
✓ **Spatial visualization** → See SST anomalies across the Pacific  
✓ **Baseline-specific** → Uses 1991-2020 climatology  
✓ **3-month smoothing** → Standard ONI methodology  

### Output Files Created

```
era5_oni_timeseries.png     # Time series + climatology visualization
colombia_pacific_sst_era5.nc # Raw ERA5 data (cached for efficiency)
```

---

## Feature 2: ETS Hyperparameter Grid Search

### What's Being Searched

```
Error types:        ['add', 'mul']           (additive, multiplicative)
Trend types:        ['add', 'mul', None]     (with/without trend)
Seasonal types:     ['add', 'mul', None]     (with/without seasonality)
Seasonal periods:   [12, 6, 3]              (monthly, bi-monthly, quarterly)
Damped trend:       [True, False]            (smooth/sharp trends)
ENSO lags:          [0-7 months]            (lag structures to test)
Cross-validation:   3-fold splits           (robustness check)
```

**Total configurations tested**: ~800-1000 per variable

### How to Use

#### Option A: Automatic Grid Search (Recommended)

```python
forecaster = MonthlyClimateForecasterETSWithENSO()
forecaster.prepare_enso_data()
forecaster.load_monthly_data()

# For each variable, run grid search
for variable in forecaster.data_wide.columns:
    series = forecaster.data_wide[variable]
    
    # This will automatically run grid search
    ets_fitted, enso_model, forecast_df, model_info = \
        forecaster.fit_ets_with_enso(
            series, 
            variable,
            forecaster.enso_features,
            use_grid_search=True  # ← Enable grid search
        )
```

#### Option B: Manual Grid Search + Save Results

```python
# Run grid search once
gs_results = forecaster.grid_search_ets_parameters(
    series=forecaster.data_wide['temperature_t_90'],
    variable_name='temperature_t_90',
    enso_features_aligned=aligned_enso,
    error_types=['add', 'mul'],
    trend_types=['add', 'mul', None],
    seasonal_types=['add', 'mul', None],
    seasonal_periods=[12, 6, 3],
    damped_trend_options=[True, False],
    enso_lag_range=range(0, 8),
    cv_splits=3
)

# Use best parameters from grid search
best_params = gs_results['best_params']

# Fit model with these parameters
ets_fitted, enso_model, forecast_df, model_info = \
    forecaster.fit_ets_with_enso(
        series,
        variable_name='temperature_t_90',
        enso_features_aligned=aligned_enso,
        best_params=best_params  # ← Use grid search results
    )
```

### Interpreting Grid Search Output

**Console Output Example:**
```
[GRID SEARCH] ETS+ENSO Hyperparameter Optimization for temperature_t_90
Testing 960 configurations...

[RESULTS] Top 10 ETS Configurations (by AIC):
────────────────────────────────────────────────────────────────────────────────
E:A T:A S:A SP:12 D:True Lag:3    | AIC:   124.3 | CV_RMSE:  0.2345
E:A T:A S:A SP:12 D:False Lag:3   | AIC:   125.1 | CV_RMSE:  0.2412
E:M T:A S:A SP:12 D:True Lag:4    | AIC:   126.8 | CV_RMSE:  0.2567
...
```

**Abbreviations:**
- `E:A/M` = Error (Additive/Multiplicative)
- `T:A/M/N` = Trend (Additive/Multiplicative/None)
- `S:A/M/N` = Seasonal (Additive/Multiplicative/None)
- `SP:X` = Seasonal Period (12=monthly, 6=bi-monthly, 3=quarterly)
- `D:T/F` = Damped Trend (True/False)
- `Lag:X` = ENSO lag in months
- `AIC` = Model fit (lower is better)
- `CV_RMSE` = Cross-validation error (lower is better)

### Output Files Created

```
grid_search_results_{variable}.csv              # Full results (all 800+ configs)
grid_search_visualization_{variable}.png        # 4-panel diagnostic plot
  ├─ Top configurations by AIC
  ├─ AIC vs CV_RMSE scatter (colored by lag)
  ├─ AIC distribution by seasonal period
  └─ Best AIC across different ENSO lags
```

### Grid Search Visualization Interpretation

**Panel 1: AIC Rankings**
- Shows top 20 configurations
- Lower AIC = better fit
- Look for consistency across similar configs

**Panel 2: AIC vs CV_RMSE**
- Each point = one configuration
- Color = ENSO lag used
- Good models: low AIC + low CV_RMSE
- Red flags: high CV_RMSE (overfitting) with low AIC

**Panel 3: Seasonal Period Comparison**
- Histogram of AIC by seasonal period (12, 6, 3 months)
- Monthly (SP=12) usually performs best for climate data
- Different colors show distributions

**Panel 4: ENSO Lag Efficiency**
- Best AIC for each ENSO lag
- Optimal lag usually 3-6 months for tropical systems
- Monotonic increase suggests diminishing returns

---

## Workflow: Complete Example

```python
from src.forecast_scripts.ETSX_ica_forecast import (
    ONIDataHandler,
    MonthlyClimateForecasterETSWithENSO
)

# ============================================================================
# STEP 1: Prepare ONI Data (ERA5-based)
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Prepare ONI Data with ERA5")
print("="*80)

forecaster = MonthlyClimateForecasterETSWithENSO(
    region_name="cundinamarca_bogota",
    forecast_months=12,
    enso_max_lag=6
)

# Fetch ERA5 ONI (automatic, cached if already downloaded)
forecaster.prepare_enso_data()
# Output: era5_oni_timeseries.png, climatology analysis

# ============================================================================
# STEP 2: Load Climate Data
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Load Climate Data")
print("="*80)

datasets = forecaster.load_monthly_data()
forecaster.extract_regional_series(datasets)

# ============================================================================
# STEP 3: Grid Search for Best ETS Configuration
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Grid Search ETS Parameters")
print("="*80)

# Option: Run grid search for first variable
first_var = forecaster.data_wide.columns[0]
gs_results = forecaster.grid_search_ets_parameters(
    series=forecaster.data_wide[first_var],
    variable_name=first_var,
    enso_features_aligned=forecaster.align_enso_with_climate_data(),
    cv_splits=3
)

best_params = gs_results['best_params']
print(f"\nBest configuration for {first_var}:")
print(f"  Error: {best_params['error']}")
print(f"  Trend: {best_params['trend']}")
print(f"  Seasonal: {best_params['seasonal']}")
print(f"  Seasonal Period: {best_params['seasonal_period']}")
print(f"  Damped Trend: {best_params['damped_trend']}")
print(f"  ENSO Lag: {best_params['enso_lag']} months")
print(f"  AIC: {best_params['aic']:.2f}")

# Output: grid_search_results_*.csv, grid_search_visualization_*.png

# ============================================================================
# STEP 4: Fit Models with Best Parameters
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Fit ETS+ENSO Models")
print("="*80)

# Option A: Use grid search for each variable
for col in forecaster.data_wide.columns:
    series = forecaster.data_wide[col]
    if len(series) < 60:
        continue
    
    ets_model, enso_model, forecast_df, model_info = \
        forecaster.fit_ets_with_enso(
            series, col,
            forecaster.align_enso_with_climate_data(),
            use_grid_search=True  # Enable grid search for each variable
        )

# ============================================================================
# STEP 5: Generate Forecasts & Diagnostics
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Generate Forecasts & Diagnostics")
print("="*80)

forecasts, upper, lower, enso_comp, ets_comp = \
    forecaster.generate_forecasts()

forecaster.visualize_forecasts(forecasts, upper, lower, enso_comp)
forecaster.visualize_enso_contribution()
forecaster.visualize_residual_diagnostics()
forecaster.visualize_data_distributions()

# ============================================================================
# STEP 6: Save Results
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Save Results")
print("="*80)

forecaster.save_results(forecasts, upper, lower, enso_comp, ets_comp)

print("\n[COMPLETE] Forecasting pipeline finished!")
print(f"Results saved to: {forecaster.output_path}")
```

---

## Choosing Between Configurations

### When to Use Grid Search

✓ **Always, for production models**  
✓ When discovering new region/variable relationships  
✓ When model performance degrades  
✓ For published research or decision-making  

### When Grid Search Takes Too Long

Grid search tests ~1000 configurations per variable, which can take:
- **2-5 minutes** per variable on modern CPU
- **10-30 minutes** for all variables

**To speed up:**

```python
# Reduce search space
gs_results = forecaster.grid_search_ets_parameters(
    series=series,
    variable_name=var_name,
    enso_features_aligned=enso_aligned,
    error_types=['add'],           # Test only additive
    trend_types=['add'],            # Only additive trend
    seasonal_types=['add'],         # Only additive seasonality
    seasonal_periods=[12],          # Only monthly
    damped_trend_options=[True],    # Only damped
    enso_lag_range=range(0, 6),    # Test 0-5 lags instead of 0-7
    cv_splits=2                     # Use 2-fold CV instead of 3
)
# Much faster: ~200 configurations instead of 960!
```

---

## Troubleshooting

### Issue: ERA5 Download Very Slow

**Solution:** The first download is large (~500MB), but it's cached.
```python
# If file exists, it will be reused automatically
# To force re-download, delete: colombia_pacific_sst_era5.nc
```

### Issue: Grid Search Produces No Results

**Solution:** Check data alignment and validity:
```python
# Verify data
print(f"Data shape: {forecaster.data_wide.shape}")
print(f"ENSO shape: {enso_aligned.shape}")
print(f"Common dates: {len(series.index.intersection(enso_aligned.index))}")

# Ensure enough data
assert len(series) > 100, "Need at least 100 observations"
```

### Issue: Best Configuration Seems Wrong

**Best practices:**
1. **Check CV_RMSE, not just AIC** - AIC can be overly optimistic
2. **Look at top 5, not just #1** - Often very similar performance
3. **Compare across variables** - Patterns should be somewhat consistent
4. **Inspect residuals** - Check visualization outputs

```python
# Examine top 5 configurations
top_5 = gs_results['top_5_params']
for i, params in enumerate(top_5):
    print(f"\nConfiguration {i+1}:")
    print(f"  AIC: {params['aic']:.2f}")
    print(f"  CV_RMSE: {params['cv_rmse']:.4f}")
    print(f"  Config: {params['error']}/{params['trend']}/{params['seasonal']}")
```

---

## Advanced: Custom Grid Search

```python
# Search only configurations relevant to your hypothesis
gs_results = forecaster.grid_search_ets_parameters(
    series=series,
    variable_name="my_variable",
    enso_features_aligned=enso_aligned,
    error_types=['add', 'mul'],    # Test both
    trend_types=['add', None],      # No multiplicative trend
    seasonal_types=['add'],         # Only additive seasonality
    seasonal_periods=[12],          # Only monthly seasonality
    damped_trend_options=[True],    # Always damped
    enso_lag_range=range(1, 7),    # Test lags 1-6 (skip lag 0)
    cv_splits=5                     # Thorough 5-fold validation
)
```

---

## Output Structure

```
articles/graficas/forecast_ets_enso_cundinamarca_bogota/
├── Era5 Data & Visualization
│   ├── era5_oni_timeseries.png           ← Full ONI analysis
│   └── colombia_pacific_sst_era5.nc       ← Raw data (cached)
│
├── Grid Search Results (per variable)
│   ├── grid_search_results_temperature_t_90.csv
│   ├── grid_search_results_precipitation.csv
│   └── grid_search_visualization_temperature_t_90.png
│
├── Forecasts
│   ├── forecast_ets_enso_cundinamarca_bogota_point.csv
│   ├── forecast_ets_enso_cundinamarca_bogota_upper_95ci.csv
│   └── forecast_ets_enso_cundinamarca_bogota_lower_95ci.csv
│
├── Diagnostics
│   ├── residual_diagnostics_all_variables.png
│   ├── residual_summary_comparison.png
│   ├── data_distributions_all_variables.png
│   └── enso_contribution_by_variable.png
│
└── Visualizations
    ├── forecasts_ets_enso_all_variables.png
    └── [more plots...]
```

---

## Reference: ETS Configuration Codes

| Code | Meaning | Use Case |
|------|---------|----------|
| AAA | Add/Add/Add | Stationary with additive errors |
| AAM | Add/Add/Mul | Proportional seasonality growth |
| AMA | Add/Mul/Add | Trend changes magnitude over time |
| AMM | Add/Mul/Mul | Both trend and seasonality change |
| ANN | Add/None/None | Simple exponential smoothing |
| MAA | Mul/Add/Add | Multiplicative errors (volatility) |
| MMM | Mul/Mul/Mul | Multiplicative growth + seasonality |

**For climate data:**
- **AAA** (Additive) → Usually best for temperature, precipitation anomalies
- **AAM** → When seasonal component grows/shrinks over time
- **Damped=True** → Prevents unrealistic long-term trends

---

## References

- **ONI/ENSO**: https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensoyears.shtml
- **ERA5 Data**: https://cds.climate.copernicus.eu/
- **ETS Methods**: Hyndman et al. (2008) "Forecasting with Exponential Smoothing"
- **Colombia Climate**: ACI-CO Methodology Documentation

---

**Questions?** Check the console output or grid search visualization files for diagnostic information.
