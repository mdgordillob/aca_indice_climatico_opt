import os
import xarray as xr
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


def load_daily_sst_files(grib_folder):
    """
    Load and combine all daily SST GRIB files.
    
    Parameters:
    - grib_folder (str): Path to folder containing SST GRIB files.
    
    Returns:
    - combined_sst (xarray.DataArray): Combined SST data with time dimension.
    """
    sst_files = sorted(Path(grib_folder).glob('era5_sst_*.grib'))
    
    if not sst_files:
        raise FileNotFoundError(f"No SST GRIB files found in {grib_folder}")
    
    print(f"Found {len(sst_files)} SST GRIB files")
    
    # Load all files and concatenate
    sst_data_list = []
    
    for file_path in sst_files:
        print(f"  Loading {file_path.name}...", end="")
        try:
            ds = xr.open_dataset(file_path, engine='cfgrib')
            if 'sst' in ds.data_vars:
                sst_data_list.append(ds['sst'])
                print(f" ✓ ({len(ds['sst'].time)} days)")
            else:
                print(f" ✗ (no sst variable)")
        except Exception as e:
            print(f" ✗ (Error: {e})")
    
    # Combine along time dimension
    combined_sst = xr.concat(sst_data_list, dim='time')
    combined_sst = combined_sst.sortby('time')
    
    print(f"\nCombined SST shape: {combined_sst.shape}")
    print(f"Time range: {combined_sst.time.values[0]} to {combined_sst.time.values[-1]}")
    
    return combined_sst


def calculate_daily_sst_anomalies(sst_data):
    """
    Calculate daily SST anomalies using climatological mean.
    
    Parameters:
    - sst_data (xarray.DataArray): Daily SST data.
    
    Returns:
    - sst_anomalies (xarray.DataArray): Daily SST anomalies.
    """
    # Get day of year for climatology
    sst_data['dayofyear'] = sst_data['time'].dt.dayofyear
    
    # Calculate climatological mean for each day of year
    mean_sst = sst_data.groupby('dayofyear').mean(dim='time')
    
    # Calculate anomalies
    sst_anomalies = sst_data.groupby('dayofyear') - mean_sst
    
    return sst_anomalies


def calculate_daily_oni_index(sst_anomalies, window=91):
    """
    Calculate daily ONI as rolling mean of SST anomalies.
    
    Parameters:
    - sst_anomalies (xarray.DataArray): Daily SST anomalies.
    - window (int): Rolling window in days (default 91 = ~3 months).
    
    Returns:
    - oni_index (xarray.DataArray): Daily ONI index.
    """
    # Calculate spatial mean
    sst_anomalies_spatial = sst_anomalies.mean(dim=['latitude', 'longitude'])
    
    # Calculate rolling mean (91 days ≈ 3 months)
    oni_index = sst_anomalies_spatial.rolling(time=window, center=True).mean()
    
    return oni_index


def resample_to_monthly(daily_oni):
    """
    Resample daily ONI to monthly values.
    
    Parameters:
    - daily_oni (xarray.DataArray): Daily ONI index.
    
    Returns:
    - monthly_oni (xarray.DataArray): Monthly ONI index.
    - monthly_df (pandas.DataFrame): Monthly ONI as DataFrame.
    """
    # Resample to monthly means
    monthly_oni = daily_oni.resample(time='MS').mean()
    
    # Convert to DataFrame
    monthly_df = pd.DataFrame({
        'time': pd.to_datetime(monthly_oni.time.values),
        'oni_daily_monthly': monthly_oni.values
    })
    
    return monthly_oni, monthly_df


def process_daily_sst(grib_folder, output_dir=None):
    """
    Main processing function: load daily SST, calculate ONI, resample to monthly.
    
    Parameters:
    - grib_folder (str): Path to folder with SST GRIB files.
    - output_dir (str): Output directory for results.
    
    Returns:
    - tuple: (daily results DataFrame, monthly results DataFrame)
    """
    
    print("=" * 80)
    print("CALCULATING ONI FROM DAILY SST DATA")
    print("=" * 80)
    
    # Load daily SST data
    print("\n1. Loading daily SST GRIB files...")
    sst_daily = load_daily_sst_files(grib_folder)
    
    # Calculate daily anomalies
    print("\n2. Calculating daily SST anomalies...")
    sst_anomalies = calculate_daily_sst_anomalies(sst_daily)
    
    # Calculate daily ONI
    print("\n3. Calculating daily ONI index (91-day rolling mean)...")
    oni_daily = calculate_daily_oni_index(sst_anomalies)
    
    # Create daily results
    daily_results = pd.DataFrame({
        'time': pd.to_datetime(oni_daily.time.values),
        'oni_daily': oni_daily.values
    })
    daily_results = daily_results.dropna()
    
    # Resample to monthly
    print("\n4. Resampling to monthly...")
    monthly_oni, monthly_results = resample_to_monthly(oni_daily)
    
    print(f"\nDaily results: {len(daily_results)} records")
    print(f"  Time range: {daily_results['time'].min()} to {daily_results['time'].max()}")
    print(f"\nMonthly results: {len(monthly_results)} records")
    print(f"  Time range: {monthly_results['time'].min()} to {monthly_results['time'].max()}")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        daily_output = os.path.join(output_dir, "oni_daily_from_raw_data.csv")
        daily_results.to_csv(daily_output, index=False)
        print(f"\n✓ Daily ONI saved to {daily_output}")
        
        monthly_output = os.path.join(output_dir, "oni_monthly_from_daily_resampled.csv")
        monthly_results.to_csv(monthly_output, index=False)
        print(f"✓ Monthly ONI (resampled from daily) saved to {monthly_output}")
    
    return daily_results, monthly_results


def compare_with_processed_data(monthly_daily, processed_csv_path, output_dir=None):
    """
    Compare monthly ONI calculated from daily data with the processed monthly data.
    
    Parameters:
    - monthly_daily (pandas.DataFrame): Monthly ONI from daily resampled data.
    - processed_csv_path (str): Path to the processed monthly CSV.
    - output_dir (str): Output directory for comparison results.
    """
    
    print("\n" + "=" * 80)
    print("COMPARING DAILY-RESAMPLED vs MONTHLY-PROCESSED ONI")
    print("=" * 80)
    
    # Load processed data
    processed = pd.read_csv(processed_csv_path)
    processed['time'] = pd.to_datetime(processed['time'])
    
    # Merge the two datasets
    comparison = monthly_daily.merge(processed, on='time', how='inner')
    
    print(f"\nCommon dates: {len(comparison)} months")
    print(f"Date range: {comparison['time'].min()} to {comparison['time'].max()}")
    
    # Calculate difference statistics
    comparison['diff'] = comparison['oni_daily_monthly'] - comparison['oni_colombia_pacific']
    comparison['abs_diff'] = abs(comparison['diff'])
    
    print(f"\nDifference Statistics:")
    print(f"  Mean difference: {comparison['diff'].mean():.6f}")
    print(f"  Max difference: {comparison['diff'].max():.6f}")
    print(f"  Min difference: {comparison['diff'].min():.6f}")
    print(f"  Std dev of difference: {comparison['diff'].std():.6f}")
    print(f"  RMSE: {np.sqrt((comparison['diff']**2).mean()):.6f}")
    
    # Calculate correlation
    correlation = comparison['oni_daily_monthly'].corr(comparison['oni_colombia_pacific'])
    print(f"  Correlation: {correlation:.6f}")
    
    # Show sample comparisons
    print(f"\nSample Comparisons:")
    print(comparison[['time', 'oni_daily_monthly', 'oni_colombia_pacific', 'diff']].head(10).to_string(index=False))
    
    # Save comparison
    if output_dir:
        comparison_output = os.path.join(output_dir, "oni_comparison_daily_vs_processed.csv")
        comparison.to_csv(comparison_output, index=False)
        print(f"\n✓ Comparison saved to {comparison_output}")
    
    return comparison


if __name__ == "__main__":
    # Get the script's directory and navigate to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Define paths
    grib_folder = os.path.join(project_root, "data", "raw", "era5")
    output_dir = os.path.join(project_root, "data", "processed")
    processed_csv = os.path.join(output_dir, "anomalies_sst_oni.csv")
    
    print(f"Project root: {project_root}")
    print("=" * 80)
    
    # Process daily SST data
    daily_results, monthly_results = process_daily_sst(grib_folder, output_dir)
    
    # Compare with processed monthly data
    comparison = compare_with_processed_data(monthly_results, processed_csv, output_dir)
    
    print('\n✓ All processes completed')
