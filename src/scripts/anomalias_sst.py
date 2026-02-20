import os
import xarray as xr
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def load_sst_data(file_path):
    """
    Load SST data from netCDF file.
    
    Parameters:
    - file_path (str): Path to the netCDF file containing SST data.
    
    Returns:
    - sst_data (xarray.DataArray): SST data.
    """
    dataset = xr.open_dataset(file_path)
    
    # Try to find the SST variable (common names: sst, skt, t2m, skin_temperature)
    sst_var = None
    possible_vars = ['sst', 'skt', 'skin_temperature', 'sea_surface_temperature']
    
    for var in possible_vars:
        if var in dataset.data_vars:
            sst_var = var
            break
    
    if sst_var is None:
        # If none of the common names found, use the first data variable
        sst_var = list(dataset.data_vars)[0]
        print(f"⚠️  SST variable name not recognized. Using '{sst_var}'")
    
    sst_data = dataset[sst_var]
    
    return sst_data


def calculate_sst_statistics(sst_data):
    """
    Calculate climatological mean and standard deviation for each month.
    
    Parameters:
    - sst_data (xarray.DataArray): Monthly SST anomalies or raw SST data.
    
    Returns:
    - mean_sst (xarray.DataArray): Climatological mean by month.
    - std_sst (xarray.DataArray): Climatological standard deviation by month.
    """
    # Determine time dimension name
    time_dim = 'valid_time' if 'valid_time' in sst_data.dims else 'time'
    
    mean_sst = sst_data.groupby(f'{time_dim}.month').mean(dim=time_dim)
    std_sst = sst_data.groupby(f'{time_dim}.month').std(dim=time_dim)
    
    return mean_sst, std_sst


def calculate_sst_anomalies(sst_data):
    """
    Calculate SST anomalies by subtracting climatological mean.
    
    Parameters:
    - sst_data (xarray.DataArray): SST data.
    
    Returns:
    - sst_anomalies (xarray.DataArray): SST anomalies.
    """
    # Determine time dimension name
    time_dim = 'valid_time' if 'valid_time' in sst_data.dims else 'time'
    
    # Get climatological mean for each month
    mean_sst = sst_data.groupby(f'{time_dim}.month').mean(dim=time_dim)
    
    # Calculate anomalies
    sst_anomalies = sst_data.groupby(f'{time_dim}.month') - mean_sst
    
    return sst_anomalies


def calculate_oni_index(sst_anomalies, window=3):
    """
    Calculate ONI (Oceanic Niño Index) as a 3-month rolling mean of SST anomalies.
    
    Parameters:
    - sst_anomalies (xarray.DataArray): SST anomalies.
    - window (int): Rolling window size (default 3 months for ONI).
    
    Returns:
    - oni_index (xarray.DataArray): ONI index values.
    """
    # Determine time dimension name
    time_dim = 'valid_time' if 'valid_time' in sst_anomalies.dims else 'time'
    
    # Calculate rolling mean
    oni_index = sst_anomalies.rolling({time_dim: window}, center=True).mean()
    
    return oni_index


def process_sst_anomalies(sst_file_path, output_dir=None):
    """
    Main processing function: load SST data, calculate anomalies and ONI index.
    
    Parameters:
    - sst_file_path (str): Path to SST netCDF file.
    - output_dir (str): Output directory for CSV file.
    
    Returns:
    - results_df (pandas.DataFrame): DataFrame with time, sst_anomaly, and oni_index.
    """
    
    print("=" * 60)
    print("CALCULATING SST ANOMALIES AND ONI INDEX")
    print("=" * 60)
    print(f"Loading SST data from: {sst_file_path}")
    
    # Load SST data
    sst_data = load_sst_data(sst_file_path)
    
    # Determine time dimension name
    time_dim = 'valid_time' if 'valid_time' in sst_data.dims else 'time'
    
    print(f"SST data shape: {sst_data.shape}")
    print(f"SST data time range: {sst_data[time_dim].values[0]} to {sst_data[time_dim].values[-1]}")
    
    # Calculate SST anomalies
    print("\nCalculating SST anomalies...")
    sst_anomalies = calculate_sst_anomalies(sst_data)
    
    # Calculate mean anomalies across spatial dimensions (if they exist)
    if 'latitude' in sst_anomalies.dims and 'longitude' in sst_anomalies.dims:
        print("Averaging across spatial dimensions...")
        sst_anomalies_mean = sst_anomalies.mean(dim=['latitude', 'longitude'])
    else:
        sst_anomalies_mean = sst_anomalies
    
    # Calculate ONI index (3-month rolling mean)
    print("Calculating ONI index (3-month rolling mean)...")
    oni_index = calculate_oni_index(sst_anomalies_mean)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'time': sst_anomalies_mean[time_dim].values,
        'sst_anomaly': sst_anomalies_mean.values,
        'oni_colombia_pacific': oni_index.values
    })
    
    # Convert time to datetime
    results_df['time'] = pd.to_datetime(results_df['time'])
    
    # Remove NaN values that resulted from rolling calculation
    results_df = results_df.dropna()
    
    # Save to CSV if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "anomalies_sst_oni.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to {output_path}")
        print(f"  Records: {len(results_df)}")
        print(f"  Time range: {results_df['time'].min()} to {results_df['time'].max()}")
    
    return results_df


if __name__ == "__main__":
    # Get the script's directory and navigate to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Define paths
    sst_file = os.path.join(project_root, "colombia_pacific_sst_era5.nc")
    output_dir = os.path.join(project_root, "data", "processed")
    
    print(f"Project root: {project_root}")
    print("=" * 60)
    
    # Process SST anomalies
    results = process_sst_anomalies(sst_file, output_dir)
    
    print("\n✓ Process completed")
    print(f"\nFirst 5 records:")
    print(results.head())
    print(f"\nLast 5 records:")
    print(results.tail())
