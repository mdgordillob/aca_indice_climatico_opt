import xarray as xr
import pdb
import os
import pandas as pd

def load_grid_data(file_path, variable):
    """
    Load grid data for a specific variable.
    """
    grid_data = xr.open_dataset(file_path, engine="cfgrib")[variable]
    return grid_data

## Temperature
def resample_to_daily_tmp(grid_data):
    """
    Resample hourly data to daily max and min.
    """
    # For Colombia we have to convert the time from UTC to UTC-5
    grid_data['time'] = grid_data.indexes['time'] - pd.Timedelta(hours=5)

    daily_max = grid_data.resample(time="1D").max(dim="time")
    daily_min = grid_data.resample(time="1D").min(dim="time")
    return daily_max, daily_min

def save_yearly_data_combined_tmp(daily_max, daily_min, year, output_dir, variable):
    """
    Save daily max and min data for a year to a single NetCDF file.
    """
    os.makedirs(output_dir, exist_ok=True)
    combined = xr.Dataset({
        f"daily_max": daily_max,
        f"daily_min": daily_min
    })
    output_file = os.path.join(output_dir, f"{variable}_daily_{year}.nc")
    combined.to_netcdf(output_file)
    print(f"Saved yearly file: {output_file}")

def process_yearly_data_tmp(file_path, year, variable, output_dir):
    """
    Process a single year's data: load, resample, and save combined max and min.
    """
    grid_data = load_grid_data(file_path, variable)
    daily_max, daily_min = resample_to_daily_tmp(grid_data)
    save_yearly_data_combined_tmp(daily_max, daily_min, year, output_dir, variable)

## Precipitation
def resample_to_daily_precipitation(grid_data):
    """
    Resample hourly precipitation data to daily sums.
    """
    #convertir tiempo a pandas datetime y ajustar a UTC-5 en Colombia
    grid_data['valid_time'] = grid_data['valid_time'] - pd.Timedelta(hours=5)

    #aplanar el grid_data combinando 'time' y 'step'
    grid_data_plain = grid_data.stack(valid_time_dim=("time", "step")).reset_index("valid_time_dim")

    #suma de precipitacion diaria por dia
    daily_sum = grid_data_plain.groupby(grid_data_plain["valid_time"].dt.floor("D")).sum(dim='valid_time_dim')

    # Renombrar la variable floor
    daily_sum = daily_sum.rename({"floor": "time"})

    return daily_sum

def save_yearly_precipitation_data(daily_sum, year, output_dir, variable):
    """
    Save daily precipitation data (sum) for a year to a single NetCDF file.
    """
    os.makedirs(output_dir, exist_ok=True)
    combined = xr.Dataset({
        f"{variable}_daily_sum": daily_sum
    })
    output_file = os.path.join(output_dir, f"{variable}_daily_{year}.nc")
    combined.to_netcdf(output_file)
    print(f"Saved yearly precipitation file: {output_file}")

def process_yearly_precipitation_data(file_path, year, variable, output_dir):
    """
    Process a single year's data: load, resample to daily sums, and save.
    """
    grid_data = load_grid_data(file_path, variable)
    daily_sum = resample_to_daily_precipitation(grid_data)
    save_yearly_precipitation_data(daily_sum, year, output_dir, variable)

## Wind
def resample_to_daily_wind(grid_data):
    """
    Resample hourly wind data to daily max.
    """
    # For Colombia we have to convert the time from UTC to UTC-5
    grid_data['time'] = grid_data.indexes['time'] - pd.Timedelta(hours=5)

    daily_mean = grid_data.resample(time="1D").mean(dim="time")
    
    # Calculate the wind speed from the U and V components
    daily_mean["wind_speed"] = (daily_mean["u10"]**2 + daily_mean["v10"]**2)**0.5

    # Return only the wind speed
    daily_mean = daily_mean.drop_vars(["u10", "v10"])

    return daily_mean


def save_yearly_wind_data(daily_mean, year, output_dir):
    """
    Save daily wind speed data for a year to a single NetCDF file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"wind_speed_daily_{year}.nc")
    daily_mean.to_netcdf(output_file)
    print(f"Saved yearly wind speed file: {output_file}")

def process_yearly_wind_data(file_path, year, output_dir, variable):
    """
    Process a single year's wind data: load, resample to daily, and save.
    """
    grid_data = load_grid_data(file_path, variable)
    daily_mean = resample_to_daily_wind(grid_data)
    save_yearly_wind_data(daily_mean, year, output_dir)


def merge_yearly_files(output_dir, merged_file, variable):
    """
    Merge all yearly NetCDF files into a single dataset.

    :param output_dir: Directory containing the yearly NetCDF files.
    :param merged_file: Path for the final merged NetCDF file.
    :param variable: The variable name (e.g., "t2m").
    """
    print(f"Merging files from: {output_dir}")
    
    # Check if directory exists and has files
    if not os.path.exists(output_dir):
        print(f"⚠️  Directory does not exist: {output_dir}")
        print(f"No files to merge. Skipping...")
        return
    
    # Get all yearly files in the directory
    nc_files = [f for f in os.listdir(output_dir) if f.endswith(".nc")]
    
    if not nc_files:
        print(f"⚠️  No .nc files found in {output_dir}")
        return
    
    print(f"Found {len(nc_files)} files to merge")
    
    yearly_files = sorted([os.path.join(output_dir, file) for file in nc_files])

    # Open all files into a list of datasets
    datasets = [xr.open_dataset(file) for file in yearly_files]

    # Concatenate datasets along the "time" dimension
    merged_dataset = xr.concat(datasets, dim="time")

    # Save the merged dataset to a single NetCDF file
    merged_dataset.to_netcdf(merged_file)
    print(f"✓ Merged dataset saved to: {merged_file}")

# Example Usage
if __name__ == "__main__":
    # Get the script's directory and navigate to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    ##### temperature
    ##input_dir_tmp = os.path.join(project_root, "data", "raw", "era5")
    ##output_dir_tmp = os.path.join(project_root, "data", "processed", "daily_by_year_tmp")
    ##merged_file_tmp = os.path.join(project_root, "data", "processed", "era5_daily_combined_tmp.nc")
    ##variable_tmp = "t2m"

    ### Process each year
    ##for year in range(1960, 1991):  # Adjust year range as needed
    ##    file_path = os.path.join(input_dir_tmp, f"era5_tmp_{year}.grib")
    ##    if os.path.exists(file_path):
    ##        process_yearly_data_tmp(file_path, year, variable_tmp, output_dir_tmp)
    
    ### Merge all yearly files into one
    ##merge_yearly_files(output_dir_tmp, merged_file_tmp, variable_tmp)

    ### rain
    input_dir_rain = os.path.join(project_root, "data", "raw", "era5")
    output_dir_rain = os.path.join(project_root, "data", "processed", "daily_by_year_rain")
    merged_file_rain = os.path.join(project_root, "data", "processed", "era5_daily_combined_rain.nc")
    variable_rain = "tp"

    print("=" * 60)
    print("PROCESSING RAINFALL DATA")
    print("=" * 60)
    
    # Process each year
    for year in range(1960, 1991):  # Adjust year range as needed
        file_path = os.path.join(input_dir_rain, f"era5_rain_{year}.grib")
        if os.path.exists(file_path):
            print(f"Processing {year}...")
            process_yearly_precipitation_data(file_path, year, variable_rain, output_dir_rain)
        else:
            print(f"⚠️  File not found: {file_path}")

    # Merge all yearly files into one
    print("\nMerging rainfall files...")
    merge_yearly_files(output_dir_rain, merged_file_rain, variable_rain)
    
    ####### wind
    ##input_dir_wind = os.path.join(project_root, "data", "raw", "era5")
    ##output_dir_wind = os.path.join(project_root, "data", "processed", "daily_by_year_wind")
    ##merged_file_wind = os.path.join(project_root, "data", "processed", "era5_daily_combined_wind.nc")
    ##variable_wind = ['u10', 'v10']

    #### Process each year
    ##for year in range(1960, 1991):  # Adjust year range as needed
    ##    file_path = os.path.join(input_dir_wind, f"era5_wind_{year}.grib")
    ##    if os.path.exists(file_path):
    ##        process_yearly_wind_data(file_path, year, output_dir_wind, variable_wind)

    #### Merge all yearly files into one
    ##merge_yearly_files(output_dir_wind, merged_file_wind, variable_wind)
    
    print("\n" + "=" * 60)
    print("✓ All processing complete!")
    print("=" * 60)