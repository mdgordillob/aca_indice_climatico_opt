import os
import cdsapi

def download_era5_temperature(years, target_folder="../../data/raw/era5/", target_filename="era5_2m_temperature.grib"):
    """
    Download ERA5 2m temperature data for specified years.

    Parameters:
    years (list): List of years to download data for.
    target_folder (str): Folder to save the downloaded data.
    target_filename (str): Filename for the downloaded data.
    """
    # Ensure the folder exists
    os.makedirs(target_folder, exist_ok=True)

    # Define the dataset and request
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": years,
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [4.0, -73.0, 3.0, -72.0]
    }

    # Create the CDS API client and download the data
    client = cdsapi.Client()
    target_path = os.path.join(target_folder, target_filename)
    client.retrieve(dataset, request).download(target_path)

    print(f"Data downloaded to {target_path}")

# Example usage
years = ["1981", "1982"]
download_era5_temperature(years, target_filename="era5_2m_temperature_81_82.grib")

years = ["1983", "1984"]
download_era5_temperature(years, target_filename="era5_2m_temperature_83_84.grib")