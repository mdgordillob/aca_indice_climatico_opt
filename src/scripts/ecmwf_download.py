import os
import cdsapi
import pdb

# Siguiendo la guía https://cds.climate.copernicus.eu/how-to-api 

# Specify the target folder and filename
target_folder = "../../data/raw/era5/"  # Replace with your desired folder path
target_filename = "era5_2m_temperature_1961_1971.grib"  # Change the filename if needed
target_path = os.path.join(target_folder, target_filename)

# Ensure the folder exists
os.makedirs(target_folder, exist_ok=True)

# Define the dataset and request
dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": ["2m_temperature"],
    "year": [
        "1961", "1962", "1963",
        "1964", "1965", "1966",
        "1967", "1968", "1969",
        "1970", "1971"
    ],
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
    "area": [13.77, -82.04, -5.68, -66.68]
}

# Create the CDS API client and download the data
client = cdsapi.Client()
client.retrieve(dataset, request).download(target_path)

print(f"Data downloaded to {target_path}")
