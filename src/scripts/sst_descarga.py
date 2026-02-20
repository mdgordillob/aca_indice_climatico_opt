import os
import cdsapi


class ERA5SSTDownloader:
    '''
    Class to download ERA5 Sea Surface Temperature data from the CDS API
    # Siguiendo la guía https://cds.climate.copernicus.eu/how-to-api
    '''
    def __init__(self, target_folder="../../data/raw/era5/", area=[13.0, -83.0, -4.6, -66.1]):
        self.target_folder = target_folder
        self.area = area
        self.client = cdsapi.Client()
        os.makedirs(self.target_folder, exist_ok=True)

    def download_sst(self, years, target_filename):
        dataset = "reanalysis-era5-single-levels"
        request = {
            "product_type": "reanalysis",
            "variable": ["sea_surface_temperature"],
            "year": years,
            "month": [f"{i:02d}" for i in range(1, 13)],
            "day": [f"{i:02d}" for i in range(1, 32)],
            "time": [f"{i:02d}:00" for i in range(24)],
            "format": "grib",
            "area": self.area
        }
        target_path = os.path.join(self.target_folder, target_filename)
        self.client.retrieve(dataset, request).download(target_path)
        print(f"Data downloaded to {target_path}")


# Example usage
if __name__ == "__main__":
    downloader = ERA5SSTDownloader()
    for year in range(1961, 2025):
        downloader.download_sst([str(year)], f"era5_sst_{year}.grib")