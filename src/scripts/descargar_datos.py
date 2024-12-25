from ecmwf_descarga import ERA5Downloader

## # Years to download
## years = ["1981", "1984"]
## 
## # Create a vector of the years to download
## years = [str(year) for year in range(1981, 1987)]
## 
## # Create a loop to download the data in gruo of 2 years
## for i in range(0, len(years), 2):
##     years_to_download = years[i:i+2]
##     print(f"Downloading data for {years_to_download}")
## 
##     # Download the data
##     downloader = ERA5Downloader()
##     downloader.download_precipitation(years_to_download, target_filename=f"era5_rain_{years_to_download[0]}_{years_to_download[-1]}.grib")
##     downloader.download_temperature(years_to_download, target_filename=f"era5_tmp_{years_to_download[0]}_{years_to_download[-1]}.grib")
##     downloader.download_wind(years_to_download, target_filename=f"era5_wind_{years_to_download[0]}_{years_to_download[-1]}.grib")

# Create a loop to download the data in gruo of 10 years
years = [str(year) for year in range(1961, 2024)]
area = [5.94, -74.99, 3.65, -72.78] # Colombia

for i in range(0, len(years), 10):
    years_to_download = years[i:i+10]
    print(f"Downloading data for {years_to_download}")

    # Download the data
    downloader = ERA5Downloader()
    downloader.download_precipitation(years_to_download, target_filename=f"era5_rain_{years_to_download[0]}_{years_to_download[-1]}.grib", area=area)
    downloader.download_temperature(years_to_download, target_filename=f"era5_tmp_{years_to_download[0]}_{years_to_download[-1]}.grib", area=area)
    downloader.download_wind(years_to_download, target_filename=f"era5_wind_{years_to_download[0]}_{years_to_download[-1]}.grib", area=area)