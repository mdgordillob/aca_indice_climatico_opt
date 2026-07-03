import os
import re

import geopandas as gpd

_shapefile_cache = {}


def get_available_years(directory, keyword):
    """Return sorted list of years found in GRIB files matching keyword."""
    years = set()
    for f in os.listdir(directory):
        if keyword in f and f.endswith('.grib'):
            m = re.search(r'(\d{4})', f)
            if m:
                years.add(int(m.group(1)))
    return sorted(years)


def get_cached_shapefile(shapefile_path):
    """Load shapefile once and cache it."""
    if shapefile_path not in _shapefile_cache:
        _shapefile_cache[shapefile_path] = gpd.read_file(shapefile_path)
    return _shapefile_cache[shapefile_path]
