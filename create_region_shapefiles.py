"""
Creates shapefiles for Colombian natural regions not yet in data/shapefiles/:
  - pacifico_4326.shp    (Pacific coastal region / Chocó Biogeographic)
  - amazonas_4326.shp    (Colombian Amazon basin)

Polygon coordinates are approximate natural-region boundaries clipped to the
ERA5-Land data extent (LAT -4.6 to 12.9, LON -83.0 to -66.25).
"""

from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon

SHPDIR = Path(__file__).resolve().parent / "data" / "shapefiles"
SHPDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Pacific (Pacífico) – western coastal strip, Chocó bioregion + Nariño coast
# Runs from the Darién border south to the Ecuadorian border, staying west of
# the Western Cordillera.  Approximate polygon in WGS-84.
# ---------------------------------------------------------------------------
pacific_coords = [
    (-80.5,  8.5),   # NW corner (gulf of Urabá / Darién)
    (-76.5,  8.5),   # NE limit (where coast meets Andes foothills)
    (-76.5,  7.0),   # east side, Chocó highland limit
    (-76.8,  5.5),   # east, towards Cordillera Occidental
    (-77.2,  4.0),   # east, Valle del Cauca coastal side
    (-77.3,  2.5),   # east, Cauca coastal fringe
    (-77.2,  1.0),   # east, Nariño / Patía river
    (-77.5, -0.5),   # east, approaching Ecuador border
    (-78.5, -1.5),   # SE corner (Colombia-Ecuador border, Pacific side)
    (-80.5, -1.5),   # SW corner
    (-80.5,  8.5),   # close polygon
]

# ---------------------------------------------------------------------------
# Amazonas – Colombian Amazon basin
# East of the Andes, south of Orinoquía, covering Caquetá, Putumayo,
# Amazonas, Vaupés, Guainía departments.
# ---------------------------------------------------------------------------
amazonas_coords = [
    (-75.5,  2.0),   # NW (where Caquetá meets Orinoquía / Andes)
    (-67.0,  2.0),   # NE (Colombia-Venezuela-Brazil junction)
    (-67.0, -4.5),   # SE (deep Amazon, Colombia-Brazil-Peru border)
    (-75.5, -4.5),   # SW (Putumayo/Ecuador/Peru border)
    (-75.5,  2.0),   # close polygon
]


def save_region(name: str, coords: list, crs: str = "EPSG:4326"):
    out = SHPDIR / f"{name}_4326.shp"
    if out.exists():
        print(f"  {out.name} already exists – skipping")
        return
    poly = Polygon(coords)
    gdf  = gpd.GeoDataFrame({"region": [name]}, geometry=[poly], crs=crs)
    gdf.to_file(out)
    print(f"  Saved {out}")


if __name__ == "__main__":
    print("Creating regional shapefiles...")
    save_region("pacifico",  pacific_coords)
    save_region("amazonas",  amazonas_coords)
    print("Done.")
