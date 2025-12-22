import os
import xarray as xr
import geopandas as gpd
import pandas as pd
import rioxarray as rio
import warnings
warnings.filterwarnings('ignore')

def check_grib_files(data_dir):
    """Check available GRIB files and their content."""
    print("=" * 80)
    print("CHECKING GRIB FILES")
    print("=" * 80)
    
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return
    
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.grib')])
    
    if not files:
        print(f"❌ No GRIB files found in {data_dir}")
        return
    
    print(f"✓ Found {len(files)} GRIB files\n")
    
    # Group by year
    years_found = {}
    for file in files:
        for year in range(1961, 2025):
            if str(year) in file:
                if year not in years_found:
                    years_found[year] = []
                years_found[year].append(file)
                break
    
    # Check for 2003 specifically
    print("📊 YEARS WITH DATA:")
    for year in sorted(years_found.keys()):
        print(f"  {year}: {len(years_found[year])} file(s)")
    
    if 2003 in years_found:
        print(f"\n✓ 2003 data found:")
        for file in years_found[2003]:
            print(f"  - {file}")
            file_path = os.path.join(data_dir, file)
            try:
                ds = xr.open_dataset(file_path, engine='cfgrib')
                print(f"    Variables: {list(ds.data_vars)}")
                print(f"    Dimensions: {dict(ds.dims)}")
                if 'time' in ds.dims:
                    print(f"    Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
                if 'u10' in ds.data_vars and 'v10' in ds.data_vars:
                    print(f"    ✓ Has wind components (u10, v10)")
                elif 'wind_speed' in ds.data_vars:
                    print(f"    ✓ Has wind_speed variable")
                else:
                    print(f"    ❌ No wind variables found")
            except Exception as e:
                print(f"    ❌ Error reading file: {e}")
    else:
        print(f"\n❌ No 2003 data found!")
        print(f"   Available years: {sorted(years_found.keys())}")


def check_shapefiles(shapefile_dir):
    """Check shapefile validity and bounds."""
    print("\n" + "=" * 80)
    print("CHECKING SHAPEFILES")
    print("=" * 80)
    
    if not os.path.exists(shapefile_dir):
        print(f"❌ Shapefile directory not found: {shapefile_dir}")
        return
    
    shapefiles = sorted([f for f in os.listdir(shapefile_dir) if f.endswith('.shp')])
    
    if not shapefiles:
        print(f"❌ No shapefiles found in {shapefile_dir}")
        return
    
    print(f"✓ Found {len(shapefiles)} shapefiles\n")
    
    for shapefile in shapefiles:
        shapefile_path = os.path.join(shapefile_dir, shapefile)
        print(f"📍 {shapefile}:")
        try:
            gdf = gpd.read_file(shapefile_path)
            print(f"   CRS: {gdf.crs}")
            bounds = gdf.total_bounds
            print(f"   Bounds: minx={bounds[0]:.2f}, miny={bounds[1]:.2f}, maxx={bounds[2]:.2f}, maxy={bounds[3]:.2f}")
            print(f"   Geometry type(s): {gdf.geometry.type.unique().tolist()}")
            print(f"   Features: {len(gdf)}")
            
            # Check if bounds are reasonable for Colombia
            if bounds[0] < -85 or bounds[2] > -60 or bounds[1] < -6 or bounds[3] > 14:
                print(f"   ⚠️  Bounds seem reasonable for region")
            else:
                print(f"   ⚠️  Bounds might be unexpected")
        except Exception as e:
            print(f"   ❌ Error reading shapefile: {e}")


def check_wind_data_overlap(data_dir, shapefile_path):
    """Check if wind data overlaps with shapefile bounds."""
    print("\n" + "=" * 80)
    print("CHECKING DATA-SHAPEFILE OVERLAP")
    print("=" * 80)
    
    try:
        # Load shapefile
        gdf = gpd.read_file(shapefile_path)
        shape_bounds = gdf.total_bounds
        print(f"\n📍 Shapefile bounds:")
        print(f"   Longitude: [{shape_bounds[0]:.2f}, {shape_bounds[2]:.2f}]")
        print(f"   Latitude: [{shape_bounds[1]:.2f}, {shape_bounds[3]:.2f}]")
        
        # Check for 2003 wind file
        wind_2003 = None
        for file in os.listdir(data_dir):
            if '2003' in file and 'wind' in file and file.endswith('.grib'):
                wind_2003 = file
                break
        
        if not wind_2003:
            print(f"\n❌ No 2003 wind file found")
            return
        
        print(f"\n🌪️  Testing file: {wind_2003}")
        file_path = os.path.join(data_dir, wind_2003)
        
        try:
            ds = xr.open_dataset(file_path, engine='cfgrib')
            
            # Get grid bounds
            if 'longitude' in ds.coords and 'latitude' in ds.coords:
                lon_min = float(ds.longitude.min())
                lon_max = float(ds.longitude.max())
                lat_min = float(ds.latitude.min())
                lat_max = float(ds.latitude.max())
                
                print(f"\n   Grid bounds:")
                print(f"   Longitude: [{lon_min:.2f}, {lon_max:.2f}]")
                print(f"   Latitude: [{lat_min:.2f}, {lat_max:.2f}]")
                
                # Check overlap
                lon_overlap = not (lon_max < shape_bounds[0] or lon_min > shape_bounds[2])
                lat_overlap = not (lat_max < shape_bounds[1] or lat_min > shape_bounds[3])
                
                if lon_overlap and lat_overlap:
                    print(f"   ✓ Grid and shapefile overlap correctly")
                else:
                    print(f"   ❌ NO OVERLAP DETECTED!")
                    print(f"      Longitude overlap: {lon_overlap}")
                    print(f"      Latitude overlap: {lat_overlap}")
            else:
                print(f"   ⚠️  Could not determine grid bounds")
        
        except Exception as e:
            print(f"   ❌ Error reading wind file: {e}")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    data_dir = os.path.join(project_root, "data", "raw", "era5")
    shapefile_dir = os.path.join(project_root, "data", "shapefiles")
    shapefile_path = os.path.join(shapefile_dir, "colombia_4326.shp")
    
    check_grib_files(data_dir)
    check_shapefiles(shapefile_dir)
    check_wind_data_overlap(data_dir, shapefile_path)
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()