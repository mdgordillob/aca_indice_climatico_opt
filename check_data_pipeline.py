"""
Diagnostic script to check which files exist and what's missing
"""
import os

def check_files():
    """Check if required files exist"""
    
    required_dirs = [
        'data/raw/era5',
        'data/processed',
        'data/processed/daily_by_year_tmp',
        'data/processed/daily_by_year_rain',
        'data/processed/daily_by_year_wind',
    ]
    
    required_files = [
        'data/processed/era5_daily_combined_tmp.nc',
        'data/processed/era5_daily_combined_rain.nc',
        'data/processed/era5_daily_combined_wind.nc',
        'data/processed/era5_temperatura_percentil.nc',
        'data/processed/era5_lluvia_percentil.nc',
        'data/processed/era5_wind_percentil.nc',
    ]
    
    print("=" * 60)
    print("DATA PIPELINE STATUS CHECK")
    print("=" * 60)
    
    print("\n📁 Directory Status:")
    for directory in required_dirs:
        exists = os.path.exists(directory)
        status = "✓ EXISTS" if exists else "✗ MISSING"
        print(f"  {status:12} - {directory}")
    
    print("\n📄 File Status:")
    for file in required_files:
        exists = os.path.exists(file)
        status = "✓ EXISTS" if exists else "✗ MISSING"
        size = f"({os.path.getsize(file) / 1024 / 1024:.1f} MB)" if exists else ""
        print(f"  {status:12} - {file} {size}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    
    raw_files = [f for f in os.listdir('data/raw/era5') if f.endswith('.grib')]
    if not raw_files:
        print("\n⚠️  NO RAW GRIB FILES FOUND!")
        print("   You must download ERA5 data first:")
        print("   1. Register at https://cds.climate.copernicus.eu/user/register")
        print("   2. Create ~/.cdsapirc with your API key")
        print("   3. Run: python src/scripts/ecmwf_descarga.py")
    else:
        print(f"\n✓ Found {len(raw_files)} raw GRIB files")
        print("  Next: Run python src/scripts/unir_archivos.py")

if __name__ == "__main__":
    check_files()