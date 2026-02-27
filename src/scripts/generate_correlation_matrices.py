"""
generate_correlation_matrices.py
────────────────────────────────
Script to generate correlation matrix plots for each region, including regional anomalies (temperature, wind, precipitation, drought),
Colombian Pacific SST index, and NOAA ONI.
Computes a correlation matrix for each region and visualizes it as a heatmap.

Inputs:
- Regional anomaly CSVs: data/processed/<region_name>/*.csv
- SST index: data/processed/sst_index_colombia_pacific.csv
- ONI data: data/processed/oni.ascii.txt

Outputs:
- Heatmap plots: data/processed/<region_name>/correlation_matrix.png
- Console output with correlation matrices
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Two levels up → project root
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
SST_CSV = os.path.join(PROCESSED_DIR, 'sst_index_colombia_pacific.csv')
ONI_FILE = os.path.join(PROCESSED_DIR, 'oni.ascii.txt')

print(f"Project root: {PROJECT_ROOT}")
print(f"Processed dir: {PROCESSED_DIR}")
print(f"SST file expected: {SST_CSV}")
print(f"ONI file expected: {ONI_FILE}\n")

# ── ONI parser ────────────────────────────────────────────────────────────────
def load_oni():
    if not os.path.exists(ONI_FILE):
        raise FileNotFoundError(f"ONI file not found: {ONI_FILE}")
    
    df = pd.read_csv(ONI_FILE, sep=r'\s+', engine='python')
    df.columns = [c.strip() for c in df.columns]
    SEAS_MAP = {'DJF':1, 'JFM':2, 'FMA':3, 'MAM':4, 'AMJ':5, 'MJJ':6,
                'JJA':7, 'JAS':8, 'ASO':9, 'SON':10, 'OND':11, 'NDJ':12}
    df['month'] = df['SEAS'].map(SEAS_MAP)
    df = df.dropna(subset=['month'])
    df['YR']   = pd.to_numeric(df['YR'], errors='coerce').astype('Int64')
    df['ANOM'] = pd.to_numeric(df['ANOM'], errors='coerce')
    df['month'] = df['month'].astype(int)
    df.loc[df['SEAS']=='NDJ','YR'] += 1
    
    # Safe datetime parsing
    df['time'] = pd.to_datetime(
        df['YR'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01',
        errors='coerce'
    )
    df = df.dropna(subset=['time', 'ANOM'])
    df['time'] = df['time'].dt.normalize()
    
    return df[['time', 'ANOM']].rename(columns={'ANOM': 'oni_anom'})

# ── Load SST index ────────────────────────────────────────────────────────────
def load_sst():
    if not os.path.exists(SST_CSV):
        raise FileNotFoundError(f"SST file not found: {SST_CSV}")
    
    df = pd.read_csv(SST_CSV)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    df['time'] = df['time'].dt.normalize()
    
    if 'sst_index' not in df.columns:
        raise ValueError(f"No 'sst_index' column found in {SST_CSV}")
    
    return df[['time', 'sst_index']]

# ── Load regional anomaly CSV with robust date cleaning ───────────────────────
def load_regional_anomaly(region_name, signal):
    file_map = {
        'temperature': 'anomalies_temperature_combined.csv',
        'wind': 'anomalies_wind_combined.csv',
        'precipitation': 'anomalies_precipitation_combined.csv',
        'drought': 'anomalies_drought_combined.csv'
    }

    file_name = file_map.get(signal)
    if not file_name:
        raise ValueError(f"Invalid signal: {signal}")

    path = os.path.join(PROCESSED_DIR, region_name, file_name)
    if not os.path.exists(path):
        print(f"  Warning: {path} not found. Skipping {signal}.")
        return None

    print(f"  Loading: {path}")
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # ── Build time column robustly ───────────────────────────

    # Case 1: Spanish columns
    if {'Año', 'Mes'}.issubset(df.columns):
        print(f"    Building time from Año + Mes")
        df['time'] = pd.to_datetime(
            df['Año'].astype(str) + '-' +
            df['Mes'].astype(str).str.zfill(2) + '-01',
            errors='coerce'
        )

    # Case 2: English columns
    elif {'year', 'month'}.issubset(df.columns):
        print(f"    Building time from year + month")
        df['time'] = pd.to_datetime(
            df['year'].astype(str) + '-' +
            df['month'].astype(str).str.zfill(2) + '-01',
            errors='coerce'
        )

    # Case 3: Already valid time column
    elif 'time' in df.columns:
        print(f"    Parsing existing time column")
        df['time'] = pd.to_datetime(df['time'], errors='coerce')

    else:
        print(f"  No usable date columns found. Skipping.")
        return None

    df = df.dropna(subset=['time'])

    if df.empty:
        print(f"    No valid dates after parsing. Skipping.")
        return None

    df['time'] = df['time'].dt.normalize()

    # ── Detect anomaly column automatically ─────────────────

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Remove year/month from candidate anomaly columns
    exclude_cols = ['Año', 'Mes', 'year', 'month']
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    if not numeric_cols:
        print(f"    No numeric anomaly column found.")
        return None

    anomaly_col = numeric_cols[0]
    print(f"    Using column '{anomaly_col}' as anomaly")

    df = df[['time', anomaly_col]].rename(
        columns={anomaly_col: f'{signal}_anomaly'}
    )

    return df

# ── Generate correlation matrix for a region ──────────────────────────────────
def generate_matrix_for_region(region_name, signals=['temperature', 'wind', 'precipitation', 'drought']):
    print(f'\n=== Processing region: {region_name} ===')
    
    try:
        # Load global data
        sst_df = load_sst()
        oni_df = load_oni()
        
        # Load regional signals
        regional_dfs = []
        for signal in signals:
            df = load_regional_anomaly(region_name, signal)
            if df is not None:
                regional_dfs.append(df)
        
        if not regional_dfs:
            print(f'  No valid regional data found for {region_name}. Skipping.')
            return
        
        # Merge all
        merged = sst_df.merge(oni_df, on='time', how='inner')
        for df in regional_dfs:
            merged = merged.merge(df, on='time', how='inner')
        
        merged = merged.dropna()
        if merged.empty:
            print(f'  No overlapping valid data after merging for {region_name}. Skipping.')
            return
        
        # Select columns for correlation
        corr_cols = [col for col in merged.columns if col.endswith('_anomaly') or col in ['sst_index', 'oni_anom']]
        corr_df = merged[corr_cols]
        
        # Compute correlation matrix
        corr_matrix = corr_df.corr()
        print(f'\nCorrelation Matrix for {region_name}:')
        print(corr_matrix.round(3).to_string())
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                    fmt='.2f', linewidths=0.5, annot_kws={'size': 9})
        plt.title(f'Correlation Matrix: {region_name} Signals vs. SST & ONI')
        output_dir = os.path.join(PROCESSED_DIR, region_name)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, 'correlation_matrix.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  → Plot saved: {out_path}')
    
    except Exception as e:
        print(f'  Error processing {region_name}: {str(e)}')

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    regions = [
        "anomalias_colombia",
        "anomalias_cundinamarca_bogota",
        "anomalias_antioquia",
        "anomalias_valle_cauca",
        "anomalias_san_andres_providencia",
        "anomalias_medellin",
        "anomalias_cali",
        "anomalias_bogota"
    ]
    
    for region in regions:
        generate_matrix_for_region(region)
    
    print('\nAll done! Check outputs in data/processed/<region_name>/')