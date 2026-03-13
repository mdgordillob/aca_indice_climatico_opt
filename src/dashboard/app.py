import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path
from PIL import Image

# ── 1. Page Configuration & Custom CSS ──────────────────────────────────────
st.set_page_config(page_title="Climate Actuarial Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 95%; }
    header {visibility: hidden;} footer {visibility: hidden;}
    .stTabs[data-baseweb="tab-list"] { gap: 8px; background-color: transparent; }
    .stTabs[data-baseweb="tab"] {
        padding-top: 10px; padding-bottom: 10px; padding-left: 20px; padding-right: 20px;
        border-radius: 8px; background-color: #f1f5f9; border: 1px solid #e2e8f0; color: #475569;
    }
    .stTabs[aria-selected="true"] { background-color: #2563eb !important; color: white !important; border: 1px solid #2563eb; }
    [data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 800; color: #1e40af; }
    h1, h2, h3, h4 { font-family: 'Inter', sans-serif; font-weight: 600; color: #0f172a; }
</style>
""", unsafe_allow_html=True)

# ── 2. Helper: Plotly Theme Engine ──────────────────────────────────────────
def apply_sleek_theme(fig):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#1e293b"),
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zerolinecolor="#cbd5e1"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zerolinecolor="#cbd5e1"),
        margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ── 3. Paths & Constants ───────────────────────────────────────────────────
# Locally:  set env var CLIMATE_BASE=C:\Users\mdgor\aca\aca_indice_climatico_opt-main
#           run once in terminal: setx CLIMATE_BASE "C:\Users\mdgor\aca\aca_indice_climatico_opt-main"
# On HF Spaces: env var is absent → falls back to relative path next to app.py
BASE = Path(os.environ.get("CLIMATE_BASE", Path(__file__).parent))
PROC = BASE / "data" / "processed"
RAW  = BASE / "data" / "raw"

REGION_FOLDERS = {
    "Colombia": "anomalias_colombia",
    "Antioquia": "anomalias_antioquia",
    "Cundinamarca & Bogotá": "anomalias_cundinamarca_bogota",
    "Valle del Cauca": "anomalias_valle_cauca",
    "Cali": "anomalias_cali",
    "Medellín": "anomalias_medellin",
    "Bogotá": "anomalias_bogota",
    "San Andrés & Providencia": "anomalias_san_andres_providencia"
}

REGION_COORDS = {
    "Colombia": {"lat": 4.5709, "lon": -74.2973, "zoom": 4.5},
    "Antioquia": {"lat": 6.2518, "lon": -75.5636, "zoom": 6.5},
    "Cundinamarca & Bogotá": {"lat": 4.6097, "lon": -74.0817, "zoom": 7.5},
    "Valle del Cauca": {"lat": 3.8, "lon": -76.5, "zoom": 7.5},
    "Cali": {"lat": 3.4516, "lon": -76.5320, "zoom": 9.5},
    "Medellín": {"lat": 6.2442, "lon": -75.5812, "zoom": 10},
    "Bogotá": {"lat": 4.7110, "lon": -74.0721, "zoom": 10},
    "San Andrés & Providencia": {"lat": 12.5847, "lon": -81.7006, "zoom": 9}
}

# ── 4. Data Loading ────────────────────────────────────────────────────────
@st.cache_data
def load_enso_data():
    try:
        sst = pd.read_csv(PROC / 'sst_index_colombia_pacific.csv', parse_dates=['time'])
        oni_path = PROC / 'oni.ascii.txt' if (PROC / 'oni.ascii.txt').exists() else RAW / 'oni.ascii.txt'
        oni_raw = pd.read_csv(oni_path, sep=r'\s+', engine='python')
        oni_raw['MON'] = oni_raw['SEAS'].map({'DJF':1,'JFM':2,'FMA':3,'MAM':4,'AMJ':5,'MJJ':6,'JJA':7,'JAS':8,'ASO':9,'SON':10,'OND':11,'NDJ':12})
        oni_raw['YR'] = pd.to_numeric(oni_raw['YR'], errors='coerce')
        oni_raw.loc[oni_raw['SEAS'] == 'NDJ', 'YR'] += 1
        oni_raw['time'] = pd.to_datetime(oni_raw['YR'].astype(str) + '-' + oni_raw['MON'].astype(str) + '-01')
        oni = oni_raw.rename(columns={'ANOM': 'oni'})[['time', 'oni']].dropna()
        df = sst.merge(oni, on='time', how='inner').dropna(subset=['sst_index', 'oni'])
        df['year'] = df['time'].dt.year; df['decade'] = (df['year'] // 10) * 10
        return df
    except Exception as e:
        st.warning(f"ENSO data load error: {e}")
        return pd.DataFrame()

@st.cache_data
def load_tuning_data():
    try:
        flags = pd.read_csv(PROC / 'enso_flag_comparison.csv', parse_dates=['time'])
        tuning = pd.read_csv(PROC / 'threshold_search_results.csv')
        return flags, tuning
    except Exception as e:
        st.warning(f"Tuning data load error: {e}")
        return None, None

def _read_and_clean_csv(filepath, cols_to_keep):
    p = Path(filepath)
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(p, encoding='latin-1')   # fallback for files saved on Windows
    df.columns = [c.strip() for c in df.columns]

    if {'Año', 'Mes'}.issubset(df.columns):
        df.rename(columns={'Año': 'year', 'Mes': 'month'}, inplace=True)
    if not {'year', 'month'}.issubset(df.columns):
        return pd.DataFrame()

    df['time'] = pd.to_datetime(df[['year', 'month']].assign(day=1), errors='coerce')
    df = df.dropna(subset=['time'])

    keep = ['time'] + [c for c in cols_to_keep if c in df.columns]
    return df[keep]

@st.cache_data
def load_regional_ica_data(region_key):
    folder_name = REGION_FOLDERS.get(region_key)
    if not folder_name:
        return pd.DataFrame()
    region_dir = PROC / folder_name

    # ── Path diagnostic: shown only when folder or files are missing ──
    if not region_dir.exists():
        st.error(f"📂 Folder not found: `{region_dir}`\n\nCheck that the folder exists and `CLIMATE_BASE` is set correctly.")
        return pd.DataFrame()

    found_files = list(region_dir.glob("*.csv"))
    if not found_files:
        st.warning(f"📄 Folder exists but contains no CSV files: `{region_dir}`")
        return pd.DataFrame()

    df_temp   = _read_and_clean_csv(region_dir / 'anomalies_temperature_combined.csv', ['t_90', 't_10'])
    df_rain   = _read_and_clean_csv(region_dir / 'anomalies_precipitation_combined.csv', ['Anomalia_Lluvia'])
    df_drought = _read_and_clean_csv(region_dir / 'anomalies_drought_combined.csv', ['Anomalia_Sequia'])
    df_wind   = _read_and_clean_csv(region_dir / 'anomalies_wind_combined.csv', ['anomalies_above'])

    if df_temp.empty and df_rain.empty and df_drought.empty and df_wind.empty:
        file_list = [f.name for f in found_files]
        st.warning(
            f"No usable data found in: {region_dir}. "
            f"Files present: {file_list}. "
            "Expected: anomalies_temperature_combined.csv, anomalies_precipitation_combined.csv, "
            "anomalies_drought_combined.csv, anomalies_wind_combined.csv"
        )
        return pd.DataFrame()

    master = pd.DataFrame({'time': pd.date_range(start='1950-01-01', end='2030-01-01', freq='MS')})
    if not df_temp.empty:   master = master.merge(df_temp,   on='time', how='left')
    if not df_rain.empty:   master = master.merge(df_rain,   on='time', how='left')
    if not df_drought.empty: master = master.merge(df_drought, on='time', how='left')
    if not df_wind.empty:   master = master.merge(df_wind,   on='time', how='left')

    master = master.dropna(how='all', subset=['t_90', 't_10', 'Anomalia_Lluvia', 'Anomalia_Sequia', 'anomalies_above'])
    master = master.sort_values('time').reset_index(drop=True)

    if {'t_90', 't_10'}.issubset(master.columns):
        master['temp_composite'] = master['t_90'] - master['t_10']

    req_cols = ['t_90', 't_10', 'anomalies_above', 'Anomalia_Lluvia', 'Anomalia_Sequia']
    if set(req_cols).issubset(master.columns):
        master['ICA'] = (master['t_90'] - master['t_10'] + master['anomalies_above'] + master['Anomalia_Lluvia'] + master['Anomalia_Sequia']) / 5

    cols_to_roll = ['t_90', 't_10', 'temp_composite', 'Anomalia_Lluvia', 'Anomalia_Sequia', 'anomalies_above', 'ICA']
    for c in cols_to_roll:
        if c in master.columns:
            master[f'{c}_5yr_avg'] = master[c].rolling(window=60, center=False).mean()

    return master

# ── 5. Sidebar Navigation ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/21/Flag_of_Colombia.svg", width=80)
    st.markdown("## Domain Selection")
    selected_region = st.selectbox("Target Region", list(REGION_FOLDERS.keys()))

    st.markdown("### Geospatial Reference")
    with st.container(border=True):
        try:
            coords = REGION_COORDS.get(selected_region, REGION_COORDS["Colombia"])
            map_data = pd.DataFrame([coords])
            fig_map = px.scatter_map(map_data, lat="lat", lon="lon", zoom=coords["zoom"],
                                        map_style="open-street-map", height=250)
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)")
            fig_map.update_traces(marker=dict(size=18, color="#2563eb", opacity=0.8))
            st.plotly_chart(fig_map, width='stretch')
        except Exception as e:
            st.caption(f"Map unavailable: {e}")

# ── 6. Header & KPI Ribbon ─────────────────────────────────────────────────
st.markdown(f"<h1>Regional Analysis: <span style='color:#2563eb;'>{selected_region}</span></h1>", unsafe_allow_html=True)
df_enso = load_enso_data()

if not df_enso.empty:
    latest = df_enso.iloc[-1]
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Record Date", latest['time'].strftime('%b %Y'))
        col2.metric("Local SST Index", f"{latest['sst_index']:+.2f} °C")
        col3.metric("NOAA ONI", f"{latest['oni']:+.2f} °C")
        col4.metric("SST/ONI Correlation", f"{df_enso['sst_index'].corr(df_enso['oni']):.3f}")

# ── 7. Main Application Tabs ───────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    "Regional Anomalies & ICA",
    "SST vs. ONI Analysis",
    "Signal Decoupling",
    "Actuarial Tuning"
])

# ═════ TAB 1: REGIONAL ANOMALIES & ICA ═════════════════════════════════════
with t1:
    df_reg = load_regional_ica_data(selected_region)

    if not df_reg.empty:
        # --- 1. Actuarial Climate Index (ICA) ---
        if 'ICA' in df_reg.columns:
            with st.container(border=True):
                st.markdown("### Actuarial Climate Index (ICA) — 5-Year Moving Average")
                fig_ica = go.Figure()
                fig_ica.add_trace(go.Scatter(x=df_reg['time'], y=df_reg['t_90_5yr_avg'], name='T90 (5-yr Avg)', line=dict(color='#64748b', width=2)))
                fig_ica.add_trace(go.Scatter(x=df_reg['time'], y=df_reg['t_10_5yr_avg'], name='T10 (5-yr Avg)', line=dict(color='#94a3b8', width=2)))
                fig_ica.add_trace(go.Scatter(x=df_reg['time'], y=df_reg['anomalies_above_5yr_avg'], name='Wind (5-yr Avg)', line=dict(color='#c2410c', width=2)))
                fig_ica.add_trace(go.Scatter(x=df_reg['time'], y=df_reg['Anomalia_Lluvia_5yr_avg'], name='Rainfall (5-yr Avg)', line=dict(color='#0369a1', width=2)))
                fig_ica.add_trace(go.Scatter(x=df_reg['time'], y=df_reg['Anomalia_Sequia_5yr_avg'], name='Drought (5-yr Avg)', line=dict(color='#d97706', width=2)))
                fig_ica.add_trace(go.Scatter(x=df_reg['time'], y=df_reg['ICA_5yr_avg'], name='ICA Composite', line=dict(color='#0f172a', width=3)))
                fig_ica.add_vline(x='1990-01-01', line_dash='dot', line_color='#94a3b8')
                st.plotly_chart(apply_sleek_theme(fig_ica), width='stretch')

        # --- 2. Temperature Anomalies ---
        if {'t_90', 't_10'}.issubset(df_reg.columns):
            with st.container(border=True):
                st.markdown("#### Temperature Standardized Anomalies (5-Year Moving Average)")
                fig_temp = go.Figure()
                fig_temp.add_trace(go.Scatter(x=df_reg['time'], y=df_reg['t_90_5yr_avg'], name='T90 Std. Anomalies', line=dict(color='#ef4444')))
                fig_temp.add_trace(go.Scatter(x=df_reg['time'], y=df_reg['t_10_5yr_avg'], name='T10 Std. Anomalies', line=dict(color='#3b82f6')))
                fig_temp.add_trace(go.Scatter(x=df_reg['time'], y=df_reg['temp_composite_5yr_avg'], name='Composite', line=dict(color='#1e293b', width=2)))
                fig_temp.add_vline(x='1990-01-01', line_dash='dot', line_color='#94a3b8')
                st.plotly_chart(apply_sleek_theme(fig_temp), width='stretch')

        # --- 3. Rainfall, Drought, Wind ---
        st.markdown("#### Component Anomalies (Raw Data vs. Moving Average)")
        c1, c2 = st.columns(2)

        def create_bar_line_plot(df, raw_col, avg_col, title):
            if raw_col not in df.columns: return None
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df['time'], y=df[raw_col], name='Std. Anomalies', marker_color='#bae6fd'))
            fig.add_trace(go.Scatter(x=df['time'], y=df[avg_col], name='5-Yr Moving Avg', line=dict(color='#1e293b', width=2)))
            fig.add_vline(x='1990-01-01', line_dash='dot', line_color='#94a3b8')
            fig.update_layout(title=title, barmode='overlay')
            return apply_sleek_theme(fig)

        with c1:
            fig_rain = create_bar_line_plot(df_reg, 'Anomalia_Lluvia', 'Anomalia_Lluvia_5yr_avg', "Maximum 5-Day Rainfall Anomalies")
            if fig_rain:
                with st.container(border=True): st.plotly_chart(fig_rain, width='stretch')

            fig_wind = create_bar_line_plot(df_reg, 'anomalies_above', 'anomalies_above_5yr_avg', "Wind Power Anomalies")
            if fig_wind:
                with st.container(border=True): st.plotly_chart(fig_wind, width='stretch')

        with c2:
            fig_drought = create_bar_line_plot(df_reg, 'Anomalia_Sequia', 'Anomalia_Sequia_5yr_avg', "Consecutive Dry Days Anomalies")
            if fig_drought:
                with st.container(border=True): st.plotly_chart(fig_drought, width='stretch')

            # --- Correlation Matrix ---
            with st.container(border=True):
                st.markdown("**Feature Correlation Matrix**")
                raw_vars = ['t_90', 't_10', 'Anomalia_Lluvia', 'Anomalia_Sequia', 'anomalies_above', 'ICA']
                available_vars = [v for v in raw_vars if v in df_reg.columns]

                if available_vars:
                    if not df_enso.empty:
                        corr_df = df_reg[available_vars + ['time']].merge(df_enso[['time', 'sst_index', 'oni']], on='time', how='inner')
                    else:
                        corr_df = df_reg[available_vars]

                    corr_matrix = corr_df.drop(columns=['time'], errors='ignore').corr()
                    fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                    st.plotly_chart(apply_sleek_theme(fig_corr), width='stretch')

    else:
        st.info(f"Anomaly datasets not located for {selected_region} in the processed data directory.")

# ═════ TAB 2: SST vs ONI INTERACTIVE ═══════════════════════════════════════
with t2:
    if not df_enso.empty:
        with st.container(border=True):
            st.markdown("### Pacific SST vs. Niño 3.4 Region (Time Series)")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df_enso['time'], y=df_enso['oni'], name='NOAA ONI', line=dict(color='#94a3b8', width=2, dash='dot'), fill='tozeroy', fillcolor='rgba(148, 163, 184, 0.05)'))
            fig1.add_trace(go.Scatter(x=df_enso['time'], y=df_enso['sst_index'], name='Col. Pacific SST', line=dict(color='#2563eb', width=2)))
            fig1.add_hline(y=0.5, line_dash="dash", line_color="#ef4444", annotation_text="El Niño")
            fig1.add_hline(y=-0.5, line_dash="dash", line_color="#3b82f6", annotation_text="La Niña")
            st.plotly_chart(apply_sleek_theme(fig1), width='stretch')

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("### Signal Alignment (Scatter Plot)")
                fig2 = px.scatter(df_enso, x='oni', y='sst_index', color='year', color_continuous_scale='Viridis')
                m, b = np.polyfit(df_enso['oni'], df_enso['sst_index'], 1)
                fig2.add_trace(go.Scatter(x=df_enso['oni'], y=m*df_enso['oni']+b, mode='lines', name='Trend', line=dict(color='#1e293b', width=2)))
                st.plotly_chart(apply_sleek_theme(fig2), width='stretch')

        with c2:
            with st.container(border=True):
                st.markdown("### 5-Year Rolling Correlation")
                roll_corr = df_enso['sst_index'].rolling(60, center=True).corr(df_enso['oni'])
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=df_enso['time'], y=roll_corr, name="Pearson r", line=dict(color='#059669', width=3), fill='tozeroy', fillcolor='rgba(5, 150, 105, 0.05)'))
                fig3.add_hline(y=df_enso['sst_index'].corr(df_enso['oni']), line_dash="dash", line_color="#64748b", annotation_text="Mean")
                st.plotly_chart(apply_sleek_theme(fig3), width='stretch')

# ═════ TAB 3: SIGNAL INSPECTION ════════════════════════════════════════════
with t3:
    if not df_enso.empty:
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("**Spatial Validation Mapping**")
                signal_geo = PROC / 'signal_geometry.png'
                if signal_geo.exists():
                    st.image(Image.open(signal_geo))
        with c2:
            with st.container(border=True):
                st.markdown("**Grid Correlation Analysis (GRIB Data)**")
                signal_corr = PROC / 'signal_spatial_corr.png'
                if signal_corr.exists():
                    st.image(Image.open(signal_corr))

        with st.container(border=True):
            st.markdown("### Decoupling Diagnostics (Physical Divergence)")
            df_enso['product'] = df_enso['sst_index'] * df_enso['oni']
            df_enso['state'] = np.where(df_enso['product'] > 0, 'Synchronized', 'Decoupled')
            fig_div = px.bar(df_enso, x='time', y='product', color='state',
                             color_discrete_map={'Synchronized': '#10b981', 'Decoupled': '#f43f5e'})
            st.plotly_chart(apply_sleek_theme(fig_div), width='stretch')

# ═════ TAB 4: ENSO FLAGS & TUNING ══════════════════════════════════════════
with t4:
    flags, tuning = load_tuning_data()
    if flags is not None and tuning is not None:
        best = tuning.iloc[0]
        st.markdown("### Threshold Optimization vs. IDEAM & NOAA ONI")

        with st.container(border=True):
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Best Macro-F1 Score", f"{best['macro_f1']:.4f}")
            k2.metric("Local El Niño Threshold", f"{best['warm_thr']:+.2f} °C")
            k3.metric("Local La Niña Threshold", f"{best['cold_thr']:+.2f} °C")
            k4.metric("Minimum Consec. Months", f"{int(best['min_consec'])}")

        with st.container(border=True):
            st.markdown("#### Calibrated Actuarial Flags")
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(x=flags['time'], y=flags['sst_index_bc'], name='Bias-Corrected SST', line=dict(color='#475569', width=1.5)))
            warm_events = flags[flags['D_warm_tuned'] == 1]
            cold_events = flags[flags['D_cold_tuned'] == 1]
            fig_time.add_trace(go.Bar(x=warm_events['time'], y=[1.5]*len(warm_events), name='Tuned El Niño', marker_color='rgba(239, 68, 68, 0.3)', width=2.6e9))
            fig_time.add_trace(go.Bar(x=cold_events['time'], y=[-1.5]*len(cold_events), name='Tuned La Niña', marker_color='rgba(37, 99, 235, 0.3)', width=2.6e9))
            fig_time.add_hline(y=best['warm_thr'], line_dash="dot", line_color="#ef4444")
            fig_time.add_hline(y=best['cold_thr'], line_dash="dot", line_color="#2563eb")
            st.plotly_chart(apply_sleek_theme(fig_time), width='stretch')

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("#### Optimization Surface (Macro-F1)")
                hm_data = tuning[tuning['min_consec'] == best['min_consec']].pivot(index='cold_thr', columns='warm_thr', values='macro_f1')
                fig_hm = px.imshow(hm_data, text_auto=".3f", color_continuous_scale='Viridis',
                                   labels=dict(x="Warm Threshold (°C)", y="Cold Threshold (°C)", color="F1-Score"))
                st.plotly_chart(apply_sleek_theme(fig_hm), width='stretch')

        with c2:
            with st.container(border=True):
                st.markdown("#### Classification Metrics (Before vs. After Tuning)")
                confusion_img = PROC / 'confusion_matrix.png'
                if confusion_img.exists():
                    st.image(Image.open(confusion_img))
    else:
        st.info("Validation metrics absent. Execute the 'compare_enso_flags.py' protocol to generate actuarial tuning datasets.")