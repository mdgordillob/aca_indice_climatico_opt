"""
Produce all paper figures for the ACI-CO article.

ACI-CO = (T90 - T10 + CDD + WP + Rx5day) / 5   (direct z-score on each component)

ENSO covariate: NOAA ERSSTv5 Niño-3.4 anomaly (1991-2020 baseline, 1961-2024)
                ersst5.nino.mth.91-20.ascii → column index 9 (Niño-3.4 ANOM)
                Local Colombian Pacific SST (sst_index_colombia_pacific.csv) is
                plotted for comparison in fig_enso_comparison only.

ENSO regression (asymmetric OLS, HC3 SEs):
    X = α + β⁺·E⁺ + β⁻·E⁻ + γ·t + ε
    E⁺ = max(E,0),  E⁻ = min(E,0)
    ENSO-neutral: X_neutral = α̂ + γ̂·t

Statistical diagnostics for the INDEX regression:
    - Overall F-test
    - t-tests for β⁺, β⁻, γ (HC3)
    - Wald test for symmetry H₀: β⁺ = β⁻
    - Durbin-Watson (first-order autocorrelation)
    - Ljung-Box Q at lags 6, 12 (serial correlation)
    - Jarque-Bera (residual normality)
    - Breusch-Pagan (heteroskedasticity)
    - ADF (stationarity of ACI and SST)

Outputs → articles/graficas/paper_figures/
"""

from pathlib import Path
import zipfile, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import geopandas as gpd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import chi2 as chi2_dist

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
SHP  = ROOT / "data" / "shapefiles"
OUT  = ROOT / "articles" / "graficas" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

if not SHP.exists():
    with zipfile.ZipFile(ROOT / "data" / "shapefiles.zip") as z:
        z.extractall(ROOT / "data")

REF_END = pd.Timestamp("1990-12-01")

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          13,
    "axes.titlesize":     14,
    "axes.labelsize":     13,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "legend.fontsize":    11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.edgecolor":     "#444444",
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     ":",
    "grid.color":         "#aaaaaa",
    "axes.prop_cycle":    plt.cycler(color=[
        "#4878d0", "#ee854a", "#6acc65", "#d65f5f",
        "#956cb4", "#8c613c", "#dc7ec0", "#797979",
    ]),
})

T_START = pd.Timestamp("1961-01-01")
T_END   = pd.Timestamp("2024-12-31")


def _set_time_xlim(ax):
    ax.set_xlim(T_START, T_END)

REGION_LABELS = {
    "colombia":            "Colombia (nacional)",
    "antioquia":           "Antioquia",
    "cundinamarca_bogota": "Cundinamarca–Bogotá",
    "valle_cauca":         "Valle del Cauca",
}

COMP_COLORS = {
    "t90":           "#d62728",
    "t10":           "#1f77b4",
    "drought":       "#8c6d31",
    "wind":          "#2ca02c",
    "precipitation": "#9467bd",
    "aci":           "black",
}

COMP_LABELS = {
    "t90":           "T90 (hot extremes)",
    "t10":           "T10 (cold extremes, negated)",
    "drought":       "CDD (drought)",
    "wind":          "WP (wind power)",
    "precipitation": "Rx5day (precipitation)",
    "aci":           "ACI-CO",
}

SST_LABEL = "Niño-3.4 anomaly – NOAA ERSSTv5 (°C)"

# ---------------------------------------------------------------------------
# 1.  DATA LOADING
# ---------------------------------------------------------------------------

def _load_region(region: str) -> pd.DataFrame:
    d = PROC / f"anomalias_{region}"

    df_t = pd.read_csv(d / "anomalies_temperature_combined.csv")
    df_t["date"] = pd.to_datetime({"year": df_t["year"], "month": df_t["month"], "day": 1})
    df_t = df_t[["date", "t_90", "t_10"]].rename(columns={"t_90": "t90", "t_10": "t10"})

    df_w = pd.read_csv(d / "anomalies_wind_combined.csv")
    df_w["date"] = pd.to_datetime({"year": df_w["year"], "month": df_w["month"], "day": 1})
    df_w = df_w[["date", "anomalies_above"]].rename(columns={"anomalies_above": "wind"})

    df_d = pd.read_csv(d / "anomalies_drought_combined.csv")
    df_d.columns = ["year", "month", "drought"]
    df_d["date"] = pd.to_datetime({"year": df_d["year"], "month": df_d["month"], "day": 1})
    df_d = df_d[["date", "drought"]]

    df_p = pd.read_csv(d / "anomalies_precipitation_combined.csv")
    df_p.columns = ["year", "month", "precipitation"]
    df_p["date"] = pd.to_datetime({"year": df_p["year"], "month": df_p["month"], "day": 1})
    df_p = df_p[["date", "precipitation"]]

    df = (df_t
          .merge(df_w, on="date")
          .merge(df_d, on="date")
          .merge(df_p, on="date"))
    df = df.sort_values("date").set_index("date")
    df["aci"] = (df["t90"] - df["t10"] + df["drought"] + df["wind"] + df["precipitation"]) / 5
    return df


def load_all_regions() -> dict:
    return {r: _load_region(r) for r in REGION_LABELS}


def load_sst() -> pd.Series:
    """
    NOAA ERSSTv5 Niño-3.4 monthly anomaly (baseline 1991-2020).
    Source: data/processed/ersst5.nino.mth.91-20.ascii
    Columns: YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM
    The Niño-3.4 ANOM is column index 8 (0-based, space-delimited).
    """
    rows = []
    with open(PROC / "ersst5.nino.mth.91-20.ascii") as f:
        next(f)  # skip header
        for line in f:
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                yr, mo = int(parts[0]), int(parts[1])
                anom = float(parts[9])   # Niño-3.4 ANOM (index 9, after the absolute value at index 8)
                rows.append((pd.Timestamp(year=yr, month=mo, day=1), anom))
            except (ValueError, IndexError):
                continue
    s = pd.Series({d: v for d, v in rows}, name="nino34_anom")
    return s.sort_index().dropna()


def load_oni() -> pd.Series:
    """
    NOAA ONI (3-month running mean of Niño-3.4, ERSSTv5, base 1991-2020).
    Source: data/processed/oni.ascii.txt
    Season label → central month: DJF→Jan, JFM→Feb, ..., NDJ→Dec
    """
    _seas_to_month = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
        "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
        "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }
    rows = []
    with open(PROC / "oni.ascii.txt") as f:
        next(f)  # skip header
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            seas, yr, anom = parts[0], int(parts[1]), float(parts[3])
            mo = _seas_to_month.get(seas)
            if mo is None:
                continue
            rows.append((pd.Timestamp(year=yr, month=mo, day=1), anom))
    s = pd.Series({d: v for d, v in rows}, name="oni")
    return s.sort_index().dropna()


def load_local_sst() -> pd.Series:
    """
    Local Colombian Pacific SST anomaly derived from ERA5-Land.
    Source: data/processed/sst_index_colombia_pacific.csv
    Column: sst_anomaly_pacific (1°N–8°N, 79°W–77°W area mean)
    Used only for the index-comparison figure; NOT used as ENSO covariate.
    """
    df = pd.read_csv(PROC / "sst_index_colombia_pacific.csv",
                     parse_dates=["time"], index_col="time")
    return df["sst_anomaly_pacific"].rename("local_sst_pacific").sort_index().dropna()


# ---------------------------------------------------------------------------
# 2.  ENSO SHADING
# ---------------------------------------------------------------------------

def _episodes(oni: pd.Series, thr: float, min_months: int = 5):
    flag = (oni >= thr if thr > 0 else oni <= thr).astype(int)
    eps, in_ep, start, run = [], False, None, 0
    for dt, v in flag.items():
        if v:
            if not in_ep:
                start, in_ep = dt, True
            run += 1
        elif in_ep:
            if run >= min_months:
                eps.append((start, dt))
            in_ep, run = False, 0
    if in_ep and run >= min_months:
        eps.append((start, oni.index[-1]))
    return eps


def shade_enso(ax, oni: pd.Series):
    for s, e in _episodes(oni, +0.5):
        ax.axvspan(s, e, alpha=0.13, color="red",  zorder=0, linewidth=0)
    for s, e in _episodes(oni, -0.5):
        ax.axvspan(s, e, alpha=0.13, color="blue", zorder=0, linewidth=0)


ENSO_PATCHES = [
    mpatches.Patch(color="red",  alpha=0.35, label="Warm SST phase (≥ +0.5°C)"),
    mpatches.Patch(color="blue", alpha=0.35, label="Cold SST phase (≤ −0.5°C)"),
]

# ---------------------------------------------------------------------------
# 3.  ENSO REGRESSION — ASYMMETRIC OLS WITH DIAGNOSTICS
# ---------------------------------------------------------------------------

def fit_enso(series: pd.Series, sst: pd.Series) -> dict:
    """
    Fit  X = α + β⁺·E⁺ + β⁻·E⁻ + γ·t + η_t,  η_t ~ AR(p)
    AR order p selected by AIC over p ∈ {0,1,2,3} via SARIMAX.
    Wald test for asymmetry (H₀: β⁺ = β⁻) computed from cov_params().
    """
    idx = series.index.intersection(sst.index)
    y = series.loc[idx].values.astype(float)
    e = sst.loc[idx].values.astype(float)
    t = np.arange(len(idx), dtype=float)

    e_pos = np.maximum(e, 0)
    e_neg = np.minimum(e, 0)
    mask = ~np.isnan(y)

    y_fit  = y[mask]
    exog   = np.column_stack([np.ones(mask.sum()), e_pos[mask], e_neg[mask], t[mask]])

    # AIC-select AR order
    best_aic, best_p, best_res = np.inf, 0, None
    for p in range(0, 4):
        try:
            mod = SARIMAX(y_fit, exog=exog, order=(p, 0, 0),
                          trend=None, enforce_stationarity=False,
                          enforce_invertibility=False)
            r = mod.fit(disp=False, maxiter=200)
            if r.aic < best_aic:
                best_aic, best_p, best_res = r.aic, p, r
        except Exception:
            continue

    res = best_res
    # In statsmodels SARIMAX: exog params are FIRST, then AR, then sigma2
    # So params[:4] = [alpha, beta_pos, beta_neg, gamma]
    alpha, bp, bn, gamma = res.params[0:4]
    cov = res.cov_params()
    pvals = res.pvalues
    tvals = res.tvalues

    resid = res.resid

    # Wald test: H₀ β⁺ = β⁻  (asymmetry)
    # contrast c such that c·params = β⁺ - β⁻
    c = np.zeros(len(res.params))
    c[1] =  1.0   # β⁺ at index 1
    c[2] = -1.0   # β⁻ at index 2
    diff    = c @ res.params
    var_diff = c @ cov @ c
    wald_stat = float(diff**2 / var_diff)
    wald_pval = float(1 - chi2_dist.cdf(wald_stat, df=1))

    # Durbin-Watson on residuals
    dw = durbin_watson(resid)

    # Ljung-Box at lags 6 and 12
    lb6  = acorr_ljungbox(resid, lags=[6],  return_df=True)
    lb12 = acorr_ljungbox(resid, lags=[12], return_df=True)

    # Jarque-Bera normality test
    jb_stat, jb_pval = stats.jarque_bera(resid)

    # ADF stationarity
    adf_y = adfuller(y_fit, autolag="AIC")
    adf_e = adfuller(e,     autolag="AIC")

    # De-ENSOed series: remove ENSO signal, keep trend + residual
    t_full = np.arange(len(series), dtype=float)
    e_full = sst.reindex(series.index).values.astype(float)
    ep_full = np.maximum(e_full, 0)
    en_full = np.minimum(e_full, 0)
    deensoed = series.values - bp * ep_full - bn * en_full
    deensoed_s = pd.Series(deensoed, index=series.index)

    # ENSO-neutral trend = rolling mean of de-ENSOed (replaces straight line)
    neutral_smooth = deensoed_s.rolling(60, min_periods=36).mean()

    # Pseudo-R² (variance explained vs unconditional variance)
    ss_tot = np.nanvar(y_fit)
    ss_res = np.nanvar(resid)
    r2 = float(1 - ss_res / ss_tot)

    return {
        "alpha":    alpha,
        "beta_pos": bp,
        "beta_neg": bn,
        "gamma":    gamma,
        "gamma_per_decade": gamma * 120,
        "ar_order": best_p,
        "aic":      best_aic,
        "pval_bp":  float(pvals[1]),
        "pval_bn":  float(pvals[2]),
        "pval_g":   float(pvals[3]),
        "tstat_bp": float(tvals[1]),
        "tstat_bn": float(tvals[2]),
        "tstat_g":  float(tvals[3]),
        "r2":           r2,
        "adj_r2":       r2,
        "n_obs":        int(mask.sum()),
        "dw":           dw,
        "lb6_stat":     float(lb6["lb_stat"].iloc[0]),
        "lb6_pval":     float(lb6["lb_pvalue"].iloc[0]),
        "lb12_stat":    float(lb12["lb_stat"].iloc[0]),
        "lb12_pval":    float(lb12["lb_pvalue"].iloc[0]),
        "jb_stat":      jb_stat,
        "jb_pval":      jb_pval,
        "wald_stat":    wald_stat,
        "wald_pval":    wald_pval,
        "adf_y_stat":   adf_y[0],
        "adf_y_pval":   adf_y[1],
        "adf_e_stat":   adf_e[0],
        "adf_e_pval":   adf_e[1],
        "neutral":      neutral_smooth,
        "deensoed":     deensoed_s,
        "observed":     series,
        "fitted":       pd.Series(res.fittedvalues, index=idx[mask]),
        "residuals":    pd.Series(resid, index=idx[mask]),
        "enso_idx":     idx,
        "sst":          sst.loc[idx],
    }


def _sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def build_enso_table(regions: dict, sst: pd.Series) -> pd.DataFrame:
    rows = []
    for region, df in regions.items():
        for comp in ["t90", "t10", "drought", "wind", "precipitation", "aci"]:
            r = fit_enso(df[comp], sst)
            rows.append({
                "region":            region,
                "component":         comp,
                "beta_pos":          round(r["beta_pos"], 3),
                "sig_bp":            _sig(r["pval_bp"]),
                "beta_neg":          round(r["beta_neg"], 3),
                "sig_bn":            _sig(r["pval_bn"]),
                "gamma_per_decade":  round(r["gamma_per_decade"], 3),
                "sig_g":             _sig(r["pval_g"]),
                "wald_pval":         round(r["wald_pval"], 4),
                "R2":                round(r["r2"], 3),
                "DW":                round(r["dw"], 3),
                "LB12_pval":         round(r["lb12_pval"], 4),
                "JB_pval":           round(r["jb_pval"], 4),
            })
    return pd.DataFrame(rows)


def print_diagnostics(r: dict, label: str = "ACI-CO"):
    """Print a formatted diagnostic table to stdout."""
    out = [
        f"\n{'='*60}",
        f"  ENSO regression diagnostics: {label}",
        f"{'='*60}",
        f"  N = {r['n_obs']}   R2 = {r['r2']:.3f}   AR order = {r['ar_order']}   AIC = {r['aic']:.1f}",
        "",
        "  Coefficients (SARIMAX MLE s.e.)",
        f"  {'':12s} {'coef':>8s}  {'t':>7s}  {'p':>8s}  {'sig':3s}",
        f"  {'b+ (warm)':12s} {r['beta_pos']:>8.4f}  {r['tstat_bp']:>7.3f}  {r['pval_bp']:>8.4f}  {_sig(r['pval_bp'])}",
        f"  {'b- (cold)':12s} {r['beta_neg']:>8.4f}  {r['tstat_bn']:>7.3f}  {r['pval_bn']:>8.4f}  {_sig(r['pval_bn'])}",
        f"  {'g (trend)':12s} {r['gamma_per_decade']:>8.4f}  {r['tstat_g']:>7.3f}  {r['pval_g']:>8.4f}  {_sig(r['pval_g'])}  [s/decade]",
        "",
        f"  Asymmetry test  Wald chi2(1) = {r['wald_stat']:.3f}   p = {r['wald_pval']:.4f}   {_sig(r['wald_pval'])}",
        "",
        "  Residual diagnostics",
        f"  Durbin-Watson        = {r['dw']:.3f}   (2 = no AC; <1.5 positive AC)",
        f"  Ljung-Box Q(6)  stat = {r['lb6_stat']:.2f}   p = {r['lb6_pval']:.4f}   {_sig(r['lb6_pval'])}",
        f"  Ljung-Box Q(12) stat = {r['lb12_stat']:.2f}   p = {r['lb12_pval']:.4f}   {_sig(r['lb12_pval'])}",
        f"  Jarque-Bera     stat = {r['jb_stat']:.2f}   p = {r['jb_pval']:.4f}   {_sig(r['jb_pval'])}",
        "",
        f"  ADF stationarity (ACI)  stat = {r['adf_y_stat']:.3f}   p = {r['adf_y_pval']:.4f}",
        f"  ADF stationarity (SST)  stat = {r['adf_e_stat']:.3f}   p = {r['adf_e_pval']:.4f}",
        f"{'='*60}",
    ]
    print("\n".join(out))


# ---------------------------------------------------------------------------
# 4.  FIGURE 1 — Colombia map with San Andrés & Providencia inset
# ---------------------------------------------------------------------------

def fig1_colombia_map():
    col   = gpd.read_file(SHP / "colombia_4326.shp")
    ant   = gpd.read_file(SHP / "antioquia_4326.shp")
    cund  = gpd.read_file(SHP / "Cundinamarca_Bogota_4326.shp")
    valle = gpd.read_file(SHP / "valle_cauca_4326.shp")
    sa    = gpd.read_file(SHP / "san_andres_providencia.shp")
    mde   = gpd.read_file(SHP / "medellin_4326.shp")
    bog   = gpd.read_file(SHP / "bogota.shp")
    cali  = gpd.read_file(SHP / "cali_4326.shp")

    fig = plt.figure(figsize=(8, 10))

    # Main mainland map
    ax = fig.add_axes([0.05, 0.05, 0.87, 0.88])
    col.plot(ax=ax,   color="#f0ede4", edgecolor="#555", linewidth=0.8)
    ant.plot(ax=ax,   color="#d62728", alpha=0.45, edgecolor="#d62728", linewidth=0.6)
    cund.plot(ax=ax,  color="#1f77b4", alpha=0.45, edgecolor="#1f77b4", linewidth=0.6)
    valle.plot(ax=ax, color="#2ca02c", alpha=0.45, edgecolor="#2ca02c", linewidth=0.6)

    # City markers
    for gdf, label, offset in [
        (mde, "Medellín", (+0.15,  0.10)),
        (bog, "Bogotá",   (+0.15, -0.20)),
        (cali, "Cali",    (+0.15,  0.10)),
    ]:
        pt = gdf.geometry.centroid.iloc[0]
        ax.plot(pt.x, pt.y, "o", color="white", markeredgecolor="black",
                markersize=6, zorder=5)
        ax.annotate(label, (pt.x + offset[0], pt.y + offset[1]),
                    fontsize=8, zorder=6)

    legend_patches = [
        mpatches.Patch(color="#d62728", alpha=0.6, label="Antioquia"),
        mpatches.Patch(color="#1f77b4", alpha=0.6, label="Cundinamarca–Bogotá"),
        mpatches.Patch(color="#2ca02c", alpha=0.6, label="Valle del Cauca"),
        mpatches.Patch(color="#f0ede4", edgecolor="#555", label="Rest of Colombia"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Figure 1 — Colombia: Focus Departments", fontweight="bold")
    ax.set_aspect("equal")

    # San Andrés & Providencia inset (top-left corner, Caribbean)
    ax_ins = fig.add_axes([0.05, 0.70, 0.22, 0.22])
    # light ocean background
    ax_ins.set_facecolor("#d0e8f8")
    col_clip = col.copy()
    col_clip.plot(ax=ax_ins, color="#f0ede4", edgecolor="#888", linewidth=0.5)
    sa.plot(ax=ax_ins, color="#e07b39", edgecolor="#c05a20", linewidth=0.6)

    # Zoom to the islands with some Caribbean context
    minx, miny, maxx, maxy = sa.total_bounds
    pad_x = (maxx - minx) * 3
    pad_y = (maxy - miny) * 4
    ax_ins.set_xlim(minx - pad_x, maxx + pad_x)
    ax_ins.set_ylim(miny - pad_y, maxy + pad_y)
    ax_ins.set_aspect("equal")
    ax_ins.set_xticks([]); ax_ins.set_yticks([])
    ax_ins.set_title("San Andrés &\nProvidencia", fontsize=6.5, pad=2)
    for spine in ax_ins.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#555")

    # Annotate on main map where the islands are relative to mainland
    # Arrow from inset to approximate Caribbean position on main map
    sa_cent_x = (minx + maxx) / 2
    sa_cent_y = (miny + maxy) / 2
    main_bounds = col.total_bounds
    if main_bounds[0] <= sa_cent_x <= main_bounds[2]:
        ax.plot(sa_cent_x, sa_cent_y, "s", color="#e07b39",
                markersize=5, markeredgecolor="#c05a20", zorder=5)
        ax.annotate("Arch. San Andrés\ny Providencia",
                    (sa_cent_x + 0.1, sa_cent_y - 0.6),
                    fontsize=6.5, color="#c05a20")

    fig.savefig(OUT / "fig1_colombia_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig1_colombia_map.png")


# ---------------------------------------------------------------------------
# 5.  FIGURE 4 — National ACI-CO
# ---------------------------------------------------------------------------

def fig4_national_aci(df_col: pd.DataFrame, oni: pd.Series):
    monthly = df_col["aci"]
    ma5     = monthly.rolling(60, min_periods=36).mean()

    # OLS secular trend on monthly series
    t_num   = np.arange(len(monthly), dtype=float)
    mask    = ~np.isnan(monthly.values)
    slope, intercept = np.polyfit(t_num[mask], monthly.values[mask], 1)
    trend_line = pd.Series(slope * t_num + intercept, index=monthly.index)
    trend_ma   = trend_line.rolling(60, min_periods=36).mean()

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.1]})

    # ── Panel (a): Monthly bars + OLS trend ──────────────────────────────────
    ax = axes[0]
    ax.bar(monthly.index, monthly.values, width=28,
           color="#3c7ebf", alpha=0.38, linewidth=0, label="Monthly ACI-CO")
    ax.plot(trend_line.index, trend_line.values,
            color="#e07b10", linewidth=2.0, linestyle="--",
            label=f"OLS trend (+{slope*120:.2f} σ/decade)")
    ax.axvline(REF_END, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Std. anomaly (σ)")
    ax.set_title("(a) ACI-CO — monthly anomaly with secular trend")
    ax.legend(fontsize=9, loc="upper left", ncol=2)

    # ── Panel (b): 5-yr MA + light ENSO shading + bootstrap bands ────────────
    ax = axes[1]
    for s, e in _episodes(oni, +0.5):
        ax.axvspan(s, e, alpha=0.07, color="red",  zorder=0, linewidth=0)
    for s, e in _episodes(oni, -0.5):
        ax.axvspan(s, e, alpha=0.07, color="blue", zorder=0, linewidth=0)

    # Load bootstrap bands if available (produced by build_validation_outputs.py)
    _boot_csv = OUT / "table6b_bootstrap_national.csv"
    boot_legend = []
    if _boot_csv.exists():
        boot_df = pd.read_csv(_boot_csv, parse_dates=["date"]).set_index("date")
        ax.fill_between(boot_df.index, boot_df["p05"], boot_df["p95"],
                        alpha=0.18, color="steelblue", zorder=1)
        ax.fill_between(boot_df.index, boot_df["p25"], boot_df["p75"],
                        alpha=0.32, color="steelblue", zorder=1)
        boot_legend = [
            mpatches.Patch(color="steelblue", alpha=0.40, label="±1.5σ residual envelope"),
            mpatches.Patch(color="steelblue", alpha=0.65, label="±0.75σ residual envelope"),
        ]

    ax.plot(ma5.index, ma5.values, color="black", linewidth=2.4,
            label="Observed (5-yr MA)", zorder=3)
    ax.plot(trend_ma.index, trend_ma.values,
            color="#e07b10", linewidth=1.6, linestyle="--",
            label="Secular trend (5-yr MA)", zorder=2)

    ax.axvline(REF_END, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.6)

    # Key event annotations
    for ts, lbl in [
        (pd.Timestamp("1997-09-01"), "1997–98\nEl Niño"),
        (pd.Timestamp("2010-06-01"), "2010–12\nLa Niña"),
        (pd.Timestamp("2015-09-01"), "2015–16\nEl Niño"),
    ]:
        yval = float(ma5.asof(ts)) if not pd.isna(ma5.asof(ts)) else 0.0
        ax.annotate(lbl, xy=(ts, yval),
                    xytext=(0, 20), textcoords="offset points",
                    fontsize=7.5, ha="center", color="#444444",
                    arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.8))

    ax.set_ylabel("Std. anomaly (σ)")
    ax.set_xlabel("Year")
    ax.set_title("(b) ACI-CO — 5-year moving average with ENSO context and uncertainty")

    legend_handles = boot_legend + [
        Line2D([], [], color="black",   linewidth=2.4, label="Observed (5-yr MA)"),
        Line2D([], [], color="#e07b10", linewidth=1.6, linestyle="--",
               label="Secular trend"),
        mpatches.Patch(color="red",  alpha=0.28, label="El Niño phase"),
        mpatches.Patch(color="blue", alpha=0.28, label="La Niña phase"),
    ]
    ax.legend(handles=legend_handles, fontsize=8.5, ncol=2, loc="upper left")

    _set_time_xlim(axes[0])
    fig.suptitle("Colombian Actuarial Climate Index (ACI-CO) — National (1961–2024)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "fig4_national_aci.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig4_national_aci.png")


# ---------------------------------------------------------------------------
# 6.  FIGURE 5 — Component time series
# ---------------------------------------------------------------------------

def fig5_components(df_col: pd.DataFrame, oni: pd.Series):
    comps = ["t90", "t10", "drought", "wind", "precipitation"]
    panel = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    fig, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
    for ax, comp, letter in zip(axes, comps, panel):
        s   = df_col[comp]
        ma5 = s.rolling(60, min_periods=36).mean()
        shade_enso(ax, oni)
        ax.bar(s.index, s.values, width=28,
               color=COMP_COLORS[comp], alpha=0.35, label="Monthly")
        ax.plot(ma5.index, ma5.values,
                color=COMP_COLORS[comp], linewidth=2, label="5-yr MA")
        ax.axvline(REF_END, color="gray", linestyle=":", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("σ")
        ax.set_title(f"{letter} {COMP_LABELS[comp]}")
        ax.legend(fontsize=7, loc="upper left",
                  handles=[mpatches.Patch(color=COMP_COLORS[comp], alpha=0.5,
                                          label="Monthly"),
                            Line2D([], [], color=COMP_COLORS[comp],
                                   linewidth=2, label="5-yr MA")])
    axes[-1].set_xlabel("Year")
    _set_time_xlim(axes[0])
    fig.suptitle("ACI-CO Component Anomalies — Colombia (national)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_components.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig5_components.png")


# ---------------------------------------------------------------------------
# 7.  FIGURE 5b — Marginal distributions
# ---------------------------------------------------------------------------

def fig5b_distributions(df_col: pd.DataFrame):
    comps = ["t90", "t10", "drought", "wind", "precipitation"]
    fig, axes = plt.subplots(1, 5, figsize=(13, 4))
    xg = np.linspace(-5, 5, 300)
    for ax, comp in zip(axes, comps):
        vals = df_col[comp].dropna().values
        sk, ku = stats.skew(vals), stats.kurtosis(vals)
        ax.hist(vals, bins=35, density=True,
                color=COMP_COLORS[comp], alpha=0.55, edgecolor="white")
        ax.plot(xg, stats.norm.pdf(xg), "k--", linewidth=1.4, label="N(0,1)")
        ax.set_title(f"{COMP_LABELS[comp]}\nskew={sk:.2f}  kurt={ku:.2f}",
                     fontsize=7.5)
        ax.set_xlabel("σ")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    fig.suptitle("Marginal distributions of ACI-CO component anomalies (national, 1961–2024)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig5b_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig5b_distributions.png")


# ---------------------------------------------------------------------------
# 8.  FIGURE 2 — ENSO decomposition (observed vs neutral)
# ---------------------------------------------------------------------------

def fig2_enso_decomposition(df_col: pd.DataFrame, sst: pd.Series, oni: pd.Series):
    r        = fit_enso(df_col["aci"], sst)
    obs      = r["observed"]
    deensoed = r["deensoed"]          # obs minus ENSO signal
    neutral  = r["neutral"]           # rolling mean of de-ENSOed (non-linear)
    obs_ma5  = obs.rolling(60, min_periods=36).mean()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    shade_enso(ax, oni)
    ax.plot(obs.index, obs.values, color="steelblue", linewidth=0.6,
            alpha=0.55, label="Observed ACI-CO")
    ax.plot(deensoed.index, deensoed.values, color="darkorange",
            linewidth=0.7, alpha=0.7, label="De-ENSOed ACI-CO")
    ax.axvline(REF_END, color="gray", linestyle=":", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Std. anomaly (σ)")
    ax.set_title("(a) ACI-CO: observed vs de-ENSOed — monthly")
    ax.legend()

    ax = axes[1]
    shade_enso(ax, oni)
    ax.plot(obs_ma5.index, obs_ma5.values, color="steelblue", linewidth=2.2,
            label="Observed (5-yr MA)")
    ax.plot(neutral.index, neutral.values, color="darkorange",
            linewidth=2.2, linestyle="--", label="De-ENSOed trend (5-yr MA)")
    ax.fill_between(obs_ma5.index,
                    obs_ma5.values, neutral.reindex(obs_ma5.index),
                    alpha=0.25, color="purple", label="ENSO contribution")
    ax.axvline(REF_END, color="gray", linestyle=":", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Std. anomaly (σ)")
    ax.set_xlabel("Year")
    ax.set_title("(b) ACI-CO: observed vs de-ENSOed trend — 5-year moving average")
    bp, bn, g, r2 = r["beta_pos"], r["beta_neg"], r["gamma_per_decade"], r["r2"]
    ax.text(0.015, 0.97,
            f"β⁺(warm) = {bp:+.3f}{_sig(r['pval_bp'])}"
            f"   β⁻(cold) = {bn:+.3f}{_sig(r['pval_bn'])}"
            f"   γ = {g:+.3f} σ/decade{_sig(r['pval_g'])}"
            f"   R² = {r2:.2f}   AR({r['ar_order']})",
            transform=ax.transAxes, fontsize=9, color="dimgray", va="top")
    ax.legend()

    _set_time_xlim(axes[0])
    fig.suptitle("ACI-CO — ENSO Decomposition (National, SST covariate)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_enso_neutral.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig2_enso_neutral.png")


# ---------------------------------------------------------------------------
# 9.  FIGURE 2b — ENSO regression scatters
# ---------------------------------------------------------------------------

def fig2b_enso_regression_scatter(df_col: pd.DataFrame, sst: pd.Series):
    comps = ["t90", "t10", "drought", "wind", "precipitation", "aci"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    eg = np.linspace(-1.5, 2.0, 200)
    ep = np.maximum(eg, 0)
    en = np.minimum(eg, 0)

    for ax, comp in zip(axes, comps):
        r   = fit_enso(df_col[comp], sst)
        idx = r["enso_idx"]
        y   = df_col[comp].loc[idx].values
        e   = r["sst"].values
        pos = e >= 0
        ax.scatter(e[pos],  y[pos],  s=8, color="#d62728", alpha=0.35, label="Warm SST")
        ax.scatter(e[~pos], y[~pos], s=8, color="#1f77b4", alpha=0.35, label="Cold SST")
        fitted = r["alpha"] + r["beta_pos"] * ep + r["beta_neg"] * en
        ax.plot(eg, fitted, color="black", linewidth=1.8, label="Fit")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
        bp, bn, r2 = r["beta_pos"], r["beta_neg"], r["r2"]
        asym = "†" if r["wald_pval"] < 0.05 else ""
        ax.set_title(f"{COMP_LABELS[comp]}\n"
                     f"β⁺={bp:+.2f}{_sig(r['pval_bp'])}"
                     f"  β⁻={bn:+.2f}{_sig(r['pval_bn'])}"
                     f"  R²={r2:.2f}{asym}",
                     fontsize=8)
        ax.set_xlabel(SST_LABEL, fontsize=7)
        ax.set_ylabel("Component anomaly (σ)")
        if comp == "t90":
            ax.legend(fontsize=7)

    fig.suptitle(
        "ENSO Regression: Component anomalies vs NOAA ERSSTv5 Niño-3.4 (1961–2024)\n"
        "† = asymmetry test H₀:β⁺=β⁻ rejected at 5%",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig2b_enso_regression_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig2b_enso_regression_scatter.png")


# ---------------------------------------------------------------------------
# 10. FIGURE 3 — ACI regression statistical diagnostics
# ---------------------------------------------------------------------------

def fig_enso_diagnostics(df_col: pd.DataFrame, sst: pd.Series):
    # Figure 3 — ENSO Phase Analysis (replaces confusing regression diagnostics).
    # Shows HOW ENSO drives ACI-CO through four complementary panels:
    #   (a) Asymmetric ENSO regression coefficients β+ / β- by component
    #   (b) ACI-CO distribution by ENSO phase (El Niño / Neutral / La Niña)
    #   (c) ACI-CO vs Niño-3.4 scatter, colored by decade + piecewise OLS fit
    #   (d) Seasonal ENSO sensitivity: monthly mean ACI-CO by ENSO phase
    comps       = ["t90", "t10", "drought", "wind", "precipitation"]
    comp_short  = ["T90", "T10", "CDD", "WP", "Rx5day"]

    # Fit asymmetric ENSO regression for each component
    idx_shared = df_col.index.intersection(sst.index)
    e_vals = sst.loc[idx_shared].values
    e_pos  = np.maximum(e_vals, 0)
    e_neg  = np.minimum(e_vals, 0)
    t_arr  = np.arange(len(idx_shared), dtype=float)
    exog   = np.column_stack([np.ones(len(idx_shared)), e_pos, e_neg, t_arr])

    betas_pos, betas_neg = [], []
    pvals_pos, pvals_neg = [], []
    for comp in comps:
        y = df_col[comp].loc[idx_shared].values.astype(float)
        mask = ~np.isnan(y)
        try:
            best_aic, best_res = np.inf, None
            for p in range(0, 3):
                try:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX as _S
                    mod = _S(y[mask], exog=exog[mask], order=(p, 0, 0),
                             trend=None, enforce_stationarity=False,
                             enforce_invertibility=False)
                    r2 = mod.fit(disp=False, maxiter=200)
                    if r2.aic < best_aic:
                        best_aic, best_res = r2.aic, r2
                except Exception:
                    continue
            betas_pos.append(float(best_res.params[1]))
            betas_neg.append(float(best_res.params[2]))
            pvals_pos.append(float(best_res.pvalues[1]))
            pvals_neg.append(float(best_res.pvalues[2]))
        except Exception:
            betas_pos.append(0.0); betas_neg.append(0.0)
            pvals_pos.append(1.0); pvals_neg.append(1.0)

    # ENSO phase classification using 3-month centred Niño-3.4
    oni3m = sst.rolling(3, center=True, min_periods=2).mean()
    phase = pd.Series("Neutral", index=sst.index, dtype=str)
    phase[oni3m >= 0.5]  = "El Niño"
    phase[oni3m <= -0.5] = "La Niña"

    aci_obs   = df_col["aci"].loc[idx_shared]
    phase_cat = phase.reindex(idx_shared)
    months    = idx_shared.month
    years     = idx_shared.year

    elnino_aci  = aci_obs[phase_cat == "El Niño"].values
    neutral_aci = aci_obs[phase_cat == "Neutral"].values
    lanina_aci  = aci_obs[phase_cat == "La Niña"].values

    fig = plt.figure(figsize=(12, 9))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.36)

    # ── (a) β+ and β- by component ──────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(comps))
    w = 0.34
    ax_a.bar(x - w/2, betas_pos, w, color="#d62728", alpha=0.78,
             label="β⁺  El Niño response")
    ax_a.bar(x + w/2, betas_neg, w, color="#1f77b4", alpha=0.78,
             label="β⁻  La Niña response")
    for i, (bp, bn, pp_, pn) in enumerate(zip(betas_pos, betas_neg,
                                               pvals_pos, pvals_neg)):
        def _star(p):
            return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        if _star(pp_):
            ypos = bp + (0.03 if bp >= 0 else -0.10)
            ax_a.text(i - w/2, ypos, _star(pp_), ha="center", fontsize=8,
                      color="#d62728")
        if _star(pn):
            ypos = bn + (0.03 if bn >= 0 else -0.10)
            ax_a.text(i + w/2, ypos, _star(pn), ha="center", fontsize=8,
                      color="#1f77b4")
    ax_a.axhline(0, color="black", linewidth=0.8)
    ax_a.set_xticks(x); ax_a.set_xticklabels(comp_short, fontsize=10)
    ax_a.set_ylabel("Response (σ per °C Niño-3.4)")
    ax_a.set_title("(a) Asymmetric ENSO response by component")
    ax_a.legend(fontsize=8.5)

    # ── (b) Distribution by ENSO phase ───────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    phase_data = [elnino_aci, neutral_aci, lanina_aci]
    phase_cols = ["#d62728", "#888888", "#1f77b4"]
    bp_obj = ax_b.boxplot(phase_data, patch_artist=True,
                           medianprops=dict(color="black", linewidth=1.8),
                           whiskerprops=dict(linewidth=1.2),
                           capprops=dict(linewidth=1.2),
                           flierprops=dict(marker="o", markersize=3,
                                           alpha=0.4, linestyle="none"))
    for patch, col in zip(bp_obj["boxes"], phase_cols):
        patch.set_facecolor(col); patch.set_alpha(0.60)
    for i, (data, col) in enumerate(zip(phase_data, phase_cols), 1):
        ax_b.scatter(i, np.mean(data), s=55, zorder=5,
                     color="white", edgecolors=col, linewidths=2.0)
    ax_b.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax_b.set_xticks([1, 2, 3])
    ax_b.set_xticklabels(["El Niño", "Neutral", "La Niña"], fontsize=10)
    ax_b.set_ylabel("ACI-CO anomaly (σ)")
    ax_b.set_title("(b) ACI-CO distribution by ENSO phase")
    for i, (data, col) in enumerate(zip(phase_data, phase_cols), 1):
        ax_b.text(i, ax_b.get_ylim()[0] + 0.04 * np.diff(ax_b.get_ylim())[0],
                  f"n={len(data)}", ha="center", fontsize=7.5, color="#555")

    # ── (c) Scatter ACI-CO vs Niño-3.4, colored by decade ───────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    decades = [(1961, 1979, "#9ecae1", "1961–79"),
               (1980, 1999, "#fc8d59", "1980–99"),
               (2000, 2013, "#74c476", "2000–13"),
               (2014, 2024, "#c994c7", "2014–24")]
    for y0, y1, col, lbl in decades:
        md = (years >= y0) & (years <= y1)
        ax_c.scatter(e_vals[md], aci_obs.values[md], s=7,
                     color=col, alpha=0.65, label=lbl, zorder=2)
    # Piecewise OLS fit for display
    valid = ~np.isnan(aci_obs.values)
    Xpw   = np.column_stack([np.ones(valid.sum()),
                              np.maximum(e_vals[valid], 0),
                              np.minimum(e_vals[valid], 0)])
    from numpy.linalg import lstsq
    c_ols, *_ = lstsq(Xpw, aci_obs.values[valid], rcond=None)
    eg = np.linspace(e_vals.min(), e_vals.max(), 300)
    ax_c.plot(eg, c_ols[0] + c_ols[1]*np.maximum(eg, 0) + c_ols[2]*np.minimum(eg, 0),
              color="black", linewidth=2.0, label="Piecewise fit", zorder=4)
    ax_c.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_c.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_c.set_xlabel("Niño-3.4 anomaly (°C)")
    ax_c.set_ylabel("ACI-CO anomaly (σ)")
    ax_c.set_title("(c) ACI-CO vs Niño-3.4 by decade")
    ax_c.legend(fontsize=7.5, ncol=2, loc="upper left")

    # ── (d) Seasonal ENSO response ───────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    elnino_m = phase_cat == "El Niño"
    lanina_m = phase_cat == "La Niña"
    mean_nino = [float(aci_obs[elnino_m & (months == m)].mean()) for m in range(1, 13)]
    mean_nina = [float(aci_obs[lanina_m & (months == m)].mean()) for m in range(1, 13)]
    mean_neut = [float(aci_obs[~elnino_m & ~lanina_m & (months == m)].mean())
                 for m in range(1, 13)]
    x_m = np.arange(1, 13)
    ax_d.plot(x_m, mean_nino, color="#d62728", marker="o", markersize=5,
              linewidth=1.8, label="El Niño months")
    ax_d.plot(x_m, mean_nina, color="#1f77b4", marker="s", markersize=5,
              linewidth=1.8, label="La Niña months")
    ax_d.plot(x_m, mean_neut, color="#888888", marker="^", markersize=4,
              linewidth=1.4, linestyle="--", label="Neutral months")
    ax_d.axhline(0, color="black", linewidth=0.6)
    ax_d.set_xticks(x_m)
    ax_d.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax_d.set_ylabel("Mean ACI-CO anomaly (σ)")
    ax_d.set_xlabel("Calendar month")
    ax_d.set_title("(d) Seasonal ENSO response (monthly composite)")
    ax_d.legend(fontsize=8.5)

    fig.suptitle(
        "Figure 3 — ENSO Phase Analysis: How El Niño / La Niña drive ACI-CO"
        "  (National, 1961–2024)",
        fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "fig3_enso_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig3_enso_diagnostics.png")


# ---------------------------------------------------------------------------
# 11. FIGURE 7a — Three focus regions
# ---------------------------------------------------------------------------

def fig7a_regional(regions: dict, oni: pd.Series):
    focus = ["cundinamarca_bogota", "antioquia", "valle_cauca"]
    comps = ["t90", "t10", "drought", "wind", "precipitation", "aci"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, region in zip(axes, focus):
        df = regions[region]
        shade_enso(ax, oni)
        for comp in comps:
            ma5 = df[comp].rolling(60, min_periods=36).mean()
            lw  = 2.5 if comp == "aci" else 1.1
            ls  = "-"  if comp == "aci" else "--"
            al  = 1.0  if comp == "aci" else 0.75
            ax.plot(ma5.index, ma5.values, color=COMP_COLORS[comp],
                    linewidth=lw, linestyle=ls, alpha=al,
                    label=COMP_LABELS[comp])
        ax.axvline(REF_END, color="gray", linestyle=":", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Std. anomaly (σ)")
        ax.set_title(REGION_LABELS[region], fontsize=10)
        ax.legend(fontsize=7, ncol=3, loc="upper left")
    axes[-1].set_xlabel("Year")
    _set_time_xlim(axes[0])
    fig.suptitle("ACI-CO and components — Focus regions (5-year moving average)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig7a_regional.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig7a_regional.png")


# ---------------------------------------------------------------------------
# 12. CORRELATION MATRIX including SST
# ---------------------------------------------------------------------------

def fig_correlation(df_col: pd.DataFrame, sst: pd.Series):
    comps  = ["t90", "t10", "drought", "wind", "precipitation", "aci"]
    labels = ["T90", "T10", "CDD", "WP", "Rx5day", "ACI-CO", "SST anomaly"]

    idx_shared = df_col.index.intersection(sst.index)
    sub = df_col.loc[idx_shared, comps].copy()
    sub["sst"] = sst.loc[idx_shared].values

    corr = sub.corr()
    pmat = pd.DataFrame(np.ones_like(corr.values),
                        index=corr.index, columns=corr.columns)
    for i, ci in enumerate(corr.columns):
        for j, cj in enumerate(corr.columns):
            if i != j:
                _, p = stats.pearsonr(sub[ci].dropna(), sub[cj].dropna())
                pmat.iloc[i, j] = p

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            p = pmat.values[i, j]
            sig = _sig(p)
            text = f"{v:.2f}{sig}"
            col  = "white" if abs(v) > 0.4 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=col)
    ax.set_title("Component + SST Pearson correlation matrix (national, 1961–2024)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_corr_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig_corr_matrix.png")


# ---------------------------------------------------------------------------
# 13. FIGURE 6 — Regional ACI-CO heatmap (region × year)
# ---------------------------------------------------------------------------

def fig6_regional_heatmap(regions: dict):
    focus = ["colombia", "antioquia", "cundinamarca_bogota", "valle_cauca"]
    labels = [REGION_LABELS[r] for r in focus]
    years = np.arange(1961, 2025)
    data = np.zeros((len(focus), len(years)))
    for i, region in enumerate(focus):
        ann = regions[region]["aci"].resample("YE").mean()
        for j, yr in enumerate(years):
            ts = str(yr)
            if ts in ann.index.strftime("%Y"):
                data[i, j] = ann.loc[ann.index.year == yr].mean()

    fig, ax = plt.subplots(figsize=(12, 3.5))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5,
                   extent=[1961, 2024, len(focus)-0.5, -0.5])
    plt.colorbar(im, ax=ax, label="Annual mean ACI-CO (σ)", shrink=0.85)
    ax.set_yticks(range(len(focus)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Year")
    ax.set_title("ACI-CO Annual Composite by Region (1961–2024)",
                 fontsize=12, fontweight="bold")
    # Annotate major El Niño episodes
    for yr, label in [(1983, "1982/83"), (1998, "1997/98"), (2016, "2015/16")]:
        ax.axvline(yr, color="darkred", linewidth=1.2, alpha=0.7, linestyle="--")
        ax.text(yr + 0.2, -0.45, label, fontsize=7, color="darkred",
                rotation=90, va="top")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_regional_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig6_regional_heatmap.png")


# ---------------------------------------------------------------------------
# 14. TABLE 6 — Mann-Kendall trend tests by component and region
# ---------------------------------------------------------------------------

def build_mk_table(regions: dict) -> pd.DataFrame:
    from scipy.stats import kendalltau

    def mk_test(series: pd.Series):
        y = series.dropna().values
        n = len(y)
        t_arr = np.arange(n)
        tau, pval = kendalltau(t_arr, y)
        # Sen slope: median of all pairwise slopes
        slopes = [(y[j] - y[i]) / (j - i)
                  for i in range(n) for j in range(i+1, n)]
        sen = np.median(slopes) * 120  # convert to σ/decade (monthly data)
        return tau, pval, sen

    rows = []
    for region, df in regions.items():
        for comp in ["t90", "t10", "drought", "wind", "precipitation", "aci"]:
            s = df[comp].dropna()
            tau, pval, sen = mk_test(s)
            rows.append({
                "region":    region,
                "component": comp,
                "tau":       round(tau, 3),
                "pval":      round(pval, 4),
                "sig":       _sig(pval),
                "sen_slope": round(sen, 3),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 15. FIGURE — ENSO INDEX COMPARISON
#     NOAA ERSSTv5 Niño-3.4 vs local Colombian Pacific SST (ERA5-Land)
# ---------------------------------------------------------------------------

def fig_enso_comparison(nino34: pd.Series, local_sst: pd.Series):
    """
    Two-panel figure comparing the two SST indices used in the study:
      (a) Time-series overlay (1961-2024) with Pearson ρ annotated.
      (b) Scatter plot: NOAA Niño-3.4 on x-axis, local Colombian Pacific on y-axis.
    """
    idx = nino34.index.intersection(local_sst.index)
    x   = nino34.loc[idx]
    y   = local_sst.loc[idx]
    rho, pval = stats.pearsonr(x.dropna(), y.reindex(x.dropna().index).dropna())

    col_nino  = "#c0392b"
    col_local = "#2980b9"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                    gridspec_kw={"width_ratios": [2, 1]})

    # Panel (a) — time series
    ax1.plot(x.index, x.values, color=col_nino,  linewidth=1.4, alpha=0.9,
             label="NOAA ERSSTv5 Niño-3.4")
    ax1.plot(y.index, y.values, color=col_local, linewidth=1.2, alpha=0.75,
             linestyle="--", label="Colombian Pacific (ERA5-Land, 1°–8°N, 77°–79°W)")
    ax1.axhline(0,  color="black", linewidth=0.5)
    ax1.axhline(+0.5, color=col_nino,  linewidth=0.6, linestyle=":", alpha=0.5)
    ax1.axhline(-0.5, color=col_nino,  linewidth=0.6, linestyle=":", alpha=0.5)
    _set_time_xlim(ax1)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("SST anomaly (°C)")
    ax1.set_title("(a) ENSO index comparison — time series")
    ax1.legend(fontsize=10)
    ax1.text(0.02, 0.97, f"ρ = {rho:.3f}  (p {'< 0.001' if pval < 0.001 else f'= {pval:.3f}'})",
             transform=ax1.transAxes, fontsize=10, va="top", color="dimgray",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"))

    # Panel (b) — scatter
    ax2.scatter(x.values, y.values, s=8, alpha=0.45,
                color="#555555", edgecolors="none")
    # OLS fit line
    m, b, *_ = stats.linregress(x.dropna(), y.reindex(x.dropna().index).dropna())
    xg = np.linspace(x.min(), x.max(), 200)
    ax2.plot(xg, m * xg + b, color="#222222", linewidth=1.6)
    ax2.axhline(0, color="black", linewidth=0.4)
    ax2.axvline(0, color="black", linewidth=0.4)
    ax2.set_xlabel("NOAA ERSSTv5 Niño-3.4 (°C)")
    ax2.set_ylabel("Colombian Pacific SST (°C)")
    ax2.set_title("(b) Scatter (1961–2024)")
    ax2.text(0.05, 0.97, f"ρ = {rho:.3f}\nslope = {m:.3f}",
             transform=ax2.transAxes, fontsize=10, va="top", color="dimgray",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"))

    fig.suptitle(
        "ENSO Index Comparison: NOAA ERSSTv5 Niño-3.4 vs Colombian Pacific SST",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_enso_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig_enso_comparison.png")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Loading regional data...")
    regions = load_all_regions()
    df_col  = regions["colombia"].loc["1961":"2024"]

    print("Loading SST and ONI indices...")
    sst       = load_sst().loc["1961":"2024"]
    oni       = load_oni().loc["1961":"2024"]
    local_sst = load_local_sst().loc["1961":"2024"]

    print("\nBuilding ENSO sensitivity table...")
    enso_table = build_enso_table(
        {k: v.loc["1961":"2024"] for k, v in regions.items()}, sst)
    enso_table.to_csv(OUT / "table3_enso_sensitivity.csv", index=False)
    print("table3_enso_sensitivity.csv")
    print(enso_table[enso_table.region == "colombia"].to_string(index=False))

    print("\n--- Regression diagnostics (ACI-CO national) ---")
    r_aci = fit_enso(df_col["aci"], sst)
    print_diagnostics(r_aci, "ACI-CO national")

    print("\nGenerating figures...")
    fig1_colombia_map()
    fig4_national_aci(df_col, oni)
    fig5_components(df_col, oni)
    fig5b_distributions(df_col)
    fig2_enso_decomposition(df_col, sst, oni)
    fig2b_enso_regression_scatter(df_col, sst)
    fig_enso_diagnostics(df_col, sst)
    fig7a_regional({k: v.loc["1961":"2024"] for k, v in regions.items()}, oni)
    fig_correlation(df_col, sst)
    fig6_regional_heatmap({k: v.loc["1961":"2024"] for k, v in regions.items()})
    fig_enso_comparison(sst, local_sst)

    print("\nBuilding Mann-Kendall trend table...")
    mk_table = build_mk_table({k: v.loc["1961":"2024"] for k, v in regions.items()})
    mk_table.to_csv(OUT / "table6_mann_kendall.csv", index=False)
    print("table6_mann_kendall.csv")
    print(mk_table[mk_table.region == "colombia"].to_string(index=False))

    print(f"\nAll outputs in: {OUT}")


if __name__ == "__main__":
    main()
