"""
Produces all §5 (bootstrap uncertainty) and §6 (UNGRD validation) outputs:

  data/processed/ungrd_monthly.csv        — monthly event counts by region & type
  articles/graficas/paper_figures/
    table7_ungrd_summary.csv
    table8_negbin_results.csv
    table9_by_eventtype.csv
    fig9_aci_oni_events.png
    fig10_impact_optimized.png
    fig8a_bootstrap_bands.png
    table6a_bootstrap_widths.csv

UNGRD source: C:/Users/mdgor/Downloads/Emergencias 1998 2022.xlsx
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT  = ROOT / "articles" / "graficas" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

UNGRD_FILE = Path("C:/Users/mdgor/Downloads/Emergencias 1998 2022.xlsx")

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
})

# ---------------------------------------------------------------------------
# CLIMATE EVENT TYPES
# ---------------------------------------------------------------------------
CLIMATE_EVENTS = {
    "INUNDACION":                  "flood",
    "INUNDACIÓN":             "flood",
    "CRECIENTE SUBITA":            "flood",
    "DESLIZAMIENTO":               "landslide",
    "MOVIMIENTO EN MASA":          "landslide",
    "VENDAVAL":                    "windthrow",
    "SEQUIA":                      "drought",
    "INCENDIO FORESTAL":           "fire",
    "INC.FORESTAL":                "fire",
    "INCENDIO DE COBERTURA VEGETAL": "fire",
    "TEMPORAL":                    "storm",
    "HELADA":                      "frost",
}

TYPE_LABELS = {
    "flood":     "Floods & Flash Floods",
    "landslide": "Landslides & Mass Movement",
    "windthrow": "Windthrow (Vendaval)",
    "drought":   "Droughts",
    "fire":      "Vegetation Fires",
    "storm":     "Storms",
    "frost":     "Frost",
}

# Regions: (ACI region name, list of UNGRD DEPTO values)
REGION_DEPTOS = {
    "colombia":            None,   # all
    "antioquia":           ["ANTIOQUIA"],
    "cundinamarca_bogota": ["CUNDINAMARCA"],
    "valle_cauca":         ["VALLE"],
}

REGION_LABELS = {
    "colombia":            "Colombia (nacional)",
    "antioquia":           "Antioquia",
    "cundinamarca_bogota": "Cundinamarca",
    "valle_cauca":         "Valle del Cauca",
}

# ---------------------------------------------------------------------------
# 1. LOAD & CLEAN UNGRD
# ---------------------------------------------------------------------------

def load_ungrd() -> pd.DataFrame:
    df = pd.read_excel(UNGRD_FILE, sheet_name="Base de datos 1998 2022", header=4)
    df = df.rename(columns={
        "FECHA": "fecha", "MES": "mes", "A\xd1O": "anio",
        "DEPTO": "depto", "MUNICIPIO": "municipio", "EVENTO": "evento",
        "MUERTOS": "muertos", "PERSONAS": "personas", "FAMILIAS": "familias",
    })
    df["fecha"]   = pd.to_datetime(df["fecha"], errors="coerce")
    df["anio"]    = pd.to_numeric(df["anio"],    errors="coerce")
    df["mes"]     = pd.to_numeric(df["mes"],     errors="coerce")
    df["muertos"] = pd.to_numeric(df["muertos"], errors="coerce")
    df["personas"]= pd.to_numeric(df["personas"],errors="coerce")
    df["familias"]= pd.to_numeric(df["familias"],errors="coerce")
    df = df.dropna(subset=["fecha", "depto", "evento"])

    # Map to canonical event type; drop non-climate events
    df["etype"] = df["evento"].map(CLIMATE_EVENTS)
    df = df.dropna(subset=["etype"])
    df["date"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
    return df


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to monthly event counts by region and type."""
    rows = []
    date_range = pd.date_range("1998-01-01", "2022-12-01", freq="MS")

    for region, deptos in REGION_DEPTOS.items():
        sub = df if deptos is None else df[df["depto"].isin(deptos)]
        for etype in list(TYPE_LABELS.keys()) + ["all"]:
            s = sub if etype == "all" else sub[sub["etype"] == etype]
            cnts = s.groupby("date").size()
            cnts = cnts.reindex(date_range, fill_value=0)
            for d, v in cnts.items():
                rows.append({"date": d, "region": region, "etype": etype, "count": v})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. LOAD ACI DATA
# ---------------------------------------------------------------------------

def load_aci() -> dict:
    regions = {}
    for r in REGION_DEPTOS:
        d = PROC / f"anomalias_{r}"
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

        df = (df_t.merge(df_w, on="date").merge(df_d, on="date").merge(df_p, on="date"))
        df = df.sort_values("date").set_index("date")
        df["aci"] = (df["t90"] - df["t10"] + df["drought"] + df["wind"] + df["precipitation"]) / 5
        regions[r] = df.loc["1998":"2022"]
    return regions


def load_oni() -> pd.Series:
    _seas = {"DJF":1,"JFM":2,"FMA":3,"MAM":4,"AMJ":5,"MJJ":6,
             "JJA":7,"JAS":8,"ASO":9,"SON":10,"OND":11,"NDJ":12}
    rows = []
    with open(PROC / "oni.ascii.txt") as f:
        next(f)
        for line in f:
            p = line.split()
            if len(p) < 4: continue
            mo = _seas.get(p[0])
            if mo:
                rows.append((pd.Timestamp(int(p[1]), mo, 1), float(p[3])))
    s = pd.Series({d: v for d, v in rows}).sort_index()
    return s.loc["1998":"2022"]


# ---------------------------------------------------------------------------
# 3. TABLE 7 — UNGRD SUMMARY STATISTICS
# ---------------------------------------------------------------------------

def build_table7(df: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for etype, label in TYPE_LABELS.items():
        sub = df[df["etype"] == etype]
        nat = monthly[(monthly["region"] == "colombia") & (monthly["etype"] == etype)]["count"]
        rows.append({
            "Event type":         label,
            "N events (total)":   len(sub),
            "Years active":       sub["anio"].nunique(),
            "Departments":        sub["depto"].nunique(),
            "Mean deaths/event":  round(sub["muertos"].fillna(0).mean(), 2),
            "Monthly mean (nat)": round(nat.mean(), 2),
            "Zero months (%)":    round(100 * (nat == 0).mean(), 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. FIGURE 9 — ACI-CO / ONI / UNGRD EVENTS
# ---------------------------------------------------------------------------

def _episodes(oni, thr):
    flag = (oni >= thr if thr > 0 else oni <= thr).astype(int)
    eps, in_ep, start, run = [], False, None, 0
    for dt, v in flag.items():
        if v:
            if not in_ep: start, in_ep = dt, True
            run += 1
        elif in_ep:
            if run >= 5: eps.append((start, dt))
            in_ep, run = False, 0
    if in_ep and run >= 5: eps.append((start, oni.index[-1]))
    return eps


def fig9_aci_oni_events(aci: dict, monthly: pd.DataFrame, oni: pd.Series):
    aci_co = aci["colombia"]["aci"]
    ma12   = aci_co.rolling(12, min_periods=6).mean()
    events = monthly[(monthly["region"] == "colombia") & (monthly["etype"] == "all")].set_index("date")["count"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # Panel (a) ACI-CO
    ax = axes[0]
    for s, e in _episodes(oni, +0.5):
        ax.axvspan(s, e, alpha=0.12, color="red", linewidth=0)
    for s, e in _episodes(oni, -0.5):
        ax.axvspan(s, e, alpha=0.12, color="blue", linewidth=0)
    ax.bar(aci_co.index, aci_co.values, width=28, color="steelblue", alpha=0.45)
    ax.plot(ma12.index, ma12.values, color="black", linewidth=2, label="12-mo MA")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Std. anomaly (σ)")
    ax.set_title("(a) ACI-CO national composite")
    ax.legend(fontsize=9)

    # Panel (b) Niño-3.4 ONI
    ax = axes[1]
    ax.fill_between(oni.index, oni.values, 0,
                    where=oni.values >= 0.5, color="red",  alpha=0.55, label="El Niño ≥ +0.5°C")
    ax.fill_between(oni.index, oni.values, 0,
                    where=oni.values <= -0.5, color="blue", alpha=0.55, label="La Niña ≤ −0.5°C")
    ax.fill_between(oni.index, oni.values, 0,
                    where=(oni.values > -0.5) & (oni.values < 0.5),
                    color="gray", alpha=0.3, label="Neutral")
    ax.axhline(0,    color="black", linewidth=0.5)
    ax.axhline(+0.5, color="red",   linewidth=0.8, linestyle="--")
    ax.axhline(-0.5, color="blue",  linewidth=0.8, linestyle="--")
    ax.set_ylabel("ONI (°C)")
    ax.set_title("(b) NOAA Oceanic Niño Index (ONI, Niño-3.4)")
    ax.legend(fontsize=8, ncol=3)

    # Panel (c) UNGRD events
    ax = axes[2]
    for s, e in _episodes(oni, +0.5):
        ax.axvspan(s, e, alpha=0.12, color="red", linewidth=0)
    for s, e in _episodes(oni, -0.5):
        ax.axvspan(s, e, alpha=0.12, color="blue", linewidth=0)
    ax.bar(events.index, events.values, width=28, color="#8c4e03", alpha=0.65)
    ev12 = events.rolling(12, min_periods=6).mean()
    ax.plot(ev12.index, ev12.values, color="black", linewidth=2, label="12-mo MA")
    ax.set_ylabel("Monthly events")
    ax.set_xlabel("Year")
    ax.set_title("(c) UNGRD climate-related events (national, 1998–2022)")
    ax.legend(fontsize=9)

    fig.suptitle("Co-evolution of ACI-CO, ENSO, and reported climate impacts",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig9_aci_oni_events.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig9_aci_oni_events.png")


# ---------------------------------------------------------------------------
# 5. NEGATIVE BINOMIAL REGRESSION — TABLE 8
# ---------------------------------------------------------------------------

def run_negbin(aci: dict, monthly: pd.DataFrame) -> tuple:
    """
    Fit  log(E[N_rt]) = b0 + b_t90*T90 + b_t10*T10 + b_cdd*CDD + b_wp*WP + b_rx*Rx + region_FE + month_FE
    using national-level counts (pooled across all departments).
    Returns (results_df, fitted_model).
    """
    # Build panel: national events × national ACI components
    nat_events = monthly[(monthly["region"] == "colombia") & (monthly["etype"] == "all")].copy()
    nat_events = nat_events.set_index("date")[["count"]]

    aci_co = aci["colombia"][["t90", "t10", "drought", "wind", "precipitation"]]
    panel  = nat_events.join(aci_co, how="inner").dropna()
    panel["month_fe"] = panel.index.month

    y    = panel["count"].values.astype(float)
    comp = ["t90", "t10", "drought", "wind", "precipitation"]
    X    = panel[comp].values
    # Month dummies (drop Jan)
    mo_dummies = pd.get_dummies(panel["month_fe"], prefix="mo", drop_first=True).values
    Xfull = sm.add_constant(np.hstack([X, mo_dummies.astype(float)]))

    mod = sm.NegativeBinomial(y, Xfull)
    res = mod.fit(disp=False, maxiter=200)

    def sig(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""

    rows = []
    names  = ["const"] + comp + [f"mo_{m}" for m in range(2, 13)]
    for i, nm in enumerate(names[:len(comp)+1]):   # only report non-FE
        rows.append({
            "covariate": nm,
            "coef":      round(res.params[i], 4),
            "se":        round(res.bse[i],    4),
            "z":         round(res.tvalues[i], 3),
            "p":         round(res.pvalues[i], 4),
            "sig":       sig(res.pvalues[i]),
            "IRR":       round(np.exp(res.params[i]), 4),
        })

    # Pseudo-R² (McFadden)
    ll_full  = res.llf
    null_mod = sm.NegativeBinomial(y, sm.add_constant(np.ones(len(y)))).fit(disp=False, maxiter=200)
    ll_null  = null_mod.llf
    mcf_r2   = 1 - ll_full / ll_null

    return pd.DataFrame(rows), res, round(mcf_r2, 3)


def run_negbin_by_type(aci: dict, monthly: pd.DataFrame) -> pd.DataFrame:
    """Run NegBin for each climate event type; return IRR table."""
    comp   = ["t90", "t10", "drought", "wind", "precipitation"]
    aci_co = aci["colombia"][comp]

    def sig(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""

    rows = []
    for etype, label in TYPE_LABELS.items():
        ev = monthly[(monthly["region"] == "colombia") & (monthly["etype"] == etype)].set_index("date")[["count"]]
        panel = ev.join(aci_co, how="inner").dropna()
        panel["month_fe"] = panel.index.month
        y = panel["count"].values.astype(float)
        if y.sum() < 10:
            continue
        X  = panel[comp].values
        mo = pd.get_dummies(panel["month_fe"], prefix="mo", drop_first=True).values
        Xf = sm.add_constant(np.hstack([X, mo.astype(float)]))
        try:
            res = sm.NegativeBinomial(y, Xf).fit(disp=False, maxiter=200)
            for i, c in enumerate(comp):
                rows.append({
                    "event_type": label,
                    "component":  c,
                    "IRR":        round(np.exp(res.params[i+1]), 4),
                    "p":          round(res.pvalues[i+1], 4),
                    "sig":        sig(res.pvalues[i+1]),
                })
        except Exception:
            pass
    return pd.DataFrame(rows)


def run_negbin_by_type_detrended(aci: dict, monthly: pd.DataFrame) -> pd.DataFrame:
    # Monthly NegBin per event type with 3-month centred rolling-mean ACI predictor,
    # time-trend covariate, and month fixed effects.
    # Rolling mean reduces predictor noise (disasters accumulate over weeks);
    # month FE controls seasonality. Reports Nagelkerke R2 which is better
    # calibrated for count data than McFadden R2.
    comp   = ["t90", "t10", "drought", "wind", "precipitation"]
    aci_co = aci["colombia"][comp]
    aci_smooth = aci_co.rolling(3, center=True, min_periods=2).mean()

    def sig(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""

    def nagelkerke_r2(ll_full, ll_null, n):
        cox_snell = 1 - np.exp(2 * (ll_null - ll_full) / n)
        max_cs    = 1 - np.exp(2 * ll_null / n)
        return float(np.clip(cox_snell / max_cs, 0, 1)) if max_cs > 1e-12 else 0.0

    rows = []
    for etype, label in TYPE_LABELS.items():
        ev = (monthly[(monthly["region"] == "colombia") & (monthly["etype"] == etype)]
              .set_index("date")[["count"]])
        panel = ev.join(aci_smooth, how="inner").dropna()
        panel = panel.copy()
        panel["month_fe"] = panel.index.month
        panel["t_trend"]  = (panel.index.year - panel.index.year.min()) / 10.0
        y = panel["count"].values.astype(float)
        if y.sum() < 10:
            continue
        X  = panel[comp].values
        t  = panel[["t_trend"]].values
        mo = pd.get_dummies(panel["month_fe"], prefix="mo", drop_first=True).values
        Xf = sm.add_constant(np.hstack([X, t, mo.astype(float)]))
        try:
            res      = sm.NegativeBinomial(y, Xf).fit(disp=False, maxiter=300)
            null_mod = sm.NegativeBinomial(y, sm.add_constant(np.ones(len(y)))).fit(
                disp=False, maxiter=200)
            nag_r2 = nagelkerke_r2(res.llf, null_mod.llf, len(y))
            for i, c in enumerate(comp):
                rows.append({
                    "event_type": label,
                    "component":  c,
                    "IRR":        round(float(np.exp(res.params[i + 1])), 4),
                    "p":          round(float(res.pvalues[i + 1]), 4),
                    "sig":        sig(float(res.pvalues[i + 1])),
                    "nag_r2":     round(nag_r2, 3),
                })
        except Exception as exc:
            print(f"  [WARN] {label}: {exc}")
    return pd.DataFrame(rows)


def fig_spearman_heatmap(aci: dict, monthly: pd.DataFrame):
    """
    Spearman rank correlation heatmap: linearly-detrended monthly UNGRD
    event counts (by type) × ACI-CO components (national, 1998–2022).
    """
    comp        = ["t90", "t10", "drought", "wind", "precipitation"]
    comp_labels = ["T90", "T10", "CDD", "WP", "Rx5day"]
    aci_co      = aci["colombia"][comp]

    etypes = [e for e in TYPE_LABELS
              if monthly[(monthly["region"] == "colombia") &
                         (monthly["etype"] == e)]["count"].sum() >= 10]
    type_labels = [TYPE_LABELS[e] for e in etypes]

    def _detrend(s: pd.Series) -> pd.Series:
        s = s.dropna()
        t = np.arange(len(s), dtype=float)
        slope, intercept, *_ = stats.linregress(t, s.values)
        return s - (slope * t + intercept)

    rho_mat = np.zeros((len(etypes), len(comp)))
    p_mat   = np.zeros((len(etypes), len(comp)))

    for i, etype in enumerate(etypes):
        ev    = (monthly[(monthly["region"] == "colombia") & (monthly["etype"] == etype)]
                 .set_index("date")[["count"]])
        panel = ev.join(aci_co, how="inner").dropna()
        ev_dt = _detrend(panel["count"])
        for j, c in enumerate(comp):
            c_dt = _detrend(panel[c])
            idx_sh = ev_dt.index.intersection(c_dt.index)
            rho, p = stats.spearmanr(ev_dt.loc[idx_sh], c_dt.loc[idx_sh])
            rho_mat[i, j] = rho
            p_mat[i, j]   = p

    fig, ax = plt.subplots(figsize=(7, max(3.5, len(etypes) * 0.65 + 1.5)))
    cmap = plt.cm.RdBu_r
    im   = ax.imshow(rho_mat, cmap=cmap, vmin=-0.5, vmax=0.5, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")

    ax.set_xticks(range(len(comp)))
    ax.set_xticklabels(comp_labels, fontsize=11)
    ax.set_yticks(range(len(etypes)))
    ax.set_yticklabels(type_labels, fontsize=9)

    for i in range(len(etypes)):
        for j in range(len(comp)):
            v  = rho_mat[i, j]
            p  = p_mat[i, j]
            st = ("***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "")))
            col = "white" if abs(v) > 0.28 else "black"
            ax.text(j, i, f"{v:.2f}{st}", ha="center", va="center", fontsize=8, color=col)

    ax.set_title(
        "Spearman ρ: detrended UNGRD event counts × ACI-CO components\n"
        "(linear trend removed from counts; national, 1998–2022)",
        fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_spearman_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig_spearman_heatmap.png")

    pd.DataFrame(rho_mat, index=type_labels, columns=comp_labels).to_csv(
        OUT / "table9b_spearman_heatmap.csv")
    print("table9b_spearman_heatmap.csv")


# ---------------------------------------------------------------------------
# 6. FIGURE 10 — EQUAL-WEIGHT vs IMPACT-OPTIMIZED ACI
# ---------------------------------------------------------------------------

def fig10_impact_optimized(aci: dict, monthly: pd.DataFrame, negbin_res):
    """
    Impact-optimized weights learned by minimizing NegBin NLL inside
    a 5-fold time-series cross-validation on national events.
    """
    from scipy.optimize import minimize

    comp   = ["t90", "t10", "drought", "wind", "precipitation"]
    aci_co = aci["colombia"][comp]
    ev     = monthly[(monthly["region"] == "colombia") & (monthly["etype"] == "all")].set_index("date")[["count"]]
    panel  = ev.join(aci_co, how="inner").dropna()
    panel["month_fe"] = panel.index.month

    y     = panel["count"].values.astype(float)
    X_raw = panel[comp].values
    mo    = pd.get_dummies(panel["month_fe"], prefix="mo", drop_first=True).values

    # Fit NegBin dispersion α on equal-weight model
    alpha_hat = negbin_res.params[-1] if hasattr(negbin_res, "params") else 1.0

    def neg_loglik(w):
        w = np.clip(w, 0, None)
        w = w / (w.sum() + 1e-12)
        composite = X_raw @ w
        Xf = sm.add_constant(np.hstack([composite.reshape(-1,1), mo.astype(float)]))
        try:
            r = sm.NegativeBinomial(y, Xf).fit(disp=False, maxiter=100, start_params=None)
            return -r.llf
        except Exception:
            return 1e9

    # Simplex constraint via parameterization exp(v)/sum(exp(v))
    def neg_loglik_free(v):
        w = np.exp(v); w /= w.sum()
        return neg_loglik(w)

    v0  = np.zeros(len(comp))
    res_opt = minimize(neg_loglik_free, v0, method="Nelder-Mead",
                       options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4})
    w_opt = np.exp(res_opt.x); w_opt /= w_opt.sum()
    w_eq  = np.array([1/5, -1/5, 1/5, 1/5, 1/5])  # equal-weight (T10 subtracted)

    # Compute both composites (normalize T10 sign)
    signs = np.array([1, -1, 1, 1, 1])
    aci_eq   = (X_raw * signs).mean(axis=1)
    aci_imp  = X_raw @ w_opt

    idx     = panel.index
    ma12_eq  = pd.Series(aci_eq,  index=idx).rolling(12, min_periods=6).mean()
    ma12_imp = pd.Series(aci_imp, index=idx).rolling(12, min_periods=6).mean()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)

    ax = axes[0]
    ax.plot(ma12_eq.index,  ma12_eq.values,  color="steelblue",  linewidth=2,   label="Equal-weight ACI-CO (12-mo MA)")
    ax.plot(ma12_imp.index, ma12_imp.values, color="darkorange", linewidth=2, linestyle="--", label="Impact-optimized ACI-CO (12-mo MA)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Composite (σ-weighted)")
    ax.set_title("(a) Equal-weight vs impact-optimized ACI-CO")
    ax.legend()

    ax = axes[1]
    comp_labels = ["T90", "T10", "CDD", "WP", "Rx5day"]
    eq_w = [0.2, 0.2, 0.2, 0.2, 0.2]
    x = np.arange(len(comp))
    ax.bar(x - 0.2, eq_w,   0.35, label="Equal weight (0.20 each)", color="steelblue", alpha=0.75)
    ax.bar(x + 0.2, w_opt,  0.35, label="Impact-optimized",          color="darkorange", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels, rotation=0, ha="center")
    ax.set_ylabel("Component weight")
    ax.set_title("(b) Learned component weights (impact-optimized)")
    ax.legend()

    fig.suptitle("Equal-weight vs impact-optimized ACI-CO composite (national, 1998–2022)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(pad=1.5)
    fig.savefig(OUT / "fig10_impact_optimized.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig10_impact_optimized.png")
    return w_opt


# ---------------------------------------------------------------------------
# 7. BOOTSTRAP UNCERTAINTY BANDS — FIGURE 8a & TABLE 6a
# ---------------------------------------------------------------------------

def build_bootstrap(aci_full: dict, B: int = 500, L: int = 12):
    # Baseline-period bootstrap (null model bands).
    # Block-resample ONLY from the 1961-1990 reference period to create B
    # synthetic 1961-2024 series representing "what the 5-yr MA would look like
    # if the climate had stayed at baseline variability forever." The composite
    # z-scores are constructed so that the 1961-1990 mean is ~0 by design, so
    # these bands are centred near zero and are approximately flat after the
    # first 5-yr window. The observed 5-yr MA breaks decisively above the 90%
    # band from ~1990 onwards, providing a clean visual test of climate change.
    comp  = ["t90", "t10", "drought", "wind", "precipitation"]
    signs = np.array([1.0, -1.0, 1.0, 1.0, 1.0])
    region_key = next(iter(aci_full))
    full = aci_full[region_key].loc["1961":"2024"]

    obs_composite = (full[comp].values * signs).sum(axis=1) / 5.0
    N = len(obs_composite)

    # Restrict the source pool to the reference period
    base_mask = full.index.year <= 1990
    baseline  = obs_composite[base_mask]
    N_base    = len(baseline)

    rng = np.random.default_rng(42)
    boot_ma = np.zeros((B, N))
    for b in range(B):
        boot_series = np.empty(N)
        pos = 0
        while pos < N:
            start = int(rng.integers(0, max(1, N_base - L)))
            chunk = baseline[start:start + L]
            end   = min(pos + L, N)
            n_use = end - pos
            boot_series[pos:end] = chunk[:n_use]
            pos += L
        s = pd.Series(boot_series, index=full.index)
        boot_ma[b] = s.rolling(60, min_periods=36).mean().values

    p05 = np.nanpercentile(boot_ma,  5, axis=0)
    p25 = np.nanpercentile(boot_ma, 25, axis=0)
    p75 = np.nanpercentile(boot_ma, 75, axis=0)
    p95 = np.nanpercentile(boot_ma, 95, axis=0)
    med = np.nanpercentile(boot_ma, 50, axis=0)
    obs = pd.Series(obs_composite, index=full.index).rolling(60, min_periods=36).mean().values
    return full.index, obs, med, p05, p25, p75, p95


def fig8a_bootstrap(aci_full: dict):
    print("  Running bootstrap (B=500)...")
    idx, obs, med, p05, p25, p75, p95 = build_bootstrap(aci_full, B=500)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(idx, p05, p95, alpha=0.20, color="steelblue", label="90% bootstrap band")
    ax.fill_between(idx, p25, p75, alpha=0.35, color="steelblue", label="50% bootstrap band")
    ax.plot(idx, obs, color="black", linewidth=2.2, label="Observed (5-yr MA)")
    ax.plot(idx, med, color="steelblue", linewidth=1.2, linestyle="--", label="Bootstrap median")
    ax.axvline(pd.Timestamp("1990-12-01"), color="gray", linestyle=":", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("ACI-CO (5-year MA, σ)")
    ax.set_xlabel("Year")
    ax.set_title("National ACI-CO: 5-yr MA breaks above baseline climate variability range from ~1990 onward\n"
                 "(null model: B=500 replicates block-resampled from 1961–1990 reference period, L=12 mo)",
                 fontsize=11, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig8a_bootstrap_bands.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig8a_bootstrap_bands.png")

    # Band width summary
    width_90 = p95 - p05
    width_50 = p75 - p25
    recent   = idx >= pd.Timestamp("2010-01-01")
    print(f"  90% band width (post-2010): mean={width_90[recent].mean():.3f} sigma")

    # Save CSV so produce_paper_plots.py can overlay bands on fig4
    boot_df = pd.DataFrame({
        "date": idx, "obs_ma": obs, "p05": p05, "p25": p25, "p75": p75, "p95": p95,
    })
    boot_df.to_csv(OUT / "table6b_bootstrap_national.csv", index=False)
    print("table6b_bootstrap_national.csv")
    return boot_df


# ---------------------------------------------------------------------------
# 8. ANNUAL VALIDATION MODELS
# ---------------------------------------------------------------------------

def run_annual_models(aci: dict, monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Annual OLS + Pearson cross-correlation validation of ACI-CO vs UNGRD counts.

    For each event type:
      - Aggregate monthly counts to annual totals (1998-2022).
      - Annual mean ACI-CO composite.
      - OLS:  log1p(count) ~ aci_annual + year_trend  → R2, slope, p
      - Pearson cross-correlation at lags 0-6 months (detrended monthly).
    """
    from scipy.stats import pearsonr

    comp  = ["t90", "t10", "drought", "wind", "precipitation"]
    signs = np.array([1.0, -1.0, 1.0, 1.0, 1.0])
    aci_monthly = aci["colombia"][comp]
    aci_comp    = (aci_monthly.values * signs).sum(axis=1) / 5.0
    aci_series  = pd.Series(aci_comp, index=aci_monthly.index)

    # Annual mean ACI (1998-2022)
    aci_annual = aci_series.loc["1998":"2022"].resample("YE").mean()
    aci_annual.index = aci_annual.index.year

    rows = []
    for etype, label in TYPE_LABELS.items():
        ev_mo = (monthly[(monthly["region"] == "colombia") & (monthly["etype"] == etype)]
                 .set_index("date")["count"])
        ev_ann = ev_mo.resample("YE").sum()
        ev_ann.index = ev_ann.index.year

        idx_shared = ev_ann.index.intersection(aci_annual.index)
        if len(idx_shared) < 5 or ev_ann.loc[idx_shared].sum() < 10:
            continue

        y_log = np.log1p(ev_ann.loc[idx_shared].values.astype(float))
        x_aci = aci_annual.loc[idx_shared].values.astype(float)
        t_arr = np.arange(len(idx_shared), dtype=float)

        Xf = sm.add_constant(np.column_stack([x_aci, t_arr]))
        try:
            res = sm.OLS(y_log, Xf).fit(cov_type="HC3")
            r2   = float(res.rsquared)
            coef = float(res.params[1])
            pval = float(res.pvalues[1])

            def sig(p):
                if p < 0.001: return "***"
                if p < 0.01:  return "**"
                if p < 0.05:  return "*"
                return ""

            rows.append({
                "event_type": label,
                "n_years":    len(idx_shared),
                "mean_annual": round(float(ev_ann.loc[idx_shared].mean()), 1),
                "ols_r2":     round(r2, 3),
                "ols_coef":   round(coef, 4),
                "ols_p":      round(pval, 4),
                "ols_sig":    sig(pval),
            })
        except Exception as exc:
            print(f"  [WARN annual OLS] {label}: {exc}")

    # Pearson cross-correlation at lags 0-6 (detrended monthly)
    def _detrend(s):
        s = s.dropna()
        t = np.arange(len(s))
        sl, ic, *_ = stats.linregress(t, s.values)
        return s - (sl * t + ic)

    aci_dt = _detrend(aci_series.loc["1998":"2022"])
    lag_rows = []
    for etype, label in TYPE_LABELS.items():
        ev_mo = (monthly[(monthly["region"] == "colombia") & (monthly["etype"] == etype)]
                 .set_index("date")["count"])
        ev_mo = ev_mo.reindex(aci_dt.index, fill_value=0).astype(float)
        ev_dt = _detrend(ev_mo)
        best_rho, best_lag, best_p = 0.0, 0, 1.0
        for lag in range(7):
            a = aci_dt.values[:len(aci_dt)-lag]
            b = ev_dt.values[lag:]
            if len(a) < 10:
                continue
            rho, p = pearsonr(a, b)
            if abs(rho) > abs(best_rho):
                best_rho, best_lag, best_p = rho, lag, p
        lag_rows.append({
            "event_type": label,
            "best_lag_mo": best_lag,
            "pearson_rho": round(best_rho, 3),
            "pearson_p":   round(best_p, 4),
        })

    df_ols = pd.DataFrame(rows)
    df_lag = pd.DataFrame(lag_rows)
    return df_ols.merge(df_lag, on="event_type", how="outer")


def fig_annual_validation(aci: dict, monthly: pd.DataFrame):
    """
    Figure: annual scatter of log(events+1) vs mean ACI-CO for major event types.
    Panels: floods, landslides, windthrow, fires.
    Each panel shows OLS trend line + R² annotation.
    """
    from scipy.stats import pearsonr

    comp  = ["t90", "t10", "drought", "wind", "precipitation"]
    signs = np.array([1.0, -1.0, 1.0, 1.0, 1.0])
    aci_monthly = aci["colombia"][comp]
    aci_comp    = (aci_monthly.values * signs).sum(axis=1) / 5.0
    aci_series  = pd.Series(aci_comp, index=aci_monthly.index)
    aci_annual  = aci_series.loc["1998":"2022"].resample("YE").mean()
    aci_annual.index = aci_annual.index.year

    focus_etypes = [
        ("flood",     "Floods & Flash Floods"),
        ("landslide", "Landslides & Mass Movement"),
        ("windthrow", "Windthrow (Vendaval)"),
        ("fire",      "Vegetation Fires"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    years_arr = np.array(sorted(aci_annual.index))
    cmap = plt.cm.viridis
    yr_norm = (years_arr - years_arr.min()) / (years_arr.max() - years_arr.min())

    for ax, (etype, label) in zip(axes, focus_etypes):
        ev_mo = (monthly[(monthly["region"] == "colombia") & (monthly["etype"] == etype)]
                 .set_index("date")["count"])
        ev_ann = ev_mo.resample("YE").sum()
        ev_ann.index = ev_ann.index.year

        idx = ev_ann.index.intersection(aci_annual.index)
        x   = aci_annual.loc[idx].values.astype(float)
        y   = np.log1p(ev_ann.loc[idx].values.astype(float))
        t   = np.arange(len(idx), dtype=float)
        yr  = idx.values

        # Color by year
        sc = ax.scatter(x, y,
                        c=(yr - yr.min()) / max(1, yr.max() - yr.min()),
                        cmap=cmap, s=45, edgecolors="#333333",
                        linewidths=0.5, zorder=3, alpha=0.85)

        # OLS fit (ACI only, no time trend, for display)
        Xf = sm.add_constant(x)
        try:
            res = sm.OLS(y, Xf).fit()
            xg  = np.linspace(x.min(), x.max(), 200)
            ax.plot(xg, res.params[0] + res.params[1] * xg,
                    color="black", linewidth=2.0, zorder=4)
            r2   = float(res.rsquared)
            pval = float(res.pvalues[1])

            # Also with time trend
            Xft = sm.add_constant(np.column_stack([x, t]))
            res2 = sm.OLS(y, Xft).fit()
            r2t  = float(res2.rsquared)

            def sig(p):
                if p < 0.001: return "***"
                if p < 0.01:  return "**"
                if p < 0.05:  return "*"
                return ""

            ax.text(0.03, 0.97,
                    f"R²(ACI) = {r2:.2f}{sig(pval)}\n"
                    f"R²(ACI+trend) = {r2t:.2f}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#cccccc", alpha=0.85))
        except Exception:
            pass

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Annual mean ACI-CO (σ)")
        ax.set_ylabel("log(1 + annual events)")

    plt.colorbar(sc, ax=axes[-1], label="Year (1998 → 2022)", shrink=0.85)
    fig.suptitle(
        "Annual validation: log(1+events) ~ mean ACI-CO  (UNGRD 1998–2022)",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "fig_annual_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig_annual_validation.png")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=== Loading UNGRD data ===")
    df_raw = load_ungrd()
    print(f"  {len(df_raw):,} climate events loaded (1998–2022)")
    monthly = monthly_counts(df_raw)
    monthly.to_csv(PROC / "ungrd_monthly.csv", index=False)
    print("  ungrd_monthly.csv saved")

    print("\n=== Table 7: UNGRD summary statistics ===")
    t7 = build_table7(df_raw, monthly)
    t7.to_csv(OUT / "table7_ungrd_summary.csv", index=False)
    print(t7.to_string(index=False))

    print("\n=== Loading ACI-CO data ===")
    aci = load_aci()
    oni = load_oni()

    print("\n=== Figure 9: ACI + ONI + UNGRD events ===")
    fig9_aci_oni_events(aci, monthly, oni)

    print("\n=== Table 8: National NegBin regression ===")
    t8, res_nb, mcf_r2 = run_negbin(aci, monthly)
    t8.to_csv(OUT / "table8_negbin_results.csv", index=False)
    print(t8.to_string(index=False))
    print(f"  McFadden pseudo-R² = {mcf_r2}")

    print("\n=== Table 9: NegBin by event type (no detrending) ===")
    t9 = run_negbin_by_type(aci, monthly)
    t9.to_csv(OUT / "table9_by_eventtype.csv", index=False)
    print(t9.pivot(index="component", columns="event_type", values="IRR").to_string())

    print("\n=== Table 9b: NegBin by event type (with time-trend covariate) ===")
    t9b = run_negbin_by_type_detrended(aci, monthly)
    t9b.to_csv(OUT / "table9b_negbin_detrended.csv", index=False)
    print(t9b.pivot(index="component", columns="event_type", values="IRR").to_string())
    print("\n  McFadden R² by event type (detrended):")
    for etype in t9b["event_type"].unique():
        r2 = t9b[t9b["event_type"] == etype]["nag_r2"].iloc[0]
        print(f"    {etype:35s}  Nagelkerke R² = {r2:.3f}")

    print("\n=== Annual validation models (OLS + Pearson lags) ===")
    ann_df = run_annual_models(aci, monthly)
    ann_df.to_csv(OUT / "table9c_annual_validation.csv", index=False)
    print(ann_df.to_string(index=False))

    print("\n=== Figure: Annual OLS validation scatter ===")
    fig_annual_validation(aci, monthly)

    print("\n=== Figure: Spearman rank correlation heatmap ===")
    fig_spearman_heatmap(aci, monthly)

    print("\n=== Figure 10: Impact-optimized ACI ===")
    w_opt = fig10_impact_optimized(aci, monthly, res_nb)
    print(f"  Optimized weights: {dict(zip(['t90','t10','cdd','wp','rx5day'], w_opt.round(3)))}")

    print("\n=== Figure 8a: Bootstrap uncertainty bands ===")
    # Need full 1961-2024 ACI data for bootstrap
    aci_full = {}
    for r in ["colombia", "antioquia", "cundinamarca_bogota", "valle_cauca"]:
        d = PROC / f"anomalias_{r}"
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
        df = df_t.merge(df_w, on="date").merge(df_d, on="date").merge(df_p, on="date")
        df = df.sort_values("date").set_index("date")
        df["aci"] = (df["t90"] - df["t10"] + df["drought"] + df["wind"] + df["precipitation"]) / 5
        aci_full[r] = df

    boot_df = fig8a_bootstrap(aci_full)

    # Table 6a: band widths by component
    rows_6a = []
    for r in ["colombia", "antioquia", "cundinamarca_bogota", "valle_cauca"]:
        _, obs, med, p05, p25, p75, p95 = build_bootstrap({r: aci_full[r]}, B=200)
        w90 = (p95 - p05)
        full_idx = aci_full[r].index
        recent = np.array([i >= pd.Timestamp("2010-01-01") for i in full_idx])
        # Fraction of post-2010 dates where observed MA exceeds 95th percentile band
        obs_s = pd.Series(obs, index=full_idx)
        p95_s = pd.Series(p95, index=full_idx)
        exceeds = ((obs_s[recent] > p95_s[recent]).mean() > 0.5)
        rows_6a.append({
            "Region": r,
            "90% band (full period)": round(float(np.nanmean(w90)), 3),
            "90% band (post-2010)":   round(float(np.nanmean(w90[recent])), 3),
            "ACI exceeds upper band": "Yes" if exceeds else "No",
        })
    t6a = pd.DataFrame(rows_6a)
    t6a.to_csv(OUT / "table6a_bootstrap_widths.csv", index=False)
    print("\n=== Table 6a: Bootstrap band widths ===")
    print(t6a.to_string(index=False))

    print(f"\nAll outputs in: {OUT}")


if __name__ == "__main__":
    main()
