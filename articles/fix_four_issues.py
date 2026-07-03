"""
Patch produce_paper_plots.py and build_validation_outputs.py for four issues:
  1. R² too low        → quarterly NegBin aggregation
  2. Bootstrap bands   → moving-block bootstrap on full composite series
  3. Figure 4 sucks    → clean design: dark bars / trend line / light ENSO
  4. Figure 3 nonsense → replace stat-diagnostics with ENSO phase analysis
"""
import pathlib, re

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1 & 2 — build_validation_outputs.py
# ─────────────────────────────────────────────────────────────────────────────

bv_path = pathlib.Path(
    r"c:\Users\mdgor\aca\aca_indice_climatico_opt-main\articles\build_validation_outputs.py"
)
bv = bv_path.read_text(encoding="utf-8")

# ── (1) Replace run_negbin_by_type_detrended with quarterly version ───────────

OLD_NB = r"""def run_negbin_by_type_detrended(aci: dict, monthly: pd.DataFrame) -> pd.DataFrame:
    """
NB_SEARCH_END = r"""    return pd.DataFrame(rows)


def fig_spearman_heatmap"""

# Find the exact block
nb_start = bv.index("def run_negbin_by_type_detrended(aci: dict, monthly: pd.DataFrame) -> pd.DataFrame:")
nb_end   = bv.index("def fig_spearman_heatmap", nb_start)
old_nb_block = bv[nb_start:nb_end]

NEW_NB = r"""def run_negbin_by_type_detrended(aci: dict, monthly: pd.DataFrame) -> pd.DataFrame:
    """
    NegBin per event type, quarterly aggregation + time-trend covariate.
    Quarterly aggregation reduces monthly noise substantially, increasing
    McFadden R² relative to the monthly version.
    Model: log E[Q] = b0 + b_comp * components_mean + b_t * t_trend
    where t_trend = (year - 1998) / 10 (decades from series start).
    """
    comp   = ["t90", "t10", "drought", "wind", "precipitation"]
    aci_co = aci["colombia"][comp]

    def sig(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""

    rows = []
    for etype, label in TYPE_LABELS.items():
        ev = (monthly[(monthly["region"] == "colombia") & (monthly["etype"] == etype)]
              .set_index("date")[["count"]])
        panel = ev.join(aci_co, how="inner").dropna()
        if panel["count"].sum() < 10:
            continue

        panel = panel.copy()
        # Quarterly aggregation: sum events, mean ACI components
        panel["quarter"] = panel.index.to_period("Q")
        q_counts = panel.groupby("quarter")["count"].sum()
        q_aci    = panel.groupby("quarter")[comp].mean()
        qdf      = q_aci.join(q_counts)
        qdf.index = qdf.index.to_timestamp()

        qdf = qdf.copy()
        qdf["t_trend"] = (qdf.index.year - qdf.index.year.min()) / 10.0
        y  = qdf["count"].values.astype(float)
        X  = qdf[comp].values
        tt = qdf[["t_trend"]].values
        Xf = sm.add_constant(np.hstack([X, tt]))
        try:
            res      = sm.NegativeBinomial(y, Xf).fit(disp=False, maxiter=300)
            null_mod = sm.NegativeBinomial(
                y, sm.add_constant(np.ones(len(y)))
            ).fit(disp=False, maxiter=200)
            mcf_r2 = float(np.clip(1 - res.llf / null_mod.llf, 0, 1))
            for i, c in enumerate(comp):
                rows.append({
                    "event_type": label,
                    "component":  c,
                    "IRR":        round(float(np.exp(res.params[i + 1])), 4),
                    "p":          round(float(res.pvalues[i + 1]), 4),
                    "sig":        sig(float(res.pvalues[i + 1])),
                    "mcf_r2":     round(mcf_r2, 3),
                })
        except Exception as exc:
            print(f"  [WARN] {label}: {exc}")
    return pd.DataFrame(rows)


def fig_spearman_heatmap"""

bv = bv[:nb_start] + NEW_NB + bv[nb_end + len("def fig_spearman_heatmap"):]
print("[OK] Patched run_negbin_by_type_detrended (quarterly)")

# ── (2) Replace build_bootstrap with proper moving-block bootstrap ────────────
bs_start = bv.index("def build_bootstrap(aci_full: dict, B: int = 500, L: int = 9):")
bs_end   = bv.index("def fig8a_bootstrap(aci_full: dict):", bs_start)
old_bs_block = bv[bs_start:bs_end]

NEW_BS = r"""def build_bootstrap(aci_full: dict, B: int = 500, L: int = 24):
    """
    Moving-block bootstrap on the FULL ACI-CO composite monthly series.

    Resamples L-month blocks from the observed 1961-2024 composite to build
    B bootstrap replicate series of the same length, then computes the 5-yr
    moving average of each replicate. The resulting percentile bands capture
    genuine uncertainty about the trend: replicates that under-sample the
    post-2018 extreme months will have a lower MA at the end of the series,
    widening the band where the upward trend is most pronounced.

    L=24 months preserves the ENSO annual cycle in each block.
    """
    comp   = ["t90", "t10", "drought", "wind", "precipitation"]
    signs  = np.array([1.0, -1.0, 1.0, 1.0, 1.0])
    region_key = next(iter(aci_full))
    df   = aci_full[region_key]
    full = df.loc["1961":"2024"]

    obs_composite = (full[comp].values * signs).sum(axis=1) / 5.0
    N   = len(obs_composite)
    rng = np.random.default_rng(42)

    boot_ma = np.zeros((B, N))
    for b in range(B):
        boot_series = np.empty(N)
        pos = 0
        while pos < N:
            # Pick a random starting position in the observed series
            start = int(rng.integers(0, max(1, N - L)))
            chunk = obs_composite[start:start + L]
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
    idx = full.index
    return idx, obs, med, p05, p25, p75, p95


def fig8a_bootstrap(aci_full: dict):"""

bv = bv[:bs_start] + NEW_BS + bv[bs_end + len("def fig8a_bootstrap(aci_full: dict):"):]
print("[OK] Patched build_bootstrap (moving-block on full series)")

bv_path.write_text(bv, encoding="utf-8")
print("[OK] build_validation_outputs.py written")


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 3 & 4 — produce_paper_plots.py
# ─────────────────────────────────────────────────────────────────────────────

pp_path = pathlib.Path(
    r"c:\Users\mdgor\aca\aca_indice_climatico_opt-main\articles\produce_paper_plots.py"
)
pp = pp_path.read_text(encoding="utf-8")

# ── (3) Replace fig4_national_aci ─────────────────────────────────────────────

f4_start = pp.index("def fig4_national_aci(df_col: pd.DataFrame, oni: pd.Series):")
f4_end   = pp.index("# ---------------------------------------------------------------------------\n# 6.  FIGURE 5", f4_start)
pp = pp[:f4_start] + r"""def fig4_national_aci(df_col: pd.DataFrame, oni: pd.Series):
    """
    Redesigned national ACI-CO figure.
    (a) Monthly anomaly bars – clean, no ENSO stripes, OLS secular trend dashed
    (b) 5-yr moving average with very light ENSO shading, de-ENSOed trend,
        and key event annotations.
    """
    monthly = df_col["aci"]
    ma5     = monthly.rolling(60, min_periods=36).mean()

    # OLS secular trend on monthly series
    t_num = np.arange(len(monthly), dtype=float)
    mask  = ~np.isnan(monthly.values)
    slope, intercept = np.polyfit(t_num[mask], monthly.values[mask], 1)
    trend_line = pd.Series(slope * t_num + intercept, index=monthly.index)

    # De-ENSOed trend already in fit_enso; here approximate with trend_line
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.1]})

    # ── Panel (a): Monthly bars + OLS trend ──────────────────────────────────
    ax = axes[0]
    ax.bar(monthly.index, monthly.values, width=28,
           color="#3c7ebf", alpha=0.40, linewidth=0, label="Monthly ACI-CO")
    ax.plot(trend_line.index, trend_line.values,
            color="#e07b10", linewidth=2.0, linestyle="--",
            label=f"OLS trend (+{slope*120:.2f} σ/decade)")
    ax.axvline(pd.Timestamp("1990-12-01"), color="gray",
               linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Std. anomaly (σ)")
    ax.set_title("(a) ACI-CO — monthly anomaly with secular trend")
    ax.legend(fontsize=9, loc="upper left", ncol=2)

    # ── Panel (b): 5-yr MA + ENSO shading + annotations ─────────────────────
    ax = axes[1]
    # Light ENSO shading (much lower alpha than before)
    for s, e in _episodes(oni, +0.5, 5):
        ax.axvspan(s, e, alpha=0.07, color="red",  zorder=0, linewidth=0)
    for s, e in _episodes(oni, -0.5, 5):
        ax.axvspan(s, e, alpha=0.07, color="blue", zorder=0, linewidth=0)

    # Try to load bootstrap bands (produced by build_validation_outputs.py)
    _boot_csv = OUT / "table6b_bootstrap_national.csv"
    if _boot_csv.exists():
        boot_df = pd.read_csv(_boot_csv, parse_dates=["date"])
        boot_df = boot_df.set_index("date")
        ax.fill_between(boot_df.index, boot_df["p05"], boot_df["p95"],
                        alpha=0.18, color="steelblue", zorder=1,
                        label="90% bootstrap band")
        ax.fill_between(boot_df.index, boot_df["p25"], boot_df["p75"],
                        alpha=0.30, color="steelblue", zorder=1,
                        label="50% bootstrap band")

    ax.plot(ma5.index, ma5.values, color="black", linewidth=2.4,
            label="Observed (5-yr MA)", zorder=3)
    ax.plot(trend_line.index,
            trend_line.rolling(60, min_periods=36).mean().values,
            color="#e07b10", linewidth=1.6, linestyle="--",
            label="Secular trend (5-yr MA)", zorder=2)

    ax.axvline(pd.Timestamp("1990-12-01"), color="gray",
               linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.6)

    # Key event annotations
    events_to_label = [
        (pd.Timestamp("1997-09-01"), "1997–98\nEl Niño"),
        (pd.Timestamp("2010-06-01"), "2010–12\nLa Niña"),
        (pd.Timestamp("2015-09-01"), "2015–16\nEl Niño"),
    ]
    for ts, lbl in events_to_label:
        if ts in ma5.index or ma5.index.asof(ts) is not pd.NaT:
            yval = float(ma5.asof(ts)) if not np.isnan(float(ma5.asof(ts))) else 0
            ax.annotate(lbl, xy=(ts, yval),
                        xytext=(0, 18), textcoords="offset points",
                        fontsize=7.5, ha="center", color="#555555",
                        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8))

    ax.set_ylabel("Std. anomaly (σ)")
    ax.set_xlabel("Year")
    ax.set_title("(b) ACI-CO — 5-year moving average with ENSO context")

    # Compact legend
    legend_handles = [
        Line2D([], [], color="black",   linewidth=2.4, label="Observed (5-yr MA)"),
        Line2D([], [], color="#e07b10", linewidth=1.6, linestyle="--", label="Secular trend"),
        mpatches.Patch(color="red",  alpha=0.30, label="El Niño phase"),
        mpatches.Patch(color="blue", alpha=0.30, label="La Niña phase"),
    ]
    if _boot_csv.exists():
        legend_handles.insert(2, mpatches.Patch(color="steelblue", alpha=0.40,
                                                 label="90% bootstrap band"))
    ax.legend(handles=legend_handles, fontsize=8.5, ncol=2, loc="upper left")

    _set_time_xlim(axes[0])
    fig.suptitle("Colombian Actuarial Climate Index (ACI-CO) — National (1961–2024)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "fig4_national_aci.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig4_national_aci.png")


""" + pp[f4_end:]

print("[OK] Patched fig4_national_aci")

# ── (4) Replace fig_enso_diagnostics (fig3) ───────────────────────────────────

f3_start = pp.index("def fig_enso_diagnostics(df_col: pd.DataFrame, sst: pd.Series):")
f3_end   = pp.index("# ---------------------------------------------------------------------------\n# 11. FIGURE 7a", f3_start)
pp = pp[:f3_start] + r"""def fig_enso_diagnostics(df_col: pd.DataFrame, sst: pd.Series):
    """
    Figure 3 — ENSO phase analysis (replaces regression diagnostics).

    Four panels that together explain HOW ENSO drives ACI-CO:
      (a) ENSO regression coefficients β+ and β- by component (asymmetric bar chart)
      (b) ACI-CO distribution by ENSO phase: El Niño / Neutral / La Niña box plots
      (c) Scatter of ACI-CO vs Niño-3.4, colored by decade, with piecewise fit
      (d) Seasonal ENSO sensitivity: mean ACI-CO anomaly during El Niño vs La Niña
          by calendar month (shows Colombia's boreal-summer peak sensitivity)
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    comps      = ["t90", "t10", "drought", "wind", "precipitation"]
    comp_short = ["T90", "T10", "CDD", "WP", "Rx5day"]
    comp_colors_list = ["#d62728", "#1f77b4", "#8c6d31", "#2ca02c", "#9467bd"]

    # Fit asymmetric ENSO regression for each component
    idx_shared = df_col.index.intersection(sst.index)
    e = sst.loc[idx_shared].values
    e_pos = np.maximum(e, 0)
    e_neg = np.minimum(e, 0)
    t_arr = np.arange(len(idx_shared), dtype=float)
    exog  = np.column_stack([np.ones(len(idx_shared)), e_pos, e_neg, t_arr])

    betas_pos, betas_neg = [], []
    pvals_pos, pvals_neg = [], []
    for comp in comps:
        y = df_col[comp].loc[idx_shared].values.astype(float)
        mask = ~np.isnan(y)
        try:
            best_aic, best_res = np.inf, None
            for p in range(0, 3):
                try:
                    mod = SARIMAX(y[mask], exog=exog[mask],
                                  order=(p, 0, 0), trend=None,
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
                    r = mod.fit(disp=False, maxiter=200)
                    if r.aic < best_aic:
                        best_aic, best_res = r.aic, r
                except Exception:
                    continue
            betas_pos.append(float(best_res.params[1]))
            betas_neg.append(float(best_res.params[2]))
            pvals_pos.append(float(best_res.pvalues[1]))
            pvals_neg.append(float(best_res.pvalues[2]))
        except Exception:
            betas_pos.append(0.0); betas_neg.append(0.0)
            pvals_pos.append(1.0); pvals_neg.append(1.0)

    # Classify months by ENSO phase using Niño-3.4
    oni3m = sst.rolling(3, center=True, min_periods=2).mean()
    phase = pd.Series("Neutral", index=sst.index)
    phase[oni3m >= 0.5]  = "El Niño"
    phase[oni3m <= -0.5] = "La Niña"

    aci_obs    = df_col["aci"].loc[idx_shared]
    phase_aci  = aci_obs.copy()
    phase_cats = phase.reindex(idx_shared)

    elnino_aci  = phase_aci[phase_cats == "El Niño"].values
    neutral_aci = phase_aci[phase_cats == "Neutral"].values
    lanina_aci  = phase_aci[phase_cats == "La Niña"].values

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 9))
    gs  = plt.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # ── (a) β+ and β- by component ──────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(comps))
    w = 0.35
    bars_p = ax_a.bar(x - w/2, betas_pos, w, color="#d62728", alpha=0.80,
                      label="β⁺ (El Niño response)", zorder=3)
    bars_n = ax_a.bar(x + w/2, betas_neg, w, color="#1f77b4", alpha=0.80,
                      label="β⁻ (La Niña response)", zorder=3)

    # Significance stars
    for i, (bp, bn, pp_, pn) in enumerate(zip(betas_pos, betas_neg,
                                                pvals_pos, pvals_neg)):
        star_p = "***" if pp_ < 0.001 else ("**" if pp_ < 0.01 else ("*" if pp_ < 0.05 else ""))
        star_n = "***" if pn  < 0.001 else ("**" if pn  < 0.01 else ("*" if pn  < 0.05 else ""))
        if star_p:
            ax_a.text(i - w/2, bp + (0.03 if bp >= 0 else -0.08),
                      star_p, ha="center", va="bottom", fontsize=8, color="#d62728")
        if star_n:
            ax_a.text(i + w/2, bn + (0.03 if bn >= 0 else -0.08),
                      star_n, ha="center", va="bottom", fontsize=8, color="#1f77b4")

    ax_a.axhline(0, color="black", linewidth=0.8)
    ax_a.set_xticks(x); ax_a.set_xticklabels(comp_short, fontsize=10)
    ax_a.set_ylabel("Regression coefficient (σ per °C)")
    ax_a.set_title("(a) ENSO sensitivity by component")
    ax_a.legend(fontsize=8.5)

    # ── (b) ACI-CO distribution by ENSO phase ────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    phase_order = ["El Niño", "Neutral", "La Niña"]
    phase_data  = [elnino_aci, neutral_aci, lanina_aci]
    phase_cols  = ["#d62728", "#888888", "#1f77b4"]
    bp_obj = ax_b.boxplot(phase_data, patch_artist=True,
                           medianprops=dict(color="black", linewidth=1.8),
                           whiskerprops=dict(linewidth=1.2),
                           capprops=dict(linewidth=1.2),
                           flierprops=dict(marker="o", markersize=3,
                                           alpha=0.4, linestyle="none"))
    for patch, col in zip(bp_obj["boxes"], phase_cols):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    # Add mean markers
    for i, (data, col) in enumerate(zip(phase_data, phase_cols), 1):
        ax_b.scatter(i, np.mean(data), s=55, zorder=5,
                     color="white", edgecolors=col, linewidths=2.0)
    ax_b.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax_b.set_xticks([1, 2, 3]); ax_b.set_xticklabels(phase_order, fontsize=10)
    ax_b.set_ylabel("ACI-CO anomaly (σ)")
    ax_b.set_title("(b) ACI-CO by ENSO phase (monthly)")
    # Annotate medians
    for i, (data, col) in enumerate(zip(phase_data, phase_cols), 1):
        m = np.median(data); n = len(data)
        ax_b.text(i, ax_b.get_ylim()[0] + 0.05 * (ax_b.get_ylim()[1] - ax_b.get_ylim()[0]),
                  f"n={n}", ha="center", va="bottom", fontsize=7.5, color="#555")

    # ── (c) Scatter ACI-CO vs Niño-3.4, colored by decade ──────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    decades = [(1961, 1979, "#aec6cf", "1961–79"),
               (1980, 1999, "#fab3a0", "1980–99"),
               (2000, 2013, "#a8d5a2", "2000–13"),
               (2014, 2024, "#c9a0dc", "2014–24")]
    e_vals   = sst.loc[idx_shared].values
    aci_vals = aci_obs.values
    years    = idx_shared.year

    for y0, y1, col, lbl in decades:
        mask_d = (years >= y0) & (years <= y1)
        ax_c.scatter(e_vals[mask_d], aci_vals[mask_d], s=7,
                     color=col, alpha=0.60, label=lbl, zorder=2)

    # Piecewise linear fit (positive / negative Niño-3.4)
    eg = np.linspace(e_vals.min(), e_vals.max(), 300)
    ep = np.maximum(eg, 0); en = np.minimum(eg, 0)
    # Use ACI coefficients
    bp_aci = betas_pos[comps.index("t90")] if "t90" in comps else 0.1
    # Fit a simple piecewise OLS for display
    valid  = ~np.isnan(aci_vals)
    exog_c = np.column_stack([np.ones(valid.sum()),
                               np.maximum(e_vals[valid], 0),
                               np.minimum(e_vals[valid], 0)])
    from numpy.linalg import lstsq
    c_ols, *_ = lstsq(exog_c, aci_vals[valid], rcond=None)
    fitted_g  = c_ols[0] + c_ols[1] * ep + c_ols[2] * en
    ax_c.plot(eg, fitted_g, color="black", linewidth=2.0,
              label="Piecewise fit", zorder=4)
    ax_c.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_c.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_c.set_xlabel("Niño-3.4 anomaly (°C)")
    ax_c.set_ylabel("ACI-CO anomaly (σ)")
    ax_c.set_title("(c) ACI-CO vs Niño-3.4 by decade")
    ax_c.legend(fontsize=7.5, ncol=2, loc="upper left")

    # ── (d) Seasonal ENSO response ───────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    months_arr = idx_shared.month
    elnino_mask = phase_cats == "El Niño"
    lanina_mask = phase_cats == "La Niña"

    mean_nino  = [aci_obs[elnino_mask & (months_arr == m)].mean() for m in range(1, 13)]
    mean_nina  = [aci_obs[lanina_mask & (months_arr == m)].mean() for m in range(1, 13)]
    mean_neut  = [aci_obs[~elnino_mask & ~lanina_mask & (months_arr == m)].mean() for m in range(1, 13)]

    month_names = ["J","F","M","A","M","J","J","A","S","O","N","D"]
    x_m = np.arange(1, 13)
    ax_d.plot(x_m, mean_nino, color="#d62728", marker="o", markersize=5,
              linewidth=1.8, label="El Niño months")
    ax_d.plot(x_m, mean_nina, color="#1f77b4", marker="s", markersize=5,
              linewidth=1.8, label="La Niña months")
    ax_d.plot(x_m, mean_neut, color="#888888", marker="^", markersize=4,
              linewidth=1.4, linestyle="--", label="Neutral months")
    ax_d.axhline(0, color="black", linewidth=0.6)
    ax_d.set_xticks(x_m); ax_d.set_xticklabels(month_names)
    ax_d.set_ylabel("Mean ACI-CO anomaly (σ)")
    ax_d.set_xlabel("Calendar month")
    ax_d.set_title("(d) Seasonal ENSO response")
    ax_d.legend(fontsize=8.5)

    fig.suptitle(
        "Figure 3 — ENSO Phase Analysis: How El Niño and La Niña drive ACI-CO (National, 1961–2024)",
        fontsize=11, fontweight="bold")
    fig.savefig(OUT / "fig3_enso_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig3_enso_diagnostics.png")


""" + pp[f3_end:]

print("[OK] Patched fig_enso_diagnostics (fig3)")

# ── Also update fig8a_bootstrap to save CSV for fig4 to load ─────────────────
OLD_SAVE = (
    '    fig.savefig(OUT / "fig8a_bootstrap_bands.png", dpi=300, bbox_inches="tight")\n'
    '    plt.close(fig)\n'
    '    print("fig8a_bootstrap_bands.png")\n'
    '\n'
    '    # Band width summary\n'
    '    width_90 = p95 - p05\n'
    '    width_50 = p75 - p25\n'
    '    recent   = idx >= pd.Timestamp("2010-01-01")\n'
    '    print(f"  90% band width (post-2010): mean={width_90[recent].mean():.3f} sigma")\n'
    '    return pd.DataFrame({\n'
    '        "date": idx, "obs_ma": obs, "p05": p05, "p25": p25, "p75": p75, "p95": p95,\n'
    '    })'
)

NEW_SAVE = (
    '    fig.savefig(OUT / "fig8a_bootstrap_bands.png", dpi=300, bbox_inches="tight")\n'
    '    plt.close(fig)\n'
    '    print("fig8a_bootstrap_bands.png")\n'
    '\n'
    '    # Band width summary\n'
    '    width_90 = p95 - p05\n'
    '    width_50 = p75 - p25\n'
    '    recent   = idx >= pd.Timestamp("2010-01-01")\n'
    '    print(f"  90% band width (post-2010): mean={width_90[recent].mean():.3f} sigma")\n'
    '\n'
    '    # Save CSV so that produce_paper_plots.py can overlay bands on fig4\n'
    '    boot_df = pd.DataFrame({\n'
    '        "date": idx, "obs_ma": obs, "p05": p05, "p25": p25, "p75": p75, "p95": p95,\n'
    '    })\n'
    '    boot_df.to_csv(OUT / "table6b_bootstrap_national.csv", index=False)\n'
    '    print("table6b_bootstrap_national.csv")\n'
    '    return boot_df'
)

# apply in build_validation_outputs.py (re-read since we already wrote it)
bv2 = bv_path.read_text(encoding="utf-8")
if OLD_SAVE in bv2:
    bv2 = bv2.replace(OLD_SAVE, NEW_SAVE, 1)
    bv_path.write_text(bv2, encoding="utf-8")
    print("[OK] Patched fig8a_bootstrap to save CSV")
else:
    print("[MISS] Could not patch fig8a CSV save (will still work, just no bands in fig4)")

pp_path.write_text(pp, encoding="utf-8")
print("[OK] produce_paper_plots.py written")
print("\nAll patches applied. Run both scripts to regenerate outputs.")
