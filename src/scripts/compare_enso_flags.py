"""
compare_enso_flags.py
─────────────────────
1. Loads Colombian Pacific SST index flags (D_warm / D_cold)
2. Contrasts against two references:
     REF-1  NOAA CPC ONI  (oni.ascii.txt — same source IDEAM cites)
     REF-2  IDEAM institutional event list (2002/2014 + bulletins to 2024)
3. Bias-corrects the index (removes long-run mean)
4. Grid-searches warm_thr × cold_thr × min_consec to maximise macro-F1 vs ONI
5. Re-runs comparison with tuned flags and saves everything

Outputs
-------
  enso_flag_comparison.csv        all flags (original + tuned) vs both refs
  threshold_search_results.csv    full grid scores
  confusion_matrix.png            before/after precision-recall-F1 (2×2)
  timeline_comparison.png         original flags vs ONI + IDEAM
  tuned_timeline.png              tuned flags vs ONI + IDEAM
  tuning_heatmap.png              macro-F1 surface over the search grid
"""

import os, urllib.request, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import product

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'data', 'processed'
)
FEATURES_CSV = os.path.join(PROCESSED_DIR, 'sst_features_colombia_pacific.csv')
SST_CSV      = os.path.join(PROCESSED_DIR, 'sst_index_colombia_pacific.csv')
ONI_FILE     = os.path.join(PROCESSED_DIR, 'oni.ascii.txt')
OUTPUT_DIR   = PROCESSED_DIR

ONI_URL = 'https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt'

# ── Grid search bounds ────────────────────────────────────────────────────────
WARM_THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)
COLD_THRESHOLDS = np.round(np.arange(-0.10, -0.91, -0.05), 2)
MIN_CONSEC_VALS = [3, 4, 5, 6, 7]

# ── IDEAM institutional event list ────────────────────────────────────────────
# Source: "Modelo Institucional El Nino - La Nina" (IDEAM 2002/2014) + bulletins
# (start_year, start_month, end_year, end_month, intensity)
# intensity: 1=debil  2=moderado  3=fuerte  4=muy fuerte
IDEAM_EL_NINO = [
    (1963,  6, 1964,  2, 1),
    (1965,  5, 1966,  4, 2),
    (1968, 11, 1970,  1, 1),
    (1972,  5, 1973,  3, 3),
    (1976,  9, 1977,  2, 1),
    (1982,  5, 1983,  7, 4),
    (1986,  9, 1988,  2, 2),
    (1991,  6, 1992,  7, 3),
    (1994,  9, 1995,  3, 2),
    (1997,  5, 1998,  5, 4),
    (2002,  6, 2003,  2, 2),
    (2004,  7, 2005,  2, 1),
    (2006,  9, 2007,  1, 1),
    (2009,  7, 2010,  3, 2),
    (2014, 10, 2016,  5, 4),
    (2018, 10, 2019,  6, 1),
    (2023,  6, 2024,  5, 3),
]
IDEAM_LA_NINA = [
    (1964,  5, 1965,  1, 2),
    (1967,  9, 1968,  3, 1),
    (1970,  7, 1972,  1, 3),
    (1973,  6, 1976,  3, 3),
    (1983,  9, 1985, 10, 2),
    (1988,  5, 1989, 12, 3),
    (1995,  9, 1996,  3, 1),
    (1998,  7, 2001,  2, 3),
    (2005, 11, 2006,  3, 1),
    (2007,  8, 2009,  6, 2),
    (2010,  7, 2012,  3, 3),
    (2016,  8, 2016, 12, 1),
    (2017, 10, 2018,  4, 1),
    (2020,  8, 2023,  3, 2),
]


# ══════════════════════════════════════════════════════════════════════════════
# Core helpers
# ══════════════════════════════════════════════════════════════════════════════

def consec_flag(series, min_run):
    """Binary array: 1 only for runs of >= min_run consecutive True values."""
    arr  = series.values if hasattr(series, 'values') else np.asarray(series)
    flag = np.zeros(len(arr), dtype=int)
    i = 0
    while i < len(arr):
        if arr[i]:
            j = i
            while j < len(arr) and arr[j]:
                j += 1
            if (j - i) >= min_run:
                flag[i:j] = 1
            i = j
        else:
            i += 1
    return flag


def confusion_metrics(pred, ref, label=''):
    tp = int(((pred == 1) & (ref == 1)).sum())
    fp = int(((pred == 1) & (ref == 0)).sum())
    fn = int(((pred == 0) & (ref == 1)).sum())
    tn = int(((pred == 0) & (ref == 0)).sum())
    prec   = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1     = (2 * prec * recall / (prec + recall)
              if not (np.isnan(prec) or np.isnan(recall) or (prec + recall) == 0)
              else np.nan)
    return dict(label=label, TP=tp, FP=fp, FN=fn, TN=tn,
                Precision=prec, Recall=recall, F1=f1)


def print_metrics(m):
    print(f"\n  {m['label']}")
    print(f"    TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}")
    p = m['Precision']
    r = m['Recall']
    f = m['F1']
    print(f"    Precision={p:.3f}  Recall={r:.3f}  F1={f:.3f}")


def _shade(ax, times, flag, color, ymin, ymax, alpha=0.55, hatch=None):
    """Shade time axis where flag == 1."""
    in_ev = False
    t0    = None
    times = list(times)
    for t, v in zip(times, flag):
        if v == 1 and not in_ev:
            in_ev, t0 = True, t
        elif v != 1 and in_ev:
            ax.axvspan(t0, t, ymin=ymin, ymax=ymax,
                       color=color, alpha=alpha, hatch=hatch, lw=0)
            in_ev = False
    if in_ev:
        ax.axvspan(t0, times[-1], ymin=ymin, ymax=ymax,
                   color=color, alpha=alpha, hatch=hatch, lw=0)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════

def load_our_flags():
    feat = pd.read_csv(FEATURES_CSV, parse_dates=['time'])
    sst  = pd.read_csv(SST_CSV,      parse_dates=['time'])
    feat['time'] = pd.to_datetime(feat['time']).dt.normalize()
    sst['time']  = pd.to_datetime(sst['time']).dt.normalize()

    print(f"  feat columns : {list(feat.columns)}")
    print(f"  sst  columns : {list(sst.columns)}")

    sst_cols = [c for c in ['time','sst_index','sst_anomaly',
                             'sst_anomaly_pacific','sst_anomaly_caribbean']
                if c in sst.columns]
    df = feat.merge(sst[sst_cols], on='time', how='left', suffixes=('_feat', ''))
    df = df.drop(columns=[c for c in df.columns if c.endswith('_feat')])
    df = df.sort_values('time').reset_index(drop=True)
    print(f"  Loaded: {len(df)} months  "
          f"({df['time'].min().date()} -> {df['time'].max().date()})")
    return df


def load_oni_reference():
    if not os.path.exists(ONI_FILE):
        print("  Downloading ONI table from NOAA CPC ...")
        try:
            urllib.request.urlretrieve(ONI_URL, ONI_FILE)
            print(f"  Saved to {ONI_FILE}")
        except Exception as e:
            print(f"  Download failed: {e}")
            return None

    try:
        df = pd.read_csv(ONI_FILE, sep=r'\s+', engine='python')
        df.columns = [c.strip() for c in df.columns]

        SEAS_MAP = {'DJF':1,'JFM':2,'FMA':3,'MAM':4,'AMJ':5,'MJJ':6,
                    'JJA':7,'JAS':8,'ASO':9,'SON':10,'OND':11,'NDJ':12}
        df['month'] = df['SEAS'].map(SEAS_MAP)
        df = df.dropna(subset=['month'])
        df['YR']    = pd.to_numeric(df['YR'],   errors='coerce').astype('Int64')
        df['ANOM']  = pd.to_numeric(df['ANOM'], errors='coerce')
        df['month'] = df['month'].astype(int)
        df.loc[df['SEAS'] == 'NDJ', 'YR'] += 1   # NDJ centre month = Jan of next year
        df['time'] = pd.to_datetime(
            df['YR'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
        )
        df = df.dropna(subset=['YR','ANOM']).sort_values('time').reset_index(drop=True)
        df['oni_warm'] = consec_flag(df['ANOM'] >= 0.5,  5)
        df['oni_cold'] = consec_flag(df['ANOM'] <= -0.5, 5)
        df['oni_anom'] = df['ANOM']

        oni = df[['time','oni_anom','oni_warm','oni_cold']]
        print(f"  ONI: {len(oni)} months  "
              f"({oni['time'].min().date()} -> {oni['time'].max().date()})")
        print(f"  ONI El Nino: {oni['oni_warm'].sum()} months  "
              f"La Nina: {oni['oni_cold'].sum()} months")
        return oni

    except Exception as e:
        print(f"  Parse failed: {e}")
        return None


def build_ideam_flags(date_range):
    df = pd.DataFrame({'time': date_range,
                       'ideam_warm': 0, 'ideam_cold': 0,
                       'ideam_intensity': np.nan}).set_index('time')
    for sy, sm, ey, em, intens in IDEAM_EL_NINO:
        mask = (df.index >= pd.Timestamp(sy, sm, 1)) & \
               (df.index <= pd.Timestamp(ey, em, 1))
        df.loc[mask, 'ideam_warm']      = 1
        df.loc[mask, 'ideam_intensity'] = intens
    for sy, sm, ey, em, intens in IDEAM_LA_NINA:
        mask = (df.index >= pd.Timestamp(sy, sm, 1)) & \
               (df.index <= pd.Timestamp(ey, em, 1))
        df.loc[mask, 'ideam_cold']      = 1
        df.loc[mask, 'ideam_intensity'] = intens
    df = df.reset_index()
    print(f"  IDEAM El Nino: {df['ideam_warm'].sum()} months  "
          f"La Nina: {df['ideam_cold'].sum()} months")
    return df


def merge_all(our_df, oni_df, ideam_df):
    df = our_df.merge(oni_df,   on='time', how='left')
    df = df.merge(ideam_df,     on='time', how='left')
    for col in ['oni_warm','oni_cold','ideam_warm','ideam_cold']:
        df[col] = df[col].fillna(0).astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Comparison helper
# ══════════════════════════════════════════════════════════════════════════════

def run_comparison(df, warm_col='D_warm', cold_col='D_cold', label_suffix=''):
    overlap = df.dropna(subset=['sst_index','oni_anom'])
    metrics_warm, metrics_cold = [], []
    for ref_w, ref_c, name, sub in [
        ('oni_warm',   'oni_cold',   f'vs. CPC ONI{label_suffix}', overlap),
        ('ideam_warm', 'ideam_cold', f'vs. IDEAM{label_suffix}',   df),
    ]:
        mw = confusion_metrics(sub[warm_col], sub[ref_w], f'El Nino  {name}')
        mc = confusion_metrics(sub[cold_col], sub[ref_c], f'La Nina  {name}')
        print_metrics(mw)
        print_metrics(mc)
        metrics_warm.append(mw)
        metrics_cold.append(mc)
    return metrics_warm, metrics_cold


def discrepancy_report(df, warm_col='D_warm', cold_col='D_cold'):
    tag = f' [{warm_col}]' if warm_col != 'D_warm' else ''
    print(f"\n{'='*72}\nDISCREPANCY REPORT{tag}\n{'='*72}")
    for ref_w, ref_c, ref_name in [
        ('oni_warm',   'oni_cold',   'CPC ONI'),
        ('ideam_warm', 'ideam_cold', 'IDEAM'),
    ]:
        print(f"\n-- vs. {ref_name} " + '-'*50)
        for desc, mask in [
            (f"{warm_col}=1 but {ref_name}=0  (FP El Nino)",
             (df[warm_col]==1) & (df[ref_w]==0)),
            (f"{ref_name} warm but {warm_col}=0  (FN El Nino)",
             (df[warm_col]==0) & (df[ref_w]==1)),
            (f"{cold_col}=1 but {ref_name}=0  (FP La Nina)",
             (df[cold_col]==1) & (df[ref_c]==0)),
            (f"{ref_name} cold but {cold_col}=0  (FN La Nina)",
             (df[cold_col]==0) & (df[ref_c]==1)),
        ]:
            sub = df.loc[mask, ['time','sst_index']]
            print(f"\n  {desc}: {len(sub)} months")
            if not sub.empty:
                print(sub.to_string(index=False))
    print(f"\n{'='*72}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Bias correction
# ══════════════════════════════════════════════════════════════════════════════

def bias_correct(df):
    bias = df['sst_index'].mean()
    df   = df.copy()
    df['sst_index_bc'] = df['sst_index'] - bias
    print(f"  Bias removed: {bias:+.4f} C")
    print(f"  Original  : mean={df['sst_index'].mean():.4f}  std={df['sst_index'].std():.4f}")
    print(f"  Corrected : mean={df['sst_index_bc'].mean():.4f}  std={df['sst_index_bc'].std():.4f}")
    return df, bias


# ══════════════════════════════════════════════════════════════════════════════
# 4. Grid search
# ══════════════════════════════════════════════════════════════════════════════

def _f1_score(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return f, p, r


def evaluate_params(idx, warm_thr, cold_thr, min_consec, oni_warm, oni_cold):
    dw = consec_flag(idx >= warm_thr, min_consec)
    dc = consec_flag(idx <= cold_thr, min_consec)
    tp_w = int(((dw==1)&(oni_warm==1)).sum())
    fp_w = int(((dw==1)&(oni_warm==0)).sum())
    fn_w = int(((dw==0)&(oni_warm==1)).sum())
    f1_w, p_w, r_w = _f1_score(tp_w, fp_w, fn_w)
    tp_c = int(((dc==1)&(oni_cold==1)).sum())
    fp_c = int(((dc==1)&(oni_cold==0)).sum())
    fn_c = int(((dc==0)&(oni_cold==1)).sum())
    f1_c, p_c, r_c = _f1_score(tp_c, fp_c, fn_c)
    return dict(
        warm_thr=round(warm_thr,2), cold_thr=round(cold_thr,2),
        min_consec=min_consec, macro_f1=(f1_w+f1_c)/2,
        f1_warm=f1_w, prec_warm=p_w, rec_warm=r_w,
        tp_warm=tp_w, fp_warm=fp_w, fn_warm=fn_w,
        f1_cold=f1_c, prec_cold=p_c, rec_cold=r_c,
        tp_cold=tp_c, fp_cold=fp_c, fn_cold=fn_c,
    )


def grid_search(df):
    overlap = df.dropna(subset=['sst_index_bc','oni_anom']).copy()
    idx = overlap['sst_index_bc']
    ow  = overlap['oni_warm'].values
    oc  = overlap['oni_cold'].values
    combos = list(product(WARM_THRESHOLDS, COLD_THRESHOLDS, MIN_CONSEC_VALS))
    print(f"  Testing {len(combos)} combinations ...", end='', flush=True)
    results = [evaluate_params(idx, wt, ct, mc, ow, oc) for wt, ct, mc in combos]
    print(" done.")
    return pd.DataFrame(results).sort_values('macro_f1', ascending=False)


def print_top(results_df, n=15):
    cols = ['warm_thr','cold_thr','min_consec','macro_f1',
            'f1_warm','prec_warm','rec_warm',
            'f1_cold','prec_cold','rec_cold']
    print(f"\n{'─'*80}\nTOP {n} COMBINATIONS\n{'─'*80}")
    print(results_df[cols].head(n).to_string(index=False,
                                              float_format='{:.3f}'.format))
    best = results_df.iloc[0]
    print(f"\n{'─'*80}\nBEST PARAMETERS\n{'─'*80}")
    print(f"  warm_thr   = {best['warm_thr']:+.2f} C   (original +0.50)")
    print(f"  cold_thr   = {best['cold_thr']:+.2f} C   (original -0.50)")
    print(f"  min_consec = {int(best['min_consec'])} seasons  (original 5)")
    print(f"  Macro-F1   = {best['macro_f1']:.4f}")
    print(f"  El Nino -> F1={best['f1_warm']:.3f}  P={best['prec_warm']:.3f}  "
          f"R={best['rec_warm']:.3f}  "
          f"(TP={int(best['tp_warm'])} FP={int(best['fp_warm'])} FN={int(best['fn_warm'])})")
    print(f"  La Nina -> F1={best['f1_cold']:.3f}  P={best['prec_cold']:.3f}  "
          f"R={best['rec_cold']:.3f}  "
          f"(TP={int(best['tp_cold'])} FP={int(best['fp_cold'])} FN={int(best['fn_cold'])})")
    return best


def apply_tuned_flags(df, best):
    col = 'sst_index_bc'
    df  = df.copy()
    df['D_warm_tuned'] = consec_flag(df[col] >= best['warm_thr'], int(best['min_consec']))
    df['D_cold_tuned'] = consec_flag(df[col] <= best['cold_thr'], int(best['min_consec']))
    mu, sig = df[col].mean(), df[col].std(ddof=1)
    df['sst_z_bc'] = (df[col] - mu) / sig
    for lag in [0, 1, 2, 3]:
        df[f'sst_z_bc_lag{lag}'] = df['sst_z_bc'].shift(lag)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. Plots
# ══════════════════════════════════════════════════════════════════════════════

def _bar_group(ax, metrics_list, title):
    labels = [m['label'] for m in metrics_list]
    x = np.arange(len(labels))
    w = 0.25
    for offset, key, color, label in [
        (-w, 'Precision', '#4dac26', 'Precision'),
        (0,  'Recall',    '#d01c8b', 'Recall'),
        (+w, 'F1',        '#f1b6da', 'F1'),
    ]:
        vals = [m[key] for m in metrics_list]
        bars = ax.bar(x + offset, vals, w, color=color, alpha=0.85, label=label)
        for b in bars:
            h = b.get_height()
            if not np.isnan(h):
                ax.text(b.get_x()+b.get_width()/2, h+0.01,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.axhline(0.5, color='grey', lw=0.7, linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)


def plot_confusion_figure(mw_b, mc_b, mw_a, mc_a, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    _bar_group(axes[0,0], mw_b, 'El Nino — BEFORE tuning')
    _bar_group(axes[0,1], mc_b, 'La Nina — BEFORE tuning')
    _bar_group(axes[1,0], mw_a, 'El Nino — AFTER tuning')
    _bar_group(axes[1,1], mc_a, 'La Nina — AFTER tuning')
    fig.suptitle('ENSO Flag Quality vs. NOAA ONI + IDEAM  |  Before & After Tuning',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Confusion figure -> {out_path}")
    plt.close()


def plot_timeline(df, title, warm_col, cold_col, warm_thr, cold_thr,
                  out_path, index_col='sst_index'):
    fig, axes = plt.subplots(3, 1, figsize=(18, 11), sharex=True)
    fig.subplots_adjust(hspace=0.40)
    times = df['time']

    # Panel 1 – index curves
    ax = axes[0]
    ax.plot(times, df[index_col], color='#2c7bb6', lw=1.2, label=index_col)
    if 'oni_anom' in df.columns:
        ax.plot(times, df['oni_anom'], color='grey', lw=0.9,
                linestyle='--', alpha=0.75, label='CPC ONI anomaly')
    ax.axhline(warm_thr,  color='#d7191c', lw=0.8, linestyle=':', alpha=0.8)
    ax.axhline(cold_thr,  color='#4575b4', lw=0.8, linestyle=':', alpha=0.8)
    ax.axhline(0, color='black', lw=0.4)
    ax.set_ylabel('C')
    ax.set_title(f'SST index vs. CPC ONI  |  {title}', fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.2)

    # Panel 2 – flags vs ONI
    ax = axes[1]
    _shade(ax, times, df[warm_col].values, '#fc8d59', 0.55, 1.0)
    _shade(ax, times, df[cold_col].values, '#91bfdb', 0.05, 0.50)
    _shade(ax, times, df['oni_warm'].values, '#d7191c', 0.55, 1.0, alpha=0.3, hatch='//')
    _shade(ax, times, df['oni_cold'].values, '#2c7bb6', 0.05, 0.50, alpha=0.3, hatch='//')
    patches = [mpatches.Patch(color='#fc8d59', label='Our El Nino'),
               mpatches.Patch(color='#91bfdb', label='Our La Nina'),
               mpatches.Patch(color='#d7191c', alpha=0.4, hatch='//', label='ONI warm'),
               mpatches.Patch(color='#2c7bb6', alpha=0.4, hatch='//', label='ONI cold')]
    ax.set_yticks([])
    ax.legend(handles=patches, fontsize=8, loc='upper left', ncol=4)
    ax.set_title(f'Flags vs. CPC ONI (hatched)  '
                 f'warm>={warm_thr:+.2f}C  cold<={cold_thr:+.2f}C', fontweight='bold')

    # Panel 3 – agreement map
    ax = axes[2]
    agree_oni   = ((df[warm_col]==df['oni_warm'])   & (df[cold_col]==df['oni_cold'])).values
    agree_ideam = ((df[warm_col]==df['ideam_warm']) & (df[cold_col]==df['ideam_cold'])).values
    for i, t in enumerate(times):
        c = ('green'  if agree_oni[i] and agree_ideam[i]
             else 'orange' if agree_oni[i] or agree_ideam[i]
             else 'red')
        ax.axvspan(t, t + pd.DateOffset(months=1), color=c, alpha=0.4, lw=0)
    patches = [mpatches.Patch(color='green',  alpha=0.5, label='Agree both'),
               mpatches.Patch(color='orange', alpha=0.5, label='Agree one'),
               mpatches.Patch(color='red',    alpha=0.5, label='Disagree both')]
    ax.set_yticks([])
    ax.legend(handles=patches, fontsize=8, loc='upper left', ncol=3)
    ax.set_title('Agreement: our flags vs. ONI + IDEAM', fontweight='bold')

    fig.suptitle(f'ENSO Flag Comparison — {title}', fontsize=13, fontweight='bold')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Timeline -> {out_path}")
    plt.close()


def plot_heatmap(results_df, best, out_path):
    wt_vals = sorted(results_df['warm_thr'].unique())
    ct_vals = sorted(results_df['cold_thr'].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(MIN_CONSEC_VALS),
                             figsize=(5*len(MIN_CONSEC_VALS), 5), sharey=True)
    for ax, mc in zip(axes, MIN_CONSEC_VALS):
        sub  = results_df[results_df['min_consec'] == mc]
        grid = np.full((len(ct_vals), len(wt_vals)), np.nan)
        for _, row in sub.iterrows():
            ri = ct_vals.index(row['cold_thr'])
            ci = wt_vals.index(row['warm_thr'])
            grid[ri, ci] = row['macro_f1']
        im = ax.imshow(grid, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(wt_vals)))
        ax.set_xticklabels([f'{v:.2f}' for v in wt_vals], rotation=90, fontsize=7)
        if ax is axes[0]:
            ax.set_yticks(range(len(ct_vals)))
            ax.set_yticklabels([f'{v:.2f}' for v in ct_vals], fontsize=7)
            ax.set_ylabel('cold_thr (C)', fontsize=9)
        ax.set_xlabel('warm_thr (C)', fontsize=9)
        ax.set_title(f'min_consec = {mc}', fontsize=10, fontweight='bold')
        if int(best['min_consec']) == mc:
            ri_b = ct_vals.index(best['cold_thr'])
            ci_b = wt_vals.index(best['warm_thr'])
            ax.add_patch(plt.Rectangle((ci_b-0.5, ri_b-0.5), 1, 1,
                                       fill=False, edgecolor='black', lw=2.5))
    plt.colorbar(im, ax=axes[-1], label='Macro-F1', shrink=0.8)
    fig.suptitle('Grid Search Macro-F1 vs. NOAA ONI  |  black box = best',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap -> {out_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('='*72)
    print('ENSO FLAG COMPARISON + TUNING — Colombian Pacific SST Index')
    print('='*72)

    print('\n[1] Loading SST features ...')
    our_df = load_our_flags()

    print('\n[2] Loading NOAA CPC ONI ...')
    oni_df = load_oni_reference()
    if oni_df is None:
        oni_df = pd.DataFrame({'time': our_df['time'],
                               'oni_anom': np.nan, 'oni_warm': 0, 'oni_cold': 0})

    print('\n[3] Building IDEAM flags ...')
    ideam_df = build_ideam_flags(our_df['time'])

    print('\n[4] Merging ...')
    df = merge_all(our_df, oni_df, ideam_df)
    if 'sst_index_x' in df.columns:
        df = df.rename(columns={'sst_index_x': 'sst_index'})
        df = df.drop(columns=['sst_index_y'], errors='ignore')
    print(f"  Shape: {df.shape}")

    # ── BEFORE tuning ─────────────────────────────────────────────
    print('\n[5] Metrics — ORIGINAL flags ...')
    mw_before, mc_before = run_comparison(df)
    discrepancy_report(df)

    # ── Bias correction ───────────────────────────────────────────
    print('\n[6] Bias correction ...')
    df, bias = bias_correct(df)

    # ── Grid search ───────────────────────────────────────────────
    print('\n[7] Grid search ...')
    results_df = grid_search(df)
    best = print_top(results_df)

    # ── Apply tuned flags ─────────────────────────────────────────
    print('\n[8] Applying tuned flags ...')
    df = apply_tuned_flags(df, best)

    # ── AFTER tuning ──────────────────────────────────────────────
    print('\n[9] Metrics — TUNED flags ...')
    mw_after, mc_after = run_comparison(df,
                                        warm_col='D_warm_tuned',
                                        cold_col='D_cold_tuned',
                                        label_suffix=' (tuned)')
    discrepancy_report(df, warm_col='D_warm_tuned', cold_col='D_cold_tuned')

    # ── Save ──────────────────────────────────────────────────────
    print('\n[10] Saving CSVs ...')
    df.to_csv(os.path.join(OUTPUT_DIR, 'enso_flag_comparison.csv'),
              index=False, date_format='%Y-%m-%d')
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'threshold_search_results.csv'),
                      index=False, float_format='%.4f')
    print('  enso_flag_comparison.csv')
    print('  threshold_search_results.csv')

    # ── Plots ─────────────────────────────────────────────────────
    print('\n[11] Generating plots ...')
    plot_confusion_figure(
        mw_before, mc_before, mw_after, mc_after,
        os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    )
    plot_timeline(
        df, 'ORIGINAL (warm>=+0.50C  cold<=-0.50C  consec=5)',
        'D_warm', 'D_cold', 0.50, -0.50,
        os.path.join(OUTPUT_DIR, 'timeline_comparison.png'),
        index_col='sst_index'
    )
    plot_timeline(
        df, f'TUNED (bias-corrected, warm>={best["warm_thr"]:+.2f}C '
            f'cold<={best["cold_thr"]:+.2f}C  consec={int(best["min_consec"])})',
        'D_warm_tuned', 'D_cold_tuned',
        best['warm_thr'], best['cold_thr'],
        os.path.join(OUTPUT_DIR, 'tuned_timeline.png'),
        index_col='sst_index_bc'
    )
    plot_heatmap(results_df, best,
                 os.path.join(OUTPUT_DIR, 'tuning_heatmap.png'))

    # ── Summary ───────────────────────────────────────────────────
    print(f'\n{"="*72}')
    print('SUMMARY — paste these into anomalias_sst_daily.py')
    print(f'{"="*72}')
    print(f'  BIAS_CORRECTION    = {bias:.4f}   # subtract from sst_index')
    print(f'  ONI_THRESHOLD_WARM = {best["warm_thr"]}')
    print(f'  ONI_THRESHOLD_COLD = {best["cold_thr"]}')
    print(f'  ONI_MIN_CONSEC     = {int(best["min_consec"])}')
    print(f'  Macro-F1 vs ONI    : {best["macro_f1"]:.4f}')
    print(f'{"="*72}')
    print('\nAll done.')
    return df, results_df, best


if __name__ == '__main__':
    df, results_df, best = main()