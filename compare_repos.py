"""
Comparison of ACI component outputs between:
  - ACI-Python-main  (original library applied to Colombia ERA5 data)
  - aca_indice_climatico_opt-main  (Colombia-adapted, no sea level)

OPT source:  data/processed/anomalias_colombia/anomalies_*.csv
             (same files graficas.py reads)
ACI source:  data/processed/aci_python_output_1961_1990.csv

Raw data differences (for reference):
  ACI-Python temperature:
    - Splits hours into day (6-21) and night (0-5, 22-23)
    - Rolling 80/40-window percentile on reference period
    - Fraction of days above (T90) / below (T10) day/night threshold
  OPT temperature:
    - Daily max and daily min (all hours)
    - Pre-computed monthly percentile grids (percentiles_max/min at q=0.1, 0.9)
    - (count_above_90_max + count_above_90_min) / 2, then standardised
  Both:
    - Same ERA5 GRIB source: hourly t2m, 0.25 deg grid, Colombia bbox
    - Wind: ACI-Python uses u10/v10 separately; OPT uses sqrt(u10^2+v10^2)
    - Precipitation: ACI-Python uses hourly accumulated tp; OPT uses daily sums
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORK_DIR = Path(__file__).resolve().parent
PROCESSED = WORK_DIR / "data" / "processed"
OPT_DIR   = PROCESSED / "anomalias_colombia"
ACI_CSV   = PROCESSED / "aci_colombia_1961_2024.csv"
OUT_DIR   = WORK_DIR / "articles" / "graficas" / "comparison_repos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_LINE = pd.Timestamp("1990-01-01")
WINDOW = 60  # 5-year moving average

# graficas.py palette (OPT originals)
C_T90    = "#6B8E8E"
C_T10    = "#A9C8C8"
C_COMP   = "black"
C_BAR    = "#A9C8C8"
C_WIND   = "#C08056"
C_RAIN   = "#D9A441"
C_DRGT   = "#EAD8C0"

# ACI-Python overlay colour (dashed)
C_ACI    = "#C0392B"


# ---------------------------------------------------------------------------
# Load OPT component CSVs  (exactly as graficas.py does)
# ---------------------------------------------------------------------------
def load_opt_temp():
    df = pd.read_csv(OPT_DIR / "anomalies_temperature_combined.csv")
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df.sort_values("date").reset_index(drop=True)

def load_opt_precip():
    df = pd.read_csv(OPT_DIR / "anomalies_precipitation_combined.csv")
    df.rename(columns={df.columns[0]: "year", df.columns[1]: "month"}, inplace=True)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df.sort_values("date").reset_index(drop=True)

def load_opt_drought():
    df = pd.read_csv(OPT_DIR / "anomalies_drought_combined.csv")
    df.rename(columns={df.columns[0]: "year", df.columns[1]: "month"}, inplace=True)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df.sort_values("date").reset_index(drop=True)

def load_opt_wind():
    df = pd.read_csv(OPT_DIR / "anomalies_wind_combined.csv")
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Load ACI-Python output and trim to 1961-1990
# ---------------------------------------------------------------------------
def load_aci():
    if not ACI_CSV.exists():
        raise FileNotFoundError(
            f"ACI output not found: {ACI_CSV}\n"
            "Run  python run_aci_colombia.py  first to compute it from raw GRIB data."
        )
    df = pd.read_csv(ACI_CSV, index_col=0, parse_dates=True)
    df.index = df.index.to_period("M").to_timestamp(how="S")
    df = df.sort_index()
    # ACI column already computed without sea level in run_aci_colombia.py
    df["ACI_no_sl"] = df["ACI"]
    return df


# ---------------------------------------------------------------------------
# Rolling helpers
# ---------------------------------------------------------------------------
def ma(series: pd.Series) -> pd.Series:
    return series.rolling(window=WINDOW, min_periods=1, center=False).mean()

def _fmt_ax(ax, title):
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Standardized Anomalies")
    ax.axvline(REFERENCE_LINE, color="gray", linestyle="dotted", linewidth=1)
    ax.grid(True, linestyle="dotted", linewidth=0.5, alpha=0.5)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend()


# ---------------------------------------------------------------------------
# Trim a dataframe to the 1961-1990 window
# ---------------------------------------------------------------------------
def trim_1990(df, date_col="date"):
    return df[df[date_col] <= "1990-12-31"].copy()


# ---------------------------------------------------------------------------
# Plot 1 — Temperature
# Mirrors graficas.py > plot_temp_anomalies, with ACI-Python overlay
# ---------------------------------------------------------------------------
def plot_temperature(df_opt: pd.DataFrame, df_aci: pd.DataFrame, out: Path):
    df_opt["T90_ma"] = ma(df_opt["t_90"])
    df_opt["T10_ma"] = ma(df_opt["t_10"])
    df_opt["comp"]   = df_opt["t_90"] - df_opt["t_10"]
    df_opt["comp_ma"] = ma(df_opt["comp"])

    t90_aci  = ma(df_aci["t90"])
    t10_aci  = ma(df_aci["t10"])
    comp_aci = ma(df_aci["t90"] - df_aci["t10"])

    plt.figure(figsize=(15, 6))
    # OPT (solid, graficas.py style)
    plt.plot(df_opt["date"], df_opt["T90_ma"],  color=C_T90,  label="OPT T90 5-yr avg")
    plt.plot(df_opt["date"], df_opt["T10_ma"],  color=C_T10,  label="OPT T10 5-yr avg")
    plt.plot(df_opt["date"], df_opt["comp_ma"], color=C_COMP, label="OPT Composite (T90-T10)")
    # ACI-Python (dashed)
    plt.plot(df_aci.index, t90_aci,  color=C_T90, linestyle="--", alpha=0.7, label="ACI-Python T90 5-yr avg")
    plt.plot(df_aci.index, t10_aci,  color=C_T10, linestyle="--", alpha=0.7, label="ACI-Python T10 5-yr avg")
    plt.plot(df_aci.index, comp_aci, color=C_ACI, linestyle="--", linewidth=1.5, label="ACI-Python Composite")

    ax = plt.gca()
    _fmt_ax(ax, "Temperature Standardized Anomalies 5-Year Moving Average\n"
                "(OPT solid | ACI-Python dashed)")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out.name}")


# ---------------------------------------------------------------------------
# Plot 2 — Precipitation
# Mirrors graficas.py > plot_rainfall_anomalies, with ACI-Python overlay
# ---------------------------------------------------------------------------
def plot_precipitation(df_opt: pd.DataFrame, df_aci: pd.DataFrame, out: Path):
    df_opt["ma"] = ma(df_opt["Anomalia_Lluvia"])

    prec_aci = ma(df_aci["precipitation"])

    plt.figure(figsize=(15, 6))
    plt.bar(df_opt["date"], df_opt["Anomalia_Lluvia"],
            color=C_BAR, alpha=1, label="OPT Standardized Anomalies", width=40)
    plt.plot(df_opt["date"], df_opt["ma"],
             color="black", linewidth=2, label="OPT 5-Year Moving Avg")
    plt.plot(df_aci.index, prec_aci,
             color=C_ACI, linewidth=2, linestyle="--", label="ACI-Python 5-Year Moving Avg")

    ax = plt.gca()
    _fmt_ax(ax, "Maximum 5-day rainfall Standardized Anomalies 5-Year Moving Average\n"
                "(OPT bars+solid | ACI-Python dashed)")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out.name}")


# ---------------------------------------------------------------------------
# Plot 3 — Drought
# ---------------------------------------------------------------------------
def plot_drought(df_opt: pd.DataFrame, df_aci: pd.DataFrame, out: Path):
    df_opt["ma"] = ma(df_opt["Anomalia_Sequia"])

    drgt_aci = ma(df_aci["drought"])

    plt.figure(figsize=(15, 6))
    plt.bar(df_opt["date"], df_opt["Anomalia_Sequia"],
            color=C_BAR, alpha=1, label="OPT Standardized Anomalies", width=40)
    plt.plot(df_opt["date"], df_opt["ma"],
             color="black", linewidth=2, label="OPT 5-Year Moving Avg")
    plt.plot(df_aci.index, drgt_aci,
             color=C_ACI, linewidth=2, linestyle="--", label="ACI-Python 5-Year Moving Avg")

    ax = plt.gca()
    _fmt_ax(ax, "Consecutive dry days Standardized Anomalies 5-Year Moving Average\n"
                "(OPT bars+solid | ACI-Python dashed)")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out.name}")


# ---------------------------------------------------------------------------
# Plot 4 — Wind
# ---------------------------------------------------------------------------
def plot_wind(df_opt: pd.DataFrame, df_aci: pd.DataFrame, out: Path):
    df_opt["ma"] = ma(df_opt["anomalies_above"])

    wind_aci = ma(df_aci["wind"])

    plt.figure(figsize=(15, 6))
    plt.bar(df_opt["date"], df_opt["anomalies_above"],
            color=C_BAR, alpha=1, label="OPT Standardized Anomalies", width=40)
    plt.plot(df_opt["date"], df_opt["ma"],
             color="black", linewidth=2, label="OPT 5-Year Moving Avg")
    plt.plot(df_aci.index, wind_aci,
             color=C_ACI, linewidth=2, linestyle="--", label="ACI-Python 5-Year Moving Avg")

    ax = plt.gca()
    _fmt_ax(ax, "Wind Power Standardized Anomalies 5-Year Moving Average\n"
                "(OPT bars+solid | ACI-Python dashed)")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out.name}")


# ---------------------------------------------------------------------------
# Plot 5 — ICA / ACI composite
# Mirrors graficas.py > plot_ICA, with ACI-Python ICA overlay
# ---------------------------------------------------------------------------
def plot_ica(df_temp: pd.DataFrame, df_prec: pd.DataFrame,
             df_drgt: pd.DataFrame, df_wind: pd.DataFrame,
             df_aci: pd.DataFrame, out: Path):
    # Build OPT ICA (graficas.py logic exactly)
    df_prec = df_prec.rename(columns={"Anomalia_Lluvia": "P_std"})
    df_drgt = df_drgt.rename(columns={"Anomalia_Sequia": "D_std"})
    df_wind = df_wind.rename(columns={"anomalies_above": "W_std"})

    ICA = df_wind[["date", "W_std"]].copy()
    ICA = ICA.merge(df_temp[["date", "t_90", "t_10"]], on="date", how="left")
    ICA = ICA.merge(df_prec[["date", "P_std"]], on="date", how="left")
    ICA = ICA.merge(df_drgt[["date", "D_std"]], on="date", how="left")
    ICA = ICA.sort_values("date").reset_index(drop=True)

    ICA["ICA"] = (ICA["t_90"] - ICA["t_10"] + ICA["W_std"] + ICA["P_std"] + ICA["D_std"]) / 5

    ICA["ICA_ma"]  = ma(ICA["ICA"])
    ICA["t90_ma"]  = ma(ICA["t_90"])
    ICA["t10_ma"]  = ma(ICA["t_10"])
    ICA["W_ma"]    = ma(ICA["W_std"])
    ICA["P_ma"]    = ma(ICA["P_std"])
    ICA["D_ma"]    = ma(ICA["D_std"])

    # ACI-Python overall index (no sea level) — stops at 1990 naturally
    aci_ma = ma(df_aci["ACI_no_sl"])

    plt.figure(figsize=(15, 6))
    # OPT components (graficas.py colours)
    plt.plot(ICA["date"], ICA["t90_ma"], color=C_T90,  alpha=0.6, linewidth=2, label="OPT T90 5-yr avg")
    plt.plot(ICA["date"], ICA["t10_ma"], color=C_T10,  alpha=0.4, linewidth=2, label="OPT T10 5-yr avg")
    plt.plot(ICA["date"], ICA["W_ma"],   color=C_WIND, alpha=0.5, linewidth=2, label="OPT Wind 5-yr avg")
    plt.plot(ICA["date"], ICA["P_ma"],   color=C_RAIN, alpha=0.5, linewidth=2, label="OPT Rainfall 5-yr avg")
    plt.plot(ICA["date"], ICA["D_ma"],   color=C_DRGT, alpha=0.5, linewidth=2, label="OPT Drought 5-yr avg")
    # OPT ICA
    plt.plot(ICA["date"], ICA["ICA_ma"], color="black", linewidth=2, label="OPT ICA 5-yr avg")
    # ACI-Python ICA overlay (stops at 1990 where data ends)
    plt.plot(df_aci.index, aci_ma, color=C_ACI, linewidth=2, linestyle="--",
             label="ACI-Python ICA 5-yr avg (no sea lvl)")

    ax = plt.gca()
    _fmt_ax(ax, "Actuarial Climate Index - 5-Year Moving Average\n"
                "(OPT solid | ACI-Python dashed, no sea level)")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading OPT component CSVs...")
    opt_temp  = load_opt_temp()
    opt_prec  = load_opt_precip()
    opt_drgt  = load_opt_drought()
    opt_wind  = load_opt_wind()

    print("Loading ACI-Python output...")
    df_aci = load_aci()

    print(f"OPT coverage: {opt_temp['date'].min().date()} -> {opt_temp['date'].max().date()}")
    print(f"ACI-Python coverage: {df_aci.index.min().date()} -> {df_aci.index.max().date()} (stops at 1990)")
    print()

    plot_temperature(opt_temp, df_aci, OUT_DIR / "comparison_temperature.png")
    plot_precipitation(opt_prec, df_aci, OUT_DIR / "comparison_precipitation.png")
    plot_drought(opt_drgt, df_aci, OUT_DIR / "comparison_drought.png")
    plot_wind(opt_wind, df_aci, OUT_DIR / "comparison_wind.png")
    plot_ica(opt_temp, opt_prec, opt_drgt, opt_wind, df_aci, OUT_DIR / "comparison_ICA.png")

    print(f"\nAll plots -> {OUT_DIR}")


if __name__ == "__main__":
    main()
