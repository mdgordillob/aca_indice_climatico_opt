# ARCHITECTURE.md

How this repository is put together: the science it implements, the actual
data flow (as opposed to the README's idealized version), where the three
articles fit, and a standing list of cleanup priorities. Companion to
`CLAUDE.md` (working conventions/guardrails for an assistant) — this file is
the "what is this and how does it fit together" reference.

## 1. The scientific target: ACI-CO

The codebase computes the **Colombian Actuarial Climate Index (ACI-CO)**, a
Colombian adaptation of the Actuaries Climate Index (ACI) framework, using
ERA5-Land reanalysis (~9km, 1961–2024). It is described, in order of
authority, by:

1. **`articles/aci_co_submission/article1.tex`** — canonical/current paper,
   "Assessing Climate Extremes in Colombia... (ACI-CO)". Use this for any
   methodology question.
2. **`articles/article1.tex`** (repo root) — an older draft of the same
   paper using LOWESS trend bands instead of the bootstrap approach in (1).
   Superseded; kept for history.
3. **`articles/Indice actuarial para Colombia.pdf`** — the *original* draft
   (Google Docs export, no `.tex` source), calling the index "ICA" rather
   than "ACI-CO". Six components (÷6) on ERA5 standard at 32km, parametric
   wind threshold (μ+1.28σ), no ENSO/bootstrap/UNGRD/CHIRPS work. This is
   the v0 the current paper superseded — useful for understanding *why*
   some scripts contain an "old" and a "new" version of the same logic
   (e.g., parametric vs. empirical wind threshold).
4. **`articles/AnnualICAForecast_Report.tex`** — a separate, later report
   pivoting the *forecasting* side to annual resolution (AR(1) vs. Holt ETS
   with ENSO). Documents a real methodological fix (monthly CDD forecasts
   were an interpolation artifact) but references `scripts/daily_forecast_to_annual.py`
   and `scripts/ar1_vs_ets_final.py`, neither of which exists in this repo —
   treat as a design document, not a description of shipped code. Also has
   a stray `\end{document}` mid-file (line 93) with more content appended
   after it; compiling it may truncate the PDF.

### The five components (per `article1.tex` §3.3)

| Component | Meaning | Sign in composite | Threshold |
|---|---|---|---|
| **T90** | fraction of days/month with T above baseline (1961–1990) 90th percentile | + | empirical P90 per grid cell × calendar month |
| **T10** | fraction of days/month with T below baseline 10th percentile | − | empirical P10 |
| **Rx5day** | max 5-day precipitation total per month, z-scored vs. 1961–1990 | + | z-score vs. monthly climatology |
| **CDD** | annual max consecutive dry days (<1mm), interpolated onto the monthly axis | + | — |
| **WP** | wind power (∝U³) exceedance | + | **empirical P90** (current) — replaces an earlier **parametric μ+1.28σ** threshold, discarded because WP is right-skewed |

Composite = unweighted mean of the 5 signed, standardized components
(divide by K=5). An earlier version divided by 6 for a placeholder
sea-level component that was dropped ("not included in current release
due to limited tide-gauge coverage") — sea level is *not* part of the
current ACI-CO despite `sealevel2.py` existing in `src/scripts/` (see §5).

### Extensions beyond the base ACI framework (the paper's 3 claimed contributions)

- **ENSO-neutral decomposition** (§4): asymmetric ARIMAX regression per
  component/region on Niño-3.4 (E⁺=max(E,0), E⁻=min(E,0)) + linear trend +
  AR(p) errors, to separate secular warming from ENSO-driven variability.
- **Bootstrap uncertainty bands** (§4.4): block-resample (L=12mo) the
  1961–1990 baseline composite into B=500 (national) / B=200 (regional)
  null-model replicates, testing whether the observed trend exceeds
  baseline-period variability. This replaced the LOWESS residual-envelope
  approach used in the root-level draft.
- **Multi-dataset robustness** (§5): ERA5-Land vs. ERA5 standard vs. CHIRPS.
- **UNGRD validation** (§7): negative-binomial regression of disaster counts
  (floods, landslides, windthrow, veg. fires — 49,734 events, 1998–2022) on
  ACI-CO components.

### Regions actually computed

Despite the paper conceptually mentioning 5 natural regions + 32
departments, the **implementation only aggregates 4 units**: national
(area-weighted, all of Colombia) + 3 departments chosen for insurance
relevance — **Antioquia, Cundinamarca-Bogotá, Valle del Cauca**. Shapefiles
also exist for Bogotá, Medellín, Cali, San Andrés/Providencia, Pacífico, and
Amazonas, but `data/processed/anomalias_{bogota,medellin,san_andres_providencia}/`
are empty — never populated.

## 2. Actual data flow

The README describes an idealized 8-step flow; several of the scripts it
names don't exist (see §5). The real flow, reconstructed from what scripts
actually read/write:

```
Stage 0  Download           data/raw/
  ecmwf_descarga.py  ──▶ data/raw/era5/*.grib          (CDS API, ERA5-Land Colombia bbox — canonical)
  sst_descarga.py    ──▶ data/raw/era5/era5_sst_*.grib  (SST, for ENSO index)
  get_aux_data.py    ──▶ data/raw/auxiliary/            (DEM/NDVI/etc., for downscaling)
  IDEAM station data ──▶ data/raw/ideam/*.rds           (read via pyreadr)
  [earthub_descarga.py / display.py — alternate Earth Data Hub downloaders; broken/misnamed, see §5]

Stage 1  Merge/resample     data/raw/ → data/processed/
  unir_archivos.py   ──▶ era5_daily_combined_{tmp,rain,wind}.nc  (hourly→daily, UTC-5 day/night split)
  run_aci_colombia.py (untracked) independently re-merges into data/processed/aci_daily/,
                          aci_colombia_1961_2024.csv — a from-scratch, faithful reimplementation
                          of the original ACI-Python algorithm, used as a cross-check against
                          the OPT pipeline below (see compare_aci_colombia.py, compare_repos.py)

Stage 2  Percentiles (baseline 1961-1990)
  calcular_percentil_{temperatura,lluvia,viento}.py ──▶ era5_*_percentil.nc, percentiles*.csv

Stage 3  Anomalies, per region
  calcular_anomalias_{temperatura,lluvia,viento}.py, orchestrated by
  calcular_anomalias_regiones.py over data/shapefiles/*.shp
  ──▶ data/processed/anomalias_<region>/anomalies_{temperature,precipitation,drought,wind}_combined.csv
      (populated: antioquia, cali, colombia, cundinamarca_bogota, valle_cauca;
       empty: bogota, medellin, san_andres_providencia)

Stage 4  Daily variants (newer, untracked)
  produce_daily_anomalies.py, append_2025_2026.py ──▶ data/processed/daily_anomalies/*.xlsx

Stage 5  ENSO / SST index
  anomalias_sst_daily.py, compare_enso_flags.py ──▶ sst_index_colombia_pacific.csv,
      enso_flag_comparison.csv, validated against NOAA CPC oni.ascii.txt

Stage 6  Downscaling (separate track, high-res 0.01°)
  downscale_era5_temperature.py, downscale_era5_tp.py (RandomForest, DEM/NDVI/IDEAM)
  ──▶ data/processed/downscaled/, validated in data/processed/era5_ideam_comparison/
      via bias_era5_ideam.py

Stage 7  Visualization / export
  graficas.py ──▶ articles/graficas/<region>/*.png, xlsx exports
  generate_correlation_matrices.py ──▶ correlation heatmaps per region

Stage 8  Forecasting  (two independent families — see CLAUDE.md)
  forecast_scripts/forecast_ica_{monthly,daily}.py   (AutoTS ensembles)
  forecast_scripts/ETSX_ica_forecast.py, ETSX_daily_ica_forecast.py
                                                       (SARIMAX/ETS + ENSO exogenous)
  ──▶ articles/graficas/forecast_*/

Dashboard (read-only consumer, not a pipeline stage)
  src/dashboard/app.py — Streamlit, reads Stage 3/5 CSVs directly
```

`data/indices/` and `data/zones/` exist but are effectively empty — no
script currently writes to those exact paths; they read as intended-but-unused
output locations (perhaps for a final composite index / zone lookup table
that was planned but never wired up).

## 3. Two parallel "correctness check" implementations

There are effectively two independent implementations of the core ACI
algorithm in this repo, and understanding *why* matters before touching
either:

- **The OPT pipeline** (`src/scripts/calcular_*`, tracked in git, README's
  documented flow) — the original project pipeline, per-region,
  percentile/anomaly based.
- **`run_aci_colombia.py`** (untracked) — a from-scratch reimplementation of
  the ACI-Python reference algorithm directly on raw GRIB, used to
  cross-validate the OPT pipeline's output. `compare_aci_colombia.py` and
  `compare_repos.py` (also untracked) exist specifically to diff the two
  and document the algorithmic differences (see `compare_repos.py`'s own
  docstring for the itemized list). This comparison is what grounds the
  paper's "validation against x-ACI" discussion (§8).

Don't assume one is "the old version" of the other — they're independent
implementations kept deliberately in sync for cross-validation, and both are
currently needed.

## 4. Where things are documented vs. not

| Doc | Status |
|---|---|
| `docs/methodology.md`, `docs/results.md`, `docs/zones_description.md` | **Empty stubs.** Methodology lives in `articles/aci_co_submission/article1.tex` instead. |
| `docs/explicacion_excel.md` | One truncated sentence, no body. |
| `docs/data_dictionary.md` | Early, incomplete fragment (HUMBOLDT-only) of `docs/Diccionario.md`. |
| `docs/Diccionario.md` | The real, complete data-source vetting document (~2400 lines, 20 candidate sources evaluated) that predates and explains the final choice of ERA5(-Land) + CHIRPS + UNGRD. Treat as canonical over `data_dictionary.md`. |
| `README.md` | Describes an idealized 8-step flow; two referenced scripts don't exist (`descargar_datos.py`, `sealevel.py` — see §5). |

## 5. Discrepancies between README/docstrings and reality

- README step 1 says run `descargar_datos.py`, which "invokes
  `ecmwf_descarga.py`" — `descargar_datos.py` doesn't exist;
  `ecmwf_descarga.py` is the actual entry point (it has its own `__main__`).
- README step 6 says run `sealevel.py` for sea-level analysis via psmsl.org —
  doesn't exist. Only `sealevel2.py` exists, and it doesn't do sea-level
  analysis at all — it's a Mann-Kendall trend + Q-Q normality check on
  precipitation-like series, with leftover Jupyter cell markers ("Celda 13"),
  suggesting it was pasted from a notebook and misnamed. Sea level isn't
  part of the current ACI-CO methodology (§1) — this script appears to be a
  dead end, not a gap to fill.
- `src/scripts/earthub_descarga.py` has a whole-file indentation error (no
  enclosing block) and cannot run as-is, on top of embedding a live-looking
  credential (see `CLAUDE.md`).
- `src/scripts/display.py` is not a plotting module despite the name — it's
  a Greece-precipitation download script (also embeds the same credential),
  and produced a 562MB `.nc` file at repo root as a side effect.
- `src/scripts/era5datasets.py`'s internal docstring says
  `"""compare_land_only.py"""`; `get_aux_data.py`'s header comment says
  `get_auxiliary_data.py`. Both are stale renames — trust the actual
  filename/git path, not the in-file comment.
- `graficas_altaresolucion.py` is a 4-line dead stub (imports only, no code).
- `ETSX_seasonal_motif_forecast.py` is very likely the "broken two-stage
  Lasso→ETS" approach that `ETSX_daily_ica_forecast.py`'s own docstring says
  it fixed — i.e., one of these two is superseded, not a live alternative.

## 6. Cleanup priorities (standing list)

Ordered roughly by urgency/impact. Confirm scope with the user before acting
on any of these — see `CLAUDE.md`'s "Git tracking gap" section first.

1. **Secrets**: rotate the two Earth Data Hub PAT tokens
   (`src/scripts/display.py:13`, `src/scripts/earthub_descarga.py:15`) and
   the CDS key in `GRID_SEARCH_AND_ERA5_GUIDE.md:44`; scrub from git history
   if the user wants them fully gone (rotation alone is sufficient for
   safety; history scrubbing is a separate, more invasive step).
2. **Commit the real pipeline**: `run_aci_colombia.py`, `compare_aci_colombia.py`,
   `compare_repos.py`, `produce_daily_anomalies.py`, `append_2025_2026.py`,
   `reproduce.py`, `plot_repo_anomalies.py`, `create_region_shapefiles.py`,
   `probe_2025.py`, and the `articles/aci_co_submission/` submission scaffold
   are all currently un-backed-up working-tree-only files.
3. **Purge committed generated junk**: `presentation.aux/.log/.out`,
   `texput.log`, `output.log`, `comparison.png`, `storm_daniel.png`, `.vs/`,
   `proyecto-indices/` (empty since the first commit),
   `colombia_pacific_sst_era5.nc`/`colombia_pacific_oni.csv` (data files
   committed at root instead of under `data/`).
4. **Fix `.gitignore`**: currently contains a literal PowerShell heredoc
   command wrapped around the real ignore rules (an `Out-File` invocation
   got committed instead of its output) — rewrite cleanly. Note that
   `*.nc`/`data/shapefiles/*` are "ignored" but still tracked, because the
   rule was added after those files were already committed.
5. **Delete disk-only bloat** (not committed, safe to remove locally):
   `acaopt_env.zip` (348MB), `acaopt_env.7z` (197MB),
   `tp_greece_sept_1993_2023.nc` (563MB, unrelated Greece test data),
   `__pycache__/`.
6. **Consolidate duplicated helpers** into `src/utils/` (currently empty):
   `get_available_years`, `get_cached_shapefile`, `get_cached_percentiles`
   are copy-pasted across `calcular_anomalias_temperatura.py`,
   `calcular_anomalias_viento.py`, `anomalies_precipitation.py`.
7. **Resolve naming/content mismatches**: `display.py` (not a display
   module), `sealevel2.py` (not sea level), stale internal docstrings in
   `era5datasets.py`/`get_aux_data.py` (§5).
8. **Decide the fate of superseded scripts**: `graficas_altaresolucion.py`
   (dead stub), `ETSX_seasonal_motif_forecast.py` (likely superseded by
   `ETSX_daily_ica_forecast.py`), `docs/data_dictionary.md` (superseded by
   `docs/Diccionario.md`), root `articles/article1.tex` (superseded by
   `articles/aci_co_submission/article1.tex`).
9. **Fill the empty doc stubs** (`docs/methodology.md`, `docs/results.md`,
   `docs/zones_description.md`) from `article1.tex`, or delete them if the
   paper is considered sufficient documentation going forward.
10. **Populate or drop** the empty `data/processed/anomalias_{bogota,medellin,san_andres_providencia}/`
    directories and the unused `data/indices/`, `data/zones/` locations.
