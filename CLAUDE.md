# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

A research pipeline that builds the **Colombian Actuarial Climate Index (ACI-CO)**
from ERA5-Land reanalysis data (1961–2024) — a Colombian adaptation of the
Actuaries Climate Index (ACI) framework, extended with ENSO calibration, a
bootstrap uncertainty band, multi-dataset robustness checks, and validation
against Colombia's national disaster registry (UNGRD). The code, the papers in
`articles/`, and the docs in `docs/` describe the *same* project at different
points in its evolution — see `ARCHITECTURE.md` for how they relate and where
they've drifted apart.

This is an active research codebase, not a packaged library: there's no test
suite, several parallel/experimental implementations of the same stage, and
real working scripts that were never committed to git (see "Git tracking gap"
below). Read `ARCHITECTURE.md` before assuming a script is dead just because
it isn't referenced by the README.

## Before touching methodology-related code

The authoritative description of the ACI-CO methodology is
`articles/aci_co_submission/article1.tex` (§3–§7), **not** the root-level
`articles/article1.tex` (an older LOWESS-based draft) and **not** the empty
stubs in `docs/methodology.md` / `docs/results.md` / `docs/zones_description.md`.
If you're asked to change how a component (T90, T10, Rx5day, CDD, WP) is
computed, or how the composite/ENSO decomposition/uncertainty bands work,
read the relevant section of that `.tex` file first. Don't infer the
methodology from docstrings alone — several scripts contain stale comments
(wrong filenames, superseded approaches) noted in `ARCHITECTURE.md`.

## Git tracking gap — read this before deleting or "restoring" anything

A significant chunk of the *current, more-correct* pipeline exists only in
the working tree and has never been `git add`ed:
`run_aci_colombia.py`, `compare_aci_colombia.py`, `compare_repos.py`,
`create_region_shapefiles.py`, `probe_2025.py`, `produce_daily_anomalies.py`,
`append_2025_2026.py`, `reproduce.py`, `plot_repo_anomalies.py`, and most of
`articles/` (the `aci_co_submission/` folder, `produce_paper_plots.py`,
`build_validation_outputs.py`, etc.).

Conversely, things that **are** committed include generated junk
(`presentation.aux/log/out`, `texput.log`, `output.log`, `comparison.png`,
`storm_daniel.png`, `.vs/`, `proyecto-indices/`, root-level `.nc`/`.csv` data
files) and a `.gitignore` that was itself accidentally overwritten with a
PowerShell heredoc command instead of its output.

**Do not** assume `git status` clutter is safe to blow away without checking
which side of this split a file is on — untracked here often means
"uncommitted work in progress," not "disposable."

## Known secrets in the repo (do not add more)

Two Earth Data Hub PAT tokens are hardcoded and **committed** in
`src/scripts/display.py:13` and `src/scripts/earthub_descarga.py:15` (same
token, duplicated). A CDS API key also appears in
`GRID_SEARCH_AND_ERA5_GUIDE.md:44`. These need rotation/history-scrubbing by
the user, independent of any code change — flag it again if you notice
yourself about to touch either file. When writing or editing download
scripts, follow the pattern in `ecmwf_descarga.py` / `sst_descarga.py`
instead: `cdsapi.Client()` reads credentials from the user's local
`~/.cdsapirc`, nothing is embedded in source.

## Conventions actually in use (descriptive, not aspirational)

- Scripts are runnable modules with a `__main__` block, not a package — there's
  no `setup.py`/`pyproject.toml` and `src/utils/` is empty despite near-identical
  helper functions (`get_available_years`, `get_cached_shapefile`, etc.) being
  copy-pasted across `calcular_anomalias_temperatura.py`,
  `calcular_anomalias_viento.py`, and `anomalies_precipitation.py`. If you
  consolidate these, put the shared version in `src/utils/` — that's clearly
  its intended purpose, just never followed through.
- Spanish is the dominant language for pipeline scripts and docstrings
  (`calcular_*`, `unir_archivos.py`); English is dominant in the newer
  forecast scripts and the papers. Match whichever a file already uses rather
  than mixing languages within one file.
- Absolute Windows paths (`C:\Users\mdgor\...`) are hardcoded in several
  one-off scripts (`era5datasets.py`, `get_aux_data.py`, `bias_era5_ideam.py`,
  `compare_res.py`, `plot_sst_oni.py`). These are machine-specific and will
  break for anyone else — don't copy this pattern into new code; use relative
  paths from the repo root or a config value instead.
- Two independent forecasting families exist side by side and are **not**
  a superset/subset of each other: `forecast_ica_{monthly,daily}.py` (AutoTS
  ensembles) and the `ETS(X)_*` family (statsmodels SARIMAX/ETS + ENSO
  exogenous regressor). `ETSX_seasonal_motif_forecast.py` (the "broken
  two-stage Lasso→ETS" approach `ETSX_daily_ica_forecast.py`'s own docstring
  says it replaced) was deleted as superseded/unused.
- `src/forecast_scripts/common/` holds logic that was byte-identical or
  near-identical across the forecast scripts: `data_loading.py`
  (`load_monthly_data`/`extract_regional_series`, shared by
  `forecast_ica_monthly.py` and `ETS_ica_forecast.py`), `climate_index.py`
  (the ICA composite formula, shared by those two plus `ETSX_ica_forecast.py`),
  `stationarity.py` (`adf_test`/`make_stationary`/`reconstruct_from_differences`,
  shared by `ETS_ica_forecast.py` and `ETSX_ica_forecast.py`), and
  `diagnostics.py` (`ModelErrorDiagnostics`, moved from the old
  `error_diagnostics.py` — still not wired into any pipeline, see below).
  Each consuming class keeps a same-named method that thinly delegates to the
  shared function, so the public API didn't change. What's genuinely
  different per file (ONI/ENSO handling, SARIMAX vs. ETS fitting, daily vs.
  monthly data shapes) was deliberately left alone — those are structurally
  parallel, not copy-pasted, and merging them would be a real redesign, not a
  mechanical extraction.
- `ModelErrorDiagnostics` (`common/diagnostics.py`) is comprehensive (9-panel
  residual diagnostics + 5 normality tests) but still isn't called by
  `ETSX_ica_forecast.py`/`ETSX_daily_ica_forecast.py`, which each still carry
  their own inline 4-panel reimplementation. Wiring those to use the shared
  class instead would be a further improvement, but changes plot output
  shape/content, so treat it as a separate, deliberate follow-up rather than
  bundling it into an unrelated change.
- `example_grid_search_era5.py` (repo root) imports `ONIDataHandler` from
  `ETSX_ica_forecast.py` expecting `_fetch_oni_from_era5`/`_fetch_oni_from_noaa`/
  `_visualize_era5_oni` methods that don't exist on the class today — this
  predates the forecast_scripts modularization and is a pre-existing break,
  not something introduced by it.
- `articles/AnnualICAForecast_Report.tex` references
  `scripts/daily_forecast_to_annual.py` and `scripts/ar1_vs_ets_final.py` —
  neither exists anywhere in the repo. Treat that report's described
  annual-AR(1)-vs-Holt-ETS pipeline as documented-but-unimplemented, not as
  something you can find and modify.

## Running things

There is no test suite (`tests/` contains only a placeholder `test.txt`) and
no CI. "Verification" for this repo means re-running the relevant pipeline
stage against `data/` and comparing outputs/plots, or using the `/verify`
skill against a concrete script invocation — don't claim a change works
without doing that.

`reproduce.py` (untracked, at repo root) is the closest thing to an
end-to-end check: it regenerates the paper's figures/tables via
`articles/produce_paper_plots.py` and `articles/build_validation_outputs.py`
and MD5-checksums them against an expected manifest.

## When asked to "clean up" this codebase

See `ARCHITECTURE.md`'s "Cleanup priorities" section for the standing list.
Confirm with the user before any destructive step (deleting untracked files,
rewriting git history for the secrets, removing scripts that look dead) —
several things that look like dead scratch code are actually load-bearing
work that just never got committed.
