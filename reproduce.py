"""
Master reproducibility runner for ACI-CO paper outputs.

Usage:
    python reproduce.py

Runs both pipeline scripts in order, then writes a manifest of all generated
output files with MD5 checksums and timestamps. Exit code 0 = success.
"""

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT  = ROOT / "articles" / "graficas" / "paper_figures"

EXPECTED_OUTPUTS = [
    "fig1_colombia_map.png",
    "fig2_enso_neutral.png",
    "fig2b_enso_regression_scatter.png",
    "fig3_enso_diagnostics.png",
    "fig4_national_aci.png",
    "fig5_components.png",
    "fig5b_distributions.png",
    "fig6_regional_heatmap.png",
    "fig7a_regional.png",
    "fig8a_bootstrap_bands.png",
    "fig9_aci_oni_events.png",
    "fig_corr_matrix.png",
    "table3_enso_sensitivity.csv",
    "table6_mann_kendall.csv",
    "table6a_bootstrap_widths.csv",
    "table7_ungrd_summary.csv",
    "table8_negbin_results.csv",
    "table9_by_eventtype.csv",
]

SCRIPTS = [
    ROOT / "articles" / "produce_paper_plots.py",
    ROOT / "articles" / "build_validation_outputs.py",
]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_script(script: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"Running: {script.name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr, file=sys.stderr)
        raise RuntimeError(f"{script.name} exited with code {result.returncode}")
    print(f"  Completed in {elapsed:.1f}s")
    return {
        "script": script.name,
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 2),
        "stdout_lines": len(result.stdout.splitlines()),
    }


def build_manifest() -> dict:
    entries = {}
    missing = []
    for name in EXPECTED_OUTPUTS:
        p = OUT / name
        if p.exists():
            entries[name] = {
                "md5": md5(p),
                "size_bytes": p.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(
                    p.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        else:
            missing.append(name)
    return entries, missing


def main():
    run_results = []
    for script in SCRIPTS:
        run_results.append(run_script(script))

    entries, missing = build_manifest()

    manifest = {
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
        "python": sys.version,
        "scripts_run": run_results,
        "outputs": entries,
        "missing": missing,
    }

    manifest_path = OUT / "reproducibility_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Manifest written: {manifest_path}")
    print(f"Files verified:   {len(entries)}/{len(EXPECTED_OUTPUTS)}")
    if missing:
        print(f"MISSING outputs:  {missing}", file=sys.stderr)
        sys.exit(1)
    else:
        print("All expected outputs present. Reproducibility check passed.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
