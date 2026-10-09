"""
run_all.py — Reproduce the entire pipeline end to end.

Runs scripts 01..19 in order (step 10 is the earlier single-plan experiment; steps 12-19
implement the planning-time evaluation, common scoring and display items). Step 03 (weather retrieval) is network-bound and
resume-safe (per-case parquet cache); re-running continues from the cache.

Usage:
  python run_all.py            # run all steps
  python run_all.py --from 04  # resume from a given step
  python run_all.py --only 06  # run a single step
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    ("01", "01_filter_osha.py"),
    ("02", "02_validate_heat_labels.py"),
    ("03", "03_match_noaa_weather.py"),
    ("04", "04_model_case_crossover.py"),
    ("05", "05_heat_attribution.py"),
    ("06", "06_dayahead_prediction.py"),
    ("07", "07_exposure_weights.py"),
    ("08", "08_parse_dslib.py"),
    ("09", "09_build_risk_calendars.py"),
    ("10", "10_schedule_optimization.py"),
    ("11", "11_validate_verify.py"),
    ("12", "12_prediction_uncertainty.py"),
    ("12b", "12b_rule_vs_injury_discrimination.py"),
    ("13", "13_climatology_calendars.py"),
    ("14", "14_nbm_forecasts.py"),
    ("15", "15_evaluation_inputs.py"),
    ("15b", "15b_bls_hours.py"),
    ("16", "16_schedule_r1.py"),
    ("16c", "16c_solver_budget.py"),
    ("17", "17_posthoc_r1.py"),
]
HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="01")
    ap.add_argument("--only", dest="only", default=None)
    args = ap.parse_args()

    for code, script in STEPS:
        if args.only and code != args.only:
            continue
        if not args.only and code < args.start:
            continue
        print(f"\n{'='*70}\nRUNNING STEP {code}: {script}\n{'='*70}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / script)], cwd=str(HERE))
        dt = time.time() - t0
        if r.returncode != 0:
            print(f"STEP {code} FAILED (exit {r.returncode}) after {dt:.0f}s", flush=True)
            return r.returncode
        print(f"STEP {code} OK ({dt:.0f}s)", flush=True)
    print("\nPIPELINE COMPLETE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
