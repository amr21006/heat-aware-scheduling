"""
08_validate_verify.py — Consolidated validation & verification report.

Reads every verification JSON produced by steps 01-06 and checks each acceptance
target from the research plan, emitting a single PASS/FAIL table. Steps not yet
run are reported as PENDING (the script is safe to run at any pipeline stage).

Outputs:
  outputs/tables/validation_report.{csv,md}
  outputs/tables/validation_report.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd
import config
import common

log = common.setup_logger("11_validate_verify")
T = config.TABLES


def load(name):
    p = T / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def check(rows, area, target, value, ok):
    status = "PENDING" if ok is None else ("PASS" if ok else "FAIL")
    rows.append({"Area": area, "Acceptance target": target,
                 "Observed": value, "Status": status})


def main():
    common.banner(log, "STEP 11 — VALIDATION & VERIFICATION REPORT")
    rows = []

    # ---- Verification 1: data integrity ----
    v1 = load("verification01_data_integrity")
    if v1:
        check(rows, "V1 Data integrity", "SHA-256 matches expected",
              v1["sha256_matches_expected"], v1["sha256_matches_expected"])
        check(rows, "V1 Data integrity", "Row count = 103,750",
              v1["row_count"], v1["row_count_matches_expected"])
        check(rows, "V1 Data integrity", "Construction NAICS 23 = 18,617",
              v1["construction_count"], v1["construction_matches_expected"])
        check(rows, "V1 Data integrity", "Environmental-heat cases = 605",
              v1["heat_count"], v1["heat_matches_expected"])
    else:
        check(rows, "V1 Data integrity", "all checks", "-", None)

    # ---- Validation 1: labels ----
    v2 = load("verification04_labeling")
    if v2:
        check(rows, "Val1 Labels", "Cohen's kappa >= 0.75",
              round(v2["population_kappa"], 3), v2["kappa_target_met"])
        check(rows, "Val1 Labels", "Precision vs burns >= 0.85",
              round(v2["precision_against_burns"], 3), v2["precision_vs_burns_target_met"])
        check(rows, "Val1 Labels", "Human-review frame >= 500", v2["sample_size"],
              v2["sample_size"] >= 500)
    else:
        check(rows, "Val1 Labels", "all checks", "-", None)

    # ---- Validation 2: weather matching ----
    v3 = load("verification02_weather_matching")
    if v3:
        check(rows, "Val2 Weather", "Median station distance <= 50 km",
              round(v3["dist_median_km"], 1), v3["dist_median_km"] <= 50)
        check(rows, "Val2 Weather", "Case-days within 100 km >= 90%",
              round(v3["within_100km_pct"], 1), v3["within_100km_pct"] >= 90)
        check(rows, "Val2 Weather", "ERA5 fallback (flagged) <= 10%",
              round(v3["pct_ERA5_fallback_casedays"], 1), v3["pct_ERA5_fallback_casedays"] <= 10)
        check(rows, "Val2 Weather", "Heat-index availability reported",
              round(v3["pct_heatindex_available_casedays"], 1), True)
    else:
        check(rows, "Val2 Weather", "all checks", "-", None)

    # ---- Validation 3: association ----
    v4 = load("verification03_association")
    if v4:
        check(rows, "Val3 Association", "Heat OR direction positive (>1)",
              round(v4["primary_OR_tmax_per5C"], 3), v4["primary_direction_positive"])
        ci = v4["primary_CI"]
        check(rows, "Val3 Association", "95% CI excludes 1",
              f"[{ci[0]:.2f}, {ci[1]:.2f}]", ci[0] > 1.0)
        check(rows, "Val3 Association", "Placebo near null (0.9-1.1)",
              round(v4["placebo_median_OR"], 3), v4["placebo_near_null"])
        if v4.get("distance_sensitivity_consistent") is not None:
            check(rows, "Val3 Association", "Distance-sensitivity consistent (OR>1)",
                  v4["distance_sensitivity_consistent"], v4["distance_sensitivity_consistent"])
    else:
        check(rows, "Val3 Association", "all checks", "-", None)

    # ---- Validation 4: prediction ----
    v5 = load("verification04b_prediction")
    if v5:
        check(rows, "Val4 Prediction", "Best test AUC reported",
              round(v5["best_auc"], 3), True)
        check(rows, "Val4 Prediction", "Weather beats month-only (+>=0.05 AUC)",
              f"{v5['full_logit_auc']:.3f} vs {v5['month_only_auc']:.3f}",
              v5["weather_improves_over_month"])
    else:
        check(rows, "Val4 Prediction", "all checks", "-", None)

    # ---- Validation 6 / Verification 6: scheduling ----
    v6 = load("verification06_scheduling")
    if v6:
        check(rows, "Val6 Scheduling", "Resource violations = 0",
              v6["total_resource_violations"], v6["total_resource_violations"] == 0)
        check(rows, "Val6 Scheduling", "Precedence violations = 0",
              v6["total_precedence_violations"], v6["total_precedence_violations"] == 0)
        check(rows, "Val6 Scheduling", "Heat-risk reduction > 0 (same duration)",
              f"{v6['mean_heat_reduction_pct_same_makespan']:.1f}%",
              v6["mean_heat_reduction_pct_same_makespan"] > 0)
        check(rows, "Val6 Scheduling", "Baseline >= CP-SAT optimal duration",
              v6["blind_ge_opt_makespan"], v6["blind_ge_opt_makespan"])
    else:
        check(rows, "Val6 Scheduling", "all checks", "-", None)

    rep = pd.DataFrame(rows)
    common.save_table(rep, "validation_report")
    n_pass = int((rep["Status"] == "PASS").sum())
    n_fail = int((rep["Status"] == "FAIL").sum())
    n_pend = int((rep["Status"] == "PENDING").sum())
    summary = {"pass": n_pass, "fail": n_fail, "pending": n_pend,
               "all_run_checks_pass": bool(n_fail == 0 and n_pend == 0)}
    common.save_json(summary, "validation_report")
    log.info(f"\n{rep.to_string(index=False)}")
    log.info(f"SUMMARY: {n_pass} PASS, {n_fail} FAIL, {n_pend} PENDING")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
