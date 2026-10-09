"""
12b_rule_vs_injury_discrimination.py — Which day cost ranks days closer to injury risk?

The scheduling experiments score schedules with a count of heat-index days, which favors a
heat-index rule by construction wherever the rule and the injury-derived cost rank hot days
differently. This step asks the question on held-out injuries instead: on the 2023-August 2025
test strata, how well does each day cost separate the case day from the referent days of the
same month and place?

Scores compared (all computed from same-day realized weather, except the day-ahead model):
  * injury-derived same-day cost, as used by the scheduler (bounded at 1 by Equation 2);
  * the same cost without the bound (the predicted probability);
  * injury-derived day-ahead cost;
  * the heat-index rule (heat index scaled between 27 and 51 C, as used by the scheduler);
  * the daily maximum heat index itself, and the daily maximum temperature.
Metrics, each with a 95% interval and a paired difference against the bounded injury-derived cost
from 1,000 bootstrap resamples of test strata (ties count one half):
  * pooled AUC: case days against referent days of all test strata;
  * within-state concordance: case days against referent days of the same state in any month,
    which is the choice set of a scheduler at one location across the seasons of a project;
  * within-stratum concordance: case day against the referent days of the same month and place.
Exploratory regional check, added after the scheduling comparison showed the rule's advantage
concentrated at the two humid locations: the within-state concordance restricted to Florida and
Georgia (the states of those locations), and to the wider set of humid Gulf and Atlantic states, with
a bootstrap over the strata of the region.

Outputs:
  outputs/tables/table_rule_vs_injury_discrimination.csv
  outputs/tables/verification12b_discrimination.json
"""
from __future__ import annotations

import sys
import warnings
from importlib import import_module

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
import common

warnings.filterwarnings("ignore")
log = common.setup_logger("12b_rule_vs_injury_discrimination")
m06 = import_module("06_dayahead_prediction")
cal09 = import_module("09_build_risk_calendars")
N_BOOT = 1000
REF = "Injury-derived same-day cost (bounded, as scheduled)"
REGIONS = {"Humid_": ("Florida", "Georgia"),
           "HumidWide_": ("Florida", "Georgia", "Alabama", "Louisiana", "Mississippi", "South Carolina")}


def concordance(y, s, strata):
    """Share of (case, referent) pairs within a stratum where the case day scores higher."""
    num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        cs, rs = s[m & (y == 1)], s[m & (y == 0)]
        if len(cs) == 0 or len(rs) == 0:
            continue
        diff = cs[:, None] - rs[None, :]
        num += (diff > 0).sum() + 0.5 * (diff == 0).sum()
        den += diff.size
    return num / den if den else np.nan


def concordance_by(y, s, groups):
    """Case days against referent days of the same group (pairs pooled over groups)."""
    num = den = 0.0
    for g in np.unique(groups):
        m = groups == g
        cs, rs = s[m & (y == 1)], s[m & (y == 0)]
        if len(cs) == 0 or len(rs) == 0:
            continue
        cs, rs = np.sort(cs), np.sort(rs)
        lo = np.searchsorted(rs, cs, side="left")
        hi = np.searchsorted(rs, cs, side="right")
        num += lo.sum() + 0.5 * (hi - lo).sum()
        den += len(cs) * len(rs)
    return num / den if den else np.nan


def main():
    common.banner(log, "STEP 12b - DISCRIMINATION OF THE DAY COSTS ON HELD-OUT INJURIES")
    raw = pd.read_csv(config.DATA_INTERMEDIATE / "weather_matches.csv")
    df = m06.add_features(raw)
    te0, te1 = config.TEST_YEARS
    test = df[df.year.between(te0, te1)].reset_index(drop=True)
    models = {k: joblib.load(config.MODELS / f"risk_model_{k}.joblib") for k in ["sameday", "dayahead"]}
    sd, da = models["sameday"], models["dayahead"]
    q = sd["reference_p99"]
    p_sd = sd["pipeline"].predict_proba(test[sd["columns"]])[:, 1]
    p_da = da["pipeline"].predict_proba(test[da["columns"]])[:, 1]
    hi = test["heat_index_filled"].to_numpy(float)
    scores = {
        REF: np.minimum(p_sd / q, 1.0),
        "Injury-derived same-day cost without the bound": p_sd,
        "Injury-derived day-ahead cost": np.minimum(p_da / da["reference_p99"], 1.0),
        "Heat-index rule (scaled 27 to 51 C)": np.array([cal09.screening_risk(v) for v in hi]),
        "Daily maximum heat index": hi,
        "Daily maximum temperature": test["tmax"].to_numpy(float),
    }
    y = test.is_case.astype(int).to_numpy()
    strata = test.case_id.to_numpy()
    state = test.state.astype(str).to_numpy()
    uniq = np.unique(strata)
    idx_by = {s: np.where(strata == s)[0] for s in uniq}
    rng = np.random.default_rng(config.RANDOM_SEED)
    boots = [np.concatenate([idx_by[s] for s in rng.choice(uniq, len(uniq), replace=True)])
             for _ in range(N_BOOT)]

    def stats(idx):
        out = {}
        for k, s in scores.items():
            out[k] = (roc_auc_score(y[idx], s[idx]), concordance(y[idx], s[idx], strata[idx]),
                      concordance_by(y[idx], s[idx], state[idx]))
        return out

    point = stats(np.arange(len(y)))
    bs = [stats(b) for b in boots]

    # regional within-state concordance, bootstrapped over the strata of the region
    regional = {}
    for pre, states in REGIONS.items():
        idx0 = np.where(np.isin(state, states))[0]
        ureg = np.unique(strata[idx0])
        by = {g: idx0[strata[idx0] == g] for g in ureg}
        rrng = np.random.default_rng(config.RANDOM_SEED)
        rboots = [np.concatenate([by[g] for g in rrng.choice(ureg, len(ureg), replace=True)])
                  for _ in range(N_BOOT)]
        reg = lambda ix: {k: concordance_by(y[ix], sc[ix], state[ix]) for k, sc in scores.items()}
        rp, rb = reg(idx0), [reg(b) for b in rboots]
        for k in scores:
            v = np.array([b[k] for b in rb])
            dv = v - np.array([b[REF] for b in rb])
            regional.setdefault(k, {}).update({
                f"{pre}strata": len(ureg), f"{pre}state_concordance": rp[k],
                f"{pre}state_concordance_lo": np.nanpercentile(v, 2.5),
                f"{pre}state_concordance_hi": np.nanpercentile(v, 97.5),
                f"{pre}state_concordance_diff_vs_ref": rp[k] - rp[REF],
                f"{pre}state_concordance_diff_lo": np.nanpercentile(dv, 2.5),
                f"{pre}state_concordance_diff_hi": np.nanpercentile(dv, 97.5)})
    rows, verif = [], {}
    for k in scores:
        auc = np.array([b[k][0] for b in bs])
        con = np.array([b[k][1] for b in bs])
        d_auc = auc - np.array([b[REF][0] for b in bs])
        d_con = con - np.array([b[REF][1] for b in bs])
        cst = np.array([b[k][2] for b in bs])
        d_cst = cst - np.array([b[REF][2] for b in bs])
        rec = {"Score": k,
               "AUC": point[k][0], "AUC_lo": np.percentile(auc, 2.5), "AUC_hi": np.percentile(auc, 97.5),
               "Concordance": point[k][1], "Concordance_lo": np.nanpercentile(con, 2.5),
               "Concordance_hi": np.nanpercentile(con, 97.5),
               "AUC_diff_vs_ref": point[k][0] - point[REF][0],
               "AUC_diff_lo": np.percentile(d_auc, 2.5), "AUC_diff_hi": np.percentile(d_auc, 97.5),
               "Concordance_diff_vs_ref": point[k][1] - point[REF][1],
               "Concordance_diff_lo": np.nanpercentile(d_con, 2.5),
               "Concordance_diff_hi": np.nanpercentile(d_con, 97.5),
               "State_concordance": point[k][2], "State_concordance_lo": np.nanpercentile(cst, 2.5),
               "State_concordance_hi": np.nanpercentile(cst, 97.5),
               "State_concordance_diff_vs_ref": point[k][2] - point[REF][2],
               "State_concordance_diff_lo": np.nanpercentile(d_cst, 2.5),
               "State_concordance_diff_hi": np.nanpercentile(d_cst, 97.5)} | regional[k]
        rows.append(rec)
        verif[k] = {kk: round(float(v), 4) for kk, v in rec.items() if kk != "Score"}
        log.info(f"{k:52s} AUC {rec['AUC']:.3f} [{rec['AUC_lo']:.3f},{rec['AUC_hi']:.3f}]  "
                 f"dAUC {rec['AUC_diff_vs_ref']:+.3f} [{rec['AUC_diff_lo']:+.3f},{rec['AUC_diff_hi']:+.3f}] | "
                 f"state {rec['State_concordance']:.3f} d {rec['State_concordance_diff_vs_ref']:+.3f} "
                 f"[{rec['State_concordance_diff_lo']:+.3f},{rec['State_concordance_diff_hi']:+.3f}] | "
                 f"stratum {rec['Concordance']:.3f} d {rec['Concordance_diff_vs_ref']:+.3f} "
                 f"[{rec['Concordance_diff_lo']:+.3f},{rec['Concordance_diff_hi']:+.3f}]")
    tbl = pd.DataFrame(rows)
    common.save_table(tbl.round(4), "table_rule_vs_injury_discrimination")
    common.save_json({"n_boot": N_BOOT, "test_rows": int(len(y)), "test_strata": int(len(uniq)),
                      "reference": REF, "scores": verif}, "verification12b_discrimination")
    return 0


if __name__ == "__main__":
    sys.exit(main())
