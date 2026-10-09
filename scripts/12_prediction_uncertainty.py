"""
12_prediction_uncertainty.py — Sample counts and bootstrap uncertainty for the day-level models.

Refits the information-set models of step 06 on the same temporal split (training 2015-2022,
held-out test 2023-August 2025) and reports, on the test set:
  * the number of rows, strata and case days in each split;
  * AUC, PR-AUC, Brier score, top-decile capture;
  * calibration slope (coefficient on the logit of the predicted probability) and
    calibration intercept (calibration-in-the-large: intercept of a logistic model with the
    logit of the predicted probability as an offset), with ideal values 1 and 0;
  * 95% percentile intervals from 1,000 bootstrap resamples of test strata, so that the case
    day and its referent days stay together;
  * reliability-curve data with a bootstrap band for the day-ahead and same-day models.

Outputs:
  outputs/tables/table_prediction_r1.{csv,md}
  outputs/tables/table_prediction_counts.{csv,md}
  outputs/tables/fig_reliability_r1.csv
  outputs/tables/verification12_prediction.json
"""
from __future__ import annotations

import sys
import warnings
from importlib import import_module

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import config
import common

warnings.filterwarnings("ignore")
log = common.setup_logger("12_prediction_uncertainty")
m06 = import_module("06_dayahead_prediction")
N_BOOT = 1000
rng = np.random.default_rng(config.RANDOM_SEED)

SPECS = {
    "Calendar only (month)": ([], ["month"]),
    "Day-ahead, persistence only": (m06.DAYAHEAD_NUM, []),
    "Day-ahead, persistence + state": (m06.DAYAHEAD_NUM, ["state"]),
    "Same-day observed weather": (m06.SAMEDAY_NUM, []),
    "Same-day observed weather + state, month and trade": (m06.SAMEDAY_NUM,
                                                           ["state", "month", "naics3"]),
}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration(y, p):
    lp = logit(p)
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    cil = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(),
                 offset=lp).fit().params[0]
    return float(slope), float(cil)


def metrics(y, p):
    slope, cil = calibration(y, p)
    return {"AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p), "Slope": slope, "Intercept": cil,
            "TopDecile": m06.top_decile(y, p)}


def main():
    common.banner(log, "STEP 12 - SAMPLE COUNTS AND BOOTSTRAP UNCERTAINTY")
    raw = pd.read_csv(config.DATA_INTERMEDIATE / "weather_matches.csv")
    df = m06.add_features(raw)
    tr0, tr1 = config.TRAIN_YEARS
    te0, te1 = config.TEST_YEARS
    train, test = df[df.year.between(tr0, tr1)], df[df.year.between(te0, te1)]

    counts = pd.DataFrame([
        {"Set": "Linked case-crossover cohort", "Rows": len(raw),
         "Strata": raw.case_id.nunique(), "Case days": int(raw.is_case.sum())},
        {"Set": "Excluded (missing daily minimum temperature)", "Rows": len(raw) - len(df),
         "Strata": None, "Case days": int(raw.is_case.sum() - df.is_case.sum())},
        {"Set": "Training, 2015-2022", "Rows": len(train), "Strata": train.case_id.nunique(),
         "Case days": int(train.is_case.sum())},
        {"Set": "Held-out test, 2023-August 2025", "Rows": len(test),
         "Strata": test.case_id.nunique(), "Case days": int(test.is_case.sum())},
    ])
    common.save_table(counts, "table_prediction_counts")
    log.info("\n" + counts.to_string(index=False))

    y_tr = train.is_case.astype(int).to_numpy()
    y_te = test.is_case.astype(int).to_numpy()
    strata = test.case_id.to_numpy()
    uniq = np.unique(strata)
    idx_by = {s: np.where(strata == s)[0] for s in uniq}
    boots = [np.concatenate([idx_by[s] for s in rng.choice(uniq, len(uniq), replace=True)])
             for _ in range(N_BOOT)]

    rows, rel_rows, verif = [], [], {}
    for name, (num, cat) in SPECS.items():
        pipe = m06.make_pipe(num, cat)
        pipe.fit(train[num + cat], y_tr)
        p = pipe.predict_proba(test[num + cat])[:, 1]
        point = metrics(y_te, p)
        bs = pd.DataFrame([metrics(y_te[b], p[b]) for b in boots])
        rec = {"Information set": name}
        for k, v in point.items():
            lo, hi = np.percentile(bs[k], [2.5, 97.5])
            rec[k] = v
            rec[f"{k}_lo"] = lo
            rec[f"{k}_hi"] = hi
        rows.append(rec)
        verif[name] = {k: round(float(v), 4) for k, v in point.items()}
        log.info(f"{name:52s} AUC {point['AUC']:.3f} [{rec['AUC_lo']:.3f},{rec['AUC_hi']:.3f}] "
                 f"slope {point['Slope']:.2f} [{rec['Slope_lo']:.2f},{rec['Slope_hi']:.2f}] "
                 f"intercept {point['Intercept']:.2f} [{rec['Intercept_lo']:.2f},"
                 f"{rec['Intercept_hi']:.2f}]")
        if name in ("Day-ahead, persistence only", "Same-day observed weather"):
            edges = np.quantile(p, np.linspace(0, 1, 9))
            bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 7)
            for b in range(8):
                mask = bins == b
                obs = [y_te[bi][bins[bi] == b].mean() for bi in boots[:N_BOOT]
                       if (bins[bi] == b).any()]
                rel_rows.append({"model": name, "bin": b,
                                 "mean_predicted": float(p[mask].mean()),
                                 "fraction_positive": float(y_te[mask].mean()),
                                 "fraction_lo": float(np.percentile(obs, 2.5)),
                                 "fraction_hi": float(np.percentile(obs, 97.5)),
                                 "n": int(mask.sum())})
    tbl = pd.DataFrame(rows)
    common.save_table(tbl.round(4), "table_prediction_r1")
    pd.DataFrame(rel_rows).to_csv(config.TABLES / "fig_reliability_r1.csv", index=False)
    common.save_json({"n_boot": N_BOOT, "point_estimates": verif,
                      "counts": counts.to_dict(orient="records")}, "verification12_prediction")
    return 0


if __name__ == "__main__":
    sys.exit(main())
