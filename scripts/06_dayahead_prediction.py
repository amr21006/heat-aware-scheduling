"""
06_dayahead_prediction.py — Day-Ahead Operational Predictive Modeling & Calibration.

Trains and evaluates machine learning models restricted strictly to information
observable prior to the workday (t-1):
  - Predictor set: previous-day weather, calendar features, and geographic coordinates
  - Temporal split: Out-of-time evaluation on held-out test period
  - Probability calibration: Isotonic regression for uninflated risk penalties
"""
from __future__ import annotations

import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
import common

warnings.filterwarnings("ignore")
log = common.setup_logger("11_dayahead_prediction")

# Features observable by the end of day t, used to score day t+1.
DAYAHEAD_NUM = ["tmax_lag1", "tmax_lag1_sq", "hot_days_prior3", "hot_days_prior7",
                "doy_sin", "doy_cos"]
# Features that require the realised observation for day t+1.
SAMEDAY_NUM = ["tmax", "tmax_sq", "tmin", "heat_index_filled", "hi_sq", "tmax_x_hi",
               "prcp_flag", "hot_days_prior3", "hot_days_prior7", "tmax_lag1",
               "doy_sin", "doy_cos"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["month"] = d["month"].astype(int)
    d["doy_sin"] = np.sin(2 * np.pi * d["doy"] / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * d["doy"] / 365.25)
    for c in ["tmax", "tmin", "tavg", "prcp", "tmax_lag1", "heat_index"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["prcp_flag"] = d["prcp_flag"].fillna(0).astype(float)
    for c in ["hot_days_prior3", "hot_days_prior7"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    d["tmax_lag1"] = d["tmax_lag1"].fillna(d["tmax"])
    d["heat_index_filled"] = d["heat_index"].fillna(d["tmax"])
    d["tmax_sq"] = d["tmax"] ** 2
    d["tmax_lag1_sq"] = d["tmax_lag1"] ** 2
    d["hi_sq"] = d["heat_index_filled"] ** 2
    d["tmax_x_hi"] = d["tmax"] * d["heat_index_filled"]
    d["state"] = d["state"].astype(str)
    d["naics3"] = d["naics3"].astype(str)
    return d.dropna(subset=["tmax", "tmin"])


def calib(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(np.log(p / (1 - p)).reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def top_decile(y, p):
    k = max(1, int(round(0.10 * len(y))))
    return float(y[np.argsort(-p)[:k]].sum() / max(1, y.sum()))


def make_pipe(num, cat):
    tr = []
    if num:
        tr.append(("num", StandardScaler(), num))
    if cat:
        tr.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    return Pipeline([("prep", ColumnTransformer(tr)),
                     ("clf", LogisticRegression(penalty="l2", C=1.0, max_iter=2000))])


def main():
    common.banner(log, "STEP 11 - DAY-AHEAD HEAT-RISK DISCRIMINATION")
    df = add_features(pd.read_csv(config.DATA_INTERMEDIATE / "weather_matches.csv"))
    tr0, tr1 = config.TRAIN_YEARS
    te0, te1 = config.TEST_YEARS
    train, test = df[df.year.between(tr0, tr1)], df[df.year.between(te0, te1)]
    y_tr = train.is_case.astype(int).to_numpy()
    y_te = test.is_case.astype(int).to_numpy()
    log.info(f"train rows={len(train)} (cases={y_tr.sum()}); "
             f"test rows={len(test)} (cases={y_te.sum()})")

    specs = {
        "Calendar only (month)": ("baseline", [], ["month"]),
        "Calendar only (state + month)": ("baseline", [], ["state", "month"]),
        "Day-ahead, persistence only": ("dayahead", DAYAHEAD_NUM, []),
        "Day-ahead, persistence + location": ("dayahead", DAYAHEAD_NUM, ["state"]),
        "Perfect foresight, observed day t+1 weather": ("sameday", SAMEDAY_NUM, []),
        "Perfect foresight + location and trade": ("sameday", SAMEDAY_NUM,
                                                   ["state", "month", "naics3"]),
    }

    rows, fitted = [], {}
    for name, (family, num, cat) in specs.items():
        pipe = make_pipe(num, cat)
        cols = num + cat
        pipe.fit(train[cols], y_tr)
        p_te = pipe.predict_proba(test[cols])[:, 1]
        slope, intercept = calib(y_te, p_te)
        rows.append({"Information set": name,
                     "Test AUC (%)": round(100 * roc_auc_score(y_te, p_te), 1),
                     "PR-AUC (%)": round(100 * average_precision_score(y_te, p_te), 1),
                     "Brier": round(brier_score_loss(y_te, p_te), 4),
                     "Calibration slope": round(slope, 2),
                     "Top-decile capture (%)": round(100 * top_decile(y_te, p_te), 1)})
        fitted[name] = (pipe, cols, family)
        log.info(f"{name:44s} AUC={rows[-1]['Test AUC (%)']:.1f} "
                 f"slope={slope:.2f} cap={rows[-1]['Top-decile capture (%)']:.1f}")

    table = pd.DataFrame(rows)
    common.save_table(table, "table_dayahead_prediction")

    # ---- calibration curve for the day-ahead model --------------------------
    pipe, cols, _ = fitted["Day-ahead, persistence only"]
    p_te = pipe.predict_proba(test[cols])[:, 1]
    frac, mean_pred = calibration_curve(y_te, p_te, n_bins=8, strategy="quantile")
    pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac}).to_csv(
        config.TABLES / "fig_dayahead_calibration.csv", index=False)

    # ---- persist both scheduling models, with the scaling constant ----------
    # The scheduling cost needs a bounded score. Predicted probabilities are divided
    # by a fixed reference quantile of the TRAINING distribution and clipped to [0,1],
    # so the same constant applies at every location and is reported in the paper.
    for key, out_name in [("Day-ahead, persistence only", "risk_model_dayahead"),
                          ("Perfect foresight, observed day t+1 weather", "risk_model_sameday")]:
        pipe, cols, _ = fitted[key]
        p_train = pipe.predict_proba(train[cols])[:, 1]
        ref = float(np.percentile(p_train, 99))
        joblib.dump({"pipeline": pipe, "columns": cols, "reference_p99": ref,
                     "information_set": key},
                    config.MODELS / f"{out_name}.joblib")
        log.info(f"saved {out_name}: reference p99 = {ref:.4f}")

    da = table.loc[table["Information set"] == "Day-ahead, persistence only"].iloc[0]
    pf = table.loc[table["Information set"] ==
                   "Perfect foresight, observed day t+1 weather"].iloc[0]
    cal = table.loc[table["Information set"] == "Calendar only (month)"].iloc[0]
    verification = {
        "calendar_only_auc": float(cal["Test AUC (%)"]),
        "dayahead_persistence_auc": float(da["Test AUC (%)"]),
        "perfect_foresight_auc": float(pf["Test AUC (%)"]),
        "dayahead_beats_calendar": bool(da["Test AUC (%)"] > cal["Test AUC (%)"] + 5),
        "dayahead_below_perfect_foresight": bool(da["Test AUC (%)"] < pf["Test AUC (%)"]),
        "n_test_cases": int(y_te.sum()),
    }
    common.save_json(verification, "verification11_dayahead")
    log.info(f"Verification: {verification}")
    log.info("\n" + table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
