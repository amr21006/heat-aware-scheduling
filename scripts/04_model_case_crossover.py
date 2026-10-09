"""
04_model_case_crossover.py — Time-stratified case-crossover association (Design A).

Conditional logistic regression with strata = case id (each heat case vs its own
same-month/same-weekday referent days). Reports odds ratios per temperature
increment with state-cluster bootstrap CIs, a nonlinear exposure-response curve,
distance-threshold sensitivity, and negative-control / placebo tests (Validation 5).

Outputs:
  outputs/tables/table04_casecrossover.{csv,md}
  outputs/tables/table04_distance_sensitivity.csv
  outputs/tables/table04_negative_controls.csv
  outputs/tables/fig03_exposure_response.csv      (exposure-response curve)
  outputs/tables/verification03_association.json
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

import config
import common

warnings.filterwarnings("ignore")
log = common.setup_logger("04_model_case_crossover")
rng = np.random.default_rng(config.RANDOM_SEED)


def prep(df: pd.DataFrame, exposure: str) -> pd.DataFrame:
    d = df.dropna(subset=[exposure]).copy()
    g = d.groupby("case_id")["is_case"]
    has_case = g.transform(lambda s: (s == 1).any())
    has_ctrl = g.transform(lambda s: (s == 0).any())
    d = d[has_case & has_ctrl]
    return d


def fit_clogit(d: pd.DataFrame, exog_cols: list[str]):
    X = d[exog_cols].astype(float)
    res = ConditionalLogit(d["is_case"].astype(int).to_numpy(),
                           X.to_numpy(), groups=d["case_id"].to_numpy()).fit(disp=0)
    return res


def or_ci(beta, se, scale=1.0):
    return (np.exp(beta * scale),
            np.exp((beta - 1.96 * se) * scale),
            np.exp((beta + 1.96 * se) * scale))


def state_cluster_bootstrap(d: pd.DataFrame, exog_cols: list[str], col_idx=0, nreps=200):
    """Percentile CI for one coefficient, resampling STATES with replacement."""
    states = d["state"].dropna().unique()
    betas = []
    for _ in range(nreps):
        samp = rng.choice(states, size=len(states), replace=True)
        parts, off = [], 0
        for s in samp:
            sub = d[d["state"] == s].copy()
            if sub.empty:
                continue
            sub["case_id"] = sub["case_id"].astype(str) + f"__b{off}"  # unique strata
            off += 1
            parts.append(sub)
        bd = pd.concat(parts, ignore_index=True)
        bd = prep(bd, exog_cols[0])
        if bd["case_id"].nunique() < 5:
            continue
        try:
            r = fit_clogit(bd, exog_cols)
            betas.append(r.params[col_idx])
        except Exception:
            continue
    betas = np.array(betas)
    if len(betas) < 20:
        return (np.nan, np.nan)
    return (np.percentile(betas, 2.5), np.percentile(betas, 97.5))


def main():
    common.banner(log, "STEP 04 — CASE-CROSSOVER CONDITIONAL LOGISTIC REGRESSION")
    wm_path = config.DATA_INTERMEDIATE / "weather_matches.csv"
    if not wm_path.exists():
        log.error("weather_matches.csv not found — run 03 first.")
        return 1
    df = pd.read_csv(wm_path)
    log.info(f"Loaded {len(df)} rows; {df['case_id'].nunique()} strata; "
             f"{int(df['is_case'].sum())} case-days")

    results_rows = []

    # ---- Primary exposure: daily max temperature ----
    for exposure, label, scale, unit in [
        ("tmax", "Daily max temperature", 5.0, "per +5°C"),
        ("heat_index", "Daily max heat index", 5.0, "per +5°C"),
        ("tmin", "Daily min temperature", 5.0, "per +5°C"),
    ]:
        d = prep(df, exposure)
        if d["case_id"].nunique() < 10:
            log.warning(f"{exposure}: too few strata ({d['case_id'].nunique()})")
            continue
        res = fit_clogit(d, [exposure])
        beta, se = res.params[0], res.bse[0]
        oR, lo, hi = or_ci(beta, se, scale)
        # state-cluster bootstrap CI
        blo, bhi = state_cluster_bootstrap(d, [exposure], 0, nreps=200)
        bOR_lo = np.exp(blo * scale) if np.isfinite(blo) else np.nan
        bOR_hi = np.exp(bhi * scale) if np.isfinite(bhi) else np.nan
        pval = res.pvalues[0]
        results_rows.append({
            "Exposure": label, "Increment": unit,
            "OR": round(oR, 3), "CI_low_model": round(lo, 3), "CI_high_model": round(hi, 3),
            "CI_low_stateboot": round(bOR_lo, 3) if np.isfinite(bOR_lo) else None,
            "CI_high_stateboot": round(bOR_hi, 3) if np.isfinite(bOR_hi) else None,
            "p_value": f"{pval:.2e}", "n_strata": int(d['case_id'].nunique()),
        })
        log.info(f"{label}: OR({unit})={oR:.3f} [{lo:.3f},{hi:.3f}] model; "
                 f"[{bOR_lo:.3f},{bOR_hi:.3f}] state-boot; p={pval:.2e}")

    # ---- Adjusted model: tmax + precip indicator ----
    d2 = prep(df, "tmax")
    d2["prcp_flag"] = d2["prcp_flag"].fillna(0).astype(float)
    res2 = fit_clogit(d2, ["tmax", "prcp_flag"])
    oR, lo, hi = or_ci(res2.params[0], res2.bse[0], 5.0)
    results_rows.append({
        "Exposure": "Daily max temp (adj. precip)", "Increment": "per +5°C",
        "OR": round(oR, 3), "CI_low_model": round(lo, 3), "CI_high_model": round(hi, 3),
        "CI_low_stateboot": None, "CI_high_stateboot": None,
        "p_value": f"{res2.pvalues[0]:.2e}", "n_strata": int(d2['case_id'].nunique()),
    })
    pflag_or, pflo, pfhi = or_ci(res2.params[1], res2.bse[1], 1.0)
    results_rows.append({
        "Exposure": "Precipitation day (adj. temp)", "Increment": "yes vs no",
        "OR": round(pflag_or, 3), "CI_low_model": round(pflo, 3), "CI_high_model": round(pfhi, 3),
        "CI_low_stateboot": None, "CI_high_stateboot": None,
        "p_value": f"{res2.pvalues[1]:.2e}", "n_strata": int(d2['case_id'].nunique()),
    })

    table4 = pd.DataFrame(results_rows)
    common.save_table(table4, "table04_casecrossover")

    # ---- Nonlinear exposure-response (natural cubic spline of tmax) ----
    try:
        from patsy import dmatrix, build_design_matrices
        d3 = prep(df, "tmax")
        basis = dmatrix("cr(tmax, df=4)", {"tmax": d3["tmax"]}, return_type="dataframe")
        di = basis.design_info                       # reuse exact basis for prediction
        spline_cols = [c for c in basis.columns if c != "Intercept"]
        d3s = pd.concat([d3[["is_case", "case_id"]].reset_index(drop=True),
                         basis[spline_cols].reset_index(drop=True)], axis=1)
        res_sp = ConditionalLogit(d3s["is_case"].astype(int).to_numpy(),
                                  d3s[spline_cols].to_numpy(),
                                  groups=d3s["case_id"].to_numpy()).fit(disp=0)
        grid = np.linspace(d3["tmax"].quantile(0.02), d3["tmax"].quantile(0.98), 60)
        ref_t = float(d3["tmax"].quantile(0.10))     # in-range reference (cool day)
        gb = pd.DataFrame(build_design_matrices([di], {"tmax": grid})[0], columns=basis.columns)
        rb = pd.DataFrame(build_design_matrices([di], {"tmax": np.array([ref_t])})[0], columns=basis.columns)
        beta = res_sp.params
        G = gb[spline_cols].to_numpy() - rb[spline_cols].to_numpy()   # centered basis vs reference
        logodds = G @ beta
        cov = np.asarray(res_sp.cov_params())
        se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", G, cov, G), 0, None))  # delta method
        er = pd.DataFrame({"tmax_c": grid, "log_OR": logodds, "OR": np.exp(logodds),
                           "OR_lo": np.exp(logodds - 1.96 * se),
                           "OR_hi": np.exp(logodds + 1.96 * se), "ref_c": ref_t})
        er.to_csv(config.TABLES / "fig03_exposure_response.csv", index=False)
        log.info(f"Exposure-response: OR at {grid[-1]:.0f}°C vs ref {ref_t:.0f}°C = "
                 f"{np.exp(logodds[-1]):.2f} [{np.exp(logodds[-1]-1.96*se[-1]):.2f}, "
                 f"{np.exp(logodds[-1]+1.96*se[-1]):.2f}]")
    except Exception as exc:
        log.warning(f"spline exposure-response skipped (insufficient/ill-conditioned data): {exc}")

    # ---- Distance-threshold sensitivity ----
    sens_rows = []
    for thr in [25, 50, 75, 100, config.MAX_SEARCH_DISTANCE_KM]:
        dd = df[(df["distance_km"].fillna(9999) <= thr)]
        dd = prep(dd, "tmax")
        if dd["case_id"].nunique() < 10:
            continue
        r = fit_clogit(dd, ["tmax"])
        oR, lo, hi = or_ci(r.params[0], r.bse[0], 5.0)
        sens_rows.append({"max_distance_km": thr, "n_strata": int(dd["case_id"].nunique()),
                          "OR_per5C": round(oR, 3), "CI_low": round(lo, 3), "CI_high": round(hi, 3)})
    sens = pd.DataFrame(sens_rows)
    common.save_table(sens, "table04_distance_sensitivity")
    log.info(f"Distance sensitivity:\n{sens}")

    # ---- Negative controls + placebo (Validation 5) ----
    neg_rows = []
    # primary heat reference
    dprimary = prep(df, "tmax")
    rprimary = fit_clogit(dprimary, ["tmax"])
    op, lp, hp = or_ci(rprimary.params[0], rprimary.bse[0], 5.0)
    neg_rows.append({"outcome": "Environmental heat (primary)", "OR_per5C": round(op, 3),
                     "CI_low": round(lp, 3), "CI_high": round(hp, 3),
                     "n_strata": int(dprimary["case_id"].nunique())})

    negf = config.DATA_INTERMEDIATE / "weather_matches_negcontrol.csv"
    if negf.exists():
        ndf = pd.read_csv(negf)
        nd = prep(ndf, "tmax")
        rn = fit_clogit(nd, ["tmax"])
        on, ln, hn = or_ci(rn.params[0], rn.bse[0], 5.0)
        neg_rows.append({"outcome": "Non-heat amputation (neg. control)", "OR_per5C": round(on, 3),
                         "CI_low": round(ln, 3), "CI_high": round(hn, 3),
                         "n_strata": int(nd["case_id"].nunique())})
    else:
        log.warning("negcontrol weather not ready yet — skipping amputation control")

    # within-stratum permutation placebo: randomly reassign the case day within each stratum
    placebo_ors = []
    base = prep(df, "tmax")
    for rep in range(200):
        perm = base.copy()
        perm["is_case"] = perm.groupby("case_id")["is_case"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        try:
            rp = fit_clogit(perm, ["tmax"])
            placebo_ors.append(np.exp(rp.params[0] * 5.0))
        except Exception:
            continue
    placebo_ors = np.array(placebo_ors)
    neg_rows.append({"outcome": "Placebo (permuted case day, 200x)",
                     "OR_per5C": round(float(np.median(placebo_ors)), 3),
                     "CI_low": round(float(np.percentile(placebo_ors, 2.5)), 3),
                     "CI_high": round(float(np.percentile(placebo_ors, 97.5)), 3),
                     "n_strata": int(base["case_id"].nunique())})
    negt = pd.DataFrame(neg_rows)
    common.save_table(negt, "table04_negative_controls")
    log.info(f"Negative controls:\n{negt}")

    verification = {
        "primary_OR_tmax_per5C": float(op),
        "primary_CI": [float(lp), float(hp)],
        "primary_direction_positive": bool(op > 1.0),
        "placebo_median_OR": float(np.median(placebo_ors)),
        "placebo_near_null": bool(0.9 <= np.median(placebo_ors) <= 1.1),
        "n_strata_primary": int(dprimary["case_id"].nunique()),
        "distance_sensitivity_consistent": bool(
            (sens["OR_per5C"] > 1).all()) if len(sens) else None,
    }
    common.save_json(verification, "verification03_association")
    log.info(f"Verification: {verification}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
