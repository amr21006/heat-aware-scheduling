"""
13_climatology_calendars.py — Climatological day costs for planning beyond the forecast range.

At a planning date the weather of a workday more than a week ahead is unknown; the planner's
best information for such a day is its climatology. For each scheduling location this step
builds daily NOAA station observations for the 1991-2020 normals period (same nearest-station
method as the realized calendars), applies the fitted day-level risk models and the heat-index
screening scale to every historical day, and averages by day of year. The averages are
smoothed with a centred 15-day circular moving window. No observation from the 2023-2025
evaluation window enters the climatology.

Outputs:
  data_external/risk_calendars/<loc>_climatology_1991_2020.csv
  outputs/tables/table_climatology_summary.{csv,md}
  outputs/tables/verification13_climatology.json
"""
from __future__ import annotations

import sys
import warnings
from datetime import date
from importlib import import_module

import joblib
import numpy as np
import pandas as pd

import config
import common

warnings.filterwarnings("ignore")
log = common.setup_logger("13_climatology_calendars")
cal09 = import_module("09_build_risk_calendars")

CAL_DIR = config.DATA_EXTERNAL / "risk_calendars"
CLIM_START, CLIM_END = date(1991, 1, 1), date(2020, 12, 31)
SMOOTH_HALF_WINDOW = 7          # days on each side of the centre day
HOT_DAY_C = 32.2


def features(cal: pd.DataFrame) -> pd.DataFrame:
    """Same feature construction as the realized calendars (09_build_risk_calendars)."""
    cal = cal.copy()
    for c in ["tmax", "tmin", "heat_index"]:
        cal[c] = cal[c].interpolate(limit_direction="both", limit=5)
    cal["prcp"] = cal["prcp"].fillna(0.0)
    cal["tavg"] = cal[["tmax", "tmin"]].mean(axis=1)
    cal["prcp_flag"] = (cal["prcp"] > 0).astype(float)
    cal["tmax_lag1"] = cal["tmax"].shift(1).fillna(cal["tmax"])
    hot = (cal["tmax"] >= HOT_DAY_C).astype(float)
    cal["hot_days_prior3"] = hot.shift(1).rolling(3, min_periods=1).sum().fillna(0)
    cal["hot_days_prior7"] = hot.shift(1).rolling(7, min_periods=1).sum().fillna(0)
    doy = cal["date"].dt.dayofyear
    cal["doy"] = doy
    cal["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    cal["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    cal["month"] = cal["date"].dt.month
    cal["heat_index_filled"] = cal["heat_index"].fillna(cal["tmax"])
    cal["tmax_sq"] = cal["tmax"] ** 2
    cal["tmax_lag1_sq"] = cal["tmax_lag1"] ** 2
    cal["hi_sq"] = cal["heat_index_filled"] ** 2
    cal["tmax_x_hi"] = cal["tmax"] * cal["heat_index_filled"]
    return cal


def circular_smooth(values: np.ndarray, half: int) -> np.ndarray:
    n = len(values)
    out = np.empty(n)
    for i in range(n):
        idx = [(i + k) % n for k in range(-half, half + 1)]
        out[i] = np.nanmean(values[idx])
    return out


def build(loc: str, lat: float, lon: float, models: dict):
    raw_f = CAL_DIR / f"{loc}_observed_1991_2020.csv"
    dates = list(pd.date_range(CLIM_START, CLIM_END, freq="D"))
    if raw_f.exists():
        raw = pd.read_csv(raw_f, parse_dates=["date"])
        prov = {"cached": True}
    else:
        log.info(f"{loc}: building {len(dates)} historical days from NOAA observations ...")
        tmax, tmin, prcp, dst = cal09.daily_observations(lat, lon, dates)
        hi, hst = cal09.heat_index_calendar(lat, lon, dates)
        raw = pd.DataFrame({"date": dates})
        raw["tmax"] = [tmax.get(d, np.nan) for d in dates]
        raw["tmin"] = [tmin.get(d, np.nan) for d in dates]
        raw["prcp"] = [prcp.get(d, np.nan) for d in dates]
        raw["heat_index"] = [hi.get(d, np.nan) for d in dates]
        raw.to_csv(raw_f, index=False)
        prov = {"cached": False, "daily_stations": dst[:6], "heat_index_stations": hst[:6]}
    cov = {c: float(raw[c].notna().mean()) for c in ["tmax", "tmin", "heat_index"]}
    log.info(f"{loc}: coverage tmax {cov['tmax']:.3f}, tmin {cov['tmin']:.3f}, "
             f"heat index {cov['heat_index']:.3f}")
    cal = features(raw).dropna(subset=["tmax", "tmin"])

    for key, bundle in models.items():
        cal[f"p_{key}"] = bundle["pipeline"].predict_proba(cal[bundle["columns"]])[:, 1]
    cal["r_screening"] = cal["heat_index_filled"].apply(cal09.screening_risk)

    g = cal.groupby("doy")
    clim = pd.DataFrame({"doy": np.arange(1, 367)})
    for col in ["p_sameday", "p_dayahead", "r_screening", "tmax", "heat_index_filled"]:
        m = g[col].mean().reindex(clim["doy"]).to_numpy(dtype=float)
        clim[f"{col}_clim"] = circular_smooth(m, SMOOTH_HALF_WINDOW)
    clim["n_years"] = g["tmax"].count().reindex(clim["doy"]).fillna(0).astype(int).to_numpy()
    clim.to_csv(CAL_DIR / f"{loc}_climatology_1991_2020.csv", index=False)
    return clim, cov, prov


def main():
    common.banner(log, "STEP 13 - CLIMATOLOGICAL DAY COSTS (1991-2020)")
    models = {k: joblib.load(config.MODELS / f"risk_model_{k}.joblib")
              for k in ["sameday", "dayahead"]}
    rows, prov_all = [], {}
    for loc, (lat, lon) in config.SCHEDULE_LOCATIONS.items():
        clim, cov, prov = build(loc, lat, lon, models)
        prov_all[loc] = {"coverage": cov, **prov}
        summer = clim[(clim["doy"] >= 152) & (clim["doy"] <= 243)]
        rows.append({"Location": loc.replace("_", " "),
                     "Years": "1991-2020",
                     "Daily maximum temperature coverage": round(cov["tmax"], 3),
                     "Heat index coverage": round(cov["heat_index"], 3),
                     "Mean summer Tmax, climatology (C)": round(summer["tmax_clim"].mean(), 1),
                     "Mean summer heat index, climatology (C)":
                         round(summer["heat_index_filled_clim"].mean(), 1),
                     "Mean summer same-day risk probability, climatology":
                         round(summer["p_sameday_clim"].mean(), 3)})
    tbl = pd.DataFrame(rows)
    common.save_table(tbl, "table_climatology_summary")
    common.save_json({"period": [CLIM_START.isoformat(), CLIM_END.isoformat()],
                      "smoothing_half_window_days": SMOOTH_HALF_WINDOW,
                      "provenance": prov_all}, "verification13_climatology")
    log.info("\n" + tbl.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
