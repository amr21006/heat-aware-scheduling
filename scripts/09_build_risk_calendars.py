"""
09_build_risk_calendars.py — Continuous Multi-Location Thermal Risk Calendars.

Generates date-indexed daily risk costs [0, 1] across geographic climate locations
using fitted day-ahead models and weather station time series.
"""
from __future__ import annotations

import sys
import warnings
from datetime import date, datetime, timedelta

import joblib
import numpy as np
import pandas as pd

import config
import common

warnings.filterwarnings("ignore")
log = common.setup_logger("12_build_risk_calendars")

CAL_DIR = config.DATA_EXTERNAL / "risk_calendars"
CAL_DIR.mkdir(parents=True, exist_ok=True)

# Continuous window: the whole held-out period in the OSHA extract.
WINDOW_START = date(2023, 1, 1)
WINDOW_END = date(2025, 8, 31)

HI_LO, HI_HI = 27.0, 51.0          # National Weather Service heat-index band anchors (deg C)
HIGH_RISK_THRESHOLD = 0.5
HOT_DAY_C = 32.2

LOC_STATE = {"TX_Austin": "Texas", "FL_Orlando": "Florida",
             "GA_Atlanta": "Georgia", "AZ_Phoenix": "Arizona", "IL_Chicago": "Illinois"}


def screening_risk(hi_c):
    if hi_c is None or (isinstance(hi_c, float) and np.isnan(hi_c)):
        return np.nan
    return float(np.clip((hi_c - HI_LO) / (HI_HI - HI_LO), 0.0, 1.0))


def daily_observations(lat, lon, dates):
    """Nearest-station daily tmax, tmin and precipitation across the whole window."""
    near = common.nearest_stations(lat, lon, config.N_NEAREST_STATIONS)
    years = sorted({d.year for d in dates})
    tmax, tmin, prcp, used = {}, {}, {}, []
    for sid, srow in near.sort_values("distance_km").iterrows():
        dist = float(srow["distance_km"])
        if dist > config.MAX_SEARCH_DISTANCE_KM:
            break
        got_any = False
        for year in years:
            sd = common.station_year_daily(sid, year)
            if sd is None or len(sd) == 0:
                continue
            for d in dates:
                if d.year != year or d not in sd.index:
                    continue
                if d not in tmax and pd.notna(sd.at[d, "tmax"]):
                    tmax[d] = float(sd.at[d, "tmax"])
                    got_any = True
                if d not in tmin and pd.notna(sd.at[d, "tmin"]):
                    tmin[d] = float(sd.at[d, "tmin"])
                if d not in prcp and "prcp" in sd.columns and pd.notna(sd.at[d, "prcp"]):
                    prcp[d] = float(sd.at[d, "prcp"])
        if got_any:
            used.append({"station_id": sid, "distance_km": round(dist, 2)})
        if len(tmax) >= len(dates) * 0.99 and len(tmin) >= len(dates) * 0.99:
            break
    return tmax, tmin, prcp, used


def heat_index_calendar(lat, lon, dates):
    """Daily-maximum heat index from NOAA Integrated Surface Database hourly records."""
    near = common.nearest_stations(lat, lon, config.N_NEAREST_STATIONS)
    out, used = {}, []
    span_start = datetime.combine(min(dates).date(), datetime.min.time())
    span_end = datetime.combine(max(dates).date() + timedelta(days=1), datetime.min.time())
    for sid, srow in near.sort_values("distance_km").iterrows():
        dist = float(srow["distance_km"])
        if dist > config.MAX_SEARCH_DISTANCE_KM:
            break
        try:
            h = common.fetch_isd_hourly(sid, span_start, span_end)
        except Exception:
            h = None
        if h is None or len(h) == 0 or "temp" not in h.columns or "rhum" not in h.columns:
            continue
        h = h.dropna(subset=["temp", "rhum"])
        if h.empty:
            continue
        hi = common.heat_index_series_c(h["temp"].astype(float), h["rhum"].astype(float))
        hi.index = h.index
        daily_max = hi.groupby(hi.index.normalize()).max()
        added = 0
        for d in dates:
            if d in out:
                continue
            v = daily_max.get(pd.Timestamp(d), np.nan)
            if pd.notna(v):
                out[d] = float(v)
                added += 1
        if added:
            used.append({"station_id": sid, "distance_km": round(dist, 2), "days": added})
        if len(out) >= len(dates) * 0.98:
            break
    return out, used


def build(loc, lat, lon, models):
    cache = CAL_DIR / f"{loc}_continuous.csv"
    dates = list(pd.date_range(WINDOW_START, WINDOW_END, freq="D"))

    if cache.exists():
        cal = pd.read_csv(cache, parse_dates=["date"])
        if len(cal) == len(dates):
            log.info(f"{loc}: reusing cached continuous calendar ({len(cal)} days)")
            return cal, {"cached": True}

    log.info(f"{loc}: building {len(dates)} days from NOAA observations ...")
    tmax, tmin, prcp, dstations = daily_observations(lat, lon, dates)
    hi, histations = heat_index_calendar(lat, lon, dates)
    log.info(f"{loc}: tmax {len(tmax)}/{len(dates)}, tmin {len(tmin)}/{len(dates)}, "
             f"heat index {len(hi)}/{len(dates)}")

    cal = pd.DataFrame({"date": dates})
    cal["tmax"] = [tmax.get(d, np.nan) for d in dates]
    cal["tmin"] = [tmin.get(d, np.nan) for d in dates]
    cal["prcp"] = [prcp.get(d, np.nan) for d in dates]
    cal["heat_index"] = [hi.get(d, np.nan) for d in dates]

    # Interpolate isolated gaps in the observed series so the schedule horizon is
    # continuous; the number of interpolated days is reported.
    gaps = {c: int(cal[c].isna().sum()) for c in ["tmax", "tmin", "heat_index"]}
    for c in ["tmax", "tmin", "heat_index"]:
        cal[c] = cal[c].interpolate(limit_direction="both", limit=5)
    cal["prcp"] = cal["prcp"].fillna(0.0)
    remaining = {c: int(cal[c].isna().sum()) for c in ["tmax", "tmin", "heat_index"]}

    cal["tavg"] = cal[["tmax", "tmin"]].mean(axis=1)
    cal["prcp_flag"] = (cal["prcp"] > 0).astype(float)
    cal["tmax_lag1"] = cal["tmax"].shift(1)
    hot = (cal["tmax"] >= HOT_DAY_C).astype(float)
    cal["hot_days_prior3"] = hot.shift(1).rolling(3, min_periods=1).sum()
    cal["hot_days_prior7"] = hot.shift(1).rolling(7, min_periods=1).sum()
    cal["tmax_lag1"] = cal["tmax_lag1"].fillna(cal["tmax"])
    cal[["hot_days_prior3", "hot_days_prior7"]] = \
        cal[["hot_days_prior3", "hot_days_prior7"]].fillna(0)

    doy = cal["date"].dt.dayofyear
    cal["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    cal["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    cal["month"] = cal["date"].dt.month
    cal["state"] = LOC_STATE[loc]
    cal["heat_index_filled"] = cal["heat_index"].fillna(cal["tmax"])
    cal["tmax_sq"] = cal["tmax"] ** 2
    cal["tmax_lag1_sq"] = cal["tmax_lag1"] ** 2
    cal["hi_sq"] = cal["heat_index_filled"] ** 2
    cal["tmax_x_hi"] = cal["tmax"] * cal["heat_index_filled"]

    cal["r_screening"] = cal["heat_index_filled"].apply(screening_risk)
    for key, bundle in models.items():
        pipe, cols, ref = bundle["pipeline"], bundle["columns"], bundle["reference_p99"]
        missing = [c for c in cols if c not in cal.columns]
        if missing:
            log.error(f"{loc}: calendar missing model columns {missing}")
            return None, None
        cal[f"r_model_{key}"] = np.clip(pipe.predict_proba(cal[cols])[:, 1] / ref, 0.0, 1.0)

    cal.to_csv(cache, index=False)
    return cal, {"cached": False, "observed_gaps": gaps, "gaps_after_interpolation": remaining,
                 "daily_stations": dstations[:4], "heat_index_stations": histations[:4]}


def main():
    common.banner(log, "STEP 12 - CONTINUOUS RISK CALENDARS ON THREE RISK SCALES")

    models = {}
    for key, fname in [("dayahead", "risk_model_dayahead.joblib"),
                       ("sameday", "risk_model_sameday.joblib")]:
        path = config.MODELS / fname
        if not path.exists():
            log.error(f"{fname} missing - run 11_dayahead_prediction.py first")
            return 1
        models[key] = joblib.load(path)
        log.info(f"loaded {key}: reference p99 = {models[key]['reference_p99']:.4f}")

    rows, provenance = [], {}
    for loc, (lat, lon) in config.SCHEDULE_LOCATIONS.items():
        cal, prov = build(loc, lat, lon, models)
        if cal is None:
            return 1
        provenance[loc] = prov
        summer = cal[cal["date"].dt.month.isin([6, 7, 8])]
        rec = {"Location": loc.replace("_", " "),
               "Calendar days": len(cal),
               "Mean summer maximum temperature (C)": round(float(summer["tmax"].mean()), 1),
               "Mean summer heat index (C)": round(float(summer["heat_index_filled"].mean()), 1)}
        for label, col in [("Day-ahead model", "r_model_dayahead"),
                           ("Same-day model", "r_model_sameday"),
                           ("Heat-index screening", "r_screening")]:
            rec[f"{label}: mean summer risk"] = round(float(summer[col].mean()), 3)
            rec[f"{label}: high-risk days"] = int((cal[col] >= HIGH_RISK_THRESHOLD).sum())
        corr = cal[["r_screening", "r_model_sameday", "r_model_dayahead"]].corr(method="spearman")
        rec["Spearman, screening vs same-day model"] = round(
            float(corr.loc["r_screening", "r_model_sameday"]), 3)
        rec["Spearman, screening vs day-ahead model"] = round(
            float(corr.loc["r_screening", "r_model_dayahead"]), 3)
        rows.append(rec)
        log.info(f"{loc}: {len(cal)} days, summer risk day-ahead="
                 f"{rec['Day-ahead model: mean summer risk']}, "
                 f"screening={rec['Heat-index screening: mean summer risk']}, "
                 f"rho={rec['Spearman, screening vs day-ahead model']}")

    tbl = pd.DataFrame(rows)
    common.save_table(tbl, "table_risk_scale_correspondence")

    verification = {
        "window_start": WINDOW_START.isoformat(),
        "window_end": WINDOW_END.isoformat(),
        "n_days": int(tbl["Calendar days"].iloc[0]),
        "locations": list(config.SCHEDULE_LOCATIONS),
        "hi_lo_c": HI_LO, "hi_hi_c": HI_HI,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "hot_day_threshold_c": HOT_DAY_C,
        "reference_p99_dayahead": models["dayahead"]["reference_p99"],
        "reference_p99_sameday": models["sameday"]["reference_p99"],
        "min_spearman_screening_vs_dayahead": float(
            tbl["Spearman, screening vs day-ahead model"].min()),
        "max_spearman_screening_vs_dayahead": float(
            tbl["Spearman, screening vs day-ahead model"].max()),
        "calendar_provenance": provenance,
    }
    common.save_json(verification, "verification12_calendars")
    log.info("\n" + tbl.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
