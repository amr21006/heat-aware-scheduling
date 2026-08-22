"""
03_match_noaa_weather.py — Match NOAA weather to OSHA cases + build case-crossover set.

For each geo-valid case:
  * generate time-stratified referent (control) days (same year+month+day-of-week);
  * build a continuous daily series over [month_start - buffer, month_end] using the
    NEAREST NOAA GHCN-Daily station with data for each date (TMAX/TMIN/PRCP);
  * compute daily-max NWS heat index from NOAA ISD-Lite hourly (best effort);
  * if no NOAA station within MAX_SEARCH_DISTANCE_KM has data, use ERA5 (flagged);
  * emit one row per case/referent day with weather + heat-history lag features.

Resume-safe: per-case results are cached to data_external/weather_cache/<tag>/.

Usage:
  python 03_match_noaa_weather.py            # full run (heat + negative control)
  python 03_match_noaa_weather.py --limit 8  # quick test on 8 heat cases
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config
import common

warnings.filterwarnings("ignore")
log = common.setup_logger("03_match_noaa_weather")

HOT_DAY_C = 32.2  # 90 F threshold for "hot day" heat-history features


def analytic_window(event_dt):
    y, m = event_dt.year, event_dt.month
    month_start = datetime(y, m, 1)
    month_end = (datetime(y + 1, 1, 1) if m == 12 else datetime(y, m + 1, 1)) - timedelta(days=1)
    win_start = month_start - timedelta(days=config.HEAT_HISTORY_BUFFER_DAYS)
    return month_start, month_end, win_start


def build_loc_daily(near, lat, lon, win_start, win_end):
    """Continuous daily series from nearest NOAA station-with-data per date, using the
    cached station-year layer. `near` is a precomputed nearest-stations frame.
    Returns (df_daily, station_meta)."""
    if near is None or near.empty:
        return None, {}
    near = near.sort_values("distance_km")
    dates = pd.date_range(win_start, win_end, freq="D")
    years = sorted({d.year for d in dates})
    result = {}
    for sid, srow in near.iterrows():                       # nearest first
        dist = float(srow["distance_km"])
        if dist > config.MAX_SEARCH_DISTANCE_KM:
            break
        if len(result) == len(dates):
            break
        parts = [common.station_year_daily(sid, y) for y in years]
        parts = [p for p in parts if p is not None and len(p)]
        if not parts:
            continue
        sd = pd.concat(parts)
        sd = sd[~sd.index.duplicated(keep="first")].sort_index()
        for d in dates:
            if d in result or d not in sd.index:
                continue
            tmax = sd.at[d, "tmax"] if "tmax" in sd.columns else np.nan
            if pd.notna(tmax):
                result[d] = (sid, dist, float(tmax),
                             float(sd.at[d, "tmin"]) if "tmin" in sd.columns and pd.notna(sd.at[d, "tmin"]) else np.nan,
                             float(sd.at[d, "prcp"]) if "prcp" in sd.columns and pd.notna(sd.at[d, "prcp"]) else np.nan,
                             "NOAA")
    # ERA5 fallback (flagged) for any dates with no NOAA station coverage
    missing = [d for d in dates if d not in result]
    if missing:
        era = common.fetch_era5_daily(lat, lon, win_start.date(), win_end.date())
        if era is not None:
            for d in missing:
                if d in era.index and pd.notna(era.at[d, "tmax"]):
                    result[d] = ("ERA5_GRID", np.nan, float(era.at[d, "tmax"]),
                                 float(era.at[d, "tmin"]) if pd.notna(era.at[d, "tmin"]) else np.nan,
                                 float(era.at[d, "prcp"]) if pd.notna(era.at[d, "prcp"]) else np.nan,
                                 "ERA5_GRID")
    if not result:
        return pd.DataFrame(columns=["station_id", "distance_km", "tmax", "tmin", "prcp", "source"]), {}
    df = pd.DataFrame(
        [(d, *result[d]) for d in sorted(result)],
        columns=["date", "station_id", "distance_km", "tmax", "tmin", "prcp", "source"]
    ).set_index("date")
    return df, {"n_stations": int(len(near)), "nearest_km": float(near["distance_km"].min())}


def build_heat_index(near, dates_needed):
    """Daily-max NWS heat index (C) from NOAA ISD-Lite hourly, nearest station w/ data.
    Returns dict {date -> hi_c} (best effort; may be empty)."""
    if near is None or near.empty:
        return {}, np.nan
    span_start = datetime(min(d.year for d in dates_needed), min(d.month for d in dates_needed), 1)
    span_end = max(dates_needed) + timedelta(days=1)
    for sid, row in near.sort_values("distance_km").iterrows():
        if row["distance_km"] > config.MAX_SEARCH_DISTANCE_KM:
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
        out = {d: float(daily_max.get(pd.Timestamp(d), np.nan)) for d in dates_needed}
        if any(pd.notna(v) for v in out.values()):
            return out, float(row["distance_km"])
    return {}, np.nan


def match_cases(cases: pd.DataFrame, tag: str, limit=None) -> pd.DataFrame:
    cache_dir = config.WEATHER_CACHE / tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    cases = cases.copy()
    cases["lat"] = pd.to_numeric(cases["Latitude"], errors="coerce")
    cases["lon"] = pd.to_numeric(cases["Longitude"], errors="coerce")
    cases["evt"] = common.parse_dates(cases["EventDate"])
    cases = cases[(cases["lat"].between(18, 72)) & (cases["lon"].between(-180, -65))
                  & cases["evt"].notna()].reset_index(drop=True)
    if limit:
        cases = cases.head(limit)
    log.info(f"[{tag}] matching {len(cases)} geo/date-valid cases ...")

    all_rows = []
    t0 = time.time()
    for i, c in cases.iterrows():
        cid = str(c["ID"])
        cache_f = cache_dir / f"{cid}.parquet"
        if cache_f.exists():
            all_rows.append(pd.read_parquet(cache_f))
            continue
        evt = c["evt"].to_pydatetime()
        case_date = pd.Timestamp(evt.date())
        referents = [pd.Timestamp(d) for d in common.referent_days(evt.date(), config.CASECROSS_MATCH_DOW)]
        analytic = [case_date] + referents
        _, month_end, win_start = analytic_window(evt)
        win_end = month_end

        near = common.nearest_stations(c["lat"], c["lon"], config.N_NEAREST_STATIONS)
        loc_daily, meta = build_loc_daily(near, c["lat"], c["lon"], win_start, win_end)
        if loc_daily is None or loc_daily.empty:
            log.warning(f"[{tag}] {cid}: no daily weather; skipped")
            continue
        hi_map, hi_dist = build_heat_index(near, analytic)

        rows = []
        for d in analytic:
            if d not in loc_daily.index:
                continue
            rec = loc_daily.loc[d]
            # heat-history lags from continuous series
            prior3 = loc_daily.loc[(loc_daily.index >= d - timedelta(days=3)) &
                                   (loc_daily.index <= d - timedelta(days=1)), "tmax"]
            prior7 = loc_daily.loc[(loc_daily.index >= d - timedelta(days=7)) &
                                   (loc_daily.index <= d - timedelta(days=1)), "tmax"]
            lag1 = loc_daily.at[d - timedelta(days=1), "tmax"] if (d - timedelta(days=1)) in loc_daily.index else np.nan
            rows.append({
                "case_id": cid, "stratum": cid,
                "date": d, "is_case": int(d == case_date),
                "lat": c["lat"], "lon": c["lon"],
                "state": str(c.get("State", "")).title(),
                "naics3": str(c.get("naics3", "") or str(c.get("Primary NAICS", ""))[:3]),
                "year": d.year, "month": d.month, "doy": d.dayofyear, "dow": d.dayofweek,
                "tmax": rec["tmax"], "tmin": rec["tmin"], "prcp": rec["prcp"],
                "tavg": np.nanmean([rec["tmax"], rec["tmin"]]),
                "heat_index": hi_map.get(d, np.nan),
                "station_id": rec["station_id"], "distance_km": rec["distance_km"],
                "hi_station_km": hi_dist, "source": rec["source"],
                "tmax_lag1": lag1,
                "hot_days_prior3": int((prior3 >= HOT_DAY_C).sum()) if len(prior3) else np.nan,
                "hot_days_prior7": int((prior7 >= HOT_DAY_C).sum()) if len(prior7) else np.nan,
                "prcp_flag": int(rec["prcp"] > 0.0) if pd.notna(rec["prcp"]) else np.nan,
            })
        if not rows:
            continue
        cdf = pd.DataFrame(rows)
        cdf.to_parquet(cache_f, index=False)
        all_rows.append(cdf)

        if (i + 1) % 25 == 0:
            rate = (i + 1) / (time.time() - t0)
            log.info(f"[{tag}] {i+1}/{len(cases)} done ({rate:.2f}/s)")

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    log.info(f"[{tag}] built {len(out)} case/referent-day rows for "
             f"{out['case_id'].nunique() if len(out) else 0} cases in {time.time()-t0:.0f}s")
    return out


def weather_report(df: pd.DataFrame, tag: str):
    """Validation 2: station distances + missingness."""
    case_days = df[df["is_case"] == 1]
    rep = {}
    rep["n_case_days"] = int(len(case_days))
    rep["n_total_rows"] = int(len(df))
    d = pd.to_numeric(case_days["distance_km"], errors="coerce").dropna()
    rep["dist_median_km"] = float(d.median()) if len(d) else None
    rep["dist_p90_km"] = float(d.quantile(0.9)) if len(d) else None
    rep["dist_max_km"] = float(d.max()) if len(d) else None
    rep["within_25km_pct"] = float((d <= 25).mean() * 100) if len(d) else None
    rep["within_50km_pct"] = float((d <= 50).mean() * 100) if len(d) else None
    rep["within_75km_pct"] = float((d <= 75).mean() * 100) if len(d) else None
    rep["within_100km_pct"] = float((d <= 100).mean() * 100) if len(d) else None
    rep["pct_ERA5_fallback_casedays"] = float((case_days["source"] == "ERA5_GRID").mean() * 100)
    rep["pct_heatindex_available_casedays"] = float(case_days["heat_index"].notna().mean() * 100)
    rep["pct_tmax_missing_casedays"] = float(case_days["tmax"].isna().mean() * 100)
    rep["controls_per_case_mean"] = float(df.groupby("case_id")["is_case"].apply(lambda s: (s == 0).sum()).mean())
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--negcontrol-n", type=int, default=600)
    ap.add_argument("--skip-negcontrol", action="store_true")
    args = ap.parse_args()

    common.banner(log, "STEP 03 — NOAA WEATHER MATCHING & CASE-CROSSOVER BUILD")

    heat = pd.read_csv(config.DATA_INTERMEDIATE / "environmental_heat_cases.csv",
                       dtype=str, keep_default_na=False)
    heat_match = match_cases(heat, "heat", limit=args.limit)
    if len(heat_match):
        heat_match.to_csv(config.DATA_INTERMEDIATE / "weather_matches.csv", index=False)
        rep = weather_report(heat_match, "heat")
        common.save_json(rep, "verification02_weather_matching")
        table3 = pd.DataFrame([
            ["Case-days matched", rep["n_case_days"]],
            ["Median nearest-station distance (km)", round(rep["dist_median_km"], 1)],
            ["90th pct station distance (km)", round(rep["dist_p90_km"], 1)],
            ["Max station distance (km)", round(rep["dist_max_km"], 1)],
            ["Case-days within 25 km (%)", round(rep["within_25km_pct"], 1)],
            ["Case-days within 50 km (%)", round(rep["within_50km_pct"], 1)],
            ["Case-days within 75 km (%)", round(rep["within_75km_pct"], 1)],
            ["Case-days within 100 km (%)", round(rep["within_100km_pct"], 1)],
            ["Case-days using ERA5 gridded fallback (%)", round(rep["pct_ERA5_fallback_casedays"], 1)],
            ["Case-days with heat index available (%)", round(rep["pct_heatindex_available_casedays"], 1)],
            ["Mean referent days per case", round(rep["controls_per_case_mean"], 2)],
        ], columns=["Metric", "Value"])
        common.save_table(table3, "table03_weather_matching")
        log.info(f"Weather report: {rep}")

    if not args.skip_negcontrol and args.limit is None:
        amput = pd.read_csv(config.DATA_INTERMEDIATE / "nonheat_amputation_cases.csv",
                            dtype=str, keep_default_na=False)
        rng = np.random.default_rng(config.RANDOM_SEED)
        amput_v = amput[pd.to_numeric(amput["Latitude"], errors="coerce").notna()]
        take = min(args.negcontrol_n, len(amput_v))
        idx = rng.choice(amput_v.index.to_numpy(), size=take, replace=False)
        neg = amput_v.loc[idx].reset_index(drop=True)
        neg_match = match_cases(neg, "negcontrol")
        if len(neg_match):
            neg_match.to_csv(config.DATA_INTERMEDIATE / "weather_matches_negcontrol.csv", index=False)
            common.save_json(weather_report(neg_match, "negcontrol"),
                             "verification02b_weather_negcontrol")
    return 0


if __name__ == "__main__":
    sys.exit(main())
