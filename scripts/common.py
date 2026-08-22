"""
common.py — Shared utilities: logging, geodesy, NWS heat index, date parsing,
NOAA/meteostat weather fetchers, Open-Meteo gridded fallback, and I/O helpers.

No analytic value is fabricated. Weather values come from NOAA (GHCN-Daily,
ISD-Lite via meteostat) or, only as a flagged fallback, ECMWF ERA5 reanalysis
via the Open-Meteo archive API.
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests

import config

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(config.LOGS / f"{name}.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


# ----------------------------------------------------------------------------
# Geodesy
# ----------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ----------------------------------------------------------------------------
# Temperature conversions & NWS heat index
# ----------------------------------------------------------------------------
def c_to_f(c):
    return c * 9.0 / 5.0 + 32.0


def f_to_c(f):
    return (f - 32.0) * 5.0 / 9.0


def heat_index_f(T: float, RH: float) -> float:
    """NWS/NOAA heat index (Rothfusz regression) in degrees Fahrenheit.

    Implements the algorithm published at
    https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml :
    compute the simple formula, average with temperature; if >= 80 F apply the
    full regression with the low-RH and high-RH adjustments.
    """
    if T is None or RH is None or (isinstance(T, float) and math.isnan(T)) \
            or (isinstance(RH, float) and math.isnan(RH)):
        return float("nan")
    hi_simple = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
    hi = (hi_simple + T) / 2.0
    if hi >= 80.0:
        hi = (-42.379 + 2.04901523 * T + 10.14333127 * RH
              - 0.22475541 * T * RH - 0.00683783 * T * T - 0.05481717 * RH * RH
              + 0.00122874 * T * T * RH + 0.00085282 * T * RH * RH
              - 0.00000199 * T * T * RH * RH)
        if (RH < 13.0) and (80.0 <= T <= 112.0):
            hi -= ((13.0 - RH) / 4.0) * math.sqrt((17.0 - abs(T - 95.0)) / 17.0)
        elif (RH > 85.0) and (80.0 <= T <= 87.0):
            hi += ((RH - 85.0) / 10.0) * ((87.0 - T) / 5.0)
    return hi


def heat_index_c(T_c, RH):
    """Heat index in Celsius from Celsius temperature and % RH (scalar)."""
    return f_to_c(heat_index_f(c_to_f(T_c), RH))


def heat_index_series_c(T_c: pd.Series, RH: pd.Series) -> pd.Series:
    """Vectorised heat index in Celsius for pandas Series."""
    out = []
    for t, r in zip(T_c.to_numpy(dtype="float64"), RH.to_numpy(dtype="float64")):
        out.append(heat_index_c(t, r))
    return pd.Series(out, index=T_c.index, dtype="float64")


# ----------------------------------------------------------------------------
# Date parsing
# ----------------------------------------------------------------------------
def parse_dates(series: pd.Series) -> pd.Series:
    """Parse OSHA EventDate (M/D/YYYY) robustly."""
    return pd.to_datetime(series, errors="coerce", format="mixed")


# ----------------------------------------------------------------------------
# Time-stratified case-crossover referent days
# ----------------------------------------------------------------------------
def referent_days(event_dt: date, match_dow: bool = True) -> list[date]:
    """Return control (referent) days: same year+month, same day-of-week, != event."""
    y, m = event_dt.year, event_dt.month
    if m == 12:
        nxt = date(y + 1, 1, 1)
    else:
        nxt = date(y, m + 1, 1)
    d = date(y, m, 1)
    days = []
    while d < nxt:
        if d != event_dt and (not match_dow or d.weekday() == event_dt.weekday()):
            days.append(d)
        d += timedelta(days=1)
    return days


# ----------------------------------------------------------------------------
# NOAA weather via meteostat
# ----------------------------------------------------------------------------
_MS = None


def _meteostat():
    global _MS
    if _MS is None:
        import meteostat
        # allow probing >10 nearby stations per location for nearest-with-data fallback
        try:
            meteostat.config.block_large_requests = False
        except Exception:
            pass
        _MS = meteostat
    return _MS


def nearest_stations(lat: float, lon: float, limit: int) -> pd.DataFrame:
    """Nearest stations with distance in km (column 'distance_km')."""
    ms = _meteostat()
    pt = ms.Point(lat, lon)
    df = ms.stations.nearby(pt, limit=limit)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    df["distance_km"] = df["distance"] / 1000.0
    return df


def fetch_ghcnd_daily(station_ids: list[str], start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """NOAA daily TMAX/TMIN/PRCP for a list of stations.

    Uses GHCN-Daily where available and falls back to daily values DERIVED from
    NOAA ISD-Lite hourly (provider DAILY_DERIVED) at airport/ASOS stations that
    lack a GHCN-Daily record. Both providers are NOAA-sourced, so this stays
    within the official-NOAA framing while greatly improving date coverage.
    Returns long df indexed (station, time) or None.
    """
    ms = _meteostat()
    P = ms.Parameter
    ts = ms.daily(station_ids, start, end,
                  parameters=[P.TMAX, P.TMIN, P.PRCP],
                  providers=[ms.Provider.GHCND, ms.Provider.DAILY_DERIVED])
    df = ts.fetch()
    return df


def station_year_daily(station_id: str, year: int) -> Optional[pd.DataFrame]:
    """NOAA daily series for ONE station for ONE calendar year, cached to parquet.

    This is the core efficiency layer for matching the full construction set:
    any two OSHA records near the same station+year share one cached fetch.
    Returns a time-indexed df (tmax,tmin,prcp) or None. A successful-but-empty
    fetch is cached as a marker; transient errors are NOT cached.
    """
    cdir = config.WEATHER_CACHE / "daily_station_year"
    cdir.mkdir(parents=True, exist_ok=True)
    f = cdir / f"{station_id}_{year}.parquet"
    if f.exists():
        try:
            df = pd.read_parquet(f)
            return df if len(df) else None
        except Exception:
            pass
    ms = _meteostat()
    P = ms.Parameter
    try:
        ts = ms.daily([station_id], datetime(year, 1, 1), datetime(year, 12, 31),
                      parameters=[P.TMAX, P.TMIN, P.PRCP],
                      providers=[ms.Provider.GHCND, ms.Provider.DAILY_DERIVED])
        df = ts.fetch()
    except Exception:
        return None  # transient — do not cache, allow retry
    if df is None or len(df) == 0:
        pd.DataFrame(columns=["tmin", "tmax", "prcp"]).to_parquet(f)
        return None
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    try:
        df.to_parquet(f)
    except Exception:
        pass
    return df


def fetch_isd_hourly(station_id: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """NOAA ISD-Lite hourly TEMP + RHUM for one station. Returns df indexed by time or None."""
    ms = _meteostat()
    P = ms.Parameter
    ts = ms.hourly(station_id, start, end,
                   parameters=[P.TEMP, P.RHUM],
                   providers=[ms.Provider.ISD_LITE])
    df = ts.fetch()
    return df


# ----------------------------------------------------------------------------
# Open-Meteo ERA5 gridded fallback (flagged, non-NOAA)
# ----------------------------------------------------------------------------
_OM_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_era5_daily(lat: float, lon: float, start: date, end: date,
                     max_retries: int = 4) -> Optional[pd.DataFrame]:
    """ERA5 daily tmax/tmin/tmean/precip via Open-Meteo. Fallback only; flagged."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum",
        "timezone": "auto",
    }
    for attempt in range(max_retries):
        try:
            r = requests.get(_OM_URL, params=params, timeout=40)
            if r.status_code == 200:
                j = r.json().get("daily", {})
                if not j or "time" not in j:
                    return None
                df = pd.DataFrame({
                    "time": pd.to_datetime(j["time"]),
                    "tmax": j.get("temperature_2m_max"),
                    "tmin": j.get("temperature_2m_min"),
                    "tavg": j.get("temperature_2m_mean"),
                    "prcp": j.get("precipitation_sum"),
                }).set_index("time")
                return df
            elif r.status_code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1))
            else:
                return None
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return None


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def save_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    """Write a results table as CSV and a GitHub-flavoured Markdown copy."""
    csv_path = config.TABLES / f"{name}.csv"
    df.to_csv(csv_path, index=index)
    try:
        md = df.to_markdown(index=index)
        (config.TABLES / f"{name}.md").write_text(md, encoding="utf-8")
    except Exception:
        pass


def save_json(obj, name: str) -> None:
    path = config.TABLES / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def banner(logger: logging.Logger, text: str) -> None:
    logger.info("=" * 70)
    logger.info(text)
    logger.info("=" * 70)


# ----------------------------------------------------------------------------
# Self-tests (Verification 2 & 3): heat index and haversine
# ----------------------------------------------------------------------------
def _self_test():
    # Haversine: Austin -> Houston ~ 235 km
    d = haversine_km(30.27, -97.74, 29.76, -95.37)
    assert 220 < d < 250, f"haversine Austin-Houston off: {d}"
    # Haversine zero
    assert haversine_km(40, -100, 40, -100) < 1e-6
    # Heat index NWS reference points
    hi = heat_index_f(90, 70)   # NWS chart ~ 106
    assert 104 <= hi <= 108, f"HI(90F,70%)={hi}"
    hi = heat_index_f(80, 40)   # NWS chart ~ 80
    assert 78 <= hi <= 82, f"HI(80F,40%)={hi}"
    hi = heat_index_f(100, 40)  # NWS chart ~ 109
    assert 106 <= hi <= 112, f"HI(100F,40%)={hi}"
    hi = heat_index_f(110, 40)  # NWS chart ~ 136
    assert 130 <= hi <= 140, f"HI(110F,40%)={hi}"
    # Low-RH adjustment should reduce HI
    assert heat_index_f(100, 10) < heat_index_f(100, 25)
    # Celsius round trip
    assert abs(heat_index_c(f_to_c(90), 70) - f_to_c(heat_index_f(90, 70))) < 1e-6
    return True


if __name__ == "__main__":
    ok = _self_test()
    print("common.py self-tests passed:", ok)
