"""
14_nbm_forecasts.py — Archived NOAA National Blend of Models (NBM) station forecasts.

Retrieves, for each scheduling location, the station block of the 12 UTC extended text
bulletin (NBE) of the NBM for every planning day, from the public NOAA archive on Amazon Web
Services (bucket noaa-nbm-grib2-pds). Station blocks in the bulletin are sorted by station
identifier, so each block is located with HTTP byte-range requests (a window around the
previous day's position, then a binary search if needed) instead of downloading the whole
bulletin. The parsed guidance gives daily maximum and minimum temperature, dew point and
24-h precipitation by lead day.

Forecast stations are the NOAA stations whose observations build each location's realized
risk calendar (ICAO identifiers): Austin KATT, Orlando KORL, Atlanta KFTY, Phoenix KPHX,
Chicago KMDW.

Resume-safe: one parquet file per bulletin date under data_external/nbm/daily/.

Outputs:
  data_external/nbm/daily/<YYYYMMDD>.parquet
  data_external/nbm/nbm_forecasts.parquet
  outputs/tables/nbm_download_report.json
"""
from __future__ import annotations

import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import requests

import config
import common
import nbm_parse

warnings.filterwarnings("ignore")
log = common.setup_logger("14_nbm_forecasts")

NBM_DIR = config.DATA_EXTERNAL / "nbm"
DAILY_DIR = NBM_DIR / "daily"
DAILY_DIR.mkdir(parents=True, exist_ok=True)
URL = "https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{d}/{h:02d}/text/blend_nbetx.t{h:02d}z"
RUN_HOURS = [12, 13, 11, 14, 10]          # preferred run first, then the nearest alternatives

STATIONS = {"TX_Austin": "KATT", "FL_Orlando": "KORL", "GA_Atlanta": "KFTY",
            "AZ_Phoenix": "KPHX", "IL_Chicago": "KMDW"}

# Planning days: the evening before the common project start (1 July 2023) through the end of
# the longest evaluated project plus the 7-day forecast window.
WINDOW_START = date(2023, 6, 30)
WINDOW_END = date(2024, 11, 20)

HDR = re.compile(rb"(?m)^[ 	]*(\S+)[ 	]+NBM V[\d.]+ NBE GUIDANCE")
SESSION = requests.Session()


def get_range(url: str, a: int, b: int):
    """Bytes a..b (inclusive) of url; returns (content, total_size) or (None, None)."""
    for attempt in range(5):
        try:
            r = SESSION.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=60)
            if r.status_code == 206:
                total = int(r.headers["Content-Range"].split("/")[-1])
                return r.content, total
            if r.status_code in (403, 404, 416):
                return None, None
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return None, None


def first_header(buf: bytes):
    """Station id and position of the first block header in a byte buffer."""
    m = HDR.search(buf)
    return (m.group(1).decode(), m.start()) if m else (None, None)


def extract_block(buf: bytes, start: int) -> str:
    nxt = HDR.search(buf, start + 10)
    end = nxt.start() if nxt else len(buf)
    return buf[start:end].decode("latin-1")


def find_block(url: str, total: int, station: str, guess: int | None):
    """Locate one station block; returns (block_text, offset) or (None, None)."""
    if guess is not None:
        a = max(0, guess - 8000)
        buf, _ = get_range(url, a, min(total - 1, guess + 12000))
        if buf:
            for m in HDR.finditer(buf):
                if m.group(1).decode() == station:
                    blk = extract_block(buf, m.start())
                    if blk.count("\n") > 10:
                        return blk, a + m.start()
    lo, hi = 0, total
    while hi - lo > 6000:
        mid = (lo + hi) // 2
        buf, _ = get_range(url, mid, min(total - 1, mid + 5000))
        if not buf:
            return None, None
        sid, _pos = first_header(buf)
        if sid is None or sid > station:
            hi = mid
        elif sid < station:
            lo = mid
        else:
            lo = max(0, mid - 100)
            break
    buf, _ = get_range(url, lo, min(total - 1, lo + 16000))
    if not buf:
        return None, None
    for m in HDR.finditer(buf):
        if m.group(1).decode() == station:
            return extract_block(buf, m.start()), lo + m.start()
    return None, None


def process_day(d: date, guesses: dict) -> dict:
    out = DAILY_DIR / f"{d.strftime('%Y%m%d')}.parquet"
    if out.exists():
        return {"date": d.isoformat(), "status": "cached"}
    for h in RUN_HOURS:
        url = URL.format(d=d.strftime("%Y%m%d"), h=h)
        head, total = get_range(url, 0, 200)
        if head is None:
            continue
        recs, offs = [], {}
        for loc, st in STATIONS.items():
            blk, off = find_block(url, total, st, guesses.get(st))
            if blk is None:
                continue
            offs[st] = off
            parsed = nbm_parse.parse_block(blk.splitlines())
            if parsed:
                for rec in parsed["days"]:
                    rec["location"] = loc
                    recs.append(rec)
        if recs:
            df = pd.DataFrame(recs)
            df["issue_date"] = d
            df["run_hour"] = h
            df.to_parquet(out, index=False)
            return {"date": d.isoformat(), "status": "ok", "run_hour": h,
                    "stations": int(df["station"].nunique()), "offsets": offs}
    return {"date": d.isoformat(), "status": "missing"}


def main():
    common.banner(log, "STEP 14 - ARCHIVED NBM STATION FORECASTS (5 SCHEDULING STATIONS)")
    days = [WINDOW_START + timedelta(days=i) for i in range((WINDOW_END - WINDOW_START).days + 1)]
    todo = [d for d in days if not (DAILY_DIR / f"{d.strftime('%Y%m%d')}.parquet").exists()]
    log.info(f"{len(days)} planning days; {len(todo)} bulletins to retrieve")

    # Seed offsets from the first bulletin, then reuse them as guesses (blocks move little).
    guesses = {}
    if todo:
        first = process_day(todo[0], {})
        guesses = first.get("offsets", {})
        log.info(f"first bulletin {first['date']}: {first['status']}, offsets {guesses}")
        todo = todo[1:]
    status = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(process_day, d, guesses): d for d in todo}
        for k, f in enumerate(as_completed(futs), 1):
            status.append(f.result())
            if k % 50 == 0:
                log.info(f"  {k}/{len(todo)} bulletins processed")

    files = sorted(DAILY_DIR.glob("*.parquet"))
    allf = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    allf.to_parquet(NBM_DIR / "nbm_forecasts.parquet", index=False)
    have = {f.stem for f in files}
    missing = [d.isoformat() for d in days if d.strftime("%Y%m%d") not in have]
    per_day = allf.groupby("issue_date")["station"].nunique()
    report = {
        "window": [WINDOW_START.isoformat(), WINDOW_END.isoformat()],
        "n_planning_days": len(days), "n_bulletins": len(files),
        "missing_dates": missing,
        "bulletins_missing_a_station": int((per_day < len(STATIONS)).sum()),
        "n_runs_not_12utc": int((allf.groupby("issue_date")["run_hour"].first() != 12).sum()),
        "stations": STATIONS,
    }
    common.save_json(report, "nbm_download_report")
    log.info(f"report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
