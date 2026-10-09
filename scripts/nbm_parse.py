"""
nbm_parse.py — Parser for NOAA National Blend of Models (NBM) extended text bulletins (NBE).

Each station block in an NBE bulletin lists 12-hourly guidance out to about 8 days.
This module extracts, for one station block, the daily maximum temperature, the morning
minimum temperature, the dew point near the time of the maximum, and the 24-h precipitation
amount, each attributed to the local calendar day it describes.

Column conventions of the NBE bulletin (NBM v4.x):
  * TXN at a 00 UTC column is the daytime maximum of the preceding local day (US time zones);
    TXN at a 12 UTC column is the overnight minimum ending that morning.
  * DPT is the dew point at the column time; the 00 UTC value (late afternoon/evening local)
    is used with the maximum temperature to estimate the daily maximum heat index.
  * Q24 at a 12 UTC column is the 24-h precipitation amount ending at that time (hundredths of
    an inch), attributed to the preceding local day.
Values are in degrees Fahrenheit and are converted to Celsius.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

HEADER_RE = re.compile(r"^\s*(\S+)\s+NBM V[\d.]+ NBE GUIDANCE\s+(\d+)/(\d+)/(\d{4})\s+(\d{4}) UTC")


def f_to_c(f):
    return (f - 32.0) * 5.0 / 9.0


def _row_values(line: str, ends: list[int]) -> list[float | None]:
    """Read the value whose right edge sits at each FHR column end (fixed-width layout)."""
    out = []
    for e in ends:
        seg = line[max(0, e - 4):e].replace("|", " ").strip()
        if seg == "" or seg.startswith("-99"):
            out.append(None)
            continue
        try:
            out.append(float(seg.split()[-1]))
        except ValueError:
            out.append(None)
    return out


def parse_block(lines: list[str]) -> dict | None:
    """Parse one station block (header line first). Returns per-local-day records."""
    while lines and not lines[0].strip():
        lines = lines[1:]
    if not lines:
        return None
    m = HEADER_RE.match(lines[0])
    if not m:
        return None
    station = m.group(1)
    run = datetime(int(m.group(4)), int(m.group(2)), int(m.group(3)),
                   int(m.group(5)[:2]), int(m.group(5)[2:]))
    rows = {}
    fhr_line = None
    for ln in lines[1:]:
        tag = ln[:5].strip()
        if tag == "FHR":
            fhr_line = ln
        elif tag:
            rows[tag] = ln
    if fhr_line is None:
        return None
    ends = [mm.end() for mm in re.finditer(r"\d+", fhr_line[5:])]
    ends = [e + 5 for e in ends]
    fhrs = [int(fhr_line[e - 3:e].strip()) for e in ends]
    get = {k: _row_values(v, ends) for k, v in rows.items() if k in ("TXN", "DPT", "Q24", "TMP")}
    days = {}
    for i, h in enumerate(fhrs):
        valid = run + timedelta(hours=h)
        txn = get.get("TXN", [None] * len(fhrs))[i]
        dpt = get.get("DPT", [None] * len(fhrs))[i]
        q24 = get.get("Q24", [None] * len(fhrs))[i]
        if valid.hour == 0:
            d = (valid - timedelta(days=1)).date()
            rec = days.setdefault(d, {})
            if txn is not None:
                rec["tmax_f"] = txn
            if dpt is not None:
                rec["dpt_f"] = dpt
        elif valid.hour == 12:
            d_min = valid.date()
            rec = days.setdefault(d_min, {})
            if txn is not None:
                rec["tmin_f"] = txn
            d_q = (valid - timedelta(days=1)).date()
            if q24 is not None:
                days.setdefault(d_q, {})["q24"] = q24
    out = []
    for d, rec in sorted(days.items()):
        lead = (d - run.date()).days
        out.append({"station": station, "run": run, "date": d, "lead_days": lead,
                    "tmax_c": f_to_c(rec["tmax_f"]) if "tmax_f" in rec else None,
                    "tmin_c": f_to_c(rec["tmin_f"]) if "tmin_f" in rec else None,
                    "dpt_c": f_to_c(rec["dpt_f"]) if "dpt_f" in rec else None,
                    "q24_in": rec["q24"] / 100.0 if "q24" in rec else None})
    return {"station": station, "run": run, "days": out}


def parse_bulletin(text: str, stations: set[str]) -> list[dict]:
    """Return parsed blocks for the requested stations only."""
    lines = text.splitlines()
    found = []
    i, n = 0, len(lines)
    while i < n:
        m = HEADER_RE.match(lines[i])
        if m and m.group(1) in stations:
            j = i + 1
            while j < n and not HEADER_RE.match(lines[j]) and lines[j].strip() not in ("1",):
                j += 1
            blk = parse_block(lines[i:j])
            if blk:
                found.append(blk)
            i = j
        else:
            i += 1
    return found
