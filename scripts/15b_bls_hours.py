"""
15b_bls_hours.py — Average weekly hours by construction subsector (worker-hours sensitivity).

The project networks record crews but not working hours. The BLS Current Employment Statistics
publish average weekly hours of all employees by three-digit construction subsector; their
2015-2024 means (not seasonally adjusted, monthly values) give each crew a daily-hours weight
(weekly hours / 5) for a worker-hours version of the primary outcome.

Series (all employees, average weekly hours, not seasonally adjusted):
  CEU2023600002  NAICS 236  Construction of buildings
  CEU2023700002  NAICS 237  Heavy and civil engineering construction
  CEU2023800002  NAICS 238  Specialty trade contractors

Outputs:
  data_external/bls_ces/ces_hours_2015_2024.json   (raw monthly values, cached)
  outputs/tables/table_ces_hours.{csv,md}
"""
from __future__ import annotations

import json
import sys
import urllib.request

import pandas as pd

import config
import common

log = common.setup_logger("15b_bls_hours")
SERIES = {"236": "CEU2023600002", "237": "CEU2023700002", "238": "CEU2023800002"}
NAMES = {"236": "Construction of buildings", "237": "Heavy and civil engineering construction",
         "238": "Specialty trade contractors"}
CACHE = config.DATA_EXTERNAL / "bls_ces" / "ces_hours_2015_2024.json"
API = "https://api.bls.gov/publicAPI/v1/timeseries/data/"


def fetch():
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    body = json.dumps({"seriesid": list(SERIES.values()), "startyear": "2015", "endyear": "2024"}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read().decode())
    assert d.get("status") == "REQUEST_SUCCEEDED", d.get("message")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return d


def main():
    common.banner(log, "STEP 15b - AVERAGE WEEKLY HOURS BY CONSTRUCTION SUBSECTOR (BLS CES)")
    d = fetch()
    rows = []
    for s in d["Results"]["series"]:
        naics = next(k for k, v in SERIES.items() if v == s["seriesID"])
        vals = [float(x["value"]) for x in s["data"] if x["period"].startswith("M") and x["period"] != "M13"]
        assert len(vals) == 120, (s["seriesID"], len(vals))
        rows.append({"NAICS": naics, "Subsector": NAMES[naics], "Series": s["seriesID"],
                     "Months": len(vals), "Mean weekly hours, 2015-2024": round(sum(vals) / len(vals), 2),
                     "Daily hours (weekly / 5)": round(sum(vals) / len(vals) / 5, 3)})
    tbl = pd.DataFrame(rows).sort_values("NAICS")
    common.save_table(tbl, "table_ces_hours")
    log.info(tbl.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
