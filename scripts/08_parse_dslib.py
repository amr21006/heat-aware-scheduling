"""
09_parse_dslib.py - Parse empirical construction project networks from DSLIB v3.2.

Source: Dynamic Scheduling Library (DSLIB) v3.2, the empirical project database of
the Operations Research & Scheduling group, Ghent University, originally described
in Batselier and Vanhoucke (2015). Every project in the database is an as-planned
baseline schedule recorded from a real project, not a generated instance.

This step converts the construction-sector projects into the same
resource-constrained project scheduling problem (RCPSP) structure already used for
the PSPLIB benchmark instances, so both evaluation settings run through one solver:

  activities  : id, name, integer duration in working days, renewable demands
  precedence  : finish-start relations with integer lags in working days
  resources   : renewable resource names and integer availabilities

Nothing is imputed. A project is rejected, with the reason recorded, whenever a
required field cannot be read from the source workbook.

Outputs:
  data_intermediate/dslib_projects/<code>.json      one normalised network per project
  outputs/tables/table_dslib_inventory.{csv,md}     parsed inventory with rejections
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl

import config
import common

warnings.filterwarnings("ignore")
log = common.setup_logger("09_parse_dslib")

DSLIB_DIR = config.DATA_EXTERNAL / "dslib"
EXCEL_DIR = DSLIB_DIR / "Excel"
INDEX_XLSX = DSLIB_DIR / "DSLIB_Analysis_Scheet.xlsx"
OUT_DIR = config.DATA_INTERMEDIATE / "dslib_projects"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOURS_PER_DAY = 8.0
MIN_ACTIVITIES = 12          # below this a network carries no sequencing freedom worth testing
MAX_ACTIVITIES = 250         # above this the per-scenario solve time is not affordable


# ---------------------------------------------------------------------------
# Field parsers
# ---------------------------------------------------------------------------
_DUR_TOKEN = re.compile(r"(\d+(?:[.,]\d+)?)\s*([dhwm])", re.I)


def parse_duration_days(text) -> float | None:
    """'5d' -> 5.0 ; '4h' -> 0.5 ; '1d 2h' -> 1.25 ; '2w' -> 10.0. None if unreadable."""
    if text is None:
        return None
    s = str(text).strip().lower()
    if s in ("", "none", "nan"):
        return None
    if re.fullmatch(r"\d+(\.\d+)?", s):          # bare number = days
        return float(s)
    total, found = 0.0, False
    for value, unit in _DUR_TOKEN.findall(s):
        v = float(value.replace(",", "."))
        found = True
        if unit == "d":
            total += v
        elif unit == "h":
            total += v / HOURS_PER_DAY
        elif unit == "w":
            total += v * 5.0                      # working week
        elif unit == "m":
            total += v * 20.0                     # working month
    return total if found else None


_PRED_TOKEN = re.compile(
    r"^\s*(\d+)\s*(FS|SS|FF|SF)?\s*([+-]\s*[\dwdhm.,\s]+)?\s*$", re.I)


def parse_predecessors(text):
    """'10FS;133FS-2w 4d' -> [(10,'FS',0.0), (133,'FS',-14.0)]. Unreadable tokens dropped."""
    if text is None:
        return [], 0
    s = str(text).strip()
    if s in ("", "None", "nan"):
        return [], 0
    out, dropped = [], 0
    for tok in re.split(r"[;,]", s):
        tok = tok.strip()
        if not tok:
            continue
        m = _PRED_TOKEN.match(tok)
        if not m:
            dropped += 1
            continue
        pid = int(m.group(1))
        rel = (m.group(2) or "FS").upper()
        lag = 0.0
        if m.group(3):
            sign = -1.0 if m.group(3).strip().startswith("-") else 1.0
            lag_days = parse_duration_days(m.group(3).strip().lstrip("+-").strip())
            lag = sign * (lag_days if lag_days is not None else 0.0)
        out.append((pid, rel, lag))
    return out, dropped


_RES_TOKEN = re.compile(r"^\s*(.+?)\s*(?:\[\s*([\d.,]+)\s*(?:#\s*([\d.,]+)\s*)?\])?\s*$")


def parse_resource_demand(text):
    """'Handlanger[2.00 #20]' -> {'Handlanger': 2.0} ; 'Bronbemaling' -> {'Bronbemaling': 1.0}."""
    if text is None:
        return {}
    s = str(text).strip()
    if s in ("", "None", "nan"):
        return {}
    demands = {}
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        m = _RES_TOKEN.match(tok)
        if not m:
            continue
        name = m.group(1).strip()
        units = float(m.group(2).replace(",", ".")) if m.group(2) else 1.0
        if name:
            demands[name] = demands.get(name, 0.0) + units
    return demands


def parse_availability(text) -> float | None:
    if text is None:
        return None
    s = str(text).strip()
    m = re.search(r"[\d.,]+", s)
    return float(m.group(0).replace(",", ".")) if m else None


# ---------------------------------------------------------------------------
# Project parser
# ---------------------------------------------------------------------------
def parse_project(path: Path, code: str, name: str, sector: str):
    """Return (network dict, None) on success or (None, reason) on rejection."""
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        return None, f"workbook unreadable: {type(exc).__name__}"

    if "Baseline Schedule" not in wb.sheetnames:
        return None, "no Baseline Schedule sheet"

    # ---- renewable resource capacities -------------------------------------
    capacity = {}
    if "Resources" in wb.sheetnames:
        ws = wb["Resources"]
        rows = list(ws.iter_rows(values_only=True))
        hdr_i = next((i for i, r in enumerate(rows)
                      if r and any(str(c).strip() == "ID" for c in r if c is not None)), None)
        if hdr_i is not None:
            hdr = [str(c).strip() if c is not None else "" for c in rows[hdr_i]]
            try:
                i_name, i_type, i_avail = (hdr.index("Name"), hdr.index("Type"),
                                           hdr.index("Availability"))
            except ValueError:
                i_name = i_type = i_avail = None
            if i_name is not None:
                for r in rows[hdr_i + 1:]:
                    if not r or r[i_name] is None:
                        continue
                    if str(r[i_type]).strip().lower() != "renewable":
                        continue
                    cap = parse_availability(r[i_avail])
                    if cap and cap > 0:
                        capacity[str(r[i_name]).strip()] = int(np.ceil(cap))
    if not capacity:
        return None, "no renewable resource with a stated availability"

    # ---- activity rows ------------------------------------------------------
    ws = wb["Baseline Schedule"]
    rows = list(ws.iter_rows(values_only=True))
    hdr_i = next((i for i, r in enumerate(rows)
                  if r and any(str(c).strip() == "ID" for c in r if c is not None)), None)
    if hdr_i is None:
        return None, "no header row in Baseline Schedule"
    hdr = [str(c).strip() if c is not None else "" for c in rows[hdr_i]]
    need = ["ID", "WBS", "Predecessors", "Duration"]
    if any(c not in hdr for c in need):
        return None, "Baseline Schedule missing a required column"
    iID, iWBS, iPred, iDur = (hdr.index(c) for c in need)
    iName = hdr.index("Name") if "Name" in hdr else None
    iDem = hdr.index("Resource Demand") if "Resource Demand" in hdr else None
    if iDem is None:
        return None, "no Resource Demand column"

    raw = []
    for r in rows[hdr_i + 1:]:
        if not r or r[iID] is None or str(r[iID]).strip() == "":
            continue
        try:
            aid = int(float(str(r[iID]).strip()))
        except ValueError:
            continue
        raw.append({
            "id": aid,
            "wbs": str(r[iWBS]).strip() if r[iWBS] is not None else "",
            "name": str(r[iName]).strip() if iName is not None and r[iName] is not None else "",
            "pred_raw": r[iPred],
            "dur_raw": r[iDur],
            "dem_raw": r[iDem],
        })
    if not raw:
        return None, "no activity rows"

    # Leaf activities only: a WBS code that is a strict prefix of another is a summary.
    codes = [a["wbs"] for a in raw if a["wbs"]]
    summary = {c for c in codes if any(o != c and o.startswith(c + ".") for o in codes)}
    leaves = [a for a in raw if a["wbs"] and a["wbs"] not in summary]
    if len(leaves) < MIN_ACTIVITIES:
        return None, f"only {len(leaves)} leaf activities (min {MIN_ACTIVITIES})"
    if len(leaves) > MAX_ACTIVITIES:
        return None, f"{len(leaves)} leaf activities (max {MAX_ACTIVITIES})"

    leaf_ids = {a["id"] for a in leaves}
    dropped_rel = 0
    acts = []
    for a in leaves:
        dur = parse_duration_days(a["dur_raw"])
        if dur is None:
            return None, f"unreadable duration on activity {a['id']}"
        dur_i = int(np.ceil(dur)) if dur > 0 else 0
        preds, dl = parse_predecessors(a["pred_raw"])
        dropped_rel += dl
        demands = {k: v for k, v in parse_resource_demand(a["dem_raw"]).items() if k in capacity}
        acts.append({"id": a["id"], "name": a["name"], "wbs": a["wbs"],
                     "duration": dur_i, "demands": demands,
                     "preds": [(p, rel, lag) for p, rel, lag in preds if p in leaf_ids]})

    # Resources actually demanded by at least one leaf activity
    used = sorted({k for a in acts for k in a["demands"]})
    if not used:
        return None, "no leaf activity demands a renewable resource"
    capacity = {k: capacity[k] for k in used}

    # Cap each demand at its resource availability so every activity is schedulable alone.
    clipped = 0
    for a in acts:
        for k, v in list(a["demands"].items()):
            v_i = max(1, int(np.ceil(v)))
            if v_i > capacity[k]:
                v_i = capacity[k]
                clipped += 1
            a["demands"][k] = v_i

    # ---- precedence as finish-start with integer lag ------------------------
    dur_by_id = {a["id"]: a["duration"] for a in acts}
    edges = []
    for a in acts:
        for p, rel, lag in a["preds"]:
            lag_i = int(round(lag))
            if rel == "FS":
                edges.append((p, a["id"], lag_i))
            elif rel == "SS":
                edges.append((p, a["id"], lag_i - dur_by_id[p]))
            elif rel == "FF":
                edges.append((p, a["id"], lag_i - dur_by_id[a["id"]]))
            else:  # SF
                edges.append((p, a["id"], lag_i - dur_by_id[p] - dur_by_id[a["id"]]))

    # Drop any edge that closes a cycle, keeping the source order; record the count.
    order = {a["id"]: i for i, a in enumerate(acts)}
    kept, cyc = [], 0
    adj = {a["id"]: set() for a in acts}

    def reaches(src, dst):
        seen, stack = set(), [src]
        while stack:
            n = stack.pop()
            if n == dst:
                return True
            if n in seen:
                continue
            seen.add(n)
            stack.extend(adj[n])
        return False

    for i, j, lag in sorted(edges, key=lambda e: (order[e[0]], order[e[1]])):
        if i == j or reaches(j, i):
            cyc += 1
            continue
        adj[i].add(j)
        kept.append({"pred": i, "succ": j, "lag": lag})

    net = {
        "code": code, "name": name, "sector": sector,
        "n_activities": len(acts),
        "resources": [{"name": k, "capacity": capacity[k]} for k in used],
        "activities": [{"id": a["id"], "name": a["name"], "wbs": a["wbs"],
                        "duration": a["duration"],
                        "demands": [{"resource": k, "units": v} for k, v in a["demands"].items()]}
                       for a in acts],
        "precedence": kept,
        "provenance": {
            "database": "DSLIB v3.2 (Dynamic Scheduling Library), OR&S Ghent University",
            "reference": "Batselier and Vanhoucke (2015)",
            "source_file": path.name,
            "relations_dropped_unparseable": dropped_rel,
            "relations_dropped_cyclic": cyc,
            "demands_clipped_to_capacity": clipped,
        },
    }
    return net, None


# ---------------------------------------------------------------------------
def main():
    common.banner(log, "STEP 09 - PARSE EMPIRICAL CONSTRUCTION PROJECT NETWORKS (DSLIB v3.2)")
    if not INDEX_XLSX.exists():
        log.error(f"DSLIB index not found at {INDEX_XLSX}")
        return 1

    wb = openpyxl.load_workbook(INDEX_XLSX, read_only=True, data_only=True)
    rows = list(wb["DSLIB"].iter_rows(values_only=True))
    hdr_i = next(i for i, r in enumerate(rows)
                 if r and any(str(c).strip() == "Code" for c in r if c is not None))
    hdr = [str(c).strip() if c is not None else "" for c in rows[hdr_i]]
    iCode, iName, iSector = hdr.index("Code"), hdr.index("Project name"), hdr.index("Sector")

    records = [r for r in rows[hdr_i + 1:] if r and r[iCode]]
    construction = [r for r in records if str(r[iSector]).lower().startswith("construction")]
    log.info(f"{len(records)} projects in DSLIB; {len(construction)} in the construction sector")

    files = {p.name: p for p in EXCEL_DIR.glob("*.xlsx")}
    inventory, parsed = [], 0
    for r in construction:
        code = str(r[iCode]).strip()
        name = str(r[iName]).strip()
        sector = str(r[iSector]).strip()
        match = [p for fn, p in files.items() if fn.startswith(code)]
        if not match:
            inventory.append({"Code": code, "Project": name, "Sector": sector,
                              "Activities": None, "Resources": None,
                              "Status": "rejected", "Reason": "no workbook in archive"})
            continue
        net, reason = parse_project(sorted(match)[0], code, name, sector)
        if net is None:
            inventory.append({"Code": code, "Project": name, "Sector": sector,
                              "Activities": None, "Resources": None,
                              "Status": "rejected", "Reason": reason})
            continue
        (OUT_DIR / f"{code}.json").write_text(json.dumps(net, indent=1), encoding="utf-8")
        parsed += 1
        inventory.append({"Code": code, "Project": name, "Sector": sector,
                          "Activities": net["n_activities"],
                          "Resources": len(net["resources"]),
                          "Status": "parsed", "Reason": ""})

    inv = pd.DataFrame(inventory)
    common.save_table(inv, "table_dslib_inventory")
    log.info(f"parsed {parsed} of {len(construction)} construction projects")
    log.info("rejection reasons:\n" +
             inv.query("Status == 'rejected'")["Reason"].value_counts().to_string())
    if parsed:
        ok = inv.query("Status == 'parsed'")
        log.info(f"activities: median={ok['Activities'].median():.0f} "
                 f"range={ok['Activities'].min()}-{ok['Activities'].max()}")
        log.info("sector composition of the parsed set:\n" +
                 ok["Sector"].str.lower().value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
