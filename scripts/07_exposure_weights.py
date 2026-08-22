"""
07_exposure_weights.py — Empirical Craft Vulnerability Weight Calibration.

Derives activity exposure weights from 4-digit NAICS construction industry incident
propensities, normalizing by the maximum observed rate across trades.
"""
from __future__ import annotations

import json
import re
import sys
import warnings

import numpy as np
import pandas as pd

import config
import common

warnings.filterwarnings("ignore")
log = common.setup_logger("13_exposure_weights")

MIN_INJURIES_FOR_INDEX = 200   # an industry needs enough injuries for a stable share

NAICS_LABEL = {
    "2361": "Residential building construction",
    "2362": "Nonresidential building construction",
    "2371": "Utility system construction",
    "2373": "Highway, street and bridge construction",
    "2379": "Other heavy and civil engineering construction",
    "2381": "Foundation, structure and building exterior contractors",
    "2382": "Building equipment contractors",
    "2383": "Building finishing contractors",
    "2389": "Other specialty trade contractors",
}

# Trade keyword -> NAICS four-digit industry. Order matters: first match wins.
TRADE_PATTERNS = [
    # Highway, street and bridge work
    (r"asfalt|asphalt|paving|pavement|wegenwerk|\bweg\b|road|street|highway|"
     r"\bbridge\b|\bbrug\b|sidewalk|trottoir|kerb|curb", "2373"),
    # Utility systems
    (r"riool|sewer|drainage|waterleiding|water main|pipeline|leiding|utility|"
     r"nutsvoorzien|kabel|cable|elektriciteitsnet|power network|telecom", "2371"),
    # Site preparation, earthmoving, lifting
    (r"grondwerk|grondwerker|graafkraan|graver|excavat|earthwork|digging|"
     r"sloop|demolit|bronbemaling|dewater|paal|pile|boor|drill|wals|roller|"
     r"kraan|crane|hoogtewerker|hoist|torenkraan|montagekraan|drilhamer|herassen", "2389"),
    # Structure, envelope and exterior
    (r"metser|metselaar|mason|bricklay|bekist|formwork|timmerman|carpenter|"
     r"beton|concrete|ijzervlechter|rebar|reinforc|staal|steel|welder|lasser|"
     r"dak|roof|scaffold|stelling|precast|monteerder|erector|gevel|facade|"
     r"raam|ramen|window|glaz|vitrage|siding|structur", "2381"),
    # Building equipment
    (r"loodgieter|plumb|sanitair|elektric|elektrisch|electric|chauffage|"
     r"chauffagist|heating|verwarming|hvac|ventilat|koeling|cooling|"
     r"brandtechnic|fire protect|veiligheidsinstall|security|automatiser|"
     r"automation|lift|elevator|technicus|technieker|installer", "2382"),
    # Interior finishing
    (r"schilder|paint|stukadoor|plaster|stucco|pleister|gyproc|drywall|"
     r"chape|screed|vloer|floor|tegel|tile|plafond|ceiling|meubel|furniture|"
     r"keuken|kitchen|schrijnwerk|joinery|isolatie|insulat|interior|finish|"
     r"behang|wallpaper", "2383"),
]
GENERIC_LABOUR = re.compile(
    r"handlanger|arbeider|arbeiders|labour|labor|werkkracht|workforce|manpower|"
    r"#\s*people|work hours|foreman|ploegbaas|team|subcontractor|onderaannemer|"
    r"personeel|crew", re.I)

_TRADE_RE = [(re.compile(p, re.I), n) for p, n in TRADE_PATTERNS]

# Project sector -> NAICS four-digit industry, refined by project-name keywords.
SECTOR_ROAD = re.compile(
    r"road|street|highway|bridge|sidewalk|pavement|asphalt|tunnel|rail|"
    r"parking|roundabout|intersection|carriageway", re.I)
SECTOR_UTILITY = re.compile(
    r"sewer|sanitary|water|canal|pipeline|utility|power|network|cable|"
    r"pumping|intake|drainage|electric|telecom|wind|solar|energy", re.I)


def trade_to_naics(resource_name: str) -> str | None:
    """Return the NAICS industry for a named trade, or None for undifferentiated labour."""
    if GENERIC_LABOUR.search(resource_name or ""):
        return None
    for rx, naics in _TRADE_RE:
        if rx.search(resource_name or ""):
            return naics
    return None


def sector_to_naics(sector: str, project_name: str) -> str:
    s = (sector or "").lower()
    if "civil" in s:
        if SECTOR_ROAD.search(project_name or ""):
            return "2373"
        if SECTOR_UTILITY.search(project_name or ""):
            return "2371"
        return "2379"
    if "residential" in s:
        return "2361"
    return "2362"      # commercial, institutional and industrial buildings


def compute_index() -> dict:
    """Heat-injury share by NAICS four-digit industry, normalised to the maximum."""
    con = pd.read_csv(config.DATA_INTERMEDIATE / "construction_filtered.csv", low_memory=False)
    heat = pd.read_csv(config.DATA_INTERMEDIATE / "environmental_heat_cases.csv", low_memory=False)
    for df in (con, heat):
        df["naics4"] = df["Primary NAICS"].astype(str).str.replace(r"\D", "", regex=True).str[:4]

    all_counts = con["naics4"].value_counts()
    heat_counts = heat["naics4"].value_counts()
    tbl = pd.DataFrame({"injuries": all_counts, "heat_injuries": heat_counts}).fillna(0)
    tbl = tbl[tbl["injuries"] >= MIN_INJURIES_FOR_INDEX].copy()
    tbl["heat_share_pct"] = 100 * tbl["heat_injuries"] / tbl["injuries"]
    tbl["exposure_index"] = (tbl["heat_share_pct"] / tbl["heat_share_pct"].max()).round(3)
    tbl = tbl.sort_values("exposure_index", ascending=False)
    tbl.index.name = "naics4"
    return tbl.reset_index()


def main():
    common.banner(log, "STEP 13 - EMPIRICAL TRADE EXPOSURE WEIGHTS")
    tbl = compute_index()
    tbl["Industry"] = tbl["naics4"].map(NAICS_LABEL).fillna("Other construction")
    out = tbl[["naics4", "Industry", "injuries", "heat_injuries",
               "heat_share_pct", "exposure_index"]].rename(columns={
                   "naics4": "NAICS", "injuries": "Severe injuries",
                   "heat_injuries": "Environmental-heat injuries",
                   "heat_share_pct": "Heat share of injuries (%)",
                   "exposure_index": "Exposure index"})
    out["Heat share of injuries (%)"] = out["Heat share of injuries (%)"].round(2)
    out["Environmental-heat injuries"] = out["Environmental-heat injuries"].astype(int)
    common.save_table(out, "table_exposure_weight_index")
    log.info("\n" + out.to_string(index=False))

    index = dict(zip(tbl["naics4"], tbl["exposure_index"]))
    payload = {
        "exposure_index_by_naics4": index,
        "naics_labels": NAICS_LABEL,
        "min_injuries_for_index": MIN_INJURIES_FOR_INDEX,
        "normalisation": "share divided by the largest industry share",
        "scenario_weights_psplib": config.OUTDOOR_WEIGHTS,
    }
    (config.DATA_INTERMEDIATE / "exposure_weight_index.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8")

    log.info(f"index range {min(index.values()):.3f}-{max(index.values()):.3f} "
             f"across {len(index)} industries")
    log.info(f"ratio of most to least exposed industry: "
             f"{max(index.values()) / min(index.values()):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
