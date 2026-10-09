"""
01_filter_osha.py — Reproduce OSHA filtering and run data-integrity verification.

Outputs:
  data_intermediate/construction_filtered.csv      (NAICS 23 subset)
  data_intermediate/environmental_heat_cases.csv   (primary outcome, 605)
  data_intermediate/burn_secondary_cases.csv       (excluded burns; secondary)
  data_intermediate/nonheat_construction.csv        (negative-control pool)
  outputs/tables/table01_filtering.{csv,md}
  outputs/tables/diag_month_distribution.csv
  outputs/tables/diag_state_distribution.csv
  outputs/tables/diag_naics_distribution.csv
  outputs/tables/verification01_data_integrity.json
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
import common

log = common.setup_logger("01_filter_osha")

REQUIRED_COLS = [
    "ID", "EventDate", "City", "State", "Zip", "Latitude", "Longitude",
    "Primary NAICS", "Hospitalized", "Amputation", "Loss of Eye",
    "Final Narrative", "NatureTitle", "EventTitle", "SourceTitle",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def classify_heat(df: pd.DataFrame) -> pd.Series:
    et = df["EventTitle"].astype(str).str.strip().str.lower()
    st = df["SourceTitle"].astype(str).str.strip().str.lower()
    nt = df["NatureTitle"].astype(str).str.strip().str.lower()
    cond_event = et.str.startswith(config.HEAT_EVENTTITLE_PREFIX)
    cond_source = st.isin(config.HEAT_SOURCETITLE_SET)
    cond_nature = nt.apply(lambda x: any(t in x for t in config.HEAT_NATURE_TERMS))
    return cond_event | cond_source | cond_nature, cond_event, cond_source, cond_nature


def classify_burn(df: pd.DataFrame) -> pd.Series:
    """Burns / hot-object thermal injuries that are NOT environmental heat."""
    nt = df["NatureTitle"].astype(str).str.strip().str.lower()
    is_burn = nt.apply(lambda x: any(t in x for t in config.BURN_NATURE_TERMS))
    is_heat, *_ = classify_heat(df)
    return is_burn & (~is_heat)


def main() -> int:
    common.banner(log, "STEP 01 — OSHA FILTERING & DATA INTEGRITY")
    verification = {}

    # --- Verification 1: checksum & row count ---
    log.info("Computing SHA256 of raw file ...")
    digest = sha256(config.RAW_CSV)
    verification["sha256"] = digest
    verification["sha256_matches_expected"] = (digest == config.EXPECTED_SHA256)
    log.info(f"SHA256 = {digest}  match={verification['sha256_matches_expected']}")

    log.info("Loading raw CSV ...")
    df = pd.read_csv(config.RAW_CSV, dtype=str, keep_default_na=False, low_memory=False)
    n_total = len(df)
    verification["row_count"] = n_total
    verification["row_count_matches_expected"] = (n_total == config.EXPECTED_ROWS)
    log.info(f"Rows = {n_total} (expected {config.EXPECTED_ROWS})")

    # required columns
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    verification["missing_required_columns"] = missing_cols
    if missing_cols:
        log.error(f"Missing required columns: {missing_cols}")
        return 1

    # --- derive helper columns ---
    df["EventDate_parsed"] = common.parse_dates(df["EventDate"])
    df["year"] = df["EventDate_parsed"].dt.year
    df["month"] = df["EventDate_parsed"].dt.month
    df["dow"] = df["EventDate_parsed"].dt.dayofweek
    df["lat"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["naics3"] = df["Primary NAICS"].astype(str).str.strip().str[:3]
    df["geo_valid"] = (
        df["lat"].between(18, 72) & df["lon"].between(-180, -65)
        & ~((df["lat"] == 0) & (df["lon"] == 0))
    )

    verification["missing_eventdate"] = int(df["EventDate_parsed"].isna().sum())

    # --- Construction filter ---
    naics = df["Primary NAICS"].astype(str).str.strip()
    constr = df[naics.str.startswith(config.CONSTRUCTION_NAICS_PREFIX)].copy()
    n_constr = len(constr)
    verification["construction_count"] = n_constr
    verification["construction_matches_expected"] = (n_constr == config.EXPECTED_CONSTRUCTION)
    log.info(f"Construction (NAICS 23) = {n_constr} (expected {config.EXPECTED_CONSTRUCTION})")

    # --- Heat classification ---
    is_heat, c_evt, c_src, c_nat = classify_heat(constr)
    heat = constr[is_heat].copy()
    heat["heat_via_event"] = c_evt[is_heat].values
    heat["heat_via_source"] = c_src[is_heat].values
    heat["heat_via_nature"] = c_nat[is_heat].values
    n_heat = len(heat)
    verification["heat_count"] = n_heat
    verification["heat_matches_expected"] = (n_heat == config.EXPECTED_HEAT)
    verification["heat_via_event"] = int(c_evt.sum())
    verification["heat_via_source"] = int(c_src.sum())
    verification["heat_via_nature"] = int(c_nat.sum())
    log.info(f"Environmental-heat analytic = {n_heat} (expected {config.EXPECTED_HEAT})")

    # geolocation among heat cases — break out null vs out-of-frame for transparency
    heat_lat_raw = heat["Latitude"].astype(str).str.strip()
    heat_lon_raw = heat["Longitude"].astype(str).str.strip()
    heat_geo_null = int(((heat_lat_raw == "") | (heat_lon_raw == "")).sum())
    heat_geo_valid = int(heat["geo_valid"].sum())
    heat_geo_outside = int(len(heat) - heat_geo_valid - heat_geo_null)
    verification["heat_geo_null"] = heat_geo_null                 # plan's "1 missing"
    verification["heat_geo_outside_conus_frame"] = heat_geo_outside  # e.g., Guam
    verification["heat_geo_valid"] = heat_geo_valid
    log.info(f"Heat geo: valid={heat_geo_valid}, null={heat_geo_null}, "
             f"outside-US-frame={heat_geo_outside}")

    # --- Burn / hot-object secondary (excluded from primary) ---
    is_burn = classify_burn(constr)
    burn = constr[is_burn].copy()
    verification["burn_secondary_count"] = int(len(burn))
    log.info(f"Burn/hot-object secondary (excluded) = {len(burn)}")

    # --- Non-heat construction pool (negative controls / label validation) ---
    nonheat = constr[~is_heat].copy()
    verification["nonheat_construction_count"] = int(len(nonheat))

    # --- amputation negative-control outcome (non-heat) ---
    amput = nonheat[nonheat["Amputation"].astype(str).str.strip().isin(["1", "1.00"])].copy()
    verification["nonheat_amputation_count"] = int(len(amput))
    log.info(f"Non-heat amputations (negative-control outcome) = {len(amput)}")

    # --- write intermediates ---
    constr.to_csv(config.DATA_INTERMEDIATE / "construction_filtered.csv", index=False)
    heat.to_csv(config.DATA_INTERMEDIATE / "environmental_heat_cases.csv", index=False)
    burn.to_csv(config.DATA_INTERMEDIATE / "burn_secondary_cases.csv", index=False)
    nonheat.to_csv(config.DATA_INTERMEDIATE / "nonheat_construction.csv", index=False)
    amput.to_csv(config.DATA_INTERMEDIATE / "nonheat_amputation_cases.csv", index=False)

    # --- filtering counts ---
    table1 = pd.DataFrame([
        ["All OSHA severe-injury records", n_total],
        ["Construction NAICS 23", n_constr],
        ["Primary environmental heat analytic definition", n_heat],
        ["  ...flagged by EventTitle", int(c_evt.sum())],
        ["  ...flagged by SourceTitle", int(c_src.sum())],
        ["  ...flagged by NatureTitle", int(c_nat.sum())],
        ["Heat cases with valid US geolocation", heat_geo_valid],
        ["  ...missing lat/lon (null)", heat_geo_null],
        ["  ...outside US weather frame (e.g., Guam)", heat_geo_outside],
        ["Heat cases with valid EventDate", int(heat["EventDate_parsed"].notna().sum())],
        ["Burn / hot-object secondary (excluded from primary)", int(len(burn))],
    ], columns=["Step", "Records"])
    common.save_table(table1, "table01_filtering")

    # --- diagnostics ---
    month_dist = (heat["month"].value_counts().sort_index()
                  .rename_axis("month").reset_index(name="records"))
    common.save_table(month_dist, "diag_month_distribution")

    state_dist = (heat["State"].str.title().value_counts()
                  .rename_axis("state").reset_index(name="records"))
    common.save_table(state_dist, "diag_state_distribution")

    # NAICS title distribution: use Primary NAICS code counts (titles not in file)
    naics_dist = (heat["Primary NAICS"].astype(str).str.strip().value_counts()
                  .rename_axis("primary_naics").reset_index(name="records")).head(20)
    common.save_table(naics_dist, "diag_naics_distribution")

    # year distribution (for temporal split sanity)
    year_dist = (heat["year"].value_counts().sort_index()
                 .rename_axis("year").reset_index(name="records"))
    common.save_table(year_dist, "diag_year_distribution")

    common.save_json(verification, "verification01_data_integrity")

    # --- console summary ---
    log.info("-" * 60)
    log.info("VERIFICATION SUMMARY")
    for k in ["sha256_matches_expected", "row_count_matches_expected",
              "construction_matches_expected", "heat_matches_expected"]:
        log.info(f"  {k}: {verification[k]}")
    log.info(f"Top states: {state_dist.head(10)['state'].tolist()}")
    log.info(f"Month distribution: {dict(zip(month_dist['month'], month_dist['records']))}")

    all_ok = all(verification[k] for k in [
        "sha256_matches_expected", "row_count_matches_expected",
        "construction_matches_expected", "heat_matches_expected"])
    log.info(f"ALL INTEGRITY CHECKS PASSED: {all_ok}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
