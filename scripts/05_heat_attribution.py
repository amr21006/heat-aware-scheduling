"""
05_heat_attribution.py — NLP Narrative Classification & Mechanism Stratification.

Classifies severe injury incident narratives into three mutually exclusive strata:
  - ambient-primary: Outdoor/unconditioned setting with no competing mechanism
  - heat-with-co-factor: Heat with non-ambient sources, enclosed spaces, or medical events
  - indeterminate: Narrative lacks sufficient setting or mechanism details

Refits the time-stratified case-crossover model within each stratum to assess
epidemiological mechanism specificity.
"""
from __future__ import annotations

import re
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

import config
import common

warnings.filterwarnings("ignore")
log = common.setup_logger("10_heat_attribution")

# ---------------------------------------------------------------------------
# Rule set (reported verbatim in the supplementary material)
# ---------------------------------------------------------------------------
HEAT_MECHANISM = [
    r"heat stress", r"heat exhaustion", r"heat stroke", r"heatstroke", r"heat syncope",
    r"heat cramp", r"heat[- ]related", r"heat illness", r"heat injury", r"overheat",
    r"overcome by (the )?heat", r"dehydrat", r"hyperthermi", r"sun ?stroke",
    r"heat exposure", r"excessive heat", r"high temperature", r"hot weather",
]
OUTDOOR_SETTING = [
    r"\boutdoor", r"\boutside\b", r"\bin the sun\b", r"\bsun\b", r"\bsunlight",
    r"roof", r"asphalt", r"paving", r"pavement", r"excavat", r"trench", r"ditch",
    r"concrete", r"sidewalk", r"road", r"highway", r"bridge", r"scaffold",
    r"brick", r"mason", r"framing", r"digging", r"landscap", r"survey",
    r"utility line", r"pipeline", r"steel erect", r"rebar", r"formwork",
    r"job ?site", r"construction site", r"field", r"yard", r"parking lot",
    r"heat index", r"ambient", r"weather", r"humid",
]
NON_AMBIENT_HEAT_SOURCE = [
    r"\battic\b", r"boiler", r"furnace", r"\boven\b", r"kiln", r"\bcrawl ?space",
    r"confined space", r"\bmanhole\b", r"\bvault\b", r"\btank\b(?!er)", r"\bvessel\b",
    r"steam", r"molten", r"welding (?:booth|enclosure)", r"unventilated",
    r"inside a (?:tank|vessel|silo|container)", r"enclosed",
    r"no air ?condition", r"without ventilation",
]
COMPETING_MECHANISM = [
    r"\bfell\b", r"\bfall\b", r"\bfalling\b", r"struck by", r"caught in",
    r"crush", r"amputat", r"lacerat", r"fractur", r"electrocut", r"electric shock",
    r"heart attack", r"cardiac", r"myocardial", r"stroke\b(?!.*heat)", r"aneurysm",
    r"seizure", r"epilep", r"diabet", r"pre[- ]?existing", r"medical condition",
    r"overdose", r"asthma", r"allerg", r"insect", r"snake", r"collision",
    r"motor vehicle", r"chemical", r"inhal", r"burn(?:ed|s)? (?:by|from)",
]
VAGUE = [r"^.{0,80}$"]

_C = {k: [re.compile(p, re.I) for p in v] for k, v in {
    "heat": HEAT_MECHANISM, "outdoor": OUTDOOR_SETTING,
    "nonambient": NON_AMBIENT_HEAT_SOURCE, "competing": COMPETING_MECHANISM}.items()}


def hits(text: str, key: str) -> list[str]:
    return [p.pattern for p in _C[key] if p.search(text)]


def classify(narrative: str, nature: str, event: str) -> tuple[str, str]:
    """Return (stratum, reason)."""
    text = f"{narrative} {nature} {event}"
    heat = hits(text, "heat")
    outdoor = hits(text, "outdoor")
    nonamb = hits(narrative, "nonambient")
    comp = hits(narrative, "competing")

    if nonamb:
        return "cofactor", f"non-ambient heat source or enclosed setting: {nonamb[:3]}"
    if comp:
        return "cofactor", f"competing mechanism: {comp[:3]}"
    if heat and outdoor:
        return "ambient_primary", f"heat mechanism {heat[:2]} with outdoor setting {outdoor[:2]}"
    if heat and len(str(narrative)) <= 120:
        return "indeterminate", "heat mechanism stated but no work setting described"
    if heat:
        return "indeterminate", "heat mechanism stated, setting not identifiable"
    return "indeterminate", "no explicit heat mechanism in the narrative"


# ---------------------------------------------------------------------------
def prep(df: pd.DataFrame, exposure: str) -> pd.DataFrame:
    d = df.dropna(subset=[exposure]).copy()
    g = d.groupby("case_id")["is_case"]
    return d[g.transform(lambda s: (s == 1).any()) & g.transform(lambda s: (s == 0).any())]


def fit_or(d: pd.DataFrame, exposure: str, scale: float = 5.0):
    if d["case_id"].nunique() < 10:
        return None
    res = ConditionalLogit(d["is_case"].astype(int).to_numpy(),
                           d[[exposure]].astype(float).to_numpy(),
                           groups=d["case_id"].to_numpy()).fit(disp=0)
    b, se = res.params[0], res.bse[0]
    return {"OR": float(np.exp(b * scale)),
            "CI_low": float(np.exp((b - 1.96 * se) * scale)),
            "CI_high": float(np.exp((b + 1.96 * se) * scale)),
            "p_value": float(res.pvalues[0]),
            "n_strata": int(d["case_id"].nunique())}


def main():
    common.banner(log, "STEP 10 - HEAT ATTRIBUTION STRATA AND SENSITIVITY")

    cases = pd.read_csv(config.DATA_INTERMEDIATE / "environmental_heat_cases.csv",
                        low_memory=False)
    log.info(f"{len(cases)} environmental-heat cases loaded")

    recs = []
    for _, r in cases.iterrows():
        narr = str(r.get("Final Narrative", "") or "")
        stratum, reason = classify(narr, str(r.get("NatureTitle", "") or ""),
                                   str(r.get("EventTitle", "") or ""))
        recs.append({"case_id": r.get("ID", r.name), "attribution": stratum,
                     "reason": reason, "narrative_length": len(narr)})
    att = pd.DataFrame(recs)

    id_col = "ID" if "ID" in cases.columns else None
    if id_col:
        att["case_id"] = cases[id_col].to_numpy()
    att.to_csv(config.DATA_INTERMEDIATE / "heat_attribution.csv", index=False)

    counts = att["attribution"].value_counts()
    strata_tbl = pd.DataFrame({
        "Attribution stratum": ["Ambient heat primary", "Heat with co-factor",
                                "Indeterminate"],
        "Definition": [
            "Heat mechanism described with an outdoor or unconditioned work setting and no competing mechanism",
            "Heat mechanism accompanied by a competing injury or medical mechanism, a non-ambient heat source, or an enclosed setting",
            "Narrative does not identify the work setting or the mechanism with sufficient detail"],
        "Cases": [int(counts.get("ambient_primary", 0)),
                  int(counts.get("cofactor", 0)),
                  int(counts.get("indeterminate", 0))]})
    strata_tbl["Share of cases (%)"] = (100 * strata_tbl["Cases"] / len(att)).round(1)
    common.save_table(strata_tbl, "table_attribution_strata")
    log.info("\n" + strata_tbl.to_string(index=False))

    # ---- refit the case-crossover model within each stratum -----------------
    wm = pd.read_csv(config.DATA_INTERMEDIATE / "weather_matches.csv")
    wm["case_id"] = wm["case_id"].astype(str)
    att["case_id"] = att["case_id"].astype(str)
    wm = wm.merge(att[["case_id", "attribution"]], on="case_id", how="left")
    unmatched = wm["attribution"].isna().sum()
    log.info(f"weather rows without an attribution label: {unmatched}")

    rows = []
    for label, subset in [
            ("All cases (primary analysis)", wm),
            ("Ambient heat primary", wm[wm["attribution"] == "ambient_primary"]),
            ("Heat with co-factor", wm[wm["attribution"] == "cofactor"]),
            ("Indeterminate", wm[wm["attribution"] == "indeterminate"]),
            ("Ambient primary + indeterminate",
             wm[wm["attribution"].isin(["ambient_primary", "indeterminate"])])]:
        d = prep(subset, "tmax")
        fit = fit_or(d, "tmax")
        if fit is None:
            log.warning(f"{label}: too few strata to fit")
            continue
        rows.append({"Case set": label, "Strata": fit["n_strata"],
                     "OR per +5 C": round(fit["OR"], 2),
                     "95% CI": f"{fit['CI_low']:.2f}-{fit['CI_high']:.2f}",
                     "p": f"{fit['p_value']:.1e}"})
        log.info(f"{label:34s} n={fit['n_strata']:4d} OR={fit['OR']:.2f} "
                 f"[{fit['CI_low']:.2f},{fit['CI_high']:.2f}]")

    sens = pd.DataFrame(rows)
    common.save_table(sens, "table_attribution_sensitivity")

    primary = next((r for r in rows if r["Case set"] == "All cases (primary analysis)"), None)
    ambient = next((r for r in rows if r["Case set"] == "Ambient heat primary"), None)
    verification = {
        "n_cases_classified": int(len(att)),
        "n_ambient_primary": int(counts.get("ambient_primary", 0)),
        "n_cofactor": int(counts.get("cofactor", 0)),
        "n_indeterminate": int(counts.get("indeterminate", 0)),
        "OR_all_cases": primary["OR per +5 C"] if primary else None,
        "OR_ambient_primary": ambient["OR per +5 C"] if ambient else None,
        "association_holds_in_ambient_primary": bool(
            ambient and float(str(ambient["95% CI"]).split("-")[0]) > 1.0),
    }
    common.save_json(verification, "verification10_attribution")
    log.info(f"Verification: {verification}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
