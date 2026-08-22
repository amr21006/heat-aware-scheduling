"""
02_validate_heat_labels.py — Outcome-label validation (research plan Validation 1).

Two deliverables:
  (A) A frozen, seed-reproducible STRATIFIED SAMPLE FRAME for human dual review,
      written with empty reviewer columns and the OSHA narrative
      (data_intermediate/label_review_sample.csv). This is the artifact two human
      reviewers fill in; Cohen's kappa is then computed on their adjudication.
  (B) An INDEPENDENT AUTOMATED NARRATIVE CLASSIFIER used as a reproducible proxy
      for manual review NOW, so the rule-based OSHA-coded label can be quantified
      (precision / recall / F1 / Cohen's kappa) before human review is available.

The automated classifier reads ONLY the Final Narrative free text; the rule-based
label reads ONLY the OSHA coded fields. Agreement between two independent signals
is evidence the environmental-heat label is not contaminated by burns.

Outputs:
  data_intermediate/label_review_sample.csv      (human-review frame; deliverable)
  data_intermediate/construction_narr_labeled.csv
  outputs/tables/table02_label_validation.{csv,md}
  outputs/tables/table02_confusion.csv
  outputs/tables/verification04_labeling.json
"""
from __future__ import annotations

import re
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, confusion_matrix

import config
import common

log = common.setup_logger("02_validate_heat_labels")

# ----------------------------------------------------------------------------
# Independent narrative classifier (reads Final Narrative only)
# ----------------------------------------------------------------------------
# NOTE: stems use \w* (not a trailing \b) so "dehydration", "ignited", "collapsed"
# match correctly. A trailing \b after a stem never matches when more letters follow.
HEAT_PAT = re.compile(
    r"\b(?:heat[\s-]?(?:stroke|exhaustion|fatigue|syncope|cramp\w*|illness|stress|"
    r"related|sickness|prostration|issue\w*|problem\w*|symptom\w*)|"
    r"heat\s+index|over[\s-]?heat\w*|heatstroke|sun[\s-]?stroke|"
    r"dehydrat\w*|hyperthermia|elevated\s+body\s+temperature|"
    r"high\s+body\s+temperature|excessive\s+heat|extreme\s+heat|"
    r"hot\s+(?:weather|day|and\s+humid|humid|temperature\w*|condition\w*|"
    r"environment\w*)|high\s+temperature|high\s+heat|"
    r"overcome\s+by\s+(?:the\s+)?heat|overcome\s+by\s+(?:the\s+)?sun|"
    r"from\s+the\s+heat|due\s+to\s+(?:the\s+)?heat|heat[\s-]?induced|"
    r"working\s+in\s+the\s+(?:heat|sun))",
    re.IGNORECASE)

# Secondary heat context (supports collapse/cramps when no burn signal present)
HEAT_CONTEXT_PAT = re.compile(
    r"\b(?:collaps\w*|faint\w*|dizz\w*|light[\s-]?headed|passed\s+out|unrespons\w*|"
    r"lost\s+consciousness|disorient\w*|nausea\w*|vomit\w*|cramp\w*|"
    r"sweat\w*|woozy|wobbl\w*)", re.IGNORECASE)
HOT_ENV_PAT = re.compile(
    r"\b(?:heat|hot|sun|humid\w*|temperature\w*|attic|roof\w*|outdoor\w*|"
    r"outside|degree\w*|excavat\w*)", re.IGNORECASE)

BURN_PAT = re.compile(
    r"\b(?:burn\w*|scald\w*|flame\w*|fire\b|ignit\w*|explos\w*|explod\w*|"
    r"torch\w*|molten|asphalt|tar\b|oxy[\s-]?fuel|propane|acetylene|steam\w*|"
    r"hot\s+water|hot\s+oil|hot\s+tar|hot\s+metal|welding\s+arc|arc\s+flash|"
    r"electrocut\w*|chemical\s+burn|caustic|acid\b)", re.IGNORECASE)


def classify_narrative(text: str) -> str:
    """Return 'heat', 'burn', or 'nonheat' from free text only."""
    if not isinstance(text, str) or not text.strip():
        return "unclear"
    has_heat = bool(HEAT_PAT.search(text))
    has_burn = bool(BURN_PAT.search(text))
    # explicit environmental-heat illness language dominates
    if has_heat and not has_burn:
        return "heat"
    if has_heat and has_burn:
        # both present: decide by which is the injury mechanism — burn words like
        # 'fire/explosion/ignited' indicate thermal contact, override generic heat
        if re.search(r"\b(fire|explos|ignit|flame|torch|scald|molten|oxy)\b", text, re.I):
            return "burn"
        return "heat"
    if has_burn:
        return "burn"
    # no explicit heat term, no burn term: check collapse-in-hot-context
    if HEAT_CONTEXT_PAT.search(text) and HOT_ENV_PAT.search(text):
        return "heat"
    return "nonheat"


def main() -> int:
    common.banner(log, "STEP 02 — OUTCOME LABEL VALIDATION")
    rng = np.random.default_rng(config.RANDOM_SEED)

    constr = pd.read_csv(config.DATA_INTERMEDIATE / "construction_filtered.csv",
                         dtype=str, keep_default_na=False)
    heat_ids = set(pd.read_csv(config.DATA_INTERMEDIATE / "environmental_heat_cases.csv",
                               dtype=str, keep_default_na=False)["ID"])
    burn_ids = set(pd.read_csv(config.DATA_INTERMEDIATE / "burn_secondary_cases.csv",
                               dtype=str, keep_default_na=False)["ID"])

    # rule-based (OSHA-coded) label
    constr["coded_label"] = np.where(constr["ID"].isin(heat_ids), "heat",
                              np.where(constr["ID"].isin(burn_ids), "burn", "nonheat"))
    constr["coded_heat"] = (constr["coded_label"] == "heat").astype(int)

    # independent narrative label
    log.info("Classifying narratives (independent free-text classifier) ...")
    constr["narr_label"] = constr["Final Narrative"].apply(classify_narrative)
    constr["narr_heat"] = (constr["narr_label"] == "heat").astype(int)

    constr.to_csv(config.DATA_INTERMEDIATE / "construction_narr_labeled.csv", index=False)

    # ---------- population-level agreement (full construction set) ----------
    y_rule = constr["coded_heat"].to_numpy()
    y_narr = constr["narr_heat"].to_numpy()
    kappa_pop = cohen_kappa_score(y_rule, y_narr)
    p, r, f1, _ = precision_recall_fscore_support(
        y_narr, y_rule, average="binary", zero_division=0)
    # treat independent narrative as reference for binary env-heat vs not
    cm = confusion_matrix(y_narr, y_rule, labels=[1, 0])  # rows=narr, cols=rule
    log.info(f"[POPULATION] kappa={kappa_pop:.3f}  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}")

    # --- Burn-contamination analysis of the PRIMARY heat label (key target) ---
    coded_heat = constr[constr["coded_label"] == "heat"]
    n_h = len(coded_heat)
    n_confirm = int((coded_heat["narr_label"] == "heat").sum())
    n_burn_contam = int((coded_heat["narr_label"] == "burn").sum())
    n_vague = int((coded_heat["narr_label"] == "nonheat").sum())
    burn_contam_rate = n_burn_contam / n_h
    precision_vs_burns = 1.0 - burn_contam_rate
    narr_confirm_rate = n_confirm / n_h
    # burn-label purity (excluded set should be ~all burn by narrative)
    coded_burn = constr[constr["coded_label"] == "burn"]
    burn_purity = float((coded_burn["narr_label"] == "burn").mean()) if len(coded_burn) else float("nan")
    log.info(f"[HEAT LABEL] confirmed={n_confirm} ({narr_confirm_rate:.1%}), "
             f"burn-contamination={n_burn_contam} ({burn_contam_rate:.1%}), "
             f"vague(for human review)={n_vague} ({n_vague/n_h:.1%})")
    log.info(f"[HEAT LABEL] precision-against-burns={precision_vs_burns:.3f}; "
             f"burn-set purity={burn_purity:.3f}")

    # ---------- stratified sample frame for human review ----------
    summer = constr["EventDate"].apply(
        lambda s: (common.parse_dates(pd.Series([s]))[0].month in config.SUMMER_MONTHS)
        if s.strip() else False)
    pools = {
        "primary_heat": constr[constr["coded_label"] == "heat"],
        "excluded_burn": constr[constr["coded_label"] == "burn"],
        "nonheat_summer": constr[(constr["coded_label"] == "nonheat") & summer],
        "nonheat_nonsummer": constr[(constr["coded_label"] == "nonheat") & (~summer)],
    }
    sample_parts = []
    for stratum, n in config.LABEL_SAMPLE.items():
        pool = pools[stratum]
        take = min(n, len(pool))
        idx = rng.choice(pool.index.to_numpy(), size=take, replace=False)
        part = pool.loc[idx, ["ID", "EventDate", "State", "Primary NAICS",
                              "NatureTitle", "EventTitle", "SourceTitle",
                              "Final Narrative", "coded_label", "narr_label"]].copy()
        part.insert(0, "stratum", stratum)
        sample_parts.append(part)
        log.info(f"  sampled {take:>3}/{n} from {stratum} (pool={len(pool)})")
    sample = pd.concat(sample_parts, ignore_index=True)
    # reviewer columns (empty — to be filled by two human reviewers)
    sample["reviewer1_label"] = ""
    sample["reviewer2_label"] = ""
    sample["adjudicated_label"] = ""
    sample.to_csv(config.DATA_INTERMEDIATE / "label_review_sample.csv", index=False)
    log.info(f"Wrote human-review sample frame: {len(sample)} records "
             "(reviewer columns empty; for dual manual adjudication).")

    # ---------- sample-level proxy metrics (narrative as stand-in reviewer) ----------
    s_rule = (sample["coded_label"] == "heat").astype(int).to_numpy()
    s_narr = (sample["narr_label"] == "heat").astype(int).to_numpy()
    kappa_s = cohen_kappa_score(s_rule, s_narr)
    ps, rs, f1s, _ = precision_recall_fscore_support(
        s_narr, s_rule, average="binary", zero_division=0)

    # ---------- Table 2 ----------
    table2 = pd.DataFrame([
        ["Primary environmental-heat cases (OSHA-coded)", n_h, "—"],
        ["Narrative-confirmed environmental heat", f"{n_confirm} ({narr_confirm_rate:.1%})", "—"],
        ["Burn / hot-object contamination of heat label", f"{n_burn_contam} ({burn_contam_rate:.1%})", "→ 0"],
        ["Vague narrative (flagged for human review)", f"{n_vague} ({n_vague/n_h:.1%})", "—"],
        ["Precision against burn contamination", round(precision_vs_burns, 3), "≥ 0.85"],
        ["Burn-set narrative purity (excluded set)", round(burn_purity, 3), "—"],
        ["Cohen's kappa (rule vs narrative), full set", round(kappa_pop, 3), "≥ 0.75"],
        ["Precision (env-heat, narrative as reference)", round(p, 3), "—"],
        ["Recall (env-heat, narrative as reference)", round(r, 3), "—"],
        ["F1 (env-heat, narrative as reference)", round(f1, 3), "—"],
        ["Cohen's kappa (stratified review sample)", round(kappa_s, 3), "≥ 0.75"],
        ["Review-sample size (for human dual adjudication)", len(sample), "≥ 500"],
    ], columns=["Metric", "Value", "Acceptance target"])
    common.save_table(table2, "table02_label_validation")

    # confusion matrix table (population)
    cm_df = pd.DataFrame(cm, index=["narrative=heat", "narrative=not-heat"],
                         columns=["coded=heat", "coded=not-heat"])
    cm_df.to_csv(config.TABLES / "table02_confusion.csv")

    # cross-tab of narrative label within each coded class (diagnostic)
    crosstab = pd.crosstab(constr["coded_label"], constr["narr_label"])
    crosstab.to_csv(config.TABLES / "table02_crosstab_coded_vs_narrative.csv")
    log.info(f"Cross-tab (coded x narrative):\n{crosstab}")

    verification = {
        "heat_cases": int(n_h),
        "narrative_confirmed_heat": int(n_confirm),
        "narrative_confirmed_rate": float(narr_confirm_rate),
        "burn_contamination_count": int(n_burn_contam),
        "burn_contamination_rate": float(burn_contam_rate),
        "precision_against_burns": float(precision_vs_burns),
        "vague_for_human_review": int(n_vague),
        "burn_set_purity": float(burn_purity),
        "population_kappa": float(kappa_pop),
        "population_precision_narr_ref": float(p),
        "population_recall_narr_ref": float(r),
        "population_f1_narr_ref": float(f1),
        "sample_kappa": float(kappa_s),
        "sample_size": int(len(sample)),
        "kappa_target_met": bool(kappa_pop >= 0.75),
        "precision_vs_burns_target_met": bool(precision_vs_burns >= 0.85),
        "note": ("Automated narrative classifier is a reproducible PROXY for the "
                 "two-reviewer manual adjudication required by the plan. The frozen "
                 "sample frame (label_review_sample.csv) is provided for human review; "
                 "final kappa should be recomputed from reviewer columns. The key "
                 "reviewer concern (burns mixed into the environmental-heat label) is "
                 "quantified by precision_against_burns."),
    }
    common.save_json(verification, "verification04_labeling")

    log.info(f"Targets — kappa>=0.75: {verification['kappa_target_met']}; "
             f"precision-vs-burns>=0.85: {verification['precision_vs_burns_target_met']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
