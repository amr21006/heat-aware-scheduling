"""
15_evaluation_inputs.py — Inputs for the common evaluation of all schedules.

Builds four inputs, each from a source independent of the 2023-2025 evaluation outcomes:
  1. Held-out exposure-response curve: the case-crossover natural-spline model of step 04 refitted
     on 2015-2022 strata only; relative odds of a severe heat injury at a given daily maximum
     temperature against a 27 C reference, held flat outside the 2nd-98th percentile of the
     training temperatures.
  2. Crew size and outdoor exposure of every renewable resource in the project networks:
     workers per unit (1 for a trade crew or an operated machine, 0 for unmanned equipment or
     site facilities, 1/8 for resources counted in work hours per day) and the share of workers
     in the matching O*NET occupation who report working outdoors, exposed to all weather
     conditions, every day (O*NET 30.0, work-context element 4.C.2.a.1.c, category 5).
  3. Rate-based trade weights: environmental-heat severe injuries per 100,000 workers by
     four-digit NAICS construction industry, using national private-sector employment from the
     BLS Quarterly Census of Employment and Wages (annual averages 2015-2024), normalized to
     the maximum.
  4. Agreement between the injury-derived trade weights and O*NET outdoor exposure.

Outputs:
  data_intermediate/eval_exposure_response_2015_2022.csv
  data_intermediate/resource_exposure.csv
  data_intermediate/rate_weight_index.json
  outputs/tables/table_resource_mapping.{csv,md}
  outputs/tables/table_rate_weights.{csv,md}
  outputs/tables/table_weight_validation.{csv,md}
  outputs/tables/verification15_evaluation_inputs.json
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
import warnings
from importlib import import_module

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from statsmodels.discrete.conditional_models import ConditionalLogit

import config
import common

warnings.filterwarnings("ignore")
log = common.setup_logger("15_evaluation_inputs")
ew = import_module("07_exposure_weights")
cc = import_module("04_model_case_crossover")

ONET_DIR = config.DATA_EXTERNAL / "onet"
BLS_DIR = config.DATA_EXTERNAL / "bls_qcew"
BLS_DIR.mkdir(parents=True, exist_ok=True)
REF_C = 27.0

# Resource name -> (O*NET-SOC code, occupation, workers per unit). First match wins.
# Workers per unit: 1 = trade crew member or operator of one machine; 0 = unmanned equipment
# or site facility; 0.125 = resource counted in work hours per day (8-hour day).
RESOURCE_RULES = [
    (r"work hours", "47-2061.00", "Construction Laborers", 0.125),
    (r"vergaderkeet|compressor|generator|water pump|cable stayed pylon|delivery barge|"
     r"pumping barge|scows|hand lift truck", None, "Unmanned equipment or site facility", 0.0),
    (r"crane|kraan|winch", "53-7021.00", "Crane and Tower Operators", 1.0),
    (r"pile|paal|paler|vibratory hammer", "47-2072.00", "Pile Driver Operators", 1.0),
    (r"dredge", "53-7031.00", "Dredge Operators", 1.0),
    (r"tug boat", "53-5011.00", "Sailors and Marine Oilers", 1.0),
    (r"truck|vrachtwagen", "53-3032.00", "Heavy and Tractor-Trailer Truck Drivers", 1.0),
    (r"road worker|asfalt|asphalt|line striper|cold milling|roller|rolles|\bwals\b",
     "47-2071.00", "Paving, Surfacing, and Tamping Equipment Operators", 1.0),
    (r"excavat|graafkraan|digger|bulldozer|wheel loader|telehandler|verreiker|manitu|"
     r"fork lift|telescope lifting|scissor lift|scherenhub|hoogtewerker|drill|tunnel|"
     r"rock socket|herassen|bronbemaling|concrete pump|betonpomp|concrete mixer|"
     r"pole machine|pruning machine|vlindertoestel|standard operator",
     "47-2073.00", "Operating Engineers and Other Construction Equipment Operators", 1.0),
    (r"roof|dakwerk", "47-2181.00", "Roofers", 1.0),
    (r"cement mason|concrete finisher|betoneerder|chapen|floor/sills",
     "47-2051.00", "Cement Masons and Concrete Finishers", 1.0),
    (r"mason|metser|bricklay", "47-2021.00", "Brickmasons and Blockmasons", 1.0),
    (r"ijzervlechter|rebar", "47-2171.00", "Reinforcing Iron and Rebar Workers", 1.0),
    (r"staalmonteur|steel worker|monteerder", "47-2221.00",
     "Structural Iron and Steel Workers", 1.0),
    (r"welder|lasser", "51-4121.00", "Welders, Cutters, Solderers, and Brazers", 1.0),
    (r"window", "47-2121.00", "Glaziers", 1.0),
    (r"bekister|carpenter|timmerman|schrijnwerker|joiner|furniture",
     "47-2031.00", "Carpenters", 1.0),
    (r"painter|schilder", "47-2141.00", "Painters, Construction and Maintenance", 1.0),
    (r"plaster|stukadoor|stucco", "47-2161.00", "Plasterers and Stucco Masons", 1.0),
    (r"tiler", "47-2044.00", "Tile and Stone Setters", 1.0),
    (r"flooring", "47-2042.00", "Floor Layers, Except Carpet, Wood, and Hard Tiles", 1.0),
    (r"plumb|loodgieter|bathroom/kitchen", "47-2152.00",
     "Plumbers, Pipefitters, and Steamfitters", 1.0),
    (r"engineer designer|designer|calculator|construction engineer", "17-2051.00",
     "Civil Engineers", 1.0),
    (r"informatician", "15-1252.00", "Software Developers", 1.0),
    (r"electric|elektric", "47-2111.00", "Electricians", 1.0),
    (r"heating|chauffagist|cooling", "49-9021.00",
     "Heating, Air Conditioning, and Refrigeration Mechanics and Installers", 1.0),
    (r"brandtechnic|veiligheidsinstall", "49-2098.00",
     "Security and Fire Alarm Systems Installers", 1.0),
    (r"technieker|technicus|automatiser", "49-9041.00", "Industrial Machinery Mechanics", 1.0),
    (r"foreman|ploegbaas", "47-1011.00", "First-Line Supervisors of Construction Trades", 1.0),
]
DEFAULT_RULE = ("47-2061.00", "Construction Laborers", 1.0)
_RULES = [(re.compile(p, re.I), soc, occ, wpu) for p, soc, occ, wpu in RESOURCE_RULES]


def map_resource(name: str):
    for rx, soc, occ, wpu in _RULES:
        if rx.search(name or ""):
            return soc, occ, wpu
    return DEFAULT_RULE


def onet_outdoor() -> pd.DataFrame:
    f = sorted(ONET_DIR.glob("Work_Context_db_*.txt"))[-1]
    wc = pd.read_csv(f, sep="\t", dtype=str)
    sel = wc[(wc["Element ID"] == "4.C.2.a.1.c") & (wc["Scale ID"] == "CXP")
             & (wc["Category"] == "5")]
    out = pd.DataFrame({"soc": sel["O*NET-SOC Code"],
                        "outdoor_every_day": sel["Data Value"].astype(float) / 100.0,
                        "onet_n": sel["N"]})
    return out.drop_duplicates("soc").set_index("soc")


def exposure_response_2015_2022() -> tuple[pd.DataFrame, dict]:
    from patsy import dmatrix, build_design_matrices
    df = pd.read_csv(config.DATA_INTERMEDIATE / "weather_matches.csv")
    df = df[df.year.between(*config.TRAIN_YEARS)]
    d = cc.prep(df, "tmax")
    basis = dmatrix("cr(tmax, df=4)", {"tmax": d["tmax"]}, return_type="dataframe")
    di = basis.design_info
    cols = [c for c in basis.columns if c != "Intercept"]
    res = ConditionalLogit(d["is_case"].astype(int).to_numpy(), basis[cols].to_numpy(),
                           groups=d["case_id"].to_numpy()).fit(disp=0)
    lo, hi = float(d["tmax"].quantile(0.02)), float(d["tmax"].quantile(0.98))
    grid = np.round(np.arange(-30.0, 55.01, 0.1), 1)
    clamped = np.clip(grid, lo, hi)
    gb = pd.DataFrame(build_design_matrices([di], {"tmax": clamped})[0], columns=basis.columns)
    rb = pd.DataFrame(build_design_matrices([di], {"tmax": np.array([REF_C])})[0],
                      columns=basis.columns)
    logodds = (gb[cols].to_numpy() - rb[cols].to_numpy()) @ res.params
    curve = pd.DataFrame({"tmax_c": grid, "relative_odds": np.exp(logodds)})
    info = {"n_strata": int(d["case_id"].nunique()), "train_tmax_p02": lo, "train_tmax_p98": hi,
            "reference_c": REF_C,
            "relative_odds_at_35c": float(curve.loc[curve.tmax_c == 35.0, "relative_odds"].iloc[0]),
            "relative_odds_at_20c": float(curve.loc[curve.tmax_c == 20.0, "relative_odds"].iloc[0])}
    return curve, info


def qcew_employment(naics: list[str], years: range) -> pd.DataFrame:
    rows = []
    for y in years:
        for n in naics:
            f = BLS_DIR / f"qcew_{y}_{n}.csv"
            if not f.exists():
                for attempt in range(4):
                    try:
                        r = requests.get(f"https://data.bls.gov/cew/data/api/{y}/a/industry/{n}.csv",
                                         headers={"User-Agent": "Mozilla/5.0 (research)"}, timeout=90)
                        if r.status_code == 200:
                            f.write_bytes(r.content)
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(3 * (attempt + 1))
            q = pd.read_csv(f, dtype={"area_fips": str, "industry_code": str})
            nat = q[(q.area_fips == "US000") & (q.own_code == 5)]
            rows.append({"year": y, "naics4": n,
                         "employment": float(nat["annual_avg_emplvl"].iloc[0])})
    return pd.DataFrame(rows)


def main():
    common.banner(log, "STEP 15 - EVALUATION INPUTS")
    verif = {}

    curve, info = exposure_response_2015_2022()
    curve.to_csv(config.DATA_INTERMEDIATE / "eval_exposure_response_2015_2022.csv", index=False)
    verif["exposure_response_2015_2022"] = info
    log.info(f"held-out exposure-response: {info}")

    onet = onet_outdoor()
    names = {}
    for f in sorted((config.DATA_INTERMEDIATE / "dslib_projects").glob("*.json")):
        net = json.loads(f.read_text(encoding="utf-8"))
        for r in net["resources"]:
            names[r["name"]] = names.get(r["name"], 0) + 1
    rec = []
    for nm, cnt in sorted(names.items(), key=lambda kv: kv[0].lower()):
        soc, occ, wpu = map_resource(nm)
        outd = float(onet.loc[soc, "outdoor_every_day"]) if soc in onet.index else np.nan
        rec.append({"resource": nm, "projects": cnt, "soc": soc, "occupation": occ,
                    "workers_per_unit": wpu, "outdoor_every_day": outd,
                    "injury_naics": ew.trade_to_naics(nm)})
    res = pd.DataFrame(rec)
    missing = res[(res.workers_per_unit > 0) & res.outdoor_every_day.isna()]
    if len(missing):
        log.error(f"O*NET value missing for {missing.soc.unique().tolist()}")
        return 1
    res["outdoor_every_day"] = res["outdoor_every_day"].fillna(0.0)
    res.to_csv(config.DATA_INTERMEDIATE / "resource_exposure.csv", index=False)
    occ_tbl = (res.groupby(["soc", "occupation"])
               .agg(Resources=("resource", "count"), Projects=("projects", "sum"),
                    Workers_per_unit=("workers_per_unit", "first"),
                    Outdoor_every_day=("outdoor_every_day", "first"))
               .reset_index().sort_values("Outdoor_every_day", ascending=False))
    occ_tbl["Outdoor_every_day"] = (100 * occ_tbl["Outdoor_every_day"]).round(1)
    common.save_table(occ_tbl.rename(columns={
        "soc": "O*NET-SOC", "occupation": "Occupation",
        "Workers_per_unit": "Workers per unit",
        "Outdoor_every_day": "Outdoors every day (%)"}), "table_resource_mapping")
    verif["resources"] = {"distinct_names": int(len(res)),
                          "default_laborer": int((res.soc == DEFAULT_RULE[0]).sum()),
                          "unmanned": int((res.workers_per_unit == 0).sum())}

    # rate-based weights
    idx = json.loads((config.DATA_INTERMEDIATE / "exposure_weight_index.json")
                     .read_text(encoding="utf-8"))["exposure_index_by_naics4"]
    naics = sorted(idx)
    emp = qcew_employment(naics, range(2015, 2025))
    mean_emp = emp.groupby("naics4")["employment"].mean()
    share = ew.compute_index().set_index("naics4")
    years_osha = 10 + 8 / 12            # January 2015 to August 2025
    rate = pd.DataFrame({"NAICS": naics,
                         "Industry": [ew.NAICS_LABEL.get(n, n) for n in naics],
                         "Mean employment 2015-2024": [round(mean_emp[n]) for n in naics],
                         "Environmental-heat injuries": [int(share.loc[n, "heat_injuries"])
                                                         for n in naics]})
    rate["Heat injuries per 100,000 workers per year"] = (
        1e5 * rate["Environmental-heat injuries"] / years_osha
        / rate["Mean employment 2015-2024"]).round(3)
    rate["Rate-based index"] = (rate["Heat injuries per 100,000 workers per year"]
                                / rate["Heat injuries per 100,000 workers per year"].max()).round(3)
    rate["Share-based index"] = [idx[n] for n in naics]
    rate = rate.sort_values("Rate-based index", ascending=False)
    common.save_table(rate, "table_rate_weights")
    (config.DATA_INTERMEDIATE / "rate_weight_index.json").write_text(json.dumps(
        {"rate_index_by_naics4": dict(zip(rate.NAICS, rate["Rate-based index"]))}, indent=1),
        encoding="utf-8")
    rho_rs = spearmanr(rate["Rate-based index"], rate["Share-based index"])
    verif["rate_vs_share_spearman"] = [float(rho_rs.statistic), float(rho_rs.pvalue)]

    # agreement of injury-derived weights with independent outdoor exposure (trade resources)
    trade = res[res.injury_naics.notna() & (res.workers_per_unit > 0)].copy()
    trade["share_index"] = trade.injury_naics.map(idx)
    trade["rate_index"] = trade.injury_naics.map(dict(zip(rate.NAICS, rate["Rate-based index"])))
    by_naics = (trade.groupby("injury_naics")
                .agg(Resources=("resource", "count"),
                     Mean_outdoor=("outdoor_every_day", "mean"),
                     Share_index=("share_index", "first"),
                     Rate_index=("rate_index", "first")).reset_index())
    by_naics["Industry"] = by_naics.injury_naics.map(ew.NAICS_LABEL)
    by_naics["Mean_outdoor"] = (100 * by_naics["Mean_outdoor"]).round(1)
    common.save_table(by_naics.rename(columns={
        "injury_naics": "NAICS", "Mean_outdoor": "Mean outdoors every day (%)",
        "Share_index": "Share-based index", "Rate_index": "Rate-based index"}),
        "table_weight_validation")
    r1 = spearmanr(trade.share_index, trade.outdoor_every_day)
    r2 = spearmanr(trade.rate_index, trade.outdoor_every_day)
    verif["resource_level_spearman_share_vs_outdoor"] = [float(r1.statistic), float(r1.pvalue),
                                                         int(len(trade))]
    verif["resource_level_spearman_rate_vs_outdoor"] = [float(r2.statistic), float(r2.pvalue)]
    common.save_json(verif, "verification15_evaluation_inputs")
    log.info("\n" + rate.to_string(index=False))
    log.info("\n" + by_naics.to_string(index=False))
    log.info(f"verification: {verif}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
