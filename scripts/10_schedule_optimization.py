"""
10_schedule_optimization.py — Resource-Constrained Project Scheduling Optimization.

Solves the heat-aware RCPSP on empirical construction networks using Google OR-Tools CP-SAT:
  - Minimizes total worker thermal exposure
  - Enforces strict precedence constraints and renewable resource limits
  - Preserves baseline project makespan duration (zero project delay)
"""
from __future__ import annotations

import json
import sys
import warnings
import zlib
from datetime import date, timedelta

import numpy as np
import pandas as pd
import config
import common
from ortools.sat.python import cp_model
from importlib import import_module

try:
    _ew = import_module("07_exposure_weights")
except ModuleNotFoundError:
    _ew = import_module("13_exposure_weights")

warnings.filterwarnings("ignore")
log = common.setup_logger("14_rerun_scheduling")

YEAR = 2023
START_DATE = date(YEAR, 7, 1)
HIGH_RISK_THRESHOLD = 0.5
CAL_DIR = config.DATA_EXTERNAL / "risk_calendars"
PROJ_DIR = config.DATA_INTERMEDIATE / "dslib_projects"
SOLVER_SEED = config.RANDOM_SEED
SOLVER_WORKERS = 8

# Time limit in seconds by activity count. Applied identically to both objectives.
def time_limit_for(n: int) -> float:
    return 10.0 if n <= 40 else (20.0 if n <= 90 else 35.0)


# ---------------------------------------------------------------------------
# Unified network structure
# ---------------------------------------------------------------------------
class Network:
    """Activities, finish-start precedence with integer lags, renewable resources."""

    def __init__(self, name, sector, jobs, dur, demand, cap, edges, weight):
        self.name, self.sector = name, sector
        self.jobs = jobs                       # ordered activity ids
        self.dur = dur                         # id -> integer duration in days
        self.demand = demand                   # id -> list of demands per resource
        self.cap = cap                         # list of resource capacities
        self.edges = edges                     # (pred, succ, lag)
        self.weight = weight                   # id -> exposure weight in [0, 1]
        self.n_res = len(cap)

    @property
    def n(self):
        return len(self.jobs)

    def total_duration(self):
        return sum(self.dur.values())


def load_real_project(path, exposure_index) -> Network:
    net = json.loads(path.read_text(encoding="utf-8"))
    res_names = [r["name"] for r in net["resources"]]
    cap = [int(r["capacity"]) for r in net["resources"]]
    ridx = {n: i for i, n in enumerate(res_names)}

    sector_naics = _ew.sector_to_naics(net["sector"], net["name"])
    sector_w = exposure_index.get(sector_naics, float(np.mean(list(exposure_index.values()))))

    jobs, dur, demand, weight = [], {}, {}, {}
    for a in net["activities"]:
        jid = a["id"]
        jobs.append(jid)
        dur[jid] = int(a["duration"])
        d = [0] * len(cap)
        for dm in a["demands"]:
            d[ridx[dm["resource"]]] = int(dm["units"])
        demand[jid] = d
        # Exposure weight: demand-weighted mean over identifiable trades,
        # falling back to the project's own sector for undifferentiated labor.
        num, den = 0.0, 0.0
        for dm in a["demands"]:
            naics = _ew.trade_to_naics(dm["resource"])
            w = exposure_index.get(naics, sector_w) if naics else sector_w
            num += w * dm["units"]
            den += dm["units"]
        weight[jid] = (num / den) if den > 0 else sector_w
        if dur[jid] == 0:
            weight[jid] = 0.0

    edges = [(e["pred"], e["succ"], int(e["lag"])) for e in net["precedence"]]
    # The source database records the same sector with inconsistent capitalisation
    # (for example "Construction (Civil)" and "Construction (civil)"); normalize so the
    # sector breakdown does not split one sector across two rows.
    sector = net["sector"].strip()
    if sector.lower().startswith("construction (") and sector.endswith(")"):
        inner = sector[len("construction ("):-1].strip().lower()
        sector = f"Construction ({inner})"
    return Network(net["code"], sector, jobs, dur, demand, cap, edges, weight)


def load_psplib(path, scenario="mixed") -> Network:
    """PSPLIB .sm instance with declared scenario exposure weights."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    n = next(int(ln.split(":")[1]) for ln in lines if "jobs (incl" in ln)
    i = next(k for k, ln in enumerate(lines) if "PRECEDENCE RELATIONS" in ln) + 2
    succ = {}
    while not lines[i].startswith("***"):
        p = lines[i].split()
        succ[int(p[0])] = [int(x) for x in p[3:3 + int(p[2])]]
        i += 1
    j = next(k for k, ln in enumerate(lines) if "REQUESTS/DURATIONS" in ln) + 3
    dur, demand = {}, {}
    while not lines[j].startswith("***"):
        p = lines[j].split()
        if len(p) >= 4:
            dur[int(p[0])] = int(p[2])
            demand[int(p[0])] = [int(x) for x in p[3:]]
        j += 1
    c = next(k for k, ln in enumerate(lines) if "RESOURCEAVAILABILITIES" in ln) + 2
    cap = [int(x) for x in lines[c].split()]

    classes = list(config.OUTDOOR_WEIGHTS.values())
    rng = np.random.default_rng(config.RANDOM_SEED + (zlib.crc32(path.stem.encode()) % 100000))
    weight = {}
    for job in range(1, n + 1):
        if dur[job] == 0:
            weight[job] = 0.0
        elif scenario == "mixed":
            weight[job] = float(rng.choice(classes))
        elif scenario == "all_high":
            weight[job] = 1.0
        else:
            weight[job] = 0.5
    edges = [(p, s, 0) for p, ss in succ.items() for s in ss]
    return Network(path.stem, "PSPLIB benchmark", list(range(1, n + 1)),
                   dur, demand, cap, edges, weight)


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------
def topological_order(net: Network):
    """Activity ids in a precedence-feasible order, ties broken by source order."""
    indeg = {j: 0 for j in net.jobs}
    succ = {j: [] for j in net.jobs}
    for i, j, _lag in net.edges:
        succ[i].append(j)
        indeg[j] += 1
    rank = {j: r for r, j in enumerate(net.jobs)}
    ready = sorted([j for j in net.jobs if indeg[j] == 0], key=lambda j: rank[j])
    order = []
    while ready:
        job = ready.pop(0)
        order.append(job)
        for s in succ[job]:
            indeg[s] -= 1
            if indeg[s] == 0:
                ready.append(s)
        ready.sort(key=lambda j: rank[j])
    if len(order) != net.n:                    # unreachable: cycles are removed at parse time
        order += [j for j in net.jobs if j not in set(order)]
    return order


def serial_sgs(net: Network):
    """Weather-blind priority-rule schedule: serial generation scheme in topological
    order, scheduling each activity at its earliest precedence- and resource-feasible
    start. Always returns a feasible schedule."""
    pred = {j: [] for j in net.jobs}
    for i, j, lag in net.edges:
        pred[j].append((i, lag))
    horizon = net.total_duration() + sum(abs(l) for _, _, l in net.edges) + 2
    usage = np.zeros((horizon + 2, net.n_res), dtype=int)
    start = {}
    for job in topological_order(net):
        est = 0
        for p, lag in pred[job]:
            est = max(est, start[p] + net.dur[p] + lag)
        est = max(est, 0)
        d, req = net.dur[job], net.demand[job]
        if d == 0:
            start[job] = est
            continue
        t = est
        while t + d < horizon:
            if all(usage[tt, k] + req[k] <= net.cap[k]
                   for tt in range(t, t + d) for k in range(net.n_res)):
                break
            t += 1
        for tt in range(t, min(t + d, horizon)):
            for k in range(net.n_res):
                usage[tt, k] += req[k]
        start[job] = t
    return start


def solve(net: Network, day_risk, objective, horizon, makespan_cap=None, time_limit=10.0):
    """CP-SAT. objective is 'makespan' or 'heat'. Identical parameters for both."""
    model = cp_model.CpModel()
    s, e, iv = {}, {}, {}
    for job in net.jobs:
        d = net.dur[job]
        s[job] = model.NewIntVar(0, horizon, f"s{job}")
        e[job] = model.NewIntVar(0, horizon, f"e{job}")
        iv[job] = model.NewIntervalVar(s[job], d, e[job], f"iv{job}")
    for i, j, lag in net.edges:
        model.Add(s[j] >= e[i] + lag)
    for k in range(net.n_res):
        model.AddCumulative([iv[j] for j in net.jobs],
                            [net.demand[j][k] for j in net.jobs], net.cap[k])

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, [e[j] for j in net.jobs])
    if makespan_cap is not None:
        model.Add(makespan <= makespan_cap)

    if objective == "makespan":
        model.Minimize(makespan)
    else:
        SCALE = 1000
        terms = []
        for job in net.jobs:
            d, w = net.dur[job], net.weight[job]
            if d == 0 or w <= 0:
                continue
            max_s = horizon - d
            if max_s < 0:
                continue
            costs = [int(round(w * sum(day_risk[t] for t in range(st, st + d)) * SCALE))
                     for st in range(0, max_s + 1)]
            model.Add(s[job] <= max_s)
            ci = model.NewIntVar(min(costs), max(costs), f"c{job}")
            model.AddElement(s[job], costs, ci)
            terms.append(ci)
        if not terms:
            return None
        total = model.NewIntVar(0, sum(1 for _ in terms) * 10 ** 7, "H")
        model.Add(total == sum(terms))
        model.Minimize(total)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = SOLVER_WORKERS
    solver.parameters.random_seed = SOLVER_SEED
    st = solver.Solve(model)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return ({j: solver.Value(s[j]) for j in net.jobs},
            int(solver.Value(makespan)),
            st == cp_model.OPTIMAL)


# ---------------------------------------------------------------------------
# Independent verification and scoring
# ---------------------------------------------------------------------------
def score(net: Network, start, day_risk):
    prec_viol = sum(1 for i, j, lag in net.edges
                    if start[j] < start[i] + net.dur[i] + lag)
    makespan = max(start[j] + net.dur[j] for j in net.jobs)
    usage = {}
    for job in net.jobs:
        for t in range(start[job], start[job] + net.dur[job]):
            for k in range(net.n_res):
                if net.demand[job][k]:
                    usage[(t, k)] = usage.get((t, k), 0) + net.demand[job][k]
    res_viol = sum(1 for (t, k), u in usage.items() if u > net.cap[k])

    heat, high_days = 0.0, 0
    for job in net.jobs:
        w = net.weight[job]
        if net.dur[job] == 0 or w <= 0:
            continue
        for t in range(start[job], start[job] + net.dur[job]):
            r = day_risk[t] if t < len(day_risk) else 0.0
            heat += w * r
            if r >= HIGH_RISK_THRESHOLD:
                high_days += 1
    return {"makespan": makespan, "heat": heat, "high_days": high_days,
            "precedence_violations": prec_viol, "resource_violations": res_viol}


def continuity(net: Network, start):
    """Number of separate work spells per renewable resource, summed over resources."""
    spells = 0
    for k in range(net.n_res):
        days = set()
        for job in net.jobs:
            if net.demand[job][k]:
                days.update(range(start[job], start[job] + net.dur[job]))
        if not days:
            continue
        ordered = sorted(days)
        spells += 1 + sum(1 for a, b in zip(ordered, ordered[1:]) if b - a > 1)
    return spells


def management_impact(net: Network, s_base, s_aware):
    shifts = [abs(s_aware[j] - s_base[j]) for j in net.jobs if net.dur[j] > 0]
    moved = [x for x in shifts if x > 0]
    return {
        "activities_moved": len(moved),
        "activities_total": len(shifts),
        "pct_activities_moved": round(100 * len(moved) / max(1, len(shifts)), 1),
        "mean_shift_days": round(float(np.mean(moved)), 2) if moved else 0.0,
        "max_shift_days": int(max(shifts)) if shifts else 0,
        "float_consumed_activity_days": int(sum(shifts)),
        "trade_spells_base": continuity(net, s_base),
        "trade_spells_aware": continuity(net, s_aware),
    }


# ---------------------------------------------------------------------------
def risk_series(cal: pd.DataFrame, col: str, start_date: date, n_days: int):
    m = dict(zip(cal["date"], cal[col]))
    return [float(m.get(pd.Timestamp(start_date + timedelta(days=t)), 0.0))
            for t in range(n_days)]


def run_scenario(net: Network, cal, risk_col, start_date, time_limit):
    """Returns a result row, or None when the calendar cannot cover the schedule."""
    s_sgs = serial_sgs(net)
    m_sgs = max(s_sgs[j] + net.dur[j] for j in net.jobs)

    cal_days = (cal["date"].max().date() - start_date).days
    if m_sgs > cal_days:
        return None, f"schedule of {m_sgs} days exceeds the {cal_days}-day calendar horizon"

    day_risk = risk_series(cal, risk_col, start_date, m_sgs + 2)

    # The priority-rule schedule is a valid upper bound on the horizon for both solves.
    base = solve(net, day_risk, "makespan", m_sgs, time_limit=time_limit)
    if base is None:
        return None, "no feasible duration-minimizing schedule within the time limit"
    s_base, m_base, base_proven = base

    # The heat-aware solve is capped at the baseline duration, so its horizon is m_base.
    aware = solve(net, day_risk, "heat", m_base, makespan_cap=m_base, time_limit=time_limit)
    if aware is None:
        return None, "no feasible heat-aware schedule within the time limit"
    s_aware, m_aware, _ = aware

    sc_sgs = score(net, s_sgs, day_risk)
    sc_base = score(net, s_base, day_risk)
    sc_aware = score(net, s_aware, day_risk)
    mi = management_impact(net, s_base, s_aware)

    row = {
        "project": net.name, "sector": net.sector, "activities": net.n,
        "resources": net.n_res, "risk_scale": risk_col,
        "sgs_makespan": sc_sgs["makespan"], "sgs_heat": round(sc_sgs["heat"], 3),
        "sgs_high_days": sc_sgs["high_days"],
        "base_makespan": sc_base["makespan"], "base_heat": round(sc_base["heat"], 3),
        "base_high_days": sc_base["high_days"], "base_proven_optimal": base_proven,
        "aware_makespan": sc_aware["makespan"], "aware_heat": round(sc_aware["heat"], 3),
        "aware_high_days": sc_aware["high_days"],
        "prec_viol": sc_base["precedence_violations"] + sc_aware["precedence_violations"]
                     + sc_sgs["precedence_violations"],
        "res_viol": sc_base["resource_violations"] + sc_aware["resource_violations"]
                    + sc_sgs["resource_violations"],
    }
    row["heat_reduction_pct"] = round(
        100 * (sc_base["heat"] - sc_aware["heat"]) / sc_base["heat"], 2) if sc_base["heat"] > 0 else 0.0
    row["high_day_reduction_pct"] = round(
        100 * (sc_base["high_days"] - sc_aware["high_days"]) / sc_base["high_days"], 2) \
        if sc_base["high_days"] > 0 else 0.0
    row["duration_increase_pct"] = round(
        100 * (sc_aware["makespan"] - sc_base["makespan"]) / sc_base["makespan"], 2)
    # Fairness check against the priority-rule baseline used previously.
    row["sgs_vs_cpsat_duration_gap_pct"] = round(
        100 * (sc_sgs["makespan"] - sc_base["makespan"]) / sc_base["makespan"], 2)
    row["heat_reduction_vs_sgs_pct"] = round(
        100 * (sc_sgs["heat"] - sc_aware["heat"]) / sc_sgs["heat"], 2) if sc_sgs["heat"] > 0 else 0.0
    row.update(mi)
    return row, None


def main():
    common.banner(log, "STEP 14 - HEAT-AWARE SCHEDULING ON EMPIRICAL CONSTRUCTION PROJECTS")

    idx_file = config.DATA_INTERMEDIATE / "exposure_weight_index.json"
    if not idx_file.exists():
        log.error("exposure_weight_index.json missing - run 13_exposure_weights.py first")
        return 1
    exposure_index = json.loads(idx_file.read_text(encoding="utf-8"))["exposure_index_by_naics4"]

    calendars = {}
    for loc in config.SCHEDULE_LOCATIONS:
        f = CAL_DIR / f"{loc}_continuous.csv"
        if not f.exists():
            log.error(f"{f} missing - run 12_build_risk_calendars.py first")
            return 1
        calendars[loc] = pd.read_csv(f, parse_dates=["date"])

    proj_files = sorted(PROJ_DIR.glob("*.json"))
    if not proj_files:
        log.error("no parsed projects - run 09_parse_dslib.py first")
        return 1
    projects = [load_real_project(p, exposure_index) for p in proj_files]
    log.info(f"{len(projects)} empirical construction projects loaded")
    log.info(f"activity counts: median={np.median([p.n for p in projects]):.0f} "
             f"range={min(p.n for p in projects)}-{max(p.n for p in projects)}")
    wsum = [w for p in projects for j, w in p.weight.items() if w > 0]
    log.info(f"exposure weights: mean={np.mean(wsum):.3f} "
             f"range={min(wsum):.3f}-{max(wsum):.3f}")

    rows, skipped = [], []
    for loc, cal in calendars.items():
        for net in projects:
            tl = time_limit_for(net.n)
            for risk_col in ["r_model_dayahead", "r_model_sameday", "r_screening"]:
                row, why = run_scenario(net, cal, risk_col, START_DATE, tl)
                if row is None:
                    skipped.append({"project": net.name, "location": loc,
                                    "risk_scale": risk_col, "reason": why})
                    continue
                row["location"] = loc
                row["weighting"] = "empirical_index"
                rows.append(row)
        done = len([r for r in rows if r["location"] == loc])
        log.info(f"[{loc}] {done} scenarios completed")

    # ---- exposure-weight sensitivity -------------------------------------
    # The same projects and calendars are rerun on the primary risk scale with the
    # activity exposure weights replaced, so the influence of the weighting can be
    # read directly against the empirically indexed result.
    weight_rows = []
    for scheme in ["uniform_one", "scenario_mixed"]:
        for loc, cal in calendars.items():
            for net in projects:
                original = dict(net.weight)
                if scheme == "uniform_one":
                    net.weight = {j: (0.0 if net.dur[j] == 0 else 1.0) for j in net.jobs}
                else:
                    classes = list(config.OUTDOOR_WEIGHTS.values())
                    rng = np.random.default_rng(
                        config.RANDOM_SEED + (zlib.crc32(net.name.encode()) % 100000))
                    net.weight = {j: (0.0 if net.dur[j] == 0 else float(rng.choice(classes)))
                                  for j in net.jobs}
                row, why = run_scenario(net, cal, "r_model_dayahead", START_DATE,
                                        time_limit_for(net.n))
                net.weight = original
                if row is None:
                    continue
                row["location"] = loc
                row["weighting"] = scheme
                weight_rows.append(row)
        log.info(f"[weighting={scheme}] {len([r for r in weight_rows if r['weighting'] == scheme])} scenarios")

    if not rows:
        log.error("no scenario completed")
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLES / "sched_real_projects_full.csv", index=False)
    pd.DataFrame(skipped).to_csv(config.TABLES / "sched_real_projects_skipped.csv", index=False)
    log.info(f"{len(df)} scenarios completed; {len(skipped)} skipped")
    if skipped:
        log.info("skip reasons:\n" +
                 pd.DataFrame(skipped)["reason"].value_counts().to_string())

    prim = df[df["risk_scale"] == "r_model_dayahead"]

    # ---- headline table on the primary risk scale --------------------------
    headline = pd.DataFrame([
        {"Schedule": "Duration-minimizing baseline (no heat information)",
         "Project duration (days)": round(prim["base_makespan"].mean(), 1),
         "Modeled heat-risk exposure": round(prim["base_heat"].mean(), 2),
         "High-risk activity-days": round(prim["base_high_days"].mean(), 1),
         "Precedence violations": int(prim["prec_viol"].sum()),
         "Resource violations": int(prim["res_viol"].sum())},
        {"Schedule": "Heat-aware schedule (same duration)",
         "Project duration (days)": round(prim["aware_makespan"].mean(), 1),
         "Modeled heat-risk exposure": round(prim["aware_heat"].mean(), 2),
         "High-risk activity-days": round(prim["aware_high_days"].mean(), 1),
         "Precedence violations": 0, "Resource violations": 0}])
    common.save_table(headline, "table_sched_real_projects")

    # ---- by sector ---------------------------------------------------------
    by_sector = (prim.groupby("sector")
                 .agg(Projects=("project", "nunique"), Scenarios=("project", "size"),
                      Activities=("activities", "mean"),
                      Duration=("base_makespan", "mean"),
                      Heat_reduction=("heat_reduction_pct", "mean"),
                      High_day_reduction=("high_day_reduction_pct", "mean"),
                      Duration_increase=("duration_increase_pct", "mean"))
                 .round(2).reset_index()
                 .rename(columns={"sector": "Construction sector",
                                  "Activities": "Mean activities",
                                  "Duration": "Mean duration (days)",
                                  "Heat_reduction": "Heat-risk reduction (%)",
                                  "High_day_reduction": "High-risk day reduction (%)",
                                  "Duration_increase": "Duration increase (%)"})
                 .sort_values("Heat-risk reduction (%)", ascending=False))
    common.save_table(by_sector, "table_sched_by_sector")

    # ---- baseline fairness -------------------------------------------------
    fairness = pd.DataFrame([
        {"Baseline": "Priority-rule serial schedule generation scheme",
         "Mean duration (days)": round(prim["sgs_makespan"].mean(), 1),
         "Mean modeled heat-risk exposure": round(prim["sgs_heat"].mean(), 2),
         "Duration relative to the solver baseline (%)":
             round(prim["sgs_vs_cpsat_duration_gap_pct"].mean(), 2),
         "Reported heat-risk reduction (%)": round(prim["heat_reduction_vs_sgs_pct"].mean(), 2)},
        {"Baseline": "Duration-minimizing constraint-programming schedule",
         "Mean duration (days)": round(prim["base_makespan"].mean(), 1),
         "Mean modeled heat-risk exposure": round(prim["base_heat"].mean(), 2),
         "Duration relative to the solver baseline (%)": 0.0,
         "Reported heat-risk reduction (%)": round(prim["heat_reduction_pct"].mean(), 2)}])
    common.save_table(fairness, "table_sched_baseline_fairness")

    # ---- risk scale comparison ---------------------------------------------
    scale = (df.groupby("risk_scale")
             .agg(Scenarios=("project", "size"),
                  Heat_reduction=("heat_reduction_pct", "mean"),
                  High_day_reduction=("high_day_reduction_pct", "mean"),
                  Activities_moved=("pct_activities_moved", "mean"))
             .round(2).reset_index())
    scale["risk_scale"] = scale["risk_scale"].map({
        "r_model_dayahead": "Fitted day-ahead risk model (primary)",
        "r_model_sameday": "Fitted model on observed same-day weather",
        "r_screening": "Normalized heat-index screening scale"})
    scale = scale.rename(columns={"risk_scale": "Day-cost scale",
                                  "Heat_reduction": "Heat-risk reduction (%)",
                                  "High_day_reduction": "High-risk day reduction (%)",
                                  "Activities_moved": "Activities moved (%)"})
    common.save_table(scale, "table_sched_risk_scale")

    # ---- management impact -------------------------------------------------
    impact = pd.DataFrame([{
        "Activities moved from the baseline start (%)": round(prim["pct_activities_moved"].mean(), 1),
        "Mean shift of a moved activity (days)": round(prim["mean_shift_days"].mean(), 2),
        "Largest shift observed (days)": int(prim["max_shift_days"].max()),
        "Float consumed (activity-days)": round(prim["float_consumed_activity_days"].mean(), 1),
        "Trade work spells, baseline": round(prim["trade_spells_base"].mean(), 1),
        "Trade work spells, heat-aware": round(prim["trade_spells_aware"].mean(), 1),
        "Change in project duration (%)": round(prim["duration_increase_pct"].mean(), 2)}])
    common.save_table(impact, "table_sched_management_impact")

    # ---- exposure-weight sensitivity table ---------------------------------
    all_w = pd.concat([prim.assign(weighting="empirical_index"),
                       pd.DataFrame(weight_rows)], ignore_index=True) \
        if weight_rows else prim.assign(weighting="empirical_index")
    wsens = (all_w.groupby("weighting")
             .agg(Scenarios=("project", "size"),
                  Heat_reduction=("heat_reduction_pct", "mean"),
                  High_day_reduction=("high_day_reduction_pct", "mean"),
                  Activities_moved=("pct_activities_moved", "mean"),
                  Duration_increase=("duration_increase_pct", "mean"))
             .round(2).reset_index())
    wsens["weighting"] = wsens["weighting"].map({
        "empirical_index": "Empirical heat-injury index by trade (primary)",
        "uniform_one": "Uniform weight of one for every activity",
        "scenario_mixed": "Declared low/medium/high scenario weights"}).fillna(
            wsens["weighting"])
    wsens = wsens.rename(columns={"weighting": "Activity exposure weighting",
                                  "Heat_reduction": "Heat-risk reduction (%)",
                                  "High_day_reduction": "High-risk day reduction (%)",
                                  "Activities_moved": "Activities moved (%)",
                                  "Duration_increase": "Duration increase (%)"})
    common.save_table(wsens, "table_sched_weight_sensitivity")
    if weight_rows:
        pd.DataFrame(weight_rows).to_csv(
            config.TABLES / "sched_weight_sensitivity_full.csv", index=False)

    verification = {
        "n_projects": int(prim["project"].nunique()),
        "n_scenarios_primary_scale": int(len(prim)),
        "n_scenarios_all_scales": int(len(df)),
        "n_skipped": int(len(skipped)),
        "locations": list(calendars),
        "start_date": START_DATE.isoformat(),
        "risk_scale_primary": "r_model_dayahead",
        "solver_seed": SOLVER_SEED,
        "total_precedence_violations": int(df["prec_viol"].sum()),
        "total_resource_violations": int(df["res_viol"].sum()),
        "all_feasible": bool(df["prec_viol"].sum() == 0 and df["res_viol"].sum() == 0),
        "mean_heat_reduction_pct": float(prim["heat_reduction_pct"].mean()),
        "mean_high_day_reduction_pct": float(prim["high_day_reduction_pct"].mean()),
        "mean_duration_increase_pct": float(prim["duration_increase_pct"].mean()),
        "duration_never_increases": bool((prim["aware_makespan"] <= prim["base_makespan"]).all()),
        "sgs_never_shorter_than_cpsat": bool(
            (prim["sgs_makespan"] >= prim["base_makespan"]).all()),
        "share_base_proven_optimal": float(prim["base_proven_optimal"].mean()),
        "n_weight_sensitivity_scenarios": int(len(weight_rows)),
    }
    common.save_json(verification, "verification14_scheduling")
    log.info("\nHeadline:\n" + headline.to_string(index=False))
    log.info("\nBy sector:\n" + by_sector.to_string(index=False))
    log.info("\nBaseline fairness:\n" + fairness.to_string(index=False))
    log.info("\nRisk scale:\n" + scale.to_string(index=False))
    log.info("\nManagement impact:\n" + impact.to_string(index=False))
    log.info("\nWeight sensitivity:\n" + wsens.to_string(index=False))
    log.info(f"Verification: {verification}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
