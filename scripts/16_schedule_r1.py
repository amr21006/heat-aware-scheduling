"""
16_schedule_r1.py — Heat-aware scheduling under retrospective and planning-time information.

For every empirical project network and location the script builds:
  * a duration-minimizing baseline (minimum makespan, then earliest starts);
  * heat-aware schedules that minimize crew-weighted modeled exposure within the baseline
    makespan, under the strategies listed in STRATEGIES:
      - retrospective strategies, which know the realized day cost of every future day;
      - planning-time strategies, which use only what is known when a start is committed:
        observed weather up to the previous day, NBM station forecasts for days 1-7 and the
        1991-2020 climatology beyond, with daily (or weekly) re-planning in which started and
        committed activities stay fixed.
Every executed schedule is then scored on observed 2023-2025 weather with measures that are
common to all strategies (see evaluate()).

Sensitivity strategies (added after the main run, reported as sensitivity analyses):
  * the injury-derived day cost without the upper bound of Equation (2) ("injury_u"), because the
    bound flattens most danger-level days at the hottest location to the same cost;
  * shift caps of 7, 28 and 42 days next to the 14-day cap, to trace the exposure-disturbance
    trade-off;
  * activity weights equal to the outdoor share of the crew (O*NET), so that the objective counts
    outdoor-weighted worker-days ("outdoor"), with the injury-derived cost and the heat-index rule;
  * a makespan allowance of 5% or 10% over the minimum makespan (sixth field of the strategy).
Running the script again computes only the strategies missing from each per-scenario file.

Solver: OR-Tools CP-SAT, one search worker per solve, fixed seed and deterministic time limits,
so results reproduce exactly; scenarios run in parallel processes.
Heat-aware solves are lexicographic: stage 1 minimizes exposure; stage 2 keeps exposure within
0.1% of the stage-1 value and minimizes the total absolute deviation from the reference schedule
(the baseline for a single plan, the previous plan when re-planning).

Outputs:
  outputs/r1_sched/<project>__<location>.json   starts and solve statistics (resume-safe)
  outputs/tables/sched_r1_metrics.csv            one row per scenario and strategy
  outputs/tables/verification16_scheduling.json
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
import warnings
from datetime import date, timedelta
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

import config
import common

warnings.filterwarnings("ignore")

START_DATE = date(2023, 7, 1)
CAL_DIR = config.DATA_EXTERNAL / "risk_calendars"
PROJ_DIR = config.DATA_INTERMEDIATE / "dslib_projects"
OUT_DIR = config.OUTPUTS / "r1_sched"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = config.RANDOM_SEED
SCALE = 1000
TOLERANCE = 0.001            # stage-2 exposure tolerance (0.1%)
SHIFT_CAP = 14               # calendar days, cap sensitivity
HOT_DAY_C = 32.2
HI_THRESHOLDS = (32.2, 39.4)  # NWS "extreme caution" and "danger" lower bounds (deg C)
LOCATIONS = list(config.SCHEDULE_LOCATIONS)

# name: (information, day cost, weights, re-planning, cap)
STRATEGIES = {
    "R-A": ("perfect", "injury", "share", None, False),
    "R-B": ("perfect", "injury", "uniform", None, False),
    "R-C": ("perfect", "heat_index", "share", None, False),
    "R-D": ("perfect", "heat_index", "uniform", None, False),
    "R-E": ("perfect", "injury", "rate", None, False),
    "R-F": ("perfect", "injury", "share", None, True),
    "O1": ("climatology", "injury", "share", None, False),
    "O1c": ("climatology", "injury", "share", None, True),
    "O2": ("dayahead", "injury", "share", 1, False),
    "O3": ("forecast", "injury", "share", 1, False),
    "O3-B": ("forecast", "injury", "uniform", 1, False),
    "O3-C": ("forecast", "heat_index", "share", 1, False),
    "O3-D": ("forecast", "heat_index", "uniform", 1, False),
    "O3-W": ("forecast", "injury", "share", 7, False),
    # sensitivity: day-cost scale without the upper bound
    "R-U": ("perfect", "injury_u", "share", None, False),
    "O1u": ("climatology", "injury_u", "share", None, False),
    "O3u": ("forecast", "injury_u", "share", 1, False),
    # sensitivity: shift caps (calendar days) around the 14-day cap
    "R-F7": ("perfect", "injury", "share", None, 7),
    "R-F28": ("perfect", "injury", "share", None, 28),
    "R-F42": ("perfect", "injury", "share", None, 42),
    "O1c7": ("climatology", "injury", "share", None, 7),
    "O1c28": ("climatology", "injury", "share", None, 28),
    "O1c42": ("climatology", "injury", "share", None, 42),
    # extension: activity weights equal to the outdoor share of the crew
    "R-O": ("perfect", "injury", "outdoor", None, False),
    "O1o": ("climatology", "injury", "outdoor", None, False),
    "O3o": ("forecast", "injury", "outdoor", 1, False),
    "O3-Do": ("forecast", "heat_index", "outdoor", 1, False),
    # extension: makespan allowance over the minimum makespan (percent)
    "R-M5": ("perfect", "injury", "share", None, False, 5),
    "R-M10": ("perfect", "injury", "share", None, False, 10),
    "O1m5": ("climatology", "injury", "share", None, False, 5),
    "O1m10": ("climatology", "injury", "share", None, False, 10),
    # start-date sensitivity: the heat-index rule under climatology
    "O1-D": ("climatology", "heat_index", "uniform", None, False),
}
START_STRATEGIES = ["R-A", "R-D", "O1", "O1-D"]   # strategies run for an alternative start date
COST_PREFIX = {"injury": "inj", "injury_u": "inju", "heat_index": "hi"}


def cap_days(capped) -> int | None:
    """Shift cap of a strategy: True means the 14-day cap, an integer a cap of that many days."""
    if capped is True:
        return SHIFT_CAP
    return int(capped) if capped else None


def static_limit(n: int) -> float:
    return 10.0 if n <= 40 else (20.0 if n <= 90 else 35.0)


REPLAN_LIMIT_1, REPLAN_LIMIT_2 = 2.0, 1.0
WALL_SAFETY_S = 3600.0     # wall-clock safety cap, far above any deterministic limit


def old_wall_cap(limit: float) -> float:
    """Wall-clock cap of the first runs (6 x limit, at least 60 s); under heavy machine load it could
    stop a solve before the deterministic limit. Solves that reached it are re-solved (--recheck)."""
    return max(60.0, 6 * limit)
LOOKAHEAD = 42             # six-week look-ahead window for re-planning (days)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class Net:
    def __init__(self, path: Path, ew, share_idx, rate_idx, rexp):
        net = json.loads(path.read_text(encoding="utf-8"))
        self.name, self.sector = net["code"], net["sector"].strip()
        s = self.sector
        if s.lower().startswith("construction (") and s.endswith(")"):
            self.sector = f"Construction ({s[len('construction ('):-1].strip().lower()})"
        res = net["resources"]
        self.cap = [int(r["capacity"]) for r in res]
        ridx = {r["name"]: i for i, r in enumerate(res)}
        self.jobs, self.dur, self.dem = [], {}, {}
        self.n_crew, self.n_out, self.w = {}, {}, {"share": {}, "uniform": {}, "rate": {}, "outdoor": {}}
        sector_naics = ew.sector_to_naics(net["sector"], net["name"])
        for a in net["activities"]:
            j = a["id"]
            self.jobs.append(j)
            self.dur[j] = int(a["duration"])
            d = [0] * len(self.cap)
            crew = outd = 0.0
            num = {"share": 0.0, "rate": 0.0}
            den = 0.0
            for dm in a["demands"]:
                u = int(dm["units"])
                d[ridx[dm["resource"]]] = u
                wpu, od = rexp.get(dm["resource"], (1.0, 0.0))
                crew += u * wpu
                outd += u * wpu * od
                naics = ew.trade_to_naics(dm["resource"])
                for key, idx in (("share", share_idx), ("rate", rate_idx)):
                    sw = idx.get(sector_naics, float(np.mean(list(idx.values()))))
                    num[key] += (idx.get(naics, sw) if naics else sw) * u
                den += u
            self.dem[j] = d
            self.n_crew[j], self.n_out[j] = crew, outd
            for key, idx in (("share", share_idx), ("rate", rate_idx)):
                sw = idx.get(sector_naics, float(np.mean(list(idx.values()))))
                self.w[key][j] = (num[key] / den) if den > 0 else sw
            self.w["uniform"][j] = 1.0
            self.w["outdoor"][j] = (outd / crew) if crew > 0 else 0.0
            if self.dur[j] == 0:
                for key in self.w:
                    self.w[key][j] = 0.0
        self.edges = [(e["pred"], e["succ"], int(e["lag"])) for e in net["precedence"]]
        self.pred = {j: [] for j in self.jobs}
        self.succ = {j: [] for j in self.jobs}
        for i, j, lag in self.edges:
            self.pred[j].append((i, lag))
            self.succ[i].append((j, lag))
        indeg = {j: len(self.pred[j]) for j in self.jobs}
        ready = [j for j in self.jobs if indeg[j] == 0]
        self.topo = []
        while ready:
            j = ready.pop(0)
            self.topo.append(j)
            for k, _ in self.succ[j]:
                indeg[k] -= 1
                if indeg[k] == 0:
                    ready.append(k)
        self.n = len(self.jobs)


def bounds(net: "Net", fixed: dict, lb: int, cmax: int):
    """Precedence-feasible earliest and latest starts given fixed starts, a release day lb for
    free activities and the deadline cmax (exact bounds; resources may tighten them further)."""
    es, ls = {}, {}
    for j in net.topo:
        if j in fixed:
            es[j] = fixed[j]
        else:
            es[j] = max([lb] + [es[p] + net.dur[p] + lag for p, lag in net.pred[j]])
    for j in reversed(net.topo):
        if j in fixed:
            ls[j] = fixed[j]
        else:
            ls[j] = min([cmax - net.dur[j]] + [ls[k] - lag - net.dur[j] for k, lag in net.succ[j]])
    return es, ls


def serial_sgs(net: Net):
    from collections import deque
    indeg = {j: 0 for j in net.jobs}
    succ = {j: [] for j in net.jobs}
    for i, j, _ in net.edges:
        succ[i].append(j)
        indeg[j] += 1
    rank = {j: r for r, j in enumerate(net.jobs)}
    ready = sorted([j for j in net.jobs if indeg[j] == 0], key=rank.get)
    order = []
    while ready:
        job = ready.pop(0)
        order.append(job)
        for s in succ[job]:
            indeg[s] -= 1
            if indeg[s] == 0:
                ready.append(s)
        ready.sort(key=rank.get)
    horizon = sum(net.dur.values()) + sum(abs(l) for *_, l in net.edges) + 2
    usage = np.zeros((horizon + 2, len(net.cap)), dtype=int)
    start = {}
    for job in order:
        est = max([start[p] + net.dur[p] + lag for p, lag in net.pred[job]] + [0])
        dd, req = net.dur[job], net.dem[job]
        if dd == 0:
            start[job] = est
            continue
        t = est
        while t + dd < horizon and any(usage[t:t + dd, k].max() + req[k] > net.cap[k]
                                        for k in range(len(net.cap)) if req[k]):
            t += 1
        usage[t:t + dd] += np.array(req)
        start[job] = t
    return start


# ---------------------------------------------------------------------------
# CP-SAT models
# ---------------------------------------------------------------------------
def base_model(net: Net, horizon: int, fixed: dict, lb: int, makespan_cap: int | None,
               es: dict | None = None, ls: dict | None = None):
    m = cp_model.CpModel()
    s, e, iv = {}, {}, {}
    for j in net.jobs:
        d = net.dur[j]
        if j in fixed:
            s[j] = m.NewConstant(fixed[j])
        elif es is not None:
            s[j] = m.NewIntVar(es[j], max(es[j], ls[j]), f"s{j}")
        else:
            s[j] = m.NewIntVar(lb, max(lb, horizon - d), f"s{j}")
        e[j] = m.NewIntVar(0, horizon, f"e{j}")
        iv[j] = m.NewIntervalVar(s[j], d, e[j], f"iv{j}")
    for i, j, lag in net.edges:
        m.Add(s[j] >= e[i] + lag)
    for k in range(len(net.cap)):
        m.AddCumulative([iv[j] for j in net.jobs], [net.dem[j][k] for j in net.jobs], net.cap[k])
    mk = m.NewIntVar(0, horizon, "mk")
    m.AddMaxEquality(mk, [e[j] for j in net.jobs])
    if makespan_cap is not None:
        m.Add(mk <= makespan_cap)
    return m, s, mk


def solver(limit: float) -> cp_model.CpSolver:
    sv = cp_model.CpSolver()
    sv.parameters.num_workers = 1
    sv.parameters.random_seed = SEED
    sv.parameters.max_deterministic_time = limit
    sv.parameters.max_time_in_seconds = WALL_SAFETY_S   # never binds: the deterministic limit decides
    sv.parameters.linearization_level = 0   # same optima, about 2.7x faster on these models
    return sv


def solve_baseline(net: Net, horizon: int, limit: float):
    m, s, mk = base_model(net, horizon, {}, 0, None)
    m.Minimize(mk)
    sv = solver(limit)
    st = sv.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    cmax, proven = int(sv.Value(mk)), st == cp_model.OPTIMAL
    hint = {j: sv.Value(s[j]) for j in net.jobs}
    m2, s2, mk2 = base_model(net, cmax, {}, 0, cmax)
    for j in net.jobs:
        m2.AddHint(s2[j], hint[j])
    m2.Minimize(sum(s2[j] for j in net.jobs))
    sv2 = solver(limit)
    st2 = sv2.Solve(m2)
    starts = {j: sv2.Value(s2[j]) for j in net.jobs} if st2 in (
        cp_model.OPTIMAL, cp_model.FEASIBLE) else hint
    return starts, cmax, proven


def act_costs(net: Net, weights: dict, cost: np.ndarray, es: dict, ls: dict, fixed: dict):
    """Integer exposure cost of each free activity for every start in [es_j, ls_j]."""
    cs = np.concatenate([[0.0], np.cumsum(cost)])
    out = {}
    for j in net.jobs:
        d, coef = net.dur[j], net.n_crew[j] * weights[j]
        if d == 0 or coef <= 0 or j in fixed:
            continue
        starts = np.arange(es[j], max(es[j], ls[j]) + 1)
        out[j] = np.rint(SCALE * coef * (cs[starts + d] - cs[starts])).astype(np.int64)
    return out


def solve_heat(net: Net, weights: dict, cost: np.ndarray, cmax: int, fixed: dict, lb: int,
               ref: dict, hint: dict | None, cap_ref: dict | None, lim1: float, lim2: float,
               cap: int = SHIFT_CAP):
    """Two-stage heat-aware solve. Returns (starts, stage-1 status optimal?)."""
    es, ls = bounds(net, fixed, lb, cmax)
    costs = act_costs(net, weights, cost, es, ls, fixed)

    def build():
        m, s, _ = base_model(net, cmax, fixed, lb, cmax, es, ls)
        terms = []
        for j, arr in costs.items():
            c = m.NewIntVar(int(arr.min()), int(arr.max()), f"c{j}")
            idx = m.NewIntVar(0, len(arr) - 1, f"x{j}")
            m.Add(idx == s[j] - es[j])
            m.AddElement(idx, [int(v) for v in arr], c)
            terms.append(c)
        if cap_ref is not None:
            for j in net.jobs:
                if j not in fixed and net.dur[j] > 0:
                    m.Add(s[j] <= cap_ref[j] + cap)
                    m.Add(s[j] >= cap_ref[j] - cap)
        if hint:
            for j in net.jobs:
                if j not in fixed:
                    m.AddHint(s[j], hint[j])
        return m, s, terms

    m, s, terms = build()
    if not terms:
        return dict(ref), True
    H = sum(terms)
    m.Minimize(H)
    sv = solver(lim1)
    st = sv.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, False
    h1 = int(sv.ObjectiveValue())
    first = {j: sv.Value(s[j]) for j in net.jobs}
    if all(first[j] == ref[j] for j in net.jobs if j not in fixed):
        return first, st == cp_model.OPTIMAL
    m2, s2, terms2 = build()
    for j in net.jobs:
        if j not in fixed:
            m2.AddHint(s2[j], first[j])
    m2.Add(sum(terms2) <= int(np.floor(h1 * (1 + TOLERANCE))))
    devs = []
    for j in net.jobs:
        if j in fixed:
            continue
        dv = m2.NewIntVar(0, cmax, f"d{j}")
        m2.AddAbsEquality(dv, s2[j] - ref[j])
        devs.append(dv)
    m2.Minimize(sum(devs))
    sv2 = solver(lim2)
    st2 = sv2.Solve(m2)
    if st2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {j: sv2.Value(s2[j]) for j in net.jobs}, st == cp_model.OPTIMAL
    return first, st == cp_model.OPTIMAL


# ---------------------------------------------------------------------------
# Information sets
# ---------------------------------------------------------------------------
def load_location(loc: str, models: dict, q: float):
    cal = pd.read_csv(CAL_DIR / f"{loc}_continuous.csv", parse_dates=["date"])
    cal = cal[cal["date"] >= pd.Timestamp(START_DATE)].reset_index(drop=True)
    sd, da = models["sameday"], models["dayahead"]
    p_sd = sd["pipeline"].predict_proba(cal[sd["columns"]])[:, 1]
    p_da = da["pipeline"].predict_proba(cal[da["columns"]])[:, 1]
    clim = pd.read_csv(CAL_DIR / f"{loc}_climatology_1991_2020.csv").set_index("doy")
    doy = cal["date"].dt.dayofyear.to_numpy()
    info = {
        "dates": cal["date"].dt.date.to_numpy(),
        "tmax": cal["tmax"].to_numpy(float), "hi": cal["heat_index_filled"].to_numpy(float),
        "inj_perfect": np.minimum(p_sd / q, 1.0),
        "inj_dayahead": np.minimum(p_da / q, 1.0),
        "inj_clim": np.minimum(clim.loc[doy, "p_sameday_clim"].to_numpy() / q, 1.0),
        "inju_perfect": p_sd / q,
        "inju_clim": clim.loc[doy, "p_sameday_clim"].to_numpy() / q,
        "hi_perfect": cal["r_screening"].to_numpy(float),
        "hi_clim": clim.loc[doy, "r_screening_clim"].to_numpy(float),
        "r_da_own": np.minimum(p_da / models["dayahead"]["reference_p99"], 1.0),
    }
    info["fc"] = forecast_costs(loc, cal, sd, q)
    return info


def forecast_costs(loc: str, cal: pd.DataFrame, sd: dict, q: float) -> dict:
    """Same-day injury-model cost and heat-index cost from the NBM bulletin issued the day
    before each planning day, for leads 1-7. Returns {plan_index: (inj[7], hi[7], inj_u[7])},
    where inj_u is the injury cost without the upper bound."""
    m06 = import_module("06_dayahead_prediction")
    cal09 = import_module("09_build_risk_calendars")
    fc = pd.read_parquet(config.DATA_EXTERNAL / "nbm" / "nbm_forecasts.parquet")
    fc = fc[fc["location"] == loc].copy()
    fc["issue_date"] = pd.to_datetime(fc["issue_date"]).dt.date
    fc["date"] = pd.to_datetime(fc["date"]).dt.date
    obs_tmax = dict(zip(cal["date"].dt.date, cal["tmax"]))
    rows, keys = [], []
    for issue, g in fc.groupby("issue_date"):
        g = g.set_index("date").sort_index()
        plan_day = issue + timedelta(days=1)
        tau = (plan_day - START_DATE).days
        if tau < 0:
            continue
        for k in range(1, 8):
            t = issue + timedelta(days=k)
            if t not in g.index or pd.isna(g.at[t, "tmax_c"]):
                continue
            tx, tn, td = g.at[t, "tmax_c"], g.at[t, "tmin_c"], g.at[t, "dpt_c"]
            rh = float(100 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * tx / (243.04 + tx)))
            hi = common.heat_index_c(tx, min(rh, 100.0))
            q24 = g.at[t, "q24_in"]

            def tmax_on(dd):
                if dd <= issue:
                    return obs_tmax.get(dd, np.nan)
                return g.at[dd, "tmax_c"] if dd in g.index else np.nan
            lag1 = tmax_on(t - timedelta(days=1))
            prior = [tmax_on(t - timedelta(days=i)) for i in range(1, 8)]
            hot = [1.0 if (v is not None and not np.isnan(v) and v >= HOT_DAY_C) else 0.0
                   for v in prior]
            doy = t.timetuple().tm_yday
            rows.append({"tmax": tx, "tmin": tn if not pd.isna(tn) else tx - 10.0,
                         "heat_index_filled": hi,
                         "prcp_flag": float(q24 is not None and not pd.isna(q24) and q24 >= 0.01),
                         "hot_days_prior3": sum(hot[:3]), "hot_days_prior7": sum(hot),
                         "tmax_lag1": lag1 if not np.isnan(lag1) else tx,
                         "doy_sin": np.sin(2 * np.pi * doy / 365.25),
                         "doy_cos": np.cos(2 * np.pi * doy / 365.25)})
            keys.append((tau, k))
    X = pd.DataFrame(rows)
    X["tmax_sq"] = X["tmax"] ** 2
    X["hi_sq"] = X["heat_index_filled"] ** 2
    X["tmax_x_hi"] = X["tmax"] * X["heat_index_filled"]
    p = sd["pipeline"].predict_proba(X[sd["columns"]])[:, 1]
    r_hi = X["heat_index_filled"].apply(cal09.screening_risk).to_numpy()
    out = {}
    for (tau, k), pv, hv in zip(keys, p, r_hi):
        inj, hh, inju = out.setdefault(tau, (np.full(7, np.nan), np.full(7, np.nan),
                                             np.full(7, np.nan)))
        inj[k - 1] = min(pv / q, 1.0)
        hh[k - 1] = hv
        inju[k - 1] = pv / q
    return out


def cost_vector(info: dict, information: str, daycost: str, tau: int, n: int) -> np.ndarray:
    """Day costs as known on the evening before day tau, for days 0..n-1."""
    pre = COST_PREFIX[daycost]
    if information == "perfect":
        return info[f"{pre}_perfect"][:n].copy()
    c = info[f"{pre}_clim"][:n].copy()
    if information == "climatology":
        return c
    if information == "dayahead":
        if tau < n:
            c[tau] = info["inj_dayahead"][tau]
        return c
    fc = info["fc"].get(tau)
    if fc is not None:
        arr = {"inj": fc[0], "hi": fc[1], "inju": fc[2]}[pre]
        for k in range(7):
            t = tau + k
            if t < n and not np.isnan(arr[k]):
                c[t] = arr[k]
    return c


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
def run_static(net, info, spec, cmax, base, limit):
    information, daycost, wkey, _, capped = spec[:5]
    allowance = spec[5] if len(spec) > 5 else 0
    if allowance:              # deadline relaxed by a percentage of the minimum makespan
        cmax = min(cmax + int(round(cmax * allowance / 100)), len(info["dates"]) - 2)
    cost = cost_vector(info, information, daycost, 0, cmax + 2)
    cd = cap_days(capped)
    starts, opt = solve_heat(net, net.w[wkey], cost, cmax, {}, 0, base, base,
                             base if cd else None, limit, limit, cd or SHIFT_CAP)
    return starts, {"solves": 1, "stage1_optimal": int(opt)}


def run_rolling(net, info, spec, cmax, base, step):
    """Re-planning with a look-ahead window: on the evening before day tau, started activities
    are fixed, activities planned to start within the next LOOKAHEAD days are re-optimized with
    the information available then, and activities planned later keep their planned starts until
    they enter the window. Starts planned for the coming re-planning interval are committed."""
    information, daycost, wkey, _, _ = spec[:5]
    plan, fixed = dict(base), {}
    solves = optimal = 0
    tau = 0
    while len(fixed) < net.n and tau <= cmax:
        free = [j for j in net.jobs if j not in fixed]
        window = [j for j in free if plan[j] <= tau + LOOKAHEAD - 1]
        if step == 1:
            eligible = any(all(p in fixed and fixed[p] + net.dur[p] + lag <= tau
                               for p, lag in net.pred[j]) for j in window)
        else:
            eligible = bool(window)
        if eligible:
            frozen = dict(fixed)
            frozen.update({j: plan[j] for j in free if j not in window})
            cost = cost_vector(info, information, daycost, tau, cmax + 2)
            new, opt = solve_heat(net, net.w[wkey], cost, cmax, frozen, tau, plan, plan, None,
                                  REPLAN_LIMIT_1, REPLAN_LIMIT_2)
            if new is not None:
                plan = new
                solves += 1
                optimal += int(opt)
        horizon_end = tau + step - 1
        for j in free:
            if plan[j] <= horizon_end:
                fixed[j] = plan[j]
        tau += step
    for j in net.jobs:
        fixed.setdefault(j, plan[j])
    return fixed, {"solves": solves, "stage1_optimal": optimal}


# ---------------------------------------------------------------------------
# Evaluation on observed weather
# ---------------------------------------------------------------------------
def evaluate(net, starts, info, ro, base):
    hi, tmax = info["hi"], info["tmax"]
    rel = ro[np.clip(np.rint(tmax * 10).astype(int) + 300, 0, len(ro) - 1)]
    inj = info["inj_perfect"]
    rda = info["r_da_own"]
    acc = {f"E1o_{t}": 0.0 for t in HI_THRESHOLDS} | {f"E1n_{t}": 0.0 for t in HI_THRESHOLDS}
    acc |= {"E2o": 0.0, "E2n": 0.0, "Hinj": 0.0, "worker_days": 0.0, "outdoor_worker_days": 0.0,
            "HR_orig": 0.0} | {f"HR_{th:.1f}": 0.0 for th in (0.3, 0.4, 0.5, 0.6, 0.7)}
    for j in net.jobs:
        d = net.dur[j]
        if d == 0:
            continue
        sl = slice(starts[j], starts[j] + d)
        n, o, w = net.n_crew[j], net.n_out[j], net.w["share"][j]
        acc["worker_days"] += n * d
        acc["outdoor_worker_days"] += o * d
        for t in HI_THRESHOLDS:
            k = float((hi[sl] >= t).sum())
            acc[f"E1o_{t}"] += o * k
            acc[f"E1n_{t}"] += n * k
        acc["E2o"] += o * rel[sl].sum()
        acc["E2n"] += n * rel[sl].sum()
        acc["Hinj"] += n * w * inj[sl].sum()
        acc["HR_orig"] += float((rda[sl] >= 0.5).sum())
        for th in (0.3, 0.4, 0.5, 0.6, 0.7):
            acc[f"HR_{th:.1f}"] += float((inj[sl] >= th).sum())
    shifts = np.array([starts[j] - base[j] for j in net.jobs if net.dur[j] > 0])
    moved = shifts[shifts != 0]
    acc["makespan"] = max(starts[j] + net.dur[j] for j in net.jobs)
    acc["n_activities"] = int(len(shifts))
    acc["moved"] = int(len(moved))
    acc["moved_earlier"] = int((moved < 0).sum())
    acc["mean_abs_shift"] = float(np.abs(moved).mean()) if len(moved) else 0.0
    acc["median_abs_shift"] = float(np.median(np.abs(moved))) if len(moved) else 0.0
    acc["max_abs_shift"] = int(np.abs(moved).max()) if len(moved) else 0
    acc["abs_shift_total"] = int(np.abs(shifts).sum())
    acc["prec_viol"] = sum(1 for i, j, lag in net.edges if starts[j] < starts[i] + net.dur[i] + lag)
    use = {}
    for j in net.jobs:
        for t in range(starts[j], starts[j] + net.dur[j]):
            for k, u in enumerate(net.dem[j]):
                if u:
                    use[(t, k)] = use.get((t, k), 0) + u
    acc["res_viol"] = sum(1 for (t, k), u in use.items() if u > net.cap[k])
    return acc


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
_G = {}


def set_start(start: str | None):
    """Use an alternative project start date, with its own output folder."""
    global START_DATE, OUT_DIR
    if start:
        START_DATE = date.fromisoformat(start)
        OUT_DIR = config.OUTPUTS / f"r1_sched_start{START_DATE:%Y%m%d}"
        OUT_DIR.mkdir(parents=True, exist_ok=True)


def init_worker(start: str | None = None):
    set_start(start)
    try:                                     # keep the machine responsive during long runs
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass
    ew = import_module("07_exposure_weights")
    share = json.loads((config.DATA_INTERMEDIATE / "exposure_weight_index.json")
                       .read_text(encoding="utf-8"))["exposure_index_by_naics4"]
    rate = json.loads((config.DATA_INTERMEDIATE / "rate_weight_index.json")
                      .read_text(encoding="utf-8"))["rate_index_by_naics4"]
    rx = pd.read_csv(config.DATA_INTERMEDIATE / "resource_exposure.csv")
    rexp = {r.resource: (float(r.workers_per_unit), float(r.outdoor_every_day))
            for r in rx.itertuples()}
    models = {k: joblib.load(config.MODELS / f"risk_model_{k}.joblib")
              for k in ["sameday", "dayahead"]}
    q = models["sameday"]["reference_p99"]
    curve = pd.read_csv(config.DATA_INTERMEDIATE / "eval_exposure_response_2015_2022.csv")
    _G.update(ew=ew, share=share, rate=rate, rexp=rexp, models=models, q=q,
              ro=curve["relative_odds"].to_numpy(), info={})


def run_strategies(net, info, names, cmax, base, lim):
    starts, stats = {}, {}
    for name in names:
        spec = STRATEGIES[name]
        ts = time.time()
        if spec[3] is None:
            st, info_s = run_static(net, info, spec, cmax, base, lim)
        else:
            st, info_s = run_rolling(net, info, spec, cmax, base, spec[3])
        info_s["seconds"] = round(time.time() - ts, 1)
        starts[name] = st if st is not None else base
        stats[name] = info_s
    return starts, stats


def _retry(fn, tries=60, wait=0.5):
    """Windows reports a file that another process has open as busy; wait and try again."""
    for k in range(tries):
        try:
            return fn()
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(wait)


def write_json(out: Path, rec: dict):
    """Write atomically, so an interrupted run never leaves a truncated file."""
    tmp = out.with_suffix(".tmp")
    _retry(lambda: tmp.write_text(json.dumps(rec), encoding="utf-8"))
    _retry(lambda: tmp.replace(out))


def read_json(out: Path) -> dict:
    return _retry(lambda: json.loads(out.read_text(encoding="utf-8")))


def task_cost(path: str, name: str | None = None) -> float:
    """Rough cost used to start the largest tasks first: activities, times three for re-planning."""
    n = len(json.loads(Path(path).read_text(encoding="utf-8"))["activities"])
    return n * (3.0 if name and STRATEGIES[name][3] is not None else 1.0)


def run_part(args):
    """One missing strategy of one existing scenario; the parent process merges the result.
    The baseline and makespan come with the task, so workers never read the files being merged."""
    path, loc, name, base_s, cmax = args
    out = OUT_DIR / f"{Path(path).stem}__{loc}.json"
    if loc not in _G["info"]:
        _G["info"][loc] = load_location(loc, _G["models"], _G["q"])
    info = _G["info"][loc]
    net = Net(Path(path), _G["ew"], _G["share"], _G["rate"], _G["rexp"])
    key = {str(j): j for j in net.jobs}
    base = {key[j]: int(v) for j, v in base_s.items()}
    starts, stats = run_strategies(net, info, [name], cmax, base, static_limit(net.n))
    st = starts[name]
    return {"out": str(out), "name": name, "metrics": evaluate(net, st, info, _G["ro"], base),
            "starts": {str(j): int(v) for j, v in st.items()}, "stats": stats[name]}


def extend_task(path, loc, rec, missing, out):
    """Add the strategies missing from an existing per-scenario file, against its stored baseline."""
    if loc not in _G["info"]:
        _G["info"][loc] = load_location(loc, _G["models"], _G["q"])
    info = _G["info"][loc]
    net = Net(Path(path), _G["ew"], _G["share"], _G["rate"], _G["rexp"])
    key = {str(j): j for j in net.jobs}
    base = {key[j]: int(v) for j, v in rec["starts"]["O0"].items()}
    t0 = time.time()
    starts, stats = run_strategies(net, info, missing, rec["cmax"], base, static_limit(net.n))
    for k, v in starts.items():
        rec["metrics"][k] = evaluate(net, v, info, _G["ro"], base)
        rec["starts"][k] = {str(j): int(s) for j, s in v.items()}
        rec["stats"][k] = stats[k]
    rec["seconds_extended"] = round(rec.get("seconds_extended", 0.0) + time.time() - t0, 1)
    write_json(out, rec)
    return rec


def run_task(args):
    path, loc, strategies = args
    out = OUT_DIR / f"{Path(path).stem}__{loc}.json"
    if out.exists():
        rec = read_json(out)
        missing = [k for k in strategies if "metrics" in rec and k not in rec["metrics"]]
        return extend_task(path, loc, rec, missing, out) if missing else rec
    if loc not in _G["info"]:
        _G["info"][loc] = load_location(loc, _G["models"], _G["q"])
    info = _G["info"][loc]
    net = Net(Path(path), _G["ew"], _G["share"], _G["rate"], _G["rexp"])
    t0 = time.time()
    sgs = serial_sgs(net)
    m_sgs = max(sgs[j] + net.dur[j] for j in net.jobs)
    ndays = len(info["dates"])
    rec = {"project": net.name, "sector": net.sector, "location": loc,
           "activities": net.n, "resources": len(net.cap)}
    if m_sgs + 2 > ndays:
        rec["skipped"] = f"priority-rule schedule of {m_sgs} days exceeds the {ndays}-day calendar"
        write_json(out, rec)
        return rec
    lim = static_limit(net.n)
    b = solve_baseline(net, m_sgs, lim)
    if b is None:
        rec["skipped"] = "no feasible baseline"
        write_json(out, rec)
        return rec
    base, cmax, proven = b
    rec.update(cmax=cmax, base_proven_optimal=bool(proven), sgs_makespan=m_sgs)
    starts, stats = run_strategies(net, info, strategies, cmax, base, lim)
    starts = {"O0": base, **starts}
    rec["metrics"] = {k: evaluate(net, v, info, _G["ro"], base) for k, v in starts.items()}
    rec["starts"] = {k: {str(j): int(s) for j, s in v.items()} for k, v in starts.items()}
    rec["stats"] = stats
    rec["seconds"] = round(time.time() - t0, 1)
    write_json(out, rec)
    return rec


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--locations", nargs="*", default=None,
                    help="restrict the run to these locations (partial run: no summary files)")
    ap.add_argument("--extend-only", action="store_true",
                    help="only add missing strategies to existing per-scenario files (partial run)")
    ap.add_argument("--start", default=None,
                    help="alternative project start date (YYYY-MM-DD); runs START_STRATEGIES only")
    ap.add_argument("--recheck", action="store_true",
                    help="re-solve single-plan strategies whose recorded time reached the old wall-clock "
                         "cap, so that the deterministic limit decides (partial run)")
    ap.add_argument("--split", action="store_true",
                    help="add missing strategies to existing files as (scenario, strategy) tasks, "
                         "largest first, merged by this process (partial run)")
    a = ap.parse_args(argv)
    set_start(a.start)
    log = common.setup_logger("16_schedule_r1")
    common.banner(log, "STEP 16 - HEAT-AWARE SCHEDULING, RETROSPECTIVE AND PLANNING-TIME")
    projects = sorted(PROJ_DIR.glob("*.json"))
    names = START_STRATEGIES if a.start else [k for k in STRATEGIES if k != "O1-D"]
    tasks = [(str(p), loc, names) for loc in (a.locations or LOCATIONS) for p in projects]
    if a.extend_only:
        tasks = [t for t in tasks if (OUT_DIR / f"{Path(t[0]).stem}__{t[1]}.json").exists()]
    partial = bool(a.locations) or a.extend_only or a.split or a.recheck
    tasks.sort(key=lambda t: -task_cost(t[0]))          # largest projects first
    if a.split or a.recheck:
        parts = []
        for path, loc, nm in tasks:
            out = OUT_DIR / f"{Path(path).stem}__{loc}.json"
            if not out.exists():
                continue
            rec = read_json(out)
            if "metrics" not in rec:
                continue
            if a.recheck:
                cap = old_wall_cap(static_limit(rec["activities"]))
                parts += [(path, loc, k, rec["starts"]["O0"], rec["cmax"]) for k, v in rec["stats"].items()
                          if STRATEGIES[k][3] is None and v["seconds"] >= 0.95 * cap]
            else:
                parts += [(path, loc, k, rec["starts"]["O0"], rec["cmax"]) for k in nm
                          if k not in rec["metrics"]]
        parts.sort(key=lambda t: -task_cost(t[0], t[2]))
        log.info(f"{len(parts)} scenario-strategy tasks, {a.workers} processes")
        t0 = time.time()
        with mp.Pool(a.workers, initializer=init_worker, initargs=(a.start,)) as pool:
            for k, r in enumerate(pool.imap_unordered(run_part, parts), 1):
                out = Path(r["out"])
                rec = read_json(out)
                rec["metrics"][r["name"]] = r["metrics"]
                rec["starts"][r["name"]] = r["starts"]
                rec["stats"][r["name"]] = r["stats"]
                write_json(out, rec)
                if k % 25 == 0:
                    log.info(f"  {k}/{len(parts)} done ({time.time() - t0:.0f}s) {out.stem} {r['name']}")
        log.info("split run finished; summary files are written by a complete run")
        return 0
    if a.pilot:
        pick = {"C2011-12", "C2012-04", "C2025-13"}
        tasks = [t for t in tasks if Path(t[0]).stem in pick and t[1] in ("TX_Austin", "GA_Atlanta")]
    log.info(f"{len(tasks)} project-location tasks, {len(STRATEGIES)} strategies, "
             f"{a.workers} processes")
    t0 = time.time()
    results = []
    with mp.Pool(a.workers, initializer=init_worker, initargs=(a.start,)) as pool:
        for k, r in enumerate(pool.imap_unordered(run_task, tasks), 1):
            results.append(r)
            if k % 10 == 0 or a.pilot:
                log.info(f"  {k}/{len(tasks)} done ({time.time() - t0:.0f}s) "
                         f"{r['project']} {r['location']} {r.get('seconds', r.get('skipped'))}")
    if partial:
        log.info("partial run finished; summary files are written by a complete run")
        return 0
    rows = []
    for r in results:
        if "metrics" not in r:
            continue
        for strat, mtr in r["metrics"].items():
            rows.append({"project": r["project"], "sector": r["sector"], "location": r["location"],
                         "activities": r["activities"], "cmax": r["cmax"],
                         "base_proven_optimal": r["base_proven_optimal"], "strategy": strat,
                         **mtr, **{f"stat_{k}": v for k, v in r["stats"].get(strat, {}).items()}})
    df = pd.DataFrame(rows)
    name = ("sched_r1_metrics_pilot" if a.pilot else
            f"sched_r1_metrics_start{START_DATE:%Y%m%d}" if a.start else "sched_r1_metrics")
    df.to_csv(config.TABLES / f"{name}.csv", index=False)
    skipped = [{"project": r["project"], "location": r["location"], "reason": r["skipped"]}
               for r in results if "skipped" in r]
    verif = {"tasks": len(tasks), "completed": int(df[["project", "location"]].drop_duplicates()
                                                   .shape[0]) if len(df) else 0,
             "skipped": skipped,
             "precedence_violations": int(df["prec_viol"].sum()) if len(df) else None,
             "resource_violations": int(df["res_viol"].sum()) if len(df) else None,
             "makespan_exceeds_baseline": int((df["makespan"] > df["cmax"]).sum()) if len(df) else None,
             "wall_seconds": round(time.time() - t0, 1)}
    common.save_json(verif, "verification16_scheduling_pilot" if a.pilot else
                     f"verification16_scheduling_start{START_DATE:%Y%m%d}" if a.start else
                     "verification16_scheduling")
    log.info(f"verification: {verif}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
