"""
16c_solver_budget.py — Optimality gaps and sensitivity of the results to the solver time budget.

Single-plan heat-aware solves stop at a deterministic time limit; when the limit is reached
before optimality is proven, the reported schedule is the best one found. This step takes a
stratified sample of scenarios whose stage-1 solve was not proven optimal, for the injury-derived
cost (R-A) and the heat-index rule (R-D) with perfect information, and:
  * re-solves stage 1 with the standard limit, recording the objective and the best bound
    (optimality gap) and checking that the stored schedule is reproduced;
  * re-solves both stages with four times the standard limit, recording the gap and scoring the
    resulting schedule with the common measures.
Outputs:
  outputs/tables/solver_budget_sensitivity.csv
  outputs/tables/verification16c_solver_budget.json
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

import config
import common

s16 = import_module("16_schedule_r1")
FACTOR = 4.0
PER_STRATUM = 10          # sampled scenarios per size class and strategy
SIZES = [("<=40", 0, 40), ("41-90", 41, 90), (">90", 91, 10 ** 6)]


def solve_with_bound(net, weights, cost, cmax, base, lim):
    """Stage 1 and stage 2 as in 16_schedule_r1.solve_heat, returning the stage-1 objective
    and best bound as well as the final starts."""
    es, ls = s16.bounds(net, {}, 0, cmax)
    costs = s16.act_costs(net, weights, cost, es, ls, {})

    def build():
        m, s, _ = s16.base_model(net, cmax, {}, 0, cmax, es, ls)
        terms = []
        for j, arr in costs.items():
            c = m.NewIntVar(int(arr.min()), int(arr.max()), f"c{j}")
            idx = m.NewIntVar(0, len(arr) - 1, f"x{j}")
            m.Add(idx == s[j] - es[j])
            m.AddElement(idx, [int(v) for v in arr], c)
            terms.append(c)
        for j in net.jobs:
            m.AddHint(s[j], base[j])
        return m, s, terms

    m, s, terms = build()
    m.Minimize(sum(terms))
    sv = s16.solver(lim)
    st = sv.Solve(m)
    obj, bnd = sv.ObjectiveValue(), sv.BestObjectiveBound()
    first = {j: sv.Value(s[j]) for j in net.jobs}
    out = {"optimal": st == cp_model.OPTIMAL, "objective": obj, "bound": bnd,
           "gap_pct": 100 * (obj - bnd) / obj if obj > 0 else 0.0}
    if all(first[j] == base[j] for j in net.jobs):
        return first, out
    m2, s2, terms2 = build()
    for j in net.jobs:
        m2.AddHint(s2[j], first[j])
    m2.Add(sum(terms2) <= int(np.floor(obj * (1 + s16.TOLERANCE))))
    devs = []
    for j in net.jobs:
        dv = m2.NewIntVar(0, cmax, f"d{j}")
        m2.AddAbsEquality(dv, s2[j] - base[j])
        devs.append(dv)
    m2.Minimize(sum(devs))
    sv2 = s16.solver(lim)
    st2 = sv2.Solve(m2)
    if st2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {j: sv2.Value(s2[j]) for j in net.jobs}, out
    return first, out


def run_item(args):
    path, loc, name = args
    rec = json.loads((s16.OUT_DIR / f"{Path(path).stem}__{loc}.json").read_text(encoding="utf-8"))
    G = s16._G
    if loc not in G["info"]:
        G["info"][loc] = s16.load_location(loc, G["models"], G["q"])
    info = G["info"][loc]
    net = s16.Net(Path(path), G["ew"], G["share"], G["rate"], G["rexp"])
    key = {str(j): j for j in net.jobs}
    base = {key[j]: int(v) for j, v in rec["starts"]["O0"].items()}
    information, daycost, wkey, _, _ = s16.STRATEGIES[name][:5]
    cmax = rec["cmax"]
    cost = s16.cost_vector(info, information, daycost, 0, cmax + 2)
    lim = s16.static_limit(net.n)
    t0 = time.time()
    st1, o1 = solve_with_bound(net, net.w[wkey], cost, cmax, base, lim)
    st4, o4 = solve_with_bound(net, net.w[wkey], cost, cmax, base, FACTOR * lim)
    stored = {key[j]: int(v) for j, v in rec["starts"][name].items()}
    m0 = rec["metrics"]["O0"]
    m1 = s16.evaluate(net, st1, info, G["ro"], base)
    m4 = s16.evaluate(net, st4, info, G["ro"], base)
    red = lambda m, k: 100 * (m0[k] - m[k]) / m0[k] if m0[k] > 0 else np.nan
    return {"project": rec["project"], "location": loc, "activities": net.n, "strategy": name,
            "reproduces_stored": st1 == stored,
            "gap_1x_pct": o1["gap_pct"], "optimal_1x": o1["optimal"],
            "gap_4x_pct": o4["gap_pct"], "optimal_4x": o4["optimal"],
            "objective_change_pct": 100 * (o4["objective"] - o1["objective"]) / o1["objective"]
            if o1["objective"] > 0 else 0.0,
            "E1o_red_1x": red(m1, "E1o_32.2"), "E1o_red_4x": red(m4, "E1o_32.2"),
            "E1o394_1x": m1["E1o_39.4"], "E1o394_4x": m4["E1o_39.4"], "E1o394_base": m0["E1o_39.4"],
            "seconds": round(time.time() - t0, 1)}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args(argv)
    log = common.setup_logger("16c_solver_budget")
    common.banner(log, "STEP 16c - OPTIMALITY GAPS AND SOLVER-BUDGET SENSITIVITY")
    rng = np.random.default_rng(config.RANDOM_SEED)
    items = []
    for name in ("R-A", "R-D"):
        pool = {g: [] for g, *_ in SIZES}
        for f in sorted(s16.OUT_DIR.glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            if "stats" not in r or name not in r["stats"] or r["stats"][name]["stage1_optimal"]:
                continue
            for g, lo, hi in SIZES:
                if lo <= r["activities"] <= hi:
                    pool[g].append((str(s16.PROJ_DIR / f"{r['project']}.json"), r["location"], name))
        for g, cand in pool.items():
            if cand:
                pick = rng.choice(len(cand), min(PER_STRATUM, len(cand)), replace=False)
                items += [cand[i] for i in sorted(pick)]
    items.sort(key=lambda t: -s16.task_cost(t[0]))
    log.info(f"{len(items)} sampled non-optimal solves, {a.workers} processes, budget x{FACTOR:g}")
    rows = []
    with mp.Pool(a.workers, initializer=s16.init_worker) as pool:
        for k, r in enumerate(pool.imap_unordered(run_item, items), 1):
            rows.append(r)
            if k % 10 == 0:
                log.info(f"  {k}/{len(items)} done")
    df = pd.DataFrame(rows)
    common.save_table(df.round(4), "solver_budget_sensitivity")
    d = df["E1o_red_4x"] - df["E1o_red_1x"]
    verif = {"items": int(len(df)), "reproduces_stored_share": float(df.reproduces_stored.mean()),
             "gap_1x_median": float(df.gap_1x_pct.median()), "gap_1x_p90": float(df.gap_1x_pct.quantile(0.9)),
             "gap_4x_median": float(df.gap_4x_pct.median()),
             "optimal_4x_share": float(df.optimal_4x.mean()),
             "objective_change_median_pct": float(df.objective_change_pct.median()),
             "E1o_red_change_mean": float(d.mean()), "E1o_red_change_median": float(d.median()),
             "E1o_red_change_abs_p90": float(d.abs().quantile(0.9))}
    common.save_json(verif, "verification16c_solver_budget")
    log.info(f"verification: {verif}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
