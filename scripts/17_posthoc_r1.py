"""
17_posthoc_r1.py — Analyses of the saved schedules (no further optimization).

  1. Activity-level comparison with the baseline: direction and size of start shifts, total
     float of moved activities (precedence network, baseline makespan as deadline), Kendall rank
     correlation of start order, and the change in outdoor worker-days on hot days by trade
     weight class.
  2. Heat-dependent durations: every schedule is re-timed with daily productivity reduced on hot
     days for the outdoor share of each crew, keeping the planned activity order, planned start
     as earliest start, precedence lags and resource capacities. Two declared loss scenarios by
     NWS heat-index band: moderate (10% at 32.2-39.4 C, 25% at 39.4 C and above) and severe
     (25% and 50%). The re-timed schedules are also scored on the primary outcome and at 39.4 C,
     so that the exposure reduction is reported under heat-dependent durations as well.
  3. Worker-hours: the primary outcome with each outdoor worker weighted by the daily hours of its
     construction subsector (BLS Current Employment Statistics, step 15b), for every schedule.

Outputs:
  outputs/tables/posthoc_activity_level.csv
  outputs/tables/posthoc_trade_class.csv
  outputs/tables/posthoc_heat_durations.csv
  outputs/tables/posthoc_gantt_example.csv
  outputs/tables/posthoc_worker_hours.csv
  outputs/tables/verification17_posthoc.json
"""
from __future__ import annotations

import json
import sys
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import config
import common

warnings.filterwarnings("ignore")
log = common.setup_logger("17_posthoc_r1")
s16 = import_module("16_schedule_r1")

LOSS = {"moderate": (0.10, 0.25), "severe": (0.25, 0.50)}
COMPARE = ["R-A", "R-B", "R-C", "R-D", "R-E", "R-F", "O1", "O1c", "O2", "O3", "O3-B", "O3-C", "O3-D", "O3-W",
           "R-U", "O1u", "O3u", "R-F7", "R-F28", "R-F42", "O1c7", "O1c28", "O1c42",
           "R-O", "O1o", "O3o", "O3-Do", "R-M5", "R-M10", "O1m5", "O1m10"]
RETIME = ["O0", "R-A", "O1", "O3", "O3-D"]


def total_float(net, cmax):
    es = {}
    order = sorted(net.jobs, key=lambda j: 0)
    indeg = {j: len(net.pred[j]) for j in net.jobs}
    succ = {j: [] for j in net.jobs}
    for i, j, lag in net.edges:
        succ[i].append((j, lag))
    ready = [j for j in net.jobs if indeg[j] == 0]
    topo = []
    while ready:
        j = ready.pop()
        topo.append(j)
        for s, _ in succ[j]:
            indeg[s] -= 1
            if indeg[s] == 0:
                ready.append(s)
    for j in topo:
        es[j] = max([es[p] + net.dur[p] + lag for p, lag in net.pred[j]] + [0])
    ls = {}
    for j in reversed(topo):
        ls[j] = min([ls[s] - lag - net.dur[j] for s, lag in succ[j]] + [cmax - net.dur[j]])
    return {j: ls[j] - es[j] for j in net.jobs}


def retime(net, starts, hi, loss):
    lo, hi_loss = loss
    day_loss = np.where(hi >= 39.4, hi_loss, np.where(hi >= 32.2, lo, 0.0))
    # precedence-consistent order: among activities whose predecessors are placed, the earliest
    # planned start first (lags can be negative, so planned starts alone are not an order)
    import heapq
    indeg = {j: len(net.pred[j]) for j in net.jobs}
    rank = {j: i for i, j in enumerate(net.jobs)}
    heap = [(starts[j], rank[j], j) for j in net.jobs if indeg[j] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, _, j = heapq.heappop(heap)
        order.append(j)
        for k, _lag in net.succ[j]:
            indeg[k] -= 1
            if indeg[k] == 0:
                heapq.heappush(heap, (starts[k], rank[k], k))
    horizon = len(hi) - 1
    usage = np.zeros((horizon + 400, len(net.cap)), dtype=int)
    hi = np.concatenate([hi, np.zeros(400)])        # days after the calendar count as not hot
    fin, span = {}, {}
    for j in order:
        d = net.dur[j]
        f = (net.n_out[j] / net.n_crew[j]) if net.n_crew[j] > 0 else 0.0
        est = max([fin[p] + lag for p, lag in net.pred[j]] + [starts[j]])
        if d == 0:
            fin[j] = est
            continue
        t = est
        while True:
            prog, D = 0.0, 0
            while prog < d - 1e-9:
                dl = day_loss[t + D] if t + D < len(day_loss) else 0.0
                prog += 1.0 - f * dl
                D += 1
            req = net.dem[j]
            if all(usage[t:t + D, k].max() + req[k] <= net.cap[k] for k in range(len(net.cap)) if req[k]):
                break
            t += 1
        usage[t:t + D] += np.array(req)
        fin[j] = t + D
        span[j] = (t, D)
    expo = {f"E1o_{thr}": float(sum(net.n_out[j] * (hi[t:t + D] >= thr).sum() for j, (t, D) in span.items()))
            for thr in (32.2, 39.4)}
    return max(fin.values()), expo


def activity_hours(project, G):
    """Daily hours of each activity: outdoor-worker-weighted mean of its resources' subsector hours."""
    ces = pd.read_csv(config.TABLES / "table_ces_hours.csv", dtype={"NAICS": str})
    hours = dict(zip(ces["NAICS"], ces["Daily hours (weekly / 5)"]))
    net = json.loads((s16.PROJ_DIR / f"{project}.json").read_text(encoding="utf-8"))
    sector_naics = G["ew"].sector_to_naics(net["sector"], net["name"])
    out = {}
    for a in net["activities"]:
        num = den = 0.0
        for dm in a["demands"]:
            wpu, od = G["rexp"].get(dm["resource"], (1.0, 0.0))
            w = int(dm["units"]) * wpu * od
            naics = G["ew"].trade_to_naics(dm["resource"]) or sector_naics
            num += w * hours[naics[:3]]
            den += w
        out[a["id"]] = num / den if den > 0 else hours[sector_naics[:3]]
    return out


def main():
    common.banner(log, "STEP 17 - POST-HOC ANALYSES OF SAVED SCHEDULES")
    s16.init_worker()
    G = s16._G
    files = sorted((config.OUTPUTS / "r1_sched").glob("*.json"))
    act_rows, trade_rows, dur_rows, wh_rows = [], [], [], []
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        if "starts" not in r:
            continue
        loc = r["location"]
        if loc not in G["info"]:
            G["info"][loc] = s16.load_location(loc, G["models"], G["q"])
        info = G["info"][loc]
        net = s16.Net(s16.PROJ_DIR / f"{r['project']}.json", G["ew"], G["share"], G["rate"], G["rexp"])
        st = {k: {int(j): s for j, s in v.items()} for k, v in r["starts"].items()}
        base = st["O0"]
        tf = total_float(net, r["cmax"])
        hi = info["hi"]
        wclass = {j: ("high" if net.w["share"][j] >= 0.7 else
                      "low" if net.w["share"][j] < 0.45 else "middle") for j in net.jobs}
        for name in COMPARE:
            if name not in st:
                continue
            s = st[name]
            act = [j for j in net.jobs if net.dur[j] > 0]
            sh = np.array([s[j] - base[j] for j in act])
            mv = sh != 0
            tau = kendalltau([base[j] for j in act], [s[j] for j in act]).statistic
            act_rows.append({
                "project": r["project"], "sector": r["sector"], "location": loc, "strategy": name,
                "activities": len(act), "moved": int(mv.sum()),
                "moved_later": int((sh > 0).sum()), "moved_earlier": int((sh < 0).sum()),
                "moved_zero_float": int(sum(1 for j, m in zip(act, mv) if m and tf[j] <= 0)),
                "zero_float_activities": int(sum(1 for j in act if tf[j] <= 0)),
                "shift_p50": float(np.median(np.abs(sh[mv]))) if mv.any() else 0.0,
                "shift_p90": float(np.percentile(np.abs(sh[mv]), 90)) if mv.any() else 0.0,
                "shift_max": int(np.abs(sh).max()) if len(sh) else 0,
                "shift_le_7": int(((np.abs(sh) <= 7) & mv).sum()),
                "shift_le_14": int(((np.abs(sh) <= 14) & mv).sum()),
                "shift_le_42": int(((np.abs(sh) <= 42) & mv).sum()),
                "kendall_tau": float(tau) if tau == tau else 1.0})
            for cls in ("high", "middle", "low"):
                js = [j for j in act if wclass[j] == cls]
                if not js:
                    continue
                e_b = sum(net.n_out[j] * (hi[base[j]:base[j] + net.dur[j]] >= 32.2).sum() for j in js)
                e_s = sum(net.n_out[j] * (hi[s[j]:s[j] + net.dur[j]] >= 32.2).sum() for j in js)
                trade_rows.append({"project": r["project"], "location": loc, "strategy": name,
                                   "weight_class": cls, "activities": len(js),
                                   "E1o_base": float(e_b), "E1o_strategy": float(e_s)})
        for name in RETIME:
            if name not in st:
                continue
            row = {"project": r["project"], "sector": r["sector"], "location": loc,
                   "strategy": name, "planned_makespan": r["cmax"]}
            for lname, lv in LOSS.items():
                mk, expo = retime(net, st[name], hi, lv)
                row[f"realized_makespan_{lname}"] = mk
                for k_, v_ in expo.items():
                    row[f"{k_}_{lname}"] = v_
            dur_rows.append(row)
        # worker-hours: outdoor workers weighted by the daily hours of their construction subsector
        hrs = activity_hours(r["project"], G)
        for name in ["O0"] + COMPARE:
            if name not in st:
                continue
            s = st[name]
            wh_rows.append({"project": r["project"], "location": loc, "strategy": name, **{
                f"E1h_{thr}": float(sum(net.n_out[j] * hrs[j] * (hi[s[j]:s[j] + net.dur[j]] >= thr).sum()
                                        for j in net.jobs if net.dur[j] > 0)) for thr in (32.2, 39.4)}})
    act = pd.DataFrame(act_rows)
    act.to_csv(config.TABLES / "posthoc_activity_level.csv", index=False)
    tr = pd.DataFrame(trade_rows)
    tr.to_csv(config.TABLES / "posthoc_trade_class.csv", index=False)
    du = pd.DataFrame(dur_rows)
    du.to_csv(config.TABLES / "posthoc_heat_durations.csv", index=False)
    pd.DataFrame(wh_rows).to_csv(config.TABLES / "posthoc_worker_hours.csv", index=False)

    # Example schedule: the building-sector scenario whose
    # primary planning-time reduction of hot-day outdoor worker-days is closest to the median.
    met = pd.read_csv(config.TABLES / "sched_r1_metrics.csv")
    b = met[met.strategy == "O0"].set_index(["project", "location"])
    p = met[met.strategy == "O3"].set_index(["project", "location"])
    red = (100 * (b["E1o_32.2"] - p["E1o_32.2"]) / b["E1o_32.2"]).dropna()
    bld = red[[("building" in s) for s in b.loc[red.index, "sector"]]]
    pick = (bld - bld.median()).abs().idxmin()
    r = json.loads((config.OUTPUTS / "r1_sched" / f"{pick[0]}__{pick[1]}.json").read_text(encoding="utf-8"))
    net = s16.Net(s16.PROJ_DIR / f"{pick[0]}.json", G["ew"], G["share"], G["rate"], G["rexp"])
    gantt = pd.DataFrame([{"project": pick[0], "location": pick[1], "activity": j,
                           "duration": net.dur[j], "weight": net.w["share"][j],
                           "outdoor_workers": net.n_out[j],
                           "start_O0": r["starts"]["O0"][str(j)], "start_O3": r["starts"]["O3"][str(j)]}
                          for j in net.jobs if net.dur[j] > 0])
    gantt.to_csv(config.TABLES / "posthoc_gantt_example.csv", index=False)
    common.save_json({"scenarios": int(len(du) / len(RETIME)) if len(du) else 0,
                      "loss_scenarios": LOSS, "gantt_example": list(pick),
                      "gantt_example_reduction_pct": float(bld.loc[pick])},
                     "verification17_posthoc")
    log.info(f"activity rows {len(act)}, trade rows {len(tr)}, duration rows {len(du)}; "
             f"Gantt example {pick}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
