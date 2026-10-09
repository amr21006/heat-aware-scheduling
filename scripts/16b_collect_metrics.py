"""
16b_collect_metrics.py — Rebuild sched_r1_metrics.csv from the per-scenario files of step 16.

Step 16 writes the same table at the end of its run; this helper rebuilds it from the
resume-safe per-scenario files (used to test the downstream steps on completed scenarios and to
re-assemble the table without re-solving).
"""
from __future__ import annotations

import json
import sys

import pandas as pd

import config


def main(name: str = "sched_r1_metrics"):
    rows, skipped = [], []
    for f in sorted((config.OUTPUTS / "r1_sched").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if "metrics" not in r:
            skipped.append({"project": r["project"], "location": r["location"], "reason": r.get("skipped")})
            continue
        for strat, mtr in r["metrics"].items():
            rows.append({"project": r["project"], "sector": r["sector"], "location": r["location"],
                         "activities": r["activities"], "cmax": r["cmax"],
                         "base_proven_optimal": r["base_proven_optimal"], "strategy": strat,
                         **mtr, **{f"stat_{k}": v for k, v in r["stats"].get(strat, {}).items()}})
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLES / f"{name}.csv", index=False)
    print(f"{name}: {df[['project', 'location']].drop_duplicates().shape[0]} scenarios, "
          f"{len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
