"""Summarises a vmapped screen (jobs/<job>/<run>/results.jsonl): per grid point, the swept
hyperparameter values, the seed-mean final exploitability, the mean over the last 10% of exact
evaluations (the selection criterion) and the seed-mean curve against env samples.

Usage: python launch/collect_screen.py <job dir> [out.json]
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def main():
    job_dir = sys.argv[1]
    run = sorted(glob.glob(os.path.join(job_dir, "*", "results.jsonl")))[-1]
    grid = json.load(open(os.path.join(os.path.dirname(run), "sweep.json")))  # one entry per instance
    swept = [k[3:] for k in grid[0] if k not in ("hp/grid_index", "hp/seed") and len({str(g[k]) for g in grid}) > 1]
    cur, steps, hp = defaultdict(lambda: defaultdict(list)), {}, {}
    with open(run) as f:
        for line in f:
            r = json.loads(line)
            if "exploitability" not in r:
                continue
            g, s = int(r["hp/grid_index"]), int(r["hp/seed"])
            cur[g][s].append(r["exploitability"])
            hp[g] = {k: r[f"hp/{k}"] for k in swept}
            if g == 0 and s == 0:
                steps.setdefault("x", []).append(r["env_steps"])
    out = []
    for g in sorted(cur):
        n = min(len(v) for v in cur[g].values())
        a = np.array([v[:n] for v in cur[g].values()])
        tail = max(1, n // 10)
        out.append(dict(grid=g, **hp[g], seeds=a.shape[0], final_mean=float(a[:, -1].mean()),
                        final_std=float(a[:, -1].std()), tail_mean=float(a[:, -tail:].mean()),
                        tail_std=float(a[:, -tail:].mean(1).std()),
                        x=[round(s / 1e6, 2) for s in steps["x"][:n]], mean=a.mean(0).round(4).tolist()))
    for o in out:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if k not in ("x", "mean")})
    if len(sys.argv) > 2:
        json.dump(out, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()
