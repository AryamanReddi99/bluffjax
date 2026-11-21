"""Collects the matched-budget runs (jobs/m_<method>_n<N>/<run>/) on the machine it runs on into
summary_<host>.json: per job, the per-seed final exploitability, the mean over the last 10% of exact
evaluations, and the seed-mean and seed-std curves against env samples; for MMD also the magnet's.

Usage (from the run folder): python launch/collect_matched.py <host tag>
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def summarise(run_dir):
    cur, mag, steps = defaultdict(list), defaultdict(list), defaultdict(list)
    with open(os.path.join(run_dir, "results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if "exploitability" not in r:
                continue
            i = int(r["seed_index"])
            cur[i].append(r["exploitability"])
            steps[i].append(r["env_steps"])
            if "magnet_exploitability" in r:
                mag[i].append(r["magnet_exploitability"])
    seeds = sorted(cur)
    length = min(len(cur[i]) for i in seeds)
    tail = max(1, length // 10)
    a = np.array([cur[i][:length] for i in seeds])
    out = {
        "seeds": len(seeds),
        "final": a[:, -1].tolist(),
        "final_mean": float(a[:, -1].mean()),
        "final_std": float(a[:, -1].std()),
        "tail_mean": float(a[:, -tail:].mean()),
        "x": [round(s / 1e6, 2) for s in steps[seeds[0]][:length]],
        "mean": a.mean(0).round(4).tolist(),
        "std": a.std(0).round(4).tolist(),
    }
    if mag:
        m = np.array([mag[i][:length] for i in seeds])
        out.update(magnet_final_mean=float(m[:, -1].mean()), magnet_final_std=float(m[:, -1].std()),
                   magnet_tail_mean=float(m[:, -tail:].mean()), magnet_mean=m.mean(0).round(4).tolist())
    return out


def main():
    host = sys.argv[1]
    out = {}
    for job_dir in sorted(glob.glob(os.path.join(HERE, "jobs", "m_*"))):
        job = os.path.basename(job_dir)
        runs = sorted(glob.glob(os.path.join(job_dir, f"{job}_*")))
        if not runs or not os.path.exists(os.path.join(runs[-1], "results.jsonl")):
            continue
        method, n = re.match(r"m_(\w+)_n(\d)", job).groups()
        try:
            out[job] = {"method": method, "num_cards": int(n), **summarise(runs[-1])}
        except (ValueError, KeyError) as e:
            print(f"{job}: skipped ({e})")
            continue
        s = out[job]
        print(f"{job:12s} final {s['final_mean']:.3f} ± {s['final_std']:.3f}  last 10% {s['tail_mean']:.3f}"
              + (f"  magnet {s['magnet_final_mean']:.3f}" if "magnet_final_mean" in s else ""))
    with open(os.path.join(HERE, f"summary_{host}.json"), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
