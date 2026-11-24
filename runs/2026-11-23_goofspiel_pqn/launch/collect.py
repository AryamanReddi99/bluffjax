"""Exact exploitability of every job of this run folder (jobs/<job>/<run>/results.jsonl): per job the
seed-mean final value, its std over seeds and the mean over each seed's last 10% of exact evaluations (the
screens' selection criterion).

Usage (from the repo root): python runs/2026-11-23_goofspiel_pqn/launch/collect.py [job prefix] [out.json]
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def collect(job_dir):
    runs = sorted(glob.glob(os.path.join(job_dir, "*", "results.jsonl")))
    if not runs:
        return None
    curves = defaultdict(list)
    with open(runs[-1]) as f:
        for line in f:
            r = json.loads(line)
            if r.get("exact_evaluated") and "exploitability" in r:
                curves[int(r["seed_index"])].append(r["exploitability"])
    if not curves:
        return None
    n = min(len(v) for v in curves.values())
    a = np.array([curves[s][:n] for s in sorted(curves)])
    tail = max(1, n // 10)
    return dict(job=os.path.basename(job_dir), seeds=a.shape[0], evals=n, final_mean=float(a[:, -1].mean()),
                final_std=float(a[:, -1].std()), tail_mean=float(a[:, -tail:].mean()), best_mean=float(a.mean(0).min()))


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    rows = [r for d in sorted(glob.glob(os.path.join(RUN, "jobs", prefix + "*"))) if (r := collect(d))]
    for r in rows:
        print(f"{r['job']:24s} seeds {r['seeds']:2d} evals {r['evals']:4d}  final {r['final_mean']:.3f} +- "
              f"{r['final_std']:.3f}  last-10% {r['tail_mean']:.3f}  best seed-mean {r['best_mean']:.3f}")
    if len(sys.argv) > 2:
        json.dump(rows, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
