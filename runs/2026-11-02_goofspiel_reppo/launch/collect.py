"""Collects the ReMAPPO runs (jobs/<job>/<run>/) into summary.json and prints
one table per job: for every grid point, over its seeds, the final
exploitability (mean, std), the mean over the last 10% of exact evaluations
(robust to the last-iterate noise of self-play) and the best seed-mean
exploitability during training.

Usage: python launch/collect.py [job name substring]
"""

import glob
import gzip
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIFORM = {n: (n * n - 1) / 6 for n in range(3, 7)}
SWEPT = ["lr", "ent_coef"]
STATIC = ["reppo_k", "nonself_weight", "clip_mode", "nonself_w_max", "num_epochs"]


def load(run_dir):
    path = os.path.join(run_dir, "results.jsonl")
    f = open(path) if os.path.exists(path) else gzip.open(path + ".gz", "rt")
    with f:
        rows = [json.loads(line) for line in f if line.strip()]
    curves = defaultdict(list)  # instance -> [(update_step, exploitability)]
    for r in rows:
        if "exploitability" in r:
            curves[int(r["seed_index"])].append((int(r["update_step"]), float(r["exploitability"])))
    insts = json.load(open(os.path.join(run_dir, "sweep.json")))
    cfg = open(os.path.join(run_dir, "config.yaml")).read()
    static = {}
    for k in STATIC:
        m = re.search(rf"^{k}: (\S+)", cfg, re.M)
        static[k] = m.group(1) if m else None
    n = int(re.search(r"^  num_cards: (\d+)", cfg, re.M).group(1))
    return curves, insts, static, n


def summarise(job, run_dir):
    curves, insts, static, n = load(run_dir)
    grid = defaultdict(list)
    for i, inst in enumerate(insts):
        grid[int(inst["hp/grid_index"])].append(i)
    points = []
    for g, members in sorted(grid.items()):
        series = [np.array([e for _, e in sorted(curves[i])]) for i in members if curves[i]]
        if not series:
            continue
        length = min(len(s) for s in series)
        stack = np.stack([s[:length] for s in series])  # seeds x evaluations
        tail = max(1, length // 10)
        mean_curve = stack.mean(0)
        steps = [u for u, _ in sorted(curves[members[0]])][:length]
        points.append({
            **{k: insts[members[0]][f"hp/{k}"] for k in SWEPT},
            "seeds": len(series),
            "final_mean": float(stack[:, -1].mean()),
            "final_std": float(stack[:, -1].std()),
            "tail_mean": float(stack[:, -tail:].mean()),
            "best_mean": float(mean_curve.min()),
            "best_update": int(steps[int(mean_curve.argmin())]),
            "curve_updates": steps,
            "curve_mean": mean_curve.tolist(),
        })
    return {"job": job, "run_dir": os.path.relpath(run_dir, HERE), "num_cards": n,
            "uniform": UNIFORM.get(n), **static, "points": points}


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    out = {}
    for job_dir in sorted(glob.glob(os.path.join(HERE, "jobs", "*"))):
        job = os.path.basename(job_dir)
        runs = sorted(glob.glob(os.path.join(job_dir, f"{job}_*")))
        if not runs or pattern not in job:
            continue
        try:
            out[job] = summarise(job, runs[-1])
        except (FileNotFoundError, AttributeError) as e:  # still running
            print(f"{job}: skipped ({e})")
            continue
        s = out[job]
        print(f"\n{job}  N={s['num_cards']} (uniform {s['uniform']:.3f})  K={s['reppo_k']} "
              f"weight={s['nonself_weight']} clip={s['clip_mode']} epochs={s['num_epochs']}")
        print(f"{'lr':>8} {'ent':>6} | {'final':>15} | {'last 10%':>8} | {'best':>7} @upd")
        for p in sorted(s["points"], key=lambda p: (p["lr"], p["ent_coef"])):
            print(f"{p['lr']:8.0e} {p['ent_coef']:6.2f} | {p['final_mean']:7.3f} ± {p['final_std']:5.3f} "
                  f"| {p['tail_mean']:8.3f} | {p['best_mean']:7.3f} {p['best_update']:5d}")
    if not pattern:
        with open(os.path.join(HERE, "summary.json"), "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
