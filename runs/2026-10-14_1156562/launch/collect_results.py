"""Collects the logged metrics of every run directory under a checkpoints root.

Usage: python collect_results.py <checkpoints root> <out.json>

For each run directory ({save_dir}/{job_type}_{timestamp}/ with config.yaml and
results.jsonl) it keeps, per metric, a curve over the run's sample counter
(env_steps, iteration, samples or updates_completed): the records are binned
into at most 100 bins of the counter, averaged within each seed and bin, then
summarised over seeds (mean, min, max, number of seeds). Runs of the same
job_type (e.g. a baseline split over two processes) are merged. Final values
(the last bin of every seed) are kept too.
"""

import collections
import json
import math
import sys
from pathlib import Path

import yaml

root, out = Path(sys.argv[1]), Path(sys.argv[2])
COUNTERS = ("env_steps", "iteration", "samples", "updates_completed")
SKIP = {"seed_index", "wall_time", "update_step"} | set(COUNTERS)
BINS = 100


def is_metric(key, value):
    if key in SKIP or not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    keep_prefix = ("game/", "game_br/", "final/")
    keep = ("exploitability", "value_vs_random", "value_br_vs_random", "_vs_random",
            "_vs_baseline", "_vs_always", "return_per_episode", "ep_length_avg",
            "episodes_truncated_frac", "sl_loss", "sl_acc", "value_loss", "td_loss",
            "entropy", "epsilon")
    if key.startswith(keep_prefix):
        return not key.split("/")[-1].startswith("n_")
    return any(k in key for k in keep)


runs = collections.defaultdict(list)  # job_type -> [(config, records)]
for cfg_path in sorted(root.glob("*/*/config.yaml")):
    run_dir = cfg_path.parent
    res = run_dir / "results.jsonl"
    if not res.exists():
        continue
    cfg = yaml.safe_load(cfg_path.read_text())
    records = []
    for line in res.read_text().splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line being written
    if records:
        runs[cfg.get("job_type", run_dir.name)].append((cfg, run_dir, records))

result = {}
for job_type, parts in runs.items():
    counter = next((c for c in COUNTERS if any(c in r for _, _, recs in parts for r in recs)), None)
    if counter is None:
        continue
    # seeds of every process of this job type, renumbered
    seeds = []
    for cfg, run_dir, recs in parts:
        by_seed = collections.defaultdict(list)
        for r in recs:
            if counter in r:
                by_seed[r.get("seed_index", 0)].append(r)
        seeds.extend(by_seed.values())
    xmax = max(r[counter] for s in seeds for r in s)
    budget = None
    cfg0 = parts[0][0]
    for key in ("num_timesteps", "num_samples", "iterations", "num_update_steps"):
        if key in cfg0:
            budget = float(cfg0[key])
            break
    edges = (budget or xmax) or 1
    metrics = sorted({k for s in seeds for r in s for k, v in r.items() if is_metric(k, v)})
    curves, finals = {}, {}
    for m in metrics:
        per_seed = []
        for s in seeds:
            bins = collections.defaultdict(list)
            for r in s:
                v = r.get(m)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                    b = min(BINS - 1, int(BINS * r[counter] / edges))
                    bins[b].append(v)
            per_seed.append({b: sum(v) / len(v) for b, v in bins.items()})
        xs = sorted({b for d in per_seed for b in d})
        if not xs:
            continue
        pts = []
        for b in xs:
            vals = [d[b] for d in per_seed if b in d]
            pts.append([round((b + 0.5) * edges / BINS, 3), round(sum(vals) / len(vals), 6),
                        round(min(vals), 6), round(max(vals), 6), len(vals)])
        curves[m] = pts
        last = [d[max(d)] for d in per_seed if d]
        if last:
            mean = sum(last) / len(last)
            sd = math.sqrt(sum((v - mean) ** 2 for v in last) / max(len(last) - 1, 1)) if len(last) > 1 else 0.0
            finals[m] = {"mean": round(mean, 6), "sd": round(sd, 6), "n": len(last)}
    result[job_type] = {
        "env_name": cfg0.get("env_name"),
        "counter": counter,
        "budget": budget,
        "progress": round(xmax / budget, 4) if budget else None,
        "num_seeds": len(seeds),
        "run_dirs": [str(p[1]) for p in parts],
        "wall_time_s": max((r.get("wall_time", 0) for s in seeds for r in s), default=None),
        "curves": curves,
        "final": finals,
    }
out.write_text(json.dumps(result))
print(f"{len(result)} job types, {out.stat().st_size / 1e6:.2f} MB")
