"""Per-seed exploitability results of the Kuhn/Leduc NFSP sweep (and of the main runs).

Usage: python collect_sweep.py <out.json> <checkpoints root> [<checkpoints root> ...]
Keeps runs whose job_type is a sweep run (sweep_*) or a main Kuhn/Leduc NFSP run.
"""

import json
import sys
from pathlib import Path

import yaml

out = Path(sys.argv[1])
KEYS = ("exploitability", "exploitability_br", "value_vs_random",
        "game/p1_bet_jack", "game/p1_call_queen", "game/p0_bet_jack", "game/p0_call_queen",
        "game/p1_equilibrium_gap", "game/p0_alpha_gap",
        "game/bluff_rate", "game/correct_fold_rate", "game/pair_raise_rate", "game/unpaired_raise_rate")
runs = []
for root in map(Path, sys.argv[2:]):
    for cfg_path in sorted(root.glob("*/*/config.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text())
        jt = cfg.get("job_type", "")
        if not (jt.startswith("sweep_") or jt in ("kuhn_ppo_nfsp", "kuhn_pqn_nfsp", "leduc_ppo_nfsp", "leduc_pqn_nfsp")):
            continue
        res = cfg_path.parent / "results.jsonl"
        if not res.exists():
            continue
        seeds = {}
        for line in res.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "exploitability" not in r:
                continue
            s = seeds.setdefault(r["seed_index"], {"steps": [], **{k: [] for k in KEYS}})
            s["steps"].append(r["env_steps"])
            for k in KEYS:
                s[k].append(r.get(k))
        runs.append({
            "job_type": jt, "game": "kuhn" if "kuhn" in jt else "leduc",
            "algo": "ppo_nfsp" if "ppo_nfsp" in jt else "pqn_nfsp",
            "ent_coef": cfg.get("ent_coef"), "anneal_lr": cfg.get("anneal_lr"),
            "start_e": cfg.get("start_e"), "end_e": cfg.get("end_e"),
            "exploration_fraction": cfg.get("exploration_fraction"),
            "num_timesteps": cfg.get("num_timesteps"), "num_seeds": cfg.get("num_seeds"),
            "main_run": not jt.startswith("sweep_"),
            "seeds": {str(i): s for i, s in sorted(seeds.items())},
        })
out.write_text(json.dumps(runs))
print(len(runs), "runs")
