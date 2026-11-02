"""results.jsonl of a PQN-NFSP Leduc run -> one entry of the leduc_push JSON.
usage: summarize.py <run_dir> <phase> <variant> <where> [out.json]"""
import json, sys, gzip, os, statistics as st
run_dir, phase, variant, where = sys.argv[1:5]
p = os.path.join(run_dir, "results.jsonl")
rows = [json.loads(l) for l in (gzip.open(p + ".gz", "rt") if not os.path.exists(p) else open(p))]
import yaml
cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
# env steps from update_step (runs before the host-side fix logged an int32
# counter that wraps past 2**31)
for r in rows:
    r["env_steps"] = (r["update_step"] + 1) * cfg["num_envs"] * cfg["num_steps_per_env_per_update"]
seeds, gap_avg, gap_mix, expl_br = [], {}, {}, {}
for s in sorted({r["seed_index"] for r in rows}):
    ev = [r for r in rows if r["seed_index"] == s and "exploitability" in r]
    ev.sort(key=lambda r: r["update_step"])
    curve = [[r["env_steps"], r["exploitability"]] for r in ev]
    seeds.append({"seed": s, "curve": curve, "final": curve[-1][1]})
    gap_avg[s] = [[r["env_steps"], r.get("br_gap_vs_avg")] for r in ev]
    gap_mix[s] = [[r["env_steps"], r.get("br_gap_vs_mix")] for r in ev]
    expl_br[s] = [[r["env_steps"], r["exploitability_br"]] for r in ev]
finals = [x["final"] for x in seeds]
last = lambda d: [v[-1][1] for v in d.values() if v and v[-1][1] is not None]
entry = {
    "phase": phase, "method": "PQN-NFSP", "variant": variant, "game": "leduc",
    "x_unit": "env_steps", "budget": float(cfg["num_timesteps"]),
    "wall_min": round(max(r["wall_time"] for r in rows) / 60, 1), "where": where,
    "seeds": seeds, "final_mean": st.mean(finals), "final_sd": st.pstdev(finals),
    "extra": {
        "br_gap_vs_avg": gap_avg, "br_gap_vs_mix": gap_mix, "exploitability_br": expl_br,
        "final_br_gap_vs_avg_mean": st.mean(last(gap_avg)) if last(gap_avg) else None,
        "final_br_gap_vs_mix_mean": st.mean(last(gap_mix)) if last(gap_mix) else None,
        "config": {k: cfg[k] for k in ("num_timesteps", "num_envs", "num_steps_per_env_per_update",
                   "num_epochs", "num_minibatches", "lr", "anneal_lr", "start_e", "end_e",
                   "exploration_fraction", "q_lambda", "max_grad_norm", "gamma",
                   "anticipatory_eta", "sl_reservoir_capacity", "sl_batch_size",
                   "sl_num_steps_per_update", "sl_lr") if k in cfg} | {"br_optimizer": cfg.get("br_optimizer", "adam")},
    },
    "run_dirs": [run_dir], "notes": "",
}
print(f"{phase}: final mean {entry['final_mean']:.4f} sd {entry['final_sd']:.4f} | per seed "
      + " ".join(f"{f:.3f}" for f in finals)
      + f" | BR gap vs avg {entry['extra']['final_br_gap_vs_avg_mean']}, vs mix {entry['extra']['final_br_gap_vs_mix_mean']} | wall {entry['wall_min']} min")
print("mean curve:", [(c[0], round(st.mean(s['curve'][i][1] for s in seeds), 4)) for i, c in enumerate(seeds[0]["curve"])])
if len(sys.argv) > 5:
    json.dump(entry, open(sys.argv[5], "w"))
