"""Collect CFR / Deep CFR / SD-CFR push results into the summary JSON."""
import glob, json, os, sys
import numpy as np

ROOT = os.path.expanduser("~/bluffjax-dev/cfr-push/runs/leduc_push")
OLD = os.path.expanduser("~/bluffjax-dev/runs/checkpoints")

def records(path):
    return [json.loads(l) for l in open(path)]

def curve(recs, key, seed_index=0):
    return [[r["iteration"], r[key]] for r in recs if r.get("seed_index", 0) == seed_index and key in r]

def entry(phase, method, variant, game, seeds, budget, wall_min, where, run_dirs, notes, extra=None):
    finals = [s["final"] for s in seeds]
    e = {"phase": phase, "method": method, "variant": variant, "game": game, "x_unit": "iterations",
         "budget": budget, "wall_min": wall_min, "where": where, "seeds": seeds,
         "final_mean": float(np.mean(finals)), "final_sd": float(np.std(finals)),
         "run_dirs": run_dirs, "notes": notes}
    if extra: e["extra"] = extra
    return e

out = []
NAMES = {"cfr": "vanilla CFR (alternating updates)", "cfr_plus": "CFR+ (RM+, linear averaging)",
         "linear": "Linear CFR", "dcfr": "Discounted CFR (1.5, 0, 2)"}
for game in ("kuhn", "leduc"):
    # baselines: the first-round runs
    for d in sorted(glob.glob(f"{OLD}/{game}/{game}_cfr_*")):
        recs = records(f"{d}/results.jsonl"); c = curve(recs, "exploitability")
        out.append(entry("baseline", "CFR", "vanilla CFR, 100 iterations (first round)", game,
                         [{"seed": 0, "curve": c, "final": c[-1][1]}], 100, recs[-1]["wall_time"] / 60,
                         "nash CPU", [d], "existing run"))
    for d in sorted(glob.glob(f"{OLD}/{game}/{game}_deep_cfr_*")):
        recs = records(f"{d}/results.jsonl")
        seeds = []
        for i in sorted({r["seed_index"] for r in recs}):
            c = curve(recs, "exploitability", i); seeds.append({"seed": i, "curve": c, "final": c[-1][1]})
        wall = max(r["wall_time"] for r in recs) / 60
        out.append(entry("baseline", "Deep CFR", "50 iterations x 100 traversals, 200 adv. steps, 1e5 memories (first round)",
                         game, seeds, 50, wall, "nash GPU", [d], "existing run (seeds run one after the other; wall_min is per seed)"))
    for v in ("cfr", "cfr_plus", "linear", "dcfr"):
        ds = glob.glob(f"{ROOT}/A_{game}_{v}/*/results.jsonl")
        if not ds: continue
        recs = records(ds[0]); c = curve(recs, "exploitability")
        first = {th: next((x for x, e in c if e <= th), None) for th in (0.05, 0.01)}
        out.append(entry("A", "CFR" if v == "cfr" else NAMES[v].split(" (")[0], NAMES[v], game,
                         [{"seed": 0, "curve": c, "final": c[-1][1]}], 2000, recs[-1]["wall_time"] / 60,
                         "nash CPU (python)", [os.path.dirname(ds[0])],
                         "deterministic; evaluated every 100 iterations; matches OpenSpiel's solver of the same name",
                         {"first_eval_iteration_below": first}))
    for tag in sorted(glob.glob(f"{ROOT}/D_{game}_*")):
        cfg_name = os.path.basename(tag)
        seeds_dc, seeds_sd, dirs, walls = [], [], [], []
        for s in range(5):
            fs = glob.glob(f"{tag}/seed{s}/*/results.jsonl")
            if not fs: continue
            recs = records(fs[0]); dirs.append(os.path.dirname(fs[0]))
            c = curve(recs, "exploitability"); sc = curve(recs, "sd_cfr/exploitability")
            if not c or c[-1][0] != 1000: continue
            seeds_dc.append({"seed": s, "curve": c, "final": c[-1][1]})
            seeds_sd.append({"seed": s, "curve": sc, "final": sc[-1][1]})
            walls.append(recs[-1]["wall_time"] / 60)
        if not seeds_dc: continue
        steps = cfg_name.split("adv")[-1]
        variant = f"1000 iterations x 1500 traversals, {steps} adv. steps x 2048, adv. memory 2e6, strategy memory 1e7, MLP 128-128"
        where = open(f"{tag}/gpu.txt").read().split("\n")[-2] if os.path.exists(f"{tag}/gpu.txt") else "cluster"
        note = "5 seeds run in parallel on one GPU; SD-CFR shares the advantage networks of the Deep CFR run (as in the SD-CFR paper); XLA deterministic GPU ops"
        if cfg_name == "D_leduc_adv3000":
            where = where + " (seeds 0, 2, 3, 4; seed 1 crashed at start-up with a shared-filesystem error and was rerun alone on nash RTX 3080, 36 min)"
        out.append(entry("D", "Deep CFR", variant, game, seeds_dc, 1000, max(walls), f"cluster {where}", dirs, note))
        out.append(entry("D", "SD-CFR", variant, game, seeds_sd, 1000, max(walls), f"cluster {where}", dirs, note))
json.dump(out, open(sys.argv[1], "w"), indent=1)
for e in out:
    print(f"{e['phase']:8s} {e['game']:5s} {e['method']:12s} {e['final_mean']:.4f} +- {e['final_sd']:.4f}  n={len(e['seeds'])}  {e['wall_min']:.1f} min  {e['variant'][:60]}")
