"""Builds the leduc_push JSON entries for phase C from the collected run dirs.
usage: combine.py <collect_dir> <out.json> [phase ...]"""
import glob, json, os, subprocess, sys
collect, out = sys.argv[1], sys.argv[2]
phases = sys.argv[3:] or ["C0", "C1", "C2", "C3a", "C3b", "C3c", "C3d", "C4", "C4b", "C5a1", "C5a2", "C5b", "C5c"]
BASE = "eps 0.06->0 linear over the run, gamma 1"
OPT = "RAdam, clip 10, lr 2.5e-4 linearly to 0"
VAR = {
    "C0": ("C0", "baseline config (Adam 2.5e-4 const, clip 0.5, eps 1->0.005 over 1%, gamma 0.99, lambda 0.9, 4x4 minibatches), rerun with BR-gap logging"),
    "C1": ("C1", BASE),
    "C2": ("C2", f"{BASE}; {OPT}"),
    "C3a": ("C3", f"{BASE}; {OPT}; 32 minibatches"),
    "C3b": ("C3", f"{BASE}; {OPT}; lambda 0.65"),
    "C3c": ("C3", f"{BASE}; {OPT}; 32 minibatches, lambda 0.65"),
    "C3d": ("C3", f"{BASE}; {OPT}; 16 minibatches"),
    "C4": ("C4", f"C3c (best of C1-C3: {BASE}; {OPT}; 32 minibatches, lambda 0.65) at 1e9 env steps, 2e6 reservoir"),
    "C4b": ("C4", f"C3a ({BASE}; {OPT}; 32 minibatches, lambda 0.9) at 1e9 env steps, 2e6 reservoir"),
    "C5a1": ("C5a", "C4 with 64 minibatches (4 epochs), 1e9 env steps, 2e6 reservoir"),
    "C5a2": ("C5a", "C4 with 32 minibatches x 2 epochs, 1e9 env steps, 2e6 reservoir"),
    "C5b": ("C5b", "C4 settings at 3e9 env steps, 3e6 reservoir"),
    "C5c": ("C5c", "C4 with lr floor: lr decays linearly to 10% (lr_final_frac 0.1), 1e9 env steps, 2e6 reservoir"),
}
entries = []
here = os.path.dirname(os.path.abspath(__file__))
for p in phases:
    dirs = glob.glob(f"{collect}/{p}/leduc_pqn_nfsp_*")
    if not dirs:
        print("missing", p); continue
    run_dir = dirs[0]
    where = "nash RTX 3080" if p == "C0" else "cluster (RTX A5000 / 3090)"
    tmp = f"/tmp/_entry_{p}.json"
    subprocess.run([sys.executable, f"{here}/summarize.py", run_dir, VAR[p][0], VAR[p][1], where, tmp], check=True)
    e = json.load(open(tmp))
    e["extra"]["run_name"] = p
    e["run_dirs"] = [f"runs/2026-10-31_leduc_push/{p}/leduc/{os.path.basename(run_dir)}"]
    entries.append(e)
# reference: the current published PQN-NFSP run (runs2), evaluated every 10 updates
ref = "/home/aryaman/bluffjax-dev/runs2/checkpoints/leduc/leduc_pqn_nfsp_2026-10-14_21-50-34"
if os.path.exists(ref):
    subprocess.run([sys.executable, f"{here}/summarize.py", ref, "baseline", "current published run (runs2, same config as C0)", "nash RTX 3080", "/tmp/_entry_ref.json"], check=True)
    e = json.load(open("/tmp/_entry_ref.json"))
    e["notes"] = ("Same config and seeds as C0; differs only by floating-point divergence (identical losses at update 0, "
                  "exploitability equal to 3 decimals), i.e. 0.183 vs 0.134 is run-to-run noise of a 5-seed mean.")
    e["run_dirs"] = [ref]
    entries.insert(0, e)
json.dump(entries, open(out, "w"))
print("wrote", out, len(entries))
