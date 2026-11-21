import glob, json, sys, collections, numpy as np
def load(job):
    d = sorted(glob.glob(f"jobs/{job}/{job}_*"))[-1]
    rows = collections.defaultdict(list)
    for line in open(d + "/results.jsonl"):
        r = json.loads(line); rows[r["seed_index"]].append(r)
    return rows
for job in sys.argv[1:]:
    try: rows = load(job)
    except IndexError: print(job, "missing"); continue
    fin = [[r["exploitability"] for r in v if "exploitability" in r][-1] for _, v in sorted(rows.items())]
    first = {k: [round(v[u]["actor_loss"], 7) for u in (0, 1, 5)] for k, v in sorted(rows.items())}
    print(job, "final per seed", [round(f, 3) for f in fin], "mean %.3f" % np.mean(fin))
    print("   actor_loss at updates 0,1,5 per seed:", first)
