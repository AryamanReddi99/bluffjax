import glob, json, sys, collections, numpy as np
for job in sys.argv[1:]:
    d = sorted(glob.glob(f"jobs/{job}/{job}_*"))[-1]
    cur = collections.defaultdict(list); mag = collections.defaultdict(list)
    for line in open(d + "/results.jsonl"):
        r = json.loads(line)
        if "exploitability" in r:
            cur[r["seed_index"]].append(r["exploitability"]); mag[r["seed_index"]].append(r.get("magnet_exploitability", np.nan))
    L = min(map(len, cur.values())); t = max(1, L // 10)
    a = np.array([v[:L] for v in cur.values()]); m = np.array([v[:L] for v in mag.values()])
    print(job, "policy final %.3f±%.3f tail %.3f | magnet final %.3f±%.3f tail %.3f" % (a[:, -1].mean(), a[:, -1].std(), a[:, -t:].mean(), m[:, -1].mean(), m[:, -1].std(), m[:, -t:].mean()))
