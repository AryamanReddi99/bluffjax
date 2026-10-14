"""Builds a runs root for the head-to-head: the PPO-NFSP / PQN-NFSP run directories
of NEW_ROOT and every other baseline's run directories of OLD_ROOT, as symlinks.

Usage: python make_h2h_root.py OLD_ROOT NEW_ROOT OUT_ROOT GAME_DIR [GAME_DIR ...]
"""
import os, sys
import yaml

old, new, out = (os.path.abspath(os.path.expanduser(a)) for a in sys.argv[1:4])
NFSP = ("_ppo_nfsp", "_pqn_nfsp")

def runs(root, game):
    base = os.path.join(root, game)
    for d in sorted(os.listdir(base)):
        cfg = os.path.join(base, d, "config.yaml")
        if os.path.isfile(cfg):
            yield os.path.join(base, d), yaml.safe_load(open(cfg))["job_type"]

for game in sys.argv[4:]:
    dst = os.path.join(out, game)
    os.makedirs(dst, exist_ok=False)
    kept = []
    for path, jt in runs(old, game):
        if not jt.endswith(NFSP):
            os.symlink(path, os.path.join(dst, os.path.basename(path))); kept.append(jt)
    for suffix in NFSP:
        found = [(p, jt) for p, jt in runs(new, game) if jt.endswith(suffix)]
        assert len(found) == 1, (game, suffix, found)
        p, jt = found[0]
        for sub in ("half", "full"):
            assert os.listdir(os.path.join(p, sub)), (p, sub)
        os.symlink(p, os.path.join(dst, os.path.basename(p))); kept.append(jt + " (new)")
    print(game, kept)
