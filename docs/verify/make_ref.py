#!/usr/bin/env python3
"""Record JAX reference rollouts (random legal actions, several episodes) with
per-step state hashes, rewards and legal-action masks; check.mjs replays them
through whlo. Step i uses key data [0, i]; episode e resets with [1, e]."""
import hashlib, json, pathlib, sys
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", False)
from bluffjax import make
from bluffjax.environments.env import ParallelEnv

name = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
env = make(name)
par = isinstance(env, ParallelEnv)
meta = json.loads((pathlib.Path(__file__).parent.parent / "games/index.json").read_text())[name]
dtypes = [l["dtype"] for l in meta["leaves"]]
reset = jax.jit(lambda k: env.reset(k)[0])
step = jax.jit(env.step_env)
avail_f = jax.jit(env.get_avail_actions)
key = lambda a, b: jnp.array([a, b], jnp.uint32)
def h(s):
    ls = jax.tree_util.tree_leaves(s)
    return hashlib.sha1(b"".join(np.ascontiguousarray(np.asarray(l).astype(dt)).tobytes() for l, dt in zip(ls, dtypes))).hexdigest()
rng = np.random.default_rng(0)
ep = 0
s = reset(key(1, ep))
rec = dict(resets=[], actions=[], hashes=[h(s)], rewards=[], avails=[], dones=[])
for i in range(STEPS):
    av = np.asarray(avail_f(s))
    rec["avails"].append(av.astype(int).ravel().tolist())
    if par:
        a = [int(rng.choice(np.flatnonzero(row))) for row in av]
    else:
        a = int(rng.choice(np.flatnonzero(av)))
    s, _, r, _, d, _ = step(key(0, i), s, jnp.asarray(a, jnp.int32))
    rec["actions"].append(a); rec["rewards"].append(np.asarray(r, np.float32).ravel().tolist()); rec["dones"].append(int(d))
    if bool(d):
        ep += 1; rec["resets"].append(i); s = reset(key(1, ep))
    rec["hashes"].append(h(s))
out = pathlib.Path(__file__).parent / "refs"; out.mkdir(exist_ok=True)
(out / f"{name}.json").write_text(json.dumps(rec))
print(f"{name}: {STEPS} steps, {ep} episodes")
