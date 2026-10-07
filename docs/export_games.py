#!/usr/bin/env python3
"""Export BluffJAX games to StableHLO so whlo can run them in the browser.

For each game this writes games/<name>/{reset,step,avail}.mlir and an entry in
games/index.json:

  reset(key: u32[2])                  -> (*state_leaves)
  step(key: u32[2], *leaves, action)  -> (*state_leaves, reward f32[num_agents], done i32[])
  avail(*leaves)                      -> avail bool[num_actions] (AEC) or [num_agents, num_actions] (parallel)

`step` wraps `step_env` (not `step`), so a finished hand stays on the table
until the page deals the next one. `action` is a scalar for turn-based (AEC)
games and an i32[num_agents] vector for simultaneous (parallel) ones. The
manifest names each state leaf by its field path so the page's renderer can
read e.g. `agent_cards` directly.

Usage:
    .venv/bin/python export_games.py              # every registered game
    .venv/bin/python export_games.py kuhn_poker
"""
import json
import pathlib
import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from jax import export

jax.config.update("jax_enable_x64", False)
# Drop source-location metadata: it is most of the module text and whlo ignores it.
jax.config.update("jax_traceback_in_locations_limit", 0)

from bluffjax import available_envs, make  # noqa: E402
from bluffjax.environments.env import ParallelEnv  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "games"


def export_game(name):
    env = make(name)
    parallel = isinstance(env, ParallelEnv)
    key = jax.random.PRNGKey(0)
    state0, _ = jax.jit(env.reset)(key)
    paths_leaves, treedef = jax.tree_util.tree_flatten_with_path(state0)
    leaves0 = [np.asarray(l) for _, l in paths_leaves]
    names = [jax.tree_util.keystr(p).lstrip(".") for p, _ in paths_leaves]
    specs = [jax.ShapeDtypeStruct(l.shape, l.dtype) for l in leaves0]

    def cast(leaves):
        return [jnp.asarray(l).astype(s.dtype).reshape(s.shape) for l, s in zip(leaves, specs)]

    def reset(key):
        state, _ = env.reset(key)
        return tuple(cast(jax.tree_util.tree_leaves(state)))

    def step(key, *args):
        *leaves, action = args
        state = jax.tree_util.tree_unflatten(treedef, leaves)
        state, _, reward, _, done, _ = env.step_env(key, state, action)
        return (
            *cast(jax.tree_util.tree_leaves(state)),
            jnp.asarray(reward, jnp.float32).reshape(env.num_agents),
            jnp.asarray(done, jnp.int32).reshape(()),
        )

    def avail(*leaves):
        state = jax.tree_util.tree_unflatten(treedef, list(leaves))
        return env.get_avail_actions(state)

    action_spec = jax.ShapeDtypeStruct((env.num_agents,) if parallel else (), jnp.int32)
    key_spec = jax.ShapeDtypeStruct((2,), jnp.uint32)
    jit = lambda f: jax.jit(f, keep_unused=True)  # noqa: E731  (keep args 1:1 with leaves)
    modules = {
        "reset": export.export(jit(reset))(key_spec).mlir_module(),
        "step": export.export(jit(step))(key_spec, *specs, action_spec).mlir_module(),
        "avail": export.export(jit(avail))(*specs).mlir_module(),
    }
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for k, text in modules.items():
        (d / f"{k}.mlir").write_text(text)
    return {
        "name": name,
        "kind": "parallel" if parallel else "aec",
        "num_agents": env.num_agents,
        "num_actions": int(env.action_space().n),
        "leaves": [
            {"name": n, "dtype": str(l.dtype), "shape": list(l.shape)} for n, l in zip(names, leaves0)
        ],
        "bytes": {k: len(v) for k, v in modules.items()},
    }


def main():
    names = sys.argv[1:] or list(available_envs())
    OUT.mkdir(exist_ok=True)
    index_path = OUT / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for name in names:
        print(f"[{name}] exporting...", flush=True)
        try:
            index[name] = export_game(name)
            info = index[name]
            print(f"[{name}] ok: {len(info['leaves'])} leaves, step {info['bytes']['step'] / 1e3:.0f} kB")
        except Exception as e:  # keep going; one broken game shouldn't stop the rest
            traceback.print_exc()
            print(f"[{name}] FAILED: {e}")
            index.pop(name, None)
    index_path.write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
