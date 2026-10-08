"""Final evaluation of NFSP average policies against a ReBeL checkpoint.

For every seed of an NFSP run (avg_policy_{i}.msgpack, NFSP's output strategy),
plays n_deals mirrored deals (each deal from both seats) against the ReBeL
opponent with test-time search, the same protocol as ReBeL's own final
evaluation (holdem_rebel.agent.play_match / summarize).

Usage: python -m bluffjax.examples.holdem_rebel.eval_nfsp <nfsp_run_dir> <rebel_checkpoint.msgpack> [n_deals] [batch]
Writes <nfsp_run_dir>/eval_vs_rebel.json.
"""

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from omegaconf import OmegaConf

from bluffjax import make
from bluffjax.examples.holdem_rebel.agent import PolicyPlayer, load_rebel, play_match, summarize
from bluffjax.networks.mlp import ActorCriticDiscreteMLP, ActorDiscreteMLP

ENV_IDS = {"hul": "texas_limit_holdem", "hunl": "texas_nolimit_holdem"}


def avg_network(game: str, algo: str, action_dim: int, hidden: int):
    """The average-policy network exactly as save_checkpoints documents it."""
    base = (ActorCriticDiscreteMLP if algo == "ppo" else ActorDiscreteMLP)(
        action_dim=action_dim, hidden_dim=hidden
    )
    if game == "hunl":
        from bluffjax.examples.HUNL.hunl_ppo_nfsp import ChipScaledInput

        return ChipScaledInput(base, 52, 100.0)
    return base


def main(run_dir: str, rebel_path: str, n_deals: int, batch: int) -> None:
    cfg = OmegaConf.to_container(OmegaConf.load(os.path.join(run_dir, "config.yaml")))
    game_name = cfg["env_name"]
    algo = "ppo" if "ppo" in cfg["job_type"] else "pqn"
    env = make(ENV_IDS[game_name], **cfg.get("env_kwargs", {}))
    sample_state, _ = env.reset(jax.random.PRNGKey(0))
    action_dim = int(env.get_avail_actions(sample_state).shape[-1])  # as in the NFSP scripts
    net = avg_network(game_name, algo, action_dim, cfg["fc_dim_size"])
    template = net.init(jax.random.PRNGKey(0), jnp.zeros((env.obs_dim,), jnp.float32))

    rebel, rebel_params, meta = load_rebel(rebel_path)
    assert meta["game"] == game_name, (meta["game"], game_name)

    def sample(params, rng, state, avail):
        out = net.apply(params, env.obs_from_state(state))
        logits = out[0] if isinstance(out, tuple) else out
        return jax.random.categorical(rng, jnp.where(avail, logits, -jnp.inf))

    player = PolicyPlayer(sample)
    match = jax.jit(
        lambda p, k: play_match(env, rebel.game, player, p, rebel, rebel_params, batch, k)
    )
    per_seed = []
    for i in range(cfg["num_seeds"]):
        with open(os.path.join(run_dir, f"avg_policy_{i}.msgpack"), "rb") as f:
            params = serialization.from_bytes(template, f.read())
        rng = jax.random.PRNGKey(10_000 + i)
        t0 = time.time()
        results = []
        for _ in range(max(1, n_deals // batch)):
            rng, k = jax.random.split(rng)
            results.append(jax.device_get(match(params, k)))
        res = summarize(results)
        res["seed"], res["seconds"] = i, time.time() - t0
        per_seed.append(res)
        print(f"seed {i}: {res['mean']:+.4f} +- {res['se']:.4f} per hand ({res['hands']} hands, "
              f"re-solves {res['resolves']}, unfinished {res['unfinished']}, {res['seconds']:.0f} s)", flush=True)
    means = np.array([r["mean"] for r in per_seed])
    summary = {
        "run_dir": run_dir, "rebel_checkpoint": rebel_path, "rebel_samples": meta.get("samples"),
        "game": game_name, "algo": algo, "units": "BB per hand" if game_name == "hul" else "chips per hand",
        "deals_per_seed": n_deals, "mean_over_seeds": float(means.mean()),
        "se_over_seeds": float(means.std(ddof=1) / np.sqrt(len(means))) if len(means) > 1 else float("nan"),
        "per_seed": per_seed,
    }
    with open(os.path.join(run_dir, "eval_vs_rebel.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"{game_name} {algo}-NFSP vs ReBeL: {summary['mean_over_seeds']:+.4f} +- {summary['se_over_seeds']:.4f} "
          f"{summary['units']} over {len(means)} seeds")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 2048, int(a[4]) if len(a) > 4 else 128)
