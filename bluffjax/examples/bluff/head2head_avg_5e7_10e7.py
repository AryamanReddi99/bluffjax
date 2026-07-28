"""
Head-to-head evaluation of Bluff checkpoints, extended with the long-run
PPO-NFSP checkpoints trained to 5e7 and 1e8 (=10e7) timesteps.

Mimics head2head_avg.py but adds two agent types on top of the original
5 (ppo_full, ppo_half, pqn_full, pqn_half, random):
  - ppo_5e7:  PPO-NFSP avg checkpoint at the 0.5 fraction of the long
              (num_timesteps=1e8) run in checkpoints/10e7 (i.e. 5e7 steps).
  - ppo_10e7: PPO-NFSP avg checkpoint at the 1.0 fraction of that same run
              (i.e. 1e8 steps).

Both new checkpoints come from a single seed (s0), unlike the 3-seed stacks
used for the original 5 agent types -- run_matchup samples uniformly over
however many seeds are actually available for each side.

Matchups among the original 5 agent types are NOT recomputed; they are
copied verbatim from the existing head2head_results_avg.json (produced by
head2head_avg.py). Only the matchups involving ppo_5e7 and/or ppo_10e7 are
newly evaluated (13 of the 28 unique pairs among the 7 agent types).
"""

import argparse
import glob
import itertools
import json
import os
import re

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import serialization
from jax import lax

from bluffjax import make
from bluffjax.networks.mlp import ActorCriticDiscreteMLP, ActorDiscreteMLP

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(HERE, "checkpoints")
LONG_RUN_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "10e7")
NUM_SEEDS = 3

RUN_DIR_RE = re.compile(r"^(ppo_nfsp|pqn_nfsp)_.*_s\d+$")

OLD_AGENT_TYPES = ["ppo_full", "ppo_half", "pqn_full", "pqn_half", "random"]
NEW_AGENT_TYPES = ["ppo_5e7", "ppo_10e7"]
AGENT_TYPES = OLD_AGENT_TYPES + NEW_AGENT_TYPES
_TYPE_INFO = {
    "ppo_full": ("ppo_nfsp", "1"),
    "ppo_half": ("ppo_nfsp", "0p5"),
    "pqn_full": ("pqn_nfsp", "1"),
    "pqn_half": ("pqn_nfsp", "0p5"),
}
_NEW_FRAC_TAG = {
    "ppo_5e7": "0p5",
    "ppo_10e7": "1",
}


def find_run_dirs(algorithm):
    dirs = sorted(
        d
        for d in os.listdir(CHECKPOINT_DIR)
        if os.path.isdir(os.path.join(CHECKPOINT_DIR, d)) and RUN_DIR_RE.match(d)
    )
    return [d for d in dirs if d.startswith(algorithm + "_")]


def load_checkpoint_stack(network, sample_obs, algorithm, frac_tag):
    run_dirs = find_run_dirs(algorithm)
    if len(run_dirs) != NUM_SEEDS:
        raise RuntimeError(
            f"Expected {NUM_SEEDS} top-level checkpoint runs for '{algorithm}' "
            f"in {CHECKPOINT_DIR}, found {len(run_dirs)}: {run_dirs}"
        )
    template_params = network.init(jax.random.PRNGKey(0), sample_obs)
    all_params = []
    for d in run_dirs:
        path = os.path.join(CHECKPOINT_DIR, d, f"avg_{frac_tag}.msgpack")
        with open(path, "rb") as f:
            params = serialization.from_bytes(template_params, f.read())
        all_params.append(params)
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *all_params)
    return stacked, run_dirs


def load_long_run_checkpoint_stack(network, sample_obs, frac_tag):
    pattern = os.path.join(LONG_RUN_CHECKPOINT_DIR, f"*_{frac_tag}_avg.msgpack")
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly 1 checkpoint matching '{pattern}', found {len(matches)}: {matches}"
        )
    path = matches[0]
    template_params = network.init(jax.random.PRNGKey(0), sample_obs)
    with open(path, "rb") as f:
        params = serialization.from_bytes(template_params, f.read())
    # Single seed: keep the leading "seed" axis of size 1 so it behaves
    # like the NUM_SEEDS-stacked checkpoints (indexing seed 0 always).
    stacked = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], params)
    run_name = os.path.splitext(os.path.basename(path))[0]
    return stacked, [run_name]


def build_agent_pool(env, ac_network, actor_network):
    _, sample_obs = env.reset(jax.random.PRNGKey(0))
    pool = {}
    checkpoints_used = {}
    for agent_type in AGENT_TYPES:
        if agent_type == "random":
            pool[agent_type] = ("random", None, None)
            continue
        if agent_type in NEW_AGENT_TYPES:
            frac_tag = _NEW_FRAC_TAG[agent_type]
            stack, run_dirs = load_long_run_checkpoint_stack(ac_network, sample_obs, frac_tag)
            pool[agent_type] = ("actor_critic", ac_network, stack)
            checkpoints_used[agent_type] = run_dirs
            continue
        algorithm, frac_tag = _TYPE_INFO[agent_type]
        network = ac_network if algorithm == "ppo_nfsp" else actor_network
        stack, run_dirs = load_checkpoint_stack(network, sample_obs, algorithm, frac_tag)
        kind = "actor_critic" if algorithm == "ppo_nfsp" else "actor"
        pool[agent_type] = (kind, network, stack)
        checkpoints_used[agent_type] = run_dirs
    return pool, checkpoints_used


def sample_action(kind, network, params, obs, action_mask, rng):
    if kind == "random":
        probs = action_mask.astype(jnp.float32)
        probs = probs / jnp.maximum(probs.sum(), 1.0)
        return distrax.Categorical(probs=probs).sample(seed=rng)
    if kind == "actor_critic":
        logits, _ = network.apply(params, obs)
    else:  # "actor"
        logits = network.apply(params, obs.astype(jnp.float32))
    logits_masked = jnp.where(action_mask, logits, -jnp.inf)
    return distrax.Categorical(logits=logits_masked).sample(seed=rng)


def make_play_episode(env, kind_a, network_a, kind_b, network_b):
    def play_episode(rng, params_a, params_b, seat_a):
        rng, rng_reset = jax.random.split(rng)
        state, obs = env.reset(rng_reset)

        def cond(carry):
            *_, done_flag = carry
            return ~done_flag

        def body(carry):
            state_s, obs_s, rng_s, winner_term, draw_term, _ = carry
            rng_s, rng_a, rng_b, rng_step = jax.random.split(rng_s, 4)
            action_mask = env.get_avail_actions(state_s)
            current_player = state_s.current_player_idx

            action_a = sample_action(kind_a, network_a, params_a, obs_s, action_mask, rng_a)
            action_b = sample_action(kind_b, network_b, params_b, obs_s, action_mask, rng_b)
            action_seat0 = jnp.where(seat_a == 0, action_a, action_b)
            action_seat1 = jnp.where(seat_a == 0, action_b, action_a)
            action = jnp.where(current_player == 0, action_seat0, action_seat1)

            next_state, next_obs, _, _, done, info = env.step(rng_step, state_s, action)
            winner_now = info["game_winner"]
            new_winner_term = jnp.where(done, winner_now, winner_term)
            draw_now = ~info["game_winner"].any()
            new_draw_term = jnp.where(done, draw_now, draw_term)
            return next_state, next_obs, rng_s, new_winner_term, new_draw_term, done

        init = (
            state,
            obs,
            rng,
            jnp.zeros(env.num_agents, dtype=jnp.bool_),
            jnp.bool_(False),
            jnp.bool_(False),
        )
        _, _, _, winner_term, draw_term, _ = lax.while_loop(cond, body, init)

        seat_b = 1 - seat_a
        a_win = winner_term[seat_a].astype(jnp.float32)
        b_win = winner_term[seat_b].astype(jnp.float32)
        draw = draw_term.astype(jnp.float32)
        return a_win, b_win, draw

    return play_episode


def _num_seeds(stack):
    if stack is None:
        return 1
    return int(jax.tree_util.tree_leaves(stack)[0].shape[0])


def run_matchup(env, agent_pool, type_a, type_b, num_episodes, rng):
    kind_a, network_a, stack_a = agent_pool[type_a]
    kind_b, network_b, stack_b = agent_pool[type_b]
    play_episode = make_play_episode(env, kind_a, network_a, kind_b, network_b)

    num_seeds_a = _num_seeds(stack_a)
    num_seeds_b = _num_seeds(stack_b)

    rng, rng_idx_a, rng_idx_b, rng_seat, rng_eps = jax.random.split(rng, 5)
    ep_rngs = jax.random.split(rng_eps, num_episodes)
    idx_a = (
        jax.random.randint(rng_idx_a, (num_episodes,), 0, num_seeds_a, dtype=jnp.int32)
        if stack_a is not None
        else jnp.zeros((num_episodes,), dtype=jnp.int32)
    )
    idx_b = (
        jax.random.randint(rng_idx_b, (num_episodes,), 0, num_seeds_b, dtype=jnp.int32)
        if stack_b is not None
        else jnp.zeros((num_episodes,), dtype=jnp.int32)
    )
    seat_a = jax.random.bernoulli(rng_seat, p=0.5, shape=(num_episodes,)).astype(jnp.int32)

    def episode_fn(ep_rng, ia, ib, sa):
        params_a = (
            None if stack_a is None else jax.tree_util.tree_map(lambda x: x[ia], stack_a)
        )
        params_b = (
            None if stack_b is None else jax.tree_util.tree_map(lambda x: x[ib], stack_b)
        )
        return play_episode(ep_rng, params_a, params_b, sa)

    a_win, b_win, draw = jax.jit(jax.vmap(episode_fn))(ep_rngs, idx_a, idx_b, seat_a)
    return np.asarray(a_win), np.asarray(b_win), np.asarray(draw)


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    mean = float(x.mean()) if n else 0.0
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    stderr = std / np.sqrt(n) if n else 0.0
    return {"avg": mean, "std_deviation": std, "std_error": float(stderr)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_episodes", type=int, default=1000, help="Episodes per newly-evaluated matchup."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--old_results",
        type=str,
        default=os.path.join(HERE, "head2head_results_avg.json"),
        help="Existing 5x5 results to reuse verbatim for the original agent types.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(HERE, "head2head_results_avg_5e7_10e7.json"),
    )
    args = parser.parse_args()

    with open(args.old_results, "r") as f:
        old_results = json.load(f)
    if old_results["agent_types"] != OLD_AGENT_TYPES:
        raise RuntimeError(
            f"Expected old results agent_types={OLD_AGENT_TYPES}, got {old_results['agent_types']}"
        )
    old_matchups = {
        (m["agent_a"], m["agent_b"]): m for m in old_results["matchups"]
    }

    with open(os.path.join(HERE, "config_ppo_nfsp.yaml"), "r") as f:
        config = yaml.safe_load(f)
    env_kwargs = config["env_kwargs"]
    fc_dim_size = config["fc_dim_size"]

    env = make(config["env_name"], **env_kwargs)
    sample_state, _ = env.reset(jax.random.PRNGKey(0))
    action_dim = int(env.get_avail_actions(sample_state).shape[-1])

    ac_network = ActorCriticDiscreteMLP(action_dim=action_dim, hidden_dim=fc_dim_size)
    actor_network = ActorDiscreteMLP(action_dim=action_dim, hidden_dim=fc_dim_size)

    agent_pool, checkpoints_used = build_agent_pool(env, ac_network, actor_network)
    checkpoints_used = {**old_results["checkpoints_used"], **checkpoints_used}

    rng = jax.random.PRNGKey(args.seed)
    matchups = []
    for type_a, type_b in itertools.combinations_with_replacement(AGENT_TYPES, 2):
        is_new = type_a in NEW_AGENT_TYPES or type_b in NEW_AGENT_TYPES
        if not is_new:
            entry = old_matchups[(type_a, type_b)]
            matchups.append(entry)
            print(
                f"{type_a:10s} vs {type_b:10s}: "
                f"win_a={entry['win_rate_a']['avg']:.3f} "
                f"win_b={entry['win_rate_b']['avg']:.3f} "
                f"draw={entry['draw_rate']['avg']:.3f}  [reused]"
            )
            continue

        rng, rng_matchup = jax.random.split(rng)
        a_win, b_win, draw = run_matchup(
            env, agent_pool, type_a, type_b, args.num_episodes, rng_matchup
        )
        entry = {
            "agent_a": type_a,
            "agent_b": type_b,
            "num_episodes": args.num_episodes,
            "win_rate_a": stats(a_win),
            "win_rate_b": stats(b_win),
            "draw_rate": stats(draw),
        }
        matchups.append(entry)
        print(
            f"{type_a:10s} vs {type_b:10s}: "
            f"win_a={entry['win_rate_a']['avg']:.3f} "
            f"win_b={entry['win_rate_b']['avg']:.3f} "
            f"draw={entry['draw_rate']['avg']:.3f}"
        )

    results = {
        "game": config["env_name"],
        "env_kwargs": env_kwargs,
        "agent_types": AGENT_TYPES,
        "num_episodes_per_matchup": args.num_episodes,
        "checkpoints_used": checkpoints_used,
        "matchups": matchups,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
