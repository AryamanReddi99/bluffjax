"""
Head-to-head evaluation of Bluff checkpoints across three training scales.

Compares 9 agent types: PPO-NFSP and PQN-NFSP each at four sample counts
(1e8, 5e7, 1e7, 5e6 -- decreasing), plus a uniform-random legal-action agent.
Both algorithms use their BR checkpoint. PPO-NFSP's br checkpoint is an
ActorCriticDiscreteMLP (stochastic policy, sampled as usual). PQN-NFSP's br
checkpoint is a QNetworkDiscreteMLP; it is evaluated greedily (epsilon=0,
ties broken uniformly), matching how bluff_pqn_nfsp.py's
sample_action_q_greedy is used at evaluation time during training.

The "1e7" checkpoints come from the original default-scale run in
checkpoints/ (num_timesteps=1e7, save_checkpoints=[0.5,1.0], so its "1"
fraction is 1e7 samples). The "5e7"/"1e8" checkpoints come from the longer
run in checkpoints/10e7/ (num_timesteps=1e8, so its "0p5"/"1" fractions are
5e7/1e8 samples respectively). Each of the 8 non-random agent types is a
3-seed stack, matching the original head2head_br.py convention --
run_matchup samples a random seed per side per episode.

For each of the 45 unique matchups (9 self-matchups + 36 cross-matchups;
ordered pairs are redundant since a "vs" b and b "vs" a are the same
evaluation), n episodes are played. Each episode independently samples a
random checkpoint seed for each side and a random seat assignment, so
results are averaged across seeds and across seat/positional bias.
"""

import argparse
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
from bluffjax.networks.mlp import ActorCriticDiscreteMLP, QNetworkDiscreteMLP

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_DIR = os.path.join(HERE, "checkpoints")
LONG_RUN_CHECKPOINT_DIR = os.path.join(HERE, "checkpoints", "10e7")
NUM_SEEDS = 3

RUN_DIR_RE = re.compile(r"^(ppo_nfsp|pqn_nfsp)_.*_s\d+$")

AGENT_TYPES = ["ppo_10e7", "ppo_5e7", "ppo_1e7", "ppo_5e6", "pqn_10e7", "pqn_5e7", "pqn_1e7", "pqn_5e6", "random"]
_TYPE_INFO = {
    "ppo_10e7": (LONG_RUN_CHECKPOINT_DIR, "ppo_nfsp", "1"),
    "ppo_5e7": (LONG_RUN_CHECKPOINT_DIR, "ppo_nfsp", "0p5"),
    "ppo_1e7": (DEFAULT_CHECKPOINT_DIR, "ppo_nfsp", "1"),
    "ppo_5e6": (DEFAULT_CHECKPOINT_DIR, "ppo_nfsp", "0p5"),
    "pqn_10e7": (LONG_RUN_CHECKPOINT_DIR, "pqn_nfsp", "1"),
    "pqn_5e7": (LONG_RUN_CHECKPOINT_DIR, "pqn_nfsp", "0p5"),
    "pqn_1e7": (DEFAULT_CHECKPOINT_DIR, "pqn_nfsp", "1"),
    "pqn_5e6": (DEFAULT_CHECKPOINT_DIR, "pqn_nfsp", "0p5"),
}


def find_run_dirs(root_dir, algorithm):
    dirs = sorted(
        d
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and RUN_DIR_RE.match(d)
    )
    return [d for d in dirs if d.startswith(algorithm + "_")]


def load_checkpoint_stack(network, sample_obs, root_dir, algorithm, frac_tag):
    run_dirs = find_run_dirs(root_dir, algorithm)
    if len(run_dirs) != NUM_SEEDS:
        raise RuntimeError(
            f"Expected {NUM_SEEDS} top-level checkpoint runs for '{algorithm}' "
            f"in {root_dir}, found {len(run_dirs)}: {run_dirs}"
        )
    template_params = network.init(jax.random.PRNGKey(0), sample_obs)
    all_params = []
    for d in run_dirs:
        path = os.path.join(root_dir, d, f"br_{frac_tag}.msgpack")
        with open(path, "rb") as f:
            params = serialization.from_bytes(template_params, f.read())
        all_params.append(params)
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *all_params)
    return stacked, [os.path.relpath(os.path.join(root_dir, d), HERE) for d in run_dirs]


def build_agent_pool(env, ac_network, q_network, fc_dim_size):
    del fc_dim_size
    _, sample_obs = env.reset(jax.random.PRNGKey(0))
    pool = {}
    checkpoints_used = {}
    for agent_type in AGENT_TYPES:
        if agent_type == "random":
            pool[agent_type] = ("random", None, None)
            continue
        root_dir, algorithm, frac_tag = _TYPE_INFO[agent_type]
        network = ac_network if algorithm == "ppo_nfsp" else q_network
        kind = "actor_critic" if algorithm == "ppo_nfsp" else "q_network"
        stack, run_dirs = load_checkpoint_stack(network, sample_obs, root_dir, algorithm, frac_tag)
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
        logits_masked = jnp.where(action_mask, logits, -jnp.inf)
        return distrax.Categorical(logits=logits_masked).sample(seed=rng)
    # kind == "q_network": greedy (epsilon=0) action, matching
    # bluff_pqn_nfsp.py's sample_action_q_greedy, with ties broken uniformly.
    q_vals = network.apply(params, obs.astype(jnp.float32))
    q_vals_masked = jnp.where(action_mask, q_vals, -jnp.inf)
    best_val = jnp.max(q_vals_masked)
    ties = (q_vals_masked == best_val) & action_mask
    tie_logits = jnp.where(ties, 0.0, -1e9)
    return jax.random.categorical(rng, tie_logits)


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


def run_matchup(env, agent_pool, type_a, type_b, num_episodes, rng):
    kind_a, network_a, stack_a = agent_pool[type_a]
    kind_b, network_b, stack_b = agent_pool[type_b]
    play_episode = make_play_episode(env, kind_a, network_a, kind_b, network_b)

    rng, rng_idx_a, rng_idx_b, rng_seat, rng_eps = jax.random.split(rng, 5)
    ep_rngs = jax.random.split(rng_eps, num_episodes)
    idx_a = (
        jax.random.randint(rng_idx_a, (num_episodes,), 0, NUM_SEEDS, dtype=jnp.int32)
        if stack_a is not None
        else jnp.zeros((num_episodes,), dtype=jnp.int32)
    )
    idx_b = (
        jax.random.randint(rng_idx_b, (num_episodes,), 0, NUM_SEEDS, dtype=jnp.int32)
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


# Old 5-type results (ppo_full, ppo_half, pqn_full, pqn_half, random) reuse:
# all 5 map directly onto the new 9-type naming (same checkpoints, same
# default-scale config) -- ppo_full/pqn_full are the "1e7" (fraction "1")
# points and ppo_half/pqn_half are the "5e6" (fraction "0p5") points.
_OLD_TO_NEW = {
    "ppo_full": "ppo_1e7",
    "ppo_half": "ppo_5e6",
    "pqn_full": "pqn_1e7",
    "pqn_half": "pqn_5e6",
    "random": "random",
}


def load_reusable_old_matchups(old_results_path):
    if not os.path.exists(old_results_path):
        return {}
    with open(old_results_path, "r") as f:
        old_results = json.load(f)
    reusable = {}
    for m in old_results["matchups"]:
        new_a = _OLD_TO_NEW.get(m["agent_a"])
        new_b = _OLD_TO_NEW.get(m["agent_b"])
        if new_a is None or new_b is None:
            continue
        reusable[(new_a, new_b)] = m
    return reusable


def reused_entry(old_matchups, type_a, type_b):
    """Look up a cached matchup in either orientation, relabeling agent_a/b
    (and swapping win_rate_a/win_rate_b if the cached orientation is
    reversed) to match the (type_a, type_b) order this run expects."""
    if (type_a, type_b) in old_matchups:
        m = old_matchups[(type_a, type_b)]
        return {
            "agent_a": type_a,
            "agent_b": type_b,
            "num_episodes": m["num_episodes"],
            "win_rate_a": m["win_rate_a"],
            "win_rate_b": m["win_rate_b"],
            "draw_rate": m["draw_rate"],
        }
    if (type_b, type_a) in old_matchups:
        m = old_matchups[(type_b, type_a)]
        return {
            "agent_a": type_a,
            "agent_b": type_b,
            "num_episodes": m["num_episodes"],
            "win_rate_a": m["win_rate_b"],
            "win_rate_b": m["win_rate_a"],
            "draw_rate": m["draw_rate"],
        }
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_episodes", type=int, default=1000, help="Episodes per unique matchup."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--old_results",
        type=str,
        default=os.path.join(HERE, "head2head_results_br.json"),
        help="Old 5-type results to reuse verbatim for the ppo_1e7/ppo_5e6/pqn_1e7/pqn_5e6/random matchups among themselves.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(HERE, "head2head_results_br_10e7.json"),
    )
    args = parser.parse_args()

    with open(os.path.join(HERE, "config_ppo_nfsp.yaml"), "r") as f:
        config = yaml.safe_load(f)
    env_kwargs = config["env_kwargs"]
    fc_dim_size = config["fc_dim_size"]

    env = make(config["env_name"], **env_kwargs)
    sample_state, _ = env.reset(jax.random.PRNGKey(0))
    action_dim = int(env.get_avail_actions(sample_state).shape[-1])

    ac_network = ActorCriticDiscreteMLP(action_dim=action_dim, hidden_dim=fc_dim_size)
    q_network = QNetworkDiscreteMLP(action_dim=action_dim, hidden_dim=fc_dim_size)

    agent_pool, checkpoints_used = build_agent_pool(env, ac_network, q_network, fc_dim_size)
    old_matchups = load_reusable_old_matchups(args.old_results)

    rng = jax.random.PRNGKey(args.seed)
    matchups = []
    for type_a, type_b in itertools.combinations_with_replacement(AGENT_TYPES, 2):
        entry = reused_entry(old_matchups, type_a, type_b)
        if entry is not None:
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
