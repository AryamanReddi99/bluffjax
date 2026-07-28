"""
Head-to-head evaluation of Kemps 'simple' checkpoints.

Compares 11 agent types: PPO-NFSP and PQN-NFSP each checkpointed at 5
points during training on the small/fast '_simple' env variant (10%, 25%,
50%, 75%, 100% of total timesteps), plus a uniform-random legal-action
agent. Both PPO-NFSP and PQN-NFSP use their avg (average-strategy)
checkpoint, since that is the NFSP output policy meant to approximate a
Nash equilibrium -- the br checkpoint is only the training-time
best-response exploiter.

Kemps is a ParallelEnv played by 4 agents in 2 fixed teams (team 0 = seats
{0, 2}, team 1 = seats {1, 3}), all acting simultaneously every step, unlike
bluff's turn-based 2-player AECEnv. Each matchup pits one agent TYPE
controlling one whole team against another agent TYPE controlling the other
team (both teammates within a team use the same policy/checkpoint).

For each of the 66 unique matchups (11 self-matchups + 55 cross-matchups;
ordered pairs are redundant since a "vs" b and b "vs" a are the same
evaluation), n episodes are played. Each episode independently samples a
random checkpoint (of the 3 saved seeds) for each side and a random seat
assignment, so results are averaged across seeds and across seat/positional
bias.
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
from bluffjax.networks.mlp import ActorCriticDiscreteMLP, ActorDiscreteMLP

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(HERE, "checkpoints", "simple")
NUM_SEEDS = 3

RUN_DIR_RE = re.compile(r"^(ppo_nfsp|pqn_nfsp)_.*_s\d+$")

AGENT_TYPES = [
    "ppo_0p1",
    "ppo_0p25",
    "ppo_0p5",
    "ppo_0p75",
    "ppo_1",
    "pqn_0p1",
    "pqn_0p25",
    "pqn_0p5",
    "pqn_0p75",
    "pqn_1",
    "random",
]
_TYPE_INFO = {
    "ppo_0p1": ("ppo_nfsp", "0p1"),
    "ppo_0p25": ("ppo_nfsp", "0p25"),
    "ppo_0p5": ("ppo_nfsp", "0p5"),
    "ppo_0p75": ("ppo_nfsp", "0p75"),
    "ppo_1": ("ppo_nfsp", "1"),
    "pqn_0p1": ("pqn_nfsp", "0p1"),
    "pqn_0p25": ("pqn_nfsp", "0p25"),
    "pqn_0p5": ("pqn_nfsp", "0p5"),
    "pqn_0p75": ("pqn_nfsp", "0p75"),
    "pqn_1": ("pqn_nfsp", "1"),
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


def build_agent_pool(env, ac_network, actor_network, fc_dim_size):
    del fc_dim_size
    _, sample_obs = env.reset(jax.random.PRNGKey(0))
    pool = {}
    checkpoints_used = {}
    for agent_type in AGENT_TYPES:
        if agent_type == "random":
            pool[agent_type] = ("random", None, None)
            continue
        algorithm, frac_tag = _TYPE_INFO[agent_type]
        network = ac_network if algorithm == "ppo_nfsp" else actor_network
        kind = "actor_critic" if algorithm == "ppo_nfsp" else "actor"
        stack, run_dirs = load_checkpoint_stack(network, sample_obs, algorithm, frac_tag)
        pool[agent_type] = (kind, network, stack)
        checkpoints_used[agent_type] = run_dirs
    return pool, checkpoints_used


def sample_action(kind, network, params, obs, action_mask, rng):
    if kind == "random":
        probs = action_mask.astype(jnp.float32)
        probs = probs / jnp.maximum(probs.sum(axis=-1, keepdims=True), 1.0)
        return distrax.Categorical(probs=probs).sample(seed=rng)
    if kind == "actor_critic":
        logits, _ = network.apply(params, obs)
    else:  # "actor"
        logits = network.apply(params, obs.astype(jnp.float32))
    logits_masked = jnp.where(action_mask, logits, -jnp.inf)
    return distrax.Categorical(logits=logits_masked).sample(seed=rng)


def make_play_episode(env, kind_a, network_a, kind_b, network_b):
    def play_episode(rng, params_a, params_b, swap):
        # swap == 0: agent type A controls team 0 (seats 0, 2); B controls
        # team 1 (seats 1, 3). swap == 1: reversed. This removes
        # team-assignment bias, analogous to bluff's seat_a.
        team0_mask = (jnp.arange(env.num_agents) % 2) == 0  # (4,) bool
        is_a_seat = jnp.where(swap == 0, team0_mask, ~team0_mask)  # (4,) bool

        rng, rng_reset = jax.random.split(rng)
        state, obs = env.reset(rng_reset)

        def cond(carry):
            *_, done_flag = carry
            return ~done_flag

        def body(carry):
            state_s, obs_s, rng_s, terminal_reward, _ = carry
            rng_s, rng_a, rng_b, rng_step = jax.random.split(rng_s, 4)
            action_mask = env.get_avail_actions(state_s)  # (4, action_dim)

            action_a = sample_action(kind_a, network_a, params_a, obs_s, action_mask, rng_a)
            action_b = sample_action(kind_b, network_b, params_b, obs_s, action_mask, rng_b)
            joint_action = jnp.where(is_a_seat, action_a, action_b)  # (4,)

            next_state, next_obs, reward, _, done, _ = env.step(
                rng_step, state_s, joint_action
            )
            new_terminal_reward = jnp.where(done, reward, terminal_reward)
            return next_state, next_obs, rng_s, new_terminal_reward, done

        init = (
            state,
            obs,
            rng,
            jnp.zeros(env.num_agents, dtype=jnp.float32),
            jnp.bool_(False),
        )
        _, _, _, terminal_reward, _ = lax.while_loop(cond, body, init)

        # terminal_reward is shared within a team and is one of {-1, 0, +1};
        # index 0 represents team 0, index 1 represents team 1.
        team_a_return = jnp.where(swap == 0, terminal_reward[0], terminal_reward[1])
        a_win = (team_a_return > 0).astype(jnp.float32)
        b_win = (team_a_return < 0).astype(jnp.float32)
        draw = (team_a_return == 0).astype(jnp.float32)
        return a_win, b_win, draw

    return play_episode


def run_matchup(env, agent_pool, type_a, type_b, num_episodes, rng):
    kind_a, network_a, stack_a = agent_pool[type_a]
    kind_b, network_b, stack_b = agent_pool[type_b]
    play_episode = make_play_episode(env, kind_a, network_a, kind_b, network_b)

    rng, rng_idx_a, rng_idx_b, rng_swap, rng_eps = jax.random.split(rng, 5)
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
    swap = jax.random.bernoulli(rng_swap, p=0.5, shape=(num_episodes,)).astype(jnp.int32)

    def episode_fn(ep_rng, ia, ib, sw):
        params_a = (
            None if stack_a is None else jax.tree_util.tree_map(lambda x: x[ia], stack_a)
        )
        params_b = (
            None if stack_b is None else jax.tree_util.tree_map(lambda x: x[ib], stack_b)
        )
        return play_episode(ep_rng, params_a, params_b, sw)

    a_win, b_win, draw = jax.jit(jax.vmap(episode_fn))(ep_rngs, idx_a, idx_b, swap)
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
        "--num_episodes", type=int, default=1000, help="Episodes per unique matchup."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(HERE, "head2head_results_avg_simple.json"),
    )
    args = parser.parse_args()

    with open(os.path.join(HERE, "config_ppo_nfsp_simple.yaml"), "r") as f:
        config = yaml.safe_load(f)
    env_kwargs = config["env_kwargs"]
    fc_dim_size = config["fc_dim_size"]

    env = make(config["env_name"], **env_kwargs)
    sample_state, _ = env.reset(jax.random.PRNGKey(0))
    action_dim = int(env.get_avail_actions(sample_state).shape[-1])

    ac_network = ActorCriticDiscreteMLP(action_dim=action_dim, hidden_dim=fc_dim_size)
    actor_network = ActorDiscreteMLP(action_dim=action_dim, hidden_dim=fc_dim_size)

    agent_pool, checkpoints_used = build_agent_pool(env, ac_network, actor_network, fc_dim_size)

    rng = jax.random.PRNGKey(args.seed)
    matchups = []
    for type_a, type_b in itertools.combinations_with_replacement(AGENT_TYPES, 2):
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
