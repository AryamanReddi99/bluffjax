"""
Parts shared by bluff_ppo_nfsp.py and bluff_pqn_nfsp.py: the NFSP reservoir
buffer, the per-game policy mixing, checkpoint saving and strict loading, and
the head-to-head evaluation.
"""

import datetime
import os
from typing import Any, Callable, NamedTuple

import jax
from flax import serialization
from jax import lax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from bluffjax.environments.bluff.bluff import Bluff
from bluffjax.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray


class SLBufferState(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    seen: IntArray
    size: IntArray


def init_sl_buffer(capacity: int, obs_dim: int, action_dim: int) -> SLBufferState:
    return SLBufferState(
        obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
        action_mask=jnp.zeros((capacity, action_dim), dtype=jnp.bool_),
        action=jnp.zeros((capacity,), dtype=jnp.int32),
        seen=jnp.array(0, dtype=jnp.int32),
        size=jnp.array(0, dtype=jnp.int32),
    )


def reservoir_append(
    buffer: SLBufferState,
    obs: FloatArray,
    action_mask: BoolArray,
    action: IntArray,
    valid: BoolArray,
    rng: PRNGKeyArray,
) -> SLBufferState:
    """Adds the valid rows to the reservoir exactly as Algorithm R would when
    processing them one at a time in order.

    The k-th valid item of the whole stream (k from 0) goes to slot k while the
    buffer is not full, and otherwise replaces slot j ~ U{0, ..., k} if
    j < capacity. Whether and where an item is written doesn't depend on the
    buffer contents, so the batch is written with one scatter; when several
    items pick the same slot the latest one wins, as in the sequential version.
    """
    capacity = buffer.action.shape[0]
    valid_i = valid.astype(jnp.int32)
    k = buffer.seen + jnp.cumsum(valid_i) - valid_i
    j = jax.random.randint(rng, k.shape, 0, k + 1, dtype=jnp.int32)
    slot = jnp.where(k < capacity, k, j)
    write = valid & (slot < capacity)
    slot = jnp.where(write, slot, capacity)
    last_writer = (
        jnp.full((capacity,), -1, dtype=jnp.int32)
        .at[slot]
        .max(jnp.where(write, k, -1), mode="drop")
    )
    write = write & (last_writer[jnp.minimum(slot, capacity - 1)] == k)
    slot = jnp.where(write, slot, capacity)  # out of range: dropped
    num_new = valid_i.sum()
    return SLBufferState(
        obs=buffer.obs.at[slot].set(obs.astype(buffer.obs.dtype), mode="drop"),
        action_mask=buffer.action_mask.at[slot].set(action_mask, mode="drop"),
        action=buffer.action.at[slot].set(action, mode="drop"),
        seen=buffer.seen + num_new,
        size=jnp.minimum(capacity, buffer.size + num_new),
    )


def sample_sl_batch(rng: PRNGKeyArray, buffer: SLBufferState, batch_size: int):
    """Uniform sample (with replacement) of (obs, action_mask, action)."""
    indices = jax.random.randint(
        rng, (batch_size,), 0, jnp.maximum(buffer.size, 1), dtype=jnp.int32
    )
    return buffer.obs[indices], buffer.action_mask[indices], buffer.action[indices]


def draw_br_mode(
    rng: PRNGKeyArray, br_mode: BoolArray, done: BoolArray, eta: float
) -> BoolArray:
    """NFSP policy mixing for the step after env.step returned `done`.

    br_mode (num_envs, num_agents) says which players follow the best response
    in the current game. When a game ends (the env then auto-resets), every
    player of the new game independently draws the BR with probability eta and
    the average policy otherwise, and keeps it for the whole game.
    """
    new_mode = jax.random.bernoulli(rng, p=eta, shape=br_mode.shape)
    return jnp.where(done[:, None], new_mode, br_mode)


def sample_logits(logits: FloatArray, action_mask: BoolArray, rng: PRNGKeyArray):
    return jax.random.categorical(rng, jnp.where(action_mask, logits, -jnp.inf))


def sample_random_legal(action_mask: BoolArray, rng: PRNGKeyArray) -> IntArray:
    return jax.random.categorical(rng, jnp.where(action_mask, 0.0, -jnp.inf))


def greedy_random_ties(
    q_vals: FloatArray, action_mask: BoolArray, rng: PRNGKeyArray
) -> IntArray:
    """argmax over the legal actions, ties broken uniformly at random."""
    q_masked = jnp.where(action_mask, q_vals, -jnp.inf)
    ties = (q_masked == q_masked.max()) & action_mask
    return jax.random.categorical(rng, jnp.where(ties, 0.0, -jnp.inf))


# =============================================================================
# Checkpoints
# =============================================================================


def load_params(path: str, template: Any) -> Any:
    """Loads flax params written by save_checkpoints, failing loudly.

    The file must exist and hold exactly the parameter tree of `template`
    (same keys, shapes and dtypes); flax's from_bytes alone silently ignores
    extra keys and doesn't check shapes.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: '{path}'")
    with open(path, "rb") as f:
        restored = serialization.msgpack_restore(f.read())
    expected = serialization.to_state_dict(template)
    if jax.tree_util.tree_structure(restored) != jax.tree_util.tree_structure(
        expected
    ):
        raise ValueError(
            f"checkpoint {path} doesn't match the network: parameter tree "
            f"{jax.tree_util.tree_structure(restored)} vs expected "
            f"{jax.tree_util.tree_structure(expected)}"
        )
    for (key, got), want in zip(
        jax.tree_util.tree_flatten_with_path(restored)[0],
        jax.tree_util.tree_leaves(expected),
    ):
        if np.shape(got) != np.shape(want) or np.asarray(got).dtype != want.dtype:
            raise ValueError(
                f"checkpoint {path} doesn't match the network at "
                f"{jax.tree_util.keystr(key)}: {np.asarray(got).dtype}"
                f"{np.shape(got)} vs expected {want.dtype}{np.shape(want)}"
            )
    return serialization.from_state_dict(template, restored)


def save_checkpoints(config: dict, avg_params: Any, br_params: Any) -> str:
    """Saves the networks of every seed after training.

    Layout of {save_dir}/{job_type}_{timestamp}/:
        avg_policy_{i}.msgpack  average policy of seed i (NFSP's output strategy)
        br_{i}.msgpack          best response of seed i
        config.yaml             resolved config of the run
    i = 0..num_seeds-1 indexes the vmapped seeds (wandb run "{seed}_{i}"). Each
    file holds the flax params of one network, without a seed axis, written with
    flax.serialization.to_bytes; restore with load_params(path, template) where
    template = network.init(rng, obs). The networks map a Bluff observation to:
        PPO-NFSP: both ActorCriticDiscreteMLP(action_dim, fc_dim_size) ->
            (action logits, value); the average policy's value head is unused.
        PQN-NFSP: average policy ActorDiscreteMLP(action_dim, fc_dim_size) ->
            action logits; BR QNetworkDiscreteMLP(action_dim, fc_dim_size) ->
            Q-values (its policy is the argmax over legal actions).
    Mask illegal actions (env.get_avail_actions) before the softmax or argmax.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(config["save_dir"], f"{config['job_type']}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    for name, params in (("avg_policy", avg_params), ("br", br_params)):
        for i in range(config["num_seeds"]):
            params_i = jax.tree_util.tree_map(lambda x: np.asarray(x[i]), params)
            with open(os.path.join(run_dir, f"{name}_{i}.msgpack"), "wb") as f:
                f.write(serialization.to_bytes(params_i))
    OmegaConf.save(OmegaConf.create(config), os.path.join(run_dir, "config.yaml"))
    return run_dir


# =============================================================================
# Evaluation
# =============================================================================

# act(params, obs, action_mask, rng) -> action, for one observation
Policy = Callable[[Any, FloatArray, BoolArray, PRNGKeyArray], IntArray]


class EvalResult(NamedTuple):
    """Per game, from the learner's point of view."""

    win: FloatArray  # the learner emptied its hand first
    loss: FloatArray  # another player did
    draw: FloatArray  # cut off at the horizon without a winner
    ret: FloatArray  # the learner's total reward over the game
    length: FloatArray  # env steps


def play_games(
    env: Bluff,
    learner: Policy,
    learner_params: Any,
    baseline: Policy,
    baseline_params: Any,
    num_games: int,
    rng: PRNGKeyArray,
) -> EvalResult:
    """Plays num_games games in parallel, the learner against baseline copies.

    The learner's seat rotates over the games, starting from a random seat
    (game g: seat (offset + g) % num_agents), and the env draws the first
    player at random, so no seat is favoured. Every game is played until it
    ends or reaches the env horizon.
    """
    n = env.num_agents
    rng_seat, rng_reset, rng_play = jax.random.split(rng, 3)
    offset = jax.random.randint(rng_seat, (), 0, n)
    seats = (offset + jnp.arange(num_games)) % n
    states, obs = jax.vmap(env.reset)(jax.random.split(rng_reset, num_games))
    games = jnp.arange(num_games)
    zeros = jnp.zeros((num_games,), dtype=jnp.float32)

    def cond(carry):
        _, _, finished, _, _ = carry
        return ~finished.all()

    def body(carry):
        states, obs, finished, result, rng = carry
        rng, rng_learner, rng_baseline, rng_step = jax.random.split(rng, 4)
        mask = jax.vmap(env.get_avail_actions)(states)
        act = jax.vmap(learner, in_axes=(None, 0, 0, 0))(
            learner_params, obs, mask, jax.random.split(rng_learner, num_games)
        )
        act_baseline = jax.vmap(baseline, in_axes=(None, 0, 0, 0))(
            baseline_params, obs, mask, jax.random.split(rng_baseline, num_games)
        )
        action = jnp.where(states.current_player_idx == seats, act, act_baseline)
        next_states, next_obs, reward, absorbing, done, info = jax.vmap(env.step_env)(
            jax.random.split(rng_step, num_games), states, action
        )
        live = ~finished
        ends = live & done
        won = info["game_winner"][games, seats]
        result = EvalResult(
            win=jnp.where(ends & won, 1.0, result.win),
            loss=jnp.where(ends & absorbing[:, 0] & ~won, 1.0, result.loss),
            draw=jnp.where(ends & ~absorbing[:, 0], 1.0, result.draw),
            ret=result.ret + jnp.where(live, reward[games, seats], 0.0),
            length=result.length + live,
        )

        def keep_finished(new, old):
            return jnp.where(finished.reshape((-1,) + (1,) * (new.ndim - 1)), old, new)

        states = jax.tree_util.tree_map(keep_finished, next_states, states)
        obs = keep_finished(next_obs, obs)
        return states, obs, finished | done, result, rng

    init = (
        states,
        obs,
        jnp.zeros((num_games,), dtype=bool),
        EvalResult(zeros, zeros, zeros, zeros, zeros),
        rng_play,
    )
    _, _, _, result, _ = lax.while_loop(cond, body, init)
    return result


def eval_metrics(name: str, result: EvalResult) -> dict[str, FloatArray]:
    """Means over the evaluation games; win_rate_decided excludes draws."""
    decided = result.win.sum() + result.loss.sum()
    return {
        f"win_rate_{name}": result.win.mean(),
        f"loss_rate_{name}": result.loss.mean(),
        f"draw_rate_{name}": result.draw.mean(),
        f"win_rate_decided_{name}": result.win.sum() / jnp.maximum(decided, 1.0),
        f"avg_return_{name}": result.ret.mean(),
        f"avg_episode_length_{name}": result.length.mean(),
    }


# =============================================================================
# Host-side logging (called through jax.experimental.io_callback)
# =============================================================================


def log_update(logger, evals_by_seed: dict, seed_val, metric_dict: dict) -> None:
    """Logs the metrics of one update of one seed.

    metric_dict["evaluated"] says whether the evaluation metrics (keys with
    "_vs_") were computed in this update; they are logged, printed and kept in
    evals_by_seed[seed] = [(env_steps, {metric: value}), ...] only then.
    """
    seed = int(seed_val)
    metrics = {k: np.asarray(v) for k, v in metric_dict.items()}
    evaluated = bool(metrics.pop("evaluated"))
    eval_keys = [k for k in metrics if "_vs_" in k]
    if evaluated:
        evals = {k: float(metrics[k]) for k in eval_keys}
        evals_by_seed.setdefault(seed, []).append((int(metrics["env_steps"]), evals))
        print(
            f"seed {seed} update {int(metrics['update_step'])} "
            f"({int(metrics['env_steps'])} env steps): {format_eval(evals)}",
            flush=True,
        )
    else:
        for k in eval_keys:
            del metrics[k]
    logger.log(seed, metrics)


def format_eval(evals: dict[str, float]) -> str:
    names = sorted(
        k[len("win_rate_"):]
        for k in evals
        if k.startswith("win_rate_") and not k.startswith("win_rate_decided_")
    )
    return "; ".join(
        f"{name} W/L/D {evals[f'win_rate_{name}']:.3f}/"
        f"{evals[f'loss_rate_{name}']:.3f}/{evals[f'draw_rate_{name}']:.3f} "
        f"(win rate of decided games {evals[f'win_rate_decided_{name}']:.3f}), "
        f"return {evals[f'avg_return_{name}']:.2f}, "
        f"length {evals[f'avg_episode_length_{name}']:.0f}"
        for name in names
    )


def print_final_evals(evals_by_seed: dict) -> None:
    """The last evaluation of every seed and the mean over seeds."""
    finals = {seed: evals[-1][1] for seed, evals in sorted(evals_by_seed.items())}
    for seed, evals in finals.items():
        print(f"final evaluation seed {seed}: {format_eval(evals)}")
    if finals:
        keys = next(iter(finals.values())).keys()
        mean = {k: float(np.mean([e[k] for e in finals.values()])) for k in keys}
        print(f"final evaluation, mean of {len(finals)} seed(s): {format_eval(mean)}")
