"""
Parts shared by bluff_ppo_nfsp.py and bluff_pqn_nfsp.py: the NFSP reservoir
buffer, the per-game policy mixing, the half- and full-budget checkpoints with
strict loading, the head-to-head evaluation, and logging.
"""

from typing import Any, Callable, NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from bluffjax.environments.bluff.bluff import Bluff
from bluffjax.examples.run_utils import (
    CHECKPOINT_STAGES,
    ResultsLog,
    load_params,
    save_seed_params,
)
from bluffjax.examples.run_utils import checkpoint_path as _checkpoint_path
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


class Checkpoint(NamedTuple):
    """The two NFSP networks at one point of training."""

    avg_params: Any  # average policy, NFSP's output strategy
    br_params: Any  # best response


def checkpoint_path(run_dir: str, stage: str, name: str, seed_index: int) -> str:
    """{run_dir}/{stage}/{name}_{seed_index}.msgpack, name avg_policy or br."""
    if name not in ("avg_policy", "br"):
        raise ValueError(f"name must be 'avg_policy' or 'br', got '{name}'")
    return _checkpoint_path(run_dir, stage, name, seed_index)


def save_checkpoints(
    run_dir: str, num_seeds: int, half: Checkpoint, full: Checkpoint
) -> str:
    """Saves the networks of every seed at half and at the full budget.

    Layout of the run directory (bluffjax.examples.run_utils.make_run_dir):
        config.yaml                  resolved config of the run
        results.jsonl                everything logged (see run_utils)
        half/avg_policy_{i}.msgpack  after half_update_steps(num_update_steps)
        half/br_{i}.msgpack          (= num_update_steps // 2) updates
        full/avg_policy_{i}.msgpack  after all num_update_steps updates
        full/br_{i}.msgpack
    Both checkpoints come from the same run: the half one is the state after
    exactly that many updates, with the learning-rate and epsilon schedules
    running over the whole run (config.yaml records num_update_steps and
    half_update_steps). i = 0..num_seeds-1 indexes the vmapped seeds (wandb
    run "{seed}_{i}"). Each file holds the flax params of one network, without
    a seed axis, written with flax.serialization.to_bytes; restore one with
    load_params(path, template) (or load_checkpoint), where template =
    network.init(rng, obs). The networks map a Bluff observation of the player
    to act (env.obs_from_state(state), env.obs_dim floats) to:
        PPO-NFSP: both ActorCriticDiscreteMLP(action_dim, fc_dim_size) ->
            (action logits, value in env reward units); the average policy's
            value head is unused.
        PQN-NFSP: average policy ActorDiscreteMLP(action_dim, fc_dim_size) ->
            action logits; BR QNetworkDiscreteMLP(action_dim, fc_dim_size) ->
            Q-values in env reward units (its policy is the argmax over legal
            actions).
    action_dim is env.action_dim; mask illegal actions
    (env.get_avail_actions(state)) before the softmax or argmax. As an
    evaluation opponent (compare_mode=checkpoint) use compare_network_type
    actor_critic for PPO-NFSP files, actor for PQN-NFSP avg_policy files and
    q_network for PQN-NFSP br files.
    """
    for stage, checkpoint in zip(CHECKPOINT_STAGES, (half, full)):
        for name, params in (
            ("avg_policy", checkpoint.avg_params),
            ("br", checkpoint.br_params),
        ):
            save_seed_params(run_dir, stage, name, params, range(num_seeds))
    return run_dir


def load_checkpoint(
    run_dir: str, stage: str, seed_index: int, avg_template: Any, br_template: Any
) -> Checkpoint:
    """Both networks of one seed and stage ("half" or "full"), loaded strictly."""
    return Checkpoint(
        avg_params=load_params(
            checkpoint_path(run_dir, stage, "avg_policy", seed_index), avg_template
        ),
        br_params=load_params(
            checkpoint_path(run_dir, stage, "br", seed_index), br_template
        ),
    )


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
    seat: IntArray  # the learner's seat


def play_games(
    env: Bluff,
    learner: Policy,
    learner_params: Any,
    baseline: Policy,
    baseline_params: Any,
    num_games: int,
    rng: PRNGKeyArray,
) -> EvalResult:
    """Plays num_games complete games in parallel, the learner in one seat and
    baseline copies in every other seat.

    The learner's seat rotates over the games, starting from a random seat
    (game g: seat (offset + g) % num_agents), and the env draws the first
    player at random, so no seat is favoured. Every game is played with
    env.step_env (no auto-reset) until it ends or reaches the env horizon (a
    draw), and the loop runs until all of them are over.
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
        result = result._replace(
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
        EvalResult(zeros, zeros, zeros, zeros, zeros, seats),
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


def log_update(
    results_log: ResultsLog, evals_by_seed: dict, seed_val, metric_dict: dict
) -> None:
    """Logs the metrics of one update of one seed to wandb and results.jsonl.

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
    results_log.log(metrics, seed)


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
