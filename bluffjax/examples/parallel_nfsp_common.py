"""
Parts shared by the training scripts in bluffjax/examples. The PPO-NFSP and
PQN-NFSP scripts of the simultaneous-move games, Goofspiel
(bluffjax/examples/goofspiel) and Kemps (bluffjax/examples/kemps), use all of
it: the per-agent training targets, the NFSP reservoir buffer and per-game
policy mixing, the half- and full-budget checkpoints with strict loading, the
head-to-head evaluation with seat rotation, and logging. The Hold'em NFSP
scripts (bluffjax/examples/HUL, bluffjax/examples/HUNL) use its evaluation
metrics (EvalResult, eval_metrics); the Kuhn Poker and Leduc Hold'em scripts
(bluffjax/examples/kuhn, bluffjax/examples/leduc) use them too, through
bluffjax/examples/kuhn_leduc_eval.py, with the sampling helpers and the
final-evaluation summary (print_final_evals).

In a ParallelEnv every agent acts at every step: observations, action masks,
actions, rewards and absorbing flags carry a num_agents axis after the env
axis, `done` is one flag per env, and env.step auto-resets a game that is
done. Every agent's own trajectory is therefore every step of its games, and
its targets only involve its own rewards and observations: the scripts
compute them independently for every (env, agent) column.
"""

from typing import Any, Callable, NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from bluffjax.environments.env import ParallelEnv
from bluffjax.examples.run_utils import (
    CHECKPOINT_STAGES,
    ResultsLog,
    half_update_steps,
    load_params,
    save_seed_params,
)
from bluffjax.examples.run_utils import checkpoint_path as _checkpoint_path
from bluffjax.metrics import rate
from bluffjax.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray


# =============================================================================
# Training targets
# =============================================================================


def per_agent_gae(
    values: FloatArray,
    rewards: FloatArray,
    dones: BoolArray,
    truncated: BoolArray,
    last_value: FloatArray,
    gamma: float,
    gae_lambda: float,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """GAE for every agent of a simultaneous-move rollout.

    Shapes: T steps, then any batch shape B, e.g. (num_envs, num_agents).
        values: (T, *B) critic value of the agent's observation at step t.
        rewards: (T, *B) the agent's reward from env.step at step t.
        dones: (T, *B) the agent's game ended or was cut off at step t (the
            env then auto-resets, so step t + 1 starts a new game).
        truncated: (T, *B) the game was cut off at step t without being over
            (only where dones is set).
        last_value: (*B,) value of the observation after the last step.

    Every agent acts at every step, so its decision at step t is followed by
    its decision at t + 1 in the same game unless the game is done at t. The
    decision at a step where the game ends is terminal (no bootstrap). At the
    rollout cut-off the last decision bootstraps from last_value. The decision
    at a step where the game is truncated has no known outcome (the env
    replaces the observation after the cut by the next game's first one): it
    is marked invalid and its advantage is 0, so the decision before it
    bootstraps from its value with the lambda trace cut there, as at a
    rollout cut-off.

    Returns:
        advantages (T, *B), value targets (T, *B), valid (T, *B).
    """

    def body(carry, x):
        next_value, next_adv = carry
        value, reward, done, trunc = x
        delta = reward + gamma * jnp.where(done, 0.0, next_value) - value
        adv = delta + gamma * gae_lambda * jnp.where(done, 0.0, next_adv)
        valid = ~(done & trunc)
        adv = jnp.where(valid, adv, 0.0)
        return (value, adv), (adv, valid)

    init = (last_value, jnp.zeros_like(last_value))
    _, (advantages, valid) = lax.scan(
        body, init, (values, rewards, dones, truncated), reverse=True
    )
    return advantages, advantages + values, valid


def per_agent_q_lambda_targets(
    q_max: FloatArray,
    rewards: FloatArray,
    dones: BoolArray,
    truncated: BoolArray,
    traces: FloatArray,
    last_q_max: FloatArray,
    gamma: float,
) -> tuple[FloatArray, BoolArray]:
    """Peng's Q(lambda) targets for every agent of a simultaneous-move rollout.

    Shapes: T steps, then any batch shape B, e.g. (num_envs, num_agents).
        q_max: (T, *B) max over the legal actions of Q(obs_t) of the agent.
        rewards, dones, truncated: as in per_agent_gae.
        traces: (T, *B) trace coefficient of the decision at step t: lambda if
            the agent took it with the epsilon-greedy BR, 0 if with the
            average policy.
        last_q_max: (*B,) max over the legal actions of Q at the observation
            after the last step.

    For the decision at step t, with c = traces[t + 1] the coefficient of the
    agent's next decision in the same game,
        G_t = r_t + gamma * ((1 - c) * q_max[t + 1] + c * G_{t + 1}),
    i.e. Peng's Q(lambda) as in PQN through BR decisions and the one-step
    Q-learning target before an average-policy decision. If the game ends at
    t, G_t = r_t. At the rollout cut-off G_t = r_t + gamma * last_q_max. The
    decision at a step where the game is truncated has no known outcome: it
    is marked invalid, and the decision before it bootstraps from q_max there
    without the trace.

    Returns:
        targets (T, *B), valid (T, *B).
    """

    def body(carry, x):
        next_q, next_return, next_trace = carry
        q_t, reward, done, trunc, trace = x
        bootstrap = (1.0 - next_trace) * next_q + next_trace * next_return
        ret = reward + gamma * jnp.where(done, 0.0, bootstrap)
        valid = ~(done & trunc)
        carry = (q_t, jnp.where(valid, ret, q_t), trace)
        return carry, (jnp.where(valid, ret, 0.0), valid)

    init = (last_q_max, last_q_max, jnp.zeros_like(last_q_max))
    _, (targets, valid) = lax.scan(
        body, init, (q_max, rewards, dones, truncated, traces), reverse=True
    )
    return targets, valid


# =============================================================================
# NFSP: reservoir buffer and policy mixing
# =============================================================================


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

    br_mode (num_envs, num_agents) says which agents follow the best response
    in the current game. When a game is done (the env then auto-resets), every
    agent of the new game independently draws the BR with probability eta and
    the average policy otherwise, and keeps it for the whole game.
    """
    new_mode = jax.random.bernoulli(rng, p=eta, shape=br_mode.shape)
    return jnp.where(done[:, None], new_mode, br_mode)


# =============================================================================
# Policies: act(params, obs, action_mask, rng) -> action, for any batch shape
# =============================================================================


def sample_logits(logits: FloatArray, action_mask: BoolArray, rng: PRNGKeyArray):
    return jax.random.categorical(rng, jnp.where(action_mask, logits, -jnp.inf))


def sample_random_legal(action_mask: BoolArray, rng: PRNGKeyArray) -> IntArray:
    return jax.random.categorical(rng, jnp.where(action_mask, 0.0, -jnp.inf))


def greedy_random_ties(
    q_vals: FloatArray, action_mask: BoolArray, rng: PRNGKeyArray
) -> IntArray:
    """argmax over the legal actions, ties broken uniformly at random."""
    q_masked = jnp.where(action_mask, q_vals, -jnp.inf)
    ties = (q_masked == q_masked.max(axis=-1, keepdims=True)) & action_mask
    return jax.random.categorical(rng, jnp.where(ties, 0.0, -jnp.inf))


def epsilon_greedy_probs(
    greedy_action: IntArray, action_mask: BoolArray, eps: FloatArray
) -> FloatArray:
    """Action distribution of the PQN scripts' epsilon-greedy BR: a uniformly
    random legal action with probability eps, greedy_action otherwise.
    greedy_action (...,), action_mask (..., num_actions) -> (..., num_actions)."""
    uniform = action_mask / action_mask.sum(axis=-1, keepdims=True)
    greedy = jax.nn.one_hot(greedy_action, action_mask.shape[-1], dtype=jnp.float32)
    return eps * uniform + (1.0 - eps) * greedy


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
    network.init(rng, obs). The networks map one agent's observation
    (env.obs_from_state(state)[agent], env.obs_dim floats) to:
        PPO-NFSP: both ActorCriticDiscreteMLP(action_dim, fc_dim_size) ->
            (action logits, value in env reward units); the
            average policy's value head is unused.
        PQN-NFSP: average policy ActorDiscreteMLP(action_dim, fc_dim_size) ->
            action logits; BR QNetworkDiscreteMLP(action_dim, fc_dim_size) ->
            Q-values in env reward units (its policy is the
            argmax over legal actions).
    action_dim is env.num_actions; mask illegal actions
    (env.get_avail_actions(state)[agent]) before the softmax or argmax. As an
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

# act(params, obs, action_mask, rng) -> action, for any batch shape
Policy = Callable[[Any, FloatArray, BoolArray, PRNGKeyArray], IntArray]


class GameRecord(NamedTuple):
    returns: FloatArray  # (num_games, num_agents) total env reward of every seat
    length: FloatArray  # (num_games,) env steps
    info: Any  # env info of each game's last step


class EvalResult(NamedTuple):
    """Per game, from the learner's point of view."""

    win: FloatArray
    loss: FloatArray
    draw: FloatArray
    ret: FloatArray  # the learner's total env reward over the game
    length: FloatArray  # env steps


def rotating_seats(rng: PRNGKeyArray, num_games: int, num_agents: int) -> BoolArray:
    """(num_games, num_agents): the learner plays seat (offset + g) % num_agents
    of game g, from a random offset, and the baseline every other seat."""
    offset = jax.random.randint(rng, (), 0, num_agents)
    seats = (offset + jnp.arange(num_games)) % num_agents
    return jax.nn.one_hot(seats, num_agents, dtype=jnp.bool_)


def rotating_teams(
    rng: PRNGKeyArray, num_games: int, team_of_seat: IntArray, num_teams: int
) -> BoolArray:
    """(num_games, num_agents): the learner plays every seat of team
    (offset + g) % num_teams in game g, from a random offset, and the baseline
    every seat of the other teams. team_of_seat: (num_agents,) team ids."""
    offset = jax.random.randint(rng, (), 0, num_teams)
    teams = (offset + jnp.arange(num_games)) % num_teams
    return team_of_seat[None, :] == teams[:, None]


def play_games(
    env: ParallelEnv,
    learner: Policy,
    learner_params: Any,
    baseline: Policy,
    baseline_params: Any,
    learner_seats: BoolArray,
    rng: PRNGKeyArray,
) -> GameRecord:
    """Plays one game per row of learner_seats (num_games, num_agents), all in
    parallel: in game g, seat p follows the learner if learner_seats[g, p] and
    the baseline otherwise. Every game is played with env.step_env (no
    auto-reset) until it is done, i.e. ends or reaches the env horizon.
    """
    num_games = learner_seats.shape[0]
    rng_reset, rng_play = jax.random.split(rng)
    states, obs = jax.vmap(env.reset)(jax.random.split(rng_reset, num_games))
    step = jax.vmap(env.step_env)
    info = jax.tree_util.tree_map(  # zeros shaped like the env's info
        lambda x: jnp.zeros(x.shape, x.dtype),
        jax.eval_shape(
            step,
            jax.random.split(rng_play, num_games),
            states,
            jnp.zeros(learner_seats.shape, dtype=jnp.int32),
        )[-1],
    )

    def where_games(games, new, old):
        return jnp.where(games.reshape((-1,) + (1,) * (new.ndim - 1)), new, old)

    def cond(carry):
        return ~carry[2].all()

    def body(carry):
        states, obs, finished, returns, length, info, rng = carry
        rng, rng_learner, rng_baseline, rng_step = jax.random.split(rng, 4)
        mask = jax.vmap(env.get_avail_actions)(states)
        action = jnp.where(
            learner_seats,
            learner(learner_params, obs, mask, rng_learner),
            baseline(baseline_params, obs, mask, rng_baseline),
        )
        next_states, next_obs, reward, _, done, step_info = step(
            jax.random.split(rng_step, num_games), states, action
        )
        live = ~finished
        ends = live & done
        returns = returns + jnp.where(live[:, None], reward, 0.0)
        length = length + live
        info = jax.tree_util.tree_map(
            lambda new, old: where_games(ends, new, old), step_info, info
        )
        states = jax.tree_util.tree_map(
            lambda new, old: where_games(live, new, old), next_states, states
        )
        obs = where_games(live, next_obs, obs)
        return states, obs, finished | done, returns, length, info, rng

    init = (
        states,
        obs,
        jnp.zeros((num_games,), dtype=bool),
        jnp.zeros(learner_seats.shape, dtype=jnp.float32),
        jnp.zeros((num_games,), dtype=jnp.float32),
        info,
        rng_play,
    )
    _, _, _, returns, length, info, _ = lax.while_loop(cond, body, init)
    return GameRecord(returns=returns, length=length, info=info)


def evaluate_goofspiel(
    env: ParallelEnv,
    learner: Policy,
    learner_params: Any,
    baseline: Policy,
    baseline_params: Any,
    num_games: int,
    rng: PRNGKeyArray,
) -> EvalResult:
    """Goofspiel: the learner plays one seat, rotating over the games (with two
    players it alternates seats), the baseline every other seat. Win: the
    learner is the unique top scorer (info["game_winner"]); loss: another
    player scored more than the learner (with three or more players, possibly
    several tied above it); draw: the learner tied for the top score. The
    return is the learner's total reward, i.e. its points minus the opponents'
    mean points."""
    rng_seats, rng_play = jax.random.split(rng)
    seats = rotating_seats(rng_seats, num_games, env.num_agents)
    record = play_games(
        env, learner, learner_params, baseline, baseline_params, seats, rng_play
    )
    points = record.info["points"]
    win = (record.info["game_winner"] & seats).any(axis=-1)
    loss = (points * seats).sum(axis=-1) < points.max(axis=-1)
    return EvalResult(
        win=win.astype(jnp.float32),
        loss=loss.astype(jnp.float32),
        draw=(~win & ~loss).astype(jnp.float32),
        ret=(record.returns * seats).sum(axis=-1),
        length=record.length,
    )


def evaluate_kemps(
    env: ParallelEnv,
    learner: Policy,
    learner_params: Any,
    baseline: Policy,
    baseline_params: Any,
    num_games: int,
    rng: PRNGKeyArray,
) -> EvalResult:
    """Kemps: the learner plays both partners of one team, rotating over the
    teams (and so the seats) from game to game, the baseline every other team.
    The env pays both partners their team's reward, so the learner's return is
    its team's: win if positive (the other team took the letter), loss if
    negative, draw if zero (a real deal or a hand cut off at the horizon)."""
    rng_seats, rng_play = jax.random.split(rng)
    team_of_seat = env._team_id(jnp.arange(env.num_agents))
    seats = rotating_teams(rng_seats, num_games, team_of_seat, env.num_teams)
    record = play_games(
        env, learner, learner_params, baseline, baseline_params, seats, rng_play
    )
    ret = (record.returns * seats).sum(axis=-1) / seats.sum(axis=-1)
    return EvalResult(
        win=(ret > 0).astype(jnp.float32),
        loss=(ret < 0).astype(jnp.float32),
        draw=(ret == 0).astype(jnp.float32),
        ret=ret,
        length=record.length,
    )


def eval_metrics(name: str, result: EvalResult) -> dict[str, FloatArray]:
    """Means over the evaluation games, as {stat}_{name}; win_rate_decided
    excludes draws and is NaN if every game was a draw (bluffjax.metrics.rate).
    name is vs_{opponent} for the output policy (NFSP's average policy) and
    br_vs_{opponent} for NFSP's best response."""
    decided = result.win.sum() + result.loss.sum()
    return {
        f"win_rate_{name}": result.win.mean(),
        f"loss_rate_{name}": result.loss.mean(),
        f"draw_rate_{name}": result.draw.mean(),
        f"win_rate_decided_{name}": rate(result.win.sum(), decided),
        f"avg_return_{name}": result.ret.mean(),
        f"avg_episode_length_{name}": result.length.mean(),
    }


# =============================================================================
# Kemps: hands cut off at the horizon
# =============================================================================


def kemps_horizon_cutoff(
    env: ParallelEnv, state: Any, action: IntArray, done: BoolArray
) -> BoolArray:
    """(num_envs,) whether env.step cut the hand off at the horizon.

    state: the (batched) states the actions were taken in; action:
    (num_envs, num_agents); done: as returned by env.step. A Kemps hand ends
    when somebody declares, on a real deal (everybody plays NOOP and the stock
    can't refill the centre) or when it reaches the horizon. The env sets
    `absorbing` in all three cases, so the cut-off is recognised here: the
    hand is done although nobody declared and it wasn't a real deal.
    """
    game_action = action // env.comm_dim
    declared = (game_action >= env.action_kemps).any(axis=-1)
    all_noop = (game_action == env.action_noop).all(axis=-1)
    can_refill = state.deck_idx + env.num_center <= state.deck.shape[-1]
    return done & ~declared & ~(all_noop & ~can_refill)


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
        f"length {evals[f'avg_episode_length_{name}']:.1f}"
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
