"""
PQN-NFSP for Werewolf.

Neural Fictitious Self-Play (Heinrich & Silver 2016, arXiv:1603.01121) with
PQN (Gallici et al. 2024, "Simplifying Deep Temporal Difference Learning") as
the best-response (BR) learner in place of DQN. All players share one
Q-network and one average-policy network; the observation is relative to the
player to act and holds its role (and, for a werewolf, its teammates).

- Policy mixing: at the start of every game each player independently draws
  the policy it follows for the whole game, the epsilon-greedy BR with
  probability anticipatory_eta and the average policy otherwise.
- BR learning: as in NFSP the Q-network learns from all of a player's
  decisions, whichever policy it followed. Targets follow each player's own
  decisions (a decision's reward is everything the player receives until its
  next decision or the end of the game; the env pays every player its team's
  outcome when the game ends, whoever moves) and bootstrap from the max over
  legal actions of Q at that next decision. With six players in two teams the
  next move is often a teammate's or an opponent's, so values are never
  negated or shared across players. The lambda-trace runs through decisions
  taken by the epsilon-greedy BR (Peng's Q(lambda), as in PQN) and is cut at
  average-policy decisions, so those give one-step Q-learning targets (see
  per_player_q_lambda_targets).
- Average policy: every BR decision (obs, action) is added to a reservoir
  buffer (Algorithm R) and the average policy is fit to it by cross-entropy.
  The average policy is NFSP's output strategy; the BR is a training device.
- Evaluation: compare_steps games against the opponent (uniformly random
  legal play or a checkpoint in every other seat), game i with the learner in
  seat i % num_agents. Roles are dealt at random, so the learner plays both
  teams; win rates are reported overall and by the learner's team. The BR is
  evaluated greedily.

Units: the env pays +10 to the winning team and -10 to the losing one;
Q-values and logged returns are in these units.

Checkpoints (save_final): see save_checkpoints.
"""

import datetime
import os
import time
from typing import Any, Callable, NamedTuple

import distrax
from flax import serialization
from flax.training.train_state import TrainState
import hydra
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax

from bluffjax.utils.typing import (
    BoolArray,
    FloatArray,
    IntArray,
    PRNGKeyArray,
)
from bluffjax import make
from bluffjax.environments.werewolf.werewolf import WEREWOLF, WerewolfState
from bluffjax.networks.mlp import (
    ActorCriticDiscreteMLP,
    ActorDiscreteMLP,
    QNetworkDiscreteMLP,
)
from bluffjax.utils.jax_utils import pytree_norm
from bluffjax.utils.paths import register_resolvers
from bluffjax.utils.wandb_multilogger import WandbMultiLogger

LOGGER = None
# EVALUATIONS[seed] = [(update, env_steps, {metric: value}), ...]
EVALUATIONS: dict[int, list[tuple[int, int, dict[str, float]]]] = {}


class Transition(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    reward: FloatArray  # (num_agents,) env reward for every player
    done: BoolArray  # the game ended at this step (the env auto-resets)
    q_max: FloatArray  # max over legal actions of Q(obs)
    player_idx: IntArray
    is_br: BoolArray  # the acting player follows the BR in this game
    br_mode: BoolArray  # (num_agents,) which players follow the BR in this game
    roles: IntArray  # (num_agents,) roles in this game
    team_won: FloatArray  # (num_agents,) 1 if the player's team won at this step
    game_length: IntArray  # steps of the game so far, after this step


class SLBufferState(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    seen: IntArray
    size: IntArray


class RunnerState(NamedTuple):
    br_train_state: TrainState
    avg_train_state: TrainState
    sl_buffer: SLBufferState
    state: WerewolfState
    obs: FloatArray
    done: BoolArray
    br_mode: BoolArray  # (num_envs, num_agents) policy drawn for the current game
    update_step: IntArray
    rng: PRNGKeyArray


class PQNUpdateState(NamedTuple):
    train_state: TrainState
    transitions: Transition
    targets: FloatArray
    train_mask: BoolArray
    rng: PRNGKeyArray


class SLBatch(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray


class SLUpdateState(NamedTuple):
    train_state: TrainState
    sl_buffer: SLBufferState
    rng: PRNGKeyArray


def _pick(arr: FloatArray, player: IntArray) -> FloatArray:
    """arr[n, player[n]] for an (N, num_agents) array."""
    return jnp.take_along_axis(arr, player[:, None], axis=-1).squeeze(-1)


def per_player_q_lambda_targets(
    q_max: FloatArray,
    rewards: FloatArray,
    dones: BoolArray,
    players: IntArray,
    traces: FloatArray,
    last_q_max: FloatArray,
    last_player: IntArray,
    gamma: float,
) -> tuple[FloatArray, BoolArray]:
    """Q(lambda) targets along each player's own decisions in a turn-based rollout.

    Shapes: T steps, N envs, P players.
        q_max: (T, N) max over legal actions of Q(obs_t) (player acting at t).
        rewards: (T, N, P) reward vector returned by env.step at step t.
        dones: (T, N) the game ended at step t (the env then auto-resets).
        players: (T, N) player acting at step t.
        traces: (T, N) trace coefficient of the decision at t: lambda if it was
            taken by the epsilon-greedy BR, 0 if by the average policy.
        last_q_max: (N,) max over legal actions of Q at the observation after
            the last step.
        last_player: (N,) player to act after the last step.

    For player p's decision at step t with next own decision t' in the same
    game, R = p's rewards from step t up to t' (rewards paid on other players'
    steps included) and
        G_t = R + gamma * ((1 - c) * q_max[t'] + c * G_t'),  c = traces[t'],
    i.e. Peng's Q(lambda) as in PQN when the next decision follows the BR, and
    the one-step Q-learning target when it follows the average policy. If the
    game ends first, G_t = R. At the rollout cut-off only the player to act has
    a next decision (G_t = R + gamma * last_q_max). The other players' trailing
    decisions in an unfinished game have no known outcome yet: they are marked
    invalid, and those players' earlier decisions bootstrap from q_max there
    without the trace.

    Returns:
        targets (T, N), valid (T, N).
    """
    num_agents = rewards.shape[-1]
    last_onehot = jax.nn.one_hot(last_player, num_agents, dtype=jnp.bool_)
    zeros = jnp.zeros(last_onehot.shape, dtype=jnp.float32)
    last_q = jnp.where(last_onehot, last_q_max[:, None], 0.0)
    # Per player, about its next decision (going backwards): q_max, return
    # (= q_max if the trace stops there), trace coefficient, whether it is
    # known, whether the game ends before it, and the reward collected since
    # the decision being processed.
    init = (last_q, last_q, zeros, last_onehot, jnp.zeros_like(last_onehot), zeros)

    def body(carry, x):
        next_q, next_return, next_trace, known, terminal, reward_acc = carry
        q_max_t, reward, done, player, trace = x
        # Steps after a game-ending step belong to the next game.
        game_over = done[:, None]
        known = known | game_over
        terminal = terminal | game_over
        reward_acc = jnp.where(game_over, 0.0, reward_acc) + reward

        not_terminal = 1.0 - _pick(terminal, player).astype(jnp.float32)
        c = _pick(next_trace, player)
        bootstrap = (1.0 - c) * _pick(next_q, player) + c * _pick(next_return, player)
        ret = _pick(reward_acc, player) + gamma * not_terminal * bootstrap
        valid = _pick(known, player)

        acting = jax.nn.one_hot(player, num_agents, dtype=jnp.bool_)
        next_q = jnp.where(acting, q_max_t[:, None], next_q)
        next_return = jnp.where(
            acting, jnp.where(valid, ret, q_max_t)[:, None], next_return
        )
        next_trace = jnp.where(acting, trace[:, None], next_trace)
        known = known | acting
        terminal = terminal & ~acting
        reward_acc = jnp.where(acting, 0.0, reward_acc)
        carry = (next_q, next_return, next_trace, known, terminal, reward_acc)
        return carry, (jnp.where(valid, ret, 0.0), valid)

    _, (targets, valid) = lax.scan(
        body, init, (q_max, rewards, dones, players, traces), reverse=True
    )
    return targets, valid


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


def load_params(path: str, template: Any, what: str) -> Any:
    """Loads one network's flax params saved by save_checkpoints. Fails unless
    the file holds exactly the template's parameter tree and shapes."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{what} checkpoint not found: '{path}'")
    with open(path, "rb") as f:
        restored = serialization.msgpack_restore(f.read())

    def leaves(tree):
        flat, _ = jax.tree_util.tree_flatten_with_path(tree)
        return {jax.tree_util.keystr(p): tuple(np.shape(v)) for p, v in flat}

    got = leaves(restored)
    expected = leaves(serialization.to_state_dict(template))
    if got != expected:
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        shapes = sorted(
            f"{k}: {got[k]} != {expected[k]}"
            for k in set(got) & set(expected)
            if got[k] != expected[k]
        )
        raise ValueError(
            f"{path} does not hold {what} parameters (missing {missing}, "
            f"unexpected {extra}, shape mismatches {shapes})"
        )
    return serialization.from_state_dict(template, restored)


def play_eval_games(
    env: Any,
    num_games: int,
    rng: PRNGKeyArray,
    learner_act: Callable,
    learner_params: Any,
    opponent_act: Callable,
    opponent_params: Any,
) -> dict[str, FloatArray]:
    """num_games games in parallel, game i with the learner in seat
    i % num_agents and the opponent in all the other seats.

    act(params, obs, action_mask, rng) -> action. Roles and the first speaker
    are dealt at random, so the learner plays both teams. Returns the learner's
    win rate overall and by its team, the fraction of games in which it was a
    werewolf, its mean return and the mean game length.
    """
    seats = jnp.arange(num_games) % env.num_agents

    def one_game(rng_game: PRNGKeyArray, seat: IntArray):
        rng_game, rng_reset = jax.random.split(rng_game)
        state, obs = env.reset(rng_reset)

        def body(carry):
            state_s, obs_s, rng_s, _, ret, length = carry
            rng_s, rng_l, rng_o, rng_step = jax.random.split(rng_s, 4)
            action_mask = env.get_avail_actions(state_s)
            action = jnp.where(
                state_s.current_player_idx == seat,
                learner_act(learner_params, obs_s, action_mask, rng_l),
                opponent_act(opponent_params, obs_s, action_mask, rng_o),
            )
            state_s, obs_s, reward, _, done, _ = env.step_env(rng_step, state_s, action)
            return state_s, obs_s, rng_s, done, ret + reward[seat], length + 1

        init = (state, obs, rng_game, jnp.bool_(False), jnp.float32(0), jnp.int32(0))
        _, _, _, _, ret, length = lax.while_loop(lambda c: ~c[3], body, init)
        return ret, length, state.roles[seat] == WEREWOLF

    ret, length, is_ww = jax.vmap(one_game)(jax.random.split(rng, num_games), seats)
    won = (ret > 0).astype(jnp.float32)
    ww = is_ww.astype(jnp.float32)
    human = 1.0 - ww
    return {
        "win_rate": won.mean(),
        "win_rate_as_werewolf": (won * ww).sum() / jnp.maximum(ww.sum(), 1.0),
        "win_rate_as_human": (won * human).sum() / jnp.maximum(human.sum(), 1.0),
        "learner_werewolf_frac": ww.mean(),
        "return": ret.mean(),
        "episode_length": length.astype(jnp.float32).mean(),
    }


def make_train(config: dict) -> Callable[[PRNGKeyArray, int], RunnerState]:
    env = make("werewolf", **config["env_kwargs"])
    sample_state, _ = env.reset(jax.random.PRNGKey(0))
    action_dim = int(env.get_avail_actions(sample_state).shape[-1])
    num_agents = env.num_agents
    config["batch_shuffle_dim"] = (
        config["num_steps_per_env_per_update"] * config["num_envs"]
    )

    br_network = QNetworkDiscreteMLP(
        action_dim=action_dim, hidden_dim=config["fc_dim_size"]
    )
    avg_network = ActorDiscreteMLP(
        action_dim=action_dim, hidden_dim=config["fc_dim_size"]
    )

    compare_enabled = config["compare"]
    compare_against_random = config["compare_mode"] == "random"
    compare_network_type = config["compare_network_type"]
    opponent = "random" if compare_against_random else "baseline"
    baseline_params = None
    baseline_network = None
    if compare_enabled and not compare_against_random:
        if compare_network_type == "actor_critic":
            baseline_network = ActorCriticDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )
        elif compare_network_type == "q_network":
            baseline_network = QNetworkDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )
        elif compare_network_type == "actor":
            baseline_network = ActorDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )
        else:
            raise ValueError(
                "compare_network_type must be one of "
                "['actor_critic', 'q_network', 'actor'], "
                f"got '{compare_network_type}'"
            )
        template = baseline_network.init(
            jax.random.PRNGKey(0), jnp.zeros((env.obs_dim,), dtype=jnp.float32)
        )
        baseline_params = load_params(
            config["compare_with"], template, f"{compare_network_type} baseline"
        )
        print(f"Loaded {compare_network_type} baseline from {config['compare_with']}")

    def linear_decay(count: int) -> float:
        frac = 1.0 - count / config["num_gradient_steps"]
        return config["lr"] * frac

    def epsilon_schedule(update_step: FloatArray) -> FloatArray:
        decay_steps = max(
            config["exploration_fraction"] * config["num_update_steps"], 1
        )
        frac = jnp.minimum(1.0, update_step.astype(jnp.float32) / decay_steps)
        return config["start_e"] + frac * (config["end_e"] - config["start_e"])

    def init_sl_buffer(obs_dim: int, action_dim: int) -> SLBufferState:
        capacity = config["sl_reservoir_capacity"]
        return SLBufferState(
            obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
            action_mask=jnp.zeros((capacity, action_dim), dtype=jnp.bool_),
            action=jnp.zeros((capacity,), dtype=jnp.int32),
            seen=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
        )

    def sample_sl_batch(
        rng: PRNGKeyArray,
        buffer: SLBufferState,
        batch_size: int,
    ) -> SLBatch:
        max_size = jnp.maximum(buffer.size, 1)
        indices = jax.random.randint(rng, (batch_size,), 0, max_size, dtype=jnp.int32)
        return SLBatch(
            obs=buffer.obs[indices],
            action_mask=buffer.action_mask[indices],
            action=buffer.action[indices],
        )

    def masked_mean(x: FloatArray, mask: FloatArray) -> FloatArray:
        denom = jnp.maximum(mask.sum(), 1.0)
        return (x * mask).sum() / denom

    # Evaluation policies: act(params, obs, action_mask, rng) -> action
    def act_actor_critic(net, params, obs, action_mask, rng):
        logits, _ = net.apply(params, obs)
        logits_masked = jnp.where(action_mask, logits, -jnp.inf)
        return distrax.Categorical(logits=logits_masked).sample(seed=rng)

    def act_actor(net, params, obs, action_mask, rng):
        logits = net.apply(params, obs)
        logits_masked = jnp.where(action_mask, logits, -jnp.inf)
        return distrax.Categorical(logits=logits_masked).sample(seed=rng)

    def act_q_greedy(net, params, obs, action_mask, rng):
        q_vals_masked = jnp.where(action_mask, net.apply(params, obs), -jnp.inf)
        ties = q_vals_masked == jnp.max(q_vals_masked)
        return jax.random.categorical(rng, jnp.where(ties, 0.0, -jnp.inf))

    def act_random(params, obs, action_mask, rng):
        return jax.random.categorical(rng, jnp.where(action_mask, 0.0, -jnp.inf))

    if compare_against_random:
        baseline_act = act_random
    else:
        baseline_act_fn = {
            "actor_critic": act_actor_critic,
            "q_network": act_q_greedy,
            "actor": act_actor,
        }[compare_network_type]

        def baseline_act(params, obs, action_mask, rng):
            return baseline_act_fn(baseline_network, params, obs, action_mask, rng)

    def run_compare_eval(
        rng: PRNGKeyArray,
        avg_params: Any,
        br_params: Any,
        opponent_params: Any,
    ) -> dict[str, FloatArray]:
        """Evaluation metrics of the average policy and the greedy BR."""
        rng_avg, rng_br = jax.random.split(rng)
        players = [
            (
                "avg",
                avg_params,
                rng_avg,
                lambda p, o, m, r: act_actor(avg_network, p, o, m, r),
            ),
            (
                "br",
                br_params,
                rng_br,
                lambda p, o, m, r: act_q_greedy(br_network, p, o, m, r),
            ),
        ]
        metrics = {}
        for name, params, rng_i, act in players:
            results = play_eval_games(
                env,
                config["compare_steps"],
                rng_i,
                act,
                params,
                baseline_act,
                opponent_params,
            )
            for key, value in results.items():
                metrics[f"eval/{key}_{name}_vs_{opponent}"] = value
        return metrics

    def logging_callback(seed_val, update, evaluated, metric_dict) -> None:
        seed_i, update = int(seed_val), int(update)
        metrics = {k: np.asarray(v) for k, v in metric_dict.items()}
        if not evaluated:
            metrics = {k: v for k, v in metrics.items() if not k.startswith("eval/")}
        else:
            evals = {k: float(v) for k, v in metrics.items() if k.startswith("eval/")}
            env_steps = int(metrics["env_steps"])
            EVALUATIONS.setdefault(seed_i, []).append((update, env_steps, evals))

            def rate(stat, name):
                return evals[f"eval/{stat}_{name}_vs_{opponent}"]

            line = ", ".join(
                f"{name} {rate('win_rate', name):.3f} "
                f"(werewolf {rate('win_rate_as_werewolf', name):.3f}, "
                f"human {rate('win_rate_as_human', name):.3f})"
                for name in ("avg", "br")
            )
            print(
                f"seed {seed_i} update {update + 1}/{config['num_update_steps']} "
                f"({env_steps} env steps): win rate vs {opponent}: {line}",
                flush=True,
            )
        if LOGGER is not None:
            LOGGER.log(seed_i, metrics)

    def train(rng: PRNGKeyArray, seed: int) -> RunnerState:
        def train_setup(
            rng_inner: PRNGKeyArray,
        ) -> tuple[
            TrainState,
            TrainState,
            WerewolfState,
            FloatArray,
            SLBufferState,
        ]:
            rng_inner, rng_reset = jax.random.split(rng_inner)
            rng_resets = jax.random.split(rng_reset, config["num_envs"])
            state, obs = jax.vmap(env.reset, in_axes=(0))(rng_resets)

            rng_inner, rng_br_init, rng_avg_init = jax.random.split(rng_inner, 3)
            br_params = br_network.init(rng_br_init, obs)
            avg_params = avg_network.init(rng_avg_init, obs)

            br_lr = linear_decay if config["anneal_lr"] else config["lr"]
            br_tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.radam(learning_rate=br_lr),
            )
            avg_tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=config["sl_lr"], eps=1e-5),
            )

            br_train_state = TrainState.create(
                apply_fn=br_network.apply, params=br_params, tx=br_tx
            )
            avg_train_state = TrainState.create(
                apply_fn=avg_network.apply, params=avg_params, tx=avg_tx
            )
            sl_buffer = init_sl_buffer(obs.shape[-1], action_dim)
            return (br_train_state, avg_train_state, state, obs, sl_buffer)

        rng, rng_setup, rng_mode = jax.random.split(rng, 3)
        br_train_state, avg_train_state, state, obs, sl_buffer = train_setup(rng_setup)
        br_mode = jax.random.bernoulli(
            rng_mode,
            p=config["anticipatory_eta"],
            shape=(config["num_envs"], num_agents),
        )

        def update_step(
            runner_state: RunnerState,
            unused: None,
        ) -> tuple[RunnerState, None]:
            def step(
                runner_state_inner: RunnerState,
                unused: None,
            ) -> tuple[RunnerState, Transition]:
                br_train_state_s = runner_state_inner.br_train_state
                avg_train_state_s = runner_state_inner.avg_train_state
                state_s = runner_state_inner.state
                obs_s = runner_state_inner.obs
                rng_s = runner_state_inner.rng

                br_mode_s = runner_state_inner.br_mode
                (
                    rng_s,
                    rng_mode,
                    rng_explore,
                    rng_random,
                    rng_avg,
                    rng_step,
                ) = jax.random.split(rng_s, 6)
                action_mask = jax.vmap(env.get_avail_actions, in_axes=(0))(state_s)

                q_vals = br_network.apply(br_train_state_s.params, obs_s)
                q_vals_masked = jnp.where(action_mask, q_vals, -jnp.inf)
                greedy_action = jnp.argmax(q_vals_masked, axis=-1)
                q_max = jnp.max(q_vals_masked, axis=-1)

                # epsilon-greedy BR over the legal actions
                eps = epsilon_schedule(
                    jnp.array(runner_state_inner.update_step, dtype=jnp.float32)
                )
                explore = jax.random.uniform(rng_explore, (config["num_envs"],)) < eps
                random_action = jax.random.categorical(
                    rng_random, jnp.where(action_mask, 0.0, -jnp.inf)
                )
                br_action = jnp.where(explore, random_action, greedy_action)

                avg_logits = avg_network.apply(avg_train_state_s.params, obs_s)
                avg_logits_masked = jnp.where(action_mask, avg_logits, -jnp.inf)
                avg_action = jax.random.categorical(rng_avg, avg_logits_masked)

                # NFSP: the acting player follows the policy it drew for this game.
                player_idx = state_s.current_player_idx
                is_br = _pick(br_mode_s, player_idx)
                action = jnp.where(is_br, br_action, avg_action)

                rng_steps = jax.random.split(rng_step, config["num_envs"])
                next_state, next_obs, reward, absorbing, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_steps, state_s, action)

                # A new game starts after done (the env auto-resets): every
                # player draws BR (prob. anticipatory_eta) or average policy.
                new_br_mode = jax.random.bernoulli(
                    rng_mode, p=config["anticipatory_eta"], shape=br_mode_s.shape
                )
                next_br_mode = jnp.where(done[:, None], new_br_mode, br_mode_s)

                transition = Transition(
                    obs=obs_s,
                    action_mask=action_mask,
                    action=action,
                    reward=reward,
                    done=done,
                    q_max=q_max,
                    player_idx=player_idx,
                    is_br=is_br,
                    br_mode=br_mode_s,
                    roles=state_s.roles,
                    team_won=info["game_winner"],
                    game_length=info["timestep"],
                )
                runner_state_inner = RunnerState(
                    br_train_state=br_train_state_s,
                    avg_train_state=avg_train_state_s,
                    sl_buffer=runner_state_inner.sl_buffer,
                    state=next_state,
                    obs=next_obs,
                    done=done,
                    br_mode=next_br_mode,
                    update_step=runner_state_inner.update_step,
                    rng=rng_s,
                )
                return runner_state_inner, transition

            runner_state, transitions = lax.scan(
                step,
                runner_state,
                None,
                config["num_steps_per_env_per_update"],
            )

            br_train_state = runner_state.br_train_state
            avg_train_state = runner_state.avg_train_state
            sl_buffer = runner_state.sl_buffer
            last_obs = runner_state.obs
            last_state = runner_state.state
            rng, rng_sl_append, rng_pqn_update, rng_sl_update, rng_compare = (
                jax.random.split(runner_state.rng, 5)
            )

            last_mask = jax.vmap(env.get_avail_actions, in_axes=(0))(last_state)
            last_q_vals = br_network.apply(br_train_state.params, last_obs)
            last_q_max = jnp.max(jnp.where(last_mask, last_q_vals, -jnp.inf), axis=-1)

            # Per-player Q(lambda) targets with each player's own reward (which
            # carries its team's outcome). Q-learning is off-policy, so (as in
            # NFSP) every decision with a known outcome trains the Q-network;
            # the trace is cut at average-policy decisions.
            targets, valid = per_player_q_lambda_targets(
                transitions.q_max,
                transitions.reward,
                transitions.done,
                transitions.player_idx,
                config["q_lambda"] * transitions.is_br.astype(jnp.float32),
                last_q_max,
                last_state.current_player_idx,
                config["gamma"],
            )
            train_mask = valid

            # NFSP: every BR decision goes to the reservoir.
            sl_buffer = reservoir_append(
                sl_buffer,
                transitions.obs.reshape(-1, transitions.obs.shape[-1]),
                transitions.action_mask.reshape(-1, transitions.action_mask.shape[-1]),
                transitions.action.reshape(-1),
                transitions.is_br.reshape(-1),
                rng_sl_append,
            )

            def update_epoch(
                pqn_state: PQNUpdateState,
                unused: None,
            ) -> tuple[PQNUpdateState, dict[str, FloatArray]]:
                rng_s, rng_permute = jax.random.split(pqn_state.rng)
                batch = (
                    pqn_state.transitions.obs,
                    pqn_state.transitions.action,
                    pqn_state.targets,
                    pqn_state.train_mask,
                )
                # (steps, envs, ...) -> (steps * envs, ...)
                batch_reshaped = jax.tree_util.tree_map(
                    lambda x: x.reshape((-1,) + x.shape[2:]), batch
                )
                permutation = jax.random.permutation(
                    rng_permute, config["batch_shuffle_dim"]
                )
                batch_shuffled = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0),
                    batch_reshaped,
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape(config["num_minibatches"], -1, *x.shape[1:]),
                    batch_shuffled,
                )

                def update_minibatch(
                    train_state: TrainState,
                    minibatch: tuple[FloatArray, IntArray, FloatArray, BoolArray],
                ) -> tuple[TrainState, dict[str, FloatArray]]:
                    obs_mb, action_mb, targets_mb, train_mask_mb = minibatch

                    def loss_fn(params):
                        q_vals = br_network.apply(params, obs_mb)
                        chosen_q = jnp.take_along_axis(
                            q_vals,
                            action_mb[..., None].astype(jnp.int32),
                            axis=-1,
                        ).squeeze(-1)
                        mask = train_mask_mb.astype(jnp.float32)
                        mask_denom = jnp.maximum(mask.sum(), 1.0)
                        td_loss = (
                            0.5 * jnp.square(chosen_q - targets_mb) * mask
                        ).sum() / mask_denom
                        return td_loss, {
                            "td_loss": td_loss,
                            "q_values": masked_mean(chosen_q, mask),
                            "mean_target": masked_mean(targets_mb, mask),
                        }

                    (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                        train_state.params
                    )
                    aux["grad_norm"] = pytree_norm(grads)
                    updated_train_state = train_state.apply_gradients(grads=grads)
                    return updated_train_state, aux

                final_train_state, batch_stats = lax.scan(
                    update_minibatch,
                    pqn_state.train_state,
                    minibatches,
                )
                pqn_state = PQNUpdateState(
                    train_state=final_train_state,
                    transitions=pqn_state.transitions,
                    targets=pqn_state.targets,
                    train_mask=pqn_state.train_mask,
                    rng=rng_s,
                )
                return pqn_state, batch_stats

            pqn_state = PQNUpdateState(
                train_state=br_train_state,
                transitions=transitions,
                targets=targets,
                train_mask=train_mask,
                rng=rng_pqn_update,
            )
            final_pqn_state, pqn_loss_info = lax.scan(
                update_epoch,
                pqn_state,
                None,
                config["num_epochs"],
            )
            br_train_state = final_pqn_state.train_state
            pqn_loss_info = jax.tree_util.tree_map(lambda x: x.mean(), pqn_loss_info)

            def update_sl_step(
                sl_state: SLUpdateState,
                unused: None,
            ) -> tuple[SLUpdateState, dict[str, FloatArray]]:
                rng_s, rng_batch = jax.random.split(sl_state.rng)
                has_data = sl_state.sl_buffer.size > 0

                def _train_step(train_state: TrainState):
                    batch = sample_sl_batch(
                        rng_batch,
                        sl_state.sl_buffer,
                        config["sl_batch_size"],
                    )

                    def sl_loss(params) -> tuple[FloatArray, dict[str, FloatArray]]:
                        logits = avg_network.apply(params, batch.obs)
                        logits_masked = jnp.where(batch.action_mask, logits, -jnp.inf)
                        log_probs = jax.nn.log_softmax(logits_masked, axis=-1)
                        action_log_probs = jnp.take_along_axis(
                            log_probs,
                            batch.action[:, None],
                            axis=-1,
                        ).squeeze(-1)
                        ce_loss = -action_log_probs.mean()
                        acc = (
                            jnp.argmax(logits_masked, axis=-1) == batch.action
                        ).mean()
                        return ce_loss, {
                            "sl_loss": ce_loss,
                            "sl_acc": acc,
                        }

                    grad_fn = jax.value_and_grad(sl_loss, has_aux=True)
                    (_, aux), grads = grad_fn(train_state.params)
                    aux["sl_grad_norm"] = pytree_norm(grads)
                    new_train_state = train_state.apply_gradients(grads=grads)
                    return new_train_state, aux

                def _skip_step(train_state: TrainState):
                    return train_state, {
                        "sl_loss": jnp.array(0.0, dtype=jnp.float32),
                        "sl_acc": jnp.array(0.0, dtype=jnp.float32),
                        "sl_grad_norm": jnp.array(0.0, dtype=jnp.float32),
                    }

                train_state, aux = lax.cond(
                    has_data,
                    _train_step,
                    _skip_step,
                    sl_state.train_state,
                )
                sl_state = SLUpdateState(
                    train_state=train_state,
                    sl_buffer=sl_state.sl_buffer,
                    rng=rng_s,
                )
                return sl_state, aux

            sl_state = SLUpdateState(
                train_state=avg_train_state,
                sl_buffer=sl_buffer,
                rng=rng_sl_update,
            )
            sl_state, sl_loss_info = lax.scan(
                update_sl_step,
                sl_state,
                None,
                config["sl_num_steps_per_update"],
            )
            avg_train_state = sl_state.train_state
            sl_loss_info = jax.tree_util.tree_map(lambda x: x.mean(), sl_loss_info)

            # Self-play metrics over the games that ended in this rollout, per
            # player, in env units.
            game_end = transitions.done[..., None]
            werewolf = transitions.roles == WEREWOLF
            ended_ww = game_end & werewolf
            ended_human = game_end & ~werewolf
            br_games = game_end & transitions.br_mode
            avg_games = game_end & ~transitions.br_mode
            games = jnp.maximum(transitions.done.sum(), 1)

            def per_game(x: FloatArray, mask: BoolArray) -> FloatArray:
                return (x * mask).sum() / jnp.maximum(mask.sum(), 1)

            metric = {
                "update_step": runner_state.update_step,
                "env_steps": (runner_state.update_step + 1)
                * config["num_envs"]
                * config["num_steps_per_env_per_update"],
                "games_completed": transitions.done.sum(),
                "ep_length_avg": jnp.where(
                    transitions.done, transitions.game_length, 0
                ).sum()
                / games,
                "win_rate_human_team_rollout": per_game(
                    transitions.team_won, ended_human
                ),
                "win_rate_werewolf_team_rollout": per_game(
                    transitions.team_won, ended_ww
                ),
                "returns_human_team_avg": per_game(transitions.reward, ended_human),
                "returns_werewolf_team_avg": per_game(transitions.reward, ended_ww),
                # Against the mixed opponents the BR should win more often.
                "br_return_per_game": per_game(transitions.reward, br_games),
                "avg_return_per_game": per_game(transitions.reward, avg_games),
                "br_action_frac_rollout": transitions.is_br.mean(),
                "q_sample_frac": train_mask.mean(),
                "sl_buffer_size": sl_buffer.size.astype(jnp.float32),
                "sl_buffer_seen": sl_buffer.seen.astype(jnp.float32),
                "epsilon": epsilon_schedule(
                    jnp.asarray(runner_state.update_step, dtype=jnp.float32)
                ),
                "br_lr": (
                    linear_decay(br_train_state.step)
                    if config["anneal_lr"]
                    else jnp.asarray(config["lr"])
                ),
            }
            metric.update(pqn_loss_info)
            metric.update(sl_loss_info)

            should_compare = compare_enabled & (
                (runner_state.update_step % config["compare_interval"] == 0)
                | (runner_state.update_step == config["num_update_steps"] - 1)
            )
            opponent_params = (
                avg_train_state.params if compare_against_random else baseline_params
            )

            def do_compare(rng_c):
                return run_compare_eval(
                    rng_c,
                    avg_train_state.params,
                    br_train_state.params,
                    opponent_params,
                )

            if compare_enabled:
                eval_metrics = lax.cond(
                    should_compare,
                    do_compare,
                    lambda rng_c: jax.tree_util.tree_map(
                        lambda x: jnp.zeros(x.shape, x.dtype),
                        jax.eval_shape(do_compare, rng_c),
                    ),
                    rng_compare,
                )
                metric.update(eval_metrics)

            jax.experimental.io_callback(
                logging_callback,
                None,
                seed,
                runner_state.update_step,
                should_compare,
                metric,
            )

            runner_state = RunnerState(
                br_train_state=br_train_state,
                avg_train_state=avg_train_state,
                sl_buffer=sl_buffer,
                state=runner_state.state,
                obs=runner_state.obs,
                done=runner_state.done,
                br_mode=runner_state.br_mode,
                update_step=runner_state.update_step + 1,
                rng=rng,
            )
            return runner_state, None

        initial_runner_state = RunnerState(
            br_train_state=br_train_state,
            avg_train_state=avg_train_state,
            sl_buffer=sl_buffer,
            state=state,
            obs=obs,
            done=jnp.zeros((config["num_envs"]), dtype=jnp.bool_),
            br_mode=br_mode,
            update_step=jnp.array(0, dtype=jnp.int32),
            rng=rng,
        )
        final_runner_state, _ = lax.scan(
            update_step,
            initial_runner_state,
            None,
            config["num_update_steps"],
        )
        return final_runner_state

    return train


def save_checkpoints(config: dict, final_runner_state: RunnerState) -> str:
    """Saves the networks of every seed after training.

    Layout of {save_dir}/{job_type}_{timestamp}/:
        avg_policy_{i}.msgpack  average policy of seed i (NFSP's output strategy)
        br_{i}.msgpack          PQN best-response Q-network of seed i
        config.yaml             resolved config of the run
    i = 0..num_seeds-1 indexes the vmapped seeds (wandb run "{seed}_{i}"). Each
    file holds the flax params of one network, without a seed axis, written with
    flax.serialization.to_bytes. The average policy is
    ActorDiscreteMLP(action_dim, fc_dim_size) mapping an env observation to
    action logits; the BR is QNetworkDiscreteMLP(action_dim, fc_dim_size)
    mapping it to Q-values in env reward units (its policy is the argmax over
    legal actions). Load one with load_params (or serialization.from_bytes(
    template, data) where template = network.init(rng, obs)), and mask illegal
    actions. As an evaluation opponent (compare_mode=checkpoint) use
    compare_network_type=actor for avg_policy_{i} and q_network for br_{i}.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(config["save_dir"], f"{config['job_type']}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    networks = {
        "avg_policy": final_runner_state.avg_train_state.params,
        "br": final_runner_state.br_train_state.params,
    }
    for name, params in networks.items():
        for i in range(config["num_seeds"]):
            params_i = jax.tree_util.tree_map(lambda x: np.asarray(x[i]), params)
            with open(os.path.join(run_dir, f"{name}_{i}.msgpack"), "wb") as f:
                f.write(serialization.to_bytes(params_i))
    OmegaConf.save(OmegaConf.create(config), os.path.join(run_dir, "config.yaml"))
    return run_dir


def print_evaluation_summary() -> None:
    """Last evaluation of every seed, and the mean over seeds."""
    if not EVALUATIONS:
        return
    finals = [EVALUATIONS[s][-1][2] for s in sorted(EVALUATIONS)]
    for key in finals[0]:
        values = np.array([f[key] for f in finals])
        print(
            f"final {key}: mean {values.mean():.3f} std {values.std():.3f} "
            f"over {len(values)} seed(s): {np.round(values, 3).tolist()}"
        )


@hydra.main(version_base=None, config_path="./", config_name="config_pqn_nfsp")
def main(config: dict) -> None:
    global LOGGER
    try:
        config = OmegaConf.to_container(config, resolve=True)
        config["num_update_steps"] = int(
            config["num_timesteps"]
            // config["num_envs"]
            // config["num_steps_per_env_per_update"]
        )
        config["num_gradient_steps"] = (
            config["num_update_steps"]
            * config["num_epochs"]
            * config["num_minibatches"]
        )
        env_steps = (
            config["num_update_steps"]
            * config["num_envs"]
            * config["num_steps_per_env_per_update"]
        )

        rng = jax.random.PRNGKey(config["seed"])
        rng_seeds = jax.random.split(rng, config["num_seeds"])
        exp_ids = jnp.arange(config["num_seeds"])

        print("Starting compile...")
        start = time.time()
        train_vjit = (
            jax.jit(jax.vmap(make_train(config))).lower(rng_seeds, exp_ids).compile()
        )
        print(f"Compile finished in {time.time() - start:.1f} s")

        job_type = f"{config['job_type']}_{config['env_name']}"
        group = f"{config['env_name']}" + datetime.datetime.now().strftime(
            "_%Y-%m-%d_%H-%M-%S"
        )
        LOGGER = WandbMultiLogger(
            project=config["project"],
            group=group,
            job_type=job_type,
            config=config,
            mode=(lambda: "online" if config["wandb"] else "disabled")(),
            seed=config["seed"],
            num_seeds=config["num_seeds"],
        )

        print("Running...")
        start = time.time()
        final_runner_state = jax.block_until_ready(train_vjit(rng_seeds, exp_ids))
        run_time = time.time() - start
        print(
            f"Trained {config['num_seeds']} seed(s) x {env_steps} env steps in "
            f"{run_time:.1f} s ({run_time / env_steps * 1e6:.2f} s per 1e6 env "
            f"steps, all seeds in parallel)"
        )
        print_evaluation_summary()
        if config["save_final"]:
            run_dir = save_checkpoints(config, final_runner_state)
            print(f"Saved checkpoints to {run_dir}")
    finally:
        if LOGGER is not None:
            LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    register_resolvers()
    main()
