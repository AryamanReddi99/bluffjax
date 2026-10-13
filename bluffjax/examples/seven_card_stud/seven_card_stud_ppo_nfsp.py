"""
PPO-NFSP for 7-Card Stud.

Neural Fictitious Self-Play (Heinrich & Silver 2016, arXiv:1603.01121) with
PPO as the best-response (BR) learner. All seats share one BR network and one
average-policy network (the observation is relative to the player to act).

- Policy mixing: at the start of every hand each player independently draws
  the policy it follows for the whole hand, the BR with probability
  anticipatory_eta and the average policy otherwise (see redraw_br_mode).
- BR learning: PPO is on-policy, so the BR actor and critic train only on
  decisions made while following the BR. Returns follow each player's own
  decisions: a decision's reward is everything the player receives until its
  next decision or the end of the hand (which may come on another player's
  step), and it bootstraps from the value of that next decision (see
  per_player_gae).
- Average policy: every BR decision (obs, action) is added to a reservoir
  buffer (Algorithm R) and the average policy is fit to it by cross-entropy.
  The average policy is NFSP's output strategy; the BR is a training device.
- Evaluation (compare): every compare_interval updates and after the last
  one, the average and BR policies each play compare_episodes complete hands
  against the opponent (uniform random legal actions, or a checkpoint) in
  every other seat, with the learner's seat rotating over the hands from a
  random seat (see play_eval_games). Logged per policy: win, loss and draw
  rates (net chips positive, negative, zero), the win rate of the decided
  hands (NaN if there were none), chips per hand and hand length, as
  {stat}_vs_{random|baseline} for the average policy and
  {stat}_br_vs_{random|baseline} for the BR.

Units: the env pays net chips per hand (antes 1/2, bring-in 1/2, bets 1 and
2); training and the logged returns use them as they are.

Game metrics (bluffjax.metrics.seven_card_stud): counted at every rollout
step and logged every update for the decisions of the players following the
average policy in their hand (game/, NFSP's output strategy) and of those
following the BR (game_br/). Their per-hand carry persists across updates,
so hands that span two rollouts are counted once and exactly.

Run directory and checkpoints: main() creates {save_dir}/{job_type}_{timestamp}/
with config.yaml and results.jsonl (everything logged, as sent to wandb), and
with save_checkpoints the networks after half and all of the updates (see
save_checkpoints); load_params restores one.
"""

import datetime
import time
from typing import Callable, NamedTuple

import distrax
from flax.training.train_state import TrainState
import hydra
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax
import wandb

from bluffjax.utils.typing import (
    Any,
    BoolArray,
    FloatArray,
    IntArray,
    PRNGKeyArray,
)
from bluffjax import make
from bluffjax.environments.seven_card_stud.seven_card_stud import SevenCardStudState
from bluffjax.examples.run_utils import (
    ResultsLog,
    half_update_steps,
    load_params,
    make_run_dir,
    save_seed_params,
)
from bluffjax.metrics import make_metrics, masked_totals, prefixed, rate
from bluffjax.networks.mlp import (
    ActorCriticDiscreteMLP,
    ActorDiscreteMLP,
    QNetworkDiscreteMLP,
)
from bluffjax.utils.jax_utils import pytree_norm
from bluffjax.utils.paths import REPO_ROOT, register_resolvers

WANDB_RUNS: list = []  # one wandb run per vmapped seed, created in main()
# results.jsonl of the run (and wandb), created in main()
RESULTS_LOG: ResultsLog | None = None
# EVALUATIONS[seed] = [(env_steps, {metric: value}), ...]
EVALUATIONS: dict[int, list[tuple[int, dict[str, float]]]] = {}


class Transition(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    br_log_prob: FloatArray  # log-prob of the action under the BR (BR decisions)
    reward: FloatArray  # (num_agents,) env reward for every player, chips
    done: BoolArray  # the hand ended at this step (the env auto-resets)
    timestep: IntArray  # steps into the hand after this step (hand length if done)
    value: FloatArray  # BR critic value of obs
    player_idx: IntArray
    is_br: BoolArray  # the acting player follows the BR in this hand
    br_mode: BoolArray  # (num_agents,) which players follow the BR in this hand
    game_counts: dict  # game-metric counts of this step, (num_agents, ...) each


class SLBufferState(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    seen: IntArray
    size: IntArray


class Checkpoint(NamedTuple):
    """The two NFSP networks at one point of training."""

    avg_params: Any  # average policy, NFSP's output strategy
    br_params: Any  # best response


class RunnerState(NamedTuple):
    br_train_state: TrainState
    avg_train_state: TrainState
    sl_buffer: SLBufferState
    state: SevenCardStudState
    obs: FloatArray
    br_mode: BoolArray  # (num_envs, num_agents) policy drawn for the current hand
    metric_carry: Any  # per-env game-metric carry of the current hand
    half: Checkpoint  # networks after half_update_steps updates
    update_step: IntArray
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


def redraw_br_mode(
    rng: PRNGKeyArray, br_mode: BoolArray, done: BoolArray, eta: float
) -> BoolArray:
    """NFSP policy mixing for the next step.

    br_mode (N, P) says which players follow the BR in the current hand. Where
    the hand just ended (done, the env auto-resets), every player draws anew:
    BR with probability eta, average policy otherwise. Within a hand it stays.
    """
    new_mode = jax.random.bernoulli(rng, p=eta, shape=br_mode.shape)
    return jnp.where(done[:, None], new_mode, br_mode)


def per_player_gae(
    values: FloatArray,
    rewards: FloatArray,
    dones: BoolArray,
    players: IntArray,
    last_value: FloatArray,
    last_player: IntArray,
    gamma: float,
    gae_lambda: float,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """GAE along each player's own decisions in a turn-based rollout.

    Shapes: T steps, N envs, P players.
        values: (T, N) critic value of obs_t for the player acting at step t.
        rewards: (T, N, P) reward vector returned by env.step at step t.
        dones: (T, N) the hand ended at step t (the env then auto-resets).
        players: (T, N) player acting at step t.
        last_value: (N,) critic value of the observation after the last step.
        last_player: (N,) player to act after the last step.

    The reward of player p's decision at step t is the sum of p's rewards from
    step t up to p's next decision t' (so rewards paid on other players' steps
    are included), and the decision bootstraps from values[t']. If the hand
    ends first the decision is terminal. At the rollout cut-off only the player
    to act has a next decision (bootstrap from last_value). The other players'
    trailing decisions in an unfinished hand have no known outcome yet: they
    are marked invalid, and those players' earlier decisions truncate their
    lambda trace at them (as GAE does at a cut-off).

    Returns:
        advantages (T, N), value targets (T, N), valid (T, N).
    """
    num_agents = rewards.shape[-1]
    last_onehot = jax.nn.one_hot(last_player, num_agents, dtype=jnp.bool_)
    zeros = jnp.zeros(last_onehot.shape, dtype=jnp.float32)
    # Per player, about its next decision (going backwards): value, advantage
    # (0 if the trace stops there), whether it is known, whether the hand ends
    # before it, and the reward collected since the decision being processed.
    init = (
        jnp.where(last_onehot, last_value[:, None], 0.0),
        zeros,
        last_onehot,
        jnp.zeros_like(last_onehot),
        zeros,
    )

    def body(carry, x):
        next_value, next_adv, known, terminal, reward_acc = carry
        value, reward, done, player = x
        # Steps after a hand-ending step belong to the next hand.
        hand_over = done[:, None]
        next_value = jnp.where(hand_over, 0.0, next_value)
        next_adv = jnp.where(hand_over, 0.0, next_adv)
        known = known | hand_over
        terminal = terminal | hand_over
        reward_acc = jnp.where(hand_over, 0.0, reward_acc) + reward

        not_terminal = 1.0 - _pick(terminal, player).astype(jnp.float32)
        delta = (
            _pick(reward_acc, player)
            + gamma * not_terminal * _pick(next_value, player)
            - value
        )
        adv = delta + gamma * gae_lambda * not_terminal * _pick(next_adv, player)
        valid = _pick(known, player)
        adv = jnp.where(valid, adv, 0.0)

        acting = jax.nn.one_hot(player, num_agents, dtype=jnp.bool_)
        next_value = jnp.where(acting, value[:, None], next_value)
        next_adv = jnp.where(acting, adv[:, None], next_adv)
        known = known | acting
        terminal = terminal & ~acting
        reward_acc = jnp.where(acting, 0.0, reward_acc)
        return (next_value, next_adv, known, terminal, reward_acc), (adv, valid)

    _, (advantages, valid) = lax.scan(
        body, init, (values, rewards, dones, players), reverse=True
    )
    return advantages, advantages + values, valid


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


# =============================================================================
# Evaluation against a baseline
# =============================================================================

# act(params, obs, action_mask, rng) -> action, for one observation
Policy = Callable[[Any, FloatArray, BoolArray, PRNGKeyArray], IntArray]


class EvalResult(NamedTuple):
    """Per hand, from the learner's point of view."""

    win: FloatArray  # the learner won chips (its return is positive)
    loss: FloatArray  # the learner lost chips
    draw: FloatArray  # neither (e.g. a split pot that returned its stake)
    ret: FloatArray  # the learner's net chips of the hand
    length: FloatArray  # env steps
    seat: IntArray  # the learner's seat


def play_eval_games(
    env: Any,
    learner: Policy,
    learner_params: Any,
    baseline: Policy,
    baseline_params: Any,
    num_games: int,
    rng: PRNGKeyArray,
) -> EvalResult:
    """Plays num_games complete hands in parallel, the learner in one seat and
    the baseline in every other seat.

    The learner's seat rotates over the hands, starting from a random seat
    (hand g: seat (offset + g) % num_agents), so every seat is used equally
    up to one hand (and the cards decide the bring-in, so no seat index is
    favoured). Every hand is played with env.step_env (no auto-reset) until
    it is over. The env pays the result of a hand when it ends, so the
    learner's return is its net chips: a win if positive, a loss if
    negative and a draw if zero.
    """
    num_agents = env.num_agents
    rng_seat, rng_games = jax.random.split(rng)
    offset = jax.random.randint(rng_seat, (), 0, num_agents)
    seats = (offset + jnp.arange(num_games)) % num_agents

    def one_hand(rng_hand: PRNGKeyArray, seat: IntArray):
        rng_reset, rng_play = jax.random.split(rng_hand)
        state, obs = env.reset(rng_reset)

        def body(carry):
            state, obs, _, ret, length, rng_s = carry
            rng_s, rng_learner, rng_baseline, rng_step = jax.random.split(rng_s, 4)
            action_mask = env.get_avail_actions(state)
            action = jnp.where(
                state.current_player_idx == seat,
                learner(learner_params, obs, action_mask, rng_learner),
                baseline(baseline_params, obs, action_mask, rng_baseline),
            )
            state, obs, reward, _, done, _ = env.step_env(rng_step, state, action)
            return state, obs, done, ret + reward[seat], length + 1, rng_s

        init = (state, obs, jnp.bool_(False), jnp.float32(0.0), jnp.int32(0), rng_play)
        _, _, _, ret, length, _ = lax.while_loop(lambda c: ~c[2], body, init)
        return ret, length

    ret, length = jax.vmap(one_hand)(jax.random.split(rng_games, num_games), seats)
    return EvalResult(
        win=(ret > 0).astype(jnp.float32),
        loss=(ret < 0).astype(jnp.float32),
        draw=(ret == 0).astype(jnp.float32),
        ret=ret,
        length=length.astype(jnp.float32),
        seat=seats,
    )


def eval_metrics(name: str, result: EvalResult) -> dict[str, FloatArray]:
    """Means over the evaluation hands, as {stat}_{name}; win_rate_decided
    excludes draws and is NaN if every hand was a draw (bluffjax.metrics.rate).
    name is vs_{opponent} for the average policy and br_vs_{opponent} for the
    BR."""
    decided = result.win.sum() + result.loss.sum()
    return {
        f"win_rate_{name}": result.win.mean(),
        f"loss_rate_{name}": result.loss.mean(),
        f"draw_rate_{name}": result.draw.mean(),
        f"win_rate_decided_{name}": rate(result.win.sum(), decided),
        f"avg_return_{name}": result.ret.mean(),
        f"avg_episode_length_{name}": result.length.mean(),
    }


def format_eval(evals: dict[str, float]) -> str:
    """One line for the policies evaluated, from eval_metrics' metrics."""
    names = sorted(
        k[len("win_rate_"):]
        for k in evals
        if k.startswith("win_rate_") and not k.startswith("win_rate_decided_")
    )
    return "; ".join(
        f"{name} W/L/D {evals[f'win_rate_{name}']:.3f}/"
        f"{evals[f'loss_rate_{name}']:.3f}/{evals[f'draw_rate_{name}']:.3f} "
        f"(win rate of decided hands {evals[f'win_rate_decided_{name}']:.3f}), "
        f"chips per hand {evals[f'avg_return_{name}']:+.3f}, "
        f"length {evals[f'avg_episode_length_{name}']:.1f}"
        for name in names
    )


def print_eval_summary() -> None:
    """The last evaluation of every seed and the mean over seeds."""
    finals = {seed: evals[-1][1] for seed, evals in sorted(EVALUATIONS.items())}
    for seed, evals in finals.items():
        print(f"final evaluation seed {seed}: {format_eval(evals)}")
    if finals:
        keys = next(iter(finals.values())).keys()
        mean = {k: float(np.mean([e[k] for e in finals.values()])) for k in keys}
        print(f"final evaluation, mean of {len(finals)} seed(s): {format_eval(mean)}")


def make_train(config: dict) -> Callable[..., RunnerState]:
    env = make("seven_card_stud", **config["env_kwargs"])
    num_agents = env.num_agents
    action_dim = env.action_space().n
    config["batch_shuffle_dim"] = (
        config["num_steps_per_env_per_update"] * config["num_envs"]
    )
    num_half = half_update_steps(config["num_update_steps"])
    game_metrics = make_metrics("seven_card_stud", env)
    # The average policy uses the same actor-critic module so that both
    # networks are evaluated the same way; only its actor is trained.
    network = ActorCriticDiscreteMLP(
        action_dim=action_dim, hidden_dim=config["fc_dim_size"]
    )
    template_obs = jnp.zeros((env.obs_dim,), dtype=jnp.float32)

    compare_enabled = config["compare"]
    compare_against_random = config["compare_mode"] == "random"
    compare_interval = config["compare_interval"]
    last_update = config["num_update_steps"] - 1
    opponent = "random" if compare_against_random else "baseline"

    def sample_actor(net: Any) -> Callable:
        def sample(params, obs, action_mask, rng) -> IntArray:
            logits = net.apply(params, obs)
            if isinstance(logits, tuple):  # actor-critic: (logits, value)
                logits = logits[0]
            return jax.random.categorical(
                rng, jnp.where(action_mask, logits, -jnp.inf)
            )

        return sample

    def sample_greedy(net: Any) -> Callable:
        def sample(params, obs, action_mask, rng) -> IntArray:
            # greedy, ties broken at random
            q_vals = jnp.where(action_mask, net.apply(params, obs), -jnp.inf)
            best = action_mask & (q_vals == q_vals.max())
            return jax.random.categorical(rng, jnp.where(best, 0.0, -jnp.inf))

        return sample

    def sample_random_legal_action(params, obs, action_mask, rng) -> IntArray:
        return jax.random.categorical(rng, jnp.where(action_mask, 0.0, -jnp.inf))

    sample_masked_action = sample_actor(network)
    baseline_params = None
    baseline_fn = sample_random_legal_action
    if compare_enabled and not compare_against_random:
        checkpoint_path = config["compare_with"]
        network_type = config["compare_network_type"]
        if network_type == "actor_critic":
            baseline_net = ActorCriticDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )
            baseline_fn = sample_actor(baseline_net)
        elif network_type == "actor":
            baseline_net = ActorDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )
            baseline_fn = sample_actor(baseline_net)
        elif network_type == "q_network":
            baseline_net = QNetworkDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )
            baseline_fn = sample_greedy(baseline_net)
        else:
            raise ValueError(
                "compare_network_type must be one of "
                "['actor_critic', 'q_network', 'actor'], "
                f"got '{network_type}'"
            )
        baseline_params = load_params(
            checkpoint_path, baseline_net.init(jax.random.PRNGKey(0), template_obs)
        )
        print(f"Loaded {network_type} baseline from {checkpoint_path}")

    def linear_decay(count: int) -> float:
        frac = 1.0 - count / config["num_gradient_steps"]
        return config["lr"] * frac

    def logging_callback(seed_val, metric_dict, evaluated) -> None:
        """Logs the metrics; the evaluation ones (keys with "_vs_") only when
        an evaluation ran."""
        seed_i = int(seed_val)
        metrics = {k: np.asarray(v) for k, v in metric_dict.items()}
        eval_keys = [k for k in metrics if "_vs_" in k]
        if bool(evaluated):
            env_steps = int(metrics["env_steps"])
            evals = {k: float(metrics[k]) for k in eval_keys}
            EVALUATIONS.setdefault(seed_i, []).append((env_steps, evals))
            print(
                f"seed {seed_i} update {int(metrics['update_step']) + 1}/"
                f"{last_update + 1} ({env_steps} env steps): {format_eval(evals)}",
                flush=True,
            )
        else:
            for k in eval_keys:
                del metrics[k]
        RESULTS_LOG.log(metrics, seed_i)

    def init_sl_buffer(obs_dim: int) -> SLBufferState:
        capacity = config["sl_reservoir_capacity"]
        return SLBufferState(
            # float16 halves the memory and is exact for these observations
            # (all features are 0/1).
            obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float16),
            action_mask=jnp.zeros((capacity, action_dim), dtype=jnp.bool_),
            action=jnp.zeros((capacity,), dtype=jnp.int32),
            seen=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
        )

    def sample_sl_batch(
        rng: PRNGKeyArray, buffer: SLBufferState, batch_size: int
    ) -> SLBatch:
        max_size = jnp.maximum(buffer.size, 1)
        indices = jax.random.randint(rng, (batch_size,), 0, max_size, dtype=jnp.int32)
        return SLBatch(
            obs=buffer.obs[indices].astype(jnp.float32),
            action_mask=buffer.action_mask[indices],
            action=buffer.action[indices],
        )

    def masked_mean(x: FloatArray, mask: FloatArray) -> FloatArray:
        denom = jnp.maximum(mask.sum(), 1.0)
        return (x * mask).sum() / denom

    def run_compare_eval(
        rng: PRNGKeyArray, avg_params: Any, br_params: Any
    ) -> dict[str, FloatArray]:
        """Evaluation metrics of the average policy ({stat}_vs_{opponent})
        and the BR ({stat}_br_vs_{opponent})."""
        rng_avg, rng_br = jax.random.split(rng)
        metrics = {}
        for prefix, act, params, rng_eval in (
            ("", sample_masked_action, avg_params, rng_avg),
            ("br_", sample_masked_action, br_params, rng_br),
        ):
            result = play_eval_games(
                env,
                act,
                params,
                baseline_fn,
                baseline_params,
                config["compare_episodes"],
                rng_eval,
            )
            metrics.update(eval_metrics(f"{prefix}vs_{opponent}", result))
        return metrics

    def train(
        rng: PRNGKeyArray, seed: int, num_updates: int | None = None
    ) -> RunnerState:
        """Runs num_updates updates (default: all of them) of one seed. The
        schedules always span config["num_update_steps"] updates."""
        rng, rng_reset, rng_br_init, rng_avg_init, rng_mode = jax.random.split(rng, 5)
        rng_resets = jax.random.split(rng_reset, config["num_envs"])
        state, obs = jax.vmap(env.reset, in_axes=(0))(rng_resets)

        br_params = network.init(rng_br_init, obs)
        avg_params = network.init(rng_avg_init, obs)
        br_lr = linear_decay if config["anneal_lr"] else config["lr"]
        br_tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=br_lr, eps=1e-5),
        )
        avg_tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=config["sl_lr"], eps=1e-5),
        )
        br_train_state = TrainState.create(
            apply_fn=network.apply, params=br_params, tx=br_tx
        )
        avg_train_state = TrainState.create(
            apply_fn=network.apply, params=avg_params, tx=avg_tx
        )
        sl_buffer = init_sl_buffer(obs.shape[-1])
        br_mode = jax.random.bernoulli(
            rng_mode,
            p=config["anticipatory_eta"],
            shape=(config["num_envs"], num_agents),
        )

        def update_step(
            runner_state: RunnerState, update: IntArray
        ) -> tuple[RunnerState, None]:
            def step(
                runner_state: RunnerState, unused: None
            ) -> tuple[RunnerState, Transition]:
                br_train_state = runner_state.br_train_state
                avg_train_state = runner_state.avg_train_state
                state = runner_state.state
                obs = runner_state.obs
                br_mode = runner_state.br_mode
                rng, rng_mode, rng_action, rng_step = jax.random.split(
                    runner_state.rng, 4
                )

                br_logits, value = network.apply(br_train_state.params, obs)
                avg_logits, _ = network.apply(avg_train_state.params, obs)
                action_mask = jax.vmap(env.get_avail_actions, in_axes=(0))(state)
                br_logits_masked = jnp.where(action_mask, br_logits, -jnp.inf)
                avg_logits_masked = jnp.where(action_mask, avg_logits, -jnp.inf)

                # NFSP: the acting player follows the policy it drew for this hand.
                player_idx = state.current_player_idx
                is_br = _pick(br_mode, player_idx)
                acting_logits = jnp.where(
                    is_br[:, None], br_logits_masked, avg_logits_masked
                )
                action = distrax.Categorical(logits=acting_logits).sample(
                    seed=rng_action
                )
                br_pi = distrax.Categorical(logits=br_logits_masked)
                br_log_prob = jnp.where(is_br, br_pi.log_prob(action), 0.0)

                rng_steps = jax.random.split(rng_step, config["num_envs"])
                next_state, next_obs, reward, _, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_steps, state, action)
                metric_carry, game_counts = jax.vmap(game_metrics.step)(
                    runner_state.metric_carry,
                    state,
                    action,
                    next_state,
                    reward,
                    done,
                    info,
                )

                transition = Transition(
                    obs=obs,
                    action_mask=action_mask,
                    action=action,
                    br_log_prob=br_log_prob,
                    reward=reward,
                    done=done,
                    timestep=info["timestep"],
                    value=value,
                    player_idx=player_idx,
                    is_br=is_br,
                    br_mode=br_mode,
                    game_counts=game_counts,
                )
                runner_state = runner_state._replace(
                    state=next_state,
                    obs=next_obs,
                    br_mode=redraw_br_mode(
                        rng_mode, br_mode, done, config["anticipatory_eta"]
                    ),
                    metric_carry=metric_carry,
                    rng=rng,
                )
                return runner_state, transition

            runner_state, transitions = lax.scan(
                step, runner_state, None, config["num_steps_per_env_per_update"]
            )

            br_train_state = runner_state.br_train_state
            avg_train_state = runner_state.avg_train_state
            sl_buffer = runner_state.sl_buffer
            last_state = runner_state.state
            rng, rng_sl_append, rng_ppo_update, rng_sl_update, rng_compare = (
                jax.random.split(runner_state.rng, 5)
            )

            _, last_val = network.apply(br_train_state.params, runner_state.obs)

            # Per-player GAE. PPO is on-policy: the BR actor and critic train
            # on BR decisions whose outcome is known.
            advantages, targets, valid = per_player_gae(
                transitions.value,
                transitions.reward,
                transitions.done,
                transitions.player_idx,
                last_val,
                last_state.current_player_idx,
                config["gamma"],
                config["gae_lambda"],
            )
            train_mask = valid & transitions.is_br

            # NFSP: every BR decision goes to the reservoir.
            sl_buffer = reservoir_append(
                sl_buffer,
                transitions.obs.reshape(-1, transitions.obs.shape[-1]),
                transitions.action_mask.reshape(-1, action_dim),
                transitions.action.reshape(-1),
                transitions.is_br.reshape(-1),
                rng_sl_append,
            )

            def update_ppo_epoch(carry, unused):
                train_state, rng = carry
                rng, rng_permute = jax.random.split(rng)
                batch = (
                    transitions.obs,
                    transitions.action_mask,
                    transitions.action,
                    transitions.br_log_prob,
                    transitions.value,
                    advantages,
                    targets,
                    train_mask,
                )
                # (steps, envs, ...) -> (steps * envs, ...)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((-1,) + x.shape[2:]), batch
                )
                permutation = jax.random.permutation(
                    rng_permute, config["batch_shuffle_dim"]
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0).reshape(
                        config["num_minibatches"], -1, *x.shape[1:]
                    ),
                    batch,
                )

                def update_minibatch(train_state: TrainState, minibatch):
                    (
                        obs_mb,
                        action_mask_mb,
                        action_mb,
                        br_log_prob_mb,
                        value_old_mb,
                        advantages_mb,
                        targets_mb,
                        train_mask_mb,
                    ) = minibatch

                    def loss(params) -> tuple[FloatArray, dict[str, FloatArray]]:
                        logits, value = network.apply(params, obs_mb)
                        logits_masked = jnp.where(action_mask_mb, logits, -jnp.inf)
                        pi = distrax.Categorical(logits=logits_masked)
                        log_prob = pi.log_prob(action_mb)

                        # Only BR decisions with a known outcome enter the loss.
                        mask = train_mask_mb.astype(jnp.float32)
                        mask_denom = jnp.maximum(mask.sum(), 1.0)

                        mean_adv = masked_mean(advantages_mb, mask)
                        std_adv = jnp.sqrt(
                            masked_mean(jnp.square(advantages_mb - mean_adv), mask)
                        )
                        gae_normalized = (advantages_mb - mean_adv) / (std_adv + 1e-8)

                        logratio = jnp.where(
                            train_mask_mb, log_prob - br_log_prob_mb, 0.0
                        )
                        ratio = jnp.exp(logratio)
                        loss_actor_raw = ratio * gae_normalized
                        loss_actor_clipped = (
                            jnp.clip(
                                ratio,
                                1.0 - config["clip_eps"],
                                1.0 + config["clip_eps"],
                            )
                            * gae_normalized
                        )
                        actor_per_item = -jnp.minimum(
                            loss_actor_raw, loss_actor_clipped
                        )
                        loss_actor = (actor_per_item * mask).sum() / mask_denom
                        entropy = (pi.entropy() * mask).sum() / mask_denom

                        value_clipped = value_old_mb + jnp.clip(
                            value - value_old_mb,
                            -config["vf_clip"],
                            config["vf_clip"],
                        )
                        value_err = jnp.maximum(
                            jnp.square(value - targets_mb),
                            jnp.square(value_clipped - targets_mb),
                        )
                        value_loss = 0.5 * (value_err * mask).sum() / mask_denom

                        total_loss = (
                            loss_actor
                            + config["vf_coef"] * value_loss
                            - config["ent_coef"] * entropy
                        )

                        kl_backward = masked_mean((ratio - 1) - logratio, mask)
                        kl_forward = masked_mean(ratio * logratio - (ratio - 1), mask)
                        clip_frac = masked_mean(
                            (jnp.abs(ratio - 1) > config["clip_eps"]).astype(
                                jnp.float32
                            ),
                            mask,
                        )
                        return total_loss, {
                            "total_loss": total_loss,
                            "value_loss": value_loss,
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "ratio_mean": masked_mean(ratio, mask),
                            "ratio_min": jnp.where(train_mask_mb, ratio, jnp.inf).min(),
                            "ratio_max": jnp.where(
                                train_mask_mb, ratio, -jnp.inf
                            ).max(),
                            "gae_mean": mean_adv,
                            "gae_std": std_adv,
                            "mean_target": masked_mean(targets_mb, mask),
                            "value_pred_mean": masked_mean(value, mask),
                            "kl_backward": kl_backward,
                            "kl_forward": kl_forward,
                            "clip_frac": clip_frac,
                            "ppo_sample_frac_minibatch": mask.mean(),
                        }

                    (_, aux), grads = jax.value_and_grad(loss, has_aux=True)(
                        train_state.params
                    )
                    aux["grad_norm"] = pytree_norm(grads)
                    return train_state.apply_gradients(grads=grads), aux

                train_state, batch_stats = lax.scan(
                    update_minibatch, train_state, minibatches
                )
                return (train_state, rng), batch_stats

            (br_train_state, _), ppo_loss_info = lax.scan(
                update_ppo_epoch,
                (br_train_state, rng_ppo_update),
                None,
                config["num_epochs"],
            )
            ppo_loss_info = jax.tree_util.tree_map(lambda x: x.mean(), ppo_loss_info)

            def update_sl_step(
                sl_state: SLUpdateState, unused: None
            ) -> tuple[SLUpdateState, dict[str, FloatArray]]:
                rng, rng_batch = jax.random.split(sl_state.rng)
                has_data = sl_state.sl_buffer.size > 0

                def _train_step(train_state: TrainState):
                    batch = sample_sl_batch(
                        rng_batch, sl_state.sl_buffer, config["sl_batch_size"]
                    )

                    def sl_loss(params) -> tuple[FloatArray, dict[str, FloatArray]]:
                        logits, _ = network.apply(params, batch.obs)
                        logits_masked = jnp.where(batch.action_mask, logits, -jnp.inf)
                        log_probs = jax.nn.log_softmax(logits_masked, axis=-1)
                        action_log_probs = jnp.take_along_axis(
                            log_probs, batch.action[:, None], axis=-1
                        ).squeeze(-1)
                        ce_loss = -action_log_probs.mean()
                        acc = (
                            jnp.argmax(logits_masked, axis=-1) == batch.action
                        ).mean()
                        return ce_loss, {"sl_loss": ce_loss, "sl_acc": acc}

                    (_, aux), grads = jax.value_and_grad(sl_loss, has_aux=True)(
                        train_state.params
                    )
                    aux["sl_grad_norm"] = pytree_norm(grads)
                    return train_state.apply_gradients(grads=grads), aux

                def _skip_step(train_state: TrainState):
                    return train_state, {
                        "sl_loss": jnp.array(0.0, dtype=jnp.float32),
                        "sl_acc": jnp.array(0.0, dtype=jnp.float32),
                        "sl_grad_norm": jnp.array(0.0, dtype=jnp.float32),
                    }

                train_state, aux = lax.cond(
                    has_data, _train_step, _skip_step, sl_state.train_state
                )
                sl_state = SLUpdateState(
                    train_state=train_state, sl_buffer=sl_state.sl_buffer, rng=rng
                )
                return sl_state, aux

            sl_state = SLUpdateState(
                train_state=avg_train_state, sl_buffer=sl_buffer, rng=rng_sl_update
            )
            sl_state, sl_loss_info = lax.scan(
                update_sl_step, sl_state, None, config["sl_num_steps_per_update"]
            )
            avg_train_state = sl_state.train_state
            sl_loss_info = jax.tree_util.tree_map(lambda x: x.mean(), sl_loss_info)

            # Self-play payoffs per finished hand in chips, split by the
            # policy the player followed in that hand. Against the mixed
            # opponents the BR should win (the average policy then loses).
            hand_end = transitions.done[..., None]
            br_hands = hand_end & transitions.br_mode
            avg_hands = hand_end & ~transitions.br_mode
            num_hands = jnp.maximum(transitions.done.sum(), 1)
            metric = {
                "update_step": runner_state.update_step,
                "env_steps": (runner_state.update_step + 1)
                * config["num_envs"]
                * config["num_steps_per_env_per_update"],
                "episodes_completed": transitions.done.sum(),
                "ep_length_avg": (transitions.timestep * transitions.done).sum()
                / num_hands,
                "return_per_episode": (transitions.reward * avg_hands).sum()
                / jnp.maximum(avg_hands.sum(), 1),
                "return_per_episode_br": (transitions.reward * br_hands).sum()
                / jnp.maximum(br_hands.sum(), 1),
                "br_action_frac_rollout": transitions.is_br.mean(),
                "ppo_sample_frac": train_mask.mean(),
                "sl_buffer_size": sl_buffer.size.astype(jnp.float32),
                "sl_buffer_seen": sl_buffer.seen.astype(jnp.float32),
                "br_lr": (
                    linear_decay(br_train_state.step)
                    if config["anneal_lr"]
                    else jnp.asarray(config["lr"])
                ),
            }
            metric.update(ppo_loss_info)
            metric.update(sl_loss_info)
            # Game metrics of this rollout's decisions, by the policy the
            # credited player follows in that hand: average policy (game/,
            # NFSP's output strategy) and BR (game_br/).
            metric.update(
                prefixed(
                    "game",
                    game_metrics.summarize(
                        masked_totals(transitions.game_counts, ~transitions.br_mode)
                    ),
                )
            )
            metric.update(
                prefixed(
                    "game_br",
                    game_metrics.summarize(
                        masked_totals(transitions.game_counts, transitions.br_mode)
                    ),
                )
            )

            # update is the scan's update index, the same for every seed (never
            # batched under the vmap over seeds), so the cond really skips the
            # evaluation in the other updates.
            evaluated = compare_enabled & (
                (update % compare_interval == 0) | (update == last_update)
            )
            if compare_enabled:
                eval_shapes = jax.eval_shape(
                    run_compare_eval,
                    rng_compare,
                    avg_train_state.params,
                    br_train_state.params,
                )
                metric.update(
                    lax.cond(
                        evaluated,
                        lambda: run_compare_eval(
                            rng_compare, avg_train_state.params, br_train_state.params
                        ),
                        lambda: jax.tree_util.tree_map(
                            lambda s: jnp.zeros(s.shape, s.dtype), eval_shapes
                        ),
                    )
                )
            jax.experimental.io_callback(
                logging_callback, None, seed, metric, evaluated
            )

            # Half-budget checkpoint: the networks after num_half updates.
            take_half = update + 1 == num_half
            half = jax.tree_util.tree_map(
                lambda new, old: jnp.where(take_half, new, old),
                Checkpoint(avg_train_state.params, br_train_state.params),
                runner_state.half,
            )
            runner_state = runner_state._replace(
                br_train_state=br_train_state,
                avg_train_state=avg_train_state,
                sl_buffer=sl_buffer,
                half=half,
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
            br_mode=br_mode,
            metric_carry=jax.vmap(lambda _: game_metrics.init_carry())(
                jnp.arange(config["num_envs"])
            ),
            half=Checkpoint(avg_params, br_params),  # replaced after num_half updates
            update_step=jnp.array(0, dtype=jnp.int32),
            rng=rng,
        )
        final_runner_state, _ = lax.scan(
            update_step,
            initial_runner_state,
            jnp.arange(
                config["num_update_steps"] if num_updates is None else num_updates
            ),
        )
        return final_runner_state

    return train


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
    avg_policy is the average policy (NFSP's output strategy), br the PPO best
    response. Both checkpoints come from the same run: the half one is the
    state after exactly that many updates, with the learning-rate schedule
    running over the whole run (config.yaml records num_update_steps and
    half_update_steps). i = 0..num_seeds-1 indexes the vmapped seeds (wandb
    run "{seed}_{i}"). Each file holds the flax params of one network, without
    a seed axis, written with flax.serialization.to_bytes; restore one with
    load_params(path, template), where template = network.init(rng, obs).
    Both networks are ActorCriticDiscreteMLP(action_dim=4,
    hidden_dim=fc_dim_size), mapping the acting player's observation
    (env.obs_from_state(state), env.obs_dim floats, which depends on
    env_kwargs.num_agents) to (action logits, value in chips); the value
    head of the average policy is unused. Mask illegal actions
    (env.get_avail_actions(state)) before the softmax. Load either one as an
    evaluation baseline with compare_mode=checkpoint and
    compare_network_type=actor_critic.
    """
    for stage, checkpoint in (("half", half), ("full", full)):
        for name, params in (
            ("avg_policy", checkpoint.avg_params),
            ("br", checkpoint.br_params),
        ):
            save_seed_params(run_dir, stage, name, params, range(num_seeds))
    return run_dir


@hydra.main(version_base=None, config_path="./", config_name="config_ppo_nfsp")
def main(config: dict) -> None:
    global WANDB_RUNS, RESULTS_LOG
    try:
        config = OmegaConf.to_container(config, resolve=True)
        config["num_update_steps"] = int(
            config["num_timesteps"]
            // config["num_envs"]
            // config["num_steps_per_env_per_update"]
        )
        config["half_update_steps"] = half_update_steps(config["num_update_steps"])
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
        train = make_train(config)  # completes config (batch_shuffle_dim)

        # The run directory and the results log come first, so that
        # results.jsonl fills while the run trains (wall_time counts from
        # here, compilation included).
        run_dir = make_run_dir(config)
        job_type = f"{config['job_type']}_{config['env_name']}"
        group = f"{config['env_name']}" + datetime.datetime.now().strftime(
            "_%Y-%m-%d_%H-%M-%S"
        )
        WANDB_RUNS = [
            wandb.init(
                project=config["project"],
                group=group,
                job_type=job_type,
                name=f"{config['seed']}_{i}",
                config=config,
                mode=None if config["wandb"] else "disabled",
                dir=str(REPO_ROOT),
                reinit="create_new",
            )
            for i in range(config["num_seeds"])
        ]
        RESULTS_LOG = ResultsLog(run_dir, WANDB_RUNS)
        print(f"Run directory: {run_dir}")

        rng = jax.random.PRNGKey(config["seed"])
        rng_seeds = jax.random.split(rng, config["num_seeds"])
        exp_ids = jnp.arange(config["num_seeds"])

        print("Starting compile...")
        start = time.time()
        train_vjit = jax.jit(jax.vmap(train)).lower(rng_seeds, exp_ids).compile()
        print(f"Compile finished in {time.time() - start:.1f} s")

        print("Running...")
        start = time.time()
        final_runner_state = jax.block_until_ready(train_vjit(rng_seeds, exp_ids))
        jax.effects_barrier()  # every update's record is in results.jsonl
        run_time = time.time() - start
        print(
            f"Trained {config['num_seeds']} seed(s) x {env_steps} env steps in "
            f"{run_time:.1f} s (all seeds in parallel)"
        )
        print_eval_summary()
        if config["save_checkpoints"]:
            save_checkpoints(
                run_dir,
                config["num_seeds"],
                final_runner_state.half,
                Checkpoint(
                    final_runner_state.avg_train_state.params,
                    final_runner_state.br_train_state.params,
                ),
            )
            print(f"Saved checkpoints to {run_dir}")
    finally:
        for run in WANDB_RUNS:
            run.finish()
        print("Finished.")


if __name__ == "__main__":
    register_resolvers()
    main()
