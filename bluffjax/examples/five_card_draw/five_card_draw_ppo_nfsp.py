"""
PPO-NFSP for 5-Card Draw (2-10 players).

Neural Fictitious Self-Play (Heinrich & Silver 2016, arXiv:1603.01121) with
PPO as the best-response (BR) learner. All seats share one BR network and one
average-policy network.

- Policy mixing: at the start of every hand each player independently draws
  the policy it follows for the whole hand, the BR with probability
  anticipatory_eta and the average policy otherwise.
- BR learning: PPO is on-policy, so the BR actor and critic train only on
  decisions made while following the BR. Returns follow each player's own
  decisions: a decision's reward is everything the player receives until its
  next decision or the end of the hand (which may come on another player's
  step), and it bootstraps from the value of that next decision (see
  per_player_gae).
- Average policy: every BR decision (obs, action) is added to a reservoir
  buffer (Algorithm R) and the average policy is fit to it by cross-entropy.
  The average policy is NFSP's output strategy; the BR is a training device.
- Evaluation (compare): the average and BR policies each play compare_episodes
  hands against the baseline (uniform random legal actions, or a checkpoint)
  in every other seat. The learner's seat rotates over the hands (hand h: seat
  h % num_agents), so every seat is used equally. Runs every compare_interval
  updates and after the last one; chips per hand, env units.

Units: the env pays raw chips per hand (stacks init_chips, blinds 1/2), and
training and logging use them as they are.

Checkpoints (save_final): see save_checkpoints. Baselines are loaded with
load_params, which requires the exact parameter tree of compare_network_type.
"""

import datetime
import os
import time
from typing import Callable, NamedTuple

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
    Any,
    BoolArray,
    FloatArray,
    IntArray,
    PRNGKeyArray,
)
from bluffjax import make
from bluffjax.environments.five_card_draw.five_card_draw import FiveCardDrawState
from bluffjax.networks.mlp import (
    ActorCriticDiscreteMLP,
    ActorDiscreteMLP,
    QNetworkDiscreteMLP,
)
from bluffjax.utils.jax_utils import pytree_norm
from bluffjax.utils.paths import register_resolvers
from bluffjax.utils.wandb_multilogger import WandbMultiLogger

LOGGER = None
# EVALUATIONS[seed] = [(env_steps, {metric name: chips per hand}), ...]
EVALUATIONS: dict[int, list[tuple[int, dict[str, float]]]] = {}


class Transition(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    br_log_prob: FloatArray  # log-prob of the action under the BR (BR decisions)
    reward: FloatArray  # (num_agents,) env reward for every player, chips
    done: BoolArray  # the hand ended at this step (the env auto-resets)
    value: FloatArray  # BR critic value of obs
    player_idx: IntArray
    is_br: BoolArray  # the acting player follows the BR in this hand
    br_mode: BoolArray  # (num_agents,) which players follow the BR in this hand
    hand_length: IntArray  # steps in the hand, at the step that ends it


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
    state: FiveCardDrawState
    obs: FloatArray
    done: BoolArray
    br_mode: BoolArray  # (num_envs, num_agents) policy drawn for the current hand
    update_step: IntArray
    rng: PRNGKeyArray


class PPOUpdateState(NamedTuple):
    train_state: TrainState
    transitions: Transition
    advantages: FloatArray
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
    to act has a next decision (bootstrap from last_value). Another player's
    trailing decision in an unfinished hand has no known outcome yet: it is
    marked invalid, and that player's earlier decisions truncate their lambda
    trace at it (as GAE does at a cut-off).

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


def load_params(path: str, template: Any, network_name: str) -> Any:
    """Restores flax params written with serialization.to_bytes. The file must
    hold exactly the parameter tree of `template` (same keys and shapes):
    flax's from_bytes alone ignores extra keys, so e.g. a Q-network file
    would load as an actor."""
    with open(path, "rb") as f:
        restored = serialization.msgpack_restore(f.read())

    def shapes(tree):
        leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
        return {jax.tree_util.keystr(k): np.shape(v) for k, v in leaves}

    want = shapes(serialization.to_state_dict(template))
    got = shapes(restored)
    if want != got:
        wrong = [k for k in want.keys() & got.keys() if want[k] != got[k]]
        raise ValueError(
            f"{path} does not hold {network_name} parameters: "
            f"missing {sorted(want.keys() - got.keys())}, "
            f"unexpected {sorted(got.keys() - want.keys())}, "
            f"wrong shapes {[(k, got[k], want[k]) for k in sorted(wrong)]}"
        )
    return serialization.from_state_dict(template, restored)


def make_train(config: dict) -> Callable[[PRNGKeyArray, int], RunnerState]:
    env = make("five_card_draw", **config["env_kwargs"])
    action_dim = env.num_actions
    num_agents = env.num_agents
    config["batch_shuffle_dim"] = (
        config["num_steps_per_env_per_update"] * config["num_envs"]
    )
    last_update = config["num_update_steps"] - 1

    # The average policy uses the same actor-critic module so that both
    # networks are evaluated the same way; only its actor is trained.
    network = ActorCriticDiscreteMLP(
        action_dim=action_dim, hidden_dim=config["fc_dim_size"]
    )

    compare_enabled = config["compare"]
    compare_against_random = config["compare_mode"] == "random"
    compare_network_type = config["compare_network_type"]
    if config["compare_mode"] not in ("random", "checkpoint"):
        raise ValueError(
            f"compare_mode must be 'random' or 'checkpoint', got {config['compare_mode']!r}"
        )

    baseline_params = None
    baseline_network = None
    if compare_enabled and not compare_against_random:
        checkpoint_path = config["compare_with"]
        if not checkpoint_path:
            raise ValueError("compare_mode=checkpoint needs compare_with (a .msgpack file)")
        baseline_networks = {
            "actor_critic": ActorCriticDiscreteMLP,
            "q_network": QNetworkDiscreteMLP,
            "actor": ActorDiscreteMLP,
        }
        if compare_network_type not in baseline_networks:
            raise ValueError(
                f"compare_network_type must be one of {list(baseline_networks)}, "
                f"got {compare_network_type!r}"
            )
        baseline_network = baseline_networks[compare_network_type](
            action_dim=action_dim, hidden_dim=config["fc_dim_size"]
        )
        template = baseline_network.init(
            jax.random.PRNGKey(0), jnp.zeros((env.obs_dim,), dtype=jnp.float32)
        )
        baseline_params = load_params(checkpoint_path, template, compare_network_type)
        print(f"Loaded {compare_network_type} baseline from {checkpoint_path}")

    def linear_decay(count: int) -> float:
        frac = 1.0 - count / config["num_gradient_steps"]
        return config["lr"] * frac

    eval_names = (
        ("avg_return_avg_vs_random", "avg_return_br_vs_random")
        if compare_against_random
        else ("avg_return_avg_vs_baseline", "avg_return_br_vs_baseline")
    )

    def logging_callback(seed_val, metric_dict, evaluated) -> None:
        """Logs the metrics; evaluation metrics only when an evaluation ran."""
        seed_i = int(seed_val)
        metrics = {k: np.asarray(v) for k, v in metric_dict.items()}
        if bool(evaluated):
            update = int(metrics["update_step"])
            env_steps = int(metrics["env_steps"])
            result = {name: float(metrics[name]) for name in eval_names}
            EVALUATIONS.setdefault(seed_i, []).append((env_steps, result))
            print(
                f"seed {seed_i} update {update + 1}/{last_update + 1} "
                f"({env_steps} env steps): chips per hand vs "
                f"{'random' if compare_against_random else 'baseline'}: "
                f"average policy {result[eval_names[0]]:+.3f}, "
                f"BR {result[eval_names[1]]:+.3f}",
                flush=True,
            )
        else:
            for name in eval_names + ("avg_eval_episode_length",):
                metrics.pop(name, None)
        LOGGER.log(seed_i, metrics)

    def init_sl_buffer(obs_dim: int) -> SLBufferState:
        capacity = config["sl_reservoir_capacity"]
        return SLBufferState(
            obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
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
            obs=buffer.obs[indices],
            action_mask=buffer.action_mask[indices],
            action=buffer.action[indices],
        )

    def masked_mean(x: FloatArray, mask: FloatArray) -> FloatArray:
        denom = jnp.maximum(mask.sum(), 1.0)
        return (x * mask).sum() / denom

    # Evaluation policies: (params, obs, action_mask, rng) -> action
    def act_actor_critic(net, params, obs, action_mask, rng):
        logits, _ = net.apply(params, obs)
        return jax.random.categorical(rng, jnp.where(action_mask, logits, -jnp.inf))

    def act_actor(net, params, obs, action_mask, rng):
        logits = net.apply(params, obs)
        return jax.random.categorical(rng, jnp.where(action_mask, logits, -jnp.inf))

    def act_q_greedy(net, params, obs, action_mask, rng):
        q_vals = jnp.where(action_mask, net.apply(params, obs), -jnp.inf)
        ties = q_vals == jnp.max(q_vals)
        return jax.random.categorical(rng, jnp.where(ties, 0.0, -jnp.inf))

    def act_random(params, obs, action_mask, rng):
        return jax.random.categorical(rng, jnp.where(action_mask, 0.0, -jnp.inf))

    learner_act = lambda params, obs, mask, rng: act_actor_critic(  # noqa: E731
        network, params, obs, mask, rng
    )
    if compare_against_random:
        baseline_act = act_random
    else:
        baseline_act = {
            "actor_critic": act_actor_critic,
            "q_network": act_q_greedy,
            "actor": act_actor,
        }[compare_network_type]
        baseline_act = lambda params, obs, mask, rng, f=baseline_act: f(  # noqa: E731
            baseline_network, params, obs, mask, rng
        )

    def play_eval_hands(
        rng: PRNGKeyArray, learner_params: Any, num_hands: int
    ) -> tuple[FloatArray, FloatArray]:
        """num_hands hands in parallel; in hand h the learner sits in seat
        h % num_agents and the baseline in every other seat. Returns the
        learner's chips and the length of every hand."""

        def one_hand(rng_hand, seat):
            rng_hand, rng_reset = jax.random.split(rng_hand)
            state, obs = env.reset(rng_reset)

            def body(carry):
                state, obs, rng_s, _, _, length = carry
                rng_s, rng_learner, rng_base, rng_step = jax.random.split(rng_s, 4)
                mask = env.get_avail_actions(state)
                action = jnp.where(
                    state.current_player_idx == seat,
                    learner_act(learner_params, obs, mask, rng_learner),
                    baseline_act(baseline_params, obs, mask, rng_base),
                )
                state, obs, reward, _, done, _ = env.step_env(rng_step, state, action)
                return state, obs, rng_s, done, reward, length + 1

            init = (
                state,
                obs,
                rng_hand,
                jnp.bool_(False),
                jnp.zeros(num_agents),
                jnp.int32(0),
            )
            _, _, _, _, reward, length = lax.while_loop(lambda c: ~c[3], body, init)
            return reward[seat], length

        seats = jnp.arange(num_hands) % num_agents
        return jax.vmap(one_hand)(jax.random.split(rng, num_hands), seats)

    def run_compare_eval(
        rng: PRNGKeyArray, avg_params: Any, br_params: Any
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Chips per hand of the average and best-response policies."""
        rng_avg, rng_br = jax.random.split(rng)
        avg_chips, avg_len = play_eval_hands(
            rng_avg, avg_params, config["compare_episodes"]
        )
        br_chips, br_len = play_eval_hands(rng_br, br_params, config["compare_episodes"])
        eval_len = (avg_len.mean() + br_len.mean()) / 2
        return avg_chips.mean(), br_chips.mean(), eval_len

    def train(rng: PRNGKeyArray, seed: int) -> RunnerState:
        rng, rng_reset, rng_br_init, rng_avg_init, rng_mode = jax.random.split(rng, 5)
        state, obs = jax.vmap(env.reset)(jax.random.split(rng_reset, config["num_envs"]))
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
            runner_state: RunnerState, unused: None
        ) -> tuple[RunnerState, None]:
            def step(
                runner_state: RunnerState, unused: None
            ) -> tuple[RunnerState, Transition]:
                rng, rng_mode, rng_action, rng_step = jax.random.split(
                    runner_state.rng, 4
                )
                state, obs, br_mode = (
                    runner_state.state,
                    runner_state.obs,
                    runner_state.br_mode,
                )
                br_logits, value = network.apply(runner_state.br_train_state.params, obs)
                avg_logits, _ = network.apply(runner_state.avg_train_state.params, obs)
                action_mask = jax.vmap(env.get_avail_actions)(state)
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
                br_log_prob = jnp.where(
                    is_br,
                    distrax.Categorical(logits=br_logits_masked).log_prob(action),
                    0.0,
                )

                next_state, next_obs, reward, _, done, info = jax.vmap(env.step)(
                    jax.random.split(rng_step, config["num_envs"]), state, action
                )

                # A new hand starts after done (the env auto-resets): every
                # player draws BR (prob. anticipatory_eta) or average policy.
                new_br_mode = jax.random.bernoulli(
                    rng_mode, p=config["anticipatory_eta"], shape=br_mode.shape
                )
                next_br_mode = jnp.where(done[:, None], new_br_mode, br_mode)

                transition = Transition(
                    obs=obs,
                    action_mask=action_mask,
                    action=action,
                    br_log_prob=br_log_prob,
                    reward=reward,
                    done=done,
                    value=value,
                    player_idx=player_idx,
                    is_br=is_br,
                    br_mode=br_mode,
                    hand_length=info["timestep"],
                )
                runner_state = runner_state._replace(
                    state=next_state,
                    obs=next_obs,
                    done=done,
                    br_mode=next_br_mode,
                    rng=rng,
                )
                return runner_state, transition

            runner_state, transitions = lax.scan(
                step, runner_state, None, config["num_steps_per_env_per_update"]
            )

            br_train_state = runner_state.br_train_state
            avg_train_state = runner_state.avg_train_state
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
                runner_state.state.current_player_idx,
                config["gamma"],
                config["gae_lambda"],
            )
            train_mask = valid & transitions.is_br

            # NFSP: every BR decision goes to the reservoir.
            sl_buffer = reservoir_append(
                runner_state.sl_buffer,
                transitions.obs.reshape(-1, transitions.obs.shape[-1]),
                transitions.action_mask.reshape(-1, action_dim),
                transitions.action.reshape(-1),
                transitions.is_br.reshape(-1),
                rng_sl_append,
            )

            def update_ppo_epoch(
                ppo_state: PPOUpdateState, unused: None
            ) -> tuple[PPOUpdateState, dict[str, FloatArray]]:
                rng, rng_permute = jax.random.split(ppo_state.rng)
                batch = (
                    ppo_state.transitions,
                    ppo_state.advantages,
                    ppo_state.targets,
                    ppo_state.train_mask,
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

                def update_minibatch(
                    train_state: TrainState,
                    minibatch: tuple[Transition, FloatArray, FloatArray, BoolArray],
                ) -> tuple[TrainState, dict[str, FloatArray]]:
                    transitions_mb, advantages_mb, targets_mb, train_mask_mb = minibatch

                    def loss(params) -> tuple[FloatArray, dict[str, FloatArray]]:
                        logits, value = network.apply(params, transitions_mb.obs)
                        logits_masked = jnp.where(
                            transitions_mb.action_mask, logits, -jnp.inf
                        )
                        pi = distrax.Categorical(logits=logits_masked)
                        log_prob = pi.log_prob(transitions_mb.action)

                        # Only BR decisions with a known outcome enter the loss.
                        mask = train_mask_mb.astype(jnp.float32)
                        mask_denom = jnp.maximum(mask.sum(), 1.0)

                        mean_adv = masked_mean(advantages_mb, mask)
                        std_adv = jnp.sqrt(
                            masked_mean(jnp.square(advantages_mb - mean_adv), mask)
                        )
                        gae_normalized = (advantages_mb - mean_adv) / (std_adv + 1e-8)

                        logratio = jnp.where(
                            train_mask_mb, log_prob - transitions_mb.br_log_prob, 0.0
                        )
                        ratio = jnp.exp(logratio)
                        loss_actor_raw = ratio * gae_normalized
                        loss_actor_clipped = (
                            jnp.clip(
                                ratio, 1.0 - config["clip_eps"], 1.0 + config["clip_eps"]
                            )
                            * gae_normalized
                        )
                        actor_per_item = -jnp.minimum(loss_actor_raw, loss_actor_clipped)
                        loss_actor = (actor_per_item * mask).sum() / mask_denom

                        entropy = (pi.entropy() * mask).sum() / mask_denom

                        value_clipped = transitions_mb.value + jnp.clip(
                            value - transitions_mb.value,
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

                        kl_backward = (((ratio - 1) - logratio) * mask).sum() / mask_denom
                        kl_forward = (
                            (ratio * logratio - (ratio - 1)) * mask
                        ).sum() / mask_denom
                        clip_frac = (
                            (jnp.abs(ratio - 1) > config["clip_eps"]).astype(jnp.float32)
                            * mask
                        ).sum() / mask_denom

                        return total_loss, {
                            "total_loss": total_loss,
                            "value_loss": value_loss,
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "ratio_mean": masked_mean(ratio, mask),
                            "ratio_min": jnp.where(train_mask_mb, ratio, jnp.inf).min(),
                            "ratio_max": jnp.where(train_mask_mb, ratio, -jnp.inf).max(),
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

                final_train_state, batch_stats = lax.scan(
                    update_minibatch, ppo_state.train_state, minibatches
                )
                ppo_state = ppo_state._replace(train_state=final_train_state, rng=rng)
                return ppo_state, batch_stats

            ppo_state = PPOUpdateState(
                train_state=br_train_state,
                transitions=transitions,
                advantages=advantages,
                targets=targets,
                train_mask=train_mask,
                rng=rng_ppo_update,
            )
            final_ppo_state, ppo_loss_info = lax.scan(
                update_ppo_epoch, ppo_state, None, config["num_epochs"]
            )
            br_train_state = final_ppo_state.train_state
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
                return sl_state._replace(train_state=train_state, rng=rng), aux

            sl_state = SLUpdateState(
                train_state=avg_train_state, sl_buffer=sl_buffer, rng=rng_sl_update
            )
            sl_state, sl_loss_info = lax.scan(
                update_sl_step, sl_state, None, config["sl_num_steps_per_update"]
            )
            avg_train_state = sl_state.train_state
            sl_loss_info = jax.tree_util.tree_map(lambda x: x.mean(), sl_loss_info)

            update = runner_state.update_step
            should_compare = compare_enabled & (
                (update % config["compare_interval"] == 0) | (update == last_update)
            )
            avg_chips, br_chips, eval_len = lax.cond(
                should_compare,
                lambda: run_compare_eval(
                    rng_compare, avg_train_state.params, br_train_state.params
                ),
                lambda: (jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)),
            )

            # Self-play payoffs per finished hand in chips, split by the policy
            # the player followed in that hand. Against the mixed opponents
            # the BR should win (the average policy then loses).
            hand_end = transitions.done[..., None]
            br_hands = hand_end & transitions.br_mode
            avg_hands = hand_end & ~transitions.br_mode
            num_hands = jnp.maximum(transitions.done.sum(), 1)
            metric = {
                "update_step": update,
                "env_steps": (update + 1)
                * config["num_envs"]
                * config["num_steps_per_env_per_update"],
                "hands_completed": transitions.done.sum(),
                "ep_length_avg": (transitions.hand_length * transitions.done).sum()
                / num_hands,
                "br_return_per_hand": (transitions.reward * br_hands).sum()
                / jnp.maximum(br_hands.sum(), 1),
                "avg_return_per_hand": (transitions.reward * avg_hands).sum()
                / jnp.maximum(avg_hands.sum(), 1),
                "br_action_frac_rollout": transitions.is_br.mean(),
                "ppo_sample_frac": train_mask.mean(),
                "sl_buffer_size": sl_buffer.size.astype(jnp.float32),
                "sl_buffer_seen": sl_buffer.seen.astype(jnp.float32),
                "br_lr": (
                    linear_decay(br_train_state.step)
                    if config["anneal_lr"]
                    else jnp.asarray(config["lr"])
                ),
                eval_names[0]: avg_chips,
                eval_names[1]: br_chips,
                "avg_eval_episode_length": eval_len,
            }
            metric.update(ppo_loss_info)
            metric.update(sl_loss_info)
            jax.experimental.io_callback(
                logging_callback, None, seed, metric, should_compare
            )

            runner_state = runner_state._replace(
                br_train_state=br_train_state,
                avg_train_state=avg_train_state,
                sl_buffer=sl_buffer,
                update_step=update + 1,
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
            update_step, initial_runner_state, None, config["num_update_steps"]
        )
        return final_runner_state

    return train


def save_checkpoints(config: dict, final_runner_state: RunnerState) -> str:
    """Saves the networks of every seed after training.

    Layout of {save_dir}/{job_type}_{timestamp}/:
        avg_policy_{i}.msgpack  average policy of seed i (NFSP's output strategy)
        br_{i}.msgpack          PPO best response of seed i
        config.yaml             resolved config of the run
    i = 0..num_seeds-1 indexes the vmapped seeds (wandb run "{seed}_{i}"). Each
    file holds the flax params of one network, without a seed axis, written with
    flax.serialization.to_bytes. Both networks are
    ActorCriticDiscreteMLP(action_dim=37, hidden_dim=fc_dim_size) mapping a raw
    env observation to (action logits, value); load either one as a baseline
    with compare_network_type=actor_critic, and mask illegal actions before the
    softmax. The value head of the average policy is unused.
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


def print_eval_summary() -> None:
    """Last evaluation of every seed, and the mean over seeds."""
    if not EVALUATIONS:
        return
    for name in EVALUATIONS[min(EVALUATIONS)][-1][1]:
        finals = [EVALUATIONS[s][-1][1][name] for s in sorted(EVALUATIONS)]
        values = " ".join(f"{v:+.3f}" for v in finals)
        print(
            f"final {name} (chips per hand): mean {np.mean(finals):+.3f} "
            f"std {np.std(finals):.3f} over {len(finals)} seed(s): {values}"
        )


@hydra.main(version_base=None, config_path="./", config_name="config_ppo_nfsp")
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
            f"{run_time:.1f} s (all seeds in parallel, evaluation included)"
        )
        print_eval_summary()
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
