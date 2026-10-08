"""
PPO-NFSP for heads-up no-limit Texas Hold'em.

Neural Fictitious Self-Play (Heinrich & Silver 2016, arXiv:1603.01121) with
PPO as the best-response (BR) learner. Both seats share one BR network and one
average-policy network.

- Policy mixing: at the start of every hand each player independently draws
  the policy it follows for the whole hand, the BR with probability
  anticipatory_eta and the average policy otherwise.
- BR learning: PPO is on-policy, so the BR actor and critic train only on
  decisions made while following the BR. Returns follow each player's own
  decisions: a decision's reward is everything the player receives until its
  next decision or the end of the hand (which may come on the opponent's step),
  and it bootstraps from the value of that next decision (see per_player_gae).
- Average policy: every BR decision (obs, action) is added to a reservoir
  buffer (Algorithm R) and the average policy is fit to it by cross-entropy.
  The average policy is NFSP's output strategy; the BR is a training device.

Units: the env pays raw chips per hand (stacks 100, blinds 1/2). Training uses
reward * reward_scale, and logged returns are in chips per hand. The two chip
counts in the observation (raw chips, up to the 100-chip stack) are divided by
the starting stack inside the networks (ChipScaledInput); the env and the
observations stored in the reservoir stay in raw chips.

Checkpoints (save_final): see save_checkpoints.
"""

import datetime
import os
import time
from typing import Callable, NamedTuple

import distrax
from flax import serialization
import flax.linen as nn
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
from bluffjax.environments.texas_nolimit_holdem.texas_nolimit_holdem import (
    TexasNoLimitHoldEmState,
)
from bluffjax.networks.mlp import ActorCriticDiscreteMLP
from bluffjax.utils.jax_utils import pytree_norm
from bluffjax.utils.paths import register_resolvers
from bluffjax.utils.wandb_multilogger import WandbMultiLogger

# ReBeL evaluation opponent
from bluffjax.examples.holdem_rebel.agent import PolicyPlayer, load_rebel, play_match
from bluffjax.examples.holdem_rebel.game import make_game

LOGGER = None


class Transition(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    br_log_prob: FloatArray  # log-prob of the action under the BR (BR decisions)
    reward: FloatArray  # (num_agents,) env reward for every player, env units
    done: BoolArray  # the hand ended at this step (the env auto-resets)
    value: FloatArray  # BR critic value of obs, in scaled reward units
    player_idx: IntArray
    is_br: BoolArray  # the acting player follows the BR in this hand
    br_mode: BoolArray  # (num_agents,) which players follow the BR in this hand
    info: dict[str, Any]


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
    state: TexasNoLimitHoldEmState
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


class ChipScaledInput(nn.Module):
    """Applies `net` to the HUNL observation with its two chip-count features
    (the acting player's chips in the pot and the largest chips in the pot, raw
    chips up to the starting stack) divided by the starting stack. The 52 card
    indicators pass through unchanged. Every use of the networks, training and
    evaluation alike, goes through this, so they all see the same inputs."""

    net: nn.Module
    num_cards: int
    init_chips: float

    @nn.compact
    def __call__(self, x: FloatArray) -> Any:
        scale = jnp.ones(x.shape[-1], dtype=jnp.float32)
        scale = scale.at[self.num_cards :].set(1.0 / self.init_chips)
        return self.net(x * scale)


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
    step t up to p's next decision t' (so rewards paid on the opponent's steps
    are included), and the decision bootstraps from values[t']. If the hand ends
    first the decision is terminal. At the rollout cut-off only the player to
    act has a next decision (bootstrap from last_value). The other player's
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


def make_train(config: dict) -> Callable[[PRNGKeyArray, int], RunnerState]:
    env = make("texas_nolimit_holdem", **config["env_kwargs"])
    sample_state, _ = env.reset(jax.random.PRNGKey(0))
    action_dim = int(env.get_avail_actions(sample_state).shape[-1])
    config["batch_shuffle_dim"] = (
        config["num_steps_per_env_per_update"] * config["num_envs"]
    )

    network = ChipScaledInput(
        ActorCriticDiscreteMLP(action_dim=action_dim, hidden_dim=config["fc_dim_size"]),
        num_cards=env.deck_size,
        init_chips=env.init_chips,
    )

    compare_mode = config["compare_mode"]
    compare_against_random = compare_mode == "random"
    compare_enabled = config["compare"]
    compare_network_type = config["compare_network_type"]

    baseline_params = None
    rebel_player = None
    if compare_enabled and (not compare_against_random):
        checkpoint_path = config["compare_with"]
        if compare_network_type == "rebel":
            # ReBeL with search at test time (bluffjax/examples/holdem_rebel).
            # Runs inside the vmap over seeds, so it can't branch per street.
            rebel_player, baseline_params, rebel_meta = load_rebel(
                checkpoint_path,
                cfr_iters=config["compare_cfr_iters"],
                solve_chunk=config["compare_solve_chunk"],
                by_street=False,
            )
            if rebel_meta["game"] != "hunl":
                raise ValueError(
                    f"{checkpoint_path} is a {rebel_meta['game']} ReBeL checkpoint"
                )
            if env.init_chips != rebel_player.game.stack:
                raise ValueError(
                    "the ReBeL opponent models 100-chip stacks, the env has "
                    f"{env.init_chips}"
                )
            print(
                f"Loaded ReBeL opponent from {checkpoint_path} "
                f"({rebel_meta['samples']} samples, "
                f"{rebel_player.cfr_iters} CFR iterations per subgame)"
            )
        elif compare_network_type == "actor_critic":
            # Load actor-critic network (same style as PPO-NFSP)
            template_obs = jnp.zeros((env.obs_dim,), dtype=jnp.float32)
            template_params = network.init(jax.random.PRNGKey(0), template_obs)
            with open(checkpoint_path, "rb") as f:
                baseline_params = serialization.from_bytes(template_params, f.read())
            print(
                f"Loaded actor-critic baseline from {checkpoint_path} "
                f"(compare_network_type=actor_critic)"
            )
        else:
            raise ValueError(
                f"compare_network_type must be 'rebel' or 'actor_critic', "
                f"got '{compare_network_type}'"
            )

    def linear_decay(count: int) -> float:
        frac = 1.0 - count / config["num_gradient_steps"]
        return config["lr"] * frac

    def wandb_callback(exp_id: int, metrics: dict) -> None:
        np_log_dict = {k: np.array(v) for k, v in metrics.items()}
        LOGGER.log(int(exp_id), np_log_dict)

    def init_sl_buffer(obs_dim: int, action_dim: int) -> SLBufferState:
        capacity = config["sl_reservoir_capacity"]
        return SLBufferState(
            # float16 halves the memory and is exact for these observations
            # (0/1 card indicators and integer chip counts up to 100).
            obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float16),
            action_mask=jnp.zeros((capacity, action_dim), dtype=jnp.bool_),
            action=jnp.zeros((capacity,), dtype=jnp.int32),
            seen=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
        )

    def append_sl_samples(
        buffer: SLBufferState,
        obs_batch: FloatArray,
        action_mask_batch: BoolArray,
        action_batch: IntArray,
        valid_batch: BoolArray,
        rng: PRNGKeyArray,
    ) -> SLBufferState:
        return reservoir_append(
            buffer, obs_batch, action_mask_batch, action_batch, valid_batch, rng
        )

    def sample_sl_batch(
        rng: PRNGKeyArray,
        buffer: SLBufferState,
        batch_size: int,
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

    def sample_masked_action(
        params: Any,
        obs: FloatArray,
        action_mask: BoolArray,
        rng: PRNGKeyArray,
    ) -> IntArray:
        logits, _ = network.apply(params, obs)
        logits_masked = jnp.where(action_mask, logits, -jnp.inf)
        return distrax.Categorical(logits=logits_masked).sample(seed=rng)

    def sample_random_legal_action(
        action_mask: BoolArray, rng: PRNGKeyArray
    ) -> IntArray:
        probs = action_mask.astype(jnp.float32)
        probs = probs / jnp.maximum(probs.sum(), 1.0)
        return distrax.Categorical(probs=probs).sample(seed=rng)

    game = make_game("hunl") if rebel_player is None else rebel_player.game

    def policy_player(sample_fn: Callable) -> PolicyPlayer:
        """Plays sample_fn(params, obs, action_mask, rng) from the env obs."""
        return PolicyPlayer(
            lambda params, rng, state, avail: sample_fn(
                params, env.obs_from_state(state), avail, rng
            )
        )

    train_player = policy_player(sample_masked_action)
    if compare_against_random:
        baseline_player = PolicyPlayer(
            lambda params, rng, state, avail: sample_random_legal_action(avail, rng)
        )
    elif compare_network_type == "rebel":
        baseline_player = rebel_player
    else:
        baseline_player = policy_player(sample_masked_action)

    def run_compare_eval(
        rng: PRNGKeyArray,
        avg_params: Any,
        br_params: Any,
        checkpoint_params: Any,
    ) -> tuple[PRNGKeyArray, FloatArray, FloatArray, FloatArray]:
        """Chips per hand of the average and best-response policies.

        compare_steps deals, each played twice with the seats swapped.
        """
        rng, rng_avg, rng_br = jax.random.split(rng, 3)
        avg = play_match(
            env, game, train_player, avg_params, baseline_player,
            checkpoint_params, config["compare_steps"], rng_avg,
        )
        br = play_match(
            env, game, train_player, br_params, baseline_player,
            checkpoint_params, config["compare_steps"], rng_br,
        )
        avg_eval_len = (avg.hand_length.mean() + br.hand_length.mean()) / 2
        return rng, avg.rewards.mean(), br.rewards.mean(), avg_eval_len

    def train(rng: PRNGKeyArray, seed: int) -> RunnerState:
        def train_setup(
            rng_inner: PRNGKeyArray,
        ) -> tuple[
            TrainState,
            TrainState,
            TexasNoLimitHoldEmState,
            FloatArray,
            SLBufferState,
        ]:
            rng_inner, rng_reset = jax.random.split(rng_inner)
            rng_resets = jax.random.split(rng_reset, config["num_envs"])
            state, obs = jax.vmap(env.reset, in_axes=(0))(rng_resets)

            rng_inner, rng_br_init, rng_avg_init = jax.random.split(rng_inner, 3)
            br_params = network.init(rng_br_init, obs)
            # The average policy uses the same actor-critic module so that both
            # networks can be evaluated the same way; only its actor is trained.
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
            sl_buffer = init_sl_buffer(obs.shape[-1], action_dim)
            return (br_train_state, avg_train_state, state, obs, sl_buffer)

        rng, rng_setup, rng_mode = jax.random.split(rng, 3)
        br_train_state, avg_train_state, state, obs, sl_buffer = train_setup(rng_setup)
        br_mode = jax.random.bernoulli(
            rng_mode,
            p=config["anticipatory_eta"],
            shape=(config["num_envs"], env.num_agents),
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
                rng_s, rng_mode, rng_action, rng_step = jax.random.split(rng_s, 4)

                br_logits, value = network.apply(br_train_state_s.params, obs_s)
                avg_logits, _ = network.apply(avg_train_state_s.params, obs_s)
                action_mask = jax.vmap(env.get_avail_actions, in_axes=(0))(state_s)

                br_logits_masked = jnp.where(action_mask, br_logits, -jnp.inf)
                avg_logits_masked = jnp.where(action_mask, avg_logits, -jnp.inf)

                # NFSP: the acting player follows the policy it drew for this hand.
                player_idx = state_s.current_player_idx
                is_br = _pick(br_mode_s, player_idx)
                acting_logits = jnp.where(
                    is_br[:, None], br_logits_masked, avg_logits_masked
                )

                acting_pi = distrax.Categorical(logits=acting_logits)
                br_pi = distrax.Categorical(logits=br_logits_masked)
                action = acting_pi.sample(seed=rng_action)
                br_log_prob = jnp.where(is_br, br_pi.log_prob(action), 0.0)

                rng_steps = jax.random.split(rng_step, config["num_envs"])
                next_state, next_obs, reward, absorbing, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_steps, state_s, action)

                # A new hand starts after done (the env auto-resets): every
                # player draws BR (prob. anticipatory_eta) or average policy.
                new_br_mode = jax.random.bernoulli(
                    rng_mode, p=config["anticipatory_eta"], shape=br_mode_s.shape
                )
                next_br_mode = jnp.where(done[:, None], new_br_mode, br_mode_s)

                transition = Transition(
                    obs=obs_s,
                    action_mask=action_mask,
                    action=action,
                    br_log_prob=br_log_prob,
                    reward=reward,
                    done=done,
                    value=value,
                    player_idx=player_idx,
                    is_br=is_br,
                    br_mode=br_mode_s,
                    info=info,
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
            rng, rng_sl_append, rng_ppo_update, rng_sl_update, rng_compare = (
                jax.random.split(runner_state.rng, 5)
            )

            _, last_val = network.apply(br_train_state.params, last_obs)

            # Per-player GAE in scaled reward units. PPO is on-policy: the BR
            # actor and critic train on BR decisions whose outcome is known.
            last_player_idx = last_state.current_player_idx
            advantages, targets, valid = per_player_gae(
                transitions.value,
                transitions.reward * config["reward_scale"],
                transitions.done,
                transitions.player_idx,
                last_val,
                last_player_idx,
                config["gamma"],
                config["gae_lambda"],
            )
            train_mask = valid & transitions.is_br

            # NFSP: every BR decision goes to the reservoir.
            obs_flat = transitions.obs.reshape(-1, transitions.obs.shape[-1])
            action_mask_flat = transitions.action_mask.reshape(
                -1, transitions.action_mask.shape[-1]
            )
            action_flat = transitions.action.reshape(-1)
            is_br_flat = transitions.is_br.reshape(-1)
            sl_buffer = append_sl_samples(
                sl_buffer,
                obs_flat,
                action_mask_flat,
                action_flat,
                is_br_flat,
                rng_sl_append,
            )

            def update_ppo_epoch(
                ppo_state: PPOUpdateState,
                unused: None,
            ) -> tuple[PPOUpdateState, dict[str, FloatArray]]:
                rng_s, rng_permute = jax.random.split(ppo_state.rng)
                batch = (
                    ppo_state.transitions,
                    ppo_state.advantages,
                    ppo_state.targets,
                    ppo_state.train_mask,
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
                    minibatch: tuple[Transition, FloatArray, FloatArray, BoolArray],
                ) -> tuple[TrainState, dict[str, FloatArray]]:
                    transitions_mb, advantages_mb, targets_mb, train_mask_mb = minibatch

                    def loss(
                        params,
                        transitions_mb,
                        advantages_mb,
                        targets_mb,
                        train_mask_mb,
                    ) -> tuple[FloatArray, dict[str, FloatArray]]:
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

                        kl_backward = (
                            ((ratio - 1) - logratio) * mask
                        ).sum() / mask_denom
                        kl_forward = (
                            (ratio * logratio - (ratio - 1)) * mask
                        ).sum() / mask_denom
                        clip_frac = (
                            (jnp.abs(ratio - 1) > config["clip_eps"]).astype(
                                jnp.float32
                            )
                            * mask
                        ).sum() / mask_denom

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

                    grad_fn = jax.value_and_grad(loss, has_aux=True)
                    (_, aux), grads = grad_fn(
                        train_state.params,
                        transitions_mb,
                        advantages_mb,
                        targets_mb,
                        train_mask_mb,
                    )
                    aux["grad_norm"] = pytree_norm(grads)
                    updated_train_state = train_state.apply_gradients(grads=grads)
                    return updated_train_state, aux

                final_train_state, batch_stats = lax.scan(
                    update_minibatch,
                    ppo_state.train_state,
                    minibatches,
                )
                ppo_state = PPOUpdateState(
                    train_state=final_train_state,
                    transitions=ppo_state.transitions,
                    advantages=ppo_state.advantages,
                    targets=ppo_state.targets,
                    train_mask=ppo_state.train_mask,
                    rng=rng_s,
                )
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
                update_ppo_epoch,
                ppo_state,
                None,
                config["num_epochs"],
            )
            br_train_state = final_ppo_state.train_state
            ppo_loss_info = jax.tree_util.tree_map(lambda x: x.mean(), ppo_loss_info)

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
                        logits, _ = network.apply(params, batch.obs)
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

            masked_returns = jnp.where(
                transitions.done[..., None], transitions.reward, 0.0
            )
            done_count = jnp.maximum(transitions.done.sum(), 1)
            returns_avg = masked_returns.sum() / done_count
            returns_avg_agent_one = masked_returns[..., 0].sum() / done_count
            returns_avg_agent_two = masked_returns[..., 1].sum() / done_count

            masked_timesteps = jnp.where(
                transitions.done, transitions.info["timestep"], 0.0
            )
            ep_length_avg = masked_timesteps.sum() / done_count

            should_compare = compare_enabled & (
                runner_state.update_step % config["compare_interval"] == 0
            )

            def do_compare(carry_rng):
                checkpoint_params = baseline_params
                if compare_against_random:
                    checkpoint_params = avg_train_state.params
                return run_compare_eval(
                    carry_rng,
                    avg_train_state.params,
                    br_train_state.params,
                    checkpoint_params,
                )

            def skip_compare(carry_rng):
                return (
                    carry_rng,
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                )

            (
                rng_compare,
                avg_chips_avg_vs_baseline,
                avg_chips_br_vs_baseline,
                avg_eval_episode_length,
            ) = lax.cond(
                should_compare,
                do_compare,
                skip_compare,
                rng_compare,
            )

            metric = {
                "returns_avg": returns_avg,
                "returns_avg_agent_one": returns_avg_agent_one,
                "returns_avg_agent_two": returns_avg_agent_two,
                "ep_length_avg": ep_length_avg,
                "br_action_frac_rollout": transitions.is_br.mean(),
                "sl_buffer_size": sl_buffer.size.astype(jnp.float32),
                "sl_buffer_seen": sl_buffer.seen.astype(jnp.float32),
                "update_step": runner_state.update_step,
                "avg_chips_avg_vs_baseline": (
                    jnp.array(0.0, dtype=jnp.float32)
                    if compare_against_random
                    else avg_chips_avg_vs_baseline
                ),
                "avg_chips_br_vs_baseline": (
                    jnp.array(0.0, dtype=jnp.float32)
                    if compare_against_random
                    else avg_chips_br_vs_baseline
                ),
                "avg_chips_avg_vs_random": (
                    avg_chips_avg_vs_baseline
                    if compare_against_random
                    else jnp.array(0.0, dtype=jnp.float32)
                ),
                "avg_chips_br_vs_random": (
                    avg_chips_br_vs_baseline
                    if compare_against_random
                    else jnp.array(0.0, dtype=jnp.float32)
                ),
                "avg_eval_episode_length": avg_eval_episode_length,
            }
            metric.update(ppo_loss_info)
            metric.update(sl_loss_info)

            # Self-play payoffs per finished hand in env units (chips), split by
            # the policy the player followed in that hand. Against the mixed
            # opponent the BR should win (the average policy then loses).
            hand_end = transitions.done[..., None]
            br_hands = hand_end & transitions.br_mode
            avg_hands = hand_end & ~transitions.br_mode
            metric.update(
                {
                    "env_steps": (runner_state.update_step + 1)
                    * config["num_envs"]
                    * config["num_steps_per_env_per_update"],
                    "hands_completed": transitions.done.sum(),
                    "br_return_per_hand": (transitions.reward * br_hands).sum()
                    / jnp.maximum(br_hands.sum(), 1),
                    "avg_return_per_hand": (transitions.reward * avg_hands).sum()
                    / jnp.maximum(avg_hands.sum(), 1),
                    "ppo_sample_frac": train_mask.mean(),
                    "br_lr": (
                        linear_decay(br_train_state.step)
                        if config["anneal_lr"]
                        else jnp.asarray(config["lr"])
                    ),
                }
            )

            def logging_callback(seed_val, metric_dict):
                wandb_callback(seed_val, dict(metric_dict))

            jax.experimental.io_callback(
                logging_callback,
                None,
                seed,
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
                rng=rng_compare,
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
        br_{i}.msgpack          PPO best response of seed i
        config.yaml             resolved config of the run
    i = 0..num_seeds-1 indexes the vmapped seeds (wandb run "{seed}_{i}"). Each
    file holds the flax params of one network, without a seed axis, written with
    flax.serialization.to_bytes. Both networks are
    ChipScaledInput(ActorCriticDiscreteMLP(action_dim, fc_dim_size), 52, 100)
    mapping a raw env observation to (action logits, value); restore with
    serialization.from_bytes(template, data) where template =
    network.init(rng, obs), and mask illegal actions before the softmax. The
    value head of the average policy is unused.
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
            f"{run_time:.1f} s ({run_time / env_steps * 1e6:.2f} s per 1e6 env "
            f"steps, all seeds in parallel)"
        )
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
