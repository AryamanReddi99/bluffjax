"""
PQN-NFSP for heads-up no-limit Texas Hold'em.

Neural Fictitious Self-Play (Heinrich & Silver 2016, arXiv:1603.01121) with
PQN (Gallici et al. 2024, "Simplifying Deep Temporal Difference Learning") as
the best-response (BR) learner in place of DQN. Both seats share one Q-network
and one average-policy network.

- Policy mixing: at the start of every hand each player independently draws
  the policy it follows for the whole hand, the epsilon-greedy BR with
  probability anticipatory_eta and the average policy otherwise.
- BR learning: as in NFSP the Q-network learns from all of a player's
  decisions, whichever policy it followed. Targets follow each player's own
  decisions (a decision's reward is everything the player receives until its
  next decision or the end of the hand, which may come on the opponent's step)
  and bootstrap from the max over legal actions of Q at that next decision.
  The lambda-trace runs through decisions taken by the epsilon-greedy BR
  (Peng's Q(lambda), as in PQN) and is cut at average-policy decisions, so
  those give one-step Q-learning targets (see per_player_q_lambda_targets).
- Average policy: every BR decision (obs, action) is added to a reservoir
  buffer (Algorithm R) and the average policy is fit to it by cross-entropy.
  The average policy is NFSP's output strategy; the BR is a training device.

Units: the env pays raw chips per hand (stacks 100, blinds 1/2). Training uses
reward * reward_scale (so Q-values are in scaled units), and logged returns
are in chips per hand. The two chip counts in the observation (raw chips, up
to the 100-chip stack) are divided by the starting stack inside the networks
(ChipScaledInput); the env and the observations stored in the reservoir stay
in raw chips.

Checkpoints (save_final): see save_checkpoints.
"""

import datetime
import os
import time
from typing import Any, Callable, NamedTuple

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
    BoolArray,
    FloatArray,
    IntArray,
    PRNGKeyArray,
)
from bluffjax import make
from bluffjax.environments.texas_nolimit_holdem.texas_nolimit_holdem import (
    TexasNoLimitHoldEmState,
)
from bluffjax.networks.mlp import (
    ActorCriticDiscreteMLP,
    ActorDiscreteMLP,
    QNetworkDiscreteMLP,
)
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
    reward: FloatArray  # (num_agents,) env reward for every player, env units
    done: BoolArray  # the hand ended at this step (the env auto-resets)
    q_max: FloatArray  # max over legal actions of Q(obs), scaled reward units
    player_idx: IntArray
    info: dict[str, Any]
    is_br: BoolArray  # the acting player follows the BR in this hand
    br_mode: BoolArray  # (num_agents,) which players follow the BR in this hand


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
        dones: (T, N) the hand ended at step t (the env then auto-resets).
        players: (T, N) player acting at step t.
        traces: (T, N) trace coefficient of the decision at t: lambda if it was
            taken by the epsilon-greedy BR, 0 if by the average policy.
        last_q_max: (N,) max over legal actions of Q at the observation after
            the last step.
        last_player: (N,) player to act after the last step.

    For player p's decision at step t with next own decision t' in the same
    hand, R = p's rewards from step t up to t' (rewards paid on the opponent's
    steps included) and
        G_t = R + gamma * ((1 - c) * q_max[t'] + c * G_t'),  c = traces[t'],
    i.e. Peng's Q(lambda) as in PQN when the next decision follows the BR, and
    the one-step Q-learning target when it follows the average policy. If the
    hand ends first, G_t = R. At the rollout cut-off only the player to act has
    a next decision (G_t = R + gamma * last_q_max). The other player's trailing
    decision in an unfinished hand has no known outcome yet: it is marked
    invalid, and that player's earlier decisions bootstrap from q_max there
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
    # known, whether the hand ends before it, and the reward collected since
    # the decision being processed.
    init = (last_q, last_q, zeros, last_onehot, jnp.zeros_like(last_onehot), zeros)

    def body(carry, x):
        next_q, next_return, next_trace, known, terminal, reward_acc = carry
        q_max_t, reward, done, player, trace = x
        # Steps after a hand-ending step belong to the next hand.
        hand_over = done[:, None]
        known = known | hand_over
        terminal = terminal | hand_over
        reward_acc = jnp.where(hand_over, 0.0, reward_acc) + reward

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


def make_train(config: dict) -> Callable[[PRNGKeyArray, int], RunnerState]:
    env = make("texas_nolimit_holdem", **config["env_kwargs"])
    sample_state, _ = env.reset(jax.random.PRNGKey(0))
    action_dim = int(env.get_avail_actions(sample_state).shape[-1])
    config["batch_shuffle_dim"] = (
        config["num_steps_per_env_per_update"] * config["num_envs"]
    )

    compare_mode = config["compare_mode"]
    compare_against_random = compare_mode == "random"
    compare_enabled = config["compare"]
    compare_network_type = config["compare_network_type"]

    baseline_params = None
    rebel_player = None
    actor_critic_network = None
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
            actor_critic_network = ActorCriticDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )
            template_obs = jnp.zeros((env.obs_dim,), dtype=jnp.float32)
            template_params = actor_critic_network.init(
                jax.random.PRNGKey(0), template_obs
            )
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

    def epsilon_schedule(update_step: FloatArray) -> FloatArray:
        decay_steps = max(
            config["exploration_fraction"] * config["num_update_steps"], 1
        )
        frac = jnp.minimum(1.0, update_step.astype(jnp.float32) / decay_steps)
        return config["start_e"] + frac * (config["end_e"] - config["start_e"])

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

    def sample_action_actor(
        avg_net: ActorDiscreteMLP,
        params: Any,
        obs: FloatArray,
        action_mask: BoolArray,
        rng: PRNGKeyArray,
    ) -> IntArray:
        logits = avg_net.apply(params, obs.astype(jnp.float32))
        logits_masked = jnp.where(action_mask, logits, -jnp.inf)
        return distrax.Categorical(logits=logits_masked).sample(seed=rng)

    def sample_action_q_greedy(
        br_net: QNetworkDiscreteMLP,
        params: Any,
        obs: FloatArray,
        action_mask: BoolArray,
        rng: PRNGKeyArray,
    ) -> IntArray:
        q_vals = br_net.apply(params, obs.astype(jnp.float32))
        q_vals_masked = jnp.where(action_mask, q_vals, -jnp.inf)
        best_val = jnp.max(q_vals_masked)
        ties = (q_vals_masked == best_val) & action_mask
        logits = jnp.where(ties, 0.0, -1e9)
        return jax.random.categorical(rng, logits)

    def sample_masked_action_actor_critic(
        ac_net: ActorCriticDiscreteMLP,
        params: Any,
        obs: FloatArray,
        action_mask: BoolArray,
        rng: PRNGKeyArray,
    ) -> IntArray:
        logits, _ = ac_net.apply(params, obs)
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

    if compare_against_random:
        baseline_player = PolicyPlayer(
            lambda params, rng, state, avail: sample_random_legal_action(avail, rng)
        )
    elif compare_network_type == "rebel":
        baseline_player = rebel_player
    else:
        baseline_player = policy_player(
            lambda params, obs, mask, rng: sample_masked_action_actor_critic(
                actor_critic_network, params, obs, mask, rng
            )
        )

    def run_compare_eval(
        rng: PRNGKeyArray,
        avg_params: Any,
        br_params: Any,
        checkpoint_params: Any,
        br_net: QNetworkDiscreteMLP,
        avg_net: ActorDiscreteMLP,
    ) -> tuple[PRNGKeyArray, FloatArray, FloatArray, FloatArray]:
        """Chips per hand of the average (actor) and best-response (greedy Q) policies.

        compare_steps deals, each played twice with the seats swapped.
        """
        avg_player = policy_player(
            lambda params, obs, mask, rng: sample_action_actor(
                avg_net, params, obs, mask, rng
            )
        )
        br_player = policy_player(
            lambda params, obs, mask, rng: sample_action_q_greedy(
                br_net, params, obs, mask, rng
            )
        )
        rng, rng_avg, rng_br = jax.random.split(rng, 3)
        avg = play_match(
            env, game, avg_player, avg_params, baseline_player,
            checkpoint_params, config["compare_steps"], rng_avg,
        )
        br = play_match(
            env, game, br_player, br_params, baseline_player,
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
            QNetworkDiscreteMLP,
            ActorDiscreteMLP,
            TexasNoLimitHoldEmState,
            FloatArray,
            SLBufferState,
        ]:
            rng_inner, rng_reset = jax.random.split(rng_inner)
            rng_resets = jax.random.split(rng_reset, config["num_envs"])
            state, obs = jax.vmap(env.reset, in_axes=(0))(rng_resets)

            rng_inner, rng_br_init, rng_avg_init = jax.random.split(rng_inner, 3)
            br_network = ChipScaledInput(
                QNetworkDiscreteMLP(
                    action_dim=action_dim, hidden_dim=config["fc_dim_size"]
                ),
                num_cards=env.deck_size,
                init_chips=env.init_chips,
            )
            avg_network = ChipScaledInput(
                ActorDiscreteMLP(
                    action_dim=action_dim, hidden_dim=config["fc_dim_size"]
                ),
                num_cards=env.deck_size,
                init_chips=env.init_chips,
            )
            br_params = br_network.init(rng_br_init, obs)
            avg_params = avg_network.init(rng_avg_init, obs)

            br_lr = linear_decay if config["anneal_lr"] else config["lr"]
            br_tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=br_lr),
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
            return (
                br_train_state,
                avg_train_state,
                br_network,
                avg_network,
                state,
                obs,
                sl_buffer,
            )

        rng, rng_setup, rng_mode = jax.random.split(rng, 3)
        (
            br_train_state,
            avg_train_state,
            br_network,
            avg_network,
            state,
            obs,
            sl_buffer,
        ) = train_setup(rng_setup)
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
                (
                    rng_s,
                    rng_mode,
                    rng_explore,
                    rng_random,
                    rng_avg,
                    rng_step,
                ) = jax.random.split(rng_s, 6)
                action_mask = jax.vmap(env.get_avail_actions, in_axes=(0))(state_s)

                q_vals = br_network.apply(
                    br_train_state_s.params, obs_s.astype(jnp.float32)
                )
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

                avg_logits = avg_network.apply(
                    avg_train_state_s.params, obs_s.astype(jnp.float32)
                )
                avg_logits_masked = jnp.where(action_mask, avg_logits, -jnp.inf)
                avg_action = jax.random.categorical(rng_avg, avg_logits_masked)

                # NFSP: the acting player follows the policy it drew for this hand.
                player_idx = state_s.current_player_idx
                is_br = _pick(br_mode_s, player_idx)
                action = jnp.where(is_br, br_action, avg_action)

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
                    reward=reward,
                    done=done,
                    q_max=q_max,
                    player_idx=player_idx,
                    info=info,
                    is_br=is_br,
                    br_mode=br_mode_s,
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
            last_q_vals = br_network.apply(
                br_train_state.params, last_obs.astype(jnp.float32)
            )
            last_q_max = jnp.max(jnp.where(last_mask, last_q_vals, -jnp.inf), axis=-1)

            # Per-player Q(lambda) targets in scaled reward units. Q-learning is
            # off-policy, so (as in NFSP) every decision with a known outcome
            # trains the Q-network; the trace is cut at average-policy decisions.
            targets, valid = per_player_q_lambda_targets(
                transitions.q_max,
                transitions.reward * config["reward_scale"],
                transitions.done,
                transitions.player_idx,
                config["q_lambda"] * transitions.is_br.astype(jnp.float32),
                last_q_max,
                last_state.current_player_idx,
                config["gamma"],
            )
            train_mask = valid

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
                        q_vals = br_network.apply(params, obs_mb.astype(jnp.float32))
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
                            "q_values": (chosen_q * mask).sum() / mask_denom,
                            "mean_target": (targets_mb * mask).sum() / mask_denom,
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
                        logits = avg_network.apply(
                            params, batch.obs.astype(jnp.float32)
                        )
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
            returns_avg_agent_one = masked_returns[..., 0].sum() / done_count
            returns_avg_agent_two = masked_returns[..., 1].sum() / done_count

            masked_timesteps = jnp.where(
                transitions.done,
                transitions.info["timestep"].astype(jnp.float32),
                0.0,
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
                    br_network,
                    avg_network,
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
            metric.update(pqn_loss_info)
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
                    "q_sample_frac": train_mask.mean(),
                    "epsilon": epsilon_schedule(
                        jnp.asarray(runner_state.update_step, dtype=jnp.float32)
                    ),
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
        br_{i}.msgpack          PQN best-response Q-network of seed i
        config.yaml             resolved config of the run
    i = 0..num_seeds-1 indexes the vmapped seeds (wandb run "{seed}_{i}"). Each
    file holds the flax params of one network, without a seed axis, written with
    flax.serialization.to_bytes. The average policy is
    ChipScaledInput(ActorDiscreteMLP(action_dim, fc_dim_size), 52, 100) mapping
    a raw env observation to action logits; the BR is
    ChipScaledInput(QNetworkDiscreteMLP(action_dim, fc_dim_size), 52, 100)
    mapping it to Q-values in units of reward_scale * env reward (its policy is
    the argmax over legal actions). Restore with serialization.from_bytes(
    template, data) where template = network.init(rng, obs), and mask illegal
    actions.
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
