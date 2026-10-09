"""
PPO-NFSP for Leduc Hold'em.

Neural Fictitious Self-Play (Heinrich & Silver 2016, arXiv:1603.01121) with
PPO as the best-response (BR) learner. Both seats share one BR network and one
average-policy network (the observation encodes the position, so it is the
acting player's perfect-recall information state).

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
- Evaluation: exact exploitability (leduc_exploitability) of the average and
  BR policies every eval_interval updates and after the last update.

Units: the env pays chips per hand. Training uses reward * reward_scale, and
logged returns and exploitability are in chips.
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
from bluffjax.environments.leduc_holdem.leduc_holdem import (
    MAX_ACTIONS_PER_ROUND,
    LeducHoldemState,
)
from bluffjax.networks.mlp import ActorCriticDiscreteMLP
from bluffjax.utils.game_utils.leduc_exploitability import (
    _INFOSET_LIST,
    _KEY_TO_IDX,
    exploitability_from_policy_array,
    policy_array_from_network,
)
from bluffjax.utils.jax_utils import pytree_norm
from bluffjax.utils.paths import REPO_ROOT, register_resolvers

WANDB_RUNS: list = []  # one wandb run per vmapped seed, created in main()
# EXPLOITABILITY[seed] = [(env_steps, {policy name: exploitability}), ...]
EXPLOITABILITY: dict[int, list[tuple[int, dict[str, float]]]] = {}


class Transition(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    br_log_prob: FloatArray  # log-prob of the action under the BR (BR decisions)
    reward: FloatArray  # (num_agents,) env reward for every player, chips
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
    state: LeducHoldemState
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
    env = make("leduc_holdem", **config["env_kwargs"])
    if env.horizon < 2 * MAX_ACTIONS_PER_ROUND:
        # A truncated hand pays 0, but exploitability is for the full game.
        raise ValueError(
            f"horizon={env.horizon} truncates Leduc hands (up to "
            f"{2 * MAX_ACTIONS_PER_ROUND} actions)"
        )
    action_dim = env.num_actions
    config["batch_shuffle_dim"] = (
        config["num_steps_per_env_per_update"] * config["num_envs"]
    )
    # The average policy uses the same actor-critic module so that both
    # networks are evaluated the same way; only its actor is trained.
    network = ActorCriticDiscreteMLP(
        action_dim=action_dim, hidden_dim=config["fc_dim_size"]
    )
    eval_interval = config["eval_interval"]
    last_update = config["num_update_steps"] - 1

    def linear_decay(count: int) -> float:
        frac = 1.0 - count / config["num_gradient_steps"]
        return config["lr"] * frac

    def logging_callback(seed_val, metric_dict, policy_arrays) -> None:
        """Logs the metrics; computes the exact exploitability when it is due."""
        seed_i = int(seed_val)
        metrics = {k: np.asarray(v) for k, v in metric_dict.items()}
        update = int(metrics["update_step"])
        if update % eval_interval == 0 or update == last_update:
            expl = {
                name: exploitability_from_policy_array(
                    np.asarray(arr, dtype=np.float64), _KEY_TO_IDX
                )
                for name, arr in policy_arrays.items()
            }
            env_steps = int(metrics["env_steps"])
            EXPLOITABILITY.setdefault(seed_i, []).append((env_steps, expl))
            metrics["exploitability_avg_policy"] = expl["avg_policy"]
            metrics["exploitability_br_policy"] = expl["br_policy"]
            print(
                f"seed {seed_i} update {update + 1}/{last_update + 1} "
                f"({env_steps} env steps): exploitability average policy "
                f"{expl['avg_policy']:.4f}, BR {expl['br_policy']:.4f}",
                flush=True,
            )
        WANDB_RUNS[seed_i].log(metrics)

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

    def train(rng: PRNGKeyArray, seed: int) -> RunnerState:
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
            shape=(config["num_envs"], env.num_agents),
        )

        def update_step(
            runner_state: RunnerState, unused: None
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
                    info=info,
                )
                runner_state = RunnerState(
                    br_train_state=br_train_state,
                    avg_train_state=avg_train_state,
                    sl_buffer=runner_state.sl_buffer,
                    state=next_state,
                    obs=next_obs,
                    done=done,
                    br_mode=next_br_mode,
                    update_step=runner_state.update_step,
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
            rng, rng_sl_append, rng_ppo_update, rng_sl_update = jax.random.split(
                runner_state.rng, 4
            )

            _, last_val = network.apply(br_train_state.params, runner_state.obs)

            # Per-player GAE in scaled reward units. PPO is on-policy: the BR
            # actor and critic train on BR decisions whose outcome is known.
            advantages, targets, valid = per_player_gae(
                transitions.value,
                transitions.reward * config["reward_scale"],
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
                batch_reshaped = jax.tree_util.tree_map(
                    lambda x: x.reshape((-1,) + x.shape[2:]), batch
                )
                permutation = jax.random.permutation(
                    rng_permute, config["batch_shuffle_dim"]
                )
                batch_shuffled = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch_reshaped
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

                final_train_state, batch_stats = lax.scan(
                    update_minibatch, ppo_state.train_state, minibatches
                )
                ppo_state = PPOUpdateState(
                    train_state=final_train_state,
                    transitions=ppo_state.transitions,
                    advantages=ppo_state.advantages,
                    targets=ppo_state.targets,
                    train_mask=ppo_state.train_mask,
                    rng=rng,
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

            # Self-play payoffs per finished hand in chips, split by the policy
            # the player followed in that hand. Against the mixed opponent the
            # BR should win (the average policy then loses).
            hand_end = transitions.done[..., None]
            br_hands = hand_end & transitions.br_mode
            avg_hands = hand_end & ~transitions.br_mode
            num_hands = jnp.maximum(transitions.done.sum(), 1)
            metric = {
                "update_step": runner_state.update_step,
                "env_steps": (runner_state.update_step + 1)
                * config["num_envs"]
                * config["num_steps_per_env_per_update"],
                "hands_completed": transitions.done.sum(),
                "hand_length": (transitions.info["timestep"] * transitions.done).sum()
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
            }
            metric.update(ppo_loss_info)
            metric.update(sl_loss_info)

            # Both policies at every infoset; their exploitability is computed
            # on the host when due.
            policy_arrays = {
                "avg_policy": policy_array_from_network(
                    network.apply, avg_train_state.params, _INFOSET_LIST
                ),
                "br_policy": policy_array_from_network(
                    network.apply, br_train_state.params, _INFOSET_LIST
                ),
            }
            jax.experimental.io_callback(
                logging_callback, None, seed, metric, policy_arrays
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
            update_step, initial_runner_state, None, config["num_update_steps"]
        )
        return final_runner_state

    return train


def print_exploitability_summary(names: list[str]) -> None:
    """Final exploitability of every seed, and the mean over seeds."""
    for name in names:
        finals = [EXPLOITABILITY[s][-1][1][name] for s in sorted(EXPLOITABILITY)]
        values = " ".join(f"{v:.4f}" for v in finals)
        print(
            f"final exploitability ({name}): mean {np.mean(finals):.4f} "
            f"std {np.std(finals):.4f} over {len(finals)} seed(s): {values}"
        )


@hydra.main(version_base=None, config_path="./", config_name="config_ppo_nfsp")
def main(config: dict) -> None:
    global WANDB_RUNS
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

        print("Running...")
        start = time.time()
        jax.block_until_ready(train_vjit(rng_seeds, exp_ids))
        run_time = time.time() - start
        print(
            f"Trained {config['num_seeds']} seed(s) x {env_steps} env steps in "
            f"{run_time:.1f} s (all seeds in parallel, exploitability included)"
        )
        print_exploitability_summary(["avg_policy", "br_policy"])
    finally:
        for run in WANDB_RUNS:
            run.finish()
        print("Finished.")


if __name__ == "__main__":
    register_resolvers()
    main()
