"""
PPO-NFSP for Bluff.

Neural Fictitious Self-Play (Heinrich & Silver 2016, arXiv:1603.01121) with
PPO as the best-response (BR) learner. All seats share one BR network and one
average-policy network (the observation is relative to the player to act).

- Policy mixing: at the start of every game each player independently draws
  the policy it follows for the whole game, the BR with probability
  anticipatory_eta and the average policy otherwise.
- BR learning: PPO is on-policy, so the BR actor and critic train only on
  decisions made while following the BR. Returns follow each player's own
  decisions: a decision's reward is everything the player receives until its
  next decision or the end of the game, and it bootstraps from the value of
  that next decision (see per_player_gae). In Bluff most rewards are paid on
  other players' steps (unchallenged claims, challenges) and a player usually
  acts several times in a row (rank, size, cards).
- Truncation: a game cut off at the env horizon is not over, so it is not
  treated as terminal; the players' last decisions before the cut have no
  known outcome and are left out of the loss, and earlier decisions bootstrap
  from their values (as at the end of a rollout).
- Average policy: every BR decision (obs, action) is added to a reservoir
  buffer (Algorithm R) and the average policy is fit to it by cross-entropy.
  The average policy is NFSP's output strategy; the BR is a training device.
- Evaluation: the average policy and the BR each play compare_episodes games
  against copies of a baseline (random legal moves or a checkpoint), with the
  learner's seat rotating over the games; wins, losses and draws (games cut off
  at the horizon) are reported separately.

Checkpoints (save_final): see bluff_nfsp_common.save_checkpoints.
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
from bluffjax.environments.bluff.bluff import PLAY, BluffState
from bluffjax.examples.bluff.bluff_nfsp_common import (
    SLBufferState,
    draw_br_mode,
    eval_metrics,
    greedy_random_ties,
    init_sl_buffer,
    load_params,
    log_update,
    play_games,
    print_final_evals,
    reservoir_append,
    sample_logits,
    sample_random_legal,
    sample_sl_batch,
    save_checkpoints,
)
from bluffjax.networks.mlp import (
    ActorCriticDiscreteMLP,
    ActorDiscreteMLP,
    QNetworkDiscreteMLP,
)
from bluffjax.utils.jax_utils import pytree_norm
from bluffjax.utils.paths import register_resolvers
from bluffjax.utils.wandb_multilogger import WandbMultiLogger

LOGGER = None
# EVALS[seed] = [(env_steps, {metric: value}), ...]
EVALS: dict[int, list[tuple[int, dict[str, float]]]] = {}


class Transition(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    br_log_prob: FloatArray  # log-prob of the action under the BR (BR decisions)
    reward: FloatArray  # (num_agents,) env reward for every player
    done: BoolArray  # the game ended or was truncated at this step (env auto-resets)
    truncated: BoolArray  # the game was cut off at the horizon at this step
    value: FloatArray  # BR critic value of obs
    player_idx: IntArray
    phase: IntArray  # env phase of the decision
    is_br: BoolArray  # the acting player follows the BR in this game
    br_mode: BoolArray  # (num_agents,) which players follow the BR in this game
    game_return: FloatArray  # (num_agents,) return of the game ending at this step
    game_length: IntArray  # length of the game ending at this step


class RunnerState(NamedTuple):
    br_train_state: TrainState
    avg_train_state: TrainState
    sl_buffer: SLBufferState
    state: BluffState
    obs: FloatArray
    br_mode: BoolArray  # (num_envs, num_agents) policy drawn for the current game
    game_return: FloatArray  # (num_envs, num_agents) rewards so far this game
    update_step: IntArray
    rng: PRNGKeyArray


class PPOUpdateState(NamedTuple):
    train_state: TrainState
    transitions: Transition
    advantages: FloatArray
    targets: FloatArray
    train_mask: BoolArray
    rng: PRNGKeyArray


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
    truncated: BoolArray,
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
        dones: (T, N) the game ended or was truncated at step t (the env then
            auto-resets).
        truncated: (T, N) the game was cut off at the horizon at step t.
        players: (T, N) player acting at step t.
        last_value: (N,) critic value of the observation after the last step.
        last_player: (N,) player to act after the last step.

    The reward of player p's decision at step t is the sum of p's rewards from
    step t up to p's next decision t' (so rewards paid on other players' steps
    are included), and the decision bootstraps from values[t']. If the game
    ends first the decision is terminal. At the rollout cut-off only the player
    to act has a next decision (bootstrap from last_value). The other players'
    trailing decisions in an unfinished game have no known outcome yet: they
    are marked invalid, and those players' earlier decisions truncate their
    lambda trace there (as GAE does at a cut-off). A game truncated at the
    horizon is handled the same way, for every player: its last decision of
    each player is invalid.

    Returns:
        advantages (T, N), value targets (T, N), valid (T, N).
    """
    num_agents = rewards.shape[-1]
    last_onehot = jax.nn.one_hot(last_player, num_agents, dtype=jnp.bool_)
    zeros = jnp.zeros(last_onehot.shape, dtype=jnp.float32)
    # Per player, about its next decision (going backwards): value, advantage
    # (0 if the trace stops there), whether its outcome is known, whether the
    # game ends before it, and the reward collected since the decision being
    # processed.
    init = (
        jnp.where(last_onehot, last_value[:, None], 0.0),
        zeros,
        last_onehot,
        jnp.zeros_like(last_onehot),
        zeros,
    )

    def body(carry, x):
        next_value, next_adv, known, terminal, reward_acc = carry
        value, reward, done, trunc, player = x
        # Steps after a game-ending step belong to the next game. A game that
        # ended is terminal for everyone; a truncated one has unknown outcomes.
        game_over = done[:, None]
        cut = (done & trunc)[:, None]
        next_value = jnp.where(game_over, 0.0, next_value)
        next_adv = jnp.where(game_over, 0.0, next_adv)
        known = jnp.where(cut, False, known | game_over)
        terminal = jnp.where(cut, False, terminal | game_over)
        reward_acc = jnp.where(game_over, 0.0, reward_acc) + reward

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
        body, init, (values, rewards, dones, truncated, players), reverse=True
    )
    return advantages, advantages + values, valid


def make_train(config: dict) -> Callable[[PRNGKeyArray, int], RunnerState]:
    env = make("bluff", **config["env_kwargs"])
    num_agents = env.num_agents
    action_dim = env.action_dim
    config["batch_shuffle_dim"] = (
        config["num_steps_per_env_per_update"] * config["num_envs"]
    )
    template_obs = jnp.zeros((env.obs_dim,), dtype=jnp.float32)

    network = ActorCriticDiscreteMLP(
        action_dim=action_dim, hidden_dim=config["fc_dim_size"]
    )

    def act_actor_critic(params, obs, mask, rng):
        return sample_logits(network.apply(params, obs)[0], mask, rng)

    # Evaluation opponent: random legal moves or a saved network.
    compare_enabled = config["compare"]
    compare_against_random = config["compare_mode"] == "random"
    baseline_params = None
    if compare_against_random:

        def act_baseline(params, obs, mask, rng):
            return sample_random_legal(mask, rng)

    elif config["compare_mode"] == "checkpoint":
        compare_network_type = config["compare_network_type"]
        if compare_network_type == "actor_critic":
            baseline_network = ActorCriticDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )
            act_baseline = act_actor_critic
        elif compare_network_type == "q_network":
            baseline_network = QNetworkDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )

            def act_baseline(params, obs, mask, rng):
                return greedy_random_ties(
                    baseline_network.apply(params, obs), mask, rng
                )

        elif compare_network_type == "actor":
            baseline_network = ActorDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )

            def act_baseline(params, obs, mask, rng):
                return sample_logits(baseline_network.apply(params, obs), mask, rng)

        else:
            raise ValueError(
                "compare_network_type must be one of "
                "['actor_critic', 'q_network', 'actor'], "
                f"got '{compare_network_type}'"
            )
        if compare_enabled:
            baseline_params = load_params(
                config["compare_with"],
                baseline_network.init(jax.random.PRNGKey(0), template_obs),
            )
            print(
                f"Loaded {compare_network_type} baseline from {config['compare_with']}"
            )
    else:
        raise ValueError(
            f"compare_mode must be 'random' or 'checkpoint', got "
            f"'{config['compare_mode']}'"
        )
    baseline_name = "random" if compare_against_random else "baseline"

    def linear_decay(count: int) -> float:
        frac = 1.0 - count / config["num_gradient_steps"]
        return config["lr"] * frac

    def run_compare_eval(
        rng: PRNGKeyArray, avg_params: Any, br_params: Any
    ) -> dict[str, FloatArray]:
        rng_avg, rng_br = jax.random.split(rng)
        metrics = {}
        for name, params, rng_eval in (
            ("avg", avg_params, rng_avg),
            ("br", br_params, rng_br),
        ):
            result = play_games(
                env,
                act_actor_critic,
                params,
                act_baseline,
                baseline_params,
                config["compare_episodes"],
                rng_eval,
            )
            metrics.update(eval_metrics(f"{name}_vs_{baseline_name}", result))
        return metrics

    def train(rng: PRNGKeyArray, seed: int) -> RunnerState:
        rng, rng_reset, rng_br_init, rng_avg_init, rng_mode = jax.random.split(rng, 5)
        rng_resets = jax.random.split(rng_reset, config["num_envs"])
        state, obs = jax.vmap(env.reset, in_axes=(0))(rng_resets)

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
        sl_buffer = init_sl_buffer(
            config["sl_reservoir_capacity"], obs.shape[-1], action_dim
        )
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
                br_mode_s = runner_state_inner.br_mode
                rng_s, rng_mode, rng_action, rng_step = jax.random.split(
                    runner_state_inner.rng, 4
                )

                br_logits, value = network.apply(br_train_state_s.params, obs_s)
                avg_logits, _ = network.apply(avg_train_state_s.params, obs_s)
                action_mask = jax.vmap(env.get_avail_actions, in_axes=(0))(state_s)

                br_logits_masked = jnp.where(action_mask, br_logits, -jnp.inf)
                avg_logits_masked = jnp.where(action_mask, avg_logits, -jnp.inf)

                # NFSP: the acting player follows the policy it drew for this game.
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

                game_return = runner_state_inner.game_return + reward
                transition = Transition(
                    obs=obs_s,
                    action_mask=action_mask,
                    action=action,
                    br_log_prob=br_log_prob,
                    reward=reward,
                    done=done,
                    truncated=info["truncated"],
                    value=value,
                    player_idx=player_idx,
                    phase=state_s.phase,
                    is_br=is_br,
                    br_mode=br_mode_s,
                    game_return=game_return,
                    game_length=info["timestep"],
                )
                runner_state_inner = RunnerState(
                    br_train_state=br_train_state_s,
                    avg_train_state=avg_train_state_s,
                    sl_buffer=runner_state_inner.sl_buffer,
                    state=next_state,
                    obs=next_obs,
                    br_mode=draw_br_mode(
                        rng_mode, br_mode_s, done, config["anticipatory_eta"]
                    ),
                    game_return=jnp.where(done[:, None], 0.0, game_return),
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

            # Per-player GAE. PPO is on-policy: the BR actor and critic train
            # on BR decisions whose outcome is known.
            advantages, targets, valid = per_player_gae(
                transitions.value,
                transitions.reward,
                transitions.done,
                transitions.truncated,
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
                ppo_state: PPOUpdateState,
                unused: None,
            ) -> tuple[PPOUpdateState, dict[str, FloatArray]]:
                rng_s, rng_permute = jax.random.split(ppo_state.rng)
                batch = (
                    ppo_state.transitions.obs,
                    ppo_state.transitions.action_mask,
                    ppo_state.transitions.action,
                    ppo_state.transitions.br_log_prob,
                    ppo_state.transitions.value,
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
                    minibatch: tuple,
                ) -> tuple[TrainState, dict[str, FloatArray]]:
                    def loss(params, minibatch):
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
                        logits, value = network.apply(params, obs_mb)
                        logits_masked = jnp.where(action_mask_mb, logits, -jnp.inf)
                        pi = distrax.Categorical(logits=logits_masked)
                        log_prob = pi.log_prob(action_mb)

                        # Only BR decisions with a known outcome enter the loss.
                        mask = train_mask_mb.astype(jnp.float32)
                        mask_denom = jnp.maximum(mask.sum(), 1.0)

                        def masked_mean(x):
                            return (x * mask).sum() / mask_denom

                        mean_adv = masked_mean(advantages_mb)
                        std_adv = jnp.sqrt(masked_mean(jnp.square(advantages_mb - mean_adv)))
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
                        loss_actor = masked_mean(
                            -jnp.minimum(loss_actor_raw, loss_actor_clipped)
                        )
                        entropy = masked_mean(pi.entropy())

                        value_clipped = value_old_mb + jnp.clip(
                            value - value_old_mb,
                            -config["vf_clip"],
                            config["vf_clip"],
                        )
                        value_err = jnp.maximum(
                            jnp.square(value - targets_mb),
                            jnp.square(value_clipped - targets_mb),
                        )
                        value_loss = 0.5 * masked_mean(value_err)

                        total_loss = (
                            loss_actor
                            + config["vf_coef"] * value_loss
                            - config["ent_coef"] * entropy
                        )
                        return total_loss, {
                            "total_loss": total_loss,
                            "value_loss": value_loss,
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "ratio_mean": masked_mean(ratio),
                            "ratio_min": jnp.where(train_mask_mb, ratio, jnp.inf).min(),
                            "ratio_max": jnp.where(
                                train_mask_mb, ratio, -jnp.inf
                            ).max(),
                            "gae_mean": mean_adv,
                            "gae_std": std_adv,
                            "mean_target": masked_mean(targets_mb),
                            "value_pred_mean": masked_mean(value),
                            "kl_backward": masked_mean((ratio - 1) - logratio),
                            "kl_forward": masked_mean(ratio * logratio - (ratio - 1)),
                            "clip_frac": masked_mean(
                                (jnp.abs(ratio - 1) > config["clip_eps"]).astype(
                                    jnp.float32
                                )
                            ),
                            "ppo_sample_frac_minibatch": mask.mean(),
                        }

                    grad_fn = jax.value_and_grad(loss, has_aux=True)
                    (_, aux), grads = grad_fn(train_state.params, minibatch)
                    aux["grad_norm"] = pytree_norm(grads)
                    updated_train_state = train_state.apply_gradients(grads=grads)
                    return updated_train_state, aux

                final_train_state, batch_stats = lax.scan(
                    update_minibatch,
                    ppo_state.train_state,
                    minibatches,
                )
                ppo_state = ppo_state._replace(train_state=final_train_state, rng=rng_s)
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
                    obs_b, mask_b, action_b = sample_sl_batch(
                        rng_batch, sl_state.sl_buffer, config["sl_batch_size"]
                    )

                    def sl_loss(params) -> tuple[FloatArray, dict[str, FloatArray]]:
                        logits, _ = network.apply(params, obs_b)
                        logits_masked = jnp.where(mask_b, logits, -jnp.inf)
                        log_probs = jax.nn.log_softmax(logits_masked, axis=-1)
                        action_log_probs = jnp.take_along_axis(
                            log_probs, action_b[:, None], axis=-1
                        ).squeeze(-1)
                        ce_loss = -action_log_probs.mean()
                        acc = (jnp.argmax(logits_masked, axis=-1) == action_b).mean()
                        return ce_loss, {"sl_loss": ce_loss, "sl_acc": acc}

                    grad_fn = jax.value_and_grad(sl_loss, has_aux=True)
                    (_, aux), grads = grad_fn(train_state.params)
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
                return sl_state._replace(train_state=train_state, rng=rng_s), aux

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

            # Evaluate every compare_interval updates and after the last one.
            if compare_enabled:
                should_compare = (
                    runner_state.update_step % config["compare_interval"] == 0
                ) | (runner_state.update_step == config["num_update_steps"] - 1)
                eval_template = jax.eval_shape(
                    run_compare_eval,
                    rng_compare,
                    avg_train_state.params,
                    br_train_state.params,
                )
                compare_metrics = lax.cond(
                    should_compare,
                    lambda: run_compare_eval(
                        rng_compare, avg_train_state.params, br_train_state.params
                    ),
                    lambda: jax.tree_util.tree_map(
                        lambda x: jnp.zeros(x.shape, x.dtype), eval_template
                    ),
                )
            else:
                should_compare = jnp.array(False)
                compare_metrics = {}

            # Self-play statistics of the games that ended in this rollout,
            # split by the policy the player followed in that game.
            game_end = transitions.done[..., None]
            br_games = game_end & transitions.br_mode
            avg_games = game_end & ~transitions.br_mode
            games_completed = transitions.done.sum()
            play_steps = transitions.phase == PLAY
            metric = {
                "update_step": runner_state.update_step,
                "env_steps": (runner_state.update_step + 1)
                * config["num_envs"]
                * config["num_steps_per_env_per_update"],
                "games_completed": games_completed,
                "games_truncated_frac": transitions.truncated.sum()
                / jnp.maximum(games_completed, 1),
                "ep_length_avg": jnp.where(transitions.done, transitions.game_length, 0).sum()
                / jnp.maximum(games_completed, 1),
                "br_return_per_game": (transitions.game_return * br_games).sum()
                / jnp.maximum(br_games.sum(), 1),
                "avg_return_per_game": (transitions.game_return * avg_games).sum()
                / jnp.maximum(avg_games.sum(), 1),
                "br_action_frac_rollout": transitions.is_br.mean(),
                "ppo_sample_frac": train_mask.mean(),
                # cards picked that are not of the claimed rank (offset 0)
                "play_action_lie_frac": (play_steps & (transitions.action != 0)).sum()
                / jnp.maximum(play_steps.sum(), 1),
                "sl_buffer_size": sl_buffer.size.astype(jnp.float32),
                "sl_buffer_seen": sl_buffer.seen.astype(jnp.float32),
                "br_lr": (
                    linear_decay(br_train_state.step)
                    if config["anneal_lr"]
                    else jnp.asarray(config["lr"])
                ),
                "evaluated": should_compare,
            }
            metric.update(compare_metrics)
            metric.update(ppo_loss_info)
            metric.update(sl_loss_info)

            jax.experimental.io_callback(logging_callback, None, seed, metric)

            runner_state = runner_state._replace(
                br_train_state=br_train_state,
                avg_train_state=avg_train_state,
                sl_buffer=sl_buffer,
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
            game_return=jnp.zeros((config["num_envs"], num_agents)),
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


def logging_callback(seed_val, metric_dict) -> None:
    log_update(LOGGER, EVALS, seed_val, metric_dict)


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
        print_final_evals(EVALS)
        if config["save_final"]:
            run_dir = save_checkpoints(
                config,
                final_runner_state.avg_train_state.params,
                final_runner_state.br_train_state.params,
            )
            print(f"Saved checkpoints to {run_dir}")
    finally:
        if LOGGER is not None:
            LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    register_resolvers()
    main()
