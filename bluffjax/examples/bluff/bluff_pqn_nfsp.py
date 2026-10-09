"""
PQN-NFSP for Bluff.

Neural Fictitious Self-Play (Heinrich & Silver 2016, arXiv:1603.01121) with
PQN (Gallici et al. 2024, "Simplifying Deep Temporal Difference Learning") as
the best-response (BR) learner in place of DQN. All seats share one Q-network
and one average-policy network (the observation is relative to the player to
act).

- Policy mixing: at the start of every game each player independently draws
  the policy it follows for the whole game, the epsilon-greedy BR with
  probability anticipatory_eta and the average policy otherwise.
- BR learning: as in NFSP the Q-network learns from all of a player's
  decisions, whichever policy it followed. Targets follow each player's own
  decisions (a decision's reward is everything the player receives until its
  next decision or the end of the game, including rewards paid on other
  players' steps) and bootstrap from the max over legal actions of Q at that
  next decision, which in Bluff is usually the same player's next sub-step.
  The lambda-trace runs through decisions taken by the epsilon-greedy BR
  (Peng's Q(lambda), as in PQN) and is cut at average-policy decisions, so
  those give one-step Q-learning targets (see per_player_q_lambda_targets).
- Truncation: a game cut off at the env horizon is not over, so it is not
  treated as terminal; the players' last decisions before the cut have no
  known outcome and are left out of the loss, and earlier decisions bootstrap
  from Q there (as at the end of a rollout).
- Average policy: every BR decision (obs, action) is added to a reservoir
  buffer (Algorithm R) and the average policy is fit to it by cross-entropy.
  The average policy is NFSP's output strategy; the BR is a training device.
- Evaluation: the average policy (sampled) and the greedy BR each play
  compare_steps games against copies of a baseline (random legal moves or a
  checkpoint), with the learner's seat rotating over the games; wins, losses
  and draws (games cut off at the horizon) are reported separately.

Checkpoints (save_final): see bluff_nfsp_common.save_checkpoints.
"""

import datetime
import time
from typing import Any, Callable, NamedTuple

from flax.training.train_state import TrainState
import hydra
import jax
from jax import lax
import jax.numpy as jnp
from omegaconf import OmegaConf
import optax

from bluffjax.utils.typing import (
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
    reward: FloatArray  # (num_agents,) env reward for every player
    done: BoolArray  # the game ended or was truncated at this step (env auto-resets)
    truncated: BoolArray  # the game was cut off at the horizon at this step
    q_max: FloatArray  # max over legal actions of Q(obs)
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


class PQNUpdateState(NamedTuple):
    train_state: TrainState
    transitions: Transition
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


def per_player_q_lambda_targets(
    q_max: FloatArray,
    rewards: FloatArray,
    dones: BoolArray,
    truncated: BoolArray,
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
        dones: (T, N) the game ended or was truncated at step t (the env then
            auto-resets).
        truncated: (T, N) the game was cut off at the horizon at step t.
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
    without the trace. A game truncated at the horizon is handled the same way,
    for every player: its last decision of each player is invalid.

    Returns:
        targets (T, N), valid (T, N).
    """
    num_agents = rewards.shape[-1]
    last_onehot = jax.nn.one_hot(last_player, num_agents, dtype=jnp.bool_)
    zeros = jnp.zeros(last_onehot.shape, dtype=jnp.float32)
    last_q = jnp.where(last_onehot, last_q_max[:, None], 0.0)
    # Per player, about its next decision (going backwards): q_max, return
    # (= q_max if the trace stops there), trace coefficient, whether its
    # outcome is known, whether the game ends before it, and the reward
    # collected since the decision being processed.
    init = (last_q, last_q, zeros, last_onehot, jnp.zeros_like(last_onehot), zeros)

    def body(carry, x):
        next_q, next_return, next_trace, known, terminal, reward_acc = carry
        q_max_t, reward, done, trunc, player, trace = x
        # Steps after a game-ending step belong to the next game. A game that
        # ended is terminal for everyone; a truncated one has unknown outcomes.
        game_over = done[:, None]
        cut = (done & trunc)[:, None]
        known = jnp.where(cut, False, known | game_over)
        terminal = jnp.where(cut, False, terminal | game_over)
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
        body, init, (q_max, rewards, dones, truncated, players, traces), reverse=True
    )
    return targets, valid


def make_train(config: dict) -> Callable[[PRNGKeyArray, int], RunnerState]:
    env = make("bluff", **config["env_kwargs"])
    num_agents = env.num_agents
    action_dim = env.action_dim
    config["batch_shuffle_dim"] = (
        config["num_steps_per_env_per_update"] * config["num_envs"]
    )
    template_obs = jnp.zeros((env.obs_dim,), dtype=jnp.float32)

    br_network = QNetworkDiscreteMLP(
        action_dim=action_dim, hidden_dim=config["fc_dim_size"]
    )
    avg_network = ActorDiscreteMLP(
        action_dim=action_dim, hidden_dim=config["fc_dim_size"]
    )

    def act_avg(params, obs, mask, rng):
        return sample_logits(avg_network.apply(params, obs), mask, rng)

    def act_br_greedy(params, obs, mask, rng):
        return greedy_random_ties(br_network.apply(params, obs), mask, rng)

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

            def act_baseline(params, obs, mask, rng):
                return sample_logits(baseline_network.apply(params, obs)[0], mask, rng)

        elif compare_network_type == "q_network":
            baseline_network = br_network
            act_baseline = act_br_greedy
        elif compare_network_type == "actor":
            baseline_network = avg_network
            act_baseline = act_avg
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

    def epsilon_schedule(update_step: FloatArray) -> FloatArray:
        decay_steps = max(
            config["exploration_fraction"] * config["num_update_steps"], 1
        )
        frac = jnp.minimum(1.0, update_step.astype(jnp.float32) / decay_steps)
        return config["start_e"] + frac * (config["end_e"] - config["start_e"])

    def run_compare_eval(
        rng: PRNGKeyArray, avg_params: Any, br_params: Any
    ) -> dict[str, FloatArray]:
        rng_avg, rng_br = jax.random.split(rng)
        metrics = {}
        for name, act, params, rng_eval in (
            ("avg", act_avg, avg_params, rng_avg),
            ("br", act_br_greedy, br_params, rng_br),
        ):
            result = play_games(
                env,
                act,
                params,
                act_baseline,
                baseline_params,
                config["compare_steps"],
                rng_eval,
            )
            metrics.update(eval_metrics(f"{name}_vs_{baseline_name}", result))
        return metrics

    def train(rng: PRNGKeyArray, seed: int) -> RunnerState:
        rng, rng_reset, rng_br_init, rng_avg_init, rng_mode = jax.random.split(rng, 5)
        rng_resets = jax.random.split(rng_reset, config["num_envs"])
        state, obs = jax.vmap(env.reset, in_axes=(0))(rng_resets)

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
                (
                    rng_s,
                    rng_mode,
                    rng_explore,
                    rng_random,
                    rng_avg,
                    rng_step,
                ) = jax.random.split(runner_state_inner.rng, 6)
                action_mask = jax.vmap(env.get_avail_actions, in_axes=(0))(state_s)

                q_vals = br_network.apply(br_train_state_s.params, obs_s)
                q_vals_masked = jnp.where(action_mask, q_vals, -jnp.inf)
                greedy_action = jnp.argmax(q_vals_masked, axis=-1)
                q_max = jnp.max(q_vals_masked, axis=-1)

                # epsilon-greedy BR over the legal actions
                eps = epsilon_schedule(
                    jnp.asarray(runner_state_inner.update_step, dtype=jnp.float32)
                )
                explore = jax.random.uniform(rng_explore, (config["num_envs"],)) < eps
                random_action = jax.random.categorical(
                    rng_random, jnp.where(action_mask, 0.0, -jnp.inf)
                )
                br_action = jnp.where(explore, random_action, greedy_action)

                avg_logits = avg_network.apply(avg_train_state_s.params, obs_s)
                avg_action = jax.random.categorical(
                    rng_avg, jnp.where(action_mask, avg_logits, -jnp.inf)
                )

                # NFSP: the acting player follows the policy it drew for this game.
                player_idx = state_s.current_player_idx
                is_br = _pick(br_mode_s, player_idx)
                action = jnp.where(is_br, br_action, avg_action)

                rng_steps = jax.random.split(rng_step, config["num_envs"])
                next_state, next_obs, reward, absorbing, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_steps, state_s, action)

                game_return = runner_state_inner.game_return + reward
                transition = Transition(
                    obs=obs_s,
                    action_mask=action_mask,
                    action=action,
                    reward=reward,
                    done=done,
                    truncated=info["truncated"],
                    q_max=q_max,
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
            rng, rng_sl_append, rng_pqn_update, rng_sl_update, rng_compare = (
                jax.random.split(runner_state.rng, 5)
            )

            last_mask = jax.vmap(env.get_avail_actions, in_axes=(0))(last_state)
            last_q_vals = br_network.apply(br_train_state.params, last_obs)
            last_q_max = jnp.max(jnp.where(last_mask, last_q_vals, -jnp.inf), axis=-1)

            # Per-player Q(lambda) targets. Q-learning is off-policy, so (as in
            # NFSP) every decision with a known outcome trains the Q-network;
            # the trace is cut at average-policy decisions.
            targets, valid = per_player_q_lambda_targets(
                transitions.q_max,
                transitions.reward,
                transitions.done,
                transitions.truncated,
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
                transitions.action_mask.reshape(-1, action_dim),
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
                            q_vals, action_mb[..., None], axis=-1
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
                    return train_state.apply_gradients(grads=grads), aux

                final_train_state, batch_stats = lax.scan(
                    update_minibatch,
                    pqn_state.train_state,
                    minibatches,
                )
                pqn_state = pqn_state._replace(train_state=final_train_state, rng=rng_s)
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
                    obs_b, mask_b, action_b = sample_sl_batch(
                        rng_batch, sl_state.sl_buffer, config["sl_batch_size"]
                    )

                    def sl_loss(params) -> tuple[FloatArray, dict[str, FloatArray]]:
                        logits = avg_network.apply(params, obs_b)
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
                "q_sample_frac": train_mask.mean(),
                # cards picked that are not of the claimed rank (offset 0)
                "play_action_lie_frac": (play_steps & (transitions.action != 0)).sum()
                / jnp.maximum(play_steps.sum(), 1),
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
                "evaluated": should_compare,
            }
            metric.update(compare_metrics)
            metric.update(pqn_loss_info)
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
