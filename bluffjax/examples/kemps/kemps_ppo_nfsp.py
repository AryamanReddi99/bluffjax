"""
PPO-NFSP for Kemps.

Neural Fictitious Self-Play (Heinrich & Silver 2016, arXiv:1603.01121) with
PPO as the best-response (BR) learner. All players share one BR network and
one average-policy network (the observation is relative to the observer, so
a player's partner is always n/2 seats on). Kemps is a simultaneous-move game
(ParallelEnv): every player acts at every step, with one action that combines
a game action (swap, NOOP, KEMPS, STOP KEMPS) and the public signal; the
policy is a distribution over all these pairs, masked with
env.get_avail_actions (a game action is legal with every signal).

- Policy mixing: at the start of every hand each player independently draws
  the policy it follows for the whole hand, the BR with probability
  anticipatory_eta and the average policy otherwise (partners draw
  independently).
- BR learning: PPO is on-policy, so the BR actor and critic train only on
  decisions made while following the BR. Every player's trajectory is every
  step of its hands, and its decision at step t gets its own reward of that
  step and bootstraps from the value of its decision at t + 1, unless the
  hand ends at t (see parallel_nfsp_common.per_agent_gae). The env pays both
  partners their team's reward (+1 / -1 with two teams) when a declaration
  ends the hand, so each player is credited with its team's outcome.
- Truncation: a hand cut off at the env horizon is not over (the env sets
  absorbing on the cut-off too, so it is recognised by
  parallel_nfsp_common.kemps_horizon_cutoff). It is not treated as terminal:
  the decisions at the cut have no known outcome and are left out of the
  loss, and the decisions before bootstrap from their values. A real deal
  (everybody plays NOOP and the stock can't refill the centre) ends the hand
  with no score and is terminal.
- Average policy: every BR decision (obs, action) is added to a reservoir
  buffer (Algorithm R) and the average policy is fit to it by cross-entropy.
  The average policy is NFSP's output strategy; the BR is a training device.
- Evaluation: the average policy and the BR each play compare_episodes hands
  as both partners of one team against a baseline (uniformly random legal
  actions or a checkpoint) playing the other team(s), the learner's team
  alternating over the hands; wins, losses and draws (real deals and hands
  cut off at the horizon) are reported separately.
- Game metrics (bluffjax/metrics/kemps.py: KEMPS and STOP KEMPS precision,
  interception and own-call rates of four of a kind, signal mutual
  information, KEMPS latency) of every update's self-play rollout,
  credited to the players they describe: game/<metric> over the players
  following the average policy in that hand, game_br/ over those following
  the BR, each rate with its number of events (n_...). Their per-hand part
  is carried across updates in the runner state; a hand cut off at the
  horizon ends without a call. They need each step's player order, which
  env.step draws from its key and doesn't return (metrics.kemps.step_order).

Units: the env pays each team k / (k - 1) * (mean(letters) - its letter) at
the end of a hand, +1 / -1 with two teams; logged returns are in these units.

Run directory, {save_dir}/{job_type}_{timestamp}/, made when the run starts
(bluffjax.examples.run_utils.make_run_dir):
    config.yaml                  the resolved config, with num_update_steps,
                                 half_update_steps and run_dir
    results.jsonl                one record per update and seed: seed_index,
                                 wall_time and every metric sent to wandb,
                                 env_steps included (run_utils.read_results)
    half/avg_policy_{i}.msgpack  if save_checkpoints: the networks of seed i
    half/br_{i}.msgpack          after half_update_steps(num_update_steps)
    full/avg_policy_{i}.msgpack  updates and after all num_update_steps, from
    full/br_{i}.msgpack          the same run (the learning rate decays over
                                 the whole run)
Every checkpoint file holds the flax params, without a seed axis, of an
ActorCriticDiscreteMLP(action_dim=env.num_actions, hidden_dim=fc_dim_size):
input one player's observation (env.obs_from_state(state)[p], env.obs_dim
floats), output (logits over the actions game_action * comm_dim + signal,
value in env reward units). Mask the illegal swaps
(env.get_avail_actions(state)[p]) before the softmax; the average policy's
value head is untrained. Load a file with run_utils.load_params(path,
network.init(rng, obs)), or both networks of a seed with
parallel_nfsp_common.load_checkpoint.
"""

import datetime
import time
from typing import Callable, NamedTuple

import distrax
from flax.training.train_state import TrainState
import hydra
import jax
from jax import lax
from jax.experimental import io_callback
import jax.numpy as jnp
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
from bluffjax.environments.kemps.kemps import KempsState
from bluffjax.examples.parallel_nfsp_common import (
    Checkpoint,
    SLBufferState,
    draw_br_mode,
    eval_metrics,
    evaluate_kemps,
    greedy_random_ties,
    init_sl_buffer,
    kemps_horizon_cutoff,
    log_update,
    per_agent_gae,
    print_final_evals,
    reservoir_append,
    sample_logits,
    sample_random_legal,
    sample_sl_batch,
    save_checkpoints,
)
from bluffjax.examples.run_utils import (
    ResultsLog,
    half_update_steps,
    load_params,
    make_run_dir,
)
from bluffjax.metrics import Counts, make_metrics, masked_totals, prefixed
from bluffjax.metrics.kemps import step_order
from bluffjax.networks.mlp import (
    ActorCriticDiscreteMLP,
    ActorDiscreteMLP,
    QNetworkDiscreteMLP,
)
from bluffjax.utils.jax_utils import pytree_norm
from bluffjax.utils.paths import REPO_ROOT, register_resolvers

# results.jsonl of the run and its wandb runs (one per vmapped seed), made in main()
RESULTS_LOG: ResultsLog | None = None
# EVALS[seed] = [(env_steps, {metric: value}), ...]
EVALS: dict[int, list[tuple[int, dict[str, float]]]] = {}


class Transition(NamedTuple):
    """One env step of every env; agent fields are (num_envs, num_agents, ...)."""

    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    br_log_prob: FloatArray  # log-prob of the action under the BR (BR decisions)
    reward: FloatArray  # env reward of every agent
    done: BoolArray  # (num_envs,) the hand ended or was cut off (env auto-resets)
    truncated: BoolArray  # the agent's hand was cut off at the horizon
    value: FloatArray  # BR critic value of obs
    is_br: BoolArray  # the agent follows the BR in this hand
    game_return: FloatArray  # env rewards of the hand so far, this step included
    game_length: IntArray  # (num_envs,) steps of the hand so far, this one included
    game_counts: Counts  # game metrics of the step, credited per agent


class RunnerState(NamedTuple):
    br_train_state: TrainState
    avg_train_state: TrainState
    sl_buffer: SLBufferState
    state: KempsState
    obs: FloatArray
    br_mode: BoolArray  # (num_envs, num_agents) policy drawn for the current hand
    game_return: FloatArray  # (num_envs, num_agents) rewards so far this hand
    game_length: IntArray  # (num_envs,) steps so far this hand
    metric_carry: Any  # per-env carry of the game metrics (current hand)
    half: Checkpoint  # networks after half_update_steps updates
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


def make_train(config: dict) -> Callable[..., RunnerState]:
    env = make("kemps", **config["env_kwargs"])
    num_agents = env.num_agents
    action_dim = env.num_actions
    num_envs = config["num_envs"]
    num_steps = config["num_steps_per_env_per_update"]
    # every agent makes a decision at every step
    config["batch_shuffle_dim"] = num_steps * num_envs * num_agents
    if config["batch_shuffle_dim"] % config["num_minibatches"]:
        raise ValueError("num_minibatches must divide the number of decisions per update")
    num_half = half_update_steps(config["num_update_steps"])
    gm = make_metrics("kemps", env)  # bluffjax/metrics/kemps.py
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
            baseline_network = network
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
        # {stat}_vs_{opponent} for the average policy, {stat}_br_vs_{opponent}
        # for the BR
        for prefix, params, rng_eval in (
            ("", avg_params, rng_avg),
            ("br_", br_params, rng_br),
        ):
            result = evaluate_kemps(
                env,
                act_actor_critic,
                params,
                act_baseline,
                baseline_params,
                config["compare_episodes"],
                rng_eval,
            )
            metrics.update(eval_metrics(f"{prefix}vs_{baseline_name}", result))
        return metrics

    def train(
        rng: PRNGKeyArray, seed: int, num_updates: int | None = None
    ) -> RunnerState:
        """Runs num_updates updates (default: all of them) of one seed. The
        schedules always span config["num_update_steps"] updates."""
        rng, rng_reset, rng_br_init, rng_avg_init, rng_mode = jax.random.split(rng, 5)
        state, obs = jax.vmap(env.reset)(jax.random.split(rng_reset, num_envs))

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
            config["sl_reservoir_capacity"], env.obs_dim, action_dim
        )
        br_mode = jax.random.bernoulli(
            rng_mode, p=config["anticipatory_eta"], shape=(num_envs, num_agents)
        )

        def update_step(
            runner_state: RunnerState,
            unused: None,
        ) -> tuple[RunnerState, None]:
            def step(
                runner_state_inner: RunnerState,
                unused: None,
            ) -> tuple[RunnerState, Transition]:
                state_s = runner_state_inner.state
                obs_s = runner_state_inner.obs  # (num_envs, num_agents, obs_dim)
                br_mode_s = runner_state_inner.br_mode
                rng_s, rng_mode, rng_action, rng_step = jax.random.split(
                    runner_state_inner.rng, 4
                )

                br_logits, value = network.apply(
                    runner_state_inner.br_train_state.params, obs_s
                )
                avg_logits, _ = network.apply(
                    runner_state_inner.avg_train_state.params, obs_s
                )
                action_mask = jax.vmap(env.get_avail_actions)(state_s)
                br_logits_masked = jnp.where(action_mask, br_logits, -jnp.inf)
                avg_logits_masked = jnp.where(action_mask, avg_logits, -jnp.inf)

                # NFSP: every agent follows the policy it drew for this hand.
                is_br = br_mode_s
                acting_logits = jnp.where(
                    is_br[..., None], br_logits_masked, avg_logits_masked
                )
                action = distrax.Categorical(logits=acting_logits).sample(
                    seed=rng_action
                )
                br_log_prob = jnp.where(
                    is_br,
                    distrax.Categorical(logits=br_logits_masked).log_prob(action),
                    0.0,
                )

                step_keys = jax.random.split(rng_step, num_envs)
                next_state, next_obs, reward, absorbing, done, info = jax.vmap(
                    env.step
                )(step_keys, state_s, action)
                # Cut off at the horizon, not over: done without absorbing,
                # or (as this env sets absorbing then too) done without a
                # declaration or a real deal.
                cutoff = kemps_horizon_cutoff(env, state_s, action, done)
                truncated = done[:, None] & (~absorbing | cutoff[:, None])

                # Game metrics of this step. They need the step's player order
                # (whose declaration counts), which env.step draws from its key
                # but doesn't return.
                order = jax.vmap(lambda key: step_order(env, key))(step_keys)
                metric_carry, game_counts = jax.vmap(gm.step)(
                    runner_state_inner.metric_carry,
                    state_s,
                    action,
                    next_state,
                    reward,
                    done,
                    {**info, "order": order},
                )

                game_return = runner_state_inner.game_return + reward
                game_length = runner_state_inner.game_length + 1
                transition = Transition(
                    obs=obs_s,
                    action_mask=action_mask,
                    action=action,
                    br_log_prob=br_log_prob,
                    reward=reward,
                    done=done,
                    truncated=truncated,
                    value=value,
                    is_br=is_br,
                    game_return=game_return,
                    game_length=game_length,
                    game_counts=game_counts,
                )
                runner_state_inner = runner_state_inner._replace(
                    state=next_state,
                    obs=next_obs,
                    br_mode=draw_br_mode(
                        rng_mode, br_mode_s, done, config["anticipatory_eta"]
                    ),
                    game_return=jnp.where(done[:, None], 0.0, game_return),
                    game_length=jnp.where(done, 0, game_length),
                    metric_carry=metric_carry,
                    rng=rng_s,
                )
                return runner_state_inner, transition

            runner_state, transitions = lax.scan(step, runner_state, None, num_steps)

            br_train_state = runner_state.br_train_state
            avg_train_state = runner_state.avg_train_state
            sl_buffer = runner_state.sl_buffer
            rng, rng_sl_append, rng_ppo_update, rng_sl_update, rng_compare = (
                jax.random.split(runner_state.rng, 5)
            )

            _, last_val = network.apply(br_train_state.params, runner_state.obs)

            # Per-agent GAE in env reward units. PPO is on-policy: the BR
            # actor and critic train on BR decisions whose outcome is known.
            advantages, targets, valid = per_agent_gae(
                transitions.value,
                transitions.reward,
                jnp.broadcast_to(
                    transitions.done[..., None], transitions.reward.shape
                ),
                transitions.truncated,
                last_val,
                config["gamma"],
                config["gae_lambda"],
            )
            train_mask = valid & transitions.is_br

            # NFSP: every BR decision goes to the reservoir.
            sl_buffer = reservoir_append(
                sl_buffer,
                transitions.obs.reshape(-1, env.obs_dim),
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
                # (steps, envs, agents, ...) -> (steps * envs * agents, ...)
                batch_reshaped = jax.tree_util.tree_map(
                    lambda x: x.reshape((-1,) + x.shape[3:]), batch
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
                        std_adv = jnp.sqrt(
                            masked_mean(jnp.square(advantages_mb - mean_adv))
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

            # Self-play statistics of the hands that ended in this rollout,
            # per agent and split by the policy it followed in that hand.
            episodes_completed = transitions.done.sum()
            agent_game_end = jnp.broadcast_to(
                transitions.done[..., None], transitions.is_br.shape
            )
            br_games = agent_game_end & transitions.is_br
            avg_games = agent_game_end & ~transitions.is_br
            metric = {
                "update_step": runner_state.update_step,
                "env_steps": (runner_state.update_step + 1) * num_envs * num_steps,
                "episodes_completed": episodes_completed,
                "episodes_truncated_frac": (
                    transitions.done & transitions.truncated.any(axis=-1)
                ).sum()
                / jnp.maximum(episodes_completed, 1),
                "ep_length_avg": jnp.where(
                    transitions.done, transitions.game_length, 0
                ).sum()
                / jnp.maximum(episodes_completed, 1),
                "return_per_episode": (transitions.game_return * avg_games).sum()
                / jnp.maximum(avg_games.sum(), 1),
                "return_per_episode_br": (transitions.game_return * br_games).sum()
                / jnp.maximum(br_games.sum(), 1),
                "br_action_frac_rollout": transitions.is_br.mean(),
                "ppo_sample_frac": train_mask.mean(),
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

            # Game metrics of this rollout, credited to the players they
            # describe: game/ for the players following the average policy
            # (NFSP's output strategy) in that hand, game_br/ for those
            # following the BR.
            for prefix, mask in (
                ("game", ~transitions.is_br),
                ("game_br", transitions.is_br),
            ):
                totals = masked_totals(transitions.game_counts, mask)
                metric.update(prefixed(prefix, gm.summarize(totals)))

            io_callback(logging_callback, None, seed, metric)

            # Half-budget checkpoint: the networks after num_half updates.
            take_half = runner_state.update_step + 1 == num_half
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
            game_return=jnp.zeros((num_envs, num_agents), dtype=jnp.float32),
            game_length=jnp.zeros((num_envs,), dtype=jnp.int32),
            metric_carry=jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (num_envs,) + x.shape), gm.init_carry()
            ),
            half=Checkpoint(avg_params, br_params),  # replaced after num_half updates
            update_step=jnp.array(0, dtype=jnp.int32),
            rng=rng,
        )
        final_runner_state, _ = lax.scan(
            update_step,
            initial_runner_state,
            None,
            config["num_update_steps"] if num_updates is None else num_updates,
        )
        return final_runner_state

    return train


def logging_callback(seed_val, metric_dict) -> None:
    log_update(RESULTS_LOG, EVALS, seed_val, metric_dict)


@hydra.main(version_base=None, config_path="./", config_name="config_ppo_nfsp")
def main(config: dict) -> None:
    global RESULTS_LOG
    wandb_runs = []
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
        # results.jsonl fills while the run trains.
        run_dir = make_run_dir(config)
        job_type = f"{config['job_type']}_{config['env_name']}"
        group = f"{config['env_name']}" + datetime.datetime.now().strftime(
            "_%Y-%m-%d_%H-%M-%S"
        )
        wandb_runs = [
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
        RESULTS_LOG = ResultsLog(run_dir, wandb_runs)
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
            f"{run_time:.1f} s ({run_time / env_steps * 1e6:.2f} s per 1e6 env "
            f"steps, all seeds in parallel)"
        )
        print_final_evals(EVALS)
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
        for run in wandb_runs:
            run.finish()
        print("Finished.")


if __name__ == "__main__":
    register_resolvers()
    main()
