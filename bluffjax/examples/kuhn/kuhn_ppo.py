"""
PPO in self-play for Kuhn Poker.

One actor-critic network plays both seats (the observation encodes the
position, so it is the acting player's information state; the env draws the
first player at random).

- Returns follow each player's own decisions: a decision's reward is everything
  the player receives until its next decision or the end of the hand (which may
  come on the opponent's step), and it bootstraps from the value of that next
  decision (see per_player_gae).
- Exact evaluation, every eval_interval updates and after the last update: the
  exploitability (kuhn_exploitability), the expected return per hand against
  uniformly random play averaged over both seats
  (kuhn_exploitability.value_vs_random) and the Kuhn policy metrics
  (bluffjax.metrics.kuhn_poker) of the policy, computed from its table at the
  12 infosets.
- Evaluation against random (compare), every compare_interval updates and
  after the last update: the policy (sampled from softmax(logits), the policy
  of the exact evaluation) plays compare_episodes complete hands against
  uniformly random legal play, its position alternating between first and
  second to act (bluffjax.examples.kuhn_leduc_eval). Its own random stream:
  training doesn't depend on it.

Units: the env pays chips per hand (at most 2), used as is for training.

Logging: main() creates the run directory {save_dir}/{job_type}_{timestamp}/
(run_utils.make_run_dir, with config.yaml) before training. Every update logs
one record per seed, to the seed's wandb run and to {run_dir}/results.jsonl
(with seed_index and wall_time): env_steps, update_step and the training
statistics. Every compare_interval updates and after the last one it also
holds the policy's evaluation against random, name vs_random:
win_rate_vs_random, loss_rate_vs_random, draw_rate_vs_random,
win_rate_decided_vs_random (wins among the hands won or lost),
avg_return_vs_random (chips per hand) and avg_episode_length_vs_random (env
steps per hand), over compare_episodes hands. Every eval_interval updates and
after the last one the record also holds the policy's exploitability,
value_vs_random (the exact expectation of avg_return_vs_random) and its policy
metrics, game/<name> (names and definitions in bluffjax.metrics.kuhn_poker;
one policy plays both seats).

Checkpoints (save_checkpoints), for every seed i = 0..num_seeds-1 (the wandb
run "{seed}_{i}"):
    {run_dir}/half/policy_{i}.msgpack  after half_update_steps(num_update_steps)
    {run_dir}/full/policy_{i}.msgpack  after all num_update_steps updates
Both come from the same run (the learning-rate schedule spans the whole run;
config.yaml records num_update_steps and half_update_steps). Each file holds
the flax params of the ActorCriticDiscreteMLP(action_dim=2,
hidden_dim=fc_dim_size), written with flax.serialization.to_bytes. It maps the
acting player's observation (KuhnPoker.obs_from_state, 9 floats; the rows of
kuhn_exploitability.INFOSET_OBS are the 12 infosets) to (logits over [pass,
bet], value in chips); the policy is softmax(logits)
(kuhn_exploitability.policy_array_from_network gives it at every infoset).
Restore one with run_utils.load_params(path, network.init(rng, jnp.zeros(9))).
"""

import datetime
import time
from typing import Any, Callable, NamedTuple

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

from bluffjax.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray
from bluffjax import make
from bluffjax.environments.kuhn_poker.kuhn_poker import KuhnState
from bluffjax.examples.kuhn_leduc_eval import (
    eval_key,
    format_returns,
    is_sampled_metric,
    maybe_vs_random_metrics,
    softmax_policy,
)
from bluffjax.examples.parallel_nfsp_common import print_final_evals
from bluffjax.examples.run_utils import (
    ResultsLog,
    half_update_steps,
    make_run_dir,
    save_seed_params,
)
from bluffjax.metrics import prefixed
from bluffjax.metrics.kuhn_poker import policy_metrics
from bluffjax.networks.mlp import ActorCriticDiscreteMLP
from bluffjax.utils.game_utils.kuhn_exploitability import (
    exploitability_from_policy_array,
    policy_array_from_network,
    value_vs_random_from_policy_array,
)
from bluffjax.utils.jax_utils import pytree_norm
from bluffjax.utils.paths import REPO_ROOT, register_resolvers

WANDB_RUNS: list = []  # one wandb run per vmapped seed, created in main()
RESULTS_LOG: ResultsLog | None = None  # results.jsonl and wandb, set in main()
# EXPLOITABILITY[seed] = [(env_steps, {policy name: exploitability}), ...]
EXPLOITABILITY: dict[int, list[tuple[int, dict[str, float]]]] = {}
# VALUE_VS_RANDOM[seed] = [(env_steps, {policy name: exact value}), ...]
VALUE_VS_RANDOM: dict[int, list[tuple[int, dict[str, float]]]] = {}
# EVALUATIONS[seed] = [(env_steps, {metric: value}), ...], against random
EVALUATIONS: dict[int, list[tuple[int, dict[str, float]]]] = {}


class Transition(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    log_prob: FloatArray
    reward: FloatArray  # (num_agents,) env reward for every player, chips
    done: BoolArray  # the hand ended at this step (the env auto-resets)
    value: FloatArray  # critic value of obs
    player_idx: IntArray
    start_player_idx: IntArray  # the hand's first player
    info: dict[str, Any]


class RunnerState(NamedTuple):
    train_state: TrainState
    state: KuhnState
    obs: FloatArray
    done: BoolArray
    half_params: Any  # network after half_update_steps(num_update_steps) updates
    update_step: IntArray
    rng: PRNGKeyArray


class UpdateState(NamedTuple):
    train_state: TrainState
    transitions: Transition
    advantages: FloatArray
    targets: FloatArray
    train_mask: BoolArray
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


def make_train(config: dict) -> Callable[..., RunnerState]:
    env = make("kuhn_poker", **config["env_kwargs"])
    config["batch_shuffle_dim"] = (
        config["num_steps_per_env_per_update"] * config["num_envs"]
    )
    network = ActorCriticDiscreteMLP(
        action_dim=env.num_actions, hidden_dim=config["fc_dim_size"]
    )
    eval_interval = config["eval_interval"]
    compare_enabled = config["compare"]
    compare_interval = config["compare_interval"]
    compare_episodes = config["compare_episodes"]
    last_update = config["num_update_steps"] - 1
    num_half = half_update_steps(config["num_update_steps"])

    def linear_decay(count: int) -> float:
        frac = 1.0 - count / config["num_gradient_steps"]
        return config["lr"] * frac

    def masked_mean(x: FloatArray, mask: FloatArray) -> FloatArray:
        denom = jnp.maximum(mask.sum(), 1.0)
        return (x * mask).sum() / denom

    def logging_callback(seed_val, metric_dict, policy_arrays, evaluated) -> None:
        """Logs the metrics, those of the evaluation against random only when
        it ran; computes the exact exploitability, value against random and
        policy metrics when they are due."""
        seed_i = int(seed_val)
        metrics = {k: np.asarray(v) for k, v in metric_dict.items()}
        update = int(metrics["update_step"])
        env_steps = int(metrics["env_steps"])
        if bool(evaluated):
            evals = {k: float(v) for k, v in metrics.items() if is_sampled_metric(k)}
            EVALUATIONS.setdefault(seed_i, []).append((env_steps, evals))
        else:
            metrics = {k: v for k, v in metrics.items() if not is_sampled_metric(k)}
        if update % eval_interval == 0 or update == last_update:
            policies = {name: np.asarray(arr) for name, arr in policy_arrays.items()}
            expl = {
                name: exploitability_from_policy_array(arr)
                for name, arr in policies.items()
            }
            value = {
                name: value_vs_random_from_policy_array(arr)
                for name, arr in policies.items()
            }
            EXPLOITABILITY.setdefault(seed_i, []).append((env_steps, expl))
            VALUE_VS_RANDOM.setdefault(seed_i, []).append((env_steps, value))
            metrics.update(prefixed("game", policy_metrics(policies["policy"])))
            metrics["exploitability"] = expl["policy"]
            metrics["value_vs_random"] = value["policy"]
            extra = f"; value vs random {value['policy']:+.4f}"
            if bool(evaluated):
                extra += (
                    f"; return per hand in {compare_episodes} hands vs random: "
                    f"{format_returns(EVALUATIONS[seed_i][-1][1])}"
                )
            print(
                f"seed {seed_i} update {update + 1}/{last_update + 1} "
                f"({env_steps} env steps): exploitability {expl['policy']:.4f}{extra}",
                flush=True,
            )
        RESULTS_LOG.log(metrics, seed_i)

    def train(
        rng: PRNGKeyArray, seed: int, num_updates: int | None = None
    ) -> RunnerState:
        """Runs num_updates updates (default: all num_update_steps) of one
        seed; the schedules always span num_update_steps."""
        rng, rng_reset, rng_network_init = jax.random.split(rng, 3)
        rng_resets = jax.random.split(rng_reset, config["num_envs"])
        state, obs = jax.vmap(env.reset, in_axes=(0))(rng_resets)
        network_params = network.init(rng_network_init, obs)
        lr = linear_decay if config["anneal_lr"] else config["lr"]
        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=lr, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply, params=network_params, tx=tx
        )

        def update_step(
            runner_state: RunnerState, unused: None
        ) -> tuple[RunnerState, None]:
            def step(
                runner_state: RunnerState, unused: None
            ) -> tuple[RunnerState, Transition]:
                train_state = runner_state.train_state
                state = runner_state.state
                obs = runner_state.obs
                rng, rng_action, rng_step = jax.random.split(runner_state.rng, 3)

                action_logits, value = network.apply(train_state.params, obs)
                action_mask = jax.vmap(env.get_avail_actions, in_axes=(0))(state)
                logits_masked = jnp.where(action_mask, action_logits, -jnp.inf)
                pi = distrax.Categorical(logits=logits_masked)
                action = pi.sample(seed=rng_action)

                rng_steps = jax.random.split(rng_step, config["num_envs"])
                next_state, next_obs, reward, _, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_steps, state, action)

                transition = Transition(
                    obs=obs,
                    action_mask=action_mask,
                    action=action,
                    log_prob=pi.log_prob(action),
                    reward=reward,
                    done=done,
                    value=value,
                    player_idx=state.current_player_idx,
                    start_player_idx=state.start_player_idx,
                    info=info,
                )
                runner_state = runner_state._replace(
                    state=next_state, obs=next_obs, done=done, rng=rng
                )
                return runner_state, transition

            runner_state, transitions = lax.scan(
                step, runner_state, None, config["num_steps_per_env_per_update"]
            )

            train_state = runner_state.train_state
            last_state = runner_state.state
            rng, rng_update = jax.random.split(runner_state.rng)

            _, last_val = network.apply(train_state.params, runner_state.obs)

            # Per-player GAE; every decision with a known outcome trains the
            # actor and the critic.
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
            train_mask = valid

            def update_epoch(
                update_state: UpdateState, unused: None
            ) -> tuple[UpdateState, dict[str, FloatArray]]:
                rng, rng_permute = jax.random.split(update_state.rng)
                batch = (
                    update_state.transitions,
                    update_state.advantages,
                    update_state.targets,
                    update_state.train_mask,
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

                        # Only decisions with a known outcome enter the loss.
                        mask = train_mask_mb.astype(jnp.float32)
                        mask_denom = jnp.maximum(mask.sum(), 1.0)

                        mean_adv = masked_mean(advantages_mb, mask)
                        std_adv = jnp.sqrt(
                            masked_mean(jnp.square(advantages_mb - mean_adv), mask)
                        )
                        gae_normalized = (advantages_mb - mean_adv) / (std_adv + 1e-8)

                        logratio = jnp.where(
                            train_mask_mb, log_prob - transitions_mb.log_prob, 0.0
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

                        # critic loss (clipped to prevent value overshooting)
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
                        }

                    (_, aux), grads = jax.value_and_grad(loss, has_aux=True)(
                        train_state.params
                    )
                    aux["grad_norm"] = pytree_norm(grads)
                    return train_state.apply_gradients(grads=grads), aux

                final_train_state, batch_stats = lax.scan(
                    update_minibatch, update_state.train_state, minibatches
                )
                update_state = UpdateState(
                    train_state=final_train_state,
                    transitions=update_state.transitions,
                    advantages=update_state.advantages,
                    targets=update_state.targets,
                    train_mask=update_state.train_mask,
                    rng=rng,
                )
                return update_state, batch_stats

            update_state = UpdateState(
                train_state=train_state,
                transitions=transitions,
                advantages=advantages,
                targets=targets,
                train_mask=train_mask,
                rng=rng_update,
            )
            final_update_state, loss_info = lax.scan(
                update_epoch, update_state, None, config["num_epochs"]
            )
            train_state = final_update_state.train_state
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)

            # Self-play payoffs per finished hand in chips. The game is
            # zero-sum, so only the first player's return is informative (its
            # Nash value is -1/18).
            hand_end = transitions.done
            num_hands = jnp.maximum(hand_end.sum(), 1)
            first_return = jnp.take_along_axis(
                transitions.reward, transitions.start_player_idx[..., None], axis=-1
            ).squeeze(-1)
            metric = {
                "update_step": runner_state.update_step,
                "env_steps": (runner_state.update_step + 1)
                * config["num_envs"]
                * config["num_steps_per_env_per_update"],
                "hands_completed": hand_end.sum(),
                "hand_length": (transitions.info["timestep"] * hand_end).sum()
                / num_hands,
                "first_player_return_per_hand": (first_return * hand_end).sum()
                / num_hands,
                "abs_return_per_hand": (jnp.abs(first_return) * hand_end).sum()
                / num_hands,
                "ppo_sample_frac": train_mask.mean(),
                "lr": (
                    linear_decay(train_state.step)
                    if config["anneal_lr"]
                    else jnp.asarray(config["lr"])
                ),
            }
            metric.update(loss_info)

            # Evaluation against random: compare_episodes hands for each
            # policy, when due, from a stream of its own (eval_key).
            update = runner_state.update_step
            evaluated = compare_enabled & (
                (update % compare_interval == 0) | (update == last_update)
            )
            if compare_enabled:
                metric.update(
                    maybe_vs_random_metrics(
                        env,
                        {
                            "vs_random": (
                                softmax_policy(network.apply),
                                train_state.params,
                            ),
                        },
                        compare_episodes,
                        eval_key(runner_state.rng),
                        evaluated,
                    )
                )

            # The policy at every infoset; its exploitability and value against
            # random are computed on the host when due.
            policy_arrays = {
                "policy": policy_array_from_network(network.apply, train_state.params)
            }
            jax.experimental.io_callback(
                logging_callback, None, seed, metric, policy_arrays, evaluated
            )

            # Half-budget checkpoint: the network after num_half updates.
            take_half = runner_state.update_step + 1 == num_half
            half_params = jax.tree_util.tree_map(
                lambda new, old: jnp.where(take_half, new, old),
                train_state.params,
                runner_state.half_params,
            )
            runner_state = runner_state._replace(
                train_state=train_state,
                half_params=half_params,
                update_step=runner_state.update_step + 1,
                rng=rng,
            )
            return runner_state, None

        initial_runner_state = RunnerState(
            train_state=train_state,
            state=state,
            obs=obs,
            done=jnp.zeros((config["num_envs"]), dtype=jnp.bool_),
            half_params=network_params,  # replaced after num_half updates
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


def save_checkpoints(run_dir: str, num_seeds: int, half_params, full_params) -> None:
    """Writes {run_dir}/{half,full}/policy_{i}.msgpack for every seed
    (leading seed axis), the layout of the module docstring."""
    save_seed_params(run_dir, "half", "policy", half_params, range(num_seeds))
    save_seed_params(run_dir, "full", "policy", full_params, range(num_seeds))


def print_exploitability_summary(names: list[str]) -> None:
    """Final exploitability and value against random of every seed, and the
    mean over seeds; the last evaluation against random."""
    for title, log in (
        ("exploitability", EXPLOITABILITY),
        ("value vs random", VALUE_VS_RANDOM),
    ):
        for name in names:
            finals = [log[s][-1][1][name] for s in sorted(log)]
            values = " ".join(f"{v:.4f}" for v in finals)
            print(
                f"final {title} ({name}): mean {np.mean(finals):.4f} "
                f"std {np.std(finals):.4f} over {len(finals)} seed(s): {values}"
            )
    print_final_evals(EVALUATIONS)


def run(config: dict) -> str:
    """Trains config["num_seeds"] seeds in parallel, logging to results.jsonl
    and wandb, saves the checkpoints and returns the run directory."""
    global WANDB_RUNS, RESULTS_LOG
    config["num_update_steps"] = int(
        config["num_timesteps"]
        // config["num_envs"]
        // config["num_steps_per_env_per_update"]
    )
    config["half_update_steps"] = half_update_steps(config["num_update_steps"])
    config["num_gradient_steps"] = (
        config["num_update_steps"] * config["num_epochs"] * config["num_minibatches"]
    )
    env_steps = (
        config["num_update_steps"]
        * config["num_envs"]
        * config["num_steps_per_env_per_update"]
    )
    run_dir = make_run_dir(config)
    for log in (EXPLOITABILITY, VALUE_VS_RANDOM, EVALUATIONS):
        log.clear()
    try:
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
        train_vjit = (
            jax.jit(jax.vmap(make_train(config))).lower(rng_seeds, exp_ids).compile()
        )
        print(f"Compile finished in {time.time() - start:.1f} s")

        print("Running...")
        start = time.time()
        final_runner_state = jax.block_until_ready(train_vjit(rng_seeds, exp_ids))
        jax.effects_barrier()  # every logging callback has run
        run_time = time.time() - start
        print(
            f"Trained {config['num_seeds']} seed(s) x {env_steps} env steps in "
            f"{run_time:.1f} s (all seeds in parallel, evaluations included)"
        )
        print_exploitability_summary(["policy"])
        if config["save_checkpoints"]:
            save_checkpoints(
                run_dir,
                config["num_seeds"],
                final_runner_state.half_params,
                final_runner_state.train_state.params,
            )
            print(f"Saved checkpoints to {run_dir}")
    finally:
        for wandb_run in WANDB_RUNS:
            wandb_run.finish()
        print("Finished.")
    return run_dir


@hydra.main(version_base=None, config_path="./", config_name="config_ppo")
def main(config: dict) -> None:
    run(OmegaConf.to_container(config, resolve=True))


if __name__ == "__main__":
    register_resolvers()
    main()
