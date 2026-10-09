"""
PQN in self-play for Leduc Hold'em.

PQN (Gallici et al. 2024, "Simplifying Deep Temporal Difference Learning"):
one Q-network with LayerNorm plays both seats epsilon-greedily (the observation
encodes the position, so it is the acting player's perfect-recall information
state) and learns from Peng's Q(lambda) targets of its own rollouts.

- Targets follow each player's own decisions: a decision's reward is everything
  the player receives until its next decision or the end of the hand (which may
  come on the opponent's step), and it bootstraps from the max over legal
  actions of Q at that next decision (see per_player_q_lambda_targets). The same
  player can act twice in a row, e.g. when it calls a raise at the end of
  round 1 and then opens round 2.
- Evaluation: exact exploitability (leduc_exploitability) of the greedy policy
  every eval_interval updates and after the last update.

Units: the env pays chips per hand. Training uses reward * reward_scale (so
Q-values are in scaled units), and logged returns and exploitability are in
chips.
"""

import datetime
import time
from typing import Any, Callable, NamedTuple

from flax.training.train_state import TrainState
import hydra
from jax import lax
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax
import wandb

from bluffjax.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray
from bluffjax import make
from bluffjax.environments.leduc_holdem.leduc_holdem import (
    MAX_ACTIONS_PER_ROUND,
    LeducHoldemState,
)
from bluffjax.networks.mlp import QNetworkDiscreteMLP
from bluffjax.utils.game_utils.leduc_exploitability import (
    _INFOSET_LIST,
    _KEY_TO_IDX,
    exploitability_from_policy_array,
    policy_array_from_qnetwork,
)
from bluffjax.utils.jax_utils import pytree_norm
from bluffjax.utils.paths import REPO_ROOT, register_resolvers

WANDB_RUNS: list = []  # one wandb run per vmapped seed, created in main()
# EXPLOITABILITY[seed] = [(env_steps, {policy name: exploitability}), ...]
EXPLOITABILITY: dict[int, list[tuple[int, dict[str, float]]]] = {}


class Transition(NamedTuple):
    obs: FloatArray
    action: IntArray
    reward: FloatArray  # (num_agents,) env reward for every player, chips
    done: BoolArray  # the hand ended at this step (the env auto-resets)
    q_max: FloatArray  # max over legal actions of Q(obs), scaled reward units
    player_idx: IntArray
    start_player_idx: IntArray  # the hand's first player
    info: dict[str, Any]


class RunnerState(NamedTuple):
    train_state: TrainState
    state: LeducHoldemState
    obs: FloatArray
    done: BoolArray
    update_step: IntArray
    rng: PRNGKeyArray


class UpdateState(NamedTuple):
    train_state: TrainState
    transitions: Transition
    targets: FloatArray
    train_mask: BoolArray
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
        dones: (T, N) the hand ended at step t (the env then auto-resets).
        players: (T, N) player acting at step t.
        traces: (T, N) trace coefficient of the decision at t: lambda if it was
            taken by the epsilon-greedy policy, 0 to cut the trace there.
        last_q_max: (N,) max over legal actions of Q at the observation after
            the last step.
        last_player: (N,) player to act after the last step.

    For player p's decision at step t with next own decision t' in the same
    hand, R = p's rewards from step t up to t' (rewards paid on the opponent's
    steps included) and
        G_t = R + gamma * ((1 - c) * q_max[t'] + c * G_t'),  c = traces[t'],
    i.e. Peng's Q(lambda) as in PQN, and the one-step Q-learning target where
    c = 0. If the hand ends first, G_t = R. At the rollout cut-off only the
    player to act has a next decision (G_t = R + gamma * last_q_max). The other
    player's trailing decision in an unfinished hand has no known outcome yet:
    it is marked invalid, and that player's earlier decisions bootstrap from
    q_max there without the trace.

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


def make_train(config: dict) -> Callable[[PRNGKeyArray, int], RunnerState]:
    env = make("leduc_holdem", **config["env_kwargs"])
    if env.horizon < 2 * MAX_ACTIONS_PER_ROUND:
        # A truncated hand pays 0, but exploitability is for the full game.
        raise ValueError(
            f"horizon={env.horizon} truncates Leduc hands (up to "
            f"{2 * MAX_ACTIONS_PER_ROUND} actions)"
        )
    config["batch_shuffle_dim"] = (
        config["num_steps_per_env_per_update"] * config["num_envs"]
    )
    network = QNetworkDiscreteMLP(
        action_dim=env.num_actions, hidden_dim=config["fc_dim_size"]
    )
    eval_interval = config["eval_interval"]
    last_update = config["num_update_steps"] - 1

    def linear_decay(count: int) -> float:
        frac = 1.0 - count / config["num_gradient_steps"]
        return config["lr"] * frac

    def epsilon_schedule(update_step: FloatArray) -> FloatArray:
        decay_steps = max(
            config["exploration_fraction"] * config["num_update_steps"], 1
        )
        frac = jnp.minimum(1.0, update_step.astype(jnp.float32) / decay_steps)
        return config["start_e"] + frac * (config["end_e"] - config["start_e"])

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
            metrics["exploitability"] = expl["greedy"]
            print(
                f"seed {seed_i} update {update + 1}/{last_update + 1} "
                f"({env_steps} env steps): exploitability {expl['greedy']:.4f}",
                flush=True,
            )
        WANDB_RUNS[seed_i].log(metrics)

    def train(rng: PRNGKeyArray, seed: int) -> RunnerState:
        rng, rng_reset, rng_network_init = jax.random.split(rng, 3)
        rng_resets = jax.random.split(rng_reset, config["num_envs"])
        state, obs = jax.vmap(env.reset, in_axes=(0))(rng_resets)
        network_params = network.init(rng_network_init, obs)
        lr = linear_decay if config["anneal_lr"] else config["lr"]
        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=lr),
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
                rng, rng_explore, rng_random, rng_step = jax.random.split(
                    runner_state.rng, 4
                )

                q_vals = network.apply(train_state.params, obs.astype(jnp.float32))
                action_mask = jax.vmap(env.get_avail_actions, in_axes=(0))(state)
                q_vals_masked = jnp.where(action_mask, q_vals, -jnp.inf)
                greedy_action = jnp.argmax(q_vals_masked, axis=-1)
                q_max = jnp.max(q_vals_masked, axis=-1)

                # epsilon-greedy over the legal actions
                eps = epsilon_schedule(
                    jnp.array(runner_state.update_step, dtype=jnp.float32)
                )
                explore = jax.random.uniform(rng_explore, (config["num_envs"],)) < eps
                random_action = jax.random.categorical(
                    rng_random, jnp.where(action_mask, 0.0, -jnp.inf)
                )
                action = jnp.where(explore, random_action, greedy_action)

                rng_steps = jax.random.split(rng_step, config["num_envs"])
                next_state, next_obs, reward, _, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_steps, state, action)

                transition = Transition(
                    obs=obs,
                    action=action,
                    reward=reward,
                    done=done,
                    q_max=q_max,
                    player_idx=state.current_player_idx,
                    start_player_idx=state.start_player_idx,
                    info=info,
                )
                runner_state = RunnerState(
                    train_state=train_state,
                    state=next_state,
                    obs=next_obs,
                    done=done,
                    update_step=runner_state.update_step,
                    rng=rng,
                )
                return runner_state, transition

            runner_state, transitions = lax.scan(
                step, runner_state, None, config["num_steps_per_env_per_update"]
            )

            train_state = runner_state.train_state
            last_state = runner_state.state
            rng, rng_update = jax.random.split(runner_state.rng)

            last_mask = jax.vmap(env.get_avail_actions, in_axes=(0))(last_state)
            last_q_vals = network.apply(
                train_state.params, runner_state.obs.astype(jnp.float32)
            )
            last_q_max = jnp.max(jnp.where(last_mask, last_q_vals, -jnp.inf), axis=-1)

            # Per-player Peng's Q(lambda) targets in scaled reward units; every
            # decision is epsilon-greedy, so the trace is never cut.
            targets, valid = per_player_q_lambda_targets(
                transitions.q_max,
                transitions.reward * config["reward_scale"],
                transitions.done,
                transitions.player_idx,
                jnp.full(transitions.q_max.shape, config["q_lambda"]),
                last_q_max,
                last_state.current_player_idx,
                config["gamma"],
            )
            train_mask = valid

            def update_epoch(
                update_state: UpdateState, unused: None
            ) -> tuple[UpdateState, dict[str, FloatArray]]:
                rng, rng_permute = jax.random.split(update_state.rng)
                batch = (
                    update_state.transitions.obs,
                    update_state.transitions.action,
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
                    minibatch: tuple[FloatArray, IntArray, FloatArray, BoolArray],
                ) -> tuple[TrainState, dict[str, FloatArray]]:
                    obs_mb, action_mb, targets_mb, train_mask_mb = minibatch

                    def loss_fn(params):
                        q_vals = network.apply(params, obs_mb.astype(jnp.float32))
                        chosen_q = jnp.take_along_axis(
                            q_vals, action_mb[..., None].astype(jnp.int32), axis=-1
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
                    update_minibatch, update_state.train_state, minibatches
                )
                update_state = UpdateState(
                    train_state=final_train_state,
                    transitions=update_state.transitions,
                    targets=update_state.targets,
                    train_mask=update_state.train_mask,
                    rng=rng,
                )
                return update_state, batch_stats

            update_state = UpdateState(
                train_state=train_state,
                transitions=transitions,
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
            # Nash value is about -0.086).
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
                "q_sample_frac": train_mask.mean(),
                "epsilon": epsilon_schedule(
                    jnp.asarray(runner_state.update_step, dtype=jnp.float32)
                ),
                "lr": (
                    linear_decay(train_state.step)
                    if config["anneal_lr"]
                    else jnp.asarray(config["lr"])
                ),
            }
            metric.update(loss_info)

            # The greedy policy at every infoset; its exploitability is
            # computed on the host when due.
            policy_arrays = {
                "greedy": policy_array_from_qnetwork(
                    network.apply, train_state.params, _INFOSET_LIST
                )
            }
            jax.experimental.io_callback(
                logging_callback, None, seed, metric, policy_arrays
            )

            runner_state = RunnerState(
                train_state=train_state,
                state=runner_state.state,
                obs=runner_state.obs,
                done=runner_state.done,
                update_step=runner_state.update_step + 1,
                rng=rng,
            )
            return runner_state, None

        initial_runner_state = RunnerState(
            train_state=train_state,
            state=state,
            obs=obs,
            done=jnp.zeros((config["num_envs"]), dtype=jnp.bool_),
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


@hydra.main(version_base=None, config_path="./", config_name="config_pqn")
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
        print_exploitability_summary(["greedy"])
    finally:
        for run in WANDB_RUNS:
            run.finish()
        print("Finished.")


if __name__ == "__main__":
    register_resolvers()
    main()
