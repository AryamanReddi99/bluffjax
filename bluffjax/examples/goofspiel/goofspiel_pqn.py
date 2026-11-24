"""
PQN in self-play for Goofspiel.

PQN (Gallici et al. 2024, "Simplifying Deep Temporal Difference Learning"):
one Q-network with LayerNorm plays every seat epsilon-greedily (the
observation is relative to the observer) and learns from Peng's Q(lambda)
targets of its own rollouts. This is goofspiel_pqn_nfsp.py without NFSP: no
average policy, no reservoir and no policy mixing, so every decision is taken
by the epsilon-greedy Q-network and the lambda-trace is never cut. With the
same config the Q-network is trained exactly as PQN-NFSP's best response,
except that it plays against itself instead of against NFSP's mixture.

Goofspiel is a simultaneous-move game (ParallelEnv): every player bids at
every step, so every player makes a decision at every step of the game.

- Targets (parallel_nfsp_common.per_agent_q_lambda_targets): a player's
  decision at step t gets its own reward of that step (the round's zero-sum
  payoff) and bootstraps from the max over legal actions of Q at its decision
  at t + 1, unless the game ends at t.
- Optimiser and schedule: br_optimizer adam (default) or radam (as in
  purejaxql's PQN); with anneal_lr the lr decays linearly from lr to
  lr * lr_final_frac (default 0). Epsilon decays linearly from start_e to
  end_e over the first exploration_fraction of the updates.
- Truncation: a game the env cuts off without ending it (done without
  absorbing; Goofspiel always ends after N rounds, so this doesn't happen
  with the current env) would not be treated as terminal: the decisions at
  the cut have no known outcome and are left out of the loss.
- Evaluation against a baseline (compare): the greedy policy (argmax Q over
  the cards in hand, ties broken at random) plays compare_episodes games
  against uniformly random legal bids or a checkpoint, its seat rotating.
- Game metrics (bluffjax/metrics/goofspiel.py) of every update's self-play
  rollout as game/<metric>; the bid entropy is that of the epsilon-greedy
  distribution the bid was sampled from.
- Exact evaluation (two players and env_kwargs.num_cards N <= 6, see
  bluffjax/utils/game_utils/goofspiel_exploitability.py): every
  exact_eval_interval updates and after the last one, the exploitability
  (NashConv / 2, exact over every decision state) of the greedy policy (ties
  to the lowest card) and its exact expected return against uniformly random
  bids (value_vs_random). The exact metrics use GE.EXACT_KEYS: with no
  average policy, exploitability_br and value_br_vs_random are the same
  greedy policy's (identical to exploitability and value_vs_random). In
  points of the point difference.

Units: the env pays each round's prize value (1-N) to its winner and
-value / (num_agents - 1) to every other player, so a game's return is the
player's points minus the opponents' mean points (with two players the point
difference). Logged returns are in these units.

Run directory, {save_dir}/{job_type}_{timestamp}/, made when the run starts
(bluffjax.examples.run_utils.make_run_dir):
    config.yaml                 the resolved config, with num_update_steps,
                                half_update_steps and run_dir
    results.jsonl               one record per update and seed: seed_index,
                                wall_time and every metric sent to wandb,
                                env_steps included (run_utils.read_results)
    half/q_network_{i}.msgpack  if save_checkpoints: the Q-network of seed i
                                after half_update_steps(num_update_steps)
    full/q_network_{i}.msgpack  updates and after all num_update_steps (the
                                same run; the schedules span the whole run)
Every checkpoint holds flax params without a seed axis of
QNetworkDiscreteMLP(action_dim=N, hidden_dim=fc_dim_size): one player's
observation (env.obs_from_state(state)[p], env.obs_dim floats) -> Q-values of
the N bids in env reward units. Mask the cards no longer in hand
(env.get_avail_actions(state)[p]) before the argmax. Load a file with
run_utils.load_params(path, network.init(rng, obs)).
"""

import datetime
import functools
import time
from typing import Any, Callable, NamedTuple

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
    BoolArray,
    FloatArray,
    IntArray,
    PRNGKeyArray,
)
from bluffjax import make
from bluffjax.environments.goofspiel.goofspiel import GoofspielState
from bluffjax.examples.parallel_nfsp_common import (
    epsilon_greedy_probs,
    eval_metrics,
    evaluate_goofspiel,
    greedy_random_ties,
    log_update,
    per_agent_q_lambda_targets,
    print_final_evals,
    print_final_exact,
    sample_logits,
    sample_random_legal,
)
from bluffjax.examples.run_utils import (
    ResultsLog,
    half_update_steps,
    load_params,
    make_run_dir,
    save_seed_params,
)
from bluffjax.metrics import Counts, make_metrics, masked_totals, prefixed
from bluffjax.networks.mlp import (
    ActorCriticDiscreteMLP,
    ActorDiscreteMLP,
    QNetworkDiscreteMLP,
)
from bluffjax.utils.jax_utils import pytree_norm
from bluffjax.utils.game_utils import goofspiel_exploitability as GE
from bluffjax.utils.paths import REPO_ROOT, register_resolvers

# results.jsonl of the run and its wandb runs (one per vmapped seed), made in main()
RESULTS_LOG: ResultsLog | None = None
# EVALS[seed] = [(env_steps, {metric: value}), ...]
EVALS: dict[int, list[tuple[int, dict[str, float]]]] = {}
# EXACT[seed] = [(env_steps, {exact metric: value}), ...]
EXACT: dict[int, list[tuple[int, dict[str, float]]]] = {}


class Transition(NamedTuple):
    """One env step of every env; agent fields are (num_envs, num_agents, ...)."""

    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    reward: FloatArray  # env reward of every agent
    done: BoolArray  # (num_envs,) the game ended or was cut off (env auto-resets)
    truncated: BoolArray  # the agent's game was cut off without being over
    q_max: FloatArray  # max over legal actions of Q(obs)
    game_return: FloatArray  # env rewards of the game so far, this step included
    game_length: IntArray  # (num_envs,) steps of the game so far, this one included
    game_counts: Counts  # game metrics of the step, credited per agent


class RunnerState(NamedTuple):
    train_state: TrainState
    state: GoofspielState
    obs: FloatArray
    game_return: FloatArray  # (num_envs, num_agents) rewards so far this game
    game_length: IntArray  # (num_envs,) steps so far this game
    metric_carry: Any  # per-env carry of the game metrics (current game)
    half_params: Any  # Q-network after half_update_steps updates
    update_step: IntArray
    rng: PRNGKeyArray


class PQNUpdateState(NamedTuple):
    train_state: TrainState
    transitions: Transition
    targets: FloatArray
    train_mask: BoolArray
    rng: PRNGKeyArray


def make_train(config: dict) -> Callable[..., RunnerState]:
    env = make("goofspiel", **config["env_kwargs"])
    num_agents = env.num_agents
    action_dim = env.num_actions
    num_envs = config["num_envs"]
    num_steps = config["num_steps_per_env_per_update"]
    # every agent makes a decision at every step
    config["batch_shuffle_dim"] = num_steps * num_envs * num_agents
    if config["batch_shuffle_dim"] % config["num_minibatches"]:
        raise ValueError("num_minibatches must divide the number of decisions per update")
    num_half = half_update_steps(config["num_update_steps"])
    gm = make_metrics("goofspiel", env)  # bluffjax/metrics/goofspiel.py
    template_obs = jnp.zeros((env.obs_dim,), dtype=jnp.float32)

    q_network = QNetworkDiscreteMLP(
        action_dim=action_dim, hidden_dim=config["fc_dim_size"]
    )

    def act_greedy(params, obs, mask, rng):
        return greedy_random_ties(q_network.apply(params, obs), mask, rng)

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
            baseline_network = q_network
            act_baseline = act_greedy
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

    # With anneal_lr the lr decays linearly from lr to lr * lr_final_frac
    # over the run (lr_final_frac 0, the default, decays to 0).
    lr_final_frac = config.get("lr_final_frac", 0.0)
    br_optimizer = config.get("br_optimizer", "adam")
    if br_optimizer not in ("adam", "radam"):
        raise ValueError(f"br_optimizer must be 'adam' or 'radam', got {br_optimizer!r}")

    def linear_decay(count: int) -> float:
        frac = 1.0 - count / config["num_gradient_steps"]
        return config["lr"] * (lr_final_frac + (1.0 - lr_final_frac) * frac)

    def epsilon_schedule(update_step: FloatArray) -> FloatArray:
        decay_steps = max(
            config["exploration_fraction"] * config["num_update_steps"], 1
        )
        frac = jnp.minimum(1.0, update_step.astype(jnp.float32) / decay_steps)
        return config["start_e"] + frac * (config["end_e"] - config["start_e"])

    def run_compare_eval(rng: PRNGKeyArray, params: Any) -> dict[str, FloatArray]:
        # {stat}_vs_{opponent} of the greedy policy
        result = evaluate_goofspiel(
            env,
            act_greedy,
            params,
            act_baseline,
            baseline_params,
            config["compare_episodes"],
            rng,
        )
        return eval_metrics(f"vs_{baseline_name}", result)

    def greedy_table_probs(params: Any):
        """tables -> (S, N): the greedy policy at every decision state."""
        return lambda t: GE.greedy_probs(
            GE.apply_in_chunks(lambda o: q_network.apply(params, o), t.obs),
            t.legal,
        )

    def train(
        rng: PRNGKeyArray,
        seed: int,
        num_updates: int | None = None,
        exact_tables: GE.DeviceTables | None = None,
    ) -> RunnerState:
        """Runs num_updates updates (default: all of them) of one seed. The
        schedules always span config["num_update_steps"] updates. With
        exact_tables (GE.tables_for_training; an argument, not a closure, as
        they are large) the exact metrics are computed every
        exact_eval_interval updates and after the last one."""
        rng, rng_reset, rng_init = jax.random.split(rng, 3)
        state, obs = jax.vmap(env.reset)(jax.random.split(rng_reset, num_envs))

        params = q_network.init(rng_init, obs)
        lr = linear_decay if config["anneal_lr"] else config["lr"]
        # br_optimizer: adam (default) or radam (purejaxql's PQN)
        opt = optax.radam if br_optimizer == "radam" else optax.adam
        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            opt(learning_rate=lr),
        )
        train_state = TrainState.create(apply_fn=q_network.apply, params=params, tx=tx)

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
                rng_s, rng_explore, rng_random, rng_step = jax.random.split(
                    runner_state_inner.rng, 4
                )
                action_mask = jax.vmap(env.get_avail_actions)(state_s)

                q_vals = q_network.apply(runner_state_inner.train_state.params, obs_s)
                q_vals_masked = jnp.where(action_mask, q_vals, -jnp.inf)
                greedy_action = jnp.argmax(q_vals_masked, axis=-1)
                q_max = jnp.max(q_vals_masked, axis=-1)

                # epsilon-greedy over the legal actions, per agent
                eps = epsilon_schedule(
                    jnp.asarray(runner_state_inner.update_step, dtype=jnp.float32)
                )
                explore = jax.random.uniform(rng_explore, (num_envs, num_agents)) < eps
                random_action = jax.random.categorical(
                    rng_random, jnp.where(action_mask, 0.0, -jnp.inf)
                )
                action = jnp.where(explore, random_action, greedy_action)

                next_state, next_obs, reward, absorbing, done, info = jax.vmap(
                    env.step
                )(jax.random.split(rng_step, num_envs), state_s, action)
                # done without absorbing: cut off at the horizon, not over
                truncated = done[:, None] & ~absorbing

                metric_carry, game_counts = jax.vmap(gm.step)(
                    runner_state_inner.metric_carry,
                    state_s,
                    action,
                    next_state,
                    reward,
                    done,
                    info,
                    epsilon_greedy_probs(greedy_action, action_mask, eps),
                )

                game_return = runner_state_inner.game_return + reward
                game_length = runner_state_inner.game_length + 1
                transition = Transition(
                    obs=obs_s,
                    action_mask=action_mask,
                    action=action,
                    reward=reward,
                    done=done,
                    truncated=truncated,
                    q_max=q_max,
                    game_return=game_return,
                    game_length=game_length,
                    game_counts=game_counts,
                )
                runner_state_inner = runner_state_inner._replace(
                    state=next_state,
                    obs=next_obs,
                    game_return=jnp.where(done[:, None], 0.0, game_return),
                    game_length=jnp.where(done, 0, game_length),
                    metric_carry=metric_carry,
                    rng=rng_s,
                )
                return runner_state_inner, transition

            runner_state, transitions = lax.scan(step, runner_state, None, num_steps)

            train_state = runner_state.train_state
            rng, rng_pqn_update, rng_compare = jax.random.split(runner_state.rng, 3)

            last_mask = jax.vmap(env.get_avail_actions)(runner_state.state)
            last_q_vals = q_network.apply(train_state.params, runner_state.obs)
            last_q_max = jnp.max(jnp.where(last_mask, last_q_vals, -jnp.inf), axis=-1)

            # Per-agent Peng's Q(lambda) targets in env reward units; every
            # decision is the epsilon-greedy Q-network's, so the trace is never cut.
            targets, valid = per_agent_q_lambda_targets(
                transitions.q_max,
                transitions.reward,
                jnp.broadcast_to(
                    transitions.done[..., None], transitions.reward.shape
                ),
                transitions.truncated,
                jnp.full(transitions.reward.shape, config["q_lambda"], jnp.float32),
                last_q_max,
                config["gamma"],
            )
            train_mask = valid

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
                    minibatch: tuple[FloatArray, IntArray, FloatArray, BoolArray],
                ) -> tuple[TrainState, dict[str, FloatArray]]:
                    obs_mb, action_mb, targets_mb, train_mask_mb = minibatch

                    def loss_fn(params):
                        q_vals = q_network.apply(params, obs_mb)
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
                train_state=train_state,
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
            train_state = final_pqn_state.train_state
            pqn_loss_info = jax.tree_util.tree_map(lambda x: x.mean(), pqn_loss_info)

            # Evaluate every compare_interval updates and after the last one.
            if compare_enabled:
                should_compare = (
                    runner_state.update_step % config["compare_interval"] == 0
                ) | (runner_state.update_step == config["num_update_steps"] - 1)
                eval_template = jax.eval_shape(
                    run_compare_eval, rng_compare, train_state.params
                )
                compare_metrics = lax.cond(
                    should_compare,
                    lambda: run_compare_eval(rng_compare, train_state.params),
                    lambda: jax.tree_util.tree_map(
                        lambda x: jnp.zeros(x.shape, x.dtype), eval_template
                    ),
                )
            else:
                should_compare = jnp.array(False)
                compare_metrics = {}

            # Self-play statistics of the games that ended in this rollout.
            episodes_completed = transitions.done.sum()
            agent_game_end = jnp.broadcast_to(
                transitions.done[..., None], transitions.reward.shape
            )
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
                "return_per_episode": (transitions.game_return * agent_game_end).sum()
                / jnp.maximum(agent_game_end.sum(), 1),
                "q_sample_frac": train_mask.mean(),
                "epsilon": epsilon_schedule(
                    jnp.asarray(runner_state.update_step, dtype=jnp.float32)
                ),
                "lr": (
                    linear_decay(train_state.step)
                    if config["anneal_lr"]
                    else jnp.asarray(config["lr"])
                ),
                "evaluated": should_compare,
            }
            metric.update(compare_metrics)
            metric.update(pqn_loss_info)

            # Exact exploitability and value against random (small games) of
            # the greedy policy, in both the average-policy and BR slots of
            # GE.EXACT_KEYS.
            if exact_tables is not None:
                should_exact = (
                    runner_state.update_step % config["exact_eval_interval"] == 0
                ) | (runner_state.update_step == config["num_update_steps"] - 1)
                metric["exact_evaluated"] = should_exact
                greedy = greedy_table_probs(train_state.params)
                metric.update(
                    GE.maybe_exact_metrics(exact_tables, should_exact, greedy, greedy)
                )

            # Game metrics of this rollout (every player follows the
            # epsilon-greedy Q-network).
            totals = masked_totals(
                transitions.game_counts,
                jnp.ones(transitions.reward.shape, dtype=bool),
            )
            metric.update(prefixed("game", gm.summarize(totals)))

            io_callback(
                functools.partial(logging_callback, config=config), None, seed, metric
            )

            # Half-budget checkpoint: the Q-network after num_half updates.
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
            game_return=jnp.zeros((num_envs, num_agents), dtype=jnp.float32),
            game_length=jnp.zeros((num_envs,), dtype=jnp.int32),
            metric_carry=jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (num_envs,) + x.shape), gm.init_carry()
            ),
            half_params=params,  # replaced after num_half updates
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


def logging_callback(seed_val, metric_dict, config) -> None:
    log_update(
        RESULTS_LOG, EVALS, seed_val, metric_dict, config, EXACT, GE.EXACT_KEYS
    )


@hydra.main(version_base=None, config_path="./", config_name="config_pqn")
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
        exact_tables = GE.tables_for_training(
            config, make("goofspiel", **config["env_kwargs"])
        )
        print(
            "Exact exploitability: "
            + (
                f"{len(exact_tables.swap)} decision states, every "
                f"{config['exact_eval_interval']} updates"
                if exact_tables is not None
                else "off"
            )
        )
        train_vjit = (
            jax.jit(
                jax.vmap(
                    lambda r, s, t: train(r, s, exact_tables=t), in_axes=(0, 0, None)
                )
            )
            .lower(rng_seeds, exp_ids, exact_tables)
            .compile()
        )
        print(f"Compile finished in {time.time() - start:.1f} s")

        print("Running...")
        start = time.time()
        final_runner_state = jax.block_until_ready(
            train_vjit(rng_seeds, exp_ids, exact_tables)
        )
        jax.effects_barrier()  # every update's record is in results.jsonl
        run_time = time.time() - start
        print(
            f"Trained {config['num_seeds']} seed(s) x {env_steps} env steps in "
            f"{run_time:.1f} s ({run_time / env_steps * 1e6:.2f} s per 1e6 env "
            f"steps, all seeds in parallel)"
        )
        print_final_evals(EVALS)
        print_final_exact(EXACT)
        if config["save_checkpoints"]:
            seeds = range(config["num_seeds"])
            save_seed_params(
                run_dir, "half", "q_network", final_runner_state.half_params, seeds
            )
            save_seed_params(
                run_dir,
                "full",
                "q_network",
                final_runner_state.train_state.params,
                seeds,
            )
            print(f"Saved checkpoints to {run_dir}")
    finally:
        for run in wandb_runs:
            run.finish()
        print("Finished.")


if __name__ == "__main__":
    register_resolvers()
    main()
