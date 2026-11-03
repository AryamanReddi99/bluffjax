"""
Self-play PPO with optimistic Adam for Goofspiel.

goofspiel_ppo.py with Adam replaced by optimistic Adam (Daskalakis et al.
2018, "Training GANs with Optimism"); everything else (rollout, GAE, epochs,
minibatches, losses, schedules, sweeps, evaluation, logging, checkpoints) is
unchanged, and with optimism: 0 the script is goofspiel_ppo.py.

Update. Every minibatch step takes one gradient g_t of the PPO loss (one
backward pass), clips it to global norm max_grad_norm as goofspiel_ppo.py
does, and applies, with lr_t the scheduled learning rate and o = optimism,

    m_t = b1 m_{t-1} + (1 - b1) g_t,   v_t = b2 v_{t-1} + (1 - b2) g_t^2
    r_t = (m_t / (1 - b1^t)) / (sqrt(v_t / (1 - b2^t)) + eps)   (Adam's step)
    theta_{t+1} = theta_t - lr_t ((1 + o) r_t - o r_{t-1})

(b1 0.9, b2 0.999, eps 1e-5 as goofspiel_ppo.py; r_{t-1} is the previous
minibatch step's Adam step, kept in the optimizer state; at the first step
r_{t-1} = r_t, so the first step is Adam's). o = 1, the default, is the
textbook optimistic Adam, theta_{t+1} = theta_t - lr_t (2 r_t - r_{t-1}), i.e.
2 AdamStep(g_t) - AdamStep(g_{t-1}); o = 0 is Adam. This is optax's
optimistic_adam_v2(lr_t, alpha=1, beta=o, nesterov=False) (optax 0.2.6:
scale_by_adam, then scale_by_optimistic_gradient, then the learning rate;
nesterov=False as optax.adam). With a learning-rate schedule both steps take
the current lr_t (the paper's constant-rate form; the lr is constant by
default). optax's deprecated optimistic_adam(lr, optimism) is the same
update with its `optimism` the absolute rate o * lr (default lr, i.e. o = 1;
update -lr r_t - optimism (r_t - r_{t-1})), Nesterov Adam by default and no
lr schedules; it is not used.
The previous step crosses update boundaries: the optimizer state, r_{t-1}
included, carries over from the last minibatch of an update to the first
of the next, as Adam's moments do.

Knob (sweepable): optimism (o above).

Below, goofspiel_ppo.py's description, which holds unchanged.

All players share one actor-critic (the observation is relative to the
observer) and every player's every decision trains it: Goofspiel is a
simultaneous-move game (ParallelEnv), so every player bids at every step.
The policy evaluated is the current one; there is no average policy.

- PPO: per-agent GAE in env reward units (parallel_nfsp_common.per_agent_gae:
  the decision at step t gets the round's zero-sum payoff and bootstraps from
  the agent's decision at t + 1 unless the game ends at t), clipped
  surrogate, clipped value loss, entropy bonus, advantages normalised per
  minibatch. A game the env cuts off without ending it (never with the
  default horizon) would be left out of the loss.
- Schedules (linear over all gradient steps): learning rate from lr to
  lr * lr_final_frac, entropy coefficient from ent_coef to
  ent_coef * ent_final_frac (1.0: constant).
- Sweeps: `sweep` maps hyperparameters to lists of values; the run trains
  every point of their grid (itertools.product, in the order the keys are
  given) for num_seeds seeds, all vmapped in one program. Only the scalar
  hyperparameters of SWEEPABLE can be swept this way (the others change
  shapes; run them as separate jobs). Instance i is grid point
  i // num_seeds, seed i % num_seeds; it is logged as seed_index i with its
  hyperparameters as hp/<name> and its grid point and seed as
  hp/grid_index, hp/seed. At most max_vmap_instances run in one program
  (GPU memory); more run as consecutive chunks.
- Exact evaluation (two players and env_kwargs.num_cards N <= 6;
  bluffjax/utils/game_utils/goofspiel_exploitability.py): every
  exact_eval_interval updates and after the last one, exploitability
  (NashConv / 2, exact over every decision state) of the policy's softmax
  over the cards in hand, and its exact expected return against uniformly
  random bids (value_vs_random). Points of the point difference.
- Evaluation against random: compare_episodes games against uniformly
  random legal bids, the learner's seat alternating (win_rate_vs_random,
  avg_return_vs_random, ...; parallel_nfsp_common.evaluate_goofspiel).
- Game metrics (bluffjax/metrics/goofspiel.py: bid-prize Spearman
  correlation, bid entropy, overbid margin, burn rate) of every update's
  self-play rollout, as game/<metric>, each rate with its count (n_...).

Units: the env pays each round's prize value (1-N) to its winner and -value
to the other player, so a game's return is the point difference.

Run directory {save_dir}/{job_type}_{timestamp}/ (run_utils.make_run_dir):
    config.yaml, results.jsonl   as in the NFSP scripts, one record per update
                                 and instance (seed_index = instance)
    sweep.json                   the hyperparameters of every instance
    half/policy_{i}.msgpack      if save_checkpoints: instance i's
    full/policy_{i}.msgpack      ActorCriticDiscreteMLP(action_dim=N,
                                 hidden_dim=fc_dim_size) after half and all
                                 of the updates
"""

import datetime
import functools
import itertools
import json
import os
import time
from typing import Callable, NamedTuple

import distrax
from flax.training.train_state import TrainState
import hydra
import jax
from jax import lax
from jax.experimental import io_callback
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax
import wandb

from bluffjax import make
from bluffjax.environments.goofspiel.goofspiel import GoofspielState
from bluffjax.examples.parallel_nfsp_common import (
    eval_metrics,
    evaluate_goofspiel,
    log_update,
    per_agent_gae,
    print_final_evals,
    print_final_exact,
    sample_logits,
    sample_random_legal,
)
from bluffjax.examples.run_utils import (
    ResultsLog,
    half_update_steps,
    make_run_dir,
    save_seed_params,
)
from bluffjax.metrics import Counts, make_metrics, masked_totals, prefixed
from bluffjax.networks.mlp import ActorCriticDiscreteMLP
from bluffjax.utils.game_utils import goofspiel_exploitability as GE
from bluffjax.utils.jax_utils import pytree_norm
from bluffjax.utils.paths import REPO_ROOT, register_resolvers
from bluffjax.utils.typing import Any, BoolArray, FloatArray, IntArray, PRNGKeyArray

# Scalar hyperparameters a sweep may vary (traced per instance).
SWEEPABLE = (
    "lr",
    "lr_final_frac",
    "ent_coef",
    "ent_final_frac",
    "gamma",
    "gae_lambda",
    "clip_eps",
    "vf_coef",
    "max_grad_norm",
    "optimism",
)

RESULTS_LOG: ResultsLog | None = None
EVALS: dict[int, list[tuple[int, dict[str, float]]]] = {}
EXACT: dict[int, list[tuple[int, dict[str, float]]]] = {}
INSTANCES: list[dict[str, float]] = []  # hp/<name> of every instance


class HParams(NamedTuple):
    lr: FloatArray
    lr_final_frac: FloatArray
    ent_coef: FloatArray
    ent_final_frac: FloatArray
    gamma: FloatArray
    gae_lambda: FloatArray
    clip_eps: FloatArray
    vf_coef: FloatArray
    max_grad_norm: FloatArray
    optimism: FloatArray


class Transition(NamedTuple):
    """One env step of every env; agent fields are (num_envs, num_agents, ...)."""

    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    log_prob: FloatArray
    reward: FloatArray
    done: BoolArray  # (num_envs,)
    truncated: BoolArray
    value: FloatArray
    game_return: FloatArray
    game_length: IntArray
    game_counts: Counts


class RunnerState(NamedTuple):
    train_state: TrainState
    state: GoofspielState
    obs: FloatArray
    game_return: FloatArray
    game_length: IntArray
    metric_carry: Any
    half: Any  # params after half_update_steps updates
    update_step: IntArray
    rng: PRNGKeyArray


def instances(config: dict) -> list[dict[str, float]]:
    """hp/<name> of every instance: the sweep grid times the seeds."""
    sweep = dict(config.get("sweep") or {})
    unknown = set(sweep) - set(SWEEPABLE)
    if unknown:
        raise ValueError(f"cannot sweep {sorted(unknown)}; sweepable: {SWEEPABLE}")
    keys = list(sweep)
    out = []
    for g, values in enumerate(itertools.product(*(sweep[k] for k in keys))):
        point = {k: float(config[k]) for k in SWEEPABLE}
        point.update({k: float(v) for k, v in zip(keys, values)})
        for s in range(config["num_seeds"]):
            out.append({**{f"hp/{k}": v for k, v in point.items()}, "hp/grid_index": g, "hp/seed": s})
    return out


def hparams_of(insts: list[dict[str, float]]) -> HParams:
    return HParams(
        **{k: jnp.asarray([i[f"hp/{k}"] for i in insts], jnp.float32) for k in SWEEPABLE}
    )


def make_train(config: dict) -> Callable[..., RunnerState]:
    env = make("goofspiel", **config["env_kwargs"])
    num_agents = env.num_agents
    action_dim = env.num_actions
    num_envs = config["num_envs"]
    num_steps = config["num_steps_per_env_per_update"]
    num_updates_total = config["num_update_steps"]
    batch = num_steps * num_envs * num_agents
    if batch % config["num_minibatches"]:
        raise ValueError("num_minibatches must divide the number of decisions per update")
    num_gradient_steps = num_updates_total * config["num_epochs"] * config["num_minibatches"]
    num_half = half_update_steps(num_updates_total)
    gm = make_metrics("goofspiel", env)
    network = ActorCriticDiscreteMLP(action_dim=action_dim, hidden_dim=config["fc_dim_size"])

    def act(params, obs, mask, rng):
        return sample_logits(network.apply(params, obs)[0], mask, rng)

    def act_random(params, obs, mask, rng):
        return sample_random_legal(mask, rng)

    def train(
        rng: PRNGKeyArray,
        seed: IntArray,
        hp: HParams,
        exact_tables: GE.DeviceTables | None = None,
        num_updates: int | None = None,
    ) -> RunnerState:
        """One instance: hp holds its scalar hyperparameters (traced)."""

        def frac_left(count):
            return 1.0 - count / num_gradient_steps

        def lr_schedule(count):
            return hp.lr * (hp.lr_final_frac + (1.0 - hp.lr_final_frac) * frac_left(count))

        def ent_schedule(count):
            return hp.ent_coef * (
                hp.ent_final_frac + (1.0 - hp.ent_final_frac) * frac_left(count)
            )

        rng, rng_reset, rng_init = jax.random.split(rng, 3)
        state, obs = jax.vmap(env.reset)(jax.random.split(rng_reset, num_envs))
        params = network.init(rng_init, obs)
        tx = optax.chain(
            optax.clip_by_global_norm(hp.max_grad_norm),
            optax.optimistic_adam_v2(
                learning_rate=lr_schedule, alpha=1.0, beta=hp.optimism, eps=1e-5,
                nesterov=False,
            ),
        )
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        def update_step(runner_state: RunnerState, unused) -> tuple[RunnerState, None]:
            def step(rs: RunnerState, unused) -> tuple[RunnerState, Transition]:
                rng_s, rng_action, rng_step = jax.random.split(rs.rng, 3)
                logits, value = network.apply(rs.train_state.params, rs.obs)
                action_mask = jax.vmap(env.get_avail_actions)(rs.state)
                logits = jnp.where(action_mask, logits, -jnp.inf)
                pi = distrax.Categorical(logits=logits)
                action = pi.sample(seed=rng_action)
                log_prob = pi.log_prob(action)
                next_state, next_obs, reward, absorbing, done, info = jax.vmap(env.step)(
                    jax.random.split(rng_step, num_envs), rs.state, action
                )
                truncated = done[:, None] & ~absorbing
                metric_carry, game_counts = jax.vmap(gm.step)(
                    rs.metric_carry, rs.state, action, next_state, reward, done, info,
                    jax.nn.softmax(logits, axis=-1),
                )
                game_return = rs.game_return + reward
                game_length = rs.game_length + 1
                transition = Transition(
                    obs=rs.obs, action_mask=action_mask, action=action, log_prob=log_prob,
                    reward=reward, done=done, truncated=truncated, value=value,
                    game_return=game_return, game_length=game_length, game_counts=game_counts,
                )
                rs = rs._replace(
                    state=next_state,
                    obs=next_obs,
                    game_return=jnp.where(done[:, None], 0.0, game_return),
                    game_length=jnp.where(done, 0, game_length),
                    metric_carry=metric_carry,
                    rng=rng_s,
                )
                return rs, transition

            runner_state, tr = lax.scan(step, runner_state, None, num_steps)
            ts = runner_state.train_state
            rng, rng_epochs, rng_eval = jax.random.split(runner_state.rng, 3)
            _, last_val = network.apply(ts.params, runner_state.obs)
            advantages, targets, valid = per_agent_gae(
                tr.value,
                tr.reward,
                jnp.broadcast_to(tr.done[..., None], tr.reward.shape),
                tr.truncated,
                last_val,
                hp.gamma,
                hp.gae_lambda,
            )
            ent_coef = ent_schedule(ts.step)

            def epoch(carry, unused):
                ts, rng = carry
                rng, rng_perm = jax.random.split(rng)
                data = (tr.obs, tr.action_mask, tr.action, tr.log_prob, tr.value,
                        advantages, targets, valid)
                data = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[3:]), data)
                perm = jax.random.permutation(rng_perm, batch)
                data = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, perm, axis=0).reshape(
                        (config["num_minibatches"], -1) + x.shape[1:]
                    ),
                    data,
                )

                def minibatch(ts, mb):
                    obs_b, mask_b, act_b, logp_old, v_old, adv_b, tgt_b, valid_b = mb
                    w = valid_b.astype(jnp.float32)
                    denom = jnp.maximum(w.sum(), 1.0)

                    def mean(x):
                        return (x * w).sum() / denom

                    def loss_fn(params):
                        logits, value = network.apply(params, obs_b)
                        pi = distrax.Categorical(logits=jnp.where(mask_b, logits, -jnp.inf))
                        logp = pi.log_prob(act_b)
                        adv_mean = mean(adv_b)
                        adv_std = jnp.sqrt(mean(jnp.square(adv_b - adv_mean)))
                        adv_n = (adv_b - adv_mean) / (adv_std + 1e-8)
                        logratio = jnp.where(valid_b, logp - logp_old, 0.0)
                        ratio = jnp.exp(logratio)
                        actor = mean(-jnp.minimum(
                            ratio * adv_n,
                            jnp.clip(ratio, 1.0 - hp.clip_eps, 1.0 + hp.clip_eps) * adv_n,
                        ))
                        entropy = mean(pi.entropy())
                        v_clip = v_old + jnp.clip(value - v_old, -config["vf_clip"], config["vf_clip"])
                        value_loss = 0.5 * mean(jnp.maximum(
                            jnp.square(value - tgt_b), jnp.square(v_clip - tgt_b)
                        ))
                        total = actor + hp.vf_coef * value_loss - ent_coef * entropy
                        return total, {
                            "total_loss": total,
                            "actor_loss": actor,
                            "value_loss": value_loss,
                            "entropy": entropy,
                            "kl_backward": mean((ratio - 1) - logratio),
                            "clip_frac": mean((jnp.abs(ratio - 1) > hp.clip_eps).astype(jnp.float32)),
                        }

                    (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
                    aux["grad_norm"] = pytree_norm(grads)
                    return ts.apply_gradients(grads=grads), aux

                ts, aux = lax.scan(minibatch, ts, data)
                return (ts, rng), aux

            (ts, _), loss_info = lax.scan(epoch, (ts, rng_epochs), None, config["num_epochs"])
            loss_info = jax.tree_util.tree_map(jnp.mean, loss_info)
            step_idx = runner_state.update_step
            last = step_idx == num_updates_total - 1

            # Evaluation against random.
            if config["compare"]:
                should_compare = (step_idx % config["compare_interval"] == 0) | last

                def compare():
                    return eval_metrics(
                        "vs_random",
                        evaluate_goofspiel(env, act, ts.params, act_random, None,
                                           config["compare_episodes"], rng_eval),
                    )

                template = jax.eval_shape(compare)
                compare_metrics = lax.cond(
                    should_compare, compare,
                    lambda: jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), template),
                )
            else:
                should_compare, compare_metrics = jnp.array(False), {}

            episodes = tr.done.sum()
            agent_end = jnp.broadcast_to(tr.done[..., None], tr.game_return.shape)
            metric = {
                "update_step": step_idx,
                "episodes_completed": episodes,
                "ep_length_avg": jnp.where(tr.done, tr.game_length, 0).sum() / jnp.maximum(episodes, 1),
                "return_abs_per_episode": (jnp.abs(tr.game_return) * agent_end).sum()
                / jnp.maximum(agent_end.sum(), 1),
                "lr": lr_schedule(ts.step),
                "ent_coef": ent_coef,
                "evaluated": should_compare,
            }
            metric.update(compare_metrics)
            metric.update(loss_info)
            metric.update(prefixed("game", gm.summarize(
                masked_totals(tr.game_counts, jnp.ones(tr.game_return.shape, bool))
            )))
            if exact_tables is not None:
                should_exact = (step_idx % config["exact_eval_interval"] == 0) | last
                metric["exact_evaluated"] = should_exact

                def probs(t):
                    return GE.policy_probs(
                        GE.apply_in_chunks(lambda o: network.apply(ts.params, o)[0], t.obs),
                        t.legal,
                    )

                def compute():
                    p = probs(exact_tables)
                    return {
                        "exploitability": GE.exploitability(exact_tables, p),
                        "value_vs_random": GE.value_vs_random(exact_tables, p),
                    }

                template = jax.eval_shape(compute)
                metric.update(lax.cond(
                    should_exact, compute,
                    lambda: jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), template),
                ))
            io_callback(functools.partial(logging_callback, config=config), None, seed, metric)

            take_half = step_idx + 1 == num_half
            half = jax.tree_util.tree_map(
                lambda new, old: jnp.where(take_half, new, old), ts.params, runner_state.half
            )
            runner_state = runner_state._replace(
                train_state=ts, half=half, update_step=step_idx + 1, rng=rng
            )
            return runner_state, None

        init = RunnerState(
            train_state=train_state,
            state=state,
            obs=obs,
            game_return=jnp.zeros((num_envs, num_agents), jnp.float32),
            game_length=jnp.zeros((num_envs,), jnp.int32),
            metric_carry=jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (num_envs,) + x.shape), gm.init_carry()
            ),
            half=params,
            update_step=jnp.array(0, jnp.int32),
            rng=rng,
        )
        final, _ = lax.scan(
            update_step, init, None, num_updates_total if num_updates is None else num_updates
        )
        return final

    return train


def logging_callback(seed_val, metric_dict, config) -> None:
    metric_dict = dict(metric_dict)
    if INSTANCES:
        metric_dict.update(INSTANCES[int(seed_val)])
    log_update(RESULTS_LOG, EVALS, seed_val, metric_dict, config, EXACT, GE.EXACT_KEYS)


def print_sweep_summary(insts: list[dict]) -> None:
    """Final exploitability of every grid point (mean, std over its seeds)."""
    if not EXACT:
        return
    by_grid: dict[int, list[float]] = {}
    for i, inst in enumerate(insts):
        if i in EXACT:
            by_grid.setdefault(int(inst["hp/grid_index"]), []).append(
                EXACT[i][-1][1]["exploitability"]
            )
    sweep_keys = sorted({k for i in insts for k in i if k.startswith("hp/")} - {"hp/seed", "hp/grid_index"})
    for g, vals in sorted(by_grid.items(), key=lambda kv: np.mean(kv[1])):
        point = next(i for i in insts if i["hp/grid_index"] == g)
        hps = " ".join(f"{k[3:]}={point[k]:g}" for k in sweep_keys)
        print(f"grid {g}: final exploitability {np.mean(vals):.4f} ± {np.std(vals):.4f} | {hps}")


@hydra.main(version_base=None, config_path="./", config_name="config_optppo")
def main(config: dict) -> None:
    global RESULTS_LOG
    wandb_runs = []
    try:
        config = OmegaConf.to_container(config, resolve=True)
        config["num_update_steps"] = int(
            config["num_timesteps"] // config["num_envs"] // config["num_steps_per_env_per_update"]
        )
        config["half_update_steps"] = half_update_steps(config["num_update_steps"])
        insts = instances(config)
        INSTANCES[:] = insts
        for log in (EVALS, EXACT):
            log.clear()
        train = make_train(config)
        run_dir = make_run_dir(config)
        with open(os.path.join(run_dir, "sweep.json"), "w") as f:
            json.dump(insts, f, indent=1)
        group = config["env_name"] + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        wandb_runs = [
            wandb.init(
                project=config["project"], group=group,
                job_type=f"{config['job_type']}_{config['env_name']}",
                name=f"{config['seed']}_{i}", config={**config, **inst},
                mode=None if config["wandb"] else "disabled", dir=str(REPO_ROOT),
                reinit="create_new",
            )
            for i, inst in enumerate(insts)
        ]
        RESULTS_LOG = ResultsLog(run_dir, wandb_runs)
        print(f"Run directory: {run_dir}; {len(insts)} instance(s)")

        exact_tables = GE.tables_for_training(config, make("goofspiel", **config["env_kwargs"]))
        print("Exact exploitability: " + (
            f"{len(exact_tables.swap)} decision states, every {config['exact_eval_interval']} updates"
            if exact_tables is not None else "off"
        ))
        # Instance i: seed key i % num_seeds, so grid points share seeds.
        seed_keys = jax.random.split(jax.random.PRNGKey(config["seed"]), config["num_seeds"])
        rngs = seed_keys[jnp.asarray([int(i["hp/seed"]) for i in insts])]
        ids = jnp.arange(len(insts))
        hp = hparams_of(insts)
        # At most max_vmap_instances in one program (GPU memory), in chunks
        # of equal size where possible.
        num_chunks = -(-len(insts) // config["max_vmap_instances"])
        size = -(-len(insts) // num_chunks)
        train_vjit = jax.jit(
            jax.vmap(lambda r, s, h, t: train(r, s, h, t), in_axes=(0, 0, 0, None))
        )
        start = time.time()
        finals = []
        for c in range(num_chunks):
            sl = slice(c * size, min(len(insts), (c + 1) * size))
            args = (rngs[sl], ids[sl], jax.tree_util.tree_map(lambda x: x[sl], hp), exact_tables)
            finals.append(jax.block_until_ready(train_vjit(*args)))
            jax.effects_barrier()
            print(f"Chunk {c + 1}/{num_chunks} (instances {sl.start}..{sl.stop - 1}) "
                  f"done after {time.time() - start:.1f} s", flush=True)
        final = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs), *finals)
        env_steps = config["num_update_steps"] * config["num_envs"] * config["num_steps_per_env_per_update"]
        print(f"Trained {len(insts)} instance(s) x {env_steps} env steps in {time.time() - start:.1f} s")
        print_final_evals(EVALS)
        print_final_exact(EXACT)
        print_sweep_summary(insts)
        if config["save_checkpoints"]:
            for stage, params in (("half", final.half), ("full", final.train_state.params)):
                save_seed_params(run_dir, stage, "policy", params, range(len(insts)))
            print(f"Saved checkpoints to {run_dir}")
    finally:
        for run in wandb_runs:
            run.finish()
        print("Finished.")


if __name__ == "__main__":
    register_resolvers()
    main()
