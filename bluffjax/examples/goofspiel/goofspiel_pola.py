"""
POLA-DiCE (Zhao et al., 2022) self-play for two-player Goofspiel.

A port of rrmarl-pola-new's SMAX POLA-DiCE (src/rrmarl/jax/experiments/smax/
pola/pola_new/pola_new.py, fresh rollouts only) to the zero-sum,
simultaneous-move, shared-network setting of goofspiel_ppo.py. Everything that
is not the actor update (rollout, per-agent GAE, critic loss, schedules,
sweeps, evaluation, logging, checkpoints) is goofspiel_reppo.py's.

Each update approximates POLA's two nested proximal problems,

    phi*(theta') = argmax_phi    J_opp(theta', phi) - beta_in  * KL(pi_phi    || pi_old)
    theta+       = argmax_theta' J_self(theta', phi*(theta'))
                                 - beta_out * KL(pi_theta' || pi_old) + ent_coef * H(pi_theta')

where in each env one seat is self (plays theta') and the other the opponent
(plays phi, a copy of the shared actor), J_self and J_opp are the two seats'
expected returns (zero-sum: J_opp = -J_self in expectation, but each is
estimated with its own seat's advantages), old is the network at the start of
the update (theta_old), and the KL and entropy terms are means over the
decisions of the update's base rollout (every seat). The objectives are
averaged over the choice of self seat (goofspiel_dice.seat_dice), so every
decision carries the direct policy gradient and the shaping term flows
through phi.

- Lookahead: phi*(theta') is inner_steps steps from theta_old of the actor's
  own optimizer (global-norm clipping and Adam with the train state's current
  Adam state and learning rate; a copy with eps_root = 1e-12, since
  differentiating through Adam's sqrt at a zero second moment is NaN), each
  step scaled by inner_scale, on J_opp minus the beta_in KL. It stays
  differentiable in theta', and the gradient that flows through phi*(theta')
  is the opponent-shaping term.
- Outer update: num_epochs x num_minibatches Adam steps on theta', each on
  -(J_self - beta_out KL + ent_coef H) + vf_coef * value loss, differentiating
  through the lookahead (recomputed from theta_old on the minibatch's envs).
  No ratio clipping; the critic (its own trunk in the shared params) is
  trained on the base rollout by goofspiel_ppo.py's clipped value loss
  (vf_clip).
- Objectives: Loaded DiCE (goofspiel_dice.py) on per-seat GAE advantages
  from the critic (parallel_nfsp_common.per_agent_gae), normalised per
  minibatch over every valid decision of both seats, importance-corrected
  against whichever policy sampled the data, with episode starts from the done
  flags (envs auto-reset inside step; games last N steps) and dependence on
  earlier actions discounted by dice_lambda. Minibatches split environments,
  so every column keeps its whole sequence.
- Fresh rollouts (POLA proper; inner_steps + 1 rollouts per update): rollout
  B0 is self-play at theta_old; inner step 1 uses B0, inner step m > 1 a new
  rollout B_{m-1} with the opponent at phi_{m-1} (the lookahead of theta_old
  through m - 1 steps, on the whole rollouts), and the outer objective a new
  rollout B_M with the opponent at phi_M. In B_1..B_M one random seat per env
  plays theta_old, the other phi (fixed for the update; Goofspiel's seats are
  symmetric and observations relative to the observer, so the choice only
  decorrelates seat and role). fresh_self_seats: both (pola_new: average J
  over both choices of self seat in every rollout, the choice that did not
  sample the data importance-weighted) or sampled (only the seat that played
  theta_old is self in B_1..B_M). The KL, entropy and value terms use B0.
- inner_scale 0 removes the lookahead (phi = theta_old, shaping gradient 0):
  the update is PPO with a KL penalty and no clipping. beta_in = beta_out = 0
  with inner_steps 1 is LOLA-DiCE.

Budget. num_timesteps counts every env sample, fresh rollouts included:

    num_update_steps = num_timesteps // (num_envs * num_steps_per_env_per_update
                                         * (inner_steps + 1))

and env_steps (logged) counts all inner_steps + 1 rollouts of every update.
A minibatch step counts as two backward passes (the lookahead's inner
gradient and the outer gradient through it), so with inner_steps 1 POLA and
PPO at the same num_timesteps make the same number of backward passes: 1e9
samples are 7,629 updates (1024 envs x 64 steps x 2 rollouts) x 4 epochs x 4
minibatches x 2 = 244,128, PPO's 15,258 x 4 x 4. (With inner_steps M a
minibatch step makes M + 1, so the match holds for M = 1 only.) The shaping
diagnostics (shaping_diagnostics: true) add an uncounted backward pass.

Knobs (scalar ones are sweepable, see SWEEPABLE):
- inner_steps (static): lookahead steps M (M + 1 rollouts per update).
- inner_scale: the lookahead step as a multiple of one actor Adam step.
- beta_in, beta_out: proximal KL coefficients of the opponent and of self.
- dice_lambda: DiCE's discount on dependence on earlier actions.
- ent_coef, ent_final_frac: entropy bonus of the outer objective (pola_new
  has none; self-play collapses to deterministic bids without it).
- fresh_self_seats (static): both | sampled, above.
- shaping_diagnostics (static): also compute the direct gradient (the
  opponent frozen at phi*) to log the shaping gradient.

With inner_steps 1, beta_in has no effect: the one lookahead step is taken at
phi = theta_old, where the gradient of KL(pi_phi || pi_old) is 0 and does not
depend on theta'. It matters from inner_steps 2.

Logged per update (means over the minibatch steps) on top of
goofspiel_ppo.py's rollout, game, exact and vs-random metrics:
dice_objective (J_self at phi*), inner_objective (J_opp at phi* on the last
inner rollout), inner_gain (J_opp at phi* minus at theta_old on B0), kl_theta
(KL(pi_theta' || pi_old) on the minibatch, before each step), kl_theta_after
(after the update, on B0), kl_phi (KL(pi_phi* || pi_old)), entropy,
value_loss, actor_loss, total_loss, grad_norm, and with shaping_diagnostics
direct_grad_norm (the actor loss's gradient with the opponent frozen at
phi*), shaping_grad_norm (the total gradient minus that and the value loss's
gradient: what flows through phi*(theta')) and shaping_cos (their cosine);
all before clipping.

Exact evaluation, evaluation against random, game metrics (of B0, the
self-play rollout), sweeps, run directory, results.jsonl, sweep.json and the
half/full checkpoints are as in goofspiel_reppo.py (see its docstring).
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
from bluffjax.examples.goofspiel import goofspiel_dice as D
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
    "vf_coef",
    "max_grad_norm",
    "inner_scale",
    "beta_in",
    "beta_out",
    "dice_lambda",
)

RESULTS_LOG: ResultsLog | None = None
EVALS: dict[int, list[tuple[int, dict[str, float]]]] = {}
EXACT: dict[int, list[tuple[int, dict[str, float]]]] = {}
INSTANCES: list[dict[str, float]] = []  # hp/<name> of every instance

# Logit of an illegal bid in the losses (finite, so masked KL and entropy
# terms have finite derivatives; exp of it is exactly 0 in float32).
MASKED_LOGIT = -1e9


class HParams(NamedTuple):
    lr: FloatArray
    lr_final_frac: FloatArray
    ent_coef: FloatArray
    ent_final_frac: FloatArray
    gamma: FloatArray
    gae_lambda: FloatArray
    vf_coef: FloatArray
    max_grad_norm: FloatArray
    inner_scale: FloatArray
    beta_in: FloatArray
    beta_out: FloatArray
    dice_lambda: FloatArray


class Transition(NamedTuple):
    """One env step of every env; agent fields are (num_envs, num_agents, ...)."""

    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    log_prob: FloatArray  # under whichever parameters sampled the bid
    reward: FloatArray
    done: BoolArray  # (num_envs,)
    truncated: BoolArray
    value: FloatArray
    game_return: FloatArray
    game_length: IntArray
    game_counts: Counts


class Rollout(NamedTuple):
    """One rollout, ready for the losses: leaves (T, num_envs, num_agents,
    ...) except starts (T, num_envs) and self_cols (num_envs, num_agents)."""

    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    log_prob: FloatArray
    value: FloatArray
    adv: FloatArray  # per-seat GAE, 0 where invalid
    targets: FloatArray
    valid: BoolArray
    starts: BoolArray  # a game begins at step t (and at the rollout's first step)
    self_cols: FloatArray  # weight of each choice of self seat in the DiCE mean


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


def rollouts_per_update(config: dict) -> int:
    return config["inner_steps"] + 1


def num_update_steps(config: dict) -> int:
    """Updates in num_timesteps env samples, every rollout counted."""
    return int(
        config["num_timesteps"]
        // (config["num_envs"] * config["num_steps_per_env_per_update"] * rollouts_per_update(config))
    )


def make_train(config: dict) -> Callable[..., RunnerState]:
    env = make("goofspiel", **config["env_kwargs"])
    num_agents = env.num_agents
    action_dim = env.num_actions
    num_envs = config["num_envs"]
    num_steps = config["num_steps_per_env_per_update"]
    num_updates_total = config["num_update_steps"]
    num_mb = config["num_minibatches"]
    if num_envs % num_mb:
        raise ValueError("num_minibatches must divide num_envs (minibatches split environments)")
    inner_steps = config["inner_steps"]
    num_gradient_steps = num_updates_total * config["num_epochs"] * num_mb
    num_half = half_update_steps(num_updates_total)
    gm = make_metrics("goofspiel", env)
    network = ActorCriticDiscreteMLP(action_dim=action_dim, hidden_dim=config["fc_dim_size"])
    # env_steps (run_utils.env_steps_after) counts every rollout of an update
    log_config = {
        **config,
        "num_steps_per_env_per_update": num_steps * rollouts_per_update(config),
    }

    def act(params, obs, mask, rng):
        return sample_logits(network.apply(params, obs)[0], mask, rng)

    def act_random(params, obs, mask, rng):
        return sample_random_legal(mask, rng)

    def log_pi(params, obs, mask):
        """Log-probabilities over the bids, illegal ones at exp() = 0."""
        return jax.nn.log_softmax(jnp.where(mask, network.apply(params, obs)[0], MASKED_LOGIT))

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

        def make_tx(eps_root):
            return optax.chain(
                optax.clip_by_global_norm(hp.max_grad_norm),
                optax.adam(learning_rate=lr_schedule, eps=1e-5, eps_root=eps_root),
            )

        tx = make_tx(0.0)
        # The same optimizer with a small eps_root, so that differentiating
        # through its step is finite (the lookahead; same state structure).
        lookahead_tx = make_tx(1e-12)

        rng, rng_reset, rng_init = jax.random.split(rng, 3)
        state, obs = jax.vmap(env.reset)(jax.random.split(rng_reset, num_envs))
        params = network.init(rng_init, obs)
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        # ------------------------------------------------------------ losses

        def masked_mean(x, valid):
            w = valid.astype(jnp.float32)
            return (x * w).sum() / jnp.maximum(w.sum(), 1.0)

        def kl_to_old(params, b, old_log_pi, lp=None):
            """Mean KL(pi_params || pi_old) over b's valid decisions (lp:
            log_pi(params) on b, if already computed)."""
            if lp is None:
                lp = log_pi(params, b.obs, b.action_mask)
            kl = jnp.where(b.action_mask, jnp.exp(lp) * (lp - old_log_pi), 0.0).sum(-1)
            return masked_mean(kl, b.valid)

        def norm_adv(b):
            """Per-seat advantages normalised over b's valid decisions."""
            mean = masked_mean(b.adv, b.valid)
            std = jnp.sqrt(masked_mean(jnp.square(b.adv - mean), b.valid))
            return lax.stop_gradient(jnp.where(b.valid, (b.adv - mean) / (std + 1e-8), 0.0))

        def seat_objective(self_params, opp_params, b, role):
            """J by DiCE with self_params at the self seat and opp_params at the
            other, averaged over the choices of self seat (b.self_cols): the
            self seat's (role self) or the opponent's (role opponent) return."""

            def log_ratio(params):
                lp = jnp.take_along_axis(
                    log_pi(params, b.obs, b.action_mask), b.action[..., None], axis=-1
                )[..., 0]
                return jnp.where(b.valid, lp - b.log_prob, 0.0)

            adv = norm_adv(b)
            if role == "opponent":
                adv = D.opponent_advantage(adv)
            return D.seat_dice(
                log_ratio(self_params), log_ratio(opp_params), adv, b.starts,
                hp.dice_lambda, b.self_cols,
            )

        def inner_objective(phi, theta, b, kl_b, old_log_pi):
            return seat_objective(theta, phi, b, "opponent") - hp.beta_in * kl_to_old(
                phi, kl_b, old_log_pi
            )

        def lookahead(theta, theta_old, inner, kl_b, old_log_pi, opt_state):
            """The opponent's proximal update as a function of theta: from
            theta_old, one step of the actor's own optimizer (its state
            opt_state, step scaled by inner_scale) on its objective per rollout
            in `inner`. Differentiable in theta, so its dependence on theta
            carries the shaping term."""
            phi = theta_old
            for b in inner:
                g = jax.grad(inner_objective)(phi, theta, b, kl_b, old_log_pi)
                neg = jax.tree_util.tree_map(lambda x: -x, g)  # optax minimises; phi ascends
                updates, opt_state = lookahead_tx.update(neg, opt_state, phi)
                phi = jax.tree_util.tree_map(lambda p, u: p + hp.inner_scale * u, phi, updates)
            return phi

        def policy_terms(params, phi, base, old_log_pi, outer, ent_coef):
            """-(J_self at phi - beta_out KL + ent_coef H): the actor's loss."""
            obj = seat_objective(params, phi, outer, "self")
            lp = log_pi(params, base.obs, base.action_mask)
            kl = kl_to_old(params, base, old_log_pi, lp)
            entropy = masked_mean(
                -jnp.where(base.action_mask, jnp.exp(lp) * lp, 0.0).sum(-1), base.valid
            )
            actor = -(obj - hp.beta_out * kl)
            return actor - ent_coef * entropy, {
                "actor_loss": actor,
                "dice_objective": obj,
                "kl_theta": kl,
                "entropy": entropy,
            }

        def value_terms(params, base):
            """vf_coef * goofspiel_ppo.py's clipped value loss on base."""
            _, value = network.apply(params, base.obs)
            v_old = base.value
            v_clip = v_old + jnp.clip(value - v_old, -config["vf_clip"], config["vf_clip"])
            value_loss = 0.5 * masked_mean(
                jnp.maximum(jnp.square(value - base.targets), jnp.square(v_clip - base.targets)),
                base.valid,
            )
            return hp.vf_coef * value_loss, value_loss

        # ------------------------------------------------------------ data

        def collect(rs: RunnerState, phi, self_seat) -> tuple[RunnerState, Transition]:
            """One rollout of num_steps steps. The seats where self_seat (bool,
            (num_envs, num_agents)) is True bid with the train state's
            parameters, the others with phi (phi None: every seat with the
            train state's parameters)."""
            params = rs.train_state.params

            def step(rs: RunnerState, unused) -> tuple[RunnerState, Transition]:
                rng_s, rng_action, rng_step = jax.random.split(rs.rng, 3)
                logits, value = network.apply(params, rs.obs)
                if phi is not None:
                    logits = jnp.where(self_seat[..., None], logits, network.apply(phi, rs.obs)[0])
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

            return lax.scan(step, rs, None, num_steps)

        def to_rollout(rs: RunnerState, tr: Transition, self_cols) -> Rollout:
            """Per-seat GAE (the critic of the train state) and episode starts."""
            _, last_val = network.apply(rs.train_state.params, rs.obs)
            advantages, targets, valid = per_agent_gae(
                tr.value,
                tr.reward,
                jnp.broadcast_to(tr.done[..., None], tr.reward.shape),
                tr.truncated,
                last_val,
                hp.gamma,
                hp.gae_lambda,
            )
            starts = jnp.concatenate([jnp.ones_like(tr.done[:1]), tr.done[:-1]], axis=0)
            return Rollout(
                obs=tr.obs, action_mask=tr.action_mask, action=tr.action, log_prob=tr.log_prob,
                value=tr.value, adv=advantages, targets=targets, valid=valid,
                starts=starts.astype(bool), self_cols=self_cols,
            )

        def env_slice(b: Rollout, idx) -> Rollout:
            """Keep the environments in idx."""
            sliced = jax.tree_util.tree_map(lambda x: jnp.take(x, idx, axis=1), b)
            return sliced._replace(self_cols=jnp.take(b.self_cols, idx, axis=0))

        # ------------------------------------------------------------ update

        def update_step(runner_state: RunnerState, unused) -> tuple[RunnerState, None]:
            ts = runner_state.train_state
            theta_old = lax.stop_gradient(ts.params)
            opt_old = lax.stop_gradient(ts.opt_state)  # the opponent's optimizer state
            all_cols = jnp.ones((num_envs, num_agents), jnp.float32)

            # B0: self-play at theta_old. It defines the KL terms, trains the
            # critic and gives the update's metrics.
            runner_state, tr = collect(runner_state, None, None)
            base = to_rollout(runner_state, tr, all_cols)
            old_log_pi = lax.stop_gradient(log_pi(theta_old, base.obs, base.action_mask))

            # Fresh rollouts: one random seat per env plays theta_old, the other phi_m.
            rng, rng_seat = jax.random.split(runner_state.rng)
            runner_state = runner_state._replace(rng=rng)
            self_seat = jax.nn.one_hot(
                jax.random.randint(rng_seat, (num_envs,), 0, num_agents), num_agents
            ).astype(bool)
            fresh_cols = all_cols if config["fresh_self_seats"] == "both" else self_seat.astype(jnp.float32)
            inner = [base]
            for m in range(1, inner_steps + 1):
                phi_m = lax.stop_gradient(
                    lookahead(theta_old, theta_old, inner, base, old_log_pi, opt_old)
                )
                runner_state, tr_m = collect(runner_state, phi_m, self_seat)
                fresh = to_rollout(runner_state, tr_m, fresh_cols)
                if m < inner_steps:
                    inner.append(fresh)
            outer = fresh

            rng, rng_epochs, rng_eval = jax.random.split(runner_state.rng, 3)
            ent_coef = ent_schedule(ts.step)
            mb_envs = num_envs // num_mb

            def epoch(carry, unused):
                ts, rng = carry
                rng, rng_perm = jax.random.split(rng)
                perm = jax.random.permutation(rng_perm, num_envs).reshape(num_mb, mb_envs)

                def minibatch(ts, idx):
                    base_b = env_slice(base, idx)
                    old_b = jnp.take(old_log_pi, idx, axis=1)
                    inner_b = [env_slice(b, idx) for b in inner]
                    outer_b = env_slice(outer, idx)

                    def loss_fn(params):
                        phi = lookahead(params, theta_old, inner_b, base_b, old_b, opt_old)
                        actor, aux = policy_terms(params, phi, base_b, old_b, outer_b, ent_coef)
                        critic, aux["value_loss"] = value_terms(params, base_b)
                        aux["total_loss"] = actor + critic
                        return actor + critic, (aux, phi)

                    (_, (aux, phi)), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
                    phi_sg = lax.stop_gradient(phi)
                    aux["grad_norm"] = pytree_norm(grads)
                    aux["kl_phi"] = kl_to_old(phi_sg, base_b, old_b)
                    aux["inner_objective"] = seat_objective(ts.params, phi_sg, inner_b[-1], "opponent")
                    aux["inner_gain"] = seat_objective(
                        theta_old, phi_sg, inner_b[0], "opponent"
                    ) - seat_objective(theta_old, theta_old, inner_b[0], "opponent")
                    if config["shaping_diagnostics"]:
                        # The direct policy gradient (the opponent frozen at
                        # phi*) and the shaping remainder; the critic's trunk
                        # gets the value-loss gradient only.
                        direct = jax.grad(
                            lambda p: policy_terms(p, phi_sg, base_b, old_b, outer_b, ent_coef)[0]
                        )(ts.params)
                        value_grads = jax.grad(lambda p: value_terms(p, base_b)[0])(ts.params)
                        shaping = jax.tree_util.tree_map(
                            lambda g, d, v: g - d - v, grads, direct, value_grads
                        )
                        dot = sum(
                            (a * b).sum()
                            for a, b in zip(jax.tree_util.tree_leaves(shaping),
                                            jax.tree_util.tree_leaves(direct))
                        )
                        aux["direct_grad_norm"] = pytree_norm(direct)
                        aux["shaping_grad_norm"] = pytree_norm(shaping)
                        aux["shaping_cos"] = dot / jnp.maximum(
                            aux["direct_grad_norm"] * aux["shaping_grad_norm"], 1e-12
                        )
                    return ts.apply_gradients(grads=grads), aux

                ts, aux = lax.scan(minibatch, ts, perm)
                return (ts, rng), aux

            (ts, _), loss_info = lax.scan(epoch, (ts, rng_epochs), None, config["num_epochs"])
            loss_info = jax.tree_util.tree_map(jnp.mean, loss_info)
            loss_info["kl_theta_after"] = kl_to_old(ts.params, base, old_log_pi)
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
            io_callback(functools.partial(logging_callback, config=log_config), None, seed, metric)

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


@hydra.main(version_base=None, config_path="./", config_name="config_pola")
def main(config: dict) -> None:
    global RESULTS_LOG
    wandb_runs = []
    try:
        config = OmegaConf.to_container(config, resolve=True)
        if config["inner_steps"] < 1:
            raise ValueError(f"inner_steps must be >= 1, got {config['inner_steps']}")
        if config["fresh_self_seats"] not in ("both", "sampled"):
            raise ValueError(f"unknown fresh_self_seats {config['fresh_self_seats']!r}")
        if config["env_kwargs"]["num_agents"] != 2:
            raise ValueError("goofspiel_pola.py is for two players")
        config["num_update_steps"] = num_update_steps(config)
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
        print(f"Run directory: {run_dir}; {len(insts)} instance(s), "
              f"{config['num_update_steps']} updates of {rollouts_per_update(config)} rollouts")

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
        # Chunks of unequal size compile separately, so their static fields (the optimizer's
        # functions) differ: merge the arrays under the first chunk's structure.
        final = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(finals[0]),
            [jnp.concatenate(xs) for xs in zip(*(jax.tree_util.tree_leaves(f) for f in finals))],
        )
        env_steps = (config["num_update_steps"] * config["num_envs"]
                     * config["num_steps_per_env_per_update"] * rollouts_per_update(config))
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
