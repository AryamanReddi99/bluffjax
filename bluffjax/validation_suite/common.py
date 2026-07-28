"""
Shared utilities for the per-game rule-conformance validation suites
(bluffjax/validation_suite/<game>_validation.py).

Provides the agent-loading, checkpoint-discovery, and rollout-driving
machinery common to every game's validation script. Game-specific rule
checks and edge-case tests live in each game's own file; only the
game-agnostic plumbing lives here.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from flax import serialization

from bluffjax.networks.mlp import ActorCriticDiscreteMLP, ActorDiscreteMLP, QNetworkDiscreteMLP

RUN_DIR_RE = re.compile(r"^(ppo_nfsp|pqn_nfsp)_.*_s\d+$")


# ---------------------------------------------------------------------------
# Agents: uniform-random legal-action policy, or a loaded checkpoint network.
# ---------------------------------------------------------------------------


def sample_action(
    kind: str,
    network: Optional[Any],
    params: Optional[Any],
    obs: jnp.ndarray,
    avail: jnp.ndarray,
    rng: jnp.ndarray,
) -> jnp.ndarray:
    """Pick a legal action for one of the supported agent kinds.

    kind == "random": uniform over legal actions.
    kind == "actor"/"actor_critic": sample the policy's categorical logits,
        masked to legal actions (how avg-policy / PPO-BR checkpoints are
        ordinarily evaluated).
    kind == "q_network": greedy w.r.t. Q-values among legal actions, ties
        broken uniformly (mirrors <game>_pqn_nfsp.py's sample_action_q_greedy).
    """
    if kind == "random":
        logits = jnp.where(avail, 0.0, -1e9)
        return jax.random.categorical(rng, logits)
    if kind in ("actor", "actor_critic"):
        out = network.apply(params, obs)
        logits = out[0] if isinstance(out, tuple) else out
        logits = jnp.where(avail, logits, -jnp.inf)
        return jax.random.categorical(rng, logits)
    if kind == "q_network":
        q_vals = network.apply(params, obs.astype(jnp.float32))
        q_vals_masked = jnp.where(avail, q_vals, -jnp.inf)
        best_val = jnp.max(q_vals_masked)
        ties = (q_vals_masked == best_val) & avail
        tie_logits = jnp.where(ties, 0.0, -1e9)
        return jax.random.categorical(rng, tie_logits)
    raise ValueError(f"Unknown agent kind: {kind}")


def infer_network_kind(checkpoint_path: str) -> str:
    """Infer whether a checkpoint is an avg policy ("actor"), a PPO best
    response ("actor_critic"), or a PQN best response ("q_network"), from the
    save-path convention used by every <game>_ppo_nfsp.py / <game>_pqn_nfsp.py
    training script: `<run_dir>/{avg,br}_<frac_tag>.msgpack` where run_dir is
    prefixed `ppo_nfsp_` or `pqn_nfsp_`."""
    fname = os.path.basename(checkpoint_path)
    parent = os.path.basename(os.path.dirname(checkpoint_path))
    haystack = f"{parent}/{fname}"
    if fname.startswith("avg") or "_avg" in fname:
        return "actor"
    if "ppo" in haystack:
        return "actor_critic"
    if "pqn" in haystack:
        return "q_network"
    raise ValueError(
        f"Cannot infer network kind for '{checkpoint_path}'; pass network_type explicitly."
    )


def build_network(kind: str, action_dim: int, hidden_dim: int = 128):
    if kind == "actor":
        return ActorDiscreteMLP(action_dim=action_dim, hidden_dim=hidden_dim)
    if kind == "actor_critic":
        return ActorCriticDiscreteMLP(action_dim=action_dim, hidden_dim=hidden_dim)
    if kind == "q_network":
        return QNetworkDiscreteMLP(action_dim=action_dim, hidden_dim=hidden_dim)
    raise ValueError(f"Unknown network kind: {kind}")


def load_checkpoint_params(network, sample_obs: jnp.ndarray, checkpoint_path: str):
    template_params = network.init(jax.random.PRNGKey(0), sample_obs)
    with open(checkpoint_path, "rb") as f:
        return serialization.from_bytes(template_params, f.read())


def discover_checkpoints(checkpoint_root: str) -> list[tuple[str, str]]:
    """Find (checkpoint_path, network_kind) pairs under the standard
    `ppo_nfsp_*_s<seed>` / `pqn_nfsp_*_s<seed>` checkpoint directories, so the
    checkpoint-driven tests can run automatically wherever training outputs
    happen to exist, and skip cleanly where they don't (e.g. a game whose
    checkpoints haven't been generated yet)."""
    found = []
    if not os.path.isdir(checkpoint_root):
        return found
    for d in sorted(os.listdir(checkpoint_root)):
        run_dir = os.path.join(checkpoint_root, d)
        if not (os.path.isdir(run_dir) and RUN_DIR_RE.match(d)):
            continue
        for fname in sorted(os.listdir(run_dir)):
            if not fname.endswith(".msgpack"):
                continue
            path = os.path.join(run_dir, fname)
            try:
                kind = infer_network_kind(path)
            except ValueError:
                continue
            found.append((path, kind))
    return found


def one_checkpoint_per_algorithm_and_kind(
    checkpoints: list[tuple[str, str]], checkpoint_root: str
):
    """Dedupe discovered checkpoints down to one representative per
    (algorithm, network-kind) combo, so the default pytest run stays fast --
    exhaustive per-checkpoint validation is available via each script's
    standalone CLI. Returns a list of pytest.param(...) ready for
    parametrize."""
    seen: set[tuple[str, str]] = set()
    params = []
    for path, kind in checkpoints:
        algorithm = "ppo_nfsp" if "ppo" in os.path.basename(os.path.dirname(path)) else "pqn_nfsp"
        key = (algorithm, kind)
        if key in seen:
            continue
        seen.add(key)
        params.append(
            pytest.param((path, kind, algorithm), id=os.path.relpath(path, checkpoint_root))
        )
    return params


def training_env_kwargs(examples_dir: str, algorithm: str) -> tuple[dict, int]:
    """Load env_kwargs/fc_dim_size from the config used to train checkpoints,
    so a loaded checkpoint's network is applied to observations of the exact
    shape/size it was trained on."""
    cfg_name = "config_ppo_nfsp.yaml" if algorithm == "ppo_nfsp" else "config_pqn_nfsp.yaml"
    with open(os.path.join(examples_dir, cfg_name), "r") as f:
        config = yaml.safe_load(f)
    return config["env_kwargs"], config["fc_dim_size"]


# ---------------------------------------------------------------------------
# Violation bookkeeping shared by every game's RuleChecker.
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    episode: int
    step: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"[episode {self.episode}, step {self.step}] ({self.rule}) {self.message}"


class RuleCheckerBase:
    """Common violation-collection machinery. Game-specific subclasses add
    their own `_check_*` methods and a `validate_transition`/`validate_pre_step`
    dispatcher."""

    def __init__(self, env):
        self.env = env
        self.violations: list[Violation] = []
        self._episode = -1

    def start_episode(self) -> None:
        self._episode += 1

    def _fail(self, step: int, rule: str, message: str) -> None:
        self.violations.append(Violation(self._episode, step, rule, message))

    def assert_no_violations(self, extra_context: str = "") -> None:
        if self.violations:
            preview = "\n".join(str(v) for v in self.violations[:20])
            more = (
                f"\n... and {len(self.violations) - 20} more" if len(self.violations) > 20 else ""
            )
            raise AssertionError(
                f"{len(self.violations)} rule violation(s) found{extra_context}:\n{preview}{more}"
            )


# ---------------------------------------------------------------------------
# Rollout drivers: AEC (one acting agent per step) and Parallel (all agents
# act simultaneously every step, e.g. Goofspiel/Kemps).
# ---------------------------------------------------------------------------


def rollout_and_validate_aec(
    env,
    kind: str,
    network: Optional[Any],
    params: Optional[Any],
    num_episodes: int,
    checker,
    seed: int = 0,
):
    """Runs `num_episodes` full episodes of an AECEnv with the given agent
    (all seats controlled by the same policy, i.e. self-play), validating
    every transition against `checker`. Returns `checker`.

    Calls `env.step_env` directly rather than the public `env.step`, since
    `AECEnv.step` auto-resets on `done` (substituting in an unrelated fresh
    episode's state at exactly the one transition -- a win, or a horizon
    truncation -- most in need of checking). Episode boundaries are handled
    explicitly here instead.
    """
    rng = jax.random.PRNGKey(seed)

    for _ in range(num_episodes):
        checker.start_episode()
        rng, reset_rng = jax.random.split(rng)
        state, obs = env.reset(reset_rng)
        if hasattr(checker, "check_reset"):
            checker.check_reset(state)

        for step in range(env.horizon + 1):
            if bool(state.done):
                break
            if hasattr(checker, "validate_pre_step"):
                checker.validate_pre_step(step, state)

            avail = env.get_avail_actions(state)
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            action = sample_action(kind, network, params, obs, avail, act_rng)
            next_state, next_obs, reward, absorbing, done, info = env.step_env(
                step_rng, state, action
            )

            checker.validate_transition(
                step, state, int(action), avail, next_state, reward, bool(done)
            )
            state, obs = next_state, next_obs

    return checker


def rollout_and_validate_parallel(
    env,
    kind: str,
    network: Optional[Any],
    params: Optional[Any],
    num_episodes: int,
    checker,
    seed: int = 0,
):
    """Same as `rollout_and_validate_aec` but for ParallelEnv games
    (Goofspiel, Kemps) where every agent acts simultaneously each step and
    `obs`/`avail` are stacked per-agent arrays. All agents share one policy
    (self-play): each agent samples its own action independently from its
    own observation/avail-mask slice using the same `kind`/`network`/`params`.
    """
    rng = jax.random.PRNGKey(seed)

    for _ in range(num_episodes):
        checker.start_episode()
        rng, reset_rng = jax.random.split(rng)
        state, obs = env.reset(reset_rng)
        if hasattr(checker, "check_reset"):
            checker.check_reset(state)

        for step in range(env.horizon + 1):
            if bool(state.done):
                break
            if hasattr(checker, "validate_pre_step"):
                checker.validate_pre_step(step, state)

            avail = env.get_avail_actions(state)
            rng, act_rng = jax.random.split(rng)
            agent_rngs = jax.random.split(act_rng, env.num_agents)
            actions = jnp.stack(
                [
                    sample_action(kind, network, params, obs[a], avail[a], agent_rngs[a])
                    for a in range(env.num_agents)
                ]
            )
            rng, step_rng = jax.random.split(rng)
            next_state, next_obs, reward, absorbing, done, info = env.step_env(
                step_rng, state, actions
            )

            checker.validate_transition(
                step, state, np.asarray(actions), avail, next_state, reward, bool(done)
            )
            state, obs = next_state, next_obs

    return checker
