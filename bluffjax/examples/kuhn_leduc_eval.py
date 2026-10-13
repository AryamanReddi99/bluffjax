"""
Sampled evaluation against uniformly random legal play, shared by the Kuhn
Poker and Leduc Hold'em training scripts (bluffjax/examples/kuhn,
bluffjax/examples/leduc), next to their exact evaluation (exploitability and
value_vs_random of bluffjax.utils.game_utils.{kuhn,leduc}_exploitability).

vs_random_metrics(env, policies, num_games, rng) plays, for every policy,
num_games complete hands (each played with env.step_env, without auto-reset,
until it is over) in parallel against an opponent that picks uniformly among
its legal actions. The env draws the first player of every hand at random and
the observation encodes the position, so the learner's position is what
rotates: in hand g it acts first if (offset + g) % 2 == 0 and second otherwise,
from a random offset, i.e. it alternates between the two positions and, with
an even num_games, plays each in exactly half of the hands (its env seat is
(start_player_idx + position) % 2). Per policy `name` it returns, as
parallel_nfsp_common.eval_metrics does for the other games:
    win_rate_{name}             share of hands with a positive return
    loss_rate_{name}            share of hands with a negative return
    draw_rate_{name}            share of hands with return 0 (a Leduc split pot)
    win_rate_decided_{name}     wins / (wins + losses)
    avg_return_{name}           mean return per hand, chips (env units)
    avg_episode_length_{name}   mean env steps per hand
The expectation of avg_return_{name} is the exact value_vs_random of the
policy's table. The scripts call it every compare_interval updates and after
the last update (maybe_vs_random_metrics), with the key eval_key(rng), a
stream that no split of rng reaches, so the evaluation doesn't change
training (a run with compare=false draws the same random numbers and trains
the same networks).
"""

from typing import Any, Callable, NamedTuple

import jax
from jax import lax
import jax.numpy as jnp

from bluffjax.environments.env import AECEnv
from bluffjax.examples.parallel_nfsp_common import (
    EvalResult,
    eval_metrics,
    sample_logits,
    sample_random_legal,
)
from bluffjax.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray

# act(params, obs, action_mask, rng) -> action, for any batch shape
Policy = Callable[[Any, FloatArray, BoolArray, PRNGKeyArray], IntArray]

# fold_in data of the evaluation key. jax.random.split(rng, n)[i] can equal
# fold_in(rng, i) (partitionable threefry), so the constant is far above any n.
EVAL_STREAM = 0x5EED_E7A1


def eval_key(rng: PRNGKeyArray) -> PRNGKeyArray:
    """The evaluation's key, independent of every split of rng."""
    return jax.random.fold_in(rng, EVAL_STREAM)


def softmax_policy(apply_fn: Callable) -> Policy:
    """Samples from softmax(logits) over the legal actions, the policy whose
    table policy_array_from_network gives. apply_fn(params, obs) returns the
    logits or (logits, value)."""

    def act(params, obs, action_mask, rng):
        out = apply_fn(params, obs)
        logits = out[0] if isinstance(out, tuple) else out
        return sample_logits(logits, action_mask, rng)

    return act


def greedy_policy(apply_fn: Callable) -> Policy:
    """argmax of the Q-values over the legal actions, ties to the first one,
    the policy whose table policy_array_from_qnetwork gives."""

    def act(params, obs, action_mask, rng):
        q_vals = apply_fn(params, obs)
        return jnp.argmax(jnp.where(action_mask, q_vals, -jnp.inf), axis=-1)

    return act


def random_policy(params, obs, action_mask, rng) -> IntArray:
    """Uniform over the legal actions (the evaluation opponent)."""
    return sample_random_legal(action_mask, rng)


class GameRecord(NamedTuple):
    returns: FloatArray  # (num_games, num_agents) total env reward of every seat
    length: IntArray  # (num_games,) env steps until the hand was over
    learner_position: IntArray  # (num_games,) 0: the learner acts first
    learner_seat: IntArray  # (num_games,) the learner's env player index
    done: BoolArray  # (num_games,) the hand is over (always true at the end)


def play_vs_random(
    env: AECEnv,
    learner: Policy,
    learner_params: Any,
    num_games: int,
    rng: PRNGKeyArray,
) -> GameRecord:
    """Plays num_games hands in parallel, the learner in position
    (offset + g) % num_agents of hand g (random offset) and uniformly random
    legal play in the other seats, each with env.step_env until it is over."""
    num_agents = env.num_agents
    rng_offset, rng_reset, rng_play = jax.random.split(rng, 3)
    offset = jax.random.randint(rng_offset, (), 0, num_agents)
    position = (offset + jnp.arange(num_games)) % num_agents
    states, obs = jax.vmap(env.reset)(jax.random.split(rng_reset, num_games))
    seat = (states.start_player_idx + position) % num_agents
    step = jax.vmap(env.step_env)

    def where_games(games, new, old):
        return jnp.where(games.reshape((-1,) + (1,) * (new.ndim - 1)), new, old)

    def cond(carry):
        return ~carry[2].all()

    def body(carry):
        states, obs, finished, returns, length, rng = carry
        rng, rng_learner, rng_random, rng_step = jax.random.split(rng, 4)
        mask = jax.vmap(env.get_avail_actions)(states)
        action = jnp.where(
            states.current_player_idx == seat,
            learner(learner_params, obs, mask, rng_learner),
            random_policy(None, obs, mask, rng_random),
        )
        next_states, next_obs, reward, _, done, _ = step(
            jax.random.split(rng_step, num_games), states, action
        )
        live = ~finished
        returns = returns + jnp.where(live[:, None], reward, 0.0)
        length = length + live.astype(jnp.int32)
        states = jax.tree_util.tree_map(
            lambda new, old: where_games(live, new, old), next_states, states
        )
        obs = where_games(live, next_obs, obs)
        return states, obs, finished | (live & done), returns, length, rng

    init = (
        states,
        obs,
        jnp.zeros((num_games,), dtype=jnp.bool_),
        jnp.zeros((num_games, num_agents), dtype=jnp.float32),
        jnp.zeros((num_games,), dtype=jnp.int32),
        rng_play,
    )
    _, _, finished, returns, length, _ = lax.while_loop(cond, body, init)
    return GameRecord(
        returns=returns,
        length=length,
        learner_position=position,
        learner_seat=seat,
        done=finished,
    )


def evaluate_vs_random(
    env: AECEnv,
    learner: Policy,
    learner_params: Any,
    num_games: int,
    rng: PRNGKeyArray,
) -> EvalResult:
    """Per hand, from the learner's seat: win (positive return), loss
    (negative), draw (0), return and length (play_vs_random)."""
    record = play_vs_random(env, learner, learner_params, num_games, rng)
    ret = jnp.take_along_axis(
        record.returns, record.learner_seat[:, None], axis=-1
    ).squeeze(-1)
    return EvalResult(
        win=(ret > 0).astype(jnp.float32),
        loss=(ret < 0).astype(jnp.float32),
        draw=(ret == 0).astype(jnp.float32),
        ret=ret,
        length=record.length.astype(jnp.float32),
    )


def vs_random_metrics(
    env: AECEnv,
    policies: dict[str, tuple[Policy, Any]],
    num_games: int,
    rng: PRNGKeyArray,
) -> dict[str, FloatArray]:
    """eval_metrics(name, ...) of num_games hands against random for every
    {name: (policy, params)} (names like "avg_vs_random"), independent keys."""
    metrics = {}
    for key, (name, (act, params)) in zip(
        jax.random.split(rng, len(policies)), policies.items()
    ):
        result = evaluate_vs_random(env, act, params, num_games, key)
        metrics.update(eval_metrics(name, result))
    return metrics


def maybe_vs_random_metrics(
    env: AECEnv,
    policies: dict[str, tuple[Policy, Any]],
    num_games: int,
    rng: PRNGKeyArray,
    due: BoolArray,
) -> dict[str, FloatArray]:
    """vs_random_metrics if due, zeros of the same shapes otherwise (lax.cond,
    so the games are only played when due)."""

    def run():
        return vs_random_metrics(env, policies, num_games, rng)

    shapes = jax.eval_shape(run)
    return lax.cond(
        due,
        run,
        lambda: jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), shapes),
    )


def is_sampled_metric(key: str) -> bool:
    """Whether a logged key is one of vs_random_metrics' (names *_vs_random;
    the exact value_*vs_random keys are not)."""
    return key.endswith("_vs_random") and not key.startswith("value_")


def format_returns(evals: dict[str, float]) -> str:
    """'<name> +x.xxx, ...' of the avg_return_<name> entries of evals."""
    prefix = "avg_return_"
    return ", ".join(
        f"{k[len(prefix):]} {v:+.3f}" for k, v in evals.items() if k.startswith(prefix)
    )
