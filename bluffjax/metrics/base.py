"""Common interface of the game-specific behaviour metrics.

The metrics of a game are computed from environment steps, so every training
script (and the head-to-head evaluator) measures them the same way:

    gm = make_metrics("leduc", env)                # or <Game>Metrics(env)
    carry = gm.init_carry()                         # one env, before a game
    carry, counts = gm.step(carry, state, action, next_state, reward, done, info)
    ...                                             # vmap over envs, scan over steps
    totals = masked_totals(all_counts, mask)        # sum over steps, envs, players
    rates = gm.summarize(totals)                    # {"bluff_rate": ..., ...}

step() sees one env: `state` is the state before the step, `action`, `reward`,
`done` and `info` are exactly what was passed to / returned by env.step, and
`next_state` is the state returned by env.step. Because the environments reset
automatically, `next_state` is the first state of the next game when `done`;
everything about the game that just ended has to come from `state`, `action`,
`reward`, `info` and the carry.

`counts` maps an event name to a float array of shape (num_agents, *extra):
the events of this step, credited to the player whose behaviour they describe
(a bluff to the bettor, a correct fold to the folder, a successful bluff to
the bluffer even though the fold that made it succeed comes later). Credit by
player lets a training script keep only the decisions of NFSP's average policy
and an evaluator split a match by side. Extra axes hold tables (e.g. a joint
count of signal and hand for a mutual information).

The carry is a per-env pytree for what a metric has to remember across the
steps of one game (e.g. whether a player has already put chips in voluntarily
preflop). step() has to return it reset when `done`.

summarize() turns summed counts into rates. A rate with no events is NaN, not
0, so that "never happened" can't be mistaken for a measured value.
"""

from typing import Any

import jax.numpy as jnp

from bluffjax.utils.typing import Array, BoolArray, FloatArray

Counts = dict[str, FloatArray]


class GameMetrics:
    """Base class; one subclass per game (see bluffjax/metrics/<game>.py)."""

    #: names of the rates summarize() returns, in logging order
    rate_names: tuple[str, ...] = ()
    #: True if step() needs `probs`, the action probabilities of the policy
    #: each player followed at this step, shape (num_agents, num_actions)
    needs_probs: bool = False

    def __init__(self, env: Any) -> None:
        self.env = env
        self.num_agents = env.num_agents

    def init_carry(self) -> Any:
        """Carry of one env at the start of a game (default: nothing)."""
        return ()

    def step(
        self,
        carry: Any,
        state: Any,
        action: Array,
        next_state: Any,
        reward: FloatArray,
        done: BoolArray,
        info: dict,
        probs: FloatArray | None = None,
    ) -> tuple[Any, Counts]:
        raise NotImplementedError

    def summarize(self, totals: Counts) -> dict[str, FloatArray]:
        raise NotImplementedError


def masked_totals(counts: Counts, mask: BoolArray | FloatArray) -> Counts:
    """Sums counts over the leading axes of `mask`, keeping only credited
    players where mask is true.

    counts: name -> (..., num_agents, *extra); mask: (..., num_agents), e.g.
    (num_steps, num_envs, num_agents) with mask = ~is_br for the decisions of
    NFSP's average policy. Returns name -> (*extra).
    """
    out = {}
    for name, value in counts.items():
        m = mask.astype(value.dtype).reshape(
            mask.shape + (1,) * (value.ndim - mask.ndim)
        )
        out[name] = (value * m).sum(axis=tuple(range(mask.ndim)))
    return out


def rate(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """numerator / denominator, NaN when there were no events."""
    numerator = jnp.asarray(numerator, jnp.float32)
    denominator = jnp.asarray(denominator, jnp.float32)
    return jnp.where(
        denominator > 0, numerator / jnp.where(denominator > 0, denominator, 1.0), jnp.nan
    )


def credit(num_agents: int, player: Array, value: Array = 1.0) -> FloatArray:
    """(num_agents,) array with `value` at index `player` and 0 elsewhere (all
    zeros if `player` isn't a valid index). A one-hot product rather than a
    scatter, so that XLA fuses it with the rest of the rollout step."""
    return (jnp.arange(num_agents) == player) * jnp.asarray(value, jnp.float32)


def prefixed(prefix: str, rates: dict[str, FloatArray]) -> dict[str, FloatArray]:
    """{"bluff_rate": x} -> {"<prefix>/bluff_rate": x} for logging."""
    return {f"{prefix}/{name}": value for name, value in rates.items()}
