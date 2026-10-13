"""Kuhn Poker behaviour metrics, computed exactly from the policy table.

Conventions (bluffjax.environments.kuhn_poker and kuhn_exploitability): cards
0, 1, 2 are the Jack, Queen and King; action 0 is pass (check, or fold when
facing a bet) and action 1 is bet (bet, or call when facing a bet). The env
draws the first player of every hand at random. "The first player" below is
whoever acts first in the hand (player 0 of kuhn_exploitability's infosets "0",
"1", "2", "0pb", "1pb", "2pb") and "the second player" the other one ("0p",
"1p", "2p", "0b", "1b", "2b").

A policy is given as kuhn_exploitability takes it: a (12, 2) array of
[P(pass), P(bet)] per information set with rows in INFOSETS order (e.g. from
policy_array_from_network or policy_array_from_qnetwork), or a callable
infoset key -> [p_pass, p_bet] (e.g. nash_equilibrium(alpha)). Every row must
be a probability distribution. policy_metrics(policy_p0, policy_p1) reads the
first player's six information sets from policy_p0 and the second player's six
from policy_p1, so for a match with A acting first against B it is
policy_metrics(A, B); policy_p1 defaults to policy_p0 (self-play, one policy in
both seats, which is what the training scripts log).

Write b(I) = P(bet or call) = P(action 1) at information set I, named by the
card and the betting so far (p = pass, b = bet). Kuhn Poker's Nash equilibria
(Kuhn 1950; nash_equilibrium in kuhn_exploitability) are, for some alpha in
[0, 1/3]:

    first player:   b(J) = alpha        b(Q) = 0               b(K) = 3 alpha
      after check-bet: b(Jpb) = 0      b(Qpb) = alpha + 1/3   b(Kpb) = 1
    second player, after a check:  b(Jp) = 1/3   b(Qp) = 0     b(Kp) = 1
                   facing a bet:   b(Jb) = 0     b(Qb) = 1/3   b(Kb) = 1

The second player's equilibrium strategy is unique; the first player's
equilibrium strategies are this one-parameter family.

Metrics (exact functions of the table, no sampling), returned by
policy_metrics in METRIC_NAMES order:
- p0_bet_jack: b(J) of the first player: P(bet | Jack, first to act), the
  first player's bluff (equilibrium alpha).
- p0_bet_king: b(K) of the first player: P(bet | King, first to act)
  (equilibrium 3 alpha).
- p1_bet_jack: b(Jp) of the second player: P(bet | Jack, after a check), the
  second player's bluff (equilibrium 1/3).
- p0_call_queen: b(Qpb) of the first player: P(call | Queen, after it checked
  and the second player bet), a bluff catch (equilibrium alpha + 1/3).
- p1_call_queen: b(Qb) of the second player: P(call | Queen, facing a bet),
  a bluff catch (equilibrium 1/3).
- p1_equilibrium_gap: max over the second player's six information sets of
  |b(I) - b*(I)| with b* its unique equilibrium; 0 exactly when the second
  player plays its equilibrium strategy.
- p0_alpha_gap: min over alpha in [0, 1/3] of the max over the first player's
  six information sets of |b(I) - b*_alpha(I)|; 0 exactly when the first
  player plays the equilibrium strategy of some alpha. (Equilibrium play at an
  information set the first player's own strategy never reaches doesn't
  matter: at alpha = 1/3 it always bets the King and never meets Kpb, but the
  metric still compares b(Kpb) with 1.)
- p0_alpha: the alpha of that minimum. Only J, K and Qpb depend on alpha (with
  slopes 1, 3 and 1); p0_alpha is the alpha in [0, 1/3] that minimises their
  largest gap h(alpha) = max(|b(J) - alpha|, |b(K) - 3 alpha|,
  |b(Qpb) - 1/3 - alpha|). It is unique (h is convex and piecewise linear with
  no flat piece) and it also attains p0_alpha_gap, which is
  max(h(p0_alpha), |b(Q)|, |b(Jpb)|, |1 - b(Kpb)|): the three alpha-free terms
  only add a constant.

The minimum over alpha is computed exactly: h is minimised over [0, 1/3] at an
end of the interval or at a breakpoint of h, i.e. where one of its terms is 0
or two of them are equal (x_i - k_i alpha = +-(x_j - k_j alpha)); every such
point is evaluated.
"""

import itertools
from typing import Callable

import numpy as np

from bluffjax.utils.game_utils.kuhn_exploitability import ACTION_BET, INFOSETS

Policy = np.ndarray | Callable[[str], np.ndarray]

METRIC_NAMES: tuple[str, ...] = (
    "p0_bet_jack",
    "p0_bet_king",
    "p1_bet_jack",
    "p0_call_queen",
    "p1_call_queen",
    "p1_equilibrium_gap",
    "p0_alpha_gap",
    "p0_alpha",
)

ALPHA_MAX = 1.0 / 3.0

# The first player's equilibrium b*_alpha(I) = offset + slope * alpha.
FIRST_PLAYER_EQUILIBRIUM: dict[str, tuple[float, float]] = {
    "0": (0.0, 1.0),
    "1": (0.0, 0.0),
    "2": (0.0, 3.0),
    "0pb": (0.0, 0.0),
    "1pb": (1.0 / 3.0, 1.0),
    "2pb": (1.0, 0.0),
}
# The second player's (unique) equilibrium b*(I).
SECOND_PLAYER_EQUILIBRIUM: dict[str, float] = {
    "0p": 1.0 / 3.0,
    "1p": 0.0,
    "2p": 1.0,
    "0b": 0.0,
    "1b": 1.0 / 3.0,
    "2b": 1.0,
}

_ROW = {key: i for i, key in enumerate(INFOSETS)}
# Largest deviation of a row sum from 1 that is accepted (float32 softmax).
_SUM_TOLERANCE = 1e-5


def policy_table(policy: Policy) -> np.ndarray:
    """The (12, 2) float64 table [P(pass), P(bet)] of a policy, rows in
    INFOSETS order, from an array or a callable; checks it is a policy."""
    if callable(policy):
        table = np.stack([np.asarray(policy(k), dtype=np.float64) for k in INFOSETS])
    else:
        table = np.asarray(policy, dtype=np.float64)
    if table.shape != (len(INFOSETS), 2):
        raise ValueError(f"expected shape ({len(INFOSETS)}, 2), got {table.shape}")
    if not np.isfinite(table).all() or (table < 0).any():
        raise ValueError("policy probabilities must be finite and non-negative")
    if np.abs(table.sum(axis=-1) - 1.0).max() > _SUM_TOLERANCE:
        raise ValueError("every row of the policy must sum to 1")
    return table


def bet_probabilities(policy: Policy) -> dict[str, float]:
    """b(I) = P(bet or call) at every information set, keyed by infoset."""
    table = policy_table(policy)
    return {key: float(table[_ROW[key], ACTION_BET]) for key in INFOSETS}


def first_player_alpha_fit(bets: dict[str, float]) -> tuple[float, float]:
    """(p0_alpha, p0_alpha_gap) of the first player's b(I) (see the module
    docstring): the alpha in [0, 1/3] minimising the largest gap at the
    alpha-dependent information sets, and the largest gap over all six there."""
    # b(I) - b*_alpha(I) = x - k * alpha
    terms = [
        (bets[key] - offset, slope)
        for key, (offset, slope) in FIRST_PLAYER_EQUILIBRIUM.items()
        if slope != 0.0
    ]
    constant = max(
        abs(bets[key] - offset)
        for key, (offset, slope) in FIRST_PLAYER_EQUILIBRIUM.items()
        if slope == 0.0
    )
    candidates = [0.0, ALPHA_MAX] + [x / k for x, k in terms]
    for (x1, k1), (x2, k2) in itertools.combinations(terms, 2):
        if k1 != k2:
            candidates.append((x1 - x2) / (k1 - k2))
        candidates.append((x1 + x2) / (k1 + k2))
    alphas = np.clip(np.asarray(candidates, dtype=np.float64), 0.0, ALPHA_MAX)
    fit = np.max([np.abs(x - k * alphas) for x, k in terms], axis=0)
    best = int(np.argmin(fit))
    return float(alphas[best]), float(max(constant, fit[best]))


def policy_metrics(policy_p0: Policy, policy_p1: Policy | None = None) -> dict[str, float]:
    """The metrics of the module docstring, in METRIC_NAMES order.

    policy_p0 gives the first player's strategy (its rows "0", "1", "2", "0pb",
    "1pb", "2pb"), policy_p1 the second player's (rows "0p", "1p", "2p", "0b",
    "1b", "2b"); policy_p1 = None uses policy_p0 for both (self-play).
    """
    first = bet_probabilities(policy_p0)
    second = first if policy_p1 is None else bet_probabilities(policy_p1)
    p0_alpha, p0_alpha_gap = first_player_alpha_fit(first)
    p1_gap = max(abs(second[k] - v) for k, v in SECOND_PLAYER_EQUILIBRIUM.items())
    metrics = {
        "p0_bet_jack": first["0"],
        "p0_bet_king": first["2"],
        "p1_bet_jack": second["0p"],
        "p0_call_queen": first["1pb"],
        "p1_call_queen": second["1b"],
        "p1_equilibrium_gap": float(p1_gap),
        "p0_alpha_gap": p0_alpha_gap,
        "p0_alpha": p0_alpha,
    }
    return {name: metrics[name] for name in METRIC_NAMES}
