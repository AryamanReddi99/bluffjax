"""Game-specific behaviour metrics (bluffjax-metrics.md).

Kuhn Poker and Leduc Hold'em compute theirs exactly from a policy table
(bluffjax.metrics.kuhn_poker, bluffjax.metrics.leduc_holdem). Every other game
accumulates them from environment steps through the GameMetrics interface in
bluffjax.metrics.base; make_metrics(env_id, env) returns the one for a game.
"""

import importlib
from typing import Any

from bluffjax.metrics.base import (
    Counts,
    GameMetrics,
    credit,
    masked_totals,
    prefixed,
    rate,
)

# env id -> module defining `Metrics`, a GameMetrics subclass
_STEP_METRICS = {
    "texas_limit_holdem": "bluffjax.metrics.texas_limit_holdem",
    "texas_nolimit_holdem": "bluffjax.metrics.texas_nolimit_holdem",
    "five_card_draw": "bluffjax.metrics.five_card_draw",
    "seven_card_stud": "bluffjax.metrics.seven_card_stud",
    "goofspiel": "bluffjax.metrics.goofspiel",
    "bluff": "bluffjax.metrics.bluff",
    "kemps": "bluffjax.metrics.kemps",
    "werewolf": "bluffjax.metrics.werewolf",
}


def make_metrics(env_id: str, env: Any) -> GameMetrics:
    """The step-based metrics of a game, for an env made with make(env_id)."""
    if env_id not in _STEP_METRICS:
        raise ValueError(
            f"no step-based metrics for '{env_id}' (Kuhn Poker and Leduc Hold'em "
            f"compute theirs from the policy table); available: {sorted(_STEP_METRICS)}"
        )
    return importlib.import_module(_STEP_METRICS[env_id]).Metrics(env)


__all__ = [
    "Counts",
    "GameMetrics",
    "credit",
    "make_metrics",
    "masked_totals",
    "prefixed",
    "rate",
]
