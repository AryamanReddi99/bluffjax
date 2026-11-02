"""Single Deep CFR's average strategy (Steinberger 2019,
https://arxiv.org/abs/1901.07621), computed exactly at every infoset of a small
game, and row-wise regret matching; shared by kuhn_deep_cfr.py and
leduc_deep_cfr.py (--sd_cfr)."""

import numpy as np


def regret_matching_table(
    advantages: np.ndarray, legal_masks: np.ndarray
) -> np.ndarray:
    """Deep CFR's regret_matching row by row (the max-advantage legal action,
    ties split, when no advantage is positive): (n, NUM_ACTIONS) advantages and legal masks
    to (n, NUM_ACTIONS) float64 strategies."""
    advantages = np.asarray(advantages, dtype=np.float64)
    positive = np.where(legal_masks, np.maximum(advantages, 0.0), 0.0)
    total = positive.sum(axis=-1, keepdims=True)
    masked = np.where(legal_masks, advantages, -np.inf)
    best = (masked == masked.max(axis=-1, keepdims=True)).astype(np.float64)
    best /= best.sum(axis=-1, keepdims=True)
    return np.where(total > 0, positive / np.where(total > 0, total, 1.0), best)


class SDCFRAverage:
    """Single Deep CFR's average strategy at every infoset (rows in an infoset
    list): the t-weighted average of a player's strategies sigma^t, each
    weighted at infoset I by the player's own reach probability of I under
    sigma^t, as tabular CFR's average strategy. Infosets that no added
    strategy reaches get the uniform strategy."""

    def __init__(
        self,
        players: np.ndarray,
        parent: np.ndarray,
        parent_action: np.ndarray,
        legal_masks: np.ndarray,
    ) -> None:
        self._players = players
        self._parent = parent
        self._parent_action = parent_action
        self._legal_masks = legal_masks
        n, num_actions = legal_masks.shape
        self._num = np.zeros((n, num_actions), dtype=np.float64)
        self._den = np.zeros(n, dtype=np.float64)
        # Rows grouped by the number of own decisions before them
        depth = np.zeros(n, dtype=np.int64)
        for row in range(n):
            r = row
            while parent[r] >= 0:
                depth[row] += 1
                r = parent[r]
        self._levels = [np.flatnonzero(depth == d) for d in range(1, depth.max() + 1)]

    def reach(self, strategy: np.ndarray) -> np.ndarray:
        """Each infoset's own reach probability under `strategy` (rows of
        its player's infosets)."""
        reach = np.ones(len(self._parent), dtype=np.float64)
        for rows in self._levels:
            parents = self._parent[rows]
            reach[rows] = reach[parents] * strategy[parents, self._parent_action[rows]]
        return reach

    def add(self, player: int, strategy: np.ndarray, weight: float) -> None:
        """Adds `player`'s strategy (its rows of `strategy`) with `weight`."""
        rows = self._players == player
        w = weight * self.reach(strategy)[rows]
        self._num[rows] += w[:, None] * strategy[rows]
        self._den[rows] += w

    def policy(self) -> np.ndarray:
        uniform = self._legal_masks / self._legal_masks.sum(axis=-1, keepdims=True)
        den = self._den[:, None]
        return np.where(den > 0, self._num / np.where(den > 0, den, 1.0), uniform)
