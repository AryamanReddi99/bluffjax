"""
Cards and hands for heads-up Hold'em ReBeL.

A player's private state is one of the 1,326 two-card combinations. Beliefs and
values are vectors over those combinations, indexed by HAND_CARDS. Card ids
follow the environments (0..51, suit = card // 13).

Hands that share a card with each other or with the board can't coexist. All
counterfactual values here handle that card removal exactly, in O(52 + 1326)
per vector, with inclusion-exclusion over the 51 hands that contain each card.
"""

import functools
import itertools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from bluffjax.utils.game_utils.poker_utils import (
    _COMB_7_5,
    _card_rank,
    _card_suit,
    _score_five_card_hand,
)
from bluffjax.utils.typing import BoolArray, FloatArray, IntArray

NUM_CARDS = 52
NUM_HANDS = 1326


def _build_hand_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hand_cards = np.array(
        [(a, b) for a in range(NUM_CARDS) for b in range(a + 1, NUM_CARDS)],
        dtype=np.int32,
    )
    hand_index = -np.ones((NUM_CARDS, NUM_CARDS), dtype=np.int32)
    hand_index[hand_cards[:, 0], hand_cards[:, 1]] = np.arange(NUM_HANDS)
    hand_index[hand_cards[:, 1], hand_cards[:, 0]] = np.arange(NUM_HANDS)
    card_hands = np.stack(
        [np.nonzero((hand_cards == c).any(axis=1))[0] for c in range(NUM_CARDS)]
    ).astype(np.int32)
    return hand_cards, hand_index, card_hands


_HAND_CARDS_NP, _HAND_INDEX_NP, _CARD_HANDS_NP = _build_hand_tables()
HAND_CARDS = jnp.asarray(_HAND_CARDS_NP)  # (1326, 2), first card < second
HAND_INDEX = jnp.asarray(_HAND_INDEX_NP)  # (52, 52) -> hand id, -1 on the diagonal
CARD_HANDS = jnp.asarray(_CARD_HANDS_NP)  # (52, 51) hands containing each card


def hand_index(cards: IntArray) -> IntArray:
    """Hand id of two hole cards (any order)."""
    return HAND_INDEX[cards[..., 0], cards[..., 1]]


def cards_onehot(cards: IntArray, count: IntArray) -> BoolArray:
    """(52,) mask of the first `count` entries of `cards`."""
    used = jnp.arange(cards.shape[-1]) < count
    safe = jnp.where(used, cards, 0)
    return jnp.zeros(NUM_CARDS, dtype=bool).at[safe].max(used)


def hands_not_blocked(card_mask: BoolArray) -> BoolArray:
    """(1326,) True for hands that share no card with `card_mask` (52,)."""
    return ~(card_mask[HAND_CARDS[:, 0]] | card_mask[HAND_CARDS[:, 1]])


def normalize(x: FloatArray) -> FloatArray:
    """Normalise the last axis to sum to one (all-zero rows stay zero)."""
    total = jnp.sum(x, axis=-1, keepdims=True)
    return jnp.where(total > 0.0, x / jnp.where(total > 0.0, total, 1.0), 0.0)


def card_sums(x: FloatArray) -> FloatArray:
    """(..., 52): total weight of the hands holding each card."""
    return jnp.take(x, CARD_HANDS, axis=-1).sum(axis=-1)


def compatible_mass(x: FloatArray) -> FloatArray:
    """(..., 1326): sum of x over the hands that share no card with each hand.

    Inclusion-exclusion: a hand h' != h shares at most one card with h, and h
    itself was subtracted twice, so it is added back once.
    """
    cs = card_sums(x)
    total = jnp.sum(x, axis=-1, keepdims=True)
    return (
        total
        - jnp.take(cs, HAND_CARDS[:, 0], axis=-1)
        - jnp.take(cs, HAND_CARDS[:, 1], axis=-1)
        + x
    )


# =============================================================================
# Hand strength
# =============================================================================

_BINOM = np.array(
    [[math.comb(n, k) for k in range(6)] for n in range(NUM_CARDS + 1)],
    dtype=np.int64,
)
_BINOM_J = jnp.asarray(_BINOM.astype(np.int32))


@functools.lru_cache(maxsize=1)
def rank5_table() -> IntArray:
    """Score of every 5-card hand, indexed by the colex rank of its sorted cards.

    Scores come from the environments' own evaluator, so ties and orderings
    match the env's showdowns exactly. Takes about half a second to build.
    """
    combos = np.array(list(itertools.combinations(range(NUM_CARDS), 5)), np.int32)
    colex = (_BINOM[combos, np.arange(1, 6)]).sum(axis=1)
    score = jax.jit(
        jax.vmap(lambda c: _score_five_card_hand(_card_rank(c), _card_suit(c)))
    )
    with jax.ensure_compile_time_eval():  # may be first called while tracing
        chunks = [
            np.asarray(score(jnp.asarray(combos[i : i + (1 << 18)])))
            for i in range(0, len(combos), 1 << 18)
        ]
        table = np.zeros(len(combos), dtype=np.int32)
        table[colex] = np.concatenate(chunks)
        return jnp.asarray(table)


def _seven_card_score(table: IntArray, cards: IntArray) -> IntArray:
    """Best 5-of-7 score, same 21 subsets as poker_utils._score_seven_card_hand."""
    cards = jnp.sort(cards)
    combos = cards[_COMB_7_5]  # (21, 5), each row sorted
    colex = jnp.sum(_BINOM_J[combos, jnp.arange(1, 6)], axis=-1)
    return jnp.max(table[colex])


def hand_strengths(table: IntArray, board: IntArray) -> IntArray:
    """(1326,) showdown score of every hand on a 5-card board, -1 if blocked."""
    blocked = ~hands_not_blocked(cards_onehot(board, 5))
    cards = jnp.concatenate(
        [jnp.broadcast_to(board, (NUM_HANDS, 5)), HAND_CARDS], axis=1
    )
    scores = jax.vmap(lambda c: _seven_card_score(table, c))(cards)
    return jnp.where(blocked, -1, scores)


# =============================================================================
# Showdown values
# =============================================================================


class ShowdownTables(NamedTuple):
    """Sorted orders for one board, used to evaluate showdowns in O(n)."""

    order: IntArray  # (1326,) hands sorted by strength
    lo: IntArray  # (1326,) number of hands strictly weaker
    hi: IntArray  # (1326,) number of hands weaker or tied
    card_order: IntArray  # (52, 51) hands holding each card, sorted by strength
    card_lo: IntArray  # (1326, 2) same counts within each of the hand's cards
    card_hi: IntArray  # (1326, 2)


def showdown_tables(strength: IntArray) -> ShowdownTables:
    order = jnp.argsort(strength)
    sorted_s = strength[order]
    lo = jnp.searchsorted(sorted_s, strength, side="left")
    hi = jnp.searchsorted(sorted_s, strength, side="right")
    s_card = strength[CARD_HANDS]  # (52, 51)
    card_perm = jnp.argsort(s_card, axis=1)
    card_order = jnp.take_along_axis(CARD_HANDS, card_perm, axis=1)
    sorted_card_s = jnp.take_along_axis(s_card, card_perm, axis=1)

    def counts(card, s):
        row = sorted_card_s[card]
        return (
            jnp.searchsorted(row, s, side="left"),
            jnp.searchsorted(row, s, side="right"),
        )

    clo, chi = jax.vmap(
        lambda cards, s: jax.vmap(lambda c: counts(c, s))(cards)
    )(HAND_CARDS, strength)
    return ShowdownTables(order, lo, hi, card_order, clo, chi)


def showdown_value(x: FloatArray, t: ShowdownTables) -> FloatArray:
    """(..., 1326): opponent weight beaten minus opponent weight that wins.

    x (..., 1326) is the opponent's reach. Ties count zero. Multiply by the
    amount each player has in the pot to get counterfactual showdown values.
    """
    lead = x.shape[:-1]
    cum = jnp.concatenate(
        [jnp.zeros(lead + (1,), x.dtype), jnp.cumsum(jnp.take(x, t.order, -1), -1)],
        axis=-1,
    )
    total = cum[..., -1:]
    below = jnp.take(cum, t.lo, axis=-1)
    above = total - jnp.take(cum, t.hi, axis=-1)
    xc = jnp.take(x, t.card_order.reshape(-1), axis=-1).reshape(lead + (52, 51))
    cumc = jnp.concatenate(
        [jnp.zeros(lead + (52, 1), x.dtype), jnp.cumsum(xc, axis=-1)], axis=-1
    ).reshape(lead + (52 * 52,))
    win = below
    lose = above
    for k in range(2):
        base = HAND_CARDS[:, k] * 52
        row_total = jnp.take(cumc, base + 51, axis=-1)
        win = win - jnp.take(cumc, base + t.card_lo[:, k], axis=-1)
        lose = lose - (row_total - jnp.take(cumc, base + t.card_hi[:, k], axis=-1))
    return win - lose
