"""Poker hand scores: the JAX evaluators against a plain-Python reference.

ref_score scores the best five cards of a hand by the usual poker rules, in
the evaluators' encoding: category * 15**5 plus the tiebreak ranks as base-15
digits (category 1 = high card, ..., 9 = straight flush).
"""

import itertools
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax.utils.game_utils.poker_utils import (
    _card_rank,
    _card_suit,
    _compare_hands,
    _score_five_card_hand,
    _score_seven_card_hand,
    _straight_high,
)

RANKS = "23456789TJQKA"
SUITS = "cdhs"  # suit = card // 13


def card(rank: int, suit: int) -> int:
    """Card index for rank 2-14 (14 = ace) and suit 0-3 (the env's encoding)."""
    return suit * 13 + (0 if rank == 14 else rank - 1)


def parse(text: str) -> list[int]:
    """Cards from text such as "Ah Td 2c"."""
    return [card(RANKS.index(r) + 2, SUITS.index(s)) for r, s in text.split()]


def rank_of(c: int) -> int:
    return 14 if c % 13 == 0 else c % 13 + 1


def five_card_key(cards) -> tuple[int, ...]:
    """(category, tiebreak ranks) of five cards."""
    counts = Counter(rank_of(c) for c in cards)
    groups = sorted(counts.items(), key=lambda rc: (rc[1], rc[0]), reverse=True)
    shape = [n for _, n in groups]
    order = [r for r, _ in groups]
    flush = len({c // 13 for c in cards}) == 1
    straight = 0
    if len(order) == 5 and order[0] - order[4] == 4:
        straight = order[0]
    elif order == [14, 5, 4, 3, 2]:
        straight = 5
    if straight and flush:
        return (9, straight)
    if shape == [4, 1]:
        return (8, *order)
    if shape == [3, 2]:
        return (7, *order)
    if flush:
        return (6, *order)
    if straight:
        return (5, straight)
    if shape == [3, 1, 1]:
        return (4, *order)
    if shape == [2, 2, 1]:
        return (3, *order)
    if shape == [2, 1, 1, 1]:
        return (2, *order)
    return (1, *order)


def encode(key: tuple[int, ...]) -> int:
    category, *tiebreaks = key
    digits = tiebreaks + [0] * (5 - len(tiebreaks))
    return category * 15**5 + sum(d * 15 ** (4 - i) for i, d in enumerate(digits))


def ref_score(cards) -> int:
    """Score of the best five cards of a 5- to 7-card hand."""
    return max(encode(five_card_key(c)) for c in itertools.combinations(cards, 5))


score_hands = jax.jit(jax.vmap(_score_seven_card_hand))
score_five_card_hands = jax.jit(
    jax.vmap(lambda c: _score_five_card_hand(_card_rank(c), _card_suit(c)))
)


# (7 cards, best five-card hand as (category, tiebreak ranks))
CRAFTED = [
    ("Ah Qd 9c 7s 5h 3d 2c", (1, 14, 12, 9, 7, 5)),
    ("Ah Ad 9c 7s 5h 3d 2c", (2, 14, 9, 7, 5)),
    ("Kh Ad 2c 3s 4h 9d 9c", (2, 9, 14, 13, 4)),  # no wrap-around straight
    ("Jh Jd 4c 4s Ah 8d 2c", (3, 11, 4, 14)),
    ("Ah Ad Kc Ks 4h 4d 2c", (3, 14, 13, 4)),  # three pairs, kicker from a pair
    ("8h 8d 6c 6s 4h 4d Ac", (3, 8, 6, 14)),  # three pairs, kicker a single
    ("Qh Qd Qc 9s 6h 4d 2c", (4, 12, 9, 6)),
    ("Ah 2d 3c 4s 5h 9d Kc", (5, 5)),  # wheel
    ("Ah 2d 3c 4s 5h 6d Kc", (5, 6)),
    ("Th Jd Qc Ks Ah 2d 3c", (5, 14)),  # broadway
    ("5h 6d 7c 8s 9h 9d 9c", (5, 9)),  # straight beats trips
    ("5h 6d 6c 7s 8h 9d 9c", (5, 9)),  # straight beats two pair
    ("9h Th Jd Qc Kh 2h 4h", (6, 13, 10, 9, 4, 2)),  # flush beats straight
    ("Ah Kh 9h 5h 2h Ad Ac", (6, 14, 13, 9, 5, 2)),  # flush beats trips
    ("Kh Kd 9h 7h 4h 2h Qc", (6, 13, 9, 7, 4, 2)),
    ("2s 5s 7s 9s Js Ks Ad", (6, 13, 11, 9, 7, 5)),  # six-card flush
    ("2d 4d 6d 8d Td Qd Ad", (6, 14, 12, 10, 8, 6)),  # seven-card flush
    ("Kh Kd Kc 7s 7h 7d 2c", (7, 13, 7)),  # two trips
    ("5h 5d 5c Js Jh 9d 9c", (7, 5, 11)),  # trips and two pairs
    ("9h 9d 9c 9s Kh Kd Kc", (8, 9, 13)),  # quads and trips
    ("3h 3d 3c 3s 7h 7d Qc", (8, 3, 12)),
    ("3h 3d 3c 3s Kh Kd Qc", (8, 3, 13)),  # kicker from a pair
    ("Ah Ad Ac As 2h 3d 4c", (8, 14, 4)),
    ("Ah 2h 3h 4h 5h 9d Kc", (9, 5)),  # steel wheel
    ("5c 6c 7c 8c 9c Td Jh", (9, 9)),  # beats the higher plain straight
    ("3c 4c 5c 6c 7c Kc Ac", (9, 7)),  # beats the higher flush cards
    ("Ts Js Qs Ks As 2d 2c", (9, 14)),
]


@pytest.mark.parametrize("text, key", CRAFTED)
def test_crafted_hands(text: str, key: tuple[int, ...]) -> None:
    cards = parse(text)
    assert ref_score(cards) == encode(key)
    assert int(_score_seven_card_hand(jnp.asarray(cards, jnp.int32))) == encode(key)


@pytest.mark.parametrize("n", [5, 6, 7])
def test_random_hands_match_reference(n: int) -> None:
    """Uniform deals, plus deals from two suits and from the top five ranks,
    which are mostly flushes and full houses, quads and straights."""
    rng = np.random.default_rng(n)
    decks = [
        np.arange(52),
        np.arange(26),
        np.array([card(r, s) for r in range(10, 15) for s in range(4)]),
    ]
    hands = np.stack(
        [rng.choice(deck, n, replace=False) for deck in decks for _ in range(1000)]
    ).astype(np.int32)
    scores = np.asarray(score_hands(jnp.asarray(hands)))
    assert scores.tolist() == [ref_score(h) for h in hands.tolist()]
    if n == 7:
        # Also the best of the 21 subsets under the five-card evaluator.
        subsets = np.array(list(itertools.combinations(range(7), 5)))
        five = np.asarray(score_five_card_hands(jnp.asarray(hands[:, subsets].reshape(-1, 5))))
        assert np.array_equal(scores, five.reshape(-1, 21).max(axis=1))


def test_five_card_hand_classes() -> None:
    """One hand per five-card hand class: 7,462 distinct scores in poker order,
    the same from both evaluators."""
    hands = []
    for ranks in itertools.combinations_with_replacement(range(2, 15), 5):
        if len(set(ranks)) == 1:
            continue
        held = Counter()
        hand = []
        for r in ranks:
            hand.append(card(r, held[r]))
            held[r] += 1
        if len(set(ranks)) == 5:
            hand[0] = card(ranks[0], 1)  # not a flush
        hands.append(hand)
    hands += [[card(r, 0) for r in ranks] for ranks in itertools.combinations(range(2, 15), 5)]
    keys = [five_card_key(h) for h in hands]
    assert len(set(keys)) == len(hands) == 7462

    scores = np.asarray(score_hands(jnp.asarray(hands, jnp.int32)))
    assert np.array_equal(scores, np.asarray(score_five_card_hands(jnp.asarray(hands, jnp.int32))))
    assert scores.tolist() == [encode(k) for k in keys]
    order = sorted(range(len(hands)), key=lambda i: keys[i])
    assert np.all(np.diff(scores[order]) > 0)


def test_straight_high_all_rank_sets() -> None:
    masks = np.array(list(itertools.product([False, True], repeat=13)))
    highs = np.asarray(jax.jit(jax.vmap(_straight_high))(jnp.asarray(masks)))
    for mask, high in zip(masks, highs):
        ranks = {r for r, held in zip(range(2, 15), mask) if held}
        if 14 in ranks:
            ranks.add(1)
        runs = [h for h in range(5, 15) if set(range(h - 4, h + 1)) <= ranks]
        assert high == max(runs, default=0)


def test_compare_hands_splits_ties_and_skips_folded() -> None:
    board = parse("2c 3d 4h 5s 6c")
    hole = [parse("Kh Kd"), parse("9h 9d"), parse("7h 8d")]
    hands = jnp.asarray([h + board for h in hole], jnp.int32)
    winners = _compare_hands(hands, jnp.array([False, False, False]))
    assert winners.tolist() == [False, False, True]
    winners = _compare_hands(hands, jnp.array([False, False, True]))
    assert winners.tolist() == [True, True, False]
