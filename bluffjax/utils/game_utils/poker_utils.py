import jax
import jax.numpy as jnp
from bluffjax.utils.typing import IntArray, BoolArray


@jax.jit
def _card_rank(card: IntArray) -> IntArray:
    mod = card % 13
    return jnp.where(mod == 0, 14, mod + 1)


@jax.jit
def _card_suit(card: IntArray) -> IntArray:
    return card // 13


def _encode_score(category: IntArray, tiebreaks: IntArray) -> IntArray:
    powers = jnp.array([15**4, 15**3, 15**2, 15**1, 15**0], dtype=jnp.int32)
    return category * (15**5) + jnp.sum(tiebreaks * powers)


@jax.jit
def _score_five_card_hand(ranks: IntArray, suits: IntArray) -> IntArray:
    ranks_sorted = jnp.sort(ranks)
    ranks_sorted_desc = ranks_sorted[::-1]
    rank_vals = jnp.arange(2, 15, dtype=jnp.int32)
    counts = jnp.bincount(ranks - 2, length=13)

    is_flush = jnp.all(suits == suits[0])
    unique_count = jnp.sum(counts > 0)
    max_rank = jnp.max(ranks_sorted)
    min_rank = jnp.min(ranks_sorted)
    is_wheel = jnp.all(ranks_sorted == jnp.array([2, 3, 4, 5, 14]))
    is_straight = (unique_count == 5) & ((max_rank - min_rank == 4) | is_wheel)
    straight_high = jnp.where(is_wheel, 5, max_rank)

    has_four = jnp.any(counts == 4)
    has_three = jnp.any(counts == 3)
    num_pairs = jnp.sum(counts == 2)
    has_full_house = has_three & (num_pairs >= 1)
    has_two_pair = num_pairs == 2
    has_pair = num_pairs == 1

    ranks_count4 = jnp.where(counts == 4, rank_vals, -1)
    quad_rank = jnp.max(ranks_count4)

    ranks_count3 = jnp.where(counts == 3, rank_vals, -1)
    trip_rank = jnp.max(ranks_count3)

    ranks_count2 = jnp.where(counts == 2, rank_vals, -1)
    pair_ranks_desc = jnp.sort(ranks_count2)[::-1]
    high_pair = pair_ranks_desc[0]
    low_pair = pair_ranks_desc[1]

    ranks_count1 = jnp.where(counts == 1, rank_vals, -1)
    kickers_desc = jnp.sort(ranks_count1)[::-1]

    score_high = _encode_score(1, ranks_sorted_desc)
    score_pair = _encode_score(
        2, jnp.array([high_pair, kickers_desc[0], kickers_desc[1], kickers_desc[2], 0])
    )
    score_two_pair = _encode_score(
        3, jnp.array([high_pair, low_pair, kickers_desc[0], 0, 0])
    )
    score_trips = _encode_score(
        4, jnp.array([trip_rank, kickers_desc[0], kickers_desc[1], 0, 0])
    )
    score_straight = _encode_score(5, jnp.array([straight_high, 0, 0, 0, 0]))
    score_flush = _encode_score(6, ranks_sorted_desc)
    score_full_house = _encode_score(7, jnp.array([trip_rank, high_pair, 0, 0, 0]))
    score_four = _encode_score(8, jnp.array([quad_rank, kickers_desc[0], 0, 0, 0]))
    score_straight_flush = _encode_score(9, jnp.array([straight_high, 0, 0, 0, 0]))

    score = score_high
    score = jnp.where(has_pair, score_pair, score)
    score = jnp.where(has_two_pair, score_two_pair, score)
    score = jnp.where(has_three, score_trips, score)
    score = jnp.where(is_straight, score_straight, score)
    score = jnp.where(is_flush, score_flush, score)
    score = jnp.where(has_full_house, score_full_house, score)
    score = jnp.where(has_four, score_four, score)
    score = jnp.where(is_straight & is_flush, score_straight_flush, score)
    return score


def _top_ranks(present: BoolArray, k: int) -> IntArray:
    """The k highest ranks flagged in present (13 flags, 2 to ace), descending.

    Padded with -1 when fewer than k ranks are flagged.
    """
    rank_vals = jnp.arange(2, 15, dtype=jnp.int32)
    # A flagged rank's place in descending order is the number flagged above it.
    place = jnp.cumsum(present[::-1])[::-1] - present
    in_place = present & (place == jnp.arange(k)[:, None])
    return jnp.max(jnp.where(in_place, rank_vals, -1), axis=1)


def _straight_high(present: BoolArray) -> IntArray:
    """High card of the best straight among the ranks flagged in present.

    present has 13 flags, 2 to ace. The wheel (A-2-3-4-5) is 5 high. Returns 0
    if there is no straight.
    """
    # Bit r - 1 holds rank r, and bit 0 holds the ace again, playing low.
    bits = jnp.sum(jnp.where(present, 2 << jnp.arange(13), 0)) | present[12]
    highs = jnp.arange(5, 15, dtype=jnp.int32)
    has_run = ((bits >> (highs - 5)) & 0b11111) == 0b11111
    return jnp.max(jnp.where(has_run, highs, 0))


@jax.jit
def _score_seven_card_hand(hand: IntArray) -> IntArray:
    """Score of the best five cards of a 5- to 7-card hand.

    Same encoding as _score_five_card_hand, and equal to its maximum over all
    five-card subsets, but read off the rank and suit counts directly.
    """
    ranks = _card_rank(hand)
    suits = _card_suit(hand)
    rank_vals = jnp.arange(2, 15, dtype=jnp.int32)
    is_rank = ranks[:, None] == rank_vals
    counts = jnp.sum(is_rank, axis=0)

    # Only one suit can hold five of at most seven cards.
    suit_counts = jnp.sum(suits[:, None] == jnp.arange(4), axis=0)
    flush_suit = jnp.argmax(suit_counts)
    is_flush = suit_counts[flush_suit] >= 5
    flush_ranks = jnp.any(is_rank & (suits == flush_suit)[:, None], axis=0)

    straight_high = _straight_high(counts > 0)
    straight_flush_high = jnp.where(is_flush, _straight_high(flush_ranks), 0)

    quad_rank = jnp.max(jnp.where(counts == 4, rank_vals, -1))
    trip_rank = jnp.max(jnp.where(counts == 3, rank_vals, -1))
    pair_ranks_desc = _top_ranks(counts == 2, 2)
    high_pair = pair_ranks_desc[0]
    low_pair = pair_ranks_desc[1]
    kickers_desc = _top_ranks(counts == 1, 5)
    flush_desc = _top_ranks(flush_ranks, 5)

    # The pair of a full house is the best other rank held at least twice, so
    # a second trips can play as the pair.
    full_house_pair = jnp.max(
        jnp.where((counts >= 2) & (rank_vals != trip_rank), rank_vals, -1)
    )
    # Quads and two pair play the best remaining card as the kicker, which
    # can come from another pair.
    quad_kicker = jnp.max(
        jnp.where((counts > 0) & (rank_vals != quad_rank), rank_vals, -1)
    )
    two_pair_kicker = jnp.max(
        jnp.where(
            (counts > 0) & (rank_vals != high_pair) & (rank_vals != low_pair),
            rank_vals,
            -1,
        )
    )

    has_four = quad_rank > 0
    has_three = trip_rank > 0
    num_pairs = jnp.sum(counts == 2)
    has_full_house = has_three & (full_house_pair > 0)

    score_high = _encode_score(1, kickers_desc)
    score_pair = _encode_score(
        2, jnp.array([high_pair, kickers_desc[0], kickers_desc[1], kickers_desc[2], 0])
    )
    score_two_pair = _encode_score(
        3, jnp.array([high_pair, low_pair, two_pair_kicker, 0, 0])
    )
    score_trips = _encode_score(
        4, jnp.array([trip_rank, kickers_desc[0], kickers_desc[1], 0, 0])
    )
    score_straight = _encode_score(5, jnp.array([straight_high, 0, 0, 0, 0]))
    score_flush = _encode_score(6, flush_desc)
    score_full_house = _encode_score(
        7, jnp.array([trip_rank, full_house_pair, 0, 0, 0])
    )
    score_four = _encode_score(8, jnp.array([quad_rank, quad_kicker, 0, 0, 0]))
    score_straight_flush = _encode_score(
        9, jnp.array([straight_flush_high, 0, 0, 0, 0])
    )

    score = score_high
    score = jnp.where(num_pairs >= 1, score_pair, score)
    score = jnp.where(num_pairs >= 2, score_two_pair, score)
    score = jnp.where(has_three, score_trips, score)
    score = jnp.where(straight_high > 0, score_straight, score)
    score = jnp.where(is_flush, score_flush, score)
    score = jnp.where(has_full_house, score_full_house, score)
    score = jnp.where(has_four, score_four, score)
    score = jnp.where(straight_flush_high > 0, score_straight_flush, score)
    return score


@jax.jit
def _get_bring_in_idx(door_cards: IntArray) -> IntArray:
    """Index of player with lowest door card. Ties broken by suit: clubs < diamonds < hearts < spades."""
    ranks = _card_rank(door_cards)  # 2-14 (2 low, Ace high)
    suits = door_cards // 13
    # For bring-in, 2 is lowest. composite = (rank-2)*4 + suit so lower rank = lower composite
    composite = (ranks - 2) * 4 + suits
    return jnp.argmin(composite)


@jax.jit
def _score_visible_upcards(cards: IntArray) -> IntArray:
    """Score 2-4 visible upcards for first-to-act determination. Higher = better."""
    ranks = _card_rank(cards)
    rank_vals = jnp.arange(2, 15, dtype=jnp.int32)
    counts = jnp.bincount(ranks - 2, length=13)

    has_four = jnp.any(counts == 4)
    has_three = jnp.any(counts == 3)
    num_pairs = jnp.sum(counts == 2)
    has_two_pair = num_pairs == 2
    has_pair = num_pairs >= 1

    ranks_count4 = jnp.where(counts == 4, rank_vals, -1)
    quad_rank = jnp.max(ranks_count4)

    ranks_count3 = jnp.where(counts == 3, rank_vals, -1)
    trip_rank = jnp.max(ranks_count3)

    ranks_count2 = jnp.where(counts == 2, rank_vals, -1)
    pair_ranks_desc = jnp.sort(ranks_count2)[::-1]
    high_pair = pair_ranks_desc[0]
    low_pair = pair_ranks_desc[1]

    ranks_count1 = jnp.where(counts == 1, rank_vals, -1)
    kickers_desc = jnp.sort(ranks_count1)[::-1]
    k0 = jnp.maximum(kickers_desc[0], 0)
    k1 = jnp.maximum(kickers_desc[1], 0)
    k2 = jnp.maximum(kickers_desc[2], 0)
    k3 = jnp.maximum(kickers_desc[3], 0)

    score_high = _encode_score(1, jnp.array([k0, k1, k2, k3, 0]))
    score_pair = _encode_score(2, jnp.array([high_pair, k0, k1, k2, 0]))
    score_two_pair = _encode_score(3, jnp.array([high_pair, low_pair, k0, 0, 0]))
    score_trips = _encode_score(4, jnp.array([trip_rank, k0, k1, 0, 0]))
    score_four = _encode_score(5, jnp.array([quad_rank, k0, 0, 0, 0]))

    score = score_high
    score = jnp.where(has_pair, score_pair, score)
    score = jnp.where(has_two_pair, score_two_pair, score)
    score = jnp.where(has_three, score_trips, score)
    score = jnp.where(has_four, score_four, score)
    return score


def _compare_hands(all_hands: IntArray, folded: BoolArray) -> BoolArray:
    def score_for_player(player_cards: IntArray, is_folded: BoolArray) -> IntArray:
        return jnp.where(is_folded, -1, _score_seven_card_hand(player_cards))

    scores = jax.vmap(score_for_player)(all_hands, folded)
    max_score = jnp.max(scores)
    return (scores == max_score) & (~folded)


def _compare_five_card_hands(all_hands: IntArray, folded: BoolArray) -> BoolArray:
    """Compare 5-card hands. all_hands shape (num_agents, 5)."""

    def score_for_player(player_cards: IntArray, is_folded: BoolArray) -> IntArray:
        ranks = _card_rank(player_cards)
        suits = _card_suit(player_cards)
        return jnp.where(is_folded, -1, _score_five_card_hand(ranks, suits))

    scores = jax.vmap(score_for_player)(all_hands, folded)
    max_score = jnp.max(scores)
    return (scores == max_score) & (~folded)
