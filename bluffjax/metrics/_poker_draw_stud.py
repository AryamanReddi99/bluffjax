"""Poker definitions shared by the 5-Card Draw and 7-Card Stud metrics.

- Aggressive action: a bet or raise of any size. Antes and blinds are forced
  and not actions. In Stud the bring-in player's first decision is to post
  the bring-in or complete: posting is the forced minimum and is neither an
  aggressive action nor a call; completing the bring-in (by the bring-in
  player, or by a later player after it posted) is a raise.
- Would lose at showdown (at the time of an action): the acting player's
  final hand is strictly worse, by the env's hand scorer, than the final hand
  of at least one opponent still in the hand (not folded; all-in players are
  in). Ties don't lose.
- Bluff: an aggressive action made with a hand that would lose at showdown.
- Successful bluff: the hand then ends without a showdown because every
  opponent still in folds to it, i.e. the actions after the bluff until the
  end of the hand are all folds (heads-up: the opponent folds directly to the
  bluff). A later call, bet or raise answers the bluff. (Both envs allow
  folding when checking is free, so "no further bet or raise" alone would
  also count a bluff that was called and then won because the caller folded
  for free on a later street; that isn't a bluff that made anyone fold.)
  Only the last aggressive action of a hand can succeed.

When counts are credited (both games): an event's outcome (a bluff's,
snow's or steal's success, how the hand ends) is only known when the hand
ends, so every count of a hand, of its events and of their outcomes alike,
is accumulated in the carry and credited, to the player it describes, at
the step where the hand ends (the step with `done`, whose next_state is
already the next hand after the env's auto-reset); every other step credits
nothing. The counts of a rollout therefore cover exactly the hands that
ended in it, and every rate of a rollout is a share of events of those
hands (a success rate can't exceed 1). A hand still running when training
stops is not counted.

These helpers work on one env (no batch axis) and any number of players.

Cost. The metrics run at every env step, vmapped over the envs, where
jnp.sort is slow; the scorers below give exactly the values of the env's
sort-based scorers without sorting (tests compare them on every possible
input).
"""

import jax.numpy as jnp

from bluffjax.metrics.base import Counts
from bluffjax.utils.game_utils.poker_utils import (
    _card_rank,
    _encode_score,
    _score_seven_card_hand,
    _top_ranks,
)
from bluffjax.utils.typing import BoolArray, IntArray

#: no player (e.g. no unanswered bet or raise)
NO_PLAYER = -1
#: a poker_utils hand score is category * 15**5 + tiebreaks, so score //
#: CATEGORY_BASE is its category: 1 high card, 2 pair, 3 two pair, 4 trips,
#: 5 straight, 6 flush, 7 full house, 8 quads, 9 straight flush
CATEGORY_BASE = 15**5
HIGH_CARD = 1


def zero_counts(names: tuple[str, ...], num_agents: int) -> Counts:
    """Counts of a hand before its first action: name -> zeros (num_agents,)."""
    return {name: jnp.zeros(num_agents, jnp.float32) for name in names}


def hand_end_counts(
    hand: Counts, step: Counts, done: BoolArray
) -> tuple[Counts, Counts]:
    """Adds a step's event counts to those of the hand in progress (the
    carry's, same names). Returns the hand's counts after the step (to keep
    in the carry, which the caller resets when `done`) and the counts to
    credit at this step: all of the hand's when it ends here, zeros
    otherwise."""
    hand = {name: hand[name] + step[name] for name in hand}
    return hand, {name: done * value for name, value in hand.items()}


def five_card_score(cards: IntArray) -> IntArray:
    """poker_utils._score_five_card_hand of a 5-card hand (the env's showdown
    scorer), computed by _score_seven_card_hand, which is equal to it on
    every 5-card hand (tests/test_poker_utils.py) and doesn't sort."""
    return _score_seven_card_hand(cards)


def visible_score(cards: IntArray) -> IntArray:
    """poker_utils._score_visible_upcards of 1-4 face-up cards (the env's
    betting-order scorer), with its sorts replaced by _top_ranks: the same
    categories (high card, pair, two pair, trips, quads) and tiebreaks."""
    ranks = _card_rank(cards)
    rank_vals = jnp.arange(2, 15, dtype=jnp.int32)
    counts = jnp.sum(ranks[:, None] == rank_vals, axis=0)
    quad_rank = jnp.max(jnp.where(counts == 4, rank_vals, -1))
    trip_rank = jnp.max(jnp.where(counts == 3, rank_vals, -1))
    num_pairs = jnp.sum(counts == 2)
    high_pair, low_pair = _top_ranks(counts == 2, 2)
    k = jnp.maximum(_top_ranks(counts == 1, 4), 0)
    score = _encode_score(1, jnp.array([k[0], k[1], k[2], k[3], 0]))
    score = jnp.where(
        num_pairs >= 1, _encode_score(2, jnp.array([high_pair, k[0], k[1], k[2], 0])), score
    )
    score = jnp.where(
        num_pairs == 2, _encode_score(3, jnp.array([high_pair, low_pair, k[0], 0, 0])), score
    )
    score = jnp.where(
        trip_rank > 0, _encode_score(4, jnp.array([trip_rank, k[0], k[1], 0, 0])), score
    )
    score = jnp.where(
        quad_rank > 0, _encode_score(5, jnp.array([quad_rank, k[0], 0, 0, 0])), score
    )
    return score


def would_lose(scores: IntArray, in_hand: BoolArray, player: IntArray) -> BoolArray:
    """True if another player in the hand has a strictly better score.

    scores: (num_agents,) final-hand scores (higher is better); in_hand:
    (num_agents,) players still in the hand (player's own flag is ignored).
    """
    others = in_hand & (jnp.arange(scores.shape[0]) != player)
    return jnp.any(others & (scores > scores[player]))


def unanswered_aggressor(
    aggressor: IntArray, player: IntArray, aggressive: BoolArray, fold: BoolArray
) -> IntArray:
    """The player whose bet or raise has been followed only by folds.

    After player's action: player itself if it bet or raised, unchanged if it
    folded, NO_PLAYER after a check or call (that answers the bet or raise).
    """
    return jnp.where(aggressive, player, jnp.where(fold, aggressor, NO_PLAYER))


def won_without_showdown(
    folded_after: BoolArray, done: BoolArray
) -> tuple[BoolArray, IntArray]:
    """(the hand ended because all players but one folded, that player).

    folded_after: (num_agents,) who has folded after the step's action (when
    the hand is done, the env's next_state is already the next hand).
    """
    remaining = ~folded_after
    return done & (remaining.sum() == 1), jnp.argmax(remaining)
