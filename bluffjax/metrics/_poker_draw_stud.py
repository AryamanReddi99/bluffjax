"""Poker definitions shared by the 5-Card Draw and 7-Card Stud metrics.

- Aggressive action: a bet or raise of any size. Antes, blinds and the Stud
  bring-in are forced and not actions; completing the bring-in is a raise.
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

These helpers work on one env (no batch axis) and any number of players.
"""

import jax.numpy as jnp

from bluffjax.utils.typing import BoolArray, IntArray

#: no player (e.g. no unanswered bet or raise)
NO_PLAYER = -1
#: score // CATEGORY_BASE is the category of a poker_utils hand score:
#: 1 high card, 2 pair, 3 two pair, 4 trips, 5 straight, 6 flush, 7 full
#: house, 8 quads, 9 straight flush
CATEGORY_BASE = 15**5
HIGH_CARD = 1


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
