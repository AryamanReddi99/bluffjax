"""
Parts shared by the Limit and No-Limit Hold'em metrics
(bluffjax.metrics.texas_limit_holdem, bluffjax.metrics.texas_nolimit_holdem):
final hand strengths, bluffs and their success, correct folds, showdowns.

Definitions (common to the poker games of the suite):
- Aggressive action: a bet or raise of any size, all-in included. Posting
  the blinds is not an action.
- Would lose at showdown, at the time of an action: the acting player's final
  hand (its hole cards and the whole board, which the env deals in advance)
  scores strictly lower, with the env's own hand scorer, than the final hand
  of at least one opponent still in the hand (not folded; all-in players are
  in). Ties don't lose.
- Bluff: an aggressive action made with a hand that would lose at showdown.
- Successful bluff: the hand ends without showdown with every opponent still
  in having folded to the bluff: every action after the bluff, up to the end
  of the hand, is a fold (heads-up: the opponent folds directly to it). A
  call, check, bet or raise by anyone after the bluff makes it unsuccessful.
- Correct fold: a fold made with a hand that would lose at showdown against
  the opponents still in at the fold.

When counts are credited: a hand's counts are only known (bluff success, a
showdown win) or only meaningful together (VPIP of the hand) when it ends,
so every count of a hand is accumulated in the carry and credited, to the
player it describes, at the step where the hand ends. The counts of a
rollout therefore cover exactly the hands that ended in it, and every rate
of a rollout is a share of events of those hands. A hand still running when
training stops is not counted.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bluffjax.metrics.base import Counts
from bluffjax.utils.game_utils.poker_utils import _score_seven_card_hand
from bluffjax.utils.typing import BoolArray, FloatArray, IntArray

STREETS = ("preflop", "flop", "turn", "river")
NUM_STREETS = len(STREETS)


def final_scores(state) -> IntArray:
    """(num_agents,) env hand score of every player's final hand: its two
    hole cards and the five board cards (pre-dealt), scored like the env's
    showdown (poker_utils._score_seven_card_hand, higher is better)."""
    board = jnp.concatenate(
        [state.flop_cards, jnp.stack([state.turn_card, state.river_card])]
    )
    num_agents = state.agent_cards.shape[0]
    hands = jnp.concatenate(
        [state.agent_cards, jnp.broadcast_to(board, (num_agents, 5))], axis=1
    )
    return jax.vmap(_score_seven_card_hand)(hands.astype(jnp.int32))


def would_lose(scores: IntArray, player: IntArray, in_hand: BoolArray) -> BoolArray:
    """Whether `player`'s final hand is strictly worse than the final hand of
    at least one other player in `in_hand` (ties don't lose)."""
    num_agents = scores.shape[0]
    opponents = in_hand & (jnp.arange(num_agents) != player)
    best_opponent = jnp.max(jnp.where(opponents, scores, -1))
    return scores[player] < best_opponent


class BluffCarry(NamedTuple):
    """Bluff and fold counts of the hand in progress, and the bluff that can
    still succeed."""

    aggressive: FloatArray  # (num_agents, 4) bets and raises by player, street
    bluffs: FloatArray  # (num_agents, 4)
    folds: FloatArray  # (num_agents,)
    correct_folds: FloatArray  # (num_agents,)
    live: BoolArray  # the last bet or raise was a bluff and only folds followed
    bluffer: IntArray  # who made that bet or raise
    bluff_street: IntArray  # on which street


def init_bluff_carry(num_agents: int) -> BluffCarry:
    return BluffCarry(
        aggressive=jnp.zeros((num_agents, NUM_STREETS), jnp.float32),
        bluffs=jnp.zeros((num_agents, NUM_STREETS), jnp.float32),
        folds=jnp.zeros((num_agents,), jnp.float32),
        correct_folds=jnp.zeros((num_agents,), jnp.float32),
        live=jnp.array(False),
        bluffer=jnp.int32(0),
        bluff_street=jnp.int32(0),
    )


class ActionInfo(NamedTuple):
    """What one env action is, from the state before it."""

    actor: IntArray  # player acting
    street: IntArray  # 0 preflop .. 3 river
    aggressive: BoolArray  # a bet or raise (any size, all-in included)
    call: BoolArray  # puts chips in to match a bet (not a check)
    fold: BoolArray


def bluff_step(
    carry: BluffCarry,
    act: ActionInfo,
    lose: BoolArray,
    in_hand_after: BoolArray,
    done: BoolArray,
) -> tuple[BluffCarry, Counts]:
    """One action in the bluff / fold bookkeeping.

    lose: the actor would lose at showdown against the opponents in the hand
    before the action. in_hand_after: (num_agents,) players not folded after
    the action. Returns the carry (not yet reset at the end of the hand) and
    the hand's counts when `done` (zeros otherwise): aggressive, bluffs and
    bluff_successes by (player, street), folds and correct_folds by player.
    """
    num_agents = carry.folds.shape[0]
    who = jax.nn.one_hot(act.actor, num_agents, dtype=jnp.float32)
    where = who[:, None] * jax.nn.one_hot(act.street, NUM_STREETS, dtype=jnp.float32)
    bluff = act.aggressive & lose
    passive = ~act.aggressive & ~act.fold  # call or check
    carry = BluffCarry(
        aggressive=carry.aggressive + act.aggressive * where,
        bluffs=carry.bluffs + bluff * where,
        folds=carry.folds + act.fold * who,
        correct_folds=carry.correct_folds + (act.fold & lose) * who,
        # Only the last bet or raise can still win uncontested, and only if
        # nothing but folds follows it.
        live=jnp.where(act.aggressive, bluff, carry.live & ~passive),
        bluffer=jnp.where(act.aggressive, act.actor, carry.bluffer),
        bluff_street=jnp.where(act.aggressive, act.street, carry.bluff_street),
    )
    uncontested = (in_hand_after.sum() == 1) & in_hand_after[carry.bluffer]
    success = done & carry.live & uncontested
    successes = (
        success
        * jax.nn.one_hot(carry.bluffer, num_agents, dtype=jnp.float32)[:, None]
        * jax.nn.one_hot(carry.bluff_street, NUM_STREETS, dtype=jnp.float32)
    )
    counts = {
        "aggressive": done * carry.aggressive,
        "bluffs": done * carry.bluffs,
        "bluff_successes": successes,
        "folds": done * carry.folds,
        "correct_folds": done * carry.correct_folds,
    }
    return carry, counts


def reset_where(done: BoolArray, carry, init):
    """carry, or the initial carry where the hand ended."""
    return jax.tree_util.tree_map(lambda c, i: jnp.where(done, i, c), carry, init)


def showdown_counts(
    in_hand_after: BoolArray, reward: FloatArray, done: BoolArray
) -> Counts:
    """Counts of a hand at its end: hands (every player dealt in),
    showdown_hands (every player dealt in, if the hand reached showdown),
    showdowns (players in the showdown) and showdown_wins (players in the
    showdown with a positive reward), by player."""
    showdown = done & (in_hand_after.sum() >= 2)
    ones = jnp.ones_like(reward, dtype=jnp.float32)
    in_showdown = showdown & in_hand_after
    return {
        "hands": done * ones,
        "showdown_hands": showdown * ones,
        "showdowns": in_showdown.astype(jnp.float32),
        "showdown_wins": (in_showdown & (reward > 0)).astype(jnp.float32),
    }
