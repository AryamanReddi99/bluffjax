"""7-Card Stud metrics (bluffjax-metrics.md), from environment steps.

Actions: 0 call, 1 raise, 2 fold, 3 check. The antes are posted at the deal.
The hand's first action is the bring-in's decision (it can't fold or check):
call posts the bring-in, the forced minimum, which is neither an aggressive
action nor a call (it answers no bet; nothing is counted for it), and raise
completes to one small bet. Aggressive actions are the raises (action 1):
completing the bring-in (by the bring-in at its first decision, or by a later
player after the bring-in was posted), the first bet of a later street and
every raise. Aggressive action, would lose at showdown, bluff and bluff
success are defined in bluffjax.metrics._poker_draw_stud. Each rate below is
numerator / denominator, NaN without events, and summarize() also returns
the denominator as n_<name>.

Final hands. The env shuffles the deck at the deal and deals each later
street from its front to the players still in at that point (a fold changes
who gets which card; with 8+ players a short deck turns a street into one
community card). The final hand of every player at the time of an action is
therefore the seven cards they would hold at a showdown if nobody else
folded: the streets after the current one dealt from the deck to the players
still in, exactly as the env deals them (SevenCardStud._deal_street). Heads-up
it is the actual final hand of every hand that reaches a showdown. Scored
with the env's showdown scorer, poker_utils._score_seven_card_hand.

Board. A player's board on the current street is the score of its face-up
cards as the env ranks them for the betting order
(SevenCardStud._first_to_act_by_upcards: poker_utils._score_visible_upcards
of the 2-4 up cards, or the five-card score when a 7th-street community card
makes a five-card board), on 3rd street _score_visible_upcards of the door
card alone (the scorers' exact values, computed without sorting). The
bettor's board is ahead if it scores strictly higher than the best board
among the opponents still in.

Per action, credited to the acting player (wins to the bettor, counted at the
step that ends the hand; so the counts of one update's rollout can hold the
win of a bluff or steal made in the previous rollout):
- bluff_rate = bluffs / aggressive.
- bluff_success = bluff_wins / bluffs: the hand then ends without a showdown
  and the actions after the bluff were all folds.
- correct_fold_rate = correct_folds / folds: folds by a player whose final
  hand (if it had stayed in) would have lost against the players still in at
  the fold.
- bluff_rate_board_ahead = bluffs_board_ahead / aggressive_board_ahead and
  bluff_success_board_ahead = bluff_wins_board_ahead / bluffs_board_ahead:
  the same for aggressive actions made with the board ahead;
  bluff_rate_board_not_ahead, bluff_success_board_not_ahead (denominators
  aggressive_board_not_ahead, bluffs_board_not_ahead) for the others (board
  tied or behind).
- steal_success = steal_wins / steals: a steal is the first raise on 3rd
  street (raise count of 3rd street 0 before it), i.e. the completion of the
  bring-in, by the bring-in player instead of posting or by a later player
  after it posted (the bring-in has no option, so it never raises a bring-in
  that everyone just called); it wins if every other player then folds
  before anyone calls or raises.

Per hand, credited to every player (all are dealt in), counted at the step
that ends the hand:
- fold_street_<k> = fold_street[k - 3] / hands for k = 3..7: share of hands
  that end because all players but one folded, the last fold on street k.
- showdown_rate = showdowns / hands: share of hands that reach a showdown.
  The six rates sum to 1.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bluffjax.environments.seven_card_stud.seven_card_stud import SevenCardStudState
from bluffjax.metrics._poker_draw_stud import (
    NO_PLAYER,
    five_card_score,
    unanswered_aggressor,
    visible_score,
    won_without_showdown,
    would_lose,
)
from bluffjax.metrics.base import Counts, GameMetrics, credit, rate
from bluffjax.utils.game_utils.poker_utils import _score_seven_card_hand
from bluffjax.utils.typing import Any, Array, BoolArray, FloatArray, IntArray

CALL, RAISE, FOLD, CHECK = 0, 1, 2, 3
NUM_STREETS = 5  # 3rd to 7th street, stage 0 to 4
FIRST_STREET = 3


def final_hands(env: Any, state: SevenCardStudState) -> IntArray:
    """(num_agents, 7) cards of every player at a showdown if nobody else
    folds: the streets after state.stage dealt by the env's own rule to the
    players still in (folded players' later slots stay -1)."""
    dealt = (state.agent_cards, state.num_dealt, state.community)
    for street in range(1, NUM_STREETS):
        cards, num_dealt, community = dealt
        new = env._deal_street(
            state.replace(agent_cards=cards, num_dealt=num_dealt, community=community),
            jnp.int32(street),
            state.folded,
        )
        dealt = jax.tree_util.tree_map(
            lambda a, b: jnp.where(street > state.stage, a, b), new, dealt
        )
    return dealt[0]


def board_scores(state: SevenCardStudState) -> IntArray:
    """(num_agents,) score of every player's face-up cards on the current
    street, as SevenCardStud._first_to_act_by_upcards ranks them (on 3rd
    street the door card alone): _score_visible_upcards of 1-4 cards, or the
    five-card score of a 7th-street board with a community card.

    The scores are the env scorers' values (computed without sorting, see
    _poker_draw_stud). A community card on 7th street needs more than 7
    players (7 cards each for at most 7 players fit in the deck), so the
    five-card board is only scored then.
    """
    up = jnp.maximum(state.agent_cards[:, 2:], 0)  # face-up card of streets 0-4
    boards = [jax.vmap(visible_score)(up[:, :k]) for k in (1, 2, 3, 4)]
    num_up = jnp.minimum(state.stage + 1, 4)
    scores = jnp.select([num_up == k for k in (1, 2, 3)], boards[:3], boards[3])
    if 7 * state.agent_cards.shape[0] > 52:
        five_card_board = (state.stage == 4) & state.community[4]
        scores = jnp.where(five_card_board, jax.vmap(five_card_score)(up), scores)
    return scores


class SevenCardStudCarry(NamedTuple):
    aggressor: IntArray  # bettor of the bet or raise followed only by folds
    aggressor_bluffed: BoolArray  # that bet or raise was a bluff
    aggressor_ahead: BoolArray  # ... made with the board ahead
    aggressor_steal: BoolArray  # ... the first raise on 3rd street


class Metrics(GameMetrics):
    rate_names = (
        "bluff_rate",
        "bluff_success",
        "correct_fold_rate",
        "bluff_rate_board_ahead",
        "bluff_success_board_ahead",
        "bluff_rate_board_not_ahead",
        "bluff_success_board_not_ahead",
        "steal_success",
        *(f"fold_street_{k}" for k in range(FIRST_STREET, FIRST_STREET + NUM_STREETS)),
        "showdown_rate",
    )

    def init_carry(self) -> SevenCardStudCarry:
        return SevenCardStudCarry(
            aggressor=jnp.int32(NO_PLAYER),
            aggressor_bluffed=jnp.bool_(False),
            aggressor_ahead=jnp.bool_(False),
            aggressor_steal=jnp.bool_(False),
        )

    def step(
        self,
        carry: SevenCardStudCarry,
        state: SevenCardStudState,
        action: Array,
        next_state: SevenCardStudState,
        reward: FloatArray,
        done: BoolArray,
        info: dict,
        probs: FloatArray | None = None,
    ) -> tuple[SevenCardStudCarry, Counts]:
        n = self.num_agents
        player = state.current_player_idx
        in_hand = ~state.folded
        others = in_hand & (jnp.arange(n) != player)

        # Posting the bring-in (CALL at the hand's first action) counts as
        # nothing: not aggressive, no fold, and as no bet is open before it,
        # the carry has no aggressor that it could answer.
        aggressive = action == RAISE
        fold = action == FOLD
        scores = jax.vmap(_score_seven_card_hand)(
            jnp.maximum(final_hands(self.env, state), 0)
        )
        loses = would_lose(scores, in_hand, player)
        bluff = aggressive & loses
        board = board_scores(state)
        ahead = board[player] > jnp.max(jnp.where(others, board, -1))
        steal = aggressive & (state.stage == 0) & (state.raise_nums[0] == 0)

        last = SevenCardStudCarry(
            aggressor=unanswered_aggressor(carry.aggressor, player, aggressive, fold),
            aggressor_bluffed=jnp.where(aggressive, bluff, carry.aggressor_bluffed),
            aggressor_ahead=jnp.where(aggressive, ahead, carry.aggressor_ahead),
            aggressor_steal=jnp.where(aggressive, steal, carry.aggressor_steal),
        )
        folded_after = state.folded.at[player].set(state.folded[player] | fold)
        uncontested, winner = won_without_showdown(folded_after, done)
        won = uncontested & (last.aggressor == winner)
        bluff_won = won & last.aggressor_bluffed

        everyone = jnp.ones(n, dtype=jnp.float32)
        street = jax.nn.one_hot(state.stage, NUM_STREETS, dtype=jnp.float32)
        counts = {
            "aggressive": credit(n, player, aggressive),
            "bluffs": credit(n, player, bluff),
            "bluff_wins": credit(n, winner, bluff_won),
            "folds": credit(n, player, fold),
            "correct_folds": credit(n, player, fold & loses),
            "aggressive_board_ahead": credit(n, player, aggressive & ahead),
            "bluffs_board_ahead": credit(n, player, bluff & ahead),
            "bluff_wins_board_ahead": credit(n, winner, bluff_won & last.aggressor_ahead),
            "aggressive_board_not_ahead": credit(n, player, aggressive & ~ahead),
            "bluffs_board_not_ahead": credit(n, player, bluff & ~ahead),
            "bluff_wins_board_not_ahead": credit(
                n, winner, bluff_won & ~last.aggressor_ahead
            ),
            "steals": credit(n, player, steal),
            "steal_wins": credit(n, winner, won & last.aggressor_steal),
            "hands": everyone * done,
            "fold_street": everyone[:, None] * street * uncontested,
            "showdowns": everyone * (done & ~uncontested),
        }
        carry = jax.tree_util.tree_map(
            lambda reset, kept: jnp.where(done, reset, kept), self.init_carry(), last
        )
        return carry, counts

    def summarize(self, totals: Counts) -> dict[str, FloatArray]:
        t = totals
        out = {
            "bluff_rate": rate(t["bluffs"], t["aggressive"]),
            "n_aggressive": t["aggressive"],
            "bluff_success": rate(t["bluff_wins"], t["bluffs"]),
            "n_bluffs": t["bluffs"],
            "correct_fold_rate": rate(t["correct_folds"], t["folds"]),
            "n_folds": t["folds"],
        }
        for board in ("board_ahead", "board_not_ahead"):
            out[f"bluff_rate_{board}"] = rate(
                t[f"bluffs_{board}"], t[f"aggressive_{board}"]
            )
            out[f"n_aggressive_{board}"] = t[f"aggressive_{board}"]
            out[f"bluff_success_{board}"] = rate(
                t[f"bluff_wins_{board}"], t[f"bluffs_{board}"]
            )
            out[f"n_bluffs_{board}"] = t[f"bluffs_{board}"]
        out["steal_success"] = rate(t["steal_wins"], t["steals"])
        out["n_steals"] = t["steals"]
        for k in range(NUM_STREETS):
            out[f"fold_street_{FIRST_STREET + k}"] = rate(t["fold_street"][k], t["hands"])
        out["showdown_rate"] = rate(t["showdowns"], t["hands"])
        out["n_hands"] = t["hands"]
        return out
