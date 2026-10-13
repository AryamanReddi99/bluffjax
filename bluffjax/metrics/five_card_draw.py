"""5-Card Draw metrics (bluffjax-metrics.md), from environment steps.

Hands are scored with the env's showdown scorer (the values of
poker_utils._score_five_card_hand, computed without sorting: see
_poker_draw_stud.five_card_score), and a hand's category is score // 15**5:
high card (1) < pair < two pair < trips < straight < flush < full house <
quads < straight flush (9). Aggressive
action, would lose at showdown, bluff and bluff success are defined in
bluffjax.metrics._poker_draw_stud. Every count is credited to one player
(shape (num_agents,)); each rate below is numerator / denominator, NaN
without events, and summarize() also returns the denominator as n_<name>.

The draw. The env gives every player still in the hand (all-in players
included) exactly one draw action per hand, 5 + k with keep pattern k (bit j
keeps the j-th smallest card), so a draw is one step at the draw stage and
is counted once per player per hand. Credited to the drawer.
- cards_drawn = cards / draws: mean number of cards exchanged per draw
  (0 to 5).
- draw_improve_rate = draws_improved / draws_known: share of draws after
  which the hand's category is higher than before (a better kicker or a
  higher pair in the same category doesn't count). The hand after the draw
  is read from next_state, or, when the draw ends the hand (a showdown right
  after the draw because fewer than two players can bet; next_state is then
  already the next hand), rebuilt from the stock as the env deals it: the
  kept cards plus the next cards of shuffled_deck. That is exact unless the
  stock runs out at that draw and the env reshuffles discards with a key the
  metric doesn't see, which needs 6 or more players; such a draw isn't
  "known" and is left out of this rate only. With up to 5 players
  draws_known = draws.
- snow_rate = snows / high_card_draws: share of the draws taken with a
  high-card hand (category 1 before the draw) in which the player draws 0
  cards (stands pat).
- snow_success = snow_wins / snows: share of snows after which the snower
  wins the hand without a showdown (every other player folds). Credited to
  the snower.

Post-draw betting (stage 2), when every player still in has drawn and the
hands are final. Aggressive actions are the raises (half pot, pot, all-in;
every legal raise raises the highest bet). Would lose: the acting player's
current hand scores strictly below the current hand of at least one other
player who hasn't folded (all-in players included). Credited to the acting
player, except wins, credited to the bluffer.
- postdraw_bluff_rate = postdraw_bluffs / postdraw_aggressive.
- postdraw_bluff_success = postdraw_bluff_wins / postdraw_bluffs: the hand
  then ends without a showdown and the actions after the bluff were all
  folds.
- postdraw_correct_fold_rate = postdraw_correct_folds / postdraw_folds:
  share of post-draw folds by a player whose hand would have lost against the
  players still in at the fold.
Betting before the draw isn't measured: the final hands aren't known then.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bluffjax.environments.five_card_draw.five_card_draw import (
    DECK_SIZE,
    FOLD,
    HAND_SIZE,
    NUM_BETTING_ACTIONS,
    FiveCardDrawState,
)
from bluffjax.metrics._poker_draw_stud import (
    CATEGORY_BASE,
    HIGH_CARD,
    NO_PLAYER,
    credit,
    five_card_score,
    unanswered_aggressor,
    won_without_showdown,
    would_lose,
)
from bluffjax.metrics.base import Counts, GameMetrics, rate
from bluffjax.utils.typing import Array, BoolArray, FloatArray, IntArray

DRAW_STAGE = 1
POST_DRAW_STAGE = 2
# raise half the pot, raise the pot, all-in
FIRST_RAISE, LAST_RAISE = 1, 3


def hand_after_draw(
    state: FiveCardDrawState, player: IntArray, keep: BoolArray
) -> tuple[IntArray, BoolArray]:
    """player's cards after keeping `keep` (5 flags over its sorted hand) and
    drawing the rest, without the env's random key.

    The env (FiveCardDraw._draw) deals the replacements from the stock,
    shuffled_deck[deck_idx:], in order; only when the stock has fewer cards
    than are drawn does it reshuffle discards with its key. Returns the cards
    (in no particular order) and whether they are exact (the stock was big
    enough).
    """
    hand = state.agent_cards[player]
    discard = ~keep
    # the j-th discarded card is replaced by the j-th card of the stock
    nth = jnp.cumsum(discard) - 1
    stock = state.shuffled_deck[jnp.minimum(state.deck_idx + nth, DECK_SIZE - 1)]
    exact = state.deck_idx + discard.sum() <= DECK_SIZE
    return jnp.where(keep, hand, stock), exact


class FiveCardDrawCarry(NamedTuple):
    snowed: BoolArray  # (num_agents,) stood pat with a high card at this hand's draw
    aggressor: IntArray  # bettor of the post-draw bet or raise followed only by folds
    aggressor_bluffed: BoolArray  # that bet or raise was a bluff


class Metrics(GameMetrics):
    rate_names = (
        "cards_drawn",
        "draw_improve_rate",
        "snow_rate",
        "snow_success",
        "postdraw_bluff_rate",
        "postdraw_bluff_success",
        "postdraw_correct_fold_rate",
    )

    def init_carry(self) -> FiveCardDrawCarry:
        return FiveCardDrawCarry(
            snowed=jnp.zeros(self.num_agents, dtype=bool),
            aggressor=jnp.int32(NO_PLAYER),
            aggressor_bluffed=jnp.bool_(False),
        )

    def step(
        self,
        carry: FiveCardDrawCarry,
        state: FiveCardDrawState,
        action: Array,
        next_state: FiveCardDrawState,
        reward: FloatArray,
        done: BoolArray,
        info: dict,
        probs: FloatArray | None = None,
    ) -> tuple[FiveCardDrawCarry, Counts]:
        n = self.num_agents
        player = state.current_player_idx
        in_hand = ~state.folded
        scores = jax.vmap(five_card_score)(state.agent_cards)

        # The draw: one action per player still in, at the draw stage.
        draw = (
            (state.stage == DRAW_STAGE)
            & (action >= NUM_BETTING_ACTIONS)
            & in_hand[player]
        )
        keep = (((action - NUM_BETTING_ACTIONS) >> jnp.arange(HAND_SIZE)) & 1) == 1
        num_drawn = HAND_SIZE - keep.sum()
        rebuilt, rebuilt_exact = hand_after_draw(state, player, keep)
        new_hand = jnp.where(done, rebuilt, next_state.agent_cards[player])
        known = draw & (~done | rebuilt_exact)
        category = scores[player] // CATEGORY_BASE
        improved = five_card_score(new_hand) // CATEGORY_BASE > category
        high_card_draw = draw & (category == HIGH_CARD)
        snow = high_card_draw & (num_drawn == 0)

        # Post-draw betting: hands are final.
        post_draw = state.stage == POST_DRAW_STAGE
        aggressive = post_draw & (action >= FIRST_RAISE) & (action <= LAST_RAISE)
        fold = post_draw & (action == FOLD)
        loses = would_lose(scores, in_hand, player)
        bluff = aggressive & loses
        aggressor = jnp.where(
            post_draw,
            unanswered_aggressor(carry.aggressor, player, aggressive, fold),
            carry.aggressor,
        )
        aggressor_bluffed = jnp.where(aggressive, bluff, carry.aggressor_bluffed)
        snowed = carry.snowed.at[player].set(carry.snowed[player] | snow)

        # A fold at either betting stage can end the hand.
        any_fold = (state.stage != DRAW_STAGE) & (action == FOLD)
        folded_after = state.folded.at[player].set(state.folded[player] | any_fold)
        uncontested, winner = won_without_showdown(folded_after, done)
        bluff_won = uncontested & (aggressor == winner) & aggressor_bluffed
        snow_won = uncontested & snowed[winner]

        counts = {
            "draws": credit(n, player, draw),
            "cards": credit(n, player, jnp.where(draw, num_drawn, 0)),
            "draws_known": credit(n, player, known),
            "draws_improved": credit(n, player, known & improved),
            "high_card_draws": credit(n, player, high_card_draw),
            "snows": credit(n, player, snow),
            "snow_wins": credit(n, winner, snow_won),
            "postdraw_aggressive": credit(n, player, aggressive),
            "postdraw_bluffs": credit(n, player, bluff),
            "postdraw_bluff_wins": credit(n, winner, bluff_won),
            "postdraw_folds": credit(n, player, fold),
            "postdraw_correct_folds": credit(n, player, fold & loses),
        }
        carry = jax.tree_util.tree_map(
            lambda reset, kept: jnp.where(done, reset, kept),
            self.init_carry(),
            FiveCardDrawCarry(snowed, aggressor, aggressor_bluffed),
        )
        return carry, counts

    def summarize(self, totals: Counts) -> dict[str, FloatArray]:
        t = totals
        return {
            "cards_drawn": rate(t["cards"], t["draws"]),
            "n_draws": t["draws"],
            "draw_improve_rate": rate(t["draws_improved"], t["draws_known"]),
            "n_draws_known": t["draws_known"],
            "snow_rate": rate(t["snows"], t["high_card_draws"]),
            "n_high_card_draws": t["high_card_draws"],
            "snow_success": rate(t["snow_wins"], t["snows"]),
            "n_snows": t["snows"],
            "postdraw_bluff_rate": rate(t["postdraw_bluffs"], t["postdraw_aggressive"]),
            "n_postdraw_aggressive": t["postdraw_aggressive"],
            "postdraw_bluff_success": rate(
                t["postdraw_bluff_wins"], t["postdraw_bluffs"]
            ),
            "n_postdraw_bluffs": t["postdraw_bluffs"],
            "postdraw_correct_fold_rate": rate(
                t["postdraw_correct_folds"], t["postdraw_folds"]
            ),
            "n_postdraw_folds": t["postdraw_folds"],
        }
