"""Goofspiel metrics (bluffjax-metrics.md, Goofspiel).

The rules they rely on (bluffjax/environments/goofspiel/goofspiel.py): every
player holds the cards 1..13 (index c has value c + 1) and the prize deck, a
uniformly random permutation of 1..13, is revealed one card per round: the
prize of a round is shown (in the observation) before the players bid. All
players bid one card from their hand at once and bid cards are used up, so
over the 13 rounds of a game each player bids every card exactly once. The
highest bid wins the prize if no other player bid the same card; if two or
more players tie for the highest bid (whatever the others bid) the prize is
burned: nobody scores it. A game always lasts all 13 rounds (the env has no
cut-off).

A player's bid is the card the env plays: its action if that card is still in
hand, otherwise the lowest card in hand (the env's rule for an illegal bid;
the training scripts only sample legal ones).

Every metric is credited per player (counts of shape (num_agents,)):

bid_prize_spearman: Spearman's rank correlation between a player's bids and
    the prizes over the 13 rounds of a completed game. Both are permutations
    of 1..13, so rho = 1 - 6 * sum_r d_r^2 / (n (n^2 - 1)) with
    d_r = bid_r - prize_r and n = 13; the carry accumulates sum_r d_r^2. Mean
    over (completed game, player), credited to the player at the game's last
    step; n_player_games counts the pairs.
bid_entropy: mean over a player's decisions with at least two legal bids (the
    cards still in hand: every round but the last) of
    H(pi(.|s)) / log(number of legal bids), where pi is the action
    distribution of the policy the player follows at that step (`probs`,
    natural logs, 0 log 0 = 0): 1 for a uniform choice among the cards in
    hand, 0 for a deterministic bid. Credited to the bidder; n_choices
    counts the decisions.
overbid_margin: mean over the prizes won (by a sole highest bid) of the
    winning bid minus the highest bid of the other players, in card values
    (at least 1); credited to the winner; n_won_prizes counts the prizes.
burn_rate: share of rounds whose highest bid is tied (the prize is burned),
    every round credited to every player; n_player_rounds counts the
    (round, player) pairs.
"""

from typing import NamedTuple

import jax.numpy as jnp

from bluffjax.metrics.base import Counts, GameMetrics, rate
from bluffjax.utils.typing import Array, BoolArray, FloatArray, IntArray


class GoofspielCarry(NamedTuple):
    # (num_agents,) sum over the rounds of this game so far of (bid - prize)^2
    sum_sq_diff: FloatArray


def played_bids(env, state, action: IntArray) -> IntArray:
    """(num_agents,) the card each player bids, as Goofspiel.step_env plays
    it: the action if that card is in hand, else the lowest card in hand."""
    in_range = (action >= 0) & (action < env.deck_size)
    in_hand = in_range & state.player_hands[
        jnp.arange(env.num_agents), jnp.clip(action, 0, env.deck_size - 1)
    ]
    return jnp.where(in_hand, action, jnp.argmax(state.player_hands, axis=1))


def normalized_entropy(probs: FloatArray, num_legal: IntArray) -> FloatArray:
    """H(probs) / log(num_legal) over the last axis (0 log 0 = 0); only
    meaningful where num_legal >= 2."""
    positive = probs > 0
    plogp = jnp.where(positive, probs * jnp.log(jnp.where(positive, probs, 1.0)), 0.0)
    return -plogp.sum(-1) / jnp.log(jnp.maximum(num_legal, 2).astype(jnp.float32))


class Metrics(GameMetrics):
    rate_names = ("bid_prize_spearman", "bid_entropy", "overbid_margin", "burn_rate")
    needs_probs = True

    def init_carry(self) -> GoofspielCarry:
        return GoofspielCarry(sum_sq_diff=jnp.zeros(self.num_agents, jnp.float32))

    def step(
        self,
        carry: GoofspielCarry,
        state,
        action: Array,
        next_state,
        reward: FloatArray,
        done: BoolArray,
        info: dict,
        probs: FloatArray | None = None,
    ) -> tuple[GoofspielCarry, Counts]:
        if probs is None:
            raise ValueError(
                "Goofspiel metrics need probs, the action distribution each "
                "player followed (for bid_entropy)"
            )
        n = self.env.deck_size
        ones = jnp.ones(self.num_agents, jnp.float32)
        bid = played_bids(self.env, state, action)
        prize = state.deck[state.current_round]

        # Spearman: both sequences are permutations of the 13 card values.
        sum_sq_diff = carry.sum_sq_diff + jnp.square(bid - prize).astype(jnp.float32)
        completed = done & (state.current_round == n - 1)
        rho = 1.0 - 6.0 * sum_sq_diff / (n * (n * n - 1))

        # The env's rule: the highest bid wins only if nobody else bid it.
        top = bid.max()
        is_top = bid == top
        sole = is_top.sum() == 1
        won = is_top & sole
        highest_other = jnp.where(is_top, -1, bid).max()  # with a sole top
        margin = (top - highest_other).astype(jnp.float32)

        num_legal = self.env.get_avail_actions(state).sum(-1)
        choice = num_legal >= 2

        counts = {
            "spearman_sum": jnp.where(completed, rho, 0.0),
            "player_games": completed * ones,
            "entropy_sum": jnp.where(choice, normalized_entropy(probs, num_legal), 0.0),
            "choices": choice.astype(jnp.float32),
            "margin_sum": jnp.where(won, margin, 0.0),
            "won_prizes": won.astype(jnp.float32),
            "burned": (~sole) * ones,
            "player_rounds": ones,
        }
        carry = GoofspielCarry(sum_sq_diff=jnp.where(done, 0.0, sum_sq_diff))
        return carry, counts

    def summarize(self, totals: Counts) -> dict[str, FloatArray]:
        return {
            "bid_prize_spearman": rate(totals["spearman_sum"], totals["player_games"]),
            "n_player_games": totals["player_games"],
            "bid_entropy": rate(totals["entropy_sum"], totals["choices"]),
            "n_choices": totals["choices"],
            "overbid_margin": rate(totals["margin_sum"], totals["won_prizes"]),
            "n_won_prizes": totals["won_prizes"],
            "burn_rate": rate(totals["burned"], totals["player_rounds"]),
            "n_player_rounds": totals["player_rounds"],
        }
