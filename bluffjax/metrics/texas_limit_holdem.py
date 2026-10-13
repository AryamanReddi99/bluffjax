"""
Behaviour metrics of Limit Hold'em (texas_limit_holdem), from env steps.

The env deals the whole board in advance and draws the small blind uniformly
at random for every hand (there is no button moving between hands); the big
blind is the next seat. The env's turn order is the standard one: heads-up
the small blind acts first preflop and the big blind first on the flop, turn
and river; with more players the player after the big blind acts first
preflop, and the small blind (or the next player still in after it) on the
later streets. Positions are named by seat relative to the small blind:
"sb", "bb", then "pos2", "pos3", ... with more than two players. Env actions:
0 call, 1 raise (a bet when nothing is to call), 2 fold, 3 check.

Every count of a hand is credited at the step where the hand ends (see
bluffjax.metrics._holdem), to the player it describes; every rate below is
NaN when its denominator is 0, and summarize() also returns that denominator
as n_<denominator>.

- vpip_<pos>: share of the hands dealt to a player in position <pos> in
  which it voluntarily put chips in preflop, i.e. called (action 0 with chips
  to call) or raised preflop at least once. Posting a blind and checking
  don't count. Denominator: hands in that position (n_hands_<pos>).
- pfr_<pos>: share of those hands in which it raised preflop at least once
  (n_hands_<pos>).
- bluff_rate_<street> (preflop, flop, turn, river): share of the player's
  aggressive actions (raises, action 1) on that street that are bluffs:
  made when its final hand is strictly worse than the final hand of at least
  one opponent still in the hand (ties don't lose), scored with the env's
  hand scorer (n_aggressive_<street>).
- bluff_success_<street>: share of the player's bluffs on that street after
  which the hand ends without showdown with every opponent still in folding
  to the bluff, i.e. every action after it is a fold (heads-up: the opponent
  folds directly to it). Credited to the bluffer (n_bluffs_<street>).
- bluff_rate, bluff_success: the same over all streets (n_aggressive,
  n_bluffs).
- correct_fold_rate: share of the player's folds made with a hand that
  would lose at showdown against the opponents still in at the fold. Folding
  when checking is free (legal in the env) is a fold too (n_folds).
- showdown_rate: share of the hands dealt to the player that reach showdown
  (at least two players left at the end), credited to every player dealt in
  (n_hands).
- showdown_win_rate: share of the player's showdowns (hands reaching
  showdown with the player still in) where its reward is positive; a split
  pot is not a win (n_showdowns).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bluffjax.metrics._holdem import (
    NUM_STREETS,
    STREETS,
    ActionInfo,
    BluffCarry,
    bluff_step,
    final_scores,
    init_bluff_carry,
    reset_where,
    showdown_counts,
    would_lose,
)
from bluffjax.metrics.base import Counts, GameMetrics, rate
from bluffjax.utils.typing import Array, BoolArray, FloatArray

CALL, RAISE, FOLD, CHECK = 0, 1, 2, 3


def position_names(num_agents: int) -> tuple[str, ...]:
    """Name of each position, by seat relative to the small blind."""
    return ("sb", "bb") + tuple(f"pos{k}" for k in range(2, num_agents))


class LimitHoldemCarry(NamedTuple):
    bluff: BluffCarry
    vpip: BoolArray  # (num_agents,) called or raised preflop in this hand
    pfr: BoolArray  # (num_agents,) raised preflop in this hand


class Metrics(GameMetrics):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.positions = position_names(self.num_agents)
        self.rate_names = (
            tuple(f"vpip_{p}" for p in self.positions)
            + tuple(f"pfr_{p}" for p in self.positions)
            + ("bluff_rate",)
            + tuple(f"bluff_rate_{s}" for s in STREETS)
            + ("bluff_success",)
            + tuple(f"bluff_success_{s}" for s in STREETS)
            + ("correct_fold_rate", "showdown_rate", "showdown_win_rate")
        )

    def init_carry(self) -> LimitHoldemCarry:
        n = self.num_agents
        return LimitHoldemCarry(
            bluff=init_bluff_carry(n),
            vpip=jnp.zeros((n,), bool),
            pfr=jnp.zeros((n,), bool),
        )

    def step(
        self,
        carry: LimitHoldemCarry,
        state,
        action: Array,
        next_state,
        reward: FloatArray,
        done: BoolArray,
        info: dict,
        probs: FloatArray | None = None,
    ) -> tuple[LimitHoldemCarry, Counts]:
        n = self.num_agents
        actor = state.current_player_idx
        to_call = jnp.max(state.chips_in) - state.chips_in[actor]
        act = ActionInfo(
            actor=actor,
            street=jnp.clip(state.stage, 0, NUM_STREETS - 1),
            aggressive=action == RAISE,
            call=(action == CALL) & (to_call > 0),
            fold=action == FOLD,
        )
        who = jax.nn.one_hot(actor, n, dtype=bool)
        in_hand = ~state.folded
        in_hand_after = in_hand & ~(who & act.fold)
        lose = would_lose(final_scores(state), actor, in_hand)
        bluff, counts = bluff_step(carry.bluff, act, lose, in_hand_after, done)

        preflop = state.stage == 0
        vpip = carry.vpip | (who & preflop & (act.aggressive | act.call))
        pfr = carry.pfr | (who & preflop & act.aggressive)
        # (player, position) table of this hand
        position = (jnp.arange(n) - state.small_blind_idx) % n
        seat = done * jax.nn.one_hot(position, n, dtype=jnp.float32)
        counts.update(
            hands_by_position=seat,
            vpip_by_position=vpip[:, None] * seat,
            pfr_by_position=pfr[:, None] * seat,
        )
        counts.update(showdown_counts(in_hand_after, reward, done))
        carry = LimitHoldemCarry(bluff=bluff, vpip=vpip, pfr=pfr)
        return reset_where(done, carry, self.init_carry()), counts

    def summarize(self, totals: Counts) -> dict[str, FloatArray]:
        out = {}
        for k, pos in enumerate(self.positions):
            hands = totals["hands_by_position"][k]
            out[f"vpip_{pos}"] = rate(totals["vpip_by_position"][k], hands)
            out[f"pfr_{pos}"] = rate(totals["pfr_by_position"][k], hands)
            out[f"n_hands_{pos}"] = hands
        aggressive, bluffs = totals["aggressive"], totals["bluffs"]
        successes = totals["bluff_successes"]
        out["bluff_rate"] = rate(bluffs.sum(), aggressive.sum())
        out["n_aggressive"] = aggressive.sum()
        out["bluff_success"] = rate(successes.sum(), bluffs.sum())
        out["n_bluffs"] = bluffs.sum()
        for k, street in enumerate(STREETS):
            out[f"bluff_rate_{street}"] = rate(bluffs[k], aggressive[k])
            out[f"n_aggressive_{street}"] = aggressive[k]
            out[f"bluff_success_{street}"] = rate(successes[k], bluffs[k])
            out[f"n_bluffs_{street}"] = bluffs[k]
        out["correct_fold_rate"] = rate(totals["correct_folds"], totals["folds"])
        out["n_folds"] = totals["folds"]
        out["showdown_rate"] = rate(totals["showdown_hands"], totals["hands"])
        out["n_hands"] = totals["hands"]
        out["showdown_win_rate"] = rate(totals["showdown_wins"], totals["showdowns"])
        out["n_showdowns"] = totals["showdowns"]
        return {k: jnp.asarray(v, jnp.float32) for k, v in out.items()}
