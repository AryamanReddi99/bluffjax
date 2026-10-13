"""
Behaviour metrics of No-Limit Hold'em (texas_nolimit_holdem), from env steps.

Env actions: 0 check/call, 1 raise half pot, 2 raise pot, 3 all-in, 4 fold
(a "half-pot raise" puts floor(pot / 2) chips in, a "pot raise" the pot,
both counting the chips to call). Every stack starts at init_chips; the env
deals the whole board in advance.

Every count of a hand is credited at the step where the hand ends (see
bluffjax.metrics._holdem), to the player it describes; every rate below is
NaN when its denominator is 0, and summarize() also returns that denominator
as n_<denominator>.

- raise_share_<size> (half_pot, pot, allin): share of the player's
  aggressive actions (actions 1-3, every one of them raises) made with the
  env action of that size (n_aggressive).
- allin_called_rate: share of the player's all-in bets that an opponent
  calls. An all-in bet is an aggressive action after which the bettor has no
  chips behind: the all-in action, or a half-pot or pot raise of the whole
  remaining stack. It is called when an opponent then calls it (action 0
  with chips to call before any other bet or raise); credited to the all-in
  player (n_allin_bets).
- allin_call_win_rate: share of the player's calls of an all-in bet (a call
  of a bet whose bettor is all-in) after which its reward for the hand is
  positive; credited to the caller (n_allin_calls).
- bluff_rate: share of the player's aggressive actions that are bluffs, made
  when its final hand is strictly worse than the final hand of at least one
  opponent still in the hand (not folded; all-in players are in; ties don't
  lose), scored with the env's hand scorer (n_aggressive).
- bluff_success: share of the player's bluffs after which the hand ends
  without showdown with every opponent still in folding to the bluff, i.e.
  every action after it is a fold (heads-up: the opponent folds directly to
  it). Credited to the bluffer (n_bluffs).
- correct_fold_rate: share of the player's folds made with a hand that
  would lose at showdown against the opponents still in at the fold (n_folds).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bluffjax.metrics._holdem import (
    NUM_STREETS,
    ActionInfo,
    BluffCarry,
    bluff_step,
    final_scores,
    init_bluff_carry,
    reset_where,
    would_lose,
)
from bluffjax.metrics.base import Counts, GameMetrics, rate
from bluffjax.utils.typing import Array, BoolArray, FloatArray, IntArray

CHECK_CALL, HALF_POT, POT, ALL_IN, FOLD = 0, 1, 2, 3, 4
RAISE_SIZES = ("half_pot", "pot", "allin")  # actions 1, 2, 3


class NoLimitHoldemCarry(NamedTuple):
    bluff: BluffCarry
    raise_sizes: FloatArray  # (num_agents, 3) aggressive actions by size
    allin_bets: FloatArray  # (num_agents,)
    allin_called: FloatArray  # (num_agents,) all-in bets of the player called
    allin_calls: FloatArray  # (num_agents,) calls of an all-in by the player
    allin_bettor: IntArray  # all-in player whose bet is to be matched, -1 if none
    allin_open: BoolArray  # that bet hasn't been called yet


class Metrics(GameMetrics):
    rate_names = (
        tuple(f"raise_share_{s}" for s in RAISE_SIZES)
        + ("allin_called_rate", "allin_call_win_rate")
        + ("bluff_rate", "bluff_success", "correct_fold_rate")
    )

    def init_carry(self) -> NoLimitHoldemCarry:
        n = self.num_agents
        return NoLimitHoldemCarry(
            bluff=init_bluff_carry(n),
            raise_sizes=jnp.zeros((n, len(RAISE_SIZES)), jnp.float32),
            allin_bets=jnp.zeros((n,), jnp.float32),
            allin_called=jnp.zeros((n,), jnp.float32),
            allin_calls=jnp.zeros((n,), jnp.float32),
            allin_bettor=jnp.int32(-1),
            allin_open=jnp.array(False),
        )

    def step(
        self,
        carry: NoLimitHoldemCarry,
        state,
        action: Array,
        next_state,
        reward: FloatArray,
        done: BoolArray,
        info: dict,
        probs: FloatArray | None = None,
    ) -> tuple[NoLimitHoldemCarry, Counts]:
        n = self.num_agents
        actor = state.current_player_idx
        # chips the action puts in, as in env.step_env
        to_call = jnp.max(state.round_raised) - state.round_raised[actor]
        pot = jnp.sum(state.chips_in)
        remaining = state.remaining_chips[actor]
        bet = jnp.stack([to_call, jnp.floor(pot / 2.0), pot, remaining, 0.0])[action]
        aggressive = (action >= HALF_POT) & (action <= ALL_IN)
        act = ActionInfo(
            actor=actor,
            street=jnp.clip(state.stage, 0, NUM_STREETS - 1),
            aggressive=aggressive,
            call=(action == CHECK_CALL) & (to_call > 0),
            fold=action == FOLD,
        )
        who = jax.nn.one_hot(actor, n, dtype=jnp.float32)
        in_hand = ~state.folded
        in_hand_after = in_hand & ~((who > 0) & act.fold)
        lose = would_lose(final_scores(state), actor, in_hand)
        bluff, counts = bluff_step(carry.bluff, act, lose, in_hand_after, done)

        allin_bet = aggressive & (remaining - bet <= 0)
        calls_allin = act.call & (carry.allin_bettor >= 0)
        newly_called = calls_allin & carry.allin_open
        size = jax.nn.one_hot(action - HALF_POT, len(RAISE_SIZES), dtype=jnp.float32)
        carry = NoLimitHoldemCarry(
            bluff=bluff,
            raise_sizes=carry.raise_sizes + aggressive * who[:, None] * size,
            allin_bets=carry.allin_bets + allin_bet * who,
            allin_called=carry.allin_called
            + newly_called
            * jax.nn.one_hot(carry.allin_bettor, n, dtype=jnp.float32),
            allin_calls=carry.allin_calls + calls_allin * who,
            # a new bet or raise is the one to match from now on
            allin_bettor=jnp.where(
                aggressive, jnp.where(allin_bet, actor, -1), carry.allin_bettor
            ),
            allin_open=jnp.where(
                aggressive, allin_bet, carry.allin_open & ~newly_called
            ),
        )
        counts.update(
            raise_sizes=done * carry.raise_sizes,
            allin_bets=done * carry.allin_bets,
            allin_called=done * carry.allin_called,
            allin_calls=done * carry.allin_calls,
            allin_call_wins=done * carry.allin_calls * (reward > 0),
        )
        return reset_where(done, carry, self.init_carry()), counts

    def summarize(self, totals: Counts) -> dict[str, FloatArray]:
        out = {}
        aggressive = totals["aggressive"].sum()
        for k, size in enumerate(RAISE_SIZES):
            out[f"raise_share_{size}"] = rate(totals["raise_sizes"][k], aggressive)
        out["n_aggressive"] = aggressive
        out["allin_called_rate"] = rate(totals["allin_called"], totals["allin_bets"])
        out["n_allin_bets"] = totals["allin_bets"]
        out["allin_call_win_rate"] = rate(
            totals["allin_call_wins"], totals["allin_calls"]
        )
        out["n_allin_calls"] = totals["allin_calls"]
        bluffs = totals["bluffs"].sum()
        out["bluff_rate"] = rate(bluffs, aggressive)
        out["bluff_success"] = rate(totals["bluff_successes"].sum(), bluffs)
        out["n_bluffs"] = bluffs
        out["correct_fold_rate"] = rate(totals["correct_folds"], totals["folds"])
        out["n_folds"] = totals["folds"]
        return {k: jnp.asarray(v, jnp.float32) for k, v in out.items()}
