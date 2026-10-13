"""Checks for the Limit and No-Limit Hold'em game metrics
(bluffjax/metrics/texas_limit_holdem.py, texas_nolimit_holdem.py, _holdem.py):

- scripted hands, heads-up and with three players: every event is detected,
  credited to the right player (and position / street / size), counted once,
  only at the step where the hand ends, and the carry is reset at the end of
  a hand, also across env.step's auto-reset;
- uniform-random play against an independent per-hand reference
  implementation of the definitions (exact counts, 2 and 3 players);
- uniform-random heads-up play against values derived by hand or from the
  betting rules (VPIP / PFR, showdown rate, bluff success) and from the
  probability of a tie (bluff rate, correct folds, showdown and all-in wins);
- play_match's per-side metric counts.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.metrics import make_metrics, masked_totals
from bluffjax.metrics._holdem import STREETS
from bluffjax.utils.game_utils.poker_utils import _score_seven_card_hand

RANKS = "23456789TJQKA"
SUITS = "cdhs"
LHE, NLHE = "texas_limit_holdem", "texas_nolimit_holdem"
# Limit actions
CALL, RAISE, FOLD, CHECK = 0, 1, 2, 3
# No-Limit actions
NL_CALL, HALF, POT, ALLIN, NL_FOLD = 0, 1, 2, 3, 4


def card(name: str) -> int:
    """Env card id of e.g. "As" (rank = id % 13 with the ace at 0, suit = id // 13)."""
    rank = RANKS.index(name[0]) + 2
    return SUITS.index(name[1]) * 13 + (0 if rank == 14 else rank - 1)


def deal(env, sb: int, holes: list[str], board: str):
    """A first state with the small blind at seat sb and the given cards."""
    key = 0
    while True:
        state, _ = env.reset(jax.random.PRNGKey(key))
        if int(state.small_blind_idx) == sb:
            break
        key += 1
    return set_cards(state, holes, board)


def set_cards(state, holes: list[str], board: str):
    dtype = state.agent_cards.dtype
    b = [card(c) for c in board.split()]
    return state.replace(
        agent_cards=jnp.array([[card(c) for c in h.split()] for h in holes], dtype),
        flop_cards=jnp.array(b[:3], state.flop_cards.dtype),
        turn_card=jnp.asarray(b[3], state.turn_card.dtype),
        river_card=jnp.asarray(b[4], state.river_card.dtype),
    )


_JIT = {}


def _jitted(env, gm):
    key = (id(env), id(gm))
    if key not in _JIT:
        _JIT[key] = (jax.jit(env.step), jax.jit(gm.step), env, gm)
    return _JIT[key][:2]


def play(env, gm, state, actions, carry=None):
    """Plays the actions with env.step (auto-reset) and gm.step. Returns the
    per-step counts (numpy), dones, rewards, the carry and the next state."""
    env_step, metric_step = _jitted(env, gm)
    carry = gm.init_carry() if carry is None else carry
    steps, dones, rewards = [], [], []
    for t, a in enumerate(actions):
        assert bool(env.get_avail_actions(state)[a]), f"action {a} at step {t} is illegal"
        nxt, _, reward, _, done, info = env_step(jax.random.PRNGKey(1000 + t), state, jnp.int32(a))
        carry, counts = metric_step(carry, state, jnp.int32(a), nxt, reward, done, info)
        steps.append({k: np.asarray(v) for k, v in counts.items()})
        dones.append(bool(done))
        rewards.append(np.asarray(reward))
        state = nxt
    return steps, dones, rewards, carry, state


def play_hand(env, gm, state, actions, carry=None):
    """A whole hand: done exactly at the last action, no counts before it.
    Returns the hand's counts, its rewards, the carry and the next state."""
    steps, dones, rewards, carry, state = play(env, gm, state, actions, carry)
    assert dones == [False] * (len(actions) - 1) + [True]
    for counts in steps[:-1]:
        for name, value in counts.items():
            assert not value.any(), f"{name} credited before the hand ended"
    return steps[-1], rewards[-1], carry, state


def check_counts(counts, expected):
    """counts: name -> (num_agents, *extra); expected: name -> {index: value};
    every entry not listed must be 0."""
    for name in expected:
        assert name in counts, name
    for name, value in counts.items():
        want = np.zeros_like(value)
        for index, v in expected.get(name, {}).items():
            want[index] = v
        np.testing.assert_array_equal(value, want, err_msg=name)


def assert_initial(gm, carry):
    for got, want in zip(jax.tree.leaves(carry), jax.tree.leaves(gm.init_carry())):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


# Final hands on the board 3s 8d 9c Jh 4s: "Ah Kh" ace high > "7c 2d" jack
# high J9874 > "6c 2h" jack high J9864.
BOARD = "3s 8d 9c Jh 4s"
BEST, MID, WORST = "Ah Kh", "7c 2d", "6c 2h"
PRE, FLOP, TURN, RIVER = range(4)


def test_board_hand_order():
    def score(hole):
        return int(_score_seven_card_hand(jnp.array([card(c) for c in (hole + " " + BOARD).split()])))

    assert score(BEST) > score(MID) > score(WORST)


@pytest.fixture(scope="module")
def lhe():
    env = make(LHE, num_agents=2)
    return env, make_metrics(LHE, env)


@pytest.fixture(scope="module")
def nlhe():
    env = make(NLHE, num_agents=2)
    return env, make_metrics(NLHE, env)


SB_HAND = {"hands_by_position": {(0, 0): 1, (1, 1): 1}, "hands": {0: 1, 1: 1}}


# =============================================================================
# Limit Hold'em, heads-up
# =============================================================================


def test_lhe_river_bluff_succeeds(lhe):
    """SB limps, everything is checked to the river, the SB bets the worse
    hand and the BB folds: one river bluff, successful, by the SB; the BB's
    fold is incorrect."""
    env, gm = lhe
    state = deal(env, 0, [MID, BEST], BOARD)
    actions = [CALL, CHECK, CHECK, CHECK, CHECK, CHECK, RAISE, FOLD]
    counts, reward, carry, _ = play_hand(env, gm, state, actions)
    check_counts(counts, {
        **SB_HAND,
        "aggressive": {(0, RIVER): 1},
        "bluffs": {(0, RIVER): 1},
        "bluff_successes": {(0, RIVER): 1},
        "folds": {1: 1},
        "vpip_by_position": {(0, 0): 1},
    })
    np.testing.assert_array_equal(reward, [2.0, -2.0])  # the BB's blind, in chips
    assert_initial(gm, carry)
    rates = gm.summarize(counts_total([counts]))
    assert rates["bluff_rate"] == 1 and rates["bluff_rate_river"] == 1
    assert rates["bluff_success"] == 1 and rates["bluff_success_river"] == 1
    assert np.isnan(rates["bluff_rate_flop"]) and rates["n_aggressive_flop"] == 0
    assert rates["correct_fold_rate"] == 0 and rates["n_folds"] == 1
    assert rates["vpip_sb"] == 1 and rates["vpip_bb"] == 0 and rates["pfr_sb"] == 0
    assert rates["n_hands_sb"] == 1 and rates["n_hands_bb"] == 1
    assert rates["showdown_rate"] == 0 and np.isnan(rates["showdown_win_rate"])
    assert set(gm.rate_names) <= set(rates)


def counts_total(hands):
    """Sum of per-hand counts over hands and players (all players kept)."""
    return {
        k: sum(h[k] for h in hands).sum(axis=0) for k in hands[0]
    }


def test_lhe_value_bet_and_correct_fold(lhe):
    env, gm = lhe
    state = deal(env, 0, [BEST, MID], BOARD)
    actions = [CALL, CHECK, CHECK, CHECK, CHECK, CHECK, RAISE, FOLD]
    counts, _, _, _ = play_hand(env, gm, state, actions)
    check_counts(counts, {
        **SB_HAND,
        "aggressive": {(0, RIVER): 1},
        "folds": {1: 1},
        "correct_folds": {1: 1},
        "vpip_by_position": {(0, 0): 1},
    })


def test_lhe_bluff_reraised_and_folded(lhe):
    """SB raises the worse hand, BB re-raises, SB folds: the SB's bluff fails
    and its fold is correct; the BB's raise is no bluff."""
    env, gm = lhe
    state = deal(env, 0, [MID, BEST], BOARD)
    counts, _, _, _ = play_hand(env, gm, state, [RAISE, RAISE, FOLD])
    check_counts(counts, {
        **SB_HAND,
        "aggressive": {(0, PRE): 1, (1, PRE): 1},
        "bluffs": {(0, PRE): 1},
        "folds": {0: 1},
        "correct_folds": {0: 1},
        "vpip_by_position": {(0, 0): 1, (1, 1): 1},
        "pfr_by_position": {(0, 0): 1, (1, 1): 1},
    })


def test_lhe_reraise_bluff_succeeds(lhe):
    """The BB re-raises the worse hand and the SB folds the better one."""
    env, gm = lhe
    state = deal(env, 0, [BEST, MID], BOARD)
    counts, _, _, _ = play_hand(env, gm, state, [RAISE, RAISE, FOLD])
    check_counts(counts, {
        **SB_HAND,
        "aggressive": {(0, PRE): 1, (1, PRE): 1},
        "bluffs": {(1, PRE): 1},
        "bluff_successes": {(1, PRE): 1},
        "folds": {0: 1},
        "vpip_by_position": {(0, 0): 1, (1, 1): 1},
        "pfr_by_position": {(0, 0): 1, (1, 1): 1},
    })


def test_lhe_called_bluff_is_no_success(lhe):
    """A preflop bluff is called; the caller folds on the flop without facing
    a bet. The bluffer wins uncontested, but not by the bluff."""
    env, gm = lhe
    state = deal(env, 0, [MID, BEST], BOARD)
    counts, reward, _, _ = play_hand(env, gm, state, [RAISE, CALL, CHECK, FOLD])
    check_counts(counts, {
        **SB_HAND,
        "aggressive": {(0, PRE): 1},
        "bluffs": {(0, PRE): 1},
        "folds": {1: 1},
        "vpip_by_position": {(0, 0): 1, (1, 1): 1},  # the BB called a raise
        "pfr_by_position": {(0, 0): 1},
    })
    assert reward[0] > 0


def test_lhe_showdown_win_and_tie(lhe):
    env, gm = lhe
    passive = [CALL, CHECK] + [CHECK] * 6
    counts, reward, _, _ = play_hand(env, gm, deal(env, 0, [MID, BEST], BOARD), passive)
    check_counts(counts, {
        **SB_HAND,
        "vpip_by_position": {(0, 0): 1},
        "showdown_hands": {0: 1, 1: 1},
        "showdowns": {0: 1, 1: 1},
        "showdown_wins": {1: 1},
    })
    np.testing.assert_array_equal(reward, [-2.0, 2.0])  # one big blind each, in chips
    # both play the royal flush on the board: a split pot is no win
    counts, reward, _, _ = play_hand(
        env, gm, deal(env, 0, ["2c 3d", "2h 3h"], "Ts Js Qs Ks As"), passive
    )
    np.testing.assert_array_equal(reward, [0.0, 0.0])
    check_counts(counts, {
        **SB_HAND,
        "vpip_by_position": {(0, 0): 1},
        "showdown_hands": {0: 1, 1: 1},
        "showdowns": {0: 1, 1: 1},
    })


def test_lhe_positions_follow_the_small_blind(lhe):
    """Seat 1 posts the small blind and limps, seat 0 (BB) raises, seat 1
    calls and folds the flop."""
    env, gm = lhe
    state = deal(env, 1, [BEST, MID], BOARD)
    assert int(state.current_player_idx) == 1
    counts, _, _, _ = play_hand(env, gm, state, [CALL, RAISE, CALL, FOLD])
    check_counts(counts, {
        "hands_by_position": {(1, 0): 1, (0, 1): 1},
        "hands": {0: 1, 1: 1},
        "vpip_by_position": {(1, 0): 1, (0, 1): 1},
        "pfr_by_position": {(0, 1): 1},
        "aggressive": {(0, PRE): 1},
        "folds": {1: 1},
        "correct_folds": {1: 1},
    })
    rates = gm.summarize(counts_total([counts]))
    assert rates["vpip_sb"] == 1 and rates["vpip_bb"] == 1
    assert rates["pfr_sb"] == 0 and rates["pfr_bb"] == 1


def test_lhe_carry_resets_across_auto_reset(lhe):
    """Two hands in a row through env.step's auto-reset: the second hand's
    counts don't include the first's."""
    env, gm = lhe
    state = deal(env, 0, [MID, BEST], BOARD)
    _, _, carry, state = play_hand(env, gm, state, [RAISE, RAISE, CALL] + [CHECK] * 5 + [RAISE, FOLD])
    assert_initial(gm, carry)
    assert int(state.stage) == 0 and not bool(state.folded.any())  # the next hand
    sb = int(state.small_blind_idx)
    holes = [WORST, BEST] if sb == 0 else [BEST, WORST]
    state = set_cards(state, holes, BOARD)
    counts, _, carry, _ = play_hand(env, gm, state, [FOLD], carry)
    check_counts(counts, {
        "hands_by_position": {(sb, 0): 1, (1 - sb, 1): 1},
        "hands": {0: 1, 1: 1},
        "folds": {sb: 1},
        "correct_folds": {sb: 1},
    })
    assert_initial(gm, carry)


# =============================================================================
# Limit Hold'em, three players
# =============================================================================


@pytest.fixture(scope="module")
def lhe3():
    env = make(LHE, num_agents=3)
    return env, make_metrics(LHE, env)


THREE = {
    "hands_by_position": {(0, 0): 1, (1, 1): 1, (2, 2): 1},
    "hands": {0: 1, 1: 1, 2: 1},
}


def test_lhe3_position_names(lhe3):
    _, gm = lhe3
    assert gm.positions == ("sb", "bb", "pos2")
    assert "vpip_pos2" in gm.rate_names


def test_lhe3_bluff_two_opponents_fold(lhe3):
    """Seat 2 (first preflop) and the SB limp, the BB checks; on the flop the
    BB folds; on the turn the SB bets a hand worse than seat 2's and seat 2
    folds: a successful bluff."""
    env, gm = lhe3
    state = deal(env, 0, [MID, WORST, BEST], BOARD)
    assert int(state.current_player_idx) == 2
    actions = [CALL, CALL, CHECK, CHECK, FOLD, CHECK, RAISE, FOLD]
    counts, _, _, _ = play_hand(env, gm, state, actions)
    check_counts(counts, {
        **THREE,
        "vpip_by_position": {(0, 0): 1, (2, 2): 1},
        "aggressive": {(0, TURN): 1},
        "bluffs": {(0, TURN): 1},
        "bluff_successes": {(0, TURN): 1},
        "folds": {1: 1, 2: 1},
        "correct_folds": {1: 1},  # seat 1 is worst; seat 2 had the best hand
    })


def test_lhe3_would_lose_only_against_players_in(lhe3):
    """The SB's turn bet beats the only opponent left (seat 2) but not seat 1,
    who folded: no bluff, and seat 2's fold is correct."""
    env, gm = lhe3
    state = deal(env, 0, [MID, BEST, WORST], BOARD)
    actions = [CALL, CALL, CHECK, CHECK, FOLD, CHECK, RAISE, FOLD]
    counts, _, _, _ = play_hand(env, gm, state, actions)
    check_counts(counts, {
        **THREE,
        "vpip_by_position": {(0, 0): 1, (2, 2): 1},
        "aggressive": {(0, TURN): 1},
        "folds": {1: 1, 2: 1},
        "correct_folds": {2: 1},  # seat 1 folded the best hand
    })


def test_lhe3_bluff_called_by_one_to_showdown(lhe3):
    """The SB bluffs the flop, seat 1 folds, seat 2 calls; checked down to a
    showdown that seat 2 wins. Every seat dealt in is credited the showdown
    hand, only the two left the showdown."""
    env, gm = lhe3
    state = deal(env, 0, [WORST, MID, BEST], BOARD)
    actions = [CALL, CALL, CHECK, RAISE, FOLD, CALL, CHECK, CHECK, CHECK, CHECK]
    counts, reward, _, _ = play_hand(env, gm, state, actions)
    check_counts(counts, {
        **THREE,
        "vpip_by_position": {(0, 0): 1, (2, 2): 1},
        "aggressive": {(0, FLOP): 1},
        "bluffs": {(0, FLOP): 1},
        "folds": {1: 1},
        "correct_folds": {1: 1},  # worse than seat 2, still in
        "showdown_hands": {0: 1, 1: 1, 2: 1},
        "showdowns": {0: 1, 2: 1},
        "showdown_wins": {2: 1},
    })
    assert reward[2] > 0 > reward[0]


# =============================================================================
# No-Limit Hold'em
# =============================================================================


def test_nlhe_allin_called(nlhe):
    """SB shoves the worse hand, BB calls and wins; then the other way round."""
    env, gm = nlhe
    counts, reward, carry, _ = play_hand(env, gm, deal(env, 0, [MID, BEST], BOARD), [ALLIN, NL_CALL])
    np.testing.assert_array_equal(reward, [-100.0, 100.0])
    check_counts(counts, {
        "aggressive": {(0, PRE): 1},
        "bluffs": {(0, PRE): 1},
        "raise_sizes": {(0, 2): 1},
        "allin_bets": {0: 1},
        "allin_called": {0: 1},
        "allin_calls": {1: 1},
        "allin_call_wins": {1: 1},
    })
    assert_initial(gm, carry)
    rates = gm.summarize(counts_total([counts]))
    assert rates["raise_share_allin"] == 1 and rates["raise_share_pot"] == 0
    assert rates["allin_called_rate"] == 1 and rates["allin_call_win_rate"] == 1
    assert rates["bluff_rate"] == 1 and rates["bluff_success"] == 0
    assert np.isnan(rates["correct_fold_rate"]) and rates["n_folds"] == 0
    assert set(gm.rate_names) <= set(rates)
    counts, reward, _, _ = play_hand(env, gm, deal(env, 0, [BEST, MID], BOARD), [ALLIN, NL_CALL])
    np.testing.assert_array_equal(reward, [100.0, -100.0])
    check_counts(counts, {
        "aggressive": {(0, PRE): 1},
        "raise_sizes": {(0, 2): 1},
        "allin_bets": {0: 1},
        "allin_called": {0: 1},
        "allin_calls": {1: 1},
    })


def test_nlhe_allin_bluff_folded(nlhe):
    env, gm = nlhe
    counts, _, _, _ = play_hand(env, gm, deal(env, 0, [MID, BEST], BOARD), [ALLIN, NL_FOLD])
    check_counts(counts, {
        "aggressive": {(0, PRE): 1},
        "bluffs": {(0, PRE): 1},
        "bluff_successes": {(0, PRE): 1},
        "raise_sizes": {(0, 2): 1},
        "allin_bets": {0: 1},
        "folds": {1: 1},
    })


def test_nlhe_raise_sizes(nlhe):
    """Preflop: SB pot raise, BB half-pot raise (a bluff), SB calls. Flop: SB
    checks, BB shoves (a bluff), SB folds the better hand."""
    env, gm = nlhe
    state = deal(env, 0, [BEST, MID], BOARD)
    actions = [POT, HALF, NL_CALL, NL_CALL, ALLIN, NL_FOLD]
    counts, _, _, _ = play_hand(env, gm, state, actions)
    check_counts(counts, {
        "aggressive": {(0, PRE): 1, (1, PRE): 1, (1, FLOP): 1},
        "bluffs": {(1, PRE): 1, (1, FLOP): 1},
        "bluff_successes": {(1, FLOP): 1},
        "raise_sizes": {(0, 1): 1, (1, 0): 1, (1, 2): 1},
        "allin_bets": {1: 1},
        "folds": {0: 1},
    })
    rates = gm.summarize(counts_total([counts]))
    np.testing.assert_allclose(
        [rates["raise_share_half_pot"], rates["raise_share_pot"], rates["raise_share_allin"]],
        [1 / 3, 1 / 3, 1 / 3], rtol=1e-6,
    )
    assert rates["n_aggressive"] == 3 and rates["allin_called_rate"] == 0
    assert np.isnan(rates["allin_call_win_rate"])


def test_nlhe_pot_raise_of_the_whole_stack_is_an_allin_bet():
    """With 4-chip stacks the SB's pot raise (3 chips) is its whole stack."""
    env = make(NLHE, num_agents=2, init_chips=4)
    gm = make_metrics(NLHE, env)
    state = deal(env, 0, [MID, BEST], BOARD)
    counts, _, _, _ = play_hand(env, gm, state, [POT, NL_CALL])
    check_counts(counts, {
        "aggressive": {(0, PRE): 1},
        "bluffs": {(0, PRE): 1},
        "raise_sizes": {(0, 1): 1},
        "allin_bets": {0: 1},
        "allin_called": {0: 1},
        "allin_calls": {1: 1},
        "allin_call_wins": {1: 1},
    })


@pytest.fixture(scope="module")
def nlhe3():
    env = make(NLHE, num_agents=3)
    return env, make_metrics(NLHE, env)


def test_nlhe3_allin_called_once_by_two(nlhe3):
    """Seat 2 shoves; the SB folds, the BB calls; then the same with both
    calling: the all-in is called once, each call is credited to its caller."""
    env, gm = nlhe3
    state = deal(env, 0, [WORST, BEST, MID], BOARD)
    counts, _, _, _ = play_hand(env, gm, state, [ALLIN, NL_FOLD, NL_CALL])
    check_counts(counts, {
        "aggressive": {(2, PRE): 1},
        "bluffs": {(2, PRE): 1},
        "raise_sizes": {(2, 2): 1},
        "allin_bets": {2: 1},
        "allin_called": {2: 1},
        "allin_calls": {1: 1},
        "allin_call_wins": {1: 1},
        "folds": {0: 1},
        "correct_folds": {0: 1},
    })
    counts, reward, _, _ = play_hand(env, gm, state, [ALLIN, NL_CALL, NL_CALL])
    np.testing.assert_array_equal(reward, [-100.0, 200.0, -100.0])
    check_counts(counts, {
        "aggressive": {(2, PRE): 1},
        "bluffs": {(2, PRE): 1},
        "raise_sizes": {(2, 2): 1},
        "allin_bets": {2: 1},
        "allin_called": {2: 1},
        "allin_calls": {0: 1, 1: 1},
        "allin_call_wins": {1: 1},
    })


# =============================================================================
# Uniform-random play against a reference implementation
# =============================================================================


def random_rollout(env, gm, num_envs, num_steps, seed):
    """Uniform-random legal play with auto-reset, as in the training scripts
    (env.step_env, then the next hand's first state when done). Returns the
    metric counts (T, N, P, ...) and the steps' records (numpy)."""

    @jax.jit
    def run(key):
        k_reset, k_play = jax.random.split(key)
        state, _ = jax.vmap(env.reset)(jax.random.split(k_reset, num_envs))
        carry = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_envs,) + x.shape), gm.init_carry()
        )

        def one(c, k):
            state, carry = c
            k_a, k_s, k_r = jax.random.split(k, 3)
            mask = jax.vmap(env.get_avail_actions)(state)
            action = jax.random.categorical(k_a, jnp.where(mask, 0.0, -jnp.inf))
            raw, _, reward, _, done, info = jax.vmap(env.step_env)(
                jax.random.split(k_s, num_envs), state, action
            )
            reset, _ = jax.vmap(env.reset)(jax.random.split(k_r, num_envs))
            nxt = jax.tree.map(
                lambda r, x: jnp.where(done.reshape((-1,) + (1,) * (x.ndim - 1)), r, x),
                reset, raw,
            )
            carry, counts = jax.vmap(gm.step)(carry, state, action, nxt, reward, done, info)
            record = dict(state=state, action=action, reward=reward, done=done, after=raw)
            return (nxt, carry), (counts, record)

        _, out = jax.lax.scan(one, (state, carry), jax.random.split(k_play, num_steps))
        return out

    counts, record = run(jax.random.PRNGKey(seed))
    return jax.tree.map(np.asarray, counts), jax.tree.map(np.asarray, record)


def final_scores_np(state):
    """(T, N, P) env scores of the players' final hands."""
    board = np.concatenate(
        [state.flop_cards, state.turn_card[..., None], state.river_card[..., None]], -1
    )
    P = state.agent_cards.shape[-2]
    hands = np.concatenate(
        [state.agent_cards, np.broadcast_to(board[..., None, :], board.shape[:-1] + (P, 5))], -1
    )
    flat = hands.reshape(-1, 7).astype(np.int32)
    return np.asarray(jax.jit(jax.vmap(_score_seven_card_hand))(flat)).reshape(hands.shape[:-1])


def reference_counts(record, limit: bool):
    """The metric counts from the definitions, hand by hand (numpy loops).
    Returns name -> (P, *extra), summed over all hands that ended."""
    st, after = record["state"], record["after"]
    action, reward, done = record["action"], record["reward"], record["done"]
    T, N = done.shape
    P = st.agent_cards.shape[-2]
    scores = final_scores_np(st)
    out = {
        "aggressive": np.zeros((P, 4)), "bluffs": np.zeros((P, 4)),
        "bluff_successes": np.zeros((P, 4)), "folds": np.zeros(P), "correct_folds": np.zeros(P),
    }
    if limit:
        out.update({
            "hands_by_position": np.zeros((P, P)), "vpip_by_position": np.zeros((P, P)),
            "pfr_by_position": np.zeros((P, P)), "hands": np.zeros(P), "showdown_hands": np.zeros(P),
            "showdowns": np.zeros(P), "showdown_wins": np.zeros(P),
        })
    else:
        out.update({
            "raise_sizes": np.zeros((P, 3)), "allin_bets": np.zeros(P), "allin_called": np.zeros(P),
            "allin_calls": np.zeros(P), "allin_call_wins": np.zeros(P),
        })
    for n in range(N):
        hand = []
        for t in range(T):
            a, p = int(action[t, n]), int(st.current_player_idx[t, n])
            folded = st.folded[t, n]
            others = [q for q in range(P) if q != p and not folded[q]]
            lose = any(scores[t, n, p] < scores[t, n, q] for q in others)
            if limit:
                to_call = st.chips_in[t, n].max() - st.chips_in[t, n, p]
                kind = {CALL: "call", RAISE: "aggr", FOLD: "fold", CHECK: "check"}[a]
                allin = False
            else:
                to_call = st.round_raised[t, n].max() - st.round_raised[t, n, p]
                kind = {NL_CALL: "call", HALF: "aggr", POT: "aggr", ALLIN: "aggr", NL_FOLD: "fold"}[a]
                allin = kind == "aggr" and after.remaining_chips[t, n, p] <= 0
            if kind == "call" and to_call == 0:
                kind = "check"
            hand.append(dict(p=p, a=a, kind=kind, street=int(st.stage[t, n]), lose=lose, allin=allin,
                             sb=int(st.small_blind_idx[t, n])))
            if not done[t, n]:
                continue
            in_hand = ~after.folded[t, n]
            r = reward[t, n]
            for i, h in enumerate(hand):
                p, s = h["p"], h["street"]
                if h["kind"] == "aggr":
                    out["aggressive"][p, s] += 1
                    if h["lose"]:
                        out["bluffs"][p, s] += 1
                        rest = hand[i + 1:]
                        if all(x["kind"] == "fold" for x in rest) and in_hand.sum() == 1 and in_hand[p]:
                            out["bluff_successes"][p, s] += 1
                if h["kind"] == "fold":
                    out["folds"][p] += 1
                    out["correct_folds"][p] += h["lose"]
            if limit:
                sb = hand[0]["sb"]
                showdown = in_hand.sum() >= 2
                for p in range(P):
                    pos = (p - sb) % P
                    pre = [h for h in hand if h["p"] == p and h["street"] == 0]
                    out["hands_by_position"][p, pos] += 1
                    out["vpip_by_position"][p, pos] += any(h["kind"] in ("call", "aggr") for h in pre)
                    out["pfr_by_position"][p, pos] += any(h["kind"] == "aggr" for h in pre)
                    out["hands"][p] += 1
                    out["showdown_hands"][p] += showdown
                    out["showdowns"][p] += showdown and in_hand[p]
                    out["showdown_wins"][p] += showdown and in_hand[p] and r[p] > 0
            else:
                for i, h in enumerate(hand):
                    p = h["p"]
                    if h["kind"] == "aggr":
                        out["raise_sizes"][p, h["a"] - 1] += 1
                        if h["allin"]:
                            out["allin_bets"][p] += 1
                            # called: an opponent calls before any other bet or raise
                            for x in hand[i + 1:]:
                                if x["kind"] == "aggr":
                                    break
                                if x["kind"] == "call":
                                    out["allin_called"][p] += 1
                                    break
                    if h["kind"] == "call":
                        last_aggr = [x for x in hand[:i] if x["kind"] == "aggr"]
                        if last_aggr and last_aggr[-1]["allin"]:
                            out["allin_calls"][p] += 1
                            out["allin_call_wins"][p] += r[p] > 0
            hand = []
    return out


@pytest.mark.parametrize(
    "env_id,num_agents,num_envs,num_steps",
    [(LHE, 2, 128, 128), (LHE, 3, 128, 128), (NLHE, 2, 128, 128), (NLHE, 3, 128, 128)],
)
def test_random_play_matches_reference(env_id, num_agents, num_envs, num_steps):
    env = make(env_id, num_agents=num_agents)
    gm = make_metrics(env_id, env)
    counts, record = random_rollout(env, gm, num_envs, num_steps, seed=num_agents)
    got = {k: v.sum(axis=(0, 1)) for k, v in counts.items()}
    want = reference_counts(record, limit=env_id == LHE)
    assert set(got) == set(want)
    for name in want:
        np.testing.assert_array_equal(got[name], want[name], err_msg=name)
    # the cases the reference has to agree on actually occur
    assert record["done"].sum() > 2000
    for name in ("bluffs", "bluff_successes", "correct_folds"):
        per_player = want[name].reshape(num_agents, -1).sum(-1)
        assert (per_player > 20).all(), (name, per_player)
    if env_id == LHE:
        assert (want["showdown_wins"] > 20).all() and (want["bluffs"] > 0).all()
        assert (want["vpip_by_position"] > 20).all() and (want["pfr_by_position"] > 20).all()
    else:
        assert (want["allin_called"] > 20).all() and (want["allin_call_wins"] > 10).all()
    # counts only at hand ends
    for name, value in counts.items():
        assert not value[~record["done"]].any(), name


# =============================================================================
# Uniform-random heads-up play against derived values
# =============================================================================


@lru_cache(None)
def _lhe_random_tree(street, raises, nr, facing):
    """Heads-up Limit Hold'em under uniform-random legal play, from a decision
    (env rules: call or check, raise below 4 raises a round, fold always;
    a round closes when both players have called or checked since the last
    raise). Returns the expected (aggressive actions, aggressive actions the
    opponent folds to, showdowns) until the end of the hand."""
    legal = ["call" if facing else "check", "fold"] + (["raise"] if raises < 4 else [])
    total = np.zeros(3)
    for a in legal:
        if a == "fold":
            continue
        if a == "raise":
            responses = 3 if raises + 1 < 4 else 2
            nxt = _lhe_random_tree(street, raises + 1, 1, True) + [1.0, 1.0 / responses, 0.0]
        elif nr + 1 >= 2:  # round closed
            nxt = np.array([0.0, 0.0, 1.0]) if street == 3 else _lhe_random_tree(street + 1, 0, 0, False)
        else:
            nxt = _lhe_random_tree(street, raises, nr + 1, False)
        total = total + nxt / len(legal)
    return total


@pytest.fixture(scope="module")
def p_lose():
    """P(a player's final hand is strictly worse than the other's) =
    (1 - P(tie)) / 2 for a random heads-up deal, by Monte Carlo, and its SE."""
    keys = jax.random.split(jax.random.PRNGKey(123), 400_000)
    deals = jax.vmap(lambda k: jax.random.permutation(k, 52)[:9])(keys)
    board = deals[:, 4:]
    score = jax.jit(jax.vmap(_score_seven_card_hand))
    s0 = score(jnp.concatenate([deals[:, :2], board], 1))
    s1 = score(jnp.concatenate([deals[:, 2:4], board], 1))
    lose = np.asarray(s0 < s1, np.float64)
    return lose.mean(), lose.std() / np.sqrt(len(lose))


def _ratio_check(num, den, value, value_se=0.0, z=4.5):
    """sum(num) / sum(den) over hands (per-hand arrays) is within z standard
    errors (delta method, hands independent) of value."""
    r = num.sum() / den.sum()
    se = np.sqrt(np.var(num - r * den) / len(num)) / den.mean()
    assert abs(r - value) < z * np.sqrt(se**2 + value_se**2), (r, value, se)
    return r


@pytest.fixture(scope="module")
def random_hands():
    """Per-hand counts (summed over players) of 2-player random play."""
    out = {}
    for env_id in (LHE, NLHE):
        env = make(env_id, num_agents=2)
        gm = make_metrics(env_id, env)
        counts, record = random_rollout(env, gm, 1024, 128, seed=7)
        ended = record["done"]
        out[env_id] = {k: v[ended] for k, v in counts.items()}  # (hands, P, ...)
    return out


def test_lhe_random_play_values(random_hands, p_lose):
    h = random_hands[LHE]
    hands = h["hands_by_position"]  # (hands, P, positions)
    assert len(hands) > 30_000
    by_pos = lambda name, pos: h[name][:, :, pos].sum(1)  # noqa: E731
    one = hands[:, :, 0].sum(1)
    # SB's first action: call, raise or fold; the BB acts after a limp
    # (check / raise / fold) or a raise (call / raise / fold).
    _ratio_check(by_pos("vpip_by_position", 0), one, 2 / 3)
    _ratio_check(by_pos("vpip_by_position", 1), one, 1 / 9 + 2 / 9)
    # the SB raises first (1/3) or re-raises a limp-raise (1/27); the BB
    # raises after a limp (1/9) or re-raises (1/9)
    _ratio_check(by_pos("pfr_by_position", 0), one, 1 / 3 + 1 / 27)
    _ratio_check(by_pos("pfr_by_position", 1), one, 2 / 9)
    aggressive, folded_to, showdown = _lhe_random_tree(0, 0, 0, True)
    _ratio_check(h["showdown_hands"].sum(1), h["hands"].sum(1), showdown)
    # cards don't change random play: bluffs succeed like any bet, and a
    # bet, a fold or a showdown is made with a losing hand w.p. P(lose)
    bluffs = h["bluffs"].sum((1, 2))
    _ratio_check(h["aggressive"].sum((1, 2)), np.ones(len(one)), aggressive)
    _ratio_check(h["bluff_successes"].sum((1, 2)), bluffs, folded_to / aggressive)
    lose, lose_se = p_lose
    _ratio_check(bluffs, h["aggressive"].sum((1, 2)), lose, lose_se)
    _ratio_check(h["correct_folds"].sum(1), h["folds"].sum(1), lose, lose_se)
    _ratio_check(h["showdown_wins"].sum(1), h["showdowns"].sum(1), lose, lose_se)
    for street in range(4):
        _ratio_check(h["bluffs"][:, :, street].sum(1), h["aggressive"][:, :, street].sum(1), lose, lose_se)


def test_nlhe_random_play_values(random_hands, p_lose):
    h = random_hands[NLHE]
    lose, lose_se = p_lose
    aggressive = h["aggressive"].sum((1, 2))
    _ratio_check(h["bluffs"].sum((1, 2)), aggressive, lose, lose_se)
    _ratio_check(h["correct_folds"].sum(1), h["folds"].sum(1), lose, lose_se)
    _ratio_check(h["allin_call_wins"].sum(1), h["allin_calls"].sum(1), lose, lose_se)
    # heads-up the opponent of an all-in can only call or fold (equal stacks)
    _ratio_check(h["allin_called"].sum(1), h["allin_bets"].sum(1), 0.5)
    assert (h["allin_called"].sum(1) == h["allin_calls"].sum(1)).all()
    # the SB opens preflop with call, pot, all-in or fold (half pot is no raise)
    sizes = h["raise_sizes"].sum((0, 1))
    assert sizes.sum() == aggressive.sum() and (sizes > 1000).all()


# =============================================================================
# play_match counts each side
# =============================================================================


def test_play_match_counts_by_side():
    from bluffjax.examples.holdem_rebel.agent import heuristic_players, play_match
    from bluffjax.examples.holdem_rebel.game import make_game

    game = make_game("hul")
    env = make(LHE, num_agents=2)
    gm = make_metrics(LHE, env)
    players = heuristic_players(game)
    n = 64
    res = jax.jit(
        lambda k: play_match(env, game, players["always_raise"], None, players["always_call"],
                             None, n, k, metrics=gm)
    )(jax.random.PRNGKey(0))
    a, b = jax.tree.map(np.asarray, res.counts_a), jax.tree.map(np.asarray, res.counts_b)
    assert int(res.unfinished) == 0
    # every deal is played from both seats: n hands in each position per side
    np.testing.assert_array_equal(a["hands_by_position"], [n, n])
    np.testing.assert_array_equal(b["hands_by_position"], [n, n])
    # the raiser raises preflop in every hand, the caller calls it
    np.testing.assert_array_equal(a["pfr_by_position"], [n, n])
    np.testing.assert_array_equal(b["pfr_by_position"], [0, 0])
    np.testing.assert_array_equal(b["vpip_by_position"], [n, n])
    assert a["aggressive"].sum() >= 2 * n and b["aggressive"].sum() == 0
    assert a["folds"] == b["folds"] == 0
    assert a["showdown_hands"] == b["showdown_hands"] == 2 * n
    # per hand one of the two wins unless it is a split pot
    assert a["showdown_wins"] + b["showdown_wins"] <= 2 * n
    rewards = np.asarray(res.rewards)
    assert a["showdown_wins"] == (rewards > 0).sum()
    without = jax.jit(
        lambda k: play_match(env, game, players["random"], None, players["random"], None, 4, k)
    )(jax.random.PRNGKey(1))
    assert without.counts_a is None and without.counts_b is None


def test_street_names():
    assert STREETS == ("preflop", "flop", "turn", "river")


def test_masked_totals_by_policy(lhe):
    """masked_totals keeps the counts of the players selected per hand."""
    env, gm = lhe
    state = deal(env, 0, [MID, BEST], BOARD)
    counts, _, _, _ = play_hand(env, gm, state, [RAISE, RAISE, FOLD])
    batch = {k: v[None] for k, v in counts.items()}  # one step
    only_bb = masked_totals(batch, np.array([[False, True]]))
    assert float(only_bb["aggressive"].sum()) == 1 and float(only_bb["bluffs"].sum()) == 0
    assert float(only_bb["folds"]) == 0
