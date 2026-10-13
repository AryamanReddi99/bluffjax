"""Checks for the Limit and No-Limit Hold'em environments (texas_limit_holdem,
texas_nolimit_holdem): the standard turn order and whole hands played under it.

Turn order: heads-up the small blind is the button and acts first preflop, the
big blind acts first on the flop, turn and river. With 3 or more players the
button sits just before the small blind; preflop the player after the big
blind acts first, on later streets the first player after the button still in
the hand (the small blind, or the next player still in after it).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make

LHE, NLHE = "texas_limit_holdem", "texas_nolimit_holdem"
# Limit actions
CALL, RAISE, FOLD, CHECK = 0, 1, 2, 3
# No-Limit actions
NL_CALL, HALF, POT, ALLIN, NL_FOLD = 0, 1, 2, 3, 4

RANKS = "23456789TJQKA"
SUITS = "cdhs"
# On this board "Ah Kh" (ace high) beats "7c 2d" (jack high).
BOARD = "3s 8d 9c Jh 4s"
BEST, WORSE = "Ah Kh", "7c 2d"


def card(name: str) -> int:
    """Env card id of e.g. "As" (rank = id % 13 with the ace at 0, suit = id // 13)."""
    rank = RANKS.index(name[0]) + 2
    return SUITS.index(name[1]) * 13 + (0 if rank == 14 else rank - 1)


def reset_with_sb(env, sb: int):
    """A first state with the small blind at seat sb."""
    key = 0
    while True:
        state, _ = env.reset(jax.random.PRNGKey(key))
        if int(state.small_blind_idx) == sb:
            return state
        key += 1


def set_cards(state, holes: list[str], board: str):
    b = [card(c) for c in board.split()]
    return state.replace(
        agent_cards=jnp.array(
            [[card(c) for c in h.split()] for h in holes], state.agent_cards.dtype
        ),
        flop_cards=jnp.array(b[:3], state.flop_cards.dtype),
        turn_card=jnp.asarray(b[3], state.turn_card.dtype),
        river_card=jnp.asarray(b[4], state.river_card.dtype),
    )


def act(env, state, action: int):
    """env.step_env after checking that the action is legal."""
    assert bool(env.get_avail_actions(state)[action]), (
        f"action {action} is illegal for player {int(state.current_player_idx)}"
    )
    state, _, reward, _, done, _ = env.step_env(
        jax.random.PRNGKey(0), state, jnp.int32(action)
    )
    return state, np.asarray(reward), bool(done)


def passive_action(env_id: str, env, state) -> int:
    """Check, or call when there is something to call."""
    if env_id == NLHE:
        return NL_CALL
    return CALL if bool(env.get_avail_actions(state)[CALL]) else CHECK


def expected_order(num_agents: int, sb: int, stage: int, in_hand) -> list[int]:
    """Players in the order they act in a betting round where everyone still
    in acts once (no raises)."""
    if stage == 0:
        first = (sb + 2) % num_agents  # the player after the big blind
    else:
        button = sb if num_agents == 2 else (sb - 1) % num_agents
        first = (button + 1) % num_agents
    seats = [(first + k) % num_agents for k in range(num_agents)]
    return [p for p in seats if in_hand[p]]


def play_passive(env_id: str, env, state, folder: int | None = None):
    """Plays a hand in which every player checks or calls, except `folder`,
    who folds at its first (preflop) turn. Returns the actors of every
    street, the final state and the rewards."""
    actors = {stage: [] for stage in range(4)}
    fold = NL_FOLD if env_id == NLHE else FOLD
    for _ in range(4 * env.num_agents):
        player, stage = int(state.current_player_idx), int(state.stage)
        actors[stage].append(player)
        a = fold if player == folder else passive_action(env_id, env, state)
        state, reward, done = act(env, state, a)
        if done:
            return actors, state, reward
    raise AssertionError("the hand did not end")


@pytest.mark.parametrize("env_id", [LHE, NLHE])
@pytest.mark.parametrize("num_agents", [2, 3, 4, 6])
def test_first_actor_of_every_street(env_id: str, num_agents: int) -> None:
    """Checked and called down from every small-blind seat: who acts first on
    every street, and the order of the players after it."""
    env = make(env_id, num_agents=num_agents)
    for sb in range(num_agents):
        state = reset_with_sb(env, sb)
        bb = (sb + 1) % num_agents
        assert int(state.current_player_idx) == (sb + 2) % num_agents
        actors, final, reward = play_passive(env_id, env, state)
        everyone = np.ones(num_agents, bool)
        for stage in range(4):
            assert actors[stage] == expected_order(num_agents, sb, stage, everyone), stage
        if num_agents == 2:
            # standard heads-up order: the SB (button) first preflop, the BB
            # first on the flop, turn and river
            assert [actors[s][0] for s in range(4)] == [sb, bb, bb, bb]
        else:
            assert [actors[s][0] for s in range(4)] == [(sb + 2) % num_agents] + [sb] * 3
        # everyone put in the big blind; the pot goes to the showdown winners
        np.testing.assert_array_equal(np.asarray(final.chips_in), np.full(num_agents, 2.0))
        assert abs(reward.sum()) < 1e-5 and int(final.stage) == 4


@pytest.mark.parametrize("env_id", [LHE, NLHE])
@pytest.mark.parametrize("fold_position", [0, 1, 2])  # SB, BB, the third seat
def test_first_actor_after_a_preflop_fold_three_players(env_id: str, fold_position: int) -> None:
    """With three players, one folds preflop and the others check it down:
    after the flop the first player still in from the small blind on acts
    first (the BB when the SB folded, else the SB)."""
    env = make(env_id, num_agents=3)
    for sb in range(3):
        folder = (sb + fold_position) % 3
        actors, final, reward = play_passive(env_id, env, reset_with_sb(env, sb), folder)
        in_hand = np.arange(3) != folder
        assert actors[0] == expected_order(3, sb, 0, np.ones(3, bool))
        for stage in (1, 2, 3):
            assert actors[stage] == expected_order(3, sb, stage, in_hand), stage
        first = (sb + 1) % 3 if fold_position == 0 else sb
        assert [actors[s][0] for s in (1, 2, 3)] == [first] * 3
        assert np.array_equal(np.asarray(final.folded), ~in_hand)
        # the folder loses what it put in (its blind, or nothing)
        chips = np.asarray(final.chips_in)
        assert chips[folder] == [1.0, 2.0, 0.0][fold_position]
        assert reward[folder] == -chips[folder] and abs(reward.sum()) < 1e-5


# Whole heads-up hands: (actor, action, chips in after it [SB, BB], stage
# after it; 4 = showdown), actors "S" (small blind) and "B" (big blind).
LHE_HAND = [
    ("S", RAISE, (4, 2), 0),
    ("B", CALL, (4, 4), 1),  # flop: the BB acts first
    ("B", RAISE, (4, 6), 1),  # bets 2
    ("S", RAISE, (8, 6), 1),  # raises 2
    ("B", CALL, (8, 8), 2),
    ("B", CHECK, (8, 8), 2),
    ("S", RAISE, (12, 8), 2),  # bets 4 on the turn
    ("B", CALL, (12, 12), 3),
    ("B", CHECK, (12, 12), 3),
    ("S", CHECK, (12, 12), 4),  # showdown
]
LHE_FOLD_HAND = [
    ("S", CALL, (2, 2), 0),
    ("B", CHECK, (2, 2), 1),
    ("B", RAISE, (2, 4), 1),  # the BB bets the flop first
    ("S", FOLD, (2, 4), 1),  # folds on the flop
]
NLHE_HAND = [
    ("S", NL_CALL, (2, 2), 0),  # limps
    ("B", NL_CALL, (2, 2), 1),  # checks; flop: the BB acts first
    ("B", HALF, (2, 4), 1),  # floor(4 / 2) = 2
    ("S", POT, (8, 4), 1),  # the pot, 6 (2 of them to call)
    ("B", NL_CALL, (8, 8), 2),
    ("B", NL_CALL, (8, 8), 2),  # checks
    ("S", NL_CALL, (8, 8), 3),  # checks
    ("B", ALLIN, (8, 100), 3),
    ("S", NL_CALL, (100, 100), 4),  # calls all in: showdown
]


@pytest.mark.parametrize(
    "env_id,script,sb_hand,reward_sb",
    [
        (LHE, LHE_HAND, BEST, 12.0),  # the SB wins the BB's 12 chips
        (LHE, LHE_FOLD_HAND, BEST, -2.0),  # the SB folds its 2 chips
        (NLHE, NLHE_HAND, WORSE, -100.0),  # the BB wins the SB's whole stack
    ],
    ids=["limit_showdown", "limit_flop_fold", "nolimit_allin_showdown"],
)
def test_heads_up_hand(env_id: str, script, sb_hand: str, reward_sb: float) -> None:
    """Every action of a whole hand from both small-blind seats: who acts,
    chips in the pot (and stacks in No-Limit), the street, and the payoffs."""
    env = make(env_id, num_agents=2)
    bb_hand = WORSE if sb_hand == BEST else BEST
    for sb in (0, 1):
        bb = 1 - sb
        seat = {"S": sb, "B": bb}
        holes = [sb_hand, bb_hand] if sb == 0 else [bb_hand, sb_hand]
        state = set_cards(reset_with_sb(env, sb), holes, BOARD)
        for t, (who, action, chips, stage) in enumerate(script):
            assert int(state.current_player_idx) == seat[who], t
            state, reward, done = act(env, state, action)
            want = np.zeros(2)
            want[sb], want[bb] = chips
            np.testing.assert_array_equal(np.asarray(state.chips_in), want, err_msg=str(t))
            if env_id == NLHE:
                np.testing.assert_array_equal(np.asarray(state.remaining_chips), 100 - want)
            last = t == len(script) - 1
            assert done == last, t
            if not done:
                assert int(state.stage) == stage, t
                assert not reward.any(), t
        if script[-1][3] == 4:  # the river was played out: showdown
            assert int(state.stage) == 4
        want = np.zeros(2)
        want[sb], want[bb] = reward_sb, -reward_sb
        np.testing.assert_array_equal(reward, want)
