"""7-Card Stud metrics (bluffjax.metrics.seven_card_stud): scripted hands for
every metric (detection, credit, no double counting, carry reset at the end
of a hand, including the env's auto-reset), a plain-Python replay of random
rollouts for 2 to 10 players that recomputes every count with an independent
deal and hand evaluator, and the values known under uniform random play."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.metrics import make_metrics, masked_totals
from bluffjax.metrics.seven_card_stud import final_hands
from bluffjax.tests.test_seven_card_stud import best_hand, board_key, five_card_key
from bluffjax.utils.game_utils.poker_utils import _get_bring_in_idx

CALL, RAISE, FOLD, CHECK = 0, 1, 2, 3
SUITS = "cdhs"
RANKS = {str(r): r for r in range(2, 10)} | {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
STREETS = (3, 4, 5, 6, 7)


def c(name: str) -> int:
    """'Ks' -> the env's card index (suit * 13 + rank - 1, ace = 0)."""
    rank, suit = RANKS[name[0]], SUITS.index(name[1])
    return suit * 13 + (0 if rank == 14 else rank - 1)


@functools.lru_cache(maxsize=None)
def _setup(n: int):
    env = make("seven_card_stud", num_agents=n)
    gm = make_metrics("seven_card_stud", env)
    return env, gm, jax.jit(env.step), jax.jit(gm.step), jax.jit(env.get_avail_actions)


def table(n: int, deck_names: str):
    """Start of a hand dealt from a deck that begins with deck_names: player p
    gets cards 3p, 3p + 1 (hole) and 3p + 2 (door), and each later street
    goes to the players still in, clockwise from the bring-in's left. The
    antes are in and the bring-in is to act (post or complete)."""
    env = _setup(n)[0]
    first = [c(x) for x in deck_names.split()]
    deck = first + [x for x in range(52) if x not in first]
    assert sorted(deck) == list(range(52))
    deck = jnp.array(deck, dtype=jnp.int32)
    bring_in = int(_get_bring_in_idx(deck[2 : 3 * n : 3]))
    state, _ = env.reset(jax.random.PRNGKey(0))
    return state.replace(
        deck=deck,
        agent_cards=jnp.full((n, 7), -1, jnp.int32).at[:, :3].set(deck[: 3 * n].reshape(n, 3)),
        bring_in_idx=jnp.int32(bring_in),
        current_player_idx=jnp.int32(bring_in),
    )


class Play:
    """Steps env.step (auto-reset, as in training) through Metrics.step."""

    def __init__(self, n: int, state):
        self.n = n
        self.env, self.gm, self._step, self._mstep, self._avail = _setup(n)
        self.state = state
        self.carry = self.gm.init_carry()
        self.totals = None
        self.t = 0

    def act(self, player: int, action: int, key=None) -> bool:
        assert int(self.state.current_player_idx) == player, (int(self.state.current_player_idx), player)
        assert bool(self._avail(self.state)[action]), action
        self.t += 1
        key = jax.random.PRNGKey(10_000 + self.t) if key is None else key
        nxt, _, reward, _, done, info = self._step(key, self.state, jnp.int32(action))
        self.carry, counts = self._mstep(
            self.carry, self.state, jnp.int32(action), nxt, reward, done, info
        )
        counts = {k: np.asarray(v) for k, v in counts.items()}
        self.totals = counts if self.totals is None else {
            k: self.totals[k] + counts[k] for k in counts
        }
        self.state = nxt
        self.last_reward = np.asarray(reward)
        return bool(done)

    def count(self, name: str) -> list:
        return self.totals[name].tolist()

    def street_ended(self) -> list:
        """Per player: the street (3-7) on which hands ended by a fold."""
        return [[s for s, x in zip(STREETS, row) for _ in range(int(x))] for row in self.totals["fold_street"]]

    def carry_is_reset(self) -> bool:
        return all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(
                jax.tree_util.tree_leaves(self.carry),
                jax.tree_util.tree_leaves(self.gm.init_carry()),
            )
        )


# Heads-up deals: player 0 has the 2c door card and is the bring-in: it acts
# first on 3rd street (posts or completes), then player 1; later streets go
# to player 1, then player 0.
FIFTH_STREET = (
    "3c 4d 2c  Kh Ks Kd"  # 3rd street: player 0, player 1
    "  5s 9h  7c 9d  8h Jc  Th Qd"  # 4th to 7th street: player 1, player 0
)


def test_bluff_on_fifth_street_ends_the_hand():
    """Player 0 posts the bring-in, player 1 calls (3rd street ends: no
    option for the bring-in). Player 0 pairs its board on 5th street and
    bets; its final hand (a pair of nines) loses to player 1's kings: a
    bluff with the board ahead. Player 1 folds: the hand ends by a fold on
    5th street, the bluff succeeds and the fold was not correct."""
    p = Play(2, table(2, FIFTH_STREET))
    p.act(0, CALL)  # posts the bring-in
    p.act(1, CALL)
    assert int(p.state.stage) == 1
    p.act(1, CHECK)  # 4th street: K-5 shows above 2-9
    p.act(0, CHECK)
    p.act(0, RAISE)  # 5th street: the pair of nines acts first
    assert p.act(1, FOLD)
    assert p.last_reward[0] > 0
    assert p.count("aggressive") == [1, 0]
    assert p.count("bluffs") == [1, 0] and p.count("bluff_wins") == [1, 0]
    assert p.count("aggressive_board_ahead") == [1, 0]
    assert p.count("bluffs_board_ahead") == [1, 0]
    assert p.count("bluff_wins_board_ahead") == [1, 0]
    assert p.count("aggressive_board_not_ahead") == [0, 0]
    assert p.count("folds") == [0, 1] and p.count("correct_folds") == [0, 0]
    assert p.count("steals") == [0, 0]
    assert p.count("hands") == [1, 1] and p.count("showdowns") == [0, 0]
    assert p.street_ended() == [[5], [5]]
    assert p.carry_is_reset()
    rates = p.gm.summarize({k: v.sum(0) for k, v in p.totals.items()})
    assert rates["fold_street_5"] == 1 and rates["showdown_rate"] == 0
    assert np.isnan(rates["steal_success"]) and rates["n_steals"] == 0
    assert rates["bluff_success_board_ahead"] == 1
    assert np.isnan(rates["bluff_rate_board_not_ahead"])


def test_the_bring_in_completes_with_a_losing_hand():
    """The bring-in completes instead of posting: an aggressive action, a
    steal (the first raise of 3rd street) and, with the pair of nines against
    trip kings, a bluff, made with the board behind (the lowest door card).
    Player 1 folds at once: steal and bluff win; the fold was not correct."""
    p = Play(2, table(2, FIFTH_STREET))
    assert p.act(0, RAISE) is False
    assert p.act(1, FOLD)
    assert p.last_reward.tolist() == [0.5, -0.5]  # the ante, in chips
    assert p.count("aggressive") == [1, 0] and p.count("steals") == [1, 0]
    assert p.count("steal_wins") == [1, 0]
    assert p.count("bluffs") == [1, 0] and p.count("bluff_wins") == [1, 0]
    assert p.count("aggressive_board_not_ahead") == [1, 0]
    assert p.count("bluff_wins_board_not_ahead") == [1, 0]
    assert p.count("aggressive_board_ahead") == [0, 0]
    assert p.count("folds") == [0, 1] and p.count("correct_folds") == [0, 0]
    assert p.street_ended() == [[3], [3]]


# player 0: aces (brings in with the 2c); player 1: king high
STEAL = "Ah As 2c  3h 4s Kd  6d 5h  8c 7s  Jd Tc  9s Qh"


def test_posting_the_bring_in_counts_nothing():
    """Posting the bring-in is the forced minimum: no aggressive action, no
    steal, and it answers nothing (the carry stays as it was). Player 1
    folds to it: the hand ends on 3rd street without a bet or raise, so
    there is no bluff or steal to win; the fold (king high against aces)
    is correct."""
    p = Play(2, table(2, STEAL))
    p.act(0, CALL)
    assert p.carry_is_reset()
    assert not any(v.any() for v in p.totals.values())
    assert p.act(1, FOLD)
    assert p.last_reward.tolist() == [0.5, -0.5]
    for k in ("aggressive", "steals", "bluffs", "bluff_wins", "steal_wins"):
        assert p.count(k) == [0, 0], k
    assert p.count("folds") == [0, 1] and p.count("correct_folds") == [0, 1]
    assert p.street_ended() == [[3], [3]]


def test_the_bring_in_completes_and_wins_immediately():
    """The bring-in's completion with aces: a steal, no bluff, with the
    board behind; player 1 folds (correctly) and the steal wins."""
    p = Play(2, table(2, STEAL))
    p.act(0, RAISE)
    assert p.act(1, FOLD)
    assert p.count("steals") == [1, 0] and p.count("steal_wins") == [1, 0]
    assert p.count("aggressive_board_not_ahead") == [1, 0]
    assert p.count("bluffs") == [0, 0] and p.count("bluff_wins") == [0, 0]
    assert p.count("correct_folds") == [0, 1]


def test_third_street_completion_wins_immediately():
    """After the bring-in posts, player 1 completes (a steal and, with king
    high against aces, a bluff, made with the board ahead) and the bring-in
    folds."""
    p = Play(2, table(2, STEAL))
    p.act(0, CALL)
    p.act(1, RAISE)  # completes the bring-in
    assert p.act(0, FOLD)
    assert p.last_reward.tolist() == [-1.0, 1.0]
    assert p.count("steals") == [0, 1] and p.count("steal_wins") == [0, 1]
    assert p.count("bluffs") == [0, 1] and p.count("bluff_wins") == [0, 1]
    assert p.count("bluffs_board_ahead") == [0, 1]  # a king against a deuce
    assert p.count("folds") == [1, 0] and p.count("correct_folds") == [0, 0]
    assert p.street_ended() == [[3], [3]]


def test_called_steal_then_a_free_fold_is_no_success():
    """The completion is called; on 4th street the caller folds although it
    could check: the hand ends without a showdown, but the steal (and bluff)
    was answered by the call."""
    p = Play(2, table(2, STEAL))
    p.act(0, CALL)
    p.act(1, RAISE)
    p.act(0, CALL)
    p.act(1, CHECK)  # K-6 above 2-5
    assert p.act(0, FOLD)
    assert p.last_reward[1] > 0
    assert p.count("steals") == [0, 1] and p.count("steal_wins") == [0, 0]
    assert p.count("bluffs") == [0, 1] and p.count("bluff_wins") == [0, 0]
    assert p.street_ended() == [[4], [4]]


@pytest.mark.parametrize("bring_in_completes", (False, True))
def test_reraise_is_not_a_steal_and_answers_one(bring_in_completes):
    """A raise over the completion (whoever completed) is not a steal and
    answers the steal: the steal doesn't win when the other player folds."""
    p = Play(2, table(2, STEAL))
    if bring_in_completes:
        p.act(0, RAISE)
        p.act(1, RAISE)
        assert p.act(0, FOLD)
        assert p.count("steals") == [1, 0] and p.count("steal_wins") == [0, 0]
        assert p.count("bluff_wins") == [0, 1]  # king high re-raises aces: a won bluff
    else:
        p.act(0, CALL)
        p.act(1, RAISE)
        p.act(0, RAISE)
        assert p.act(1, FOLD)
        assert p.count("steals") == [0, 1] and p.count("steal_wins") == [0, 0]
        assert p.count("bluff_wins") == [0, 0]  # the re-raise with aces is no bluff
    assert p.count("aggressive") == [1, 1]


# player 0 makes a heart flush with the face-down 7th-street card; player 1
# shows the best board throughout with a pair of queens in the hole
SHOWDOWN = "3h 5h 2c  Qs Qd Kd  7c 9h  8d Jh  Tc 4s  6s Kh"


def test_showdown_and_a_bluff_against_a_hand_made_on_seventh_street():
    p = Play(2, table(2, SHOWDOWN))
    p.act(0, CALL)
    p.act(1, RAISE)  # loses to the flush made on 7th street: a bluff
    p.act(0, CALL)
    for _ in range(4):  # 4th to 7th street: the king high board acts first
        p.act(1, CHECK)
        done = p.act(0, CHECK)
    assert done and p.last_reward.tolist() == [1.5, -1.5]  # chips
    assert p.count("showdowns") == [1, 1] and p.count("hands") == [1, 1]
    assert p.street_ended() == [[], []]
    assert p.count("bluffs") == [0, 1] and p.count("bluffs_board_ahead") == [0, 1]
    assert p.count("bluff_wins") == [0, 0]
    assert p.count("steals") == [0, 1] and p.count("steal_wins") == [0, 0]
    rates = p.gm.summarize({k: v.sum(0) for k, v in p.totals.items()})
    assert rates["showdown_rate"] == 1 and rates["n_hands"] == 2
    assert all(rates[f"fold_street_{k}"] == 0 for k in STREETS)


# Three players, player 0 brings in (2c), then players 1 and 2. With all three
# in, player 0 makes a club flush; after player 2 folds on 3rd street every
# later card shifts (player 0 gets player 2's) and player 0 ends with jack high.
THREE = (
    "3c 4c 2c  Qh Qs Kd  3h 4h 7s"
    "  7h 9s 9c  8d Td Jc  6s 5d Ah  5s Jh Kc"
)


def test_three_players_final_hands_at_the_time_of_the_action():
    """Player 0 posts the bring-in. Player 1's completion (pair of queens)
    loses to player 0's flush in the deal with all three in: a bluff,
    although after player 2's fold player 1 would have won. Player 2's fold
    is correct (jack high). Player 0 calls, which ends 3rd street. On 4th
    street player 0 (now jack high against queens) bluffs with the board
    behind and player 1 folds."""
    env = _setup(3)[0]
    state = table(3, THREE)
    hands = np.asarray(final_hands(env, state))
    assert [best_hand(h)[0] for h in hands.tolist()] == [6, 2, 1]  # flush, pair, high
    p = Play(3, state)
    p.act(0, CALL)
    p.act(1, RAISE)
    p.act(2, FOLD)
    p.act(0, CALL)
    assert int(p.state.stage) == 1
    after_fold = np.asarray(final_hands(env, p.state))
    assert best_hand(after_fold[1].tolist()) > best_hand(after_fold[0].tolist())
    p.act(1, CHECK)  # 4th street: K-7 shows above 2-9
    p.act(0, RAISE)
    assert p.act(1, FOLD)
    assert p.last_reward.tolist() == [2.0, -1.5, -0.5]  # chips
    assert p.count("aggressive") == [1, 1, 0]
    assert p.count("bluffs") == [1, 1, 0]
    assert p.count("bluffs_board_ahead") == [0, 1, 0]
    assert p.count("bluffs_board_not_ahead") == [1, 0, 0]
    assert p.count("bluff_wins") == [1, 0, 0]
    assert p.count("bluff_wins_board_not_ahead") == [1, 0, 0]
    assert p.count("folds") == [0, 1, 1] and p.count("correct_folds") == [0, 0, 1]
    assert p.count("steals") == [0, 1, 0] and p.count("steal_wins") == [0, 0, 0]
    # credited to every player dealt in, also the one who folded on 3rd street
    assert p.count("hands") == [1, 1, 1]
    assert p.street_ended() == [[4], [4], [4]]


def test_carry_resets_between_hands():
    """A hand ends with the bring-in's (player 0's) successful completion, a
    bluff and a steal: the carry is back to its initial value. In the next
    hand (the env's auto-reset deal) player 0 is the bring-in again, posts,
    and player 1 folds at once: player 0 wins without a bet or raise, so no
    further bluff or steal is counted or won."""
    env = _setup(2)[0]
    # a key whose reset deal makes player 0 the bring-in again
    key = next(
        k for k in map(jax.random.PRNGKey, range(100))
        if int(env.reset(jax.random.split(k)[1])[0].bring_in_idx) == 0
    )
    p = Play(2, table(2, FIFTH_STREET))
    p.act(0, RAISE)
    assert p.act(1, FOLD, key=key)
    assert p.count("bluff_wins") == [1, 0] and p.count("steal_wins") == [1, 0]
    assert p.carry_is_reset()
    before = {k: v.copy() for k, v in p.totals.items()}
    assert int(p.state.bring_in_idx) == 0 and int(p.state.current_player_idx) == 0
    p.act(0, CALL)
    assert p.carry_is_reset()
    assert p.act(1, FOLD)
    assert p.last_reward[0] > 0
    for k in ("bluffs", "bluff_wins", "steal_wins", "aggressive", "steals"):
        np.testing.assert_array_equal(p.totals[k], before[k], err_msg=k)
    assert p.count("hands") == [2, 2]
    assert p.street_ended() == [[3, 3], [3, 3]]


# ---------------------------------------------------------------- random play


def random_rollout(n: int, envs: int, steps: int, seed: int):
    """Uniform random legal play over auto-resetting envs, with Metrics.step
    in the loop as in the training scripts. Also returns the env's state
    after every step before the auto-reset (the true end of a hand)."""
    env, gm = _setup(n)[:2]
    state, _ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(seed), envs))
    carry = jax.vmap(lambda _: gm.init_carry())(jnp.arange(envs))

    def one(c, key):
        state, carry = c
        k_action, k_step = jax.random.split(key)
        avail = jax.vmap(env.get_avail_actions)(state)
        action = jax.random.categorical(k_action, jnp.where(avail, 0.0, -jnp.inf))
        keys = jax.random.split(k_step, envs)
        nxt, _, reward, _, done, info = jax.vmap(env.step)(keys, state, action)
        raw = jax.vmap(lambda k, s, a: env.step_env(jax.random.split(k)[0], s, a)[0])(
            keys, state, action
        )
        carry, counts = jax.vmap(gm.step)(carry, state, action, nxt, reward, done, info)
        return (nxt, carry), (state, action, raw, done, counts)

    run = jax.jit(lambda s, c, k: jax.lax.scan(one, (s, c), jax.random.split(k, steps)))
    _, out = run(state, carry, jax.random.PRNGKey(seed + 1))
    return jax.tree_util.tree_map(np.asarray, out)


def deal_rest(n, s):
    """Plain Python: every player's cards if nobody else folds, from the
    env's deck and the deal rule in its docstring."""
    cards = [list(map(int, h)) for h in s.agent_cards]
    community = list(map(bool, s.community))
    num_dealt = int(s.num_dealt)
    order = [(int(s.bring_in_idx) + 1 + i) % n for i in range(n)]
    active = [q for q in order if not s.folded[q]]
    for street in range(int(s.stage) + 1, 5):
        if 52 - num_dealt - len(active) < 4 - street:  # one community card
            for q in active:
                cards[q][street + 2] = int(s.deck[num_dealt])
            num_dealt += 1
            community[street] = True
        else:
            for i, q in enumerate(active):
                cards[q][street + 2] = int(s.deck[num_dealt + i])
            num_dealt += len(active)
    return cards


def board(s, q):
    """Plain Python: player q's face-up cards on the current street."""
    up = [int(x) for x in s.agent_cards[q][2:]]
    if int(s.stage) == 4 and s.community[4]:
        return five_card_key(up)
    return board_key(up[: min(int(s.stage) + 1, 4)])


NAMES = (
    "aggressive", "bluffs", "bluff_wins", "folds", "correct_folds",
    "aggressive_board_ahead", "bluffs_board_ahead", "bluff_wins_board_ahead",
    "aggressive_board_not_ahead", "bluffs_board_not_ahead", "bluff_wins_board_not_ahead",
    "steals", "steal_wins", "hands", "showdowns",
)


def reference_counts(n, states, actions, raws, dones):
    T, N = dones.shape
    out = {k: np.zeros((T, N, n), np.float32) for k in NAMES}
    out["fold_street"] = np.zeros((T, N, n, 5), np.float32)
    best = functools.lru_cache(maxsize=None)(lambda h: best_hand(list(h)))
    for e in range(N):
        aggressor, flags = None, None
        for t in range(T):
            s = jax.tree_util.tree_map(lambda x: x[t, e], states)
            p, a, stage = int(s.current_player_idx), int(actions[t, e]), int(s.stage)
            active = [q for q in range(n) if not s.folded[q]]
            others = [q for q in active if q != p]
            if a in (RAISE, FOLD):
                final = deal_rest(n, s)
                keys = {q: best(tuple(sorted(final[q]))) for q in active}
                loses = any(keys[q] > keys[p] for q in others)
                ahead = board(s, p) > max(board(s, q) for q in others)
            if a == RAISE:
                steal = stage == 0 and int(s.raise_nums[0]) == 0
                side = "board_ahead" if ahead else "board_not_ahead"
                out["aggressive"][t, e, p] = out[f"aggressive_{side}"][t, e, p] = 1
                out["bluffs"][t, e, p] = out[f"bluffs_{side}"][t, e, p] = loses
                out["steals"][t, e, p] = steal
                aggressor, flags = p, (loses, side, steal)
            elif a == FOLD:
                out["folds"][t, e, p] = 1
                out["correct_folds"][t, e, p] = loses
            elif stage == 0 and a == CALL and max(s.chips_in) == 0.5:
                # the bring-in posts, the forced minimum: neither aggressive
                # nor a call; the hand's first action, so there is no bet yet
                assert aggressor is None and p == int(s.bring_in_idx)
            else:  # a call or check answers the last bet or raise
                aggressor = None
            if dones[t, e]:
                left = [q for q in range(n) if not raws.folded[t, e][q]]
                out["hands"][t, e] = 1
                if len(left) == 1:
                    out["fold_street"][t, e, :, stage] = 1
                    if aggressor == left[0]:
                        loses, side, steal = flags
                        out["bluff_wins"][t, e, aggressor] = loses
                        out[f"bluff_wins_{side}"][t, e, aggressor] = loses
                        out["steal_wins"][t, e, aggressor] = steal
                else:
                    out["showdowns"][t, e] = 1
                aggressor, flags = None, None
    return out


@pytest.mark.parametrize("n, envs, steps", [(2, 64, 300), (3, 48, 300), (8, 16, 300), (10, 16, 300)])
def test_counts_match_plain_python_replay(n, envs, steps):
    states, actions, raws, dones, counts = random_rollout(n, envs, steps, seed=n)
    ref = reference_counts(n, states, actions, raws, dones)
    for k, v in ref.items():
        np.testing.assert_array_equal(counts[k], v, err_msg=k)
    # steals by the bring-in (completing instead of posting) and by others
    steal = (actions == RAISE) & (states.stage == 0) & (states.raise_nums[..., 0] == 0)
    by_bring_in = states.current_player_idx == states.bring_in_idx
    assert (steal & by_bring_in).any() and (steal & ~by_bring_in).any()
    totals = {k: v.sum() for k, v in ref.items()}
    for k in ("bluff_wins_board_ahead", "bluff_wins_board_not_ahead", "correct_folds",
              "showdowns", "steals"):
        assert totals[k] > 0, k
    if n <= 3:  # a steal rarely makes nine opponents fold
        assert totals["steal_wins"] > 0
    if n >= 9:  # decisions whose predicted deal has community cards on 6th and 7th
        in_hand = (~states.folded).sum(-1)
        decided = np.isin(actions, (RAISE, FOLD))
        assert (decided & (states.stage <= 2) & (in_hand >= 9)).sum() > 50


def test_final_hands_with_community_cards():
    """Ten players all in on 3rd street: 4th and 5th street go to each player,
    6th and 7th street are one community card each (the deck runs short)."""
    env = _setup(10)[0]
    state = env.reset(jax.random.PRNGKey(3))[0]
    deck = np.asarray(state.deck)
    order = [(int(state.bring_in_idx) + 1 + i) % 10 for i in range(10)]
    hands = np.asarray(final_hands(env, state))
    np.testing.assert_array_equal(hands[:, :3], deck[:30].reshape(10, 3))
    for i, p in enumerate(order):
        assert hands[p, 3] == deck[30 + i] and hands[p, 4] == deck[40 + i]
    np.testing.assert_array_equal(hands[:, 5], deck[50])
    np.testing.assert_array_equal(hands[:, 6], deck[51])
    # one player out: nine get 4th and 5th street, then the deck is short again
    folded = state.replace(folded=state.folded.at[order[0]].set(True))
    hands = np.asarray(final_hands(env, folded))
    assert (hands[order[0], 3:] == -1).all()
    for i, p in enumerate(order[1:]):
        assert hands[p, 3] == deck[30 + i] and hands[p, 4] == deck[39 + i]
    np.testing.assert_array_equal(hands[order[1:], 5], deck[48])
    np.testing.assert_array_equal(hands[order[1:], 6], deck[49])


@pytest.mark.parametrize("n", (2, 3, 10))
def test_final_hands_are_the_showdown_hands(n):
    """At every step of a hand that reaches a showdown with no fold after
    that step, final_hands gives the hands shown down (heads-up: every
    step of every hand that reaches a showdown)."""
    env = _setup(n)[0]
    states, actions, raws, dones, _ = random_rollout(n, 64, 300, seed=20 + n)
    predicted = np.asarray(jax.jit(jax.vmap(jax.vmap(lambda s: final_hands(env, s))))(states))
    checked = 0
    for e in range(dones.shape[1]):
        start = 0
        for end in np.flatnonzero(dones[:, e]):
            shown = raws.agent_cards[end, e]
            left = ~raws.folded[end, e]
            if left.sum() > 1:  # a showdown
                for t in range(start, end + 1):
                    # a fold at step t itself changes the deal after it
                    later_folds = (actions[t : end + 1, e] == FOLD).any()
                    if not later_folds:
                        np.testing.assert_array_equal(predicted[t, e][left], shown[left])
                        checked += 1
                    elif n == 2:
                        raise AssertionError("heads-up showdown after a fold")
            start = end + 1
    assert checked > 100


def p_fold_on_street(third: bool) -> float:
    """Heads-up, uniform random legal actions: probability that a betting
    round ends with a fold, at most 4 raises per street. On 3rd street the
    bring-in first posts the bring-in or completes (probability 1/2 each; it
    can't fold or check) and the other player then faces that bet (the
    bring-in has no option); later streets start with equal chips."""

    def p(facing, acted, raises):  # acted: players who bet, called or checked since the last bet
        legal = ["fold", "call" if facing else "check"] + (["raise"] if raises < 4 else [])
        total = 0.0
        for a in legal:
            if a == "fold":
                total += 1.0
            elif a == "raise":
                total += p(True, 1, raises + 1)
            elif acted + 1 < 2:
                total += p(False, acted + 1, raises)
        return total / len(legal)

    if third:  # posted (no raise yet) or completed (the first raise)
        return (p(True, 1, 0) + p(True, 1, 1)) / 2
    return p(False, 0, 0)


def test_uniform_random_play_values():
    """Heads-up uniform random play: the street on which a hand ends follows
    from the betting rounds alone (exact probabilities below); a first raise
    on 3rd street (the bring-in's completion, or a completion after it
    posted) is answered by fold, call or raise, so steal_success = 1/3;
    a bluff succeeds when the opponent folds at once, with probability 1/3,
    or 1/2 after the 4th raise of a street (no raise left)."""
    env, gm = _setup(2)[:2]
    states, actions, raws, dones, counts = random_rollout(2, 1024, 300, seed=3)
    rates = gm.summarize(masked_totals(counts, np.ones(dones.shape + (2,), bool)))
    rates = {k: float(v) for k, v in rates.items()}

    third, later = p_fold_on_street(True), p_fold_on_street(False)
    alive = 1 - third
    expected = {"fold_street_3": third}
    for k in (4, 5, 6, 7):
        expected[f"fold_street_{k}"] = alive * later
        alive *= 1 - later
    expected["showdown_rate"] = alive
    hands = rates["n_hands"] / 2  # every hand is credited to both players
    assert hands > 30_000
    for name, prob in expected.items():
        se = np.sqrt(prob * (1 - prob) / hands)
        assert abs(rates[name] - prob) < 4 * se + 1e-9, (name, rates[name], prob)
    assert sum(rates[name] for name in expected) == pytest.approx(1.0, abs=1e-6)

    assert rates["n_steals"] > 10_000
    assert abs(rates["steal_success"] - 1 / 3) < 4 * np.sqrt(2 / 9 / rates["n_steals"])

    bluff = counts["bluffs"].sum(-1) > 0
    stage = states.stage[bluff]
    capped = states.raise_nums[bluff, stage] == 3
    p_win = np.where(capped, 1 / 2, 1 / 3)
    assert capped.any() and (~capped).any()
    se = np.sqrt((p_win * (1 - p_win)).sum()) / len(p_win)
    assert abs(rates["bluff_success"] - p_win.mean()) < 4 * se
    for name in gm.rate_names:
        assert 0 <= rates[name] <= 1, name


# ---------------------------------------------------------------- scorers


def test_board_scores_are_the_env_scorers_values():
    """visible_score equals poker_utils._score_visible_upcards on every board
    of 1 to 4 cards; the 5-card community board uses five_card_score (equal
    to _score_five_card_hand, see test_metrics_five_card_draw)."""
    import itertools

    from bluffjax.metrics._poker_draw_stud import visible_score
    from bluffjax.utils.game_utils.poker_utils import _score_visible_upcards

    fast, env_scorer = jax.jit(jax.vmap(visible_score)), jax.jit(jax.vmap(_score_visible_upcards))
    for k in (1, 2, 3, 4):
        boards = jnp.asarray(np.array(list(itertools.combinations(range(52), k)), np.int32))
        np.testing.assert_array_equal(np.asarray(fast(boards)), np.asarray(env_scorer(boards)), err_msg=str(k))

