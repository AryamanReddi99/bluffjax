"""5-Card Draw metrics (bluffjax.metrics.five_card_draw): scripted hands for
every metric (detection, credit, no double counting, carry reset at the end of
a hand, including the env's auto-reset), a plain-Python replay of random
rollouts that recomputes every count from the true post-step states with an
independent hand evaluator, and the values known under uniform random play."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.metrics import make_metrics, masked_totals
from bluffjax.metrics.five_card_draw import hand_after_draw
from bluffjax.tests.test_seven_card_stud import five_card_key

CALL, HALF_POT, POT, ALL_IN, FOLD = 0, 1, 2, 3, 4
KEEP_ALL = 36
SUITS = "cdhs"
RANKS = {str(r): r for r in range(2, 10)} | {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}


def c(name: str) -> int:
    """'Ks' -> the env's card index (suit * 13 + rank - 1, ace = 0)."""
    rank, suit = RANKS[name[0]], SUITS.index(name[1])
    return suit * 13 + (0 if rank == 14 else rank - 1)


def cards(names: str) -> list[int]:
    return [c(n) for n in names.split()]


def keep(hand: list[int], kept: list[int]) -> int:
    """Draw action keeping `kept` of `hand` (bit j = j-th smallest card)."""
    return 5 + sum(1 << j for j, card in enumerate(sorted(hand)) if card in kept)


@functools.lru_cache(maxsize=None)
def _setup(n: int):
    env = make("five_card_draw", num_agents=n)
    gm = make_metrics("five_card_draw", env)
    return env, gm, jax.jit(env.step), jax.jit(gm.step), jax.jit(env.get_avail_actions)


def table(n: int, hands: list[str], stock: str = "", sb: int = 0):
    """Start of a hand where player p holds hands[p], the stock starts with
    `stock` and player sb posts the small blind."""
    env = _setup(n)[0]
    held = [cards(h) for h in hands]
    first = [x for h in held for x in sorted(h)] + cards(stock)
    assert len(set(first)) == len(first)
    deck = first[: 5 * n] + first[5 * n :] + [x for x in range(52) if x not in first]
    key = next(k for k in range(200) if int(env.reset(jax.random.PRNGKey(k))[0].small_blind_idx) == sb)
    state, _ = env.reset(jax.random.PRNGKey(key))
    return state.replace(
        agent_cards=jnp.array([sorted(h) for h in held], dtype=jnp.int32),
        shuffled_deck=jnp.array(deck, dtype=jnp.int32),
    )


class Play:
    """Steps env.step (auto-reset, as in training) through Metrics.step."""

    def __init__(self, n: int, state):
        self.env, self.gm, self._step, self._mstep, self._avail = _setup(n)
        self.state = state
        self.carry = self.gm.init_carry()
        self.totals = None
        self.dones = []
        self.t = 0

    def act(self, player: int, action: int):
        assert int(self.state.current_player_idx) == player, (int(self.state.current_player_idx), player)
        assert bool(self._avail(self.state)[action]), action
        self.t += 1
        nxt, _, reward, _, done, info = self._step(
            jax.random.PRNGKey(10_000 + self.t), self.state, jnp.int32(action)
        )
        self.carry, counts = self._mstep(
            self.carry, self.state, jnp.int32(action), nxt, reward, done, info
        )
        counts = {k: np.asarray(v) for k, v in counts.items()}
        self.totals = counts if self.totals is None else {
            k: self.totals[k] + counts[k] for k in counts
        }
        self.state = nxt
        self.dones.append(bool(done))
        self.last_reward = np.asarray(reward)
        return bool(done)

    def count(self, name: str) -> list[float]:
        return self.totals[name].tolist()

    def carry_is_reset(self) -> bool:
        init = self.gm.init_carry()
        return all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(jax.tree_util.tree_leaves(self.carry), jax.tree_util.tree_leaves(init))
        )


# Heads-up: the small blind (seat sb) is the dealer; it acts first before the
# draw, the big blind draws first and bets first after the draw.
A, B = 0, 1
SNOWER = "Ks Qh 9d 7c 4s"  # king high, no straight or flush draw
PAIR = "8s 8h 3d 5c 2s"


def test_snow_bet_and_fold():
    """A stands pat with king high, bets after the draw, B (pair of eights)
    folds: one snow and its success, one post-draw bluff and its success,
    B's fold is not a correct fold; B's draw of 3 doesn't improve it."""
    p = Play(2, table(2, [SNOWER, PAIR], stock="Jc 6d Tc", sb=A))
    p.act(A, CALL)
    p.act(B, CALL)  # check: the draw
    p.act(B, keep(cards(PAIR), cards("8s 8h")))
    p.act(A, KEEP_ALL)
    p.act(B, CALL)  # check
    p.act(A, POT)
    assert not any(p.dones)
    assert p.act(B, FOLD)
    assert p.last_reward[A] > 0
    assert p.count("draws") == [1, 1]
    assert p.count("cards") == [0, 3]
    assert p.count("draws_known") == [1, 1]
    assert p.count("draws_improved") == [0, 0]
    assert p.count("high_card_draws") == [1, 0]
    assert p.count("snows") == [1, 0]
    assert p.count("snow_wins") == [1, 0]
    assert p.count("postdraw_aggressive") == [1, 0]
    assert p.count("postdraw_bluffs") == [1, 0]
    assert p.count("postdraw_bluff_wins") == [1, 0]
    assert p.count("postdraw_folds") == [0, 1]
    assert p.count("postdraw_correct_folds") == [0, 0]
    assert p.carry_is_reset()
    rates = p.gm.summarize({k: v.sum(0) for k, v in p.totals.items()})
    assert rates["cards_drawn"] == 1.5 and rates["n_draws"] == 2
    assert rates["snow_rate"] == 1 and rates["snow_success"] == 1
    assert rates["postdraw_bluff_rate"] == 1 and rates["postdraw_bluff_success"] == 1
    assert rates["postdraw_correct_fold_rate"] == 0 and rates["n_postdraw_folds"] == 1


def test_pair_to_trips_improves_and_a_better_pair_does_not():
    """B keeps its eights and draws the third; A draws five and makes a pair
    of kings from a pair of threes (same category). B's value bet isn't a
    bluff; A's fold to it is correct; nobody snowed."""
    hand_a, hand_b = "3c 3d Kh 9s 5c", "8s 8h 3h 5d 2c"
    p = Play(2, table(2, [hand_a, hand_b], stock="8d Jc 6d Ks Kd 4h 7c Th", sb=A))
    p.act(A, CALL)
    p.act(B, CALL)
    p.act(B, keep(cards(hand_b), cards("8s 8h")))
    p.act(A, keep(cards(hand_a), []))
    assert sorted(np.asarray(p.state.agent_cards[A]).tolist()) == sorted(cards("Ks Kd 4h 7c Th"))
    p.act(B, HALF_POT)
    assert p.act(A, FOLD)
    assert p.count("cards") == [5, 3]
    assert p.count("draws_improved") == [0, 1]
    assert p.count("high_card_draws") == [0, 0]
    assert p.count("snows") == [0, 0]
    assert p.count("postdraw_aggressive") == [0, 1]
    assert p.count("postdraw_bluffs") == [0, 0]
    assert p.count("postdraw_bluff_wins") == [0, 0]
    assert p.count("postdraw_folds") == [1, 0]
    assert p.count("postdraw_correct_folds") == [1, 0]


def test_called_bluff_and_snow_that_reaches_showdown():
    p = Play(2, table(2, [SNOWER, PAIR], stock="Jc 6d Tc", sb=A))
    p.act(A, CALL)
    p.act(B, CALL)
    p.act(B, keep(cards(PAIR), cards("8s 8h")))
    p.act(A, KEEP_ALL)
    p.act(B, CALL)
    p.act(A, HALF_POT)
    assert p.act(B, CALL)  # showdown, B wins
    assert p.last_reward[B] > 0
    assert p.count("snows") == [1, 0] and p.count("snow_wins") == [0, 0]
    assert p.count("postdraw_bluffs") == [1, 0] and p.count("postdraw_bluff_wins") == [0, 0]
    assert p.count("postdraw_folds") == [0, 0]
    assert p.carry_is_reset()


def test_reraise_answers_a_bet_and_can_be_a_successful_bluff():
    """A (two pair) bets, B (pair) raises all-in as a bluff, A folds: B's bluff
    succeeds, A's bet was no bluff and A's fold was wrong."""
    hand_a, hand_b = "9s 9h 4d 4c Ks", "8s 8h 3d 5c 2s"
    p = Play(2, table(2, [hand_a, hand_b], sb=B))  # B is the dealer now
    p.act(B, CALL)
    p.act(A, CALL)
    p.act(A, KEEP_ALL)  # the big blind draws first
    p.act(B, KEEP_ALL)
    p.act(A, POT)  # and bets first
    p.act(B, ALL_IN)
    assert p.act(A, FOLD)
    assert p.count("postdraw_aggressive") == [1, 1]
    assert p.count("postdraw_bluffs") == [0, 1]
    assert p.count("postdraw_bluff_wins") == [0, 1]
    assert p.count("postdraw_folds") == [1, 0]
    assert p.count("postdraw_correct_folds") == [0, 0]
    # two pair and a pair stand pat: no high-card draws, no snows
    assert p.count("high_card_draws") == [0, 0] and p.count("snows") == [0, 0]


def test_ties_neither_lose_nor_bluff():
    """Same ranks, different suits: a bet with the tied hand is no bluff and a
    fold of it is not a correct fold."""
    hand_a, hand_b = "Ks Qh 9d 7c 4s", "Kd Qc 9h 7s 4c"
    p = Play(2, table(2, [hand_a, hand_b], sb=A))
    p.act(A, CALL)
    p.act(B, CALL)
    p.act(B, KEEP_ALL)
    p.act(A, KEEP_ALL)
    p.act(B, POT)
    assert p.act(A, FOLD)
    assert p.count("snows") == [1, 1] and p.count("snow_wins") == [0, 1]
    assert p.count("postdraw_bluffs") == [0, 0]
    assert p.count("postdraw_correct_folds") == [0, 0] and p.count("postdraw_folds") == [1, 0]


def test_draw_that_ends_the_hand_is_read_from_the_stock():
    """Both all-in before the draw: the hand goes to showdown right after the
    second draw, so next_state is already the next hand. A's improvement
    (pair of fives to trips) on that last draw is still seen; B snows."""
    hand_a, hand_b = "5s 5h Kd 9c 2s", "Ah Jd 8c 6s 3h"
    p = Play(2, table(2, [hand_a, hand_b], stock="5d Qc 7h", sb=A))
    p.act(A, ALL_IN)
    p.act(B, CALL)  # all-in call: the draw, then the showdown
    p.act(B, KEEP_ALL)
    assert p.act(A, keep(cards(hand_a), cards("5s 5h")))
    assert p.last_reward[A] > 0  # trips beat ace high
    assert int(p.state.timestep) == 0  # the next hand
    assert p.count("draws") == [1, 1] and p.count("draws_known") == [1, 1]
    assert p.count("draws_improved") == [1, 0]
    assert p.count("snows") == [0, 1] and p.count("snow_wins") == [0, 0]
    assert p.count("postdraw_aggressive") == [0, 0]  # the all-in was before the draw
    assert p.carry_is_reset()


def test_three_players_bluff_against_one_better_hand():
    """Post-draw, the small blind bets a pair of nines that beats the big blind
    but not UTG (trips): a bluff. Both fold: the bluff succeeds; the big
    blind's fold is correct, UTG's is not. Credit goes to the right seats."""
    sb, bb, utg = 1, 2, 0
    hands = {sb: "9s 9h 4d 3c Ks", bb: "8s 8h 2d 5c Js", utg: "Qd Qc Qh 6s 7d"}
    p = Play(3, table(3, [hands[i] for i in range(3)], sb=sb))
    p.act(utg, CALL)
    p.act(sb, CALL)
    p.act(bb, CALL)
    for player in (sb, bb, utg):  # the dealer (UTG) draws last
        p.act(player, KEEP_ALL)
    p.act(sb, POT)  # first after the dealer
    p.act(bb, FOLD)
    assert p.act(utg, FOLD)
    assert p.count("postdraw_aggressive") == [0, 1, 0]
    assert p.count("postdraw_bluffs") == [0, 1, 0]
    assert p.count("postdraw_bluff_wins") == [0, 1, 0]
    assert p.count("postdraw_folds") == [1, 0, 1]
    assert p.count("postdraw_correct_folds") == [0, 0, 1]
    assert p.count("draws") == [1, 1, 1] and p.count("cards") == [0, 0, 0]


def test_three_players_called_bluff_and_fold_against_the_players_left():
    """The small blind bluffs, the big blind calls, UTG folds: showdown, no
    success. UTG's fold is judged against the two players still in."""
    sb, bb, utg = 1, 2, 0
    hands = {sb: "9s 9h 4d 3c Ks", bb: "Qd Qc 2d 5c Js", utg: "8s 8h Th 6s 7d"}
    p = Play(3, table(3, [hands[i] for i in range(3)], sb=sb))
    p.act(utg, CALL)
    p.act(sb, CALL)
    p.act(bb, CALL)
    for player in (sb, bb, utg):
        p.act(player, KEEP_ALL)
    p.act(sb, HALF_POT)
    p.act(bb, CALL)
    assert p.act(utg, FOLD)  # the betting round is over: showdown
    assert p.last_reward[bb] > 0
    assert p.count("postdraw_bluffs") == [0, 1, 0]
    assert p.count("postdraw_bluff_wins") == [0, 0, 0]
    assert p.count("postdraw_correct_folds") == [1, 0, 0]


def test_carry_resets_between_hands():
    """A snows and loses at showdown; in the next hand (the env's auto-reset
    state) B folds before the draw: no stale snow win, and nothing before the
    draw is counted."""
    p = Play(2, table(2, [SNOWER, PAIR], stock="Jc 6d Tc", sb=A))
    p.act(A, CALL)
    p.act(B, CALL)
    p.act(B, keep(cards(PAIR), cards("8s 8h")))
    p.act(A, KEEP_ALL)
    assert np.asarray(p.carry.snowed).tolist() == [True, False]
    p.act(B, CALL)
    assert p.act(A, CALL)  # showdown
    assert p.carry_is_reset()
    before = {k: v.copy() for k, v in p.totals.items()}
    first = int(p.state.current_player_idx)
    assert p.act(first, FOLD)  # the small blind folds: the big blind wins
    assert p.last_reward[1 - first] > 0
    for k in before:
        np.testing.assert_array_equal(p.totals[k], before[k], err_msg=k)
    assert p.count("snow_wins") == [0, 0]


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
        # env.step runs step_env with the first half of its key
        raw = jax.vmap(lambda k, s, a: env.step_env(jax.random.split(k)[0], s, a)[0])(
            keys, state, action
        )
        carry, counts = jax.vmap(gm.step)(carry, state, action, nxt, reward, done, info)
        return (nxt, carry), (state, action, raw, done, counts)

    run = jax.jit(lambda s, c, k: jax.lax.scan(one, (s, c), jax.random.split(k, steps)))
    _, out = run(state, carry, jax.random.PRNGKey(seed + 1))
    return jax.tree_util.tree_map(np.asarray, out)


def reference_counts(n: int, states, actions, raws, dones) -> dict[str, np.ndarray]:
    """Every count of the metric recomputed per step from the true states,
    in plain Python with an independent hand evaluator."""
    T, N = dones.shape
    names = ("draws", "cards", "draws_known", "draws_improved", "high_card_draws",
             "snows", "snow_wins", "postdraw_aggressive", "postdraw_bluffs",
             "postdraw_bluff_wins", "postdraw_folds", "postdraw_correct_folds")
    out = {k: np.zeros((T, N, n), np.float32) for k in names}
    key = functools.lru_cache(maxsize=None)(lambda hand: five_card_key(list(hand)))
    for e in range(N):
        snowed, aggressor, bluffed = set(), None, False
        for t in range(T):
            hands = [tuple(sorted(int(x) for x in h)) for h in states.agent_cards[t, e]]
            folded = states.folded[t, e]
            p, a, stage = int(states.current_player_idx[t, e]), int(actions[t, e]), int(states.stage[t, e])
            if stage == 1 and a >= 5 and not folded[p]:
                kept = [(a - 5) >> j & 1 for j in range(5)]
                drawn = 5 - sum(kept)
                after = tuple(sorted(int(x) for x in raws.agent_cards[t, e][p]))
                assert set(after) >= {h for h, k in zip(hands[p], kept) if k}
                # the env reshuffles discards (with its own key) only when the
                # stock is short; the metric can't see such a hand-ending draw
                known = not (dones[t, e] and int(states.deck_idx[t, e]) + drawn > 52)
                out["draws"][t, e, p] = 1
                out["cards"][t, e, p] = drawn
                out["draws_known"][t, e, p] = known
                out["draws_improved"][t, e, p] = known and key(after)[0] > key(hands[p])[0]
                if key(hands[p])[0] == 1:
                    out["high_card_draws"][t, e, p] = 1
                    if drawn == 0:
                        out["snows"][t, e, p] = 1
                        snowed.add(p)
            if stage == 2:
                loses = any(key(hands[q]) > key(hands[p]) for q in range(n) if q != p and not folded[q])
                if a in (1, 2, 3):
                    out["postdraw_aggressive"][t, e, p] = 1
                    out["postdraw_bluffs"][t, e, p] = loses
                    aggressor, bluffed = p, loses
                elif a == 4:
                    out["postdraw_folds"][t, e, p] = 1
                    out["postdraw_correct_folds"][t, e, p] = loses
                else:
                    aggressor = None
            if dones[t, e]:
                left = [q for q in range(n) if not raws.folded[t, e][q]]
                if len(left) == 1:
                    w = left[0]
                    out["snow_wins"][t, e, w] = w in snowed
                    out["postdraw_bluff_wins"][t, e, w] = aggressor == w and bluffed
                snowed, aggressor, bluffed = set(), None, False
    return out


@pytest.mark.parametrize("n, envs, steps", [(2, 64, 300), (3, 48, 300), (6, 32, 300), (10, 16, 300)])
def test_counts_match_plain_python_replay(n, envs, steps):
    states, actions, raws, dones, counts = random_rollout(n, envs, steps, seed=n)
    ref = reference_counts(n, states, actions, raws, dones)
    for k, v in ref.items():
        np.testing.assert_array_equal(counts[k], v, err_msg=k)
    totals = {k: v.sum() for k, v in ref.items()}
    assert totals["draws_improved"] > 0 and totals["snows"] > 0
    if n <= 3:  # rare with more players (most hands end before the draw)
        assert totals["snow_wins"] > 0 and totals["postdraw_bluff_wins"] > 0
        assert totals["postdraw_correct_folds"] > 0
    if n == 10:  # hand-ending draws after a reshuffle: unknown, left out
        assert totals["draws_known"] < totals["draws"]
    else:
        assert totals["draws_known"] == totals["draws"]


@pytest.mark.parametrize("n", (2, 5, 10))
def test_hand_after_draw_matches_the_env(n):
    """The hand rebuilt from the stock is the env's hand after the draw
    whenever the stock is big enough (always with up to 5 players)."""
    states, actions, raws, dones, _ = random_rollout(n, 32, 200, seed=10 + n)
    env = _setup(n)[0]
    draw = (states.stage == 1) & (actions >= 5)
    t, e = np.nonzero(draw)
    assert len(t) > 100
    pick = lambda tree: jax.tree_util.tree_map(lambda x: x[t, e], tree)  # noqa: E731
    sub = pick(states)
    player = sub.current_player_idx
    keep = ((actions[t, e][:, None] - 5) >> np.arange(5)) & 1 == 1
    rebuilt, exact = jax.vmap(hand_after_draw)(sub, jnp.asarray(player), jnp.asarray(keep))
    rebuilt, exact = np.sort(np.asarray(rebuilt), 1), np.asarray(exact)
    truth = np.sort(raws.agent_cards[t, e, player], 1)
    np.testing.assert_array_equal(rebuilt[exact], truth[exact])
    assert exact.all() == (n <= 5)


def test_uniform_random_play_values():
    """Uniform random play, heads-up. Draws are uniform over the 32 keep
    patterns whatever the hand: cards_drawn = 2.5 and snow_rate = 1/32.
    After the draw the actions don't depend on the cards and the two final
    hands are exchangeable, so a bet is a bluff and a fold correct with
    probability (1 - P(tie)) / 2."""
    env, gm = _setup(2)[:2]
    *_, dones, counts = random_rollout(2, 1024, 400, seed=7)
    rates = gm.summarize(masked_totals(counts, np.ones(dones.shape + (2,), bool)))
    rates = {k: float(v) for k, v in rates.items()}

    def close(value, p, n, slack=0.0):
        return abs(value - p) < 4 * np.sqrt(p * (1 - p) / n) + slack

    assert rates["n_draws"] > 50_000
    assert abs(rates["cards_drawn"] - 2.5) < 4 * np.sqrt(1.25 / rates["n_draws"])
    assert close(rates["snow_rate"], 1 / 32, rates["n_high_card_draws"])
    for name, n in (("postdraw_bluff_rate", "n_postdraw_aggressive"),
                    ("postdraw_correct_fold_rate", "n_postdraw_folds")):
        assert rates[n] > 5_000
        assert close(rates[name], 0.5, rates[n], slack=0.005), (name, rates[name])
    assert rates["n_draws_known"] == rates["n_draws"]
    for name in gm.rate_names:
        assert 0 <= rates[name] <= (5 if name == "cards_drawn" else 1)
