"""7-Card Stud: rules, observation and seat neutrality.

RefStud below is a plain-Python model of the game (deal, order of play,
betting, payoffs and observation). Random rollouts of the env are replayed
through it and compared at every step, for 2 to 10 players.
"""

import itertools
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make

PLAYER_COUNTS = (2, 3, 7, 8, 9, 10)
SLOT_STREET = (0, 0, 0, 1, 2, 3, 4)
SLOT_FACE_UP = (False, False, True, True, True, True, False)


def card(rank: int, suit: int) -> int:
    """Card index for rank 2-14 (14 = ace) and suit 0-3 (the env's encoding)."""
    return suit * 13 + (0 if rank == 14 else rank - 1)


def rank_of(c: int) -> int:
    return 14 if c % 13 == 0 else c % 13 + 1


def _groups(cards):
    """(rank, count) sorted by count, then rank, both descending."""
    counts = Counter(rank_of(c) for c in cards)
    return sorted(counts.items(), key=lambda rc: (rc[1], rc[0]), reverse=True)


def board_key(cards):
    """Strength of 2-4 face-up cards: quads, trips, two pair, pair, high card."""
    groups = _groups(cards)
    shape = [n for _, n in groups]
    if shape[0] == 4:
        category = 5
    elif shape[0] == 3:
        category = 4
    elif shape[:2] == [2, 2]:
        category = 3
    elif shape[0] == 2:
        category = 2
    else:
        category = 1
    return (category, *[r for r, _ in groups])


def five_card_key(cards):
    groups = _groups(cards)
    shape = [n for _, n in groups]
    order = [r for r, _ in groups]
    flush = len({c // 13 for c in cards}) == 1
    straight = None
    if len(order) == 5:
        if order[0] - order[4] == 4:
            straight = order[0]
        elif order == [14, 5, 4, 3, 2]:
            straight = 5
    if straight and flush:
        return (9, straight)
    if shape[0] == 4:
        return (8, *order)
    if shape[:2] == [3, 2]:
        return (7, *order)
    if flush:
        return (6, *order)
    if straight:
        return (5, straight)
    if shape[0] == 3:
        return (4, *order)
    if shape[:2] == [2, 2]:
        return (3, *order)
    if shape[0] == 2:
        return (2, *order)
    return (1, *order)


def best_hand(cards):
    return max(five_card_key(c) for c in itertools.combinations(cards, 5))


class RefStud:
    """Plain-Python 7-Card Stud, written from the rules in the env docstring."""

    def __init__(self, n: int, deck):
        self.n = n
        self.deck = [int(c) for c in deck]
        self.cards = [[-1] * 7 for _ in range(n)]
        for p in range(n):
            self.cards[p][:3] = self.deck[3 * p : 3 * p + 3]
        self.num_dealt = 3 * n
        self.community = [False] * 5
        self.folded = [False] * n
        door = [self.cards[p][2] for p in range(n)]
        self.bring_in = min(range(n), key=lambda p: (rank_of(door[p]), door[p] // 13))
        self.chips = [0.5] * n
        self.chips[self.bring_in] += 0.5
        self.stage = 0
        self.raises = [0] * 5
        self.not_raised = 0
        self.done = False
        self.current = self.next_active(self.bring_in)

    def after(self, p):
        return [(p + 1 + i) % self.n for i in range(self.n)]

    def active(self, order=None):
        return [p for p in (order or range(self.n)) if not self.folded[p]]

    def next_active(self, p):
        return self.active(self.after(p))[0]

    def face_up(self, slot):
        return SLOT_FACE_UP[slot] or self.community[SLOT_STREET[slot]]

    def legal(self):
        c, top = self.current, max(self.chips)
        return [self.chips[c] < top, self.raises[self.stage] < 4, True, self.chips[c] == top]

    def deal(self, street):
        players = self.active(self.after(self.bring_in))
        if 52 - self.num_dealt - len(players) < 4 - street:
            self.community[street] = True
            for p in players:
                self.cards[p][street + 2] = self.deck[self.num_dealt]
            self.num_dealt += 1
        else:
            for p in players:
                self.cards[p][street + 2] = self.deck[self.num_dealt]
                self.num_dealt += 1

    def first_to_act(self):
        num_up = min(self.stage + 1, 4) + (self.stage == 4 and self.community[4])
        key = board_key if num_up <= 4 else five_card_key
        best = None
        for p in self.active(self.after(self.bring_in)):  # ties: first in this order
            if best is None or key(self.cards[p][2 : 2 + num_up]) > key(
                self.cards[best][2 : 2 + num_up]
            ):
                best = p
        return best

    def step(self, action):
        c, top = self.current, max(self.chips)
        if action == 0:
            self.chips[c] = top
            self.not_raised += 1
        elif action == 1:
            self.chips[c] = top + (2 if self.stage >= 2 else 1)
            self.raises[self.stage] += 1
            self.not_raised = 1
        elif action == 2:
            self.folded[c] = True
        else:
            self.not_raised += 1
        active = self.active()
        round_over = self.not_raised >= len(active)
        if round_over:
            self.stage += 1
            self.not_raised = 0
        self.done = len(active) <= 1 or self.stage >= 5
        if self.done:
            if len(active) == 1:
                winners = active
            else:
                hands = {p: best_hand(self.cards[p]) for p in active}
                winners = [p for p in active if hands[p] == max(hands.values())]
            pot = sum(self.chips)
            return [
                ((pot / len(winners) if p in winners else 0.0) - self.chips[p]) / 2
                for p in range(self.n)
            ]
        if round_over:
            self.deal(self.stage)
            self.current = self.first_to_act()
        else:
            self.current = self.next_active(c)
        return [0.0] * self.n

    def obs(self, viewer=None):
        v = self.current if viewer is None else viewer
        n = self.n
        obs = np.zeros(76 + 262 * n, dtype=np.float32)
        for slot, c in enumerate(self.cards[v]):
            if c >= 0 and not self.face_up(slot):
                obs[c] = 1
        for j in range(n):
            for t in range(5):
                c = self.cards[(v + j) % n][t + 2]
                if c >= 0 and self.face_up(t + 2):
                    obs[52 + (5 * j + t) * 52 + c] = 1
        base = 52 + 260 * n
        for j in range(1, n):
            obs[base + j - 1] = self.folded[(v + j) % n]
        base += n - 1
        for s in range(5):
            obs[base + 5 * s + self.raises[s]] = 1
        obs[base + 25 + (v - self.bring_in) % n] = 1
        return obs


def decode_obs(obs, n):
    """Own face-down cards, face-up cards by seat and street, folded flags."""
    obs = np.asarray(obs)
    down = set(np.flatnonzero(obs[:52]))
    up = obs[52 : 52 + 260 * n].reshape(n, 5, 52)
    up = [[set(np.flatnonzero(up[j, t])) for t in range(5)] for j in range(n)]
    folded = obs[52 + 260 * n : 52 + 260 * n + n - 1]
    return down, up, folded


def _rollout(env, batch, steps, seed):
    """Random rollouts of single hands (frozen after the end); a third of the
    envs never fold, a third rarely fold and a third play uniformly."""
    n_envs = batch
    fold_weight = jnp.array([0.0, 0.1, 1.0])[jnp.arange(n_envs) % 3]
    state, _ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(seed), n_envs))

    def one(carry, key):
        state, ended = carry
        avail = jax.vmap(env.get_avail_actions)(state)
        obs = jax.vmap(env.obs_from_state)(state)
        weights = jnp.where(avail, 1.0, 0.0).at[:, 2].multiply(fold_weight)
        k_action, k_step = jax.random.split(key)
        action = jax.random.categorical(
            k_action, jnp.where(weights > 0, jnp.log(weights), -jnp.inf)
        )
        nxt, _, reward, _, done, _ = jax.vmap(env.step_env)(
            jax.random.split(k_step, n_envs), state, action
        )
        keep = lambda new, old: jnp.where(
            ended.reshape((n_envs,) + (1,) * (new.ndim - 1)), old, new
        )
        nxt = jax.tree_util.tree_map(keep, nxt, state)
        record = (state, obs, avail, action, reward, done, ended)
        return (nxt, ended | done), record

    keys = jax.random.split(jax.random.PRNGKey(seed + 1), steps)
    _, record = jax.lax.scan(jax.jit(one), (state, jnp.zeros(n_envs, bool)), keys)
    return jax.tree_util.tree_map(np.asarray, record)


@pytest.fixture(scope="module")
def rollouts():
    """Per player count: (env, rollout record). 24 hands each."""
    out = {}
    for n in PLAYER_COUNTS:
        env = make("seven_card_stud", num_agents=n)
        out[n] = (env, _rollout(env, 24, 40 + 30 * n, seed=n))
    return out


def _live(ended):
    """(t, b) of the decision points: steps before the hand ended."""
    return list(zip(*np.nonzero(~ended)))


def _select(states, points):
    """The states at a list of (t, b)."""
    index = tuple(np.array(points).T)
    return jax.tree_util.tree_map(lambda x: x[index], states)


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_env_matches_reference(rollouts, n):
    """Deal, community cards, order of play, legal actions, observation and
    payoffs equal RefStud at every step of every hand."""
    env, (states, obs, avail, action, reward, done, ended) = rollouts[n]
    assert obs.shape[-1] == env.obs_dim == env.observation_space().n == 76 + 262 * n
    steps, batch = action.shape
    finished = 0
    community_streets = Counter()
    for b in range(batch):
        ref = RefStud(n, states.deck[0, b])
        for t in range(steps):
            assert not ended[t, b]
            assert int(states.current_player_idx[t, b]) == ref.current
            assert int(states.stage[t, b]) == ref.stage
            np.testing.assert_array_equal(states.agent_cards[t, b], ref.cards)
            np.testing.assert_array_equal(states.community[t, b], ref.community)
            assert int(states.num_dealt[t, b]) == ref.num_dealt
            np.testing.assert_array_equal(avail[t, b], ref.legal())
            np.testing.assert_array_equal(obs[t, b], ref.obs())
            ref_reward = ref.step(int(action[t, b]))
            np.testing.assert_allclose(reward[t, b], ref_reward, atol=1e-5)
            assert bool(done[t, b]) == ref.done
            if ref.done:
                finished += 1
                community_streets.update(s for s in range(5) if ref.community[s])
                break
    assert finished >= batch - 2  # (nearly) every hand was played to the end
    # The deck runs short only with 8 or more players (on 7th street) or 9 or
    # more (also on 6th street); the never-fold hands reach those streets.
    if n <= 7:
        assert not community_streets
    else:
        assert community_streets[4] > 0 and set(community_streets) <= ({4} if n == 8 else {3, 4})
        if n >= 9:
            assert community_streets[3] > 0


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_cards_are_dealt_once_and_hands_are_complete(rollouts, n):
    """No card is held twice except a community card, which every player
    still in the hand holds in the same slot; players at a showdown hold 7."""
    env, (states, _, _, _, _, done, ended) = rollouts[n]
    for t, b in _live(ended):
        cards = states.agent_cards[t, b]
        folded = states.folded[t, b]
        dealt = cards[cards >= 0]
        deck = states.deck[t, b]
        assert sorted(set(dealt.tolist())) == sorted(deck[: states.num_dealt[t, b]])
        for slot in range(7):
            column = cards[:, slot]
            column = column[column >= 0]
            if states.community[t, b][SLOT_STREET[slot]] and slot >= 3:
                assert len(set(column.tolist())) == 1
            else:
                assert len(set(column.tolist())) == len(column)
        others = [c for slot in range(7) for c in cards[:, slot] if c >= 0
                  and not (slot >= 3 and states.community[t, b][SLOT_STREET[slot]])]
        assert len(others) == len(set(others))
        # every player still in has exactly the cards of the streets so far
        stage = int(states.stage[t, b])
        np.testing.assert_array_equal((cards[~folded] >= 0).sum(1), 3 + stage)


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_rewards_zero_sum_and_masks_non_empty(rollouts, n):
    env, (states, _, avail, _, reward, done, ended) = rollouts[n]
    live = ~ended
    assert np.abs(reward[live].sum(-1)).max() < 1e-5
    assert np.all(reward[live & ~done] == 0)
    assert np.all(avail[live].any(-1))
    current = states.current_player_idx[live]
    assert not np.any(states.folded[live][np.arange(len(current)), current])


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_observation_shows_only_dealt_cards(rollouts, n):
    """At every street a player sees its own dealt cards (3 on 3rd street ...
    7 on 7th), face down or up, and of the others only cards dealt face up."""
    env, (states, obs, _, _, _, done, ended) = rollouts[n]
    for t, b in _live(ended):
        cards = states.agent_cards[t, b]
        current = int(states.current_player_idx[t, b])
        stage = int(states.stage[t, b])
        undealt = set(states.deck[t, b][states.num_dealt[t, b] :].tolist())
        down, up, _ = decode_obs(obs[t, b], n)
        own_up = set().union(*up[0])
        assert len(down) + len(own_up) == 3 + stage
        assert down | own_up == set(cards[current][cards[current] >= 0].tolist())
        assert not (down | set().union(*[set().union(*u) for u in up])) & undealt
        assert len(down) == 2 + (stage == 4 and not states.community[t, b][4])


@pytest.mark.parametrize("n", (3, 8, 10))
def test_face_up_cards_are_what_others_see(rollouts, n):
    """What a player's observation marks as its own face-up cards is exactly
    what every other player sees of it; its face-down cards are seen by no one."""
    env, (states, _, _, _, _, done, ended) = rollouts[n]
    obs_as = jax.jit(
        jax.vmap(
            lambda s: jax.vmap(
                lambda v: env.obs_from_state(s.replace(current_player_idx=v))
            )(jnp.arange(n))
        )
    )
    live = _live(ended)[::5]
    sample = _select(states, live)
    views = np.asarray(obs_as(sample))  # (states, viewer, obs)
    for i in range(len(live)):
        decoded = [decode_obs(views[i, v], n) for v in range(n)]
        for p in range(n):
            down_p, up_p, _ = decoded[p]
            for v in range(n):
                down_v, up_v, folded_v = decoded[v]
                assert up_v[(p - v) % n] == up_p[0]
                if v != p:
                    seen = down_v | set().union(*[set().union(*u) for u in up_v])
                    assert not seen & down_p
                    assert folded_v[(p - v) % n - 1] == sample.folded[i, p]


@pytest.mark.parametrize("n", (2, 8, 10))
def test_observation_ignores_hidden_cards(rollouts, n):
    """Shuffling the undealt deck and the other players' face-down cards
    (everything the player to act can't see) doesn't change its observation."""
    env, (states, obs, _, _, _, done, ended) = rollouts[n]
    rng = np.random.default_rng(n)
    live = _live(ended)[::3]
    deck = states.deck[tuple(np.array(live).T)].copy()
    agent_cards = states.agent_cards[tuple(np.array(live).T)].copy()
    for i, (t, b) in enumerate(live):
        current = int(states.current_player_idx[t, b])
        face_up = [SLOT_FACE_UP[s] or states.community[t, b][SLOT_STREET[s]] for s in range(7)]
        hidden = [(p, s) for p in range(n) for s in range(7)
                  if p != current and agent_cards[i, p, s] >= 0 and not face_up[s]]
        num_dealt = int(states.num_dealt[t, b])
        pool = [agent_cards[i, p, s] for p, s in hidden] + deck[i, num_dealt:].tolist()
        pool = rng.permutation(pool)
        for (p, s), c in zip(hidden, pool[: len(hidden)]):
            agent_cards[i, p, s] = c
        deck[i, num_dealt:] = pool[len(hidden) :]
    sample = _select(states, live)
    shuffled = sample.replace(deck=jnp.asarray(deck), agent_cards=jnp.asarray(agent_cards))
    assert not np.array_equal(agent_cards, np.asarray(sample.agent_cards))
    np.testing.assert_array_equal(
        np.asarray(jax.vmap(env.obs_from_state)(shuffled)),
        obs[tuple(np.array(live).T)],
    )


def _rotate(state, k, n):
    """The same table with every player moved k seats clockwise."""
    roll = lambda x: jnp.roll(x, k, axis=0)
    return state.replace(
        agent_cards=roll(state.agent_cards),
        chips_in=roll(state.chips_in),
        folded=roll(state.folded),
        absorbing=roll(state.absorbing),
        bring_in_idx=(state.bring_in_idx + k) % n,
        current_player_idx=(state.current_player_idx + k) % n,
    )


@pytest.mark.parametrize("n", (2, 3, 7, 8, 10))
def test_seat_rotation_equivariance(rollouts, n):
    """Rotating every per-player field by k seats: the player to act sees the
    same observation and legal actions, and after the same action the next
    state (who acts next, the cards dealt, ...) and the rewards are rotated by
    k. So nothing depends on a player's absolute index."""
    env, (states, obs, avail, action, _, done, ended) = rollouts[n]
    live = _live(ended)
    sample = _select(states, live)

    def check(state, a):
        nxt, next_obs, reward, _, d, _ = env.step_env(jax.random.PRNGKey(0), state, a)
        oks = []
        for k in range(1, n):
            rot = _rotate(state, k, n)
            rot_next, rot_next_obs, rot_reward, _, rot_d, _ = env.step_env(
                jax.random.PRNGKey(0), rot, a
            )
            want = _rotate(nxt, k, n)
            same_state = jax.tree_util.tree_map(
                lambda x, y: jnp.all(x == y), rot_next, want
            )
            oks.append(
                jnp.stack(
                    [
                        jnp.all(env.obs_from_state(rot) == env.obs_from_state(state)),
                        jnp.all(env.get_avail_actions(rot) == env.get_avail_actions(state)),
                        jnp.allclose(rot_reward, jnp.roll(reward, k), atol=1e-6),
                        rot_d == d,
                        d | jnp.all(jnp.stack(jax.tree_util.tree_leaves(same_state))),
                        d | jnp.all(rot_next_obs == next_obs),
                    ]
                )
            )
        return jnp.stack(oks)

    ok = np.asarray(jax.jit(jax.vmap(check))(sample, jnp.asarray(action[tuple(np.array(live).T)])))
    assert ok.all(), ok.reshape(-1, 6).mean(0)


def _table(env, hands):
    """Start of a hand in which, if nobody folds, player p is dealt hands[p]
    (7 cards by slot: hole, hole, door, 4th, 5th, 6th, 7th street)."""
    n = env.num_agents
    from bluffjax.utils.game_utils.poker_utils import _get_bring_in_idx

    bring_in = int(_get_bring_in_idx(jnp.array([h[2] for h in hands])))
    order = [(bring_in + 1 + i) % n for i in range(n)]
    deck = [hands[p][s] for p in range(n) for s in range(3)]
    for street in range(1, 5):
        deck += [hands[p][street + 2] for p in order if hands[p][street + 2] >= 0]
    deck += [c for c in range(52) if c not in deck]
    assert sorted(deck) == list(range(52))
    state, _ = env.reset(jax.random.PRNGKey(0))
    deck = jnp.array(deck, dtype=jnp.int32)
    agent_cards = jnp.full((n, 7), -1, jnp.int32).at[:, :3].set(deck[: 3 * n].reshape(n, 3))
    return state.replace(
        deck=deck,
        agent_cards=agent_cards,
        bring_in_idx=jnp.int32(bring_in),
        chips_in=jnp.full(n, 0.5).at[bring_in].add(0.5),
        current_player_idx=jnp.int32((bring_in + 1) % n),
    )


def _first_to_act(env, state, street):
    """Everyone calls or checks until street; returns who acts first there."""
    step = jax.jit(env.step_env)
    while int(state.stage) < street:
        avail = np.asarray(env.get_avail_actions(state))
        state, *_ = step(jax.random.PRNGKey(0), state, jnp.int32(0 if avail[0] else 3))
    return state


def _filler(used, count, rng):
    free = [c for c in range(52) if c not in used]
    return [int(c) for c in rng.choice(free, count, replace=False)]


TIE_CASES = {
    # 4th street: seats 1 and 3 both show K-7 (seat 0 brings in with the 2c)
    "4th street, two tied": (
        4, 1, 1,
        {0: (card(2, 0), card(3, 1)), 1: (card(13, 3), card(7, 2)),
         2: (card(9, 0), card(5, 1)), 3: (card(13, 2), card(7, 3))},
    ),
    # the bring-in (2c) is tied with seat 2 on a pair of twos: it comes last
    "4th street, bring-in tied": (
        4, 1, 2,
        {0: (card(2, 0), card(2, 1)), 1: (card(5, 1), card(9, 2)),
         2: (card(2, 2), card(2, 3)), 3: (card(6, 3), card(11, 2))},
    ),
}


@pytest.mark.parametrize("name", list(TIE_CASES))
def test_tied_boards_go_first_after_the_bring_in(name):
    """With tied best boards, the tied player first clockwise after the
    bring-in acts first, wherever the table sits (rotated over all seats)."""
    n, street, first, upcards = TIE_CASES[name]
    env = make("seven_card_stud", num_agents=n)
    rng = np.random.default_rng(0)
    base = {}
    used = {c for cards in upcards.values() for c in cards}
    for p, (door, fourth) in upcards.items():
        hole = _filler(used, 2, rng)
        used |= set(hole)
        base[p] = [*hole, door, fourth]
    for p in range(n):  # later streets: anything
        later = _filler(used, 3, rng)
        used |= set(later)
        base[p] += later
    for k in range(n):
        hands = [base[(p - k) % n] for p in range(n)]  # seat p + k gets base[p]
        state = _first_to_act(env, _table(env, hands), street)
        assert int(state.bring_in_idx) == k
        assert int(state.current_player_idx) == (first + k) % n


def test_seventh_street_community_card_board():
    """8 players, nobody folds: 7th street is one face-up community card and
    the best five-card board (here a straight made by the community card)
    acts first; rotated over all seats."""
    n = 8
    env = make("seven_card_stud", num_agents=n)
    rng = np.random.default_rng(1)
    pair_k = [card(13, 0), card(13, 1), card(2, 1), card(3, 2)]  # best on 6th
    wheel = [card(4, 0), card(5, 1), card(6, 2), card(7, 3)]  # 4-8 straight on 7th
    community = card(8, 0)
    for _ in range(200):  # random other boards with no pair and no 4-8 draw
        used = set(pair_k + wheel + [community])
        boards = {1: pair_k, 4: wheel}
        for p in (0, 2, 3, 5, 6, 7):
            boards[p] = _filler(used, 4, rng)
            used |= set(boards[p])
        hands = {}
        for p in range(n):
            hole = _filler(used, 2, rng)
            used |= set(hole)
            hands[p] = [*hole, *boards[p], -1]
        keys6 = {p: board_key(hands[p][2:6]) for p in range(n)}
        keys7 = {p: five_card_key(hands[p][2:6] + [community]) for p in range(n)}
        if (max(keys6, key=keys6.get) == 1 and sorted(keys6.values())[-2] < keys6[1]
                and max(keys7, key=keys7.get) == 4
                and sorted(keys7.values())[-2] < keys7[4]):
            break
    else:
        raise AssertionError("no deal found")
    for k in (0, 3, 6):
        rotated = [hands[(p - k) % n] for p in range(n)]
        state = _table(env, rotated)
        deck = list(np.asarray(state.deck))
        deck.remove(community)
        deck.insert(48, community)  # the 7th-street card
        state = state.replace(deck=jnp.array(deck, dtype=jnp.int32))
        sixth = _first_to_act(env, state, 3)
        assert int(sixth.current_player_idx) == (1 + k) % n
        seventh = _first_to_act(env, sixth, 4)
        assert bool(seventh.community[4]) and not bool(seventh.community[3])
        np.testing.assert_array_equal(np.asarray(seventh.agent_cards[:, 6]), community)
        assert int(seventh.current_player_idx) == (4 + k) % n


def test_tie_breaks_are_seat_neutral():
    """Over random deals in which everyone sees 4th street, tied best boards
    go to every seat equally often (the old rule gave them to index 0)."""
    n, deals = 4, 60000
    env = make("seven_card_stud", num_agents=n)
    states, _ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(7), deals))

    def call_round(state):
        def body(state, _):
            avail = env.get_avail_actions(state)
            a = jnp.where(avail[0], 0, 3)
            return env.step_env(jax.random.PRNGKey(0), state, a)[0], None
        return jax.lax.scan(body, state, None, n)[0]

    states = jax.jit(jax.vmap(call_round))(states)
    assert np.all(np.asarray(states.stage) == 1)
    boards = np.asarray(states.agent_cards[:, :, 2:4])
    first = np.asarray(states.current_player_idx)
    bring_in = np.asarray(states.bring_in_idx)
    tie_seats = []
    for i in range(deals):
        keys = [board_key(boards[i, p]) for p in range(n)]
        best = max(keys)
        tied = [p for p in range(n) if keys[p] == best]
        if len(tied) > 1:
            order = [(bring_in[i] + 1 + j) % n for j in range(n)]
            assert first[i] == next(p for p in order if p in tied)
            tie_seats.append(first[i])
    share = np.bincount(tie_seats, minlength=n) / len(tie_seats)
    assert len(tie_seats) > 500
    assert np.all(np.abs(share - 1 / n) < 0.05), share
    seats = np.bincount(bring_in, minlength=n) / deals
    assert np.all(np.abs(seats - 1 / n) < 0.01), seats


def test_player_count_limits():
    for n in (1, 17):
        with pytest.raises(ValueError):
            make("seven_card_stud", num_agents=n)
    env = make("seven_card_stud", num_agents=16)
    state, obs = env.reset(jax.random.PRNGKey(0))
    assert obs.shape == (env.obs_dim,)
