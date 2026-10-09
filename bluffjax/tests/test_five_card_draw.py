"""5-Card Draw: all-in betting, the draw (no duplicate cards with 2-10 players),
seat order, payoffs, legal actions and seat-rotation equivariance."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make

CALL, HALF_POT, POT, ALL_IN, FOLD = 0, 1, 2, 3, 4
DISCARD_ALL, KEEP_ALL = 5, 36
PER_PLAYER = (
    "agent_cards",
    "chips_in",
    "round_raised",
    "remaining_chips",
    "folded",
    "all_in",
    "absorbing",
)
SEAT_INDEX = ("small_blind_idx", "current_player_idx", "draw_start_idx")


@functools.lru_cache(maxsize=None)
def _env(num_agents: int, init_chips: int = 100):
    env = make("five_card_draw", num_agents=num_agents, init_chips=init_chips)
    return env, jax.jit(env.step_env), jax.jit(env.get_avail_actions)


class Hand:
    """Plays one hand step by step from a reset key."""

    def __init__(self, num_agents: int, seed: int = 0, init_chips: int = 100):
        self.env, self._step, self._avail = _env(num_agents, init_chips)
        self.state, _ = self.env.reset(jax.random.PRNGKey(seed))
        self.n = num_agents
        self.sb = int(self.state.small_blind_idx)
        self.t = 0

    def seat(self, offset: int) -> int:
        """Seat `offset` places after the small blind (0 = SB, 1 = BB, 2 = UTG)."""
        return (self.sb + offset) % self.n

    @property
    def player(self) -> int:
        return int(self.state.current_player_idx)

    @property
    def stage(self) -> int:
        return int(self.state.stage)

    def mask(self) -> np.ndarray:
        return np.asarray(self._avail(self.state))

    def act(self, action: int, by: int | None = None):
        if by is not None:
            assert self.player == by, (self.player, by)
        assert self.mask()[action], (action, self.mask())
        self.t += 1
        self.state, _, reward, _, done, _ = self._step(
            jax.random.PRNGKey(1000 + self.t), self.state, jnp.int32(action)
        )
        return np.asarray(reward), bool(done)


def _check_showdown(hand: Hand, reward: np.ndarray, players: list[int]) -> None:
    """Zero-sum payoff, pot split among the best hands of `players`."""
    chips = np.asarray(hand.state.chips_in)
    assert abs(reward.sum()) < 1e-4
    winners = reward + chips > 0
    assert winners.any() and set(np.flatnonzero(winners)) <= set(players)
    np.testing.assert_allclose(reward + chips, winners * chips.sum() / winners.sum())


def test_heads_up_shove_lets_big_blind_respond():
    hand = Hand(2, seed=0)
    sb, bb = hand.seat(0), hand.seat(1)
    assert hand.player == sb  # heads-up the small blind acts first pre-draw
    _, done = hand.act(ALL_IN, by=sb)
    assert not done and hand.stage == 0 and hand.player == bb
    mask = hand.mask()
    assert mask[CALL] and mask[FOLD] and not mask[[HALF_POT, POT, ALL_IN]].any()

    folded = Hand(2, seed=0)
    folded.act(ALL_IN, by=sb)
    reward, done = folded.act(FOLD, by=bb)
    assert done and reward[sb] == 2 and reward[bb] == -2

    _, done = hand.act(CALL, by=bb)
    # Both are all-in but still draw (non-dealer first), then show down.
    assert not done and hand.stage == 1 and hand.player == bb
    _, done = hand.act(DISCARD_ALL, by=bb)
    assert not done and hand.stage == 1 and hand.player == sb
    reward, done = hand.act(KEEP_ALL, by=sb)
    assert done
    np.testing.assert_array_equal(np.asarray(hand.state.chips_in), [100, 100])
    _check_showdown(hand, reward, [sb, bb])


@pytest.mark.parametrize("raise_action", [POT, HALF_POT])
def test_pot_raise_that_puts_raiser_all_in_lets_others_respond(raise_action):
    # Stacks of 4: the small blind's pot raise (3 chips) or the big blind's
    # half-pot raise after a limp (2 chips) is all-in.
    hand = Hand(2, seed=3, init_chips=4)
    sb, bb = hand.seat(0), hand.seat(1)
    if raise_action == POT:
        raiser, other = sb, bb
    else:
        hand.act(CALL, by=sb)
        raiser, other = bb, sb
    _, done = hand.act(raise_action, by=raiser)
    assert bool(hand.state.all_in[raiser])
    assert not done and hand.stage == 0 and hand.player == other
    hand.act(CALL, by=other)
    assert hand.stage == 1  # both all-in: the draw, then showdown
    hand.act(KEEP_ALL, by=bb)
    reward, done = hand.act(KEEP_ALL, by=sb)
    assert done
    np.testing.assert_array_equal(np.asarray(hand.state.chips_in), [4, 4])
    _check_showdown(hand, reward, [sb, bb])


def test_three_way_shove_and_all_in_call_let_big_blind_act():
    hand = Hand(3, seed=1)
    sb, bb, utg = hand.seat(0), hand.seat(1), hand.seat(2)
    hand.act(ALL_IN, by=utg)
    _, done = hand.act(CALL, by=sb)  # all-in call
    assert not done and hand.stage == 0 and hand.player == bb
    _, done = hand.act(FOLD, by=bb)
    # Draw from the first player after the dealer (UTG is the dealer 3-handed).
    assert not done and hand.stage == 1 and hand.player == sb
    hand.act(DISCARD_ALL, by=sb)
    reward, done = hand.act(KEEP_ALL, by=utg)
    assert done and reward[bb] == -2
    _check_showdown(hand, reward, [sb, utg])


def test_all_in_reraise_reopens_the_betting_for_the_raiser():
    hand = Hand(3, seed=2)
    sb, bb, utg = hand.seat(0), hand.seat(1), hand.seat(2)
    hand.act(POT, by=utg)  # not all-in
    hand.act(ALL_IN, by=sb)
    _, done = hand.act(FOLD, by=bb)
    assert not done and hand.stage == 0 and hand.player == utg
    _, done = hand.act(CALL, by=utg)
    assert not done and hand.stage == 1


def test_four_way_all_ins_and_folds():
    hand = Hand(4, seed=5)
    sb, bb, utg, co = (hand.seat(i) for i in range(4))
    hand.act(ALL_IN, by=utg)
    hand.act(CALL, by=co)  # all-in call
    hand.act(FOLD, by=sb)
    _, done = hand.act(CALL, by=bb)
    assert not done and hand.stage == 1
    order = []
    while not done:
        order.append(hand.player)
        reward, done = hand.act(DISCARD_ALL)
    # Dealer is the cut-off (the seat before the SB); the SB folded.
    assert order == [bb, utg, co]
    assert reward[sb] == -1
    _check_showdown(hand, reward, [bb, utg, co])


@pytest.mark.parametrize("num_agents", [2, 3, 6])
def test_draw_and_post_draw_order(num_agents):
    hand = Hand(num_agents, seed=4)
    while hand.stage == 0:
        hand.act(CALL)
    # Heads-up the small blind is the dealer: the big blind draws and bets
    # first. Otherwise the small blind does.
    first = hand.seat(1) if num_agents == 2 else hand.seat(0)
    assert hand.player == first
    order = []
    while hand.stage == 1:
        order.append(hand.player)
        hand.act(KEEP_ALL)
    assert order == [(first + i) % num_agents for i in range(num_agents)]
    assert hand.stage == 2 and hand.player == first
    for i in range(num_agents - 1):
        assert hand.player == (first + i) % num_agents
        _, done = hand.act(CALL)
        assert not done
    _, done = hand.act(CALL)
    assert done


def test_stock_reshuffle_uses_earlier_discards():
    # 6 players: 22 cards left after the deal. Everyone limps and draws 5, so
    # the 5th drawer gets the last 2 stock cards and 3 of the earlier players'
    # discards, and the 6th draws from what is left of those.
    hand = Hand(6, seed=7)
    while hand.stage == 0:
        hand.act(CALL)
    stock = set(np.asarray(hand.state.shuffled_deck)[30:].tolist())
    pile = set()
    for i in range(6):
        p = hand.player
        old = set(np.asarray(hand.state.agent_cards[p]).tolist())
        hand.act(DISCARD_ALL, by=p)
        new = set(np.asarray(hand.state.agent_cards[p]).tolist())
        if len(stock) >= 5:
            assert new <= stock
            stock -= new
            pile |= old
        else:
            # The stock runs out: the pile (not the player's own discards)
            # becomes the new stock.
            assert i == 4 and stock <= new and len(new & pile) == 5 - len(stock)
            stock = pile - new
            pile = old
    assert len(set(np.asarray(hand.state.agent_cards).ravel().tolist())) == 30
    assert stock == set(np.asarray(hand.state.shuffled_deck)[int(hand.state.deck_idx):].tolist())
    assert pile == set(np.flatnonzero(np.asarray(hand.state.discards)).tolist())


def test_ten_players_redraw_own_discards_only_when_needed():
    # 10 players: 2 cards left after the deal and no earlier discards, so the
    # first drawer gets the 2 stock cards and 3 of its own discards back.
    hand = Hand(10, seed=8)
    while hand.stage == 0:
        hand.act(CALL)
    stock = set(np.asarray(hand.state.shuffled_deck)[50:].tolist())
    p = hand.player
    old = set(np.asarray(hand.state.agent_cards[p]).tolist())
    hand.act(DISCARD_ALL, by=p)
    new = set(np.asarray(hand.state.agent_cards[p]).tolist())
    assert stock <= new and len(new & old) == 3
    # The next drawer's 5 cards come from the 2 discards left in the stock
    # and its own.
    q = hand.player
    q_old = set(np.asarray(hand.state.agent_cards[q]).tolist())
    hand.act(DISCARD_ALL, by=q)
    q_new = set(np.asarray(hand.state.agent_cards[q]).tolist())
    assert len(q_new & (old - new)) == 2 and len(q_new & q_old) == 3


@functools.lru_cache(maxsize=None)
def _rollout(num_agents: int, num_envs: int = 128, num_steps: int = 160):
    """Records every step of auto-resetting rollouts. Half the envs play
    uniformly random legal actions, half call and discard everything."""
    env, _, _ = _env(num_agents)

    @jax.jit
    def run(key):
        k0, k1 = jax.random.split(key)
        state, _ = jax.vmap(env.reset)(jax.random.split(k0, num_envs))

        def body(state, k):
            k_act, k_step, k_reset = jax.random.split(k, 3)
            mask = jax.vmap(env.get_avail_actions)(state)
            random_action = jax.random.categorical(
                k_act, jnp.where(mask, 0.0, -jnp.inf), axis=-1
            )
            draw_all = jnp.where(mask[:, DISCARD_ALL], DISCARD_ALL, CALL)
            action = jnp.where(
                jnp.arange(num_envs) < num_envs // 2, random_action, draw_all
            )
            nxt, obs, reward, _, done, _ = jax.vmap(env.step_env)(
                jax.random.split(k_step, num_envs), state, action
            )
            fresh, _ = jax.vmap(env.reset)(jax.random.split(k_reset, num_envs))
            rec = dict(
                state=state,
                mask=mask,
                next=nxt,
                next_mask=jax.vmap(env.get_avail_actions)(nxt),
                reward=reward,
                done=done,
            )
            nxt = jax.tree_util.tree_map(
                lambda f, s: jnp.where(done.reshape((-1,) + (1,) * (s.ndim - 1)), f, s),
                fresh,
                nxt,
            )
            return nxt, rec

        _, rec = jax.lax.scan(body, state, jax.random.split(k1, num_steps))
        return rec

    rec = run(jax.random.PRNGKey(num_agents))
    return jax.tree_util.tree_map(np.asarray, rec)


def _deck_counts(s, num_agents: int) -> np.ndarray:
    """How often each card is in a hand, the stock or the discard pile."""
    cards = s.agent_cards.reshape(-1, 5 * num_agents)
    deck = s.shuffled_deck.reshape(-1, 52)
    rows = np.arange(len(cards))[:, None]
    counts = np.zeros((len(cards), 52), dtype=int)
    np.add.at(counts, (rows, cards), 1)
    in_stock = np.arange(52)[None] >= s.deck_idx.reshape(-1, 1)
    np.add.at(counts, (rows, deck), in_stock.astype(int))
    return counts + s.discards.reshape(-1, 52)


@pytest.mark.parametrize("num_agents", range(2, 11))
def test_random_hands(num_agents):
    rec = _rollout(num_agents)
    n = num_agents
    # Hands, stock and discard pile hold every card exactly once.
    assert (_deck_counts(rec["state"], n) == 1).all()
    assert (_deck_counts(rec["next"], n) == 1).all()

    s, mask, done, reward = rec["state"], rec["mask"], rec["done"], rec["reward"]
    T, B = done.shape
    cur = s.current_player_idx
    folded = np.take_along_axis(s.folded, cur[..., None], -1)[..., 0]
    all_in = np.take_along_axis(s.all_in, cur[..., None], -1)[..., 0]
    betting = s.stage != 1
    assert mask.any(-1).all()  # the acting player always has a legal action
    assert not folded.any() and not (betting & all_in).any()
    assert (mask[..., :5].any(-1) == betting).all()
    assert (mask[..., 5:].all(-1) == ~betting).all()
    assert not rec["next_mask"][done].any()  # nothing is legal once it's over
    np.testing.assert_allclose(reward.sum(-1), 0.0, atol=1e-3)  # zero-sum
    assert (reward[~done] == 0).all()

    # Hand ends: everyone still in has put in the same (all-in players too,
    # as stacks are equal), and the pot goes to the best hands among them.
    ends = done.reshape(-1)
    last = jax.tree_util.tree_map(lambda x: x.reshape((T * B,) + x.shape[2:])[ends], rec["next"])
    assert ends.sum() > B
    chips, live = last.chips_in, ~last.folded
    r = reward.reshape(T * B, n)[ends]
    top = np.where(live, chips, -1).max(-1, keepdims=True)
    assert np.where(live, chips == top, True).all()
    winners = r + chips > 1e-4
    assert winners.any(-1).all() and not (winners & ~live).any()
    share = chips.sum(-1, keepdims=True) / winners.sum(-1, keepdims=True)
    np.testing.assert_allclose(r + chips, winners * share, atol=1e-3)
    # Showdowns come after the draw, also when everyone left went all-in
    # before it (those end in stage 1, right after the draw).
    showdown = live.sum(-1) >= 2
    assert (last.stage[showdown] != 0).all()
    all_in_showdown = showdown & (last.all_in | ~live).all(-1)
    assert (last.stage[all_in_showdown] == 1).any()


def test_random_start_player():
    env, _, _ = _env(4)
    state, _ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 4000))
    counts = np.bincount(np.asarray(state.current_player_idx), minlength=4)
    assert (np.abs(counts - 1000) < 120).all(), counts


def _rotate(state, k: int, n: int):
    roll = {f: jnp.roll(getattr(state, f), k, axis=0) for f in PER_PLAYER}
    shift = {f: (getattr(state, f) + k) % n for f in SEAT_INDEX}
    return state.replace(**roll, **shift)


@pytest.mark.parametrize("num_agents", [2, 3, 6])
def test_seat_rotation_equivariance(num_agents):
    """Rotating every seat by k and stepping with the same action gives the
    rotated next state and rewards and the same observation and legal actions."""
    n = num_agents
    env, _, _ = _env(n)
    rec = _rollout(n)
    flat = jax.tree_util.tree_map(
        lambda x: x.reshape((-1,) + x.shape[2:]), rec["state"]
    )
    idx = np.random.default_rng(0).choice(len(flat.stage), 2000, replace=False)
    states = jax.tree_util.tree_map(lambda x: jnp.asarray(x[idx]), flat)

    def check(s, key):
        k_act, k_step = jax.random.split(key)
        mask = env.get_avail_actions(s)
        a = jax.random.categorical(k_act, jnp.where(mask, 0.0, -jnp.inf))
        s1, o1, r1, _, d1, _ = env.step_env(k_step, s, a)
        ok = []
        for k in range(1, n):
            sk = _rotate(s, k, n)
            same_view = (env.obs_from_state(sk) == env.obs_from_state(s)).all() & (
                env.get_avail_actions(sk) == mask
            ).all()
            s2, o2, r2, _, d2, _ = env.step_env(k_step, sk, a)
            want = _rotate(s1, k, n)
            same_next = jnp.stack(
                [
                    (x == y).all()
                    for x, y in zip(
                        jax.tree_util.tree_leaves(s2), jax.tree_util.tree_leaves(want)
                    )
                ]
            ).all()
            ok.append(
                same_view
                & same_next
                & (o2 == o1).all()
                & jnp.allclose(r2, jnp.roll(r1, k))
                & (d2 == d1)
            )
        return jnp.stack(ok)

    keys = jax.random.split(jax.random.PRNGKey(99), len(idx))
    ok = np.asarray(jax.jit(jax.vmap(check))(states, keys))
    assert ok.all(), f"{(~ok).any(1).mean():.4f} of states not equivariant"


def test_invalid_configs_raise():
    with pytest.raises(ValueError):
        make("five_card_draw", num_agents=11)
    with pytest.raises(ValueError):
        make("five_card_draw", num_agents=1)
    with pytest.raises(ValueError):
        make("five_card_draw", init_chips=2)
