"""Kemps: seat-neutral observation, teams of two, seat-neutral declarations,
card conservation and zero-sum team rewards."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax.environments.kemps.kemps import Kemps

R = 13  # ranks
S = 4  # suits
NOOP = R * R
KEMPS = R * R + 1


def STOP(d):
    """STOP KEMPS against the team of the player d seats after the caller."""
    return R * R + 1 + d


def act(env, game_actions, signal=0):
    return jnp.array(game_actions, dtype=jnp.int32) * env.comm_dim + signal


def deal(env, fours=None):
    """Hands with no two cards of a rank, except that each seat in `fours`
    ({seat: rank}) holds four of a kind. Returns hands, centre, stock."""
    fours = fours or {}
    pool = [r * S + s for s in range(S) for r in range(R) if r not in fours.values()]
    hands = []
    for seat in range(env.num_agents):
        if seat in fours:
            hands.append([fours[seat] * S + s for s in range(S)])
        else:
            hands.append(pool[:4])
            pool = pool[4:]
    return hands, pool[:4], pool[4:]


def make_state(env, hands, center, stock, deck_idx=0):
    state, _ = env.reset(jax.random.PRNGKey(0))
    hands = jnp.array(hands, dtype=jnp.int32)
    center = jnp.array(center, dtype=jnp.int32)
    count = lambda cards: jnp.bincount(cards // S, length=R)
    return state.replace(
        agent_hands=hands,
        agent_hand_counts=jax.vmap(count)(hands),
        center_cards=center,
        center_counts=count(center),
        deck=jnp.array(stock, dtype=jnp.int32),
        deck_idx=jnp.int32(deck_idx),
    )


def rotate(state, k):
    """Move every player k seats on."""
    roll = lambda x: jnp.roll(x, k, axis=0)
    return state.replace(
        agent_hands=roll(state.agent_hands),
        agent_hand_counts=roll(state.agent_hand_counts),
        communication=roll(state.communication),
        absorbing=roll(state.absorbing),
    )


def sample_actions(env, key, state, p_declare, p_noop):
    """Each player declares with p_declare (uniform over KEMPS and the STOPs),
    else plays NOOP with p_noop, else a uniform legal swap; random signal."""
    n = env.num_agents
    k_swap, k_decl, k_u, k_sig = jax.random.split(key, 4)
    swap_avail = env.get_avail_actions(state)[:, :: env.comm_dim][:, :NOOP]
    swap = jax.random.categorical(k_swap, jnp.where(swap_avail, 0.0, -jnp.inf))
    declare = KEMPS + jax.random.randint(k_decl, (n,), 0, n // 2)
    u = jax.random.uniform(k_u, (n,))
    game = jnp.where(u < p_declare, declare, jnp.where(u < p_declare + p_noop, NOOP, swap))
    return game * env.comm_dim + jax.random.randint(k_sig, (n,), 0, env.comm_dim)


def rollout(env, seed, num_envs, num_steps, p_declare, p_noop):
    """Auto-resetting rollouts; returns the states before each step, actions and outputs."""
    keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    state, _ = jax.vmap(env.reset)(keys)

    def one(state, key):
        k_act, k_step = jax.random.split(key)
        action = jax.vmap(lambda k, s: sample_actions(env, k, s, p_declare, p_noop))(
            jax.random.split(k_act, num_envs), state
        )
        out = jax.vmap(env.step)(jax.random.split(k_step, num_envs), state, action)
        return out[0], (state, action, out[2], out[4])

    _, (states, actions, rewards, dones) = jax.jit(
        lambda s: jax.lax.scan(one, s, jax.random.split(jax.random.PRNGKey(seed + 1), num_steps))
    )(state)
    return states, actions, rewards, dones


@pytest.mark.parametrize("n", [4, 6, 8])
def test_sizes(n):
    env = Kemps(num_agents=n)
    assert env.obs_dim == 2 * 52 + n * 2
    assert env.num_actions == (R * R + 1 + n // 2) * 2
    state, obs = env.reset(jax.random.PRNGKey(0))
    assert obs.shape == (n, env.obs_dim)
    assert env.get_avail_actions(state).shape == (n, env.num_actions)


@pytest.mark.parametrize("n", [4, 6])
def test_observation_is_relative_to_the_observer(n):
    """Moving every player k seats on moves the observations with them, and the
    own-hand block holds the observer's own hand whatever its seat."""
    env = Kemps(num_agents=n)
    keys = jax.random.split(jax.random.PRNGKey(1), 64)
    states, _ = jax.vmap(env.reset)(keys)
    signals = jax.random.randint(jax.random.PRNGKey(2), (64, n), 0, env.comm_dim)
    states = states.replace(communication=jax.nn.one_hot(signals, env.comm_dim))
    obs = jax.vmap(env.obs_from_state)(states)
    own = jax.vmap(jax.vmap(lambda h: jnp.zeros(52).at[h].set(1.0)))(states.agent_hands)
    np.testing.assert_array_equal(obs[:, :, :52], own)
    for k in range(1, n):
        obs_rot = jax.vmap(env.obs_from_state)(jax.vmap(lambda s: rotate(s, k))(states))
        np.testing.assert_array_equal(obs_rot, jnp.roll(obs, k, axis=1))


@pytest.mark.parametrize("n", [4, 6, 8])
def test_teams_are_opposite_pairs(n):
    """Seat i's KEMPS is right exactly when seat i + n/2 holds four of a kind;
    partners always get the same reward and rewards sum to zero."""
    env = Kemps(num_agents=n)
    k = n // 2
    for caller in range(n):
        for holder in range(n):
            state = make_state(env, *deal(env, {holder: 0}))
            game = [NOOP] * n
            game[caller] = KEMPS
            _, _, rew, _, done, _ = env.step_env(jax.random.PRNGKey(0), state, act(env, game))
            rew = np.asarray(rew)
            assert bool(done)
            assert (rew[caller] > 0) == (holder == (caller + k) % n)
            np.testing.assert_allclose(rew[:k], rew[k:])
            assert abs(rew.sum()) < 1e-6


@pytest.mark.parametrize("n", [4, 6, 8])
def test_stop_kemps_accuses_the_team_d_seats_on(n):
    env = Kemps(num_agents=n)
    k = n // 2
    for caller in range(n):
        for d in range(1, k):
            for holder in range(n):
                state = make_state(env, *deal(env, {holder: 0}))
                game = [NOOP] * n
                game[caller] = STOP(d)
                _, _, rew, _, _, _ = env.step_env(jax.random.PRNGKey(0), state, act(env, game))
                accused = {(caller + d) % n, (caller + d + k) % n}
                assert (rew[caller] > 0) == (holder in accused)


def test_reward_values():
    # Two teams: +1 / -1.
    env = Kemps(num_agents=4)
    state = make_state(env, *deal(env, {2: 0}))
    _, _, rew, _, _, _ = env.step_env(jax.random.PRNGKey(0), state, act(env, [KEMPS, NOOP, NOOP, NOOP]))
    np.testing.assert_allclose(rew, [1, -1, 1, -1])
    _, _, rew, _, _, _ = env.step_env(jax.random.PRNGKey(0), state, act(env, [NOOP, KEMPS, NOOP, NOOP]))
    np.testing.assert_allclose(rew, [1, -1, 1, -1])  # seat 1's partner (3) has no four
    _, _, rew, _, _, _ = env.step_env(jax.random.PRNGKey(0), state, act(env, [NOOP, STOP(1), NOOP, NOOP]))
    np.testing.assert_allclose(rew, [-1, 1, -1, 1])
    # Three teams {0,3}, {1,4}, {2,5}.
    env = Kemps(num_agents=6)
    state = make_state(env, *deal(env, {3: 0}))
    cases = [
        ([KEMPS, NOOP, NOOP, NOOP, NOOP, NOOP], [1, -0.5, -0.5, 1, -0.5, -0.5]),  # right KEMPS
        ([NOOP, KEMPS, NOOP, NOOP, NOOP, NOOP], [0.5, -1, 0.5, 0.5, -1, 0.5]),  # wrong KEMPS
        ([NOOP, NOOP, STOP(1), NOOP, NOOP, NOOP], [-1, 0.5, 0.5, -1, 0.5, 0.5]),  # right STOP
        ([NOOP, NOOP, STOP(2), NOOP, NOOP, NOOP], [0.5, 0.5, -1, 0.5, 0.5, -1]),  # wrong STOP
    ]
    for game, expected in cases:
        _, _, rew, _, _, _ = env.step_env(jax.random.PRNGKey(0), state, act(env, game))
        np.testing.assert_allclose(rew, expected, atol=1e-6)


def _scenarios():
    """Two or three right declarations in one step, seen from seat 0's team."""
    return [
        # Seat 0 calls KEMPS on partner 2's four; seat 1 calls STOP on team {0, 2}.
        (4, {2: 0}, {0: KEMPS, 1: STOP(1)}, 0.0),
        # Seats 0 and 1 both call KEMPS, both partners hold four.
        (4, {2: 0, 3: 1}, {0: KEMPS, 1: KEMPS}, 0.0),
        # Seat 0 holds four; seat 0 (wrong) and its partner 2 (right) call KEMPS.
        (4, {0: 0}, {0: KEMPS, 2: KEMPS}, 0.0),
        # Seat 0 calls KEMPS on partner 3's four, seat 1 calls STOP on team {0, 3}
        # (right), seat 2 calls STOP on team {1, 4} (wrong): (1 - 1 + 0.5) / 3.
        (6, {3: 0}, {0: KEMPS, 1: STOP(2), 2: STOP(2)}, 1 / 6),
    ]


def _rotated_case(env, fours, calls, p):
    n = env.num_agents
    state = make_state(env, *deal(env, {(s + p) % n: r for s, r in fours.items()}))
    game = [NOOP] * n
    for seat, call in calls.items():
        game[(seat + p) % n] = call
    return state, act(env, game)


@pytest.mark.parametrize("n, fours, calls, expected", _scenarios())
def test_simultaneous_declarations_exact(n, fours, calls, expected):
    """Over every player order, the crafted case moved p seats on gives the
    rewards moved p seats on, so seat 0's team expects the same at every p."""
    env = Kemps(num_agents=n)
    orders = jnp.array(list(itertools.permutations(range(n))), dtype=jnp.int32)
    step = jax.jit(jax.vmap(env._step_in_order, in_axes=(None, None, 0)))
    state, action = _rotated_case(env, fours, calls, 0)
    base = np.asarray(step(state, action, orders)[2])
    np.testing.assert_allclose(base[:, 0].mean(), expected, atol=1e-6)
    for p in range(1, n):
        state, action = _rotated_case(env, fours, calls, p)
        rew = np.asarray(step(state, action, (orders + p) % n)[2])
        np.testing.assert_allclose(rew, np.roll(base, p, axis=1), atol=1e-6)


@pytest.mark.parametrize("n, fours, calls, expected", _scenarios())
def test_simultaneous_declarations_seat_neutral(n, fours, calls, expected):
    """Through step_env's random player order, seat p's team gets the same
    expected reward when the case is moved p seats on."""
    env = Kemps(num_agents=n)
    keys = jax.random.split(jax.random.PRNGKey(3), 4000)
    step = jax.jit(jax.vmap(env.step_env, in_axes=(0, None, None)))
    for p in range(n):
        state, action = _rotated_case(env, fours, calls, p)
        rew = np.asarray(step(keys, state, action)[2])[:, p]
        assert abs(rew.mean() - expected) < 0.08, (p, rew.mean())  # about 5 standard errors


def test_declaration_is_judged_before_this_steps_swaps():
    """Partner 2 completes four of a kind by a swap in the same step as seat 0's
    KEMPS: the call is wrong and the swap doesn't happen."""
    env = Kemps(num_agents=4)
    # Rank r is cards 4r..4r+3. Seat 2 holds three of rank 9 and a king (rank
    # 12); the centre holds the fourth 9.
    hands = [[0, 4, 8, 12], [16, 20, 24, 28], [36, 37, 38, 49], [1, 5, 9, 13]]
    center = [39, 44, 45, 46]
    used = {c for h in hands for c in h} | set(center)
    stock = [c for c in range(52) if c not in used]
    state = make_state(env, hands, center, stock)
    swap_49_for_9 = 12 * R + 9
    nxt, _, rew, _, done, _ = env.step_env(
        jax.random.PRNGKey(0), state, act(env, [KEMPS, NOOP, swap_49_for_9, NOOP])
    )
    assert bool(done)
    np.testing.assert_allclose(rew, [-1, 1, -1, 1])
    np.testing.assert_array_equal(nxt.agent_hands, state.agent_hands)


def test_swap_conflict_goes_to_the_first_in_order():
    env = Kemps(num_agents=4)
    hands = [[0, 4, 8, 12], [16, 20, 24, 28], [32, 36, 40, 44], [1, 5, 9, 13]]
    center = [48, 17, 21, 25]  # one king (rank 12)
    used = {c for h in hands for c in h} | set(center)
    stock = [c for c in range(52) if c not in used]
    state = make_state(env, hands, center, stock)
    action = act(env, [0 * R + 12, 4 * R + 12, NOOP, NOOP])  # seats 0 and 1 want the king
    for first, second in [(0, 1), (1, 0)]:
        order = jnp.array([first, second, 2, 3])
        nxt = env._step_in_order(state, action, order)[0]
        assert 48 in np.asarray(nxt.agent_hands[first])
        np.testing.assert_array_equal(nxt.agent_hands[second], state.agent_hands[second])
        assert int(nxt.agent_hand_counts[first, 12]) == 1
        assert int(nxt.center_counts[12]) == 0


def test_sweep_and_real_deal():
    env = Kemps(num_agents=4)
    state = make_state(env, *deal(env))
    stock_len = state.deck.shape[0]
    all_noop = act(env, [NOOP] * 4)
    # The centre is swept and refilled from the stock.
    state = state.replace(deck_idx=jnp.int32(stock_len - 4))
    nxt, _, rew, _, done, _ = env.step_env(jax.random.PRNGKey(0), state, all_noop)
    assert not bool(done)
    np.testing.assert_array_equal(nxt.center_cards, state.deck[-4:])
    assert int(nxt.deck_idx) == stock_len
    # The stock is empty: the hand ends with no score.
    nxt, _, rew, absorbing, done, _ = env.step_env(jax.random.PRNGKey(0), nxt, all_noop)
    assert bool(done) and bool(absorbing.all())
    np.testing.assert_array_equal(rew, 0.0)


@pytest.mark.parametrize("n", [4, 6])
def test_step_rotation_equivariance(n):
    """Moving every player k seats on (state, actions and player order) moves
    the next state, observations and rewards k seats on."""
    env = Kemps(num_agents=n)
    states, _, _, _ = rollout(env, seed=n, num_envs=64, num_steps=32, p_declare=0.02, p_noop=0.5)
    states = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), states)
    num = states.timestep.shape[0]
    keys = jax.random.split(jax.random.PRNGKey(7), num)
    actions = jax.vmap(lambda k, s: sample_actions(env, k, s, 0.3, 0.2))(keys, states)
    orders = jax.vmap(lambda k: jax.random.permutation(k, n))(jax.random.split(keys[0], num))
    step = jax.jit(jax.vmap(env._step_in_order))
    nxt, obs, rew, _, done, _ = step(states, actions, orders)
    declarers = (actions // env.comm_dim >= KEMPS).sum(axis=1)
    assert (declarers >= 2).mean() > 0.1  # the tie-break is exercised
    for k in range(1, n):
        nxt_r, obs_r, rew_r, _, done_r, _ = step(
            jax.vmap(lambda s: rotate(s, k))(states), jnp.roll(actions, k, axis=1), (orders + k) % n
        )
        np.testing.assert_allclose(rew_r, jnp.roll(rew, k, axis=1), atol=1e-6)
        np.testing.assert_array_equal(obs_r, jnp.roll(obs, k, axis=1))
        np.testing.assert_array_equal(done_r, done)
        np.testing.assert_array_equal(nxt_r.agent_hands, jnp.roll(nxt.agent_hands, k, axis=1))
        np.testing.assert_array_equal(nxt_r.center_cards, nxt.center_cards)
        np.testing.assert_array_equal(nxt_r.deck_idx, nxt.deck_idx)


@pytest.mark.parametrize("n", [4, 6, 8])
def test_card_conservation(n):
    """Hands, centre and stock never share a card, the rank counts match the
    cards, and sweeps, Real Deals and cut-offs all happen."""
    horizon = 40
    env = Kemps(num_agents=n, horizon=horizon)
    # Everyone plays NOOP in a quarter of the steps, so the centre is swept often.
    states, actions, rewards, dones = rollout(
        env, seed=10 + n, num_envs=32, num_steps=300, p_declare=0.0, p_noop=0.25 ** (1 / n)
    )
    stock_len = states.deck.shape[-1]

    def check(s):
        stock_live = jnp.arange(stock_len) >= s.deck_idx
        seen = (
            jnp.zeros(52).at[s.agent_hands.ravel()].add(1.0).at[s.center_cards].add(1.0)
            .at[s.deck].add(stock_live.astype(jnp.float32))
        )
        counts_ok = (s.agent_hand_counts == jax.vmap(lambda h: jnp.bincount(h // S, length=R))(s.agent_hands)).all()
        counts_ok &= (s.center_counts == jnp.bincount(s.center_cards // S, length=R)).all()
        total_ok = seen.sum() == 4 * n + 4 + stock_live.sum()
        return (seen.max() <= 1) & counts_ok & total_ok

    ok = jax.vmap(jax.vmap(check))(states)
    assert bool(ok.all())
    real_deal = dones & (states.timestep + 1 < horizon)
    assert int(real_deal.sum()) > 0
    assert int((dones & ~real_deal).sum()) > 0
    np.testing.assert_array_equal(rewards, 0.0)  # nobody declared
    assert int(states.deck_idx.max()) == stock_len


@pytest.mark.parametrize("n", [4, 6, 8])
def test_zero_sum_team_rewards(n):
    env = Kemps(num_agents=n)
    k = n // 2
    _, _, rewards, dones = rollout(env, seed=20 + n, num_envs=64, num_steps=64, p_declare=0.1, p_noop=0.3)
    rewards = np.asarray(rewards).reshape(-1, n)
    dones = np.asarray(dones).reshape(-1)
    np.testing.assert_allclose(rewards.sum(axis=1), 0.0, atol=1e-5)
    np.testing.assert_allclose(rewards[:, :k], rewards[:, k:])
    assert not np.abs(rewards[~dones]).any()
    decided = rewards[np.abs(rewards).sum(axis=1) > 0]
    assert len(decided) > 100 and dones.sum() >= len(decided)
    allowed = {1.0, -1.0, 1 / (k - 1), -1 / (k - 1)}
    assert all(any(abs(v - a) < 1e-5 for a in allowed) for v in decided.ravel())
    assert np.all(np.isclose(np.abs(decided).max(axis=1), 1.0))
