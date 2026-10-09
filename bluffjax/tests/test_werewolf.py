"""Tests of the Werewolf environment.

Most checks run on random legal play: every reachable state of a few thousand
games is checked against the rules, so they cover all roles at all indices.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.environments.werewolf.werewolf import (
    DOCTOR,
    PHASE_ACCUSE,
    PHASE_NIGHT,
    PHASE_VOTE,
    SEER,
    VILLAGER,
    WEREWOLF,
)

VALID = [(n, k) for n in range(5, 9) for k in range(1, 4) if 2 * k < n]
NUM_GAMES = 600
NUM_STEPS = 128  # longer than any 8-player game


def _env(n=6, k=2, **kw):
    return make("werewolf", num_agents=n, num_werewolves=k, **kw)


def _np(tree):
    return jax.tree_util.tree_map(np.asarray, tree)


@functools.lru_cache(maxsize=None)
def random_games(n, k, num_games=NUM_GAMES, num_steps=NUM_STEPS, seed=0):
    """Random legal play of num_games games with step_env (frozen once over).

    Returns numpy records (T, B, ...): the state before each step, its legal
    actions, the action, rewards, done, info, the state after the step, and
    whether the step belongs to the game (live).
    """
    env = _env(n, k)
    B = num_games
    state, _ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(seed), B))

    def one(carry, key):
        s, over = carry
        avail = jax.vmap(env.get_avail_actions)(s)
        k_act, k_step = jax.random.split(key)
        logits = jnp.where(avail, 0.0, -jnp.inf)
        action = jax.random.categorical(k_act, logits, axis=-1)
        ns, _, reward, _, done, info = jax.vmap(env.step_env)(
            jax.random.split(k_step, B), s, action
        )
        keep = lambda new, old: jnp.where(
            over.reshape((B,) + (1,) * (new.ndim - 1)), old, new
        )
        ns = jax.tree_util.tree_map(keep, ns, s)
        rec = dict(
            state=s,
            avail=avail,
            action=action,
            reward=reward,
            done=done,
            info=info,
            next=ns,
            live=~over,
        )
        return (ns, over | done), rec

    keys = jax.random.split(jax.random.PRNGKey(seed + 1), num_steps)
    (_, over), rec = jax.jit(
        lambda s: jax.lax.scan(one, (s, jnp.zeros(B, bool)), keys)
    )(state)
    assert np.asarray(over).all(), "some games did not finish"
    return _np(rec)


def _select(tree, mask):
    return jax.tree_util.tree_map(lambda x: x[mask], tree)


def _winner(alive, roles):
    num_ww = (alive & (roles == WEREWOLF)).sum(-1)
    num_humans = (alive & (roles != WEREWOLF)).sum(-1)
    return np.where(num_ww == 0, 0, np.where(num_ww >= num_humans, 1, -1))


def _absolute(avail_rel, cp):
    """Relative target mask (B, n) -> absolute target mask."""
    n = avail_rel.shape[-1]
    abs_idx = (cp[:, None] + np.arange(n)[None, :]) % n
    out = np.zeros_like(avail_rel)
    np.put_along_axis(out, abs_idx, avail_rel, axis=-1)
    return out


# ---------------------------------------------------------------- night


@pytest.mark.parametrize("n,k", [(6, 2), (7, 3), (5, 1)])
def test_night_targets_every_role_and_index(n, k):
    """Night masks name the intended players for every role at every index,
    and the chosen target is the player the action names."""
    rec = random_games(n, k)
    m = rec["live"] & (rec["state"].phase == PHASE_NIGHT)
    s, nxt = _select(rec["state"], m), _select(rec["next"], m)
    avail, action = rec["avail"][m], rec["action"][m]
    rows = np.arange(len(action))
    cp, slot = s.current_player_idx, s.night_subphase
    assert (s.night_order[rows, np.minimum(slot, k + 1)] == cp).all()
    assert s.alive[rows, cp].all()

    role = s.roles[rows, cp]
    idx = np.arange(n)[None, :]
    expected = np.where(
        (slot == 0)[:, None],
        s.alive,  # doctor: any living player, itself included
        np.where(
            (slot == 1)[:, None],
            s.alive & (idx != cp[:, None]),  # seer: any other living player
            s.alive & (s.roles != WEREWOLF),  # werewolf: any living human
        ),
    )
    assert (
        role == np.where(slot == 0, DOCTOR, np.where(slot == 1, SEER, WEREWOLF))
    ).all()
    np.testing.assert_array_equal(_absolute(avail[:, :n], cp), expected)
    assert not avail[:, n].any(), "no-op offered at night"

    # Every role acts at every index.
    for r in (DOCTOR, SEER, WEREWOLF):
        assert (np.bincount(cp[role == r], minlength=n) > 0).all()

    # The action names the player it targets.
    target = (cp + action) % n
    assert (action < n).all()
    np.testing.assert_array_equal(nxt.doctor_target[slot == 0], target[slot == 0])
    sm = slot == 1
    np.testing.assert_array_equal(nxt.seer_target[sm], target[sm])
    seen = nxt.seer_results[rows[sm], target[sm]]
    is_ww = s.roles[rows[sm], target[sm]] == WEREWOLF
    np.testing.assert_array_equal(seen, np.where(is_ww, 1.0, -1.0))
    wm = slot >= 2
    np.testing.assert_array_equal(
        nxt.werewolf_targets[rows[wm], slot[wm] - 2], target[wm]
    )


@pytest.mark.parametrize("n,k", [(6, 2), (7, 3)])
def test_night_victim(n, k):
    """The victim is a most-picked target of the living werewolves, a human,
    and survives exactly when the doctor protected it."""
    rec = random_games(n, k)
    s, nxt = rec["state"], rec["next"]
    resolved = (
        rec["live"]
        & (s.phase == PHASE_NIGHT)
        & ((nxt.phase != PHASE_NIGHT) | rec["done"])
    )
    s, nxt = _select(s, resolved), _select(nxt, resolved)
    died = s.alive & ~nxt.alive
    assert (died.sum(-1) <= 1).all()
    ww = s.night_order[:, 2:]
    picks = np.where(np.take_along_axis(s.alive, ww, -1), nxt.werewolf_targets, -1)
    counts = (picks[:, :, None] == np.arange(n)[None, None, :]).sum(1)
    most = counts == counts.max(-1, keepdims=True)
    victim = died.argmax(-1)
    rows = np.arange(len(victim))
    has_victim = died.any(-1)
    assert most[rows, victim][has_victim].all()
    assert (s.roles[rows, victim][has_victim] != WEREWOLF).all()
    assert (victim[has_victim] != nxt.doctor_target[has_victim]).all()
    # Nobody died: the doctor protected a most-picked target.
    saved = ~has_victim
    assert saved.any() and has_victim.any()
    assert (nxt.doctor_target[saved] >= 0).all()
    assert most[rows[saved], nxt.doctor_target[saved]].all()


@pytest.mark.parametrize(
    "dead, actor, slot, legal",
    [
        # roles [V, WW, D, V, S, WW], night order [D=2, S=4, WW=1, WW=5]
        ([], 1, 2, [0, 2, 3, 4]),
        ([], 5, 3, [0, 2, 3, 4]),
        ([], 4, 1, [0, 1, 2, 3, 5]),
        ([], 2, 0, [0, 1, 2, 3, 4, 5]),
        ([3], 1, 2, [0, 2, 4]),
        ([3], 4, 1, [0, 1, 2, 5]),
        ([3], 2, 0, [0, 1, 2, 4, 5]),
    ],
)
def test_crafted_night_masks(dead, actor, slot, legal):
    """The review's example: the werewolf at index 1 could target itself and
    its partner but not the doctor."""
    env = _env()
    state, _ = env.reset(jax.random.PRNGKey(0))
    alive = np.ones(6, bool)
    alive[dead] = False
    state = state.replace(
        roles=jnp.array([VILLAGER, WEREWOLF, DOCTOR, VILLAGER, SEER, WEREWOLF]),
        alive=jnp.array(alive),
        night_order=jnp.array([2, 4, 1, 5]),
        night_subphase=jnp.int32(slot),
        current_player_idx=jnp.int32(actor),
    )
    avail = np.asarray(env.get_avail_actions(state))
    targets = sorted((actor + np.flatnonzero(avail[:6])) % 6)
    assert targets == legal
    assert not avail[6]


# ---------------------------------------------------------------- turns


def _segments(rec, b):
    """Phases of game b in order: (phase, actors, alive at the start, state)."""
    s = rec["state"]
    out = []
    for t in np.flatnonzero(rec["live"][:, b]):
        phase = s.phase[t, b]
        if not out or out[-1][0] != phase:
            st = jax.tree_util.tree_map(lambda x: x[t, b], s)
            out.append((phase, [], s.alive[t, b].copy(), st))
        out[-1][1].append(int(s.current_player_idx[t, b]))
    return out


def _seat_order(first, alive):
    n = len(alive)
    return [(first + i) % n for i in range(n) if alive[(first + i) % n]]


@pytest.mark.parametrize("n,k", [(6, 2), (8, 1)])
def test_dead_players_never_act(n, k):
    rec = random_games(n, k)
    live = rec["live"]
    s = rec["state"]
    cp = s.current_player_idx
    actor_alive = np.take_along_axis(s.alive, cp[..., None], -1)[..., 0]
    assert actor_alive[live].all()
    assert rec["avail"][live].any(-1).all()


@pytest.mark.parametrize("n,k", [(6, 2), (8, 1)])
def test_turn_order(n, k):
    """Each night, the living night actors act once in night order. Each day,
    every living player accuses once and votes once, in seat order from the
    day's first speaker, who is the next living player after the previous
    day's first speaker. The first day starts at a uniformly random index."""
    rec = random_games(n, k)
    first_day_speaker = []
    num_rotations = 0
    for b in range(NUM_GAMES):
        prev_first = None
        segs = _segments(rec, b)
        for i, (phase, actors, alive, st) in enumerate(segs):
            # A game only ends on the last action of a night or a vote.
            if phase == PHASE_NIGHT:
                assert actors == [int(p) for p in st.night_order if alive[p]]
                continue
            first = actors[0]
            assert actors == _seat_order(first, alive)
            if phase == PHASE_ACCUSE:
                if prev_first is None:
                    first_day_speaker.append(first)
                else:
                    assert first == _seat_order(prev_first + 1, alive)[0]
                    num_rotations += 1
                prev_first = first
            else:
                assert segs[i - 1][0] == PHASE_ACCUSE
                assert first == segs[i - 1][1][0]
    assert len(first_day_speaker) > NUM_GAMES // 2 and num_rotations > 200
    counts = np.bincount(first_day_speaker, minlength=n)
    p = 1.0 / n
    sigma = np.sqrt(p * (1 - p) / len(first_day_speaker))
    assert np.abs(counts / len(first_day_speaker) - p).max() < 5 * sigma, counts


# ---------------------------------------------------------------- endings


@pytest.mark.parametrize("n,k", VALID)
def test_games_end_as_soon_as_a_team_has_won(n, k):
    """Every step: the game is over exactly when a team has won, winners get
    +10 and losers -10 at that step, and nothing is paid otherwise."""
    rec = random_games(n, k)
    live = rec["live"]
    s, nxt = rec["state"], rec["next"]
    won = _winner(nxt.alive, nxt.roles)
    np.testing.assert_array_equal(rec["done"][live], (won >= 0)[live])
    np.testing.assert_array_equal(nxt.game_winner[live], won[live])
    team_won = np.where(nxt.roles == WEREWOLF, won[..., None] == 1, won[..., None] == 0)
    expected = np.where((won >= 0)[..., None], np.where(team_won, 10.0, -10.0), 0.0)
    np.testing.assert_array_equal(rec["reward"][live], expected[live])
    np.testing.assert_array_equal(
        rec["info"]["game_winner"][live], (team_won & (won >= 0)[..., None])[live]
    )
    assert (_winner(s.alive, s.roles)[live] < 0).all()


def _ww_night_state(env, **kw):
    """roles [V, WW, D, V, S, WW], night order [D=2, S=4, WW=1, WW=5]."""
    state, _ = env.reset(jax.random.PRNGKey(0))
    return state.replace(
        roles=jnp.array([VILLAGER, WEREWOLF, DOCTOR, VILLAGER, SEER, WEREWOLF]),
        night_order=jnp.array([2, 4, 1, 5]),
        phase=jnp.int32(PHASE_NIGHT),
        **kw,
    )


def test_night_checks_for_a_human_win():
    """If the night removes the last werewolf the humans win at once (only an
    illegal action, the last werewolf attacking itself, gets there)."""
    env = _env()
    state = _ww_night_state(
        env,
        alive=jnp.array([True, True, True, True, True, False]),
        night_subphase=jnp.int32(2),
        current_player_idx=jnp.int32(1),
    )
    ns, _, reward, _, done, _ = env.step_env(jax.random.PRNGKey(1), state, 0)
    assert bool(done) and int(ns.game_winner) == 0
    np.testing.assert_array_equal(reward, [10, -10, 10, 10, 10, -10])


@pytest.mark.parametrize("at_horizon", [False, True])
def test_night_kill_to_parity_wins_for_the_werewolves(at_horizon):
    """A win on the horizon step is still a win."""
    env = _env(horizon=50)
    state = _ww_night_state(
        env,
        alive=jnp.array([True, True, True, True, False, True]),
        night_subphase=jnp.int32(3),
        current_player_idx=jnp.int32(5),
        werewolf_targets=jnp.array([0, -1]),
        doctor_target=jnp.int32(3),
        timestep=jnp.int32(49 if at_horizon else 10),
    )
    ns, _, reward, _, done, info = env.step_env(jax.random.PRNGKey(1), state, 1)
    assert bool(done) and int(ns.game_winner) == 1
    assert not bool(ns.alive[0])
    np.testing.assert_array_equal(reward, [-10, 10, -10, -10, -10, 10])
    np.testing.assert_array_equal(info["game_winner"], [0, 1, 0, 0, 0, 1])


def test_timeout_reports_no_winner():
    horizon = 6  # the first night (4 steps) never ends a 6-player game
    env = _env(horizon=horizon)
    state, _ = env.reset(jax.random.PRNGKey(3))
    rng = jax.random.PRNGKey(4)
    for t in range(horizon):
        rng, k_act, k_step = jax.random.split(rng, 3)
        avail = env.get_avail_actions(state)
        action = jax.random.categorical(k_act, jnp.where(avail, 0.0, -jnp.inf))
        state, _, reward, absorbing, done, info = env.step_env(k_step, state, action)
        np.testing.assert_array_equal(reward, np.zeros(6))
        assert bool(done) == (t == horizon - 1)
    assert int(state.game_winner) == -1
    assert np.asarray(absorbing).all()
    np.testing.assert_array_equal(info["game_winner"], np.zeros(6))


# ---------------------------------------------------------------- set-up


@pytest.mark.parametrize("n,k", VALID)
def test_player_and_werewolf_counts(n, k):
    env = _env(n, k)
    assert env.obs_dim == 7 + 6 * n and env.num_actions == n + 1
    states, obs = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 200))
    states = _np(states)
    assert obs.shape == (200, 7 + 6 * n)
    counts = np.stack([(states.roles == r).sum(-1) for r in range(4)], -1)
    np.testing.assert_array_equal(
        counts, np.broadcast_to([n - 2 - k, k, 1, 1], counts.shape)
    )
    order_roles = np.take_along_axis(states.roles, states.night_order, -1)
    np.testing.assert_array_equal(
        order_roles, np.broadcast_to([DOCTOR, SEER] + [WEREWOLF] * k, order_roles.shape)
    )
    assert (
        np.sort(states.night_order, -1)[:, 1:]
        != np.sort(states.night_order, -1)[:, :-1]
    ).all()
    random_games(n, k)  # every game finishes (asserted there)


@pytest.mark.parametrize("n,k", [(5, 3), (6, 3), (4, 2), (6, 0)])
def test_invalid_counts_raise(n, k):
    with pytest.raises(ValueError):
        _env(n, k)


@pytest.mark.parametrize("n,k", [(6, 2), (7, 3), (8, 1)])
def test_roles_start_and_night_order_are_uniform(n, k):
    B = 20000
    env = _env(n, k)
    states, _ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(5), B))
    states = _np(states)
    expected = {VILLAGER: n - 2 - k, WEREWOLF: k, DOCTOR: 1, SEER: 1}
    for role, count in expected.items():
        p = count / n
        freq = (states.roles == role).mean(0)
        assert np.abs(freq - p).max() < 5 * np.sqrt(p * (1 - p) / B), (role, freq)
    if k >= 2:
        # The werewolf with the lower index acts first half of the time.
        lower_first = (states.night_order[:, 2] < states.night_order[:, 3]).mean()
        assert abs(lower_first - 0.5) < 5 * np.sqrt(0.25 / B), lower_first
    p = 1 / n
    freq = np.bincount(states.start_player_idx, minlength=n) / B
    assert np.abs(freq - p).max() < 5 * np.sqrt(p * (1 - p) / B), freq


# ---------------------------------------------------------------- seat symmetry

ROTATED_FIELDS = {
    # per-player arrays (accusations and votes also hold player indices)
    "roles",
    "alive",
    "seer_results",
    "absorbing",
    "accusations",
    "votes",
    # player indices
    "night_order",
    "doctor_target",
    "seer_target",
    "werewolf_targets",
    "current_player_idx",
    "start_player_idx",
}
OTHER_FIELDS = {
    "phase",
    "night_subphase",
    "phase_progress",
    "done",
    "game_winner",
    "timestep",
}


def rotate(s, k, n):
    """Moves every player k seats on: per-player arrays roll by k and player
    indices (>= 0) are shifted by k."""
    roll = lambda x: jnp.roll(x, k, axis=0)
    shift = lambda x: jnp.where(x >= 0, (x + k) % n, x)
    return s.replace(
        roles=roll(s.roles),
        alive=roll(s.alive),
        seer_results=roll(s.seer_results),
        absorbing=roll(s.absorbing),
        accusations=shift(roll(s.accusations)),
        votes=shift(roll(s.votes)),
        night_order=shift(s.night_order),
        doctor_target=shift(s.doctor_target),
        seer_target=shift(s.seer_target),
        werewolf_targets=shift(s.werewolf_targets),
        current_player_idx=shift(s.current_player_idx),
        start_player_idx=shift(s.start_player_idx),
    )


@pytest.mark.parametrize("n,k", [(6, 2), (7, 3)])
def test_seat_rotation_equivariance(n, k):
    """Rotating every per-player field by k seats leaves the observation and
    the legal actions unchanged, and step_env with the same action and key
    gives the rotated next state and rewards."""
    env = _env(n, k)
    state_fields = set(env.reset(jax.random.PRNGKey(0))[0].__dataclass_fields__)
    assert state_fields == ROTATED_FIELDS | OTHER_FIELDS
    rec = random_games(n, k)
    m = rec["live"]
    states = _select(rec["state"], m)
    actions = rec["action"][m]
    num = len(actions)
    shifts = np.arange(num) % (n - 1) + 1
    keys = jax.random.split(jax.random.PRNGKey(7), num)

    def check(s, a, shift, key):
        r = rotate(s, shift, n)
        s1, o1, rew1, _, d1, info1 = env.step_env(key, s, a)
        s2, o2, rew2, _, d2, info2 = env.step_env(key, r, a)
        same_next = jax.tree_util.tree_map(
            lambda x, y: jnp.all(x == y), rotate(s1, shift, n), s2
        )
        return dict(
            obs=jnp.all(env.obs_from_state(s) == env.obs_from_state(r)),
            avail=jnp.all(env.get_avail_actions(s) == env.get_avail_actions(r)),
            next_state=jnp.all(jnp.stack(jax.tree_util.tree_leaves(same_next))),
            next_obs=jnp.all(o1 == o2),
            rewards=jnp.all(jnp.roll(rew1, shift) == rew2),
            winner=jnp.all(
                jnp.roll(info1["game_winner"], shift) == info2["game_winner"]
            ),
            done=d1 == d2,
        )

    ok = _np(jax.jit(jax.vmap(check))(states, actions, jnp.asarray(shifts), keys))
    for name, v in ok.items():
        assert v.all(), f"{name}: {np.mean(~v):.4f} of {num} states differ"
    assert rec["done"][m].sum() > 100  # terminal steps are covered


@functools.lru_cache(maxsize=None)
def random_outcomes(n, k, num_games, num_steps, seed):
    """Roles and total rewards of num_games random games."""
    env = _env(n, k)
    B = num_games
    state, _ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(seed), B))

    def one(carry, key):
        s, over, total = carry
        avail = jax.vmap(env.get_avail_actions)(s)
        k_act, k_step = jax.random.split(key)
        action = jax.random.categorical(k_act, jnp.where(avail, 0.0, -jnp.inf), axis=-1)
        ns, _, reward, _, done, _ = jax.vmap(env.step_env)(
            jax.random.split(k_step, B), s, action
        )
        ns = jax.tree_util.tree_map(
            lambda new, old: jnp.where(
                over.reshape((B,) + (1,) * (new.ndim - 1)), old, new
            ),
            ns,
            s,
        )
        total = total + jnp.where(over[:, None], 0.0, reward)
        return (ns, over | done, total), None

    keys = jax.random.split(jax.random.PRNGKey(seed + 1), num_steps)
    init = (state, jnp.zeros(B, bool), jnp.zeros((B, n)))
    (_, over, total), _ = jax.jit(lambda c: jax.lax.scan(one, c, keys))(init)
    assert np.asarray(over).all()
    return np.asarray(state.roles), np.asarray(total)


def test_win_rates_equal_across_indices():
    """Random play: the win rate of each index, per role, is equal within
    noise (the old night masks made a werewolf at index 0 win 81% vs 61%)."""
    roles, total = random_outcomes(6, 2, num_games=20000, num_steps=80, seed=11)
    win = total > 0
    for team in ("werewolf", "human"):
        on_team = (roles == WEREWOLF) if team == "werewolf" else (roles != WEREWOLF)
        games = on_team.sum(0)
        rate = (win & on_team).sum(0) / games
        pooled = (win & on_team).sum() / on_team.sum()
        sigma = np.sqrt(pooled * (1 - pooled) / games)
        assert (np.abs(rate - pooled) < 5 * sigma).all(), (team, rate, pooled)
