"""Goofspiel: full-game invariants, observation contents, illegal bids,
zero-sum rewards and seat-rotation equivariance."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax.environments.goofspiel.goofspiel import Goofspiel

N = 13  # cards and prizes


def rollout(env, seed, num_games):
    """Random legal bids for a full game. Arrays are (round, game, ...)."""
    keys = jax.random.split(jax.random.PRNGKey(seed), num_games)
    state, obs = jax.vmap(env.reset)(keys)

    def one(carry, key):
        state, obs = carry
        avail = jax.vmap(env.get_avail_actions)(state)
        k_act, k_step = jax.random.split(key)
        action = jax.random.categorical(k_act, jnp.where(avail, 0.0, -jnp.inf), axis=-1)
        nxt, nobs, rew, _, done, info = jax.vmap(env.step_env)(
            jax.random.split(k_step, num_games), state, action
        )
        return (nxt, nobs), dict(
            state=state, obs=obs, avail=avail, action=action, rew=rew, done=done, info=info, next_obs=nobs
        )

    (final, final_obs), out = jax.jit(
        lambda c: jax.lax.scan(one, c, jax.random.split(jax.random.PRNGKey(seed + 1), N))
    )((state, obs))
    out = jax.tree_util.tree_map(np.asarray, out)
    return final, np.asarray(final_obs), out


def tally(prizes, actions):
    """Points from the rules, independent of the env: (round + 1, game, player)."""
    rounds, games, n = actions.shape
    points = np.zeros((rounds + 1, games, n))
    for t in range(rounds):
        top = actions[t].max(axis=1, keepdims=True)
        unique = (actions[t] == top).sum(axis=1, keepdims=True) == 1
        won = (actions[t] == top) & unique
        points[t + 1] = points[t] + won * (prizes[:, t, None] + 1)
    return points


def rel(x):
    """(games, n, ...) -> (games, observer i, j, ...) holding player (i + j) % n."""
    n = x.shape[1]
    return x[:, (np.arange(n)[:, None] + np.arange(n)[None, :]) % n]


@pytest.mark.parametrize("n", [2, 3])
def test_full_game(n):
    """13 rounds; each prize is revealed exactly once; no player's mask is ever
    empty before the end; every card is played exactly once."""
    env = Goofspiel(num_agents=n)
    final, final_obs, out = rollout(env, seed=n, num_games=256)
    avail = out["avail"]  # (13, games, n, 13)
    np.testing.assert_array_equal(avail.sum(-1), np.broadcast_to((N - np.arange(N))[:, None, None], avail.shape[:-1]))
    assert not out["done"][:-1].any() and out["done"][-1].all()
    assert not np.asarray(jax.vmap(env.get_avail_actions)(final)).any()
    assert not np.asarray(final.player_hands).any()
    # The prize shown in round t is deck[t], and the 13 prizes are all different.
    prize_block = out["obs"][..., 0, :N]  # (13, games, 13)
    np.testing.assert_array_equal(prize_block.sum(-1), 1.0)
    shown = prize_block.argmax(-1).T  # (games, 13)
    np.testing.assert_array_equal(shown, out["state"].deck[0])
    np.testing.assert_array_equal(np.sort(shown, axis=1), np.broadcast_to(np.arange(N), shown.shape))
    # After the last round no prize is shown and all 13 have been contested.
    np.testing.assert_array_equal(final_obs[..., :N], 0.0)
    np.testing.assert_array_equal(final_obs[..., N : 2 * N], 1.0)


@pytest.mark.parametrize("n", [2, 3])
def test_observation_contents(n):
    """Contested prizes, cards bid and points (relative to the observer) match
    the history; points match a tally made from the rules."""
    env = Goofspiel(num_agents=n)
    assert env.obs_dim == N * (n + 2) + n
    if n == 2:
        assert env.obs_dim == 54
    final, final_obs, out = rollout(env, seed=10 + n, num_games=128)
    prizes = np.asarray(out["state"].deck[0])  # (games, 13)
    actions = out["action"]  # (13, games, n)
    points = tally(prizes, actions)
    obs = np.concatenate([out["obs"], final_obs[None]], axis=0)  # (14, games, n, dim)
    for t in range(N + 1):
        contested = np.zeros((128, N))
        np.put_along_axis(contested, prizes[:, :t], 1.0, axis=1)
        np.testing.assert_array_equal(obs[t, :, :, N : 2 * N], np.broadcast_to(contested[:, None], (128, n, N)))
        used = np.zeros((128, n, N))
        for u in range(t):
            used[np.arange(128)[:, None], np.arange(n)[None], actions[u]] = 1.0
        cards_bid = obs[t, :, :, 2 * N : (n + 2) * N].reshape(128, n, n, N)
        np.testing.assert_array_equal(cards_bid, rel(used))
        np.testing.assert_allclose(obs[t, :, :, (n + 2) * N :] * 91, rel(points[t]), atol=1e-4)


@pytest.mark.parametrize("n", [2, 3])
def test_zero_sum_rewards_and_outcome(n):
    env = Goofspiel(num_agents=n)
    _, _, out = rollout(env, seed=20 + n, num_games=256)
    prizes = np.asarray(out["state"].deck[0])
    points = tally(prizes, out["action"])
    rew = out["rew"]
    np.testing.assert_allclose(rew.sum(-1), 0.0, atol=1e-4)
    opponents_mean = (points[-1].sum(-1, keepdims=True) - points[-1]) / (n - 1)
    np.testing.assert_allclose(rew.sum(0), points[-1] - opponents_mean, atol=1e-4)
    np.testing.assert_allclose(out["info"]["points"], points[1:], atol=1e-4)
    final = points[-1]
    unique_top = (final == final.max(-1, keepdims=True)) & (
        (final == final.max(-1, keepdims=True)).sum(-1, keepdims=True) == 1
    )
    np.testing.assert_array_equal(out["info"]["game_winner"][-1], unique_top)
    assert not out["info"]["game_winner"][:-1].any()
    if n == 2:
        np.testing.assert_allclose(rew.sum(0)[:, 0], final[:, 0] - final[:, 1], atol=1e-4)
        assert 0 < (final[:, 0] == final[:, 1]).sum() < 256  # draws happen and have no winner


def test_illegal_bid_cannot_score():
    """A card already played is replaced by the lowest card still in hand."""
    env = Goofspiel()
    state, _ = env.reset(jax.random.PRNGKey(0))
    state = state.replace(deck=jnp.arange(N)[::-1])  # prizes 13, 12, 11, ...
    # Card index c is worth c + 1. Round 1: 13 beats 1 for the 13-point prize.
    state, _, rew, _, _, _ = env.step_env(jax.random.PRNGKey(1), state, jnp.array([12, 0]))
    np.testing.assert_allclose(rew, [13, -13])
    # Round 2: player 0 bids its 13 again, so it plays its lowest card, the 1,
    # and loses the 12-point prize to the 3.
    state, obs, rew, _, _, info = env.step_env(jax.random.PRNGKey(2), state, jnp.array([12, 2]))
    np.testing.assert_allclose(rew, [-12, 12])
    np.testing.assert_allclose(info["points"], [13, 12])
    used0 = np.zeros(N, dtype=bool)
    used0[[12, 0]] = True
    np.testing.assert_array_equal(state.player_hands[0], ~used0)
    np.testing.assert_array_equal(obs[0, 2 * N : 3 * N], used0)
    # Round 3: player 0's bid is out of range and player 1 bids its 3 again; both
    # play their lowest card, the 2, and the tie discards the prize.
    state, _, rew, _, _, info = env.step_env(jax.random.PRNGKey(3), state, jnp.array([N, 2]))
    np.testing.assert_allclose(rew, [0, 0])
    np.testing.assert_allclose(info["points"], [13, 12])
    assert not bool(state.player_hands[0, 1]) and not bool(state.player_hands[1, 1])
    assert int(state.player_hands.sum()) == 2 * (N - 3)


def test_tie_discards_the_prize():
    env = Goofspiel()
    state, _ = env.reset(jax.random.PRNGKey(0))
    nxt, _, rew, _, _, info = env.step_env(jax.random.PRNGKey(1), state, jnp.array([5, 5]))
    np.testing.assert_array_equal(rew, 0.0)
    np.testing.assert_array_equal(info["points"], 0.0)
    assert not bool(nxt.player_hands[0, 5]) and not bool(nxt.player_hands[1, 5])


def test_num_decks_is_removed():
    with pytest.raises(TypeError):
        Goofspiel(num_decks=2)


def rotate(state, k):
    roll = lambda x: jnp.roll(x, k, axis=0)
    return state.replace(
        player_hands=roll(state.player_hands),
        points=roll(state.points),
        winners=roll(state.winners),
        absorbing=roll(state.absorbing),
    )


@pytest.mark.parametrize("n", [2, 3])
def test_step_rotation_equivariance(n):
    """Moving every player k seats on moves the rewards, observations and
    outcome k seats on."""
    env = Goofspiel(num_agents=n)
    _, _, out = rollout(env, seed=30 + n, num_games=128)
    states = jax.tree_util.tree_map(lambda x: jnp.asarray(x).reshape((-1,) + x.shape[2:]), out["state"])
    actions = jnp.asarray(out["action"]).reshape(-1, n)
    keys = jax.random.split(jax.random.PRNGKey(5), actions.shape[0])
    step = jax.jit(jax.vmap(env.step_env))
    _, obs, rew, _, done, info = step(keys, states, actions)
    obs0 = jax.vmap(env.obs_from_state)(states)
    for k in range(1, n):
        rot = jax.vmap(lambda s: rotate(s, k))(states)
        np.testing.assert_array_equal(jax.vmap(env.obs_from_state)(rot), jnp.roll(obs0, k, axis=1))
        _, obs_r, rew_r, _, done_r, info_r = step(keys, rot, jnp.roll(actions, k, axis=1))
        np.testing.assert_allclose(rew_r, jnp.roll(rew, k, axis=1))
        np.testing.assert_array_equal(obs_r, jnp.roll(obs, k, axis=1))
        np.testing.assert_array_equal(done_r, done)
        np.testing.assert_array_equal(info_r["game_winner"], jnp.roll(info["game_winner"], k, axis=1))
