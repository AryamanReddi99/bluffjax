"""Head-to-head evaluator of bluffjax/examples/head_to_head/games.py.

Checked here:
- every checkpoint pair plays exactly 32 complete episodes: with an env that
  pays every player 1 on the step that ends the episode, A's return is 1 in
  every episode, and the full pipeline (fake runs of every game, loaded from
  disk) has the right number of pairs and episodes in every cell;
- mirrored deals reuse the deal: episodes 2d and 2d + 1 start from the same
  state, A on side 0 and then 1, and with deterministic policies they are the
  same game with the sides swapped (A's return flips sign);
- seat / team assignment: Goofspiel's exact result with a highest-card bidder
  against a lowest-card bidder, a 5-Card Draw player that always folds loses
  every hand, A's seats are the Kemps team / Werewolf team / Bluff seat of its
  side, Bluff's seat rotates from a random offset;
- Random vs Random is symmetric (win = loss, return 0 within 4 SE; in Bluff
  a third of the decided games each).
"""

import copy
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from bluffjax.environments.werewolf.werewolf import WEREWOLF
from bluffjax.examples.head_to_head import games as h2h
from bluffjax.examples.run_utils import save_params

HIDDEN = 16
ENV_KWARGS = {
    "five_card_draw": dict(num_agents=2, init_chips=100),
    "seven_card_stud": dict(num_agents=2),
    "goofspiel": dict(num_agents=2),
    "bluff": dict(num_agents=3, horizon=150),
    "kemps": dict(num_agents=4, comm_dim=2, horizon=60),
    "werewolf": dict(num_agents=6, num_werewolves=2, horizon=200),
}
MIRRORED = [g for g in h2h.GAMES if h2h.GAMES[g].mirrored]


def _setup(game_key, hidden=HIDDEN):
    game = h2h.GAMES[game_key]
    env = h2h.make_env(game_key, ENV_KWARGS[game_key])
    nets = h2h.make_networks(game, env, hidden)
    ppo_t, pqn_t = h2h.templates(nets, env.obs_dim)
    return game, env, nets, ppo_t, pqn_t


def _zeros(t):
    return jax.tree_util.tree_map(jnp.zeros_like, t)


def random_participant(ppo_t, pqn_t):
    return h2h.ParticipantParams(_zeros(ppo_t), _zeros(pqn_t), jnp.int32(h2h.RANDOM))


def bias_participant(ppo_t, pqn_t, bias):
    """A PQN-kind participant whose logits are `bias` whatever the
    observation (zero kernels, the bias of the last Dense layer)."""
    params = _zeros(pqn_t)
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    last = max(
        (jax.tree_util.keystr(p) for p, _ in flat if "bias" in jax.tree_util.keystr(p)),
        key=lambda k: int(k.split("Dense_")[1].split("'")[0]),
    )
    params = jax.tree_util.tree_map_with_path(
        lambda p, x: jnp.asarray(bias, jnp.float32) if jax.tree_util.keystr(p) == last else x,
        params,
    )
    return h2h.ParticipantParams(_zeros(ppo_t), params, jnp.int32(h2h.PQN))


def _np(tree):
    return jax.tree_util.tree_map(np.asarray, tree)


def _play_pair(game, env, nets, pa, pb, key, initial=False):
    return _np(
        jax.jit(
            lambda k: h2h.play_pair(game, env, nets, pa, pb, k, return_initial_state=initial)
        )(key)
    )


def counting_env(env):
    """A copy of env whose step_env pays every player 1 on the step with done
    and 0 otherwise, so a return counts the ends of the episode."""
    base = type(env)

    class CountingEnv(base):
        def step_env(self, rng, state, action):
            state, obs, reward, absorbing, done, info = base.step_env(self, rng, state, action)
            return state, obs, jnp.where(done, 1.0, 0.0) * jnp.ones_like(reward), absorbing, done, info

    counting = copy.copy(env)
    counting.__class__ = CountingEnv
    return counting


# =============================================================================
# 32 complete episodes per pair
# =============================================================================


@pytest.mark.parametrize("game_key", list(h2h.GAMES))
def test_each_pair_plays_32_complete_episodes(game_key):
    game, env, nets, ppo_t, pqn_t = _setup(game_key)
    env = counting_env(env)
    rnd = random_participant(ppo_t, pqn_t)
    res = _play_pair(game, env, nets, rnd, rnd, jax.random.PRNGKey(3))
    assert res.ret.shape == (h2h.EPISODES_PER_PAIR,)
    assert res.done.all()
    np.testing.assert_array_equal(res.ret, 1.0)  # each episode ended exactly once
    assert (res.length >= 1).all() and (res.length <= env.horizon).all()


def _write_fake_runs(root, game_key, num_seeds):
    game, env, nets, ppo_t, pqn_t = _setup(game_key)
    for label, job, kind in h2h.BASELINES:
        run_dir = os.path.join(root, game_key, f"{game_key}_{job}_2026-01-01_00-00-00")
        os.makedirs(run_dir)
        OmegaConf.save(
            OmegaConf.create(
                dict(
                    job_type=f"{game_key}_{job}",
                    env_name=game_key,
                    env_kwargs=ENV_KWARGS[game_key],
                    fc_dim_size=HIDDEN,
                    num_seeds=num_seeds,
                    seed=0,
                )
            ),
            os.path.join(run_dir, "config.yaml"),
        )
        net = nets.ppo if kind == h2h.PPO else nets.pqn
        obs = jnp.zeros((env.obs_dim,))
        for s, stage in enumerate(h2h.STAGES):
            for i in range(num_seeds):
                params = net.init(jax.random.PRNGKey(100 * s + i + 7 * kind), obs)
                save_params(run_dir, stage, "avg_policy", params, i)
                save_params(run_dir, stage, "br", params, i)  # unused


@pytest.mark.parametrize("game_key", list(h2h.GAMES))
def test_full_pipeline_cells(tmp_path, game_key):
    num_seeds = 2
    _write_fake_runs(str(tmp_path / "runs"), game_key, num_seeds)
    out = h2h.evaluate_game(str(tmp_path / "runs"), game_key, str(tmp_path / "out"), chunk=32)
    with open(tmp_path / "out" / f"{game_key}.json") as f:
        assert json.load(f).keys() == out.keys()
    assert out["participants"] == [
        "PPO-NFSP (half)", "PPO-NFSP (full)", "PQN-NFSP (half)", "PQN-NFSP (full)", "Random"
    ]
    assert out["checkpoints"] == {p: (1 if p == "Random" else num_seeds) for p in out["participants"]}
    n = len(out["participants"])
    for r in range(n):
        for c in range(n):
            cell = out["cells"][r][c]
            k_r = 1 if r == n - 1 else num_seeds
            k_c = 1 if c == n - 1 else num_seeds
            pairs = k_r * (k_r - 1) if (r == c and k_r > 1) else k_r * k_c
            assert cell["pairs"] == pairs
            assert cell["episodes"] == 32 * pairs
            assert abs(cell["win"] + cell["loss"] + cell["draw"] - 1) < 1e-6
            assert ("extra" in cell) == (game_key == "werewolf")
            for v in (cell["win"], cell["loss"], cell["draw"], cell["return"]):
                assert isinstance(v, float) and np.isfinite(v)
    assert np.isnan(out["cells"][n - 1][n - 1]["win_se"])  # a single pair


def test_make_pairs_skips_same_checkpoint_on_diagonal():
    parts = [h2h.Participant("A", [0, 1, 2]), h2h.Participant("R", [3])]
    pairs = h2h.make_pairs(parts)
    assert [(r, c, a, b) for r, c, a, b in pairs if r == c == 0] == [
        (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 1, 0), (0, 0, 1, 2), (0, 0, 2, 0), (0, 0, 2, 1)
    ]
    assert [(a, b) for r, c, a, b in pairs if r == c == 1] == [(3, 3)]
    assert len([p for p in pairs if (p[0], p[1]) == (0, 1)]) == 3


# =============================================================================
# Mirrored deals
# =============================================================================


@pytest.mark.parametrize("game_key", MIRRORED)
def test_mirrored_deals_reuse_the_deal(game_key):
    game, env, nets, ppo_t, pqn_t = _setup(game_key)
    rnd = random_participant(ppo_t, pqn_t)
    res, init = _play_pair(game, env, nets, rnd, rnd, jax.random.PRNGKey(5), initial=True)
    np.testing.assert_array_equal(res.side, np.arange(32) % 2)
    leaves = jax.tree_util.tree_leaves(init)
    for leaf in leaves:
        np.testing.assert_array_equal(leaf[0::2], leaf[1::2])
    # different deals differ
    assert any(not np.array_equal(leaf[0], leaf[2]) for leaf in leaves)
    # independent action sampling: the two games of a deal don't all coincide
    assert not np.array_equal(res.length[0::2], res.length[1::2]) or not np.array_equal(
        res.ret[0::2], -res.ret[1::2]
    )


@pytest.mark.parametrize("game_key", MIRRORED)
def test_mirrored_deterministic_games_swap_sides(game_key):
    """The same deterministic policy in every seat: the two games of a deal
    are the same game (same deal and env randomness), so A's return flips."""
    game, env, nets, ppo_t, pqn_t = _setup(game_key)
    det = bias_participant(ppo_t, pqn_t, 1e3 * jnp.arange(h2h.action_dim_of(env)))
    res = _play_pair(game, env, nets, det, det, jax.random.PRNGKey(9))
    np.testing.assert_array_equal(res.length[0::2], res.length[1::2])
    np.testing.assert_allclose(res.ret[0::2], -res.ret[1::2], atol=1e-5)
    np.testing.assert_array_equal(res.win[0::2], res.loss[1::2])


# =============================================================================
# Seat / team assignment
# =============================================================================


def test_goofspiel_seats_exact_result():
    game, env, nets, ppo_t, pqn_t = _setup("goofspiel")
    high = bias_participant(ppo_t, pqn_t, 1e3 * jnp.arange(13))  # bids 13, 12, ...
    low = bias_participant(ppo_t, pqn_t, -1e3 * jnp.arange(13))  # bids 1, 2, ...
    res, init = _play_pair(game, env, nets, high, low, jax.random.PRNGKey(1), initial=True)
    for g in range(32):
        prizes = init.deck[g] + 1
        a_bid, b_bid = 13 - np.arange(13), 1 + np.arange(13)
        a_pts = prizes[a_bid > b_bid].sum()
        b_pts = prizes[a_bid < b_bid].sum()
        assert res.ret[g] == a_pts - b_pts
        assert res.win[g] == float(a_pts > b_pts)
        assert res.loss[g] == float(a_pts < b_pts)
    np.testing.assert_array_equal(res.a_seats, np.eye(2, dtype=bool)[np.arange(32) % 2])
    h2h.check_routing(game, env, res)


def test_five_card_draw_folder_loses_every_hand():
    """A player that always folds against one that always checks / calls
    (and keeps its cards) loses its blind in every hand, in both seats."""
    game, env, nets, ppo_t, pqn_t = _setup("five_card_draw")
    from bluffjax.environments.five_card_draw.five_card_draw import FOLD

    folder = bias_participant(ppo_t, pqn_t, 1e3 * jax.nn.one_hot(FOLD, env.num_actions))
    caller = bias_participant(ppo_t, pqn_t, 1e3 * jax.nn.one_hot(0, env.num_actions))
    res = _play_pair(game, env, nets, folder, caller, jax.random.PRNGKey(2))
    assert (res.ret < 0).all() and (res.loss == 1).all()
    h2h.check_routing(game, env, res)
    res = _play_pair(game, env, nets, caller, folder, jax.random.PRNGKey(2))
    assert (res.ret > 0).all() and (res.win == 1).all()
    h2h.check_routing(game, env, res)


def test_kemps_teams():
    game, env, nets, ppo_t, pqn_t = _setup("kemps")
    rnd = random_participant(ppo_t, pqn_t)
    res = _play_pair(game, env, nets, rnd, rnd, jax.random.PRNGKey(4))
    team0 = np.array([True, False, True, False])  # partners sit opposite
    expected = np.where((np.arange(32) % 2 == 0)[:, None], team0, ~team0)
    np.testing.assert_array_equal(res.a_seats, expected)
    assert (res.a_moves[res.a_seats] == np.repeat(res.length, 2)).all()
    h2h.check_routing(game, env, res)


def test_werewolf_teams():
    game, env, nets, ppo_t, pqn_t = _setup("werewolf")
    rnd = random_participant(ppo_t, pqn_t)
    res, init = _play_pair(game, env, nets, rnd, rnd, jax.random.PRNGKey(6), initial=True)
    is_ww = init.roles == WEREWOLF
    side0 = (np.arange(32) % 2 == 0)[:, None]
    np.testing.assert_array_equal(res.a_seats, np.where(side0, is_ww, ~is_ww))
    h2h.check_routing(game, env, res)
    # A as the humans with a random werewolf opponent: the humans took decisions
    assert (res.a_moves[1::2].sum(-1) > 0).all() and (res.b_moves[1::2].sum(-1) > 0).all()


def test_bluff_seat_rotates():
    game, env, nets, ppo_t, pqn_t = _setup("bluff")
    rnd = random_participant(ppo_t, pqn_t)
    play = jax.jit(jax.vmap(lambda k: h2h.play_pair(game, env, nets, rnd, rnd, k)))
    res = _np(play(jax.random.split(jax.random.PRNGKey(8), 12)))
    for row in res.side:
        np.testing.assert_array_equal(row, (row[0] + np.arange(32)) % 3)
    assert set(res.side[:, 0]) == {0, 1, 2}
    np.testing.assert_array_equal(res.a_seats, np.eye(3, dtype=bool)[res.side])
    h2h.check_routing(game, env, res)


def test_routing_check_catches_wrong_seat():
    game, env, nets, ppo_t, pqn_t = _setup("goofspiel")
    rnd = random_participant(ppo_t, pqn_t)
    res = _play_pair(game, env, nets, rnd, rnd, jax.random.PRNGKey(1))
    bad = res._replace(a_moves=res.b_moves, b_moves=res.a_moves)
    with pytest.raises(AssertionError):
        h2h.check_routing(game, env, bad)


# =============================================================================
# Random vs Random
# =============================================================================


@pytest.mark.parametrize("game_key", list(h2h.GAMES))
def test_random_vs_random_symmetric(game_key):
    game, env, nets, ppo_t, pqn_t = _setup(game_key)
    rnd = random_participant(ppo_t, pqn_t)
    num_pairs = 24
    play = jax.jit(jax.vmap(lambda k: h2h.play_pair(game, env, nets, rnd, rnd, k)))
    res = _np(play(jax.random.split(jax.random.PRNGKey(11), num_pairs)))

    def z(x, mean):
        m = x.mean(axis=1)  # pair means
        return (m.mean() - mean) / (m.std(ddof=1) / np.sqrt(num_pairs))

    if game_key == "bluff":
        # A is one of three random players: a third of the decided games
        assert abs(z(res.win - (res.win + res.loss) / 3, 0.0)) < 4
    else:
        assert abs(z(res.win - res.loss, 0.0)) < 4
        assert abs(z(res.ret, 0.0)) < 4
