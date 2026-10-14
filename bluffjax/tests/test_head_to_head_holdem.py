"""Checks for the Hold'em head-to-head evaluator (examples/head_to_head/holdem.py)."""

import json
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from bluffjax import make
from bluffjax.examples.head_to_head import holdem as H
from bluffjax.examples.holdem_rebel import cards as C
from bluffjax.examples.holdem_rebel import game as G
from bluffjax.examples.holdem_rebel.agent import RebelPlayer, save_checkpoint
from bluffjax.examples.holdem_rebel.eval_nfsp import ENV_IDS, avg_network
from bluffjax.examples.run_utils import save_params

GAMES = ["hul", "hunl"]
HIDDEN = 16


def _env(game):
    return make(ENV_IDS[game], num_agents=2)


def _action_dim(env):
    st, _ = env.reset(jax.random.PRNGKey(0))
    return int(env.get_avail_actions(st).shape[-1])


def _net_params(game, kind, env, key):
    net = avg_network(game, kind, env, _action_dim(env), HIDDEN)
    return net, net.init(key, jnp.zeros((env.obs_dim,), jnp.float32))


def _call_params(game, kind, env):
    """Average-network params that always check/call: zero weights, output
    bias favouring env action 0 (call / check-call), then 3 (Limit check)."""
    net, params = _net_params(game, kind, env, jax.random.PRNGKey(0))
    a = _action_dim(env)
    bias = np.zeros(a, np.float32)
    bias[0] = 200.0
    if game == "hul":
        bias[3] = 100.0
    return jax.tree.map(lambda x: jnp.asarray(bias) if x.shape == (a,) else jnp.zeros_like(x), params)


def _pool(env, game, members):
    """A Pool of PQN-style networks (one group) plus Random."""
    net = avg_network(game, "pqn", env, _action_dim(env), HIDDEN)
    stack = jax.tree.map(lambda *x: jnp.stack(x), *members)
    player = H.PoolPlayer(env, [net])
    return H.Pool(player, (stack,), {})


@pytest.mark.parametrize("game", GAMES)
def test_mirrored_deals_reuse_the_deal(game):
    """With both sides always calling, a hand's result depends only on the
    deal, so A's rewards from the two seats of a deal are exact opposites;
    with random players they are not (independent action randomness)."""
    env = _env(game)
    call = _call_params(game, "pqn", env)
    pool = _pool(env, game, [call])  # member 0 = always call, member 1 = Random
    rgame = G.make_game(game)
    m = H.Matcher(env, rgame, pool, None, [], n_deals=16)
    r = m.play(H.Side("pool"), H.Side("pool"), [0, 0, 0], [0, 0, 0], 3, jax.random.PRNGKey(1))
    assert r.shape == (3, 16, 2)
    assert np.array_equal(r[..., 0], -r[..., 1])
    assert np.count_nonzero(r) > 0
    # different pairs and deals get different cards
    assert len({tuple(x) for x in r[..., 0]}) > 1
    rr = m.play(H.Side("pool"), H.Side("pool"), [1, 1, 1], [1, 1, 1], 3, jax.random.PRNGKey(2))
    assert not np.array_equal(rr[..., 0], -rr[..., 1])


@pytest.mark.parametrize("game", GAMES)
def test_pool_player_samples_each_members_policy(game):
    """Hand h is played by member idx[h]: the same actions as sampling that
    network's masked softmax with the same keys (Random: uniform legal)."""
    env = _env(game)
    nets_params = [_net_params(game, "pqn", env, jax.random.PRNGKey(s)) for s in range(3)]
    net = nets_params[0][0]
    members = [p for _, p in nets_params]
    pool = _pool(env, game, members)
    n = 64
    states, _ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(3), n))
    avail = jax.vmap(env.get_avail_actions)(states)
    idx = jnp.asarray(np.arange(n) % 4, jnp.int32)  # 3 networks + Random
    rng = jax.random.PRNGKey(4)
    got = np.asarray(pool.player.act((pool.stacks, idx), None, None, rng, states, avail))
    keys = jax.random.split(rng, n)
    for h in range(n):
        s = jax.tree.map(lambda x: x[h], states)
        if h % 4 < 3:
            logits = net.apply(members[h % 4], env.obs_from_state(s))
        else:
            logits = jnp.zeros(avail.shape[-1])
        want = int(jax.random.categorical(keys[h], jnp.where(avail[h], logits, -jnp.inf)))
        assert got[h] == want
        assert bool(avail[h, got[h]])


def test_cell_metrics():
    pairs = [np.array([[1.0, -1.0], [0.0, 2.0]]), np.array([[-2.0, -2.0], [0.0, 0.0]])]
    c = H.cell_metrics(pairs)
    assert c["win"] == pytest.approx(0.25) and c["loss"] == pytest.approx(0.375)
    assert c["draw"] == pytest.approx(0.375) and c["return"] == pytest.approx(-0.25)
    assert c["return_se"] == pytest.approx(np.std([0.5, -1.0], ddof=1) / math.sqrt(2))
    assert c["win_se"] == pytest.approx(np.std([0.5, 0.0], ddof=1) / math.sqrt(2))
    assert c["pairs"] == 2 and c["episodes"] == 8
    assert math.isnan(H.cell_metrics(pairs[:1])["return_se"])


def _fake_runs(root, game, num_seeds=2, rebel_seeds=(0, 1)):
    gdir = os.path.join(root, H.GAMES[game]["dir"])
    env = _env(game)
    for kind in ("ppo", "pqn"):
        run = os.path.join(gdir, f"{game}_{kind}_nfsp_2026-01-01_00-00-00")
        os.makedirs(run)
        with open(os.path.join(run, "config.yaml"), "w") as f:
            yaml.safe_dump({"job_type": f"{game}_{kind}_nfsp", "env_name": game, "seed": 0,
                            "num_seeds": num_seeds, "fc_dim_size": HIDDEN,
                            "env_kwargs": {"num_agents": 2}}, f)
        for s, stage in enumerate(("half", "full")):
            for i in range(num_seeds):
                _, p = _net_params(game, kind, env, jax.random.PRNGKey(100 * s + i))
                save_params(run, stage, "avg_policy", p, i)
    rgame = G.make_game(game)
    player = RebelPlayer(rgame, hidden_dim=8, num_layers=1, cfr_iters=2)
    for seed in rebel_seeds:
        run = os.path.join(gdir, f"{game}_rebel_2026-01-01_00-00-0{seed}")
        os.makedirs(run)
        with open(os.path.join(run, "config.yaml"), "w") as f:
            yaml.safe_dump({"job_type": f"{game}_rebel", "env_name": game, "seed": seed}, f)
        for s, stage in enumerate(("half", "full")):
            params = player.net.init(jax.random.PRNGKey(7 + 10 * seed + s), jnp.zeros((G.PUBLIC_DIM,)),
                                     jnp.zeros((2, C.NUM_HANDS)))
            save_checkpoint(os.path.join(run, stage, "value_net_0.msgpack"), params,
                            {"game": game, "max_raises": rgame.max_raises, "value_hidden_dim": 8,
                             "value_num_layers": 1, "cfr_iters": 2, "samples": 0, "seed": seed})
    os.makedirs(os.path.join(gdir, "unrelated"))  # ignored: no config.yaml


def test_participants_fail_loudly(tmp_path):
    _fake_runs(str(tmp_path), "hul", rebel_seeds=(0,))
    assert not H.rebel_runs_complete(str(tmp_path), "hul", 2)
    with pytest.raises(RuntimeError, match="ReBeL seeds"):
        H.participants(str(tmp_path), "hul", 2)
    parts, _ = H.participants(str(tmp_path), "hul", 1)
    assert [p.name for p in parts] == [
        "PPO-NFSP (half)", "PPO-NFSP (full)", "PQN-NFSP (half)", "PQN-NFSP (full)",
        "ReBeL (half)", "ReBeL (full)", "Random"]
    assert parts[1].paths[1].endswith(os.path.join("full", "avg_policy_1.msgpack"))
    assert H.rebel_runs_complete(str(tmp_path), "hul", 1)


def test_checkpoint_pairs():
    parts = [H.Participant("A", "ppo", ("a0", "a1", "a2")), H.Participant("Random", "random", (None,))]
    pairs = H.checkpoint_pairs(parts)
    assert len([p for p in pairs if p[0] == p[2] == 0]) == 6  # 3 x 3 minus i == j
    assert all(not (p[0] == p[2] == 0 and p[1] == p[3]) for p in pairs)
    assert (1, 0, 1, 0) in pairs  # Random plays itself
    assert len(pairs) == 6 + 3 + 3 + 1


@pytest.mark.parametrize("game", GAMES)
def test_end_to_end(game, tmp_path):
    """A full matrix on fake runs (2 seeds, tiny ReBeL networks, 2 deals per pair)."""
    root, out = str(tmp_path / "runs"), str(tmp_path / "out")
    _fake_runs(root, game)
    res = H.run(root, game, out, n_deals=2, num_rebel_seeds=2, solve_chunk=16, pool_pairs_per_call=16)
    with open(os.path.join(out, f"{game}.json")) as f:
        saved = json.loads(f.read())
    assert saved == json.loads(json.dumps(res))
    assert set(saved) == {"game", "game_key", "units", "episodes_per_pair", "protocol", "participants",
                          "checkpoints", "cells", "commit", "runs_root", "generated"}
    names = saved["participants"]
    assert len(names) == 7 and names[-1] == "Random" and saved["episodes_per_pair"] == 4
    assert saved["checkpoints"] == {n: (1 if n == "Random" else 2) for n in names}
    for r in range(7):
        for c in range(7):
            cell = saved["cells"][r][c]
            nr, nc = saved["checkpoints"][names[r]], saved["checkpoints"][names[c]]
            want = nr * nc - (nr if r == c and nr > 1 else 0)
            assert cell["pairs"] == want and cell["episodes"] == 4 * want
            assert cell["win"] + cell["loss"] + cell["draw"] == pytest.approx(1.0)
            for k in ("win", "loss", "draw", "return", "win_se", "return_se"):
                assert isinstance(cell[k], float)
    with open(os.path.join(out, "checks", f"{game}.json")) as f:
        checks = json.load(f)
    assert checks["every_pair_all_episodes_complete"] and checks["pairs_total"] == sum(
        saved["cells"][r][c]["pairs"] for r in range(7) for c in range(7))
