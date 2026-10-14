"""Head-to-head evaluation of the Kuhn and Leduc baselines
(bluffjax.examples.head_to_head.kuhn_leduc).

- Infosets: on every reachable env decision (every deal, first player and
  legal action sequence), the table row looked up from the env is the acting
  player's infoset in the exploitability code's model, with the same legal
  actions and the observation the networks' tables are computed on; the map
  is a bijection onto all 12 / 936 rows.
- Exact outcomes: exact_outcomes equals the exploitability code's
  expected_returns (both of A's positions) and a walk of the env's tree with
  the same tables (return, win, loss and draw); it is antisymmetric and 0 for
  uniform play.
- Play: every pair plays 32 complete hands, A in seat 0 then 1 of each deal,
  the two hands of a deal from the same reset key (same cards and first
  player, so A plays both positions); deterministic Kuhn policies give the
  returns they must; sampled means match the exact ones within 4.5 SE.
- Loading: from fake run directories of every baseline (random networks,
  CFR tables, ReBeL value networks), the tables are the scripts' own
  policy_array functions, the logged-value check passes and catches a wrong
  value, and the CLI writes the JSON of the output schema with consistent
  cells.
"""

import itertools
import json
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from bluffjax import make
from bluffjax.examples.head_to_head import kuhn_leduc as h2h
from bluffjax.examples.run_utils import save_params
from bluffjax.utils.game_utils import kuhn_exploitability as ke
from bluffjax.utils.game_utils import leduc_exploitability as le


@pytest.fixture(scope="module")
def games():
    return {k: f() for k, f in h2h.GAMES.items()}


@pytest.fixture(scope="module")
def envs(games):
    return {k: make(g.env_id) for k, g in games.items()}


@pytest.fixture(scope="module")
def walks(games, envs):
    return {k: h2h.walk_env(envs[k], games[k]) for k in games}


def random_table(game, seed, sharp=1.0):
    rng = np.random.default_rng(seed)
    t = rng.gamma(sharp, size=game.legal.shape) * game.legal
    return t / t.sum(axis=-1, keepdims=True)


# ------------------------------------------------------------- infosets
@pytest.mark.parametrize("key", ["kuhn", "leduc"])
def test_infosets_are_a_bijection(games, envs, key):
    line = h2h.check_infosets(envs[key], games[key])
    assert f"{games[key].num_rows} infosets" in line
    assert games[key].num_rows == {"kuhn": 12, "leduc": 936}[key]


def test_unknown_code_has_no_row(games):
    g = games["kuhn"]
    assert (g.lookup >= 0).sum() == 12
    g = games["leduc"]
    assert (g.lookup >= 0).sum() == 936


# ---------------------------------------------------------------- exact
def _model_value(key, a, b):
    """A's expected return, mean over A first and A second, from the
    exploitability code's expected_returns."""
    if key == "kuhn":
        pa, pb = ke.policy_from_array(a), ke.policy_from_array(b)
        return (ke.expected_returns(pa, pb)[0] + ke.expected_returns(pb, pa)[1]) / 2
    pa = le._policy_from_array_callable(a, le._KEY_TO_IDX)
    pb = le._policy_from_array_callable(b, le._KEY_TO_IDX)
    return float(le.expected_returns(pa, pb)[0] + le.expected_returns(pb, pa)[1]) / 2


@pytest.mark.parametrize("key", ["kuhn", "leduc"])
def test_exact_matches_expected_returns(games, key):
    g = games[key]
    tables = np.stack([random_table(g, s) for s in range(2)] + [h2h.uniform_table(g)])
    out = h2h.exact_outcomes(g, tables)
    for a, b in [(0, 1), (1, 0)] if key == "leduc" else itertools.product(range(3), repeat=2):
        assert out["return"][a, b] == pytest.approx(_model_value(key, tables[a], tables[b]), abs=1e-10)
    np.testing.assert_allclose(out["return"], -out["return"].T, atol=1e-12)
    np.testing.assert_allclose(out["win"], out["loss"].T, atol=1e-12)
    np.testing.assert_allclose(out["win"] + out["loss"] + out["draw"], 1.0, atol=1e-12)
    assert out["return"][2, 2] == pytest.approx(0.0, abs=1e-12)
    if key == "kuhn":
        np.testing.assert_array_equal(out["draw"], 0.0)


@pytest.mark.parametrize("key", ["kuhn", "leduc"])
def test_exact_matches_env_walk(games, walks, key):
    """The exact outcomes from the env's own tree: for every terminal state,
    the deal's probability times both tables' probabilities of the actions
    along the path (A's table at A's seat), for A in seat 0 and seat 1."""
    g = games[key]
    _, terminals, deals = walks[key]
    tables = np.stack([random_table(g, 10), random_table(g, 11, sharp=0.3)])
    want = h2h.exact_outcomes(g, tables)
    for a, b in [(0, 1), (1, 0), (0, 0)]:
        got = {"return": 0.0, "win": 0.0, "loss": 0.0, "draw": 0.0}
        for a_seat in (0, 1):
            for d, path, ret, steps in zip(terminals["deal"], terminals["path"],
                                           terminals["returns"], terminals["steps"]):
                prob = 1.0 / len(deals)
                for action, (row, seat) in zip(path, steps):
                    prob *= tables[a if seat == a_seat else b][row, action]
                r = ret[a_seat]
                got["return"] += prob * r / 2
                got["win"] += prob * (r > 0) / 2
                got["loss"] += prob * (r < 0) / 2
                got["draw"] += prob * (r == 0) / 2
        for k, v in got.items():
            assert v == pytest.approx(want[k][a, b], abs=1e-9), (k, a, b)


# ----------------------------------------------------------------- play
def _play(env, game, tables, pairs, seed=0, num_deals=h2h.NUM_DEALS):
    pairs = np.asarray(pairs)
    out = jax.jit(
        lambda t, a, b, k: h2h.play_pairs(env, game, t, a, b, k, num_deals)
    )(jnp.asarray(tables), pairs[:, 0], pairs[:, 1], jax.random.PRNGKey(seed))
    return {k: np.asarray(v) for k, v in out.items()}


def test_kuhn_deterministic_policies(games, envs):
    g, env = games["kuhn"], envs["kuhn"]
    bet = np.tile([0.0, 1.0], (12, 1))
    check = np.tile([1.0, 0.0], (12, 1))
    out = _play(env, g, np.stack([bet, check]), [(0, 1), (1, 0)])
    h2h.check_episodes(out)
    # always bet vs always pass: A bets first (B folds) or B checks, A
    # bets, B folds; A wins the ante every hand
    np.testing.assert_array_equal(out["a_return"][0], 1.0)
    np.testing.assert_array_equal(out["a_return"][1], -1.0)
    deals = out["deal"][0]
    start = deals[:, _kuhn_start_column(env)]
    position = (out["a_seat"][0] - start) % 2
    np.testing.assert_array_equal(out["length"][0], np.where(position == 0, 2, 3))
    np.testing.assert_array_equal(position[0::2], 1 - position[1::2])


def _kuhn_start_column(env):
    """Column of start_player_idx in play_pairs' flattened initial state."""
    state, _ = env.reset(jax.random.PRNGKey(0))
    leaves, cols = jax.tree_util.tree_leaves_with_path(state), 0
    for path, x in leaves:
        if "start_player_idx" in jax.tree_util.keystr(path):
            return cols
        cols += int(np.size(x))
    raise AssertionError


@pytest.mark.parametrize("key", ["kuhn", "leduc"])
def test_mirrored_hands_reuse_the_deal(games, envs, key):
    g, env = games[key], envs[key]
    tables = np.stack([random_table(g, 0), random_table(g, 1)])
    out = _play(env, g, tables, [(0, 1), (1, 0), (0, 0)], seed=3)
    h2h.check_episodes(out)
    assert out["a_return"].shape == (3, h2h.EPISODES_PER_PAIR)
    # the deal really comes from the reset key: same key, same cards and
    # first player; and different deals differ
    reset_keys, act_keys = h2h.mirrored_keys(jax.random.PRNGKey(5), h2h.NUM_DEALS)
    states, _ = jax.vmap(env.reset)(reset_keys)
    cards = np.asarray(states.agent_hands if key == "kuhn" else states.shuffled_deck[:, :3])
    start = np.asarray(states.start_player_idx)
    np.testing.assert_array_equal(cards[0::2], cards[1::2])
    np.testing.assert_array_equal(start[0::2], start[1::2])
    assert len({tuple(c) + (s,) for c, s in zip(cards[0::2], start[0::2])}) > 1
    assert len({tuple(np.asarray(k)) for k in act_keys}) == h2h.EPISODES_PER_PAIR
    # play_pairs: within a pair and across pairs, the deals differ
    deals = out["deal"]
    assert len({deals[0, 2 * d].tobytes() for d in range(h2h.NUM_DEALS)}) > 1
    assert not np.array_equal(deals[0], deals[1])


@pytest.mark.parametrize("key", ["kuhn", "leduc"])
def test_sampled_returns_match_exact(games, envs, key):
    g, env = games[key], envs[key]
    tables = np.stack([random_table(g, 20, sharp=0.5), random_table(g, 21), h2h.uniform_table(g)])
    pairs = [(0, 1), (1, 0), (0, 2), (2, 2)]
    num_deals = 4000
    out = _play(env, g, tables, pairs, seed=7, num_deals=num_deals)
    assert out["done"].all() and (out["invalid"] == 0).all()
    exact = h2h.exact_outcomes(g, tables)
    for p, (a, b) in enumerate(pairs):
        ret = out["a_return"][p]
        deal_means = ret.reshape(num_deals, 2).mean(axis=1)  # mirrored hands are correlated
        se = deal_means.std(ddof=1) / math.sqrt(num_deals)
        assert abs(ret.mean() - exact["return"][a, b]) < 4.5 * se, (key, a, b)
        for name, x in (("win", ret > 0), ("loss", ret < 0), ("draw", ret == 0)):
            m = x.reshape(num_deals, 2).mean(axis=1)
            se = max(m.std(ddof=1) / math.sqrt(num_deals), 1e-9)
            assert abs(m.mean() - exact[name][a, b]) < 4.5 * se + 1e-12, (key, name, a, b)


def test_unknown_infoset_is_flagged(games, envs):
    g = games["kuhn"]
    lookup = g.lookup.copy()
    lookup[lookup == 0] = -1  # infoset "0" (J, first to act) unknown
    broken = h2h.dataclasses.replace(g, lookup=lookup)
    out = _play(envs["kuhn"], broken, np.stack([h2h.uniform_table(g)] * 2), [(0, 1)])
    assert out["invalid"].sum() > 0
    with pytest.raises(AssertionError):
        h2h.check_episodes(out)


# -------------------------------------------------------------- helpers
def test_cell_pairs():
    assert h2h.cell_pairs(5, 5, True) == [(i, j) for i in range(5) for j in range(5) if i != j]
    assert len(h2h.cell_pairs(5, 5, False)) == 25
    assert h2h.cell_pairs(1, 1, True) == [(0, 0)]
    assert h2h.cell_pairs(1, 5, False) == [(0, j) for j in range(5)]


def test_normalize_table(games):
    g = games["leduc"]
    t = random_table(g, 0)
    np.testing.assert_allclose(h2h.normalize_table(g, t), t, atol=1e-15)
    with pytest.raises(ValueError):
        h2h.normalize_table(g, np.ones_like(t))  # illegal mass, rows sum to 3
    with pytest.raises(ValueError):
        h2h.normalize_table(g, t[:-1])
    np.testing.assert_allclose(h2h.uniform_table(g).sum(-1), 1.0)


# --------------------------------------------- fake runs and the CLI
NUM_SEEDS = 2


def _write_run(root, game, suffix, config, files, logged):
    """A run directory: config.yaml, results.jsonl (logged: list of
    records) and {stage: {name: tree}} checkpoints."""
    run_dir = os.path.join(root, game, f"{game}_{suffix}_2026-01-01_00-00-00")
    os.makedirs(run_dir)
    config = {"job_type": f"{game}_{suffix}", "num_seeds": NUM_SEEDS, **config}
    OmegaConf.save(OmegaConf.create(config), os.path.join(run_dir, "config.yaml"))
    for stage, trees in files.items():
        for name, tree in trees.items():
            save_params(run_dir, stage, name, tree)
    with open(os.path.join(run_dir, "results.jsonl"), "w") as f:
        for record in logged:
            f.write(json.dumps(record) + "\n")
    return run_dir


def _fake_runs(root, game: h2h.Game, with_logs=True):
    """Every baseline of the game with random networks / tables; the full
    checkpoints' exploitability and value_vs_random logged as the scripts do
    (computed with the scripts' own functions)."""
    from bluffjax.networks.mlp import ActorCriticDiscreteMLP, ActorDiscreteMLP, QNetworkDiscreteMLP

    key = game.key
    expected = {}  # participant -> list of tables (the scripts' functions)
    rl = {"ppo_nfsp": ("avg_policy", ActorCriticDiscreteMLP, False),
          "pqn_nfsp": ("avg_policy", ActorDiscreteMLP, False),
          "ppo": ("policy", ActorCriticDiscreteMLP, False),
          "pqn": ("q_network", QNetworkDiscreteMLP, True)}
    labels = dict((s, l) for l, s in h2h.BASELINES)
    for n, (suffix, (name, cls, greedy)) in enumerate(rl.items()):
        net = cls(action_dim=game.num_actions, hidden_dim=16)
        files, logged = {"half": {}, "full": {}}, []
        for stage in h2h.STAGES:
            tables = []
            for i in range(NUM_SEEDS):
                params = net.init(jax.random.PRNGKey(100 * n + 10 * i + (stage == "full")),
                                  jnp.zeros(game.obs_dim))
                params = jax.tree_util.tree_map(lambda x: 3.0 * x, params)
                files[stage][f"{name}_{i}"] = params
                if key == "kuhn":
                    f = ke.policy_array_from_qnetwork if greedy else ke.policy_array_from_network
                    t = np.asarray(f(net.apply, params), dtype=np.float64)
                else:
                    f = le.policy_array_from_qnetwork if greedy else le.policy_array_from_network
                    apply = net.apply if greedy else (
                        lambda p, o: (net.apply(p, o), None) if cls is ActorDiscreteMLP
                        else net.apply(p, o))
                    t = np.asarray(f(apply, params, le._INFOSET_LIST), dtype=np.float64)
                tables.append(t)
                if stage == "full" and with_logs and (key == "kuhn" or i == 0):
                    logged.append({"seed_index": i, "update_step": 9,
                                   "exploitability": game.exploitability(t),
                                   "value_vs_random": game.value_vs_random(t)})
            expected[f"{labels[suffix]} ({stage})"] = tables
        _write_run(root, key, suffix, {"fc_dim_size": 16, "num_update_steps": 10,
                                       "half_update_steps": 5}, files, logged)
    # CFR: one table per stage
    if key == "kuhn":
        from bluffjax.examples.kuhn import kuhn_cfr as cfr
    else:
        from bluffjax.examples.leduc import leduc_cfr as cfr
    files, logged = {}, []
    for s, stage in enumerate(h2h.STAGES):
        t = random_table(game, 50 + s)
        files[stage] = {"average_policy": cfr.policy_checkpoint(t, 5 * (s + 1))}
        expected[f"CFR ({stage})"] = [t]
        if with_logs:
            logged.append({"seed_index": 0, "iteration": 5 * (s + 1),
                           "exploitability": game.exploitability(t),
                           "value_vs_random": game.value_vs_random(t)})
    _write_run(root, key, "cfr", {"iterations": 10, "half_iterations": 5}, files, logged)
    # Deep CFR
    if key == "kuhn":
        from bluffjax.examples.kuhn import kuhn_deep_cfr as dcfr
        make_net = lambda: dcfr.MLP((16, 16), dcfr.NUM_ACTIONS)
        x0 = dcfr.INFOSET_TENSORS[:1]
    else:
        from bluffjax.examples.leduc import leduc_deep_cfr as dcfr
        make_net = lambda: dcfr.policy_network((16, 16))
        x0 = jnp.zeros((1, dcfr.INFO_STATE_SIZE))
    files = {"half": {}, "full": {}}
    for s, stage in enumerate(h2h.STAGES):
        tables = []
        for i in range(NUM_SEEDS):
            params = make_net().init(jax.random.PRNGKey(700 + 10 * i + s), x0)
            files[stage][f"avg_policy_{i}"] = params
            tables.append(dcfr.policy_array(params, (16, 16)))
        expected[f"Deep CFR ({stage})"] = tables
    _write_run(root, key, "deep_cfr", {"iterations": 10, "half_iterations": 5,
                                       "hidden_sizes": [16, 16]}, files, [])
    # ReBeL (Kuhn)
    if key == "kuhn":
        from bluffjax.examples.kuhn import kuhn_rebel as rebel

        net = rebel.ValueNetwork(hidden_dim=16)
        structure = rebel.subgame_structure(1)
        files, logged = {"half": {}, "full": {}}, []
        for s, stage in enumerate(h2h.STAGES):
            tables = []
            for i in range(NUM_SEEDS):
                params = net.init(jax.random.PRNGKey(900 + 10 * i + s), jnp.zeros(rebel.PBS_DIM))
                files[stage][f"value_net_{i}"] = params
                policy = rebel.test_time_policy(rebel.network_value_fn(net, params), structure, 8)
                t = rebel.policy_array(np.asarray(policy))
                tables.append(t)
                if with_logs:
                    logged.append({"seed_index": i, "updates_completed": 5 * (s + 1),
                                   "exploitability": game.exploitability(t),
                                   "value_vs_random": game.value_vs_random(t)})
            expected[f"ReBeL ({stage})"] = tables
        _write_run(root, key, "rebel", {"value_hidden_dim": 16, "max_depth": 1, "cfr_iters": 8,
                                        "num_update_steps": 10, "half_update_steps": 5},
                   files, logged)
    expected["Random"] = [h2h.uniform_table(game)]
    return expected


@pytest.fixture(scope="module")
def kuhn_runs(tmp_path_factory, games):
    root = str(tmp_path_factory.mktemp("runs"))
    return root, _fake_runs(root, games["kuhn"])


def test_loaded_tables_are_the_scripts(games, kuhn_runs):
    root, expected = kuhn_runs
    participants = h2h.load_participants(root, games["kuhn"])
    assert [p.name for p in participants] == list(expected)
    assert [p.name for p in participants][:4] == [
        "PPO-NFSP (half)", "PPO-NFSP (full)", "PQN-NFSP (half)", "PQN-NFSP (full)"]
    for p in participants:
        assert len(p.checkpoints) == len(expected[p.name])
        for c, t in zip(p.checkpoints, expected[p.name]):
            np.testing.assert_allclose(c.table, t, atol=1e-6, err_msg=p.name)
            if p.name.startswith("PQN ("):
                np.testing.assert_array_equal(c.table, t)  # greedy: exactly the same
    lines = h2h.check_logged(games["kuhn"], participants)
    assert any(l.startswith("PQN (full): 2/2 checkpoints checked") for l in lines)
    assert any(l.startswith("PQN (half): 0/2") for l in lines)
    assert any(l.startswith("CFR (half): 1/1") for l in lines)
    assert any(l.startswith("ReBeL (half): 2/2") for l in lines)
    # a table that isn't the evaluated policy is caught
    pqn = next(p for p in participants if p.name == "PQN (full)")
    pqn.checkpoints[0].table = pqn.checkpoints[0].table[:, ::-1].copy()
    with pytest.raises(AssertionError, match="PQN"):
        h2h.check_logged(games["kuhn"], participants)


def test_two_runs_of_a_baseline_are_an_error(tmp_path, games):
    for n in range(2):
        d = tmp_path / "kuhn" / f"kuhn_ppo_{n}"
        d.mkdir(parents=True)
        OmegaConf.save(OmegaConf.create({"job_type": "kuhn_ppo"}), str(d / "config.yaml"))
    with pytest.raises(ValueError, match="several runs"):
        h2h.find_run(str(tmp_path), games["kuhn"], "ppo")
    # job_type, not the directory name, picks the run: kuhn_ppo_nfsp isn't kuhn_ppo
    assert h2h.find_run(str(tmp_path), games["kuhn"], "ppo_nfsp") is None


def _check_json(data, expected, game_key):
    names = list(expected)
    assert data["game_key"] == game_key and data["episodes_per_pair"] == 32
    assert data["participants"] == names
    assert data["checkpoints"] == {k: len(v) for k, v in expected.items()}
    assert set(data) == {"game", "game_key", "units", "episodes_per_pair", "protocol",
                         "participants", "checkpoints", "cells", "commit", "runs_root",
                         "generated"}
    cells = data["cells"]
    assert len(cells) == len(names) and all(len(r) == len(names) for r in cells)
    for r, a in enumerate(names):
        for c, b in enumerate(names):
            cell = cells[r][c]
            na, nb = len(expected[a]), len(expected[b])
            pairs = na * nb - (na if r == c and na > 1 else 0)
            assert cell["pairs"] == pairs and cell["episodes"] == 32 * pairs
            assert set(cell) == {"win", "loss", "draw", "return", "win_se", "return_se",
                                 "pairs", "episodes", "extra"}
            assert set(cell["extra"]) == {"exact_return", "exact_win", "exact_loss", "exact_draw"}
            for k in ("win", "loss", "draw", "return"):
                assert isinstance(cell[k], float) and math.isfinite(cell[k])
            assert cell["win"] + cell["loss"] + cell["draw"] == pytest.approx(1.0)
            assert math.isnan(cell["return_se"]) == (pairs == 1)
            assert cell["extra"]["exact_return"] == pytest.approx(
                -cells[c][r]["extra"]["exact_return"], abs=1e-9)
            assert cell["extra"]["exact_win"] == pytest.approx(
                cells[c][r]["extra"]["exact_loss"], abs=1e-9)


def test_cli_kuhn(tmp_path, kuhn_runs, games):
    root, expected = kuhn_runs
    out = tmp_path / "out"
    h2h.main([root, str(out), "--games", "kuhn", "--seed", "1"])
    data = json.loads((out / "kuhn.json").read_text())
    _check_json(data, expected, "kuhn")
    # the exact returns are those of the expected tables
    tables = np.concatenate([np.stack(v) for v in expected.values()])
    exact = h2h.exact_outcomes(games["kuhn"], tables)["return"]
    off = np.cumsum([0] + [len(v) for v in expected.values()])
    names = list(expected)
    r, c = names.index("PQN (full)"), names.index("CFR (half)")
    want = exact[off[r]:off[r + 1], off[c]:off[c + 1]].mean()
    assert data["cells"][r][c]["extra"]["exact_return"] == pytest.approx(want, abs=1e-6)
    # the same seed gives the same file (but the generation time)
    h2h.main([root, str(tmp_path / "again"), "--games", "kuhn", "--seed", "1"])
    again = json.loads((tmp_path / "again" / "kuhn.json").read_text())
    assert json.dumps(again["cells"]) == json.dumps(data["cells"])


def test_cli_leduc(tmp_path, games):
    root = str(tmp_path / "runs")
    expected = _fake_runs(root, games["leduc"])
    assert "ReBeL (half)" not in expected
    h2h.main([root, str(tmp_path / "out"), "--games", "leduc"])
    data = json.loads((tmp_path / "out" / "leduc.json").read_text())
    _check_json(data, expected, "leduc")
    assert data["participants"][-1] == "Random"
