"""Short real runs of every Kuhn Poker training script (CPU): the run
directory, results.jsonl and the half / full checkpoints.

- PPO-NFSP, PQN-NFSP, PPO and PQN (bluffjax/examples/kuhn): run() writes
  config.yaml, results.jsonl with one record per update and seed (env_steps
  in each, exploitability and the game/ policy metrics at every evaluation,
  game_br/ for NFSP's best response) and exactly the half/ and full/
  checkpoint files, which load strictly into the script's networks. The half
  checkpoint equals the networks of a run stopped after num_half updates
  (with the schedules of the whole run), the policy metrics logged after
  update num_half and after the last update are those of the half and full
  checkpoints, and the full checkpoint differs from the half one.
- CFR: the half / full tables are the average strategy after iterations // 2
  and all iterations, the iteration count is a flag (default 100), and the
  logged metrics are those of the logged tables.
- Deep CFR: seeds run one after the other with seed + i; the policy network is
  also trained (and evaluated) at iterations // 2; the half checkpoint is the
  final policy network of a run of iterations // 2 iterations.
- ReBeL: the half value network equals the final one of a run of
  num_update_steps // 2 updates, and the logged metrics are those of the
  test-time policy of the checkpointed networks.
"""

import functools
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax.examples import run_utils
from bluffjax.examples.kuhn import (
    kuhn_cfr,
    kuhn_deep_cfr,
    kuhn_ppo,
    kuhn_ppo_nfsp,
    kuhn_pqn,
    kuhn_pqn_nfsp,
    kuhn_rebel,
)
from bluffjax.metrics.kuhn_poker import METRIC_NAMES, policy_metrics
from bluffjax.networks.mlp import (
    ActorCriticDiscreteMLP,
    ActorDiscreteMLP,
    QNetworkDiscreteMLP,
)
from bluffjax.utils.game_utils import kuhn_exploitability as ke

GAME = [f"game/{name}" for name in METRIC_NAMES]
GAME_BR = [f"game_br/{name}" for name in METRIC_NAMES]
H = 16  # hidden units of the tiny networks

# script -> (module, {checkpoint name: (network class, policy helper)},
#            {checkpoint name: metric prefix}, final-state attribute per name)
SCRIPTS = {
    "kuhn_ppo_nfsp": (
        kuhn_ppo_nfsp,
        {
            "avg_policy": (ActorCriticDiscreteMLP, ke.policy_array_from_network),
            "br": (ActorCriticDiscreteMLP, ke.policy_array_from_network),
        },
        {"avg_policy": "game", "br": "game_br"},
        {"avg_policy": "avg_train_state", "br": "br_train_state"},
    ),
    "kuhn_pqn_nfsp": (
        kuhn_pqn_nfsp,
        {
            "avg_policy": (ActorDiscreteMLP, ke.policy_array_from_network),
            "br": (QNetworkDiscreteMLP, ke.policy_array_from_qnetwork),
        },
        {"avg_policy": "game", "br": "game_br"},
        {"avg_policy": "avg_train_state", "br": "br_train_state"},
    ),
    "kuhn_ppo": (
        kuhn_ppo,
        {"policy": (ActorCriticDiscreteMLP, ke.policy_array_from_network)},
        {"policy": "game"},
        {"policy": "train_state"},
    ),
    "kuhn_pqn": (
        kuhn_pqn,
        {"q_network": (QNetworkDiscreteMLP, ke.policy_array_from_qnetwork)},
        {"q_network": "game"},
        {"q_network": "train_state"},
    ),
}
NUM_UPDATES, NUM_ENVS, NUM_STEPS, NUM_SEEDS = 5, 16, 8, 2
NUM_HALF = 2  # half_update_steps(5)


def rl_config(name: str, save_dir: str) -> dict:
    config = dict(
        num_timesteps=NUM_UPDATES * NUM_ENVS * NUM_STEPS,
        num_envs=NUM_ENVS,
        num_steps_per_env_per_update=NUM_STEPS,
        num_epochs=1,
        num_minibatches=2,
        lr=5e-3,  # large, so that every update moves the networks
        anneal_lr=True,
        gamma=1.0,
        max_grad_norm=0.5,
        fc_dim_size=H,
        eval_interval=1,
        save_dir=save_dir,
        save_checkpoints=True,
        env_name="kuhn_poker",
        seed=0,
        num_seeds=NUM_SEEDS,
        env_kwargs={},
        project="bluffjax",
        job_type=name,
        wandb=False,
    )
    if "nfsp" in name:
        config.update(
            anticipatory_eta=0.3,
            sl_reservoir_capacity=500,
            sl_batch_size=32,
            sl_num_steps_per_update=2,
            sl_lr=3e-3,
        )
    if "ppo" in name:
        config.update(gae_lambda=0.95, clip_eps=0.2, vf_clip=0.2, ent_coef=0.01, vf_coef=1.0)
    else:
        config.update(start_e=1.0, end_e=0.05, exploration_fraction=0.5, q_lambda=0.9)
    return config


def _files(run_dir: str) -> set[str]:
    return {
        os.path.relpath(os.path.join(d, f), run_dir)
        for d, _, files in os.walk(run_dir)
        for f in files
    }


def _max_diff(a, b) -> float:
    leaves = zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    return max(float(np.abs(np.asarray(x) - np.asarray(y)).max()) for x, y in leaves)


def _assert_metrics_close(record: dict, prefix: str, expected: dict, atol: float):
    for name in METRIC_NAMES:
        assert record[f"{prefix}/{name}"] == pytest.approx(expected[name], abs=atol), name


def _by_seed(records: list[dict], key: str) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r in records:
        out.setdefault(r["seed_index"], []).append(r)
    return {i: sorted(rs, key=lambda r: r[key]) for i, rs in out.items()}


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_rl_run_dir_metrics_and_checkpoints(monkeypatch, tmp_path, name):
    mod, networks, prefixes, attrs = SCRIPTS[name]
    config = rl_config(name, str(tmp_path / "runs"))
    run_dir = mod.run(config)
    assert config["num_update_steps"] == NUM_UPDATES
    assert config["half_update_steps"] == NUM_HALF
    assert os.path.dirname(run_dir) == str(tmp_path / "runs")
    assert os.path.basename(run_dir).startswith(f"{name}_")
    assert _files(run_dir) == {"config.yaml", "results.jsonl"} | {
        f"{stage}/{net}_{i}.msgpack"
        for stage in ("half", "full")
        for net in networks
        for i in range(NUM_SEEDS)
    }

    records = _by_seed(run_utils.read_results(run_dir), "update_step")
    assert sorted(records) == list(range(NUM_SEEDS))
    for rows in records.values():
        assert [r["update_step"] for r in rows] == list(range(NUM_UPDATES))
        for u, r in enumerate(rows):
            assert r["env_steps"] == (u + 1) * NUM_ENVS * NUM_STEPS
            assert set(GAME) <= set(r)  # eval_interval = 1
            assert (set(GAME_BR) <= set(r)) == ("nfsp" in name)
            for key in GAME + GAME_BR:
                if key in r:
                    assert math.isfinite(r[key])
            for k in ("p0_bet_jack", "p0_bet_king", "p1_bet_jack", "p0_call_queen"):
                assert 0.0 <= r[f"game/{k}"] <= 1.0
            assert 0.0 <= r["game/p0_alpha"] <= 1 / 3

    # checkpoints: strict loading into the script's networks; the logged
    # metrics after update num_half - 1 (the num_half-th) and after the last
    # one are those of the half and full checkpoints
    obs = jnp.zeros((9,))
    loaded = {}
    for net, (cls, to_policy) in networks.items():
        network = cls(action_dim=2, hidden_dim=H)
        template = network.init(jax.random.PRNGKey(1), obs)
        for i in range(NUM_SEEDS):
            for stage, update in (("half", NUM_HALF - 1), ("full", NUM_UPDATES - 1)):
                path = run_utils.checkpoint_path(run_dir, stage, net, i)
                params = run_utils.load_params(path, template)
                loaded[stage, net, i] = params
                table = np.asarray(to_policy(network.apply, params))
                record = records[i][update]
                _assert_metrics_close(record, prefixes[net], policy_metrics(table), 1e-5)
                expl_key = {
                    "kuhn_ppo": "exploitability",
                    "kuhn_pqn": "exploitability",
                }.get(name, f"exploitability_{'avg' if net == 'avg_policy' else 'br'}_policy")
                assert record[expl_key] == pytest.approx(
                    ke.exploitability_from_policy_array(table), abs=1e-5
                )
            assert _max_diff(loaded["half", net, i], loaded["full", net, i]) > 1e-5
        wrong = (ActorDiscreteMLP if cls is not ActorDiscreteMLP else QNetworkDiscreteMLP)
        with pytest.raises(ValueError):
            run_utils.load_params(
                run_utils.checkpoint_path(run_dir, "full", net, 0),
                wrong(action_dim=2, hidden_dim=H).init(jax.random.PRNGKey(1), obs),
            )

    # the half checkpoint is the networks of the same run stopped after
    # num_half updates (schedules of the whole run: config is the run's)
    monkeypatch.setattr(mod, "RESULTS_LOG", run_utils.ResultsLog(str(tmp_path)))
    keys = jax.random.split(jax.random.PRNGKey(config["seed"]), NUM_SEEDS)
    train = mod.make_train(config)
    partial = jax.jit(jax.vmap(functools.partial(train, num_updates=NUM_HALF)))(
        keys, jnp.arange(NUM_SEEDS)
    )
    jax.effects_barrier()
    assert (np.asarray(partial.update_step) == NUM_HALF).all()
    for net in networks:
        params = getattr(partial, attrs[net]).params
        for i in range(NUM_SEEDS):
            at_half = jax.tree_util.tree_map(lambda x: np.asarray(x)[i], params)
            assert _max_diff(loaded["half", net, i], at_half) < 1e-6
    # its half copy in the runner state is that same state
    half_state = partial.half if "nfsp" in name else partial.half_params
    if "nfsp" in name:
        assert _max_diff(half_state.avg_params, partial.avg_train_state.params) == 0.0
    else:
        assert _max_diff(half_state, partial.train_state.params) == 0.0


def test_rl_no_checkpoints(tmp_path):
    config = rl_config("kuhn_ppo", str(tmp_path))
    config["save_checkpoints"] = False
    run_dir = kuhn_ppo.run(config)
    assert _files(run_dir) == {"config.yaml", "results.jsonl"}


# ------------------------------------------------------------------ CFR
def test_cfr_run(tmp_path):
    assert kuhn_cfr.parse_args([]).iterations == 100
    args = ["--iterations", "10", "--log_interval", "3", "--save_dir", str(tmp_path)]
    run_dir = kuhn_cfr.run(kuhn_cfr.parse_args(args))
    assert _files(run_dir) == {
        "config.yaml",
        "results.jsonl",
        "half/average_policy.msgpack",
        "full/average_policy.msgpack",
    }
    template = np.zeros((12, 2))
    half = run_utils.load_params(os.path.join(run_dir, "half/average_policy.msgpack"), template)
    full = run_utils.load_params(os.path.join(run_dir, "full/average_policy.msgpack"), template)
    np.testing.assert_array_equal(half, kuhn_cfr.run_cfr(5, log_interval=100).full_policy)
    np.testing.assert_array_equal(full, kuhn_cfr.run_cfr(10, log_interval=100).full_policy)
    assert np.abs(half - full).max() > 1e-4
    records = run_utils.read_results(run_dir)
    assert [r["iteration"] for r in records] == [0, 3, 6, 9, 10]
    assert all(r["seed_index"] == 0 for r in records)
    uniform = policy_metrics(np.full((12, 2), 0.5))
    _assert_metrics_close(records[0], "game", uniform, 0.0)
    _assert_metrics_close(records[-1], "game", policy_metrics(full), 0.0)
    assert records[-1]["exploitability"] == ke.exploitability_from_policy_array(full)
    # an odd count: the half table after 3 of 7 iterations
    run_dir = kuhn_cfr.run(kuhn_cfr.parse_args(["--iterations", "7", "--save_dir", str(tmp_path)]))
    half = run_utils.load_params(os.path.join(run_dir, "half/average_policy.msgpack"), template)
    np.testing.assert_array_equal(half, kuhn_cfr.run_cfr(3, log_interval=100).full_policy)


# ------------------------------------------------------------- Deep CFR
def _deep_cfr_args(tmp_path, *extra):
    return kuhn_deep_cfr.parse_args(
        [
            "--traversals", "4",
            "--hidden_sizes", "8", "8",
            "--batch_size", "16",
            "--advantage_train_steps", "5",
            "--policy_train_steps", "10",
            "--memory_capacity", "500",
            "--save_dir", str(tmp_path),
            *extra,
        ]
    )


def test_deep_cfr_run(tmp_path):
    assert kuhn_deep_cfr.parse_args([]).num_seeds == 5
    config = _deep_cfr_args(tmp_path, "--iterations", "6", "--eval_every", "4", "--num_seeds", "2")
    run_dir = kuhn_deep_cfr.run(config)
    assert _files(run_dir) == {"config.yaml", "results.jsonl"} | {
        f"{stage}/avg_policy_{i}.msgpack" for stage in ("half", "full") for i in range(2)
    }
    records = _by_seed(run_utils.read_results(run_dir), "iteration")
    assert sorted(records) == [0, 1]
    network = kuhn_deep_cfr.MLP((8, 8), 2)
    template = network.init(jax.random.PRNGKey(0), kuhn_deep_cfr.INFOSET_TENSORS[:1])
    for i, rows in records.items():
        assert [r["iteration"] for r in rows] == list(range(1, 7))
        assert [r["traversals"] for r in rows] == [8 * it for it in range(1, 7)]
        evaluated = [r["iteration"] for r in rows if "exploitability" in r]
        assert evaluated == [1, 3, 4, 6]  # first, half, eval_every, last
        for r in rows:
            assert (set(GAME) <= set(r)) == ("exploitability" in r)
        for stage, it in (("half", 3), ("full", 6)):
            params = run_utils.load_params(
                run_utils.checkpoint_path(run_dir, stage, "avg_policy", i), template
            )
            table = kuhn_deep_cfr.policy_array(params, (8, 8))
            _assert_metrics_close(rows[it - 1], "game", policy_metrics(table), 1e-6)
            assert rows[it - 1]["exploitability"] == pytest.approx(
                ke.exploitability_from_policy_array(table), abs=1e-6
            )

    # the half network is the final one of a 3-iteration run with the same
    # seeds, and seed index i uses seed + i
    short = kuhn_deep_cfr.run(
        _deep_cfr_args(tmp_path, "--iterations", "3", "--eval_every", "4", "--num_seeds", "2")
    )
    seed1 = kuhn_deep_cfr.run(
        _deep_cfr_args(
            tmp_path, "--iterations", "3", "--eval_every", "4", "--num_seeds", "1", "--seed", "1"
        )
    )

    def load(d, stage, i):
        return run_utils.load_params(run_utils.checkpoint_path(d, stage, "avg_policy", i), template)

    for i in range(2):
        assert _max_diff(load(run_dir, "half", i), load(short, "full", i)) == 0.0
        assert _max_diff(load(run_dir, "half", i), load(run_dir, "full", i)) > 1e-4
    assert _max_diff(load(seed1, "full", 0), load(short, "full", 1)) == 0.0
    assert _max_diff(load(short, "full", 0), load(short, "full", 1)) > 1e-4


# ---------------------------------------------------------------- ReBeL
def rebel_config(save_dir: str, num_update_steps: int) -> dict:
    return dict(
        num_update_steps=num_update_steps,
        num_subgames_per_update=4,
        cfr_iterations=4,
        max_depth=1,
        random_action_prob=0.25,
        value_lr=1e-3,
        value_lr_halving_interval=0,
        value_hidden_dim=8,
        max_grad_norm=5.0,
        replay_capacity=32,
        batch_size=8,
        num_value_train_steps=2,
        log_interval=2,
        save_checkpoints=True,
        save_dir=save_dir,
        env_name="kuhn_poker",
        seed=0,
        num_seeds=2,
        env_kwargs={},
        project="bluffjax",
        job_type="kuhn_rebel",
        wandb=False,
    )


def test_rebel_run(tmp_path):
    config = rebel_config(str(tmp_path), 4)
    run_dir = kuhn_rebel.run(config)
    assert config["half_update_steps"] == 2
    assert _files(run_dir) == {"config.yaml", "results.jsonl", "curves.json"} | {
        f"{stage}/value_network_{i}.msgpack" for stage in ("half", "full") for i in range(2)
    }
    records = _by_seed(run_utils.read_results(run_dir), "num_subgames")
    network = kuhn_rebel.ValueNetwork(hidden_dim=8)
    template = network.init(jax.random.PRNGKey(0), jnp.zeros(kuhn_rebel.PBS_DIM))
    structure = kuhn_rebel.subgame_structure(1)

    @jax.jit
    def table_of(params):
        value_fn = kuhn_rebel.network_value_fn(network, params)
        return kuhn_rebel.test_time_policy(value_fn, structure, 4)

    short = kuhn_rebel.run(rebel_config(str(tmp_path), 2))
    for i, rows in records.items():
        assert [r["num_subgames"] for r in rows] == [0, 8, 16]
        assert "value_loss" not in rows[0] and "value_loss" in rows[1]
        for r in rows:
            assert set(GAME) <= set(r)
        loaded = {}
        for stage, row in (("half", rows[1]), ("full", rows[2])):
            params = run_utils.load_params(
                run_utils.checkpoint_path(run_dir, stage, "value_network", i), template
            )
            loaded[stage] = params
            table = kuhn_rebel.policy_array(np.asarray(table_of(params)))
            _assert_metrics_close(row, "game", policy_metrics(table), 1e-5)
            assert row["exploitability"] == pytest.approx(
                ke.exploitability_from_policy_array(table), abs=1e-5
            )
        short_full = run_utils.load_params(
            run_utils.checkpoint_path(short, "full", "value_network", i), template
        )
        assert _max_diff(loaded["half"], short_full) < 1e-6
        assert _max_diff(loaded["half"], loaded["full"]) > 1e-4


def test_rebel_policy_array():
    """policy_array maps (public node, card) to the INFOSETS rows."""
    rs = np.random.RandomState(0)
    policy = rs.rand(kuhn_rebel.NUM_NODES, 3, 2)
    table = kuhn_rebel.policy_array(policy)
    for i, key in enumerate(ke.INFOSETS):
        history = tuple(ke.ACTION_PASS if c == "p" else ke.ACTION_BET for c in key[1:])
        row = policy[kuhn_rebel.HISTORIES.index(history), int(key[0])]
        np.testing.assert_allclose(table[i], row / row.sum())
    assert kuhn_rebel.policy_exploitability(policy) == ke.exploitability(
        lambda k: table[ke.INFOSETS.index(k)]
    )
