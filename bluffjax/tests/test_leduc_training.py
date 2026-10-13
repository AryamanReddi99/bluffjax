"""Run directory, results log, game metrics and half/full checkpoints of the
Leduc training scripts (bluffjax/examples/leduc), from tiny CPU runs of their
main() (Hydra or argparse, exactly as from the command line).

For every script:
- results.jsonl has one record per update (iteration) and seed, with the
  sample counter, and the exploitability, value against random and game/
  metrics (game_br/ for NFSP) exactly at the evaluation points; the logged
  game metrics, exploitability and value against random are those of the
  networks (tables) saved in the checkpoints, recomputed from the loaded
  files, so they are metrics of the right policy. The RL scripts also log
  the evaluation against random of every policy, over exactly
  compare_episodes complete hands, in every record (compare_interval 1).
- half/ and full/ hold the documented files, which load strictly into the
  script's networks; the half checkpoint is the state after exactly
  half_update_steps(budget) updates of the same run (a run stopped there
  gives the same networks) and the full one the final state.
- The rollout statistics of the RL scripts have the names shared by every
  game (episodes_completed, ep_length_avg, return_per_episode[_br] for NFSP,
  first_player_return_per_episode and abs_return_per_episode in self-play);
  the output policy's exact evaluation is exploitability / value_vs_random,
  NFSP's BR adds br.
- CFR and Deep CFR evaluate after iteration 1, every eval_interval, at the
  half point and at the end, and have the flags, config.yaml and wandb names
  shared with the Kuhn scripts and the RL scripts.
"""

import functools
import glob
import os
import sys

import yaml

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from bluffjax import make
from bluffjax.examples.leduc import (
    leduc_cfr,
    leduc_deep_cfr,
    leduc_ppo,
    leduc_ppo_nfsp,
    leduc_pqn,
    leduc_pqn_nfsp,
)
from bluffjax.examples.run_utils import ResultsLog, load_params, read_results
from bluffjax.metrics.leduc_holdem import TOTAL_NAMES, log_metrics
from bluffjax.networks.mlp import (
    ActorCriticDiscreteMLP,
    ActorDiscreteMLP,
    QNetworkDiscreteMLP,
)
from bluffjax.utils.game_utils import leduc_exploitability as leduc
from bluffjax.utils.paths import register_resolvers

U = 6  # updates; half after 3
EVAL_UPDATES = {0, 2, 4, 5}  # eval_interval 2 and the last update
HIDDEN = 16
COMPARE_EPISODES = 6
EVAL_PREFIXES = (
    "win_rate_", "loss_rate_", "draw_rate_", "win_rate_decided_", "avg_return_",
    "avg_episode_length_",
)
# checkpoint name -> name of its evaluation against random (and value_<name>)
VS_RANDOM = {
    "avg_policy": "vs_random", "br": "br_vs_random", "policy": "vs_random",
    "q_network": "vs_random",
}
# rollout statistics of every update (names shared with the other games)
NFSP_STATS = {"episodes_completed", "ep_length_avg", "return_per_episode",
              "return_per_episode_br"}
SELF_PLAY_STATS = {"episodes_completed", "ep_length_avg",
                   "first_player_return_per_episode", "abs_return_per_episode"}


def _greedy(apply):
    return lambda params: np.asarray(
        leduc.policy_array_from_qnetwork(apply, params, leduc._INFOSET_LIST)
    )


def _softmax(apply):
    return lambda params: np.asarray(
        leduc.policy_array_from_network(apply, params, leduc._INFOSET_LIST)
    )


AC = ActorCriticDiscreteMLP(3, HIDDEN)
ACTOR = ActorDiscreteMLP(3, HIDDEN)
QNET = QNetworkDiscreteMLP(3, HIDDEN)
# script -> (module, {checkpoint name: (network, policy fn, logged prefix,
# train-state attribute of the final runner state, half attribute)})
SCRIPTS = {
    "leduc_ppo_nfsp": (leduc_ppo_nfsp, {
        "avg_policy": (AC, _softmax(AC.apply), "game", "avg_train_state", "avg_params"),
        "br": (AC, _softmax(AC.apply), "game_br", "br_train_state", "br_params"),
    }),
    "leduc_pqn_nfsp": (leduc_pqn_nfsp, {
        "avg_policy": (
            ACTOR, _softmax(lambda p, o: (ACTOR.apply(p, o), None)), "game",
            "avg_train_state", "avg_params",
        ),
        "br": (QNET, _greedy(QNET.apply), "game_br", "br_train_state", "br_params"),
    }),
    "leduc_ppo": (leduc_ppo, {
        "policy": (AC, _softmax(AC.apply), "game", "train_state", None),
    }),
    "leduc_pqn": (leduc_pqn, {
        "q_network": (QNET, _greedy(QNET.apply), "game", "train_state", None),
    }),
}


def _run_hydra_main(mod, monkeypatch, tmp_path, name):
    register_resolvers()
    steps = 8 * 8  # envs x steps per update
    overrides = [
        "wandb=False", "num_seeds=2", f"num_timesteps={U * steps}", "num_envs=8",
        "num_steps_per_env_per_update=8", "num_epochs=1", "num_minibatches=2",
        f"fc_dim_size={HIDDEN}", "eval_interval=2", f"save_dir={tmp_path}/runs",
        f"hydra.run.dir={tmp_path}/hydra", "save_checkpoints=true", "seed=0",
        # large enough that every update moves the networks
        "lr=1e-3", "anneal_lr=true",
        "compare=true", f"compare_episodes={COMPARE_EPISODES}", "compare_interval=1",
    ]
    if "nfsp" in name:
        overrides += [
            "sl_reservoir_capacity=64", "sl_batch_size=8", "sl_num_steps_per_update=2",
            "sl_lr=1e-3", "anticipatory_eta=0.1",
        ]
    monkeypatch.setattr(sys, "argv", [f"{name}.py", *overrides])
    mod.main()
    jax.effects_barrier()
    (run_dir,) = glob.glob(f"{tmp_path}/runs/{name}_*")
    return run_dir


def _max_diff(a, b):
    leaves = zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    return max(float(np.abs(np.asarray(x) - np.asarray(y)).max()) for x, y in leaves)


def _assert_same(a, b):
    """Equal networks (up to the rounding of differently compiled programs)."""
    assert _max_diff(a, b) < 1e-6, _max_diff(a, b)


def _assert_different(a, b):
    assert _max_diff(a, b) > 1e-4, _max_diff(a, b)


def _assert_logged(record, prefix, policy):
    for key, want in log_metrics(prefix, policy).items():
        got = record[key]
        if np.isnan(want):
            assert np.isnan(got), key
        else:
            assert got == pytest.approx(want, rel=1e-5, abs=1e-6), key


@pytest.mark.parametrize("name", list(SCRIPTS))
def test_rl_script_run(monkeypatch, tmp_path, name):
    mod, nets = SCRIPTS[name]
    run_dir = _run_hydra_main(mod, monkeypatch, tmp_path, name)
    config = OmegaConf.to_container(OmegaConf.load(os.path.join(run_dir, "config.yaml")))
    assert config["run_dir"] == run_dir
    assert (config["num_update_steps"], config["half_update_steps"]) == (U, 3)

    # results.jsonl: one record per update and seed, eval metrics when due
    records = read_results(run_dir)
    prefixes = sorted({p for _, _, p, _, _ in nets.values()})
    for i in range(2):
        rows = [r for r in records if r["seed_index"] == i]
        assert [r["update_step"] for r in rows] == list(range(U))
        assert [r["env_steps"] for r in rows] == [64 * (u + 1) for u in range(U)]
        names = {VS_RANDOM[n] for n in nets}
        stats = NFSP_STATS if "nfsp" in name else SELF_PLAY_STATS
        exact = {"exploitability", "exploitability_br"} if "nfsp" in name else {"exploitability"}
        for u, row in enumerate(rows):
            assert stats <= set(row)
            assert 2.0 <= row["ep_length_avg"] <= 8.0 and row["episodes_completed"] > 0
            game_keys = {k for k in row if k.split("/")[0] in ("game", "game_br")}
            value_keys = {
                k for k in row if k.startswith("value_") and k.endswith("_vs_random")
            }
            if u in EVAL_UPDATES:
                assert game_keys == {f"{p}/{n}" for p in prefixes for n in TOTAL_NAMES}
                assert {k for k in row if k.startswith("exploitability")} == exact
                assert value_keys == {f"value_{n}" for n in names}
            else:
                assert not game_keys
                assert not any(k.startswith("exploitability") for k in row)
                assert not value_keys
            # the evaluation against random, every update: COMPARE_EPISODES
            # complete hands per policy
            sampled = {k for k in row if k.endswith("_vs_random")} - value_keys
            assert sampled == {p + n for p in EVAL_PREFIXES for n in names}
            for n in names:
                win, loss, draw = (row[f"{k}_rate_{n}"] for k in ("win", "loss", "draw"))
                assert win + loss + draw == pytest.approx(1.0)
                for rate in (win, loss, draw):
                    k = rate * COMPARE_EPISODES
                    assert k == pytest.approx(round(k), abs=1e-4)
                assert 2.0 <= row[f"avg_episode_length_{n}"] <= 8.0
                assert abs(row[f"avg_return_{n}"]) <= 13.0  # at most 13 chips a hand

    # checkpoints: the documented files, loading strictly into the networks
    expected = {"config.yaml", "results.jsonl"} | {
        f"{stage}/{n}_{i}.msgpack" for stage in ("half", "full") for n in nets for i in range(2)
    }
    found = {
        os.path.relpath(os.path.join(d, f), run_dir)
        for d, _, files in os.walk(run_dir) for f in files
    }
    assert found == expected
    env = make("leduc_holdem")
    obs = jnp.zeros((env.obs_dim,))
    loaded = {
        (stage, n, i): load_params(
            os.path.join(run_dir, stage, f"{n}_{i}.msgpack"),
            net.init(jax.random.PRNGKey(7), obs),
        )
        for stage in ("half", "full") for n, (net, *_) in nets.items() for i in range(2)
    }
    wrong = ActorCriticDiscreteMLP(3, 2 * HIDDEN).init(jax.random.PRNGKey(0), obs)
    with pytest.raises(ValueError):
        load_params(os.path.join(run_dir, "full", f"{next(iter(nets))}_0.msgpack"), wrong)

    # the logged game metrics are those of the saved networks: the half
    # checkpoint (after 3 updates) is what update 2 evaluated, the full one
    # what the last update evaluated
    for i in range(2):
        rows = {r["update_step"]: r for r in records if r["seed_index"] == i}
        for n, (_, policy_fn, prefix, _, _) in nets.items():
            for row, stage in ((rows[2], "half"), (rows[U - 1], "full")):
                policy = policy_fn(loaded[stage, n, i])
                _assert_logged(row, prefix, policy)
                assert row[f"value_{VS_RANDOM[n]}"] == pytest.approx(
                    leduc.value_vs_random_from_policy_array(policy, leduc._KEY_TO_IDX),
                    abs=1e-5,
                )
            _assert_different(loaded["half", n, i], loaded["full", n, i])

    # the same run stopped after 3 updates has the half networks, and all U
    # updates give the full ones (same seeds as main: split(PRNGKey(seed)))
    monkeypatch.setattr(mod, "RESULTS_LOG", ResultsLog(str(tmp_path), []))
    train = mod.make_train(dict(config))
    keys = jax.random.split(jax.random.PRNGKey(config["seed"]), 2)
    seeds = jnp.arange(2)
    for n_updates, stage in ((3, "half"), (U, "full")):
        state = jax.jit(jax.vmap(functools.partial(train, num_updates=n_updates)))(
            keys, seeds
        )
        jax.effects_barrier()
        for n, (_, _, _, attr, half_attr) in nets.items():
            params = getattr(state, attr).params
            for i in range(2):
                one = jax.tree_util.tree_map(lambda x: x[i], params)
                _assert_same(loaded[stage, n, i], one)
            if stage == "full":  # the runner state's half copy is the same
                half = state.half_params if half_attr is None else getattr(state.half, half_attr)
                for i in range(2):
                    one = jax.tree_util.tree_map(lambda x: x[i], half)
                    _assert_same(loaded["half", n, i], one)


def test_cfr_run(tmp_path):
    leduc_cfr.main(
        ["--iterations", "6", "--eval_interval", "4", "--save_dir", str(tmp_path),
         "--save_checkpoints"]
    )
    (run_dir,) = glob.glob(f"{tmp_path}/leduc_cfr_*")
    records = read_results(run_dir)
    assert [r["iteration"] for r in records] == [1, 3, 4, 6]  # 1, half, every 4, last
    assert all(r["seed_index"] == 0 for r in records)
    found = {
        os.path.relpath(os.path.join(d, f), run_dir)
        for d, _, files in os.walk(run_dir) for f in files
    }
    assert found == {
        "config.yaml", "results.jsonl",
        "half/average_policy.msgpack", "full/average_policy.msgpack",
    }
    half = leduc_cfr.load_average_policy(os.path.join(run_dir, "half", "average_policy.msgpack"))
    full = leduc_cfr.load_average_policy(os.path.join(run_dir, "full", "average_policy.msgpack"))
    # CFR is deterministic: a solver run for 3 / 6 iterations has these strategies
    for iterations, policy, record in ((3, half, records[1]), (6, full, records[3])):
        solver = leduc_cfr.run_cfr(iterations, 100)
        np.testing.assert_array_equal(policy, solver.average_policy_array())
        assert record["exploitability"] == pytest.approx(solver.exploitability(), abs=1e-12)
        assert record["value_vs_random"] == pytest.approx(
            leduc.value_vs_random_from_policy_array(policy, leduc._KEY_TO_IDX), abs=1e-12
        )
        _assert_logged(record, "game", policy)
    assert not np.array_equal(half, full)


def test_cfr_policy_array_matches_average_policy():
    """The array is the solver's average strategy at every infoset (state by
    state over the whole tree), so its exploitability is the solver's."""
    solver = leduc_cfr.run_cfr(2, 100)
    policy = solver.average_policy_array()
    np.testing.assert_allclose(policy.sum(axis=1), 1.0)

    def visit(state):
        if state.is_terminal():
            return
        if not state.is_chance_node():
            row = policy[leduc._KEY_TO_IDX[state.info_state_key(state.current_player)]]
            for a, p in solver.average_policy(state).items():
                assert row[a] == pytest.approx(p, abs=1e-15)
        for action in state.legal_actions():
            visit(state.child(action))

    visit(leduc.initial_state())
    expl = leduc.exploitability_from_policy_array(policy, leduc._KEY_TO_IDX)
    assert expl == pytest.approx(solver.exploitability(), abs=1e-12)


def test_cfr_checkpoint_rows_identify_infosets(tmp_path):
    """Rows are matched by their infoset fields, not by position."""
    from bluffjax.examples.run_utils import save_params

    policy = np.random.default_rng(0).dirichlet(np.ones(3), size=len(leduc._INFOSET_LIST))
    tree = leduc_cfr.policy_checkpoint(policy, 5)
    perm = np.random.default_rng(1).permutation(len(policy))
    shuffled = {k: (v[perm] if v.ndim else v) for k, v in tree.items()}
    path = save_params(str(tmp_path), "full", "average_policy", shuffled)
    np.testing.assert_array_equal(leduc_cfr.load_average_policy(path), policy)
    # a duplicated infoset is refused
    shuffled["player"] = shuffled["player"].copy()
    shuffled["player"][0] = 1 - shuffled["player"][0]
    path = save_params(str(tmp_path), "full", "average_policy", shuffled)
    with pytest.raises(KeyError):
        leduc_cfr.load_average_policy(path)


# Flags, config.yaml and wandb names shared by the CFR and Deep CFR scripts of
# Kuhn and Leduc (bluffjax/tests/test_kuhn_runs.py checks the Kuhn ones)
SHARED_FLAGS = {
    "iterations", "eval_interval", "seed", "save_dir", "job_type", "wandb",
    "project", "save_checkpoints",
}


class _WandbRun:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def log(self, metrics):
        pass

    def finish(self):
        pass


@pytest.mark.parametrize("name", ["leduc_cfr", "leduc_deep_cfr"])
def test_cfr_flags_config_and_wandb(monkeypatch, tmp_path, name):
    mod = {"leduc_cfr": leduc_cfr, "leduc_deep_cfr": leduc_deep_cfr}[name]
    defaults = vars(mod.parse_args([]))
    deep = name == "leduc_deep_cfr"
    assert SHARED_FLAGS | ({"num_seeds"} if deep else set()) <= set(defaults)
    assert defaults["job_type"] == name and defaults["project"] == "bluffjax"
    assert defaults["seed"] == 0 and defaults["wandb"] is False
    assert defaults["save_checkpoints"] is True
    assert (defaults["iterations"], defaults["eval_interval"]) == ((50, 5) if deep else (100, 10))
    assert defaults["save_dir"].endswith(os.path.join("checkpoints", "leduc"))
    assert mod.parse_args(["--wandb"]).wandb and not mod.parse_args(["--no-wandb"]).wandb
    runs = []
    monkeypatch.setattr(mod.wandb, "init", lambda **kw: runs.append(_WandbRun(**kw)) or runs[-1])
    argv = ["--iterations", "2", "--seed", "3", "--save_dir", str(tmp_path), "--wandb"]
    if deep:
        argv += ["--num_seeds", "2", "--traversals", "2", "--hidden_sizes", "4",
                 "--batch_size", "4", "--advantage_train_steps", "1",
                 "--policy_train_steps", "1", "--memory_capacity", "50"]
    mod.main(argv)
    (run_dir,) = glob.glob(f"{tmp_path}/{name}_*")
    with open(os.path.join(run_dir, "config.yaml")) as f:
        config = yaml.safe_load(f)
    assert config["env_name"] == "leduc_poker" and config["seed"] == 3
    assert config["num_seeds"] == (2 if deep else 1) and config["half_iterations"] == 1
    assert SHARED_FLAGS <= set(config)
    assert [r.kwargs["name"] for r in runs] == (["3_0", "3_1"] if deep else ["3_0"])
    for r in runs:
        assert r.kwargs["project"] == "bluffjax" and r.kwargs["mode"] is None
        assert r.kwargs["job_type"] == f"{name}_leduc_poker"
        assert r.kwargs["group"].startswith("leduc_poker_")
        assert r.kwargs["config"]["env_name"] == "leduc_poker"


DEEP_CFR_ARGS = [
    "--traversals", "5", "--eval_interval", "3", "--policy_train_steps", "20",
    "--advantage_train_steps", "10", "--batch_size", "16", "--memory_capacity", "500",
    "--hidden_sizes", "8", "8", "--save_checkpoints",
]


def test_deep_cfr_run(tmp_path):
    leduc_deep_cfr.main(
        ["--iterations", "4", "--num_seeds", "2", "--save_dir", f"{tmp_path}/runs", *DEEP_CFR_ARGS]
    )
    (run_dir,) = glob.glob(f"{tmp_path}/runs/leduc_deep_cfr_*")
    records = read_results(run_dir)
    for i in range(2):
        rows = [r for r in records if r["seed_index"] == i]
        assert [r["iteration"] for r in rows] == [1, 2, 3, 4]
        assert [r["traversals"] for r in rows] == [10, 20, 30, 40]
        for row in rows:  # 1, half (2), every 3, last: all evaluated
            assert {f"game/{n}" for n in TOTAL_NAMES} <= set(row)
            assert "exploitability" in row and "advantage_loss_p1" in row
            assert "value_vs_random" in row
    found = {
        os.path.relpath(os.path.join(d, f), run_dir)
        for d, _, files in os.walk(run_dir) for f in files
    }
    assert found == {"config.yaml", "results.jsonl"} | {
        f"{stage}/avg_policy_{i}.msgpack" for stage in ("half", "full") for i in range(2)
    }
    template = leduc_deep_cfr.policy_network([8, 8]).init(
        jax.random.PRNGKey(0), jnp.zeros((1, leduc_deep_cfr.INFO_STATE_SIZE))
    )
    loaded = {
        (stage, i): load_params(os.path.join(run_dir, stage, f"avg_policy_{i}.msgpack"), template)
        for stage in ("half", "full") for i in range(2)
    }
    for i in range(2):
        rows = {r["iteration"]: r for r in records if r["seed_index"] == i}
        for stage, it in (("half", 2), ("full", 4)):
            policy = leduc_deep_cfr.policy_array(loaded[stage, i], [8, 8])
            assert rows[it]["exploitability"] == pytest.approx(
                leduc_deep_cfr.exploitability(policy), rel=1e-5
            )
            assert rows[it]["value_vs_random"] == pytest.approx(
                leduc.value_vs_random_from_policy_array(policy, leduc._KEY_TO_IDX),
                abs=1e-5,
            )
            _assert_logged(rows[it], "game", policy)
    _assert_different(loaded["full", 0], loaded["full", 1])

    # seed index 1 is seed 0 + 1, and its half checkpoint is the network of
    # a run stopped after 2 iterations
    config = leduc_deep_cfr.parse_args(["--iterations", "2", "--seed", "1", *DEEP_CFR_ARGS])
    os.makedirs(tmp_path / "short")
    leduc_deep_cfr.run_deep_cfr(config, 0, None, str(tmp_path / "short"))
    short = load_params(str(tmp_path / "short" / "full" / "avg_policy_0.msgpack"), template)
    _assert_same(short, loaded["half", 1])
