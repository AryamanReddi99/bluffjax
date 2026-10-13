"""Tests of the shared run directory, results log, checkpoint helpers
(bluffjax/examples/run_utils.py) and the metric helpers (bluffjax/metrics/base.py)."""

import math
import os

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from bluffjax.examples import run_utils
from bluffjax.metrics import credit, masked_totals, prefixed, rate


def _config(tmp_path):
    return {"save_dir": str(tmp_path), "job_type": "test", "num_seeds": 2}


def test_run_dir_is_unique_and_holds_config(tmp_path):
    config = _config(tmp_path)
    first = run_utils.make_run_dir(config)
    assert config["run_dir"] == first
    second = run_utils.make_run_dir(dict(config))
    assert first != second  # same second: suffixed, never reused
    with open(os.path.join(first, "config.yaml")) as f:
        saved = yaml.safe_load(f)
    assert saved["run_dir"] == first and saved["job_type"] == "test"


class _Recorder:
    def __init__(self):
        self.logged = []

    def log(self, metrics):
        self.logged.append(metrics)


def test_results_log_writes_jsonl_and_wandb(tmp_path):
    run_dir = run_utils.make_run_dir(_config(tmp_path))
    runs = [_Recorder(), _Recorder()]
    log = run_utils.ResultsLog(run_dir, runs)
    log.log({"env_steps": np.int32(64), "rate": jnp.float32(0.25)}, seed_index=1)
    log.log({"env_steps": 128, "rate": jnp.nan, "table": jnp.arange(3)}, seed_index=0)
    records = run_utils.read_results(run_dir)
    assert [r["seed_index"] for r in records] == [1, 0]
    assert records[0]["env_steps"] == 64 and records[0]["rate"] == 0.25
    assert math.isnan(records[1]["rate"]) and records[1]["table"] == [0, 1, 2]
    assert runs[1].logged == [{"env_steps": 64, "rate": 0.25}]
    assert len(runs[0].logged) == 1 and math.isnan(runs[0].logged[0]["rate"])


def test_results_log_without_wandb(tmp_path):
    run_dir = run_utils.make_run_dir(_config(tmp_path))
    run_utils.ResultsLog(run_dir).log({"x": 1.0})
    assert run_utils.read_results(run_dir)[0]["x"] == 1.0


def test_seed_params_round_trip_and_strict_load(tmp_path):
    run_dir = run_utils.make_run_dir(_config(tmp_path))
    params = {"dense": {"kernel": jnp.arange(12.0).reshape(2, 3, 2)}}  # seed axis 2
    paths = run_utils.save_seed_params(run_dir, "half", "policy", params, range(2))
    assert paths == [
        os.path.join(run_dir, "half", "policy_0.msgpack"),
        os.path.join(run_dir, "half", "policy_1.msgpack"),
    ]
    template = {"dense": {"kernel": jnp.zeros((3, 2))}}
    loaded = run_utils.load_params(paths[1], template)
    np.testing.assert_array_equal(loaded["dense"]["kernel"], params["dense"]["kernel"][1])
    with pytest.raises(ValueError):
        run_utils.load_params(paths[0], {"dense": {"kernel": jnp.zeros((2, 3))}})
    with pytest.raises(ValueError):
        run_utils.load_params(paths[0], {"other": {"kernel": jnp.zeros((3, 2))}})
    with pytest.raises(FileNotFoundError):
        run_utils.load_params(os.path.join(run_dir, "full", "policy_0.msgpack"), template)
    with pytest.raises(ValueError):
        run_utils.checkpoint_path(run_dir, "middle", "policy", 0)
    single = run_utils.save_params(run_dir, "full", "table", {"p": jnp.ones(4)})
    assert single == os.path.join(run_dir, "full", "table.msgpack")


def test_masked_totals_rate_and_credit():
    # (steps=2, envs=3, agents=2) counts, one with an extra table axis
    counts = {
        "bets": jnp.ones((2, 3, 2)),
        "table": jnp.ones((2, 3, 2, 4)),
    }
    mask = jnp.zeros((2, 3, 2), bool).at[:, :, 0].set(True)
    totals = masked_totals(counts, mask)
    assert float(totals["bets"]) == 6.0
    np.testing.assert_array_equal(totals["table"], np.full(4, 6.0))
    assert float(rate(1.0, 4.0)) == 0.25
    assert math.isnan(float(rate(0.0, 0.0)))
    np.testing.assert_array_equal(credit(3, 1, 2.0), np.array([0.0, 2.0, 0.0]))
    assert prefixed("game", {"x": 1}) == {"game/x": 1}
