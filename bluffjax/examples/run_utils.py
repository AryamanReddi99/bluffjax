"""
Run directories, the local results log and the half- and full-budget
checkpoints, shared by every training script in bluffjax/examples.

Layout of one run, {save_dir}/{job_type}_{timestamp}/ (created when the run
starts):
    config.yaml                resolved config of the run, with run_dir
    results.jsonl              everything the run logs, one JSON object per line
    half/{name}_{i}.msgpack    networks (or tables) after half the budget
    full/{name}_{i}.msgpack    after the full budget

i is the index of the seed (0..num_seeds-1 for the vmapped seeds of one run,
the same index as the wandb run "{seed}_{i}"). Both checkpoints come from the
same run: the half one is the state after half the budget, with the learning
rate and exploration schedules running over the whole run. What the budget and
the file names are is up to each script and documented there.

results.jsonl has one line per log call:
    {"seed_index": i, "wall_time": seconds since the run started, <metrics>}
with the same metric names and values that go to wandb. Values are numbers
(or lists for array metrics); a rate with no events is NaN, which Python's
json module writes as NaN and reads back. Read it with read_results().
"""

import datetime
import gzip
import json
import os
import time
from typing import Any, Iterable

import jax
import numpy as np
from flax import serialization
from omegaconf import OmegaConf

CHECKPOINT_STAGES = ("half", "full")
RESULTS_FILE = "results.jsonl"


def half_update_steps(num_update_steps: int) -> int:
    """Number of updates (iterations) after which the half checkpoint is taken."""
    return num_update_steps // 2


def make_run_dir(config: dict) -> str:
    """Creates {save_dir}/{job_type}_{timestamp}[_n]/ and writes config.yaml.

    Sets config["run_dir"]. Two runs started in the same second get distinct
    directories (suffix _1, _2, ...); an existing directory is never reused.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = os.path.join(config["save_dir"], f"{config['job_type']}_{timestamp}")
    run_dir, n = base, 0
    while True:
        try:
            os.makedirs(run_dir)
            break
        except FileExistsError:
            n += 1
            run_dir = f"{base}_{n}"
    config["run_dir"] = run_dir
    OmegaConf.save(OmegaConf.create(config), os.path.join(run_dir, "config.yaml"))
    return run_dir


def _to_python(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


class ResultsLog:
    """Appends logged metrics to {run_dir}/results.jsonl and to wandb.

    wandb_runs[i] is the wandb run of seed index i (one per vmapped seed), or
    an empty list for no wandb. Safe to call from jax io_callback / debug
    callbacks: each call writes and flushes one line.
    """

    def __init__(self, run_dir: str, wandb_runs: list | None = None) -> None:
        self.path = os.path.join(run_dir, RESULTS_FILE)
        self.wandb_runs = wandb_runs if wandb_runs is not None else []
        self.start = time.time()

    def log(self, metrics: dict[str, Any], seed_index: int = 0) -> None:
        seed_index = int(seed_index)
        values = {k: _to_python(v) for k, v in metrics.items()}
        record = {
            "seed_index": seed_index,
            "wall_time": round(time.time() - self.start, 3),
            **values,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.wandb_runs:
            self.wandb_runs[seed_index].log(values)


def read_results(run_dir: str) -> list[dict]:
    """The records of {run_dir}/results.jsonl, in the order they were logged.

    Reads results.jsonl.gz instead when only the compressed file exists (as in
    the runs committed under runs/)."""
    path = os.path.join(run_dir, RESULTS_FILE)
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        with gzip.open(path + ".gz", "rt") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def checkpoint_path(
    run_dir: str, stage: str, name: str, seed_index: int | None = None
) -> str:
    """{run_dir}/{stage}/{name}_{seed_index}.msgpack ({name}.msgpack if no index)."""
    if stage not in CHECKPOINT_STAGES:
        raise ValueError(f"stage must be one of {CHECKPOINT_STAGES}, got '{stage}'")
    file = f"{name}.msgpack" if seed_index is None else f"{name}_{seed_index}.msgpack"
    return os.path.join(run_dir, stage, file)


def save_params(
    run_dir: str, stage: str, name: str, params: Any, seed_index: int | None = None
) -> str:
    """Writes one pytree (no seed axis) with flax.serialization.to_bytes."""
    path = checkpoint_path(run_dir, stage, name, seed_index)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    params = jax.tree_util.tree_map(np.asarray, params)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))
    return path


def save_seed_params(
    run_dir: str, stage: str, name: str, params: Any, seed_indices: Iterable[int]
) -> list[str]:
    """Writes params with a leading seed axis as one file per seed index."""
    return [
        save_params(
            run_dir,
            stage,
            name,
            jax.tree_util.tree_map(lambda x: np.asarray(x[i]), params),
            i,
        )
        for i in seed_indices
    ]


def load_params(path: str, template: Any) -> Any:
    """Loads a pytree written by save_params, failing loudly.

    The file must exist and hold exactly the tree of `template` (same keys,
    shapes and dtypes); flax's from_bytes alone silently ignores extra keys
    and doesn't check shapes.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: '{path}'")
    with open(path, "rb") as f:
        restored = serialization.msgpack_restore(f.read())
    expected = serialization.to_state_dict(template)
    if jax.tree_util.tree_structure(restored) != jax.tree_util.tree_structure(
        expected
    ):
        raise ValueError(
            f"checkpoint {path} doesn't match: tree "
            f"{jax.tree_util.tree_structure(restored)} vs expected "
            f"{jax.tree_util.tree_structure(expected)}"
        )
    for (key, got), want in zip(
        jax.tree_util.tree_flatten_with_path(restored)[0],
        jax.tree_util.tree_leaves(expected),
    ):
        want_dtype = np.asarray(want).dtype
        if np.shape(got) != np.shape(want) or np.asarray(got).dtype != want_dtype:
            raise ValueError(
                f"checkpoint {path} doesn't match at {jax.tree_util.keystr(key)}: "
                f"{np.asarray(got).dtype}{np.shape(got)} vs expected "
                f"{want_dtype}{np.shape(want)}"
            )
    return serialization.from_state_dict(template, restored)
