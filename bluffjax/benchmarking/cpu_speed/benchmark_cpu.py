"""Benchmark poker env rollout throughput (samples/s) across CPU RL stacks.

Each *sample* is one environment ``step`` with a uniformly random *legal* action.
For ``n`` logical parallel environments we run ``STEPS_PER_ENV`` steps in each
environment (``n * STEPS_PER_ENV`` total samples). Environments are grouped into
``min(n, os.cpu_count())`` worker processes so we do not spawn ``O(n)``
interpreters when ``n`` is large.

Games / backends are listed in ``BENCHMARK_TARGETS`` (Leduc / Kuhn / Texas limit
/ Texas no-limit across RLCard, OpenSpiel, and PettingZoo where applicable).

Optional dependencies: ``rlcard``, ``numpy``, ``pettingzoo``, OpenSpiel (``pyspiel``).

Run::

    python -m bluffjax.benchmarking.cpu_speed.benchmark_cpu

Each backend × parallel count ``n`` is repeated ``NUM_SEEDS`` times (module
constant). Results are saved to ``benchmark_cpu_results_<timestamp>.pkl`` as::

    { result_key: { num_envs: [int samples/s per seed, ...] , ... } , ... }

— the same nesting as ``benchmark_speed/single_gpu/benchmark_all.py`` (that
script uses bluffjax env names as ``result_key``; here keys include a backend
suffix when more than one library implements the game).
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import pickle
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

# Steps collected per environment before reporting throughput.
STEPS_PER_ENV = 1000
# Independent master seeds per (target, n) configuration (edit here).
NUM_SEEDS = 5
DEFAULT_NS = (1, 10, 100, 1000, 10000)

# Spacing between repeat-run master seeds (distinct worker/task seeds per run).
_RUN_SEED_STRIDE = 982_451_653

_BACKEND_SALT = {"rlcard": 11, "openspiel": 29, "pettingzoo": 37}


def _stable_mix(s: str) -> int:
    return int(hashlib.md5(s.encode(), usedforsecurity=False).hexdigest()[:8], 16)


@dataclass(frozen=True)
class BenchmarkTarget:
    """One line in the benchmark matrix.

    ``runner`` encodes implementation: ``rlcard|<game_id>|<num_players>``,
    ``openspiel|<load_game_name>``, ``pettingzoo|<module_basename>``.
    """

    result_key: str
    runner: str


# result_key aligns with bluffjax ``make`` names where possible, plus backend suffix.
BENCHMARK_TARGETS: list[BenchmarkTarget] = [
    BenchmarkTarget("leduc_holdem_rlcard", "rlcard|leduc-holdem|2"),
    BenchmarkTarget("leduc_holdem_openspiel", "openspiel|leduc_poker"),
    BenchmarkTarget("leduc_holdem_pettingzoo", "pettingzoo|leduc_holdem_v4"),
    BenchmarkTarget("kuhn_poker", "openspiel|kuhn_poker"),
    BenchmarkTarget("texas_limit_holdem_rlcard", "rlcard|limit-holdem|2"),
    BenchmarkTarget("texas_limit_holdem_pettingzoo", "pettingzoo|texas_holdem_v4"),
    BenchmarkTarget("texas_nolimit_holdem_rlcard", "rlcard|no-limit-holdem|2"),
    BenchmarkTarget(
        "texas_nolimit_holdem_pettingzoo",
        "pettingzoo|texas_holdem_no_limit_v6",
    ),
]


def _partition_counts(n: int, num_workers: int) -> list[int]:
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    base = n // num_workers
    rem = n % num_workers
    return [base + (1 if i < rem else 0) for i in range(num_workers)]


def _legal_rlcard(obs: object) -> list:
    legal_actions = obs["legal_actions"]  # type: ignore[index]
    if hasattr(legal_actions, "keys"):
        return list(legal_actions.keys())
    return list(legal_actions)


def _bench_rlcard_game(
    game_id: str,
    num_players: int,
    num_local_envs: int,
    steps_per_env: int,
    seed: int,
) -> int:
    import rlcard

    rng = random.Random(seed)
    total = 0
    config_factory = lambda s: {
        "allow_step_back": False,
        "seed": s,
        "game_num_players": num_players,
    }
    for i in range(num_local_envs):
        env = rlcard.make(game_id, config_factory(seed + 10_003 * i))
        obs, _player_id = env.reset()
        for _ in range(steps_per_env):
            if env.is_over():
                obs, _player_id = env.reset()
            legal = _legal_rlcard(obs)
            action = rng.choice(legal)
            obs, _player_id = env.step(action)
            total += 1
    return total


def _bench_openspiel_game(
    game_name: str,
    num_local_envs: int,
    steps_per_env: int,
    seed: int,
) -> int:
    import numpy as np
    import pyspiel
    from open_spiel.python import rl_environment

    rng = np.random.RandomState(seed)
    total = 0
    for i in range(num_local_envs):
        env = rl_environment.Environment(
            pyspiel.load_game(game_name),
            enable_legality_check=False,
        )
        env.seed(int(seed + 97 * i) & 0x7FFFFFFF)
        ts = env.reset()
        for _ in range(steps_per_env):
            if ts.last():
                env.seed(int(seed + 97 * i + total) & 0x7FFFFFFF)
                ts = env.reset()
            cur = ts.observations["current_player"]
            legal = ts.observations["legal_actions"][cur]
            action = int(rng.choice(legal))
            ts = env.step([action])
            total += 1
    return total


def _pettingzoo_make_env(which: str) -> Callable[[], object]:
    if which == "leduc_holdem_v4":
        from pettingzoo.classic import leduc_holdem_v4  # pyright: ignore[reportMissingImports]

        return lambda: leduc_holdem_v4.env(render_mode=None)
    if which == "texas_holdem_v4":
        from pettingzoo.classic import texas_holdem_v4  # pyright: ignore[reportMissingImports]

        return lambda: texas_holdem_v4.env(render_mode=None, num_players=2)
    if which == "texas_holdem_no_limit_v6":
        from pettingzoo.classic import (  # pyright: ignore[reportMissingImports]
            texas_holdem_no_limit_v6,
        )

        return lambda: texas_holdem_no_limit_v6.env(render_mode=None, num_players=2)
    raise ValueError(f"unknown pettingzoo env {which!r}")


def _bench_pettingzoo_named(
    which: str,
    num_local_envs: int,
    steps_per_env: int,
    seed: int,
) -> int:
    import numpy as np

    rng = random.Random(seed)
    total = 0
    factory = _pettingzoo_make_env(which)
    for i in range(num_local_envs):
        env = factory()
        env.reset(seed=(seed + 51_991 * i) % (2**31))
        steps = 0
        while steps < steps_per_env:
            if len(env.agents) == 0:
                env.reset(seed=(seed + 51_991 * i + steps + total) % (2**31))
                continue
            ag = env.agent_selection
            obs = env.observe(ag)
            mask = obs["action_mask"]
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                env.reset(seed=(seed + 51_991 * i + steps + total) % (2**31))
                continue
            action = int(rng.choice(legal.tolist()))
            env.step(action)
            steps += 1
            total += 1
        env.close()
    return total


def _dispatch_benchmark(runner: str, num_local: int, steps: int, seed: int) -> int:
    parts = runner.split("|")
    kind = parts[0]
    if kind == "rlcard":
        return _bench_rlcard_game(parts[1], int(parts[2]), num_local, steps, seed)
    if kind == "openspiel":
        return _bench_openspiel_game(parts[1], num_local, steps, seed)
    if kind == "pettingzoo":
        return _bench_pettingzoo_named(parts[1], num_local, steps, seed)
    raise ValueError(f"unknown runner {runner!r}")


def _worker_task(payload: tuple[str, int, int, int]) -> int:
    runner, num_local, steps, seed = payload
    return _dispatch_benchmark(runner, num_local, steps, seed)


def benchmark_runner(
    runner: str,
    n_envs: int,
    steps_per_env: int,
    *,
    num_workers: int | None = None,
    run_seed: int = 0,
) -> tuple[float, int, int]:
    cpu = os.cpu_count() or 1
    if num_workers is None:
        workers = min(n_envs, cpu)
    else:
        workers = min(max(1, num_workers), n_envs)
    workers = max(1, workers)
    counts = _partition_counts(n_envs, workers)
    expected = n_envs * steps_per_env
    family = runner.split("|", 1)[0]
    game_mix = _stable_mix(runner) % 1_000_003
    tasks: list[tuple[str, int, int, int]] = []
    for w, k in enumerate(counts):
        if k <= 0:
            continue
        mix_run = (run_seed * 0x9E3779B9) & 0x7FFFFFFF
        task_seed = (
            _BACKEND_SALT.get(family, 0) + 1_000_003 * w + 50_009 * n_envs + game_mix + mix_run
        ) & 0x7FFFFFFF
        tasks.append((runner, k, steps_per_env, task_seed))

    n_proc = len(tasks)
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(processes=n_proc) as pool:
        results = pool.map(_worker_task, tasks, chunksize=1)
    elapsed = time.perf_counter() - t0
    total = sum(results)
    if total != expected:
        raise RuntimeError(f"{runner}: sample count mismatch: got {total}, expected {expected}")
    return elapsed, total, n_proc


def _parse_ns(raw: str) -> tuple[int, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return DEFAULT_NS
    return tuple(int(x) for x in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ns",
        default=",".join(map(str, DEFAULT_NS)),
        help=f"Comma-separated parallel env counts (default: {','.join(map(str, DEFAULT_NS))})",
    )
    parser.add_argument(
        "--steps-per-env",
        type=int,
        default=STEPS_PER_ENV,
        help=f"Random legal steps per environment (default: {STEPS_PER_ENV})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes (default: min(n_envs, os.cpu_count()))",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="First master seed; later repeats use base_seed + stride × trial (default: 42)",
    )
    parser.add_argument(
        "--pkl-out",
        default=None,
        help="Pickle output path (default: benchmark_cpu_results_<timestamp>.pkl next to this file)",
    )
    args = parser.parse_args()
    if NUM_SEEDS < 1:
        raise SystemExit("NUM_SEEDS must be >= 1 (see module constant)")
    ns = _parse_ns(args.ns)
    steps = args.steps_per_env

    results: dict[str, dict[int, list[int]]] = {
        t.result_key: {n: [] for n in ns} for t in BENCHMARK_TARGETS
    }

    print(
        f"CPU cores (logical): {os.cpu_count()}\n"
        f"Steps per env: {steps}; total samples per row = n × {steps}\n"
        f"Repeats per config: {NUM_SEEDS} (base_seed={args.base_seed})\n"
    )

    for target in BENCHMARK_TARGETS:
        print(f"== {target.result_key} ({target.runner}) ==")
        for n in ns:
            rates: list[float] = []
            walls: list[float] = []
            proc_count = 0
            total_samples = 0
            try:
                for trial in range(NUM_SEEDS):
                    run_seed = args.base_seed + trial * _RUN_SEED_STRIDE
                    elapsed, total, proc_count = benchmark_runner(
                        target.runner,
                        n,
                        steps,
                        num_workers=args.workers,
                        run_seed=run_seed,
                    )
                    rate = total / elapsed
                    rates.append(rate)
                    walls.append(elapsed)
                    total_samples = total
                    results[target.result_key][n].append(int(rate))
            except Exception as e:
                print(f"  n={n:4d}  ERROR: {e}")
                continue
            if NUM_SEEDS == 1:
                print(
                    f"  n={n:4d}  processes={proc_count:4d}  "
                    f"samples/s={rates[0]:,.1f}  wall_s={walls[0]:.4f}  "
                    f"total_samples={total_samples}"
                )
            else:
                m_rate = statistics.mean(rates)
                s_rate = statistics.stdev(rates) if len(rates) > 1 else 0.0
                m_wall = statistics.mean(walls)
                s_wall = statistics.stdev(walls) if len(walls) > 1 else 0.0
                print(
                    f"  n={n:4d}  processes={proc_count:4d}  seeds={NUM_SEEDS}  "
                    f"samples/s={m_rate:,.1f} ± {s_rate:,.1f}  "
                    f"wall_s={m_wall:.4f} ± {s_wall:.4f}  "
                    f"total_samples={total_samples}"
                )
        print()

    out_path = args.pkl_out
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"benchmark_cpu_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
        )
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
