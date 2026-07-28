import gc
import sys

import jax
import jax.numpy as jnp
from jax import lax
import pickle
import os
from datetime import datetime

from pgx import make


def _device_memory_stats(device: jax.Device) -> dict[str, int | None]:
    """Return device memory stats if available."""
    stats_raw = device.memory_stats()
    if stats_raw is None:
        return {}

    stats: dict[str, int | None] = {}
    for key, value in stats_raw.items():
        if value is None:
            stats[key] = None
        else:
            stats[key] = int(value)
    return stats


def _get_stat(stats: dict[str, int | None], key: str) -> int | None:
    if key in stats:
        return stats[key]
    return None


def run_benchmark(
    env_name: str,
    num_envs: int,
    num_timesteps_per_env: int,
    seed: int,
) -> dict[str, str | int | None]:
    """
    Run one PGX rollout benchmark and return memory stats.

    Returns:
        Dictionary with memory metrics for the active JAX device.
    """
    env = make(env_name)
    device = jax.devices()[0]

    def step(carry: tuple, unused: None) -> tuple[tuple, tuple]:
        rng, state = carry
        rng, rng_action = jax.random.split(rng, 2)
        rng_actions = jax.random.split(rng_action, num_envs)

        mask = state.legal_action_mask
        logits = jnp.where(mask, 0.0, -1e9)
        action = jax.vmap(jax.random.categorical, in_axes=(0, 0))(rng_actions, logits)
        state = jax.vmap(lambda s, a: env.step(s, a), in_axes=(0, 0))(state, action)
        done = state.terminated | state.truncated
        return (rng, state), (state.observation, state.rewards, done)

    def rollout(rng: jax.Array) -> tuple:
        rng, rng_init = jax.random.split(rng)
        rng_inits = jax.random.split(rng_init, num_envs)
        state = jax.vmap(lambda k: env.init(k), in_axes=(0))(rng_inits)
        (_, _), (obs_stack, reward_stack, done_stack) = lax.scan(
            step, (rng, state), None, length=num_timesteps_per_env
        )
        return obs_stack, reward_stack, done_stack

    # JIT compile
    jit_rollout = jax.jit(rollout)

    # Encourage cleanup before a new memory measurement.
    gc.collect()
    jax.clear_caches()
    memory_before = _device_memory_stats(device)

    # Warmup / compile
    rng = jax.random.PRNGKey(seed)
    _ = jax.block_until_ready(jit_rollout(rng))
    memory_after_warmup = _device_memory_stats(device)

    # Execute a second rollout after compilation.
    rng = jax.random.PRNGKey(seed + 1)
    _ = jax.block_until_ready(jit_rollout(rng))
    memory_after_run = _device_memory_stats(device)

    before_bytes = _get_stat(memory_before, "bytes_in_use")
    warmup_bytes = _get_stat(memory_after_warmup, "bytes_in_use")
    run_bytes = _get_stat(memory_after_run, "bytes_in_use")

    peak_candidates = [
        _get_stat(memory_after_warmup, "peak_bytes_in_use"),
        _get_stat(memory_after_run, "peak_bytes_in_use"),
    ]
    peak_values = [x for x in peak_candidates if x is not None]
    peak_bytes = max(peak_values) if len(peak_values) > 0 else None

    return {
        "device_platform": device.platform,
        "device_kind": device.device_kind,
        "bytes_in_use_before": before_bytes,
        "bytes_in_use_after_warmup": warmup_bytes,
        "bytes_in_use_after_run": run_bytes,
        "delta_bytes_in_use_after_run": (
            run_bytes - before_bytes
            if run_bytes is not None and before_bytes is not None
            else None
        ),
        "peak_bytes_in_use": peak_bytes,
        "peak_pool_bytes": _get_stat(memory_after_run, "peak_pool_bytes"),
        "pool_bytes": _get_stat(memory_after_run, "pool_bytes"),
        "bytes_limit": _get_stat(memory_after_run, "bytes_limit"),
    }


def main() -> None:
    num_envs = [1, 10, 100, 1000, 10000]
    num_seeds = 1
    num_timesteps_per_env = 1000

    d = {}

    envs = [arg for arg in sys.argv[1:]]
    if not envs:
        envs = ["kuhn_poker", "leduc_holdem"]

    for env_name in envs:
        d[env_name] = {}
        for num_env in num_envs:
            d[env_name][num_env] = []
            print(f"Running {env_name} with {num_env} environments")
            for seed in range(num_seeds):
                result = run_benchmark(
                    env_name=env_name,
                    num_envs=num_env,
                    num_timesteps_per_env=num_timesteps_per_env,
                    seed=seed,
                )
                d[env_name][num_env].append(result)

            peak = result["peak_bytes_in_use"]
            if peak is not None:
                print(peak / (1024.0 * 1024.0))
        with open(
            os.path.join(
                os.path.dirname(__file__),
                f"benchmark_results_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(d[env_name], f)


if __name__ == "__main__":
    main()
