from time import perf_counter

import jax
import jax.numpy as jnp
from jax import lax
import pickle
import os
from datetime import datetime

from pgx import make


def run_benchmark(
    env_name: str,
    num_envs: int,
    num_timesteps_per_env: int,
    seed: int,
) -> float:
    """
    Benchmark a PGX poker env with vmap on each device and pmap across 4 GPUs.

    Returns:
        Samples per second across all devices
    """
    num_devices: int = 4
    available_devices: int = jax.local_device_count()
    if available_devices < num_devices:
        raise ValueError(
            f"Expected at least {num_devices} local devices, found {available_devices}."
        )
    env = make(env_name)

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

    # pmap compiles one program and runs it on each local device in parallel.
    pmap_rollout = jax.pmap(jax.jit(rollout), in_axes=0, out_axes=0)

    # Warmup / compile
    rng = jax.random.PRNGKey(seed)
    warmup_rngs = jax.random.split(rng, num_devices)
    _ = jax.block_until_ready(pmap_rollout(warmup_rngs))

    # Benchmark
    rng = jax.random.PRNGKey(seed + 1)
    benchmark_rngs = jax.random.split(rng, num_devices)
    t1 = perf_counter()
    _ = jax.block_until_ready(pmap_rollout(benchmark_rngs))
    t2 = perf_counter()

    total_samples = num_devices * num_envs * num_timesteps_per_env
    samples_per_second = total_samples / (t2 - t1)
    return samples_per_second


def main() -> None:
    num_envs = [1, 10, 100, 1000, 10000]
    num_seeds = 5
    num_timesteps_per_env = 1000

    d = {}
    for env_name in [
        "kuhn_poker",
        "leduc_holdem",
    ]:
        d[env_name] = {}
        for num_env in num_envs:
            d[env_name][num_env] = []
            print(f"Running {env_name} with {num_env} environments")
            for seed in range(num_seeds):
                samples_per_second = run_benchmark(
                    env_name=env_name,
                    num_envs=num_env,
                    num_timesteps_per_env=num_timesteps_per_env,
                    seed=seed,
                )
                d[env_name][num_env].append(int(samples_per_second))

        with open(
            os.path.join(
                os.path.dirname(__file__),
                f"benchmark_results_pmap_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(d[env_name], f)


if __name__ == "__main__":
    main()
