from time import perf_counter

import jax
import jax.numpy as jnp
from jax import lax
import pickle
import os
from datetime import datetime
import copy

from bluffjax import make

# jax.config.update("jax_default_matmul_precision", "bfloat16")
# jax.config.update("jax_exec_time_optimization_effort", 1.0)
# jax.config.update("jax_memory_fitting_effort", 1.0)




def run_benchmark(
    env_name: str,
    num_envs: int,
    num_timesteps_per_env: int,
    seed: int,
) -> float:
    """
    Run Kuhn poker benchmark with vmap and JIT.

    Returns:
        Samples per second (num_parallel_envs * num_timesteps_per_env / elapsed_time)
    """
    env = make(env_name)

    def step(carry: tuple, unused: None) -> tuple[tuple, tuple]:
        rng, state = carry
        rng, rng_step, rng_action = jax.random.split(rng, 3)
        rng_steps = jax.random.split(rng_step, num_envs)
        rng_actions = jax.random.split(rng_action, num_envs)

        mask = jax.vmap(env.get_avail_actions, in_axes=(0))(state)
        logits = jnp.where(mask, 0.0, -1e9)
        action = jax.vmap(jax.random.categorical, in_axes=(0, 0))(rng_actions, logits)
        state, obs, reward, absorbing, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(rng_steps, state, action)
        return (rng, state), (obs, reward, done)

    def rollout(rng: jax.Array) -> tuple:
        rng, rng_reset = jax.random.split(rng)
        rng_resets = jax.random.split(rng_reset, num_envs)
        state, obs = jax.vmap(env.reset, in_axes=(0))(rng_resets)
        (_, final_state), (obs_stack, reward_stack, done_stack) = lax.scan(
            step, (rng, state), None, length=num_timesteps_per_env
        )
        return obs_stack, reward_stack, done_stack

    # JIT compile
    jit_rollout = jax.jit(rollout)

    # Warmup / compile
    rng = jax.random.PRNGKey(seed)
    _ = jax.block_until_ready(jit_rollout(rng))

    # Benchmark
    rng = jax.random.PRNGKey(seed + 1)
    t1 = perf_counter()
    _ = jax.block_until_ready(jit_rollout(rng))
    t2 = perf_counter()

    total_samples = num_envs * num_timesteps_per_env
    samples_per_second = total_samples / (t2 - t1)
    return samples_per_second


def main() -> None:
    num_envs = [1, 10, 100, 1000]
    num_seeds = 10
    num_timesteps_per_env = 1000
    seed = 0

    d = {}
    for env_name in [
        # "kuhn_poker",
        # "leduc_holdem",
        # "texas_limit_holdem",
        # "texas_nolimit_holdem",
        # "five_card_draw",
        # "seven_card_stud",
        # "goofspiel",
        # "bluff",
        # "werewolf",
        "kemps",
    ]:
        d[env_name] = {}
        for num_env in num_envs:
            d[env_name][num_env] = []
            print(f"Running {env_name} with {num_env} environments")
            for seed in range(num_seeds):
                if env_name == "kemps" and num_env == 10000:
                    d[env_name][num_env].append(copy.deepcopy(d[env_name][1000][seed]))
                else:
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
                f"benchmark_results_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(d[env_name], f)


if __name__ == "__main__":
    main()
