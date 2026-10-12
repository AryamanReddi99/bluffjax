"""
How to use the BluffJAX wrappers, with random legal actions. Run it with

    uv run --extra wrappers python bluffjax/wrappers/demo.py
"""

import jax
import jax.numpy as jnp
import numpy as np

import bluffjax
from bluffjax.wrappers import JaxMARLEnv, PettingZooAECEnv, PettingZooParallelEnv

# JaxMARL: 200 steps in 16 parallel games of Leduc Hold'em, as one jitted function.
# Turn-based games follow JaxMARL's Hanabi convention: every agent acts at every
# step, and the agent not to move can only play the no-op.
env = JaxMARLEnv(bluffjax.make("leduc_holdem"))
rng = jax.random.key(0)
rng, rng_reset = jax.random.split(rng)
obs, state = jax.vmap(env.reset)(jax.random.split(rng_reset, 16))


def step(carry, unused):
    rng, state = carry
    rng, rng_act, rng_step = jax.random.split(rng, 3)
    avail = jax.vmap(env.get_avail_actions)(state)
    actions = {
        agent: jax.random.categorical(key, jnp.where(avail[agent], 0.0, -1e9))
        for agent, key in zip(env.agents, jax.random.split(rng_act, env.num_agents))
    }
    obs, state, rewards, dones, infos = jax.vmap(env.step)(
        jax.random.split(rng_step, 16), state, actions
    )
    return (rng, state), (rewards, dones["__all__"])


(_, state), (rewards, dones) = jax.jit(
    lambda rng, state: jax.lax.scan(step, (rng, state), None, length=200)
)(rng, state)
print(
    f"JaxMARL: {int(dones.sum())} hands of Leduc Hold'em, "
    f"agent_0 won {float(rewards['agent_0'].sum()):+.0f} chips in total"
)

# PettingZoo AEC: one hand of Kuhn Poker, one agent at a time.
np_rng = np.random.default_rng(0)
env = PettingZooAECEnv(bluffjax.make("kuhn_poker"))
env.reset(seed=0)
for agent in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None  # the hand is over: every agent steps None once
        print(f"PettingZoo AEC: {agent} ends the hand with reward {reward:+.0f}")
    else:
        action = np_rng.choice(np.flatnonzero(obs["action_mask"]))
        print(f"PettingZoo AEC: {agent} plays action {action}")
    env.step(action)

# PettingZoo Parallel: one game of Goofspiel, everyone bids at once.
env = PettingZooParallelEnv(bluffjax.make("goofspiel"))
obs, infos = env.reset(seed=0)
returns = dict.fromkeys(env.agents, 0.0)
while env.agents:
    actions = {
        agent: np_rng.choice(np.flatnonzero(obs[agent]["action_mask"]))
        for agent in env.agents
    }
    obs, rewards, terminations, truncations, infos = env.step(actions)
    for agent, reward in rewards.items():
        returns[agent] += reward
print(f"PettingZoo Parallel: Goofspiel returns {returns}")
