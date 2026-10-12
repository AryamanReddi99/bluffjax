"""
JaxMARL interface for BluffJAX games.

JaxMARLEnv wraps any BluffJAX game as a JaxMARL MultiAgentEnv
(https://github.com/FLAIROx/JaxMARL). It stays pure JAX, so it jit-compiles
and vmaps like the game itself, and JaxMARL code and wrappers (e.g.
jaxmarl.wrappers.baselines.LogWrapper) run on it unchanged:

- reset(key) -> (obs, state) and step(key, state, actions) -> (obs, state,
  rewards, dones, infos), which resets the game when the episode is done
  (JaxMARL's MultiAgentEnv.step), step_env without the reset, get_obs(state)
  and get_avail_actions(state). The state is the BluffJAX game state.
- Observations, actions, rewards, legal-action masks and dones are dicts keyed
  by agent ("agent_0", "agent_1", ... in the order of the game's player
  indices). dones[agent] is set when the agent has reached an absorbing state
  or the episode is done, dones["__all__"] when the episode is done.
- Every info entry has one value per agent, as JaxMARL code expects: the
  game's info, broadcast to the agents, and "truncated", True for the agents
  whose episode was cut off before an absorbing state (Bluff's horizon).
  dones doesn't distinguish truncation from termination.
- observation_space(agent) is a Box of shape (obs_dim,), unbounded because
  BluffJAX doesn't declare bounds, and action_space(agent) a Discrete.

Simultaneous-move (parallel) games map directly: agent i's observation and
legal actions are row i of the game's, and its action is the i-th action.

Turn-based (AEC) games follow JaxMARL's convention for turn-based games
(Hanabi): every agent submits an action at every step, but only the action of
the player to move (state.current_player_idx) is played. The action space has
one more action than the game, a no-op (noop_action, the last action), which
is the only legal action of the agents not to move; the player to move has
the game's legal actions and may not play the no-op. A BluffJAX turn-based
game only builds the observation of the player to move, so that agent gets it
and every other agent observes zeros: no agent ever sees an observation built
for another player. Every agent gets its reward at every step, as in the game.

Once the episode is over (only visible through step_env, since step resets),
no game action is legal: in parallel games the masks are all False, in
turn-based games every agent may only play the no-op.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box, Discrete

from bluffjax.environments.env import AECEnv, ParallelEnv
from bluffjax.utils.typing import (
    Any,
    BoolArray,
    FloatArray,
    IntArray,
    PRNGKeyArray,
    PyTree,
)


class JaxMARLEnv(MultiAgentEnv):
    """A BluffJAX game as a JaxMARL MultiAgentEnv (see the module docstring)."""

    def __init__(self, env: AECEnv | ParallelEnv) -> None:
        """
        Args:
            env (AECEnv | ParallelEnv): BluffJAX game, e.g. bluffjax.make("kuhn_poker")
        """
        super().__init__(num_agents=env.num_agents)
        self.env = env
        self.turn_based = isinstance(env, AECEnv)
        num_actions = env.action_space().n
        # Turn-based games get a no-op for the agents not to move.
        self.noop_action = num_actions if self.turn_based else None
        obs_shape = (env.observation_space().n,)
        self.observation_spaces = {
            agent: Box(-jnp.inf, jnp.inf, obs_shape) for agent in self.agents
        }
        self.action_spaces = {
            agent: Discrete(num_actions + self.turn_based) for agent in self.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: PRNGKeyArray) -> tuple[dict[str, FloatArray], PyTree]:
        state, obs = self.env.reset(key)
        return self._obs_dict(state, obs), state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, key: PRNGKeyArray, state: PyTree, actions: dict[str, IntArray]
    ) -> tuple[
        dict[str, FloatArray],
        PyTree,
        dict[str, FloatArray],
        dict[str, BoolArray],
        dict[str, Any],
    ]:
        action = jnp.stack([actions[agent] for agent in self.agents])
        if self.turn_based:
            action = action[state.current_player_idx]
        state, obs, reward, absorbing, done, info = self.env.step_env(
            key, state, action
        )
        rewards = {agent: reward[i] for i, agent in enumerate(self.agents)}
        dones = {agent: absorbing[i] | done for i, agent in enumerate(self.agents)}
        dones["__all__"] = done
        infos = {
            name: jnp.broadcast_to(value, (self.num_agents,))
            for name, value in info.items()
        }
        infos["truncated"] = done & ~absorbing
        return self._obs_dict(state, obs), state, rewards, dones, infos

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: PyTree) -> dict[str, FloatArray]:
        return self._obs_dict(state, self.env.obs_from_state(state))

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: PyTree) -> dict[str, BoolArray]:
        avail = self.env.get_avail_actions(state)
        if self.turn_based:
            acting = self._is_to_move(state) & ~state.done
            noop_only = jnp.arange(self.noop_action + 1) == self.noop_action
            avail = jnp.where(acting[:, None], jnp.append(avail, False), noop_only)
        else:
            avail = avail & ~state.done
        return {agent: avail[i] for i, agent in enumerate(self.agents)}

    def _is_to_move(self, state: PyTree) -> BoolArray:
        """(num_agents,) True for the player to move of a turn-based game."""
        return jnp.arange(self.num_agents) == state.current_player_idx

    def _obs_dict(self, state: PyTree, obs: FloatArray) -> dict[str, FloatArray]:
        """Splits the game's observation by agent (zeros for the agents not to
        move in a turn-based game)."""
        if self.turn_based:
            obs = jnp.where(self._is_to_move(state)[:, None], obs, 0.0)
        return {agent: obs[i] for i, agent in enumerate(self.agents)}
