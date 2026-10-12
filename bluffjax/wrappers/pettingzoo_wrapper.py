"""
PettingZoo interface for BluffJAX games.

PettingZooAECEnv wraps a turn-based (AEC) BluffJAX game as a PettingZoo
AECEnv, and PettingZooParallelEnv a simultaneous-move (parallel) game as a
PettingZoo ParallelEnv (https://pettingzoo.farama.org). A parallel game can
also be played one agent at a time through PettingZoo's own conversion,
pettingzoo.utils.conversions.parallel_to_aec(PettingZooParallelEnv(env)).

- Agents are "agent_0", "agent_1", ... in the order of the game's player
  indices.
- As in PettingZoo's classic games, an observation is a dict of NumPy arrays:
  "observation" (float32, shape (obs_dim,)) and "action_mask" (int8, 1 for
  each legal action). The observation space is unbounded because BluffJAX
  doesn't declare bounds. Once the episode is over no action is legal.
- In a turn-based game agent_selection is the player to move. BluffJAX only
  builds the observation of that player, so any other agent observes zeros
  with an all-zero action mask: no agent ever sees an observation built for
  another player.
- The game gives every agent a reward at every step. In the AEC API they are
  accumulated as usual: last() returns the reward the agent has collected
  since it last acted.
- terminations are the game's absorbing flags. truncations mark the agents
  whose episode ended before an absorbing state (Bluff's horizon). At the end
  of an episode every agent is terminated or truncated and leaves env.agents
  (in the AEC API after stepping None once, a dead step).
- infos: every agent gets the game's info of the last step, as NumPy arrays.
- Seeding: reset(seed) starts a JAX PRNG key from the seed, and every reset
  and step splits it (key, subkey = jax.random.split(key)) and passes the
  subkey to the game's reset or step_env, so the same seed and actions give
  the same games. reset() without a seed continues with the current key
  (with a random seed on the first reset).
- The game's reset, step_env and get_avail_actions run jit-compiled. step_env
  doesn't reset the game, so the end of the episode stays visible; call
  reset() to start the next one.
- Only render_mode=None is supported.
"""

import gymnasium
import jax
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv as PettingZooAECBase
from pettingzoo import ParallelEnv as PettingZooParallelBase

from bluffjax.environments.env import AECEnv, ParallelEnv


def _to_numpy(tree):
    """Device arrays -> writable NumPy arrays."""
    return jax.tree.map(np.array, jax.device_get(tree))


class _Wrapper:
    """What both wrappers share: spaces, seeding and the jitted game functions."""

    def __init__(self, env: AECEnv | ParallelEnv, render_mode: str | None) -> None:
        if render_mode is not None:
            raise ValueError(f"render_mode {render_mode!r} is not supported, only None")
        self.env = env
        self.render_mode = render_mode
        self.metadata = {"name": type(env).__name__, "render_modes": []}
        self.possible_agents = list(env.agents)
        num_actions = env.action_space().n
        obs_shape = (env.observation_space().n,)
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(-np.inf, np.inf, obs_shape, np.float32),
                    "action_mask": spaces.Box(0, 1, (num_actions,), np.int8),
                }
            )
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(num_actions) for agent in self.possible_agents
        }
        self._key = None

        def mask(state):
            return (env.get_avail_actions(state) & ~state.done).astype(np.int8)

        def reset(key):
            key, subkey = jax.random.split(key)
            state, obs = env.reset(subkey)
            return key, state, (obs, mask(state))

        def step(key, state, action):
            key, subkey = jax.random.split(key)
            state, obs, reward, absorbing, done, info = env.step_env(
                subkey, state, action
            )
            return key, state, (obs, mask(state), reward, absorbing, done, info)

        self._reset_fn = jax.jit(reset)
        self._step_fn = jax.jit(step)

    def _reset_game(self, seed: int | None) -> tuple:
        """Resets the game; returns the observation and the action mask."""
        if seed is not None:
            self._key = jax.random.key(seed)
        elif self._key is None:
            self._key = jax.random.key(np.random.SeedSequence().generate_state(1)[0])
        self._key, self._state, out = self._reset_fn(self._key)
        return _to_numpy(out)

    def _step_game(self, action: np.ndarray) -> tuple:
        """Steps the game; returns obs, action mask, reward, absorbing, done, info."""
        self._key, self._state, out = self._step_fn(self._key, self._state, action)
        return _to_numpy(out)

    def observation_space(self, agent: str) -> spaces.Dict:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Discrete:
        return self.action_spaces[agent]

    def render(self) -> None:
        gymnasium.logger.warn(
            "You are calling render method without specifying any render mode."
        )

    def close(self) -> None:
        pass


class PettingZooAECEnv(_Wrapper, PettingZooAECBase):
    """A turn-based BluffJAX game as a PettingZoo AECEnv (see the module docstring)."""

    def __init__(self, env: AECEnv, render_mode: str | None = None) -> None:
        """
        Args:
            env (AECEnv): turn-based BluffJAX game, e.g. bluffjax.make("kuhn_poker")
            render_mode (str | None): only None is supported
        """
        if not isinstance(env, AECEnv):
            raise TypeError(
                f"{type(env).__name__} is a parallel game, use PettingZooParallelEnv"
            )
        super().__init__(env, render_mode)

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self._obs, self._mask = self._reset_game(seed)
        self._to_move = self.possible_agents[int(self._state.current_player_idx)]
        self.agents = self.possible_agents[:]
        self.agent_selection = self._to_move
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {agent: {} for agent in self.agents}

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        if agent == self._to_move:
            return {"observation": self._obs, "action_mask": self._mask}
        return {
            "observation": np.zeros_like(self._obs),
            "action_mask": np.zeros_like(self._mask),
        }

    def step(self, action: int | None) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        self._obs, self._mask, reward, absorbing, done, info = self._step_game(
            np.int32(action)
        )
        self._to_move = self.possible_agents[int(self._state.current_player_idx)]
        self._cumulative_rewards[agent] = 0.0
        for i, other in enumerate(self.possible_agents):
            if other in self.agents:
                self.rewards[other] = float(reward[i])
                self.terminations[other] = bool(absorbing[i])
                self.truncations[other] = bool(done and not absorbing[i])
                self.infos[other] = dict(info)
        self.agent_selection = self._to_move
        self._accumulate_rewards()
        self._deads_step_first()


class PettingZooParallelEnv(_Wrapper, PettingZooParallelBase):
    """A simultaneous-move BluffJAX game as a PettingZoo ParallelEnv (see the
    module docstring)."""

    def __init__(self, env: ParallelEnv, render_mode: str | None = None) -> None:
        """
        Args:
            env (ParallelEnv): simultaneous-move BluffJAX game, e.g. bluffjax.make("goofspiel")
            render_mode (str | None): only None is supported
        """
        if not isinstance(env, ParallelEnv):
            raise TypeError(
                f"{type(env).__name__} is a turn-based game, use PettingZooAECEnv"
            )
        super().__init__(env, render_mode)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        obs, mask = self._reset_game(seed)
        self.agents = self.possible_agents[:]
        observations = {
            agent: {"observation": obs[i], "action_mask": mask[i]}
            for i, agent in enumerate(self.agents)
        }
        return observations, {agent: {} for agent in self.agents}

    def step(self, actions: dict[str, int]) -> tuple[dict, dict, dict, dict, dict]:
        # Agents that have left the game don't act; they get a placeholder.
        action = np.array(
            [actions[a] if a in self.agents else 0 for a in self.possible_agents],
            dtype=np.int32,
        )
        obs, mask, reward, absorbing, done, info = self._step_game(action)
        live = [(i, a) for i, a in enumerate(self.possible_agents) if a in self.agents]
        observations = {
            agent: {"observation": obs[i], "action_mask": mask[i]} for i, agent in live
        }
        rewards = {agent: float(reward[i]) for i, agent in live}
        terminations = {agent: bool(absorbing[i]) for i, agent in live}
        truncations = {agent: bool(done and not absorbing[i]) for i, agent in live}
        infos = {agent: dict(info) for _, agent in live}
        self.agents = [
            agent for _, agent in live if not (terminations[agent] or truncations[agent])
        ]
        return observations, rewards, terminations, truncations, infos
