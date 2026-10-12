"""Checks for the JaxMARL and PettingZoo wrappers in bluffjax/wrappers.

- JaxMARL: jitted, vmapped random rollouts match the native game step for
  step (states, so also the automatic resets, rewards, dones, infos and the
  observation and legal actions of the player to move); in turn-based games
  the agents not to move observe zeros and may only play the no-op; no game
  action is legal once the episode is over; Bluff's truncation; JaxMARL's
  LogWrapper and its per-agent infos.
- PettingZoo: PettingZoo's own API and seed tests for every game (the parallel
  games also through PettingZoo's parallel_to_aec); rollouts match the native
  game (observations, masks, accumulated rewards, terminations, truncations,
  dead steps), with the documented key chain; Bluff's truncation.
- Importing bluffjax or the JaxMARL wrapper doesn't import PettingZoo, and the
  PettingZoo wrappers don't import JaxMARL.
"""

import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import available_envs, make
from bluffjax.environments.env import AECEnv

TURN_BASED = {env_id: isinstance(make(env_id), AECEnv) for env_id in available_envs()}
AEC_ENVS = [env_id for env_id in available_envs() if TURN_BASED[env_id]]
PARALLEL_ENVS = [env_id for env_id in available_envs() if not TURN_BASED[env_id]]
NUM_GAMES = 8


def jaxmarl_rollout(env, wrapped, seed, num_steps=200):
    """Random legal actions in the native game (env.step) and the same actions
    through the wrapper (wrapped.step and wrapped.step_env), with the same
    keys. Arrays are (game, step, ...)."""

    def run(key):
        key_reset, key_steps = jax.random.split(key)
        state, obs = env.reset(key_reset)
        obs_w, state_w = wrapped.reset(key_reset)

        def one(carry, key):
            state, obs, state_w, obs_w = carry
            key_act, key_step = jax.random.split(key)
            avail = env.get_avail_actions(state)
            action = jax.random.categorical(key_act, jnp.where(avail, 0.0, -1e9))
            if wrapped.turn_based:
                to_move = jnp.arange(env.num_agents) == state.current_player_idx
                action_w = jnp.where(to_move, action, wrapped.noop_action)
            else:
                action_w = action
            actions = {agent: action_w[i] for i, agent in enumerate(wrapped.agents)}
            next_state, next_obs, rew, absorbing, done, info = env.step(
                key_step, state, action
            )
            next_obs_w, next_state_w, rew_w, done_w, info_w = wrapped.step(
                key_step, state_w, actions
            )
            # step_env with the key step uses for it: the state before the reset
            _, end_state, _, end_done, _ = wrapped.step_env(
                jax.random.split(key_step)[0], state_w, actions
            )
            out = dict(
                state=state,
                obs=obs,
                avail=avail,
                rew=rew,
                absorbing=absorbing,
                done=done,
                info=info,
                state_w=state_w,
                obs_w=obs_w,
                get_obs=wrapped.get_obs(state_w),
                avail_w=wrapped.get_avail_actions(state_w),
                rew_w=rew_w,
                done_w=done_w,
                info_w=info_w,
                next_state_w=next_state_w,
                end_state=end_state,
                end_done=end_done,
                end_obs=wrapped.get_obs(end_state),
                end_avail=wrapped.get_avail_actions(end_state),
            )
            return (next_state, next_obs, next_state_w, next_obs_w), out

        steps = jax.random.split(key_steps, num_steps)
        _, out = jax.lax.scan(one, (state, obs, state_w, obs_w), steps)
        return out

    keys = jax.random.split(jax.random.key(seed), NUM_GAMES)
    return jax.tree.map(np.asarray, jax.jit(jax.vmap(run))(keys))


def stack(d, agents):
    """Agent-keyed dict of (game, step, ...) arrays -> (game, step, agent, ...)."""
    return np.stack([d[agent] for agent in agents], axis=2)


def assert_trees_equal(x, y):
    for a, b in zip(jax.tree.leaves(x), jax.tree.leaves(y), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("env_id", available_envs())
def test_jaxmarl_matches_env(env_id):
    pytest.importorskip("jaxmarl")
    from bluffjax.wrappers import JaxMARLEnv

    env = make(env_id)
    wrapped = JaxMARLEnv(env)
    agents, n = wrapped.agents, env.num_agents
    num_actions, obs_dim = env.action_space().n, env.observation_space().n
    assert agents == env.agents and wrapped.num_agents == n
    for agent in agents:
        assert wrapped.observation_space(agent).shape == (obs_dim,)
        assert wrapped.action_space(agent).n == num_actions + TURN_BASED[env_id]
    out = jaxmarl_rollout(env, wrapped, seed=0)

    # Same states (so also the same resets), rewards and dones; some episodes ended.
    assert out["done"].any()
    assert_trees_equal(out["state"], out["state_w"])
    np.testing.assert_array_equal(stack(out["rew_w"], agents), out["rew"])
    np.testing.assert_array_equal(out["done_w"]["__all__"], out["done"])
    np.testing.assert_array_equal(
        stack(out["done_w"], agents), out["absorbing"] | out["done"][..., None]
    )
    # Infos: the game's info for every agent, and truncation.
    info_w = out["info_w"]
    for name, value in out["info"].items():
        shape = out["done"].shape + (n,)
        assert info_w[name].shape == shape
        expected = value if value.ndim == 3 else value[..., None]
        np.testing.assert_array_equal(info_w[name], np.broadcast_to(expected, shape))
    np.testing.assert_array_equal(
        info_w["truncated"], out["done"][..., None] & ~out["absorbing"]
    )
    # step_env doesn't reset: it gives the step's state, which step keeps unless done.
    not_done = ~out["done"]
    np.testing.assert_array_equal(out["end_done"]["__all__"], out["done"])
    for end, nxt in zip(jax.tree.leaves(out["end_state"]), jax.tree.leaves(out["next_state_w"])):
        np.testing.assert_array_equal(end[not_done], nxt[not_done])

    obs_w, avail_w = stack(out["obs_w"], agents), stack(out["avail_w"], agents)
    np.testing.assert_array_equal(stack(out["get_obs"], agents), obs_w)
    end_avail = stack(out["end_avail"], agents)[out["done"]]
    for agent in agents:
        assert np.asarray(wrapped.observation_space(agent).contains(out["obs_w"][agent])).all()
    if TURN_BASED[env_id]:
        to_move = np.arange(n) == out["state"].current_player_idx[..., None]
        # The player to move gets the game's observation; nobody else sees anything.
        np.testing.assert_array_equal(obs_w[to_move], out["obs"].reshape(-1, obs_dim))
        assert not obs_w[~to_move].any()
        end_to_move = np.arange(n) == out["end_state"].current_player_idx[..., None]
        assert not stack(out["end_obs"], agents)[~end_to_move].any()
        # The game's legal actions for the player to move, the no-op for the others.
        no_action = np.zeros(out["avail"].shape[:-1] + (1,), dtype=bool)
        np.testing.assert_array_equal(
            avail_w[to_move],
            np.concatenate([out["avail"], no_action], -1).reshape(-1, num_actions + 1),
        )
        noop_only = np.arange(num_actions + 1) == wrapped.noop_action
        np.testing.assert_array_equal(avail_w[~to_move], np.broadcast_to(noop_only, avail_w[~to_move].shape))
        np.testing.assert_array_equal(end_avail, np.broadcast_to(noop_only, end_avail.shape))
    else:
        np.testing.assert_array_equal(obs_w, out["obs"])
        np.testing.assert_array_equal(avail_w, out["avail"])
        assert not end_avail.any()


def test_jaxmarl_bluff_truncation():
    """Bluff cut off at its horizon: done for everyone, truncated, never absorbing."""
    pytest.importorskip("jaxmarl")
    from bluffjax.wrappers import JaxMARLEnv

    env = make("bluff", horizon=6)
    out = jaxmarl_rollout(env, JaxMARLEnv(env), seed=1, num_steps=12)
    cut_off = np.broadcast_to(np.arange(12) % 6 == 5, (NUM_GAMES, 12))
    np.testing.assert_array_equal(out["done_w"]["__all__"], cut_off)
    np.testing.assert_array_equal(out["info_w"]["truncated"], np.repeat(cut_off[..., None], 3, -1))
    assert not out["absorbing"].any()


def test_jaxmarl_log_wrapper():
    """JaxMARL's LogWrapper in a JaxMARL-style vmapped, jitted rollout: the
    logged episode returns and lengths match the rewards."""
    pytest.importorskip("jaxmarl")
    from jaxmarl.wrappers.baselines import LogWrapper

    from bluffjax.wrappers import JaxMARLEnv

    env = LogWrapper(JaxMARLEnv(make("leduc_holdem")))
    num_steps = 100

    def run(key):
        key_reset, key_steps = jax.random.split(key)
        obs, state = jax.vmap(env.reset)(jax.random.split(key_reset, NUM_GAMES))

        def one(state, key):
            key_act, key_step = jax.random.split(key)
            avail = jax.vmap(env.get_avail_actions)(state.env_state)
            keys = jax.random.split(key_act, env.num_agents)
            actions = {
                agent: jax.random.categorical(k, jnp.where(avail[agent], 0.0, -1e9))
                for agent, k in zip(env.agents, keys)
            }
            obs, state, rewards, dones, info = jax.vmap(env.step)(
                jax.random.split(key_step, NUM_GAMES), state, actions
            )
            # JaxMARL baselines flatten every info entry to one value per actor.
            info = jax.tree.map(lambda x: x.reshape(NUM_GAMES * env.num_agents), info)
            return state, (rewards, dones["__all__"], info)

        _, out = jax.lax.scan(one, state, jax.random.split(key_steps, num_steps))
        return out

    rewards, dones, info = jax.tree.map(np.asarray, jax.jit(run)(jax.random.key(2)))
    rewards = np.stack([rewards[agent] for agent in env.agents], axis=-1)
    assert dones.any()
    ep_return = np.zeros((NUM_GAMES, 2))
    ep_length = np.zeros(NUM_GAMES)
    for t in range(num_steps):
        ep_return += rewards[t]
        ep_length += 1
        logged = info["returned_episode"][t].reshape(NUM_GAMES, 2)
        np.testing.assert_array_equal(logged, np.repeat(dones[t][:, None], 2, -1))
        returns = info["returned_episode_returns"][t].reshape(NUM_GAMES, 2)
        lengths = info["returned_episode_lengths"][t].reshape(NUM_GAMES, 2)
        np.testing.assert_allclose(returns[dones[t]], ep_return[dones[t]])
        np.testing.assert_array_equal(lengths[dones[t]], np.repeat(ep_length[dones[t], None], 2, -1))
        ep_return[dones[t]] = 0.0
        ep_length[dones[t]] = 0


@pytest.mark.parametrize("env_id", available_envs())
def test_pettingzoo_api(env_id):
    """PettingZoo's own API test and seed test."""
    pettingzoo_test = pytest.importorskip("pettingzoo.test")
    from pettingzoo.utils.conversions import parallel_to_aec

    from bluffjax.wrappers import PettingZooAECEnv, PettingZooParallelEnv

    def seeded(pz_env):
        for agent in pz_env.possible_agents:
            pz_env.action_space(agent).seed(0)
        return pz_env

    if TURN_BASED[env_id]:
        pettingzoo_test.api_test(seeded(PettingZooAECEnv(make(env_id))))
        pettingzoo_test.seed_test(lambda: PettingZooAECEnv(make(env_id)))
        with pytest.raises(TypeError):
            PettingZooParallelEnv(make(env_id))
    else:
        pettingzoo_test.parallel_api_test(seeded(PettingZooParallelEnv(make(env_id))))
        pettingzoo_test.parallel_seed_test(lambda: PettingZooParallelEnv(make(env_id)))
        # Turn by turn through PettingZoo's own conversion.
        pettingzoo_test.api_test(seeded(parallel_to_aec(PettingZooParallelEnv(make(env_id)))))
        pettingzoo_test.seed_test(lambda: parallel_to_aec(PettingZooParallelEnv(make(env_id))))
        with pytest.raises(TypeError):
            PettingZooAECEnv(make(env_id))


@pytest.mark.parametrize("env_id", AEC_ENVS)
def test_pettingzoo_aec_matches_env(env_id):
    """Three episodes (seeded, then continuing the key) against the native game
    driven with the wrapper's key chain."""
    pytest.importorskip("pettingzoo")
    from bluffjax.wrappers import PettingZooAECEnv

    env = make(env_id)
    pz_env = PettingZooAECEnv(env)
    agents = pz_env.possible_agents
    rng = np.random.default_rng(0)
    key = jax.random.key(7)
    for episode in range(3):
        pz_env.reset(seed=7 if episode == 0 else None)
        key, subkey = jax.random.split(key)
        state, obs = env.reset(subkey)
        # Reward each agent has collected since it last acted.
        collected = dict.fromkeys(agents, 0.0)
        done = False
        while not done:
            agent = pz_env.agent_selection
            assert agent == agents[int(state.current_player_idx)]
            observation, reward, termination, truncation, _ = pz_env.last()
            mask = np.asarray(env.get_avail_actions(state))
            np.testing.assert_array_equal(observation["observation"], obs)
            np.testing.assert_array_equal(observation["action_mask"], mask)
            assert observation["observation"].flags.writeable
            for other in agents:
                if other != agent:
                    assert not pz_env.observe(other)["observation"].any()
                    assert not pz_env.observe(other)["action_mask"].any()
            assert reward == collected[agent] and not (termination or truncation)
            collected[agent] = 0.0
            action = rng.choice(np.flatnonzero(mask))
            pz_env.step(action)
            key, subkey = jax.random.split(key)
            state, obs, rew, absorbing, done, info = env.step_env(
                subkey, state, jnp.int32(action)
            )
            rew, absorbing, done = np.asarray(rew), np.asarray(absorbing), bool(done)
            for i, other in enumerate(agents):
                collected[other] += float(rew[i])
            assert pz_env.rewards == {a: float(rew[i]) for i, a in enumerate(agents)}
            for name, value in info.items():
                np.testing.assert_array_equal(pz_env.infos[agent][name], value)
        # Every agent ends the episode with one dead step and sees what it has
        # collected since it last acted.
        assert sorted(pz_env.agents) == sorted(agents)
        while pz_env.agents:
            agent = pz_env.agent_selection
            observation, reward, termination, truncation, _ = pz_env.last()
            i = agents.index(agent)
            assert termination == bool(absorbing[i]) and truncation == (not absorbing[i])
            assert not observation["action_mask"].any()
            if agent == agents[int(state.current_player_idx)]:
                np.testing.assert_array_equal(observation["observation"], obs)
            else:
                assert not observation["observation"].any()
            assert reward == collected[agent]
            pz_env.step(None)
            assert agent not in pz_env.agents


@pytest.mark.parametrize("env_id", PARALLEL_ENVS)
def test_pettingzoo_parallel_matches_env(env_id):
    """Three episodes (seeded, then continuing the key) against the native game
    driven with the wrapper's key chain."""
    pytest.importorskip("pettingzoo")
    from bluffjax.wrappers import PettingZooParallelEnv

    env = make(env_id)
    pz_env = PettingZooParallelEnv(env)
    agents = pz_env.possible_agents
    rng = np.random.default_rng(0)
    key = jax.random.key(7)
    for episode in range(3):
        observations, _ = pz_env.reset(seed=7 if episode == 0 else None)
        key, subkey = jax.random.split(key)
        state, obs = env.reset(subkey)
        done = False
        while not done:
            assert pz_env.agents == agents
            mask = np.asarray(env.get_avail_actions(state))
            for i, agent in enumerate(agents):
                np.testing.assert_array_equal(observations[agent]["observation"], obs[i])
                np.testing.assert_array_equal(observations[agent]["action_mask"], mask[i])
                assert observations[agent]["observation"].flags.writeable
            action = np.array([rng.choice(np.flatnonzero(m)) for m in mask])
            observations, rewards, terminations, truncations, infos = pz_env.step(
                dict(zip(agents, action))
            )
            key, subkey = jax.random.split(key)
            state, obs, rew, absorbing, done, info = env.step_env(
                subkey, state, jnp.asarray(action, jnp.int32)
            )
            rew, absorbing, done = np.asarray(rew), np.asarray(absorbing), bool(done)
            assert rewards == {a: float(rew[i]) for i, a in enumerate(agents)}
            assert terminations == {a: bool(absorbing[i]) for i, a in enumerate(agents)}
            assert truncations == {a: done and not absorbing[i] for i, a in enumerate(agents)}
            for name, value in info.items():
                np.testing.assert_array_equal(infos[agents[0]][name], value)
        assert pz_env.agents == []
        for i, agent in enumerate(agents):
            np.testing.assert_array_equal(observations[agent]["observation"], obs[i])
            assert not observations[agent]["action_mask"].any()


def test_pettingzoo_bluff_truncation():
    """Bluff cut off at its horizon: every agent truncated, none terminated."""
    pytest.importorskip("pettingzoo")
    from bluffjax.wrappers import PettingZooAECEnv

    pz_env = PettingZooAECEnv(make("bluff", horizon=6))
    pz_env.reset()  # no seed on the first reset: a random one
    num_actions, finished = 0, []
    for agent in pz_env.agent_iter():
        obs, reward, termination, truncation, info = pz_env.last()
        if termination or truncation:
            assert truncation and not termination
            finished.append(agent)
            pz_env.step(None)
        else:
            pz_env.step(np.flatnonzero(obs["action_mask"])[0])
            num_actions += 1
    assert num_actions == 6 and sorted(finished) == pz_env.possible_agents


def test_imports():
    """bluffjax needs neither library, and each wrapper only needs its own."""
    pytest.importorskip("jaxmarl")
    pytest.importorskip("pettingzoo")
    checks = [
        "import sys, bluffjax, bluffjax.wrappers; "
        "assert 'jaxmarl' not in sys.modules and 'pettingzoo' not in sys.modules",
        "import sys; from bluffjax.wrappers import JaxMARLEnv; "
        "assert 'pettingzoo' not in sys.modules",
        "import sys; from bluffjax.wrappers import PettingZooAECEnv, PettingZooParallelEnv; "
        "assert 'jaxmarl' not in sys.modules",
    ]
    for code in checks:
        subprocess.run([sys.executable, "-c", code], check=True)
