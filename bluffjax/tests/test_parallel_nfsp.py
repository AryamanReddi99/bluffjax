"""Checks for the PPO-NFSP and PQN-NFSP code of the simultaneous-move games
(bluffjax/examples/parallel_nfsp_common.py and the Goofspiel and Kemps
scripts):

- the per-agent GAE and Q(lambda) targets against a brute-force forward
  reference on training-style Goofspiel and Kemps rollouts (auto-resets,
  rollout cut-offs, hands cut off at the horizon, team rewards) and on
  crafted trajectories with hand-computed values;
- Kemps horizon cut-offs against an oracle (the same step without a horizon);
- the reservoir against sequential Algorithm R with the same draws;
- short real runs of all four scripts: BR / average-policy flags constant
  within each agent's game (also across updates) and redrawn at reset, every
  BR decision in the reservoir, PPO on BR decisions with a known outcome and
  the Q-network on all of them with the trace cut at average-policy
  decisions, truncation flags, evaluation schedule;
- the half-budget checkpoint taken after exactly half the updates with the
  schedules of the whole run, the checkpoint layout, save / strict load round
  trip;
- the evaluation's seat rotation and win / loss / draw accounting.
"""

import functools
import os

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from bluffjax import make
from bluffjax.environments.kemps.kemps import Kemps

KEY = jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def common():
    """The training code needs the `baselines` extra (distrax, optax, hydra)."""
    for module in ("distrax", "optax", "hydra"):
        pytest.importorskip(module)
    from bluffjax.examples import parallel_nfsp_common

    return parallel_nfsp_common


def _scripts():
    from bluffjax.examples.goofspiel import goofspiel_ppo_nfsp, goofspiel_pqn_nfsp
    from bluffjax.examples.kemps import kemps_ppo_nfsp, kemps_pqn_nfsp

    return {
        "goofspiel_ppo_nfsp": goofspiel_ppo_nfsp,
        "goofspiel_pqn_nfsp": goofspiel_pqn_nfsp,
        "kemps_ppo_nfsp": kemps_ppo_nfsp,
        "kemps_pqn_nfsp": kemps_pqn_nfsp,
    }


# =============================================================================
# Rollouts as in the training scripts (env.step auto-resets)
# =============================================================================


def kemps_policy_logits(env, mask, p_declare):
    """Random legal actions; a declaration is chosen with weight p_declare
    relative to every other legal action (p_declare = 0: never declare)."""
    game_action = jnp.arange(env.num_actions) // env.comm_dim
    declare = game_action >= env.action_kemps
    weight = jnp.where(declare, jnp.log(jnp.maximum(p_declare, 1e-30)), 0.0)
    weight = jnp.where(declare & (p_declare == 0), -jnp.inf, weight)
    return jnp.where(mask, weight, -jnp.inf)


def training_rollout(common, env, num_envs, num_steps, seed, p_declare=None):
    """Random play with env.step, as in the scripts; returns numpy arrays
    reward (T, N, P), done (T, N), truncated (T, N, P) (the scripts' rule),
    truncated_oracle (T, N) and game_length (T, N)."""
    is_kemps = isinstance(env, Kemps)
    big = (
        Kemps(
            num_agents=env.num_agents,
            num_ranks=env.num_ranks,
            hand_size=env.hand_size,
            num_suits=env.num_suits,
            comm_dim=env.comm_dim,
            horizon=10**9,
        )
        if is_kemps
        else None
    )

    @jax.jit
    def run(key):
        key_reset, key_play = jax.random.split(key)
        state, _ = jax.vmap(env.reset)(jax.random.split(key_reset, num_envs))

        def one(carry, k):
            s, length = carry
            mask = jax.vmap(env.get_avail_actions)(s)
            k_action, k_step = jax.random.split(k)
            logits = jnp.where(mask, 0.0, -jnp.inf)
            if is_kemps and p_declare is not None:
                logits = kemps_policy_logits(env, mask, p_declare)
            action = jax.random.categorical(k_action, logits, axis=-1)
            keys = jax.random.split(k_step, num_envs)
            ns, _, reward, absorbing, done, _ = jax.vmap(env.step)(keys, s, action)
            if is_kemps:
                cutoff = common.kemps_horizon_cutoff(env, s, action, done)
                truncated = done[:, None] & (~absorbing | cutoff[:, None])
                # oracle: the same step (same env rng) without a horizon
                step_keys = jax.vmap(lambda k: jax.random.split(k)[0])(keys)
                done_big = jax.vmap(big.step_env)(step_keys, s, action)[4]
                oracle = done & ~done_big
            else:
                truncated = done[:, None] & ~absorbing
                oracle = jnp.zeros_like(done)
            length = length + 1
            out = (reward, done, truncated, oracle, length)
            return (ns, jnp.where(done, 0, length)), out

        _, out = jax.lax.scan(
            one,
            (state, jnp.zeros((num_envs,), jnp.int32)),
            jax.random.split(key_play, num_steps),
        )
        return out

    return [np.asarray(x) for x in run(jax.random.PRNGKey(seed))]


@pytest.fixture(scope="module")
def rollouts(common):
    """Training-style rollouts: Goofspiel with 2 and 3 players, Kemps with 4
    and 6 players, horizons short enough for many cut-offs."""
    out = {}
    out["goofspiel_2p"] = training_rollout(common, make("goofspiel", num_agents=2), 6, 60, 0)
    out["goofspiel_3p"] = training_rollout(common, make("goofspiel", num_agents=3), 5, 55, 1)
    out["kemps_4p"] = training_rollout(
        common, make("kemps", num_agents=4, horizon=7), 12, 80, 2, p_declare=0.2
    )
    out["kemps_6p"] = training_rollout(
        common, make("kemps", num_agents=6, horizon=5), 10, 60, 3, p_declare=0.15
    )
    return out


# =============================================================================
# Brute-force references (forward sums over each agent's own decisions)
# =============================================================================


def brute_force_gae(values, rewards, dones, truncated, last_value, gamma, lam):
    """A_t = sum_u (gamma lam)^(u - t) delta_u over the agent's decisions from
    t to the end of its known stretch: a game end, the step before a
    truncated decision (whose value is bootstrapped, trace cut) or the
    rollout cut-off (bootstrap from last_value)."""
    T = values.shape[0]
    flat = lambda x: x.reshape(T, -1)
    v, r, d, tr = flat(values), flat(rewards), flat(dones), flat(truncated)
    lv = last_value.reshape(-1)
    g, gl = np.float64(gamma), np.float64(gamma * lam)
    adv = np.zeros(v.shape, np.float64)
    valid = np.zeros(v.shape, bool)
    for c in range(v.shape[1]):
        for t in range(T):
            if tr[t, c]:
                continue
            total, coef, u = 0.0, 1.0, t
            while True:
                if d[u, c]:
                    total += coef * (r[u, c] - v[u, c])
                    break
                if u == T - 1:
                    total += coef * (r[u, c] + g * lv[c] - v[u, c])
                    break
                total += coef * (r[u, c] + g * v[u + 1, c] - v[u, c])
                if tr[u + 1, c]:
                    break
                coef *= gl
                u += 1
            adv[t, c] = total
            valid[t, c] = True
    return adv.reshape(values.shape), valid.reshape(values.shape)


def brute_force_q_lambda(q_max, rewards, dones, truncated, traces, last_q, gamma):
    """G_t = sum over u of (prod_{w=t+1..u} gamma c_w) *
    (r_u + gamma (1 - c_{u+1}) q_{u+1}), stopping at a game end (r_u only),
    before a truncated decision (bootstrap q there, no trace) or at the
    rollout cut-off (bootstrap last_q)."""
    T = q_max.shape[0]
    flat = lambda x: x.reshape(T, -1)
    q, r, d, tr, c = flat(q_max), flat(rewards), flat(dones), flat(truncated), flat(traces)
    lq = last_q.reshape(-1)
    g = np.float64(gamma)
    target = np.zeros(q.shape, np.float64)
    valid = np.zeros(q.shape, bool)
    for col in range(q.shape[1]):
        for t in range(T):
            if tr[t, col]:
                continue
            total, coef, u = 0.0, 1.0, t
            while True:
                if d[u, col]:
                    total += coef * r[u, col]
                    break
                if u == T - 1:
                    total += coef * (r[u, col] + g * lq[col])
                    break
                if tr[u + 1, col]:
                    total += coef * (r[u, col] + g * q[u + 1, col])
                    break
                cn = c[u + 1, col]
                total += coef * (r[u, col] + g * (1.0 - cn) * q[u + 1, col])
                coef *= g * cn
                u += 1
            target[t, col] = total
            valid[t, col] = True
    return target.reshape(q_max.shape), valid.reshape(q_max.shape)


def _broadcast_done(done, reward):
    return np.broadcast_to(done[..., None], reward.shape)


def _value_cases(rng, reward, lam):
    """(values, last values, gamma, lambda, atol) to test with. Halves with
    gamma 1 and lambda 0.5 keep float32 arithmetic exact in these short games
    when the rewards are dyadic (2 players or 2 teams), so the targets must be
    identical to the float64 reference; otherwise (Kemps with 3 teams pays
    3/2 * (1/3 - 1) in float32, random floats) they can differ by rounding."""
    dyadic = (reward * 64 == np.round(reward * 64)).all()
    halves = lambda shape: rng.integers(-8, 9, shape).astype(np.float32) / 2
    normal = lambda shape: rng.normal(size=shape).astype(np.float32)
    shape, last = reward.shape, reward.shape[1:]
    return (
        (halves(shape), halves(last), 1.0, 0.5, 0.0 if dyadic else 1e-6),
        (normal(shape), normal(last), 0.99, lam, 2e-5),
        (normal(shape), normal(last), 1.0, 1.0, 2e-5),
    )


def test_rollouts_have_the_cases_we_need(rollouts):
    for name, (reward, done, truncated, oracle, length) in rollouts.items():
        T, N, P = reward.shape
        assert done.sum() > 3 * N, name  # auto-resets
        assert (~done[-1]).any(), name  # unfinished games at the cut-off
        if name.startswith("kemps"):
            cut = truncated.any(-1)
            assert cut.sum() > 20 and (done & ~cut).sum() > 20, name
            # partners are paid the same, the teams' rewards sum to zero
            k = P // 2
            np.testing.assert_array_equal(reward[..., :k], reward[..., k:])
            np.testing.assert_allclose(reward.sum(-1), 0.0, atol=1e-6)
            assert (reward[done & ~cut] != 0).all(axis=-1).mean() > 0.9
        else:
            assert not truncated.any(), name
            assert (length[done] == 13).all(), name


@pytest.mark.parametrize("name", ["goofspiel_2p", "goofspiel_3p", "kemps_4p", "kemps_6p"])
def test_per_agent_gae_matches_brute_force(common, rollouts, name):
    reward, done, truncated, _, _ = rollouts[name]
    dones = _broadcast_done(done, reward)
    rng = np.random.default_rng(0)
    for values, last_value, gamma, lam, atol in _value_cases(rng, reward, 0.95):
        adv, targets, valid = common.per_agent_gae(
            jnp.asarray(values), jnp.asarray(reward), jnp.asarray(dones),
            jnp.asarray(truncated), jnp.asarray(last_value), gamma, lam,
        )
        ref_adv, ref_valid = brute_force_gae(
            values, reward, dones, truncated, last_value, gamma, lam
        )
        np.testing.assert_array_equal(np.asarray(valid), ref_valid)
        np.testing.assert_allclose(np.asarray(adv), ref_adv, rtol=0, atol=atol)
        np.testing.assert_allclose(
            np.asarray(targets)[ref_valid], (ref_adv + values)[ref_valid],
            rtol=0, atol=atol,
        )
        np.testing.assert_array_equal(np.asarray(adv)[~ref_valid], 0.0)
    np.testing.assert_array_equal(ref_valid, ~truncated)


@pytest.mark.parametrize("name", ["goofspiel_2p", "goofspiel_3p", "kemps_4p", "kemps_6p"])
def test_per_agent_q_lambda_matches_brute_force(common, rollouts, name):
    reward, done, truncated, _, _ = rollouts[name]
    dones = _broadcast_done(done, reward)
    rng = np.random.default_rng(1)
    for q_max, last_q, gamma, lam, atol in _value_cases(rng, reward, 0.9):
        # BR / average-policy agents per game, as in the scripts
        game = np.concatenate([np.zeros((1,) + done.shape[1:], int), np.cumsum(done, 0)[:-1]])
        is_br = rng.random((game.max() + 1,) + reward.shape[1:]) < 0.5
        is_br = np.take_along_axis(is_br, game[..., None].repeat(reward.shape[-1], -1), 0)
        traces = np.float32(lam) * is_br.astype(np.float32)
        targets, valid = common.per_agent_q_lambda_targets(
            jnp.asarray(q_max), jnp.asarray(reward), jnp.asarray(dones),
            jnp.asarray(truncated), jnp.asarray(traces), jnp.asarray(last_q), gamma,
        )
        ref_t, ref_v = brute_force_q_lambda(
            q_max, reward, dones, truncated, traces, last_q, gamma
        )
        np.testing.assert_array_equal(np.asarray(valid), ref_v)
        np.testing.assert_allclose(np.asarray(targets), ref_t, rtol=0, atol=atol)


def _crafted():
    """One Kemps-like env, 4 agents (team 0 = agents 0, 2; team 1 = 1, 3):
        steps 0-2: a hand that team 0 wins at step 2 (done, rewards +1/-1);
        steps 3-4: a new hand, cut off at the horizon at step 4 (done, 0);
        steps 5-6: a new hand, unfinished at the rollout cut-off.
    """
    T, P = 7, 4
    rewards = np.zeros((T, 1, P), np.float32)
    rewards[2, 0] = [1.0, -1.0, 1.0, -1.0]
    done = np.zeros((T, 1), bool)
    done[[2, 4], 0] = True
    truncated = np.zeros((T, 1, P), bool)
    truncated[4, 0] = True
    values = (np.arange(T)[:, None, None] + 0.25 * np.arange(P)).astype(np.float32)
    values = np.broadcast_to(values, (T, 1, P)).copy()
    return rewards, _broadcast_done(done, rewards), truncated, values


def test_per_agent_gae_crafted(common):
    rewards, dones, truncated, values = _crafted()
    last_value = np.array([[10.0, 20.0, 30.0, 40.0]], np.float32)
    adv, targets, valid = map(
        np.asarray,
        common.per_agent_gae(values, rewards, dones, truncated, last_value, 1.0, 1.0),
    )
    np.testing.assert_array_equal(valid[:, 0, 0], [1, 1, 1, 1, 0, 1, 1])
    # first hand: Monte Carlo return of each agent's own (team) reward
    for t in range(3):
        np.testing.assert_array_equal(targets[t, 0], [1.0, -1.0, 1.0, -1.0])
    # the decision before the cut-off bootstraps from the value at the cut
    np.testing.assert_array_equal(targets[3, 0], values[4, 0])
    np.testing.assert_array_equal(adv[4, 0], 0.0)
    # the third hand bootstraps from the value after the last step
    np.testing.assert_array_equal(targets[5, 0], last_value[0])
    np.testing.assert_array_equal(targets[6, 0], last_value[0])
    # lambda = 0.5: A_0 = d0 + d1 / 2 + d2 / 4
    adv, targets, _ = map(
        np.asarray,
        common.per_agent_gae(values, rewards, dones, truncated, last_value, 1.0, 0.5),
    )
    d = [values[1] - values[0], values[2] - values[1], rewards[2] - values[2]]
    np.testing.assert_allclose(adv[0], d[0] + d[1] / 2 + d[2] / 4, rtol=0, atol=1e-6)
    np.testing.assert_allclose(adv[3], values[4] - values[3], rtol=0, atol=1e-6)


def test_per_agent_q_lambda_crafted(common):
    rewards, dones, truncated, q_max = _crafted()
    lam = 0.5
    # agents 0 and 1 follow the BR in every hand, agents 2 and 3 the average policy
    traces = np.broadcast_to(np.float32([lam, lam, 0.0, 0.0]), q_max.shape).copy()
    last_q = np.array([[10.0, 20.0, 30.0, 40.0]], np.float32)
    targets, valid = map(
        np.asarray,
        common.per_agent_q_lambda_targets(
            q_max, rewards, dones, truncated, traces, last_q, 1.0
        ),
    )
    np.testing.assert_array_equal(valid[:, 0, 0], [1, 1, 1, 1, 0, 1, 1])
    r2 = rewards[2, 0]
    np.testing.assert_array_equal(targets[2, 0], r2)  # terminal
    # BR agents: Peng's Q(lambda) through the next (BR) decision
    g1 = (1 - lam) * q_max[2, 0] + lam * r2
    np.testing.assert_allclose(targets[1, 0, :2], g1[:2])
    np.testing.assert_allclose(
        targets[0, 0, :2], ((1 - lam) * q_max[1, 0] + lam * g1)[:2]
    )
    # average-policy agents: one-step Q-learning targets
    np.testing.assert_array_equal(targets[1, 0, 2:], q_max[2, 0, 2:])
    np.testing.assert_array_equal(targets[0, 0, 2:], q_max[1, 0, 2:])
    # before the cut-off: bootstrap from q there without the trace
    np.testing.assert_array_equal(targets[3, 0], q_max[4, 0])
    # rollout cut-off
    np.testing.assert_array_equal(targets[6, 0], last_q[0])
    np.testing.assert_allclose(
        targets[5, 0, :2], ((1 - lam) * q_max[6, 0] + lam * last_q[0])[:2]
    )
    np.testing.assert_array_equal(targets[5, 0, 2:], q_max[6, 0, 2:])


# =============================================================================
# Kemps horizon cut-offs
# =============================================================================


@pytest.mark.parametrize("name", ["kemps_4p", "kemps_6p"])
def test_kemps_cutoff_matches_oracle(rollouts, name):
    reward, done, truncated, oracle, length = rollouts[name]
    np.testing.assert_array_equal(truncated, np.broadcast_to(oracle[..., None], truncated.shape))
    horizon = {"kemps_4p": 7, "kemps_6p": 5}[name]
    assert (length[oracle] == horizon).all()
    assert (reward[oracle] == 0).all()
    assert (length[done] <= horizon).all()


@pytest.mark.parametrize("horizon, cut", [(8, True), (9, False), (20, False)])
def test_kemps_real_deal_is_not_a_cutoff(common, horizon, cut):
    """Everybody plays NOOP: 8 sweeps empty the 32-card stock, the 9th NOOP
    step is a real deal. With horizon 8 the hand is cut off first."""
    env = make("kemps", num_agents=4, horizon=horizon)
    state, _ = env.reset(KEY)
    noop = jnp.full((4,), env.action_noop * env.comm_dim, jnp.int32)
    step = jax.jit(env.step_env)
    for t in range(9):
        next_state, _, reward, absorbing, done, _ = step(KEY, state, noop)
        if done:
            break
        state = next_state
    assert t + 1 == min(horizon, 9) and bool(done)
    np.testing.assert_array_equal(np.asarray(reward), 0.0)
    batched = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], state)
    assert bool(common.kemps_horizon_cutoff(env, batched, noop[None], done[None])[0]) == cut


# =============================================================================
# Reservoir
# =============================================================================


def test_reservoir_matches_algorithm_r(common):
    capacity, dim = 50, 3
    buffer = common.init_sl_buffer(capacity, dim, 4)
    ref = np.zeros((capacity, dim), np.float32)
    seen = 0
    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(3)
    append = jax.jit(common.reservoir_append)
    size = 32
    for batch in range(12):
        obs = rng.normal(size=(size, dim)).astype(np.float32)
        valid = rng.random(size) < rng.uniform(0.1, 0.9)
        key, k = jax.random.split(key)
        buffer = append(
            buffer, jnp.asarray(obs), jnp.ones((size, 4), bool),
            jnp.arange(size, dtype=jnp.int32), jnp.asarray(valid), k,
        )
        # Algorithm R with the same draws j ~ U{0..k} for the k-th valid item
        v = valid.astype(np.int32)
        k_items = seen + np.cumsum(v) - v
        j = np.asarray(jax.random.randint(k, (size,), 0, jnp.asarray(k_items) + 1, dtype=jnp.int32))
        for i in range(size):
            if not valid[i]:
                continue
            if seen < capacity:
                ref[seen] = obs[i]
            elif j[i] < capacity:
                ref[j[i]] = obs[i]
            seen += 1
        assert int(buffer.seen) == seen and int(buffer.size) == min(seen, capacity)
        np.testing.assert_array_equal(np.asarray(buffer.obs), ref)
    assert seen > 3 * capacity


# =============================================================================
# Short real runs of the scripts
# =============================================================================


class _Recorder:
    """Stands in for the wandb run of a seed."""

    def __init__(self):
        self.rows = []

    def log(self, metrics):
        self.rows.append(metrics)


def tiny_config(name, **extra):
    game, algo = name.split("_")[0], name.split("_")[1]
    config = dict(
        num_envs=16, num_steps_per_env_per_update=16, num_epochs=2,
        num_minibatches=2, lr=2.5e-4, anneal_lr=True, gamma=1.0,
        max_grad_norm=0.5, reward_scale=0.1 if game == "goofspiel" else 1.0,
        fc_dim_size=16, anticipatory_eta=0.3, sl_reservoir_capacity=200,
        sl_batch_size=32, sl_num_steps_per_update=2, sl_lr=3e-4,
        env_name=game, seed=0, num_seeds=2,
        env_kwargs={"num_agents": 2} if game == "goofspiel" else {"num_agents": 4, "horizon": 4},
        compare=True, compare_mode="random", compare_with="",
        compare_network_type="actor_critic" if algo == "ppo" else "actor",
        compare_episodes=16, compare_interval=2, project="bluffjax",
        job_type=name, wandb=False, save_checkpoints=True, num_update_steps=5,
    )
    if algo == "ppo":
        config.update(gae_lambda=0.95, clip_eps=0.2, vf_clip=0.2, ent_coef=0.0, vf_coef=1.0)
    else:
        config.update(start_e=0.5, end_e=0.05, exploration_fraction=0.5, q_lambda=0.9)
    config.update(extra)
    config["num_gradient_steps"] = (
        config["num_update_steps"] * config["num_epochs"] * config["num_minibatches"]
    )
    return config


def _is_constant_per_game(is_br, done):
    """is_br (T, N, P) is constant within each agent's game; returns the flag
    of every (env, agent, game) that started in the rollout."""
    T, N, P = is_br.shape
    game = np.concatenate([np.zeros((1, N), int), np.cumsum(done, 0)[:-1]])
    flags = {}
    for (t, n, p), f in np.ndenumerate(is_br):
        key = (n, p, game[t, n])
        assert flags.setdefault(key, f) == f, (t, n, p)
    # drop the game in progress at the start (drawn at init)
    return np.array([f for (n, p, g), f in flags.items() if g > 0])


SCRIPT_NAMES = ["goofspiel_ppo_nfsp", "goofspiel_pqn_nfsp", "kemps_ppo_nfsp", "kemps_pqn_nfsp"]


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_real_run_mixing_targets_and_reservoir(common, monkeypatch, name):
    mod = _scripts()[name]
    is_ppo = "ppo" in name
    seen = {"targets": [], "reservoir": []}

    def record(key, *arrays):
        jax.debug.callback(lambda *a: seen[key].append([np.asarray(x) for x in a]), *arrays)

    reservoir = mod.reservoir_append

    def reservoir_spy(buffer, obs, action_mask, action, valid, rng):
        record("reservoir", valid, action_mask, action)
        return reservoir(buffer, obs, action_mask, action, valid, rng)

    monkeypatch.setattr(mod, "reservoir_append", reservoir_spy)
    if is_ppo:
        gae = mod.per_agent_gae

        def targets_spy(values, rewards, dones, truncated, last_value, g, lam):
            out = gae(values, rewards, dones, truncated, last_value, g, lam)
            record("targets", rewards, dones, truncated, out[2])
            return out

        monkeypatch.setattr(mod, "per_agent_gae", targets_spy)
    else:
        q_targets = mod.per_agent_q_lambda_targets

        def targets_spy(q_max, rewards, dones, truncated, traces, last_q, g):
            out = q_targets(q_max, rewards, dones, truncated, traces, last_q, g)
            record("targets", rewards, dones, truncated, out[1], traces)
            return out

        monkeypatch.setattr(mod, "per_agent_q_lambda_targets", targets_spy)
    logger = _Recorder()
    monkeypatch.setattr(mod, "WANDB_RUNS", [logger])
    config = tiny_config(name)
    jax.block_until_ready(jax.jit(mod.make_train(config))(KEY, jnp.int32(0)))
    jax.effects_barrier()

    U = config["num_update_steps"]
    T, N = config["num_steps_per_env_per_update"], config["num_envs"]
    P = config["env_kwargs"]["num_agents"]
    assert len(seen["targets"]) == len(seen["reservoir"]) == len(logger.rows) == U
    is_br = np.concatenate([r[0].reshape(T, N, P) for r in seen["reservoir"]])
    rewards = np.concatenate([r[0] for r in seen["targets"]])
    dones = np.concatenate([r[1] for r in seen["targets"]])
    truncated = np.concatenate([r[2] for r in seen["targets"]])
    done = dones[..., 0]
    assert (dones == done[..., None]).all()
    # mixing: constant within each agent's game, also across updates, and
    # redrawn independently for every agent at every reset
    flags = _is_constant_per_game(is_br, done)
    assert len(flags) > 100
    eta = config["anticipatory_eta"]
    assert abs(flags.mean() - eta) < 4 * np.sqrt(eta * (1 - eta) / len(flags))
    # game ends and cut-offs
    length = np.zeros_like(done, dtype=int)
    run = np.zeros(N, int)
    for t in range(len(done)):
        run += 1
        length[t] = run
        run = np.where(done[t], 0, run)
    if name.startswith("goofspiel"):
        game_end = np.arange(len(done)) % 13 == 12  # every game is 13 rounds
        np.testing.assert_array_equal(done, np.broadcast_to(game_end[:, None], done.shape))
        assert not truncated.any()
    else:
        horizon = config["env_kwargs"]["horizon"]
        cut = truncated.any(-1)
        assert (truncated == cut[..., None]).all() and (cut <= done).all()
        assert (length[cut] == horizon).all() and (rewards[cut] == 0).all()
        assert ((length[done & ~cut] < horizon) | (rewards[done & ~cut] != 0).any(-1)).all()
        assert cut.sum() > 5 and (done & ~cut).sum() > 5
    for u, row in enumerate(logger.rows):
        valid = seen["targets"][u][3]
        br = seen["reservoir"][u][0].reshape(T, N, P)
        np.testing.assert_array_equal(valid, ~seen["targets"][u][2])
        if is_ppo:
            assert row["ppo_sample_frac"] == np.float32((valid & br).mean())
        else:
            assert row["q_sample_frac"] == np.float32(valid.mean())
            np.testing.assert_array_equal(
                seen["targets"][u][4], np.float32(config["q_lambda"]) * br.astype(np.float32)
            )
        # every BR decision goes to the reservoir with its legal mask
        mask, action = seen["reservoir"][u][1], seen["reservoir"][u][2]
        assert np.take_along_axis(mask, action[:, None], 1).all()
        assert row["sl_buffer_seen"] == is_br[: (u + 1) * T].sum()
        has_eval = "win_rate_avg_vs_random" in row
        assert has_eval == (u % config["compare_interval"] == 0 or u == U - 1)
        if has_eval:
            for who in ("avg", "br"):
                total = sum(row[f"{k}_rate_{who}_vs_random"] for k in ("win", "loss", "draw"))
                assert total == pytest.approx(1.0)
    # the per-agent returns logged for ended games are the env's
    assert logger.rows[-1]["games_completed"] == done[-T:].sum()


def _params_close(a, b, atol=1e-6):
    leaves = zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    return all(np.allclose(np.asarray(x), np.asarray(y), rtol=0, atol=atol) for x, y in leaves)


def _max_diff(a, b):
    leaves = zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    return max(float(np.abs(np.asarray(x) - np.asarray(y)).max()) for x, y in leaves)


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_half_checkpoint_and_round_trip(common, monkeypatch, tmp_path, name):
    mod = _scripts()[name]
    monkeypatch.setattr(mod, "WANDB_RUNS", [_Recorder(), _Recorder()])
    U = 5  # half after 2 updates (floor)
    config = tiny_config(name, num_update_steps=U, compare=False)
    config["save_dir"] = str(tmp_path)
    config["half_update_steps"] = common.half_update_steps(U)
    assert config["half_update_steps"] == 2
    train = mod.make_train(config)
    keys = jax.random.split(KEY, 2)
    seeds = jnp.arange(2)
    full = jax.jit(jax.vmap(train))(keys, seeds)
    partial = {
        n: jax.jit(jax.vmap(functools.partial(train, num_updates=n)))(keys, seeds)
        for n in (1, 2, 3)
    }
    jax.effects_barrier()
    E, M = config["num_epochs"], config["num_minibatches"]
    assert (np.asarray(full.update_step) == U).all()
    assert (np.asarray(full.br_train_state.step) == U * E * M).all()
    assert (np.asarray(partial[2].br_train_state.step) == 2 * E * M).all()
    for net, attr in (("avg_params", "avg_train_state"), ("br_params", "br_train_state")):
        half = getattr(full.half, net)
        at = {n: getattr(partial[n], attr).params for n in partial}
        # the half checkpoint is the state of the run after exactly 2 updates
        assert _params_close(half, at[2]), _max_diff(half, at[2])
        assert _max_diff(half, at[1]) > 1e-4 and _max_diff(half, at[3]) > 1e-4
        assert _max_diff(half, getattr(full, attr).params) > 1e-4
    # the learning rate kept decaying over the whole run (not restarted)
    lr = lambda step: config["lr"] * (1 - step / config["num_gradient_steps"])
    assert lr(2 * E * M) == pytest.approx(0.6 * config["lr"])

    full_ckpt = common.Checkpoint(full.avg_train_state.params, full.br_train_state.params)
    run_dir = common.save_checkpoints(config, full.half, full_ckpt)
    assert os.path.basename(run_dir).startswith(f"{name}_")
    expected = {"config.yaml"} | {
        f"{stage}/{net}_{i}.msgpack"
        for stage in ("half", "full") for net in ("avg_policy", "br") for i in range(2)
    }
    found = {
        os.path.relpath(os.path.join(d, f), run_dir)
        for d, _, files in os.walk(run_dir) for f in files
    }
    assert found == expected
    from omegaconf import OmegaConf

    assert OmegaConf.to_container(OmegaConf.load(os.path.join(run_dir, "config.yaml"))) == config

    from bluffjax.networks.mlp import ActorCriticDiscreteMLP, ActorDiscreteMLP, QNetworkDiscreteMLP

    env = make(config["env_name"], **config["env_kwargs"])
    obs = jnp.zeros((env.obs_dim,))
    A, H = env.num_actions, config["fc_dim_size"]
    if "ppo" in name:
        avg_net = br_net = ActorCriticDiscreteMLP(A, H)
        wrong = QNetworkDiscreteMLP(A, H)
    else:
        avg_net, br_net = ActorDiscreteMLP(A, H), QNetworkDiscreteMLP(A, H)
        wrong = ActorCriticDiscreteMLP(A, H)
    avg_t = avg_net.init(jax.random.PRNGKey(5), obs)
    br_t = br_net.init(jax.random.PRNGKey(6), obs)
    for stage, ckpt in (("half", full.half), ("full", full_ckpt)):
        for i in range(2):
            loaded = common.load_checkpoint(run_dir, stage, i, avg_t, br_t)
            pairs = ((loaded.avg_params, ckpt.avg_params), (loaded.br_params, ckpt.br_params))
            for got, want in pairs:
                jax.tree_util.tree_map(
                    lambda g, w: np.testing.assert_array_equal(np.asarray(g), np.asarray(w)[i]),
                    got, want,
                )
    # strict loading
    with pytest.raises(ValueError):
        common.load_params(
            common.checkpoint_path(run_dir, "full", "br", 0), wrong.init(KEY, obs)
        )
    with pytest.raises(ValueError):
        common.load_params(
            common.checkpoint_path(run_dir, "full", "avg_policy", 0),
            type(avg_net)(A, 2 * H).init(KEY, obs),
        )
    with pytest.raises(FileNotFoundError):
        common.load_checkpoint(run_dir, "full", 2, avg_t, br_t)
    with pytest.raises(ValueError):
        common.checkpoint_path(run_dir, "quarter", "br", 0)
    # a second save in the same second gets its own directory
    assert common.save_checkpoints(config, full.half, full_ckpt) != run_dir

    # the saved networks as evaluation opponents (compare_mode=checkpoint)
    types = {"br": "actor_critic", "avg_policy": "actor_critic"}
    if "pqn" in name:
        types = {"br": "q_network", "avg_policy": "actor"}
    for net, network_type in types.items():
        logger = _Recorder()
        monkeypatch.setattr(mod, "WANDB_RUNS", [logger])
        eval_config = tiny_config(
            name, num_update_steps=1, compare_mode="checkpoint",
            compare_with=common.checkpoint_path(run_dir, "full", net, 1),
            compare_network_type=network_type, compare_episodes=8,
        )
        jax.block_until_ready(jax.jit(mod.make_train(eval_config))(KEY, jnp.int32(0)))
        jax.effects_barrier()
        row = logger.rows[0]
        for who in ("avg", "br"):
            total = sum(row[f"{k}_rate_{who}_vs_baseline"] for k in ("win", "loss", "draw"))
            assert total == pytest.approx(1.0)
        wrong_type = "q_network" if network_type != "q_network" else "actor"
        with pytest.raises(ValueError):
            mod.make_train(dict(eval_config, compare_network_type=wrong_type))
    with pytest.raises(FileNotFoundError):
        mod.make_train(dict(eval_config, compare_with=""))


# =============================================================================
# Evaluation
# =============================================================================


def test_rotating_seats_and_teams(common):
    seats = np.asarray(common.rotating_seats(KEY, 10, 3))
    assert (seats.sum(1) == 1).all()
    idx = seats.argmax(1)
    np.testing.assert_array_equal(idx, (idx[0] + np.arange(10)) % 3)
    env = make("kemps", num_agents=6)
    team_of_seat = env._team_id(jnp.arange(6))
    teams = np.asarray(common.rotating_teams(KEY, 9, team_of_seat, env.num_teams))
    assert (teams.sum(1) == 2).all()
    for g in range(9):
        a, b = np.flatnonzero(teams[g])
        assert int(env._partner_idx(a)) == b  # the learner plays a team of partners
    first = np.flatnonzero(teams[:, 0])
    np.testing.assert_array_equal(np.diff(first), 3)


def _random_policy(common):
    return lambda params, obs, mask, rng: common.sample_random_legal(mask, rng)


def test_goofspiel_evaluation(common):
    env = make("goofspiel", num_agents=2)
    rand = _random_policy(common)

    def match_prize(params, obs, mask, rng):
        """Bid the card equal to the prize (always still in hand)."""
        prize = jnp.argmax(obs[..., :13], axis=-1)
        return jnp.where(jnp.take_along_axis(mask, prize[..., None], -1)[..., 0], prize, -1)

    n = 2000
    result = jax.jit(
        lambda k: common.evaluate_goofspiel(env, rand, None, rand, None, n, k)
    )(KEY)
    win, loss, draw = map(np.asarray, (result.win, result.loss, result.draw))
    np.testing.assert_array_equal(win + loss + draw, 1.0)
    assert (np.asarray(result.length) == 13).all()
    se = np.sqrt(0.25 / n)
    assert abs(win.mean() - loss.mean()) < 5 * 2 * se and 0 < draw.mean() < 0.1
    assert abs(np.asarray(result.ret).mean()) < 5 * np.asarray(result.ret).std() / np.sqrt(n)
    # outcome vs return: with two players a win is a positive point difference
    ret = np.asarray(result.ret)
    np.testing.assert_array_equal(win, ret > 0)
    np.testing.assert_array_equal(loss, ret < 0)
    # the zero-sum record of every seat
    seats = common.rotating_seats(KEY, 64, 2)
    record = jax.jit(
        lambda k: common.play_games(env, match_prize, None, rand, None, seats, k)
    )(KEY)
    np.testing.assert_allclose(np.asarray(record.returns).sum(-1), 0.0, atol=1e-5)
    result = jax.jit(
        lambda k: common.evaluate_goofspiel(env, match_prize, None, rand, None, n, k)
    )(KEY)
    assert float(result.win.mean()) > 0.6 and float(result.ret.mean()) > 10.0


@pytest.mark.parametrize("horizon", [3, 200])
def test_kemps_evaluation(common, horizon):
    env = make("kemps", num_agents=4, horizon=horizon)
    rand = _random_policy(common)

    def never_declare(params, obs, mask, rng):
        return jax.random.categorical(rng, kemps_policy_logits(env, mask, 0.0))

    n = 1000
    result = jax.jit(
        lambda k: common.evaluate_kemps(env, rand, None, rand, None, n, k)
    )(KEY)
    win, loss, draw = map(np.asarray, (result.win, result.loss, result.draw))
    np.testing.assert_array_equal(win + loss + draw, 1.0)
    assert (np.asarray(result.length) <= horizon).all()
    np.testing.assert_array_equal(np.asarray(result.ret)[draw == 1], 0.0)
    if horizon == 200:
        assert abs(win.mean() - loss.mean()) < 5 * 2 * np.sqrt(0.25 / n)
    result = jax.jit(
        lambda k: common.evaluate_kemps(env, never_declare, None, never_declare, None, 64, k)
    )(KEY)
    # nobody declares (and an all-NOOP real deal is all but impossible):
    # every hand is cut off at the horizon, a draw
    np.testing.assert_array_equal(np.asarray(result.draw), 1.0)
    np.testing.assert_array_equal(np.asarray(result.length), horizon)
    result = jax.jit(
        lambda k: common.evaluate_kemps(env, never_declare, None, rand, None, n, k)
    )(KEY)
    if horizon == 200:  # random players' calls are almost always wrong
        assert float(result.win.mean()) > 0.9
    # partners are paid the same, the teams opposite amounts
    seats = common.rotating_teams(KEY, 64, env._team_id(jnp.arange(4)), 2)
    record = jax.jit(
        lambda k: common.play_games(env, never_declare, None, rand, None, seats, k)
    )(KEY)
    returns = np.asarray(record.returns)
    np.testing.assert_array_equal(returns[:, :2], returns[:, 2:])
    np.testing.assert_array_equal(returns[:, 0], -returns[:, 1])
