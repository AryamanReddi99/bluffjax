"""Checks for the Kuhn PPO, PQN, PPO-NFSP and PQN-NFSP training scripts.

- per_player_gae and per_player_q_lambda_targets match a brute-force reference
  that lists every player's decisions hand by hand, on a scripted Kuhn
  trajectory, on random Kuhn rollouts with extra non-terminal rewards and on
  random turn sequences in which the same player can act several times in a
  row.
- reservoir_append writes exactly what Algorithm R writes when it processes
  the samples one at a time with the same random draws.
- In a short NFSP training run each player keeps its BR / average-policy draw
  for the whole hand (also across updates), every BR decision goes to the
  reservoir, PPO trains on BR decisions and PQN on all of them.
- The network-to-policy helpers of kuhn_exploitability: rows follow the env's
  observations for both start players, the Q-network policy is greedy, and the
  exploitability of known policies is unchanged.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.examples.kuhn import kuhn_ppo, kuhn_ppo_nfsp, kuhn_pqn, kuhn_pqn_nfsp
from bluffjax.utils.game_utils import kuhn_exploitability as ke

GAE_FNS = {
    "kuhn_ppo": kuhn_ppo.per_player_gae,
    "kuhn_ppo_nfsp": kuhn_ppo_nfsp.per_player_gae,
}
Q_FNS = {
    "kuhn_pqn": kuhn_pqn.per_player_q_lambda_targets,
    "kuhn_pqn_nfsp": kuhn_pqn_nfsp.per_player_q_lambda_targets,
}
RESERVOIR_FNS = {
    "kuhn_ppo_nfsp": kuhn_ppo_nfsp.reservoir_append,
    "kuhn_pqn_nfsp": kuhn_pqn_nfsp.reservoir_append,
}
PASS, BET = 0, 1


# ------------------------------------------------------------------ reference
def _decisions(rewards, dones, players, last_player, n):
    """Every decision of env n as (t, kind, R), listed hand by hand.

    R is the acting player's reward summed from its decision up to (excluding)
    its next decision in the same hand, or up to the end of the hand. kind is
    "next" (the player acts again in this hand; the index of that decision is
    returned too), "terminal" (the hand ends first), "cutoff" (the rollout ends
    first and the player is the one to act next) or "unknown".
    """
    T = rewards.shape[0]
    out = {}
    start = 0
    while start < T:
        end = start
        while end < T - 1 and not dones[end, n]:
            end += 1
        finished = bool(dones[end, n])
        steps = list(range(start, end + 1))
        for p in set(int(players[t, n]) for t in steps):
            own = [t for t in steps if players[t, n] == p]
            for i, t in enumerate(own):
                stop = own[i + 1] if i + 1 < len(own) else end + 1
                R = sum(float(rewards[k, n, p]) for k in range(t, stop))
                if i + 1 < len(own):
                    out[t] = ("next", own[i + 1], R)
                elif finished:
                    out[t] = ("terminal", None, R)
                elif last_player[n] == p:
                    out[t] = ("cutoff", None, R)
                else:
                    out[t] = ("unknown", None, R)
        start = end + 1
    return out


def ref_gae(values, rewards, dones, players, last_value, last_player, gamma, lam):
    T, N = values.shape
    adv = np.zeros((T, N))
    valid = np.zeros((T, N), dtype=bool)
    for n in range(N):
        dec = _decisions(rewards, dones, players, last_player, n)
        A = {}
        for t in reversed(range(T)):
            kind, s, R = dec[t]
            v = values[t, n]
            if kind == "terminal":
                A[t] = R - v
            elif kind == "cutoff":
                A[t] = R + gamma * last_value[n] - v
            elif kind == "next":
                trace = A[s] if A[s] is not None else 0.0
                A[t] = R + gamma * values[s, n] - v + gamma * lam * trace
            else:
                A[t] = None
            valid[t, n] = A[t] is not None
            adv[t, n] = A[t] if A[t] is not None else 0.0
    return adv, adv + values, valid


def ref_q_lambda(q_max, rewards, dones, players, traces, last_q, last_player, gamma):
    T, N = q_max.shape
    out = np.zeros((T, N))
    valid = np.zeros((T, N), dtype=bool)
    for n in range(N):
        dec = _decisions(rewards, dones, players, last_player, n)
        G = {}
        for t in reversed(range(T)):
            kind, s, R = dec[t]
            if kind == "terminal":
                G[t] = R
            elif kind == "cutoff":
                G[t] = R + gamma * last_q[n]
            elif kind == "next" and G[s] is None:
                G[t] = R + gamma * q_max[s, n]
            elif kind == "next":
                c = traces[s, n]
                G[t] = R + gamma * ((1 - c) * q_max[s, n] + c * G[s])
            else:
                G[t] = None
            valid[t, n] = G[t] is not None
            out[t, n] = G[t] if G[t] is not None else 0.0
    return out, valid


def _run_gae(fn, values, rewards, dones, players, last_value, last_player, g, lam):
    out = fn(
        jnp.asarray(values, jnp.float32),
        jnp.asarray(rewards, jnp.float32),
        jnp.asarray(dones),
        jnp.asarray(players, jnp.int32),
        jnp.asarray(last_value, jnp.float32),
        jnp.asarray(last_player, jnp.int32),
        g,
        lam,
    )
    return [np.asarray(x) for x in out]


def _run_q(fn, q_max, rewards, dones, players, traces, last_q, last_player, g):
    out = fn(
        jnp.asarray(q_max, jnp.float32),
        jnp.asarray(rewards, jnp.float32),
        jnp.asarray(dones),
        jnp.asarray(players, jnp.int32),
        jnp.asarray(traces, jnp.float32),
        jnp.asarray(last_q, jnp.float32),
        jnp.asarray(last_player, jnp.int32),
        g,
    )
    return [np.asarray(x) for x in out]


def _check_all(values, q_max, traces, rewards, dones, players, last_player, rs):
    """All four target functions vs the reference for several (gamma, lambda)."""
    last_value = rs.randn(values.shape[1])
    last_q = rs.randn(values.shape[1])
    for gamma, lam in [(1.0, 1.0), (1.0, 0.95), (1.0, 0.0), (0.9, 0.5)]:
        ra, rt, rv = ref_gae(
            values, rewards, dones, players, last_value, last_player, gamma, lam
        )
        for name, fn in GAE_FNS.items():
            a, tg, v = _run_gae(
                fn, values, rewards, dones, players, last_value, last_player, gamma, lam
            )
            np.testing.assert_array_equal(v, rv, err_msg=name)
            np.testing.assert_allclose(a, ra, atol=1e-5, err_msg=name)
            np.testing.assert_allclose(tg[v], rt[rv], atol=1e-5, err_msg=name)
        tr = lam * traces
        rq, rqv = ref_q_lambda(
            q_max, rewards, dones, players, tr, last_q, last_player, gamma
        )
        for name, fn in Q_FNS.items():
            tq, vq = _run_q(
                fn, q_max, rewards, dones, players, tr, last_q, last_player, gamma
            )
            np.testing.assert_array_equal(vq, rqv, err_msg=name)
            np.testing.assert_allclose(tq[vq], rq[rqv], atol=1e-5, err_msg=name)
            assert (tq[~vq] == 0).all(), name


# ------------------------------------------------------------- Kuhn rollouts
@pytest.fixture(scope="module")
def env():
    return make("kuhn_poker")


def _scripted(env, script, seed):
    """Plays the actions in script (auto-resetting) with a fixed deal sequence."""
    step = jax.jit(env.step)
    state, _ = env.reset(jax.random.PRNGKey(seed))
    players, rewards, dones = [], [], []
    for t, a in enumerate(script):
        players.append(int(state.current_player_idx))
        state, _, r, _, d, _ = step(jax.random.PRNGKey(1000 + t), state, jnp.int32(a))
        rewards.append(np.asarray(r))
        dones.append(bool(d))
    return (
        np.array(players)[:, None],
        np.array(rewards)[:, None, :],
        np.array(dones)[:, None],
        np.array([int(state.current_player_idx)]),
    )


def test_targets_scripted_kuhn(env) -> None:
    """Pass-bet-call, bet-fold, pass-pass, then a cut after pass-bet."""
    script = [PASS, BET, BET, BET, PASS, PASS, PASS, PASS, BET]
    players, rewards, dones, last_player = _scripted(env, script, seed=3)
    np.testing.assert_array_equal(dones[:, 0], [0, 0, 1, 0, 1, 0, 1, 0, 0])
    # the cut: the opener passed, the other player bet, the opener is to act
    assert last_player[0] == players[7, 0] != players[8, 0]
    T = len(script)
    rs = np.random.RandomState(0)
    values = 0.1 * np.arange(1, T + 1)[:, None]
    q_max = np.arange(1, T + 1, dtype=np.float64)[:, None]
    traces = np.ones((T, 1))
    traces[2, 0] = 0.0  # cut the trace at the call
    _check_all(values, q_max, traces, rewards, dones, players, last_player, rs)
    # extra non-terminal rewards for both players on every step
    extra = rs.randn(*rewards.shape)
    _check_all(values, q_max, traces, rewards + extra, dones, players, last_player, rs)

    # Q(lambda = 0) by hand: the opener's pass bootstraps from its own call
    # (q_max[2] = 3), not from the next state; the bettor in the cut-off hand
    # has an unknown outcome.
    tq, vq = _run_q(
        kuhn_pqn.per_player_q_lambda_targets, q_max, rewards, dones, players,
        np.zeros((T, 1)), np.array([10.0]), last_player, 1.0,
    )
    r = rewards[:, 0, :]
    p = players[:, 0]
    expected = [
        3.0,  # opener's pass -> its call at t = 2
        r[2, p[1]],
        r[2, p[2]],
        r[4, p[3]],
        r[4, p[4]],
        r[6, p[5]],
        r[6, p[6]],
        10.0,  # opener's pass in the cut hand -> last_q
    ]
    np.testing.assert_allclose(tq[:8, 0], expected, atol=1e-6)
    np.testing.assert_array_equal(vq[:, 0], [1] * 8 + [0])


def _random_rollout(env, num_envs, num_steps, seed):
    rng = jax.random.PRNGKey(seed)
    state, _ = jax.vmap(env.reset)(jax.random.split(rng, num_envs))

    def body(state, key):
        k1, k2 = jax.random.split(key)
        mask = jax.vmap(env.get_avail_actions)(state)
        action = jax.random.categorical(k1, jnp.where(mask, 0.0, -jnp.inf))
        player = state.current_player_idx
        state, _, reward, _, done, _ = jax.vmap(env.step)(
            jax.random.split(k2, num_envs), state, action
        )
        return state, (player, reward, done)

    state, (players, rewards, dones) = jax.lax.scan(
        body, state, jax.random.split(rng, num_steps)
    )
    return (
        np.asarray(players),
        np.asarray(rewards),
        np.asarray(dones),
        np.asarray(state.current_player_idx),
    )


@pytest.mark.parametrize("seed", range(4))
def test_targets_random_kuhn(env, seed: int) -> None:
    """Random Kuhn rollouts, with extra non-terminal rewards on 20% of steps."""
    players, rewards, dones, last_player = _random_rollout(env, 48, 37, seed)
    rs = np.random.RandomState(seed)
    T, N = players.shape
    extra = (rs.rand(T, N, 2) < 0.2) * rs.randn(T, N, 2)
    traces = (rs.rand(T, N) < 0.7).astype(np.float64)
    _check_all(
        rs.randn(T, N), rs.randn(T, N), traces, rewards + extra, dones, players,
        last_player, rs,
    )


@pytest.mark.parametrize("seed", range(4))
def test_targets_random_turns(seed: int) -> None:
    """Arbitrary turn orders (the same player may act several times in a row)."""
    rs = np.random.RandomState(100 + seed)
    T, N = 29, 32
    players = rs.randint(0, 2, size=(T, N))
    dones = rs.rand(T, N) < 0.25
    rewards = (rs.rand(T, N, 2) < 0.4) * rs.randn(T, N, 2)
    last_player = rs.randint(0, 2, size=N)
    traces = (rs.rand(T, N) < 0.7).astype(np.float64)
    _check_all(
        rs.randn(T, N), rs.randn(T, N), traces, rewards, dones, players,
        last_player, rs,
    )


# ------------------------------------------------------------------ reservoir
def _algorithm_r(buf, seen, items, valid, j):
    """Sequential Algorithm R; j[i] is the draw from U{0..k} for item i."""
    buf = buf.copy()
    capacity = len(buf)
    for i in range(len(items)):
        if not valid[i]:
            continue
        if seen < capacity:
            buf[seen] = items[i]
        elif j[i] < capacity:
            buf[j[i]] = items[i]
        seen += 1
    return buf, seen


@pytest.mark.parametrize("name", sorted(RESERVOIR_FNS))
def test_reservoir_matches_algorithm_r(name: str) -> None:
    fn = jax.jit(RESERVOIR_FNS[name])
    buffer_cls = (
        kuhn_ppo_nfsp.SLBufferState if name == "kuhn_ppo_nfsp" else kuhn_pqn_nfsp.SLBufferState
    )
    capacity, batch = 37, 50
    rs = np.random.RandomState(0)
    buffer = buffer_cls(
        obs=jnp.zeros((capacity, 9), jnp.float32),
        action_mask=jnp.zeros((capacity, 2), jnp.bool_),
        action=jnp.zeros((capacity,), jnp.int32),
        seen=jnp.int32(0),
        size=jnp.int32(0),
    )
    ref_buf = np.full(capacity, -1, dtype=np.int64)
    ref_seen = 0
    item_id = 0
    for i in range(12):
        valid = rs.rand(batch) < (0.3 if i < 2 else 0.8)
        ids = np.arange(item_id, item_id + batch)
        item_id += batch
        obs = np.zeros((batch, 9), np.float32)
        obs[:, 0] = ids  # the item id, to identify what was kept
        rng = jax.random.PRNGKey(i)
        buffer = fn(
            buffer, jnp.asarray(obs), jnp.ones((batch, 2), jnp.bool_),
            jnp.asarray(ids % 2, jnp.int32), jnp.asarray(valid), rng,
        )
        # the same draws as reservoir_append
        k = ref_seen + np.cumsum(valid) - valid
        j = np.asarray(jax.random.randint(rng, k.shape, 0, k + 1, dtype=jnp.int32))
        ref_buf, ref_seen = _algorithm_r(ref_buf, ref_seen, ids, valid, j)
        kept = np.asarray(buffer.obs[:, 0]).astype(np.int64)
        filled = ref_buf >= 0
        np.testing.assert_array_equal(kept[filled], ref_buf[filled])
        np.testing.assert_array_equal(
            np.asarray(buffer.action)[filled], ref_buf[filled] % 2
        )
        assert int(buffer.seen) == ref_seen
        assert int(buffer.size) == min(capacity, ref_seen)
    assert ref_seen > 3 * capacity  # the buffer filled up and kept replacing


def test_reservoir_uniform_under_vmap() -> None:
    """Under vmap over seeds every valid item is kept with prob capacity / n."""
    capacity, batch, num_batches, trials = 8, 16, 4, 4000
    rs = np.random.RandomState(1)
    valid = rs.rand(num_batches, batch) < 0.6
    n = int(valid.sum())

    def run(rng):
        buffer = kuhn_pqn_nfsp.SLBufferState(
            obs=jnp.full((capacity, 1), -1.0, jnp.float32),
            action_mask=jnp.zeros((capacity, 2), jnp.bool_),
            action=jnp.zeros((capacity,), jnp.int32),
            seen=jnp.int32(0),
            size=jnp.int32(0),
        )
        for b in range(num_batches):
            ids = jnp.arange(b * batch, (b + 1) * batch, dtype=jnp.float32)[:, None]
            buffer = kuhn_pqn_nfsp.reservoir_append(
                buffer, ids, jnp.ones((batch, 2), jnp.bool_),
                jnp.zeros((batch,), jnp.int32), jnp.asarray(valid[b]),
                jax.random.fold_in(rng, b),
            )
        return buffer.obs[:, 0]

    kept = np.asarray(
        jax.jit(jax.vmap(run))(jax.random.split(jax.random.PRNGKey(0), trials))
    ).astype(np.int64)
    counts = np.bincount(kept.ravel(), minlength=num_batches * batch)
    assert (counts[~valid.ravel()] == 0).all()
    freq = counts[valid.ravel()] / trials
    expected = capacity / n
    # binomial standard error ~0.0055; 5 sigma
    assert np.abs(freq - expected).max() < 5 * np.sqrt(expected * (1 - expected) / trials)
    assert (np.sort(kept, axis=1)[:, 1:] != np.sort(kept, axis=1)[:, :-1]).all()


# ------------------------------------------------------------ NFSP mixing
class _Recorder:
    """Stands in for the wandb run of seed 0."""

    def __init__(self):
        self.rows = []

    def log(self, metrics):
        self.rows.append(metrics)


def _nfsp_config(**extra):
    config = dict(
        num_envs=64, num_steps_per_env_per_update=16, num_epochs=1,
        num_minibatches=2, lr=2.5e-4, anneal_lr=True, gamma=1.0,
        max_grad_norm=0.5, fc_dim_size=16, anticipatory_eta=0.3,
        sl_reservoir_capacity=500, sl_batch_size=32, sl_num_steps_per_update=2,
        sl_lr=3e-4, eval_interval=2, env_kwargs={}, num_update_steps=4,
    )
    config.update(extra)
    config["num_gradient_steps"] = 4 * config["num_epochs"] * config["num_minibatches"]
    return config


def _check_per_hand(is_br, dones, players):
    """is_br is constant over each player's decisions within a hand."""
    hand = np.concatenate(
        [np.zeros((1, dones.shape[1]), int), np.cumsum(dones, 0)[:-1]], 0
    )
    flags = {}
    for (t, n), f in np.ndenumerate(is_br):
        key = (n, hand[t, n], players[t, n])
        assert flags.setdefault(key, f) == f, (t, n)
    frac = np.mean(list(flags.values()))
    assert 0.2 < frac < 0.4, frac  # anticipatory_eta = 0.3 per player and hand


@pytest.mark.parametrize("name", ["kuhn_ppo_nfsp", "kuhn_pqn_nfsp"])
def test_nfsp_mixing_per_hand(monkeypatch, name: str) -> None:
    mod = {"kuhn_ppo_nfsp": kuhn_ppo_nfsp, "kuhn_pqn_nfsp": kuhn_pqn_nfsp}[name]
    seen = {"targets": [], "reservoir": []}

    def record(key, *arrays):
        jax.debug.callback(
            lambda *a: seen[key].append([np.asarray(x) for x in a]), *arrays
        )

    reservoir = mod.reservoir_append

    def reservoir_spy(buffer, obs, action_mask, action, valid, rng):
        record("reservoir", valid)
        return reservoir(buffer, obs, action_mask, action, valid, rng)

    monkeypatch.setattr(mod, "reservoir_append", reservoir_spy)
    if name == "kuhn_ppo_nfsp":
        gae = mod.per_player_gae

        def targets_spy(values, rewards, dones, players, last_value, last_player, g, lam):
            out = gae(values, rewards, dones, players, last_value, last_player, g, lam)
            record("targets", dones, players, out[2])
            return out

        monkeypatch.setattr(mod, "per_player_gae", targets_spy)
        config = _nfsp_config(
            gae_lambda=0.95, clip_eps=0.2, vf_clip=0.2, ent_coef=0.0, vf_coef=1.0
        )
    else:
        q_targets = mod.per_player_q_lambda_targets

        def targets_spy(q_max, rewards, dones, players, traces, last_q, last_player, g):
            out = q_targets(q_max, rewards, dones, players, traces, last_q, last_player, g)
            record("targets", dones, players, out[1], traces)
            return out

        monkeypatch.setattr(mod, "per_player_q_lambda_targets", targets_spy)
        config = _nfsp_config(
            start_e=1.0, end_e=0.05, exploration_fraction=0.5, q_lambda=0.9
        )
    logger = _Recorder()
    monkeypatch.setattr(mod, "WANDB_RUNS", [logger])
    jax.block_until_ready(
        jax.jit(mod.make_train(config))(jax.random.PRNGKey(0), jnp.int32(0))
    )
    jax.effects_barrier()

    T, N = config["num_steps_per_env_per_update"], config["num_envs"]
    assert len(seen["targets"]) == len(seen["reservoir"]) == len(logger.rows) == 4
    is_br = np.concatenate([r[0].reshape(T, N) for r in seen["reservoir"]])
    dones = np.concatenate([r[0] for r in seen["targets"]])
    players = np.concatenate([r[1] for r in seen["targets"]])
    _check_per_hand(is_br, dones, players)  # hands span updates too
    for u, row in enumerate(logger.rows):
        valid = seen["targets"][u][2]
        br = seen["reservoir"][u][0].reshape(T, N)
        if name == "kuhn_ppo_nfsp":
            assert row["ppo_sample_frac"] == np.float32((valid & br).mean())
        else:
            assert row["q_sample_frac"] == np.float32(valid.mean())
            np.testing.assert_array_equal(
                seen["targets"][u][3], np.float32(0.9) * br.astype(np.float32)
            )
        has_expl = "exploitability_avg_policy" in row
        assert has_expl == (u % 2 == 0 or u == 3)
        if has_expl:
            assert 0.0 <= row["exploitability_avg_policy"] < 1.0


# ------------------------------------------------------- policy helpers
def _env_infosets(env):
    """(obs, infoset key) for every decision of every deal and start player."""
    step = jax.jit(env.step_env)
    out = []
    for hands, start in itertools.product(itertools.permutations(range(3), 2), (0, 1)):
        for history in [(), (PASS,), (BET,), (PASS, BET)]:
            state, _ = env.reset(jax.random.PRNGKey(0))
            state = state.replace(
                agent_hands=jnp.array(hands, jnp.int32),
                start_player_idx=jnp.int32(start),
                current_player_idx=jnp.int32(start),
            )
            for a in history:
                state, *_ = step(jax.random.PRNGKey(0), state, jnp.int32(a))
            card = hands[int(state.current_player_idx)]
            out.append(
                (np.asarray(env.obs_from_state(state)), ke.infoset_key(card, history))
            )
    return out


def test_infoset_obs_match_env(env) -> None:
    """Row i of INFOSET_OBS is the env observation at INFOSETS[i] for either
    start player (the model's player 0 is the env player who acts first)."""
    seen = set()
    for obs, key in _env_infosets(env):
        np.testing.assert_array_equal(obs, ke.INFOSET_OBS[ke.INFOSETS.index(key)])
        seen.add(key)
    assert seen == set(ke.INFOSETS)


def test_policy_array_from_qnetwork_is_greedy() -> None:
    rs = np.random.RandomState(0)
    q_table = rs.randn(len(ke.INFOSETS), 2).astype(np.float32)
    rows = {tuple(o): q for o, q in zip(ke.INFOSET_OBS, q_table)}

    def apply(params, obs):
        return jnp.asarray(np.stack([rows[tuple(o)] for o in np.asarray(obs)]))

    policy = np.asarray(ke.policy_array_from_qnetwork(apply, None))
    np.testing.assert_array_equal(policy, np.eye(2)[q_table.argmax(-1)])

    # a real Q-network, inside jit
    net = kuhn_pqn.QNetworkDiscreteMLP(action_dim=2, hidden_dim=16)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 9)))
    policy = np.asarray(
        jax.jit(lambda p: ke.policy_array_from_qnetwork(net.apply, p))(params)
    )
    q = np.asarray(net.apply(params, jnp.asarray(ke.INFOSET_OBS)))
    np.testing.assert_array_equal(policy, np.eye(2)[q.argmax(-1)])
    assert set(np.unique(policy)) <= {0.0, 1.0}


def test_policy_array_from_network() -> None:
    """softmax(logits) at every infoset, for actor-critic and actor networks."""
    for net in (
        kuhn_ppo.ActorCriticDiscreteMLP(action_dim=2, hidden_dim=16),
        kuhn_pqn_nfsp.ActorDiscreteMLP(action_dim=2, hidden_dim=16),
    ):
        params = net.init(jax.random.PRNGKey(1), jnp.zeros((1, 9)))
        params = jax.tree_util.tree_map(lambda x: x * 3.0, params)  # not uniform
        policy = np.asarray(
            jax.jit(lambda p: ke.policy_array_from_network(net.apply, p))(params)
        )
        for i, key in enumerate(ke.INFOSETS):
            out = net.apply(params, jnp.asarray(ke._infoset_to_obs(key)))
            logits = np.asarray(out[0] if isinstance(out, tuple) else out, np.float64)
            probs = np.exp(logits - logits.max())
            np.testing.assert_allclose(policy[i], probs / probs.sum(), atol=1e-6)


def test_exploitability_from_policy_array() -> None:
    uniform = np.full((len(ke.INFOSETS), 2), 0.5)
    assert abs(ke.exploitability_from_policy_array(uniform) - 0.458333) < 1e-6
    for alpha in (0.0, 0.1, 1.0 / 3.0):
        nash = ke.nash_equilibrium(alpha)
        table = np.stack([nash(k) for k in ke.INFOSETS])
        assert abs(ke.exploitability_from_policy_array(table)) < 1e-12
    rs = np.random.RandomState(0)
    for _ in range(5):
        p_bet = rs.rand(len(ke.INFOSETS))
        table = np.stack([1 - p_bet, p_bet], axis=-1)
        direct = ke.exploitability(lambda k: table[ke.INFOSETS.index(k)])
        assert ke.exploitability_from_policy_array(table) == pytest.approx(direct)
