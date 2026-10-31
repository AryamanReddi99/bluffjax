"""Exact best-response quality of Leduc NFSP (bluffjax/utils/game_utils/
leduc_br_gap.py) and the PQN-NFSP options that use it:

- hand_mixture is the behaviour strategy of the per-hand mixture: against any
  fixed policy its value equals (1 - eta) * value vs avg + eta * value vs BR
  (values are linear in realization plans), and it reduces to avg / br at
  eta = 0 / 1 on the infosets those policies reach;
- an exact best-response table has gap 0, a uniform table a positive gap, and
  the exact BR value against the average policy is its exploitability;
- a short PQN-NFSP run with log_br_gap and br_optimizer=radam logs the gap
  metrics at exactly the evaluation points, consistent with the logged
  exploitability, and radam changes the Q-network but not the default path.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from bluffjax.examples.leduc import leduc_pqn_nfsp
from bluffjax.examples.run_utils import ResultsLog, read_results
from bluffjax.utils.game_utils import leduc_exploitability as L
from bluffjax.utils.game_utils.leduc_br_gap import br_gap_metrics, hand_mixture

LEGAL = L._infoset_inputs(L._INFOSET_LIST)[1]


def _random_table(seed: int, sharp: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=LEGAL.shape) * sharp
    logits = np.where(LEGAL, logits, -np.inf)
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def _value(table_i: np.ndarray, table_opp: np.ndarray, i: int) -> float:
    f_i = L._policy_from_array_callable(table_i, L._KEY_TO_IDX)
    f_o = L._policy_from_array_callable(table_opp, L._KEY_TO_IDX)
    root = L.initial_state()
    return L._state_values(root, lambda s: f_i(s) if s.current_player == i else f_o(s))[i]


def _exact_br_table(avg: np.ndarray) -> np.ndarray:
    table = np.zeros_like(avg)
    f = L._policy_from_array_callable(avg, L._KEY_TO_IDX)
    for i in (0, 1):
        solver = L.BestResponseSolver(L.initial_state(), i, f)
        for key in solver.infosets:
            table[L._KEY_TO_IDX[key], solver.best_response_action(key)] = 1.0
    # infosets no opponent action reaches: any legal action
    empty = table.sum(1) == 0
    table[empty] = LEGAL[empty] / LEGAL[empty].sum(1, keepdims=True)
    return table


@pytest.mark.parametrize("eta", [0.1, 0.5])
def test_hand_mixture_is_per_hand_mixing(eta):
    avg, br, me = _random_table(0), _random_table(1, sharp=3.0), _random_table(2)
    mix = hand_mixture(avg, br, eta)
    assert np.allclose(mix.sum(1), 1.0) and np.all(mix[~LEGAL] == 0)
    for i in (0, 1):
        want = (1 - eta) * _value(me, avg, i) + eta * _value(me, br, i)
        assert _value(me, mix, i) == pytest.approx(want, abs=1e-10)
    # a per-infoset average is not the same strategy
    naive = (1 - eta) * avg + eta * br
    assert not np.allclose(naive, mix)


def test_hand_mixture_limits():
    avg, br = _random_table(3), _random_table(4)
    assert np.allclose(hand_mixture(avg, br, 0.0), avg)
    assert np.allclose(hand_mixture(avg, br, 1.0), br)  # br reaches every infoset
    assert np.allclose(hand_mixture(avg, avg, 0.3), avg)


def test_br_gap_metrics():
    avg = _random_table(5)
    exact = _exact_br_table(avg)
    m = br_gap_metrics(avg, exact, 0.1)
    assert m["br_value_vs_avg"] == pytest.approx(
        L.exploitability_from_policy_array(avg, L._KEY_TO_IDX), abs=1e-10
    )
    assert m["br_gap_vs_avg"] == pytest.approx(0.0, abs=1e-10)
    assert m["greedy_value_vs_avg"] == pytest.approx(m["br_value_vs_avg"], abs=1e-10)
    assert m["br_gap_vs_mix"] >= -1e-10
    uniform = LEGAL / LEGAL.sum(1, keepdims=True)
    m_u = br_gap_metrics(avg, uniform, 0.1)
    assert m_u["br_gap_vs_avg"] > 0.1 and m_u["br_gap_vs_mix"] > 0.1
    assert m_u["br_value_vs_avg"] == pytest.approx(m["br_value_vs_avg"], abs=1e-10)


def _config(**kw):
    config = dict(
        num_timesteps=4 * 64, num_envs=8, num_steps_per_env_per_update=8,
        num_epochs=1, num_minibatches=2, lr=1e-3, anneal_lr=True, start_e=0.06,
        end_e=0.0, exploration_fraction=1.0, q_lambda=0.65, max_grad_norm=10.0,
        gamma=1.0, fc_dim_size=16, anticipatory_eta=0.1, sl_reservoir_capacity=64,
        sl_batch_size=8, sl_num_steps_per_update=2, sl_lr=1e-3, eval_interval=2,
        compare=False, compare_episodes=2, compare_interval=1, env_kwargs={},
    )
    config.update(kw)
    config["num_update_steps"] = 4
    config["num_gradient_steps"] = 4 * config["num_epochs"] * config["num_minibatches"]
    return config


def _run(monkeypatch, tmp_path, **kw):
    log_dir = tmp_path / str(len(list(tmp_path.iterdir())))
    log_dir.mkdir()
    monkeypatch.setattr(leduc_pqn_nfsp, "RESULTS_LOG", ResultsLog(str(log_dir), []))
    for log in (leduc_pqn_nfsp.EXPLOITABILITY, leduc_pqn_nfsp.VALUE_VS_RANDOM):
        log.clear()
    train = leduc_pqn_nfsp.make_train(_config(**kw))
    state = jax.jit(train)(jax.random.PRNGKey(0), jnp.int32(0))
    jax.effects_barrier()
    return state, read_results(str(log_dir))


def test_pqn_nfsp_options(monkeypatch, tmp_path):
    base, rows = _run(monkeypatch, tmp_path)  # defaults: adam, no gap metrics
    assert not any(k.startswith("br_gap") for r in rows for k in r)
    radam, rows_r = _run(monkeypatch, tmp_path, br_optimizer="radam", log_br_gap=True)
    adam2, _ = _run(monkeypatch, tmp_path, br_optimizer="adam")
    leaves = lambda s: jax.tree_util.tree_leaves(s.br_train_state.params)
    # br_optimizer=adam is the default path; radam changes the Q-network only
    assert all(np.array_equal(a, b) for a, b in zip(leaves(base), leaves(adam2)))
    assert any(not np.allclose(a, b) for a, b in zip(leaves(base), leaves(radam)))
    gap_keys = {f"{p}_vs_{o}" for p in ("br_value", "greedy_value", "br_gap") for o in ("avg", "mix")}
    for r in rows_r:
        if r["update_step"] in (0, 2, 3):
            assert gap_keys <= set(r)
            assert r["br_value_vs_avg"] == pytest.approx(r["exploitability"], abs=1e-8)
            assert r["br_gap_vs_avg"] >= -1e-9 and r["br_gap_vs_mix"] >= -1e-9
        else:
            assert not gap_keys & set(r)
    with pytest.raises(ValueError):
        leduc_pqn_nfsp.make_train(_config(br_optimizer="sgd"))
