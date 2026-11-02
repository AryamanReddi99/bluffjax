"""The Leduc push options of leduc_ppo_nfsp: the reservoir target (sl_target),
the entropy schedule (ent_coef_final) and the logged best-response gaps
(log_br_gap), from tiny CPU runs of make_train and exact checks of the gaps."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from bluffjax.examples.leduc import leduc_ppo_nfsp
from bluffjax.examples.run_utils import ResultsLog, read_results
from bluffjax.networks.mlp import ActorCriticDiscreteMLP
from bluffjax.utils.game_utils import leduc_exploitability as L
from bluffjax.utils.paths import register_resolvers

HIDDEN = 16
LEGAL = L._infoset_inputs(L._INFOSET_LIST)[1]
UNIFORM = LEGAL / LEGAL.sum(-1, keepdims=True)


def _config(**overrides) -> dict:
    register_resolvers()
    path = leduc_ppo_nfsp.__file__.replace("leduc_ppo_nfsp.py", "config_ppo_nfsp.yaml")
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    config.pop("hydra")
    config.update(
        num_envs=8, num_steps_per_env_per_update=8, num_epochs=1, num_minibatches=2,
        fc_dim_size=HIDDEN, sl_reservoir_capacity=4096, sl_batch_size=8,
        sl_num_steps_per_update=1, compare=False, eval_interval=1000,
        anticipatory_eta=0.5, num_update_steps=3, half_update_steps=1,
        num_gradient_steps=6,
    )
    config.update(overrides)
    return config


def _train(monkeypatch, tmp_path, config, num_updates):
    monkeypatch.setattr(leduc_ppo_nfsp, "RESULTS_LOG", ResultsLog(str(tmp_path), []))
    train = leduc_ppo_nfsp.make_train(config)
    state = jax.jit(functools.partial(train, num_updates=num_updates))(
        jax.random.PRNGKey(0), jnp.int32(0)
    )
    jax.effects_barrier()
    return state


def _exact_br_array(opp_policy: np.ndarray) -> np.ndarray:
    """A deterministic exact best response to opp_policy at every infoset of
    both seats (uniform where unreachable)."""
    out = UNIFORM.copy()
    opp = L._policy_from_array_callable(opp_policy, L._KEY_TO_IDX)
    for seat in (0, 1):
        solver = L.BestResponseSolver(L.initial_state(), seat, opp)
        for key in solver.infosets:
            row = np.zeros(3)
            row[solver.best_response_action(key)] = 1.0
            out[L._KEY_TO_IDX[key]] = row
    return out


def test_br_gap_values_exact():
    rng = np.random.default_rng(0)
    avg = np.where(LEGAL, rng.random(LEGAL.shape), 0.0)
    avg /= avg.sum(-1, keepdims=True)
    br = _exact_br_array(avg)
    gaps = leduc_ppo_nfsp.br_gap_values(avg, br, eta=0.0)
    # an exact BR to the average has no gap, sampled or greedy
    assert gaps["br_gap_sampled_avg"] == pytest.approx(0.0, abs=1e-9)
    assert gaps["br_gap_greedy_avg"] == pytest.approx(0.0, abs=1e-9)
    # eta = 0: the mixture is the average policy
    assert gaps["br_gap_sampled_mix"] == pytest.approx(0.0, abs=1e-9)
    # a soft version of it loses value; its greedy version doesn't
    soft = 0.6 * br + 0.4 * UNIFORM
    gaps = leduc_ppo_nfsp.br_gap_values(avg, soft, eta=0.0)
    assert gaps["br_gap_sampled_avg"] > 1e-3
    assert gaps["br_gap_greedy_avg"] == pytest.approx(0.0, abs=1e-9)
    # against the average the exact BR value is its exploitability, so the
    # uniform policy's gap is its exploitability minus its value (0) there
    gaps = leduc_ppo_nfsp.br_gap_values(UNIFORM, UNIFORM, eta=0.1)
    expl = L.exploitability_from_policy_array(UNIFORM, L._KEY_TO_IDX)
    assert gaps["br_gap_sampled_avg"] == pytest.approx(expl, abs=1e-9)
    assert gaps["br_gap_sampled_mix"] == pytest.approx(expl, abs=1e-9)


def test_ent_coef_schedule():
    coef = leduc_ppo_nfsp.ent_coef_schedule(
        {"ent_coef": 0.1, "ent_coef_final": 0.01, "ent_decay_start": 0.5,
         "num_update_steps": 101}
    )
    values = [float(coef(jnp.int32(u))) for u in (0, 50, 75, 100, 200)]
    assert values[0] == pytest.approx(0.1) and values[1] == pytest.approx(0.1)
    assert values[2] == pytest.approx(0.1 + (75 - 50.5) / 49.5 * (0.01 - 0.1))
    assert values[3] == pytest.approx(0.01) and values[4] == pytest.approx(0.01)
    const = leduc_ppo_nfsp.ent_coef_schedule({"ent_coef": 0.1, "num_update_steps": 10})
    assert float(const(jnp.int32(7))) == pytest.approx(0.1)


@pytest.mark.parametrize(
    "sl_target, temperature", [("greedy", 0.1), ("sharpened", 1e-4), ("sampled", 0.1)]
)
def test_reservoir_target(monkeypatch, tmp_path, sl_target, temperature):
    """With lr 0 the BR keeps its initial (random) network, so every reservoir
    entry can be checked against it: greedy (and sharpened at a tiny
    temperature) store the argmax of the masked logits, sampled stores
    actions that are often not the argmax."""
    config = _config(lr=0.0, sl_target=sl_target, sl_target_temperature=temperature)
    state = _train(monkeypatch, tmp_path, config, num_updates=2)
    buf = state.sl_buffer
    n = int(buf.size)
    assert n > 20 and int(buf.seen) == n  # no overwrites with this capacity
    net = ActorCriticDiscreteMLP(3, HIDDEN)
    logits, _ = net.apply(state.br_train_state.params, buf.obs[:n])
    argmax = np.asarray(jnp.argmax(jnp.where(buf.action_mask[:n], logits, -jnp.inf), -1))
    stored = np.asarray(buf.action[:n])
    assert np.asarray(buf.action_mask[:n])[np.arange(n), stored].all()  # legal
    if sl_target == "sampled":
        # a near-uniform initial policy: many samples are not the argmax
        assert (stored != argmax).mean() > 0.2
    else:
        assert (stored == argmax).all()


def test_br_gap_logged_and_entropy_decay(monkeypatch, tmp_path):
    config = _config(
        sl_target="greedy", gamma=1.0, log_br_gap=True, eval_interval=2,
        ent_coef=0.1, ent_coef_final=0.01, ent_decay_start=0.0,
    )
    _train(monkeypatch, tmp_path, config, num_updates=3)
    rows = read_results(str(tmp_path))
    assert [r["update_step"] for r in rows] == [0, 1, 2]
    gap_keys = {f"br_gap_{b}_{o}" for b in ("sampled", "greedy") for o in ("avg", "mix")}
    for r in rows:
        if r["update_step"] in (0, 2):
            assert gap_keys <= set(r)
            assert all(0.0 - 1e-9 <= r[k] <= 26.0 for k in gap_keys)
        else:
            assert not gap_keys & set(r)
    assert [r["ent_coef"] for r in rows] == pytest.approx([0.1, 0.055, 0.01])
    assert rows[-1]["sl_target_differs_frac"] > 0.0


def test_learner_seat_scheme(monkeypatch, tmp_path):
    """learner_seat: exactly one BR seat in every hand, so about half of all
    decisions are BR decisions (all of them go to the reservoir)."""
    config = _config(br_mode_scheme="learner_seat", anticipatory_eta=0.1)
    state = _train(monkeypatch, tmp_path, config, num_updates=2)
    assert (np.asarray(state.br_mode).sum(-1) == 1).all()
    decisions = 2 * config["num_envs"] * config["num_steps_per_env_per_update"]
    frac = int(state.sl_buffer.seen) / decisions
    assert 0.3 < frac < 0.7, frac
    rows = read_results(str(tmp_path))
    assert all(0.3 < r["br_action_frac_rollout"] < 0.7 for r in rows)
    modes = leduc_ppo_nfsp.draw_br_modes(
        jax.random.PRNGKey(1), (4000, 2), "learner_seat", 0.1
    )
    assert (np.asarray(modes).sum(-1) == 1).all()
    assert np.asarray(modes)[:, 0].mean() == pytest.approx(0.5, abs=0.03)
    with pytest.raises(ValueError):
        leduc_ppo_nfsp.make_train(_config(br_mode_scheme="nope"))


@pytest.mark.parametrize("target_kl, epochs_run", [(None, 3), (1e-12, 1), (1e9, 3)])
def test_target_kl(monkeypatch, tmp_path, target_kl, epochs_run):
    """target_kl stops an update's epochs after the first whose mean KL is
    above it: a tiny target leaves one epoch per update (num_minibatches
    optimizer steps), a huge one all of them."""
    config = _config(num_epochs=3, target_kl=target_kl, lr=1e-2, anticipatory_eta=0.5)
    state = _train(monkeypatch, tmp_path, config, num_updates=2)
    steps = 2 * epochs_run * config["num_minibatches"]
    assert int(state.br_train_state.step) == steps
    rows = read_results(str(tmp_path))
    if target_kl is None:
        assert "ppo_epochs_run" not in rows[0]
    else:
        assert [r["ppo_epochs_run"] for r in rows] == [epochs_run] * 2
        assert all(np.isfinite(r["kl_backward"]) for r in rows)


@pytest.mark.parametrize("anneal", [False, True])
def test_sl_lr_anneal(monkeypatch, tmp_path, anneal):
    """sl_anneal_lr: the average policy's learning rate decays linearly from
    sl_lr to 0 over all num_update_steps x sl_num_steps_per_update SL steps
    (logged sl_lr = the next step's rate); off, it is the constant sl_lr."""
    config = _config(sl_anneal_lr=anneal, sl_lr=1e-3, sl_num_steps_per_update=4)
    state = _train(monkeypatch, tmp_path, config, num_updates=3)
    total = config["num_update_steps"] * config["sl_num_steps_per_update"]
    assert int(state.avg_train_state.step) == total
    rows = read_results(str(tmp_path))
    want = [1e-3 * (1 - 4 * (u + 1) / total) if anneal else 1e-3 for u in range(3)]
    assert [r["sl_lr"] for r in rows] == pytest.approx(want, abs=1e-9)
    if anneal:  # the last SL step used rate 1e-3 / total, not 1e-3
        (tmp_path / "c").mkdir()
        constant = _train(
            monkeypatch, tmp_path / "c",
            _config(sl_lr=1e-3, sl_num_steps_per_update=4), num_updates=3,
        )
        diff = jax.tree_util.tree_map(
            lambda a, b: float(jnp.abs(a - b).max()),
            state.avg_train_state.params, constant.avg_train_state.params,
        )
        assert max(jax.tree_util.tree_leaves(diff)) > 1e-6
