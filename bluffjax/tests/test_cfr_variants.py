"""CFR variants (CFR+, Linear CFR, Discounted CFR) of kuhn_cfr.py and
leduc_cfr.py, and Single Deep CFR (--sd_cfr) of kuhn_deep_cfr.py and
leduc_deep_cfr.py.

The reference exploitabilities are OpenSpiel's (open_spiel 2.0.2, python
CFRSolver, CFRPlusSolver, LCFRSolver and DCFRSolver, exploitability.
exploitability), computed once with the same number of iterations.
"""

import glob
import os

import numpy as np
import pytest

from bluffjax.examples.kuhn import kuhn_cfr, kuhn_deep_cfr
from bluffjax.examples.leduc import leduc_cfr, leduc_deep_cfr
from bluffjax.examples.sd_cfr import SDCFRAverage, regret_matching_table
from bluffjax.utils.game_utils import leduc_exploitability as leduc

OPENSPIEL_LEDUC_20 = {
    "cfr": 0.453145653416075,
    "cfr_plus": 0.17365537344197066,
    "linear": 0.25597780680552035,
    "dcfr": 0.16679681798943993,
}
OPENSPIEL_KUHN_100 = {
    "cfr": 0.008225977315915206,
    "cfr_plus": 0.0011944041011116846,
    "linear": 0.001089027365053341,
    "dcfr": 0.0016663419703252247,
}


@pytest.mark.parametrize("variant", sorted(OPENSPIEL_LEDUC_20))
def test_leduc_cfr_variants_match_openspiel(variant):
    solver = leduc_cfr.CFRSolver(variant)
    for _ in range(20):
        solver.run_iteration()
    expl = leduc.exploitability_from_policy_array(
        solver.average_policy_array(), leduc._KEY_TO_IDX
    )
    assert expl == pytest.approx(OPENSPIEL_LEDUC_20[variant], abs=1e-12)


@pytest.mark.parametrize("variant", sorted(OPENSPIEL_KUHN_100))
def test_kuhn_cfr_variants_match_openspiel(variant):
    result = kuhn_cfr.run_cfr(100, eval_interval=100, variant=variant)
    assert result.exploitability_log[-1][1] == pytest.approx(
        OPENSPIEL_KUHN_100[variant], abs=1e-12
    )


def test_cfr_variant_flag(tmp_path):
    for mod in (kuhn_cfr, leduc_cfr):
        assert mod.parse_args([]).variant == "cfr"
    run_dir = leduc_cfr.run(
        leduc_cfr.parse_args(
            ["--iterations", "3", "--variant", "cfr_plus", "--save_dir", str(tmp_path)]
        )
    )
    solver = leduc_cfr.CFRSolver("cfr_plus")
    for _ in range(3):
        solver.run_iteration()
    np.testing.assert_array_equal(
        leduc_cfr.load_average_policy(os.path.join(run_dir, "full/average_policy.msgpack")),
        solver.average_policy_array(),
    )


def test_regret_matching_table_matches_regret_matching():
    rng = np.random.default_rng(0)
    advantages = rng.normal(size=(200, 3))
    advantages[:50] = -np.abs(advantages[:50])  # no positive advantage
    advantages[50:60] = 0.0  # all-zero: uniform
    advantages[60:70, 1] = advantages[60:70, 2] = -0.1  # tied maxima
    advantages[60:70, 0] = -1.0
    masks = rng.random((200, 3)) < 0.7
    masks[np.arange(200), rng.integers(3, size=200)] = True
    table = regret_matching_table(advantages, masks)
    for i in range(200):
        np.testing.assert_allclose(
            table[i], leduc_deep_cfr.regret_matching(advantages[i], masks[i])
        )


@pytest.mark.parametrize("variant", ["cfr", "cfr_plus"])
def test_sd_cfr_average_is_cfr_average_strategy(variant):
    """Fed tabular CFR's current strategies with CFR's averaging weights,
    SDCFRAverage's own-reach-weighted average is CFR's average strategy."""
    sd = SDCFRAverage(
        leduc_deep_cfr.INFOSET_PLAYERS,
        *leduc_deep_cfr._own_parents(),
        leduc_deep_cfr.INFOSET_LEGAL_MASKS,
    )
    solver = leduc_cfr.CFRSolver(variant)
    gamma = leduc_cfr.VARIANTS[variant]["gamma"]
    root = leduc.initial_state()
    for t in range(1, 6):
        # leduc_cfr.CFRSolver.run_iteration, adding each player's strategy
        solver.iteration = t
        solver._avg_weight = float(t) ** gamma
        for p in (0, 1):
            sd.add(p, solver.current_policy_array(p), float(t) ** gamma)
            solver._strategy_cache = {}
            solver._cfr(root, p, (1.0, 1.0))
            solver._update_regrets(p, float(t))
        solver._strategy_cache = {}
    reference = leduc_cfr.CFRSolver(variant)
    for _ in range(5):
        reference.run_iteration()
    np.testing.assert_array_equal(solver.average_policy_array(), reference.average_policy_array())
    np.testing.assert_allclose(sd.policy(), reference.average_policy_array(), atol=1e-12)


def test_kuhn_sd_cfr_average_by_hand():
    parent, parent_action = kuhn_deep_cfr._own_parents()
    sd = SDCFRAverage(
        kuhn_deep_cfr.INFOSET_PLAYERS, parent, parent_action, kuhn_deep_cfr.INFOSET_LEGAL_MASKS
    )
    rng = np.random.default_rng(1)
    strategies = [rng.dirichlet(np.ones(2), size=12) for _ in range(3)]
    for t, s in enumerate(strategies, start=1):
        sd.add(0, s, t)
        sd.add(1, s, t)
    row = {k: i for i, k in enumerate(kuhn_cfr.INFOSETS)}
    policy = sd.policy()
    for k in kuhn_cfr.INFOSETS:
        i = row[k]
        if len(k) == 3:  # player 0 after pass, bet: reached by passing first
            w = [t * s[row[k[0]], kuhn_cfr.ACTION_PASS] for t, s in enumerate(strategies, 1)]
        else:
            w = [t for t in range(1, 4)]
        expected = sum(wi * s[i] for wi, s in zip(w, strategies)) / sum(w)
        np.testing.assert_allclose(policy[i], expected, rtol=1e-12)


SMALL_DEEP_CFR = [
    "--traversals", "5", "--policy_train_steps", "20", "--advantage_train_steps", "10",
    "--batch_size", "16", "--memory_capacity", "500", "--hidden_sizes", "8", "8",
    "--num_seeds", "1", "--iterations", "4", "--eval_interval", "2",
]


@pytest.mark.parametrize("game", ["kuhn", "leduc"])
def test_deep_cfr_sd_cfr_run(tmp_path, game):
    """--sd_cfr logs and saves the SD-CFR average and leaves Deep CFR's own
    results unchanged; --strategy_memory_capacity sizes the strategy memory."""
    mod, cfr_mod = {
        "kuhn": (kuhn_deep_cfr, kuhn_cfr),
        "leduc": (leduc_deep_cfr, leduc_cfr),
    }[game]
    defaults = mod.parse_args([])  # the Leduc push's best settings
    assert defaults.sd_cfr and defaults.strategy_memory_capacity == 10_000_000
    # A small run without SD-CFR and with the strategy memory as large as the
    # advantage memories (the behaviour of strategy_memory_capacity None).
    base = mod.parse_args(SMALL_DEEP_CFR + ["--no-sd_cfr", "--strategy_memory_capacity", "500"])
    assert not base.sd_cfr
    plain = mod.run_deep_cfr(base, 0)
    args = mod.parse_args(
        SMALL_DEEP_CFR
        + ["--sd_cfr", "--strategy_memory_capacity", "700", "--save_dir", str(tmp_path)]
    )
    solver = mod.DeepCFRSolver(args)
    assert solver._strategy_memory.capacity == 700
    assert solver._advantage_memories[0].capacity == 500
    run_dir = mod.run(args)
    sd_args = mod.parse_args(SMALL_DEEP_CFR + ["--sd_cfr", "--strategy_memory_capacity", "500"])
    with_sd = mod.run_deep_cfr(sd_args, 0)
    assert with_sd["exploitability"] == plain["exploitability"]
    assert len(with_sd["sd_cfr_exploitability"]) == len(with_sd["exploitability"])
    path = glob.glob(os.path.join(run_dir, "full", "sd_cfr_policy_0.msgpack"))[0]
    policy = cfr_mod.load_average_policy(path)
    np.testing.assert_allclose(policy.sum(axis=1), 1.0)
    if game == "leduc":
        expl = leduc_deep_cfr.exploitability(policy)
    else:
        expl = kuhn_deep_cfr.exploitability(kuhn_deep_cfr.policy_from_array(policy))
    assert expl == pytest.approx(with_sd["final_sd_cfr_exploitability"], rel=1e-9)
