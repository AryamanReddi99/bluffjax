"""Tests for ReBeL on Leduc Hold'em (bluffjax/examples/leduc/leduc_rebel.py).

Run on CPU: JAX_PLATFORMS=cpu pytest bluffjax/tests/test_leduc_rebel.py
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax.examples.leduc import leduc_rebel as lr
from bluffjax.utils.game_utils import leduc_exploitability as leduc


def _walk(state, visit):
    """Calls visit(state) on every state of the game tree."""
    visit(state)
    if state.is_terminal():
        return
    for action in state.legal_actions():
        _walk(state.child(action), visit)


def test_round_trees_match_game() -> None:
    """Every betting state of leduc_exploitability's game (OpenSpiel's rules)
    is a node of the round trees with the same actor and legal actions, and
    every terminal pays what the trees say."""
    seen = {1: set(), 2: set()}

    def visit(s):
        if s.is_chance_node():
            return
        if not s.is_terminal():
            round_ = s.round_
            tree = lr.ROUND1 if round_ == 1 else lr.ROUND2
            seq = s.round1_sequence if round_ == 1 else s.round2_sequence
            n = tree.index[seq]
            seen[round_].add(n)
            assert tree.kind[n] == lr.DECISION
            assert tree.actor[n] == s.current_player
            assert tuple(np.nonzero(tree.legal[n])[0]) == s.legal_actions()
            if round_ == 2:
                n1 = lr.ROUND1.index[s.round1_sequence]
                assert lr.ROUND1.kind[n1] == lr.ROUND_END
            return
        c0, c1 = s.private_cards
        n1 = lr.ROUND1.index[s.round1_sequence]
        if s.public_card < 0:
            assert lr.ROUND1.kind[n1] == lr.FOLD
            contrib = lr.ANTE + lr.ROUND1.extra[n1]
            folder = lr.ROUND1.folder[n1]
        else:
            z = list(lr.ROUND_ENDS).index(n1)
            n2 = lr.ROUND2.index[s.round2_sequence]
            contrib = lr.ROUND2_START[z] + lr.ROUND2.extra[n2]
            folder = lr.ROUND2.folder[n2]
            if lr.ROUND2.kind[n2] == lr.ROUND_END:
                assert contrib[0] == contrib[1]
                pay = lr.SHOWDOWN[s.public_card, c0, c1] * contrib[0]
                assert s.returns() == pytest.approx((pay, -pay))
                return
            assert lr.ROUND2.kind[n2] == lr.FOLD
        pay = -contrib[0] if folder == 0 else contrib[1]
        assert s.returns() == pytest.approx((pay, -pay))

    _walk(leduc.initial_state(), visit)
    assert seen[1] == set(lr.ROUND1.decisions)
    assert seen[2] == set(lr.ROUND2.decisions)
    assert lr.ROUND1.num_nodes == lr.ROUND2.num_nodes == 15
    assert len(lr.ROUND_ENDS) == 5
    assert sorted(lr.ROUND2_START.tolist()) == [1.0, 3.0, 3.0, 5.0, 5.0]


def test_range_updates() -> None:
    """Ranges after a sampled leaf are Bayes' rule on the played profile, and
    the public card removes itself from both ranges."""
    rng = np.random.default_rng(0)
    raw = rng.random((lr.NUM_NODES, lr.NUM_CARDS, lr.NUM_ACTIONS)) + 0.05
    raw = np.where(lr.ROUND1.legal[:, None, :], raw, 0.0)
    policy = jnp.asarray(raw / np.maximum(raw.sum(-1, keepdims=True), 1e-30), jnp.float32)
    ranges = lr.initial_ranges()

    nodes, leaf_ranges = jax.vmap(
        lambda k: lr.sample_leaf(k, policy, ranges, 0.25)
    )(jax.random.split(jax.random.PRNGKey(0), 256))
    nodes, leaf_ranges = np.asarray(nodes), np.asarray(leaf_ranges)
    assert set(nodes.tolist()) <= set(lr.ROUND_ENDS.tolist()) | set(
        np.nonzero(lr.ROUND1.kind == lr.FOLD)[0].tolist()
    )
    for node, got in zip(nodes, leaf_ranges):
        expected = np.full((2, lr.NUM_CARDS), 1.0 / lr.NUM_CARDS)
        history = lr.ROUND1.histories[node]
        n = 0
        for a in history:
            expected[lr.ROUND1.actor[n]] *= np.asarray(policy)[n, :, a]
            n = lr.ROUND1.child[n, a]
        expected /= expected.sum(-1, keepdims=True)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-7)

    board = lr.board_ranges(jnp.asarray(leaf_ranges[0]), 3)
    assert np.all(np.asarray(board)[:, 3] == 0.0)
    np.testing.assert_allclose(np.asarray(board).sum(-1), 1.0, rtol=1e-6)
    # A player with no reach keeps the fallback range.
    zero = jnp.zeros((2, lr.NUM_CARDS)).at[1].set(1.0)
    fallback = lr.normalize_ranges(zero, lr.initial_ranges())
    np.testing.assert_allclose(np.asarray(fallback[0]), 1.0 / 6)
    np.testing.assert_allclose(np.asarray(fallback[1]), 1.0 / 6)


def test_chance_leaf_values_match_enumeration() -> None:
    """With a value function giving the check-down value of round 2, the
    counterfactual values of the round-1 end nodes match a direct sum over
    opponent cards and public cards."""
    rng = np.random.default_rng(1)
    reach = jnp.asarray(rng.random((lr.NUM_NODES, 2, lr.NUM_CARDS)), jnp.float32)

    def check_down(round_end, board, ranges):
        pot = jnp.asarray(lr.ROUND2_START)[round_end]
        compat = jnp.asarray(lr.COMPAT2)[board]
        sign = jnp.asarray(lr.SHOWDOWN)[board]
        opp = ranges[:, ::-1]  # (K, 2, 6) opponent's range per player
        num = jnp.einsum("kio,kco->kic", opp, compat * sign)
        den = jnp.einsum("kio,kco->kic", opp, compat)
        return pot[:, None, None] * jnp.where(den > 0, num / jnp.where(den > 0, den, 1), 0)

    got = np.asarray(lr.chance_leaf_values(check_down, reach))
    reach = np.asarray(reach)
    for zi, node in enumerate(lr.ROUND_ENDS):
        pot = lr.ROUND2_START[zi]
        for i, c in itertools.product(range(2), range(lr.NUM_CARDS)):
            want = 0.0
            for o in range(lr.NUM_CARDS):
                if o == c:
                    continue
                boards = [b for b in range(lr.NUM_CARDS) if b not in (c, o)]
                ev = np.mean([pot * lr.SHOWDOWN[b, c, o] for b in boards])
                want += reach[node, 1 - i, o] * ev
            assert got[node, i, c] == pytest.approx(want, abs=1e-5)
    others = [n for n in range(lr.NUM_NODES) if n not in lr.ROUND_ENDS]
    assert np.all(got[others] == 0.0)


def test_round2_values_zero_sum() -> None:
    """The root values of a solved round-2 subgame (the training target) are
    zero-sum under the PBS deal distribution, E_0 + E_1 = 0: exactly for the
    average-policy target, and up to the alternating-update bias (negative,
    shrinking with the iterations) for the iterate target. The public card
    has no value."""
    rng = np.random.default_rng(2)
    ranges = lr.board_ranges(jnp.asarray(rng.random((2, lr.NUM_CARDS)), jnp.float32), 2)
    r = np.asarray(ranges)
    joint = r[0][:, None] * r[1][None, :] * lr.COMPAT2[2]
    p = (joint.sum(1) / joint.sum(), joint.sum(0) / joint.sum())

    def total(sol):
        v, mask = np.asarray(sol.values), np.asarray(sol.value_mask)
        assert not mask[0, 2] and not mask[1, 2]
        return sum(np.sum(p[i] * np.where(mask[i], v[i], 0.0)) for i in range(2))

    avg = lr.solve_round2(jnp.int32(1), jnp.int32(2), ranges, 256, "average_policy")
    assert total(avg) == pytest.approx(0.0, abs=1e-5)
    gaps = [total(lr.solve_round2(jnp.int32(1), jnp.int32(2), ranges, t)) for t in (64, 1024)]
    assert gaps[0] < gaps[1] < 0.0


def test_evaluator_matches_openspiel() -> None:
    """The exact test-time policy maps onto OpenSpiel's infosets; OpenSpiel
    and leduc_exploitability agree on its exploitability; with the exact
    value function it matches the audit prototype's number."""
    pyspiel = pytest.importorskip("pyspiel")
    from open_spiel.python import policy as os_policy
    from open_spiel.python.algorithms import exploitability as os_expl

    policy = jax.device_get(
        jax.jit(lambda: lr.test_time_policy(lr.exact_value_fn(128), 16))()
    )
    table = lr.policy_array(policy)
    ours = leduc.exploitability_from_policy_array(table, leduc._KEY_TO_IDX)

    game = pyspiel.load_game("leduc_poker")
    tab = os_policy.TabularPolicy(game)
    row_of = {key[:5]: i for i, (key, _) in enumerate(leduc._INFOSET_LIST)}

    def walk(state, cards, board, seq1, seq2):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                if len(cards) < 2:
                    walk(state.child(a), cards + (a,), board, seq1, seq2)
                else:
                    walk(state.child(a), cards, a, seq1, seq2)
            return
        p = state.current_player()
        legal = state.legal_actions()
        row = table[row_of[(p, cards[p], board, seq1, seq2)]]
        tab.policy_for_key(state.information_state_string(p))[:] = row
        for a in legal:
            if board < 0:
                walk(state.child(a), cards, board, seq1 + (a,), seq2)
            else:
                walk(state.child(a), cards, board, seq1, seq2 + (a,))

    walk(game.new_initial_state(), (), -1, (), ())
    theirs = os_expl.exploitability(game, tab)
    assert ours == pytest.approx(theirs, abs=1e-9)
    # Independent audit prototype: 0.27991 at T = 16 with 128 oracle
    # iterations (CPU, float32). CFR-D with iterate-dependent leaf values is
    # numerically chaotic at small T (float32 vs float64 moves either code by
    # up to 20%), so only a loose band is checked.
    assert 0.2 < ours < 0.35
    assert table.shape == (len(leduc._INFOSET_LIST), 3)
    np.testing.assert_allclose(table.sum(-1), 1.0, rtol=1e-9)


def test_short_training_run(tmp_path) -> None:
    config = {
        "num_update_steps": 2,
        "num_games": 8,
        "cfr_iters": 4,
        "eval_cfr_iters": 4,
        "eval_matmul_precision": "highest",
        "explore_prob": 0.25,
        "value_target": "iterates",
        "value_lr": 3e-4,
        "lr_halve_at": [0.5],
        "value_hidden_dim": 16,
        "value_num_layers": 2,
        "max_grad_norm": 5.0,
        "replay_capacity": 64,
        "batch_size": 8,
        "num_value_train_steps": 2,
        "eval_interval": 1,
        "save_checkpoints": True,
        "save_dir": str(tmp_path),
        "env_name": "leduc_poker",
        "seed": 0,
        "num_seeds": 2,
        "env_kwargs": {},
        "project": "test",
        "job_type": "leduc_rebel",
        "wandb": False,
    }
    run_dir = lr.run(config)
    from bluffjax.examples.run_utils import read_results

    records = read_results(run_dir)
    assert len(records) == 2 * 3  # 2 seeds x (before training + 2 blocks)
    assert all(np.isfinite(r["exploitability"]) and r["exploitability"] > 0 for r in records)
    assert records[-1]["updates_completed"] == 2
    for stage in ("half", "full"):
        for i in range(2):
            path = f"{run_dir}/{stage}/value_net_{i}.msgpack"
            from bluffjax.examples.run_utils import load_params

            network = lr.ValueNetwork(hidden_dim=16, num_layers=2)
            load_params(path, network.init(jax.random.PRNGKey(0), jnp.zeros(lr.PBS_DIM)))
