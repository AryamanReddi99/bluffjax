"""Checks for the Hold'em ReBeL baseline (bluffjax/examples/holdem_rebel)."""

import itertools
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.examples.holdem_rebel import cards as C
from bluffjax.examples.holdem_rebel import game as G
from bluffjax.examples.holdem_rebel import solver as SV
from bluffjax.examples.holdem_rebel.agent import (
    RebelPlayer,
    heuristic_players,
    load_rebel,
    public_view,
    save_checkpoint,
)
from bluffjax.utils.game_utils.poker_utils import _score_seven_card_hand

GAMES = ["hul", "hunl"]


def _zero_net(pub, beliefs):
    return jnp.zeros(beliefs.shape)


@pytest.mark.parametrize("name", GAMES)
def test_rules_match_env(name: str) -> None:
    """Legal actions, transitions, round ends and payoffs equal env.step_env."""
    game = G.make_game(name)
    env = make(game.env_id, num_agents=2)
    table = C.rank5_table()
    legal_f = jax.jit(lambda s: G.legal_actions(game, s))
    step_f = jax.jit(lambda s, a: G.step(game, s, a))
    view_f = jax.jit(lambda s: G.bet_state_from_env(game, s))
    env_step = jax.jit(env.step_env)
    rng = np.random.default_rng(0)
    for h in range(60):
        st, _ = env.reset(jax.random.PRNGKey(h))
        seats = [int(st.small_blind_idx), 1 - int(st.small_blind_idx)]
        while True:
            bs = view_f(st)
            avail = np.asarray(env.get_avail_actions(st))
            mine = np.asarray(legal_f(bs))
            dominated = np.zeros_like(avail)
            dominated[game.fold_action] = avail[game.fold_action] and not mine[game.fold_action]
            assert np.array_equal(mine, avail & ~dominated)
            choices = np.nonzero(avail)[0]
            w = np.where(choices == game.fold_action, 0.15, 1.0)
            a = int(rng.choice(choices, p=w / w.sum()))
            new, outcome = step_f(bs, jnp.int32(a))
            st2, _, rew, _, done, _ = env_step(jax.random.PRNGKey(0), st, jnp.int32(a))
            env_pay = np.asarray(rew)[seats]
            chips = np.asarray(new.chips)
            if int(outcome) == G.CONTINUE:
                assert not bool(done) and int(st2.stage) == int(st.stage)
                for f, x in zip(G.BetState._fields, view_f(st2)):
                    assert np.allclose(np.asarray(x), np.asarray(getattr(new, f))), f
            elif int(outcome) == G.FOLDED:
                folder = int(bs.actor)
                pay = np.zeros(2)
                pay[folder], pay[1 - folder] = -chips[folder], chips[folder]
                assert bool(done) and np.allclose(env_pay, pay * game.reward_per_chip)
                break
            elif int(st.stage) == 3 or bool(new.allin.all()):
                assert bool(done) and chips[0] == chips[1]
                board = jnp.concatenate(
                    [st.flop_cards, jnp.stack([st.turn_card, st.river_card])]
                ).astype(jnp.int32)
                s = np.asarray(C.hand_strengths(table, board))
                hands = [int(C.hand_index(st.agent_cards[k].astype(jnp.int32))) for k in seats]
                sign = np.sign(s[hands[0]] - s[hands[1]])
                pay = np.array([sign, -sign]) * chips[0] * game.reward_per_chip
                assert np.allclose(env_pay, pay)
                break
            else:
                assert not bool(done) and int(st2.stage) == int(st.stage) + 1
                root = G.round_root(game, st2.stage, chips[0])
                for f, x in zip(G.BetState._fields, view_f(st2)):
                    assert np.allclose(np.asarray(x), np.asarray(getattr(root, f))), f
            st = st2


@pytest.mark.parametrize("name", GAMES)
def test_subgame_trees_match_env(name: str) -> None:
    """Every node of the search tree (from many roots) matches an env-driven DFS."""
    game = G.make_game(name)
    env = make(game.env_id, num_agents=2)
    tpl = G.build_template(game)
    strings = {0: ()}
    for k in range(1, len(tpl.parent)):
        strings[k] = strings[int(tpl.parent[k])] + (int(tpl.action[k]),)
    index = {v: k for k, v in strings.items()}
    tree_f = jax.jit(lambda r: G.build_tree(game, tpl, r))
    env_step = jax.jit(env.step_env)
    rng = np.random.default_rng(1)
    for r in range(12):
        st, _ = env.reset(jax.random.PRNGKey(100 + r))
        for _ in range(int(rng.integers(0, 6))):
            avail = np.asarray(env.get_avail_actions(st))
            a = int(rng.choice([a for a in np.nonzero(avail)[0] if a != game.fold_action]))
            st2, _, _, _, done, _ = env_step(jax.random.PRNGKey(0), st, jnp.int32(a))
            if bool(done):
                break
            st = st2
        tree = jax.device_get(tree_f(G.bet_state_from_env(game, st)))
        assert not bool(tree.template_gap)
        seen, stack = set(), [((), st)]
        sb = int(st.small_blind_idx)
        while stack:
            s, est = stack.pop()
            k = index[s]
            seen.add(k)
            assert int(tree.kind[k]) == G.DECISION
            ki = int(tpl.internal_id[k])
            avail = np.asarray(env.get_avail_actions(est))
            behind = bool(G.legal_actions(game, G.bet_state_from_env(game, est))[game.fold_action])
            for a in np.nonzero(avail)[0]:
                a, slot = int(a), game.env_slot[int(a)]
                if a == game.fold_action and not behind:
                    continue
                capped = slot in game.raise_slots and (
                    sum(x in game.raise_slots for x in s) >= game.max_raises
                )
                if capped or (slot == game.allin_slot and game.allin_slot in s):
                    assert not bool(tree.legal[ki, slot])
                    continue
                assert bool(tree.legal[ki, slot])
                k2 = index[s + (slot,)]
                e2, _, _, _, done, _ = env_step(jax.random.PRNGKey(0), est, jnp.int32(a))
                assert np.allclose(tree.chips[k2], np.asarray(e2.chips_in)[[sb, 1 - sb]])
                if a == game.fold_action:
                    assert int(tree.kind[k2]) == G.FOLD and bool(done)
                    seen.add(k2)
                elif bool(done) or int(e2.stage) != int(est.stage):
                    assert int(tree.kind[k2]) == G.ROUND_END
                    seen.add(k2)
                else:
                    stack.append((s + (slot,), e2))
        assert seen == set(np.nonzero(np.asarray(tree.kind) != G.INVALID)[0].tolist())


def test_card_removal_and_showdown_values() -> None:
    rng = np.random.default_rng(2)
    hc = np.asarray(C.HAND_CARDS)
    share = (hc[:, None, :, None] == hc[None, :, None, :]).any(-1).any(-1)
    board = rng.choice(52, 5, replace=False)
    s = np.asarray(C.hand_strengths(C.rank5_table(), jnp.asarray(board, jnp.int32)))
    for h in rng.choice(1326, 30, replace=False):
        if not np.isin(hc[h], board).any():
            ref = _score_seven_card_hand(jnp.asarray(np.concatenate([hc[h], board]), jnp.int32))
            assert s[h] == int(ref)
    x = rng.random((2, 1326)).astype(np.float32) * (~np.isin(hc, board).any(1))
    sign = np.sign(s[:, None] - s[None, :]) * ~share
    v = np.asarray(C.showdown_value(jnp.asarray(x), C.showdown_tables(jnp.asarray(s))))
    ok = s >= 0
    assert np.allclose(v[:, ok], (x @ sign.T)[:, ok], atol=1e-3)
    m = np.asarray(C.compatible_mass(jnp.asarray(x)))
    assert np.allclose(m, x @ (~share).T, atol=1e-3)


@pytest.mark.parametrize("name", GAMES)
def test_polarised_river_equilibrium(name: str) -> None:
    """Value hand vs bluff-catcher: closed-form bluff/call frequencies and values."""
    game = G.make_game(name)
    tpl = G.build_template(game)
    a, b, c = (int(C.hand_index(jnp.array(p))) for p in ([12, 38], [2, 29], [10, 35]))
    board = jnp.array([1, 19, 34, 49, 25], jnp.int32)  # 2c 7d 9h Js Kd
    x = jnp.zeros((2, 1326)).at[0, a].set(0.5).at[0, b].set(0.5).at[1, c].set(1.0)
    sg = SV.make_subgame(game, tpl, G.round_root(game, 3, 10.0), board, x)
    sol = jax.jit(lambda s: SV.solve(game, tpl, s, _zero_net, jax.random.PRNGKey(0), 1500))(sg)
    pol = np.asarray(sol.avg_policy)
    bet = 1 if game.is_limit else game.allin_slot  # 4 chips in Limit, all-in in No-Limit
    size = float(np.asarray(sol.tree.chips)[tpl.children[0, bet], 0]) - 10.0
    pot = 20.0
    node = tpl.internal_id[tpl.children[0, bet]]
    assert pol[0, bet, a] > 0.99
    assert abs(pol[0, bet, b] - size / (pot + size)) < 0.01
    assert abs(pol[node, G.PASSIVE_SLOT, c] - pot / (pot + size)) < 0.01
    posterior = np.asarray(C.normalize(x[0] * sol.avg_policy[0, bet]))[b]
    assert abs(posterior - size / (pot + 2 * size)) < 0.01
    call = pot / (pot + size)
    v = np.asarray(sol.values) * game.value_scale / game.reward_per_chip  # chips
    assert abs(v[0, a] - (10 + call * size)) < 0.05 * (10 + call * size)
    assert abs(v[0, b] + 10) < 0.2


@pytest.mark.parametrize("name", GAMES)
def test_river_cfr_converges(name: str) -> None:
    game = G.make_game(name)
    tpl = G.build_template(game)
    rng = np.random.default_rng(3)
    board = rng.choice(52, 5, replace=False)
    hc = np.asarray(C.HAND_CARDS)
    x = rng.random((2, 1326)) ** 3 * (~np.isin(hc, board).any(1))
    sg = SV.make_subgame(
        game, tpl, G.round_root(game, 3, 12.0), jnp.asarray(board, jnp.int32),
        jnp.asarray(x, jnp.float32),
    )
    expl = []
    for iters in (16, 256):
        sol = SV.solve(game, tpl, sg, _zero_net, jax.random.PRNGKey(0), iters)
        _, br = SV.evaluate_profile(game, tpl, sg, sol.avg_policy, _zero_net)
        expl.append(float(br.sum()) / 2)
    pot = 24.0 * game.reward_per_chip
    assert expl[1] < expl[0] / 5 and expl[1] < 0.005 * pot


def test_chance_values_match_enumeration() -> None:
    """All-in turn: the chance-node value is the exact equity-weighted payoff."""
    game = G.make_game("hunl")
    rng = np.random.default_rng(4)
    hc = np.asarray(C.HAND_CARDS)
    board4 = rng.permutation(52)[:4]
    ok = ~np.isin(hc, board4).any(1)
    sel = [rng.choice(np.nonzero(ok)[0], 4, replace=False) for _ in range(2)]
    x = np.zeros((2, 1326))
    for p in range(2):
        x[p, sel[p]] = rng.random(4) + 0.1
    x /= x.sum(1, keepdims=True)
    board = jnp.asarray(np.concatenate([board4, [-1]]), jnp.int32)
    ch = SV.chance_children(jnp.int32(2), board, jax.random.PRNGKey(0), 1)
    res = SV.chance_values(
        game, None, jnp.int32(2), board, jnp.float32(100.0), jnp.bool_(True),
        jnp.asarray(x, jnp.float32), ch, exact_showdown=True,
    )
    for p in range(2):
        for h in sel[p]:
            num = den = 0.0
            for h2, r in itertools.product(sel[1 - p], range(52)):
                cards = set(hc[h]) | set(hc[h2])
                if len(cards) < 4 or r in cards or r in board4:
                    continue
                b5 = np.concatenate([board4, [r]]).astype(np.int32)
                s1 = _score_seven_card_hand(jnp.asarray(np.concatenate([hc[h], b5])))
                s2 = _score_seven_card_hand(jnp.asarray(np.concatenate([hc[h2], b5])))
                num += x[1 - p, h2] * np.sign(int(s1) - int(s2))
                den += x[1 - p, h2]
            assert abs(float(res.values[p, h]) - num / den) < 1e-5


@pytest.mark.parametrize("name", GAMES)
def test_agent_ignores_hidden_information(name: str, tmp_path) -> None:
    """Action probabilities and beliefs don't change with the opponent's cards
    or undealt board cards; checkpoints round-trip."""
    game = G.make_game(name)
    env = make(game.env_id, num_agents=2)
    player = RebelPlayer(game, hidden_dim=32, num_layers=1, cfr_iters=8, solve_chunk=1)
    params = player.net.init(
        jax.random.PRNGKey(0), jnp.zeros((G.PUBLIC_DIM,)), jnp.zeros((2, C.NUM_HANDS))
    )
    path = save_checkpoint(
        os.path.join(tmp_path, f"{name}.msgpack"), params,
        {"game": name, "max_raises": game.max_raises, "value_hidden_dim": 32,
         "value_num_layers": 1, "cfr_iters": 8, "samples": 0, "seed": 0},
    )
    player, loaded, _ = load_rebel(path, solve_chunk=1)
    assert all(jax.tree.leaves(jax.tree.map(np.array_equal, params, loaded)))
    with pytest.raises(FileNotFoundError):
        load_rebel(os.path.join(tmp_path, "missing.msgpack"))
    opp = heuristic_players(game)["always_call"]
    solve, probs, observe = jax.jit(player.solve), jax.jit(player.action_probs), jax.jit(player.observe)

    def decisions(st, seat):
        out, state, need = [], player.init(1), True
        view = lambda s: jax.tree.map(lambda v: v[None], public_view(game, s, jnp.int32(seat)))  # noqa: E731
        for t in range(30):
            if need:
                state = solve(loaded, state, view(st), jnp.array([True]), jax.random.PRNGKey(t))
                need = False
            if int(st.current_player_idx) == seat:
                out.append((int(st.stage), np.asarray(probs(state, view(st))), np.asarray(state.beliefs)))
                a = int(np.argmax(out[-1][1][0]))
                a = int(G.slot_action(game, G.bet_state_from_env(game, st), a))
            else:
                one = jax.tree.map(lambda v: v[None], st)
                a = int(opp.act(None, None, None, jax.random.PRNGKey(0), one,
                                env.get_avail_actions(st)[None])[0])
            state = observe(state, view(st), jnp.array([a]), jnp.array([True]))
            st2, _, _, _, done, _ = env.step_env(jax.random.PRNGKey(0), st, jnp.int32(a))
            if bool(done):
                return out
            need = int(st2.stage) != int(st.stage)
            st = st2
        return out

    rng = np.random.default_rng(5)
    for h in range(4):
        st, _ = env.reset(jax.random.PRNGKey(h))
        seat = h % 2
        base = decisions(st, seat)
        cards = np.asarray(st.agent_cards).astype(int)
        board = [int(c) for c in np.asarray(st.flop_cards)] + [int(st.turn_card), int(st.river_card)]
        used = set(cards[seat]) | set(board)
        new = cards.copy()
        new[1 - seat] = rng.choice([c for c in range(52) if c not in used], 2, replace=False)
        other = decisions(st.replace(agent_cards=jnp.asarray(new, st.agent_cards.dtype)), seat)
        assert len(other) == len(base)
        for (_, p1, b1), (_, p2, b2) in zip(base, other):
            assert np.array_equal(p1, p2) and np.array_equal(b1, b2)
        used = set(cards.ravel()) | set(board[:3])
        t, r = rng.choice([c for c in range(52) if c not in used], 2, replace=False)
        other = decisions(
            st.replace(turn_card=jnp.asarray(t, st.turn_card.dtype),
                       river_card=jnp.asarray(r, st.river_card.dtype)), seat
        )
        for (s1, p1, b1), (_, p2, b2) in zip(base, other):
            if s1 >= 2:
                break
            assert np.array_equal(p1, p2) and np.array_equal(b1, b2)
