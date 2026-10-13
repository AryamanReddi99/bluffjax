"""Kemps metrics (bluffjax/metrics/kemps.py):

- scripted hands with hand-computed values: calls judged on the hands at the
  start of the step and credited to the caller, every call of a step counted
  for the precisions but only the one that counts (the first declarer in the
  step's player order) ending the hand; interception and own-call credit by
  team, with two and three teams; the latency from the first step the
  partner held four of a kind; hands cut off at the horizon or ended by a
  real deal end without a call; four of a kind completed in a hand's last
  step doesn't count; the carry reset at the end of a hand;
- step_order is the order env.step uses (the counted declaration and the
  env's rewards agree for every key);
- the signal table and its mutual information;
- random play through env.step (auto-reset, horizon cut-offs, real deals):
  every count against a brute-force recomputation from the recorded hands,
  actions and orders, the env's rewards against the declaration that counts,
  and the mutual information of signals independent of the hand (G-test) or
  equal to it (the entropy of holding four of a kind).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.examples.parallel_nfsp_common import kemps_horizon_cutoff
from bluffjax.environments.kemps.kemps import Kemps
from bluffjax.metrics import make_metrics, masked_totals
from bluffjax.metrics.kemps import Metrics, mutual_information_bits, step_order

KEY = jax.random.PRNGKey(0)
S = 4  # suits


def noop(env):
    return env.action_noop


def kemps(env):
    return env.action_kemps


def stop(env, d):
    """STOP KEMPS against the team of the player d seats after the caller."""
    return env.action_kemps + d


def act(env, game_actions, signals=0):
    return jnp.array(game_actions, jnp.int32) * env.comm_dim + jnp.asarray(signals, jnp.int32)


def deal(env, fours=None, hands=None):
    """Hands without two cards of a rank, except that each seat in `fours`
    ({seat: rank}) holds four of a kind and each seat in `hands` the given
    cards. Returns hands, centre, stock (cards rank * 4 + suit)."""
    fours, hands = fours or {}, dict(hands or {})
    used = {c for h in hands.values() for c in h}
    taken = set(fours.values()) | {c // S for c in used}
    pool = [
        r * S + s for s in range(S) for r in range(env.num_ranks)
        if r not in taken and r * S + s not in used
    ]
    out = []
    for seat in range(env.num_agents):
        if seat in fours:
            out.append([fours[seat] * S + s for s in range(S)])
        elif seat in hands:
            out.append(list(hands[seat]))
        else:
            out.append(pool[:4])
            pool = pool[4:]
    left = [c for c in range(env.deck_size) if c not in {x for h in out for x in h}]
    return out, left[:4], left[4:]


def make_state(env, hands, center, stock, deck_idx=0):
    state, _ = env.reset(KEY)
    hands = jnp.array(hands, jnp.int32)
    center = jnp.array(center, jnp.int32)
    count = lambda cards: jnp.bincount(cards // S, length=env.num_ranks)
    return state.replace(
        agent_hands=hands,
        agent_hand_counts=jax.vmap(count)(hands),
        center_cards=center,
        center_counts=count(center),
        deck=jnp.array(stock, jnp.int32),
        deck_idx=jnp.int32(deck_idx),
    )


def play(env, gm, state, steps, carry=None):
    """Plays the scripted steps [(game_actions, order[, signals])] with
    env._step_in_order (no auto-reset). Returns per-step counts (numpy),
    rewards, dones and the carry after each step."""
    carry = gm.init_carry() if carry is None else carry
    out = []
    for step in steps:
        game_actions, order = step[0], jnp.array(step[1], jnp.int32)
        action = act(env, game_actions, step[2] if len(step) > 2 else 0)
        nxt, _, reward, _, done, info = env._step_in_order(state, action, order)
        carry, counts = gm.step(carry, state, action, nxt, reward, done, {**info, "order": order})
        out.append(dict(
            counts={k: np.asarray(v) for k, v in counts.items()},
            reward=np.asarray(reward), done=bool(done), carry=carry,
        ))
        state = nxt
    return out


HAND_LEVEL = ("opponent_quad_hands", "intercepted", "own_quad_hands", "quad_called", "latency_sum", "quads_called")


def assert_counts(counts, **want):
    for name, value in want.items():
        np.testing.assert_array_equal(counts[name], value, err_msg=name)


def assert_init_carry(gm, carry):
    for got, want in zip(jax.tree_util.tree_leaves(carry), jax.tree_util.tree_leaves(gm.init_carry())):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_make_metrics_and_summary_names():
    env = make("kemps", num_agents=4)
    gm = make_metrics("kemps", env)
    assert isinstance(gm, Metrics) and not gm.needs_probs
    state = make_state(env, *deal(env))
    (step,) = play(env, gm, state, [([noop(env), 0, 0, 0], [0, 1, 2, 3])])
    totals = {k: jnp.asarray(v).sum(0) for k, v in step["counts"].items()}
    out = gm.summarize(totals)
    assert set(gm.rate_names) <= set(out)
    assert {"n_kemps_calls", "n_stop_calls", "n_opponent_quad_hands", "n_own_quad_hands",
            "n_decisions", "n_quads_called"} <= set(out)
    assert {f"n_signal_{s}_quad_{q}" for s in range(2) for q in range(2)} <= set(out)
    with pytest.raises(KeyError):
        gm.step(gm.init_carry(), state, act(env, [noop(env)] * 4), state, jnp.zeros(4), False, {})


def simultaneous_case(env):
    """Seat 2 holds four of a kind: seat 0's KEMPS (on partner 2) and seat 1's
    STOP KEMPS (against team {0, 2}) are right, seat 3's KEMPS (on partner 1)
    is wrong."""
    state = make_state(env, *deal(env, {2: 0}))
    game = [kemps(env), stop(env, 1), noop(env), kemps(env)]
    return state, game


@pytest.mark.parametrize(
    "order, ends, reward",
    [
        ([1, 0, 3, 2], "intercepted", [-1, 1, -1, 1]),  # seat 1's STOP counts
        ([0, 1, 3, 2], "called", [1, -1, 1, -1]),  # seat 0's KEMPS counts
        ([3, 2, 1, 0], "wrong", [1, -1, 1, -1]),  # seat 3's wrong KEMPS counts
    ],
)
def test_simultaneous_declarations(order, ends, reward):
    env = Kemps(num_agents=4)
    gm = Metrics(env)
    state, game = simultaneous_case(env)
    (step,) = play(env, gm, state, [(game, order)])
    assert step["done"]
    np.testing.assert_array_equal(step["reward"], reward)  # the env agrees
    c = step["counts"]
    # every call counts for the precisions, whatever the order
    assert_counts(
        c, kemps_calls=[1, 0, 0, 1], kemps_right=[1, 0, 0, 0],
        stop_calls=[0, 1, 0, 0], stop_right=[0, 1, 0, 0],
        own_quad_hands=[1, 0, 1, 0], opponent_quad_hands=[0, 1, 0, 1],
    )
    assert_counts(
        c,
        intercepted=[0, 1, 0, 1] if ends == "intercepted" else [0, 0, 0, 0],
        quad_called=[1, 0, 1, 0] if ends == "called" else [0, 0, 0, 0],
        quads_called=[1, 0, 0, 0] if ends == "called" else [0, 0, 0, 0],
        latency_sum=[0, 0, 0, 0],  # held from the first step: latency 0
    )
    assert_init_carry(gm, step["carry"])


def completing_swap_case(env):
    """Seat 2 holds three 0s and a 12; the centre holds the fourth 0. Returns
    the state and seat 2's swap of its 12 for the 0."""
    hands = {2: [0, 1, 2, 12 * S]}
    hands_, center, stock = deal(env, hands=hands)
    rest = [c for c in center + stock if c != 3]
    center = [3] + rest[:3]
    state = make_state(env, hands_, center, rest[3:])
    return state, 12 * env.num_ranks + 0


def test_latency_counts_from_the_first_step_the_partner_holds_four():
    """Seat 2 completes four of a kind in step 0 (so holds it from step 1),
    two all-NOOP sweeps follow, seat 0 calls KEMPS in step 3: latency 2."""
    env = Kemps(num_agents=4)
    gm = Metrics(env)
    state, swap = completing_swap_case(env)
    n = noop(env)
    order = [0, 1, 2, 3]
    steps = play(env, gm, state, [
        ([n, n, swap, n], order),
        ([n, n, n, n], order),
        ([n, n, n, n], order),
        ([kemps(env), n, n, n], order),
    ])
    assert [s["done"] for s in steps] == [False, False, False, True]
    np.testing.assert_array_equal(steps[-1]["reward"], [1, -1, 1, -1])
    np.testing.assert_array_equal(np.asarray(steps[1]["carry"].first_four), [-1, -1, 1, -1])
    for s in steps[:3]:  # nothing credited before the hand ends
        for name in HAND_LEVEL:
            np.testing.assert_array_equal(s["counts"][name], 0, err_msg=name)
    assert_counts(
        steps[-1]["counts"],
        kemps_calls=[1, 0, 0, 0], kemps_right=[1, 0, 0, 0],
        quad_called=[1, 0, 1, 0], own_quad_hands=[1, 0, 1, 0],
        opponent_quad_hands=[0, 1, 0, 1], intercepted=[0, 0, 0, 0],
        latency_sum=[2, 0, 0, 0], quads_called=[1, 0, 0, 0],
    )
    assert_init_carry(gm, steps[-1]["carry"])
    # the signal table: who signalled what, holding four or not
    np.testing.assert_array_equal(steps[1]["counts"]["signal_quad"][:, 0], [[1, 0], [1, 0], [0, 1], [1, 0]])


def test_four_completed_in_the_calling_step_does_not_count():
    """Seat 0 calls KEMPS in the step its partner completes four of a kind:
    the call is judged before the swap (wrong), and the team never held four
    at the start of a step of the hand."""
    env = Kemps(num_agents=4)
    gm = Metrics(env)
    state, swap = completing_swap_case(env)
    (step,) = play(env, gm, state, [([kemps(env), noop(env), swap, noop(env)], [0, 1, 2, 3])])
    assert step["done"]
    np.testing.assert_array_equal(step["reward"], [-1, 1, -1, 1])
    assert_counts(step["counts"], kemps_calls=[1, 0, 0, 0], kemps_right=[0, 0, 0, 0])
    for name in HAND_LEVEL:
        np.testing.assert_array_equal(step["counts"][name], 0, err_msg=name)


def test_cut_off_hand_ends_without_a_call():
    """Seat 1 holds four of a kind at step 0, swaps one away, and the hand is
    cut off at the horizon (2 steps): team {1, 3} held four in a hand that
    ended without a call."""
    env = Kemps(num_agents=4, horizon=2)
    gm = Metrics(env)
    hands, center, stock = deal(env, {1: 5})
    state = make_state(env, hands, center, stock)
    gain = int(center[0]) // S
    swap = 5 * env.num_ranks + gain
    n = noop(env)
    steps = play(env, gm, state, [([n, swap, n, n], [0, 1, 2, 3]), ([n, n, n, n], [0, 1, 2, 3])])
    assert [s["done"] for s in steps] == [False, True]
    np.testing.assert_array_equal(steps[-1]["reward"], 0.0)
    # the scripts' rule agrees: a cut-off
    before_last = env._step_in_order(state, act(env, [n, swap, n, n]), jnp.arange(4))[0]
    cut = kemps_horizon_cutoff(
        env, jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], before_last),
        act(env, [n] * 4)[None], jnp.array([True]),
    )
    assert bool(cut[0])
    assert_counts(
        steps[-1]["counts"],
        own_quad_hands=[0, 1, 0, 1], opponent_quad_hands=[1, 0, 1, 0],
        intercepted=[0] * 4, quad_called=[0] * 4, quads_called=[0] * 4, latency_sum=[0] * 4,
    )
    assert_init_carry(gm, steps[-1]["carry"])


def test_declaration_at_the_horizon_is_a_call():
    env = Kemps(num_agents=4, horizon=1)
    gm = Metrics(env)
    state = make_state(env, *deal(env, {2: 0}))
    (step,) = play(env, gm, state, [([kemps(env)] + [noop(env)] * 3, [0, 1, 2, 3])])
    assert step["done"]
    np.testing.assert_array_equal(step["reward"], [1, -1, 1, -1])
    assert_counts(step["counts"], quad_called=[1, 0, 1, 0], quads_called=[1, 0, 0, 0])


def test_real_deal_ends_without_a_call():
    env = Kemps(num_agents=4)
    gm = Metrics(env)
    hands, center, stock = deal(env, {2: 0})
    state = make_state(env, hands, center, stock, deck_idx=len(stock) - 3)  # can't refill
    (step,) = play(env, gm, state, [([noop(env)] * 4, [0, 1, 2, 3])])
    assert step["done"]
    np.testing.assert_array_equal(step["reward"], 0.0)
    assert_counts(
        step["counts"], own_quad_hands=[1, 0, 1, 0], opponent_quad_hands=[0, 1, 0, 1],
        intercepted=[0] * 4, quad_called=[0] * 4, quads_called=[0] * 4,
    )


@pytest.mark.parametrize(
    "first, intercepted, quad_called",
    [
        (1, [0, 1, 1, 0, 1, 1], [0] * 6),  # seat 1 stops team {0, 3}
        (2, [0, 1, 1, 0, 1, 1], [0] * 6),  # seat 2 too, STOP KEMPS 1
        (5, [1, 0, 1, 1, 0, 1], [0] * 6),  # seat 5 stops team {1, 4}
        (0, [0] * 6, [1, 0, 0, 1, 0, 0]),  # seat 0's KEMPS on partner 3
    ],
)
def test_three_teams(first, intercepted, quad_called):
    """Teams {0, 3}, {1, 4}, {2, 5}; seats 3 and 4 hold four of a kind. Seat 1
    calls STOP KEMPS 2 (team of seat 3), seat 2 STOP KEMPS 1 (team of seat
    3), seat 5 STOP KEMPS 2 (team of seat 1), seat 0 KEMPS: all right."""
    env = Kemps(num_agents=6)
    gm = Metrics(env)
    state = make_state(env, *deal(env, {3: 0, 4: 1}))
    game = [kemps(env), stop(env, 2), stop(env, 1), noop(env), noop(env), stop(env, 2)]
    order = [first] + [s for s in range(6) if s != first]
    (step,) = play(env, gm, state, [(game, order)])
    c = step["counts"]
    assert_counts(
        c, kemps_calls=[1, 0, 0, 0, 0, 0], kemps_right=[1, 0, 0, 0, 0, 0],
        stop_calls=[0, 1, 1, 0, 0, 1], stop_right=[0, 1, 1, 0, 0, 1],
        own_quad_hands=[1, 1, 0, 1, 1, 0],
        # other teams that held four: team 2 sees both, teams 0 and 1 each other
        opponent_quad_hands=[1, 1, 2, 1, 1, 2],
        intercepted=intercepted, quad_called=quad_called,
    )
    # the env's letters agree: the accused (or, for KEMPS, every other) team pays
    letter_team = {1: 0, 2: 0, 5: 1}.get(first)
    team_reward = np.asarray(step["reward"])[:3]
    if letter_team is None:
        np.testing.assert_allclose(team_reward, [1, -0.5, -0.5])
    else:
        assert team_reward[letter_team] == pytest.approx(-1.0)
    # a wrong STOP KEMPS: seat 1 accuses team {2, 5} (STOP KEMPS 1)
    game = [noop(env), stop(env, 1)] + [noop(env)] * 4
    (step,) = play(env, gm, state, [(game, [1, 0, 2, 3, 4, 5])])
    assert_counts(step["counts"], stop_calls=[0, 1, 0, 0, 0, 0], stop_right=[0] * 6, intercepted=[0] * 6)


def test_step_order_is_the_order_env_step_uses():
    env = Kemps(num_agents=4)
    gm = Metrics(env)
    state, game = simultaneous_case(env)
    action = act(env, game)
    keys = jax.random.split(KEY, 400)

    @jax.jit
    @jax.vmap
    def run(key):
        _, _, reward, _, done, info = env.step(key, state, action)
        order = step_order(env, key)
        by_order = env._step_in_order(state, action, order)[2]
        _, counts = gm.step(gm.init_carry(), state, action, state, reward, done, {**info, "order": order})
        return reward, by_order, counts

    reward, by_order, counts = run(keys)
    reward, by_order = np.asarray(reward), np.asarray(by_order)
    np.testing.assert_array_equal(reward, by_order)
    intercepted = np.asarray(counts["intercepted"])[:, 1] == 1
    called = np.asarray(counts["quad_called"])[:, 0] == 1
    assert not (intercepted & called).any()
    # team 0 pays iff the STOP KEMPS counted, wins iff a KEMPS did (its right one
    # or seat 3's wrong one)
    np.testing.assert_array_equal(intercepted, reward[:, 0] == -1)
    assert 0.2 < intercepted.mean() < 0.47 and 0.2 < called.mean() < 0.47


def test_mutual_information_bits():
    np.testing.assert_allclose(mutual_information_bits(jnp.array([[10.0, 30.0], [20.0, 60.0]])), 0.0, atol=1e-6)
    a, b = 30.0, 10.0
    p = a / (a + b)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    np.testing.assert_allclose(mutual_information_bits(jnp.array([[a, 0.0], [0.0, b]])), entropy, rtol=1e-6)
    table = np.random.default_rng(0).integers(0, 50, (3, 2)).astype(np.float64)
    pj = table / table.sum()
    want = (pj * np.log2(pj / pj.sum(1, keepdims=True) / pj.sum(0, keepdims=True))).sum()
    np.testing.assert_allclose(mutual_information_bits(jnp.asarray(table)), want, rtol=1e-5)
    assert np.isnan(mutual_information_bits(jnp.zeros((2, 2))))


# =============================================================================
# Random play
# =============================================================================


def collector_logits(env, state, mask, beta, p_declare):
    """Swaps that collect a rank are likelier (logit beta * (count of the
    gained rank - count of the lost rank)), NOOP has logit 0 and each
    declaration log(p_declare); every signal is equally likely."""
    R, k = env.num_ranks, env.num_teams
    counts = state.agent_hand_counts.astype(jnp.float32)
    lose, gain = jnp.arange(R * R) // R, jnp.arange(R * R) % R
    swap = beta * (counts[:, gain] - counts[:, lose])
    other = jnp.concatenate(
        [jnp.zeros((env.num_agents, 1)), jnp.full((env.num_agents, k), jnp.log(p_declare))], axis=1
    )
    logits = jnp.repeat(jnp.concatenate([swap, other], axis=1), env.comm_dim, axis=1)
    return jnp.where(mask, logits, -jnp.inf)


def random_rollout(env, num_envs, num_steps, seed, signal="random", beta=2.0, p_declare=0.05):
    """Training-style play through env.step (auto-reset) with the metrics
    vmapped over envs and the order from step_order. signal: "random"
    (uniform, independent of the hand) or "quad" (1 iff holding four of a
    kind). Returns numpy arrays."""
    gm = Metrics(env)

    @jax.jit
    def run(key):
        k_reset, k_play = jax.random.split(key)
        state, _ = jax.vmap(env.reset)(jax.random.split(k_reset, num_envs))
        carry = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (num_envs,) + x.shape), gm.init_carry()
        )

        def one(c, k):
            state, carry = c
            k_act, k_sig, k_step = jax.random.split(k, 3)
            mask = jax.vmap(env.get_avail_actions)(state)
            logits = jax.vmap(lambda s, m: collector_logits(env, s, m, beta, p_declare))(state, mask)
            game_action = jax.random.categorical(k_act, logits) // env.comm_dim
            has_four = (state.agent_hand_counts >= 4).any(-1)
            if signal == "random":
                sig = jax.random.randint(k_sig, game_action.shape, 0, env.comm_dim)
            else:
                sig = has_four.astype(jnp.int32)
            action = game_action * env.comm_dim + sig
            keys = jax.random.split(k_step, num_envs)
            nxt, _, reward, _, done, info = jax.vmap(env.step)(keys, state, action)
            order = jax.vmap(lambda kk: step_order(env, kk))(keys)
            carry, counts = jax.vmap(gm.step)(
                carry, state, action, nxt, reward, done, {**info, "order": order}
            )
            rec = dict(
                has_four=has_four, game_action=game_action, signal=sig, order=order,
                done=done, reward=reward, cutoff=kemps_horizon_cutoff(env, state, action, done),
            )
            return (nxt, carry), (counts, rec)

        _, out = jax.lax.scan(one, (state, carry), jax.random.split(k_play, num_steps))
        return out

    counts, rec = jax.tree_util.tree_map(np.asarray, run(jax.random.PRNGKey(seed)))
    return gm, counts, rec


def brute_force(env, rec):
    """The counts of every step from the definitions, loop by loop."""
    has_four, ga, sig = rec["has_four"], rec["game_action"], rec["signal"]
    order, done = rec["order"], rec["done"]
    T, E, P = ga.shape
    k, K = env.num_teams, env.action_kemps
    names = ["kemps_calls", "kemps_right", "stop_calls", "stop_right", "opponent_quad_hands",
             "intercepted", "own_quad_hands", "quad_called", "latency_sum", "quads_called"]
    out = {name: np.zeros((T, E, P)) for name in names}
    out["signal_quad"] = np.zeros((T, E, P, env.comm_dim, 2))
    rewards = np.zeros((T, E, P))
    for e in range(E):
        team_had = np.zeros(k, bool)
        first = np.full(P, -1)
        t_hand = 0
        for t in range(T):
            hf = has_four[t, e]
            tf = np.array([hf[j] or hf[j + k] for j in range(k)])
            for p in range(P):
                partner = (p + k) % P
                if ga[t, e, p] == K:
                    out["kemps_calls"][t, e, p] = 1
                    out["kemps_right"][t, e, p] = hf[partner]
                elif ga[t, e, p] > K:
                    out["stop_calls"][t, e, p] = 1
                    out["stop_right"][t, e, p] = tf[(p + ga[t, e, p] - K) % k]
                out["signal_quad"][t, e, p, sig[t, e, p], int(hf[p])] = 1
                if first[p] < 0 and hf[p]:
                    first[p] = t_hand
            team_had |= tf
            if done[t, e]:
                declarers = [p for p in order[t, e] if ga[t, e, p] >= K]
                letters = np.zeros(k)
                for p in range(P):
                    out["own_quad_hands"][t, e, p] = team_had[p % k]
                    out["opponent_quad_hands"][t, e, p] = team_had.sum() - team_had[p % k]
                if declarers:
                    c = declarers[0]
                    if ga[t, e, c] == K:
                        right = hf[(c + k) % P]
                        letters = np.ones(k) if right else np.zeros(k)
                        letters[c % k] = 1 - right
                        if right:
                            for p in range(P):
                                out["quad_called"][t, e, p] = p % k == c % k
                            out["quads_called"][t, e, c] = 1
                            out["latency_sum"][t, e, c] = t_hand - first[(c + k) % P]
                    else:
                        accused = (c + ga[t, e, c] - K) % k
                        right = tf[accused]
                        letters[accused if right else c % k] = 1
                        if right:
                            for p in range(P):
                                out["intercepted"][t, e, p] = p % k != accused
                    rewards[t, e] = (k / (k - 1) * (letters.mean() - letters))[np.arange(P) % k]
                team_had[:] = False
                first[:] = -1
                t_hand = 0
            else:
                t_hand += 1
    return out, rewards


@pytest.mark.parametrize("num_agents, num_ranks, horizon", [(4, 6, 10), (6, 8, 8), (4, 13, 12)])
def test_random_play_matches_brute_force(num_agents, num_ranks, horizon):
    env = Kemps(num_agents=num_agents, num_ranks=num_ranks, horizon=horizon)
    gm, counts, rec = random_rollout(env, num_envs=48, num_steps=150, seed=num_agents + num_ranks)
    want, rewards = brute_force(env, rec)
    for name, value in want.items():
        np.testing.assert_array_equal(counts[name], value, err_msg=name)
    # the declaration that counts gives the env's rewards
    np.testing.assert_allclose(rec["reward"], rewards, atol=1e-6)
    # hands cut off at the horizon end without a call
    cut = rec["cutoff"]
    assert cut.sum() > 5 and not (rec["game_action"][cut] >= env.action_kemps).any()
    for name in HAND_LEVEL[1:2] + HAND_LEVEL[3:]:
        assert not counts[name][cut].any(), name
    if num_ranks < 13:
        # the cases are there (with few ranks four of a kind is common): calls
        # of both kinds, right and wrong, hands that end with an interception
        # and with an own call, and cut-off hands in which a team held four
        totals = {k: v.sum((0, 1)) for k, v in counts.items()}
        assert totals["kemps_right"].sum() > 5
        assert (totals["kemps_calls"] - totals["kemps_right"]).sum() > 5
        assert totals["stop_right"].sum() > 5
        assert (totals["stop_calls"] - totals["stop_right"]).sum() > 5
        assert totals["intercepted"].sum() > 0 and totals["quads_called"].sum() > 0
        assert (counts["own_quad_hands"].sum(-1) > 0)[cut].sum() > 0
        out = gm.summarize(masked_totals(counts, np.ones(rec["game_action"].shape, bool)))
        for name in gm.rate_names:
            assert np.isfinite(float(out[name])), name
        assert float(out["kemps_latency"]) >= 0


def test_random_signals_carry_no_information():
    """Signals drawn independently of the hand: the plug-in MI is about
    chi2(1) / (2 ln 2 N) bits (G-test), although players often hold four of a
    kind; signalling exactly that gives the entropy of holding it."""
    env = Kemps(num_agents=4, num_ranks=6, horizon=20)
    gm, counts, rec = random_rollout(env, num_envs=64, num_steps=200, seed=11, signal="random")
    out = gm.summarize(masked_totals(counts, np.ones(rec["game_action"].shape, bool)))
    n = float(out["n_decisions"])
    q = rec["has_four"].mean()
    assert n == rec["has_four"].size and 0.05 < q < 0.95
    g = 2 * n * np.log(2) * float(out["signal_mi"])
    assert 0 <= g < 20  # P(chi2(1) > 20) < 1e-5
    gm, counts, rec = random_rollout(env, num_envs=64, num_steps=200, seed=12, signal="quad")
    out = gm.summarize(masked_totals(counts, np.ones(rec["game_action"].shape, bool)))
    q = rec["has_four"].mean()
    entropy = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
    np.testing.assert_allclose(float(out["signal_mi"]), entropy, rtol=1e-4)
    assert float(out["n_signal_0_quad_1"]) == 0 and float(out["n_signal_1_quad_0"]) == 0


def test_masked_totals_by_player():
    """game/ and game_br/ in the scripts: the counts of the masked players only."""
    env = Kemps(num_agents=4, num_ranks=6, horizon=10)
    gm, counts, rec = random_rollout(env, num_envs=16, num_steps=60, seed=3)
    mask = np.zeros(rec["game_action"].shape, bool)
    mask[..., [0, 2]] = True  # team 0
    out = gm.summarize(masked_totals(counts, mask))
    assert float(out["n_kemps_calls"]) == (rec["game_action"][..., [0, 2]] == env.action_kemps).sum()
    assert float(out["n_decisions"]) == mask.sum()
    np.testing.assert_allclose(
        float(out["quad_call_rate"]),
        counts["quad_called"][..., [0, 2]].sum() / counts["own_quad_hands"][..., [0, 2]].sum(),
        rtol=1e-6,
    )
