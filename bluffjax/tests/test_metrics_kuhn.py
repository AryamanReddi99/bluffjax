"""Checks for the exact Kuhn Poker policy metrics (bluffjax/metrics/kuhn_poker.py).

- The equilibrium family nash_equilibrium(alpha) of kuhn_exploitability: the
  frequencies are their equilibrium values, both gaps are 0 and alpha is
  recovered; the equilibrium values in the metric module are the ones of
  nash_equilibrium, and a first player with p0_alpha_gap = 0 is unexploitable.
- Uniform random play: every frequency is 1/2, both gaps 1/2, alpha 1/4.
- Hand-made policies with values worked out by hand, and random policies
  against a brute-force minimisation over a fine grid of alpha.
- Seats: the first player's metrics come from policy_p0 only and the second
  player's from policy_p1 only; policy_p1 defaults to policy_p0.
- Monte Carlo: hands played in the env (make("kuhn_poker")) with a fixed
  policy table, the acting player's row picked from its observation as the
  training scripts do, give the table's frequencies within 5 standard errors
  for situations defined from the env state (card, who acts first, the actions
  so far), so the table rows mean what the metrics assume.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.metrics import kuhn_poker as km
from bluffjax.utils.game_utils import kuhn_exploitability as ke

ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.25, 1.0 / 3.0)
FIRST = ("0", "1", "2", "0pb", "1pb", "2pb")
SECOND = ("0p", "1p", "2p", "0b", "1b", "2b")


def table_from_bets(bets: dict[str, float]) -> np.ndarray:
    """(12, 2) policy array with P(bet) = bets[key] (0.5 where missing)."""
    p_bet = np.array([bets.get(k, 0.5) for k in ke.INFOSETS], dtype=np.float64)
    return np.stack([1.0 - p_bet, p_bet], axis=-1)


def nash_table(alpha: float) -> np.ndarray:
    nash = ke.nash_equilibrium(alpha)
    return np.stack([nash(k) for k in ke.INFOSETS])


def brute_force_alpha(bets: dict[str, float], num: int = 300_001):
    """min over a grid of alpha of the largest first-player gap, and of the
    largest gap at the alpha-dependent infosets (h of the module docstring)."""
    alphas = np.linspace(0.0, 1.0 / 3.0, num)
    gaps = np.stack(
        [
            np.abs(bets[k] - (offset + slope * alphas))
            for k, (offset, slope) in km.FIRST_PLAYER_EQUILIBRIUM.items()
        ]
    )
    full = gaps.max(axis=0)
    dependent = np.stack(
        [
            g
            for g, (_, slope) in zip(gaps, km.FIRST_PLAYER_EQUILIBRIUM.values())
            if slope
        ]
    ).max(axis=0)
    return alphas, full, dependent


# ------------------------------------------------------------ known values
def test_equilibrium_values_match_nash_equilibrium():
    """The module's equilibrium table is kuhn_exploitability's nash_equilibrium."""
    for alpha in ALPHAS:
        nash = ke.nash_equilibrium(alpha)
        for key, (offset, slope) in km.FIRST_PLAYER_EQUILIBRIUM.items():
            assert nash(key)[ke.ACTION_BET] == pytest.approx(offset + slope * alpha, abs=1e-15)
        for key, value in km.SECOND_PLAYER_EQUILIBRIUM.items():
            assert nash(key)[ke.ACTION_BET] == pytest.approx(value, abs=1e-15)
    assert set(km.FIRST_PLAYER_EQUILIBRIUM) == set(FIRST)
    assert set(km.SECOND_PLAYER_EQUILIBRIUM) == set(SECOND)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_equilibrium_family(alpha):
    for policy in (nash_table(alpha), ke.nash_equilibrium(alpha)):
        m = km.policy_metrics(policy)
        assert list(m) == list(km.METRIC_NAMES)
        assert m["p0_bet_jack"] == pytest.approx(alpha, abs=1e-12)
        assert m["p0_bet_king"] == pytest.approx(3 * alpha, abs=1e-12)
        assert m["p1_bet_jack"] == pytest.approx(1 / 3, abs=1e-12)
        assert m["p0_call_queen"] == pytest.approx(alpha + 1 / 3, abs=1e-12)
        assert m["p1_call_queen"] == pytest.approx(1 / 3, abs=1e-12)
        assert m["p1_equilibrium_gap"] == pytest.approx(0.0, abs=1e-12)
        assert m["p0_alpha_gap"] == pytest.approx(0.0, abs=1e-12)
        assert m["p0_alpha"] == pytest.approx(alpha, abs=1e-12)


def test_uniform_random():
    m = km.policy_metrics(np.full((12, 2), 0.5))
    assert m == km.policy_metrics(ke.uniform_random_policy())
    for name in ("p0_bet_jack", "p0_bet_king", "p1_bet_jack", "p0_call_queen", "p1_call_queen"):
        assert m[name] == 0.5
    # second player: |1/2 - 0|, |1/2 - 1| at four infosets
    assert m["p1_equilibrium_gap"] == pytest.approx(0.5, abs=1e-15)
    # first player: |b(Q) - 0| = 1/2 doesn't depend on alpha; the alpha-dependent
    # gaps |1/2 - a|, |1/2 - 3a|, |1/6 - a| are balanced at a = 1/4 (1/4 each)
    assert m["p0_alpha_gap"] == pytest.approx(0.5, abs=1e-15)
    assert m["p0_alpha"] == pytest.approx(0.25, abs=1e-15)


def test_hand_made_policies():
    # Bets the King twice as often as the equilibrium for its Jack bluffs.
    # h(a) = max(|0.1 - a|, |0.6 - 3a|, |1/2 - 1/3 - a|): the first two cross
    # at a = 0.175 with gap 0.075 (the third is then 1/120).
    bets = {"0": 0.1, "1": 0.0, "2": 0.6, "0pb": 0.0, "1pb": 0.5, "2pb": 1.0}
    bets.update({"0p": 0.5, "1p": 0.2, "2p": 0.9, "0b": 0.05, "1b": 0.6, "2b": 0.7})
    m = km.policy_metrics(table_from_bets(bets))
    assert m["p0_bet_jack"] == 0.1 and m["p0_bet_king"] == 0.6
    assert m["p0_call_queen"] == 0.5
    assert m["p1_bet_jack"] == 0.5 and m["p1_call_queen"] == 0.6
    assert m["p0_alpha"] == pytest.approx(0.175, abs=1e-12)
    assert m["p0_alpha_gap"] == pytest.approx(0.075, abs=1e-12)
    # second player: 1/6, 0.2, 0.1, 0.05, 0.6 - 1/3, 0.3
    assert m["p1_equilibrium_gap"] == pytest.approx(0.3, abs=1e-12)

    # A pure first player that bets the Queen and the King, calls with the
    # Jack and folds the Queen and the King after check-bet: b(Q) = b(Jpb) = 1
    # and 1 - b(Kpb) = 1 dominate. Its alpha balances |1 - 3a| against
    # |0 - 1/3 - a| (|0 - a| is smaller) at a = 1/6, where both are 1/2.
    bets = {"0": 0.0, "1": 1.0, "2": 1.0, "0pb": 1.0, "1pb": 0.0, "2pb": 0.0}
    m = km.policy_metrics(table_from_bets(bets))
    assert m["p0_alpha_gap"] == pytest.approx(1.0, abs=1e-12)
    assert m["p0_alpha"] == pytest.approx(1 / 6, abs=1e-12)

    # Only the Queen call is off: the equilibrium of alpha = 0.1 except
    # b(Qpb) = 0.2. h balances |0.1 - a| against |0.2 - 1/3 - a| (and
    # |0.3 - 3a|): at a = 0.1 the gap is 0.2333; moving a lowers the Queen
    # gap only as fast as it raises the Jack and King gaps, so a = 0.1 - d
    # with 3d = 0.2333 - d: d = 0.058333, gap 3d = 0.175.
    bets = {"0": 0.1, "1": 0.0, "2": 0.3, "0pb": 0.0, "1pb": 0.2, "2pb": 1.0}
    m = km.policy_metrics(table_from_bets(bets))
    assert m["p0_alpha"] == pytest.approx(0.1 - 0.7 / 12, abs=1e-12)
    assert m["p0_alpha_gap"] == pytest.approx(0.175, abs=1e-12)

    # A first player outside [0, 1/3]: always bets the Jack and the King,
    # otherwise equilibrium-like. The gaps 1 - a and |1 - 3a| are smallest at
    # the upper end a = 1/3 (gap 2/3; the Queen call there: |1/3 + 1/3 - 2/3| = 0).
    bets = {"0": 1.0, "1": 0.0, "2": 1.0, "0pb": 0.0, "1pb": 2 / 3, "2pb": 1.0}
    m = km.policy_metrics(table_from_bets(bets))
    assert m["p0_alpha"] == pytest.approx(1 / 3, abs=1e-15)
    assert m["p0_alpha_gap"] == pytest.approx(2 / 3, abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_alpha_fit_against_brute_force(seed):
    rs = np.random.RandomState(seed)
    for trial in range(40):
        p_bet = rs.rand(12)
        if trial % 4 == 0:  # near the equilibrium family
            alpha = rs.rand() / 3
            p_bet = nash_table(alpha)[:, 1] + 0.05 * rs.randn(12)
        p_bet = np.clip(p_bet, 0.0, 1.0)
        table = np.stack([1 - p_bet, p_bet], -1)
        bets = dict(zip(ke.INFOSETS, p_bet))
        m = km.policy_metrics(table)
        alphas, full, dependent = brute_force_alpha(bets)
        step = alphas[1]
        # exact minimum: never above the grid minimum, and at most the
        # largest slope (3) times half a grid step below it
        assert m["p0_alpha_gap"] <= full.min() + 1e-12
        assert m["p0_alpha_gap"] >= full.min() - 3 * step
        assert 0.0 <= m["p0_alpha"] <= 1 / 3
        # p0_alpha minimises h and attains the full minimum
        h = max(
            abs(bets[k] - (o + s * m["p0_alpha"]))
            for k, (o, s) in km.FIRST_PLAYER_EQUILIBRIUM.items()
            if s
        )
        g = max(
            abs(bets[k] - (o + s * m["p0_alpha"]))
            for k, (o, s) in km.FIRST_PLAYER_EQUILIBRIUM.items()
        )
        assert h <= dependent.min() + 1e-12
        assert g == pytest.approx(m["p0_alpha_gap"], abs=1e-12)
        # h grows at least linearly away from its minimiser: the grid argmin
        # is within a grid step of p0_alpha
        assert abs(alphas[dependent.argmin()] - m["p0_alpha"]) <= step + 1e-12
        # second player
        gap = max(abs(bets[k] - v) for k, v in km.SECOND_PLAYER_EQUILIBRIUM.items())
        assert m["p1_equilibrium_gap"] == pytest.approx(gap, abs=1e-15)


def test_zero_gaps_are_unexploitable():
    """A first player in the family can't be exploited (the second player's
    best response wins 1/18, the game value), one off it can; likewise for the
    second player (the first player's best response wins -1/18)."""
    rs = np.random.RandomState(0)
    for alpha in ALPHAS:
        nash = nash_table(alpha)
        mixed = nash.copy()
        mixed[[ke.INFOSETS.index(k) for k in SECOND]] = rs.dirichlet([1, 1], size=6)
        assert km.policy_metrics(mixed)["p0_alpha_gap"] < 1e-12
        br1, _ = ke.best_response_value_infoset(1, ke.policy_from_array(mixed))
        assert br1 == pytest.approx(1 / 18, abs=1e-12)
    for _ in range(10):
        table = rs.dirichlet([1, 1], size=12)
        assert km.policy_metrics(table)["p0_alpha_gap"] > 1e-3
        br1, _ = ke.best_response_value_infoset(1, ke.policy_from_array(table))
        br0, _ = ke.best_response_value_infoset(0, ke.policy_from_array(table))
        assert br1 > 1 / 18 + 1e-6 and br0 > -1 / 18 + 1e-6
    table = rs.dirichlet([1, 1], size=12)
    table[[ke.INFOSETS.index(k) for k in SECOND]] = nash_table(0.0)[
        [ke.INFOSETS.index(k) for k in SECOND]
    ]
    assert km.policy_metrics(table)["p1_equilibrium_gap"] < 1e-12
    br0, _ = ke.best_response_value_infoset(0, ke.policy_from_array(table))
    assert br0 == pytest.approx(-1 / 18, abs=1e-12)


def test_seats():
    rs = np.random.RandomState(1)
    a = rs.dirichlet([1, 1], size=12)
    b = rs.dirichlet([1, 1], size=12)
    m = km.policy_metrics(a, b)
    first = [ke.INFOSETS.index(k) for k in FIRST]
    second = [ke.INFOSETS.index(k) for k in SECOND]
    # replacing A's second-player rows or B's first-player rows changes nothing
    a2, b2 = a.copy(), b.copy()
    a2[second] = rs.dirichlet([1, 1], size=6)
    b2[first] = rs.dirichlet([1, 1], size=6)
    assert km.policy_metrics(a2, b2) == m
    ab = a.copy()
    ab[second] = b[second]
    assert km.policy_metrics(ab) == m
    assert km.policy_metrics(a) == km.policy_metrics(a, a)
    assert m["p0_bet_jack"] == a[ke.INFOSETS.index("0"), 1]
    assert m["p1_bet_jack"] == b[ke.INFOSETS.index("0p"), 1]
    assert m["p1_call_queen"] == b[ke.INFOSETS.index("1b"), 1]
    assert m["p0_call_queen"] == a[ke.INFOSETS.index("1pb"), 1]


def test_rejects_non_policies():
    with pytest.raises(ValueError):
        km.policy_metrics(np.full((9, 2), 0.5))
    with pytest.raises(ValueError):
        km.policy_metrics(np.full((12, 2), 0.6))
    bad = np.full((12, 2), 0.5)
    bad[3] = [1.5, -0.5]
    with pytest.raises(ValueError):
        km.policy_metrics(bad)
    with pytest.raises(ValueError):
        km.policy_metrics(np.full((12, 2), 0.5), np.full((12, 2), np.nan))
    # float32 softmax output is accepted
    logits = jnp.asarray(np.random.RandomState(0).randn(12, 2), jnp.float32)
    km.policy_metrics(np.asarray(jax.nn.softmax(logits)))


# ------------------------------------------------------------ Monte Carlo
NUM_HANDS = 120_000
NUM_STEPS = 3  # a Kuhn hand has at most 3 actions


@pytest.fixture(scope="module")
def env_hands():
    """One hand in each of NUM_HANDS envs (env.step, as in training) with a
    fixed policy table, the acting player's row picked by matching its
    observation with INFOSET_OBS (the rows policy_array_from_network evaluates
    a network on). Per step (NUM_STEPS, NUM_HANDS): the acting player's card,
    the opponent's card, its position (0 = first to act), the env timestep,
    both pots before the step, its pot after it (meaningless at the hand's
    end, where env.step has already reset), the action, its reward, done, and
    whether the hand is still being played."""
    env = make("kuhn_poker")
    rs = np.random.RandomState(7)
    p_bet = rs.uniform(0.15, 0.85, size=12)
    table = np.stack([1 - p_bet, p_bet], -1)
    infoset_obs = jnp.asarray(ke.INFOSET_OBS)
    bet_prob = jnp.asarray(p_bet, jnp.float32)

    @jax.jit
    def play(key):
        k_reset, k_play = jax.random.split(key)
        state, obs = jax.vmap(env.reset)(jax.random.split(k_reset, NUM_HANDS))
        idx = jnp.arange(NUM_HANDS)

        def body(carry, k):
            state, obs, active = carry
            match = (obs[:, None, :] == infoset_obs[None]).all(-1)
            row = jnp.argmax(match, axis=-1)
            k_action, k_step = jax.random.split(k)
            action = (
                jax.random.uniform(k_action, (NUM_HANDS,)) < bet_prob[row]
            ).astype(jnp.int32)
            player = state.current_player_idx
            next_state, next_obs, reward, _, done, _ = jax.vmap(env.step)(
                jax.random.split(k_step, NUM_HANDS), state, action
            )
            record = dict(
                matched=match.sum(-1),
                card=state.agent_hands[idx, player],
                other_card=state.agent_hands[idx, 1 - player],
                position=(player - state.start_player_idx) % 2,
                timestep=state.timestep,
                own_pot=state.pot[idx, player],
                opp_pot=state.pot[idx, 1 - player],
                next_own_pot=next_state.pot[idx, player],
                action=action,
                reward=reward[idx, player],
                done=done,
                active=active,
            )
            return (next_state, next_obs, active & ~done), record

        _, records = jax.lax.scan(
            body,
            (state, obs, jnp.ones(NUM_HANDS, jnp.bool_)),
            jax.random.split(k_play, NUM_STEPS),
        )
        return records

    records = {k: np.asarray(v) for k, v in play(jax.random.PRNGKey(3)).items()}
    # the betting before each step of the hand, "p" = action 0, "b" = action 1
    letters = np.where(records["action"] == 1, "b", "p")
    history = [np.full(NUM_HANDS, "", dtype="<U3")]
    for t in range(1, NUM_STEPS):
        history.append(np.char.add(history[-1], letters[t - 1]))
    records["history"] = np.stack(history)
    records["betting"] = np.char.add(records["history"], letters)  # including the step
    return table, records


def test_env_play_is_what_the_metrics_assume(env_hands):
    """The env conventions the metrics rely on: action 1 puts a chip in (a bet
    or a call), action 0 doesn't (a check or a fold), cards 0 < 1 < 2, and the
    hands end exactly after pp, bp, bb, pbp and pbb."""
    _, r = env_hands
    act = r["active"]
    assert (r["matched"][act] == 1).all()  # every observation is one infoset row
    steps = np.broadcast_to(np.arange(NUM_STEPS)[:, None], act.shape)
    np.testing.assert_array_equal(r["timestep"][act], steps[act])
    live = act & ~r["done"]
    np.testing.assert_array_equal(
        r["next_own_pot"][live] - r["own_pot"][live], r["action"][live]
    )
    assert set(r["betting"][live]) == {"p", "b", "pb"}
    end = act & r["done"]
    assert set(r["betting"][end]) == {"pp", "bp", "bb", "pbp", "pbb"}
    assert (r["done"][-1] | ~act[-1]).all()  # every hand ended within NUM_STEPS
    higher = r["card"] > r["other_card"]
    for betting, payoff in (("pp", 1.0), ("bb", 2.0), ("pbb", 2.0)):  # showdowns
        m = end & (r["betting"] == betting)
        np.testing.assert_array_equal(r["reward"][m], np.where(higher[m], payoff, -payoff))
    for betting in ("bp", "pbp"):  # the acting player folds and loses its ante
        m = end & (r["betting"] == betting)
        np.testing.assert_array_equal(r["reward"][m], -1.0)


def test_env_frequencies_match_metrics(env_hands):
    table, r = env_hands
    act = r["active"]
    exact = km.policy_metrics(table)

    def check(mask, expected):
        """The frequency of action 1 over mask is expected within 5 standard
        errors; returns that bound."""
        n = int(mask.sum())
        assert n > 1000
        freq = r["action"][mask].mean()
        bound = 5 * np.sqrt(expected * (1 - expected) / n)
        assert abs(freq - expected) < bound, (freq, expected, n)
        return bound

    # the five frequencies, in situations read off the env state
    first = act & (r["position"] == 0)
    second = act & (r["position"] == 1)
    opening = r["timestep"] == 0
    after_check = (r["timestep"] == 1) & (r["opp_pot"] == 1)
    facing_bet = (r["opp_pot"] == 2) & (r["own_pot"] == 1)
    jack, queen, king = (r["card"] == c for c in range(3))
    check(first & opening & jack, exact["p0_bet_jack"])
    check(first & opening & king, exact["p0_bet_king"])
    check(second & after_check & jack, exact["p1_bet_jack"])
    check(first & facing_bet & queen, exact["p0_call_queen"])
    check(second & facing_bet & queen, exact["p1_call_queen"])

    # every infoset, keyed by the env's card and the actions so far; the gap
    # metrics of the empirical table are within the sampling error (they are
    # 1-Lipschitz in the table entries under the max norm)
    empirical = np.zeros((12, 2))
    tolerance = 0.0
    for i, key in enumerate(ke.INFOSETS):
        card, betting = int(key[0]), key[1:]
        mask = act & (r["card"] == card) & (r["history"] == betting)
        # the first player acts after an even number of actions
        assert (r["position"][mask] == len(betting) % 2).all()
        tolerance = max(tolerance, check(mask, table[i, 1]))
        freq = r["action"][mask].mean()
        empirical[i] = [1 - freq, freq]
    sampled = km.policy_metrics(empirical)
    for name in ("p1_equilibrium_gap", "p0_alpha_gap"):
        assert abs(sampled[name] - exact[name]) <= tolerance, name
