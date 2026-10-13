"""Evaluation against uniformly random play of the Kuhn and Leduc scripts.

Exact value (value_vs_random in bluffjax.utils.game_utils.{kuhn,leduc}_
exploitability): the expected return per hand of a policy table against
uniform random legal play, averaged over both seats.
- Uniform random play scores exactly 0, and a best response to uniform play
  scores the exploitability of uniform play (0.458333 in Kuhn, 2.373611 in
  Leduc): it is (BR value as player 0 + BR value as player 1) / 2.
- Kuhn's equilibria nash_equilibrium(alpha) score 1/9 + alpha / 6, derived by
  hand in test_kuhn_equilibrium_value.
- Leduc's value_vs_random_from_policy_array (a sum over the terminal
  histories) equals value_vs_random, the traversal with the exploitability's
  _state_values and _policy_from_array_callable, for random, deterministic
  and degenerate tables and any row order (key_to_idx).
- Brute force in the env: walking every deal, start player and action of
  make("kuhn_poker") / make("leduc_holdem") with env.step_env, the learner
  choosing its action from its env observation, gives the same values per
  position (first / second to act) and averaged, for random policies.

Sampled evaluation (bluffjax.examples.kuhn_leduc_eval):
- play_vs_random plays exactly num_games complete hands, the learner's
  position alternating from a random offset, in the seat that has it.
- The learner acts exactly at its seat: an always-bet Kuhn learner's hands
  look as they must from either position.
- Uniform random play against itself gives return ~0 and win rate ~ loss
  rate; for random networks (sampled softmax and greedy Q, the scripts'
  policies) the mean return per hand matches value_vs_random of their tables
  (policy_array_from_network / policy_array_from_qnetwork) within 4.5
  standard errors, per position and averaged.
- vs_random_metrics' names and values, maybe_vs_random_metrics only playing
  when due, and training doesn't depend on the evaluation (compare on / off
  give the same networks).
- In the training scripts (a spy on play_vs_random), every evaluation plays
  exactly compare_episodes complete hands per policy (NFSP: average policy and
  BR), alternating the learner's position, every compare_interval updates and
  after the last one, with the networks of that update.
"""

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.examples import kuhn_leduc_eval as kle
from bluffjax.examples.kuhn import kuhn_ppo, kuhn_pqn, kuhn_pqn_nfsp
from bluffjax.examples.leduc import leduc_ppo_nfsp
from bluffjax.examples.run_utils import ResultsLog
from bluffjax.utils.game_utils import kuhn_exploitability as ke
from bluffjax.utils.game_utils import leduc_exploitability as le

PASS, BET = 0, 1
FOLD, CALL, RAISE = 0, 1, 2
METRIC_PREFIXES = (
    "win_rate_",
    "loss_rate_",
    "draw_rate_",
    "win_rate_decided_",
    "avg_return_",
    "avg_episode_length_",
)


@pytest.fixture(scope="module")
def kuhn():
    return make("kuhn_poker")


@pytest.fixture(scope="module")
def leduc():
    return make("leduc_holdem")


# --------------------------------------------------------------- policies
def obs_params(seed: int, obs_dim: int, num_actions: int, scale: float = 1.5):
    """Weights of a random function of the env observation."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    return (
        scale * jax.random.normal(k1, (obs_dim, 16)),
        scale * jax.random.normal(k2, (16, num_actions)),
    )


def obs_apply(params, obs):
    """Logits (or Q-values) of the observation."""
    w1, w2 = params
    return jnp.tanh(obs @ w1) @ w2


def kuhn_table(params, greedy=False) -> np.ndarray:
    fn = ke.policy_array_from_qnetwork if greedy else ke.policy_array_from_network
    return np.asarray(fn(obs_apply, params), dtype=np.float64)


def leduc_table(params, greedy=False) -> np.ndarray:
    if greedy:
        table = le.policy_array_from_qnetwork(obs_apply, params, le._INFOSET_LIST)
    else:
        table = le.policy_array_from_network(
            lambda p, o: (obs_apply(p, o), None), params, le._INFOSET_LIST
        )
    return np.asarray(table, dtype=np.float64)


def kuhn_values_per_position(table: np.ndarray) -> np.ndarray:
    """Exact return of the table's policy against uniform as first / second."""
    policy = ke.policy_from_array(table)
    uniform = ke.uniform_random_policy()
    return np.array(
        [ke.expected_returns(policy, uniform)[0], ke.expected_returns(uniform, policy)[1]]
    )


def leduc_values_per_position(table: np.ndarray) -> np.ndarray:
    policy = le._policy_from_array_callable(table, le._KEY_TO_IDX)
    return np.array(
        [
            le.expected_returns(policy, le.uniform_policy)[0],
            le.expected_returns(le.uniform_policy, policy)[1],
        ]
    )


def best_response_to_uniform_kuhn() -> np.ndarray:
    """(12, 2) table: each player's best response to uniform play."""
    actions = {}
    for player in (0, 1):
        _, br = ke.best_response_value_infoset(player, ke.uniform_random_policy())
        actions.update(br)
    assert set(actions) == set(ke.INFOSETS)
    return np.eye(2)[[actions[k] for k in ke.INFOSETS]]


def best_response_to_uniform_leduc() -> np.ndarray:
    root = le.initial_state()
    solvers = [le.BestResponseSolver(root, p, le.uniform_policy) for p in (0, 1)]
    table = np.zeros((len(le._INFOSET_LIST), 3))
    for i, (key, _) in enumerate(le._INFOSET_LIST):
        table[i, solvers[key[0]].best_response_action(key)] = 1.0
    return table


# ------------------------------------------------------------ exact values
def test_kuhn_uniform_and_best_response():
    uniform = np.full((12, 2), 0.5)
    assert ke.value_vs_random_from_policy_array(uniform) == 0.0
    assert ke.value_vs_random(ke.uniform_random_policy()) == 0.0
    # uniform against itself: zero-sum, each seat's value is the other's loss
    first, second = ke.expected_returns(*[ke.uniform_random_policy()] * 2)
    assert first == pytest.approx(-second, abs=1e-15)
    br = best_response_to_uniform_kuhn()
    expl = ke.exploitability_from_policy_array(uniform)
    assert expl == pytest.approx(0.458333, abs=1e-6)
    assert ke.value_vs_random_from_policy_array(br) == pytest.approx(expl, abs=1e-12)
    # nothing does better than the best response
    rs = np.random.RandomState(0)
    for _ in range(20):
        p = rs.rand(12)
        table = np.stack([1 - p, p], axis=-1)
        assert ke.value_vs_random_from_policy_array(table) < expl


@pytest.mark.parametrize("alpha", [0.0, 0.05, 0.1, 0.2, 0.25, 1.0 / 3.0])
def test_kuhn_equilibrium_value(alpha):
    """By hand, with cards J < Q < K, the equilibrium nash_equilibrium(alpha)
    against a player that passes / bets (and folds / calls) with probability
    1/2 each, over the 6 equally likely deals:

    As first player (bets J with alpha, K with 3 alpha, never Q; calls a bet
    after passing with K, with Q w.p. 1/3 + alpha, never with J). A bet earns
    1/2 * (+1, fold) + 1/2 * (+-2, call); a pass earns 1/2 * (+-1, check back)
    + 1/2 * (call: +-2, fold: -1).
        J (both deals): bet -1/2, pass -1                 -> -1 + alpha / 2
        Q vs J: pass, 1/2 * 1 + 1/2 * (2 c - (1 - c)), c = 1/3 + alpha
                                                          -> (1 + 3 alpha) / 2
        Q vs K: pass, 1/2 * (-1) + 1/2 * (-2 c - (1 - c)) -> -7/6 - alpha / 2
        K (both deals): bet 3/2, pass 3/2                 -> 3/2
    Sum (1/3 + 2 alpha) / 6 = 1/18 + alpha / 3.
    As second player (after a pass bets J w.p. 1/3, never Q, always K; calls
    a bet never with J, w.p. 1/3 with Q, always with K), facing a bet w.p.
    1/2 and a pass w.p. 1/2 (after its own bet the opponent folds or calls):
        J (both deals): 1/2 * (-1) + 1/2 * (1/3 * (-1/2) + 2/3 * (-1)) -> -11/12
        Q vs J: 1/2 * (1/3 * 2 - 2/3) + 1/2 * 1                    -> 1/2
        Q vs K: 1/2 * (1/3 * (-2) - 2/3) + 1/2 * (-1)              -> -7/6
        K (both deals): 1/2 * 2 + 1/2 * (1/2 * 1 + 1/2 * 2)          -> 7/4
    Sum 1 / 6. The average over both seats is 1/9 + alpha / 6.
    """
    table = np.stack([ke.nash_equilibrium(alpha)(k) for k in ke.INFOSETS])
    first, second = kuhn_values_per_position(table)
    assert first == pytest.approx(1 / 18 + alpha / 3, abs=1e-12)
    assert second == pytest.approx(1 / 6, abs=1e-12)
    assert ke.value_vs_random_from_policy_array(table) == pytest.approx(
        1 / 9 + alpha / 6, abs=1e-12
    )
    assert ke.value_vs_random(ke.nash_equilibrium(alpha)) == pytest.approx(
        1 / 9 + alpha / 6, abs=1e-12
    )


def test_leduc_uniform_and_best_response():
    num = len(le._INFOSET_LIST)
    for uniform in (np.ones((num, 3)), np.zeros((num, 3))):  # rows renormalised
        assert le.value_vs_random_from_policy_array(uniform, le._KEY_TO_IDX) == 0.0
    first, second = le.expected_returns(le.uniform_policy, le.uniform_policy)
    assert first == pytest.approx(-second, abs=1e-12)
    br = best_response_to_uniform_leduc()
    value = le.value_vs_random_from_policy_array(br, le._KEY_TO_IDX)
    assert value == pytest.approx(le.exploitability_uniform(), abs=1e-9)
    assert value == pytest.approx(2.373611111111111, abs=1e-9)
    rs = np.random.default_rng(0)
    for _ in range(3):
        table = rs.dirichlet(np.ones(3), size=num)
        assert le.value_vs_random_from_policy_array(table, le._KEY_TO_IDX) < value


def test_leduc_fast_value_is_the_traversal():
    num = len(le._INFOSET_LIST)
    rs = np.random.default_rng(1)
    tables = [
        np.ones((num, 3)),
        best_response_to_uniform_leduc(),
        rs.dirichlet(np.ones(3), size=num),
        rs.dirichlet(0.2 * np.ones(3), size=num) * rs.integers(1, 5, size=(num, 1)),
        np.eye(3)[rs.integers(0, 3, size=num)],  # illegal folds: uniform rows
    ]
    for table in tables:
        reference = le.value_vs_random(le._policy_from_array_callable(table, le._KEY_TO_IDX))
        fast = le.value_vs_random_from_policy_array(table, le._KEY_TO_IDX)
        assert fast == pytest.approx(reference, abs=1e-12)
        # the rows in another order, with the matching key_to_idx
        perm = rs.permutation(num)
        inverse = np.argsort(perm)  # infoset i is row inverse[i] of table[perm]
        key_to_idx = {key: int(inverse[i]) for i, (key, _) in enumerate(le._INFOSET_LIST)}
        assert le.value_vs_random_from_policy_array(
            table[perm], key_to_idx
        ) == pytest.approx(fast, abs=1e-12)
    # the histories partition the game: under uniform play their probabilities
    # sum to 1, and every one pays a zero-sum return
    h = le._terminal_histories()
    reach = h["chance"] * (1.0 / h["num_legal"]).prod(axis=-1)
    assert reach.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_array_equal(h["returns"].sum(-1), 0.0)


# -------------------------------------------------- brute force in the env
def _kuhn_env_values(env, table):
    """Exact return per hand of the table's policy (row picked from the env
    observation, as in the scripts) against uniform play, as first / second
    to act, by walking every deal, start player and action of the env."""
    step = jax.jit(env.step_env)
    obs_fn = jax.jit(env.obs_from_state)
    rows = {o.tobytes(): i for i, o in enumerate(ke.INFOSET_OBS)}

    def walk(state, learner_seat):
        player = int(state.current_player_idx)
        if player == learner_seat:
            probs = table[rows[np.asarray(obs_fn(state)).tobytes()]]
        else:
            probs = np.array([0.5, 0.5])
        value = 0.0
        for a in (PASS, BET):
            nxt, _, reward, _, done, _ = step(jax.random.PRNGKey(0), state, jnp.int32(a))
            v = float(reward[learner_seat])
            if not bool(done):
                v += walk(nxt, learner_seat)
            value += probs[a] * v
        return value

    values = np.zeros(2)
    base, _ = env.reset(jax.random.PRNGKey(0))
    for hands in itertools.permutations(range(3), 2):  # cards of seats 0 and 1
        for start in (0, 1):
            state = base.replace(
                agent_hands=jnp.array(hands, jnp.int32),
                start_player_idx=jnp.int32(start),
                current_player_idx=jnp.int32(start),
            )
            for position in (0, 1):
                values[position] += walk(state, (start + position) % 2) / 12
    return values


def test_kuhn_env_brute_force(kuhn):
    tables = [np.full((12, 2), 0.5), best_response_to_uniform_kuhn()]
    tables += [kuhn_table(obs_params(s, 9, 2)) for s in range(2)]
    tables += [kuhn_table(obs_params(5, 9, 2), greedy=True)]
    for table in tables:
        env_values = _kuhn_env_values(kuhn, table)
        np.testing.assert_allclose(env_values, kuhn_values_per_position(table), atol=1e-6)
        assert ke.value_vs_random_from_policy_array(table) == pytest.approx(
            env_values.mean(), abs=1e-6
        )


def _leduc_env_values(env, table):
    """As _kuhn_env_values for Leduc: every ordered (card of seat 0, card of
    seat 1, public card) and start player, both learner positions at once."""
    step = jax.jit(env.step_env)
    obs_fn = jax.jit(env.obs_from_state)
    mask_fn = jax.jit(env.get_avail_actions)
    rows = {}
    for i, (key, _) in enumerate(le._INFOSET_LIST):
        rows.setdefault(le._infoset_key_to_obs(key).tobytes(), i)

    def walk(state, learner_seats):
        """Values (2,) of the learner in seat learner_seats[p], p = position."""
        player = int(state.current_player_idx)
        legal = np.asarray(mask_fn(state))
        uniform = legal / legal.sum()
        row = table[rows[np.asarray(obs_fn(state)).tobytes()]] * legal
        row = row / row.sum()
        probs = [row if seat == player else uniform for seat in learner_seats]
        value = np.zeros(2)
        for a in np.flatnonzero(legal):
            nxt, _, reward, _, done, _ = step(jax.random.PRNGKey(0), state, jnp.int32(a))
            v = np.asarray(reward)[list(learner_seats)].astype(np.float64)
            if not bool(done):
                v = v + walk(nxt, learner_seats)
            value += np.array([probs[0][a], probs[1][a]]) * v
        return value

    base, _ = env.reset(jax.random.PRNGKey(0))
    values = np.zeros(2)
    deals = list(itertools.permutations(range(6), 3))
    for c0, c1, public in deals:
        rest = [c for c in range(6) if c not in (c0, c1, public)]
        for start in (0, 1):
            state = base.replace(
                agent_cards=jnp.array([c0, c1], jnp.int32),
                shuffled_deck=jnp.array([c0, c1, public, *rest], jnp.int32),
                start_player_idx=jnp.int32(start),
                current_player_idx=jnp.int32(start),
            )
            values += walk(state, (start, 1 - start)) / (2 * len(deals))
    return values


def test_leduc_env_brute_force(leduc):
    table = leduc_table(obs_params(0, 24, 3))
    env_values = _leduc_env_values(leduc, table)
    np.testing.assert_allclose(env_values, leduc_values_per_position(table), atol=1e-6)
    assert le.value_vs_random_from_policy_array(table, le._KEY_TO_IDX) == pytest.approx(
        env_values.mean(), abs=1e-6
    )


# ------------------------------------------------------ sampled evaluation
@pytest.mark.parametrize("game", ["kuhn", "leduc"])
@pytest.mark.parametrize("num_games", [7, 10])
def test_plays_complete_games_with_rotating_seats(request, game, num_games):
    env = request.getfixturevalue(game)
    obs_dim = 9 if game == "kuhn" else 24
    act = kle.softmax_policy(obs_apply)
    params = obs_params(1, obs_dim, env.num_actions)
    play = jax.jit(functools.partial(kle.play_vs_random, env, act), static_argnums=1)
    offsets = set()
    for seed in range(10):
        rec = jax.tree_util.tree_map(
            np.asarray, play(params, num_games, jax.random.PRNGKey(seed))
        )
        assert rec.returns.shape == (num_games, 2) and rec.length.shape == (num_games,)
        assert rec.done.all()  # every game is complete
        np.testing.assert_array_equal(rec.returns.sum(-1), 0.0)  # zero-sum, all paid
        assert (rec.length >= 2).all() and (rec.length <= (3 if game == "kuhn" else 8)).all()
        offset = rec.learner_position[0]
        offsets.add(int(offset))
        np.testing.assert_array_equal(
            rec.learner_position, (offset + np.arange(num_games)) % 2
        )
        counts = np.bincount(rec.learner_position, minlength=2)
        assert abs(counts[0] - counts[1]) == num_games % 2
    assert offsets == {0, 1}

    result = jax.jit(
        functools.partial(kle.evaluate_vs_random, env, act), static_argnums=1
    )(params, num_games, jax.random.PRNGKey(0))
    assert result.ret.shape == (num_games,)
    np.testing.assert_array_equal(result.win + result.loss + result.draw, 1.0)


def test_learner_plays_its_seat(kuhn):
    """An always-bet learner: as first to act every hand is bet then a reply
    (2 steps, +1 or +-2); as second the opponent either bets (the learner
    calls: 2 steps, +-2) or passes (the learner bets: 3 steps, +1 or +-2)."""

    def always_bet(params, obs, mask, rng):
        return jnp.full(obs.shape[:-1], BET, jnp.int32)

    rec = jax.tree_util.tree_map(
        np.asarray, jax.jit(lambda k: kle.play_vs_random(kuhn, always_bet, None, 4000, k))(
            jax.random.PRNGKey(0)
        )
    )
    ret = rec.returns[np.arange(4000), rec.learner_seat]
    first = rec.learner_position == 0
    assert (rec.length[first] == 2).all()
    assert set(np.unique(ret[first])) == {-2.0, 1.0, 2.0}
    second = ~first
    assert set(np.unique(rec.length[second])) == {2, 3}
    assert set(np.unique(ret[second & (rec.length == 2)])) == {-2.0, 2.0}
    assert set(np.unique(ret[second & (rec.length == 3)])) == {-2.0, 1.0, 2.0}
    # and the mean returns per position are the exact ones
    table = np.tile([0.0, 1.0], (12, 1))
    exact = kuhn_values_per_position(table)
    for position, mask in ((0, first), (1, second)):
        se = ret[mask].std(ddof=1) / np.sqrt(mask.sum())
        assert abs(ret[mask].mean() - exact[position]) < 4.5 * se


@pytest.mark.parametrize("game", ["kuhn", "leduc"])
def test_random_against_random(request, game):
    env = request.getfixturevalue(game)
    n = 20000
    result = jax.tree_util.tree_map(
        np.asarray,
        jax.jit(lambda k: kle.evaluate_vs_random(env, kle.random_policy, None, n, k))(
            jax.random.PRNGKey(1)
        ),
    )
    se = result.ret.std(ddof=1) / np.sqrt(n)
    assert abs(result.ret.mean()) < 4.5 * se
    diff = result.win - result.loss
    assert abs(diff.mean()) < 4.5 * diff.std(ddof=1) / np.sqrt(n)
    if game == "kuhn":
        assert result.draw.sum() == 0  # no ties in Kuhn
    else:
        assert 0.0 < result.draw.mean() < 0.3  # split pots


@pytest.mark.parametrize("game", ["kuhn", "leduc"])
@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
def test_monte_carlo_matches_exact_value(request, game, greedy, seed):
    """The scripts' sampled policies (softmax, greedy Q) on random networks:
    the mean return per hand of play_vs_random, per position and over all
    hands, is the exact value of the network's table within 4.5 SE."""
    env = request.getfixturevalue(game)
    obs_dim = 9 if game == "kuhn" else 24
    params = obs_params(seed, obs_dim, env.num_actions)
    if game == "kuhn":
        table = kuhn_table(params, greedy)
        exact = kuhn_values_per_position(table)
        assert ke.value_vs_random_from_policy_array(table) == pytest.approx(
            exact.mean(), abs=1e-12
        )
    else:
        table = leduc_table(params, greedy)
        exact = leduc_values_per_position(table)
        assert le.value_vs_random_from_policy_array(
            table, le._KEY_TO_IDX
        ) == pytest.approx(exact.mean(), abs=1e-12)
    act = (kle.greedy_policy if greedy else kle.softmax_policy)(obs_apply)
    n = 30000
    rec = jax.tree_util.tree_map(
        np.asarray,
        jax.jit(lambda k: kle.play_vs_random(env, act, params, n, k))(
            jax.random.PRNGKey(100 + seed)
        ),
    )
    ret = rec.returns[np.arange(n), rec.learner_seat]
    for position in (0, 1):
        r = ret[rec.learner_position == position]
        assert abs(r.mean() - exact[position]) < 4.5 * r.std(ddof=1) / np.sqrt(len(r))
    # positions are balanced (n even), so the mean estimates the average
    se = np.sqrt(
        sum(ret[rec.learner_position == p].var(ddof=1) for p in (0, 1)) / (2 * n)
    )
    assert abs(ret.mean() - exact.mean()) < 4.5 * se


def test_metrics_names_and_due(kuhn):
    act = kle.softmax_policy(obs_apply)
    params = obs_params(0, 9, 2)
    policies = {"avg_vs_random": (act, params), "br_vs_random": (act, params)}
    n = 10

    @jax.jit
    def run(rng, due):
        return kle.maybe_vs_random_metrics(kuhn, policies, n, rng, due)

    metrics = {k: float(v) for k, v in run(jax.random.PRNGKey(0), True).items()}
    assert set(metrics) == {
        p + name for p in METRIC_PREFIXES for name in policies
    }
    assert all(kle.is_sampled_metric(k) for k in metrics)
    for name in policies:
        win, loss, draw = (metrics[f"{k}_rate_{name}"] for k in ("win", "loss", "draw"))
        assert win + loss + draw == pytest.approx(1.0)
        assert draw == 0.0
        assert (win * n) == pytest.approx(round(win * n))  # n games
        assert metrics[f"win_rate_decided_{name}"] == pytest.approx(win)
        assert 2.0 <= metrics[f"avg_episode_length_{name}"] <= 3.0
        assert -2.0 <= metrics[f"avg_return_{name}"] <= 2.0
    zeros = run(jax.random.PRNGKey(0), False)
    assert all(float(v) == 0.0 for v in zeros.values())
    for key in ("value_vs_random", "value_avg_vs_random", "exploitability"):
        assert not kle.is_sampled_metric(key)
    # the key is a stream of its own
    rng = jax.random.PRNGKey(3)
    key = kle.eval_key(rng)
    for n_split in (2, 4, 5, 6, 1024):
        for k in jax.random.split(rng, n_split):
            assert not np.array_equal(np.asarray(k), np.asarray(key))


def _small_config(**extra):
    config = dict(
        num_envs=16, num_steps_per_env_per_update=8, num_epochs=1,
        num_minibatches=2, lr=5e-3, anneal_lr=True, gamma=0.99,
        max_grad_norm=0.5, fc_dim_size=16, eval_interval=2,
        compare=True, compare_episodes=4, compare_interval=1,
        env_kwargs={}, num_update_steps=3,
    )
    config.update(extra)
    config["num_gradient_steps"] = 3 * config["num_epochs"] * config["num_minibatches"]
    return config


@pytest.mark.parametrize("name", ["kuhn_ppo", "kuhn_pqn_nfsp"])
def test_training_does_not_depend_on_the_evaluation(monkeypatch, tmp_path, name):
    mod = {"kuhn_ppo": kuhn_ppo, "kuhn_pqn_nfsp": kuhn_pqn_nfsp}[name]
    if name == "kuhn_ppo":
        extra = dict(gae_lambda=0.95, clip_eps=0.2, vf_clip=0.2, ent_coef=0.0, vf_coef=1.0)
    else:
        extra = dict(
            start_e=1.0, end_e=0.05, exploration_fraction=0.5, q_lambda=0.9,
            anticipatory_eta=0.3, sl_reservoir_capacity=200, sl_batch_size=16,
            sl_num_steps_per_update=2, sl_lr=3e-3,
        )
    monkeypatch.setattr(mod, "RESULTS_LOG", ResultsLog(str(tmp_path)))
    finals = []
    for compare in (True, False):
        train = mod.make_train(_small_config(compare=compare, **extra))
        finals.append(jax.jit(train)(jax.random.PRNGKey(0), jnp.int32(0)))
    jax.effects_barrier()
    attrs = ["train_state"] if name == "kuhn_ppo" else ["avg_train_state", "br_train_state"]
    for attr in attrs:
        a, b = (getattr(f, attr).params for f in finals)
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
            # equal up to the rounding of differently compiled programs (a
            # different random stream would move them by far more)
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-6)


PPO_KEYS = dict(gae_lambda=0.95, clip_eps=0.2, vf_clip=0.2, ent_coef=0.0, vf_coef=1.0)
PQN_KEYS = dict(start_e=1.0, end_e=0.05, exploration_fraction=0.5, q_lambda=0.9)
NFSP_KEYS = dict(
    anticipatory_eta=0.3, sl_reservoir_capacity=200, sl_batch_size=16,
    sl_num_steps_per_update=2, sl_lr=3e-3,
)


@pytest.mark.parametrize(
    "mod, extra, num_policies",
    [
        (kuhn_ppo, PPO_KEYS, 1),
        (kuhn_pqn, PQN_KEYS, 1),
        (kuhn_pqn_nfsp, {**PQN_KEYS, **NFSP_KEYS}, 2),
        (leduc_ppo_nfsp, {**PPO_KEYS, **NFSP_KEYS}, 2),
    ],
    ids=["kuhn_ppo", "kuhn_pqn", "kuhn_pqn_nfsp", "leduc_ppo_nfsp"],
)
def test_scripts_play_complete_games(monkeypatch, tmp_path, mod, extra, num_policies):
    calls = []
    play = kle.play_vs_random

    def spy(env, learner, learner_params, num_games, rng):
        record = play(env, learner, learner_params, num_games, rng)
        jax.debug.callback(
            lambda *a: calls.append([np.asarray(x) for x in a]),
            record.done, record.length, record.learner_position,
        )
        return record

    monkeypatch.setattr(kle, "play_vs_random", spy)
    rows = []

    class Recorder:
        def log(self, metrics):
            rows.append(metrics)

    monkeypatch.setattr(mod, "RESULTS_LOG", ResultsLog(str(tmp_path), [Recorder()]))
    config = _small_config(
        num_update_steps=5, compare_episodes=6, compare_interval=3, **extra
    )
    config["num_gradient_steps"] = 5 * config["num_epochs"] * config["num_minibatches"]
    jax.block_until_ready(jax.jit(mod.make_train(config))(jax.random.PRNGKey(0), jnp.int32(0)))
    jax.effects_barrier()
    evaluated = [0, 3, 4]  # compare_interval 3 and the last update
    assert len(calls) == num_policies * len(evaluated)
    for done, length, position in calls:
        assert done.shape == (6,) and done.all()  # 6 complete hands
        assert (length >= 2).all()
        offset = position[0]
        np.testing.assert_array_equal(position, (offset + np.arange(6)) % 2)
    assert len(rows) == 5
    for u, row in enumerate(rows):
        sampled = [k for k in row if kle.is_sampled_metric(k)]
        assert len(sampled) == (6 * num_policies if u in evaluated else 0)
