"""Exact Leduc Hold'em metrics (bluffjax.metrics.leduc_holdem).

- Hand-made deterministic policies whose metrics are derived by hand below.
- A brute-force reference that walks the game model's tree deal by deal
  (written independently of the vectorised tables) agrees to rounding for
  random policies, in self-play and head to head, total and per seat.
- Hands sampled from the env (make("leduc_holdem")) with fixed stochastic
  policies, counted from the env's states (the cards it dealt, the public card
  shuffled_deck[2] it will show, its hand ranking and its legal actions), give
  the same rates and counts per hand within a confidence interval. This checks
  the mapping between the policy tables, the information sets and the env,
  the random first player and seat mapping, and that "would lose" uses the
  public card the env actually deals.

Deal probabilities used in the hand derivations (6 cards, 2 per rank; a
player's card, the opponent's card and the public card are a uniformly random
ordered triple of distinct cards):
- P(the opponent holds the same rank) = 1/5, and then nobody can pair: a tie.
  With different ranks there is never a tie, so by symmetry
  P(would lose) = P(would win) = (1 - 1/5) / 2 = 2/5.
- P(own card pairs the public card) = P(the opponent doesn't hold the twin)
  * P(the public card is the twin) = 4/5 * 1/4 = 1/5. Both players can't pair.
- A Jack loses unless it pairs or ties: P(lose | J) = 4/5 * 3/4 = 3/5, and
  P(lose | J, the opponent holds no Jack) = 3/4.
"""

import itertools
import time
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.environments.leduc_holdem.leduc_holdem import _rank_hand
from bluffjax.metrics import leduc_holdem as lm
from bluffjax.utils.game_utils import leduc_exploitability as leduc

FOLD, CALL, RAISE = 0, 1, 2
NAN = float("nan")


# ------------------------------------------------------------------ policies
def table_policy(choose) -> np.ndarray:
    """(num_infosets, 3) deterministic policy: choose(info) -> action, where
    info has the infoset's position, card, public card (-1 in round 1),
    round and legal actions."""
    policy = np.zeros((len(leduc._INFOSET_LIST), 3))
    for i, (key, legal) in enumerate(leduc._INFOSET_LIST):
        info = dict(
            position=key[0], card=key[1], public=key[2], round=key[6], legal=legal
        )
        action = choose(info)
        assert action in legal
        policy[i, action] = 1.0
    return policy


def always_raise(info):
    return RAISE if RAISE in info["legal"] else CALL


def always_call(info):
    return CALL


def raise_or_fold(info):
    """Bet when not facing a bet, fold when facing one."""
    return FOLD if FOLD in info["legal"] else RAISE


def jack_bluffer(info):
    """In round 1 bet a Jack when not facing a bet; otherwise call."""
    if info["round"] == 1 and FOLD not in info["legal"] and info["card"] // 2 == 0:
        return RAISE
    return CALL


def pair_raiser(info):
    """In round 2 raise whenever legal with a pair; otherwise call."""
    paired = info["round"] == 2 and info["card"] // 2 == info["public"] // 2
    return RAISE if paired and RAISE in info["legal"] else CALL


def random_policy(seed: int, concentration: float = 1.0) -> np.ndarray:
    """Random rows (Dirichlet over the legal actions) at every infoset."""
    rng = np.random.default_rng(seed)
    policy = rng.dirichlet(np.full(3, concentration), size=len(leduc._INFOSET_LIST))
    for i, (_, legal) in enumerate(leduc._INFOSET_LIST):
        policy[i, [a for a in range(3) if a not in legal]] = 0.0
    return policy


def assert_metrics(got: dict, expected: dict, suffix: str = "") -> None:
    for name, want in expected.items():
        value = got[name + suffix]
        if np.isnan(want):
            assert np.isnan(value), (name + suffix, value)
        else:
            assert value == pytest.approx(want, abs=1e-12), (name + suffix, value, want)


# --------------------------------------------------------- hand derivations
def test_names():
    assert lm.TOTAL_NAMES == (
        "bluff_rate", "n_aggressive_per_hand",
        "bluff_success", "n_bluffs_per_hand",
        "correct_fold_rate", "n_folds_per_hand",
        "pair_raise_rate", "n_paired_decisions_per_hand",
        "unpaired_raise_rate", "n_unpaired_decisions_per_hand",
    )
    got = lm.policy_metrics(random_policy(0))
    assert set(got) == {n + s for n in lm.TOTAL_NAMES for s in ("", "_p0", "_p1")}
    logged = lm.log_metrics("game", random_policy(0))
    assert list(logged) == [f"game/{n}" for n in lm.TOTAL_NAMES]


def test_always_raise():
    """Every hand goes raise, re-raise, call in both rounds: each position
    raises once per round whatever its cards, so a raise is a bluff with
    P(would lose) = 2/5, nobody folds, and both round-2 first raises are
    paired with P = 1/5."""
    got = lm.policy_metrics(table_policy(always_raise))
    expected = dict(
        bluff_rate=2 / 5, n_aggressive_per_hand=4.0,
        bluff_success=0.0, n_bluffs_per_hand=8 / 5,
        correct_fold_rate=NAN, n_folds_per_hand=0.0,
        pair_raise_rate=1.0, n_paired_decisions_per_hand=2 / 5,
        unpaired_raise_rate=1.0, n_unpaired_decisions_per_hand=8 / 5,
    )
    assert_metrics(got, expected)
    # each seat is in each position half of the time: half of every count
    for suffix in ("_p0", "_p1"):
        assert_metrics(
            got,
            {k: v / 2 if k.startswith("n_") else v for k, v in expected.items()},
            suffix,
        )


def test_always_call():
    """No raises (bluff rates NaN), no folds (NaN); round 2 is check-check."""
    got = lm.policy_metrics(table_policy(always_call))
    assert_metrics(got, dict(
        bluff_rate=NAN, n_aggressive_per_hand=0.0,
        bluff_success=NAN, n_bluffs_per_hand=0.0,
        correct_fold_rate=NAN, n_folds_per_hand=0.0,
        pair_raise_rate=0.0, n_paired_decisions_per_hand=2 / 5,
        unpaired_raise_rate=0.0, n_unpaired_decisions_per_hand=8 / 5,
    ))


def test_raise_or_fold():
    """The first player bets and the second folds: every bluff succeeds, the
    folder would have lost with P = 2/5, and round 2 is never reached."""
    got = lm.policy_metrics(table_policy(raise_or_fold))
    expected = dict(
        bluff_rate=2 / 5, n_aggressive_per_hand=1.0,
        bluff_success=1.0, n_bluffs_per_hand=2 / 5,
        correct_fold_rate=2 / 5, n_folds_per_hand=1.0,
        pair_raise_rate=NAN, n_paired_decisions_per_hand=0.0,
        unpaired_raise_rate=NAN, n_unpaired_decisions_per_hand=0.0,
    )
    assert_metrics(got, expected)
    for suffix in ("_p0", "_p1"):
        assert_metrics(
            got,
            {k: v / 2 if k.startswith("n_") else v for k, v in expected.items()},
            suffix,
        )


def test_jack_bluffer_uses_the_public_card():
    """The first player bets a Jack (P = 1/3), losing with P = 3/5; otherwise
    the second player bets a Jack (P = 2/3 * 2/5 = 4/15), losing with P = 3/4:
    bluffs 1/5 + 1/5 out of 1/3 + 4/15 = 3/5 raises, bluff_rate = 2/3. With
    the public card ignored a Jack would always lose unless tied: 8/9."""
    got = lm.policy_metrics(table_policy(jack_bluffer))
    assert_metrics(got, dict(
        bluff_rate=2 / 3, n_aggressive_per_hand=3 / 5,
        bluff_success=0.0, n_bluffs_per_hand=2 / 5,
        correct_fold_rate=NAN, n_folds_per_hand=0.0,
        pair_raise_rate=0.0, n_paired_decisions_per_hand=2 / 5,
        unpaired_raise_rate=0.0, n_unpaired_decisions_per_hand=8 / 5,
    ))


def test_pair_raiser():
    """A paired player raises once and the other calls. A pair can't lose, so
    there are no bluffs (bluff_success is NaN). Round-2 decisions with raising
    legal: paired 1/5 + 1/5; unpaired 1 when the first player pairs, 2 when
    the second does (check, raise, call) and 2 when nobody does (P = 3/5):
    1/5 + 2/5 + 6/5 = 9/5."""
    got = lm.policy_metrics(table_policy(pair_raiser))
    assert_metrics(got, dict(
        bluff_rate=0.0, n_aggressive_per_hand=2 / 5,
        bluff_success=NAN, n_bluffs_per_hand=0.0,
        correct_fold_rate=NAN, n_folds_per_hand=0.0,
        pair_raise_rate=1.0, n_paired_decisions_per_hand=2 / 5,
        unpaired_raise_rate=0.0, n_unpaired_decisions_per_hand=9 / 5,
    ))


def test_head_to_head_seats():
    """Seat 0 always raises, seat 1 raises or folds. When seat 0 opens it
    bets and seat 1 folds; when seat 1 opens it bets, seat 0 re-raises and
    seat 1 folds. Per hand: seat 0 raises once (always answered by a fold),
    seat 1 raises 1/2 (answered by a raise) and folds once; nobody's action
    depends on the cards, so every rate of losing hands is 2/5."""
    got = lm.policy_metrics(table_policy(always_raise), table_policy(raise_or_fold))
    assert_metrics(got, dict(
        bluff_rate=2 / 5, n_aggressive_per_hand=1.0,
        bluff_success=1.0, n_bluffs_per_hand=2 / 5,
        correct_fold_rate=NAN, n_folds_per_hand=0.0,
        pair_raise_rate=NAN, n_paired_decisions_per_hand=0.0,
    ), "_p0")
    assert_metrics(got, dict(
        bluff_rate=2 / 5, n_aggressive_per_hand=1 / 2,
        bluff_success=0.0, n_bluffs_per_hand=1 / 5,
        correct_fold_rate=2 / 5, n_folds_per_hand=1.0,
        pair_raise_rate=NAN, n_paired_decisions_per_hand=0.0,
    ), "_p1")
    assert_metrics(got, dict(
        bluff_rate=2 / 5, n_aggressive_per_hand=3 / 2,
        bluff_success=2 / 3, n_bluffs_per_hand=3 / 5,
        correct_fold_rate=2 / 5, n_folds_per_hand=1.0,
        unpaired_raise_rate=NAN, n_unpaired_decisions_per_hand=0.0,
    ))
    # swapping the policies swaps the seats
    swapped = lm.policy_metrics(table_policy(raise_or_fold), table_policy(always_raise))
    for name in lm.TOTAL_NAMES:
        for a, b in (("_p0", "_p1"), ("_p1", "_p0"), ("", "")):
            assert swapped[name + a] == pytest.approx(got[name + b], nan_ok=True)


def test_rows_are_renormalised_like_exploitability():
    """Mass on illegal actions is dropped and an all-zero row is uniform."""
    policy = random_policy(1)
    scaled = policy * np.random.default_rng(2).uniform(0.1, 5.0, (len(policy), 1))
    noisy = scaled.copy()
    for i, (_, legal) in enumerate(leduc._INFOSET_LIST):
        if FOLD not in legal:
            noisy[i, FOLD] = 7.0
    zeros = policy.copy()
    zeros[::7] = 0.0
    uniform = zeros.copy()
    for i in range(0, len(policy), 7):
        legal = list(leduc._INFOSET_LIST[i][1])
        uniform[i, legal] = 1.0 / len(legal)
    for a, b in ((policy, noisy), (uniform, zeros)):
        ga, gb = lm.policy_metrics(a), lm.policy_metrics(b)
        assert ga == pytest.approx(gb, abs=1e-12, nan_ok=True)
    with pytest.raises(ValueError):
        lm.policy_metrics(policy[:-1])


# ------------------------------------------------------- brute-force reference
def reference_metrics(policy_p0, policy_p1):
    """Walks the model's tree for every start seat and deal, accumulating the
    expected count of every event per seat, and the rates from them."""
    policies = [lm.normalize_policy(policy_p0), lm.normalize_policy(policy_p1)]
    counts = defaultdict(float)

    def hand_loses(own, opp, public):
        own_pair, opp_pair = own // 2 == public // 2, opp // 2 == public // 2
        if own_pair != opp_pair:
            return opp_pair
        return not own_pair and own // 2 < opp // 2

    def walk(state, prob, start, public, pending_bluffer):
        if state.is_terminal():
            return
        if state.is_chance_node():
            walk(state.child(public), prob, start, public, pending_bluffer)
            return
        pos = state.current_player
        seat = (start + pos) % 2
        pi = policies[seat][leduc._KEY_TO_IDX[state.info_state_key(pos)]]
        lose = hand_loses(state.private_cards[pos], state.private_cards[1 - pos], public)
        if pending_bluffer is not None:
            assert pending_bluffer == 1 - seat
            counts["n_successful_bluffs_per_hand", pending_bluffer] += prob * pi[FOLD]
        counts["n_aggressive_per_hand", seat] += prob * pi[RAISE]
        counts["n_bluffs_per_hand", seat] += prob * pi[RAISE] * lose
        counts["n_folds_per_hand", seat] += prob * pi[FOLD]
        counts["n_correct_folds_per_hand", seat] += prob * pi[FOLD] * lose
        if state.round_ == 2 and RAISE in state.legal_actions():
            kind = "paired" if state.private_cards[pos] // 2 == public // 2 else "unpaired"
            counts[f"n_{kind}_decisions_per_hand", seat] += prob
            counts[f"n_{kind}_raises_per_hand", seat] += prob * pi[RAISE]
        for action in state.legal_actions():
            bluffer = seat if action == RAISE and lose else None
            walk(state.child(action), prob * pi[action], start, public, bluffer)

    for start in (0, 1):
        for c0, c1, public in itertools.permutations(range(6), 3):
            root = leduc.initial_state().child(c0).child(c1)
            walk(root, 1 / 240, start, public, None)

    out = {}
    rates = {
        "bluff_rate": ("n_bluffs_per_hand", "n_aggressive_per_hand"),
        "bluff_success": ("n_successful_bluffs_per_hand", "n_bluffs_per_hand"),
        "correct_fold_rate": ("n_correct_folds_per_hand", "n_folds_per_hand"),
        "pair_raise_rate": ("n_paired_raises_per_hand", "n_paired_decisions_per_hand"),
        "unpaired_raise_rate": ("n_unpaired_raises_per_hand", "n_unpaired_decisions_per_hand"),
    }
    for suffix, seats in (("", (0, 1)), ("_p0", (0,)), ("_p1", (1,))):
        total = lambda name: sum(counts[name, s] for s in seats)
        for rate, (num, den) in rates.items():
            out[rate + suffix] = total(num) / total(den) if total(den) > 0 else NAN
            out[den + suffix] = total(den)
    return out


@pytest.mark.parametrize("seeds", [(3, 3), (4, 5), (6, 7)])
def test_matches_brute_force_reference(seeds):
    a = random_policy(seeds[0], concentration=0.5)
    b = random_policy(seeds[1], concentration=0.5)
    got = lm.policy_metrics(a, b) if seeds[0] != seeds[1] else lm.policy_metrics(a)
    want = reference_metrics(a, b)
    assert set(got) == set(want)
    for name in want:
        assert got[name] == pytest.approx(want[name], rel=1e-10, abs=1e-14), name
    if seeds[0] == seeds[1]:  # self-play: the seats are symmetric
        for name in lm.TOTAL_NAMES:
            scale = 0.5 if name.startswith("n_") else 1.0
            assert got[name + "_p0"] == pytest.approx(scale * got[name], rel=1e-10)
            assert got[name + "_p1"] == pytest.approx(scale * got[name], rel=1e-10)


def test_is_fast():
    a, b = random_policy(8), random_policy(9)
    lm.policy_metrics(a, b)  # tables are built on the first call
    start = time.perf_counter()
    for _ in range(20):
        lm.policy_metrics(a, b)
    per_call = (time.perf_counter() - start) / 20
    print(f"policy_metrics: {per_call * 1e3:.2f} ms per call")
    assert per_call < 0.05


# ------------------------------------------------------------ Monte Carlo
NUM_HANDS = 1 << 18
MAX_STEPS = 8  # 4 actions per round at most
EVENTS = (
    "n_aggressive_per_hand", "n_bluffs_per_hand", "n_successful_bluffs_per_hand",
    "n_folds_per_hand", "n_correct_folds_per_hand", "n_paired_decisions_per_hand",
    "n_paired_raises_per_hand", "n_unpaired_decisions_per_hand",
    "n_unpaired_raises_per_hand", "n_bluffs_public_ignored_per_hand",
)


def _obs_policy_params(seed):
    """Random weights of a policy that is a function of the env observation."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    return (
        1.5 * jax.random.normal(k1, (24, 16)),
        1.5 * jax.random.normal(k2, (16, 3)),
    )


def _obs_policy_logits(params, obs):
    w1, w2 = params
    return jnp.tanh(obs @ w1) @ w2


def _policy_table(params):
    """The obs policy at every infoset, via leduc_exploitability's helper."""
    return np.asarray(
        leduc.policy_array_from_network(
            lambda p, o: (_obs_policy_logits(p, o), None), params, leduc._INFOSET_LIST
        )
    )


@jax.jit
def _sample_hands(rng, params_p0, params_p1):
    """Plays NUM_HANDS hands in the env, seat s acting with params_ps.

    Returns per-hand event counts (NUM_HANDS, num_events, 2 seats) and the
    number of round-2 states whose public card isn't shuffled_deck[2].
    """
    env = make("leduc_holdem")
    params = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), params_p0, params_p1)

    def one_hand(rng):
        rng_reset, rng_play = jax.random.split(rng)
        state, obs = env.reset(rng_reset)
        public = state.shuffled_deck[2]  # what the env will show in round 2
        rank = _rank_hand(state.agent_cards, public)  # the env's showdown ranking
        lose = rank < rank[::-1]
        lose_public_ignored = state.agent_cards // 2 < state.agent_cards[::-1] // 2
        paired = state.agent_cards // 2 == public // 2

        def step(carry, rng):
            state, obs, alive, bluffer, counts, mismatch = carry
            rng_act, rng_step = jax.random.split(rng)
            seat = state.current_player_idx
            mask = env.get_avail_actions(state)
            seat_params = jax.tree_util.tree_map(lambda x: x[seat], params)
            logits = jnp.where(mask, _obs_policy_logits(seat_params, obs), -jnp.inf)
            action = jax.random.categorical(rng_act, logits)
            is_raise, is_fold = action == RAISE, action == FOLD
            option = (state.stage == 2) & mask[RAISE]
            seat_events = jnp.stack([
                is_raise,
                is_raise & lose[seat],
                jnp.zeros((), bool),
                is_fold,
                is_fold & lose[seat],
                option & paired[seat],
                option & paired[seat] & is_raise,
                option & ~paired[seat],
                option & ~paired[seat] & is_raise,
                is_raise & lose_public_ignored[seat],
            ])
            new = jnp.zeros((len(EVENTS), 2)).at[:, seat].set(
                (seat_events & alive).astype(jnp.float32)
            )
            # a successful bluff: the reply to the bluffer's raise is a fold
            success = alive & (bluffer >= 0) & is_fold
            new = new.at[2, jnp.maximum(bluffer, 0)].add(success.astype(jnp.float32))
            mismatch += alive & (state.stage == 2) & (state.public_card != public)
            next_state, next_obs, _, _, done, _ = env.step(rng_step, state, action)
            bluffer = jnp.where(alive & is_raise & lose[seat], seat, -1)
            carry = (next_state, next_obs, alive & ~done, bluffer, counts + new, mismatch)
            return carry, None

        init = (state, obs, jnp.bool_(True), jnp.int32(-1),
                jnp.zeros((len(EVENTS), 2)), jnp.int32(0))
        (_, _, alive, _, counts, mismatch), _ = jax.lax.scan(
            step, init, jax.random.split(rng_play, MAX_STEPS)
        )
        return counts, mismatch + alive  # every hand must be over

    return jax.vmap(one_hand)(jax.random.split(rng, NUM_HANDS))


def _ratio_ci(num, den):
    """Ratio of sums and its delta-method standard error over iid hands."""
    r = num.sum() / den.sum()
    resid = num - r * den
    se = np.sqrt(resid.var(ddof=1) / len(num)) / den.mean()
    return r, se


@pytest.mark.parametrize("head_to_head", [False, True])
def test_monte_carlo_in_the_env(head_to_head):
    params_p0 = _obs_policy_params(10)
    params_p1 = _obs_policy_params(11) if head_to_head else params_p0
    table_p0, table_p1 = _policy_table(params_p0), _policy_table(params_p1)
    exact = lm.policy_metrics(table_p0, table_p1 if head_to_head else None)
    counts, problems = _sample_hands(jax.random.PRNGKey(12), params_p0, params_p1)
    counts = np.asarray(counts, dtype=np.float64)
    assert int(np.asarray(problems).sum()) == 0
    idx = {name: i for i, name in enumerate(EVENTS)}
    rates = {
        "bluff_rate": ("n_bluffs_per_hand", "n_aggressive_per_hand"),
        "bluff_success": ("n_successful_bluffs_per_hand", "n_bluffs_per_hand"),
        "correct_fold_rate": ("n_correct_folds_per_hand", "n_folds_per_hand"),
        "pair_raise_rate": ("n_paired_raises_per_hand", "n_paired_decisions_per_hand"),
        "unpaired_raise_rate": ("n_unpaired_raises_per_hand", "n_unpaired_decisions_per_hand"),
    }
    checked = 0
    for suffix, seats in (("", [0, 1]), ("_p0", [0]), ("_p1", [1])):
        per_hand = {k: counts[:, i, seats].sum(axis=-1) for k, i in idx.items()}
        for rate, (num, den) in rates.items():
            r, se = _ratio_ci(per_hand[num], per_hand[den])
            assert se < 0.01, (rate + suffix, se)  # the check has power
            assert abs(r - exact[rate + suffix]) < 4.5 * se, (rate + suffix, r, exact[rate + suffix], se)
            mean = per_hand[den].mean()
            se_mean = per_hand[den].std(ddof=1) / np.sqrt(len(per_hand[den]))
            assert abs(mean - exact[den + suffix]) < 4.5 * se_mean, (den + suffix, mean)
            checked += 1
        if suffix == "":
            # the same check would fail if "would lose" ignored the public card
            r, se = _ratio_ci(
                per_hand["n_bluffs_public_ignored_per_hand"],
                per_hand["n_aggressive_per_hand"],
            )
            assert abs(r - exact["bluff_rate"]) > 8 * se, r
    assert checked == 15
    # every metric has plenty of events in this sample
    assert min(exact[d] for d in lm.DENOMINATORS.values()) > 0.05
