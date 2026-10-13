"""Goofspiel metrics (bluffjax/metrics/goofspiel.py):

- scripted rounds with hand-computed values: wins and their margin credited
  to the winner, burns (the env's tie rule, also with three players) credited
  to everybody, illegal bids played as the env plays them;
- the bid entropy of given distributions, and no credit when only one card is
  left;
- scripted games through env.step (auto-reset): Spearman's rho credited once
  per game and player at its last step, the carry reset across the game
  boundary;
- uniform-random play: every count against a brute-force recomputation from
  the recorded bids and prizes, and the rates against their closed-form
  values (rho 0, bid entropy 1, burn rate and overbid margin from the
  distribution of independent uniform bids).
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.environments.goofspiel.goofspiel import Goofspiel
from bluffjax.metrics import make_metrics, masked_totals
from bluffjax.metrics.goofspiel import Metrics, normalized_entropy, played_bids

N = 13
KEY = jax.random.PRNGKey(0)


def uniform_probs(env, state):
    mask = env.get_avail_actions(state)
    return mask / mask.sum(-1, keepdims=True)


def start(env, prizes=None):
    """A fresh game whose prize deck starts with `prizes` (card indices)."""
    state, _ = env.reset(KEY)
    if prizes is not None:
        rest = [c for c in range(N) if c not in prizes]
        state = state.replace(deck=jnp.array(list(prizes) + rest, dtype=jnp.int32))
    return state


def one_step(gm, env, state, bids, probs=None, carry=None):
    action = jnp.array(bids, dtype=jnp.int32)
    nxt, _, reward, _, done, info = env.step(KEY, state, action)
    probs = uniform_probs(env, state) if probs is None else probs
    carry = gm.init_carry() if carry is None else carry
    carry, counts = gm.step(carry, state, action, nxt, reward, done, info, probs)
    return nxt, reward, carry, {k: np.asarray(v) for k, v in counts.items()}


def test_make_metrics_and_summary_names():
    env = make("goofspiel", num_agents=2)
    gm = make_metrics("goofspiel", env)
    assert isinstance(gm, Metrics) and gm.needs_probs
    state = start(env)
    _, _, _, counts = one_step(gm, env, state, [3, 4])
    out = gm.summarize({k: jnp.asarray(v).sum(0) for k, v in counts.items()})
    assert set(gm.rate_names) <= set(out)
    assert set(out) == set(gm.rate_names) | {
        "n_player_games", "n_choices", "n_won_prizes", "n_player_rounds"
    }
    with pytest.raises(ValueError):
        gm.step(gm.init_carry(), state, jnp.array([3, 4]), state, jnp.zeros(2), False, {})


@pytest.mark.parametrize(
    "bids, won, margin, burned, reward_sign",
    [
        ([7, 3], [1, 0], [4, 0], 0, [1, -1]),  # card 8 beats card 4
        ([3, 7], [0, 1], [0, 4], 0, [-1, 1]),
        ([6, 6], [0, 0], [0, 0], 1, [0, 0]),  # tie: the prize is burned
        ([12, 0], [1, 0], [12, 0], 0, [1, -1]),
    ],
)
def test_round_two_players(bids, won, margin, burned, reward_sign):
    env = Goofspiel(num_agents=2)
    gm = Metrics(env)
    state = start(env, prizes=[4])  # the prize is the card 5
    _, reward, carry, counts = one_step(gm, env, state, bids)
    np.testing.assert_array_equal(np.sign(np.asarray(reward)), reward_sign)
    np.testing.assert_array_equal(counts["won_prizes"], won)
    np.testing.assert_array_equal(counts["margin_sum"], margin)
    np.testing.assert_array_equal(counts["burned"], [burned, burned])
    np.testing.assert_array_equal(counts["player_rounds"], [1, 1])
    # not the last round: no Spearman yet, (bid - prize)^2 in the carry
    np.testing.assert_array_equal(counts["player_games"], 0)
    np.testing.assert_array_equal(counts["spearman_sum"], 0)
    np.testing.assert_array_equal(carry.sum_sq_diff, (np.array(bids) - 4) ** 2)


@pytest.mark.parametrize(
    "bids, won, margin, burned",
    [
        ([6, 6, 9], [0, 0, 1], [0, 0, 3], 0),  # sole top 10 over the tied 7s
        ([9, 9, 2], [0, 0, 0], [0, 0, 0], 1),  # tied top: burned, nobody wins
        ([9, 9, 9], [0, 0, 0], [0, 0, 0], 1),
        ([0, 11, 10], [0, 1, 0], [0, 1, 0], 0),  # margin to the highest other bid
    ],
)
def test_round_three_players(bids, won, margin, burned):
    env = Goofspiel(num_agents=3)
    gm = Metrics(env)
    state = start(env, prizes=[2])
    _, reward, _, counts = one_step(gm, env, state, bids)
    # the env agrees: only a sole highest bid scores
    np.testing.assert_array_equal(np.asarray(reward) > 0, np.array(won, bool))
    np.testing.assert_array_equal(counts["won_prizes"], won)
    np.testing.assert_array_equal(counts["margin_sum"], margin)
    np.testing.assert_array_equal(counts["burned"], [burned] * 3)


def test_illegal_bid_is_played_as_the_env_plays_it():
    """A bid of a card no longer in hand is the lowest card in hand."""
    env = Goofspiel(num_agents=2)
    gm = Metrics(env)
    state = start(env, prizes=[0, 1])
    state, _, carry, _ = one_step(gm, env, state, [0, 5])
    # both cards are used now: each player plays its lowest card instead
    bid = np.asarray(played_bids(env, state, jnp.array([0, 5])))
    np.testing.assert_array_equal(bid, [1, 0])
    nxt, _, carry, counts = one_step(gm, env, state, [0, 1], carry=carry)
    # both play the 2: a tie
    assert not bool(nxt.player_hands[0, 1]) and bool(nxt.player_hands[0, 2])
    np.testing.assert_array_equal(counts["burned"], [1, 1])
    np.testing.assert_array_equal(carry.sum_sq_diff, [0 + 0, 25 + 0])


def test_bid_entropy_of_given_distributions():
    env = Goofspiel(num_agents=2)
    gm = Metrics(env)
    state = start(env)
    # four cards left: 9..12
    state = state.replace(
        player_hands=jnp.zeros((2, N), bool).at[:, 9:].set(True),
        current_round=jnp.int32(9),
    )
    probs = jnp.zeros((2, N))
    probs = probs.at[0, 9:].set(0.25)  # uniform over the four: 1
    probs = probs.at[1, 10].set(0.5).at[1, 12].set(0.5)  # log 2 / log 4
    _, _, _, counts = one_step(gm, env, state, [9, 10], probs=probs)
    np.testing.assert_allclose(counts["entropy_sum"], [1.0, 0.5], rtol=1e-6)
    np.testing.assert_array_equal(counts["choices"], [1, 1])
    deterministic = jnp.zeros((2, N)).at[:, 11].set(1.0)
    _, _, _, counts = one_step(gm, env, state, [11, 11], probs=deterministic)
    np.testing.assert_array_equal(counts["entropy_sum"], [0.0, 0.0])
    # one card left (the last round): no choice, not counted
    last = state.replace(
        player_hands=jnp.zeros((2, N), bool).at[:, 4].set(True),
        current_round=jnp.int32(12),
    )
    one = jnp.zeros((2, N)).at[:, 4].set(1.0)
    _, _, _, counts = one_step(gm, env, last, [4, 4], probs=one)
    np.testing.assert_array_equal(counts["choices"], [0, 0])
    np.testing.assert_array_equal(counts["entropy_sum"], [0, 0])
    np.testing.assert_allclose(
        normalized_entropy(jnp.full((13,), 1 / 13), jnp.int32(13)), 1.0, rtol=1e-6
    )


def scripted_games(env, policies, num_games):
    """Plays num_games games in a row through env.step (auto-reset) with
    policies[p](prize, round) -> card. Returns per-step counts (stacked),
    carries, dones, prizes and bids."""
    gm = Metrics(env)
    state = start(env)
    carry = gm.init_carry()
    out = {"counts": [], "carry": [], "done": [], "prize": [], "bid": []}
    step = jax.jit(gm.step)
    for t in range(N * num_games):
        prize = int(state.deck[state.current_round])
        bids = [policy(prize, int(state.current_round)) for policy in policies]
        action = jnp.array(bids, dtype=jnp.int32)
        nxt, _, reward, _, done, info = env.step(jax.random.PRNGKey(t), state, action)
        carry, counts = step(carry, state, action, nxt, reward, done, info, uniform_probs(env, state))
        out["counts"].append({k: np.asarray(v) for k, v in counts.items()})
        out["carry"].append(np.asarray(carry.sum_sq_diff))
        out["done"].append(bool(done))
        out["prize"].append(prize)
        out["bid"].append(bids)
        state = nxt
    counts = {k: np.stack([c[k] for c in out["counts"]]) for k in out["counts"][0]}
    return counts, np.array(out["carry"]), np.array(out["done"]), np.array(out["prize"]), np.array(out["bid"])


def test_spearman_over_scripted_games_and_carry_reset():
    """Player 0 bids the prize (rho = 1), player 1 the mirror card
    (rho = -1), player 2 its cards in order (rho = the rank correlation of
    the prize order with 1..13). Two games in a row: rho is credited once per
    game and player, at the last round, and the carry restarts at 0."""
    env = Goofspiel(num_agents=3)
    policies = [
        lambda prize, r: prize,
        lambda prize, r: N - 1 - prize,
        lambda prize, r: r,
    ]
    counts, carry, done, prize, bid = scripted_games(env, policies, num_games=2)
    np.testing.assert_array_equal(np.flatnonzero(done), [N - 1, 2 * N - 1])
    np.testing.assert_array_equal(counts["player_games"].sum(1), np.where(done, 3, 0))
    for g in range(2):
        rounds = slice(g * N, (g + 1) * N)
        rho = counts["spearman_sum"][(g + 1) * N - 1]
        want_2 = np.corrcoef(np.arange(N), prize[rounds])[0, 1]
        np.testing.assert_allclose(rho, [1.0, -1.0, want_2], atol=1e-6)
        # every card bid once (a permutation), so Pearson on the cards is Spearman
        for p in range(3):
            np.testing.assert_allclose(
                rho[p], np.corrcoef(bid[rounds, p], prize[rounds])[0, 1], atol=1e-6
            )
    assert (prize[:N] != prize[N:]).any()  # a new deck after the reset
    # the carry: sum of squared differences so far, 0 after each game's end
    sq = (bid - prize[:, None]) ** 2
    for t in range(2 * N):
        game_start = (t // N) * N
        want = 0 if done[t] else sq[game_start : t + 1].sum(0)
        np.testing.assert_array_equal(carry[t], want)
    np.testing.assert_array_equal(counts["spearman_sum"][~done], 0.0)
    # burns: the rounds whose top bid is tied, credited to all three
    top = bid.max(1, keepdims=True)
    tied = ((bid == top).sum(1) > 1).astype(float)
    np.testing.assert_array_equal(counts["burned"], np.repeat(tied[:, None], 3, 1))


def random_rollout(env, num_envs, num_games, seed):
    """Uniformly random legal bids for num_games full games in every env,
    through env.step as in training, with the metrics vmapped over envs."""
    gm = Metrics(env)
    num_steps = N * num_games

    @jax.jit
    def run(key):
        k_reset, k_play = jax.random.split(key)
        state, _ = jax.vmap(env.reset)(jax.random.split(k_reset, num_envs))
        carry = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (num_envs,) + x.shape), gm.init_carry()
        )

        def one(c, k):
            state, carry = c
            k_act, k_step = jax.random.split(k)
            probs = jax.vmap(lambda s: uniform_probs(env, s))(state)
            action = jax.random.categorical(k_act, jnp.log(probs))
            nxt, _, reward, _, done, info = jax.vmap(env.step)(
                jax.random.split(k_step, num_envs), state, action
            )
            carry, counts = jax.vmap(gm.step)(
                carry, state, action, nxt, reward, done, info, probs
            )
            prize = jax.vmap(lambda s: s.deck[s.current_round])(state)
            return (nxt, carry), (counts, action, prize, done)

        _, out = jax.lax.scan(one, (state, carry), jax.random.split(k_play, num_steps))
        return out

    counts, action, prize, done = jax.tree_util.tree_map(np.asarray, run(jax.random.PRNGKey(seed)))
    return gm, counts, action, prize, done


def independent_bids_reference(n):
    """Per round with n independent uniform bids on 1..13: P(tied top) and
    E[top - highest other bid | sole top], by enumeration."""
    burned, margins = 0, []
    for bids in itertools.product(range(N), repeat=n):
        top = max(bids)
        if bids.count(top) > 1:
            burned += 1
        else:
            margins.append(top - max(b for b in bids if b != top))
    return burned / N**n, float(np.mean(margins))


@pytest.mark.parametrize("n", [2, 3])
def test_uniform_random_play(n):
    env = Goofspiel(num_agents=n)
    num_envs, num_games = 200, 8
    gm, counts, bid, prize, done = random_rollout(env, num_envs, num_games, seed=n)
    T = N * num_games
    np.testing.assert_array_equal(done, np.broadcast_to((np.arange(T) % N == N - 1)[:, None], done.shape))

    # brute force from the recorded bids and prizes
    top = bid.max(-1, keepdims=True)
    is_top = bid == top
    sole = is_top.sum(-1, keepdims=True) == 1
    won = is_top & sole
    other = np.where(is_top, -1, bid).max(-1, keepdims=True)
    np.testing.assert_array_equal(counts["won_prizes"], won)
    np.testing.assert_array_equal(counts["margin_sum"], np.where(won, top - other, 0))
    np.testing.assert_array_equal(counts["burned"], np.broadcast_to(~sole, bid.shape))
    np.testing.assert_array_equal(counts["choices"], np.broadcast_to((np.arange(T) % N != N - 1)[:, None, None], bid.shape))
    games_bid = bid.reshape(num_games, N, num_envs, n)
    games_prize = prize.reshape(num_games, N, num_envs)
    rho = np.array([
        [[np.corrcoef(games_bid[g, :, e, p], games_prize[g, :, e])[0, 1] for p in range(n)] for e in range(num_envs)]
        for g in range(num_games)
    ])
    np.testing.assert_allclose(counts["spearman_sum"][done], rho.reshape(-1, n), atol=1e-5)

    totals = masked_totals(counts, np.ones(bid.shape, bool))
    out = {k: float(v) for k, v in gm.summarize(totals).items()}
    num_pg = num_envs * num_games * n
    assert out["n_player_games"] == num_pg
    assert out["n_player_rounds"] == T * num_envs * n
    assert out["n_choices"] == (N - 1) * num_games * num_envs * n
    # uniform choice among the cards in hand: normalised entropy 1
    assert out["bid_entropy"] == pytest.approx(1.0, abs=1e-5)
    # bids and prizes independent: E[rho] = 0, Var[rho] = 1 / (N - 1) for
    # every (game, player), and independent across players given the prizes
    assert abs(out["bid_prize_spearman"]) < 5 * np.sqrt(1 / (N - 1) / num_pg)
    # every round's bids are independent and uniform on 1..13 (each player's
    # bids are a uniformly random permutation, independent of the others')
    p_burn, margin = independent_bids_reference(n)
    if n == 2:
        assert p_burn == pytest.approx(1 / 13) and margin == pytest.approx(14 / 3)
    num_rounds = T * num_envs
    assert abs(out["burn_rate"] - p_burn) < 5 * np.sqrt(p_burn * (1 - p_burn) / num_rounds)
    margins = np.where(won, top - other, 0).sum(-1)[sole[..., 0]]
    assert abs(out["overbid_margin"] - margin) < 5 * margins.std() / np.sqrt(margins.size)


def test_masked_totals_split_by_player():
    """game/ and game_br/ in the scripts: masking by player keeps exactly the
    credited player's events."""
    env = Goofspiel(num_agents=2)
    gm, counts, bid, prize, done = random_rollout(env, num_envs=8, num_games=2, seed=7)
    mask = np.zeros(bid.shape, bool)
    mask[..., 0] = True
    first = gm.summarize(masked_totals(counts, mask))
    both = gm.summarize(masked_totals(counts, np.ones_like(mask)))
    assert float(first["n_won_prizes"]) == counts["won_prizes"][..., 0].sum()
    assert float(first["n_player_games"]) == done.sum()
    assert float(both["n_player_games"]) == 2 * done.sum()
    np.testing.assert_allclose(
        float(first["bid_prize_spearman"]),
        counts["spearman_sum"][..., 0].sum() / done.sum(), rtol=1e-6,
    )
