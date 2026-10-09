"""Checks for the Bluff environment and the Bluff NFSP training code.

- crafted claims: the win bonus, caught lies and truthful claims, free leads,
  truncation, the observation during card selection and public information;
- random games against a step-by-step reference of the rules and of the
  documented observation layout, card conservation, legal-action masks and
  rewards;
- seat rotation: shifting every per-player field by k seats shifts the next
  state and the rewards by k and leaves the observation unchanged;
- the NFSP targets (per-player GAE and Q(lambda)) against a brute-force
  per-player reference, the reservoir against Algorithm R, the per-game policy
  mixing, checkpoint loading and the evaluation's win/loss/draw accounting.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from bluffjax import make
from bluffjax.environments.bluff.bluff import (
    CHALLENGE,
    CLAIM,
    MAX_CLAIM,
    NAME_RANK,
    PLAY,
    BluffState,
)

KEY = jax.random.PRNGKey(0)
CHALLENGE_ACTION, PASS = 0, 1


@pytest.fixture(scope="module")
def env2():
    return make("bluff", num_agents=2, horizon=200)


@pytest.fixture(scope="module")
def env3():
    return make("bluff", num_agents=3, horizon=200)


_JIT = {}


def _jit(env, name):
    if (id(env), name) not in _JIT:
        _JIT[(id(env), name)] = jax.jit(getattr(env, name))
    return _JIT[(id(env), name)]


def step(env, state, action):
    return _jit(env, "step_env")(KEY, state, jnp.int32(action))


def obs_of(env, state):
    return np.asarray(_jit(env, "obs_from_state")(state))


def mask_of(env, state):
    return np.asarray(_jit(env, "get_avail_actions")(state))


def counts(d, num_ranks=13):
    c = np.zeros(num_ranks, np.float32)
    for rank, n in d.items():
        c[rank] = n
    return c


def state_with(env, hands, pile, **fields):
    """A reset state with the given hands and pile (all cards accounted for)."""
    state, _ = env.reset(KEY)
    hands = np.asarray(hands, np.float32)
    pile = np.asarray(pile, np.float32)
    assert (hands.sum(0) + pile == env.cards_per_rank).all()
    state = state.replace(
        agent_hands=jnp.asarray(hands),
        agent_hand_sizes=jnp.asarray(hands.sum(1), jnp.int32),
        pile_hand=jnp.asarray(pile),
        pile_size=jnp.int32(pile.sum()),
    )
    return state.replace(
        **{k: jnp.asarray(v, getattr(state, k).dtype) for k, v in fields.items()}
    )


def two_player_endgame(env, own, pile, current_rank=4):
    """Player 0 to claim (rank current_rank required) holding `own`; player 1
    holds the rest of the deck except `pile`."""
    own, pile = counts(own), counts(pile)
    other = env.cards_per_rank - own - pile
    return state_with(
        env,
        [own, other],
        pile,
        pile_claims=counts({(current_rank - 1) % 13: pile.sum()}),
        phase=CLAIM,
        current_player_idx=0,
        start_player_idx=1,
        challenge_target_idx=0,
        has_current_rank=True,
        current_rank=current_rank,
        free_lead=False,
    )


def obs_blocks(env):
    """Slices of the observation, from the layout in bluff.py's docstring."""
    n, cr, d = env.num_agents, env.cards_per_rank * env.num_ranks, env.deck_size
    sizes = [
        ("own_hand", cr),
        ("picked", cr),
        ("pile_claims", cr),
        ("pile_size", d),
        ("hand_sizes", n * d),
        ("claim_size", MAX_CLAIM),
        ("claimant", n),
        ("phase", 4),
        ("free_lead", 1),
        ("revealed", cr),
        ("loser", n),
        ("lie", 1),
    ]
    out, start = {}, 0
    for name, size in sizes:
        out[name] = slice(start, start + size)
        start += size
    assert start == env.obs_dim
    return out


def rank_thermo(c, num_suits=4):
    """Suit-major thermometer of per-rank (offset-indexed) counts."""
    c = np.asarray(c)
    return (np.arange(num_suits)[:, None] < c[None, :]).astype(np.float32).reshape(-1)


def thermo(x, length):
    return (np.arange(length) < x).astype(np.float32)


# =============================================================================
# Crafted claims
# =============================================================================


def test_obs_size(env2, env3):
    assert env2.obs_dim == 378 and env3.obs_dim == 432
    assert env2.action_dim == 13
    for env in (env2, env3):
        _, obs = env.reset(KEY)
        assert obs.shape == (env.obs_dim,)
        obs_blocks(env)


def test_win_bonus_when_last_claim_is_not_challenged(env2):
    """+10 is paid once, when the last claim is resolved, not while it waits."""
    state = two_player_endgame(env2, own={4: 1}, pile={2: 2})
    state, _, r, absorbing, done, info = step(env2, state, 0)  # claim 1 card
    assert int(state.phase) == PLAY and not done and np.all(np.asarray(r) == 0)
    state, _, r, absorbing, done, info = step(env2, state, 0)  # play the 4
    assert int(state.phase) == CHALLENGE and int(state.current_player_idx) == 1
    assert int(state.agent_hand_sizes[0]) == 0
    # The hand is empty, but the claim can still be challenged.
    assert np.all(np.asarray(r) == 0) and not done and not np.asarray(absorbing).any()
    assert not np.asarray(info["game_winner"]).any()
    _, _, r, absorbing, done, info = step(env2, state, PASS)
    np.testing.assert_array_equal(np.asarray(r), [1.0 + 10.0, 0.0])
    assert done and np.asarray(absorbing).all() and not info["truncated"]
    np.testing.assert_array_equal(np.asarray(info["game_winner"]), [True, False])


def test_win_bonus_when_truthful_last_claim_is_challenged(env2):
    state = two_player_endgame(env2, own={4: 1}, pile={2: 2})
    state, *_ = step(env2, state, 0)
    state, *_ = step(env2, state, 0)
    state, _, r, absorbing, done, info = step(env2, state, CHALLENGE_ACTION)
    # the claim was true: the challenger picks up the pile (2 + 1 cards)
    np.testing.assert_array_equal(np.asarray(r), [1.0 + 10.0, -3.0])
    assert done and np.asarray(absorbing).all()
    np.testing.assert_array_equal(np.asarray(info["game_winner"]), [True, False])
    assert int(info["challenge_status"]) == 0


def test_caught_lie_on_last_card_pays_no_bonus(env2):
    """Caught lying with the last card: pick up the pile, no bonus, play goes on."""
    state = two_player_endgame(env2, own={5: 1}, pile={2: 2})
    state, *_ = step(env2, state, 0)  # claim one 4
    assert np.asarray(mask_of(env2, state)).nonzero()[0].tolist() == [1]
    state, _, r, *_ = step(env2, state, 1)  # play the 5 (offset 1): a lie
    assert np.all(np.asarray(r) == 0)
    state, obs, r, absorbing, done, info = step(env2, state, CHALLENGE_ACTION)
    np.testing.assert_array_equal(np.asarray(r), [-3.0, 1.0])
    assert not done and not np.asarray(absorbing).any()
    assert not np.asarray(info["game_winner"]).any() and int(info["challenge_status"]) == 1
    np.testing.assert_array_equal(np.asarray(state.agent_hands[0]), counts({2: 2, 5: 1}))
    assert int(state.pile_size) == 0 and not np.asarray(state.pile_claims).any()
    # The challenger makes the next claim, a free lead.
    assert int(state.current_player_idx) == 1 and int(state.phase) == NAME_RANK
    assert bool(state.free_lead) and mask_of(env2, state)[:13].all()
    # Everyone sees the turned-over card, who picked up the pile and that it was a lie.
    b = obs_blocks(env2)
    np.testing.assert_array_equal(obs[b["revealed"]], rank_thermo(counts({1: 1})))
    np.testing.assert_array_equal(obs[b["loser"]], [0.0, 1.0])  # seat 1 = player 0
    assert obs[b["lie"]][0] == 1.0 and obs[b["free_lead"]][0] == 1.0


def test_uncaught_lie_on_last_card_wins(env2):
    state = two_player_endgame(env2, own={5: 1}, pile={2: 2})
    state, *_ = step(env2, state, 0)
    state, *_ = step(env2, state, 1)
    _, _, r, absorbing, done, info = step(env2, state, PASS)
    np.testing.assert_array_equal(np.asarray(r), [11.0, 0.0])
    assert done and np.asarray(absorbing).all()
    np.testing.assert_array_equal(np.asarray(info["game_winner"]), [True, False])


def _three_player_claim(env3, played):
    """Player 1 claims three 7s playing `played`; returns the state after the play."""
    hands = np.zeros((3, 13), np.float32)
    hands[1] = counts({7: 3, 2: 1, 9: 2})
    hands[0] = counts({7: 1, 3: 4})
    pile = counts({5: 2, 6: 1})
    hands[2] = env3.cards_per_rank - hands[0] - hands[1] - pile
    state = state_with(
        env3,
        hands,
        pile,
        pile_claims=counts({5: 2, 6: 1}),
        phase=CLAIM,
        current_player_idx=1,
        challenge_target_idx=1,
        has_current_rank=True,
        current_rank=7,
        free_lead=False,
    )
    state, *_ = step(env3, state, 2)  # claim 3 cards
    for rank in played:
        state, _, r, _, done, _ = step(env3, state, (rank - 7) % 13)
        assert np.all(np.asarray(r) == 0) and not done
    assert int(state.phase) == CHALLENGE and int(state.current_player_idx) == 2
    assert int(state.pile_size) == 6 and float(state.pile_claims[7]) == 3.0
    return state


@pytest.mark.parametrize("played,lie", [((7, 7, 2), True), ((7, 7, 7), False)])
def test_challenge_outcome(env3, played, lie):
    """The second challenger in turn order challenges a three-card claim."""
    state = _three_player_claim(env3, played)
    state, _, r, _, done, _ = step(env3, state, PASS)  # player 2 passes
    assert int(state.current_player_idx) == 0 and np.all(np.asarray(r) == 0)
    pile = np.asarray(state.pile_hand)
    hands_before = np.asarray(state.agent_hands)
    state, _, r, absorbing, done, info = step(env3, state, CHALLENGE_ACTION)
    loser, winner = (1, 0) if lie else (0, 1)
    expected = np.zeros(3)
    expected[winner] += 1.0
    expected[loser] -= 6.0
    np.testing.assert_array_equal(np.asarray(r), expected)
    assert not done and not np.asarray(info["game_winner"]).any()
    np.testing.assert_array_equal(
        np.asarray(state.agent_hands[loser]), hands_before[loser] + pile
    )
    assert int(state.current_player_idx) == winner and int(state.phase) == NAME_RANK
    assert int(state.challenge_loser_idx) == loser and bool(state.challenge_lie) == lie
    np.testing.assert_array_equal(
        np.asarray(state.challenge_hand), counts({r_: played.count(r_) for r_ in played})
    )
    # cards stay in the deck
    total = np.asarray(state.agent_hands).sum(0) + np.asarray(state.pile_hand)
    np.testing.assert_array_equal(total, np.full(13, 4.0))


def test_nobody_challenges(env3):
    state = _three_player_claim(env3, (7, 2, 9))
    state, *_ = step(env3, state, PASS)
    state, _, r, _, done, info = step(env3, state, PASS)
    np.testing.assert_array_equal(np.asarray(r), [0.0, 3.0, 0.0])
    assert not done and int(info["challenge_status"]) == 2
    assert int(state.current_player_idx) == 2 and int(state.phase) == CLAIM
    assert int(state.current_rank) == 8 and not bool(state.free_lead)
    assert int(state.pile_size) == 6


def test_free_lead_can_claim_any_rank_and_lie_with_one_card(env2):
    """At a free lead the claimed rank is named first, independently of the cards."""
    own = counts({0: 2, 1: 1})
    state = state_with(
        env2,
        [own, 4.0 - own],
        np.zeros(13),
        phase=NAME_RANK,
        current_player_idx=0,
        challenge_target_idx=0,
        has_current_rank=False,
        free_lead=True,
    )
    assert mask_of(env2, state).all()  # any of the 13 ranks
    state, *_ = step(env2, state, 9)  # no reference rank yet: offset = rank
    assert int(state.phase) == CLAIM and int(state.current_rank) == 9
    state, *_ = step(env2, state, 0)  # one card
    # offsets are relative to the named rank: 4 -> rank 0, 5 -> rank 1
    assert mask_of(env2, state).nonzero()[0].tolist() == [4, 5]
    state, *_ = step(env2, state, 4)  # play a 0, claimed as a 9
    assert int(state.claim_rank) == 9 and float(state.pile_claims[9]) == 1.0
    state, _, r, _, done, _ = step(env2, state, CHALLENGE_ACTION)
    np.testing.assert_array_equal(np.asarray(r), [-1.0, 1.0])
    assert bool(state.challenge_lie) and int(state.challenge_loser_idx) == 0
    # the next free lead names its rank relative to the challenged one (9)
    assert int(state.current_player_idx) == 1 and int(state.phase) == NAME_RANK
    state, *_ = step(env2, state, 3)
    assert int(state.current_rank) == 12


def test_free_lead_truthful_claim(env2):
    own = counts({0: 2, 1: 1})
    state = state_with(
        env2,
        [own, 4.0 - own],
        np.zeros(13),
        phase=NAME_RANK,
        current_player_idx=0,
        challenge_target_idx=0,
        has_current_rank=False,
        free_lead=True,
    )
    state, *_ = step(env2, state, 0)  # name rank 0
    state, *_ = step(env2, state, 1)  # two cards
    state, *_ = step(env2, state, 0)
    state, *_ = step(env2, state, 0)
    state, _, r, _, done, _ = step(env2, state, CHALLENGE_ACTION)
    np.testing.assert_array_equal(np.asarray(r), [1.0, -2.0])
    assert not bool(state.challenge_lie) and int(state.challenge_loser_idx) == 1
    assert int(state.current_player_idx) == 0  # the claimant leads again


def test_truncation_is_not_absorbing(env2):
    state, _ = env2.reset(KEY)
    state = state.replace(timestep=env2.horizon - 1)
    _, _, r, absorbing, done, info = step(env2, state, 0)
    assert done and info["truncated"] and not np.asarray(absorbing).any()
    assert not np.asarray(info["game_winner"]).any() and np.all(np.asarray(r) == 0)
    # A game that ends on the last step is over, not truncated.
    state = two_player_endgame(env2, own={4: 1}, pile={2: 2})
    state, *_ = step(env2, state, 0)
    state, *_ = step(env2, state, 0)
    state = state.replace(timestep=env2.horizon - 1)
    _, _, r, absorbing, done, info = step(env2, state, PASS)
    assert done and not info["truncated"] and np.asarray(absorbing).all()
    assert float(r[0]) == 11.0


def test_obs_changes_during_card_selection(env2):
    own = counts({4: 3, 7: 2, 11: 1})
    pile = counts({3: 2})
    state = two_player_endgame(env2, own={4: 3, 7: 2, 11: 1}, pile={3: 2})
    b = obs_blocks(env2)
    size3, obs3, *_ = step(env2, state, 2)  # claim three 4s
    _, obs1, *_ = step(env2, state, 0)  # claim one 4
    assert not np.array_equal(obs1, obs3)
    np.testing.assert_array_equal(np.asarray(obs3)[b["claim_size"]], [1, 1, 1, 0])
    seen = [np.asarray(obs3)]
    state = size3
    for offset in (0, 3):  # a 4, then a 7 (a lie)
        state, obs, *_ = step(env2, state, offset)
        seen.append(np.asarray(obs))
    for a, c in zip(seen, seen[1:]):
        assert not np.array_equal(a, c)
    picked = counts({0: 1, 3: 1})  # offsets from the claimed rank 4
    np.testing.assert_array_equal(seen[2][b["picked"]], rank_thermo(picked))
    np.testing.assert_array_equal(
        seen[2][b["own_hand"]], rank_thermo(np.roll(own, -4) - picked)
    )
    # the public hand size only changes when the cards are played
    np.testing.assert_array_equal(
        seen[2][b["hand_sizes"]][:52], thermo(own.sum(), 52)
    )
    np.testing.assert_array_equal(
        seen[2][b["hand_sizes"]][52:], thermo(52 - own.sum() - pile.sum(), 52)
    )
    # after the last card the opponent sees the claim, not the cards
    state, obs, *_ = step(env2, state, 0)
    obs = np.asarray(obs)
    assert int(state.phase) == CHALLENGE and not obs[b["picked"]].any()
    np.testing.assert_array_equal(obs[b["claim_size"]], [1, 1, 1, 0])
    np.testing.assert_array_equal(obs[b["claimant"]], [0, 1])
    np.testing.assert_array_equal(
        obs[b["hand_sizes"]][52:], thermo(own.sum() - 3, 52)
    )


# =============================================================================
# Random games
# =============================================================================


def random_games(env, num_games, num_steps, seed):
    """Random legal play from reset with step_env; finished games stay frozen."""

    @jax.jit
    def run(key):
        key_reset, key_play = jax.random.split(key)
        state, _ = jax.vmap(env.reset)(jax.random.split(key_reset, num_games))

        def one(carry, k):
            s, finished = carry
            mask = jax.vmap(env.get_avail_actions)(s)
            k_action, k_step = jax.random.split(k)
            action = jax.random.categorical(
                k_action, jnp.where(mask, 0.0, -jnp.inf), axis=-1
            )
            ns, _, reward, absorbing, done, info = jax.vmap(env.step_env)(
                jax.random.split(k_step, num_games), s, action
            )

            def keep(new, old):
                return jnp.where(finished.reshape((-1,) + (1,) * (new.ndim - 1)), old, new)

            ns = jax.tree_util.tree_map(keep, ns, s)
            rec = dict(
                state=s,
                mask=mask,
                action=action,
                reward=reward,
                absorbing=absorbing,
                done=done,
                info=info,
                next_state=ns,
                live=~finished,
            )
            return (ns, finished | done), rec

        _, rec = jax.lax.scan(
            one,
            (state, jnp.zeros(num_games, bool)),
            jax.random.split(key_play, num_steps),
        )
        return rec

    return jax.tree_util.tree_map(np.asarray, run(jax.random.PRNGKey(seed)))


@pytest.fixture(scope="module")
def games():
    """Random games: 2 and 3 players played to the end, and 3 players cut off."""
    out = {}
    for name, n, horizon, num_games, num_steps in (
        ("2p", 2, 2500, 16, 2500),
        ("3p", 3, 3000, 12, 3000),
        ("3p_cut", 3, 150, 24, 150),
    ):
        env = make("bluff", num_agents=n, horizon=horizon)
        out[name] = (env, random_games(env, num_games, num_steps, seed=n * 7 + horizon))
    return out


def _at(tree, t, g):
    return jax.tree_util.tree_map(lambda x: x[t, g], tree)


def ref_step(env, s, a):
    """The rules of bluff.py's docstring, one step at a time, on numpy values."""
    n, num_ranks = env.num_agents, env.num_ranks
    s = {k: np.array(v, copy=True) for k, v in s.items()}
    r = np.zeros(n, np.float32)
    p = int(s["current_player_idx"])
    ref = int(s["current_rank"]) if s["has_current_rank"] else 0
    phase = int(s["phase"])
    resolved = False
    if phase == NAME_RANK:
        assert 0 <= a < num_ranks
        s["current_rank"], s["has_current_rank"], s["phase"] = (ref + a) % num_ranks, True, CLAIM
    elif phase == CLAIM:
        assert 0 <= a < min(MAX_CLAIM, s["agent_hand_sizes"][p])
        s["claim_size"], s["claim_rank"] = a + 1, s["current_rank"]
        s["pending_play_hand"][:] = 0
        s["pending_play_count"], s["challenge_status"] = 0, 3
        s["challenge_target_idx"], s["phase"] = p, PLAY
    elif phase == PLAY:
        rank = (ref + a) % num_ranks
        assert s["agent_hands"][p, rank] > s["pending_play_hand"][rank]
        s["pending_play_hand"][rank] += 1
        s["pending_play_count"] += 1
        if s["pending_play_count"] == s["claim_size"]:
            s["agent_hands"][p] -= s["pending_play_hand"]
            s["pile_hand"] += s["pending_play_hand"]
            s["pile_claims"][s["claim_rank"]] += s["claim_size"]
            s["phase"], s["current_player_idx"] = CHALLENGE, (p + 1) % n
    else:
        target = int(s["challenge_target_idx"])
        if a == CHALLENGE_ACTION:
            truthful = s["pending_play_hand"][s["claim_rank"]] == s["claim_size"]
            loser, winner = (p, target) if truthful else (target, p)
            r[winner] += 1.0
            r[loser] -= s["pile_hand"].sum()
            s["agent_hands"][loser] += s["pile_hand"]
            s["pile_hand"][:] = 0
            s["pile_claims"][:] = 0
            s["challenge_hand"] = s["pending_play_hand"].copy()
            s["challenge_loser_idx"], s["challenge_lie"] = loser, not truthful
            s["challenge_status"] = 0 if truthful else 1
            s["current_rank"], s["free_lead"] = s["claim_rank"], True
            s["phase"], s["current_player_idx"], s["challenge_target_idx"] = NAME_RANK, winner, winner
            resolved = True
        elif (p + 1) % n == target:
            r[target] += s["claim_size"]
            nxt = (target + 1) % n
            s["current_rank"] = (s["claim_rank"] + 1) % num_ranks
            s["free_lead"], s["challenge_status"] = False, 2
            s["phase"], s["current_player_idx"], s["challenge_target_idx"] = CLAIM, nxt, nxt
            resolved = True
        else:
            s["current_player_idx"] = (p + 1) % n
        if resolved:
            s["pending_play_hand"][:] = 0
            s["pending_play_count"], s["claim_size"] = 0, 0
    s["agent_hand_sizes"] = s["agent_hands"].sum(1).astype(np.int32)
    s["pile_size"] = int(s["pile_hand"].sum())
    over = resolved and bool((s["agent_hand_sizes"] == 0).any())
    winner = (s["agent_hand_sizes"] == 0) & over
    r += 10.0 * winner
    s["timestep"] = int(s["timestep"]) + 1
    truncated = (not over) and s["timestep"] >= env.horizon
    s["game_winner"] = winner
    s["absorbing"] = np.full(n, over)
    s["done"] = over or truncated
    return s, r, truncated


def ref_obs(env, s):
    """The observation as documented in bluff.py, from numpy state values."""
    n, num_ranks, d = env.num_agents, env.num_ranks, env.deck_size
    p = int(s["current_player_idx"])
    ref = int(s["current_rank"]) if s["has_current_rank"] else 0
    phase = int(s["phase"])

    def ranks(c):  # offset o -> rank (ref + o) % num_ranks
        return rank_thermo(np.asarray(c)[(ref + np.arange(num_ranks)) % num_ranks])

    picked = s["pending_play_hand"] if phase == PLAY else np.zeros(num_ranks)
    claimant = np.zeros(n)
    if phase == CHALLENGE:
        claimant[(int(s["challenge_target_idx"]) - p) % n] = 1
    loser = np.zeros(n)
    if s["challenge_loser_idx"] >= 0:
        loser[(int(s["challenge_loser_idx"]) - p) % n] = 1
    claim = thermo(s["claim_size"], MAX_CLAIM) if phase in (PLAY, CHALLENGE) else np.zeros(MAX_CLAIM)
    return np.concatenate(
        [
            ranks(s["agent_hands"][p] - picked),
            ranks(picked),
            ranks(s["pile_claims"]),
            thermo(s["pile_size"], d),
            *[thermo(s["agent_hand_sizes"][(p + k) % n], d) for k in range(n)],
            claim,
            claimant,
            np.eye(4)[phase],
            [float(s["free_lead"])],
            ranks(s["challenge_hand"]),
            loser,
            [float(s["challenge_lie"])],
        ]
    ).astype(np.float32)


def _state_dict(state):
    return {k: getattr(state, k) for k in BluffState.__dataclass_fields__}


@pytest.mark.parametrize("name", ["2p", "3p", "3p_cut"])
def test_random_games_match_reference(games, name):
    """Every transition and reward of random games equals the reference rules."""
    env, rec = games[name]
    steps = 0
    for t, g in zip(*np.nonzero(rec["live"])):
        s = _state_dict(_at(rec["state"], t, g))
        expected, r, truncated = ref_step(env, s, int(rec["action"][t, g]))
        got = _state_dict(_at(rec["next_state"], t, g))
        for k in expected:
            np.testing.assert_array_equal(got[k], expected[k], err_msg=f"{k} at {t}, {g}")
        np.testing.assert_array_equal(rec["reward"][t, g], r)
        assert bool(rec["info"]["truncated"][t, g]) == truncated
        np.testing.assert_array_equal(rec["info"]["game_winner"][t, g], expected["game_winner"])
        steps += 1
    assert steps > 3000


@pytest.mark.parametrize("name", ["2p", "3p"])
def test_random_games_observation_layout(games, name):
    env, rec = games[name]
    live = np.argwhere(rec["live"])
    rows = live[np.random.default_rng(0).choice(len(live), 1500, replace=False)]
    states = jax.tree_util.tree_map(lambda x: x[rows[:, 0], rows[:, 1]], rec["state"])
    obs = np.asarray(jax.jit(jax.vmap(env.obs_from_state))(states))
    for i in range(len(rows)):
        s = _state_dict(jax.tree_util.tree_map(lambda x: x[i], states))
        np.testing.assert_array_equal(obs[i], ref_obs(env, s))
    phases = np.asarray(states.phase)
    assert set(phases.tolist()) == {CLAIM, PLAY, CHALLENGE, NAME_RANK}


@pytest.mark.parametrize("name", ["2p", "3p", "3p_cut"])
def test_random_games_invariants(games, name):
    """Cards are conserved, the player to act has a legal action, and the win
    bonus is paid exactly once per finished game, at its end."""
    env, rec = games[name]
    live = rec["live"]
    n = env.num_agents
    nxt = rec["next_state"]
    total = nxt.agent_hands.sum(2) + nxt.pile_hand  # (T, G, R)
    assert (total[live] == env.cards_per_rank).all()
    assert (nxt.agent_hand_sizes[live] == nxt.agent_hands[live].sum(-1)).all()
    assert (nxt.pile_size[live] == nxt.pile_hand[live].sum(-1)).all()
    assert rec["mask"][live].any(-1).all()
    first = rec["state"]
    assert (first.agent_hand_sizes[0] == env.deck_size // n).all()
    assert (first.pile_size[0] == env.deck_size % n).all()

    done = rec["done"] & live
    over = done & rec["absorbing"][..., 0]
    truncated = done & ~rec["absorbing"][..., 0]
    assert (rec["info"]["truncated"][done] == truncated[done]).all()
    assert not (rec["absorbing"] & ~rec["done"][..., None])[live].any()
    winners = rec["info"]["game_winner"] & live[..., None]
    assert (winners.sum(-1) == over).all()  # one winner per finished game, else none
    reward = np.where(live[..., None], rec["reward"], 0.0)
    # Without the bonus a step pays at most +4 (an unchallenged claim).
    assert (reward[~winners] <= MAX_CLAIM).all()
    assert ((reward[winners] >= 11) & (reward[winners] <= 10 + MAX_CLAIM)).all()
    assert rec["done"].any(0).all()  # the horizon is the number of recorded steps
    if name == "3p_cut":
        assert truncated.sum() > 10
    else:
        assert over.sum() >= 0.8 * live.shape[1]


def test_start_player_is_uniform():
    env = make("bluff", num_agents=3)
    states, _ = jax.vmap(env.reset)(jax.random.split(KEY, 6000))
    start = np.asarray(states.current_player_idx)
    assert (start == np.asarray(states.start_player_idx)).all()
    freq = np.bincount(start, minlength=3) / len(start)
    assert np.abs(freq - 1 / 3).max() < 5 * np.sqrt(2 / 9 / len(start))


# =============================================================================
# Seat rotation
# =============================================================================

SEATLESS = {
    "pile_hand", "pile_claims", "pile_size", "phase", "claim_size", "claim_rank",
    "pending_play_hand", "pending_play_count", "has_current_rank", "current_rank",
    "free_lead", "challenge_status", "challenge_hand", "challenge_lie", "done",
    "timestep",
}
PER_PLAYER = {"agent_hands", "agent_hand_sizes", "absorbing", "game_winner"}
PLAYER_INDEX = {
    "current_player_idx", "start_player_idx", "challenge_target_idx", "challenge_loser_idx",
}


def rotate(state, k, n):
    """Every player moves k seats on: player i's data becomes player (i + k)'s."""
    fields = {f: jnp.roll(getattr(state, f), k, axis=-1 if f in ("absorbing", "game_winner", "agent_hand_sizes") else -2) for f in PER_PLAYER}
    for f in PLAYER_INDEX:
        idx = getattr(state, f)
        fields[f] = jnp.where(idx >= 0, (idx + k) % n, idx)
    return state.replace(**fields)


@pytest.mark.parametrize("name", ["2p", "3p"])
def test_seat_rotation_equivariance(games, name):
    """step(rotate(s), a) == rotate(step(s, a)); the observation and the legal
    actions don't depend on the absolute seats."""
    assert set(BluffState.__dataclass_fields__) == SEATLESS | PER_PLAYER | PLAYER_INDEX
    env, rec = games[name]
    n = env.num_agents
    live = np.argwhere(rec["live"])
    rows = live[np.random.default_rng(1).choice(len(live), 3000, replace=False)]
    states = jax.tree_util.tree_map(lambda x: x[rows[:, 0], rows[:, 1]], rec["state"])
    actions = rec["action"][rows[:, 0], rows[:, 1]]

    @jax.jit
    def check(state, action, key):
        out = []
        s1, o1, r1, a1, d1, _ = env.step_env(key, state, action)
        for k in range(1, n):
            rs = rotate(state, k, n)
            s2, o2, r2, a2, d2, _ = env.step_env(key, rs, action)
            same_state = jnp.stack(
                jax.tree_util.tree_leaves(
                    jax.tree_util.tree_map(jnp.array_equal, rotate(s1, k, n), s2)
                )
            ).all()
            out.append(
                jnp.stack(
                    [
                        same_state,
                        s2.current_player_idx == (s1.current_player_idx + k) % n,
                        jnp.array_equal(r2, jnp.roll(r1, k)),
                        d1 == d2,
                        jnp.array_equal(env.obs_from_state(rs), env.obs_from_state(state)),
                        jnp.array_equal(o1, o2),
                        jnp.array_equal(
                            env.get_avail_actions(rs), env.get_avail_actions(state)
                        ),
                    ]
                )
            )
        return jnp.stack(out)

    ok = np.asarray(
        jax.vmap(check)(states, jnp.asarray(actions), jax.random.split(KEY, len(rows)))
    )
    assert ok.all(), ok.reshape(-1, ok.shape[-1]).mean(0)


# =============================================================================
# NFSP training code
# =============================================================================


@pytest.fixture(scope="module")
def nfsp():
    """The training scripts need the `baselines` extra (distrax, optax, hydra)."""
    for module in ("distrax", "optax", "hydra"):
        pytest.importorskip(module)
    from bluffjax.examples.bluff import bluff_nfsp_common, bluff_ppo_nfsp, bluff_pqn_nfsp

    return bluff_nfsp_common, bluff_ppo_nfsp, bluff_pqn_nfsp


def training_rollout(env, num_envs, num_steps, seed):
    """Random play with env.step (auto-reset), as in the training scripts."""

    @jax.jit
    def run(key):
        key_reset, key_play = jax.random.split(key)
        state, _ = jax.vmap(env.reset)(jax.random.split(key_reset, num_envs))

        def one(s, k):
            mask = jax.vmap(env.get_avail_actions)(s)
            k_action, k_step = jax.random.split(k)
            action = jax.random.categorical(
                k_action, jnp.where(mask, 0.0, -jnp.inf), axis=-1
            )
            ns, _, reward, _, done, info = jax.vmap(env.step)(
                jax.random.split(k_step, num_envs), s, action
            )
            return ns, (reward, done, info["truncated"], s.current_player_idx)

        last, (reward, done, truncated, player) = jax.lax.scan(
            one, state, jax.random.split(key_play, num_steps)
        )
        return reward, done, truncated, player, last.current_player_idx

    return [np.asarray(x) for x in run(jax.random.PRNGKey(seed))]


def _decision_end(dones, truncated, players, last_player, t, n):
    """End step u of player players[t, n]'s decision at t and how it ends."""
    T = dones.shape[0]
    p = players[t, n]
    u = t
    while True:
        if dones[u, n]:
            return u, "truncated" if truncated[u, n] else "terminal"
        if u + 1 == T:
            return u, "cutoff" if last_player[n] == p else "unknown"
        if players[u + 1, n] == p:
            return u, "next"
        u += 1


def brute_force_gae(values, rewards, dones, truncated, players, last_value, last_player, gamma, lam):
    T, N, _ = rewards.shape
    adv = np.zeros((T, N), np.float32)
    valid = np.zeros((T, N), bool)
    g, gl = np.float32(gamma), np.float32(gamma * lam)
    for n in range(N):
        for t in reversed(range(T)):
            p = players[t, n]
            u, end = _decision_end(dones, truncated, players, last_player, t, n)
            if end in ("truncated", "unknown"):
                continue
            ret = np.float32(0.0)
            for w in range(u, t - 1, -1):
                ret = np.float32(ret + rewards[w, n, p])
            if end == "terminal":
                v_next, a_next, nt = np.float32(0), np.float32(0), np.float32(0)
            elif end == "cutoff":
                v_next, a_next, nt = last_value[n], np.float32(0), np.float32(1)
            else:
                v_next, a_next, nt = values[u + 1, n], adv[u + 1, n], np.float32(1)
            delta = np.float32(np.float32(ret + np.float32(g * nt) * v_next) - values[t, n])
            adv[t, n] = np.float32(delta + np.float32(gl * nt) * a_next)
            valid[t, n] = True
    return adv, valid


def brute_force_q_lambda(q_max, rewards, dones, truncated, players, traces, last_q, last_player, gamma):
    T, N, _ = rewards.shape
    target = np.zeros((T, N), np.float32)
    valid = np.zeros((T, N), bool)
    g = np.float32(gamma)
    one = np.float32(1.0)
    for n in range(N):
        for t in reversed(range(T)):
            p = players[t, n]
            u, end = _decision_end(dones, truncated, players, last_player, t, n)
            if end in ("truncated", "unknown"):
                continue
            ret = np.float32(0.0)
            for w in range(u, t - 1, -1):
                ret = np.float32(ret + rewards[w, n, p])
            if end == "terminal":
                target[t, n] = ret
            else:
                if end == "cutoff":
                    c, q_next, g_next = np.float32(0), last_q[n], last_q[n]
                else:
                    c, q_next = traces[u + 1, n], q_max[u + 1, n]
                    g_next = target[u + 1, n] if valid[u + 1, n] else q_next
                boot = np.float32(np.float32((one - c) * q_next) + np.float32(c * g_next))
                target[t, n] = np.float32(ret + np.float32(g * one) * boot)
            valid[t, n] = True
    return target, valid


@pytest.fixture(scope="module")
def rollouts():
    """Training-style rollouts (auto-reset, cut off mid-game) with many truncations."""
    out = []
    for n, horizon, seed in ((2, 60, 0), (3, 45, 1), (2, 5000, 2)):
        env = make("bluff", num_agents=n, horizon=horizon)
        out.append(training_rollout(env, 12, 160, seed))
    return out


def test_per_player_gae_matches_brute_force(nfsp, rollouts):
    per_player_gae = nfsp[1].per_player_gae
    rng = np.random.default_rng(0)
    checked = 0
    for reward, done, truncated, player, last_player in rollouts:
        T, N = done.shape
        # Halves with gamma 1 and lambda 0.5 keep float32 arithmetic exact, so
        # the targets must be identical; with random floats they can differ by
        # rounding (XLA may fuse multiply-adds).
        for values, gamma, lam, atol in (
            (rng.integers(-8, 9, (T, N)).astype(np.float32) / 2, 1.0, 0.5, 0.0),
            (rng.normal(size=(T, N)).astype(np.float32), 0.99, 0.95, 1e-5),
        ):
            last_value = rng.normal(size=N).astype(np.float32)
            adv, targets, valid = per_player_gae(
                jnp.asarray(values), jnp.asarray(reward), jnp.asarray(done),
                jnp.asarray(truncated), jnp.asarray(player), jnp.asarray(last_value),
                jnp.asarray(last_player), gamma, lam,
            )
            ref_adv, ref_valid = brute_force_gae(
                values, reward, done, truncated, player, last_value, last_player, gamma, lam
            )
            np.testing.assert_array_equal(np.asarray(valid), ref_valid)
            np.testing.assert_allclose(np.asarray(adv), ref_adv, rtol=0, atol=atol)
            np.testing.assert_allclose(
                np.asarray(targets)[ref_valid], (ref_adv + values)[ref_valid], rtol=0, atol=atol
            )
            checked += ref_valid.sum()
    # rewards paid to players who aren't acting are a large part of all rewards
    reward, done, truncated, player, _ = rollouts[-1]
    acting = np.take_along_axis(reward, player[..., None], -1)[..., 0]
    assert np.abs(reward).sum() - np.abs(acting).sum() > 0.3 * np.abs(reward).sum()
    assert checked > 5000


def test_per_player_gae_crafted(nfsp):
    """Two players, one env: a caught lie and a truncation.

    Steps (player: action, reward vector):
        0 p0 name rank, 1 p0 size, 2 p0 card, 3 p1 challenges -> [-5, +1],
        4 p1 name rank (truncated here: done), 5 p0 (new game), 6 p0, cut-off.
    """
    rewards = np.zeros((7, 1, 2), np.float32)
    rewards[3, 0] = [-5.0, 1.0]
    dones = np.array([[0], [0], [0], [0], [1], [0], [0]], bool)
    truncated = dones.copy()
    players = np.array([[0], [0], [0], [1], [1], [0], [0]])
    values = np.arange(1, 8, dtype=np.float32)[:, None]
    per_player_gae = nfsp[1].per_player_gae
    adv, targets, valid = per_player_gae(
        values, rewards, dones, truncated, players, jnp.array([10.0]), jnp.array([0]), 1.0, 1.0
    )
    adv, targets, valid = map(np.asarray, (adv, targets, valid))
    # p0's card at step 2 is its last decision of the truncated game: its
    # outcome (the -5 on p1's step 3 and whatever follows) is unknown.
    np.testing.assert_array_equal(valid[:, 0], [1, 1, 0, 1, 0, 1, 1])
    # step 3 (p1) bootstraps from p1's value at step 4 (its last, invalid, decision)
    assert targets[3, 0] == 1.0 + values[4, 0]
    # steps 0, 1 of p0 chain to step 2 (trace cut there)
    assert targets[1, 0] == 0.0 + values[2, 0]
    assert targets[0, 0] == values[2, 0]
    # new game: step 6 bootstraps from the cut-off value, step 5 from step 6
    assert targets[6, 0] == 10.0 and targets[5, 0] == 10.0
    # The same game ending (not truncated) with p1's win bonus at step 4:
    rewards[4, 0] = [0.0, 11.0]
    adv, targets, valid = per_player_gae(
        values, rewards, dones, np.zeros_like(dones), players, jnp.array([10.0]),
        jnp.array([0]), 1.0, 1.0,
    )
    np.testing.assert_array_equal(np.asarray(valid)[:, 0], [1, 1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(np.asarray(targets)[:5, 0], [-5.0, -5.0, -5.0, 12.0, 11.0])


def test_per_player_q_lambda_matches_brute_force(nfsp, rollouts):
    per_player_q_lambda_targets = nfsp[2].per_player_q_lambda_targets
    rng = np.random.default_rng(1)
    for reward, done, truncated, player, last_player in rollouts:
        T, N = done.shape
        for q_max, gamma, lam, atol in (  # exact, then rounding (see the GAE test)
            (rng.integers(-8, 9, (T, N)).astype(np.float32) / 2, 1.0, 0.5, 0.0),
            (rng.normal(size=(T, N)).astype(np.float32), 0.99, 0.9, 1e-5),
        ):
            traces = np.where(rng.random((T, N)) < 0.5, np.float32(lam), np.float32(0))
            last_q = rng.normal(size=N).astype(np.float32)
            targets, valid = per_player_q_lambda_targets(
                jnp.asarray(q_max), jnp.asarray(reward), jnp.asarray(done),
                jnp.asarray(truncated), jnp.asarray(player), jnp.asarray(traces),
                jnp.asarray(last_q), jnp.asarray(last_player), gamma,
            )
            ref_t, ref_v = brute_force_q_lambda(
                q_max, reward, done, truncated, player, traces, last_q, last_player, gamma
            )
            np.testing.assert_array_equal(np.asarray(valid), ref_v)
            np.testing.assert_allclose(np.asarray(targets), ref_t, rtol=0, atol=atol)


def test_per_player_q_lambda_crafted(nfsp):
    """p0 acts three times in a row (rank, size, card) and is caught lying."""
    per_player_q_lambda_targets = nfsp[2].per_player_q_lambda_targets
    rewards = np.zeros((4, 1, 2), np.float32)
    rewards[3, 0] = [-5.0, 1.0]
    dones = np.array([[0], [0], [0], [0]], bool)
    players = np.array([[0], [0], [0], [1]])
    q_max = np.array([[1.0], [2.0], [3.0], [4.0]], np.float32)
    traces = np.full((4, 1), 0.5, np.float32)
    # after step 3 p1 (the challenge winner) leads: cut-off value 7 for p1
    targets, valid = per_player_q_lambda_targets(
        q_max, rewards, dones, dones, players, traces, jnp.array([7.0]), jnp.array([1]), 1.0
    )
    targets, valid = np.asarray(targets), np.asarray(valid)
    # p0's last decision (step 2) is unknown at the cut-off: p0 doesn't act
    # before it; its -5 arrives but its next value doesn't, so it is invalid.
    np.testing.assert_array_equal(valid[:, 0], [1, 1, 0, 1])
    # the same player acts next: bootstrap from its own next Q, not -Q
    assert targets[1, 0] == q_max[2, 0]
    assert targets[0, 0] == 0.5 * q_max[1, 0] + 0.5 * targets[1, 0]
    assert targets[3, 0] == 1.0 + 7.0


def test_reservoir_matches_algorithm_r(nfsp):
    common = nfsp[0]
    capacity, dim = 50, 3
    buffer = common.init_sl_buffer(capacity, dim, 4)
    ref = np.zeros((capacity, dim), np.float32)
    seen = 0
    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(3)
    append = jax.jit(common.reservoir_append)
    size = 32
    for batch in range(12):
        obs = rng.normal(size=(size, dim)).astype(np.float32)
        valid = rng.random(size) < rng.uniform(0.1, 0.9)
        key, k = jax.random.split(key)
        buffer = append(
            buffer, jnp.asarray(obs), jnp.ones((size, 4), bool),
            jnp.arange(size, dtype=jnp.int32), jnp.asarray(valid), k,
        )
        # Algorithm R with the same draws j ~ U{0..k} for the k-th valid item
        v = valid.astype(np.int32)
        k_items = seen + np.cumsum(v) - v
        j = np.asarray(jax.random.randint(k, (size,), 0, jnp.asarray(k_items) + 1, dtype=jnp.int32))
        for i in range(size):
            if not valid[i]:
                continue
            if seen < capacity:
                ref[seen] = obs[i]
            elif j[i] < capacity:
                ref[j[i]] = obs[i]
            seen += 1
        assert int(buffer.seen) == seen and int(buffer.size) == min(seen, capacity)
        np.testing.assert_array_equal(np.asarray(buffer.obs), ref)
    assert seen > 3 * capacity


def test_policy_mixing_is_drawn_once_per_game(nfsp):
    common = nfsp[0]
    env = make("bluff", num_agents=3, horizon=40)
    num_envs, num_steps, eta = 64, 400, 0.3

    @jax.jit
    def run(key):
        k_reset, k_mode, k_play = jax.random.split(key, 3)
        state, _ = jax.vmap(env.reset)(jax.random.split(k_reset, num_envs))
        mode = jax.random.bernoulli(k_mode, eta, (num_envs, 3))

        def one(carry, k):
            s, mode = carry
            k_a, k_s, k_m = jax.random.split(k, 3)
            mask = jax.vmap(env.get_avail_actions)(s)
            a = jax.random.categorical(k_a, jnp.where(mask, 0.0, -jnp.inf), axis=-1)
            ns, _, _, _, done, _ = jax.vmap(env.step)(jax.random.split(k_s, num_envs), s, a)
            return (ns, common.draw_br_mode(k_m, mode, done, eta)), (mode, done)

        _, (modes, dones) = jax.lax.scan(one, (state, mode), jax.random.split(k_play, num_steps))
        return modes, dones

    modes, dones = map(np.asarray, run(KEY))
    changed = (modes[1:] != modes[:-1]).any(-1)
    assert not (changed & ~dones[:-1]).any()  # only after a game ends
    new_games = modes[1:][dones[:-1]]
    assert len(new_games) > 500
    assert abs(new_games.mean() - eta) < 5 * np.sqrt(eta * (1 - eta) / new_games.size)


def test_load_params_is_strict(nfsp, tmp_path):
    common = nfsp[0]
    from flax import serialization
    from bluffjax.networks.mlp import ActorDiscreteMLP, QNetworkDiscreteMLP

    obs = jnp.zeros((378,))
    actor = ActorDiscreteMLP(action_dim=13, hidden_dim=16).init(KEY, obs)
    qnet = QNetworkDiscreteMLP(action_dim=13, hidden_dim=16).init(KEY, obs)
    path = tmp_path / "actor.msgpack"
    path.write_bytes(serialization.to_bytes(actor))
    loaded = common.load_params(str(path), ActorDiscreteMLP(13, 16).init(jax.random.PRNGKey(1), obs))
    jax.tree_util.tree_map(np.testing.assert_array_equal, loaded, actor)
    qpath = tmp_path / "q.msgpack"
    qpath.write_bytes(serialization.to_bytes(qnet))
    with pytest.raises(ValueError):  # extra LayerNorm parameters
        common.load_params(str(qpath), actor)
    with pytest.raises(ValueError):  # hidden size
        common.load_params(str(path), ActorDiscreteMLP(13, 32).init(KEY, obs))
    with pytest.raises(FileNotFoundError):
        common.load_params(str(tmp_path / "missing.msgpack"), actor)


@pytest.mark.parametrize("horizon", [30, 4000])
def test_evaluation_counts_wins_losses_and_draws(nfsp, horizon):
    common = nfsp[0]
    env = make("bluff", num_agents=2, horizon=horizon)

    def random_policy(params, obs, mask, rng):
        return common.sample_random_legal(mask, rng)

    result = jax.jit(
        lambda key: common.play_games(env, random_policy, None, random_policy, None, 64, key)
    )(KEY)
    win, loss, draw = map(np.asarray, (result.win, result.loss, result.draw))
    np.testing.assert_array_equal(win + loss + draw, np.ones(64))
    assert (np.asarray(result.length) <= horizon).all()
    if horizon == 30:  # no random game ends this fast: all draws, not losses
        assert draw.all()
    else:
        assert draw.sum() <= 2 and 10 < win.sum() < 54
    metrics = common.eval_metrics("x", result)
    assert float(metrics["win_rate_x"]) + float(metrics["loss_rate_x"]) + float(
        metrics["draw_rate_x"]
    ) == pytest.approx(1.0)
