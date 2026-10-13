"""Tests of the Bluff game metrics (bluffjax/metrics/bluff.py) and of how the
Bluff NFSP scripts log them and save their half- and full-budget checkpoints.

- scripted claims with env.step (auto-reset, as in training): an unchallenged
  lie, a truthful play that is challenged, final-play lies and truths, a free
  lead, the challenge order with three players, horizon cuts; each event is
  credited to the right player, once, on the right step, and nothing leaks
  into the next game;
- random play: every count equals a step-by-step reference that reads the
  plays from the actions (a lie has a pick with offset != 0, as in the
  scripts' play_action_lie_frac) and the outcomes from info["challenge_status"],
  and the known values of uniform play (recall 1/2, lie success 2^-(n-1),
  precision = share of lies among the claims challenged);
- a tiny training run of each script (CPU): results.jsonl holds the game/ and
  game_br/ metrics of every update, split by the policy of the credited
  player, and half/ and full/ checkpoints that load into the networks, the
  half one being the networks after exactly half the updates.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.environments.bluff.bluff import CHALLENGE, CLAIM, NAME_RANK, PLAY
from bluffjax.metrics import make_metrics, masked_totals
from bluffjax.metrics.bluff import Metrics

KEY = jax.random.PRNGKey(0)
CHALLENGE_ACTION, PASS = 0, 1
NAMES = (
    "plays",
    "lies",
    "final_plays",
    "final_lies",
    "resolved_lies",
    "unchallenged_lies",
    "challenges",
    "caught_lies",
    "lie_decisions",
)


@pytest.fixture(scope="module")
def env2():
    return make("bluff", num_agents=2, horizon=200)


@pytest.fixture(scope="module")
def env3():
    return make("bluff", num_agents=3, horizon=200)


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


def two_player_claim(env, own, pile=None, current_rank=4, **fields):
    """Player 0 to claim (rank current_rank required, no free lead) holding
    `own`; player 1 holds the rest of the deck except `pile`."""
    own, pile = counts(own), counts(pile or {2: 2})
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
        **fields,
    )


def three_player_claim(env):
    """Player 1 to claim three 7s (rank 7 required); players 2 then 0 are
    asked after the play."""
    hands = np.zeros((3, 13), np.float32)
    hands[1] = counts({7: 3, 2: 1, 9: 2})
    hands[0] = counts({7: 1, 3: 4})
    pile = counts({5: 2, 6: 1})
    hands[2] = env.cards_per_rank - hands[0] - hands[1] - pile
    return state_with(
        env,
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


@functools.lru_cache(maxsize=None)
def _jitted(env):
    gm = Metrics(env)
    return jax.jit(env.step), jax.jit(gm.step), gm


def play(env, state, actions, key=KEY):
    """Plays `actions` with env.step (auto-reset) and the metric's step.

    Returns the per-step counts (numpy, name -> (num_agents,)), the done flags
    and the last state."""
    env_step, metric_step, gm = _jitted(env)
    carry = gm.init_carry()
    steps, dones = [], []
    for i, a in enumerate(actions):
        a = jnp.int32(a)
        next_state, _, reward, _, done, info = env_step(
            jax.random.fold_in(key, i), state, a
        )
        carry, c = metric_step(carry, state, a, next_state, reward, done, info)
        steps.append({k: np.asarray(v) for k, v in c.items()})
        dones.append(bool(done))
        state = next_state
    return steps, dones, state


def total(steps):
    return {k: sum(s[k] for s in steps) for k in NAMES}


def rates_of(env, steps):
    """The rates over all players of the counts of `steps`."""
    return Metrics(env).summarize({k: jnp.sum(v) for k, v in total(steps).items()})


def assert_counts(got, num_agents, **want):
    """Every count is as given (per player), and zero where not given."""
    assert set(got) == set(NAMES)
    for name in NAMES:
        expected = np.asarray(want.get(name, np.zeros(num_agents)), np.float32)
        np.testing.assert_array_equal(got[name], expected, err_msg=name)


# =============================================================================
# Scripted claims
# =============================================================================


def test_unchallenged_lie(env2):
    """Claim one 4 playing a 5; the opponent passes."""
    state = two_player_claim(env2, own={5: 1, 7: 2})
    steps, dones, state = play(env2, state, [0, 1, PASS])
    assert not any(dones) and int(state.phase) == CLAIM
    assert_counts(steps[0], 2)  # choosing the size counts nothing
    assert_counts(steps[1], 2, plays=[1, 0], lies=[1, 0])  # the pick
    assert_counts(
        steps[2], 2, resolved_lies=[1, 0], unchallenged_lies=[1, 0], lie_decisions=[0, 1]
    )
    rates = rates_of(env2, steps)
    assert float(rates["lie_rate"]) == 1.0 and float(rates["lie_success"]) == 1.0
    assert float(rates["challenge_recall"]) == 0.0
    assert math.isnan(float(rates["challenge_precision"]))  # no challenge
    assert math.isnan(float(rates["final_play_lie_rate"]))
    assert float(rates["n_plays"]) == 1 and float(rates["n_lie_decisions"]) == 1


def test_truthful_play_challenged(env2):
    """Claim two 4s playing two 4s; the opponent challenges and picks up."""
    state = two_player_claim(env2, own={4: 2, 7: 1})
    steps, dones, _ = play(env2, state, [1, 0, 0, CHALLENGE_ACTION])
    assert not any(dones)
    # counted once, on the last pick of the claim
    assert_counts(steps[1], 2)
    assert_counts(steps[2], 2, plays=[1, 0])
    assert_counts(steps[3], 2, challenges=[0, 1])
    rates = rates_of(env2, steps)
    assert float(rates["lie_rate"]) == 0.0
    assert float(rates["challenge_precision"]) == 0.0
    assert math.isnan(float(rates["lie_success"]))
    assert math.isnan(float(rates["challenge_recall"]))  # no lie to catch


@pytest.mark.parametrize(
    "picks,lie", [((0, 0, 3), True), ((3, 0, 0), True), ((0, 3, 0), True), ((0, 0, 0), False)]
)
def test_any_card_of_another_rank_is_a_lie(env2, picks, lie):
    """A play is a lie as soon as one pick has a non-zero offset (the cards
    play_action_lie_frac counts), wherever it is in the play."""
    state = two_player_claim(env2, own={4: 3, 7: 2})
    steps, _, _ = play(env2, state, [2, *picks, PASS])
    assert_counts(
        total(steps),
        2,
        plays=[1, 0],
        lies=[int(lie), 0],
        resolved_lies=[int(lie), 0],
        unchallenged_lies=[int(lie), 0],
        lie_decisions=[0, int(lie)],
    )


@pytest.mark.parametrize("challenge", [False, True])
def test_final_play_lie(env2, challenge):
    """The last card, a 5 claimed as a 4: unchallenged it wins the game (the
    env resets), caught the liar picks up the pile and play goes on."""
    state = two_player_claim(env2, own={5: 1})
    action = CHALLENGE_ACTION if challenge else PASS
    steps, dones, state = play(env2, state, [0, 1, action])
    assert dones == [False, False, not challenge]
    lie = dict(plays=[1, 0], lies=[1, 0], final_plays=[1, 0], final_lies=[1, 0])
    assert_counts(steps[1], 2, **lie)
    if challenge:
        assert_counts(
            steps[2], 2, resolved_lies=[1, 0], challenges=[0, 1], caught_lies=[0, 1],
            lie_decisions=[0, 1],
        )
    else:
        assert_counts(
            steps[2], 2, resolved_lies=[1, 0], unchallenged_lies=[1, 0], lie_decisions=[0, 1]
        )
        # the next game starts from the reset state: its first steps (rank,
        # size) count nothing
        assert int(state.phase) == NAME_RANK and int(state.timestep) == 0
        more, _, _ = play(env2, state, [0, 0])
        assert_counts(total(more), 2)
    rates = rates_of(env2, steps)
    assert float(rates["final_play_lie_rate"]) == 1.0


def test_final_truthful_play_and_two_card_final_lie(env2):
    state = two_player_claim(env2, own={4: 1})
    steps, dones, _ = play(env2, state, [0, 0, PASS])
    assert dones[-1]
    assert_counts(total(steps), 2, plays=[1, 0], final_plays=[1, 0])
    # two cards, the second of another rank
    state = two_player_claim(env2, own={4: 1, 6: 1})
    steps, _, _ = play(env2, state, [1, 0, 2, CHALLENGE_ACTION])
    assert_counts(
        total(steps), 2, plays=[1, 0], lies=[1, 0], final_plays=[1, 0], final_lies=[1, 0],
        resolved_lies=[1, 0], challenges=[0, 1], caught_lies=[0, 1], lie_decisions=[0, 1],
    )
    # not final: three cards in hand, two played
    state = two_player_claim(env2, own={4: 2, 6: 1})
    steps, _, _ = play(env2, state, [1, 0, 0, PASS])
    assert_counts(total(steps), 2, plays=[1, 0])


def test_free_lead_lie(env2):
    """A free lead names its rank first (not a play); the play is a 0 claimed
    as a 9, caught by the opponent."""
    own = counts({0: 2, 1: 1})
    state = state_with(
        env2, [own, 4.0 - own], np.zeros(13), phase=NAME_RANK, current_player_idx=0,
        challenge_target_idx=0, has_current_rank=False, free_lead=True,
    )
    steps, _, state = play(env2, state, [9, 0, 4, CHALLENGE_ACTION])
    assert int(state.challenge_loser_idx) == 0 and bool(state.challenge_lie)
    assert_counts(steps[0], 2)
    assert_counts(steps[2], 2, plays=[1, 0], lies=[1, 0])
    assert_counts(
        steps[3], 2, resolved_lies=[1, 0], challenges=[0, 1], caught_lies=[0, 1],
        lie_decisions=[0, 1],
    )


@pytest.mark.parametrize(
    "played,decisions,want",
    [
        # a lie: player 2 passes, player 0 catches it; both had the chance
        ((7, 7, 2), [PASS, CHALLENGE_ACTION],
         dict(resolved_lies=[0, 1, 0], challenges=[1, 0, 0], caught_lies=[1, 0, 0],
              lie_decisions=[1, 0, 1])),
        # player 2 challenges first: player 0 is never asked
        ((7, 7, 2), [CHALLENGE_ACTION],
         dict(resolved_lies=[0, 1, 0], challenges=[0, 0, 1], caught_lies=[0, 0, 1],
              lie_decisions=[0, 0, 1])),
        # a lie nobody challenges
        ((2, 7, 7), [PASS, PASS],
         dict(resolved_lies=[0, 1, 0], unchallenged_lies=[0, 1, 0], lie_decisions=[1, 0, 1])),
        # a truthful claim challenged by the second player asked
        ((7, 7, 7), [PASS, CHALLENGE_ACTION], dict(challenges=[1, 0, 0])),
        ((7, 7, 7), [PASS, PASS], {}),
    ],
)
def test_three_players(env3, played, decisions, want):
    state = three_player_claim(env3)
    picks = [(rank - 7) % 13 for rank in played]
    steps, dones, state = play(env3, state, [2, *picks, *decisions])
    assert not any(dones)
    lie = played != (7, 7, 7)
    assert_counts(
        total(steps), 3, plays=[0, 1, 0], lies=[0, int(lie), 0], **want
    )
    # the round is over: the next claim is being made
    assert int(state.phase) in (CLAIM, NAME_RANK)


def test_horizon_cuts(env2):
    """A play completed on the last step of the horizon has no challenge round:
    it is a play (and a lie) but not a resolved lie, and the next game starts
    clean. A challenge round finished on the last step is resolved."""
    horizon = env2.horizon
    state = two_player_claim(env2, own={5: 1, 7: 2}, timestep=horizon - 2)
    steps, dones, state = play(env2, state, [0, 1])
    assert dones == [False, True] and int(state.timestep) == 0
    assert_counts(total(steps), 2, plays=[1, 0], lies=[1, 0])
    more, _, _ = play(env2, state, [0, 0])
    assert_counts(total(more), 2)

    state = two_player_claim(env2, own={5: 1, 7: 2}, timestep=horizon - 3)
    steps, dones, _ = play(env2, state, [0, 1, PASS])
    assert dones == [False, False, True]
    assert_counts(
        total(steps), 2, plays=[1, 0], lies=[1, 0], resolved_lies=[1, 0],
        unchallenged_lies=[1, 0], lie_decisions=[0, 1],
    )


def test_local_credit_matches_base():
    """The metric modules' one-hot credit equals bluffjax.metrics.base.credit."""
    from bluffjax.metrics import base, bluff, werewolf

    for module in (bluff, werewolf):
        for n in (2, 3, 6):
            for player in range(n):
                for value in (1.0, 0.0, True, False, 3, 2.5):
                    got = module.credit(n, jnp.int32(player), value)
                    want = base.credit(n, jnp.int32(player), value)
                    assert got.dtype == want.dtype == jnp.float32
                    np.testing.assert_array_equal(got, want)


def test_registered_and_summary_keys(env3):
    gm = make_metrics("bluff", env3)
    assert isinstance(gm, Metrics) and gm.init_carry() == ()
    zeros = {name: jnp.zeros(()) for name in NAMES}
    rates = gm.summarize(zeros)
    assert set(gm.rate_names) <= set(rates)
    for name in gm.rate_names:
        assert math.isnan(float(rates[name]))
    assert {k for k in rates if k.startswith("n_")} == {
        "n_plays", "n_resolved_lies", "n_challenges", "n_lie_decisions", "n_final_plays"
    }


# =============================================================================
# Random play
# =============================================================================


@functools.lru_cache(maxsize=None)
def random_rollout(num_agents, horizon, num_envs, num_steps, seed):
    """Uniform random legal play with env.step (auto-reset) and the metric's
    step, as in the training scripts. Returns numpy records (T, B, ...)."""
    env = make("bluff", num_agents=num_agents, horizon=horizon)
    gm = Metrics(env)

    @jax.jit
    def run(key):
        k_reset, k_play = jax.random.split(key)
        state, _ = jax.vmap(env.reset)(jax.random.split(k_reset, num_envs))
        carry = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (num_envs,) + jnp.shape(x)), gm.init_carry()
        )

        def one(c, k):
            s, carry = c
            k_a, k_s = jax.random.split(k)
            mask = jax.vmap(env.get_avail_actions)(s)
            a = jax.random.categorical(k_a, jnp.where(mask, 0.0, -jnp.inf), axis=-1)
            ns, _, r, _, done, info = jax.vmap(env.step)(
                jax.random.split(k_s, num_envs), s, a
            )
            carry, c_ = jax.vmap(gm.step)(carry, s, a, ns, r, done, info)
            rec = dict(
                phase=s.phase,
                player=s.current_player_idx,
                hand_sizes=s.agent_hand_sizes,
                action=a,
                done=done,
                truncated=info["truncated"],
                status=info["challenge_status"],
                counts=c_,
            )
            return (ns, carry), rec

        _, rec = jax.lax.scan(one, (state, carry), jax.random.split(k_play, num_steps))
        return rec

    return jax.tree_util.tree_map(np.asarray, run(jax.random.PRNGKey(seed)))


def reference_counts(rec, num_agents):
    """Step-by-step reference: (B, num_agents) totals of every count.

    The claim is followed from its CLAIM step: the claimant, its size, whether
    it empties the hand, and its picks (a lie if any pick has offset != 0).
    Challenge outcomes come from info["challenge_status"] (0: true claim
    challenged, 1: lie caught, 2: nobody challenged, 3: round goes on)."""
    T, B = rec["done"].shape
    out = {name: np.zeros((B, num_agents)) for name in NAMES}
    for b in range(B):
        claim = None
        for t in range(T):
            phase, p, a = rec["phase"][t, b], rec["player"][t, b], rec["action"][t, b]
            status = rec["status"][t, b]
            if phase == CLAIM:
                size = a + 1
                claim = dict(
                    claimant=p, size=size, final=rec["hand_sizes"][t, b, p] == size,
                    picks=[], lie=None,
                )
            elif phase == PLAY:
                assert claim is not None and claim["claimant"] == p
                claim["picks"].append(a)
                if len(claim["picks"]) == claim["size"]:
                    lie = any(x != 0 for x in claim["picks"])
                    claim["lie"] = lie
                    out["plays"][b, p] += 1
                    out["lies"][b, p] += lie
                    out["final_plays"][b, p] += claim["final"]
                    out["final_lies"][b, p] += claim["final"] and lie
            elif phase == CHALLENGE:
                lie, claimant = claim["lie"], claim["claimant"]
                assert p != claimant
                out["lie_decisions"][b, p] += lie
                if a == CHALLENGE_ACTION:
                    assert status == (1 if lie else 0)
                    out["challenges"][b, p] += 1
                    out["caught_lies"][b, p] += lie
                    out["resolved_lies"][b, claimant] += lie
                elif status == 2:
                    out["resolved_lies"][b, claimant] += lie
                    out["unchallenged_lies"][b, claimant] += lie
                else:
                    assert status == 3
            if rec["done"][t, b]:
                claim = None
    return out


# (num_agents, horizon, num_envs, num_steps, seed): short horizons cut many
# games (and challenge rounds), long ones reach the final plays
ROLLOUTS = {
    "2p_cut": (2, 60, 32, 300, 1),
    "3p_cut": (3, 45, 32, 300, 2),
    "2p": (2, 5000, 64, 1500, 9),
    "3p": (3, 5000, 64, 1500, 10),
}


@pytest.mark.parametrize("case", sorted(ROLLOUTS))
def test_random_play_matches_reference(case):
    num_agents, horizon = ROLLOUTS[case][:2]
    rec = random_rollout(*ROLLOUTS[case])
    ref = reference_counts(rec, num_agents)
    for name in NAMES:
        got = rec["counts"][name].sum(0)  # (B, num_agents)
        np.testing.assert_array_equal(got, ref[name], err_msg=name)
    # the rollout has the cases we need
    if horizon < 100:
        assert rec["truncated"].sum() > 50
    else:
        assert ref["final_plays"].sum() > 100 and ref["final_lies"].sum() > 10
    assert 0 < ref["unchallenged_lies"].sum() < ref["resolved_lies"].sum()
    assert 0 < ref["caught_lies"].sum() < ref["challenges"].sum()
    # every credit goes to one player
    for name in NAMES:
        assert set(np.unique(rec["counts"][name].sum(-1))) <= {0.0, 1.0}


@pytest.mark.parametrize("num_agents", [2, 3])
def test_uniform_play_values(num_agents):
    """Uniform random play challenges each claim it is asked about with
    probability 1/2, independently of the claim: recall is 1/2, a lie goes
    unchallenged with probability 2^-(num_agents - 1), and precision is the
    share of lies among the claims decided on. (Long horizon: a challenge
    round cut off at the horizon would bias the resolved lies.)"""
    rec = random_rollout(*ROLLOUTS[f"{num_agents}p"])
    gm = Metrics(make("bluff", num_agents=num_agents))
    totals = masked_totals(
        {k: jnp.asarray(v) for k, v in rec["counts"].items()},
        jnp.ones(rec["counts"]["plays"].shape, bool),
    )
    rates = {k: float(v) for k, v in gm.summarize(totals).items()}
    t = {k: float(v) for k, v in totals.items()}

    def close(value, p, n):
        assert n > 2000, n
        assert abs(value - p) < 5 * math.sqrt(p * (1 - p) / n), (value, p, n)

    close(rates["challenge_recall"], 0.5, t["lie_decisions"])
    close(rates["lie_success"], 0.5 ** (num_agents - 1), t["resolved_lies"])
    decisions = (rec["phase"] == CHALLENGE).sum()
    close(rates["challenge_precision"], t["lie_decisions"] / decisions, t["challenges"])
    # random picks lie on most claims, and most on the final ones (one card
    # of a rank the hand may not hold)
    assert 0.5 < rates["lie_rate"] < 1.0 and t["final_plays"] > 100


# =============================================================================
# Training scripts
# =============================================================================


def _script(name):
    for module in ("distrax", "optax", "hydra", "wandb"):
        pytest.importorskip(module)
    from bluffjax.examples.bluff import bluff_ppo_nfsp, bluff_pqn_nfsp

    return {"bluff_ppo_nfsp": bluff_ppo_nfsp, "bluff_pqn_nfsp": bluff_pqn_nfsp}[name]


def _networks(env, config):
    from bluffjax.networks.mlp import (
        ActorCriticDiscreteMLP,
        ActorDiscreteMLP,
        QNetworkDiscreteMLP,
    )

    A, H = env.action_dim, config["fc_dim_size"]
    if "ppo" in config["job_type"]:
        return ActorCriticDiscreteMLP(A, H), ActorCriticDiscreteMLP(A, H)
    return ActorDiscreteMLP(A, H), QNetworkDiscreteMLP(A, H)


@pytest.mark.parametrize("name", ["bluff_ppo_nfsp", "bluff_pqn_nfsp"])
def test_training_run_logs_metrics_and_checkpoints(name, tmp_path, monkeypatch):
    """See nfsp_run_checks.check_training_run."""
    from bluffjax.tests.nfsp_run_checks import check_training_run

    ppo = "ppo" in name
    records = check_training_run(
        _script(name),
        monkeypatch,
        tmp_path,
        name=name,
        config_name=name.replace("bluff_", "config_"),
        env_id="bluff",
        metrics_cls=Metrics,
        count_names=(
            "n_plays", "n_resolved_lies", "n_challenges", "n_lie_decisions",
            "n_final_plays",
        ),
        networks=_networks,
        opponent_types=("actor_critic", "actor_critic") if ppo else ("actor", "q_network"),
        overrides={
            "compare_episodes" if ppo else "compare_steps": 4,
            "env_kwargs.horizon": 60,
        },
        # games end (by the horizon) within the rollouts
        short_env_kwargs={"horizon": 20},
    )
    for r in records:
        for prefix in ("game", "game_br"):
            if r[f"{prefix}/n_plays"] > 0:
                assert 0.0 <= r[f"{prefix}/lie_rate"] <= 1.0
            else:
                assert math.isnan(r[f"{prefix}/lie_rate"])
    assert sum(r["game/n_plays"] for r in records) > 0
    assert sum(r["game_br/n_plays"] for r in records) > 0
