"""Tests of the Werewolf game metrics (bluffjax/metrics/werewolf.py) and of how
the Werewolf NFSP scripts log them and save their half- and full-budget
checkpoints.

- scripted games with env.step (auto-reset, as in training): a night on which
  the doctor saves the werewolves' target and one on which it doesn't, a day
  vote that removes a werewolf, a seer check followed by the accusation or
  the elimination of that werewolf (and one followed by neither), votes and
  accusations of both teams, games won at a vote by either team and at night
  by the werewolves (the next state is then the next game's), a dead doctor,
  horizon cuts; each event is credited to the right players, once, on the
  step that resolves it;
- random play: every count equals a step-by-step reference that takes the
  night's victim and the day's elimination from the env's own tie-break (the
  same random key), and the values known for uniform play (a save with
  probability 1 / living players, a villager vote or accusation hits a
  werewolf with probability living werewolves / other living players, the
  eliminated player is uniform over the living players, and the expected
  villager votes a living player receives);
- a tiny training run of each script (CPU), see nfsp_run_checks.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.environments.werewolf.werewolf import (
    DOCTOR,
    PHASE_ACCUSE,
    PHASE_NIGHT,
    PHASE_VOTE,
    SEER,
    VILLAGER,
    WEREWOLF,
)
from bluffjax.metrics import make_metrics
from bluffjax.metrics.werewolf import Metrics

KEY = jax.random.PRNGKey(0)
NAMES = (
    "villager_votes",
    "villager_votes_for_werewolf",
    "villager_eliminations",
    "villager_eliminations_of_werewolf",
    "villager_accusations",
    "villager_accusations_of_werewolf",
    "seer_pairs",
    "seer_hits",
    "doctor_nights",
    "doctor_saves",
    "werewolf_days",
    "werewolf_votes_received",
    "villager_days",
    "villager_votes_received",
)
# players 0, 1 werewolves, 2 the doctor, 3 the seer, 4, 5 villagers
ROLES = [WEREWOLF, WEREWOLF, DOCTOR, SEER, VILLAGER, VILLAGER]
NIGHT, ACCUSE, VOTE = PHASE_NIGHT, PHASE_ACCUSE, PHASE_VOTE


@pytest.fixture(scope="module")
def env():
    return make("werewolf", num_agents=6, num_werewolves=2)


def game_state(env, roles=ROLES, alive=None, start=4, **fields):
    """A night about to start (its first living actor to act), with the
    given roles, living players and next day's first speaker search start."""
    state, _ = env.reset(KEY)
    roles = np.asarray(roles)
    night_order = [
        int(np.flatnonzero(roles == DOCTOR)[0]),
        int(np.flatnonzero(roles == SEER)[0]),
        *np.flatnonzero(roles == WEREWOLF).tolist(),
    ]
    alive = np.ones(len(roles), bool) if alive is None else np.asarray(alive, bool)
    slot = next(i for i, p in enumerate(night_order) if alive[p])
    state = state.replace(
        roles=jnp.asarray(roles, jnp.int32),
        alive=jnp.asarray(alive),
        night_order=jnp.asarray(night_order, jnp.int32),
        phase=jnp.int32(PHASE_NIGHT),
        night_subphase=jnp.int32(slot),
        current_player_idx=jnp.int32(night_order[slot]),
        start_player_idx=jnp.int32(start),
    )
    return state.replace(
        **{k: jnp.asarray(v, getattr(state, k).dtype) for k, v in fields.items()}
    )


@functools.lru_cache(maxsize=None)
def _jitted(env):
    gm = Metrics(env)
    return jax.jit(env.step), jax.jit(gm.step), jax.jit(env.get_avail_actions)


def drive(env, state, script, key=KEY):
    """Plays the phases of `script` in order with env.step (auto-reset) and the
    metric's step. script: [(phase, {player: target or None (accuse pass)})];
    the player to act picks its target from the phase's dict.

    Returns the per-step counts (numpy, name -> (num_agents,)), the done
    flags and the last state."""
    env_step, metric_step, avail = _jitted(env)
    n = env.num_agents
    steps, dones = [], []
    for phase, choices in script:
        assert int(state.phase) == phase, (int(state.phase), phase)
        while True:
            cp = int(state.current_player_idx)
            target = choices[cp]
            action = n if target is None else (target - cp) % n
            assert bool(avail(state)[action]), (phase, cp, target)
            a = jnp.int32(action)
            next_state, _, reward, _, done, info = env_step(
                jax.random.fold_in(key, len(steps)), state, a
            )
            _, c = metric_step((), state, a, next_state, reward, done, info)
            steps.append({k: np.asarray(v) for k, v in c.items()})
            dones.append(bool(done))
            state = next_state
            if done or int(state.phase) != phase:
                break
    return steps, dones, state


def total(steps):
    return {k: sum(s[k] for s in steps) for k in NAMES}


def assert_counts(got, **want):
    """Every count is as given (per player), and zero where not given."""
    assert set(got) == set(NAMES)
    for name in NAMES:
        expected = np.asarray(want.get(name, np.zeros(6)), np.float32)
        np.testing.assert_array_equal(got[name], expected, err_msg=name)


def rates_of(steps):
    return Metrics(make("werewolf")).summarize(
        {k: jnp.sum(v) for k, v in total(steps).items()}
    )


# =============================================================================
# Scripted games
# =============================================================================

# Night 1: the doctor protects 5, the werewolves' target: saved; the seer
# checks werewolf 0. Day 1: 0 is accused by the seer and villager 4, and
# voted out by every human (the werewolves vote for 4).
NIGHT_1_SAVE = (NIGHT, {2: 5, 3: 0, 0: 5, 1: 5})
DAY_1 = [
    (ACCUSE, {4: 0, 5: None, 0: 4, 1: None, 2: 3, 3: 0}),
    (VOTE, {4: 0, 5: 0, 0: 4, 1: 4, 2: 0, 3: 0}),
]


def test_save_vote_out_and_seer_info(env):
    script = [
        NIGHT_1_SAVE,
        *DAY_1,
        # Night 2: werewolf 1 kills 4 (the doctor protects itself); the seer
        # checks 1. Day 2: nobody accuses 1, the humans vote it out: they win.
        (NIGHT, {2: 2, 3: 1, 1: 4}),
        (ACCUSE, {5: None, 1: 2, 2: None, 3: None}),
        (VOTE, {5: 1, 1: 2, 2: 1, 3: 1}),
    ]
    steps, dones, state = drive(env, game_state(env), script)
    assert dones[-1] and not any(dones[:-1])
    assert int(state.timestep) == 0  # the next game
    # the night is resolved on its last step (werewolf 1's pick), the day on
    # the last vote
    assert_counts(steps[2])
    assert_counts(steps[3], doctor_nights=[0, 0, 1, 0, 0, 0], doctor_saves=[0, 0, 1, 0, 0, 0])
    humans_day_1 = [0, 0, 1, 1, 1, 1]
    humans_day_2 = [0, 0, 1, 1, 0, 1]
    both = np.add(humans_day_1, humans_day_2)
    assert_counts(
        total(steps),
        villager_votes=both,
        villager_votes_for_werewolf=both,
        villager_eliminations=both,
        villager_eliminations_of_werewolf=both,
        villager_accusations=[0, 0, 1, 1, 1, 0],
        villager_accusations_of_werewolf=[0, 0, 0, 1, 1, 0],
        # day 1: 0 identified and accused (and voted out); day 2: 1
        # identified the night before, voted out without being accused
        seer_pairs=[0, 0, 0, 2, 0, 0],
        seer_hits=[0, 0, 0, 2, 0, 0],
        doctor_nights=[0, 0, 2, 0, 0, 0],
        doctor_saves=[0, 0, 1, 0, 0, 0],
        werewolf_days=[1, 2, 0, 0, 0, 0],
        werewolf_votes_received=[4, 3, 0, 0, 0, 0],
        villager_days=both,
    )
    rates = rates_of(steps)
    assert float(rates["doctor_save_rate"]) == 0.5
    assert float(rates["elimination_accuracy"]) == 1.0
    assert float(rates["seer_info_use"]) == 1.0
    assert float(rates["werewolf_votes_received"]) == pytest.approx(7 / 3)
    assert float(rates["villager_votes_received"]) == 0.0


def test_werewolves_win_at_the_vote(env):
    """The seer's check isn't used; a villager votes for a human; the doctor
    is voted out, which gives the werewolves parity (the env resets)."""
    script = [
        (NIGHT, {2: 2, 3: 1, 0: 5, 1: 5}),  # 5 dies, the seer checks 1
        (ACCUSE, {4: 2, 0: None, 1: None, 2: None, 3: 0}),
        (VOTE, {4: 2, 0: 2, 1: 2, 2: 0, 3: 0}),
    ]
    steps, dones, state = drive(env, game_state(env), script)
    assert dones[-1] and int(state.timestep) == 0 and bool(state.alive.all())
    assert_counts(
        total(steps),
        villager_votes=[0, 0, 1, 1, 1, 0],
        villager_votes_for_werewolf=[0, 0, 1, 1, 0, 0],
        villager_eliminations=[0, 0, 1, 1, 1, 0],
        villager_accusations=[0, 0, 0, 1, 1, 0],
        villager_accusations_of_werewolf=[0, 0, 0, 1, 0, 0],
        seer_pairs=[0, 0, 0, 1, 0, 0],  # 1 neither accused nor voted out
        doctor_nights=[0, 0, 1, 0, 0, 0],
        werewolf_days=[1, 1, 0, 0, 0, 0],
        werewolf_votes_received=[2, 0, 0, 0, 0, 0],
        villager_days=[0, 0, 1, 1, 1, 0],
        villager_votes_received=[0, 0, 1, 0, 0, 0],  # 4's vote for 2
    )
    rates = rates_of(steps)
    assert float(rates["elimination_accuracy"]) == 0.0
    assert float(rates["seer_info_use"]) == 0.0
    assert float(rates["villager_vote_accuracy"]) == pytest.approx(2 / 3)
    assert float(rates["accusation_accuracy"]) == 0.5


def test_werewolves_win_at_night(env):
    """The kill that gives the werewolves parity is not a save, although the
    next state (the next game's) has everyone alive."""
    state = game_state(env, alive=[1, 1, 1, 0, 1, 1])  # the seer is dead
    steps, dones, state = drive(env, state, [(NIGHT, {2: 2, 0: 5, 1: 5})])
    assert dones == [False, False, True] and bool(state.alive.all())
    assert_counts(total(steps), doctor_nights=[0, 0, 1, 0, 0, 0])


def test_dead_doctor_has_no_nights(env):
    state = game_state(env, alive=[1, 0, 0, 1, 1, 1])  # werewolf 1 and the doctor dead
    assert int(state.current_player_idx) == 3  # the seer acts first
    steps, dones, state = drive(env, state, [(NIGHT, {3: 0, 0: 4})])
    assert not any(dones) and not bool(state.alive[4])
    assert_counts(total(steps))


def test_split_werewolf_picks(env):
    """The werewolves pick 4 and 5 and the env breaks the tie at random; the
    doctor protects 4: it is a save exactly when the env's target was 4."""
    outcomes = set()
    for i in range(16):
        steps, _, state = drive(
            env, game_state(env), [(NIGHT, {2: 4, 3: 0, 0: 4, 1: 5})],
            key=jax.random.PRNGKey(i),
        )
        saved = bool(state.alive.all())
        assert saved or not bool(state.alive[5])
        assert_counts(
            total(steps), doctor_nights=[0, 0, 1, 0, 0, 0],
            doctor_saves=[0, 0, int(saved), 0, 0, 0],
        )
        outcomes.add(saved)
    assert outcomes == {True, False}


def test_horizon_cuts(env):
    """A vote or a night resolved on the last step of the horizon has an
    unknown outcome (the next state is the next game's and nobody won): its
    elimination, seer pairs and doctor night are left out; votes,
    accusations and the votes received that day are known."""
    horizon = env.horizon
    # 4 night steps, 6 accusations, 6 votes: the last vote at the horizon
    state = game_state(env, timestep=horizon - 16)
    steps, dones, _ = drive(env, state, [NIGHT_1_SAVE, *DAY_1])
    assert len(steps) == 16 and dones == [False] * 15 + [True]
    humans = [0, 0, 1, 1, 1, 1]
    assert_counts(
        total(steps),
        villager_votes=humans,
        villager_votes_for_werewolf=humans,
        villager_accusations=[0, 0, 1, 1, 1, 0],
        villager_accusations_of_werewolf=[0, 0, 0, 1, 1, 0],
        doctor_nights=[0, 0, 1, 0, 0, 0],
        doctor_saves=[0, 0, 1, 0, 0, 0],
        werewolf_days=[1, 1, 0, 0, 0, 0],
        werewolf_votes_received=[4, 0, 0, 0, 0, 0],
        villager_days=humans,
    )
    # the night's last step at the horizon
    state = game_state(env, timestep=horizon - 4)
    steps, dones, _ = drive(env, state, [NIGHT_1_SAVE])
    assert dones == [False, False, False, True]
    assert_counts(total(steps))


def test_registered_and_summary_keys(env):
    gm = make_metrics("werewolf", env)
    assert isinstance(gm, Metrics) and gm.init_carry() == ()
    rates = gm.summarize({name: jnp.zeros(()) for name in NAMES})
    assert set(gm.rate_names) <= set(rates)
    assert "accusation_accuracy" in gm.rate_names
    for name in gm.rate_names:
        assert math.isnan(float(rates[name]))
    assert {k for k in rates if k.startswith("n_")} == {
        "n_villager_votes", "n_villager_eliminations", "n_villager_accusations",
        "n_seer_pairs", "n_doctor_nights", "n_werewolf_days", "n_villager_days",
    }


# =============================================================================
# Random play
# =============================================================================


@functools.lru_cache(maxsize=None)
def random_rollout(n, k, num_envs, num_steps, seed):
    """Uniform random legal play with env.step (auto-reset) and the metric's
    step, as in the training scripts. Returns numpy records (T, B, ...),
    with the key passed to env.step."""
    env = make("werewolf", num_agents=n, num_werewolves=k)
    gm = Metrics(env)

    @jax.jit
    def run(key):
        k_reset, k_play = jax.random.split(key)
        state, _ = jax.vmap(env.reset)(jax.random.split(k_reset, num_envs))

        def one(s, key_t):
            k_a, k_s = jax.random.split(key_t)
            mask = jax.vmap(env.get_avail_actions)(s)
            a = jax.random.categorical(k_a, jnp.where(mask, 0.0, -jnp.inf), axis=-1)
            keys = jax.random.split(k_s, num_envs)
            ns, _, r, _, done, info = jax.vmap(env.step)(keys, s, a)
            _, c = jax.vmap(gm.step)((), s, a, ns, r, done, info)
            return ns, dict(state=s, action=a, key=keys, done=done, info=info, counts=c)

        _, rec = jax.lax.scan(one, state, jax.random.split(k_play, num_steps))
        return rec

    return env, jax.tree_util.tree_map(np.asarray, run(jax.random.PRNGKey(seed)))


def _resolutions(env, rec):
    """Every night and vote resolved in the rollout, with the env's choice:
    (t, b, kind, choice). The choice is recomputed with the env's tie-break
    (_most_chosen) and the key env.step passed to step_env."""
    s, n = rec["state"], env.num_agents
    T, B = rec["done"].shape
    events, counts, starts, keys = [], [], [], []
    for t in range(T):
        for b in range(B):
            phase, cp, a = s.phase[t, b], s.current_player_idx[t, b], rec["action"][t, b]
            target = (cp + a) % n if a < n else -1
            alive = s.alive[t, b]
            if phase == PHASE_NIGHT:
                slot, order = s.night_subphase[t, b], s.night_order[t, b]
                if any(alive[order[j]] for j in range(slot + 1, len(order))):
                    continue
                picks = s.werewolf_targets[t, b].copy()
                picks[slot - 2] = target  # the last night actor is a werewolf
                c = np.zeros(n, np.float32)
                for w, pick in zip(order[2:], picks):
                    if alive[w]:
                        c[pick] += 1
                kind = "night"
            elif phase == PHASE_VOTE and s.phase_progress[t, b] + 1 == alive.sum():
                votes = s.votes[t, b].copy()
                votes[cp] = target
                c = np.bincount(votes[alive], minlength=n).astype(np.float32)
                kind = "vote"
            else:
                continue
            events.append((t, b, kind))
            counts.append(c)
            starts.append(s.start_player_idx[t, b])
            keys.append(rec["key"][t, b])
    step_keys = jax.vmap(lambda key: jax.random.split(key)[0])(jnp.asarray(np.stack(keys)))
    choice = jax.vmap(env._most_chosen)(
        jnp.asarray(np.stack(counts)), step_keys, jnp.asarray(np.stack(starts))
    )
    ties = (np.stack(counts) == np.stack(counts).max(-1, keepdims=True)).sum(-1) > 1
    return [(*e, int(c), tie) for e, c, tie in zip(events, np.asarray(choice), ties)]


def reference_counts(env, rec):
    """Step-by-step reference: (B, num_agents) totals of every count."""
    s, n = rec["state"], env.num_agents
    T, B = rec["done"].shape
    out = {name: np.zeros((B, n)) for name in NAMES}
    for t in range(T):
        for b in range(B):
            phase, cp, a = s.phase[t, b], s.current_player_idx[t, b], rec["action"][t, b]
            target = (cp + a) % n if a < n else -1
            werewolf = s.roles[t, b] == WEREWOLF
            if phase == PHASE_VOTE and not werewolf[cp]:
                out["villager_votes"][b, cp] += 1
                out["villager_votes_for_werewolf"][b, cp] += werewolf[target]
            if phase == PHASE_ACCUSE and not werewolf[cp] and target >= 0:
                out["villager_accusations"][b, cp] += 1
                out["villager_accusations_of_werewolf"][b, cp] += werewolf[target]
    for t, b, kind, choice, _ in _resolutions(env, rec):
        roles, alive = s.roles[t, b], s.alive[t, b]
        werewolf = roles == WEREWOLF
        if kind == "night":
            doctor = int(np.flatnonzero(roles == DOCTOR)[0])
            if alive[doctor]:
                out["doctor_nights"][b, doctor] += 1
                out["doctor_saves"][b, doctor] += choice == s.doctor_target[t, b]
            continue
        cp, a = s.current_player_idx[t, b], rec["action"][t, b]
        votes = s.votes[t, b].copy()
        votes[cp] = (cp + a) % n
        for p in range(n):
            if not alive[p]:
                continue
            received = sum(
                alive[v] and not werewolf[v] and votes[v] == p for v in range(n)
            )
            if werewolf[p]:
                out["werewolf_days"][b, p] += 1
                out["werewolf_votes_received"][b, p] += received
            else:
                out["villager_days"][b, p] += 1
                out["villager_votes_received"][b, p] += received
                out["villager_eliminations"][b, p] += 1
                out["villager_eliminations_of_werewolf"][b, p] += werewolf[choice]
        seer = int(np.flatnonzero(roles == SEER)[0])
        for w in range(n):
            if alive[w] and werewolf[w] and s.seer_results[t, b, w] == 1:
                out["seer_pairs"][b, seer] += 1
                accused = (s.accusations[t, b] == w).any()
                out["seer_hits"][b, seer] += accused or choice == w
    return out


ROLLOUTS = {"6p2w": (6, 2, 96, 400, 0), "8p3w": (8, 3, 64, 400, 1), "5p1w": (5, 1, 64, 300, 2)}


@pytest.mark.parametrize("case", sorted(ROLLOUTS))
def test_random_play_matches_reference(case):
    env, rec = random_rollout(*ROLLOUTS[case])
    ref = reference_counts(env, rec)
    for name in NAMES:
        np.testing.assert_array_equal(rec["counts"][name].sum(0), ref[name], err_msg=name)
    # the rollout has the cases we need: saves and kills, ties, games won at
    # a vote by each team and at night, seer hits and misses
    events = _resolutions(env, rec)
    s, done = rec["state"], rec["done"]
    winner = rec["info"]["game_winner"].astype(bool)
    humans_won = (winner & (s.roles != WEREWOLF)).any(-1)
    ends = {(kind, bool(humans_won[t, b])) for t, b, kind, _, _ in events if done[t, b]}
    assert ends == {("vote", True), ("vote", False), ("night", False)}
    assert any(tie for _, _, kind, _, tie in events if kind == "vote")
    assert 0 < ref["doctor_saves"].sum() < ref["doctor_nights"].sum()
    assert 0 < ref["seer_hits"].sum() < ref["seer_pairs"].sum()
    assert ref["villager_votes_received"].sum() > 0


def test_uniform_play_values():
    """Known values of uniform random play (see the module docstring), summed
    over the rollout as expected counts given the state at each event."""
    env, rec = random_rollout(*ROLLOUTS["6p2w"])
    s, c = rec["state"], rec["counts"]
    werewolf = s.roles == WEREWOLF
    alive = s.alive
    A = alive.sum(-1)  # living players (T, B)
    W = (alive & werewolf).sum(-1)  # living werewolves
    H = A - W
    actor_werewolf = np.take_along_axis(werewolf, s.current_player_idx[..., None], -1)[..., 0]

    def close(observed, p, n=None):
        """observed: total of Bernoulli events with probabilities p."""
        p = np.asarray(p, np.float64)
        mean, sd = p.sum(), math.sqrt((p * (1 - p)).sum())
        assert p.size > 1000
        assert abs(observed - mean) < 5 * sd, (observed, mean, sd)

    # votes and accusations of villager-team players
    vote = (s.phase == PHASE_VOTE) & ~actor_werewolf
    assert c["villager_votes"].sum() == vote.sum()
    close(c["villager_votes_for_werewolf"].sum(), (W / (A - 1))[vote])
    accusation = c["villager_accusations"].sum(-1) > 0
    close(c["villager_accusations_of_werewolf"].sum(), (W / (A - 1))[accusation])
    # doctor: protects a uniform living player, independently of the target
    night = c["doctor_nights"].sum(-1) > 0
    close(c["doctor_saves"].sum(), (1 / A)[night])
    # eliminations: uniform over the living players
    day = c["werewolf_days"].sum(-1) > 0
    eliminated_werewolf = c["villager_eliminations_of_werewolf"].sum(-1)[day] / H[day]
    close(eliminated_werewolf.sum(), (W / A)[day])
    # villager votes received per (day, living player): each living villager
    # (other than the player) votes for it with probability 1 / (A - 1)
    for team, voters, days in (
        ("werewolf", H, W),
        ("villager", H - 1, H),
    ):
        p = np.repeat((1 / (A - 1))[day], (voters * days)[day])
        close(c[f"{team}_votes_received"].sum(), p)
        assert c[f"{team}_days"].sum() == days[day].sum()


# =============================================================================
# Training scripts
# =============================================================================


def _script(name):
    for module in ("distrax", "optax", "hydra", "wandb"):
        pytest.importorskip(module)
    from bluffjax.examples.werewolf import werewolf_ppo_nfsp, werewolf_pqn_nfsp

    return {"werewolf_ppo_nfsp": werewolf_ppo_nfsp, "werewolf_pqn_nfsp": werewolf_pqn_nfsp}[
        name
    ]


def _networks(env, config):
    from bluffjax.networks.mlp import (
        ActorCriticDiscreteMLP,
        ActorDiscreteMLP,
        QNetworkDiscreteMLP,
    )

    A, H = env.num_actions, config["fc_dim_size"]
    if "ppo" in config["job_type"]:
        return ActorCriticDiscreteMLP(A, H), ActorCriticDiscreteMLP(A, H)
    return ActorDiscreteMLP(A, H), QNetworkDiscreteMLP(A, H)


@pytest.mark.parametrize("name", ["werewolf_ppo_nfsp", "werewolf_pqn_nfsp"])
def test_training_run_logs_metrics_and_checkpoints(name, tmp_path, monkeypatch):
    """See nfsp_run_checks.check_training_run."""
    from bluffjax.tests.nfsp_run_checks import check_training_run

    ppo = "ppo" in name
    records = check_training_run(
        _script(name),
        monkeypatch,
        tmp_path,
        name=name,
        config_name=name.replace("werewolf_", "config_"),
        env_id="werewolf",
        metrics_cls=Metrics,
        count_names=(
            "n_villager_votes", "n_villager_eliminations", "n_villager_accusations",
            "n_seer_pairs", "n_doctor_nights", "n_werewolf_days", "n_villager_days",
        ),
        networks=_networks,
        opponent_types=("actor_critic", "actor_critic") if ppo else ("actor", "q_network"),
        overrides={"compare_episodes" if ppo else "compare_steps": 4},
        short_env_kwargs={},  # games end within ~40 steps
    )
    for r in records:
        for prefix in ("game", "game_br"):
            for rate, count in (
                ("villager_vote_accuracy", "n_villager_votes"),
                ("doctor_save_rate", "n_doctor_nights"),
            ):
                if r[f"{prefix}/{count}"] > 0:
                    assert 0.0 <= r[f"{prefix}/{rate}"] <= 1.0
                else:
                    assert math.isnan(r[f"{prefix}/{rate}"])
    assert sum(r["game/n_villager_votes"] for r in records) > 0
    assert sum(r["game_br/n_villager_votes"] for r in records) > 0
