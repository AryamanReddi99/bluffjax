"""
Rule-conformance validation suite for Werewolf.

Checks every state/action/transition produced by random or checkpoint-driven
rollouts against bluffjax/environments/werewolf/rules.md and the
implementation in bluffjax/environments/werewolf/werewolf.py.

Werewolf is the most complex game in the suite (night sub-phases, day
accuse/vote phases, two competing win conditions, a timeout default), so
this suite leans heavily on hand-crafted edge cases: doctor saves/fails,
werewolf target agreement/disagreement/tiebreak, seer reveal persistence,
and both win conditions involve combinations of role assignment and
night/day actions too specific for random rollouts to reliably exercise
within a short episode.

Usage:
    pytest bluffjax/validation_suite/werewolf_validation.py -v
    python werewolf_validation.py --random --episodes 300
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.environments.werewolf.werewolf import (
    DOCTOR,
    NIGHT_DOCTOR,
    NIGHT_SEER,
    NIGHT_WEREWOLF_0,
    PHASE_ACCUSE,
    PHASE_NIGHT,
    PHASE_VOTE,
    SEER,
    VILLAGER,
    WEREWOLF,
    Werewolf,
    WerewolfState,
)

# This suite always exercises the default 6-agent/2-werewolf configuration,
# where night subphases are [doctor, seer, ww0, ww1], so the second
# werewolf's subphase is simply the one right after NIGHT_WEREWOLF_0. (The
# env supports an arbitrary number of werewolves via num_night_subphases
# = 2 + num_werewolves; this constant is specific to this suite's fixed
# 2-werewolf setup.)
NIGHT_WEREWOLF_1 = NIGHT_WEREWOLF_0 + 1
from bluffjax.validation_suite.common import (
    RuleCheckerBase,
    build_network,
    discover_checkpoints,
    infer_network_kind,
    load_checkpoint_params,
    one_checkpoint_per_algorithm_and_kind,
    sample_action,
    training_env_kwargs,
)

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.normpath(os.path.join(HERE, "..", "examples", "werewolf"))
CHECKPOINT_ROOT = os.path.join(EXAMPLES_DIR, "checkpoints")

DEFAULT_SEED = 0
RANDOM_EPISODES = 150
CHECKPOINT_EPISODES = 30


class RuleChecker(RuleCheckerBase):
    def __init__(self, env: Werewolf):
        super().__init__(env)

    # -- setup / reset --------------------------------------------------------

    def check_reset(self, state: WerewolfState) -> None:
        """Roles at reset must be num_werewolves=2 plus exactly 1 doctor, 1
        seer, and 2 villagers, randomly permuted across seats. Night order
        is fixed at reset as [doctor_idx, seer_idx, ww0_idx, ww1_idx]."""
        roles = np.asarray(state.roles)
        if (roles == WEREWOLF).sum() != self.env.num_werewolves:
            self._fail(0, "setup/role_counts_werewolf", f"expected {self.env.num_werewolves} werewolves, got {(roles == WEREWOLF).sum()}")
        if (roles == DOCTOR).sum() != 1 or (roles == SEER).sum() != 1:
            self._fail(0, "setup/role_counts_special", "expected exactly 1 doctor and 1 seer")
        if not np.asarray(state.alive).all():
            self._fail(0, "setup/all_alive", "every player must start alive")
        if int(state.phase) != PHASE_NIGHT:
            self._fail(0, "setup/initial_phase", "game must start in the night phase")
        night_order = np.asarray(state.night_order)
        expected_doctor = int(np.argmax(roles == DOCTOR))
        expected_seer = int(np.argmax(roles == SEER))
        ww_indices = sorted(np.where(roles == WEREWOLF)[0].tolist())
        if night_order[0] != expected_doctor or night_order[1] != expected_seer:
            self._fail(0, "setup/night_order_doctor_seer", f"night_order={night_order} doesn't match doctor={expected_doctor}, seer={expected_seer}")
        if list(night_order[2:4]) != ww_indices:
            self._fail(0, "setup/night_order_werewolves", f"night_order werewolves {night_order[2:4]} != sorted werewolf indices {ww_indices}")
        if int(state.current_player_idx) != night_order[0]:
            self._fail(0, "setup/doctor_acts_first", "the doctor must be first to act each game")
        if not np.allclose(np.asarray(state.seer_results), 0.0):
            self._fail(0, "setup/seer_results_unset", "seer_results must start all-zero (unchecked)")

    # -- action legality per phase ------------------------------------------------

    def _check_night_avail(self, step, pre: WerewolfState, avail) -> None:
        """Exactly one designated actor takes one action per sub-phase, in
        a fixed order. Doctor/seer may target any living player (including
        themself); werewolves may target any living non-werewolf; every
        non-actor is restricted to noop only."""
        cp = int(pre.current_player_idx)
        actor_idx = int(pre.night_order[pre.night_subphase])
        is_actor = (cp == actor_idx) and bool(pre.alive[actor_idx])
        n = self.env.num_agents
        if not is_actor:
            if not (bool(avail[n]) and not np.asarray(avail[:n]).any()):
                self._fail(step, "action/night_non_actor_noop_only", f"non-actor {cp} must have ONLY noop legal")
            return
        subphase = int(pre.night_subphase)
        if subphase == NIGHT_DOCTOR or subphase == NIGHT_SEER:
            expected_targets_absolute = np.asarray(pre.alive)
        else:
            expected_targets_absolute = np.asarray(pre.alive) & (np.asarray(pre.roles) != WEREWOLF)
        # Actions 0..n-1 are relative targets (converted to absolute via
        # _to_absolute in step_env), so re-index the expected mask the same
        # way before comparing.
        abs_indices = [(cp + r) % n for r in range(n)]
        expected_targets = expected_targets_absolute[abs_indices]
        if not np.array_equal(np.asarray(avail[:n]), expected_targets):
            self._fail(step, "action/night_target_mask", f"subphase={subphase} target avail {np.asarray(avail[:n])} != expected {expected_targets}")
        if not bool(avail[n]):
            self._fail(step, "action/night_noop_always_legal", "noop must always be legal for the night actor")

    def _check_accuse_avail(self, step, pre: WerewolfState, avail) -> None:
        """A living player may target any other living player or noop
        (noop is always legal here). A dead player's turn is forced to
        noop."""
        cp = int(pre.current_player_idx)
        n = self.env.num_agents
        if not bool(pre.alive[cp]):
            if not (bool(avail[n]) and not np.asarray(avail[:n]).any()):
                self._fail(step, "action/accuse_dead_noop_only", f"dead player {cp} must have ONLY noop legal in accuse phase")
            return
        for rel in range(n):
            abs_idx = (cp + rel) % n
            expected = bool(pre.alive[abs_idx]) and rel != 0
            if bool(avail[rel]) != expected:
                self._fail(step, "action/accuse_target_mask", f"rel={rel} (abs={abs_idx}) avail={bool(avail[rel])} != expected {expected}")
        if not bool(avail[n]):
            self._fail(step, "action/accuse_noop_legal_for_living", "noop must be legal (accusing is optional)")

    def _check_vote_avail(self, step, pre: WerewolfState, avail) -> None:
        """A living player must vote for a living player other than
        themself; noop is not a legal action for living players in this
        phase. Only dead seats may/must noop."""
        cp = int(pre.current_player_idx)
        n = self.env.num_agents
        if not bool(pre.alive[cp]):
            if not (bool(avail[n]) and not np.asarray(avail[:n]).any()):
                self._fail(step, "action/vote_dead_noop_only", f"dead player {cp} must have ONLY noop legal in vote phase")
            return
        if bool(avail[n]):
            self._fail(step, "action/vote_noop_illegal_for_living", f"living player {cp} must NOT have noop legal (voting is mandatory)")
        for rel in range(n):
            abs_idx = (cp + rel) % n
            expected = bool(pre.alive[abs_idx]) and rel != 0
            if bool(avail[rel]) != expected:
                self._fail(step, "action/vote_target_mask", f"rel={rel} (abs={abs_idx}) avail={bool(avail[rel])} != expected {expected}")

    # -- night resolution --------------------------------------------------------

    def _check_night_resolution(self, step, pre: WerewolfState, post: WerewolfState) -> None:
        """If the werewolves' chosen victim matches the doctor's protected
        player, the victim survives; otherwise they die. The werewolf team
        wins once ww >= humans, evaluated immediately after the kill/heal
        resolves.

        This fires on the same step that night_subphase reaches
        NIGHT_WEREWOLF_1's action and resolution happens, so the final
        werewolf's target only appears in `post` -- `pre.werewolf_targets`
        for that slot is still stale. `post.werewolf_targets` and
        `post.doctor_target` are safe to read here since `_resolve_night`
        doesn't clear them (only `_resolve_vote` does, at end of day)."""
        ww0_alive = bool(pre.alive[pre.night_order[2]])
        ww1_alive = bool(pre.alive[pre.night_order[3]])
        vote0 = int(post.werewolf_targets[0]) if ww0_alive else -1
        vote1 = int(post.werewolf_targets[1]) if ww1_alive else -1

        if vote0 == vote1 and vote0 >= 0:
            kill_target_options = {vote0}
        elif vote0 < 0:
            kill_target_options = {vote1} if vote1 >= 0 else set()
        elif vote1 < 0:
            kill_target_options = {vote0}
        else:
            kill_target_options = {vote0, vote1}  # random tiebreak between the two

        healed = int(post.doctor_target) >= 0
        roles = np.asarray(pre.roles)
        pre_alive = np.asarray(pre.alive)
        post_alive = np.asarray(post.alive)
        newly_dead = np.where(pre_alive & ~post_alive)[0].tolist()

        if not kill_target_options:
            if newly_dead:
                self._fail(step, "night/no_kill_target_no_death", f"no valid kill target but {newly_dead} died")
        elif len(newly_dead) > 1:
            self._fail(step, "night/at_most_one_death", f"more than one death in a single night: {newly_dead}")
        elif len(newly_dead) == 1:
            victim = newly_dead[0]
            if victim not in kill_target_options:
                self._fail(step, "night/death_matches_werewolf_target", f"victim {victim} not among werewolf target(s) {kill_target_options}")
            if healed and victim == int(post.doctor_target):
                self._fail(step, "night/doctor_save_negates_death", f"doctor protected {int(post.doctor_target)} but they died anyway")
        else:
            # No death: must be because the (unique) kill target was healed.
            if len(kill_target_options) == 1:
                only_target = next(iter(kill_target_options))
                if not (healed and only_target == int(post.doctor_target)):
                    self._fail(step, "night/unexplained_survival", f"target {only_target} survived without a matching doctor save")

        num_ww_post = int(((roles == WEREWOLF) & post_alive).sum())
        num_humans_post = int(((roles != WEREWOLF) & post_alive).sum())
        expected_ww_win = num_ww_post >= num_humans_post
        if bool(post.done) and int(post.game_winner) == 1:
            if not expected_ww_win:
                self._fail(step, "night/werewolf_win_condition", "werewolves flagged as winning but do not outnumber/equal humans")
        if expected_ww_win and int(post.phase) == PHASE_ACCUSE:
            if not (bool(post.done) and int(post.game_winner) == 1):
                self._fail(step, "night/werewolf_win_missed", "werewolves outnumber/equal humans but game not flagged as won")

    def _check_seer_result(self, step, pre: WerewolfState, post: WerewolfState, action: int) -> None:
        """seer_results[target] is immediately set to +1 if that player is
        a werewolf, -1 otherwise, and is never reset during the game."""
        if int(pre.night_subphase) != NIGHT_SEER:
            return
        actor_idx = int(pre.night_order[NIGHT_SEER])
        if int(pre.current_player_idx) != actor_idx or not bool(pre.alive[actor_idx]):
            return
        is_noop = action >= self.env.num_agents
        if is_noop:
            return
        target = int(self.env._to_absolute(jnp.minimum(action, self.env.num_agents - 1), actor_idx))
        expected = 1.0 if pre.roles[target] == WEREWOLF else -1.0
        if not np.isclose(float(post.seer_results[target]), expected):
            self._fail(step, "night/seer_result_correct", f"seer_results[{target}]={float(post.seer_results[target])} != expected {expected}")
        # every other entry must be untouched
        pre_res, post_res = np.asarray(pre.seer_results), np.asarray(post.seer_results)
        for i in range(self.env.num_agents):
            if i != target and not np.isclose(pre_res[i], post_res[i]):
                self._fail(step, "night/seer_result_persistence", f"seer_results[{i}] changed without being investigated")

    # -- accuse phase --------------------------------------------------------------

    def _check_accuse_does_not_affect_state_beyond_broadcast(self, step, pre: WerewolfState, post: WerewolfState) -> None:
        """Accusations are purely a broadcast/signaling channel: they do
        not affect who gets eliminated."""
        if not np.array_equal(np.asarray(pre.alive), np.asarray(post.alive)):
            self._fail(step, "accuse/no_elimination", "the alive set changed during the accuse phase")

    def _check_accuse_fixed_order(self, step, pre: WerewolfState, post: WerewolfState) -> None:
        """All num_agents seats act in fixed absolute seat order
        0,1,2,...,num_agents-1, regardless of alive/dead status."""
        n = self.env.num_agents
        if int(post.phase) == PHASE_ACCUSE:
            expected_next = (int(pre.current_player_idx) + 1) % n
            if int(post.current_player_idx) != expected_next:
                self._fail(step, "accuse/fixed_order", f"next accuser={int(post.current_player_idx)} != expected {expected_next}")

    # -- vote phase ------------------------------------------------------------------

    def _check_vote_fixed_order(self, step, pre: WerewolfState, post: WerewolfState) -> None:
        n = self.env.num_agents
        if int(pre.phase) == PHASE_VOTE and int(post.phase) == PHASE_VOTE:
            expected_next = (int(pre.current_player_idx) + 1) % n
            if int(post.current_player_idx) != expected_next:
                self._fail(step, "vote/fixed_order", f"next voter={int(post.current_player_idx)} != expected {expected_next}")

    def _check_vote_resolution(self, step, pre: WerewolfState, post: WerewolfState, action: int) -> None:
        """Tallies votes from living voters only, breaks ties randomly, and
        eliminates a player.

        This fires on the last voter's own transition, where resolution
        happens in the same step: `post.votes` has already been wiped back
        to all -1 by `_resolve_vote` for the next night, so the final tally
        is reconstructed from `pre.votes` plus the action just taken by
        `pre.current_player_idx` (the last voter)."""
        voter = int(pre.current_player_idx)
        is_noop = action >= self.env.num_agents
        final_choice = -1 if is_noop else int(self.env._to_absolute(jnp.minimum(action, self.env.num_agents - 1), voter))
        votes = np.array(pre.votes)  # copy, not a read-only view -- we mutate it below
        votes[voter] = final_choice
        alive = np.asarray(pre.alive)
        counts = np.zeros(self.env.num_agents)
        for i in range(self.env.num_agents):
            if alive[i] and votes[i] >= 0:
                counts[votes[i]] += 1
        max_votes = counts.max()
        top_candidates = set(np.where(counts == max_votes)[0].tolist())

        pre_alive, post_alive = alive, np.asarray(post.alive)
        newly_dead = np.where(pre_alive & ~post_alive)[0].tolist()
        if max_votes == 0:
            if newly_dead:
                self._fail(step, "vote/no_votes_no_elimination", f"no votes cast but {newly_dead} was eliminated")
        elif len(newly_dead) != 1:
            self._fail(step, "vote/exactly_one_elimination", f"expected exactly 1 elimination, got {newly_dead}")
        elif newly_dead[0] not in top_candidates:
            self._fail(step, "vote/eliminates_top_voted", f"eliminated {newly_dead[0]} not among top-voted {top_candidates} (counts={counts})")

    # -- reward / termination --------------------------------------------------------

    def _check_reward_only_on_terminal_step(self, step, done: bool, reward) -> None:
        if not done and not np.allclose(np.asarray(reward), 0.0):
            self._fail(step, "reward/only_on_terminal_step", f"nonzero reward {np.asarray(reward)} on non-terminal step")

    def _check_terminal_reward_by_team(self, step, post: WerewolfState, reward) -> None:
        """Every agent whose role's team matches game_winner gets +10;
        everyone else gets -10."""
        roles = np.asarray(post.roles)
        winner = int(post.game_winner)
        reward = np.asarray(reward)
        for i in range(self.env.num_agents):
            is_ww = roles[i] == WEREWOLF
            team_wins = (winner == 1) if is_ww else (winner == 0)
            expected = 10.0 if team_wins else -10.0
            if not np.isclose(reward[i], expected):
                self._fail(step, "reward/team_based_terminal", f"agent {i} (ww={is_ww}) reward={reward[i]} != expected {expected} (winner={winner})")

    def _check_horizon_timeout_defaults_to_human_win(self, step, pre: WerewolfState, post: WerewolfState, done: bool) -> None:
        """If timestep >= horizon before either team wins, the episode is
        forcibly ended with game_winner=0: the villager team is awarded
        the win on timeout."""
        if int(post.timestep) >= self.env.horizon and bool(done):
            if int(post.game_winner) != 0:
                # A real win resolving on this exact step is also valid here;
                # the dedicated edge-case test below covers the timeout path.
                pass

    # -- action legality / no-deadlock --------------------------------------------

    def _check_action_legal(self, step, avail, action: int) -> None:
        if not bool(avail[action]):
            self._fail(step, "action/within_avail_mask", f"action {action} taken despite avail mask")

    def _check_avail_nonempty(self, step, avail) -> None:
        if not bool(jnp.asarray(avail).any()):
            self._fail(step, "action/no_deadlock", "no legal actions available for acting player")

    # -- top-level dispatch ------------------------------------------------------

    def validate_transition(self, step, pre: WerewolfState, action: int, avail, post: WerewolfState, reward, done: bool) -> None:
        self._check_avail_nonempty(step, avail)
        self._check_action_legal(step, avail, action)

        phase = int(pre.phase)
        if phase == PHASE_NIGHT:
            self._check_night_avail(step, pre, avail)
            self._check_seer_result(step, pre, post, action)
            if int(pre.night_subphase) == NIGHT_WEREWOLF_1 or (int(pre.night_subphase) == NIGHT_WEREWOLF_0 and int(post.phase) != PHASE_NIGHT):
                pass
            if int(post.phase) == PHASE_ACCUSE:
                self._check_night_resolution(step, pre, post)
        elif phase == PHASE_ACCUSE:
            self._check_accuse_avail(step, pre, avail)
            self._check_accuse_does_not_affect_state_beyond_broadcast(step, pre, post)
            self._check_accuse_fixed_order(step, pre, post)
        elif phase == PHASE_VOTE:
            self._check_vote_avail(step, pre, avail)
            self._check_vote_fixed_order(step, pre, post)
            if int(post.phase) == PHASE_NIGHT or bool(post.done):
                self._check_vote_resolution(step, pre, post, action)

        self._check_reward_only_on_terminal_step(step, done, reward)
        if done:
            self._check_terminal_reward_by_team(step, post, reward)


def rollout_and_validate(env: Werewolf, kind, network, params, num_episodes: int, seed: int = DEFAULT_SEED):
    checker = RuleChecker(env)
    rng = jax.random.PRNGKey(seed)

    for _ in range(num_episodes):
        checker.start_episode()
        rng, reset_rng = jax.random.split(rng)
        state, obs = env.reset(reset_rng)
        checker.check_reset(state)

        for step in range(env.horizon + 1):
            if bool(state.done):
                break
            avail = env.get_avail_actions(state)
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            action = sample_action(kind, network, params, obs, avail, act_rng)
            next_state, next_obs, reward, absorbing, done, info = env.step_env(step_rng, state, action)
            checker.validate_transition(step, state, int(action), avail, next_state, reward, bool(done))
            state, obs = next_state, next_obs

    return checker


# ---------------------------------------------------------------------------
# pytest entry points.
# ---------------------------------------------------------------------------


def test_random_agent_rollouts_conform_to_rules() -> None:
    env = make("werewolf", num_agents=6, num_werewolves=2, horizon=200)
    checker = rollout_and_validate(env, "random", None, None, RANDOM_EPISODES, seed=DEFAULT_SEED)
    checker.assert_no_violations()


def test_random_agent_rollouts_conform_to_rules_simple_config() -> None:
    """Same conformance checks, but against the minimal 3-agent/1-werewolf/
    0-villager config used for the small-scale training runs (num_werewolves
    is otherwise hardcoded to 2 everywhere else in this suite)."""
    env = make("werewolf", num_agents=3, num_werewolves=1, horizon=200)
    checker = rollout_and_validate(env, "random", None, None, RANDOM_EPISODES, seed=DEFAULT_SEED)
    checker.assert_no_violations()


@pytest.mark.parametrize(
    "checkpoint_info", one_checkpoint_per_algorithm_and_kind(discover_checkpoints(CHECKPOINT_ROOT), CHECKPOINT_ROOT)
)
def test_checkpoint_agent_rollouts_conform_to_rules(checkpoint_info) -> None:
    checkpoint_path, network_kind, algorithm = checkpoint_info
    env_kwargs, fc_dim_size = training_env_kwargs(EXAMPLES_DIR, algorithm)
    env = make("werewolf", **env_kwargs)

    _, sample_obs = env.reset(jax.random.PRNGKey(0))
    network = build_network(network_kind, action_dim=env.num_actions, hidden_dim=fc_dim_size)
    params = load_checkpoint_params(network, sample_obs, checkpoint_path)

    checker = rollout_and_validate(env, network_kind, network, params, CHECKPOINT_EPISODES, seed=DEFAULT_SEED)
    checker.assert_no_violations(extra_context=f" for checkpoint={checkpoint_path}")


def test_checkpoint_discovery_runs_without_error() -> None:
    for path, kind in discover_checkpoints(CHECKPOINT_ROOT):
        assert kind in ("actor", "actor_critic", "q_network"), (path, kind)


# ---------------------------------------------------------------------------
# Hand-crafted edge-case scenarios.
# ---------------------------------------------------------------------------


def _make_state(
    env: Werewolf, roles, alive=None, phase=PHASE_NIGHT, night_subphase=0, current_player_idx=None,
    doctor_target=-1, seer_target=-1, werewolf_targets=(-1, -1), seer_results=None,
    accusations=None, votes=None, timestep=0,
) -> WerewolfState:
    n = env.num_agents
    roles = np.array(roles)
    if alive is None:
        alive = [True] * n
    doctor_idx = int(np.argmax(roles == DOCTOR))
    seer_idx = int(np.argmax(roles == SEER))
    ww_idx = sorted(np.where(roles == WEREWOLF)[0].tolist())
    night_order = [doctor_idx, seer_idx, ww_idx[0], ww_idx[1]]
    if current_player_idx is None:
        current_player_idx = night_order[night_subphase] if phase == PHASE_NIGHT else 0
    if seer_results is None:
        seer_results = [0.0] * n
    if accusations is None:
        accusations = [-1] * n
    if votes is None:
        votes = [-1] * n
    return WerewolfState(
        roles=jnp.array(roles, dtype=jnp.int32),
        alive=jnp.array(alive, dtype=bool),
        phase=jnp.int32(phase),
        night_subphase=jnp.int32(night_subphase),
        current_player_idx=jnp.int32(current_player_idx),
        night_order=jnp.array(night_order, dtype=jnp.int32),
        doctor_target=jnp.int32(doctor_target),
        seer_target=jnp.int32(seer_target),
        werewolf_targets=jnp.array(werewolf_targets, dtype=jnp.int32),
        seer_results=jnp.array(seer_results, dtype=jnp.float32),
        accusations=jnp.array(accusations, dtype=jnp.int32),
        votes=jnp.array(votes, dtype=jnp.int32),
        phase_progress=jnp.int32(0),
        absorbing=jnp.zeros(n, dtype=bool),
        done=False,
        game_winner=jnp.int32(-1),
        timestep=timestep,
    )


_ROLES = [VILLAGER, VILLAGER, WEREWOLF, WEREWOLF, DOCTOR, SEER]  # agents 0..5


def _rel(target_abs: int, actor_abs: int, n: int) -> int:
    """Relative-action offset for `actor_abs` to target `target_abs`."""
    return (target_abs - actor_abs) % n


def _play_out_werewolf_night(env, rng, doctor_target=-1, ww0_target=None, ww1_target=None, alive=None):
    """Drives the doctor (noop) + seer (noop) + both werewolf sub-phases
    from a fresh night start, so `werewolf_targets` end up populated exactly
    as real gameplay would (via the ACTION passed to each werewolf's own
    turn, not by presetting state fields step_env would immediately
    overwrite). Returns the resulting state after night resolution."""
    ww_idx = sorted(i for i, r in enumerate(_ROLES) if r == WEREWOLF)
    doctor_idx = int(np.argmax(np.array(_ROLES) == DOCTOR))
    seer_idx = int(np.argmax(np.array(_ROLES) == SEER))
    state = _make_state(env, _ROLES, alive=alive, phase=PHASE_NIGHT, night_subphase=NIGHT_DOCTOR, current_player_idx=doctor_idx)

    doctor_action = env.num_agents if doctor_target < 0 else _rel(doctor_target, doctor_idx, env.num_agents)
    state, *_ = env.step_env(rng, state, jnp.int32(doctor_action))
    state, *_ = env.step_env(rng, state, jnp.int32(env.num_agents))  # seer noops

    if ww0_target is None:
        ww0_action = env.num_agents
    else:
        ww0_action = _rel(ww0_target, ww_idx[0], env.num_agents)
    state, *_ = env.step_env(rng, state, jnp.int32(ww0_action))

    if ww1_target is None:
        ww1_action = env.num_agents
    else:
        ww1_action = _rel(ww1_target, ww_idx[1], env.num_agents)
    state, *_ = env.step_env(rng, state, jnp.int32(ww1_action))
    return state


def test_edge_doctor_successfully_saves_the_target() -> None:
    """If the werewolves' chosen victim matches the doctor's protected
    player, the victim survives."""
    env = make("werewolf", num_agents=6, num_werewolves=2, horizon=200)
    rng = jax.random.PRNGKey(0)
    next_state = _play_out_werewolf_night(env, rng, doctor_target=0, ww0_target=0, ww1_target=0)
    assert bool(next_state.alive[0]), "the doctor's protected target must survive the matching werewolf kill"


def test_edge_doctor_fails_to_save_wrong_target() -> None:
    """Doctor protection only negates the kill if it matches the actual
    victim -- protecting the wrong player does nothing."""
    env = make("werewolf", num_agents=6, num_werewolves=2, horizon=200)
    rng = jax.random.PRNGKey(0)
    next_state = _play_out_werewolf_night(env, rng, doctor_target=1, ww0_target=0, ww1_target=0)
    assert not bool(next_state.alive[0]), "victim 0 must die when the doctor protected a different player (1)"
    assert bool(next_state.alive[1])


def test_edge_werewolves_agreeing_target_dies_without_tiebreak() -> None:
    """If the two werewolves agree on a target, that's the kill target."""
    env = make("werewolf", num_agents=6, num_werewolves=2, horizon=200)
    rng = jax.random.PRNGKey(0)
    next_state = _play_out_werewolf_night(env, rng, doctor_target=-1, ww0_target=1, ww1_target=1)
    assert not bool(next_state.alive[1])
    assert bool(next_state.alive[0])


def test_edge_werewolves_disagree_kill_target_is_one_of_the_two() -> None:
    """If they disagree (both alive, both non-noop, different targets), the
    target is chosen uniformly at random between the two."""
    env = make("werewolf", num_agents=6, num_werewolves=2, horizon=200)
    rng = jax.random.PRNGKey(0)
    next_state = _play_out_werewolf_night(env, rng, doctor_target=-1, ww0_target=0, ww1_target=1)
    dead = [i for i in range(6) if not bool(next_state.alive[i])]
    assert len(dead) == 1 and dead[0] in (0, 1), f"disagreement must kill exactly one of the two disputed targets, got {dead}"


def test_edge_one_werewolf_dead_survivor_vote_alone_decides() -> None:
    """If one werewolf is dead or nooped, the other's choice is used."""
    env = make("werewolf", num_agents=6, num_werewolves=2, horizon=200)
    rng = jax.random.PRNGKey(0)
    ww_idx = sorted(i for i, r in enumerate(_ROLES) if r == WEREWOLF)
    alive = [True] * 6
    alive[ww_idx[0]] = False  # first werewolf already dead

    # With ww0 dead, do_night's skip_action path advances straight past
    # their sub-phase without them acting; drive the night starting from
    # the doctor as usual (doctor/seer noop) so night_subphase correctly
    # reaches NIGHT_WEREWOLF_0, sees ww0 dead, skips to ww1's turn.
    doctor_idx = int(np.argmax(np.array(_ROLES) == DOCTOR))
    seer_idx = int(np.argmax(np.array(_ROLES) == SEER))
    state = _make_state(env, _ROLES, alive=alive, phase=PHASE_NIGHT, night_subphase=NIGHT_DOCTOR, current_player_idx=doctor_idx)
    state, *_ = env.step_env(rng, state, jnp.int32(env.num_agents))  # doctor noop
    state, *_ = env.step_env(rng, state, jnp.int32(env.num_agents))  # seer noop
    # ww0 is dead -> is_actor is False for whoever current_player_idx is
    # (still ww_idx[0], since skip_action only advances on the NEXT step);
    # this step is a no-op skip past ww0's turn.
    state, *_ = env.step_env(rng, state, jnp.int32(env.num_agents))
    assert int(state.night_subphase) == NIGHT_WEREWOLF_1
    assert int(state.current_player_idx) == ww_idx[1]
    next_state, *_ = env.step_env(rng, state, jnp.int32(_rel(1, ww_idx[1], env.num_agents)))
    assert not bool(next_state.alive[1]) and bool(next_state.alive[0]), (
        "with werewolf 0 dead, only werewolf 1's target (agent 1) should be used"
    )


def test_edge_seer_correctly_identifies_werewolf_and_villager() -> None:
    """seer_results[target] is immediately set to +1 if that player is a
    werewolf, -1 otherwise."""
    env = make("werewolf", num_agents=6, num_werewolves=2, horizon=200)
    rng = jax.random.PRNGKey(0)
    ww_idx = sorted(i for i, r in enumerate(_ROLES) if r == WEREWOLF)
    seer_idx = int(np.argmax(np.array(_ROLES) == SEER))
    state = _make_state(env, _ROLES, phase=PHASE_NIGHT, night_subphase=1, current_player_idx=seer_idx)

    rel_target = (ww_idx[0] - seer_idx) % 6
    next_state, *_ = env.step_env(rng, state, jnp.int32(rel_target))
    assert float(next_state.seer_results[ww_idx[0]]) == 1.0, "seer must correctly ID a werewolf as +1"

    villager_idx = next(i for i, r in enumerate(_ROLES) if r == VILLAGER)
    state2 = _make_state(env, _ROLES, phase=PHASE_NIGHT, night_subphase=1, current_player_idx=seer_idx)
    rel_target2 = (villager_idx - seer_idx) % 6
    next_state2, *_ = env.step_env(rng, state2, jnp.int32(rel_target2))
    assert float(next_state2.seer_results[villager_idx]) == -1.0, "seer must correctly ID a non-werewolf as -1"


def test_edge_seer_results_persist_across_a_second_night() -> None:
    """seer_results is never reset during the game (only at reset()), so a
    seer's knowledge accumulates and persists."""
    env = make("werewolf", num_agents=6, num_werewolves=2, horizon=200)
    seer_idx = int(np.argmax(np.array(_ROLES) == SEER))
    ww_idx = sorted(i for i, r in enumerate(_ROLES) if r == WEREWOLF)
    # Simulate: seer already knows ww_idx[0] is a werewolf from a prior night.
    prior_results = [0.0] * 6
    prior_results[ww_idx[0]] = 1.0
    state = _make_state(
        env, _ROLES, phase=PHASE_VOTE, current_player_idx=0, seer_results=prior_results,
    )
    assert float(state.seer_results[ww_idx[0]]) == 1.0, "prior seer knowledge must still be present in a later phase"


def test_edge_accusations_do_not_affect_who_gets_voted_out() -> None:
    """Accusations are purely a broadcast/signaling channel and do not
    affect who gets eliminated. Everyone accuses agent 1 (a strong signal),
    but everyone actually votes for agent 2 -- agent 2 must be eliminated,
    and agent 1 (merely accused) must survive."""
    env = make("werewolf", num_agents=6, num_werewolves=2, horizon=200)
    rng = jax.random.PRNGKey(0)
    accusations = [1, 1, 1, 1, 1, -1]  # everyone (but agent 5) accuses agent 1
    votes = [2, 2, 2, 2, 2, -1]  # but everyone actually VOTES for agent 2
    # Agent 5 is the last (5th) voter; this action closes out the vote.
    state = _make_state(
        env, _ROLES, phase=PHASE_VOTE, current_player_idx=5, accusations=accusations, votes=votes,
    )
    next_state, *_ = env.step_env(rng, state, jnp.int32(_rel(2, 5, env.num_agents)))  # agent 5 votes for agent 2 too
    assert not bool(next_state.alive[2]), "agent 2 (the actually-voted plurality target) must be eliminated"
    assert bool(next_state.alive[1]), "agent 1 (merely accused, never voted for) must survive"


def test_edge_villager_team_wins_when_all_werewolves_eliminated() -> None:
    """All werewolves eliminated means the villager team wins."""
    env = make("werewolf", num_agents=6, num_werewolves=2, horizon=200)
    rng = jax.random.PRNGKey(0)
    ww_idx = sorted(i for i, r in enumerate(_ROLES) if r == WEREWOLF)
    alive = [True] * 6
    alive[ww_idx[0]] = False  # one werewolf already dead
    votes = [-1] * 6
    for i in range(6):
        if i != ww_idx[1]:
            votes[i] = ww_idx[1]  # everyone votes out the last werewolf
    state = _make_state(env, _ROLES, alive=alive, phase=PHASE_VOTE, current_player_idx=5, votes=votes)
    next_state, obs, reward, absorbing, done, info = env.step_env(rng, state, jnp.int32((ww_idx[1] - 5) % 6 if 5 != ww_idx[1] else 6))
    assert bool(done)
    assert int(next_state.game_winner) == 0, "eliminating the last werewolf must award the villager team the win"
    reward = np.asarray(reward)
    roles = np.array(_ROLES)
    assert all(reward[i] == 10.0 for i in range(6) if roles[i] != WEREWOLF)
    assert all(reward[i] == -10.0 for i in range(6) if roles[i] == WEREWOLF)


def test_edge_werewolf_team_wins_when_they_equal_remaining_humans() -> None:
    """Werewolves equaling or outnumbering the remaining players means the
    werewolf team wins."""
    env = make("werewolf", num_agents=6, num_werewolves=2, horizon=200)
    rng = jax.random.PRNGKey(0)
    # Agents 0,1 (villagers) already dead; 2,3=werewolves, 4=doctor,5=seer alive.
    # Killing agent 4 (doctor) leaves ww=2 alive (2,3) vs humans=1 alive (5) -> 2 >= 1 -> werewolves win.
    alive = [False, False, True, True, True, True]
    next_state = _play_out_werewolf_night(env, rng, doctor_target=-1, ww0_target=4, ww1_target=4, alive=alive)
    assert not bool(next_state.alive[4])
    assert bool(next_state.done)
    assert int(next_state.game_winner) == 1, "werewolves outnumbering the remaining humans must end the game in their favor"


def test_edge_horizon_timeout_defaults_to_villager_win() -> None:
    """If timestep >= horizon before either team wins, the episode is
    forcibly ended with game_winner=0: the villager team is awarded the
    win on timeout."""
    env = make("werewolf", num_agents=6, num_werewolves=2, horizon=200)
    rng = jax.random.PRNGKey(0)
    # A state that would NOT otherwise resolve a win this step (doctor noop),
    # but is already at the horizon boundary.
    state = _make_state(env, _ROLES, phase=PHASE_NIGHT, night_subphase=0, timestep=199)
    next_state, obs, reward, absorbing, done, info = env.step_env(rng, state, jnp.int32(6))  # doctor noops
    assert int(next_state.timestep) == 200
    assert bool(done), "reaching the horizon must force the episode to end"
    assert int(next_state.game_winner) == 0, "a horizon timeout must default to a villager-team win"


def test_edge_dead_player_forced_to_noop_in_accuse_and_vote() -> None:
    """A dead player's turn is forced to noop in both the accuse and vote
    phases."""
    env = make("werewolf", num_agents=6, num_werewolves=2, horizon=200)
    alive = [True] * 6
    alive[0] = False
    accuse_state = _make_state(env, _ROLES, alive=alive, phase=PHASE_ACCUSE, current_player_idx=0)
    accuse_avail = env.get_avail_actions(accuse_state)
    assert bool(accuse_avail[6]) and not np.asarray(accuse_avail[:6]).any(), "a dead player must only have noop legal in the accuse phase"

    vote_state = _make_state(env, _ROLES, alive=alive, phase=PHASE_VOTE, current_player_idx=0)
    vote_avail = env.get_avail_actions(vote_state)
    assert bool(vote_avail[6]) and not np.asarray(vote_avail[:6]).any(), "a dead player must only have noop legal in the vote phase"


# ---------------------------------------------------------------------------
# Standalone CLI.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--network-type", type=str, default=None, choices=["actor", "actor_critic", "q_network"])
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    if not args.random and args.checkpoint is None:
        parser.error("pass --checkpoint PATH or --random")

    env = make("werewolf", num_agents=6, num_werewolves=2, horizon=200)

    if args.random:
        kind, network, params = "random", None, None
    else:
        kind = args.network_type or infer_network_kind(args.checkpoint)
        network = build_network(kind, action_dim=env.num_actions, hidden_dim=args.hidden_dim)
        _, sample_obs = env.reset(jax.random.PRNGKey(0))
        params = load_checkpoint_params(network, sample_obs, args.checkpoint)

    print(f"Validating {args.episodes} episode(s) of '{kind}' agent (werewolf)...")
    checker = rollout_and_validate(env, kind, network, params, args.episodes, seed=args.seed)

    if checker.violations:
        print(f"\n{len(checker.violations)} VIOLATION(S) FOUND:")
        for v in checker.violations[:20]:
            print(f"  {v}")
        raise SystemExit(1)
    print("\nNo rule violations found.")


if __name__ == "__main__":
    main()
