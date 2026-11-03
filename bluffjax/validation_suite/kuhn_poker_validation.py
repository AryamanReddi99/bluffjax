"""
Rule-conformance validation suite for Kuhn Poker.

Checks every state/action/transition produced by random or checkpoint-driven
rollouts against bluffjax/environments/kuhn_poker/rules.md and the
implementation in bluffjax/environments/kuhn_poker/kuhn_poker.py.

Kuhn Poker has no `phase` field in its state -- the phase of a hand (first
action / second action / third action) is inferred purely from
`state.timestep` (0, 1, 2). Rule checks below dispatch on timestep the same
way step_env does, but the expected outcome at each branch is hand-coded
directly from the payoff table rather than re-derived from step_env's
pot-sum formula, so the oracle stays independent of the code it's checking.

Usage:
    pytest bluffjax/validation_suite/kuhn_poker_validation.py -v
    python kuhn_poker_validation.py --random --episodes 500

Kuhn Poker has zero checkpoints on disk (KuhnPoker() takes no constructor
kwargs, and its training scripts compute exploitability directly rather than
saving avg/br .msgpack checkpoints); the checkpoint-driven test is fully
implemented but will report 0 discovered checkpoints and skip.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.environments.kuhn_poker.kuhn_poker import KuhnPoker, KuhnState
from bluffjax.validation_suite.common import (
    RuleCheckerBase,
    build_network,
    discover_checkpoints,
    infer_network_kind,
    load_checkpoint_params,
    one_checkpoint_per_algorithm_and_kind,
    rollout_and_validate_aec,
    training_env_kwargs,
)

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.normpath(os.path.join(HERE, "..", "examples", "kuhn"))
CHECKPOINT_ROOT = os.path.join(EXAMPLES_DIR, "checkpoints")

DEFAULT_SEED = 0
RANDOM_EPISODES = 300
CHECKPOINT_EPISODES = 100


class RuleChecker(RuleCheckerBase):
    """One method per rule or legal-condition check for Kuhn Poker."""

    def __init__(self, env: KuhnPoker):
        super().__init__(env)

    # -- setup / reset --------------------------------------------------------

    def check_reset(self, state: KuhnState) -> None:
        """The deck is shuffled and one card dealt privately to each player;
        the third card is set aside unseen. Dealt ranks must be distinct
        values from {0,1,2} (Jack/Queen/King), and the ante must already be
        posted (`pot = [1, 1]`)."""
        hands = np.asarray(state.agent_hands)
        if hands.shape != (2,):
            self._fail(0, "setup/hand_shape", f"agent_hands shape={hands.shape} != (2,)")
        if not np.all((hands >= 0) & (hands <= 2)):
            self._fail(0, "setup/hand_range", f"agent_hands={hands} outside [0,2]")
        if hands[0] == hands[1]:
            self._fail(0, "setup/distinct_hands", f"both players dealt rank {hands[0]}")
        pot = np.asarray(state.pot)
        if not np.array_equal(pot, [1, 1]):
            self._fail(0, "setup/ante", f"pot at reset={pot}, expected [1,1] (both antes)")
        if int(state.timestep) != 0:
            self._fail(0, "setup/initial_timestep", "hand must start at timestep 0")

    # -- structural invariants --------------------------------------------------

    def _check_hand_invariant(self, step, pre: KuhnState, post: KuhnState) -> None:
        """Hole cards are dealt once at reset and never change during a
        hand -- no rule permits a card to be swapped mid-hand in Kuhn Poker."""
        if not np.array_equal(np.asarray(pre.agent_hands), np.asarray(post.agent_hands)):
            self._fail(step, "invariant/hands_immutable", "agent_hands changed mid-hand")

    def _check_pot_bounds(self, step, post: KuhnState) -> None:
        """The ante is 1 chip, and there is a single fixed bet size of 1
        chip with no raising -- so each player's pot entry can only ever be
        1 (ante only) or 2 (ante + one bet/call), never more, never less."""
        pot = np.asarray(post.pot)
        if not np.all((pot == 1) | (pot == 2)):
            self._fail(step, "invariant/pot_bounds", f"pot={pot} has an entry outside {{1,2}}")

    def _check_turn_alternation(self, step, pre: KuhnState, post: KuhnState) -> None:
        """Play alternates every step: current_player_idx must flip 0<->1
        unconditionally, even on the terminal step."""
        expected = (int(pre.current_player_idx) + 1) % 2
        if int(post.current_player_idx) != expected:
            self._fail(
                step,
                "invariant/turn_alternation",
                f"current_player_idx={int(post.current_player_idx)} != expected {expected}",
            )

    def _check_action_legal(self, step, avail: jnp.ndarray, action: int) -> None:
        """Foundational legality requirement: the action fed to `env.step_env`
        must be one `get_avail_actions` marked legal."""
        if not bool(avail[action]):
            self._fail(
                step,
                "action/within_avail_mask",
                f"action {action} taken despite avail={np.asarray(avail)}",
            )

    def _check_avail_mask_shape(self, step, pre: KuhnState, avail: jnp.ndarray) -> None:
        """Both discrete actions -- check/fold (0) and bet/call (1) -- are
        legal at every non-terminal step; there is no phase-dependent
        restriction on the action set."""
        expected = np.array([True, True]) if not bool(pre.done) else np.array([False, False])
        if not np.array_equal(np.asarray(avail), expected):
            self._fail(
                step,
                "action/avail_mask_always_both",
                f"avail={np.asarray(avail)} != expected {expected} for done={bool(pre.done)}",
            )

    # -- payoff-table oracle (the central rule-conformance check) ---------------

    def _expected_outcome(self, hands, a0: int, a1: int, a2: Optional[int]):
        """Hand-coded directly from the payoff table, independent of
        kuhn_poker.py's own pot-sum-based branching, so a bug in that
        branching logic can't escape detection by an oracle that shares it.

        hands: (2,) array of dealt ranks. a0/a1 are the first two actions (by
        the first-to-act and second-to-act player respectively); a2 is the
        optional third action (by the first-to-act player again), or None if
        the hand ended after two actions.

        Returns (winner_relative_idx, pot_total): winner_relative_idx is 0
        (first-to-act) or 1 (second-to-act); pot_total is the chips in the
        pot when the hand concluded (2, 3, or 4).
        """
        higher_is_first = hands[0] > hands[1]  # relative to first-to-act
        if a0 == 0 and a1 == 0:
            # pass, pass -> showdown, pot = 2 (both antes only)
            return (0 if higher_is_first else 1), 2
        if a0 == 0 and a1 == 1 and a2 == 0:
            # pass, bet, pass -> first-to-act folds -> second-to-act wins, pot = 3
            return 1, 3
        if a0 == 0 and a1 == 1 and a2 == 1:
            # pass, bet, bet -> showdown, pot = 4
            return (0 if higher_is_first else 1), 4
        if a0 == 1 and a1 == 0:
            # bet, pass -> second-to-act folds -> first-to-act wins, pot = 3
            return 0, 3
        if a0 == 1 and a1 == 1:
            # bet, bet -> showdown, pot = 4
            return (0 if higher_is_first else 1), 4
        raise AssertionError(f"Unreachable action sequence: a0={a0}, a1={a1}, a2={a2}")

    def check_hand_outcome(
        self, first_to_act: int, hands, actions: list[int], final_state: KuhnState, final_reward
    ) -> None:
        """Cross-checks the fully-played-out hand's winner and reward against
        the payoff table (see `_expected_outcome`), independent of
        kuhn_poker.py's own internal computation."""
        a0 = actions[0]
        a1 = actions[1]
        a2 = actions[2] if len(actions) > 2 else None
        other = 1 - first_to_act
        hands_relative = [hands[first_to_act], hands[other]]
        winner_rel, pot_total = self._expected_outcome(hands_relative, a0, a1, a2)
        winner_abs = (first_to_act + winner_rel) % 2
        loser_abs = 1 - winner_abs

        # KuhnState has no explicit `game_winner` field; the winner is
        # implied by which agent received a positive terminal reward.
        actual_reward = np.asarray(final_reward)
        if not (actual_reward[winner_abs] > 0 and actual_reward[loser_abs] < 0):
            self._fail(
                len(actions) - 1,
                "payoff/winner",
                f"expected winner={winner_abs} (rel {winner_rel}) but reward={actual_reward} "
                f"for hands={hands}, actions={actions}",
            )

        # Recompute each player's total contribution independently from the
        # action sequence (ante 1 + 1 more iff that player ever bet/called).
        contributed = [1, 1]
        # first_to_act contributes on a0 (and a2 if present); other player on a1.
        contributed[0] += a0
        contributed[1] += a1
        if a2 is not None:
            contributed[0] += a2
        expected_winner_reward = pot_total - contributed[winner_rel]
        expected_loser_reward = -contributed[1 - winner_rel]

        reward = np.asarray(final_reward)
        if not np.isclose(reward[winner_abs], expected_winner_reward):
            self._fail(
                len(actions) - 1,
                "payoff/winner_reward",
                f"winner reward={reward[winner_abs]} != expected {expected_winner_reward} "
                f"(pot_total={pot_total}, contributed={contributed})",
            )
        if not np.isclose(reward[loser_abs], expected_loser_reward):
            self._fail(
                len(actions) - 1,
                "payoff/loser_reward",
                f"loser reward={reward[loser_abs]} != expected {expected_loser_reward}",
            )
        if not np.isclose(reward.sum(), 0.0, atol=1e-4):
            self._fail(
                len(actions) - 1,
                "payoff/zero_sum",
                f"rewards {reward} do not sum to zero, violating 'the game is zero-sum in chips'",
            )

    # -- non-terminal steps must not pay out ------------------------------------

    def _check_reward_only_on_terminal_step(self, step, post: KuhnState, reward) -> None:
        """No rule grants a reward before the hand concludes; reward must be
        exactly [0,0] unless this step just produced a winner."""
        if not bool(post.all_absorbing):
            if not np.allclose(np.asarray(reward), 0.0):
                self._fail(
                    step,
                    "reward/only_on_terminal_step",
                    f"nonzero reward {np.asarray(reward)} on a non-terminal transition",
                )

    # -- horizon boundary --------------------------------------------------------

    def check_horizon_boundary(self, env: KuhnPoker) -> None:
        """The horizon check `next_timestep > self.horizon` is a strict '>',
        not '>='. With horizon=10, done-via-horizon should not trigger when
        next_timestep == 10, only once it exceeds 10. Unreachable under
        legal play (a hand always resolves within 3 actions), so this uses
        a synthetic pre-horizon state to isolate the boundary directly."""
        rng = jax.random.PRNGKey(0)
        state, obs = env.reset(rng)
        # Synthetic mid-hand state at timestep=9 isolates the horizon clause
        # from showdown/fold logic.
        state = state.replace(timestep=jnp.int32(9))
        next_state, *_ = env.step_env(rng, state, jnp.int32(0))
        if bool(next_state.done):
            self._fail(
                -1,
                "horizon/boundary_not_yet",
                f"done=True at next_timestep={int(next_state.timestep)} (==horizon={env.horizon}); "
                "expected done=False since the check is strict '>'",
            )
        state10 = state.replace(timestep=jnp.int32(10))
        next_state10, *_ = env.step_env(rng, state10, jnp.int32(0))
        if not bool(next_state10.done):
            self._fail(
                -1,
                "horizon/boundary_exceeded",
                f"done=False at next_timestep={int(next_state10.timestep)} (>horizon={env.horizon}); "
                "expected done=True",
            )


# ---------------------------------------------------------------------------
# Rollout driver (Kuhn-specific: tracks the per-hand action sequence needed
# for the payoff-table oracle, on top of the generic per-step checks).
# ---------------------------------------------------------------------------


def rollout_and_validate(env: KuhnPoker, kind, network, params, num_episodes: int, seed: int = DEFAULT_SEED):
    from bluffjax.validation_suite.common import sample_action

    checker = RuleChecker(env)
    rng = jax.random.PRNGKey(seed)

    for _ in range(num_episodes):
        checker.start_episode()
        rng, reset_rng = jax.random.split(rng)
        state, obs = env.reset(reset_rng)
        checker.check_reset(state)

        first_to_act = int(state.start_player_idx)
        hands = np.asarray(state.agent_hands)
        actions_taken: list[int] = []

        for step in range(env.horizon + 1):
            if bool(state.done):
                break
            checker._check_avail_mask_shape(step, state, env.get_avail_actions(state))

            avail = env.get_avail_actions(state)
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            action = sample_action(kind, network, params, obs, avail, act_rng)
            checker._check_action_legal(step, avail, int(action))
            actions_taken.append(int(action))

            next_state, next_obs, reward, absorbing, done, info = env.step_env(
                step_rng, state, action
            )

            checker._check_hand_invariant(step, state, next_state)
            checker._check_pot_bounds(step, next_state)
            checker._check_turn_alternation(step, state, next_state)
            checker._check_reward_only_on_terminal_step(step, next_state, reward)

            if bool(next_state.all_absorbing):
                checker.check_hand_outcome(first_to_act, hands, actions_taken, next_state, reward)

            state, obs = next_state, next_obs

    checker.check_horizon_boundary(env)
    return checker


# ---------------------------------------------------------------------------
# pytest entry points.
# ---------------------------------------------------------------------------


def test_random_agent_rollouts_conform_to_rules() -> None:
    """Uniform-random legal-action rollouts must never violate any game
    rule. KuhnPoker() takes no constructor kwargs (fixed 2 players, fixed
    horizon=10), so there is nothing to parametrize over."""
    env = make("kuhn_poker")
    checker = rollout_and_validate(env, "random", None, None, RANDOM_EPISODES, seed=DEFAULT_SEED)
    checker.assert_no_violations()


@pytest.mark.parametrize("checkpoint_info", one_checkpoint_per_algorithm_and_kind(
    discover_checkpoints(CHECKPOINT_ROOT), CHECKPOINT_ROOT
))
def test_checkpoint_agent_rollouts_conform_to_rules(checkpoint_info) -> None:
    """Same rule-conformance check, driven by a trained checkpoint's policy.
    Kuhn Poker currently has zero saved checkpoints on disk (its training
    scripts compute exploitability in-memory rather than persisting avg/br
    .msgpack files), so pytest collects zero test cases here until a
    ppo_nfsp_*/pqn_nfsp_* checkpoint appears under
    bluffjax/examples/kuhn/checkpoints/."""
    checkpoint_path, network_kind, algorithm = checkpoint_info
    env_kwargs, fc_dim_size = training_env_kwargs(EXAMPLES_DIR, algorithm)
    env = make("kuhn_poker", **env_kwargs)

    _, sample_obs = env.reset(jax.random.PRNGKey(0))
    network = build_network(network_kind, action_dim=env.num_actions, hidden_dim=fc_dim_size)
    params = load_checkpoint_params(network, sample_obs, checkpoint_path)

    checker = rollout_and_validate(
        env, network_kind, network, params, CHECKPOINT_EPISODES, seed=DEFAULT_SEED
    )
    checker.assert_no_violations(extra_context=f" for checkpoint={checkpoint_path}")


def test_checkpoint_discovery_runs_without_error() -> None:
    for path, kind in discover_checkpoints(CHECKPOINT_ROOT):
        assert kind in ("actor", "actor_critic", "q_network"), (path, kind)


# ---------------------------------------------------------------------------
# Hand-crafted edge-case scenarios.
# ---------------------------------------------------------------------------


def _make_state(hands, start_player_idx=0, pot=(1, 1), timestep=0) -> KuhnState:
    return KuhnState(
        agent_hands=jnp.array(hands, dtype=jnp.int32),
        pot=jnp.array(pot, dtype=jnp.int32),
        start_player_idx=jnp.int32(start_player_idx),
        current_player_idx=jnp.int32(start_player_idx),
        returns=jnp.zeros(2, dtype=jnp.float32),
        absorbing=jnp.zeros(2, dtype=bool),
        all_absorbing=False,
        done=False,
        timestep=jnp.int32(timestep),
    )


def test_edge_check_check_showdown_higher_card_wins() -> None:
    """Check-check goes to showdown, and the higher card wins +1. King
    (rank 2) vs Jack (rank 0); both check."""
    env = make("kuhn_poker")
    rng = jax.random.PRNGKey(0)
    state = _make_state(hands=[2, 0])  # player 0 = King, player 1 = Jack

    s1, *_ = env.step_env(rng, state, jnp.int32(0))  # player 0 checks
    assert int(s1.timestep) == 1
    final, obs, reward, absorbing, done, info = env.step_env(rng, s1, jnp.int32(0))  # player 1 checks
    assert bool(final.all_absorbing), "hand must conclude after check-check"
    assert np.isclose(float(reward[0]), 1.0) and np.isclose(float(reward[1]), -1.0), (
        "check-check showdown: King should win; net payoff should be +/-1 (pot=2, each contributed 1)"
    )


def test_edge_bet_fold_bettor_wins_uncontested() -> None:
    """Bet-fold: the bettor wins +1 uncontested. Deal the worse card to the
    bettor to prove the win comes from the fold, not the card -- folding
    forfeits regardless of who actually held the better hand."""
    env = make("kuhn_poker")
    rng = jax.random.PRNGKey(0)
    state = _make_state(hands=[0, 2])  # player 0 = Jack (worse), player 1 = King

    s1, *_ = env.step_env(rng, state, jnp.int32(1))  # player 0 bets
    final, obs, reward, absorbing, done, info = env.step_env(rng, s1, jnp.int32(0))  # player 1 folds
    assert bool(final.all_absorbing)
    assert np.isclose(float(reward[0]), 1.0) and np.isclose(float(reward[1]), -1.0), (
        "the bettor wins uncontested on a fold, even holding the worse card"
    )


def test_edge_check_bet_fold_original_actor_folds() -> None:
    """Check-bet-fold: the player who checked faces a bet and folds,
    forfeiting the pot to the opponent despite having invested only
    their ante."""
    env = make("kuhn_poker")
    rng = jax.random.PRNGKey(0)
    state = _make_state(hands=[2, 0])  # player 0 = King (better card, but will fold anyway)

    s1, *_ = env.step_env(rng, state, jnp.int32(0))  # player 0 checks
    s2, *_ = env.step_env(rng, s1, jnp.int32(1))  # player 1 bets
    final, obs, reward, absorbing, done, info = env.step_env(rng, s2, jnp.int32(0))  # player 0 folds
    assert bool(final.all_absorbing)
    assert np.isclose(float(reward[1]), 1.0) and np.isclose(float(reward[0]), -1.0), (
        "player 0 folded after check-bet -> player 1 wins despite holding the worse card"
    )


def test_edge_check_bet_call_showdown_pot_of_four() -> None:
    """Check-bet-call goes to showdown with 2 chips from each player in the
    pot, and the higher card wins +2."""
    env = make("kuhn_poker")
    rng = jax.random.PRNGKey(0)
    state = _make_state(hands=[0, 2])  # player 1 = King

    s1, *_ = env.step_env(rng, state, jnp.int32(0))  # player 0 checks
    s2, *_ = env.step_env(rng, s1, jnp.int32(1))  # player 1 bets
    final, obs, reward, absorbing, done, info = env.step_env(rng, s2, jnp.int32(1))  # player 0 calls
    assert bool(final.all_absorbing)
    assert np.isclose(float(reward[1]), 2.0) and np.isclose(float(reward[0]), -2.0), (
        "call after a check-bet line: pot=4, each contributed 2 -> net +/-2"
    )


def test_edge_horizon_boundary_is_strict_greater_than() -> None:
    """kuhn_poker.py: `next_done = ... | (next_timestep > self.horizon)` --
    confirms the exact off-by-one: done-via-horizon must NOT fire at
    next_timestep == horizon (10), only once it strictly exceeds it (11)."""
    env = make("kuhn_poker")
    rng = jax.random.PRNGKey(0)
    state = _make_state(hands=[2, 0], timestep=9)
    at_boundary, *_ = env.step_env(rng, state, jnp.int32(0))
    assert int(at_boundary.timestep) == 10
    assert not bool(at_boundary.done), "done must not trigger when next_timestep == horizon"

    state_one_more = _make_state(hands=[2, 0], timestep=10)
    past_boundary, *_ = env.step_env(rng, state_one_more, jnp.int32(0))
    assert int(past_boundary.timestep) == 11
    assert bool(past_boundary.done), "done must trigger once next_timestep exceeds horizon"


# ---------------------------------------------------------------------------
# Standalone CLI.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--network-type", type=str, default=None, choices=["actor", "actor_critic", "q_network"]
    )
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    if not args.random and args.checkpoint is None:
        parser.error("pass --checkpoint PATH or --random")

    env = make("kuhn_poker")

    if args.random:
        kind, network, params = "random", None, None
    else:
        kind = args.network_type or infer_network_kind(args.checkpoint)
        network = build_network(kind, action_dim=env.num_actions, hidden_dim=args.hidden_dim)
        _, sample_obs = env.reset(jax.random.PRNGKey(0))
        params = load_checkpoint_params(network, sample_obs, args.checkpoint)

    print(f"Validating {args.episodes} episode(s) of '{kind}' agent (kuhn_poker)...")
    checker = rollout_and_validate(env, kind, network, params, args.episodes, seed=args.seed)

    if checker.violations:
        print(f"\n{len(checker.violations)} VIOLATION(S) FOUND:")
        for v in checker.violations[:20]:
            print(f"  {v}")
        raise SystemExit(1)
    print("\nNo rule violations found.")


if __name__ == "__main__":
    main()
