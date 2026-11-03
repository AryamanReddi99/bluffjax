"""
Rule-conformance validation suite for Leduc Hold'em.

Checks every state/action/transition produced by random or checkpoint-driven
rollouts against bluffjax/environments/leduc_holdem/rules.md and the
implementation in bluffjax/environments/leduc_holdem/leduc_holdem.py.

Usage:
    pytest bluffjax/validation_suite/leduc_holdem_validation.py -v
    python leduc_holdem_validation.py --random --episodes 500

Note: leduc_holdem currently has zero checkpoints on disk (exploitability is
evaluated in-memory during training rather than persisted), so the
checkpoint-driven test below will report 0 discovered checkpoints and skip.
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.environments.leduc_holdem.leduc_holdem import LeducHoldem, LeducHoldemState
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
EXAMPLES_DIR = os.path.normpath(os.path.join(HERE, "..", "examples", "leduc"))
CHECKPOINT_ROOT = os.path.join(EXAMPLES_DIR, "checkpoints")

DEFAULT_SEED = 0
RANDOM_EPISODES = 300
CHECKPOINT_EPISODES = 100


class RuleChecker(RuleCheckerBase):
    def __init__(self, env: LeducHoldem):
        super().__init__(env)

    # -- setup / reset --------------------------------------------------------

    def check_reset(self, state: LeducHoldemState) -> None:
        """Deck is 6 cards (2 suits x 3 ranks), each player is dealt one
        private hole card, both players post a 1-chip ante, and player 0
        acts first."""
        cards = np.asarray(state.shuffled_deck)
        if not np.array_equal(np.sort(cards), np.arange(6)):
            self._fail(0, "setup/deck", f"shuffled_deck={cards} is not a permutation of 0..5")
        hole = np.asarray(state.agent_cards)
        if hole[0] == hole[1]:
            self._fail(0, "setup/distinct_hole_cards", f"both players dealt card {hole[0]}")
        if int(state.public_card) != -1:
            self._fail(0, "setup/no_public_card_yet", "public_card must be unrevealed (-1) at reset")
        if not np.array_equal(np.asarray(state.ante), [1.0, 1.0]):
            self._fail(0, "setup/ante", f"ante at reset={np.asarray(state.ante)}, expected [1,1]")
        if int(state.stage) != 1:
            self._fail(0, "setup/initial_stage", "hand must start in stage 1 (pre-flop)")
        if int(state.current_player_idx) != 0:
            self._fail(0, "setup/player_0_first", "Player 0 must act first in round 1")

    # -- action legality --------------------------------------------------------

    def _check_fold_only_when_facing_bet(self, step, pre: LeducHoldemState, avail) -> None:
        """Fold is only a legal action when facing a bet: avail[0] ==
        (stakes > my_ante)."""
        my_ante = float(pre.ante[pre.current_player_idx])
        expected = bool(pre.stakes) and float(pre.stakes) > my_ante
        if bool(avail[0]) != expected:
            self._fail(
                step,
                "action/fold_only_facing_bet",
                f"avail[fold]={bool(avail[0])} != expected {expected} "
                f"(stakes={float(pre.stakes)}, my_ante={my_ante})",
            )

    def _check_call_always_legal(self, step, avail) -> None:
        """Call is always a legal action."""
        if not bool(avail[1]):
            self._fail(step, "action/call_always_legal", "call (action 1) must always be legal")

    def _check_raise_cap(self, step, pre: LeducHoldemState, avail) -> None:
        """Raise is legal iff num_raises < MAX_RAISES (i.e. 0 or 1 raises
        already made this round); MAX_RAISES=2."""
        expected = int(pre.num_raises) < self.env.max_raises
        if bool(avail[2]) != expected:
            self._fail(
                step,
                "action/raise_cap",
                f"avail[raise]={bool(avail[2])} != expected {expected} "
                f"(num_raises={int(pre.num_raises)}, max_raises={self.env.max_raises})",
            )
        if int(pre.num_raises) > self.env.max_raises:
            self._fail(
                step, "action/raise_cap_exceeded", f"num_raises={int(pre.num_raises)} > max"
            )

    # -- chip arithmetic ---------------------------------------------------------

    def _check_call_amount(self, step, pre: LeducHoldemState, post: LeducHoldemState, player: int) -> None:
        """A call brings the caller's ante up to exactly `stakes` (the
        current outstanding bet level), no more, no less."""
        expected = max(float(pre.stakes) - float(pre.ante[player]), 0.0)
        actual = float(post.ante[player]) - float(pre.ante[player])
        if not np.isclose(actual, expected, atol=1e-4):
            self._fail(
                step,
                "chips/call_amount",
                f"call added {actual} chips, expected exactly {expected} (= stakes - my_ante)",
            )
        if not np.isclose(float(post.stakes), float(pre.stakes), atol=1e-4):
            self._fail(step, "chips/call_no_stakes_change", "a call must not change `stakes`")

    def _check_raise_amount(self, step, pre: LeducHoldemState, post: LeducHoldemState, player: int) -> None:
        """A raise brings the raiser's ante up to `stakes` (the call amount)
        then adds one more `raise_amount` on top, and `stakes` itself
        increases by that same `raise_amount`. The increment is smaller in
        round 1 and larger in round 2 (canonically 2 and 4 chips)."""
        raise_amt = self.env.raise_amount_r1 if int(pre.stage) == 1 else self.env.raise_amount_r2
        expected_ante_delta = max(float(pre.stakes) - float(pre.ante[player]), 0.0) + raise_amt
        actual_ante_delta = float(post.ante[player]) - float(pre.ante[player])
        if not np.isclose(actual_ante_delta, expected_ante_delta, atol=1e-4):
            self._fail(
                step,
                "chips/raise_amount",
                f"raise added {actual_ante_delta} chips, expected {expected_ante_delta} "
                f"(call + {raise_amt}-chip increment for stage {int(pre.stage)})",
            )
        expected_stakes = float(pre.stakes) + raise_amt
        if not np.isclose(float(post.stakes), expected_stakes, atol=1e-4):
            self._fail(
                step,
                "chips/raise_new_stakes",
                f"stakes after raise={float(post.stakes)} != expected {expected_stakes}",
            )
        if int(post.num_raises) != int(pre.num_raises) + 1:
            self._fail(step, "chips/raise_count_increment", "num_raises must increment by exactly 1")

    def _check_fold_effect(self, step, pre: LeducHoldemState, post: LeducHoldemState, player: int) -> None:
        """Folding forfeits the hand without moving any chips."""
        if not bool(post.folded[player]):
            self._fail(step, "fold/marks_folded", f"player {player} folded but folded[{player}] is False")
        if not np.allclose(np.asarray(post.ante), np.asarray(pre.ante)):
            self._fail(step, "fold/no_chip_change", "folding must not change ante amounts")
        if not np.isclose(float(post.stakes), float(pre.stakes)):
            self._fail(step, "fold/no_stakes_change", "folding must not change stakes")

    # -- public card ---------------------------------------------------------------

    def _check_public_card_reveal(self, step, pre: LeducHoldemState, post: LeducHoldemState) -> None:
        """A public card is revealed face-up at the start of round 2, drawn
        from the card held back for that purpose. Once revealed it must
        equal shuffled_deck[2] exactly, must never subsequently change, and
        must never coincide with either player's hole card."""
        if int(pre.public_card) == -1 and int(post.public_card) != -1:
            expected = int(pre.shuffled_deck[2])
            if int(post.public_card) != expected:
                self._fail(
                    step,
                    "public_card/reveal_source",
                    f"public_card revealed as {int(post.public_card)} != shuffled_deck[2]={expected}",
                )
            if int(post.public_card) in (int(post.agent_cards[0]), int(post.agent_cards[1])):
                self._fail(
                    step,
                    "public_card/no_overlap",
                    f"public_card={int(post.public_card)} coincides with a hole card {np.asarray(post.agent_cards)}",
                )
        elif int(pre.public_card) != -1:
            if int(post.public_card) != int(pre.public_card):
                self._fail(step, "public_card/immutable", "public_card changed after being revealed")

    # -- turn order / stage transitions ----------------------------------------------

    def _check_stage_monotonic(self, step, pre: LeducHoldemState, post: LeducHoldemState) -> None:
        """Stage can only stay the same or advance from 1 to 2 -- it never
        resets backward and never skips past 2 (there are only 2 betting
        rounds)."""
        if int(post.stage) < int(pre.stage) or int(post.stage) > 2:
            self._fail(
                step,
                "stage/monotonic",
                f"stage went from {int(pre.stage)} to {int(post.stage)} (must be non-decreasing, max 2)",
            )

    def _check_game_done_condition(self, step, pre: LeducHoldemState, post: LeducHoldemState) -> None:
        """The hand ends when a player folds or round 2's betting closes.
        A stage-1 betting-round close must transition to stage 2, not end
        the hand -- the game must never end while still in stage 1 unless
        someone folded."""
        remaining = 2 - int(np.asarray(post.folded).sum())
        if bool(post.done) and remaining > 1 and int(post.stage) == 1:
            self._fail(
                step,
                "win/no_fold_ends_in_stage1",
                "hand ended in stage 1 with both players still active and nobody folded",
            )

    # -- action legality / avail meta-checks -----------------------------------------

    def _check_action_legal(self, step, avail, action: int) -> None:
        if not bool(avail[action]):
            self._fail(
                step,
                "action/within_avail_mask",
                f"action {action} taken despite avail={np.asarray(avail)}",
            )

    def _check_avail_nonempty(self, step, avail) -> None:
        if not bool(jnp.asarray(avail).any()):
            self._fail(step, "action/no_deadlock", "no legal actions available for acting player")

    def _check_no_negative_chips(self, step, state: LeducHoldemState) -> None:
        if bool((jnp.asarray(state.ante) < 0).any()):
            self._fail(step, "chips/no_negative", "a player's ante went negative")

    # -- terminal reward ----------------------------------------------------------

    def _check_zero_sum_reward(self, step, reward) -> None:
        """Each player's reward is final_money - STARTING_MONEY, which is
        zero-sum by construction."""
        if not np.allclose(np.asarray(reward), 0.0):
            total = float(np.asarray(reward).sum())
            if not np.isclose(total, 0.0, atol=1e-3):
                self._fail(step, "reward/zero_sum", f"terminal rewards {np.asarray(reward)} sum to {total}, not 0")

    def _check_reward_only_on_terminal_step(self, step, done: bool, reward) -> None:
        if not done and not np.allclose(np.asarray(reward), 0.0):
            self._fail(step, "reward/only_on_terminal_step", f"nonzero reward {np.asarray(reward)} on non-terminal step")

    # -- top-level dispatch ------------------------------------------------------

    def validate_transition(self, step, pre: LeducHoldemState, action: int, avail, post: LeducHoldemState, reward, done: bool) -> None:
        self._check_avail_nonempty(step, avail)
        self._check_action_legal(step, avail, action)
        self._check_fold_only_when_facing_bet(step, pre, avail)
        self._check_call_always_legal(step, avail)
        self._check_raise_cap(step, pre, avail)

        player = int(pre.current_player_idx)
        if action == 0:
            self._check_fold_effect(step, pre, post, player)
        elif action == 1:
            self._check_call_amount(step, pre, post, player)
        elif action == 2:
            self._check_raise_amount(step, pre, post, player)

        self._check_public_card_reveal(step, pre, post)
        self._check_stage_monotonic(step, pre, post)
        self._check_game_done_condition(step, pre, post)
        self._check_no_negative_chips(step, post)
        self._check_reward_only_on_terminal_step(step, done, reward)
        if done:
            self._check_zero_sum_reward(step, reward)


def rollout_and_validate(env: LeducHoldem, kind, network, params, num_episodes: int, seed: int = DEFAULT_SEED):
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
    env = make("leduc_holdem")
    checker = rollout_and_validate(env, "random", None, None, RANDOM_EPISODES, seed=DEFAULT_SEED)
    checker.assert_no_violations()


@pytest.mark.parametrize(
    "checkpoint_info", one_checkpoint_per_algorithm_and_kind(discover_checkpoints(CHECKPOINT_ROOT), CHECKPOINT_ROOT)
)
def test_checkpoint_agent_rollouts_conform_to_rules(checkpoint_info) -> None:
    """Same check, driven by a trained checkpoint. leduc_holdem currently has
    zero checkpoints on disk (exploitability is evaluated in-memory during
    training rather than persisted), so this test discovers none and skips;
    it will run automatically once checkpoints appear under
    bluffjax/examples/leduc/checkpoints/."""
    checkpoint_path, network_kind, algorithm = checkpoint_info
    env_kwargs, fc_dim_size = training_env_kwargs(EXAMPLES_DIR, algorithm)
    env = make("leduc_holdem", **env_kwargs)

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
    agent_cards, public_card=-1, ante=(1.0, 1.0), stakes=1.0, stage=1,
    num_calls=0, num_raises=0, folded=(False, False), current_player_idx=0, timestep=0,
) -> LeducHoldemState:
    return LeducHoldemState(
        agent_cards=jnp.array(agent_cards, dtype=jnp.int32),
        public_card=jnp.int32(public_card),
        shuffled_deck=jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.int32),
        ante=jnp.array(ante, dtype=jnp.float32),
        stakes=jnp.float32(stakes),
        stage=jnp.int32(stage),
        num_calls=jnp.int32(num_calls),
        num_raises=jnp.int32(num_raises),
        folded=jnp.array(folded, dtype=bool),
        current_player_idx=jnp.int32(current_player_idx),
        absorbing=jnp.zeros(2, dtype=bool),
        done=False,
        timestep=jnp.int32(timestep),
    )


def test_edge_pair_beats_high_card_regardless_of_pair_rank() -> None:
    """A pair (private card's rank equals the public card's rank) beats any
    high-card hand. Player 0 pairs Jacks (the lowest rank) against player
    1's unpaired King -- Jacks-pair must still win."""
    env = make("leduc_holdem")
    rng = jax.random.PRNGKey(0)
    # Cards: 0=J1,1=J2,2=Q1,3=Q2,4=K1,5=K2 (rank = card // 2).
    state = _make_state(agent_cards=[0, 4], public_card=1, stage=2, ante=(3.0, 3.0), stakes=3.0)
    # Both check to reach showdown immediately.
    s1, *_ = env.step_env(rng, state, jnp.int32(1))  # player 0 calls/checks
    final, obs, reward, absorbing, done, info = env.step_env(rng, s1, jnp.int32(1))  # player 1 calls/checks
    assert bool(done)
    assert reward[0] > 0 and reward[1] < 0, (
        f"player 0's Jacks-pair should beat player 1's King-high, got reward={np.asarray(reward)}"
    )


def test_edge_tie_splits_pot_evenly() -> None:
    """Ties split the pot. Two different-suit Kings against the same public
    Jack both score King-high identically -- net reward must be exactly 0
    for both (each gets back exactly what they put in)."""
    env = make("leduc_holdem")
    rng = jax.random.PRNGKey(0)
    state = _make_state(agent_cards=[4, 5], public_card=0, stage=2, ante=(3.0, 3.0), stakes=3.0)
    s1, *_ = env.step_env(rng, state, jnp.int32(1))
    final, obs, reward, absorbing, done, info = env.step_env(rng, s1, jnp.int32(1))
    assert bool(done)
    assert np.allclose(np.asarray(reward), 0.0, atol=1e-4), (
        f"a tied showdown must return each player's own ante, net reward 0; got {np.asarray(reward)}"
    )


def test_edge_raise_cap_blocks_third_raise() -> None:
    """Each round allows at most 2 raises. After 2 raises, action 2 (raise)
    must become illegal for the next actor."""
    env = make("leduc_holdem")
    rng = jax.random.PRNGKey(0)
    state = _make_state(agent_cards=[0, 2], stage=1, ante=(1.0, 1.0), stakes=1.0, num_raises=0)

    s1, *_ = env.step_env(rng, state, jnp.int32(2))  # player 0 raises (1st raise)
    assert int(s1.num_raises) == 1
    avail1 = env.get_avail_actions(s1)
    assert bool(avail1[2]), "raise should still be legal after only 1 raise"

    s2, *_ = env.step_env(rng, s1, jnp.int32(2))  # player 1 raises (2nd raise)
    assert int(s2.num_raises) == 2
    avail2 = env.get_avail_actions(s2)
    assert not bool(avail2[2]), "raise must be illegal once num_raises reaches MAX_RAISES=2"
    assert bool(avail2[1]), "call must remain legal"


def test_edge_fold_illegal_when_not_facing_a_bet() -> None:
    """Fold is only a legal action when facing a bet. At the start of a
    fresh betting round (stakes == my_ante, nothing to call), folding must
    not be offered."""
    env = make("leduc_holdem")
    state = _make_state(agent_cards=[0, 2], stage=2, ante=(3.0, 3.0), stakes=3.0)
    avail = env.get_avail_actions(state)
    assert not bool(avail[0]), "fold must be illegal when the player faces no outstanding bet"


def test_edge_raise_size_differs_by_round() -> None:
    """The raise increment is smaller in round 1 and larger in round 2
    (canonically 2 and 4 chips)."""
    env = make("leduc_holdem")
    rng = jax.random.PRNGKey(0)

    r1_state = _make_state(agent_cards=[0, 2], stage=1, ante=(1.0, 1.0), stakes=1.0)
    r1_next, *_ = env.step_env(rng, r1_state, jnp.int32(2))
    assert np.isclose(float(r1_next.stakes), 1.0 + env.raise_amount_r1), (
        f"round-1 raise should add {env.raise_amount_r1} to stakes"
    )

    r2_state = _make_state(agent_cards=[0, 2], stage=2, ante=(3.0, 3.0), stakes=3.0)
    r2_next, *_ = env.step_env(rng, r2_state, jnp.int32(2))
    assert np.isclose(float(r2_next.stakes), 3.0 + env.raise_amount_r2), (
        f"round-2 raise should add {env.raise_amount_r2} to stakes"
    )


# ---------------------------------------------------------------------------
# Standalone CLI.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--network-type", type=str, default=None, choices=["actor", "actor_critic", "q_network"])
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    if not args.random and args.checkpoint is None:
        parser.error("pass --checkpoint PATH or --random")

    env = make("leduc_holdem")

    if args.random:
        kind, network, params = "random", None, None
    else:
        kind = args.network_type or infer_network_kind(args.checkpoint)
        network = build_network(kind, action_dim=env.num_actions, hidden_dim=args.hidden_dim)
        _, sample_obs = env.reset(jax.random.PRNGKey(0))
        params = load_checkpoint_params(network, sample_obs, args.checkpoint)

    print(f"Validating {args.episodes} episode(s) of '{kind}' agent (leduc_holdem)...")
    checker = rollout_and_validate(env, kind, network, params, args.episodes, seed=args.seed)

    if checker.violations:
        print(f"\n{len(checker.violations)} VIOLATION(S) FOUND:")
        for v in checker.violations[:20]:
            print(f"  {v}")
        raise SystemExit(1)
    print("\nNo rule violations found.")


if __name__ == "__main__":
    main()
