"""
Rule-conformance validation suite for Texas No-Limit Hold'em.

Checks every state/action/transition produced by random or checkpoint-driven
rollouts against bluffjax/environments/texas_nolimit_holdem/rules.md and the
implementation in
bluffjax/environments/texas_nolimit_holdem/texas_nolimit_holdem.py.

Usage:
    pytest bluffjax/validation_suite/texas_nolimit_holdem_validation.py -v
    python texas_nolimit_holdem_validation.py --random --episodes 300

Note: texas_nolimit_holdem currently has zero checkpoints on disk. The
checkpoint-driven test is fully implemented but will report 0 discovered
checkpoints and skip.
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.environments.texas_nolimit_holdem.texas_nolimit_holdem import (
    TexasNoLimitHoldem,
    TexasNoLimitHoldEmState,
)
from bluffjax.utils.game_utils.poker_utils import _compare_hands
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
EXAMPLES_DIR = os.path.normpath(os.path.join(HERE, "..", "examples", "texas_nolimit_holdem"))
CHECKPOINT_ROOT = os.path.join(EXAMPLES_DIR, "checkpoints")

DEFAULT_SEED = 0
RANDOM_EPISODES = 150
CHECKPOINT_EPISODES = 40


class RuleChecker(RuleCheckerBase):
    def __init__(self, env: TexasNoLimitHoldem):
        super().__init__(env)

    # -- setup / reset --------------------------------------------------------

    def check_reset(self, state: TexasNoLimitHoldEmState) -> None:
        """Both players' hole cards and all five community cards are dealt
        up front in reset(), and blinds are posted before any action."""
        hole = np.asarray(state.agent_cards)
        community = np.concatenate([np.asarray(state.flop_cards), [int(state.turn_card)], [int(state.river_card)]])
        all_cards = np.concatenate([hole.reshape(-1), community])
        if len(set(all_cards.tolist())) != all_cards.size:
            self._fail(0, "setup/distinct_cards", "dealt hole + community cards contain a duplicate")
        sb = int(state.small_blind_idx)
        bb = (sb + 1) % self.env.num_agents
        chips_in = np.asarray(state.chips_in)
        if not np.isclose(chips_in[sb], self.env.small_blind):
            self._fail(0, "setup/small_blind_posted", f"seat {sb} chips_in={chips_in[sb]}")
        if not np.isclose(chips_in[bb], self.env.big_blind):
            self._fail(0, "setup/big_blind_posted", f"seat {bb} chips_in={chips_in[bb]}")
        remaining = np.asarray(state.remaining_chips)
        expected_remaining = self.env.init_chips - chips_in
        if not np.allclose(remaining, expected_remaining):
            self._fail(0, "setup/starting_stack", f"remaining_chips={remaining} != init_chips - blinds = {expected_remaining}")
        if self.env.num_agents == 2 and int(state.current_player_idx) != sb:
            self._fail(0, "setup/heads_up_sb_acts_first", "with 2 players, small blind must act first pre-flop")
        if int(state.stage) != 0:
            self._fail(0, "setup/initial_stage", "hand must start at stage 0 (pre-flop)")

    # -- action legality --------------------------------------------------------

    def _check_betting_avail(self, step, pre: TexasNoLimitHoldEmState, avail) -> None:
        """Action 0 (check/call) is legal whenever the player is still
        active. Actions 1/2 (raise-half/raise-pot) are gated on stack size
        and on being a genuine raise. Action 3 (all-in) is only offered
        when the player has more than enough to call -- it's a shove for a
        raise, not a call for less. Action 4 (fold) is always legal while
        active."""
        cur = int(pre.current_player_idx)
        player_round = float(pre.round_raised[cur])
        max_round = float(np.max(pre.round_raised))
        player_remain = float(pre.remaining_chips[cur])
        pot = float(np.sum(pre.chips_in))
        half_pot = float(np.floor(pot / 2.0))
        diff = max_round - player_round
        is_active = not bool(pre.folded[cur]) and not bool(pre.all_in[cur]) and player_remain > 0

        can_raise_any = is_active and (diff < player_remain)
        can_raise_pot = can_raise_any and (pot <= player_remain)
        can_raise_half = can_raise_any and (half_pot <= player_remain) and ((half_pot + player_round) > max_round)

        expectations = {0: is_active, 1: can_raise_half, 2: can_raise_pot, 3: can_raise_any, 4: is_active}
        for action, expected in expectations.items():
            if bool(avail[action]) != expected:
                self._fail(step, f"action/betting_avail_{action}", f"avail[{action}]={bool(avail[action])} != expected {expected}")

    def _check_bet_amount(self, step, pre: TexasNoLimitHoldEmState, post: TexasNoLimitHoldEmState, player: int, action: int) -> None:
        """Check/call adds `diff`; raise-half adds `floor(pot/2)`; raise-pot
        adds `pot`; all-in adds the player's entire remaining stack -- each
        is the player's whole new contribution."""
        max_round = float(np.max(pre.round_raised))
        player_round = float(pre.round_raised[player])
        diff = max_round - player_round
        pot = float(np.sum(pre.chips_in))
        half_pot = float(np.floor(pot / 2.0))
        remain = float(pre.remaining_chips[player])
        expected_bet = {0: diff, 1: half_pot, 2: pot, 3: remain}.get(action)
        if expected_bet is None:
            return
        actual_bet = float(post.chips_in[player]) - float(pre.chips_in[player])
        if not np.isclose(actual_bet, expected_bet, atol=1e-4):
            self._fail(step, "chips/bet_amount", f"action={action} added {actual_bet} chips, expected {expected_bet}")

    def _check_fold_effect(self, step, pre: TexasNoLimitHoldEmState, post: TexasNoLimitHoldEmState, player: int) -> None:
        if not bool(post.folded[player]):
            self._fail(step, "fold/marks_folded", f"player {player} took fold action but folded flag not set")
        if not np.allclose(np.asarray(post.chips_in), np.asarray(pre.chips_in)):
            self._fail(step, "fold/no_chip_change", "folding must not move chips")

    def _check_all_in_flag_and_no_negative_stack(self, step, post: TexasNoLimitHoldEmState) -> None:
        remaining = np.asarray(post.remaining_chips)
        if (remaining < 0).any():
            self._fail(step, "chips/no_negative_stack", f"remaining_chips went negative: {remaining}")
        for p in range(self.env.num_agents):
            should_be_all_in = remaining[p] <= 0
            if bool(post.all_in[p]) != should_be_all_in:
                self._fail(step, "chips/all_in_flag", f"player {p} remaining={remaining[p]} but all_in={bool(post.all_in[p])}")

    # -- community-card reveal (dealt up front, "run out" automatically) --------------

    def _check_community_cards_immutable(self, step, pre: TexasNoLimitHoldEmState, post: TexasNoLimitHoldEmState) -> None:
        """All five community cards are dealt up front in reset(), so the
        cards used at showdown are fixed from the start of the hand -- an
        early all-in is automatically run out to a full 5-card board with no
        special-case logic needed."""
        if not np.array_equal(np.asarray(post.flop_cards), np.asarray(pre.flop_cards)):
            self._fail(step, "cards/flop_immutable", "flop_cards changed mid-hand")
        if int(post.turn_card) != int(pre.turn_card) or int(post.river_card) != int(pre.river_card):
            self._fail(step, "cards/turn_river_immutable", "turn_card/river_card changed mid-hand")

    # -- termination / showdown -----------------------------------------------------

    def _check_reward_only_on_terminal_step(self, step, done: bool, reward) -> None:
        if not done and not np.allclose(np.asarray(reward), 0.0):
            self._fail(step, "reward/only_on_terminal_step", f"nonzero reward {np.asarray(reward)} on non-terminal step")

    def _check_zero_sum_reward(self, step, reward) -> None:
        total = float(np.asarray(reward).sum())
        if not np.isclose(total, 0.0, atol=1e-3):
            self._fail(step, "reward/zero_sum", f"terminal rewards {np.asarray(reward)} sum to {total}, not 0")

    def _check_game_done_condition(self, step, post: TexasNoLimitHoldEmState, done: bool) -> None:
        num_active = int((~np.asarray(post.folded)).sum())
        num_playable = int((~np.asarray(post.folded) & ~np.asarray(post.all_in)).sum())
        expected = (num_active <= 1) or (num_playable == 0) or (int(post.stage) > 3)
        if bool(done) != expected:
            self._fail(
                step,
                "win/termination_condition",
                f"done={bool(done)} != expected {expected} (num_active={num_active}, num_playable={num_playable}, stage={int(post.stage)})",
            )

    def _check_showdown_awards_winner(self, step, pre: TexasNoLimitHoldEmState, post: TexasNoLimitHoldEmState, reward) -> None:
        folded = np.asarray(post.folded)
        num_active = int((~folded).sum())
        community = np.concatenate([np.asarray(post.flop_cards), [int(post.turn_card)], [int(post.river_card)]])
        pot_total = float(np.asarray(post.chips_in).sum())
        if num_active > 1:
            all_hands = np.concatenate([np.asarray(post.agent_cards), np.tile(community, (self.env.num_agents, 1))], axis=1)
            winners_mask = np.asarray(_compare_hands(jnp.array(all_hands), jnp.array(folded)))
            winners = [p for p in range(self.env.num_agents) if winners_mask[p]]
        else:
            winners = [p for p in range(self.env.num_agents) if not folded[p]]
        expected_reward = np.array([-float(post.chips_in[p]) for p in range(self.env.num_agents)])
        for w in winners:
            expected_reward[w] += pot_total / len(winners)
        if not np.allclose(np.asarray(reward), expected_reward, atol=1e-3):
            self._fail(step, "showdown/pot_awarded_correctly", f"winners={winners}, expected {expected_reward}, got {np.asarray(reward)}")

    # -- action legality / no-deadlock --------------------------------------------

    def _check_action_legal(self, step, avail, action: int) -> None:
        if not bool(avail[action]):
            self._fail(step, "action/within_avail_mask", f"action {action} taken despite avail mask")

    def _check_avail_nonempty(self, step, avail) -> None:
        if not bool(jnp.asarray(avail).any()):
            self._fail(step, "action/no_deadlock", "no legal actions available for acting player")

    # -- top-level dispatch ------------------------------------------------------

    def validate_transition(self, step, pre: TexasNoLimitHoldEmState, action: int, avail, post: TexasNoLimitHoldEmState, reward, done: bool) -> None:
        self._check_avail_nonempty(step, avail)
        self._check_action_legal(step, avail, action)
        self._check_betting_avail(step, pre, avail)

        player = int(pre.current_player_idx)
        if action == 4:
            self._check_fold_effect(step, pre, post, player)
        else:
            self._check_bet_amount(step, pre, post, player, action)
        self._check_all_in_flag_and_no_negative_stack(step, post)
        self._check_community_cards_immutable(step, pre, post)
        self._check_game_done_condition(step, post, done)
        self._check_reward_only_on_terminal_step(step, done, reward)
        if done:
            self._check_zero_sum_reward(step, reward)
            self._check_showdown_awards_winner(step, pre, post, reward)


def rollout_and_validate(env: TexasNoLimitHoldem, kind, network, params, num_episodes: int, seed: int = DEFAULT_SEED):
    checker = RuleChecker(env)
    rng = jax.random.PRNGKey(seed)

    for _ in range(num_episodes):
        checker.start_episode()
        rng, reset_rng = jax.random.split(rng)
        state, obs = env.reset(reset_rng)
        checker.check_reset(state)

        for step in range(min(env.horizon, 500) + 1):
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
    env = make("texas_nolimit_holdem", num_agents=2, horizon=100_000, init_chips=100)
    checker = rollout_and_validate(env, "random", None, None, RANDOM_EPISODES, seed=DEFAULT_SEED)
    checker.assert_no_violations()


@pytest.mark.parametrize(
    "checkpoint_info", one_checkpoint_per_algorithm_and_kind(discover_checkpoints(CHECKPOINT_ROOT), CHECKPOINT_ROOT)
)
def test_checkpoint_agent_rollouts_conform_to_rules(checkpoint_info) -> None:
    """texas_nolimit_holdem currently has zero checkpoints on disk; this is
    fully wired up and will run automatically once any appear."""
    checkpoint_path, network_kind, algorithm = checkpoint_info
    env_kwargs, fc_dim_size = training_env_kwargs(EXAMPLES_DIR, algorithm)
    env = make("texas_nolimit_holdem", **env_kwargs)

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
    env: TexasNoLimitHoldem, agent_cards, chips_in, round_raised, remaining_chips,
    flop_cards=(40, 41, 42), turn_card=43, river_card=44,
    small_blind_idx=0, current_player_idx=None, stage=0, folded=None, all_in=None,
) -> TexasNoLimitHoldEmState:
    n = env.num_agents
    if folded is None:
        folded = [False] * n
    if all_in is None:
        all_in = [False] * n
    if current_player_idx is None:
        current_player_idx = small_blind_idx
    return TexasNoLimitHoldEmState(
        flop_cards=jnp.array(flop_cards, dtype=jnp.int32),
        turn_card=jnp.int32(turn_card),
        river_card=jnp.int32(river_card),
        agent_cards=jnp.array(agent_cards, dtype=jnp.int32),
        chips_in=jnp.array(chips_in, dtype=jnp.float32),
        round_raised=jnp.array(round_raised, dtype=jnp.float32),
        remaining_chips=jnp.array(remaining_chips, dtype=jnp.float32),
        small_blind_idx=jnp.int32(small_blind_idx),
        current_player_idx=jnp.int32(current_player_idx),
        stage=jnp.int32(stage),
        folded=jnp.array(folded, dtype=bool),
        all_in=jnp.array(all_in, dtype=bool),
        not_raise_num=jnp.int32(0),
        absorbing=jnp.zeros(n, dtype=bool),
        done=False,
        timestep=0,
    )


def test_edge_raise_half_pot_illegal_when_it_does_not_exceed_call() -> None:
    """Raise-half-pot must be a genuine raise, not a tie or under-call.
    Constructs a pot where half_pot exactly equals the call amount, which
    must make the raise illegal."""
    env = make("texas_nolimit_holdem", num_agents=2, horizon=100_000, init_chips=100)
    state = _make_state(
        env, agent_cards=[[0, 1], [2, 3]], chips_in=[5.0, 5.0], round_raised=[0.0, 5.0],
        remaining_chips=[95.0, 95.0], stage=0, current_player_idx=0,
    )
    avail = env.get_avail_actions(state)
    assert not bool(avail[1]), "half-pot raise must be illegal when it would only match the call, not exceed it"
    assert bool(avail[0]), "check/call must still be legal"


def test_edge_all_in_only_offered_as_a_shove_not_a_call() -> None:
    """All-in is only offered when the player has more than enough to call
    -- it's a shove for a raise, not a call for less. If the player's
    entire remaining stack is exactly the call amount, all-in must not be
    separately offered; check/call already covers that case."""
    env = make("texas_nolimit_holdem", num_agents=2, horizon=100_000, init_chips=100)
    state = _make_state(
        env, agent_cards=[[0, 1], [2, 3]], chips_in=[95.0, 100.0], round_raised=[0.0, 5.0],
        remaining_chips=[5.0, 0.0], stage=0, current_player_idx=0,
    )
    avail = env.get_avail_actions(state)
    assert not bool(avail[3]), "all-in must be illegal when remaining chips exactly equal the call amount"
    assert bool(avail[0]), "check/call (which uses the whole remaining stack here) must be legal"


def test_edge_early_all_in_runs_out_full_board_automatically() -> None:
    """An early all-in is automatically run out to a full 5-card board with
    no special-case logic needed: since all 5 community cards are dealt at
    reset() regardless of stage, a showdown triggered by an all-in during
    the pre-flop round still uses the complete board."""
    env = make("texas_nolimit_holdem", num_agents=2, horizon=100_000, init_chips=100)
    rng = jax.random.PRNGKey(0)
    state = _make_state(
        env, agent_cards=[[0, 1], [2, 3]], chips_in=[1.0, 2.0], round_raised=[1.0, 2.0],
        remaining_chips=[99.0, 98.0], stage=0, current_player_idx=0,
    )
    all_in_state, *_ = env.step_env(rng, state, jnp.int32(3))  # player 0 goes all-in
    call_state, obs, reward, absorbing, done, info = env.step_env(rng, all_in_state, jnp.int32(0))  # player 1 calls
    assert bool(done)
    # `stage` advances mechanically, but the board is fixed at reset()
    # regardless of stage, so the full 5-card board is already available.
    assert np.array_equal(np.asarray(call_state.flop_cards), np.asarray(state.flop_cards))
    assert int(call_state.turn_card) == int(state.turn_card)
    assert int(call_state.river_card) == int(state.river_card)


def test_edge_fold_gives_pot_to_sole_remaining_player_without_showdown() -> None:
    """Showdown happens only among non-folded players; a fold leaving one
    player standing awards the pot uncontested."""
    env = make("texas_nolimit_holdem", num_agents=2, horizon=100_000, init_chips=100)
    rng = jax.random.PRNGKey(0)
    state = _make_state(
        env, agent_cards=[[0, 1], [2, 3]], chips_in=[6.0, 4.0], round_raised=[6.0, 4.0],
        remaining_chips=[94.0, 96.0], stage=1, current_player_idx=1,
    )
    next_state, obs, reward, absorbing, done, info = env.step_env(rng, state, jnp.int32(4))  # player 1 folds
    assert bool(done)
    assert np.isclose(float(reward[0]), 4.0)
    assert np.isclose(float(reward[1]), -4.0)


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

    env = make("texas_nolimit_holdem", num_agents=2, horizon=100_000, init_chips=100)

    if args.random:
        kind, network, params = "random", None, None
    else:
        kind = args.network_type or infer_network_kind(args.checkpoint)
        network = build_network(kind, action_dim=env.num_actions, hidden_dim=args.hidden_dim)
        _, sample_obs = env.reset(jax.random.PRNGKey(0))
        params = load_checkpoint_params(network, sample_obs, args.checkpoint)

    print(f"Validating {args.episodes} episode(s) of '{kind}' agent (texas_nolimit_holdem)...")
    checker = rollout_and_validate(env, kind, network, params, args.episodes, seed=args.seed)

    if checker.violations:
        print(f"\n{len(checker.violations)} VIOLATION(S) FOUND:")
        for v in checker.violations[:20]:
            print(f"  {v}")
        raise SystemExit(1)
    print("\nNo rule violations found.")


if __name__ == "__main__":
    main()
