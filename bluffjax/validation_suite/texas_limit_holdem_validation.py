"""
Rule-conformance validation suite for Texas Limit Hold'em.

Checks every state/action/transition produced by random or checkpoint-driven
rollouts against bluffjax/environments/texas_limit_holdem/rules.md and the
implementation in bluffjax/environments/texas_limit_holdem/texas_limit_holdem.py.

Usage:
    pytest bluffjax/validation_suite/texas_limit_holdem_validation.py -v
    python texas_limit_holdem_validation.py --random --episodes 300

Note: texas_limit_holdem currently has zero checkpoints on disk. The
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
from bluffjax.environments.texas_limit_holdem.texas_limit_holdem import TexasLimitHoldem, TexasLimitHoldEmState
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
EXAMPLES_DIR = os.path.normpath(os.path.join(HERE, "..", "examples", "texas_limit_holdem"))
CHECKPOINT_ROOT = os.path.join(EXAMPLES_DIR, "checkpoints")

DEFAULT_SEED = 0
RANDOM_EPISODES = 150
CHECKPOINT_EPISODES = 40


class RuleChecker(RuleCheckerBase):
    def __init__(self, env: TexasLimitHoldem):
        super().__init__(env)

    # -- setup / reset --------------------------------------------------------

    def check_reset(self, state: TexasLimitHoldEmState) -> None:
        """Blinds are posted before cards are looked at; each player is dealt
        2 hole cards; 5 community cards (3 flop, 1 turn, 1 river) are dealt
        but unrevealed. With num_agents=2, current_player_idx starts at the
        small blind seat pre-flop."""
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
        if self.env.num_agents == 2 and int(state.current_player_idx) != sb:
            self._fail(0, "setup/heads_up_sb_acts_first", "with 2 players, small blind must act first pre-flop")
        if int(state.stage) != 0:
            self._fail(0, "setup/initial_stage", "hand must start at stage 0 (pre-flop)")

    # -- action legality / arithmetic -------------------------------------------------

    def _check_betting_avail(self, step, pre: TexasLimitHoldEmState, avail) -> None:
        cur = int(pre.current_player_idx)
        player_chips = float(pre.chips_in[cur])
        max_chips = float(np.max(pre.chips_in))
        can_raise = int(pre.raise_nums[pre.stage]) < self.env.allowed_raise_num
        can_call = player_chips < max_chips
        can_check = player_chips == max_chips
        expectations = {0: can_call, 1: can_raise, 2: True, 3: can_check}
        for action, expected in expectations.items():
            if bool(avail[action]) != expected:
                self._fail(step, f"action/betting_avail_{action}", f"avail[{action}]={bool(avail[action])} != expected {expected}")

    def _check_raise_cap_never_exceeded(self, step, state: TexasLimitHoldEmState) -> None:
        if int(np.max(np.asarray(state.raise_nums))) > self.env.allowed_raise_num:
            self._fail(step, "action/raise_cap_exceeded", f"raise_nums={np.asarray(state.raise_nums)} exceeds cap {self.env.allowed_raise_num}")

    def _check_bet_amount(self, step, pre: TexasLimitHoldEmState, post: TexasLimitHoldEmState, player: int, action: int) -> None:
        """Preflop and flop bets/raises equal the big blind; turn and river
        bets/raises equal twice the big blind."""
        max_chips = float(np.max(pre.chips_in))
        player_chips = float(pre.chips_in[player])
        raise_amount = self.env.raise_amount * 2 if int(pre.stage) >= 2 else self.env.raise_amount
        if action == 0:
            expected_delta = max_chips - player_chips
        elif action == 1:
            expected_delta = max_chips - player_chips + raise_amount
        else:
            return
        actual_delta = float(post.chips_in[player]) - float(pre.chips_in[player])
        if not np.isclose(actual_delta, expected_delta, atol=1e-4):
            self._fail(step, "chips/bet_amount", f"action={action} (stage={int(pre.stage)}) added {actual_delta}, expected {expected_delta}")
        if action == 1 and int(post.raise_nums[pre.stage]) != int(pre.raise_nums[pre.stage]) + 1:
            self._fail(step, "chips/raise_count_increment", "raise_nums[stage] must increment by exactly 1 on a raise")

    def _check_fold_and_check_no_chip_change(self, step, pre: TexasLimitHoldEmState, post: TexasLimitHoldEmState, player: int, action: int) -> None:
        if action in (2, 3) and not np.isclose(float(post.chips_in[player]), float(pre.chips_in[player])):
            self._fail(step, "chips/fold_check_no_change", f"action={action} must not move chips")
        if action == 2 and not bool(post.folded[player]):
            self._fail(step, "fold/marks_folded", f"player {player} folded but folded flag not set")

    # -- community-card reveal --------------------------------------------------------

    def _check_community_cards_immutable(self, step, pre: TexasLimitHoldEmState, post: TexasLimitHoldEmState) -> None:
        """Flop/turn/river cards are all dealt up front at reset and never
        change value once dealt; only their visibility (gated by stage)
        changes."""
        if not np.array_equal(np.asarray(post.flop_cards), np.asarray(pre.flop_cards)):
            self._fail(step, "cards/flop_immutable", "flop_cards changed mid-hand")
        if int(post.turn_card) != int(pre.turn_card) or int(post.river_card) != int(pre.river_card):
            self._fail(step, "cards/turn_river_immutable", "turn_card/river_card changed mid-hand")

    def _check_postflop_first_actor_is_small_blind(self, step, pre: TexasLimitHoldEmState, post: TexasLimitHoldEmState) -> None:
        """Postflop first-to-act is always the small blind -- a deliberate
        simplification versus standard heads-up play, where the big blind
        acts first postflop."""
        if int(post.stage) != int(pre.stage) and int(post.stage) >= 1:
            sb = int(pre.small_blind_idx)
            if not bool(post.folded[sb]):
                if int(post.current_player_idx) != sb:
                    self._fail(
                        step,
                        "turn_order/postflop_starts_with_small_blind",
                        f"new stage {int(post.stage)}: current_player_idx={int(post.current_player_idx)} != small_blind {sb}",
                    )

    # -- termination / showdown -----------------------------------------------------

    def _check_reward_only_on_terminal_step(self, step, done: bool, reward) -> None:
        if not done and not np.allclose(np.asarray(reward), 0.0):
            self._fail(step, "reward/only_on_terminal_step", f"nonzero reward {np.asarray(reward)} on non-terminal step")

    def _check_zero_sum_reward(self, step, reward) -> None:
        total = float(np.asarray(reward).sum())
        if not np.isclose(total, 0.0, atol=1e-3):
            self._fail(step, "reward/zero_sum", f"terminal rewards {np.asarray(reward)} sum to {total}, not 0")

    def _check_game_done_condition(self, step, post: TexasLimitHoldEmState, done: bool) -> None:
        """A hand ends when only one player remains, or all 4 betting
        rounds are complete."""
        num_active = int((~np.asarray(post.folded)).sum())
        expected = (num_active <= 1) or (int(post.stage) >= 4)
        if bool(done) != expected:
            self._fail(step, "win/termination_condition", f"done={bool(done)} != expected {expected}")

    def _check_showdown_awards_winner(self, step, pre: TexasLimitHoldEmState, post: TexasLimitHoldEmState, reward) -> None:
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
        expected_reward = expected_reward / self.env.big_blind
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

    def validate_transition(self, step, pre: TexasLimitHoldEmState, action: int, avail, post: TexasLimitHoldEmState, reward, done: bool) -> None:
        self._check_avail_nonempty(step, avail)
        self._check_action_legal(step, avail, action)
        self._check_betting_avail(step, pre, avail)

        player = int(pre.current_player_idx)
        self._check_bet_amount(step, pre, post, player, action)
        self._check_fold_and_check_no_chip_change(step, pre, post, player, action)
        self._check_raise_cap_never_exceeded(step, post)
        self._check_community_cards_immutable(step, pre, post)
        self._check_postflop_first_actor_is_small_blind(step, pre, post)
        self._check_game_done_condition(step, post, done)
        self._check_reward_only_on_terminal_step(step, done, reward)
        if done:
            self._check_zero_sum_reward(step, reward)
            self._check_showdown_awards_winner(step, pre, post, reward)


def rollout_and_validate(env: TexasLimitHoldem, kind, network, params, num_episodes: int, seed: int = DEFAULT_SEED):
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
    env = make("texas_limit_holdem", num_agents=2, horizon=100_000)
    checker = rollout_and_validate(env, "random", None, None, RANDOM_EPISODES, seed=DEFAULT_SEED)
    checker.assert_no_violations()


@pytest.mark.parametrize(
    "checkpoint_info", one_checkpoint_per_algorithm_and_kind(discover_checkpoints(CHECKPOINT_ROOT), CHECKPOINT_ROOT)
)
def test_checkpoint_agent_rollouts_conform_to_rules(checkpoint_info) -> None:
    """texas_limit_holdem currently has zero checkpoints on disk; this is
    fully wired up and will run automatically once any appear."""
    checkpoint_path, network_kind, algorithm = checkpoint_info
    env_kwargs, fc_dim_size = training_env_kwargs(EXAMPLES_DIR, algorithm)
    env = make("texas_limit_holdem", **env_kwargs)

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
    env: TexasLimitHoldem, agent_cards, chips_in, flop_cards=(40, 41, 42), turn_card=43, river_card=44,
    small_blind_idx=0, current_player_idx=None, stage=0, raise_nums=None, folded=None,
) -> TexasLimitHoldEmState:
    n = env.num_agents
    if folded is None:
        folded = [False] * n
    if raise_nums is None:
        raise_nums = [0] * env.num_betting_rounds
    if current_player_idx is None:
        current_player_idx = small_blind_idx
    return TexasLimitHoldEmState(
        flop_cards=jnp.array(flop_cards, dtype=jnp.int32),
        turn_card=jnp.int32(turn_card),
        river_card=jnp.int32(river_card),
        agent_cards=jnp.array(agent_cards, dtype=jnp.int32),
        chips_in=jnp.array(chips_in, dtype=jnp.float32),
        small_blind_idx=jnp.int32(small_blind_idx),
        current_player_idx=jnp.int32(current_player_idx),
        stage=jnp.int32(stage),
        raise_nums=jnp.array(raise_nums, dtype=jnp.int32),
        folded=jnp.array(folded, dtype=bool),
        not_raise_num=jnp.int32(0),
        absorbing=jnp.zeros(n, dtype=bool),
        done=False,
        timestep=0,
    )


def test_edge_raise_cap_blocks_fifth_bet() -> None:
    """The academic HULHE benchmark keeps a fixed cap of 4 raises per
    round even heads-up."""
    env = make("texas_limit_holdem", num_agents=2, horizon=100_000)
    state = _make_state(env, agent_cards=[[0, 1], [2, 3]], chips_in=[5.0, 5.0], stage=0, raise_nums=[4, 0, 0, 0])
    avail = env.get_avail_actions(state)
    assert not bool(avail[1]), "raise must be illegal once raise_nums[stage] reaches the cap of 4"


def test_edge_raise_size_doubles_from_turn_onward() -> None:
    """A raise on the turn or river costs twice the base raise amount used
    preflop and on the flop."""
    env = make("texas_limit_holdem", num_agents=2, horizon=100_000)
    rng = jax.random.PRNGKey(0)
    preflop = _make_state(env, agent_cards=[[0, 1], [2, 3]], chips_in=[1.0, 2.0], stage=0, current_player_idx=0)
    next_preflop, *_ = env.step_env(rng, preflop, jnp.int32(1))
    assert np.isclose(float(next_preflop.chips_in[0]) - 1.0, (2.0 - 1.0) + env.raise_amount)

    turn = _make_state(env, agent_cards=[[0, 1], [2, 3]], chips_in=[1.0, 2.0], stage=2, current_player_idx=0)
    next_turn, *_ = env.step_env(rng, turn, jnp.int32(1))
    assert np.isclose(float(next_turn.chips_in[0]) - 1.0, (2.0 - 1.0) + env.raise_amount * 2), (
        "a raise on the turn (stage 2) must use double the base raise amount"
    )


def test_edge_postflop_first_actor_is_always_small_blind() -> None:
    """Postflop first-to-act is always the small blind, a deliberate
    deviation from standard heads-up play."""
    env = make("texas_limit_holdem", num_agents=2, horizon=100_000)
    rng = jax.random.PRNGKey(0)
    # Both players check preflop to close the round and advance to the flop.
    state = _make_state(env, agent_cards=[[0, 1], [2, 3]], chips_in=[2.0, 2.0], stage=0, current_player_idx=0, small_blind_idx=0)
    s1, *_ = env.step_env(rng, state, jnp.int32(3))  # player 0 (SB) checks
    s2, *_ = env.step_env(rng, s1, jnp.int32(3))  # player 1 (BB) checks -> round closes
    assert int(s2.stage) == 1, "both checking preflop must advance to the flop"
    assert int(s2.current_player_idx) == 0, "the small blind (player 0) must act first on the flop in this environment"


def test_edge_fold_gives_pot_to_sole_remaining_player_without_showdown() -> None:
    """If all but one player folds at any point before showdown, that
    player wins the pot uncontested."""
    env = make("texas_limit_holdem", num_agents=2, horizon=100_000)
    rng = jax.random.PRNGKey(0)
    state = _make_state(env, agent_cards=[[0, 1], [2, 3]], chips_in=[6.0, 4.0], stage=1, current_player_idx=1)
    next_state, obs, reward, absorbing, done, info = env.step_env(rng, state, jnp.int32(2))  # player 1 folds
    assert bool(done)
    assert np.isclose(float(reward[0]), 4.0 / env.big_blind)
    assert np.isclose(float(reward[1]), -4.0 / env.big_blind)


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

    env = make("texas_limit_holdem", num_agents=2, horizon=100_000)

    if args.random:
        kind, network, params = "random", None, None
    else:
        kind = args.network_type or infer_network_kind(args.checkpoint)
        network = build_network(kind, action_dim=env.num_actions, hidden_dim=args.hidden_dim)
        _, sample_obs = env.reset(jax.random.PRNGKey(0))
        params = load_checkpoint_params(network, sample_obs, args.checkpoint)

    print(f"Validating {args.episodes} episode(s) of '{kind}' agent (texas_limit_holdem)...")
    checker = rollout_and_validate(env, kind, network, params, args.episodes, seed=args.seed)

    if checker.violations:
        print(f"\n{len(checker.violations)} VIOLATION(S) FOUND:")
        for v in checker.violations[:20]:
            print(f"  {v}")
        raise SystemExit(1)
    print("\nNo rule violations found.")


if __name__ == "__main__":
    main()
