"""
Rule-conformance validation suite for Seven Card Stud poker.

Checks every state/action/transition produced by random or checkpoint-driven
rollouts against bluffjax/environments/seven_card_stud/rules.md and the
implementation in bluffjax/environments/seven_card_stud/seven_card_stud.py.

Usage:
    pytest bluffjax/validation_suite/seven_card_stud_validation.py -v
    python seven_card_stud_validation.py --random --episodes 300
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.environments.seven_card_stud.seven_card_stud import SevenCardStud, SevenCardStudState
from bluffjax.utils.game_utils.poker_utils import _card_rank, _compare_hands, _get_bring_in_idx, _score_seven_card_hand
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
EXAMPLES_DIR = os.path.normpath(os.path.join(HERE, "..", "examples", "seven_card_stud"))
CHECKPOINT_ROOT = os.path.join(EXAMPLES_DIR, "checkpoints")

DEFAULT_SEED = 0
RANDOM_EPISODES = 150
CHECKPOINT_EPISODES = 40


class RuleChecker(RuleCheckerBase):
    def __init__(self, env: SevenCardStud):
        super().__init__(env)

    # -- setup / reset --------------------------------------------------------

    def check_reset(self, state: SevenCardStudState) -> None:
        """Every player posts `ante=0.5`; the player with the lowest door
        card (index 2) additionally posts `bring_in=0.5`, breaking ties by
        suit clubs<diamonds<hearts<spades."""
        hands = np.asarray(state.agent_cards)
        if len(set(hands.reshape(-1).tolist())) != hands.size:
            self._fail(0, "setup/distinct_cards", "dealt hands contain a duplicate card")
        chips_in = np.asarray(state.chips_in)
        bring_in_idx = int(state.bring_in_idx)
        for p in range(self.env.num_agents):
            expected = self.env.ante + (self.env.bring_in if p == bring_in_idx else 0.0)
            if not np.isclose(chips_in[p], expected):
                self._fail(0, "setup/ante_and_bring_in", f"player {p} chips_in={chips_in[p]}, expected {expected}")

        door_cards = hands[:, 2]
        expected_bring_in = int(_get_bring_in_idx(jnp.array(door_cards)))
        if bring_in_idx != expected_bring_in:
            self._fail(
                0,
                "setup/bring_in_selection",
                f"bring_in_idx={bring_in_idx} != recomputed lowest-door-card idx {expected_bring_in} "
                f"(door_cards={door_cards})",
            )
        if int(state.stage) != 0:
            self._fail(0, "setup/initial_stage", "hand must start at stage 0 (3rd street)")

    # -- action legality --------------------------------------------------------

    def _check_betting_avail(self, step, pre: SevenCardStudState, avail) -> None:
        """Action indices are 0=call, 1=raise, 2=fold, 3=check. Raise is
        gated on the per-street raise cap (4); call and check are mutually
        exclusive based on whether the player is already matched to the
        current max."""
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

    def _check_raise_cap_never_exceeded(self, step, state: SevenCardStudState) -> None:
        if int(np.max(np.asarray(state.raise_nums))) > self.env.allowed_raise_num:
            self._fail(step, "action/raise_cap_exceeded", f"raise_nums={np.asarray(state.raise_nums)} exceeds cap {self.env.allowed_raise_num}")

    def _check_bet_amount(self, step, pre: SevenCardStudState, post: SevenCardStudState, player: int, action: int) -> None:
        """Bets are the small bet on 3rd and 4th street, and the big bet
        (double) from 5th street onward. A call matches `max_chips`; a
        raise matches `max_chips` and adds one further increment."""
        max_chips = float(np.max(pre.chips_in))
        player_chips = float(pre.chips_in[player])
        raise_amount = self.env.big_bet if int(pre.stage) >= 2 else self.env.small_bet
        if action == 0:  # call
            expected_delta = max_chips - player_chips
        elif action == 1:  # raise
            expected_delta = max_chips - player_chips + raise_amount
        else:
            return
        actual_delta = float(post.chips_in[player]) - float(pre.chips_in[player])
        if not np.isclose(actual_delta, expected_delta, atol=1e-4):
            self._fail(
                step,
                "chips/bet_amount",
                f"action={action} (stage={int(pre.stage)}) added {actual_delta}, expected {expected_delta}",
            )
        if action == 1 and int(post.raise_nums[pre.stage]) != int(pre.raise_nums[pre.stage]) + 1:
            self._fail(step, "chips/raise_count_increment", "raise_nums[stage] must increment by exactly 1 on a raise")

    def _check_fold_and_check_no_chip_change(self, step, pre: SevenCardStudState, post: SevenCardStudState, player: int, action: int) -> None:
        if action in (2, 3) and not np.isclose(float(post.chips_in[player]), float(pre.chips_in[player])):
            self._fail(step, "chips/fold_check_no_change", f"action={action} must not move chips")
        if action == 2 and not bool(post.folded[player]):
            self._fail(step, "fold/marks_folded", f"player {player} folded but folded flag not set")

    # -- first-to-act-by-upcards ------------------------------------------------------

    def _check_first_to_act_by_upcards(self, step, pre: SevenCardStudState, post: SevenCardStudState) -> None:
        """On every subsequent street (4th-7th), the player whose exposed
        up cards make the best poker hand acts first."""
        if int(post.stage) == int(pre.stage) + 1 and int(post.raise_nums[int(post.stage)] if int(post.stage) < 5 else 0) == 0:
            # A street just closed into a NEW stage (round_over transition).
            new_stage = int(post.stage)
            if new_stage >= 1 and new_stage <= 4:
                expected_first = int(self.env._get_first_to_act_by_upcards(post.agent_cards, post.folded, jnp.int32(new_stage)))
                if bool(post.folded[expected_first]):
                    return  # env falls through to next non-folded player; not independently re-derived here
                if int(post.current_player_idx) != expected_first:
                    self._fail(
                        step,
                        "turn_order/first_to_act_by_upcards",
                        f"new stage {new_stage}: current_player_idx={int(post.current_player_idx)} != "
                        f"best-upcards player {expected_first}",
                    )

    # -- termination / showdown -----------------------------------------------------

    def _check_reward_only_on_terminal_step(self, step, done: bool, reward) -> None:
        if not done and not np.allclose(np.asarray(reward), 0.0):
            self._fail(step, "reward/only_on_terminal_step", f"nonzero reward {np.asarray(reward)} on non-terminal step")

    def _check_zero_sum_reward(self, step, reward) -> None:
        total = float(np.asarray(reward).sum())
        if not np.isclose(total, 0.0, atol=1e-3):
            self._fail(step, "reward/zero_sum", f"terminal rewards {np.asarray(reward)} sum to {total}, not 0")

    def _check_game_done_condition(self, step, post: SevenCardStudState, done: bool) -> None:
        """The hand ends when only one player remains unfolded, or stage
        reaches 5 (after 7th-street betting completes)."""
        num_active = int((~np.asarray(post.folded)).sum())
        expected = (num_active <= 1) or (int(post.stage) >= 5)
        if bool(done) != expected:
            self._fail(step, "win/termination_condition", f"done={bool(done)} != expected {expected} (num_active={num_active}, stage={int(post.stage)})")

    def _check_showdown_awards_winner(self, step, pre: SevenCardStudState, post: SevenCardStudState, reward) -> None:
        """Independently recomputes the showdown winner via
        `_compare_hands` (best-5-of-7) and confirms the pot was awarded
        accordingly."""
        folded = np.asarray(post.folded)
        num_active = int((~folded).sum())
        pot_total = float(np.asarray(post.chips_in).sum())
        if num_active > 1:
            winners_mask = np.asarray(_compare_hands(post.agent_cards, jnp.array(folded)))
            winners = [p for p in range(self.env.num_agents) if winners_mask[p]]
        else:
            winners = [p for p in range(self.env.num_agents) if not folded[p]]
        expected_reward = np.array([-float(post.chips_in[p]) for p in range(self.env.num_agents)])
        for w in winners:
            expected_reward[w] += (pot_total / len(winners))
        expected_reward = expected_reward / self.env.big_bet
        if not np.allclose(np.asarray(reward), expected_reward, atol=1e-3):
            self._fail(
                step,
                "showdown/pot_awarded_correctly",
                f"winners={winners}, expected reward {expected_reward}, got {np.asarray(reward)}",
            )

    # -- action legality / no-deadlock --------------------------------------------

    def _check_action_legal(self, step, avail, action: int) -> None:
        if not bool(avail[action]):
            self._fail(step, "action/within_avail_mask", f"action {action} taken despite avail mask")

    def _check_avail_nonempty(self, step, avail) -> None:
        if not bool(jnp.asarray(avail).any()):
            self._fail(step, "action/no_deadlock", "no legal actions available for acting player")

    # -- top-level dispatch ------------------------------------------------------

    def validate_transition(self, step, pre: SevenCardStudState, action: int, avail, post: SevenCardStudState, reward, done: bool) -> None:
        self._check_avail_nonempty(step, avail)
        self._check_action_legal(step, avail, action)
        self._check_betting_avail(step, pre, avail)

        player = int(pre.current_player_idx)
        self._check_bet_amount(step, pre, post, player, action)
        self._check_fold_and_check_no_chip_change(step, pre, post, player, action)
        self._check_raise_cap_never_exceeded(step, post)
        self._check_first_to_act_by_upcards(step, pre, post)
        self._check_game_done_condition(step, post, done)
        self._check_reward_only_on_terminal_step(step, done, reward)
        if done:
            self._check_zero_sum_reward(step, reward)
            self._check_showdown_awards_winner(step, pre, post, reward)


def rollout_and_validate(env: SevenCardStud, kind, network, params, num_episodes: int, seed: int = DEFAULT_SEED):
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
    env = make("seven_card_stud", num_agents=2, horizon=100_000, init_chips=100)
    checker = rollout_and_validate(env, "random", None, None, RANDOM_EPISODES, seed=DEFAULT_SEED)
    checker.assert_no_violations()


@pytest.mark.parametrize(
    "checkpoint_info", one_checkpoint_per_algorithm_and_kind(discover_checkpoints(CHECKPOINT_ROOT), CHECKPOINT_ROOT)
)
def test_checkpoint_agent_rollouts_conform_to_rules(checkpoint_info) -> None:
    checkpoint_path, network_kind, algorithm = checkpoint_info
    env_kwargs, fc_dim_size = training_env_kwargs(EXAMPLES_DIR, algorithm)
    env = make("seven_card_stud", **env_kwargs)

    _, sample_obs = env.reset(jax.random.PRNGKey(0))
    network = build_network(network_kind, action_dim=4, hidden_dim=fc_dim_size)
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
    env: SevenCardStud, agent_cards, chips_in, bring_in_idx=0, current_player_idx=None,
    stage=0, raise_nums=None, folded=None,
) -> SevenCardStudState:
    n = env.num_agents
    if folded is None:
        folded = [False] * n
    if raise_nums is None:
        raise_nums = [0] * env.num_betting_rounds
    if current_player_idx is None:
        current_player_idx = (bring_in_idx + 1) % n
    return SevenCardStudState(
        agent_cards=jnp.array(agent_cards, dtype=jnp.int32),
        chips_in=jnp.array(chips_in, dtype=jnp.float32),
        bring_in_idx=jnp.int32(bring_in_idx),
        current_player_idx=jnp.int32(current_player_idx),
        stage=jnp.int32(stage),
        raise_nums=jnp.array(raise_nums, dtype=jnp.int32),
        folded=jnp.array(folded, dtype=bool),
        not_raise_num=jnp.int32(0),
        absorbing=jnp.zeros(n, dtype=bool),
        done=False,
        timestep=0,
    )


def test_edge_bring_in_tiebreak_by_suit() -> None:
    """Ties among lowest door cards are broken by suit rank, low to high:
    clubs, diamonds, hearts, spades. Two players both showing a rank-2 door
    card (the lowest possible), one in clubs (suit 0) and one in spades
    (suit 3) -- clubs must win the tie and post the bring-in."""
    # card = suit*13 + (rank-2), so rank 2 is card%13==1 for any suit.
    clubs_2 = 1       # suit 0 (clubs), mod=1 -> rank 2
    spades_2 = 3 * 13 + 1  # suit 3 (spades), rank 2
    door_cards = jnp.array([clubs_2, spades_2])
    bring_in_idx = int(_get_bring_in_idx(door_cards))
    assert bring_in_idx == 0, "clubs (suit 0) must win a rank tie over spades (suit 3) for the bring-in"


def test_edge_raise_cap_blocks_fifth_bet() -> None:
    """Betting is capped at one bet plus a fixed number of raises per round
    (`allowed_raise_num=4`). After 4 raises in a street, raising must
    become illegal."""
    env = make("seven_card_stud", num_agents=2, horizon=100_000, init_chips=100)
    state = _make_state(
        env, agent_cards=[[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]],
        chips_in=[5.0, 5.0], stage=0, raise_nums=[4, 0, 0, 0, 0],
    )
    avail = env.get_avail_actions(state)
    assert not bool(avail[1]), "raise must be illegal once raise_nums[stage] reaches the cap of 4"


def test_edge_big_bet_kicks_in_at_stage_2() -> None:
    """Bets are the small bet on 3rd and 4th street, and the big bet
    (double the small bet) from 5th street onward."""
    env = make("seven_card_stud", num_agents=2, horizon=100_000, init_chips=100)
    rng = jax.random.PRNGKey(0)
    small_street = _make_state(
        env, agent_cards=[[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]], chips_in=[1.0, 2.0],
        stage=1, current_player_idx=0,
    )
    next_small, *_ = env.step_env(rng, small_street, jnp.int32(1))  # player 0 raises
    assert np.isclose(float(next_small.chips_in[0]) - 1.0, (2.0 - 1.0) + env.small_bet)

    big_street = _make_state(
        env, agent_cards=[[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]], chips_in=[1.0, 2.0],
        stage=2, current_player_idx=0,
    )
    next_big, *_ = env.step_env(rng, big_street, jnp.int32(1))  # player 0 raises
    assert np.isclose(float(next_big.chips_in[0]) - 1.0, (2.0 - 1.0) + env.big_bet), (
        "a raise on stage>=2 (5th street+) must use the big bet, not the small bet"
    )


def test_edge_fold_gives_pot_to_sole_remaining_player_without_showdown() -> None:
    """Showdown only occurs among remaining players; a fold that leaves
    exactly one player standing awards the pot without hand evaluation."""
    env = make("seven_card_stud", num_agents=2, horizon=100_000, init_chips=100)
    rng = jax.random.PRNGKey(0)
    state = _make_state(
        env, agent_cards=[[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]],
        chips_in=[3.0, 2.0], stage=1, current_player_idx=1,
    )
    next_state, obs, reward, absorbing, done, info = env.step_env(rng, state, jnp.int32(2))  # player 1 folds
    assert bool(done)
    assert np.isclose(float(reward[0]), 2.0 / env.big_bet)
    assert np.isclose(float(reward[1]), -2.0 / env.big_bet)


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

    env = make("seven_card_stud", num_agents=2, horizon=100_000, init_chips=100)

    if args.random:
        kind, network, params = "random", None, None
    else:
        kind = args.network_type or infer_network_kind(args.checkpoint)
        network = build_network(kind, action_dim=4, hidden_dim=args.hidden_dim)
        _, sample_obs = env.reset(jax.random.PRNGKey(0))
        params = load_checkpoint_params(network, sample_obs, args.checkpoint)

    print(f"Validating {args.episodes} episode(s) of '{kind}' agent (seven_card_stud)...")
    checker = rollout_and_validate(env, kind, network, params, args.episodes, seed=args.seed)

    if checker.violations:
        print(f"\n{len(checker.violations)} VIOLATION(S) FOUND:")
        for v in checker.violations[:20]:
            print(f"  {v}")
        raise SystemExit(1)
    print("\nNo rule violations found.")


if __name__ == "__main__":
    main()
