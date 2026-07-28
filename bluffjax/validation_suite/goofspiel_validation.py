"""
Rule-conformance validation suite for Goofspiel (GOPS).

Checks every state/action/transition produced by random or checkpoint-driven
rollouts against bluffjax/environments/goofspiel/rules.md and the
implementation in bluffjax/environments/goofspiel/goofspiel.py.

Goofspiel is a ParallelEnv: both agents bid simultaneously each round
(`action` is a (num_agents,) vector), so a rollout step advances both
players' bids together and resolves the round in one transition.

Usage:
    pytest bluffjax/validation_suite/goofspiel_validation.py -v
    python goofspiel_validation.py --random --episodes 500
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.environments.goofspiel.goofspiel import Goofspiel, GoofspielState
from bluffjax.validation_suite.common import (
    RuleCheckerBase,
    build_network,
    discover_checkpoints,
    infer_network_kind,
    load_checkpoint_params,
    one_checkpoint_per_algorithm_and_kind,
    rollout_and_validate_parallel,
    training_env_kwargs,
)

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.normpath(os.path.join(HERE, "..", "examples", "goofspiel"))
CHECKPOINT_ROOT = os.path.join(EXAMPLES_DIR, "checkpoints")

DEFAULT_SEED = 0
RANDOM_EPISODES = 300
CHECKPOINT_EPISODES = 60


class RuleChecker(RuleCheckerBase):
    def __init__(self, env: Goofspiel):
        super().__init__(env)

    # -- setup / reset --------------------------------------------------------

    def check_reset(self, state: GoofspielState) -> None:
        """Both players' hands start with the full rank-1..13 deck, and
        the prize/reveal order (`deck`) is a random permutation freshly
        drawn each episode."""
        hands = np.asarray(state.player_hands)
        if not hands.all():
            self._fail(0, "setup/full_hands", "both players must start with every card in hand")
        deck = np.asarray(state.deck)
        if not np.array_equal(np.sort(deck), np.arange(self.env.deck_size)):
            self._fail(0, "setup/deck_permutation", f"deck={deck} is not a permutation of 0..{self.env.deck_size - 1}")
        if int(state.current_round) != 0:
            self._fail(0, "setup/initial_round", "current_round must start at 0")
        if not np.allclose(np.asarray(state.points), 0.0):
            self._fail(0, "setup/initial_points", "points must start at 0")
        if np.asarray(state.winners).any():
            self._fail(0, "setup/initial_winners", "winners must start all-False")

    # -- per-round mechanics --------------------------------------------------------

    def _check_card_removed_after_bid(self, step, pre: GoofspielState, post: GoofspielState, actions) -> None:
        """Each bid card can only be used once: once played, it is removed
        from that player's hand for the rest of the game, regardless of
        whether that agent won, lost, or tied the round."""
        for agent in range(self.env.num_agents):
            card = int(actions[agent])
            if not bool(pre.player_hands[agent, card]):
                self._fail(
                    step,
                    "action/bid_owned_card",
                    f"agent {agent} bid card {card} which it did not hold",
                )
            if bool(post.player_hands[agent, card]):
                self._fail(
                    step,
                    "invariant/card_removed_after_bid",
                    f"agent {agent}'s bid card {card} was not removed from hand after bidding",
                )
        hands_before = np.asarray(pre.player_hands).sum(axis=1)
        hands_after = np.asarray(post.player_hands).sum(axis=1)
        if not np.array_equal(hands_after, hands_before - 1):
            self._fail(
                step,
                "invariant/hand_shrinks_by_one",
                f"hand sizes {hands_before} -> {hands_after}, expected exactly -1 per agent",
            )

    def _check_tie_rule_and_reward(self, step, pre: GoofspielState, actions, reward) -> None:
        """The player with the strictly highest bid card wins the prize
        card; if two or more players tie for the highest bid, the prize
        is discarded instead. Recomputed here from the raw action vector
        rather than trusting step_env's own reward."""
        prize_value = int(pre.deck[pre.current_round]) + 1
        max_bid = int(np.max(actions))
        num_max_bidders = int(np.sum(actions == max_bid))
        reward = np.asarray(reward)
        if num_max_bidders > 1:
            if not np.allclose(reward, 0.0):
                self._fail(
                    step,
                    "reward/tie_discards_prize",
                    f"tied bid {max_bid} (by {num_max_bidders} agents) should discard the prize "
                    f"(reward all 0), got {reward}",
                )
        else:
            winner = int(np.argmax(actions == max_bid))
            expected = np.zeros(self.env.num_agents)
            expected[winner] = float(prize_value)
            if not np.allclose(reward, expected):
                self._fail(
                    step,
                    "reward/unique_winner_gets_prize",
                    f"expected reward {expected} (winner={winner}, prize={prize_value}), got {reward}",
                )
        if not (1 <= prize_value <= self.env.deck_size):
            self._fail(step, "invariant/prize_value_range", f"prize_value={prize_value} outside [1,{self.env.deck_size}]")

    def _check_points_accumulate(self, step, pre: GoofspielState, post: GoofspielState, reward) -> None:
        expected = np.asarray(pre.points) + np.asarray(reward)
        if not np.allclose(np.asarray(post.points), expected, atol=1e-4):
            self._fail(
                step,
                "invariant/points_accumulate",
                f"points {np.asarray(post.points)} != pre.points + reward = {expected}",
            )

    # -- termination -----------------------------------------------------------------

    def _check_round_advance_and_termination(self, step, pre: GoofspielState, post: GoofspielState, done: bool) -> None:
        """The game lasts 13 rounds; termination is purely round-count-based
        (`new_round >= num_cards`), independent of the env's `horizon`
        attribute."""
        if int(post.current_round) != int(pre.current_round) + 1:
            self._fail(step, "invariant/round_advances", "current_round must advance by exactly 1 each step")
        expected_done = int(post.current_round) >= self.env.num_cards
        if bool(done) != expected_done:
            self._fail(
                step,
                "win/termination_at_round_count",
                f"done={bool(done)} != expected {expected_done} at current_round={int(post.current_round)}",
            )
        absorbing_vals = set(np.asarray(post.absorbing).tolist())
        if absorbing_vals not in ({True}, {False}):
            self._fail(step, "invariant/absorbing_all_or_nothing", f"absorbing={np.asarray(post.absorbing)} not uniform")

    def _check_terminal_winners_flag(self, step, pre: GoofspielState, post: GoofspielState, done: bool) -> None:
        """Ties for the overall game are possible if final point totals are
        equal. `winners` should only update on the terminal step (multiple
        True entries allowed on a game-level tie), and stay unchanged from
        the previous state otherwise."""
        if done:
            expected = np.asarray(post.points) == np.max(np.asarray(post.points))
            if not np.array_equal(np.asarray(post.winners), expected):
                self._fail(
                    step,
                    "win/winners_flag_at_terminal",
                    f"winners={np.asarray(post.winners)} != (points == max(points))={expected}",
                )
        else:
            if not np.array_equal(np.asarray(post.winners), np.asarray(pre.winners)):
                self._fail(step, "win/winners_unchanged_mid_game", "winners flag changed on a non-terminal step")

    # -- action legality --------------------------------------------------------

    def _check_action_legal(self, step, avail, actions) -> None:
        for agent in range(self.env.num_agents):
            if not bool(avail[agent, actions[agent]]):
                self._fail(
                    step,
                    "action/within_avail_mask",
                    f"agent {agent} bid {actions[agent]} despite avail={np.asarray(avail[agent])}",
                )

    def _check_avail_nonempty(self, step, avail, done_before: bool) -> None:
        if done_before:
            return
        for agent in range(self.env.num_agents):
            if not bool(jnp.asarray(avail[agent]).any()):
                self._fail(step, "action/no_deadlock", f"agent {agent} has no legal bid available")

    # -- top-level dispatch ------------------------------------------------------

    def validate_transition(self, step, pre: GoofspielState, actions, avail, post: GoofspielState, reward, done: bool) -> None:
        self._check_avail_nonempty(step, avail, bool(pre.done))
        self._check_action_legal(step, avail, actions)
        self._check_card_removed_after_bid(step, pre, post, actions)
        self._check_tie_rule_and_reward(step, pre, actions, reward)
        self._check_points_accumulate(step, pre, post, reward)
        self._check_round_advance_and_termination(step, pre, post, done)
        self._check_terminal_winners_flag(step, pre, post, done)


def rollout_and_validate(env: Goofspiel, kind, network, params, num_episodes: int, seed: int = DEFAULT_SEED):
    checker = RuleChecker(env)
    return rollout_and_validate_parallel(env, kind, network, params, num_episodes, checker, seed=seed)


# ---------------------------------------------------------------------------
# pytest entry points.
# ---------------------------------------------------------------------------


def test_random_agent_rollouts_conform_to_rules() -> None:
    env = make("goofspiel", num_agents=2, horizon=14, num_decks=1)
    checker = rollout_and_validate(env, "random", None, None, RANDOM_EPISODES, seed=DEFAULT_SEED)
    checker.assert_no_violations()


@pytest.mark.parametrize(
    "checkpoint_info", one_checkpoint_per_algorithm_and_kind(discover_checkpoints(CHECKPOINT_ROOT), CHECKPOINT_ROOT)
)
def test_checkpoint_agent_rollouts_conform_to_rules(checkpoint_info) -> None:
    checkpoint_path, network_kind, algorithm = checkpoint_info
    env_kwargs, fc_dim_size = training_env_kwargs(EXAMPLES_DIR, algorithm)
    env = make("goofspiel", **env_kwargs)

    _, sample_obs = env.reset(jax.random.PRNGKey(0))
    network = build_network(network_kind, action_dim=env.num_actions, hidden_dim=fc_dim_size)
    params = load_checkpoint_params(network, sample_obs[0], checkpoint_path)

    checker = rollout_and_validate(env, network_kind, network, params, CHECKPOINT_EPISODES, seed=DEFAULT_SEED)
    checker.assert_no_violations(extra_context=f" for checkpoint={checkpoint_path}")


def test_checkpoint_discovery_runs_without_error() -> None:
    for path, kind in discover_checkpoints(CHECKPOINT_ROOT):
        assert kind in ("actor", "actor_critic", "q_network"), (path, kind)


# ---------------------------------------------------------------------------
# Hand-crafted edge-case scenarios.
# ---------------------------------------------------------------------------


def _make_state(deck, current_round, player_hands=None, points=(0.0, 0.0)) -> GoofspielState:
    if player_hands is None:
        player_hands = jnp.ones((2, 13), dtype=bool)
    return GoofspielState(
        player_hands=jnp.array(player_hands, dtype=bool),
        deck=jnp.array(deck, dtype=jnp.int32),
        current_round=jnp.int32(current_round),
        points=jnp.array(points, dtype=jnp.float32),
        winners=jnp.zeros(2, dtype=bool),
        absorbing=jnp.zeros(2, dtype=bool),
        done=False,
        timestep=0,
    )


def test_edge_tied_bid_discards_prize() -> None:
    """If two or more players tie for the highest bid, the prize card is
    discarded and no one gets it. Both players bid rank-7 (0-indexed card
    7); neither should score, but both must still lose that card."""
    env = make("goofspiel", num_agents=2, horizon=14, num_decks=1)
    rng = jax.random.PRNGKey(0)
    deck = list(range(13))  # prize this round (round 0) = deck[0]+1 = 1
    state = _make_state(deck=deck, current_round=0)

    next_state, obs, reward, absorbing, done, info = env.step_env(rng, state, jnp.array([7, 7]))
    assert np.allclose(np.asarray(reward), 0.0), "a tied bid must award the prize to no one"
    assert not bool(next_state.player_hands[0, 7]) and not bool(next_state.player_hands[1, 7]), (
        "both players' card 7 must be discarded from hand despite the tie"
    )
    assert np.allclose(np.asarray(next_state.points), 0.0)


def test_edge_unique_highest_bid_wins_exact_prize_value() -> None:
    """The player with the strictly highest bid card wins the prize card
    and scores points equal to its rank."""
    env = make("goofspiel", num_agents=2, horizon=14, num_decks=1)
    rng = jax.random.PRNGKey(0)
    deck = [11] + list(range(12))  # prize this round = deck[0]+1 = 12
    state = _make_state(deck=deck, current_round=0)

    next_state, obs, reward, absorbing, done, info = env.step_env(rng, state, jnp.array([9, 3]))
    assert np.allclose(np.asarray(reward), [12.0, 0.0]), (
        f"agent 0 bid higher (9 > 3) and should win prize value 12; got reward={np.asarray(reward)}"
    )
    assert np.allclose(np.asarray(next_state.points), [12.0, 0.0])


def test_edge_game_ends_exactly_after_13_rounds() -> None:
    """The game lasts 13 rounds: done must be False through round 12
    (0-indexed) and True only once current_round reaches 13."""
    env = make("goofspiel", num_agents=2, horizon=14, num_decks=1)
    rng = jax.random.PRNGKey(0)
    deck = list(range(13))

    second_to_last = _make_state(deck=deck, current_round=11)
    s, obs, reward, absorbing, done, info = env.step_env(rng, second_to_last, jnp.array([0, 1]))
    assert int(s.current_round) == 12
    assert not bool(done), "game must not be done with only 12 of 13 rounds played"

    last_round = _make_state(deck=deck, current_round=12)
    s2, obs2, reward2, absorbing2, done2, info2 = env.step_env(rng, last_round, jnp.array([2, 3]))
    assert int(s2.current_round) == 13
    assert bool(done2), "game must be done immediately after the 13th round"


def test_edge_terminal_game_level_tie_marks_both_winners() -> None:
    """Ties for the overall game are possible if final point totals are
    equal. If both players end with equal cumulative points, `winners`
    must flag both as True, not pick an arbitrary single winner."""
    env = make("goofspiel", num_agents=2, horizon=14, num_decks=1)
    rng = jax.random.PRNGKey(0)
    deck = list(range(13))
    # Equal points entering the final round; a tied final bid keeps them equal.
    state = _make_state(deck=deck, current_round=12, points=(5.0, 5.0))

    final_state, obs, reward, absorbing, done, info = env.step_env(rng, state, jnp.array([4, 4]))
    assert bool(done)
    assert np.allclose(np.asarray(final_state.points), [5.0, 5.0])
    assert list(map(bool, final_state.winners)) == [True, True], (
        "a final-points tie must flag both agents as winners"
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
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    if not args.random and args.checkpoint is None:
        parser.error("pass --checkpoint PATH or --random")

    env = make("goofspiel", num_agents=2, horizon=14, num_decks=1)

    if args.random:
        kind, network, params = "random", None, None
    else:
        kind = args.network_type or infer_network_kind(args.checkpoint)
        network = build_network(kind, action_dim=env.num_actions, hidden_dim=args.hidden_dim)
        _, sample_obs = env.reset(jax.random.PRNGKey(0))
        params = load_checkpoint_params(network, sample_obs[0], args.checkpoint)

    print(f"Validating {args.episodes} episode(s) of '{kind}' agent (goofspiel)...")
    checker = rollout_and_validate(env, kind, network, params, args.episodes, seed=args.seed)

    if checker.violations:
        print(f"\n{len(checker.violations)} VIOLATION(S) FOUND:")
        for v in checker.violations[:20]:
            print(f"  {v}")
        raise SystemExit(1)
    print("\nNo rule violations found.")


if __name__ == "__main__":
    main()
