"""
Rule-conformance validation suite for Five Card Draw poker.

Checks every state/action/transition produced by random or checkpoint-driven
rollouts against bluffjax/environments/five_card_draw/rules.md and the
implementation in bluffjax/environments/five_card_draw/five_card_draw.py.

Usage:
    pytest bluffjax/validation_suite/five_card_draw_validation.py -v
    python five_card_draw_validation.py --random --episodes 300
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.environments.five_card_draw.five_card_draw import FiveCardDraw, FiveCardDrawState
from bluffjax.utils.game_utils.poker_utils import _card_rank, _score_five_card_hand
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
EXAMPLES_DIR = os.path.normpath(os.path.join(HERE, "..", "examples", "five_card_draw"))
CHECKPOINT_ROOT = os.path.join(EXAMPLES_DIR, "checkpoints")

DEFAULT_SEED = 0
RANDOM_EPISODES = 150
CHECKPOINT_EPISODES = 40


class RuleChecker(RuleCheckerBase):
    def __init__(self, env: FiveCardDraw):
        super().__init__(env)

    # -- setup / reset --------------------------------------------------------

    def check_reset(self, state: FiveCardDrawState) -> None:
        """The small blind is chosen uniformly at random; with num_agents=2,
        current_player_idx reduces to the small blind seat, which acts
        first pre-draw."""
        hands = np.asarray(state.agent_cards)
        all_cards = hands.reshape(-1)
        if len(set(all_cards.tolist())) != all_cards.size:
            self._fail(0, "setup/distinct_cards", "dealt hands contain a duplicate card")
        sb = int(state.small_blind_idx)
        chips_in = np.asarray(state.chips_in)
        bb = (sb + 1) % self.env.num_agents
        if not np.isclose(chips_in[sb], self.env.small_blind):
            self._fail(0, "setup/small_blind_posted", f"small blind seat {sb} chips_in={chips_in[sb]}")
        if not np.isclose(chips_in[bb], self.env.big_blind):
            self._fail(0, "setup/big_blind_posted", f"big blind seat {bb} chips_in={chips_in[bb]}")
        if self.env.num_agents == 2 and int(state.current_player_idx) != sb:
            self._fail(0, "setup/heads_up_sb_acts_first", "with 2 players, small blind must act first pre-draw")
        if int(state.stage) != 0:
            self._fail(0, "setup/initial_stage", "hand must start in stage 0 (pre-draw betting)")

    # -- betting action legality/arithmetic -------------------------------------------

    def _check_betting_avail(self, step, pre: FiveCardDrawState, avail) -> None:
        """Check/call and fold are always legal in a betting stage;
        raise-half/raise-pot/all-in are gated on stack size and, for
        half/pot, on genuinely exceeding the call."""
        cur = int(pre.current_player_idx)
        player_round = float(pre.round_raised[cur])
        max_round = float(np.max(pre.round_raised))
        player_remain = float(pre.remaining_chips[cur])
        pot = float(np.sum(pre.chips_in))
        half_pot = float(np.floor(pot / 2.0))
        diff = max_round - player_round

        can_raise = player_remain > diff
        can_raise_pot = can_raise and (pot <= player_remain)
        can_raise_half = can_raise and (half_pot <= player_remain) and ((half_pot + player_round) > max_round)

        expectations = {0: True, 1: can_raise_half, 2: can_raise_pot, 3: can_raise, 4: True}
        for action, expected in expectations.items():
            if bool(avail[action]) != expected:
                self._fail(
                    step,
                    f"action/betting_avail_{action}",
                    f"avail[{action}]={bool(avail[action])} != expected {expected}",
                )

    def _check_bet_amount(self, step, pre: FiveCardDrawState, post: FiveCardDrawState, player: int, action: int) -> None:
        """Check/call adds exactly `diff`; raise-half adds `floor(pot/2)`;
        raise-pot adds `pot`; all-in adds the player's entire remaining
        stack. Each amount is the player's whole new contribution, not
        stacked on top of a call."""
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
            self._fail(
                step,
                "chips/bet_amount",
                f"action={action} added {actual_bet} chips, expected {expected_bet}",
            )
        if action in (1, 2, 3):
            # A raise/all-in sets not_raise_num to 1, unless that same action
            # also closes the betting round (e.g. raising against an
            # opponent already all-in, leaving only 1 playable player), in
            # which case the round-over reset zeroes it back out immediately.
            num_playable_post = int((~np.asarray(post.folded) & ~np.asarray(post.all_in)).sum())
            expected_nnr = 0 if 1 >= num_playable_post else 1
            if int(post.not_raise_num) != expected_nnr:
                self._fail(
                    step,
                    "chips/raise_resets_not_raise_num",
                    f"after a raise/all-in, not_raise_num={int(post.not_raise_num)} != expected {expected_nnr} "
                    f"(num_playable_post={num_playable_post})",
                )

    def _check_fold_effect(self, step, pre: FiveCardDrawState, post: FiveCardDrawState, player: int) -> None:
        if not bool(post.folded[player]):
            self._fail(step, "fold/marks_folded", f"player {player} took fold action but folded flag not set")
        if not np.allclose(np.asarray(post.chips_in), np.asarray(pre.chips_in)):
            self._fail(step, "fold/no_chip_change", "folding must not move chips")

    def _check_all_in_flag(self, step, post: FiveCardDrawState, player: int) -> None:
        remaining = float(post.remaining_chips[player])
        should_be_all_in = remaining <= 0
        if bool(post.all_in[player]) != should_be_all_in:
            self._fail(
                step,
                "chips/all_in_flag",
                f"player {player} remaining_chips={remaining} but all_in={bool(post.all_in[player])}",
            )

    # -- draw phase ---------------------------------------------------------------

    def _check_draw_pattern(self, step, pre: FiveCardDrawState, post: FiveCardDrawState, player: int, action: int) -> None:
        """`pattern = action - 5`; bit j (LSB) of the pattern is 1=keep/0=discard
        for the player's *sorted* hand at position j. Action 5 discards all 5
        cards; action 36 stands pat (keeps all). Discarded cards are replaced
        sequentially from the deck."""
        sorted_hand = np.sort(np.asarray(pre.agent_cards[player]))
        pattern = action - 5
        keep_bits = [(pattern >> j) & 1 for j in range(5)]
        num_discard = 5 - sum(keep_bits)
        kept = sorted(c for c, k in zip(sorted_hand, keep_bits) if k)
        expected_new_cards = np.asarray(pre.shuffled_deck)[int(pre.deck_idx) : int(pre.deck_idx) + num_discard]
        expected_hand = sorted(list(kept) + expected_new_cards.tolist())
        actual_hand = sorted(np.asarray(post.agent_cards[player]).tolist())
        if actual_hand != expected_hand:
            self._fail(
                step,
                "draw/pattern_applied",
                f"action={action} (pattern={pattern:05b}, num_discard={num_discard}): "
                f"expected hand {expected_hand}, got {actual_hand}",
            )
        if int(post.deck_idx) != int(pre.deck_idx) + num_discard:
            self._fail(step, "draw/deck_idx_advances", f"deck_idx should advance by num_discard={num_discard}")

    def _check_draw_advances_or_moves_to_stage2(self, step, pre: FiveCardDrawState, post: FiveCardDrawState) -> None:
        """Players draw in turn starting from the first non-folded player
        after the small blind; once every non-folded player has drawn, the
        env auto-advances to stage 2."""
        if int(pre.stage) == 1 and int(post.stage) == 1:
            if int(post.current_player_idx) == int(pre.current_player_idx):
                self._fail(step, "draw/advances_to_next_player", "draw phase did not advance to the next player")

    # -- termination / showdown -----------------------------------------------------

    def _check_reward_only_on_terminal_step(self, step, done: bool, reward) -> None:
        if not done and not np.allclose(np.asarray(reward), 0.0):
            self._fail(step, "reward/only_on_terminal_step", f"nonzero reward {np.asarray(reward)} on non-terminal step")

    def _check_zero_sum_reward(self, step, reward) -> None:
        """Reward per player equals chips received at showdown minus their
        total pot contribution, so rewards across players must sum to zero."""
        total = float(np.asarray(reward).sum())
        if not np.isclose(total, 0.0, atol=1e-3):
            self._fail(step, "reward/zero_sum", f"terminal rewards {np.asarray(reward)} sum to {total}, not 0")

    def _check_showdown_awards_winner(self, step, pre: FiveCardDrawState, post: FiveCardDrawState, reward) -> None:
        """Independently recomputes the showdown winner via
        `_score_five_card_hand` and confirms the pot was awarded accordingly.
        This checks that the env correctly plumbs the score into chip
        payouts, not the correctness of the ranking formula itself."""
        folded = np.asarray(post.folded)
        num_active = int((~folded).sum())
        if num_active > 1:
            scores = []
            for p in range(self.env.num_agents):
                if folded[p]:
                    scores.append(-1)
                else:
                    ranks = np.asarray(_card_rank(post.agent_cards[p]))
                    suits = np.asarray(post.agent_cards[p]) // 13
                    scores.append(int(_score_five_card_hand(jnp.array(ranks), jnp.array(suits))))
            max_score = max(scores)
            winners = [p for p, s in enumerate(scores) if s == max_score and not folded[p]]
        else:
            winners = [p for p in range(self.env.num_agents) if not folded[p]]

        pot_total = float(np.asarray(post.chips_in).sum())
        expected_reward = np.full(self.env.num_agents, 0.0)
        for p in range(self.env.num_agents):
            expected_reward[p] = -float(post.chips_in[p])
        for w in winners:
            expected_reward[w] += pot_total / len(winners)

        if not np.allclose(np.asarray(reward), expected_reward, atol=1e-3):
            self._fail(
                step,
                "showdown/pot_awarded_correctly",
                f"winners={winners} (scores={scores if num_active > 1 else 'n/a'}), "
                f"expected reward {expected_reward}, got {np.asarray(reward)}",
            )

    # -- action legality / no-deadlock --------------------------------------------

    def _check_action_legal(self, step, avail, action: int) -> None:
        if not bool(avail[action]):
            self._fail(step, "action/within_avail_mask", f"action {action} taken despite avail mask")

    def _check_avail_nonempty(self, step, avail) -> None:
        if not bool(jnp.asarray(avail).any()):
            self._fail(step, "action/no_deadlock", "no legal actions available for acting player")

    # -- top-level dispatch ------------------------------------------------------

    def validate_transition(self, step, pre: FiveCardDrawState, action: int, avail, post: FiveCardDrawState, reward, done: bool) -> None:
        self._check_avail_nonempty(step, avail)
        self._check_action_legal(step, avail, action)

        player = int(pre.current_player_idx)
        stage = int(pre.stage)
        if stage in (0, 2):
            self._check_betting_avail(step, pre, avail)
            if action == 4:
                self._check_fold_effect(step, pre, post, player)
            else:
                self._check_bet_amount(step, pre, post, player, action)
                self._check_all_in_flag(step, post, player)
        elif stage == 1:
            self._check_draw_pattern(step, pre, post, player, action)
            self._check_draw_advances_or_moves_to_stage2(step, pre, post)

        self._check_reward_only_on_terminal_step(step, done, reward)
        if done:
            self._check_zero_sum_reward(step, reward)
            self._check_showdown_awards_winner(step, pre, post, reward)


def rollout_and_validate(env: FiveCardDraw, kind, network, params, num_episodes: int, seed: int = DEFAULT_SEED):
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
    env = make("five_card_draw", num_agents=2, horizon=100_000, init_chips=100)
    checker = rollout_and_validate(env, "random", None, None, RANDOM_EPISODES, seed=DEFAULT_SEED)
    checker.assert_no_violations()


@pytest.mark.parametrize(
    "checkpoint_info", one_checkpoint_per_algorithm_and_kind(discover_checkpoints(CHECKPOINT_ROOT), CHECKPOINT_ROOT)
)
def test_checkpoint_agent_rollouts_conform_to_rules(checkpoint_info) -> None:
    checkpoint_path, network_kind, algorithm = checkpoint_info
    env_kwargs, fc_dim_size = training_env_kwargs(EXAMPLES_DIR, algorithm)
    env = make("five_card_draw", **env_kwargs)

    _, sample_obs = env.reset(jax.random.PRNGKey(0))
    network = build_network(network_kind, action_dim=env.num_actions, hidden_dim=fc_dim_size)
    params = load_checkpoint_params(network, sample_obs, checkpoint_path)

    checker = rollout_and_validate(env, network_kind, network, params, CHECKPOINT_EPISODES, seed=DEFAULT_SEED)
    checker.assert_no_violations(extra_context=f" for checkpoint={checkpoint_path}")


def test_checkpoint_discovery_runs_without_error() -> None:
    for path, kind in discover_checkpoints(CHECKPOINT_ROOT):
        assert kind in ("actor", "actor_critic", "q_network"), (path, kind)


def test_score_five_card_hand_orders_categories_correctly() -> None:
    """Unit check on `_score_five_card_hand`, the poker hand-ranking utility
    used by every showdown in this environment: standard category ordering,
    high card < pair < two pair < trips < straight < flush < full house <
    quads < straight flush."""
    def card(rank: int, suit: int) -> int:
        """rank in 2..14 (2..10, 11=J, 12=Q, 13=K, 14=A); suit in 0..3.
        Inverts `_card_rank`'s `mod = card % 13; rank = 14 if mod==0 else mod+1`."""
        mod = 0 if rank == 14 else rank - 1
        return suit * 13 + mod

    def score(cards) -> int:
        cards = jnp.array(cards)
        ranks = _card_rank(cards)
        suits = cards // 13
        return int(_score_five_card_hand(ranks, suits))

    # Hands are built from (rank, suit) pairs rather than raw card integers,
    # since raw integers are deceptive here: [0, 13, 26, 39] looks like 4
    # unrelated cards but is secretly four Aces (mod=0 means Ace for every suit).
    high_card = score([card(14, 0), card(2, 1), card(4, 2), card(6, 3), card(9, 0)])
    pair = score([card(14, 0), card(14, 1), card(2, 0), card(4, 1), card(6, 2)])
    two_pair = score([card(14, 0), card(14, 1), card(2, 0), card(2, 1), card(6, 2)])
    trips = score([card(14, 0), card(14, 1), card(14, 2), card(2, 0), card(6, 1)])
    straight = score([card(14, 0), card(2, 1), card(3, 2), card(4, 3), card(5, 0)])  # wheel A2345, mixed suits
    flush = score([card(14, 0), card(3, 0), card(5, 0), card(7, 0), card(9, 0)])  # all clubs, non-sequential
    full_house = score([card(14, 0), card(14, 1), card(14, 2), card(2, 0), card(2, 1)])
    quads = score([card(14, 0), card(14, 1), card(14, 2), card(14, 3), card(2, 0)])
    straight_flush = score([card(14, 0), card(2, 0), card(3, 0), card(4, 0), card(5, 0)])  # wheel, all clubs

    assert high_card < pair < two_pair < trips < straight < flush < full_house < quads < straight_flush


# ---------------------------------------------------------------------------
# Hand-crafted edge-case scenarios.
# ---------------------------------------------------------------------------


def _make_state(
    env: FiveCardDraw,
    agent_cards,
    chips_in,
    round_raised,
    remaining_chips,
    small_blind_idx=0,
    current_player_idx=0,
    stage=0,
    folded=None,
    all_in=None,
    not_raise_num=0,
    shuffled_deck=None,
    deck_idx=None,
    draw_start_idx=0,
) -> FiveCardDrawState:
    n = env.num_agents
    if folded is None:
        folded = [False] * n
    if all_in is None:
        all_in = [False] * n
    used = set(np.asarray(agent_cards).reshape(-1).tolist())
    remaining_deck = [c for c in range(52) if c not in used]
    if shuffled_deck is None:
        shuffled_deck = jnp.array(list(agent_cards[0]) + list(agent_cards[1]) + remaining_deck, dtype=jnp.int32)[:52]
    if deck_idx is None:
        deck_idx = 5 * n
    return FiveCardDrawState(
        agent_cards=jnp.array(agent_cards, dtype=jnp.int32),
        chips_in=jnp.array(chips_in, dtype=jnp.float32),
        round_raised=jnp.array(round_raised, dtype=jnp.float32),
        remaining_chips=jnp.array(remaining_chips, dtype=jnp.float32),
        small_blind_idx=jnp.int32(small_blind_idx),
        current_player_idx=jnp.int32(current_player_idx),
        stage=jnp.int32(stage),
        folded=jnp.array(folded, dtype=bool),
        all_in=jnp.array(all_in, dtype=bool),
        not_raise_num=jnp.int32(not_raise_num),
        shuffled_deck=shuffled_deck,
        deck_idx=jnp.int32(deck_idx),
        draw_start_idx=jnp.int32(draw_start_idx),
        absorbing=jnp.zeros(n, dtype=bool),
        done=False,
        timestep=0,
    )


def test_edge_stand_pat_keeps_exact_hand() -> None:
    """Action 36 is pattern 11111 (discard none, stand pat); the hand must
    be byte-for-byte unchanged."""
    env = make("five_card_draw", num_agents=2, horizon=100_000, init_chips=100)
    rng = jax.random.PRNGKey(0)
    hand0 = [0, 5, 10, 15, 20]
    state = _make_state(
        env, agent_cards=[hand0, [1, 6, 11, 16, 21]], chips_in=[2.0, 2.0], round_raised=[0.0, 0.0],
        remaining_chips=[98.0, 98.0], stage=1, current_player_idx=0, draw_start_idx=0,
    )
    next_state, *_ = env.step_env(rng, state, jnp.int32(36))
    assert sorted(np.asarray(next_state.agent_cards[0]).tolist()) == sorted(hand0), "stand pat must leave the hand untouched"
    assert int(next_state.deck_idx) == int(state.deck_idx), "stand pat must not consume any deck cards"


def test_edge_discard_all_replaces_entire_hand() -> None:
    """Action 5 is pattern 00000 (discard all 5 cards); all 5 replacement
    cards come from the deck at deck_idx."""
    env = make("five_card_draw", num_agents=2, horizon=100_000, init_chips=100)
    rng = jax.random.PRNGKey(0)
    hand0 = [0, 5, 10, 15, 20]
    deck = jnp.array(hand0 + [1, 6, 11, 16, 21] + [30, 31, 32, 33, 34] + list(range(52)), dtype=jnp.int32)[:52]
    state = _make_state(
        env, agent_cards=[hand0, [1, 6, 11, 16, 21]], chips_in=[2.0, 2.0], round_raised=[0.0, 0.0],
        remaining_chips=[98.0, 98.0], stage=1, current_player_idx=0, draw_start_idx=0,
        shuffled_deck=deck, deck_idx=10,
    )
    next_state, *_ = env.step_env(rng, state, jnp.int32(5))
    assert sorted(np.asarray(next_state.agent_cards[0]).tolist()) == [30, 31, 32, 33, 34]
    assert int(next_state.deck_idx) == 15


def test_edge_raise_half_pot_illegal_when_it_does_not_exceed_call() -> None:
    """Raise-half-pot is available only if it actually exceeds a call
    (half_pot + player_round > max_round); it must be a genuine raise, not
    just a call. Constructs a pot large enough that half_pot equals the
    call amount exactly, which must make the action illegal."""
    env = make("five_card_draw", num_agents=2, horizon=100_000, init_chips=100)
    # pot = chips_in.sum() = 10; half_pot = 5. player_round=0, max_round=5 -> half_pot+0 == max_round -> NOT a raise.
    state = _make_state(
        env, agent_cards=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], chips_in=[5.0, 5.0], round_raised=[0.0, 5.0],
        remaining_chips=[95.0, 95.0], stage=0, current_player_idx=0,
    )
    avail = env.get_avail_actions(state)
    assert not bool(avail[1]), "half-pot raise must be illegal when it would only match the call, not exceed it"
    assert bool(avail[0]), "check/call must still be legal"


def test_edge_fold_gives_pot_to_sole_remaining_player_without_showdown() -> None:
    """If only one player remains, they win the pot uncontested, with no
    showdown."""
    env = make("five_card_draw", num_agents=2, horizon=100_000, init_chips=100)
    rng = jax.random.PRNGKey(0)
    state = _make_state(
        env, agent_cards=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], chips_in=[6.0, 4.0], round_raised=[6.0, 4.0],
        remaining_chips=[94.0, 96.0], stage=0, current_player_idx=1, not_raise_num=0,
    )
    next_state, obs, reward, absorbing, done, info = env.step_env(rng, state, jnp.int32(4))  # player 1 folds
    assert bool(done)
    assert np.isclose(float(reward[0]), 4.0), "player 0 wins player 1's entire contribution uncontested"
    assert np.isclose(float(reward[1]), -4.0)


def test_edge_tied_hands_split_pot_evenly() -> None:
    """The pot splits equally among tied winners. Two identical-rank
    high-card hands of different suits should tie and split the pot 50/50."""
    env = make("five_card_draw", num_agents=2, horizon=100_000, init_chips=100)
    rng = jax.random.PRNGKey(0)
    # Both hands share the rank multiset {A,2,4,6,8} (mod-13 values 0,1,2,3,8)
    # but with suits arranged differently, so neither hand is a flush and the
    # two are guaranteed to tie on high card.
    p0 = [0, 13 + 1, 26 + 3, 39 + 5, 0 + 7]  # ranks: A(club),2(diamond),4(heart),6(spade),8(club) approx distinct suits
    p1 = [13 + 0, 1, 26 + 3 - 0 + 0, 39 + 5, 7]  # deliberately construct identical rank multiset, different arrangement
    p0 = [0, 14, 28, 42, 8]     # ranks (mod 13): 0,1,2,3,8 -> suits 0,1,2,3,0
    p1 = [13, 1, 41, 29, 21]    # ranks (mod 13): 0,1,2,3,8 -> suits 1,0,3,2,1 (same rank multiset, no flush either side)
    state = _make_state(
        env, agent_cards=[p0, p1], chips_in=[5.0, 5.0], round_raised=[0.0, 0.0],
        remaining_chips=[95.0, 95.0], stage=2, current_player_idx=0, not_raise_num=2,
    )
    next_state, obs, reward, absorbing, done, info = env.step_env(rng, state, jnp.int32(0))  # check -> round closes -> showdown
    assert bool(done)
    assert np.isclose(float(reward[0]), 0.0, atol=1e-3) and np.isclose(float(reward[1]), 0.0, atol=1e-3), (
        f"identical-rank hands should tie and split the pot evenly (net 0 each); got {np.asarray(reward)}"
    )


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

    env = make("five_card_draw", num_agents=2, horizon=100_000, init_chips=100)

    if args.random:
        kind, network, params = "random", None, None
    else:
        kind = args.network_type or infer_network_kind(args.checkpoint)
        network = build_network(kind, action_dim=env.num_actions, hidden_dim=args.hidden_dim)
        _, sample_obs = env.reset(jax.random.PRNGKey(0))
        params = load_checkpoint_params(network, sample_obs, args.checkpoint)

    print(f"Validating {args.episodes} episode(s) of '{kind}' agent (five_card_draw)...")
    checker = rollout_and_validate(env, kind, network, params, args.episodes, seed=args.seed)

    if checker.violations:
        print(f"\n{len(checker.violations)} VIOLATION(S) FOUND:")
        for v in checker.violations[:20]:
            print(f"  {v}")
        raise SystemExit(1)
    print("\nNo rule violations found.")


if __name__ == "__main__":
    main()
