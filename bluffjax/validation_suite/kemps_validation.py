"""
Rule-conformance validation suite for Kemps.

Checks every state/action/transition produced by random or checkpoint-driven
rollouts against bluffjax/environments/kemps/rules.md and the implementation
in bluffjax/environments/kemps/kemps.py.

Covers card-uniqueness and rank-count invariants, KEMPS/STOP-KEMPS resolution
correctness (recomputed independently from raw hand contents rather than by
re-deriving kemps.py's own formula), the KEMPS-vs-CAUGHT simultaneous-
declaration precedence rule, swap legality (invalid swap attempts must be
silent no-ops, not corruption), and center-refresh vs. stock-exhaustion
termination.

Kemps is a ParallelEnv: all 4 agents act simultaneously every step.

Usage:
    pytest bluffjax/validation_suite/kemps_validation.py -v
    python kemps_validation.py --random --episodes 300
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.environments.kemps.kemps import Kemps, KempsState
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
EXAMPLES_DIR = os.path.normpath(os.path.join(HERE, "..", "examples", "kemps"))
CHECKPOINT_ROOT = os.path.join(EXAMPLES_DIR, "checkpoints")

DEFAULT_SEED = 0
RANDOM_EPISODES = 150
# Kemps is a 4-agent ParallelEnv, so each step needs 4 separate (non-batched)
# policy calls in the plain-python rollout loop -- proportionately slower
# per episode than single-policy-call-per-step games, hence the lower
# episode count.
CHECKPOINT_EPISODES = 12


class RuleChecker(RuleCheckerBase):
    def __init__(self, env: Kemps):
        super().__init__(env)

    # -- setup / reset --------------------------------------------------------

    def check_reset(self, state: KempsState) -> None:
        """Deals 4 face-down cards to each player and 4 face-up cards in a
        row in the center at reset; communication starts all-zero."""
        self._check_card_uniqueness(-1, state)
        self._check_rank_counts_consistent(-1, state)
        if not np.allclose(np.asarray(state.communication), 0.0):
            self._fail(0, "setup/no_initial_signal", "communication must start all-zero")
        if int(state.deck_idx) != 0:
            self._fail(0, "setup/deck_idx_zero", "deck_idx must start at 0")

    # -- structural invariants ------------------------------------------------

    def _check_card_uniqueness(self, step, state: KempsState) -> None:
        """No card index may appear MORE THAN ONCE across {all agent_hands,
        center_cards, remaining deck} -- catches a swap bug that duplicates a
        card even when a count-based check would miss it (duplicate-for-
        duplicate swaps still sum correctly).

        Deliberately `<= 1`, not `== 1`: an all-NOOP refresh discards the
        old center cards for good, so the live card count shrinks by
        `num_center` on every refresh, and requiring all 52 cards to always
        be findable somewhere would be wrong once a refresh has occurred
        (see `_check_live_card_count_shrinks_only_on_refresh` for the
        matching monotonic-count check)."""
        all_cards = jnp.concatenate(
            [state.agent_hands.reshape(-1), state.center_cards.reshape(-1), state.deck[state.deck_idx :]]
        )
        counts = np.asarray(jnp.bincount(all_cards, length=self.env.deck_size))
        if not np.all(counts <= 1):
            self._fail(
                step,
                "conservation/no_card_duplication",
                f"{int((counts > 1).sum())} card(s) appear in more than one place at once",
            )

    def _check_live_card_count_shrinks_only_on_refresh(self, step, pre: KempsState, post: KempsState, all_noop: bool, any_declare: bool) -> None:
        """The total number of 'live' cards (in some hand, in the center, or
        still in the undealt deck) must stay exactly constant on any
        swap-only step, and must shrink by exactly `num_center` (the
        discarded old center) on an all-NOOP refresh step."""
        def live_count(state):
            return int(np.asarray(state.agent_hands).size + np.asarray(state.center_cards).size + (state.deck.shape[0] - int(state.deck_idx)))

        pre_count, post_count = live_count(pre), live_count(post)
        if any_declare:
            return  # declarations end the hand; no card movement to check
        if all_noop:
            # Either a successful refresh (old center of size num_center
            # discarded) or a stock-exhausted draw (nothing moves).
            if post_count not in (pre_count, pre_count - self.env.num_center):
                self._fail(step, "conservation/live_count_on_refresh", f"live card count {pre_count} -> {post_count} on an all-NOOP step")
        else:
            if post_count != pre_count:
                self._fail(step, "conservation/live_count_on_swap", f"live card count {pre_count} -> {post_count} on a swap-only step (must stay constant)")

    def _check_rank_counts_consistent(self, step, state: KempsState) -> None:
        """agent_hand_counts/center_counts (rank histograms, used for O(1)
        swap-legality checks) must always match the ranks implied by the
        actual card indices."""
        def ranks_of(cards):
            return cards // self.env.num_suits

        for agent in range(self.env.num_agents):
            recomputed = np.asarray(
                jnp.bincount(ranks_of(state.agent_hands[agent]), length=self.env.num_ranks, minlength=self.env.num_ranks)
            )
            if not np.array_equal(np.asarray(state.agent_hand_counts[agent]), recomputed):
                self._fail(step, "invariant/hand_counts_match", f"agent_hand_counts[{agent}] drifted from actual hand")
        recomputed_center = np.asarray(
            jnp.bincount(ranks_of(state.center_cards), length=self.env.num_ranks, minlength=self.env.num_ranks)
        )
        if not np.array_equal(np.asarray(state.center_counts), recomputed_center):
            self._fail(step, "invariant/center_counts_match", "center_counts drifted from actual center cards")

    def _check_hand_size_constant(self, step, state: KempsState) -> None:
        """Every agent always holds exactly `hand_size` cards -- a swap
        trades one-for-one, and a refresh replaces the whole center, not a
        player's hand."""
        sizes = np.asarray(state.agent_hands).shape[1]
        if sizes != self.env.hand_size:
            self._fail(step, "invariant/hand_size_constant", f"hand width={sizes} != hand_size={self.env.hand_size}")

    # -- declaration resolution -----------------------------------------------------

    def _check_declaration_resolution(self, step, pre: KempsState, actions, reward) -> None:
        """If multiple calls are declared in the same step, whichever type
        the lowest-indexed declaring agent used is the one that resolves (if
        only one type was declared, that type resolves regardless of index);
        among agents declaring the winning type, the lowest-indexed one is
        "the" caller. KEMPS checks the caller's partner for 4-of-a-kind;
        CAUGHT checks the whole opposing team. Both are recomputed
        independently from `agent_hand_counts`, not from kemps.py's own
        `resolve_kemps`/`resolve_stop`."""
        num_ranks_sq = self.env.num_ranks * self.env.num_ranks
        action_kemps = num_ranks_sq + 1
        action_stop = num_ranks_sq + 2
        game_action = np.asarray(actions) // self.env.comm_dim

        kemps_callers = np.where(game_action == action_kemps)[0]
        stop_callers = np.where(game_action == action_stop)[0]
        if len(kemps_callers) == 0 and len(stop_callers) == 0:
            return  # no declaration this step

        kemps_caller = kemps_callers.min() if len(kemps_callers) else None
        stop_caller = stop_callers.min() if len(stop_callers) else None

        if kemps_caller is not None and (stop_caller is None or kemps_caller < stop_caller):
            resolving_type, caller = "kemps", int(kemps_caller)
        else:
            resolving_type, caller = "stop", int(stop_caller)

        team = lambda idx: idx % 2
        hand_counts = np.asarray(pre.agent_hand_counts)
        reward = np.asarray(reward)

        if resolving_type == "kemps":
            partner = (caller + 2) % self.env.num_agents
            partner_has_4 = bool((hand_counts[partner] == 4).any())
            win = partner_has_4
        else:
            opp_team = 1 - team(caller)
            opp_mask = np.array([team(a) == opp_team for a in range(self.env.num_agents)])
            opp_has_4 = bool(((hand_counts == 4).any(axis=1) & opp_mask).any())
            win = opp_has_4

        caller_team = team(caller)
        expected = np.array(
            [(1.0 if win else -1.0) if team(a) == caller_team else (-1.0 if win else 1.0) for a in range(self.env.num_agents)]
        )
        if not np.allclose(reward, expected, atol=1e-4):
            self._fail(
                step,
                f"declaration/{resolving_type}_resolution",
                f"resolving_type={resolving_type}, caller={caller}, win={win}: "
                f"expected reward {expected}, got {reward}",
            )
        if not np.isclose(reward.sum(), 0.0, atol=1e-4):
            self._fail(step, "declaration/zero_sum", f"declaration reward {reward} does not sum to 0")

    # -- swap legality -----------------------------------------------------------

    def _check_swap_legality_and_noop_safety(self, step, pre: KempsState, post: KempsState, actions, any_declare: bool, all_noop: bool) -> None:
        """An invalid swap attempt (a rank the agent doesn't hold, or a rank
        absent from the center) must be a silent no-op for that agent -- it
        must not corrupt that agent's hand, the center, or any other
        agent's cards."""
        if any_declare or all_noop:
            return
        num_ranks_sq = self.env.num_ranks * self.env.num_ranks
        game_action = np.asarray(actions) // self.env.comm_dim
        pre_hand_counts = np.asarray(pre.agent_hand_counts)
        pre_center_counts = np.asarray(pre.center_counts)
        for agent in range(self.env.num_agents):
            a = int(game_action[agent])
            if a >= num_ranks_sq:
                continue  # noop/declare, not a swap attempt
            lose_rank, gain_rank = a // self.env.num_ranks, a % self.env.num_ranks
            has_lose = pre_hand_counts[agent, lose_rank] >= 1
            center_has_gain = pre_center_counts[gain_rank] >= 1
            if not (has_lose and center_has_gain):
                if not np.array_equal(np.asarray(post.agent_hands[agent]), np.asarray(pre.agent_hands[agent])):
                    self._fail(
                        step,
                        "swap/invalid_attempt_is_noop",
                        f"agent {agent}'s invalid swap (lose={lose_rank}, gain={gain_rank}, "
                        f"has_lose={has_lose}, center_has_gain={center_has_gain}) changed their hand",
                    )

    # -- center refresh / stock exhaustion ------------------------------------------

    def _check_all_noop_behavior(self, step, pre: KempsState, post: KempsState, all_noop: bool, any_declare: bool, reward) -> None:
        """If every agent chose NOOP, the center is swept and 4 new cards
        dealt; if the stock can't supply 4 more, the hand ends in a
        scoreless draw."""
        if any_declare or not all_noop:
            return
        deck_len = int(pre.deck.shape[0])
        can_deal = int(pre.deck_idx) + self.env.num_center <= deck_len
        if can_deal:
            expected_center = np.asarray(pre.deck)[int(pre.deck_idx) : int(pre.deck_idx) + self.env.num_center]
            if not np.array_equal(np.asarray(post.center_cards), expected_center):
                self._fail(step, "refresh/center_from_deck", "all-noop center refresh did not deal the next deck cards")
            if not np.allclose(np.asarray(reward), 0.0):
                self._fail(step, "refresh/no_reward", "a plain center refresh must give reward 0 to everyone")
        else:
            if not np.allclose(np.asarray(reward), 0.0):
                self._fail(step, "draw/scoreless", "a stock-exhausted draw must give reward 0 to everyone")

    # -- action legality / no-deadlock --------------------------------------------

    def _check_action_legal(self, step, avail, actions) -> None:
        for agent in range(self.env.num_agents):
            if not bool(avail[agent, actions[agent]]):
                self._fail(step, "action/within_avail_mask", f"agent {agent} took illegal action {actions[agent]}")

    def _check_avail_nonempty(self, step, avail) -> None:
        for agent in range(self.env.num_agents):
            if not bool(jnp.asarray(avail[agent]).any()):
                self._fail(step, "action/no_deadlock", f"agent {agent} has no legal action (NOOP should always be legal)")

    # -- top-level dispatch ------------------------------------------------------

    def validate_transition(self, step, pre: KempsState, actions, avail, post: KempsState, reward, done: bool) -> None:
        self._check_avail_nonempty(step, avail)
        self._check_action_legal(step, avail, actions)

        num_ranks_sq = self.env.num_ranks * self.env.num_ranks
        action_noop, action_kemps, action_stop = num_ranks_sq, num_ranks_sq + 1, num_ranks_sq + 2
        game_action = np.asarray(actions) // self.env.comm_dim
        any_declare = bool(((game_action == action_kemps) | (game_action == action_stop)).any())
        all_noop = bool((game_action == action_noop).all())

        self._check_declaration_resolution(step, pre, actions, reward)
        self._check_swap_legality_and_noop_safety(step, pre, post, actions, any_declare, all_noop)
        self._check_all_noop_behavior(step, pre, post, all_noop, any_declare, reward)
        self._check_live_card_count_shrinks_only_on_refresh(step, pre, post, all_noop, any_declare)
        self._check_card_uniqueness(step, post)
        self._check_rank_counts_consistent(step, post)
        self._check_hand_size_constant(step, post)


def rollout_and_validate(env: Kemps, kind, network, params, num_episodes: int, seed: int = DEFAULT_SEED):
    checker = RuleChecker(env)
    return rollout_and_validate_parallel(env, kind, network, params, num_episodes, checker, seed=seed)


# ---------------------------------------------------------------------------
# pytest entry points.
# ---------------------------------------------------------------------------


def test_random_agent_rollouts_conform_to_rules() -> None:
    env = make("kemps", num_agents=4, horizon=200)
    checker = rollout_and_validate(env, "random", None, None, RANDOM_EPISODES, seed=DEFAULT_SEED)
    checker.assert_no_violations()


@pytest.mark.parametrize(
    "checkpoint_info", one_checkpoint_per_algorithm_and_kind(discover_checkpoints(CHECKPOINT_ROOT), CHECKPOINT_ROOT)
)
def test_checkpoint_agent_rollouts_conform_to_rules(checkpoint_info) -> None:
    checkpoint_path, network_kind, algorithm = checkpoint_info
    env_kwargs, fc_dim_size = training_env_kwargs(EXAMPLES_DIR, algorithm)
    env = make("kemps", **env_kwargs)

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


def _hand_for_rank(rank, num_suits):
    """4 distinct-suit cards of the given rank, e.g. rank 0 -> [0,1,2,3]."""
    return [rank * num_suits + s for s in range(num_suits)]


def _make_state(env: Kemps, agent_hands, center_cards, deck=None, deck_idx=0) -> KempsState:
    agent_hands = jnp.array(agent_hands, dtype=jnp.int32)
    center_cards = jnp.array(center_cards, dtype=jnp.int32)

    def counts_of(cards):
        ranks = cards // env.num_suits
        return jnp.bincount(ranks.astype(jnp.int32), length=env.num_ranks, minlength=env.num_ranks)

    agent_hand_counts = jnp.stack([counts_of(h) for h in agent_hands])
    center_counts = counts_of(center_cards)
    if deck is None:
        deck = jnp.arange(env.deck_size, dtype=jnp.int32)
    return KempsState(
        agent_hands=agent_hands,
        agent_hand_counts=agent_hand_counts,
        center_cards=center_cards,
        center_counts=center_counts,
        deck=jnp.array(deck, dtype=jnp.int32),
        deck_idx=jnp.int32(deck_idx),
        communication=jnp.zeros((env.num_agents, env.comm_dim), dtype=jnp.float32),
        absorbing=jnp.zeros(env.num_agents, dtype=bool),
        done=False,
        timestep=0,
    )


def _noop_action(env: Kemps) -> int:
    num_ranks_sq = env.num_ranks * env.num_ranks
    return (num_ranks_sq) * env.comm_dim  # NOOP with comm_signal=0


def _declare_action(env: Kemps, kind: str) -> int:
    num_ranks_sq = env.num_ranks * env.num_ranks
    game_action = num_ranks_sq + (1 if kind == "kemps" else 2)
    return game_action * env.comm_dim


def test_edge_truthful_kemps_call_rewards_caller_team() -> None:
    """KEMPS checks whether the caller's partner truly holds four-of-a-kind;
    if true, the caller's team gets +1 and the other team -1. Agent 0 calls
    KEMPS; its partner (agent 2, since partner=(idx+2)%4) genuinely holds
    four 7s."""
    env = make("kemps", num_agents=4, horizon=200)
    rng = jax.random.PRNGKey(0)
    hands = [
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        _hand_for_rank(7, env.num_suits),  # agent 2 (agent 0's partner): four 7s
        [16, 17, 18, 19],
    ]
    center = [20, 21, 22, 23]
    state = _make_state(env, hands, center)

    actions = jnp.array([_declare_action(env, "kemps"), _noop_action(env), _noop_action(env), _noop_action(env)])
    next_state, obs, reward, absorbing, done, info = env.step_env(rng, state, actions)

    assert bool(done), "a resolved declaration must end the hand"
    reward = np.asarray(reward)
    assert reward[0] == 1.0 and reward[2] == 1.0, "caller's team (0,2) should each get +1"
    assert reward[1] == -1.0 and reward[3] == -1.0, "opposing team (1,3) should each get -1"


def test_edge_false_kemps_call_penalizes_caller_team() -> None:
    """If the KEMPS call is wrong, the outcome reverses: agent 0 calls
    KEMPS but its partner (agent 2) does NOT hold four-of-a-kind, so the
    caller's team loses."""
    env = make("kemps", num_agents=4, horizon=200)
    rng = jax.random.PRNGKey(0)
    hands = [
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [0, 4, 8, 12],  # agent 2: no four-of-a-kind
        [16, 17, 18, 19],
    ]
    center = [20, 21, 22, 23]
    state = _make_state(env, hands, center)

    actions = jnp.array([_declare_action(env, "kemps"), _noop_action(env), _noop_action(env), _noop_action(env)])
    next_state, obs, reward, absorbing, done, info = env.step_env(rng, state, actions)

    reward = np.asarray(reward)
    assert reward[0] == -1.0 and reward[2] == -1.0, "a false KEMPS call must penalize the caller's own team"
    assert reward[1] == 1.0 and reward[3] == 1.0


def test_edge_truthful_caught_call_rewards_accusing_team() -> None:
    """CAUGHT checks whether any member of the opposing team currently
    holds four-of-a-kind; if true, the accusing team gets +1 and the
    accused team -1."""
    env = make("kemps", num_agents=4, horizon=200)
    rng = jax.random.PRNGKey(0)
    hands = [
        [8, 9, 10, 11],
        _hand_for_rank(3, env.num_suits),  # agent 1: genuinely has four-of-a-kind
        [12, 13, 14, 15],
        [16, 17, 18, 19],
    ]
    center = [20, 21, 22, 23]
    state = _make_state(env, hands, center)

    # Agent 0 (team 0) accuses the opposing team (team 1 = agents 1,3).
    actions = jnp.array([_declare_action(env, "stop"), _noop_action(env), _noop_action(env), _noop_action(env)])
    next_state, obs, reward, absorbing, done, info = env.step_env(rng, state, actions)

    reward = np.asarray(reward)
    assert reward[0] == 1.0 and reward[2] == 1.0, "accusing team (0,2) should each get +1"
    assert reward[1] == -1.0 and reward[3] == -1.0, "accused team (1,3) should each get -1"


def test_edge_lowest_index_declaration_type_resolves_on_conflict() -> None:
    """Whichever type the lowest-indexed declaring agent used is the one
    that resolves. Agent 1 declares CAUGHT and agent 2 declares KEMPS in the
    same step -- since 1 < 2, CAUGHT must resolve, not KEMPS. This is
    subtler than kemps.py's own inline comment ("KEMPS takes precedence if
    both declared") suggests: the actual rule compares caller indices."""
    env = make("kemps", num_agents=4, horizon=200)
    rng = jax.random.PRNGKey(0)
    # Each hand holds 4 DISTINCT ranks (0-3), one per suit -- deliberately
    # NOT four-of-a-kind for any agent (a naive [8,9,10,11]-style "junk"
    # hand is a trap here: with num_suits=4, any 4 consecutive card indices
    # aligned to a multiple of 4 are secretly a real four-of-a-kind).
    hands = [
        [0, 4, 8, 12],   # agent 0: ranks 0,1,2,3 (suit 0) -- no quad
        [1, 5, 9, 13],   # agent 1: ranks 0,1,2,3 (suit 1) -- no quad
        [2, 6, 10, 14],  # agent 2: ranks 0,1,2,3 (suit 2) -- no quad
        [3, 7, 11, 15],  # agent 3: ranks 0,1,2,3 (suit 3) -- no quad
    ]
    center = [16, 20, 24, 28]
    state = _make_state(env, hands, center)

    # Agent 1 (team 1) declares CAUGHT (accusing team 0 = agents 0,2); no
    # member of team 0 actually holds four-of-a-kind here, so if CAUGHT
    # resolves, the false accusation should penalize agent 1's own team.
    # Agent 2 (team 0) simultaneously declares KEMPS.
    actions = jnp.array([_noop_action(env), _declare_action(env, "stop"), _declare_action(env, "kemps"), _noop_action(env)])
    next_state, obs, reward, absorbing, done, info = env.step_env(rng, state, actions)

    reward = np.asarray(reward)
    # If CAUGHT (agent 1, team 1) resolves and is false: team 1 gets -1, team 0 gets +1.
    assert reward[1] == -1.0 and reward[3] == -1.0, (
        f"CAUGHT (lower-indexed declarer, agent 1) should resolve, not KEMPS (agent 2); reward={reward}"
    )
    assert reward[0] == 1.0 and reward[2] == 1.0


def test_edge_invalid_swap_attempt_is_silent_noop() -> None:
    """A swap targeting a rank the agent doesn't hold (or a rank absent
    from center) must not corrupt state -- it's simply not applied for
    that agent, while other agents' legal swaps in the same step still go
    through."""
    env = make("kemps", num_agents=4, horizon=200)
    rng = jax.random.PRNGKey(0)
    hands = [
        [0, 4, 8, 12],  # agent 0: ranks 0,1,2,3 -- does NOT hold rank 10
        [16, 20, 24, 28],  # agent 1: holds rank 4 (card 16), will do a valid swap
        [32, 36, 40, 44],
        [1, 5, 9, 13],
    ]
    center = [48, 49, 50, 51]  # rank 12 cards
    state = _make_state(env, hands, center)

    # Agent 0 attempts an ILLEGAL swap: lose_rank=10 (not held), gain_rank=12 (in center).
    illegal_swap = (10 * env.num_ranks + 12) * env.comm_dim
    # Agent 1 attempts a LEGAL swap: lose_rank=4 (holds card 16), gain_rank=12 (in center).
    legal_swap = (4 * env.num_ranks + 12) * env.comm_dim

    actions = jnp.array([illegal_swap, legal_swap, _noop_action(env), _noop_action(env)])
    next_state, obs, reward, absorbing, done, info = env.step_env(rng, state, actions)

    assert np.array_equal(np.asarray(next_state.agent_hands[0]), np.asarray(hands[0])), (
        "agent 0's illegal swap attempt must be a silent no-op, leaving their hand unchanged"
    )
    assert not np.array_equal(np.asarray(next_state.agent_hands[1]), np.asarray(hands[1])), (
        "agent 1's legal swap should still be applied even though agent 0's attempt was illegal"
    )
    assert not bool(done)


def test_edge_all_noop_refreshes_center_from_deck() -> None:
    """Once nobody wants the exposed center cards, they are discarded for
    the round and four fresh cards are dealt. All 4 agents NOOP -> the
    center is fully replaced by the next 4 undealt deck cards."""
    env = make("kemps", num_agents=4, horizon=200)
    rng = jax.random.PRNGKey(0)
    hands = [[0, 4, 8, 12], [16, 20, 24, 28], [32, 36, 40, 44], [1, 5, 9, 13]]
    center = [48, 49, 50, 51]
    remaining_deck = jnp.array([2, 3, 6, 7, 10, 11], dtype=jnp.int32)
    state = _make_state(env, hands, center, deck=remaining_deck, deck_idx=0)

    actions = jnp.array([_noop_action(env)] * 4)
    next_state, obs, reward, absorbing, done, info = env.step_env(rng, state, actions)

    assert np.array_equal(np.asarray(next_state.center_cards), [2, 3, 6, 7]), (
        "all-noop must deal the next 4 cards from the deck into the center"
    )
    assert int(next_state.deck_idx) == 4
    assert np.allclose(np.asarray(reward), 0.0)
    assert not bool(done)


def test_edge_all_noop_with_exhausted_stock_ends_in_scoreless_draw() -> None:
    """If the stock can't supply 4 more cards, the hand ends in a
    scoreless draw."""
    env = make("kemps", num_agents=4, horizon=200)
    rng = jax.random.PRNGKey(0)
    hands = [[0, 4, 8, 12], [16, 20, 24, 28], [32, 36, 40, 44], [1, 5, 9, 13]]
    center = [48, 49, 50, 51]
    # `deck` keeps the same fixed length reset() always produces (JAX traces
    # against a static shape); stock exhaustion is represented by deck_idx
    # leaving too few cards *within* that array, not by shrinking it.
    remaining_deck = jnp.array([2, 3, 6, 7, 10, 11], dtype=jnp.int32)
    state = _make_state(env, hands, center, deck=remaining_deck, deck_idx=3)  # only 3 cards left (idx 3,4,5), need 4

    actions = jnp.array([_noop_action(env)] * 4)
    next_state, obs, reward, absorbing, done, info = env.step_env(rng, state, actions)

    assert bool(done), "an exhausted stock on an all-noop round must end the episode"
    assert np.allclose(np.asarray(reward), 0.0), "the stock-exhausted draw must be scoreless"


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

    env = make("kemps", num_agents=4, horizon=200)

    if args.random:
        kind, network, params = "random", None, None
    else:
        kind = args.network_type or infer_network_kind(args.checkpoint)
        network = build_network(kind, action_dim=env.num_actions, hidden_dim=args.hidden_dim)
        _, sample_obs = env.reset(jax.random.PRNGKey(0))
        params = load_checkpoint_params(network, sample_obs[0], args.checkpoint)

    print(f"Validating {args.episodes} episode(s) of '{kind}' agent (kemps)...")
    checker = rollout_and_validate(env, kind, network, params, args.episodes, seed=args.seed)

    if checker.violations:
        print(f"\n{len(checker.violations)} VIOLATION(S) FOUND:")
        for v in checker.violations[:20]:
            print(f"  {v}")
        raise SystemExit(1)
    print("\nNo rule violations found.")


if __name__ == "__main__":
    main()
