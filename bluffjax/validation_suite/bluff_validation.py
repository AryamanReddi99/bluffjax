"""
Rule-conformance validation suite for the Bluff environment.

Purpose
-------
This suite drives full episodes of Bluff with either a uniform-random
policy *or* a trained checkpoint (PPO-NFSP / PQN-NFSP, "br" or "avg"
network), and checks, transition by transition, that every state, action,
and state change is consistent with the rules described in

    bluffjax/environments/bluff/rules.md

and implemented in

    bluffjax/environments/bluff/bluff.py

Each check method on `RuleChecker` below documents the specific rule it
verifies. Where possible, a bookkeeping field (e.g. `agent_hand_sizes`,
`pile_size`, `claim_rank`) is independently recomputed from primitive state
(`agent_hands`, `pile_hand`, `pending_play_hand`) and compared, so a bug
that corrupts one field but not the other is still caught.

Usage
-----
As a pytest suite (random-agent checks always run; checkpoint checks run
automatically if any checkpoints are found under
`bluffjax/examples/bluff/checkpoints/`, otherwise they are skipped):

    pytest bluffjax/validation_suite/bluff_validation.py -v

As a standalone script, to validate one specific checkpoint over N episodes:

    python bluff_validation.py --checkpoint path/to/br_1.msgpack \\
        --network-type actor_critic --episodes 500 --num-agents 2

    python bluff_validation.py --random --episodes 200 --num-agents 4
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from flax import serialization

from bluffjax import make
from bluffjax.environments.bluff.bluff import Bluff, BluffState
from bluffjax.networks.mlp import ActorCriticDiscreteMLP, ActorDiscreteMLP, QNetworkDiscreteMLP

HERE = os.path.dirname(os.path.abspath(__file__))
BLUFF_EXAMPLES_DIR = os.path.normpath(os.path.join(HERE, "..", "examples", "bluff"))
CHECKPOINT_ROOT = os.path.join(BLUFF_EXAMPLES_DIR, "checkpoints")

DEFAULT_HORIZON = 200
DEFAULT_SEED = 0
RANDOM_EPISODES_PER_CONFIG = 60
CHECKPOINT_EPISODES = 20

RUN_DIR_RE = re.compile(r"^(ppo_nfsp|pqn_nfsp)_.*_s\d+$")


# ---------------------------------------------------------------------------
# Agents: uniform-random legal-action policy, or a loaded checkpoint network.
# ---------------------------------------------------------------------------


def sample_action(
    kind: str,
    network: Optional[Any],
    params: Optional[Any],
    obs: jnp.ndarray,
    avail: jnp.ndarray,
    rng: jnp.ndarray,
) -> jnp.ndarray:
    """Pick a legal action for one of the supported agent kinds.

    kind == "random": uniform over legal actions.
    kind == "actor"/"actor_critic": sample the policy's categorical logits,
        masked to legal actions (this is how avg-policy / PPO-BR checkpoints
        are ordinarily evaluated).
    kind == "q_network": greedy w.r.t. Q-values among legal actions, ties
        broken uniformly (mirrors bluff_pqn_nfsp.py's sample_action_q_greedy).
    """
    if kind == "random":
        logits = jnp.where(avail, 0.0, -1e9)
        return jax.random.categorical(rng, logits)
    if kind in ("actor", "actor_critic"):
        out = network.apply(params, obs)
        logits = out[0] if isinstance(out, tuple) else out
        logits = jnp.where(avail, logits, -jnp.inf)
        return jax.random.categorical(rng, logits)
    if kind == "q_network":
        q_vals = network.apply(params, obs.astype(jnp.float32))
        q_vals_masked = jnp.where(avail, q_vals, -jnp.inf)
        best_val = jnp.max(q_vals_masked)
        ties = (q_vals_masked == best_val) & avail
        tie_logits = jnp.where(ties, 0.0, -1e9)
        return jax.random.categorical(rng, tie_logits)
    raise ValueError(f"Unknown agent kind: {kind}")


def infer_network_kind(checkpoint_path: str) -> str:
    """Infer whether a checkpoint is an avg policy ("actor"), a PPO best
    response ("actor_critic"), or a PQN best response ("q_network"), from the
    save-path convention used by bluff_ppo_nfsp.py / bluff_pqn_nfsp.py:
    `<run_dir>/{avg,br}_<frac_tag>.msgpack` where run_dir is prefixed
    `ppo_nfsp_` or `pqn_nfsp_`."""
    fname = os.path.basename(checkpoint_path)
    parent = os.path.basename(os.path.dirname(checkpoint_path))
    haystack = f"{parent}/{fname}"
    if fname.startswith("avg") or "_avg" in fname:
        return "actor"
    if "ppo" in haystack:
        return "actor_critic"
    if "pqn" in haystack:
        return "q_network"
    raise ValueError(
        f"Cannot infer network kind for '{checkpoint_path}'; pass network_type explicitly."
    )


def build_network(kind: str, action_dim: int, hidden_dim: int = 128):
    if kind == "actor":
        return ActorDiscreteMLP(action_dim=action_dim, hidden_dim=hidden_dim)
    if kind == "actor_critic":
        return ActorCriticDiscreteMLP(action_dim=action_dim, hidden_dim=hidden_dim)
    if kind == "q_network":
        return QNetworkDiscreteMLP(action_dim=action_dim, hidden_dim=hidden_dim)
    raise ValueError(f"Unknown network kind: {kind}")


def load_checkpoint_params(network, sample_obs: jnp.ndarray, checkpoint_path: str):
    template_params = network.init(jax.random.PRNGKey(0), sample_obs)
    with open(checkpoint_path, "rb") as f:
        return serialization.from_bytes(template_params, f.read())


def discover_checkpoints() -> list[tuple[str, str]]:
    """Find (checkpoint_path, network_kind) pairs under the standard
    `ppo_nfsp_*_s<seed>` / `pqn_nfsp_*_s<seed>` checkpoint directories, so
    the checkpoint-driven tests run automatically when training outputs
    exist and skip cleanly otherwise."""
    found = []
    if not os.path.isdir(CHECKPOINT_ROOT):
        return found
    for d in sorted(os.listdir(CHECKPOINT_ROOT)):
        run_dir = os.path.join(CHECKPOINT_ROOT, d)
        if not (os.path.isdir(run_dir) and RUN_DIR_RE.match(d)):
            continue
        for fname in sorted(os.listdir(run_dir)):
            if not fname.endswith(".msgpack"):
                continue
            path = os.path.join(run_dir, fname)
            try:
                kind = infer_network_kind(path)
            except ValueError:
                continue
            found.append((path, kind))
    return found


def _training_env_kwargs(algorithm: str) -> dict:
    """Load env_kwargs/fc_dim_size from the config used to train checkpoints,
    so a loaded checkpoint's network is applied to observations of the exact
    shape it was trained on."""
    cfg_name = "config_ppo_nfsp.yaml" if algorithm == "ppo_nfsp" else "config_pqn_nfsp.yaml"
    with open(os.path.join(BLUFF_EXAMPLES_DIR, cfg_name), "r") as f:
        config = yaml.safe_load(f)
    return config["env_kwargs"], config["fc_dim_size"]


# ---------------------------------------------------------------------------
# Rule checker
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    episode: int
    step: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"[episode {self.episode}, step {self.step}] ({self.rule}) {self.message}"


@dataclass
class Coverage:
    """Counts of which rule branches actually fired across a batch of
    rollouts, so a clean validation run can be distinguished from a vacuous
    one that never exercised, say, a resolved challenge or a forced-rank
    turn."""

    free_rank_turns: int = 0
    forced_rank_turns: int = 0
    challenge_resolutions_claim_true: int = 0
    challenge_resolutions_claim_false: int = 0
    no_challenge_resolutions: int = 0
    wins: int = 0
    truncations: int = 0
    steps: int = 0


class RuleChecker:
    """Validates one BluffState transition at a time.

    Call `start_episode()` at the start of each episode, then
    `validate_transition(...)` after every `env.step`. Violations are
    collected rather than raised immediately, so a full diagnostic report
    can be produced for a checkpoint that systematically violates the
    rules instead of stopping at the first offending step.
    """

    def __init__(self, env: Bluff):
        self.env = env
        self.violations: list[Violation] = []
        self.coverage = Coverage()
        self._episode = -1
        self._first_play_seen = False
        self._expected_claim_rank: Optional[int] = None
        self._turn_is_free: Optional[bool] = None

    def start_episode(self) -> None:
        self._episode += 1
        self._first_play_seen = False
        self._expected_claim_rank = None
        self._turn_is_free = None

    def _fail(self, step: int, rule: str, message: str) -> None:
        self.violations.append(Violation(self._episode, step, rule, message))

    # -- structural / bookkeeping invariants ---------------------------------

    def _check_hand_size_bookkeeping(self, step, pre: BluffState) -> None:
        """`agent_hand_sizes` must always equal `agent_hands.sum(axis=1)`.
        This underlies every hand-size-dependent rule below (claim cap, win
        condition), so a drift between the two would silently break them."""
        recomputed = np.asarray(pre.agent_hands).sum(axis=1).astype(np.int32)
        tracked = np.asarray(pre.agent_hand_sizes)
        if not np.array_equal(tracked, recomputed):
            self._fail(
                step,
                "bookkeeping/hand_size",
                f"agent_hand_sizes={tracked} != recomputed sum of agent_hands={recomputed}",
            )

    def _check_card_conservation(self, step, pre: BluffState) -> None:
        """The full deck is dealt out at the start of the game, with any
        remainder placed in the central pile. Cards only ever move between
        `agent_hands` and `pile_hand` (a play moves hand->pile, a resolved
        challenge moves pile->loser's hand); none are created or destroyed,
        so their total must always equal `deck_size`."""
        total = float(jnp.asarray(pre.agent_hands).sum() + jnp.asarray(pre.pile_hand).sum())
        if not np.isclose(total, self.env.deck_size):
            self._fail(
                step,
                "conservation/card_count",
                f"agent_hands.sum() + pile_hand.sum() = {total} != deck_size={self.env.deck_size}",
            )

    def _check_no_negative_counts(self, step, pre: BluffState) -> None:
        """A player cannot hold or discard a negative number of cards of any
        rank; this would only occur if a bug allowed playing more copies of a
        rank than were actually held."""
        if bool((jnp.asarray(pre.agent_hands) < 0).any()):
            self._fail(step, "conservation/no_negative", "agent_hands has a negative entry")
        if bool((jnp.asarray(pre.pile_hand) < 0).any()):
            self._fail(step, "conservation/no_negative", "pile_hand has a negative entry")

    def _check_pile_claims_consistent(self, step, pre: BluffState) -> None:
        """Every card added to `pile_hand` by a finished play adds one unit
        to `pile_claims` at the claimed rank, and both are reset together on
        challenge resolution, so `pile_size` and `pile_claims.sum()` should
        track each other exactly. The one exception is the leftover cards
        dealt straight into the central pile at reset: they were never
        claimed by a play, so they leave a fixed-size gap between the two
        that persists until the first challenge resolution empties the
        pile."""
        pile_claims_sum = float(jnp.asarray(pre.pile_claims).sum())
        pile_size = float(pre.pile_size)
        leftover = float(self.env.deck_size % self.env.num_agents)
        gap = pile_size - pile_claims_sum
        if not (np.isclose(gap, 0.0) or np.isclose(gap, leftover)):
            self._fail(
                step,
                "bookkeeping/pile_claims",
                f"pile_size - pile_claims.sum() = {gap}, expected 0 or the initial "
                f"unclaimed leftover-deal size ({leftover})",
            )

    def _check_current_player_in_range(self, step, pre: BluffState) -> None:
        """current_player_idx must always name a real seat."""
        idx = int(pre.current_player_idx)
        if not (0 <= idx < self.env.num_agents):
            self._fail(
                step, "bookkeeping/player_index", f"current_player_idx={idx} out of range"
            )

    # -- reset / setup --------------------------------------------------------

    def check_reset(self, state: BluffState) -> None:
        """Cards are dealt in equal shares to each player, with any leftover
        going directly into the central pile: at reset, every hand has
        exactly `deck_size // num_agents` cards and the pile has exactly the
        `deck_size % num_agents` remainder, which must not be added to any
        player's hand."""
        expected_hand = self.env.deck_size // self.env.num_agents
        expected_pile = self.env.deck_size % self.env.num_agents
        hand_sizes = np.asarray(state.agent_hand_sizes)
        if not np.all(hand_sizes == expected_hand):
            self._fail(
                0,
                "setup/even_deal",
                f"agent_hand_sizes at reset={hand_sizes}, expected all == {expected_hand}",
            )
        if int(state.pile_size) != expected_pile:
            self._fail(
                0,
                "setup/leftover_to_pile",
                f"pile_size at reset={int(state.pile_size)}, expected {expected_pile}",
            )
        if int(state.phase) != 0:
            self._fail(0, "setup/initial_phase", "game must start in phase 0 (claim)")

    # -- phase 0: claim size --------------------------------------------------

    def _check_claim_size_bounds(self, step, pre: BluffState, post: BluffState) -> None:
        """Claim size is 1 to 4 cards, capped by the active player's current
        hand size: a player can never claim more cards than they hold, and
        never more than 4 regardless of hand size."""
        hand_size_before = int(pre.agent_hand_sizes[pre.current_player_idx])
        claim_size = int(post.claim_size)
        if not (1 <= claim_size <= 4):
            self._fail(step, "phase0/claim_size_range", f"claim_size={claim_size} not in [1,4]")
        cap = min(4, max(hand_size_before, 1))
        if claim_size > cap:
            self._fail(
                step,
                "phase0/claim_size_cap",
                f"claim_size={claim_size} exceeds cap min(4, hand_size={hand_size_before})={cap}",
            )

    # -- phase 1: play cards ---------------------------------------------------

    def _check_played_card_is_owned(
        self, step, pre: BluffState, action: int, player: int
    ) -> None:
        """The player places physical cards from their hand onto the pile,
        and those cards may be of any rank(s) actually in their hand. A play
        is only legal if, after subtracting cards already committed earlier
        this same turn, the player still physically holds a card of the
        chosen rank."""
        rank = int(self.env._action_offset_to_rank(pre, jnp.asarray(action)))
        available = float(pre.agent_hands[player, rank] - pre.pending_play_hand[rank])
        if available <= 0:
            self._fail(
                step,
                "phase1/play_owned_card",
                f"player {player} played rank {rank} with only {available} copies left unplayed this turn",
            )

    def _record_play_start(self, pre: BluffState, action: int) -> None:
        """At the first physical card placed this turn, determine whether
        the claimed rank is free-choice or forced, and precompute the value
        it must resolve to, so it can be checked once the play phase
        finishes."""
        keep_rank = bool(pre.has_current_rank) and not bool(pre.rank_choice_pending)
        if keep_rank:
            # Forced: the claimed rank must equal pre.current_rank, which the
            # engine already advanced past the previous round's claim.
            self._turn_is_free = False
            self._expected_claim_rank = int(pre.current_rank)
        else:
            # Free choice: the claimed rank is whatever rank the first card
            # placed happens to be.
            self._turn_is_free = True
            self._expected_claim_rank = int(
                self.env._action_offset_to_rank(pre, jnp.asarray(action))
            )

    def _check_play_phase_finish(self, step, pre: BluffState, post: BluffState) -> None:
        """The player plays exactly `claim_size` physical cards each turn,
        and the resulting claimed rank matches whichever of the two rules
        above (free or forced) applied at the start of this play phase."""
        if int(post.pending_play_count) != int(pre.claim_size):
            self._fail(
                step,
                "phase1/play_count_matches_claim",
                f"pending_play_count={int(post.pending_play_count)} != claim_size={int(pre.claim_size)}",
            )
        played_total = float(jnp.asarray(post.pending_play_hand).sum())
        if not np.isclose(played_total, float(pre.claim_size)):
            self._fail(
                step,
                "phase1/play_count_matches_claim",
                f"pending_play_hand.sum()={played_total} != claim_size={int(pre.claim_size)}",
            )
        if self._expected_claim_rank is None:
            self._fail(
                step,
                "phase1/rank_tracking",
                "play phase finished without ever observing its first play sub-step",
            )
            return
        actual_rank = int(post.claim_rank)
        if actual_rank != self._expected_claim_rank:
            rule = "phase1/free_rank_choice" if self._turn_is_free else "phase1/forced_rank_increment"
            self._fail(
                step,
                rule,
                f"claim_rank={actual_rank} != expected {self._expected_claim_rank} "
                f"(turn_is_free={self._turn_is_free})",
            )
        if self._turn_is_free:
            self.coverage.free_rank_turns += 1
        else:
            self.coverage.forced_rank_turns += 1
        self._first_play_seen = False
        self._expected_claim_rank = None
        self._turn_is_free = None

    # -- phase 2: challenge ----------------------------------------------------

    def _check_challenge_action_space(self, step, pre: BluffState) -> None:
        """Each polled player may only pass or challenge, i.e. exactly the
        two action slots {0, 1} are legal, nothing else."""
        avail = np.asarray(self.env.get_avail_actions(pre))
        expected = np.arange(self.env.action_dim) < 2
        if not np.array_equal(avail, expected):
            self._fail(
                step,
                "phase2/binary_action_space",
                f"challenge-phase avail mask={avail} != expected pass/challenge-only mask={expected}",
            )

    def _check_first_challenger(self, step, pre: BluffState, post: BluffState) -> None:
        """Starting with the player immediately after the one who just
        played, each other player is asked in turn to pass or challenge.
        Checked at the phase1->phase2 transition."""
        claimant = int(pre.current_player_idx)
        expected_first = (claimant + 1) % self.env.num_agents
        if int(post.current_player_idx) != expected_first:
            self._fail(
                step,
                "phase2/first_challenger_order",
                f"first player polled={int(post.current_player_idx)} != next player after "
                f"claimant {claimant} = {expected_first}",
            )
        if int(post.challenge_target_idx) != claimant:
            self._fail(
                step,
                "phase2/challenge_target",
                f"challenge_target_idx={int(post.challenge_target_idx)} != claimant {claimant}",
            )

    def _check_target_never_polled(self, step, pre: BluffState) -> None:
        """Challengers are polled sequentially, in fixed turn order starting
        after the player who just moved. The claimant
        (`challenge_target_idx`) must never be the player being asked to
        pass/challenge on their own claim."""
        if int(pre.current_player_idx) == int(pre.challenge_target_idx):
            self._fail(
                step,
                "phase2/target_excluded",
                "the claimant was asked to challenge their own claim",
            )

    def _check_challenge_resolution(
        self, step, pre: BluffState, post: BluffState, reward: np.ndarray
    ) -> None:
        """The first player to challenge immediately resolves the round: the
        played cards are compared against the claimed rank and count. If
        they all match, the claim was true and the challenger takes the
        entire accumulated pile into their hand; otherwise the claim was
        false and the player who made the claim takes the pile. The pile is
        then emptied, and the round's winner becomes the next round's
        starting player with a free choice of rank."""
        challenger = int(pre.current_player_idx)
        target = int(pre.challenge_target_idx)
        claim_is_true = bool(pre.pending_play_hand[pre.claim_rank] == pre.claim_size)
        loser, winner = (challenger, target) if claim_is_true else (target, challenger)

        if int(post.current_player_idx) != winner or int(post.challenge_target_idx) != winner:
            self._fail(
                step,
                "phase2/resolution_winner_starts_next",
                f"winner={winner} but next current_player_idx={int(post.current_player_idx)}, "
                f"challenge_target_idx={int(post.challenge_target_idx)}",
            )
        if not bool(post.rank_choice_pending):
            self._fail(
                step,
                "phase2/resolution_free_rank_next",
                "winner of a resolved challenge should get a free rank choice next turn",
            )

        pile_before = jnp.asarray(pre.pile_hand)
        expected_loser_hand = jnp.asarray(pre.agent_hands)[loser] + pile_before
        actual_loser_hand = jnp.asarray(post.agent_hands)[loser]
        if not bool(jnp.allclose(expected_loser_hand, actual_loser_hand)):
            self._fail(
                step,
                "phase2/resolution_pile_to_loser",
                f"loser {loser}'s hand after resolution != their hand + the entire pile",
            )
        winner_hand_before = jnp.asarray(pre.agent_hands)[winner]
        winner_hand_after = jnp.asarray(post.agent_hands)[winner]
        if not bool(jnp.allclose(winner_hand_before, winner_hand_after)):
            self._fail(
                step,
                "phase2/resolution_winner_unaffected",
                f"winner {winner}'s hand changed during challenge resolution",
            )
        if float(jnp.asarray(post.pile_hand).sum()) != 0.0 or int(post.pile_size) != 0:
            self._fail(step, "phase2/resolution_empties_pile", "pile not emptied after resolution")
        if float(jnp.asarray(post.pile_claims).sum()) != 0.0:
            self._fail(
                step, "phase2/resolution_empties_pile", "pile_claims not emptied after resolution"
            )

        pile_size_before = float(pre.pile_size)
        won_game_winner = float(reward[winner])
        won_expected = self.env._reward_challenge + (
            self.env._reward_win if int(post.agent_hand_sizes[winner]) == 0 else 0.0
        )
        lost_expected = -pile_size_before * self.env._reward_card + (
            self.env._reward_win if int(post.agent_hand_sizes[loser]) == 0 else 0.0
        )
        if not np.isclose(won_game_winner, won_expected, atol=1e-4):
            self._fail(
                step,
                "reward/challenge_resolution",
                f"winner {winner} reward={won_game_winner} != expected {won_expected} "
                f"(+{self.env._reward_challenge} challenge bonus)",
            )
        if not np.isclose(float(reward[loser]), lost_expected, atol=1e-4):
            self._fail(
                step,
                "reward/challenge_resolution",
                f"loser {loser} reward={float(reward[loser])} != expected {lost_expected} "
                f"(-pile_size {pile_size_before} * reward_card)",
            )
        for agent in range(self.env.num_agents):
            if agent in (winner, loser):
                continue
            if not np.isclose(float(reward[agent]), 0.0, atol=1e-4):
                self._fail(
                    step,
                    "reward/challenge_resolution_bystanders",
                    f"bystander agent {agent} received nonzero reward {float(reward[agent])} "
                    "on a challenge resolution they were not party to",
                )

        if claim_is_true:
            self.coverage.challenge_resolutions_claim_true += 1
        else:
            self.coverage.challenge_resolutions_claim_false += 1

    def _check_no_challenge_resolution(
        self, step, pre: BluffState, post: BluffState, reward: np.ndarray
    ) -> None:
        """If every other player passes, no challenge occurs: the played
        cards remain on the pile (the pile keeps accumulating across
        rounds), and play passes to the next player after the one who
        played, whose claimed rank is forced to increment."""
        target = int(pre.challenge_target_idx)
        expected_next = (target + 1) % self.env.num_agents
        if int(post.current_player_idx) != expected_next or int(post.challenge_target_idx) != expected_next:
            self._fail(
                step,
                "phase2/no_challenge_next_player",
                f"next player after an unchallenged play={int(post.current_player_idx)} != "
                f"expected next-after-claimant {expected_next}",
            )
        pile_before = jnp.asarray(pre.pile_hand)
        pile_after = jnp.asarray(post.pile_hand)
        if not bool(jnp.allclose(pile_before, pile_after)):
            self._fail(
                step,
                "phase2/no_challenge_pile_persists",
                "pile contents changed on an unchallenged (no-challenge) resolution; "
                "the pile must keep accumulating, not empty or mutate",
            )
        expected_rank = (int(pre.claim_rank) + 1) % self.env.num_ranks
        if int(post.current_rank) != expected_rank or bool(post.rank_choice_pending):
            self._fail(
                step,
                "phase2/no_challenge_forced_rank",
                f"next round's forced rank={int(post.current_rank)} "
                f"(rank_choice_pending={bool(post.rank_choice_pending)}) != expected forced "
                f"rank {expected_rank} with rank_choice_pending=False",
            )
        expected_target_reward = float(pre.pending_play_hand.sum()) * self.env._reward_card + (
            self.env._reward_win if int(post.agent_hand_sizes[target]) == 0 else 0.0
        )
        if not np.isclose(float(reward[target]), expected_target_reward, atol=1e-4):
            self._fail(
                step,
                "reward/no_challenge",
                f"target {target} reward={float(reward[target])} != expected "
                f"{expected_target_reward} (claim_size * reward_card [+ win bonus])",
            )
        for agent in range(self.env.num_agents):
            if agent == target:
                continue
            if not np.isclose(float(reward[agent]), 0.0, atol=1e-4):
                self._fail(
                    step,
                    "reward/no_challenge_bystanders",
                    f"bystander agent {agent} received nonzero reward {float(reward[agent])} "
                    "on an unchallenged play they were not the claimant of",
                )
        self.coverage.no_challenge_resolutions += 1

    # -- win condition ---------------------------------------------------------

    def _check_win_condition(self, step, post: BluffState, done: bool) -> None:
        """A winner is only ever declared at the moment a round returns to
        phase 0, i.e. after the challenge phase concludes. The episode may
        end on a win (as opposed to horizon truncation) only at that phase-0
        boundary; a player sitting at 0 cards mid-round -- e.g. right after
        playing their last card, while still in phase 2 awaiting a
        challenge that could hand them back the pile -- must not end the
        episode there."""
        game_winner = np.asarray(post.game_winner)
        recomputed_winner = np.asarray(post.agent_hand_sizes) == 0
        if not np.array_equal(game_winner, recomputed_winner):
            self._fail(
                step,
                "win/matches_hand_sizes",
                f"game_winner={game_winner} != (agent_hand_sizes==0)={recomputed_winner}",
            )
        # A horizon truncation can legitimately coincide with a non-phase-0
        # state where a bystander's hand happens to be empty mid-round; only
        # an actual win requires landing on a phase-0 boundary.
        game_over = bool(game_winner.any()) and int(post.phase) == 0
        truncated = int(post.timestep) >= self.env.horizon
        expected_done = game_over or truncated
        if bool(done) != expected_done:
            self._fail(
                step,
                "win/done_matches_primitives",
                f"done={bool(done)} != expected {expected_done} "
                f"(game_over={game_over}, truncated={truncated}, phase={int(post.phase)}, "
                f"timestep={int(post.timestep)}, horizon={self.env.horizon})",
            )
        if game_over:
            self.coverage.wins += 1
        if truncated and not game_over:
            self.coverage.truncations += 1

    def _check_reward_only_at_round_boundary(
        self, step, post: BluffState, reward: np.ndarray
    ) -> None:
        """Rewards -- the per-card reward, the challenge-resolution
        reward/penalty, and the terminal win bonus -- fire exactly once, at
        the conclusion of a round. Every round-concluding transition (a
        resolved challenge or an unchallenged play) lands in phase 0, so any
        transition that does not land in phase 0 -- a claim, an intermediate
        card placement, or a challenge-phase pass that doesn't complete the
        poll -- must leave every agent's reward at exactly zero."""
        if int(post.phase) != 0:
            nonzero = np.asarray(reward) != 0.0
            if nonzero.any():
                self._fail(
                    step,
                    "reward/only_at_round_boundary",
                    f"nonzero reward {np.asarray(reward)} on a transition that did not "
                    f"conclude a round (post.phase={int(post.phase)} != 0)",
                )

    # -- action legality (applies to every phase) -------------------------------

    def _check_action_legal(self, step, avail: jnp.ndarray, action: int) -> None:
        """Whatever policy is driving the rollout (random or a trained
        checkpoint), the action it hands to `env.step` must be one
        `get_avail_actions` marked legal for the current phase."""
        if not bool(avail[action]):
            self._fail(
                step,
                "action/within_avail_mask",
                f"action {action} taken despite avail mask={np.asarray(avail)} marking it illegal",
            )

    def _check_avail_nonempty(self, step, avail: jnp.ndarray) -> None:
        """No reachable non-terminal state should ever leave the acting
        player with zero legal actions (that would be a deadlock in the
        game's turn cycle)."""
        if not bool(jnp.asarray(avail).any()):
            self._fail(step, "action/no_deadlock", "no legal actions available for acting player")

    # -- top-level dispatch ------------------------------------------------------

    def validate_pre_step(self, step: int, pre: BluffState) -> None:
        self._check_hand_size_bookkeeping(step, pre)
        self._check_card_conservation(step, pre)
        self._check_no_negative_counts(step, pre)
        self._check_pile_claims_consistent(step, pre)
        self._check_current_player_in_range(step, pre)

    def validate_transition(
        self,
        step: int,
        pre: BluffState,
        action: int,
        avail: jnp.ndarray,
        post: BluffState,
        reward: np.ndarray,
        done: bool,
    ) -> None:
        self.coverage.steps += 1
        self._check_avail_nonempty(step, avail)
        self._check_action_legal(step, avail, action)
        self._check_reward_only_at_round_boundary(step, post, np.asarray(reward))

        phase = int(pre.phase)
        if phase == 0:
            self._check_claim_size_bounds(step, pre, post)
        elif phase == 1:
            player = int(pre.current_player_idx)
            self._check_played_card_is_owned(step, pre, action, player)
            if int(pre.pending_play_count) == 0:
                self._record_play_start(pre, action)
            if int(post.phase) == 2:
                self._check_play_phase_finish(step, pre, post)
        elif phase == 2:
            self._check_challenge_action_space(step, pre)
            self._check_target_never_polled(step, pre)
            challenged = int(action) == 0
            if challenged:
                self._check_challenge_resolution(step, pre, post, np.asarray(reward))
            elif int(post.phase) == 0:
                self._check_no_challenge_resolution(step, pre, post, np.asarray(reward))
            # else: a plain pass, moving the poll to the next player in order
            # (no separate invariant beyond what _check_target_never_polled
            # and the eventual resolution/no-challenge checks already cover).
        # the phase1 -> phase2 transition (first challenger set) is checked
        # independently of which sub-branch of phase 1/2 handling above ran:
        if phase == 1 and int(post.phase) == 2:
            self._check_first_challenger(step, pre, post)

        self._check_win_condition(step, post, done)

    def assert_no_violations(self, extra_context: str = "") -> None:
        if self.violations:
            preview = "\n".join(str(v) for v in self.violations[:20])
            more = f"\n... and {len(self.violations) - 20} more" if len(self.violations) > 20 else ""
            raise AssertionError(
                f"{len(self.violations)} rule violation(s) found{extra_context}:\n{preview}{more}"
            )


# ---------------------------------------------------------------------------
# Rollout driver.
# ---------------------------------------------------------------------------


def rollout_and_validate(
    env: Bluff,
    kind: str,
    network: Optional[Any],
    params: Optional[Any],
    num_episodes: int,
    seed: int = DEFAULT_SEED,
) -> RuleChecker:
    """Runs `num_episodes` full episodes with the given agent (all seats
    controlled by the same policy, i.e. self-play), validating every
    transition against `RuleChecker`. Returns the checker so callers can
    inspect `.violations` and `.coverage`.

    Uses `env.step_env` directly rather than the public `env.step`, which
    auto-resets on `done` and would substitute in an unrelated fresh
    episode's state right at the transition (a win or a horizon truncation)
    that most needs checking. `step_env` returns the true resulting state;
    episode boundaries are handled explicitly below instead.
    """
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
            checker.validate_pre_step(step, state)

            avail = env.get_avail_actions(state)
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            action = sample_action(kind, network, params, obs, avail, act_rng)
            next_state, next_obs, reward, absorbing, done, info = env.step_env(
                step_rng, state, action
            )

            checker.validate_transition(
                step, state, int(action), avail, next_state, reward, bool(done)
            )
            state, obs = next_state, next_obs

    return checker


# ---------------------------------------------------------------------------
# Hand-crafted edge-case scenarios.
#
# Everything above validates statistically: random or checkpoint play is
# rolled out for many episodes and every transition is checked, but rare
# situations -- a bluff caught on the player's last card, several
# unchallenged rounds stacking up before a challenge lands, the claimed rank
# wrapping King -> Ace -- may take a very large number of episodes to occur
# by chance, or may never occur under a converged, near-greedy checkpoint
# policy. The tests below instead construct a `BluffState` directly
# (bypassing `env.reset()`'s randomness) to land exactly on one of these
# situations every run, then assert the exact expected outcome.
#
# These states are synthetic, not dealt: card totals are not required to sum
# to `env.deck_size` (that global conservation property is already checked
# exhaustively by `RuleChecker` over real, reset()-dealt rollouts above).
# ---------------------------------------------------------------------------


def _hand_vector(env: Bluff, counts: dict[int, int]) -> jnp.ndarray:
    v = np.zeros(env.num_ranks, dtype=np.float32)
    for rank, count in counts.items():
        v[rank] = count
    return jnp.asarray(v)


def _make_state(
    env: Bluff,
    *,
    phase: int,
    current_player_idx: int,
    agent_hand_counts: list[dict[int, int]],
    challenge_target_idx: int = 0,
    claim_size: int = 0,
    claim_rank: int = 0,
    pending_play_counts: Optional[dict[int, int]] = None,
    pending_play_count: int = 0,
    has_current_rank: bool = False,
    current_rank: int = 0,
    rank_choice_pending: bool = True,
    pile_counts: Optional[dict[int, int]] = None,
    pile_claim_counts: Optional[dict[int, int]] = None,
    round_start_player_idx: int = 0,
    start_player_idx: int = 0,
    timestep: int = 0,
) -> BluffState:
    """Builds a fully-specified BluffState for one scripted scenario. Every
    field `BluffState` declares must be supplied (it has no defaults); this
    fills in the handful that matter for a given scenario and neutral/empty
    values for the rest."""
    agent_hands = jnp.stack([_hand_vector(env, c) for c in agent_hand_counts])
    agent_hand_sizes = agent_hands.sum(axis=1).astype(jnp.int32)
    pile_hand = _hand_vector(env, pile_counts or {})
    pile_claims = _hand_vector(env, pile_claim_counts or {})
    return BluffState(
        pile_hand=pile_hand,
        pile_claims=pile_claims,
        pile_size=pile_hand.sum().astype(jnp.int32),
        agent_hands=agent_hands,
        agent_hand_sizes=agent_hand_sizes,
        phase=jnp.array(phase, dtype=jnp.int32),
        current_player_idx=jnp.array(current_player_idx, dtype=jnp.int32),
        start_player_idx=jnp.array(start_player_idx, dtype=jnp.int32),
        round_start_player_idx=jnp.array(round_start_player_idx, dtype=jnp.int32),
        challenge_target_idx=jnp.array(challenge_target_idx, dtype=jnp.int32),
        claim_size=jnp.array(claim_size, dtype=jnp.int32),
        claim_rank=jnp.array(claim_rank, dtype=jnp.int32),
        pending_play_hand=_hand_vector(env, pending_play_counts or {}),
        pending_play_count=jnp.array(pending_play_count, dtype=jnp.int32),
        has_current_rank=jnp.array(has_current_rank),
        current_rank=jnp.array(current_rank, dtype=jnp.int32),
        rank_choice_pending=jnp.array(rank_choice_pending),
        challenge_status=jnp.array(3, dtype=jnp.int32),
        challenge_hand=env._empty_hand(),
        absorbing=jnp.zeros((env.num_agents,), dtype=bool),
        done=jnp.array(False),
        game_winner=jnp.zeros((env.num_agents,), dtype=bool),
        timestep=timestep,
    )


def _edge_case_env() -> Bluff:
    return make("bluff", num_agents=3, horizon=100)


def test_edge_caught_bluffing_on_final_card_does_not_win() -> None:
    """If someone challenges and the claim was a lie, the player who emptied
    their hand must take the entire pile back -- so they no longer have 0
    cards, and the game continues. This is the single most important nuance
    of the win condition, and one a policy is unlikely to stumble into on
    its own, since it requires a hand reduced to the played cards, a false
    claim, and a challenger who calls it, all at once.

    Setup: it's a forced-rank round (current_rank=5, following a prior
    no-challenge round), and the active player's only remaining card is rank
    7 -- a lie relative to the forced claim of rank 5."""
    env = _edge_case_env()
    state = _make_state(
        env,
        phase=1,
        current_player_idx=0,
        agent_hand_counts=[{7: 1}, {0: 2, 1: 2}, {2: 2, 3: 2}],
        challenge_target_idx=0,
        claim_size=1,
        has_current_rank=True,
        current_rank=5,
        rank_choice_pending=False,
        pile_counts={9: 2},
        pile_claim_counts={9: 2},
        timestep=10,
    )
    rng = jax.random.PRNGKey(0)

    # Play the only (lying) card: abs_rank = (current_rank + action) % num_ranks == 7.
    action = (7 - 5) % env.num_ranks
    avail = env.get_avail_actions(state)
    assert bool(avail[action]), "the player's only card (rank 7) should be a legal play"
    played_state, _, _, _, _, _ = env.step_env(rng, state, action)

    assert int(played_state.phase) == 2, "claim_size=1 reached -> straight to challenge phase"
    assert int(played_state.agent_hand_sizes[0]) == 0, "player 0 played their only card"
    assert int(played_state.pile_size) == 3, "2 preexisting + 1 just-played card"
    assert int(played_state.claim_rank) == 5, "forced rank kept, not the lying card's true rank 7"
    assert int(played_state.current_player_idx) == 1, "first challenger polled is next after claimant"

    # Player 1 challenges the (false) claim.
    resolved_state, _, _, _, done, _ = env.step_env(rng, played_state, 0)
    assert int(resolved_state.agent_hand_sizes[0]) == 3, (
        "the claim was a lie -> player 0 must take the ENTIRE pile (3 cards) back into their hand"
    )
    assert not bool(resolved_state.game_winner.any()), "no one has 0 cards anymore -> no winner"
    assert not bool(done), "the game must continue, not end, when a final-card bluff is caught"
    assert int(resolved_state.current_player_idx) == 1, "challenger (winner) starts the next round"
    assert bool(resolved_state.rank_choice_pending), "winner of a resolved challenge gets a free rank choice"
    assert int(resolved_state.pile_size) == 0, "pile is emptied on any resolved challenge"


def test_edge_true_claim_challenged_on_final_card_still_wins() -> None:
    """If someone challenges and the claim turns out to be true, that player
    ends the round with 0 cards and the game ends in their favor. Same
    scenario as above, but the physical card genuinely matches the forced
    rank, so a mistaken challenge should not prevent the win -- it should
    instead penalize the challenger."""
    env = _edge_case_env()
    state = _make_state(
        env,
        phase=1,
        current_player_idx=0,
        agent_hand_counts=[{5: 1}, {0: 2, 1: 2}, {2: 2, 3: 2}],
        challenge_target_idx=0,
        claim_size=1,
        has_current_rank=True,
        current_rank=5,
        rank_choice_pending=False,
        pile_counts={9: 2},
        pile_claim_counts={9: 2},
        timestep=10,
    )
    rng = jax.random.PRNGKey(0)

    action = (5 - 5) % env.num_ranks  # play the genuine rank-5 card
    played_state = env.step_env(rng, state, action)[0]
    assert int(played_state.agent_hand_sizes[0]) == 0

    resolved_state, _, reward, _, done, _ = env.step_env(rng, played_state, 0)
    assert int(resolved_state.agent_hand_sizes[0]) == 0, "winner's (already-empty) hand is untouched"
    assert int(resolved_state.agent_hand_sizes[1]) == 4 + 3, (
        "wrong challenger takes the entire 3-card pile on top of their existing 4 cards"
    )
    assert list(map(bool, resolved_state.game_winner)) == [True, False, False]
    assert int(resolved_state.phase) == 0
    assert bool(done), "phase-0 boundary reached with a player at 0 cards -> game ends"
    assert np.isclose(float(reward[0]), env._reward_challenge + env._reward_win), (
        "winner gets the fixed challenge bonus plus the terminal win bonus"
    )
    assert np.isclose(float(reward[1]), -3.0 * env._reward_card), (
        "loser is penalized proportionally to the pile size they must take"
    )


def test_edge_unchallenged_final_card_wins_with_single_reward_win_payout() -> None:
    """If nobody challenges the play that empties a player's hand, that
    player ends the round with 0 cards and the game ends in their favor.
    Also checks that the terminal win bonus is paid out exactly once, at
    this phase-0 boundary transition, not on either of the two intervening
    challenge-phase passes leading up to it."""
    env = _edge_case_env()
    state = _make_state(
        env,
        phase=2,
        current_player_idx=1,
        agent_hand_counts=[{}, {0: 3, 1: 3}, {2: 3, 3: 3}],
        challenge_target_idx=0,
        claim_size=1,
        claim_rank=3,
        pending_play_counts={3: 1},
        pending_play_count=1,
        has_current_rank=True,
        current_rank=3,
        rank_choice_pending=False,
        pile_counts={3: 1},
        pile_claim_counts={3: 1},
        timestep=20,
    )
    rng = jax.random.PRNGKey(0)

    after_p1_pass = env.step_env(rng, state, 1)[0]
    assert int(after_p1_pass.current_player_idx) == 2, "poll moves to the next player, not straight to a win"
    assert not bool(after_p1_pass.done)
    assert float(env.step_env(rng, state, 1)[2][0]) == 0.0, "no reward yet -- poll hasn't concluded"

    final_state, _, reward, _, done, _ = env.step_env(rng, after_p1_pass, 1)
    assert int(final_state.phase) == 0, "every player declined -> unchallenged resolution"
    assert list(map(bool, final_state.game_winner)) == [True, False, False]
    assert bool(done)
    assert np.isclose(float(reward[0]), 1 * env._reward_card + env._reward_win), (
        "target gets claim_size * reward_card plus the win bonus, exactly once"
    )
    assert np.isclose(float(reward[1]), 0.0) and np.isclose(float(reward[2]), 0.0)


def test_edge_rank_wraparound_king_to_ace() -> None:
    """The forced claimed rank increments one rank above the previous
    round's claimed rank, mod 13, so King wraps back to Ace. This pins the
    wraparound down directly: the previous round's claim was King (rank
    index 12), so this round's forced rank must be exactly Ace (rank
    index 0)."""
    env = _edge_case_env()
    assert env.num_ranks == 13
    state = _make_state(
        env,
        phase=0,
        current_player_idx=1,
        agent_hand_counts=[{0: 2}, {6: 1}, {1: 2, 2: 2}],
        challenge_target_idx=1,
        claim_rank=12,
        has_current_rank=True,
        current_rank=0,  # already advanced to (12 + 1) % 13 by the prior no-challenge resolution
        rank_choice_pending=False,
        timestep=30,
    )
    rng = jax.random.PRNGKey(0)

    claimed_state = env.step_env(rng, state, 0)[0]  # only 1 card in hand -> claim_size forced to 1
    assert int(claimed_state.claim_size) == 1

    # Play the only card (rank 6); the forced declared rank must still be
    # Ace (0), regardless of the physical card's true rank.
    action = (6 - int(claimed_state.current_rank)) % env.num_ranks
    final_state = env.step_env(rng, claimed_state, action)[0]
    assert int(final_state.phase) == 2
    assert int(final_state.claim_rank) == 0, "King -> Ace wraparound (12 + 1) % 13 must equal 0"


def test_edge_multi_round_pile_only_last_round_checked_on_challenge() -> None:
    """A challenge compares only this round's played cards against the
    claimed rank and count -- it must inspect only `pending_play_hand` (this
    round's physical cards), never the cumulative `pile_hand`, even though
    the entire accumulated pile is what actually changes hands. Three rounds
    of genuine, unchallenged claims (ranks 1, 2, 3) are stacked on the pile
    first, then a 4th round's claim (rank 4) is a lie -- a scenario a random
    rollout would only rarely stumble into, since it needs several
    consecutive no-challenge rounds before the eventual challenge."""
    env = _edge_case_env()
    state = _make_state(
        env,
        phase=2,
        current_player_idx=1,
        agent_hand_counts=[{7: 5}, {0: 5}, {1: 5}],
        challenge_target_idx=0,
        claim_size=2,
        claim_rank=4,
        pending_play_counts={7: 2},  # this round's true cards: rank 7 (a lie vs claimed rank 4)
        pending_play_count=2,
        has_current_rank=True,
        current_rank=4,
        rank_choice_pending=False,
        pile_counts={1: 2, 2: 2, 3: 2, 7: 2},  # 3 earlier genuine rounds + this round's 2 cards
        pile_claim_counts={1: 2, 2: 2, 3: 2, 4: 2},
        timestep=40,
    )
    assert int(state.pile_size) == 8
    rng = jax.random.PRNGKey(0)

    resolved_state = env.step_env(rng, state, 0)[0]  # player 1 challenges
    assert int(resolved_state.agent_hand_sizes[0]) == 5 + 8, (
        "claimant (loser) takes the ENTIRE 8-card accumulated pile, not just this round's 2 cards"
    )
    assert int(resolved_state.agent_hand_sizes[1]) == 5, "challenger (winner) is unaffected"
    assert int(resolved_state.pile_size) == 0


def test_edge_claim_size_cap_at_exactly_four_cards() -> None:
    """Claim size is capped both at 4 and at the player's current hand size.
    Boundary case: a hand of exactly 4 should permit a claim of exactly 4 --
    the two caps coincide rather than one clipping the other."""
    env = _edge_case_env()
    state = _make_state(
        env, phase=0, current_player_idx=0, agent_hand_counts=[{0: 2, 1: 2}, {}, {}], timestep=5
    )
    assert int(state.agent_hand_sizes[0]) == 4
    avail = np.asarray(env.get_avail_actions(state))
    assert avail[:4].all() and not avail[4:].any(), "all 4 claim-size actions legal, nothing beyond"

    rng = jax.random.PRNGKey(0)
    next_state = env.step_env(rng, state, 3)[0]  # action index 3 -> claim_size 4
    assert int(next_state.claim_size) == 4


def test_edge_claim_size_forced_to_one_with_single_card() -> None:
    """Same cap rule at the opposite boundary -- a player down to their last
    card must have claim_size forced to exactly 1 (only action index 0
    legal)."""
    env = _edge_case_env()
    state = _make_state(
        env, phase=0, current_player_idx=0, agent_hand_counts=[{5: 1}, {}, {}], timestep=5
    )
    assert int(state.agent_hand_sizes[0]) == 1
    avail = np.asarray(env.get_avail_actions(state))
    assert avail[0] and not avail[1:].any(), "only claim_size=1 is legal with a single card in hand"

    rng = jax.random.PRNGKey(0)
    next_state = env.step_env(rng, state, 0)[0]
    assert int(next_state.claim_size) == 1


# ---------------------------------------------------------------------------
# pytest entry points.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_agents", [2, 3, 4, 5, 6])
def test_random_agent_rollouts_conform_to_rules(num_agents: int) -> None:
    """Uniform-random legal-action rollouts must never violate any rule of
    the game, for every supported table size. Random play exercises rare
    branches -- free vs. forced rank, both challenge outcomes, no-challenge
    continuations -- far more uniformly than a converged policy would."""
    env = make("bluff", num_agents=num_agents, horizon=DEFAULT_HORIZON)
    checker = rollout_and_validate(
        env, "random", None, None, RANDOM_EPISODES_PER_CONFIG, seed=DEFAULT_SEED + num_agents
    )
    checker.assert_no_violations(extra_context=f" for num_agents={num_agents}")

    cov = checker.coverage
    assert cov.steps > 0
    assert cov.free_rank_turns > 0, "no free-rank-choice turns were ever exercised"
    assert cov.forced_rank_turns > 0, "no forced-rank-increment turns were ever exercised"
    assert cov.challenge_resolutions_claim_true > 0, "no true-claim challenge resolutions seen"
    assert cov.challenge_resolutions_claim_false > 0, "no false-claim (bluff caught) resolutions seen"
    assert cov.no_challenge_resolutions > 0, "no unchallenged plays were ever exercised"


def _discovered_checkpoint_params():
    """One representative checkpoint per (algorithm, network-kind) combo --
    e.g. a single PPO-NFSP 'br' (actor_critic) checkpoint, a single PQN-NFSP
    'br' (q_network) checkpoint, etc. -- rather than every seed/fraction on
    disk, so the default pytest run stays fast. Use the standalone CLI (see
    module docstring) to validate a specific checkpoint exhaustively over
    many more episodes."""
    seen: set[tuple[str, str]] = set()
    params = []
    for path, kind in discover_checkpoints():
        algorithm = "ppo_nfsp" if "ppo" in os.path.basename(os.path.dirname(path)) else "pqn_nfsp"
        key = (algorithm, kind)
        if key in seen:
            continue
        seen.add(key)
        params.append(pytest.param((path, kind, algorithm), id=os.path.relpath(path, CHECKPOINT_ROOT)))
    return params


@pytest.mark.parametrize("checkpoint_info", _discovered_checkpoint_params())
def test_checkpoint_agent_rollouts_conform_to_rules(checkpoint_info) -> None:
    """Same rule-conformance check as above, but driven by a trained
    checkpoint's policy instead of random play: a learned agent may adopt
    strategies very different from uniform random, but must still only ever
    produce rule-legal states, actions, and transitions. Skips cleanly if no
    checkpoints are present."""
    checkpoint_path, network_kind, algorithm = checkpoint_info
    env_kwargs, fc_dim_size = _training_env_kwargs(algorithm)
    env = make("bluff", **env_kwargs)

    _, sample_obs = env.reset(jax.random.PRNGKey(0))
    network = build_network(network_kind, action_dim=env.action_dim, hidden_dim=fc_dim_size)
    params = load_checkpoint_params(network, sample_obs, checkpoint_path)

    checker = rollout_and_validate(
        env, network_kind, network, params, CHECKPOINT_EPISODES, seed=DEFAULT_SEED
    )
    checker.assert_no_violations(extra_context=f" for checkpoint={checkpoint_path}")
    assert checker.coverage.steps > 0


def test_checkpoint_discovery_runs_without_error() -> None:
    """Sanity check on the discovery/inference helpers themselves: every
    discovered checkpoint must be classifiable into a known network kind."""
    for path, kind in discover_checkpoints():
        assert kind in ("actor", "actor_critic", "q_network"), (path, kind)


if not discover_checkpoints():
    test_checkpoint_agent_rollouts_conform_to_rules = pytest.mark.skip(
        reason="no trained checkpoints found under bluffjax/examples/bluff/checkpoints/"
    )(test_checkpoint_agent_rollouts_conform_to_rules)


# ---------------------------------------------------------------------------
# Standalone CLI.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .msgpack checkpoint.")
    parser.add_argument(
        "--network-type",
        type=str,
        default=None,
        choices=["actor", "actor_critic", "q_network"],
        help="Override network kind inference for --checkpoint.",
    )
    parser.add_argument("--random", action="store_true", help="Use a uniform-random agent instead.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not args.random and args.checkpoint is None:
        parser.error("pass --checkpoint PATH or --random")

    env = make("bluff", num_agents=args.num_agents, horizon=args.horizon)
    _, sample_obs = env.reset(jax.random.PRNGKey(0))

    if args.random:
        kind, network, params = "random", None, None
    else:
        kind = args.network_type or infer_network_kind(args.checkpoint)
        network = build_network(kind, action_dim=env.action_dim, hidden_dim=args.hidden_dim)
        params = load_checkpoint_params(network, sample_obs, args.checkpoint)

    print(f"Validating {args.episodes} episode(s) of '{kind}' agent, num_agents={args.num_agents}...")
    checker = rollout_and_validate(env, kind, network, params, args.episodes, seed=args.seed)

    print(f"\nSteps checked: {checker.coverage.steps}")
    print(f"Coverage: {checker.coverage}")
    if checker.violations:
        print(f"\n{len(checker.violations)} VIOLATION(S) FOUND:")
        by_rule = Counter(v.rule for v in checker.violations)
        for rule, count in by_rule.most_common():
            print(f"  {rule}: {count}")
        print("\nFirst 20:")
        for v in checker.violations[:20]:
            print(f"  {v}")
        raise SystemExit(1)
    print("\nNo rule violations found.")


if __name__ == "__main__":
    main()
