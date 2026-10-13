"""Bluff metrics (bluffjax-metrics.md).

A claim ("play") takes several steps of the claimant (bluffjax/environments/
bluff/bluff.py): naming the rank on a free lead (phase NAME_RANK), choosing the
claim size (CLAIM), picking the cards one at a time (PLAY), then one step per
other player in turn order, starting after the claimant, until one of them
challenges or all have passed (CHALLENGE). A play is counted once, on its last
card pick, when the cards go to the pile. It is a lie when not all of its
cards are of the claimed rank, i.e. when one of its picks had a rank offset
other than 0 (the cards the training scripts' play_action_lie_frac counts, per
pick); this is the env's own test (pending_play_hand[claim_rank] ==
claim_size is a true claim). Everything is read from the state before the
step and the action, so nothing is carried between steps and auto-reset
next states don't matter.

Rates, with the events counted and who they are credited to:
    lie_rate             lies / plays, credited to the claimant (on its last
                         card pick). n_plays.
    lie_success          lies that nobody challenges / lies whose challenge
                         round finished (everyone passed, or someone
                         challenged), credited to the liar, on the step that
                         resolves the claim. A claim whose challenge round is
                         cut off at the horizon has no outcome and is left
                         out. n_resolved_lies.
    challenge_precision  challenges of a lie / challenges, credited to the
                         challenger. n_challenges.
    challenge_recall     challenges of a lie / challenge decisions about a lie,
                         credited to the player deciding: a player had the
                         chance to challenge an opponent's lie when it was
                         asked (its CHALLENGE step); players after the first
                         challenger are never asked. Heads-up, this is the
                         share of the opponent's lies (with a finished
                         challenge round) that the player catches.
                         n_lie_decisions.
    final_play_lie_rate  lies / plays among the plays that would empty the
                         claimant's hand (claim size = hand size; the play
                         wins the game unless it is caught), credited to the
                         claimant. Compare with lie_rate. n_final_plays.
A rate with no events is NaN.
"""

import jax.numpy as jnp

from bluffjax.environments.bluff.bluff import CHALLENGE, PLAY, BluffState
from bluffjax.metrics.base import Counts, GameMetrics, credit, rate
from bluffjax.utils.typing import Array, BoolArray, FloatArray

CHALLENGE_ACTION = 0  # in phase CHALLENGE; any other action passes


class Metrics(GameMetrics):
    rate_names = (
        "lie_rate",
        "lie_success",
        "challenge_precision",
        "challenge_recall",
        "final_play_lie_rate",
    )

    def step(
        self,
        carry: tuple,
        state: BluffState,
        action: Array,
        next_state: BluffState,
        reward: FloatArray,
        done: BoolArray,
        info: dict,
        probs: FloatArray | None = None,
    ) -> tuple[tuple, Counts]:
        n = self.num_agents
        action = jnp.asarray(action, jnp.int32)
        player = state.current_player_idx

        # Last card pick of a play: the claim goes to the pile.
        card_rank = self.env._offset_to_rank(state, action)
        claimed_cards = state.pending_play_hand[state.claim_rank] + (
            card_rank == state.claim_rank
        )
        play = (state.phase == PLAY) & (state.pending_play_count + 1 >= state.claim_size)
        play_lie = play & (claimed_cards < state.claim_size)
        final = play & (state.agent_hand_sizes[player] == state.claim_size)

        # A challenge decision about the pending claim (the cards are on the
        # pile and in pending_play_hand).
        asked = state.phase == CHALLENGE
        claimant = state.challenge_target_idx
        claim_lie = state.pending_play_hand[state.claim_rank] != state.claim_size
        challenges = asked & (action == CHALLENGE_ACTION)
        last_pass = (
            asked & (action != CHALLENGE_ACTION) & ((player + 1) % n == claimant)
        )
        resolved_lie = (challenges | last_pass) & claim_lie

        counts = {
            "plays": credit(n, player, play),
            "lies": credit(n, player, play_lie),
            "final_plays": credit(n, player, final),
            "final_lies": credit(n, player, final & play_lie),
            "resolved_lies": credit(n, claimant, resolved_lie),
            "unchallenged_lies": credit(n, claimant, last_pass & claim_lie),
            "challenges": credit(n, player, challenges),
            "caught_lies": credit(n, player, challenges & claim_lie),
            "lie_decisions": credit(n, player, asked & claim_lie),
        }
        return carry, counts

    def summarize(self, totals: Counts) -> dict[str, FloatArray]:
        return {
            "lie_rate": rate(totals["lies"], totals["plays"]),
            "n_plays": totals["plays"],
            "lie_success": rate(totals["unchallenged_lies"], totals["resolved_lies"]),
            "n_resolved_lies": totals["resolved_lies"],
            "challenge_precision": rate(totals["caught_lies"], totals["challenges"]),
            "n_challenges": totals["challenges"],
            "challenge_recall": rate(totals["caught_lies"], totals["lie_decisions"]),
            "n_lie_decisions": totals["lie_decisions"],
            "final_play_lie_rate": rate(totals["final_lies"], totals["final_plays"]),
            "n_final_plays": totals["final_plays"],
        }
