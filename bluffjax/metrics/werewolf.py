"""Werewolf metrics (bluffjax-metrics.md).

The env (bluffjax/environments/werewolf/werewolf.py) has werewolves, one
doctor, one seer and villagers; the doctor, the seer and the villagers are the
villager team (the humans). A game cycles night -> accuse -> vote:
- night: the doctor protects a living player (itself included), the seer
  checks another living player, each werewolf picks a living human; the
  werewolves' target is their most picked player (ties broken at random by
  the env) and dies unless the doctor protected it. The night is resolved on
  the step of its last living actor (always a werewolf).
- accuse: each living player accuses another living player or passes (the
  no-op). Accusations are separate from votes, and every player sees who has
  been accused so far that day.
- vote: each living player votes for another living player (no pass); the
  most voted player (ties broken at random) is eliminated on the step of the
  last vote.
The humans win when no werewolf is alive, the werewolves when they are at
least as many as the living humans.

Everything is read from the state before the step, the action, and (for who
died) the next state, so nothing is carried between steps. When a night or a
vote ends the game the next state is already the next game's; what happened
follows from the winner: a vote won by the humans eliminated the last living
werewolf, one won by the werewolves eliminated a human, a night that ends
the game killed a human (so no save). Only at a horizon cut (no winner) is it
unknown; such a night or vote is left out of the rates that need its outcome
(with the default horizon of 200 no game gets there: every day eliminates a
player).

Rates, with the events counted and who they are credited to:
    villager_vote_accuracy   votes for a werewolf / votes, over the votes of
                             villager-team players, credited to the voter.
                             n_villager_votes.
    elimination_accuracy     day eliminations of a werewolf / day
                             eliminations, credited to every villager-team
                             player alive at the vote (each elimination
                             counts once per such player).
                             n_villager_eliminations.
    accusation_accuracy      accusations of a werewolf / accusations (passes
                             are not accusations), over the accusations of
                             villager-team players, credited to the accuser.
                             n_villager_accusations.
    seer_info_use            over (day, werewolf) pairs where the werewolf is
                             alive at the start of the day and the seer has
                             identified it on an earlier night (the night just
                             before the day included), the share of pairs in
                             which that werewolf is accused (by any player)
                             or voted out that day; credited to the seer
                             (alive or not: the definition doesn't require
                             it). n_seer_pairs.
    doctor_save_rate         nights on which the doctor protects the
                             werewolves' target / nights on which the doctor
                             is alive (and so acts), credited to the doctor.
                             The target is saved exactly when nobody dies that
                             night. n_doctor_nights.
    werewolf_votes_received  votes of villager-team players received by living
                             werewolves / (day, living werewolf) pairs,
                             credited to the werewolf. n_werewolf_days.
    villager_votes_received  the same for living villager-team players, as the
                             reference (a player can't vote for itself, so a
                             villager has one possible villager voter fewer).
                             n_villager_days.
A rate with no events is NaN.
"""

import jax.numpy as jnp

from bluffjax.environments.werewolf.werewolf import (
    DOCTOR,
    PHASE_ACCUSE,
    PHASE_NIGHT,
    PHASE_VOTE,
    SEER,
    WEREWOLF,
    WerewolfState,
)
from bluffjax.metrics.base import Counts, GameMetrics, rate
from bluffjax.utils.typing import Array, BoolArray, FloatArray


def credit(num_agents: int, player: Array, value: Array = 1.0) -> FloatArray:
    """(num_agents,) array with `value` at index `player` and 0 elsewhere, as
    bluffjax.metrics.base.credit (player is always a valid index here), but as
    a one-hot product instead of a scatter: XLA fuses it with the rest of the
    step, which takes a few percent off the training scripts' run time."""
    return (jnp.arange(num_agents) == player) * jnp.asarray(value, jnp.float32)


class Metrics(GameMetrics):
    rate_names = (
        "villager_vote_accuracy",
        "elimination_accuracy",
        "accusation_accuracy",
        "seer_info_use",
        "doctor_save_rate",
        "werewolf_votes_received",
        "villager_votes_received",
    )

    def step(
        self,
        carry: tuple,
        state: WerewolfState,
        action: Array,
        next_state: WerewolfState,
        reward: FloatArray,
        done: BoolArray,
        info: dict,
        probs: FloatArray | None = None,
    ) -> tuple[tuple, Counts]:
        n = self.num_agents
        idx = jnp.arange(n)
        action = jnp.asarray(action, jnp.int32)
        cp = state.current_player_idx
        # Relative action -> absolute target (-1 for the accuse-phase pass)
        target = jnp.where(action < n, (cp + action) % n, -1)
        werewolf = state.roles == WEREWOLF
        human = ~werewolf
        alive = state.alive
        at_werewolf = (target >= 0) & werewolf[jnp.maximum(target, 0)]
        by_human = human[cp]

        # Outcome of a night or vote resolved at this step (see the docstring)
        winner = info["game_winner"].astype(bool)  # players whose team won
        team_won = winner.any()
        humans_won = (winner & human).any()
        known = ~done | team_won
        died = alive & ~next_state.alive  # only meaningful if not done

        # Votes and accusations, one per step
        vote = (state.phase == PHASE_VOTE) & by_human
        accusation = (state.phase == PHASE_ACCUSE) & by_human & (target >= 0)

        # Day: resolved on the last vote
        day = (state.phase == PHASE_VOTE) & (
            state.phase_progress + 1 >= alive.sum()
        )
        votes = state.votes.at[cp].set(target)  # today's votes, this one included
        human_voter = alive & human
        votes_received = (
            (human_voter[:, None] & (votes[:, None] == idx[None, :]))
            .sum(axis=0)
            .astype(jnp.float32)
        )
        # (n,): the werewolf eliminated today, if any
        eliminated_werewolf = jnp.where(
            done, humans_won & alive & werewolf, died & werewolf
        )
        day_known = day & known
        accused = (state.accusations[:, None] == idx[None, :]).any(axis=0)
        seer = jnp.argmax(state.roles == SEER)
        identified = alive & werewolf & (state.seer_results == 1.0)
        seer_hits = identified & (accused | eliminated_werewolf)

        # Night: resolved on the step of its last living actor
        night = (state.phase == PHASE_NIGHT) & (
            self.env._next_night_slot(
                state.night_subphase, state.alive, state.night_order
            )
            >= self.env.num_night_actors
        )
        doctor = jnp.argmax(state.roles == DOCTOR)
        doctor_night = night & alive[doctor] & known
        saved = ~done & ~died.any()

        counts = {
            "villager_votes": credit(n, cp, vote),
            "villager_votes_for_werewolf": credit(n, cp, vote & at_werewolf),
            "villager_eliminations": (human_voter & day_known).astype(jnp.float32),
            "villager_eliminations_of_werewolf": (
                human_voter & day_known & eliminated_werewolf.any()
            ).astype(jnp.float32),
            "villager_accusations": credit(n, cp, accusation),
            "villager_accusations_of_werewolf": credit(
                n, cp, accusation & at_werewolf
            ),
            "seer_pairs": credit(n, seer, day_known * identified.sum()),
            "seer_hits": credit(n, seer, day_known * seer_hits.sum()),
            "doctor_nights": credit(n, doctor, doctor_night),
            "doctor_saves": credit(n, doctor, doctor_night & saved),
            "werewolf_days": (day & alive & werewolf).astype(jnp.float32),
            "werewolf_votes_received": jnp.where(
                day & alive & werewolf, votes_received, 0.0
            ),
            "villager_days": (day & alive & human).astype(jnp.float32),
            "villager_votes_received": jnp.where(
                day & alive & human, votes_received, 0.0
            ),
        }
        return carry, counts

    def summarize(self, totals: Counts) -> dict[str, FloatArray]:
        return {
            "villager_vote_accuracy": rate(
                totals["villager_votes_for_werewolf"], totals["villager_votes"]
            ),
            "n_villager_votes": totals["villager_votes"],
            "elimination_accuracy": rate(
                totals["villager_eliminations_of_werewolf"],
                totals["villager_eliminations"],
            ),
            "n_villager_eliminations": totals["villager_eliminations"],
            "accusation_accuracy": rate(
                totals["villager_accusations_of_werewolf"],
                totals["villager_accusations"],
            ),
            "n_villager_accusations": totals["villager_accusations"],
            "seer_info_use": rate(totals["seer_hits"], totals["seer_pairs"]),
            "n_seer_pairs": totals["seer_pairs"],
            "doctor_save_rate": rate(totals["doctor_saves"], totals["doctor_nights"]),
            "n_doctor_nights": totals["doctor_nights"],
            "werewolf_votes_received": rate(
                totals["werewolf_votes_received"], totals["werewolf_days"]
            ),
            "n_werewolf_days": totals["werewolf_days"],
            "villager_votes_received": rate(
                totals["villager_votes_received"], totals["villager_days"]
            ),
            "n_villager_days": totals["villager_days"],
        }
