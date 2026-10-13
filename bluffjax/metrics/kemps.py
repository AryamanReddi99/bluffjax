"""Kemps metrics (bluffjax-metrics.md, Kemps).

The rules they rely on (bluffjax/environments/kemps/kemps.py): n = 2k players
form k teams of two partners sitting opposite each other (team t is the seats
{t, t + k}; seat i's partner is seat i + k mod n). There are no turns: at
every step every player acts, with one action game_action * comm_dim +
signal; the signal (0..comm_dim-1) is public and seen by everybody at the
next step. The game actions are the swaps, NOOP, KEMPS (claims that the
caller's partner holds four of a kind) and STOP KEMPS d for d = 1..k-1
(claims that a player of the team of seat caller + d, i.e. seat caller + d or
caller + d + k, holds four of a kind; never the caller's own team, and with
two teams always the other one). Anybody may declare at any step. A
declaration ends the hand at once and is judged on the hands at the start of
the step (that step's swaps don't happen). When several players declare in
the same step only one declaration counts: the first declarer's in the step's
player order, a uniformly random permutation that the env draws from the key
of env.step and doesn't return (see step_order). A hand also ends without a
call when everybody plays NOOP and the stock can't refill the centre (a real
deal) or when it is cut off at the horizon; the env reports both as
absorbing, like a declaration (parallel_nfsp_common.kemps_horizon_cutoff).
Here a hand ends with a call exactly when somebody declared in its last step.

"Holds four of a kind" is always about the hands at the start of a step: a
player holds four of a kind at step t if its hand at the start of step t has
four cards of one rank, and a team holds four of a kind if either partner
does. A call is correct as the env judges it: KEMPS if the caller's partner
holds four of a kind, STOP KEMPS if a player of the accused team does.

Metrics, each credited per player (counts of shape (num_agents,), and
(num_agents, comm_dim, 2) for the signal table):

kemps_precision: share of KEMPS calls that are correct. Every KEMPS action is
    a call, also one that loses the race to a simultaneous declaration (it is
    the caller's decision and is judged the same way). Credited to the
    caller; n_kemps_calls counts the calls.
stop_precision: share of STOP KEMPS calls that are correct, every STOP KEMPS
    action counted as for KEMPS. Credited to the caller; n_stop_calls.
interception_rate: over the (hand, team T) such that T holds four of a kind
    at the start of some step of the hand, the share in which the hand ends
    with a correct STOP KEMPS against T (the declaration that counts accuses
    T and is correct). Credited, at the hand's last step, to every player of
    the other teams; n_opponent_quad_hands counts the (hand, T, player
    outside T) triples.
quad_call_rate: over the same (hand, team T), the share in which the hand
    ends with T's own correct KEMPS (the declaration that counts is a correct
    KEMPS by a player of T). Credited to both players of T;
    n_own_quad_hands counts the (hand, T, player of T) triples.
signal_mi: mutual information, in bits, between a player's signal and
    whether it holds four of a kind at that step, from the joint counts of
    (signal, holds four of a kind) over the credited players' decisions
    (every step, the last step of a hand included). Credited to the
    signalling player. n_decisions counts the decisions and
    n_signal_{s}_quad_{q} the cells of the table (q = 1: holds four of a
    kind), so that the MI of any sum of tables can be recomputed with
    mutual_information_bits. It is the plug-in estimate: with a signal
    independent of the hand its expectation is about
    (comm_dim - 1) / (2 ln 2 n_decisions) bits, not 0.
kemps_latency: over the hands that end with a correct KEMPS (the declaration
    that counts), the number of steps from the first step of the hand at
    which the caller's partner held four of a kind to the call (0: called at
    the first step it held them). Credited to the caller; n_quads_called
    counts the hands.
A hand cut off at the horizon or ended by a real deal ends without a call:
its (hand, team) pairs count only in the denominators of interception_rate
and quad_call_rate.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bluffjax.metrics.base import Counts, GameMetrics, credit, rate
from bluffjax.utils.typing import (
    Array,
    BoolArray,
    FloatArray,
    IntArray,
    PRNGKeyArray,
)


def step_order(env, rng: PRNGKeyArray) -> IntArray:
    """(num_agents,) the player order of env.step(rng, state, action), first
    to act first: who swaps first and whose declaration counts.

    Kemps doesn't return it: ParallelEnv.step splits rng into
    (rng_step, rng_reset) and Kemps.step_env draws the order as
    jax.random.permutation(rng_step, num_agents). (For env.step_env(rng, ...)
    without the auto-reset the order is jax.random.permutation(rng, n).)
    Callers pass it to Metrics.step as info["order"].
    """
    rng_step, _ = jax.random.split(rng)
    return jax.random.permutation(rng_step, env.num_agents)


def holds_four(env, state) -> BoolArray:
    """(num_agents,) whether each player's hand has four cards of one rank."""
    return (state.agent_hand_counts >= 4).any(axis=-1)


def mutual_information_bits(table: FloatArray) -> FloatArray:
    """Plug-in mutual information, in bits, of the joint counts table[x, y];
    NaN for an empty table."""
    table = jnp.asarray(table, jnp.float32)
    total = table.sum()
    p = table / jnp.where(total > 0, total, 1.0)
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    positive = p > 0
    ratio = jnp.where(positive, p, 1.0) / jnp.where(positive, px * py, 1.0)
    mi = jnp.where(positive, p * jnp.log2(ratio), 0.0).sum()
    return jnp.where(total > 0, mi, jnp.nan)


class KempsCarry(NamedTuple):
    # (num_teams,) the team held four of a kind at the start of some step of
    # this hand so far
    team_had_four: BoolArray
    # (num_agents,) first step of this hand at which the player held four of
    # a kind, -1 if it hasn't yet
    first_four: IntArray


class Metrics(GameMetrics):
    rate_names = (
        "kemps_precision",
        "stop_precision",
        "interception_rate",
        "quad_call_rate",
        "signal_mi",
        "kemps_latency",
    )
    needs_probs = False

    def init_carry(self) -> KempsCarry:
        return KempsCarry(
            team_had_four=jnp.zeros(self.env.num_teams, dtype=bool),
            first_four=jnp.full(self.num_agents, -1, dtype=jnp.int32),
        )

    def step(
        self,
        carry: KempsCarry,
        state,
        action: Array,
        next_state,
        reward: FloatArray,
        done: BoolArray,
        info: dict,
        probs: FloatArray | None = None,
    ) -> tuple[KempsCarry, Counts]:
        """info must hold "order", the step's player order (step_order)."""
        if "order" not in info:
            raise KeyError(
                "Kemps metrics need info['order'], the player order of the step, "
                "which env.step doesn't return: pass "
                "dict(info, order=bluffjax.metrics.kemps.step_order(env, rng)) "
                "with rng the key given to env.step"
            )
        env = self.env
        n, k = self.num_agents, env.num_teams
        seats = jnp.arange(n)
        team = env._team_id(seats)
        partner = env._partner_idx(seats)
        game_action = action // env.comm_dim
        signal = action % env.comm_dim

        has_four = holds_four(env, state)
        team_has_four = has_four[:k] | has_four[k:]

        # Every declaration, judged on the hands at the start of the step.
        is_kemps = game_action == env.action_kemps
        is_stop = game_action > env.action_kemps
        accused = env._team_id(seats + game_action - env.action_kemps)  # STOP only
        kemps_right = is_kemps & has_four[partner]
        stop_right = is_stop & team_has_four[accused]

        # The declaration that counts and ends the hand: the first declarer
        # in the step's order, as in Kemps._declaration_rewards.
        order = info["order"]
        is_declare = is_kemps | is_stop
        caller = order[jnp.argmax(is_declare[order])]
        called = done & is_declare.any()
        ends_kemps_right = called & kemps_right[caller]
        ends_stop_right = called & stop_right[caller]

        team_had_four = carry.team_had_four | team_has_four
        first_four = jnp.where(
            (carry.first_four < 0) & has_four, state.timestep, carry.first_four
        )
        latency = state.timestep - first_four[partner[caller]]

        own_had_four = team_had_four[team]
        signal_one_hot = jax.nn.one_hot(signal, env.comm_dim, dtype=jnp.float32)
        quad_one_hot = jax.nn.one_hot(has_four.astype(jnp.int32), 2, dtype=jnp.float32)
        signal_quad = signal_one_hot[:, :, None] * quad_one_hot[:, None, :]
        counts = {
            "kemps_calls": is_kemps.astype(jnp.float32),
            "kemps_right": kemps_right.astype(jnp.float32),
            "stop_calls": is_stop.astype(jnp.float32),
            "stop_right": stop_right.astype(jnp.float32),
            # teams other than the player's that held four of a kind this hand
            "opponent_quad_hands": jnp.where(
                done, team_had_four.sum() - own_had_four, 0
            ).astype(jnp.float32),
            "intercepted": (ends_stop_right & (team != accused[caller])).astype(
                jnp.float32
            ),
            "own_quad_hands": (done & own_had_four).astype(jnp.float32),
            "quad_called": (ends_kemps_right & (team == team[caller])).astype(
                jnp.float32
            ),
            "signal_quad": signal_quad,
            "latency_sum": credit(n, caller, jnp.where(ends_kemps_right, latency, 0)),
            "quads_called": credit(n, caller, ends_kemps_right),
        }
        carry = KempsCarry(
            team_had_four=jnp.where(done, False, team_had_four),
            first_four=jnp.where(done, -1, first_four),
        )
        return carry, counts

    def summarize(self, totals: Counts) -> dict[str, FloatArray]:
        table = totals["signal_quad"]
        out = {
            "kemps_precision": rate(totals["kemps_right"], totals["kemps_calls"]),
            "n_kemps_calls": totals["kemps_calls"],
            "stop_precision": rate(totals["stop_right"], totals["stop_calls"]),
            "n_stop_calls": totals["stop_calls"],
            "interception_rate": rate(
                totals["intercepted"], totals["opponent_quad_hands"]
            ),
            "n_opponent_quad_hands": totals["opponent_quad_hands"],
            "quad_call_rate": rate(totals["quad_called"], totals["own_quad_hands"]),
            "n_own_quad_hands": totals["own_quad_hands"],
            "signal_mi": mutual_information_bits(table),
            "n_decisions": table.sum(),
            "kemps_latency": rate(totals["latency_sum"], totals["quads_called"]),
            "n_quads_called": totals["quads_called"],
        }
        for s in range(self.env.comm_dim):
            for q in range(2):
                out[f"n_signal_{s}_quad_{q}"] = table[s, q]
        return out
