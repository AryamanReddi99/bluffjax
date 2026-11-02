"""
ReBeL (Recursive Belief-based Learning) for Leduc Hold'em.

Implements ReBeL from Brown, Bakhtin, Lerer and Gong, "Combining Deep
Reinforcement Learning and Search for Imperfect-Information Games" (NeurIPS
2020, Algorithm 2: Linear CFR-D), in the same way as kuhn/kuhn_rebel.py and
with the betting-round subgames and chance-node leaves of
holdem_rebel/ (the game is OpenSpiel's leduc_poker, as
bluffjax.utils.game_utils.leduc_exploitability):

- Cards 0..5 are J J Q Q K K (rank = card // 2). Each player antes 1; two
  betting rounds with raises of 2 and 4, at most 2 raises per round, fold
  only when facing a bet; player 0 acts first in both rounds. After round 1
  one public card is dealt; a pair with it wins, else the higher rank.
- A public belief state (PBS) is a public state plus each player's range:
  the normalized probability that they reach it with each private card. The
  deal distribution at a PBS is proportional to ranges[0, c0] *
  ranges[1, c1] for distinct cards that aren't the public card.
- Subgames run to the end of the current betting round (15 public nodes,
  RoundTree). Round-1 subgames end at folds and at the 5 round-end nodes;
  a round-end node is a chance node valued by the value network at the 6
  PBSs of the start of round 2 that follow it (one per public card),
  weighted by card removal (chance_leaf_values). Round-2 subgames end at
  folds and showdowns only, so they are solved without the network and their
  root values are exact.
- The value network maps a start-of-round-2 PBS (pot, public card, both
  ranges) to the value of every infostate in it: values[i, c] is player i's
  expected return (chips, divided by VALUE_SCALE) holding card c, with the
  opponent's card drawn from their range without c and the public card.
- Subgames are solved with T iterations of alternating-update Linear CFR-D
  (as kuhn_rebel): in each half-step the leaf PBSs follow from the current
  policy profile. The training target is the linearly averaged root value.
- Self play (one hand per game and update): the initial PBS is the same in
  every hand, so its subgame is solved once per update (the network is
  fixed within an update), as holdem_rebel does. Each game samples a
  half-step, iteration t's with probability proportional to t + 1, draws a
  deal from the root PBS and plays that profile to a leaf of round 1, with
  one random player taking a uniformly random action with probability
  explore_prob at each of their decisions. Beliefs follow the profile. At a
  round-end node the public card is dealt (a deal is drawn from the PBS,
  then the card from the 4 left), the round-2 subgame at the new PBS is
  solved and its root values are the training example. The example of the
  initial PBS is not used: the network is never queried there.
- At test time ReBeL plays the same way without exploration: it solves the
  round-1 subgame, samples iteration t, plays pi^t and passes the beliefs of
  pi^t on to round 2, where it solves the round-2 subgame and samples an
  iteration again. test_time_policy computes the expected policy of that
  random procedure exactly, by enumerating the sampled iterations and
  weighting by reach (as kuhn_rebel.test_time_policy), and its exploitability
  is logged.

Deviations from the paper: no policy network or warm start (optional in the
paper); subgames are whole betting rounds (the paper's Liar's Dice subgames
have depth 2, its poker subgames are betting rounds). An action that a
player's policy never takes with any card in their range leaves that
player's range unchanged.

Evaluation (exact, at the evaluation points: before training and every
eval_interval updates): the exploitability (leduc_exploitability, the
OpenSpiel definition (BR_0 + BR_1) / 2) of the test-time policy with
eval_cfr_iters CFR iterations per subgame, its exact expected return against
uniformly random play averaged over both seats and its Leduc policy metrics
(bluffjax.metrics.leduc_holdem, game/<name>). The test-time policy is
computed outside the training jit, at eval_matmul_precision.

Logging: main() creates the run directory {save_dir}/{job_type}_{timestamp}/
(run_utils.make_run_dir, with config.yaml). Every evaluation writes one
record per seed to {run_dir}/results.jsonl (with seed_index and wall_time)
and to the seed's wandb run: updates_completed, num_examples (examples added
to the replay buffer so far, the sample count), num_subgames (subgames
solved so far, round 1 and round 2), replay_size, the mean value_loss and
hands_to_round2 (games that reached round 2) in the updates since the last
evaluation (left out before training), exploitability, value_vs_random and
the policy metrics. After training, {run_dir}/curves.json holds the config,
the training time and every seed's curves.

Checkpoints (save_checkpoints), for every seed i = 0..num_seeds-1:
    {run_dir}/half/value_net_{i}.msgpack  after half_update_steps(num_update_steps)
    {run_dir}/full/value_net_{i}.msgpack  after all num_update_steps updates
Each holds the flax params of ValueNetwork(hidden_dim=value_hidden_dim),
mapping encode_pbs (PBS_DIM floats) to values (2, 6) / VALUE_SCALE. ReBeL's
policy is test_time_policy(network_value_fn(network, params), cfr_iters).
Restore one with run_utils.load_params(path, network.init(rng,
jnp.zeros(PBS_DIM))).
"""

import datetime
import json
import os
import time
from typing import Any, Callable, NamedTuple

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import hydra
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax
import wandb

from bluffjax.examples.run_utils import (
    ResultsLog,
    half_update_steps,
    make_run_dir,
    save_seed_params,
)
from bluffjax.metrics import prefixed
from bluffjax.metrics.leduc_holdem import policy_metrics
from bluffjax.utils.game_utils import leduc_exploitability as leduc
from bluffjax.utils.paths import REPO_ROOT, register_resolvers
from bluffjax.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray

# =============================================================================
# Game: cards, betting rounds
# =============================================================================

NUM_CARDS = 6
NUM_ACTIONS = 3  # leduc_exploitability's ids: 0 fold, 1 call/check, 2 raise
NUM_PLAYERS = 2
ACTION_FOLD = leduc.ACTION_FOLD
ACTION_CALL = leduc.ACTION_CALL
ACTION_RAISE = leduc.ACTION_RAISE
RANK = np.arange(NUM_CARDS) // 2
ANTE = float(leduc.K_ANTE)
MAX_RAISES = leduc.K_MAX_RAISES

# Node kinds of a betting round
DECISION, FOLD, ROUND_END = 0, 1, 2

# Largest contribution of a player (ante + 2 raises of 2 + 2 raises of 4);
# network outputs and targets are values in chips divided by this.
VALUE_SCALE = ANTE + MAX_RAISES * leduc.K_FIRST_RAISE_AMOUNT + MAX_RAISES * (
    leduc.K_SECOND_RAISE_AMOUNT
)

# Reach masses at or below this count as zero.
PROB_EPS = 1e-12


class RoundTree:
    """Public tree of one betting round (numpy, static), nodes breadth-first.

    histories[n]: actions of the round before node n; actor[n]: player to act
    (at decisions); kind[n]: DECISION, FOLD or ROUND_END; extra[n]: chips each
    player put in during the round before node n; folder[n]: the player who
    folded (FOLD nodes, else -1); child[n, a]: node after action a (-1 if
    illegal); parent / parent_action of every node but the root (-1).
    """

    def __init__(self, raise_amount: float):
        histories: list[tuple[int, ...]] = [()]
        actor, kind, extra, raises, folder = [0], [DECISION], [(0.0, 0.0)], [0], [-1]
        edges = []
        i = 0
        while i < len(histories):
            n = i
            i += 1
            if kind[n] != DECISION:
                continue
            p = actor[n]
            e = extra[n]
            facing = e[1 - p] > e[p]
            for a in range(NUM_ACTIONS):
                if a == ACTION_FOLD and not facing:
                    continue
                if a == ACTION_RAISE and raises[n] >= MAX_RAISES:
                    continue
                new = list(e)
                k = DECISION
                if a == ACTION_FOLD:
                    k = FOLD
                elif a == ACTION_CALL:
                    new[p] = e[1 - p]
                    # A check opens the round; any later call closes it.
                    k = ROUND_END if histories[n] else DECISION
                else:
                    new[p] = e[1 - p] + raise_amount
                histories.append(histories[n] + (a,))
                actor.append(1 - p)
                kind.append(k)
                extra.append(tuple(new))
                raises.append(raises[n] + (a == ACTION_RAISE))
                folder.append(p if a == ACTION_FOLD else -1)
                edges.append((n, a, len(histories) - 1))
        self.histories = tuple(histories)
        self.index = {h: k for k, h in enumerate(histories)}
        self.num_nodes = len(histories)
        self.actor = np.array(actor, np.int32)
        self.kind = np.array(kind, np.int32)
        self.extra = np.array(extra, np.float32)
        self.folder = np.array(folder, np.int32)
        self.child = -np.ones((self.num_nodes, NUM_ACTIONS), np.int32)
        self.parent = -np.ones(self.num_nodes, np.int32)
        self.parent_action = -np.ones(self.num_nodes, np.int32)
        for n, a, k in edges:
            self.child[n, a] = k
            self.parent[k] = n
            self.parent_action[k] = a
        self.legal = self.child >= 0
        self.decisions = tuple(int(n) for n in np.nonzero(self.kind == DECISION)[0])
        self.round_ends = tuple(int(n) for n in np.nonzero(self.kind == ROUND_END)[0])
        self.max_depth = max(len(h) for h in histories)


ROUND1 = RoundTree(float(leduc.K_FIRST_RAISE_AMOUNT))
ROUND2 = RoundTree(float(leduc.K_SECOND_RAISE_AMOUNT))
NUM_NODES = ROUND1.num_nodes  # both rounds have the same shape
ROUND_ENDS = np.array(ROUND1.round_ends, np.int32)  # round-1 end nodes
NUM_ROUND_ENDS = len(ROUND_ENDS)
# Chips each player has in at the start of round 2 after each round-1 end
# node, and the index of that pot among POT_LEVELS (network input).
ROUND2_START = ANTE + ROUND1.extra[ROUND_ENDS, 0]
POT_LEVELS = np.unique(ROUND2_START)
POT_INDEX = np.searchsorted(POT_LEVELS, ROUND2_START).astype(np.int32)

# Card removal: COMPAT1[c, o] = 1 if a player holding c can face o in round 1;
# COMPAT2[b, c, o] = 1 if c, o and the public card b are distinct; NOT_BOARD[b, c]
# = 1 if c != b. SHOWDOWN[b, c, o]: sign of the showdown for c against o.
COMPAT1 = 1.0 - np.eye(NUM_CARDS, dtype=np.float32)
COMPAT2 = np.zeros((NUM_CARDS, NUM_CARDS, NUM_CARDS), np.float32)
SHOWDOWN = np.zeros((NUM_CARDS, NUM_CARDS, NUM_CARDS), np.float32)
for _b in range(NUM_CARDS):
    for _c in range(NUM_CARDS):
        for _o in range(NUM_CARDS):
            if len({_b, _c, _o}) == 3:
                COMPAT2[_b, _c, _o] = 1.0
                _sc = 10 + RANK[_c] if RANK[_c] == RANK[_b] else RANK[_c]
                _so = 10 + RANK[_o] if RANK[_o] == RANK[_b] else RANK[_o]
                SHOWDOWN[_b, _c, _o] = np.sign(_sc - _so)
NOT_BOARD = 1.0 - np.eye(NUM_CARDS, dtype=np.float32)


def opponent_mass(ranges: FloatArray, compat: FloatArray) -> FloatArray:
    """mass[..., i, c]: opponent range on the cards compatible with c."""
    return jnp.einsum("...io,co->...ic", ranges[..., ::-1, :], compat)


def normalize_ranges(reach: FloatArray, fallback: FloatArray) -> FloatArray:
    """Bayes' rule over each player's card: normalizes reach probabilities
    (..., 2, 6) into ranges. A player whose reach is zero for every card
    (after an action the policy never takes) gets the fallback range."""
    total = jnp.sum(reach, axis=-1, keepdims=True)
    reached = total > PROB_EPS
    return jnp.where(reached, reach / jnp.where(reached, total, 1.0), fallback)


def initial_ranges() -> FloatArray:
    return jnp.full((NUM_PLAYERS, NUM_CARDS), 1.0 / NUM_CARDS)


def board_ranges(ranges: FloatArray, board: IntArray) -> FloatArray:
    """Ranges after the public card `board` is dealt (card removal)."""
    mask = jnp.asarray(NOT_BOARD)[board]
    return normalize_ranges(ranges * mask, mask / jnp.sum(mask))


# =============================================================================
# Value network
# =============================================================================

PBS_DIM = len(POT_LEVELS) + NUM_CARDS + NUM_PLAYERS * NUM_CARDS


def encode_pbs(round_end: IntArray, board: IntArray, ranges: FloatArray) -> FloatArray:
    """Value-network input for the start-of-round-2 PBS after round-1 end node
    round_end (index into ROUND_ENDS) and public card `board`: one-hot pot,
    one-hot public card and both ranges."""
    pot = jax.nn.one_hot(jnp.asarray(POT_INDEX)[round_end], len(POT_LEVELS))
    card = jax.nn.one_hot(board, NUM_CARDS)
    flat = ranges.reshape(ranges.shape[:-2] + (NUM_PLAYERS * NUM_CARDS,))
    return jnp.concatenate([pot, card, flat], axis=-1)


class ValueNetwork(nn.Module):
    """MLP (LayerNorm, GeLU) mapping a PBS encoding to values (..., 2, 6) / VALUE_SCALE."""

    hidden_dim: int = 256
    num_layers: int = 2

    @nn.compact
    def __call__(self, x: FloatArray) -> FloatArray:
        for _ in range(self.num_layers):
            x = nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)
        x = nn.Dense(
            NUM_PLAYERS * NUM_CARDS,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        return x.reshape(x.shape[:-1] + (NUM_PLAYERS, NUM_CARDS))


# value_fn(round_end (K,), board (K,), ranges (K, 2, 6)) -> values (K, 2, 6) in chips
ValueFn = Callable[[IntArray, IntArray, FloatArray], FloatArray]


def network_value_fn(network: nn.Module, params) -> ValueFn:
    def value_fn(round_end, board, ranges):
        return network.apply(params, encode_pbs(round_end, board, ranges)) * VALUE_SCALE

    return value_fn


# =============================================================================
# Linear CFR-D on a betting round
# =============================================================================


class SubgameSolution(NamedTuple):
    """
    profiles: (T, 2, 15, 6, 3) policy profile of player i's half-step on
        iteration t, (pi_0^t, pi_1^t) for i = 0 and (pi_0^(t+1), pi_1^t) for
        i = 1. None if not kept.
    policy_sum: (15, 6, 3) sum over t of (t + 1) * own reach * pi^t (the
        unnormalized Linear CFR average policy).
    values: (2, 6) linearly averaged root infostate values in chips, 0 where
        masked.
    value_mask: (2, 6) root infostates whose value is defined.
    """

    profiles: FloatArray
    policy_sum: FloatArray
    values: FloatArray
    value_mask: BoolArray


def regret_matching(tree: RoundTree, regrets: FloatArray) -> FloatArray:
    """Policy proportional to positive regret over legal actions, uniform if none."""
    legal = jnp.asarray(tree.legal)[:, None, :]
    positive = jnp.where(legal, jnp.maximum(regrets, 0.0), 0.0)
    total = jnp.sum(positive, axis=-1, keepdims=True)
    uniform = legal / jnp.maximum(jnp.sum(legal, axis=-1, keepdims=True), 1)
    return jnp.where(total > 0.0, positive / jnp.where(total > 0.0, total, 1.0), uniform)


def compute_reaches(tree: RoundTree, policy: FloatArray, root_reach: FloatArray) -> FloatArray:
    """reach[n, i, c]: root_reach[i, c] times player i's probability of
    playing to node n holding c."""
    rows = [root_reach]
    for n in range(1, tree.num_nodes):
        parent = tree.parent[n]
        rows.append(
            rows[parent]
            .at[tree.actor[parent]]
            .multiply(policy[parent, :, tree.parent_action[n]])
        )
    return jnp.stack(rows)


def counterfactual_values(
    tree: RoundTree,
    policy: FloatArray,
    reach: FloatArray,
    player: int,
    start: FloatArray,
    compat: FloatArray,
    showdown: FloatArray | None,
    leaf_cfv: FloatArray | None,
) -> list:
    """values[n][c]: counterfactual value of `player` holding c at node n, the
    sum over compatible opponent cards of the opponent's reach times the
    player's expected return. start: chips each player had in at the start
    of the round. Round ends are showdowns (`showdown` (6, 6)) or, in round 1,
    leaves with counterfactual values leaf_cfv[n, player]."""
    values: list = [None] * tree.num_nodes
    for n in reversed(range(tree.num_nodes)):
        opp = reach[n, 1 - player]
        if tree.kind[n] == FOLD:
            contrib = start + tree.extra[n]
            folder = tree.folder[n]
            pay = -contrib[player] if folder == player else contrib[folder]
            values[n] = pay * (compat @ opp)
        elif tree.kind[n] == ROUND_END:
            if leaf_cfv is None:
                pot = start[0] + tree.extra[n][0]  # equal contributions
                values[n] = pot * ((compat * showdown) @ opp)
            else:
                values[n] = leaf_cfv[n, player]
        else:
            children = [
                (a, tree.child[n, a]) for a in range(NUM_ACTIONS) if tree.legal[n, a]
            ]
            if tree.actor[n] == player:
                values[n] = sum(policy[n, :, a] * values[k] for a, k in children)
            else:
                values[n] = sum(values[k] for _, k in children)
    return values


def solve_round(
    tree: RoundTree,
    ranges: FloatArray,
    start: FloatArray,
    compat: FloatArray,
    showdown: FloatArray | None,
    leaf_fn: Callable[[FloatArray], FloatArray] | None,
    num_iterations: int,
    keep_profiles: bool = True,
    value_target: str = "iterates",
) -> SubgameSolution:
    """Alternating-update Linear CFR-D on the betting-round subgame rooted at
    a PBS with `ranges`. On iteration t player 0 and then player 1 update
    their regrets with weight t + 1, each from the current profile. leaf_fn
    maps the reach (15, 2, 6) of the current profile to the counterfactual
    values (15, 2, 6) of the round-end leaves (round 1); without it round ends
    are showdowns (round 2).

    value_target: "iterates" averages each player's root values of their
    half-steps with linear weights (the paper's v(beta_r) and the released
    code's root_values_means). With alternating updates each player's value
    is measured just after the opponent adapted to them, which biases both
    players' values down by O(CFR error) (at 512 iterations about 0.03 chips
    per player in a round-2 subgame). "average_policy" instead takes the root
    values of the final linear-average profile (exact for that profile in
    leafless subgames)."""
    actor = jnp.asarray(tree.actor)
    own_range = jnp.ones_like(ranges)
    root_mass = opponent_mass(ranges, compat)
    value_mask = root_mass > PROB_EPS

    def half_step(player: int, regrets, policy_sum, value_sum, profile, weight):
        reach = compute_reaches(tree, profile, ranges)
        leaf_cfv = None if leaf_fn is None else leaf_fn(reach)
        values = counterfactual_values(
            tree, profile, reach, player, start, compat, showdown, leaf_cfv
        )
        instant = jnp.zeros_like(regrets)
        for n in tree.decisions:
            if tree.actor[n] != player:
                continue
            for a in range(NUM_ACTIONS):
                if tree.legal[n, a]:
                    instant = instant.at[n, :, a].set(values[tree.child[n, a]] - values[n])
        regrets = regrets + weight * instant
        own_reach = compute_reaches(tree, profile, own_range)
        mine = jnp.asarray((tree.kind == DECISION) & (tree.actor == player))
        policy_sum = policy_sum + weight * jnp.where(
            mine[:, None, None], own_reach[:, player, :, None] * profile, 0.0
        )
        root_values = values[0] / jnp.where(value_mask[player], root_mass[player], 1.0)
        value_sum = value_sum.at[player].add(weight * root_values)
        return regrets, policy_sum, value_sum

    def iteration(carry, t):
        regrets, policy_sum, value_sum = carry
        weight = (t + 1).astype(jnp.float32)
        profile_p0 = regret_matching(tree, regrets)
        regrets, policy_sum, value_sum = half_step(
            0, regrets, policy_sum, value_sum, profile_p0, weight
        )
        # Player 0's new policy, player 1's current one.
        profile_p1 = jnp.where(
            (actor == 0)[:, None, None], regret_matching(tree, regrets), profile_p0
        )
        regrets, policy_sum, value_sum = half_step(
            1, regrets, policy_sum, value_sum, profile_p1, weight
        )
        profiles = jnp.stack([profile_p0, profile_p1]) if keep_profiles else None
        return (regrets, policy_sum, value_sum), profiles

    zeros = jnp.zeros((tree.num_nodes, NUM_CARDS, NUM_ACTIONS))
    init = (zeros, zeros, jnp.zeros((NUM_PLAYERS, NUM_CARDS)))
    (_, policy_sum, value_sum), profiles = lax.scan(
        iteration, init, jnp.arange(num_iterations)
    )
    total_weight = num_iterations * (num_iterations + 1) / 2.0
    if value_target == "iterates":
        root_values = value_sum / total_weight
    elif value_target == "average_policy":
        average = average_policy(tree, policy_sum)
        reach = compute_reaches(tree, average, ranges)
        leaf_cfv = None if leaf_fn is None else leaf_fn(reach)
        root_values = jnp.stack(
            [
                counterfactual_values(
                    tree, average, reach, p, start, compat, showdown, leaf_cfv
                )[0]
                / jnp.where(value_mask[p], root_mass[p], 1.0)
                for p in range(NUM_PLAYERS)
            ]
        )
    else:
        raise ValueError(f"unknown value_target {value_target}")
    values = jnp.where(value_mask, root_values, 0.0)
    return SubgameSolution(profiles, policy_sum, values, value_mask)


def average_policy(tree: RoundTree, policy_sum: FloatArray) -> FloatArray:
    """Normalized average policy, uniform over legal actions where unreached."""
    total = jnp.sum(policy_sum, axis=-1, keepdims=True)
    uniform = regret_matching(tree, jnp.zeros_like(policy_sum))
    return jnp.where(total > 0.0, policy_sum / jnp.where(total > 0.0, total, 1.0), uniform)


def chance_leaf_values(value_fn: ValueFn, reach: FloatArray) -> FloatArray:
    """Counterfactual values (15, 2, 6) of the round-1 end nodes (zero
    elsewhere) given the reach (15, 2, 6) of a round-1 profile.

    The public card b is uniform among the 4 cards the players don't hold, so
    cfv(c) = 1/4 sum_{b != c} m_b(c) v_b(c), with v_b the network's values at
    the start-of-round-2 PBS (ranges restricted to b's compatible cards) and
    m_b(c) the opponent's reach on the cards compatible with c and b.
    """
    ends = jnp.asarray(ROUND_ENDS)
    end_reach = reach[ends]  # (Z, 2, 6)
    boards = jnp.arange(NUM_CARDS)
    restricted = end_reach[:, None] * jnp.asarray(NOT_BOARD)[None, :, None, :]  # (Z, B, 2, 6)
    ranges = jax.vmap(jax.vmap(board_ranges, in_axes=(None, 0)), in_axes=(0, None))(
        end_reach, boards
    )
    z_index = jnp.broadcast_to(jnp.arange(NUM_ROUND_ENDS)[:, None], (NUM_ROUND_ENDS, NUM_CARDS))
    b_index = jnp.broadcast_to(boards[None, :], (NUM_ROUND_ENDS, NUM_CARDS))
    values = value_fn(
        z_index.reshape(-1), b_index.reshape(-1), ranges.reshape(-1, NUM_PLAYERS, NUM_CARDS)
    ).reshape(NUM_ROUND_ENDS, NUM_CARDS, NUM_PLAYERS, NUM_CARDS)
    mass = jnp.einsum("zbio,bco->zbic", restricted[:, :, ::-1], jnp.asarray(COMPAT2))
    own_ok = jnp.asarray(NOT_BOARD)[None, :, None, :]
    cfv = 0.25 * jnp.sum(own_ok * mass * values, axis=1)  # (Z, 2, 6)
    return jnp.zeros((NUM_NODES, NUM_PLAYERS, NUM_CARDS)).at[ends].set(cfv)


def solve_round1(value_fn: ValueFn, num_iterations: int, keep_profiles: bool = True):
    """The round-1 subgame at the initial PBS."""
    return solve_round(
        ROUND1,
        initial_ranges(),
        jnp.full(NUM_PLAYERS, ANTE),
        jnp.asarray(COMPAT1),
        None,
        lambda reach: chance_leaf_values(value_fn, reach),
        num_iterations,
        keep_profiles,
    )


def solve_round2(
    round_end: IntArray,
    board: IntArray,
    ranges: FloatArray,
    num_iterations: int,
    value_target: str = "iterates",
) -> SubgameSolution:
    """The round-2 subgame at the PBS after round-1 end node round_end (index
    into ROUND_ENDS) and public card board, with board-restricted `ranges`.
    It has no leaves, so its values are exact (up to the CFR iterations)."""
    start = jnp.asarray(ROUND2_START)[round_end]
    return solve_round(
        ROUND2,
        ranges,
        jnp.full(NUM_PLAYERS, start),
        jnp.asarray(COMPAT2)[board],
        jnp.asarray(SHOWDOWN)[board],
        None,
        num_iterations,
        keep_profiles=False,
        value_target=value_target,
    )


def exact_value_fn(num_iterations: int, value_target: str = "iterates") -> ValueFn:
    """The value function a perfect network would learn: solve the round-2
    subgame exactly (for tests and diagnostics)."""

    def value_fn(round_end, board, ranges):
        return jax.vmap(lambda z, b, r: solve_round2(z, b, r, num_iterations, value_target).values)(
            round_end, board, ranges
        )

    return value_fn


# =============================================================================
# Self play: sampling
# =============================================================================


def sample_profile(rng: PRNGKeyArray, solution: SubgameSolution, num_iterations: int):
    """Policy profile of a random half-step, iteration t's chosen with
    probability proportional to t + 1 (as kuhn_rebel.sample_profile)."""
    weights = jnp.repeat(jnp.arange(1, num_iterations + 1, dtype=jnp.float32), NUM_PLAYERS)
    step = jax.random.categorical(rng, jnp.log(weights))
    return solution.profiles[step // NUM_PLAYERS, step % NUM_PLAYERS]


def sample_deal(rng: PRNGKeyArray, ranges: FloatArray, compat: FloatArray) -> IntArray:
    """Cards (c0, c1) from the PBS deal distribution with card removal."""
    joint = ranges[0][:, None] * ranges[1][None, :] * compat
    joint = jnp.where(jnp.sum(joint) > PROB_EPS, joint, compat)
    deal = jax.random.categorical(rng, jnp.log(joint.reshape(-1)))
    return jnp.stack([deal // NUM_CARDS, deal % NUM_CARDS])


def sample_leaf(
    rng: PRNGKeyArray, policy: FloatArray, ranges: FloatArray, explore_prob: float
) -> tuple[IntArray, FloatArray]:
    """ReBeL's SampleLeaf in the round-1 subgame: draws a deal from the PBS and
    plays `policy` from the root to a fold or round-end node. A random player
    takes a uniformly random legal action with probability explore_prob at
    each of their decisions. Returns the node and the ranges there, which
    follow from `policy` alone."""
    tree = ROUND1
    rng_explorer, rng_deal, rng_walk = jax.random.split(rng, 3)
    explorer = jax.random.randint(rng_explorer, (), 0, NUM_PLAYERS)
    deal = sample_deal(rng_deal, ranges, jnp.asarray(COMPAT1))
    legal = jnp.asarray(tree.legal)
    node = jnp.int32(0)
    for rng_step in jax.random.split(rng_walk, tree.max_depth):
        rng_explore, rng_random, rng_policy = jax.random.split(rng_step, 3)
        player = jnp.asarray(tree.actor)[node]
        explore = (player == explorer) & (jax.random.uniform(rng_explore) < explore_prob)
        action = jnp.where(
            explore,
            jax.random.categorical(rng_random, jnp.where(legal[node], 0.0, -jnp.inf)),
            jax.random.categorical(
                rng_policy, jnp.where(legal[node], jnp.log(policy[node, deal[player]]), -jnp.inf)
            ),
        )
        is_decision = jnp.asarray(tree.kind)[node] == DECISION
        node = jnp.where(is_decision, jnp.asarray(tree.child)[node, action], node)
    reach = compute_reaches(tree, policy, ranges)
    return node, normalize_ranges(reach[node], ranges)


def deal_board(rng: PRNGKeyArray, ranges: FloatArray) -> IntArray:
    """Public card at a round-1 end PBS: a deal from the PBS, then one of the
    4 other cards uniformly (the chance distribution given the PBS)."""
    rng_deal, rng_card = jax.random.split(rng)
    deal = sample_deal(rng_deal, ranges, jnp.asarray(COMPAT1))
    left = jnp.ones(NUM_CARDS).at[deal].set(0.0)
    return jax.random.categorical(rng_card, jnp.where(left > 0, 0.0, -jnp.inf))


# =============================================================================
# Test-time policy and exact evaluation
# =============================================================================


class TestTimePolicy(NamedTuple):
    """Expected behaviour policy of ReBeL at test time, unnormalized
    (numerator) rows: round1 (15, 6, 3) by (round-1 node, card); round2
    (5, 6, 15, 6, 3) by (round-1 end, public card, round-2 node, card)."""

    round1: FloatArray
    round2: FloatArray


def test_time_policy(value_fn: ValueFn, num_iterations: int) -> TestTimePolicy:
    """At test time ReBeL solves the round-1 subgame at the initial PBS,
    samples iteration t with probability proportional to t + 1, plays
    pi^t = (pi_0^t, pi_1^t) and passes its beliefs on to round 2, where it
    solves the round-2 subgame at the new PBS (no leaves) and samples an
    iteration again. Each player's expected policy weights every sampled
    iteration by its probability and the player's own reach; inside a
    subgame that is the Linear CFR average policy."""
    weights = jnp.arange(1, num_iterations + 1, dtype=jnp.float32)
    probs = weights / jnp.sum(weights)
    solution = solve_round1(value_fn, num_iterations)
    profiles = solution.profiles[:, 0]  # (T, 15, 6, 3) pi^t
    ones = jnp.ones((NUM_PLAYERS, NUM_CARDS))
    own = jax.vmap(lambda p: compute_reaches(ROUND1, p, ones))(profiles)  # (T, 15, 2, 6)
    reach = jax.vmap(lambda p: compute_reaches(ROUND1, p, initial_ranges()))(profiles)
    ends = jnp.asarray(ROUND_ENDS)
    end_reach = reach[:, ends]  # (T, Z, 2, 6)
    end_own = own[:, ends]

    def round2_average(z, b, end_reach_t):
        ranges = board_ranges(normalize_ranges(end_reach_t, initial_ranges()), b)
        solution2 = solve_round2(z, b, ranges, num_iterations)
        return solution2.policy_sum / jnp.sum(weights)

    shape = (num_iterations, NUM_ROUND_ENDS, NUM_CARDS)
    z_index = jnp.broadcast_to(jnp.arange(NUM_ROUND_ENDS)[None, :, None], shape)
    b_index = jnp.broadcast_to(jnp.arange(NUM_CARDS)[None, None, :], shape)
    r_index = jnp.broadcast_to(end_reach[:, :, None], shape + (NUM_PLAYERS, NUM_CARDS))
    averages = jax.vmap(round2_average)(
        z_index.reshape(-1), b_index.reshape(-1), r_index.reshape(-1, NUM_PLAYERS, NUM_CARDS)
    ).reshape(shape + (NUM_NODES, NUM_CARDS, NUM_ACTIONS))
    acting_own = end_own[:, :, ROUND2.actor, :]  # (T, Z, 15, 6) own reach of each node's actor
    round2 = jnp.einsum("t,tznc,tzbnca->zbnca", probs, acting_own, averages)
    round1 = solution.policy_sum / jnp.sum(weights)
    return TestTimePolicy(round1=round1, round2=round2)


def _infoset_index() -> tuple[np.ndarray, ...]:
    """For every row of leduc_exploitability._INFOSET_LIST: (round, round-1
    node, round-1 end index, public card, round-2 node, card)."""
    rows = []
    end_of = {int(n): i for i, n in enumerate(ROUND_ENDS)}
    for key, _ in leduc._INFOSET_LIST:
        _, card, public, seq1, seq2 = key[:5]
        n1 = ROUND1.index[tuple(seq1)]
        if public < 0:
            rows.append((0, n1, 0, 0, 0, card))
        else:
            rows.append((1, n1, end_of[n1], public, ROUND2.index[tuple(seq2)], card))
    return tuple(np.array(col, np.int32) for col in zip(*rows))


INFOSET_INDEX = _infoset_index()


def policy_array(policy: TestTimePolicy) -> np.ndarray:
    """The test-time policy as leduc_exploitability's (936, 3) policy array
    (rows in _INFOSET_LIST order, renormalized over legal actions)."""
    round_, n1, z, board, n2, card = INFOSET_INDEX
    r1 = np.asarray(policy.round1, np.float64)[n1, card]
    r2 = np.asarray(policy.round2, np.float64)[z, board, n2, card]
    rows = np.where((round_ == 0)[:, None], r1, r2)
    legal = np.zeros_like(rows, dtype=bool)
    for i, (_, actions) in enumerate(leduc._INFOSET_LIST):
        legal[i, list(actions)] = True
    rows = np.where(legal, rows, 0.0)
    total = rows.sum(-1, keepdims=True)
    uniform = legal / legal.sum(-1, keepdims=True)
    return np.where(total > 0, rows / np.where(total > 0, total, 1.0), uniform)


def evaluate_policy(policy: TestTimePolicy) -> dict[str, float]:
    """Exact exploitability, value against random and policy metrics."""
    table = policy_array(policy)
    out = {
        "exploitability": float(leduc.exploitability_from_policy_array(table, leduc._KEY_TO_IDX)),
        "value_vs_random": float(leduc.value_vs_random_from_policy_array(table, leduc._KEY_TO_IDX)),
    }
    out.update(prefixed("game", policy_metrics(table)))
    return out


# =============================================================================
# Training loop
# =============================================================================


class ReplayBuffer(NamedTuple):
    """Circular buffer of (PBS encoding, values / VALUE_SCALE, value mask)."""

    inputs: FloatArray
    targets: FloatArray
    masks: BoolArray
    position: IntArray
    size: IntArray


class RunnerState(NamedTuple):
    train_state: TrainState
    replay_buffer: ReplayBuffer
    half_params: Any  # value network after half_update_steps(num_update_steps)
    update_step: IntArray
    num_examples: IntArray
    rng: PRNGKeyArray


def make_train(config: dict):
    """Returns (network, init(rng) -> RunnerState, train_block(state) ->
    (state, metrics)), the block being eval_interval updates; both are meant
    to be jitted and vmapped over seeds."""
    num_iterations = config["cfr_iters"]
    num_games = config["num_games"]
    capacity = config["replay_capacity"]
    batch_size = config["batch_size"]
    eval_interval = config["eval_interval"]
    if config["num_update_steps"] % eval_interval != 0:
        raise ValueError("num_update_steps must be a multiple of eval_interval")
    if capacity < num_games:
        raise ValueError("replay_capacity must be at least num_games")
    num_half = half_update_steps(config["num_update_steps"])

    network = ValueNetwork(
        hidden_dim=config["value_hidden_dim"], num_layers=config["value_num_layers"]
    )
    steps_per_update = config["num_value_train_steps"]
    total_steps = config["num_update_steps"] * steps_per_update
    learning_rate = optax.piecewise_constant_schedule(
        config["value_lr"],
        {int(f * total_steps): 0.5 for f in config["lr_halve_at"]},
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(learning_rate=learning_rate),
    )

    def value_loss(params, inputs, targets, masks):
        predictions = network.apply(params, inputs)
        errors = optax.huber_loss(predictions, targets, delta=1.0)
        return jnp.sum(jnp.where(masks, errors, 0.0)) / jnp.maximum(jnp.sum(masks), 1)

    def init(rng: PRNGKeyArray) -> RunnerState:
        rng, rng_init = jax.random.split(rng)
        params = network.init(rng_init, jnp.zeros(PBS_DIM))
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
        buffer = ReplayBuffer(
            inputs=jnp.zeros((capacity, PBS_DIM)),
            targets=jnp.zeros((capacity, NUM_PLAYERS, NUM_CARDS)),
            masks=jnp.zeros((capacity, NUM_PLAYERS, NUM_CARDS), bool),
            position=jnp.int32(0),
            size=jnp.int32(0),
        )
        return RunnerState(train_state, buffer, params, jnp.int32(0), jnp.int32(0), rng)

    def self_play(rng: PRNGKeyArray, value_fn: ValueFn):
        """One hand per game: round-1 leaf, public card, round-2 example."""
        solution = solve_round1(value_fn, num_iterations)

        def game(rng):
            rng_profile, rng_leaf, rng_board = jax.random.split(rng, 3)
            profile = sample_profile(rng_profile, solution, num_iterations)
            node, ranges = sample_leaf(rng_leaf, profile, initial_ranges(), config["explore_prob"])
            valid = jnp.asarray(ROUND1.kind)[node] == ROUND_END
            z = jnp.argmax(jnp.asarray(ROUND_ENDS) == node)
            board = deal_board(rng_board, ranges)
            ranges2 = board_ranges(ranges, board)
            solution2 = solve_round2(z, board, ranges2, num_iterations, config["value_target"])
            return (
                encode_pbs(z, board, ranges2),
                solution2.values / VALUE_SCALE,
                solution2.value_mask & valid,
                valid,
            )

        return jax.vmap(game)(jax.random.split(rng, num_games))

    def update_step(runner_state: RunnerState, _):
        train_state, buffer = runner_state.train_state, runner_state.replay_buffer
        rng, rng_play, rng_train = jax.random.split(runner_state.rng, 3)
        value_fn = network_value_fn(network, train_state.params)
        inputs, targets, masks, valid = self_play(rng_play, value_fn)
        rank = jnp.cumsum(valid) - 1
        index = jnp.where(valid, (buffer.position + rank) % capacity, capacity)
        added = jnp.sum(valid).astype(jnp.int32)
        buffer = ReplayBuffer(
            inputs=buffer.inputs.at[index].set(inputs, mode="drop"),
            targets=buffer.targets.at[index].set(targets, mode="drop"),
            masks=buffer.masks.at[index].set(masks, mode="drop"),
            position=(buffer.position + added) % capacity,
            size=jnp.minimum(buffer.size + added, capacity),
        )

        def train_minibatch(train_state, rng):
            batch = jax.random.randint(rng, (batch_size,), 0, jnp.maximum(buffer.size, 1))
            loss, grads = jax.value_and_grad(value_loss)(
                train_state.params, buffer.inputs[batch], buffer.targets[batch], buffer.masks[batch]
            )
            return train_state.apply_gradients(grads=grads), loss

        train_state, losses = lax.scan(
            train_minibatch, train_state, jax.random.split(rng_train, steps_per_update)
        )
        take_half = runner_state.update_step + 1 == num_half
        half_params = jax.tree.map(
            lambda new, old: jnp.where(take_half, new, old),
            train_state.params,
            runner_state.half_params,
        )
        runner_state = RunnerState(
            train_state,
            buffer,
            half_params,
            runner_state.update_step + 1,
            runner_state.num_examples + added,
            rng,
        )
        return runner_state, {"value_loss": jnp.mean(losses), "hands_to_round2": added}

    def train_block(runner_state: RunnerState):
        runner_state, metrics = lax.scan(update_step, runner_state, None, eval_interval)
        return runner_state, {
            "value_loss": jnp.mean(metrics["value_loss"]),
            "hands_to_round2": jnp.sum(metrics["hands_to_round2"]),
        }

    return network, init, train_block


WANDB_RUNS: list = []


def run(config: dict) -> str:
    """Trains config["num_seeds"] seeds in parallel, evaluates them exactly at
    every eval_interval updates, saves the curves and checkpoints and returns
    the run directory."""
    global WANDB_RUNS
    config["half_update_steps"] = half_update_steps(config["num_update_steps"])
    run_dir = make_run_dir(config)
    num_seeds = config["num_seeds"]
    eval_iters = config["eval_cfr_iters"] or config["cfr_iters"]
    try:
        group = config["env_name"] + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        WANDB_RUNS = [
            wandb.init(
                project=config["project"],
                group=group,
                job_type=f"{config['job_type']}_{config['env_name']}",
                name=f"{config['seed']}_{i}",
                config=config,
                mode=None if config["wandb"] else "disabled",
                dir=str(REPO_ROOT),
                reinit="create_new",
            )
            for i in range(num_seeds)
        ]
        log = ResultsLog(run_dir, WANDB_RUNS)
        print(f"Run directory: {run_dir}")

        network, init, train_block = make_train(config)
        rng_seeds = jax.random.split(jax.random.PRNGKey(config["seed"]), num_seeds)
        runner_state = jax.jit(jax.vmap(init))(rng_seeds)
        print("Compiling...")
        block = jax.jit(jax.vmap(train_block)).lower(runner_state).compile()
        evaluate_params = jax.jit(
            lambda params: test_time_policy(network_value_fn(network, params), eval_iters)
        )
        num_blocks = config["num_update_steps"] // config["eval_interval"]
        curves = [
            {k: [] for k in ("updates", "num_examples", "exploitability", "value_vs_random", "value_loss")}
            for _ in range(num_seeds)
        ]
        start = time.time()
        train_time = 0.0
        train_metrics = None
        for b in range(num_blocks + 1):
            if b > 0:
                t0 = time.time()
                runner_state, train_metrics = jax.block_until_ready(block(runner_state))
                train_time += time.time() - t0
            for i in range(num_seeds):
                params = jax.tree.map(lambda x: x[i], runner_state.train_state.params)
                t0 = time.time()
                with jax.default_matmul_precision(config["eval_matmul_precision"]):
                    policy = jax.device_get(evaluate_params(params))
                metrics = evaluate_policy(policy)
                updates = int(runner_state.update_step[i])
                metrics.update(
                    {
                        "updates_completed": updates,
                        "num_examples": int(runner_state.num_examples[i]),
                        # round-1 solves (one per update) + round-2 solves
                        "num_subgames": updates + int(runner_state.num_examples[i]),
                        "replay_size": int(runner_state.replay_buffer.size[i]),
                        "eval_seconds": time.time() - t0,
                        "train_seconds": train_time,
                    }
                )
                if train_metrics is not None:
                    metrics["value_loss"] = float(train_metrics["value_loss"][i])
                    metrics["hands_to_round2"] = int(train_metrics["hands_to_round2"][i])
                log.log(metrics, i)
                c = curves[i]
                c["updates"].append(updates)
                c["num_examples"].append(metrics["num_examples"])
                c["exploitability"].append(metrics["exploitability"])
                c["value_vs_random"].append(metrics["value_vs_random"])
                c["value_loss"].append(metrics.get("value_loss"))
            expl = [c["exploitability"][-1] for c in curves]
            print(
                f"updates {int(runner_state.update_step[0]):6d}  exploitability "
                + " ".join(f"{e:.4f}" for e in expl)
                + f"  mean {np.mean(expl):.4f}  ({(time.time() - start) / 60:.1f} min)",
                flush=True,
            )
        runtime = time.time() - start
        with open(os.path.join(run_dir, "curves.json"), "w") as f:
            json.dump(
                {"config": config, "runtime_seconds": runtime, "train_seconds": train_time, "seeds": curves},
                f,
            )
        if config["save_checkpoints"]:
            seeds = range(num_seeds)
            save_seed_params(run_dir, "half", "value_net", runner_state.half_params, seeds)
            save_seed_params(run_dir, "full", "value_net", runner_state.train_state.params, seeds)
            print(f"Saved value networks to {run_dir}")
    finally:
        for wandb_run in WANDB_RUNS:
            wandb_run.finish()
        print("Finished.")
    return run_dir


@hydra.main(version_base=None, config_path="./", config_name="config_rebel")
def main(config: dict) -> None:
    run(OmegaConf.to_container(config, resolve=True))


if __name__ == "__main__":
    register_resolvers()
    main()
