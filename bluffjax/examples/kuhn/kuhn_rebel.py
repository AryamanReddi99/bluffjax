"""
ReBeL (Recursive Belief-based Learning) for Kuhn Poker.

Implements ReBeL from Brown, Bakhtin, Lerer and Gong, "Combining Deep
Reinforcement Learning and Search for Imperfect-Information Games" (NeurIPS
2020), following the authors' Liar's Dice code (facebookresearch/rebel,
csrc/liars_dice):

- A public belief state (PBS) is a public betting history plus each player's
  range: the normalized probability that they reach it with each private
  card. With card removal the deal distribution at the PBS is proportional
  to ranges[0, c0] * ranges[1, c1] for c0 != c1.
- The value network maps a PBS to the value of every infostate in it:
  values[i, c] is player i's expected return when holding card c, with the
  opponent's card drawn from their range without c.
- A subgame is rooted at a PBS and extends max_depth actions; non-terminal
  nodes at the depth limit are leaves. It is solved with T iterations of
  Linear CFR-D with alternating updates: in each half-step the leaf PBSs
  follow from the current policy profile and the value network supplies the
  leaf infostate values (CFR-D, not CFR-AVG).
- The training target for the root PBS is the average of its infostate
  values over the T iterations, weighted linearly as in Linear CFR.
- Self play: sample a half-step, iteration t's with probability
  proportional to t + 1, draw a deal from the root PBS and play that
  half-step's policy profile down to a leaf, with one random player taking a
  uniformly random action with probability random_action_prob. The leaf
  PBS, with beliefs updated by the profile, is the root of the next
  subgame. A terminal leaf ends the game.
- At test time ReBeL plays the same way without exploration, sampling
  iteration t and playing its profile. The expected policy of that random
  procedure is computed exactly, by enumerating the sampled iterations and
  weighting by reach, and its exploitability is logged.

Deviations from the paper: there is no policy network or warm start, which
the paper lists as optional. An action that a player's policy never takes
with any card in their range leaves that player's range unchanged.
"""

import datetime
import json
import os
import time
from typing import Callable, NamedTuple

import flax.linen as nn
from flax import serialization
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import hydra
from hydra.core.hydra_config import HydraConfig
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax
import wandb

from bluffjax.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray
from bluffjax.utils.game_utils.kuhn_exploitability import (
    ACTION_BET,
    ACTION_PASS,
    exploitability,
    get_current_player,
    get_returns,
    infoset_key,
    is_terminal,
)
from bluffjax.utils.paths import REPO_ROOT, register_resolvers

# =============================================================================
# Public tree
# =============================================================================

NUM_CARDS = 3
NUM_ACTIONS = 2
NUM_PLAYERS = 2

# Every betting history of Kuhn Poker, parents before children. The public
# state of Kuhn is its betting history, so these are the public tree's nodes.
HISTORIES: tuple[tuple[int, ...], ...] = (
    (),
    (ACTION_PASS,),
    (ACTION_BET,),
    (ACTION_PASS, ACTION_PASS),
    (ACTION_PASS, ACTION_BET),
    (ACTION_BET, ACTION_PASS),
    (ACTION_BET, ACTION_BET),
    (ACTION_PASS, ACTION_BET, ACTION_PASS),
    (ACTION_PASS, ACTION_BET, ACTION_BET),
)
NUM_NODES = len(HISTORIES)
ROOT_NODE = 0
MAX_GAME_LENGTH = max(len(h) for h in HISTORIES)

IS_TERMINAL = np.array([is_terminal(h) for h in HISTORIES])
# Acting player at each node; 0 at terminal nodes, where it is never used.
ACTING_PLAYER = np.array(
    [0 if is_terminal(h) else get_current_player(h) for h in HISTORIES]
)
CHILD = np.zeros((NUM_NODES, NUM_ACTIONS), dtype=np.int32)
PARENT = np.full(NUM_NODES, -1)
PARENT_ACTION = np.full(NUM_NODES, -1)
for _node, _history in enumerate(HISTORIES):
    if _history:
        PARENT[_node] = HISTORIES.index(_history[:-1])
        PARENT_ACTION[_node] = _history[-1]
        CHILD[PARENT[_node], _history[-1]] = _node

# Non-terminal nodes are the public states a PBS can be at, in the order of
# the value network's one-hot encoding.
PUBLIC_STATES: tuple[int, ...] = tuple(
    n for n in range(NUM_NODES) if not IS_TERMINAL[n]
)
PUBLIC_STATE_INDEX = np.zeros(NUM_NODES, dtype=np.int32)
PUBLIC_STATE_INDEX[list(PUBLIC_STATES)] = np.arange(len(PUBLIC_STATES))

# CARD_REMOVAL[c, o] = 1 if a player holding c can face an opponent holding o.
CARD_REMOVAL = 1.0 - np.eye(NUM_CARDS, dtype=np.float32)

# PAYOFF[i, n, c, o]: return of player i holding card c against an opponent
# holding card o at terminal node n; 0 at non-terminal nodes and for c == o.
PAYOFF = np.zeros((NUM_PLAYERS, NUM_NODES, NUM_CARDS, NUM_CARDS), np.float32)
for _node in np.nonzero(IS_TERMINAL)[0]:
    for _c0 in range(NUM_CARDS):
        for _c1 in range(NUM_CARDS):
            if _c0 != _c1:
                _r0, _r1 = get_returns((_c0, _c1), HISTORIES[_node])
                PAYOFF[0, _node, _c0, _c1] = _r0
                PAYOFF[1, _node, _c1, _c0] = _r1

# Reach masses at or below this count as zero.
PROB_EPS = 1e-12


class SubgameStructure(NamedTuple):
    """Masks over public nodes of the subgame rooted at each node, [root, node]."""

    decision: np.ndarray
    leaf: np.ndarray
    terminal: np.ndarray


def subgame_structure(max_depth: int) -> SubgameStructure:
    """
    Subgames extending max_depth actions below their root. Non-terminal nodes
    at the depth limit are leaves, valued by the value network.
    """
    if max_depth < 1:
        raise ValueError(f"max_depth must be at least 1, got {max_depth}")
    decision = np.zeros((NUM_NODES, NUM_NODES), dtype=bool)
    leaf = np.zeros((NUM_NODES, NUM_NODES), dtype=bool)
    terminal = np.zeros((NUM_NODES, NUM_NODES), dtype=bool)
    for root, root_history in enumerate(HISTORIES):
        for node, history in enumerate(HISTORIES):
            depth = len(history) - len(root_history)
            if history[: len(root_history)] != root_history or depth > max_depth:
                continue
            if IS_TERMINAL[node]:
                terminal[root, node] = True
            elif depth < max_depth:
                decision[root, node] = True
            else:
                leaf[root, node] = True
    return SubgameStructure(decision=decision, leaf=leaf, terminal=terminal)


# =============================================================================
# Public belief states and the value network
# =============================================================================

PBS_DIM = len(PUBLIC_STATES) + NUM_PLAYERS * NUM_CARDS


class PBS(NamedTuple):
    """Public belief state: public node and both players' ranges, (2, 3)."""

    node: IntArray
    ranges: FloatArray


def initial_pbs() -> PBS:
    return PBS(
        node=jnp.int32(ROOT_NODE),
        ranges=jnp.full((NUM_PLAYERS, NUM_CARDS), 1.0 / NUM_CARDS),
    )


def encode_pbs(node: IntArray, ranges: FloatArray) -> FloatArray:
    """Value network input: one-hot public state and both ranges."""
    public_state = jax.nn.one_hot(
        jnp.asarray(PUBLIC_STATE_INDEX)[node], len(PUBLIC_STATES)
    )
    flat_ranges = ranges.reshape(ranges.shape[:-2] + (NUM_PLAYERS * NUM_CARDS,))
    return jnp.concatenate([public_state, flat_ranges], axis=-1)


def normalize_ranges(reach: FloatArray, fallback: FloatArray) -> FloatArray:
    """
    Bayes' rule over each player's card: normalizes reach probabilities,
    (..., 2, 3), into ranges. A player whose reach is zero for every card,
    which happens after an action the policy never takes, keeps the fallback
    range.
    """
    total = jnp.sum(reach, axis=-1, keepdims=True)
    reached = total > PROB_EPS
    return jnp.where(reached, reach / jnp.where(reached, total, 1.0), fallback)


class ValueNetwork(nn.Module):
    """MLP mapping a PBS encoding to infostate values, (..., 2, 3)."""

    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x: FloatArray) -> FloatArray:
        for _ in range(2):
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


# value_fn(nodes (K,), ranges (K, 2, 3)) -> infostate values (K, 2, 3)
ValueFn = Callable[[IntArray, FloatArray], FloatArray]


def network_value_fn(network: nn.Module, params) -> ValueFn:
    def value_fn(nodes: IntArray, ranges: FloatArray) -> FloatArray:
        return network.apply(params, encode_pbs(nodes, ranges))

    return value_fn


# =============================================================================
# Depth-limited subgame solving with Linear CFR-D
# =============================================================================


class SubgameSolution(NamedTuple):
    """
    profiles: (T, 2, 9, 3, 2) policy profile of player i's half-step on CFR
        iteration t, (pi_0^t, pi_1^t) for i = 0 and (pi_0^(t+1), pi_1^t) for
        i = 1. None if not kept.
    policy_sum: (9, 3, 2) sum over t of (t + 1) * own reach * pi^t, the
        unnormalized Linear CFR average policy.
    values: (2, 3) linearly averaged root infostate values, 0 where masked.
    value_mask: (2, 3) root infostates whose value is defined: the opponent
        has nonzero range on the cards left after card removal.
    """

    profiles: FloatArray
    policy_sum: FloatArray
    values: FloatArray
    value_mask: BoolArray


def iteration_weights(num_iterations: int) -> FloatArray:
    """Linear CFR weight of iteration t = 0..T-1."""
    return jnp.arange(1, num_iterations + 1, dtype=jnp.float32)


def regret_matching(regrets: FloatArray) -> FloatArray:
    """Policy proportional to positive regret, uniform if there is none."""
    positive = jnp.maximum(regrets, 0.0)
    total = jnp.sum(positive, axis=-1, keepdims=True)
    return jnp.where(
        total > 0.0,
        positive / jnp.where(total > 0.0, total, 1.0),
        1.0 / NUM_ACTIONS,
    )


def compute_reaches(
    policy: FloatArray,
    root: IntArray,
    ranges: FloatArray,
    decision: BoolArray,
) -> FloatArray:
    """
    reach[i, n, c]: probability that player i holds card c under `ranges` and
    plays from the subgame root to node n. Zero outside the subgame.
    """
    rows = []
    for n in range(NUM_NODES):
        if n == ROOT_NODE:
            below = jnp.zeros_like(ranges)
        else:
            parent = PARENT[n]
            factor = (
                jnp.ones_like(ranges)
                .at[ACTING_PLAYER[parent]]
                .set(policy[parent, :, PARENT_ACTION[n]])
            )
            below = jnp.where(decision[parent], rows[parent] * factor, 0.0)
        rows.append(jnp.where(root == n, ranges, below))
    return jnp.stack(rows, axis=1)


def opponent_mass(reach: FloatArray) -> FloatArray:
    """mass[i, ..., c]: opponent reach on the cards other than c."""
    return jnp.einsum("i...o,co->i...c", reach[::-1], CARD_REMOVAL)


def counterfactual_values(
    policy: FloatArray,
    reach: FloatArray,
    leaf_values: FloatArray,
    decision: BoolArray,
    leaf: BoolArray,
    terminal: BoolArray,
) -> FloatArray:
    """
    values[i, n, c]: counterfactual value of player i holding card c at node
    n, the sum over opponent cards o != c of the opponent's reach times player
    i's expected return. At leaves the value network's infostate values,
    leaf_values[n, i, c], are scaled by the opponent's reach on the cards
    other than c.
    """
    terminal_values = jnp.einsum("inco,ino->inc", PAYOFF, reach[::-1])
    leaf_cfv = opponent_mass(reach) * jnp.swapaxes(leaf_values, 0, 1)
    rows: list = [None] * NUM_NODES
    for n in reversed(range(NUM_NODES)):
        if IS_TERMINAL[n]:
            rows[n] = jnp.where(terminal[n], terminal_values[:, n], 0.0)
            continue
        children = jnp.stack([rows[CHILD[n, a]] for a in range(NUM_ACTIONS)], -1)
        weights = jnp.ones_like(children).at[ACTING_PLAYER[n]].set(policy[n])
        internal = jnp.sum(children * weights, axis=-1)
        rows[n] = jnp.where(
            leaf[n], leaf_cfv[:, n], jnp.where(decision[n], internal, 0.0)
        )
    return jnp.stack(rows, axis=1)


def instantaneous_regrets(values: FloatArray, decision: BoolArray) -> FloatArray:
    """regrets[n, c, a]: acting player's value of a minus their value at n."""
    rows = []
    for n in range(NUM_NODES):
        player = ACTING_PLAYER[n]
        child_values = values[player][CHILD[n]].T
        rows.append(
            jnp.where(decision[n], child_values - values[player, n][:, None], 0.0)
        )
    return jnp.stack(rows)


def solve_subgame(
    pbs: PBS,
    value_fn: ValueFn,
    structure: SubgameStructure,
    num_iterations: int,
    leaf_nodes: tuple[int, ...],
    keep_profiles: bool = True,
) -> SubgameSolution:
    """
    Linear CFR-D with alternating updates in the subgame rooted at pbs. On
    iteration t player 0 and then player 1 update their regrets with weight
    t + 1, each from the current profile, whose leaf PBSs are valued by
    value_fn. leaf_nodes are the nodes value_fn is queried at; every leaf of
    the subgame must be among them. keep_profiles stores every half-step's
    profile.
    """
    decision = jnp.asarray(structure.decision)[pbs.node]
    leaf = jnp.asarray(structure.leaf)[pbs.node]
    terminal = jnp.asarray(structure.terminal)[pbs.node]
    leaf_index = jnp.asarray(leaf_nodes, dtype=jnp.int32)
    own_range = jnp.ones_like(pbs.ranges)
    root_mass = opponent_mass(pbs.ranges)
    value_mask = root_mass > PROB_EPS

    def half_step(player: int, regrets, policy_sum, value_sum, profile, weight):
        # Player's regrets, average policy and root values from `profile`.
        player_nodes = decision & (jnp.asarray(ACTING_PLAYER) == player)
        reach = compute_reaches(profile, pbs.node, pbs.ranges, decision)
        leaf_values = jnp.zeros((NUM_NODES, NUM_PLAYERS, NUM_CARDS))
        if leaf_nodes:
            leaf_ranges = normalize_ranges(
                jnp.swapaxes(reach[:, leaf_index], 0, 1), pbs.ranges
            )
            leaf_values = leaf_values.at[leaf_index].set(
                value_fn(leaf_index, leaf_ranges)
            )
        values = counterfactual_values(
            profile, reach, leaf_values, decision, leaf, terminal
        )
        regrets = regrets + weight * instantaneous_regrets(values, player_nodes)
        own_reach = compute_reaches(profile, pbs.node, own_range, decision)
        policy_sum = policy_sum + weight * jnp.where(
            player_nodes[:, None, None], own_reach[player][..., None] * profile, 0.0
        )
        root_values = values[player, pbs.node] / jnp.where(
            value_mask[player], root_mass[player], 1.0
        )
        value_sum = value_sum.at[player].add(weight * root_values)
        return regrets, policy_sum, value_sum

    def iteration(carry, t):
        regrets, policy_sum, value_sum = carry
        weight = (t + 1).astype(jnp.float32)
        profile_p0 = regret_matching(regrets)
        regrets, policy_sum, value_sum = half_step(
            0, regrets, policy_sum, value_sum, profile_p0, weight
        )
        # Player 0's new policy, player 1's current one.
        profile_p1 = jnp.where(
            (jnp.asarray(ACTING_PLAYER) == 0)[:, None, None],
            regret_matching(regrets),
            profile_p0,
        )
        regrets, policy_sum, value_sum = half_step(
            1, regrets, policy_sum, value_sum, profile_p1, weight
        )
        profiles = jnp.stack([profile_p0, profile_p1]) if keep_profiles else None
        return (regrets, policy_sum, value_sum), profiles

    zeros = jnp.zeros((NUM_NODES, NUM_CARDS, NUM_ACTIONS))
    init = (zeros, zeros, jnp.zeros((NUM_PLAYERS, NUM_CARDS)))
    (_, policy_sum, value_sum), profiles = lax.scan(
        iteration, init, jnp.arange(num_iterations)
    )
    values = jnp.where(
        value_mask, value_sum / jnp.sum(iteration_weights(num_iterations)), 0.0
    )
    return SubgameSolution(
        profiles=profiles,
        policy_sum=policy_sum,
        values=values,
        value_mask=value_mask,
    )


def sample_profile(
    rng: PRNGKeyArray, solution: SubgameSolution, num_iterations: int
) -> FloatArray:
    """
    Policy profile of a random half-step, iteration t's chosen with
    probability proportional to t + 1. Self play samples among all half-steps
    so that every leaf PBS the search queries can become a subgame root.
    """
    weights = jnp.repeat(iteration_weights(num_iterations), NUM_PLAYERS)
    step = jax.random.categorical(rng, jnp.log(weights))
    return solution.profiles[step // NUM_PLAYERS, step % NUM_PLAYERS]


def sample_deal(rng: PRNGKeyArray, ranges: FloatArray) -> IntArray:
    """Cards (c0, c1) from the PBS deal distribution, with card removal."""
    joint = ranges[0][:, None] * ranges[1][None, :] * CARD_REMOVAL
    joint = jnp.where(jnp.sum(joint) > PROB_EPS, joint, CARD_REMOVAL)
    deal = jax.random.categorical(rng, jnp.log(joint.reshape(-1)))
    return jnp.stack([deal // NUM_CARDS, deal % NUM_CARDS])


def sample_leaf(
    rng: PRNGKeyArray,
    pbs: PBS,
    policy: FloatArray,
    structure: SubgameStructure,
    random_action_prob: float,
) -> PBS:
    """
    ReBeL's SampleLeaf: draws a deal from pbs and plays `policy` from the
    subgame root to a leaf or terminal node. A random player takes a uniform
    random action with probability random_action_prob at each of their
    decisions. Beliefs at the reached node follow from `policy` alone.
    """
    rng_explorer, rng_deal, rng_walk = jax.random.split(rng, 3)
    explorer = jax.random.randint(rng_explorer, (), 0, NUM_PLAYERS)
    deal = sample_deal(rng_deal, pbs.ranges)
    decision = jnp.asarray(structure.decision)[pbs.node]
    node = pbs.node
    for rng_step in jax.random.split(rng_walk, MAX_GAME_LENGTH):
        rng_explore, rng_random, rng_policy = jax.random.split(rng_step, 3)
        player = jnp.asarray(ACTING_PLAYER)[node]
        explore = (player == explorer) & (
            jax.random.uniform(rng_explore) < random_action_prob
        )
        action = jnp.where(
            explore,
            jax.random.randint(rng_random, (), 0, NUM_ACTIONS),
            jax.random.categorical(rng_policy, jnp.log(policy[node, deal[player]])),
        )
        node = jnp.where(decision[node], jnp.asarray(CHILD)[node, action], node)
    reach = compute_reaches(policy, pbs.node, pbs.ranges, decision)
    return PBS(node=node, ranges=normalize_ranges(reach[:, node], pbs.ranges))


# =============================================================================
# Test-time policy
# =============================================================================


def test_time_policy(
    value_fn: ValueFn,
    structure: SubgameStructure,
    num_iterations: int,
) -> FloatArray:
    """
    Expected behavior policy, (9, 3, 2), of ReBeL at test time: at every
    public state reached it solves the subgame rooted at the current PBS,
    samples an iteration t with probability proportional to t + 1, plays
    pi^t = (pi_0^t, pi_1^t) and passes the beliefs of pi^t on to the next
    subgame, as the authors' evaluation does. The sampled iterations are
    enumerated exactly; each player's policy is the average over them
    weighted by the player's own reach.
    """
    weights = iteration_weights(num_iterations)
    probs = weights / jnp.sum(weights)
    numerator = jnp.zeros((NUM_NODES, NUM_CARDS, NUM_ACTIONS))
    denominator = jnp.zeros((NUM_NODES, NUM_CARDS))

    def expand(root: int, ranges: FloatArray, weight: FloatArray, own: FloatArray):
        # A batch of PBSs at public node `root`: ranges (N, 2, 3), probability
        # of the iterations sampled so far (N,) and each player's own reach
        # from the start of the game (N, 2, 3).
        nonlocal numerator, denominator
        leaf_nodes = tuple(int(n) for n in np.nonzero(structure.leaf[root])[0])
        solution = jax.vmap(
            lambda r: solve_subgame(
                PBS(jnp.int32(root), r),
                value_fn,
                structure,
                num_iterations,
                leaf_nodes,
                keep_profiles=bool(leaf_nodes),
            )
        )(ranges)
        # Inside the subgame the reach-weighted average over the sampled
        # iteration is the Linear CFR average policy.
        scale = weight / jnp.sum(weights)
        for n in np.nonzero(structure.decision[root])[0]:
            played = (
                scale[:, None, None]
                * own[:, ACTING_PLAYER[n], :, None]
                * solution.policy_sum[:, n]
            )
            numerator = numerator.at[n].add(jnp.sum(played, axis=0))
            denominator = denominator.at[n].add(jnp.sum(played, axis=(0, 2)))
        if not leaf_nodes:
            return
        decision = jnp.asarray(structure.decision[root])
        reaches = jax.vmap(
            jax.vmap(compute_reaches, in_axes=(0, None, None, None)),
            in_axes=(0, None, 0, None),
        )
        profiles = solution.profiles[:, :, 0]
        reach = reaches(profiles, root, ranges, decision)
        own_reach = reaches(profiles, root, jnp.ones_like(ranges), decision)
        batch = ranges.shape[0] * num_iterations
        for z in leaf_nodes:
            child_ranges = normalize_ranges(reach[:, :, :, z], ranges[:, None])
            child_weight = weight[:, None] * probs[None, :]
            child_own = own[:, None] * own_reach[:, :, :, z]
            expand(
                z,
                child_ranges.reshape(batch, NUM_PLAYERS, NUM_CARDS),
                child_weight.reshape(batch),
                child_own.reshape(batch, NUM_PLAYERS, NUM_CARDS),
            )

    start = initial_pbs()
    expand(
        ROOT_NODE,
        start.ranges[None],
        jnp.ones(1),
        jnp.ones((1, NUM_PLAYERS, NUM_CARDS)),
    )
    reached = denominator[..., None] > 0.0
    return jnp.where(
        reached, numerator / jnp.where(reached, denominator[..., None], 1.0), 0.5
    )


def policy_exploitability(policy: np.ndarray) -> float:
    """Exploitability of a (9, 3, 2) policy over (public node, card)."""
    policy = np.asarray(policy, dtype=np.float64)
    table = {
        infoset_key(card, HISTORIES[node]): policy[node, card]
        / policy[node, card].sum()
        for node in PUBLIC_STATES
        for card in range(NUM_CARDS)
    }
    return float(exploitability(lambda key: table[key]))


# =============================================================================
# Training loop
# =============================================================================


class ReplayBuffer(NamedTuple):
    """Circular buffer of (PBS encoding, infostate values, value mask)."""

    inputs: FloatArray
    targets: FloatArray
    masks: BoolArray
    position: IntArray
    size: IntArray


class RunnerState(NamedTuple):
    train_state: TrainState
    replay_buffer: ReplayBuffer
    pbs: PBS
    update_step: IntArray
    rng: PRNGKeyArray


def make_train(
    config: dict,
) -> Callable[[PRNGKeyArray, IntArray], tuple[RunnerState, dict]]:
    structure = subgame_structure(config["max_depth"])
    # Nodes that are a leaf of some subgame; the value network is queried
    # there on every CFR iteration of self play.
    leaf_nodes = tuple(int(n) for n in np.nonzero(structure.leaf.any(axis=0))[0])
    num_iterations = config["cfr_iterations"]
    num_subgames = config["num_subgames_per_update"]
    capacity = config["replay_capacity"]
    batch_size = config["batch_size"]
    log_interval = config["log_interval"]
    if capacity < num_subgames:
        raise ValueError("replay_capacity must be at least num_subgames_per_update")
    if config["num_update_steps"] % log_interval != 0:
        raise ValueError("num_update_steps must be a multiple of log_interval")

    network = ValueNetwork(hidden_dim=config["value_hidden_dim"])
    # The learning rate halves every value_lr_halving_interval updates (as in
    # the ReBeL paper); 0 keeps it constant.
    learning_rate = config["value_lr"]
    if config["value_lr_halving_interval"] > 0:
        learning_rate = optax.exponential_decay(
            init_value=config["value_lr"],
            transition_steps=config["value_lr_halving_interval"]
            * config["num_value_train_steps"],
            decay_rate=0.5,
            staircase=True,
        )
    tx = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(learning_rate=learning_rate),
    )

    def selfplay_step(rng: PRNGKeyArray, pbs: PBS, value_fn: ValueFn):
        solution = solve_subgame(pbs, value_fn, structure, num_iterations, leaf_nodes)
        rng_profile, rng_leaf = jax.random.split(rng)
        next_pbs = sample_leaf(
            rng_leaf,
            pbs,
            sample_profile(rng_profile, solution, num_iterations),
            structure,
            config["random_action_prob"],
        )
        done = jnp.asarray(IS_TERMINAL)[next_pbs.node]
        next_pbs = jax.tree.map(
            lambda reset, x: jnp.where(done, reset, x), initial_pbs(), next_pbs
        )
        example = (
            encode_pbs(pbs.node, pbs.ranges),
            solution.values,
            solution.value_mask,
        )
        return example, next_pbs, done

    def value_loss(params, inputs, targets, masks) -> FloatArray:
        predictions = network.apply(params, inputs)
        errors = optax.huber_loss(predictions, targets, delta=1.0)
        return jnp.sum(jnp.where(masks, errors, 0.0)) / jnp.maximum(jnp.sum(masks), 1)

    def train(rng: PRNGKeyArray, exp_id: IntArray) -> tuple[RunnerState, dict]:
        rng, rng_init = jax.random.split(rng)
        params = network.init(rng_init, jnp.zeros(PBS_DIM))
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
        replay_buffer = ReplayBuffer(
            inputs=jnp.zeros((capacity, PBS_DIM)),
            targets=jnp.zeros((capacity, NUM_PLAYERS, NUM_CARDS)),
            masks=jnp.zeros((capacity, NUM_PLAYERS, NUM_CARDS), dtype=jnp.bool_),
            position=jnp.int32(0),
            size=jnp.int32(0),
        )
        pbs = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_subgames,) + x.shape), initial_pbs()
        )

        def update_step(runner_state: RunnerState, _) -> tuple[RunnerState, dict]:
            train_state = runner_state.train_state
            buffer = runner_state.replay_buffer
            rng, rng_selfplay, rng_train = jax.random.split(runner_state.rng, 3)

            # Self play: every game solves the subgame at its current PBS.
            value_fn = network_value_fn(network, train_state.params)
            (inputs, targets, masks), pbs, done = jax.vmap(
                lambda rng, pbs: selfplay_step(rng, pbs, value_fn)
            )(jax.random.split(rng_selfplay, num_subgames), runner_state.pbs)
            index = (buffer.position + jnp.arange(num_subgames)) % capacity
            buffer = ReplayBuffer(
                inputs=buffer.inputs.at[index].set(inputs),
                targets=buffer.targets.at[index].set(targets),
                masks=buffer.masks.at[index].set(masks),
                position=(buffer.position + num_subgames) % capacity,
                size=jnp.minimum(buffer.size + num_subgames, capacity),
            )

            def train_minibatch(train_state: TrainState, rng: PRNGKeyArray):
                batch = jax.random.randint(rng, (batch_size,), 0, buffer.size)
                loss, grads = jax.value_and_grad(value_loss)(
                    train_state.params,
                    buffer.inputs[batch],
                    buffer.targets[batch],
                    buffer.masks[batch],
                )
                return train_state.apply_gradients(grads=grads), loss

            train_state, losses = lax.scan(
                train_minibatch,
                train_state,
                jax.random.split(rng_train, config["num_value_train_steps"]),
            )
            metrics = {
                "value_loss": jnp.mean(losses),
                "games_finished": jnp.sum(done),
            }
            runner_state = RunnerState(
                train_state=train_state,
                replay_buffer=buffer,
                pbs=pbs,
                update_step=runner_state.update_step + 1,
                rng=rng,
            )
            return runner_state, metrics

        def evaluate(runner_state: RunnerState, train_metrics: dict) -> dict:
            params = runner_state.train_state.params
            start = initial_pbs()
            # Player 0's game value as predicted at the initial PBS; the value
            # of Kuhn Poker is -1/18.
            root_values = network.apply(params, encode_pbs(start.node, start.ranges))
            metrics = {
                **train_metrics,
                "update_step": runner_state.update_step,
                "num_subgames": runner_state.update_step * num_subgames,
                "replay_size": runner_state.replay_buffer.size,
                "root_value_p0": jnp.mean(root_values[0]),
            }
            policy = test_time_policy(
                network_value_fn(network, params), structure, num_iterations
            )

            def logging_callback(exp_id, metrics, policy):
                log_dict = {k: np.array(v) for k, v in metrics.items()}
                log_dict = {k: v for k, v in log_dict.items() if np.isfinite(v)}
                log_dict["exploitability"] = policy_exploitability(policy)
                WANDB_RUNS[int(exp_id)].log(log_dict)

            jax.experimental.io_callback(
                logging_callback, None, exp_id, metrics, policy
            )
            return {**metrics, "policy": policy}

        def log_step(carry, _) -> tuple[tuple[RunnerState, dict], dict]:
            # Evaluates before the block's updates, with the previous block's
            # training metrics, so that log calls arrive in order.
            runner_state, train_metrics = carry
            metrics = evaluate(runner_state, train_metrics)
            runner_state, block_metrics = lax.scan(
                update_step, runner_state, None, log_interval
            )
            train_metrics = {
                "value_loss": jnp.mean(block_metrics["value_loss"]),
                "games_finished": jnp.sum(block_metrics["games_finished"]),
            }
            return (runner_state, train_metrics), metrics

        runner_state = RunnerState(
            train_state=train_state,
            replay_buffer=replay_buffer,
            pbs=pbs,
            update_step=jnp.int32(0),
            rng=rng,
        )
        # The untrained value network is evaluated first, then every
        # log_interval updates.
        no_training = {
            "value_loss": jnp.float32(jnp.nan),
            "games_finished": jnp.int32(0),
        }
        (runner_state, train_metrics), metrics = lax.scan(
            log_step,
            (runner_state, no_training),
            None,
            config["num_update_steps"] // log_interval,
        )
        final_metrics = evaluate(runner_state, train_metrics)
        metrics = jax.tree.map(
            lambda rest, last: jnp.concatenate([rest, last[None]]),
            metrics,
            final_metrics,
        )
        return runner_state, metrics

    return train


WANDB_RUNS: list = []  # one wandb run per vmapped seed, created in main()


@hydra.main(version_base=None, config_path="./", config_name="config_rebel")
def main(config: dict) -> None:
    try:
        config = OmegaConf.to_container(config, resolve=True)
        rng = jax.random.PRNGKey(config["seed"])
        rng_seeds = jax.random.split(rng, config["num_seeds"])
        exp_ids = jnp.arange(config["num_seeds"])

        print("Starting compile...")
        train_vjit = (
            jax.jit(jax.vmap(make_train(config))).lower(rng_seeds, exp_ids).compile()
        )
        print("Compile finished...")

        job_type = f"{config['job_type']}_{config['env_name']}"
        group = f"{config['env_name']}" + datetime.datetime.now().strftime(
            "_%Y-%m-%d_%H-%M-%S"
        )
        global WANDB_RUNS
        WANDB_RUNS = [
            wandb.init(
                project=config["project"],
                group=group,
                job_type=job_type,
                name=f"{config['seed']}_{i}",
                config=config,
                mode=None if config["wandb"] else "disabled",
                dir=str(REPO_ROOT),
                reinit="create_new",
            )
            for i in range(config["num_seeds"])
        ]

        print("Running...")
        start_time = time.time()
        runner_state, metrics = jax.block_until_ready(train_vjit(rng_seeds, exp_ids))
        runtime = time.time() - start_time
        print(f"Training took {runtime:.1f}s")

        # Exploitability curve of each seed, saved next to the Hydra logs.
        curves = []
        for i in range(config["num_seeds"]):
            value_loss = np.asarray(metrics["value_loss"][i])
            curves.append(
                {
                    "num_subgames": np.asarray(metrics["num_subgames"][i]).tolist(),
                    "exploitability": [
                        policy_exploitability(p)
                        for p in np.asarray(metrics["policy"][i])
                    ],
                    "value_loss": [
                        None if np.isnan(v) else float(v) for v in value_loss
                    ],
                    "root_value_p0": np.asarray(metrics["root_value_p0"][i]).tolist(),
                }
            )
            print(
                f"Seed {i}: final exploitability {curves[i]['exploitability'][-1]:.5f}"
            )
        curves_path = os.path.join(HydraConfig.get().runtime.output_dir, "curves.json")
        with open(curves_path, "w") as f:
            json.dump(
                {"config": config, "runtime_seconds": runtime, "seeds": curves}, f
            )
        print(f"Saved exploitability curves to {curves_path}")

        if config["save_final"]:
            os.makedirs(config["save_dir"], exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                config["save_dir"], f"kuhn_rebel_{timestamp}.msgpack"
            )
            params = jax.tree.map(lambda x: x[0], runner_state.train_state.params)
            with open(save_path, "wb") as f:
                f.write(serialization.to_bytes(params))
            print(f"Saved value network to {save_path}")
    finally:
        for run in WANDB_RUNS:
            run.finish()
        print("Finished.")


if __name__ == "__main__":
    register_resolvers()
    main()
