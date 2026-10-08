"""
Betting rules and search trees for heads-up Limit and No-Limit Hold'em.

The rules mirror texas_limit_holdem.py and texas_nolimit_holdem.py exactly
(the tests compare them with env.step_env), with players indexed by position:
0 = small blind, 1 = big blind. As in the environments, the small blind acts
first on every street.

A ReBeL subgame covers the rest of the current betting round. Its shape comes
from a static template of action strings; the betting state, legality and the
kind of every node are computed for the actual root inside jit, so one compiled
solver handles every root (start of a round, or mid-round for re-solves).

Trees are written in action slots. In No-Limit a slot is an env action. In
Limit, check and call share the "passive" slot because exactly one of them is
legal at any decision, which halves the tree.

The search tree differs from the environment in two documented ways:
- folding when checking is free is left out (it is strictly dominated);
- No-Limit caps non-all-in raises (half pot / pot) at `max_raises` per round
  in the tree. All-in, call and fold stay available. The test-time agent
  re-solves when an opponent goes beyond the cap. In Limit the cap equals the
  environment's own (4 raises), so the Limit tree is the full betting round.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from bluffjax.examples.holdem_rebel.cards import NUM_CARDS, cards_onehot
from bluffjax.utils.typing import BoolArray, FloatArray, IntArray

# Node kinds
INVALID = 0
DECISION = 1
FOLD = 2
ROUND_END = 3  # betting round closed (showdown on the river, else a leaf)

# Step outcomes
CONTINUE = 0
FOLDED = 1
CLOSED = 2


@dataclass(frozen=True)
class HoldemGame:
    """Static description of one heads-up Hold'em variant."""

    name: str  # "hul" or "hunl"
    env_id: str
    num_actions: int  # env actions
    env_slot: tuple[int, ...]  # slot of each env action
    raise_slots: tuple[int, ...]  # capped in the tree
    allin_slot: int  # -1 if none
    fold_slot: int
    fold_action: int  # env action
    max_raises: int  # cap on raise slots per round in the search tree
    reward_per_chip: float  # env reward units per chip
    value_scale: float  # largest possible contribution, in reward units
    stack: float  # chips per player (inf for Limit)

    @property
    def is_limit(self) -> bool:
        return self.name == "hul"

    @property
    def num_slots(self) -> int:
        return max(self.env_slot) + 1


PASSIVE_SLOT = 0  # check or call


def make_game(name: str, max_raises: int | None = None) -> HoldemGame:
    """max_raises applies to No-Limit only; Limit always uses the env's cap."""
    if name == "hul":
        # Env actions: 0 call, 1 raise, 2 fold, 3 check. Slots: 0 check/call,
        # 1 raise, 2 fold. 4 raises per round is the env's cap. The largest
        # contribution is 1 + 4 bets of 1 BB preflop and on the flop and 2 BB
        # on the turn and river: 25 BB.
        return HoldemGame(
            name="hul",
            env_id="texas_limit_holdem",
            num_actions=4,
            env_slot=(0, 1, 2, 0),
            raise_slots=(1,),
            allin_slot=-1,
            fold_slot=2,
            fold_action=2,
            max_raises=4,
            reward_per_chip=0.5,
            value_scale=25.0,
            stack=float("inf"),
        )
    if name == "hunl":
        # Env actions = slots: 0 check/call, 1 raise half pot, 2 raise pot,
        # 3 all-in, 4 fold. Stacks of 100 chips, blinds 1/2.
        return HoldemGame(
            name="hunl",
            env_id="texas_nolimit_holdem",
            num_actions=5,
            env_slot=(0, 1, 2, 3, 4),
            raise_slots=(1, 2),
            allin_slot=3,
            fold_slot=4,
            fold_action=4,
            max_raises=3 if max_raises is None else max_raises,
            reward_per_chip=1.0,
            value_scale=100.0,
            stack=100.0,
        )
    raise ValueError(f"unknown game {name}")


# =============================================================================
# Betting state and rules
# =============================================================================


class BetState(NamedTuple):
    """Public betting state within a round, by position (0 = SB, 1 = BB)."""

    chips: FloatArray  # (2,) total chips in the pot
    round_bets: FloatArray  # (2,) chips put in this round (No-Limit)
    raises: IntArray  # raises made this round (Limit cap)
    nrn: IntArray  # env not_raise_num
    actor: IntArray  # position to act
    street: IntArray  # 0 preflop .. 3 river
    allin: BoolArray  # (2,) (No-Limit)


def preflop_root() -> BetState:
    blinds = jnp.array([1.0, 2.0])
    return BetState(
        chips=blinds,
        round_bets=blinds,
        raises=jnp.int32(0),
        nrn=jnp.int32(0),
        actor=jnp.int32(0),
        street=jnp.int32(0),
        allin=jnp.zeros(2, dtype=bool),
    )


def round_root(game: HoldemGame, street: IntArray, chips: FloatArray) -> BetState:
    """Start of a betting round with equal contributions (preflop: blinds)."""
    pre = preflop_root()
    post = BetState(
        chips=jnp.full(2, chips, dtype=jnp.float32),
        round_bets=jnp.zeros(2),
        raises=jnp.int32(0),
        nrn=jnp.int32(0),
        actor=jnp.int32(0),
        street=jnp.asarray(street, jnp.int32),
        allin=jnp.full(2, chips >= game.stack),
    )
    return jax.tree.map(lambda a, b: jnp.where(street == 0, a, b), pre, post)


def bet_state_from_env(game: HoldemGame, env_state) -> BetState:
    """Public betting state of an env state (reads public fields only)."""
    sb = env_state.small_blind_idx
    seats = jnp.stack([sb, 1 - sb])
    stage = env_state.stage
    if game.is_limit:
        round_bets = jnp.zeros(2)
        raises = env_state.raise_nums[jnp.minimum(stage, 3)]
        allin = jnp.zeros(2, dtype=bool)
    else:
        round_bets = env_state.round_raised[seats].astype(jnp.float32)
        raises = jnp.int32(0)
        allin = env_state.all_in[seats]
    return BetState(
        chips=env_state.chips_in[seats].astype(jnp.float32),
        round_bets=round_bets,
        raises=jnp.asarray(raises, jnp.int32),
        nrn=jnp.asarray(env_state.not_raise_num, jnp.int32),
        actor=((env_state.current_player_idx - sb) % 2).astype(jnp.int32),
        street=jnp.asarray(stage, jnp.int32),
        allin=allin,
    )


def legal_actions(game: HoldemGame, s: BetState) -> BoolArray:
    """Env-legal actions, without folds when checking is free."""
    p = s.actor
    if game.is_limit:
        behind = s.chips[p] < jnp.max(s.chips)
        return jnp.array([behind, s.raises < 4, behind, ~behind])
    player_round = s.round_bets[p]
    max_round = jnp.max(s.round_bets)
    remain = game.stack - s.chips[p]
    pot = jnp.sum(s.chips)
    half = jnp.floor(pot / 2.0)
    active = (~s.allin[p]) & (remain > 0)
    diff = max_round - player_round
    can_any = diff < remain
    can_pot = can_any & (pot <= remain)
    can_half = can_any & (half <= remain) & ((half + player_round) > max_round)
    behind = player_round < max_round
    return jnp.array(
        [active, active & can_half, active & can_pot, active & can_any,
         active & behind]
    )


def slot_action(game: HoldemGame, s: BetState, slot: IntArray) -> IntArray:
    """Env action a slot stands for in state s."""
    if game.is_limit:
        behind = s.chips[s.actor] < jnp.max(s.chips)
        passive = jnp.where(behind, 0, 3)
        return jnp.where(slot == PASSIVE_SLOT, passive, slot)
    return slot


def legal_slots(game: HoldemGame, s: BetState) -> BoolArray:
    legal = legal_actions(game, s)
    slots = jnp.zeros(game.num_slots, bool)
    for a, k in enumerate(game.env_slot):
        slots = slots.at[k].set(slots[k] | legal[a])
    return slots


def step(game: HoldemGame, s: BetState, a: IntArray) -> tuple[BetState, IntArray]:
    """Apply env action a for the player to act. Returns (state, outcome)."""
    p = s.actor
    if game.is_limit:
        max_c = jnp.max(s.chips)
        raise_amount = jnp.where(s.street >= 2, 4.0, 2.0)
        new_c = jnp.where(a == 1, max_c + raise_amount, max_c)
        chips = jnp.where((a == 0) | (a == 1), s.chips.at[p].set(new_c), s.chips)
        raises = s.raises + (a == 1)
        nrn = jnp.where(a == 1, 1, jnp.where((a == 0) | (a == 3), s.nrn + 1, s.nrn))
        closed = nrn >= 2
        allin = s.allin
        round_bets = s.round_bets
    else:
        player_round = s.round_bets[p]
        max_round = jnp.max(s.round_bets)
        remain = game.stack - s.chips[p]
        pot = jnp.sum(s.chips)
        half = jnp.floor(pot / 2.0)
        bet = jnp.select(
            [a == 0, a == 1, a == 2, a == 3],
            [max_round - player_round, half, pot, remain],
            0.0,
        )
        chips = s.chips.at[p].add(bet)
        round_bets = s.round_bets.at[p].add(bet)
        now_allin = (game.stack - chips[p]) <= 0
        allin = jnp.where(a == 4, s.allin, s.allin.at[p].set(now_allin))
        nrn = jnp.where(
            a == 0,
            jnp.where(now_allin, s.nrn, s.nrn + 1),
            jnp.where(now_allin, 0, 1),
        )
        nrn = jnp.where(a == 4, s.nrn, nrn)
        raises = s.raises  # No-Limit has no raise cap; the tree caps raises itself
        playable = jnp.sum(~allin)
        closed = nrn >= playable
    outcome = jnp.where(a == game.fold_action, FOLDED, jnp.where(closed, CLOSED, CONTINUE))
    new = BetState(
        chips=chips,
        round_bets=round_bets,
        raises=jnp.asarray(raises, jnp.int32),
        nrn=jnp.asarray(nrn, jnp.int32),
        actor=1 - p,
        street=s.street,
        allin=allin,
    )
    return new, outcome


# =============================================================================
# Static template of action strings
# =============================================================================


class Template(NamedTuple):
    """Slot strings a betting-round subgame can contain (numpy, static).

    Nodes are in breadth-first order, so each depth is a contiguous range of
    node ids, and internal nodes (those with children) of each depth are a
    contiguous range of internal ids.
    """

    parent: np.ndarray  # (N,) -1 for the root
    action: np.ndarray  # (N,) slot leading to the node
    depth: np.ndarray  # (N,)
    children: np.ndarray  # (N, S) template node or -1
    internal: np.ndarray  # (Ni,) template ids of nodes that have children
    internal_id: np.ndarray  # (N,) index into internal, -1 otherwise
    node_start: np.ndarray  # (D + 2,) first node of each depth
    internal_start: np.ndarray  # (D + 2,) first internal id of each depth
    passive_nodes: np.ndarray  # (L,) nodes reached by check/call
    fold_nodes: np.ndarray  # (F,) nodes reached by fold
    max_depth: int


def build_template(game: HoldemGame) -> Template:
    nodes = [()]
    i = 0
    while i < len(nodes):
        s = nodes[i]
        i += 1
        if s and s[-1] == game.fold_slot:
            continue
        if s and s[-1] == PASSIVE_SLOT and len(s) > 1:
            continue  # a check/call after the first action closes the round
        if s and s[-1] == game.allin_slot:
            options = [PASSIVE_SLOT, game.fold_slot]  # only call or fold an all-in
        else:
            options = list(range(game.num_slots))
        n_raises = sum(a in game.raise_slots for a in s)
        for a in options:
            if a in game.raise_slots and n_raises >= game.max_raises:
                continue
            if a == game.allin_slot and game.allin_slot in s:
                continue
            nodes.append(s + (a,))
    index = {s: k for k, s in enumerate(nodes)}
    n = len(nodes)
    parent = np.array([index[s[:-1]] if s else -1 for s in nodes], np.int32)
    action = np.array([s[-1] if s else -1 for s in nodes], np.int32)
    depth = np.array([len(s) for s in nodes], np.int32)
    assert np.all(np.diff(depth) >= 0), "template must be breadth-first"
    children = -np.ones((n, game.num_slots), np.int32)
    for k, s in enumerate(nodes):
        if s:
            children[parent[k], s[-1]] = k
    internal = np.nonzero((children >= 0).any(axis=1))[0].astype(np.int32)
    internal_id = -np.ones(n, np.int32)
    internal_id[internal] = np.arange(len(internal))
    max_depth = int(depth.max())
    node_start = np.searchsorted(depth, np.arange(max_depth + 2)).astype(np.int32)
    internal_start = np.searchsorted(depth[internal], np.arange(max_depth + 2)).astype(
        np.int32
    )
    passive = np.array(
        [k for k, s in enumerate(nodes) if s and s[-1] == PASSIVE_SLOT], np.int32
    )
    folds = np.array(
        [k for k, s in enumerate(nodes) if s and s[-1] == game.fold_slot], np.int32
    )
    return Template(
        parent=parent,
        action=action,
        depth=depth,
        children=children,
        internal=internal,
        internal_id=internal_id,
        node_start=node_start,
        internal_start=internal_start,
        passive_nodes=passive,
        fold_nodes=folds,
        max_depth=max_depth,
    )


# =============================================================================
# Subgame tree for a concrete root
# =============================================================================


class Tree(NamedTuple):
    """Betting-round subgame for one root, laid out on the template."""

    kind: IntArray  # (N,) INVALID / DECISION / FOLD / ROUND_END
    chips: FloatArray  # (N, 2) chips in the pot by position at each node
    allin: BoolArray  # (N,) both players all-in (No-Limit)
    legal: BoolArray  # (Ni, S) slots in the tree at each decision node
    actor: IntArray  # (N,) position acting at the node (meaningful at decisions)
    street: IntArray  # ()
    template_gap: BoolArray  # () a continuing node had no template children


def build_tree(game: HoldemGame, tpl: Template, root: BetState) -> Tree:
    n = len(tpl.parent)
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)), root)
    kind = jnp.zeros(n, jnp.int32).at[0].set(DECISION)
    for d in range(1, tpl.max_depth + 1):
        lo, hi = int(tpl.node_start[d]), int(tpl.node_start[d + 1])
        par = tpl.parent[lo:hi]
        slot = jnp.asarray(tpl.action[lo:hi])
        par_states = jax.tree.map(lambda x: x[par], states)
        legal = jax.vmap(lambda s: legal_slots(game, s))(par_states)
        ok = (kind[par] == DECISION) & jnp.take_along_axis(legal, slot[:, None], 1)[:, 0]
        actions = jax.vmap(lambda s, k: slot_action(game, s, k))(par_states, slot)
        child_states, outcome = jax.vmap(lambda s, a: step(game, s, a))(
            par_states, actions
        )
        child_kind = jnp.select(
            [outcome == FOLDED, outcome == CLOSED], [FOLD, ROUND_END], DECISION
        )
        kind = kind.at[lo:hi].set(jnp.where(ok, child_kind, INVALID))
        states = jax.tree.map(lambda s, c: s.at[lo:hi].set(c), states, child_states)
    is_internal = jnp.zeros(n, bool).at[tpl.internal].set(True)
    # A continuing node must have children in the template. The template is
    # built so that this always holds; the tests check template_gap.
    gap = jnp.any((kind == DECISION) & ~is_internal)
    kind = jnp.where((kind == DECISION) & ~is_internal, INVALID, kind)
    internal_states = jax.tree.map(lambda x: x[tpl.internal], states)
    legal = jax.vmap(lambda s: legal_slots(game, s))(internal_states)
    legal = legal & (tpl.children[tpl.internal] >= 0)
    legal = legal & (kind[tpl.internal] == DECISION)[:, None]
    return Tree(
        kind=kind,
        chips=states.chips,
        allin=states.allin[:, 0] & states.allin[:, 1],
        legal=legal,
        actor=states.actor,
        street=root.street,
        template_gap=gap,
    )


# =============================================================================
# Public features of a PBS (value-network input, without the beliefs)
# =============================================================================

PUBLIC_DIM = NUM_CARDS + 4 + 3


def num_board_cards(street: IntArray) -> IntArray:
    return jnp.select([street == 0, street == 1, street == 2], [0, 3, 4], 5)


def public_features(
    game: HoldemGame,
    street: IntArray,
    board: IntArray,
    end_of_round: BoolArray,
    allin: BoolArray,
    chips: FloatArray,
) -> FloatArray:
    """Board cards, street, start/end of round, all-in, contribution / scale.

    PBSs the network sees are at round boundaries, where both players have the
    same contribution `chips` (per player, in chips).
    """
    return jnp.concatenate(
        [
            cards_onehot(board, num_board_cards(street)).astype(jnp.float32),
            jax.nn.one_hot(street, 4, dtype=jnp.float32),
            jnp.stack(
                [
                    jnp.asarray(end_of_round, jnp.float32),
                    jnp.asarray(allin, jnp.float32),
                    chips * game.reward_per_chip / game.value_scale,
                ]
            ),
        ]
    )
