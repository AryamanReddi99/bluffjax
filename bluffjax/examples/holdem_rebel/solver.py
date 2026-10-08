"""
Value network and subgame solving for heads-up Hold'em ReBeL.

Units: all values are in env reward units divided by `game.value_scale`
(the largest possible contribution), so they lie in [-1, 1].

A PBS is a public state plus each player's normalised reach over the 1,326
hands (zero for hands blocked by the board). The joint deal is
P(h0, h1) ~ x0(h0) x1(h1) [h0, h1 and the board share no card].

The network maps a PBS to per-hand values for both players: the expected
payoff of holding h given the PBS, when both play an equilibrium of the
subgame below it. In CFR these become counterfactual values by multiplying
with the opponent's compatible reach mass.
"""

import itertools
from typing import Callable, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from bluffjax.examples.holdem_rebel.cards import (
    NUM_HANDS,
    ShowdownTables,
    cards_onehot,
    compatible_mass,
    hand_strengths,
    hands_not_blocked,
    normalize,
    rank5_table,
    showdown_tables,
    showdown_value,
)
from bluffjax.examples.holdem_rebel.game import (
    DECISION,
    FOLD,
    ROUND_END,
    BetState,
    HoldemGame,
    Template,
    Tree,
    build_tree,
    num_board_cards,
    public_features,
)
from bluffjax.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray

# (pub (..., PUBLIC_DIM), beliefs (..., 2, 1326)) -> values (..., 2, 1326)
ValueFn = Callable[[FloatArray, FloatArray], FloatArray]

# Hands whose compatible opponent mass is below this get no value target.
MIN_MASS = 1e-6


class ValueNetwork(nn.Module):
    """MLP with LayerNorm and GeLU (as in ReBeL), both players' values out."""

    hidden_dim: int = 512
    num_layers: int = 3

    @nn.compact
    def __call__(self, pub: FloatArray, beliefs: FloatArray) -> FloatArray:
        lead = beliefs.shape[:-2]
        # Beliefs are probabilities over 1,326 hands; scale so uniform is ~1.
        x = jnp.concatenate(
            [pub, beliefs.reshape(lead + (2 * NUM_HANDS,)) * NUM_HANDS], axis=-1
        )
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)
        out = nn.Dense(
            2 * NUM_HANDS,
            kernel_init=nn.initializers.variance_scaling(1e-4, "fan_in", "normal"),
            bias_init=nn.initializers.zeros,
        )(x)
        return out.reshape(lead + (2, NUM_HANDS))


def regret_matching(regrets: FloatArray, legal: BoolArray) -> FloatArray:
    """regrets (..., S, H), legal (..., S) -> policy (..., S, H)."""
    legal = legal[..., None]
    pos = jnp.where(legal, jnp.maximum(regrets, 0.0), 0.0)
    total = jnp.sum(pos, axis=-2, keepdims=True)
    n_legal = jnp.maximum(jnp.sum(legal, axis=-2, keepdims=True), 1)
    uniform = jnp.where(legal, 1.0 / n_legal, 0.0)
    return jnp.where(total > 0.0, pos / jnp.where(total > 0.0, total, 1.0), uniform)


def uniform_policy(legal: BoolArray) -> FloatArray:
    return regret_matching(jnp.zeros(legal.shape + (NUM_HANDS,)), legal)


def normalize_policy(sum_policy: FloatArray, legal: BoolArray) -> FloatArray:
    total = jnp.sum(sum_policy, axis=-2, keepdims=True)
    uniform = uniform_policy(legal)
    return jnp.where(total > 0.0, sum_policy / jnp.where(total > 0.0, total, 1.0), uniform)


class Subgame(NamedTuple):
    """Everything the solver needs about one betting-round subgame."""

    tree: Tree
    beliefs: FloatArray  # (2, 1326) root reach, normalised, board-masked
    hand_ok: BoolArray  # (1326,) hands not blocked by the board
    is_river: BoolArray  # ()
    showdown: ShowdownTables  # for the river board (unused before the river)
    leaf_pub: FloatArray  # (L, PUBLIC_DIM) features of the end-of-round leaves


class Solution(NamedTuple):
    values: FloatArray  # (2, 1326) root values per hand (value-net target)
    value_mask: BoolArray  # (2, 1326) where the target is defined
    policy: FloatArray  # (Ni, S, 1326) policy of the sampled iteration
    avg_policy: FloatArray  # (Ni, S, 1326) average policy
    tree: Tree


def make_subgame(
    game: HoldemGame,
    tpl: Template,
    root: BetState,
    board: IntArray,
    beliefs: FloatArray,
    with_showdown: bool = True,
) -> Subgame:
    """Subgame at root. board has -1 for undealt cards; beliefs by position."""
    tree = build_tree(game, tpl, root)
    street = root.street
    board = jnp.where(board >= 0, board, 0).astype(jnp.int32)
    hand_ok = hands_not_blocked(cards_onehot(board, num_board_cards(street)))
    beliefs = normalize(beliefs * hand_ok)
    if with_showdown:
        tables = showdown_tables(hand_strengths(rank5_table(), board))
    else:
        zeros = jnp.zeros(NUM_HANDS, jnp.int32)
        tables = ShowdownTables(
            zeros, zeros, zeros, jnp.zeros((52, 51), jnp.int32),
            jnp.zeros((NUM_HANDS, 2), jnp.int32), jnp.zeros((NUM_HANDS, 2), jnp.int32),
        )
    passive = tpl.passive_nodes
    leaf_pub = jax.vmap(
        lambda allin, chips: public_features(game, street, board, True, allin, chips)
    )(tree.allin[passive], tree.chips[passive, 0])
    return Subgame(
        tree=tree,
        beliefs=beliefs,
        hand_ok=hand_ok,
        is_river=street == 3,
        showdown=tables,
        leaf_pub=leaf_pub,
    )


class _Ops:
    """Tree passes for one template, on per-depth arrays (no full-tree scatters).

    Node ids of depth d are the range [node_start[d], node_start[d + 1]) and
    internal ids of depth d are [internal_start[d], internal_start[d + 1]),
    so every pass works on contiguous slices and small static gathers.
    """

    def __init__(self, game: HoldemGame, tpl: Template):
        self.game, self.tpl = game, tpl
        D = tpl.max_depth
        ns, ist = tpl.node_start, tpl.internal_start
        self.D = D
        self.node_rng = [(int(ns[d]), int(ns[d + 1])) for d in range(D + 1)]
        self.int_rng = [(int(ist[d]), int(ist[d + 1])) for d in range(D + 1)]
        self.par_node = [None]
        self.par_int = [None]
        self.act = [None]
        for d in range(1, D + 1):
            lo, hi = self.node_rng[d]
            par = tpl.parent[lo:hi]
            self.par_node.append(par - self.node_rng[d - 1][0])
            self.par_int.append(tpl.internal_id[par] - self.int_rng[d - 1][0])
            self.act.append(tpl.action[lo:hi])
        self.int_local = []
        self.child_local = []
        for d in range(D + 1):
            ilo, ihi = self.int_rng[d]
            nodes = tpl.internal[ilo:ihi]
            self.int_local.append(nodes - self.node_rng[d][0])
            if d < D:
                ch = tpl.children[nodes]
                nlo, nhi = self.node_rng[d + 1]
                self.child_local.append(np.where(ch >= 0, ch - nlo, nhi - nlo))
            else:
                self.child_local.append(None)
        n = len(tpl.parent)
        f, p = len(tpl.fold_nodes), len(tpl.passive_nodes)
        src = np.zeros(n, np.int32)
        src[tpl.fold_nodes] = 1 + np.arange(f)
        src[tpl.passive_nodes] = 1 + f + np.arange(p)
        self.leaf_src = src

    def reaches(self, policy: FloatArray, beliefs: FloatArray, actor: IntArray):
        """Per-depth reach of both players, [(n_d, 2, 1326)]."""
        xs = [beliefs[None]]
        for d in range(1, self.D + 1):
            ilo, ihi = self.int_rng[d - 1]
            f = policy[ilo:ihi][self.par_int[d], self.act[d]]  # (n_d, 1326)
            a = actor[ilo:ihi][self.par_int[d]]
            mine = a[:, None] == jnp.arange(2)[None, :]
            xs.append(
                xs[d - 1][self.par_node[d]] * jnp.where(mine[:, :, None], f[:, None], 1.0)
            )
        return xs

    def internal_reach(self, xs) -> FloatArray:
        """(Ni, 2, 1326) reach at internal nodes."""
        parts = [xs[d][self.int_local[d]] for d in range(self.D + 1) if len(self.int_local[d])]
        return jnp.concatenate(parts, axis=0)

    def leaf_values(
        self, sg: Subgame, xs, player: IntArray, value_fn: ValueFn, leaf_mode: str
    ) -> list:
        """Per-depth counterfactual values of `player` at folds and round ends."""
        game, tpl = self.game, self.tpl
        scale = game.reward_per_chip / game.value_scale
        x = jnp.concatenate(xs, axis=0)  # (N, 2, 1326)
        folds, passive = tpl.fold_nodes, tpl.passive_nodes
        xo_f = x[folds, 1 - player]
        folder = sg.tree.actor[tpl.parent[folds]]
        chips_f = sg.tree.chips[folds]
        pay = jnp.where(
            folder == player,
            -chips_f[:, player],
            jnp.take_along_axis(chips_f, folder[:, None], 1)[:, 0],
        ) * scale
        v_fold = pay[:, None] * compatible_mass(xo_f)
        v_fold = jnp.where((sg.tree.kind[folds] == FOLD)[:, None], v_fold, 0.0)

        x_p = x[passive]
        xo_p = x_p[:, 1 - player]
        if leaf_mode in ("net", "both"):
            net = value_fn(sg.leaf_pub, normalize(x_p))[:, player]
            v_net = net * compatible_mass(xo_p)
        if leaf_mode in ("showdown", "both"):
            sd = showdown_value(xo_p, sg.showdown)
            v_sd = sd * (sg.tree.chips[passive, 0] * scale)[:, None]
        v_end = {"net": lambda: v_net, "showdown": lambda: v_sd,
                 "both": lambda: jnp.where(sg.is_river, v_sd, v_net)}[leaf_mode]()
        v_end = jnp.where((sg.tree.kind[passive] == ROUND_END)[:, None], v_end, 0.0)
        v = jnp.concatenate([jnp.zeros((1, NUM_HANDS)), v_fold, v_end])[self.leaf_src]
        v = v * sg.hand_ok
        return [v[lo:hi] for lo, hi in self.node_rng]

    def backward(
        self,
        sg: Subgame,
        v_leaf: list,
        policy: FloatArray,
        actor: IntArray,
        player: IntArray,
        best_response: bool = False,
    ) -> tuple[FloatArray, FloatArray]:
        """Root values (1326,) and per-action regrets (Ni, S, 1326) of player."""
        tpl = self.tpl
        is_dec = sg.tree.kind[tpl.internal] == DECISION
        legal = sg.tree.legal
        v_next = v_leaf[self.D]
        insts = []
        for d in reversed(range(self.D)):
            ilo, ihi = self.int_rng[d]
            vl = v_leaf[d]
            if ihi == ilo:
                v_next = vl
                continue
            vpad = jnp.concatenate([v_next, jnp.zeros((1, NUM_HANDS))], axis=0)
            vc = vpad[self.child_local[d]]  # (ni_d, S, 1326)
            lg = legal[ilo:ihi][:, :, None]
            mine = (actor[ilo:ihi] == player)[:, None, None]
            if best_response:
                best = jnp.max(jnp.where(lg, vc, -jnp.inf), axis=1)
                summed = jnp.sum(jnp.where(lg, vc, 0.0), axis=1)
                val = jnp.where(mine[:, 0], best, summed)
                val = jnp.where(jnp.isfinite(val), val, 0.0)
            else:
                val = jnp.sum(jnp.where(mine, policy[ilo:ihi], 1.0) * lg * vc, axis=1)
            loc = self.int_local[d]
            val = jnp.where(is_dec[ilo:ihi][:, None], val, vl[loc])
            insts.append(vc - val[:, None, :])
            v_next = vl.at[loc].set(val)
        inst = jnp.concatenate(insts[::-1], axis=0)
        return v_next[0], inst


def solve(
    game: HoldemGame,
    tpl: Template,
    sg: Subgame,
    value_fn: ValueFn,
    rng: PRNGKeyArray,
    cfr_iters: int,
    leaf_mode: str = "both",
) -> Solution:
    """Alternating-update Linear CFR-D on one subgame (ReBeL Algorithm 2).

    Each iteration updates the small blind, then the big blind (as in the
    released ReBeL code). Leaf values come from the value network at the leaf
    PBS reached by the current policy (CFR-D). Regrets and the average policy
    use linear weights and the root values are averaged with linear weights.
    The policy after iteration t ~ P(t) proportional to t + 1 (t in
    0..cfr_iters-1, t = 0 is uniform) is returned for sampling the next PBS
    and for acting (safe search, Sec. 6).

    leaf_mode: "net" (no river), "showdown" (river) or "both" (select by street).
    """
    ops = _Ops(game, tpl)
    actor = sg.tree.actor[tpl.internal]
    is_dec = sg.tree.kind[tpl.internal] == DECISION
    legal = sg.tree.legal

    def own_reach(xs):  # (Ni, 1, 1326) acting player's reach at internal nodes
        xi = ops.internal_reach(xs)
        return jnp.take_along_axis(xi, actor[:, None, None], axis=1)

    uniform = uniform_policy(legal)
    sum_policy = uniform * own_reach(ops.reaches(uniform, sg.beliefs, actor))

    num_steps = 2 * cfr_iters
    k_sample = jax.random.categorical(
        rng, jnp.log(jnp.arange(1, cfr_iters + 1, dtype=jnp.float32))
    )

    def add_average(sum_policy, policy, xs, player, s_update):
        # Linear averaging of `player`'s policy computed at step s_update.
        t = jnp.asarray(s_update // 2 + 1, jnp.float32)
        own = (is_dec & (actor == player))[:, None, None]
        return jnp.where(own, sum_policy * (t / (t + 1.0)) + policy * own_reach(xs), sum_policy)

    def body(carry, s):
        regrets, policy, sum_policy, root_mean, snap = carry
        player = s % 2
        n_prev = s // 2
        snap = jnp.where(s == 2 * k_sample, policy, snap)
        xs = ops.reaches(policy, sg.beliefs, actor)
        # The other player updated at s - 1; its new reach is in xs.
        sum_policy = jnp.where(
            s > 0, add_average(sum_policy, policy, xs, 1 - player, s - 1), sum_policy
        )
        v_leaf = ops.leaf_values(sg, xs, player, value_fn, leaf_mode)
        root, inst = ops.backward(sg, v_leaf, policy, actor, player)
        alpha = 2.0 / (n_prev + 2.0)
        root_mean = root_mean.at[player].add((root - root_mean[player]) * alpha)
        own = (is_dec & (actor == player))[:, None, None]
        regrets = jnp.where(own & legal[:, :, None], regrets + inst, regrets)
        t = (n_prev + 1.0).astype(jnp.float32)
        policy = jnp.where(own, regret_matching(regrets, legal), policy)
        regrets = jnp.where(own, regrets * (t / (t + 1.0)), regrets)
        return (regrets, policy, sum_policy, root_mean, snap), None

    init = (jnp.zeros_like(uniform), uniform, sum_policy, jnp.zeros((2, NUM_HANDS)), uniform)
    (_, policy, sum_policy, root_mean, snap), _ = lax.scan(
        body, init, jnp.arange(num_steps)
    )
    last = num_steps - 1
    xs = ops.reaches(policy, sg.beliefs, actor)
    sum_policy = add_average(sum_policy, policy, xs, last % 2, last)
    mass = compatible_mass(sg.beliefs[::-1])  # (2, 1326) opponent mass per hand
    mask = sg.hand_ok[None, :] & (mass > MIN_MASS)
    values = jnp.where(mask, root_mean / jnp.where(mask, mass, 1.0), 0.0)
    return Solution(
        values=values,
        value_mask=mask,
        policy=snap,
        avg_policy=normalize_policy(sum_policy, legal),
        tree=sg.tree,
    )


def solve_batch(
    game: HoldemGame,
    tpl: Template,
    roots: BetState,
    boards: IntArray,
    beliefs: FloatArray,
    value_fn: ValueFn,
    rng: PRNGKeyArray,
    cfr_iters: int,
    chunk: int,
    by_street: bool = True,
) -> tuple[Solution, FloatArray]:
    """Solve a batch of subgames, `chunk` at a time. Returns (solutions, root beliefs).

    With by_street, subgames are ordered so that chunks without a river
    subgame skip the showdown code and the others evaluate both leaf kinds.
    This branches on chunk contents, so use by_street=False under vmap.
    """
    n = beliefs.shape[0]
    keys = jax.random.split(rng, n)

    def one(args, mode):
        root, board, belief, key = args
        sg = make_subgame(game, tpl, root, board, belief, with_showdown=mode != "net")
        return solve(game, tpl, sg, value_fn, key, cfr_iters, mode), sg.beliefs

    args = (roots, boards, beliefs, keys)
    if not by_street:
        return lax.map(lambda a: one(a, "both"), args, batch_size=chunk)
    n_chunks = -(-n // chunk)
    order = jnp.argsort(roots.street == 3, stable=True)
    pad = jnp.concatenate([order, jnp.full(n_chunks * chunk - n, order[-1])])
    chunked = jax.tree.map(lambda x: x[pad].reshape((n_chunks, chunk) + x.shape[1:]), args)

    def run_chunk(c):
        has_river = jnp.any(c[0].street == 3)
        return lax.cond(
            has_river,
            lambda: jax.vmap(lambda *a: one(a, "both"))(*c),
            lambda: jax.vmap(lambda *a: one(a, "net"))(*c),
        )

    out = lax.map(run_chunk, chunked)
    out = jax.tree.map(lambda x: x.reshape((n_chunks * chunk,) + x.shape[2:])[:n], out)
    inverse = jnp.argsort(order)
    return jax.tree.map(lambda x: x[inverse], out)


def evaluate_profile(
    game: HoldemGame,
    tpl: Template,
    sg: Subgame,
    policy: FloatArray,
    value_fn: ValueFn,
) -> tuple[FloatArray, FloatArray]:
    """(values of the profile, best-response values), each (2,), reward units.

    Values are the players' expected payoffs at the root PBS. With exact
    leaves (river subgames) the exploitability of the profile is mean(br),
    since the game is zero-sum.
    """
    ops = _Ops(game, tpl)
    actor = sg.tree.actor[tpl.internal]
    xs = ops.reaches(policy, sg.beliefs, actor)
    mass = compatible_mass(sg.beliefs[::-1])
    z = jnp.sum(sg.beliefs[0] * mass[0])

    def per_player(p):
        v = ops.leaf_values(sg, xs, p, value_fn, "both")
        on = ops.backward(sg, v, policy, actor, p)[0]
        br = ops.backward(sg, v, policy, actor, p, best_response=True)[0]
        return jnp.sum(sg.beliefs[p] * on) / z, jnp.sum(sg.beliefs[p] * br) / z

    on, br = jax.vmap(per_player)(jnp.arange(2))
    return on * game.value_scale, br * game.value_scale


# =============================================================================
# Chance nodes between betting rounds
# =============================================================================


def _all_flops() -> np.ndarray:
    return np.array(list(itertools.combinations(range(52), 3)), np.int32)


class ChanceChildren(NamedTuple):
    cards: IntArray  # (K, 3) cards added to the board, -1 where unused
    ok: BoolArray  # (K,) child is a legal deal given the board


def chance_children(
    street: IntArray, board: IntArray, rng: PRNGKeyArray, num_flops: int
) -> ChanceChildren:
    """Next-street deals from `street` (0 preflop, 1 flop, 2 turn).

    Turn and river cards are enumerated (52 entries, board cards excluded).
    Flops are either all 22,100 (num_flops=0) or num_flops sampled uniformly.
    """
    if num_flops == 0:
        flops = jnp.asarray(_all_flops())
    else:
        keys = jax.random.split(rng, num_flops)
        flops = jax.vmap(lambda k: jax.random.choice(k, 52, (3,), replace=False))(
            keys
        ).astype(jnp.int32)
    n_flops = flops.shape[0]
    k = max(n_flops, 52)
    flops = jnp.concatenate([flops, -jnp.ones((k - n_flops, 3), jnp.int32)])
    singles = jnp.full((k, 3), -1, jnp.int32).at[:52, 0].set(jnp.arange(52))
    on_board = cards_onehot(jnp.where(board >= 0, board, 0), num_board_cards(street))
    single_ok = (jnp.arange(k) < 52) & ~on_board[jnp.clip(jnp.arange(k), 0, 51)]
    flop_ok = jnp.arange(k) < n_flops
    is_pre = street == 0
    return ChanceChildren(
        cards=jnp.where(is_pre, flops, singles),
        ok=jnp.where(is_pre, flop_ok, single_ok),
    )


def add_cards(board: IntArray, street: IntArray, cards: IntArray) -> IntArray:
    """Board after dealing `cards` (-1 = none) at the next free positions."""
    nb = num_board_cards(street)
    for j in range(3):
        pos = jnp.clip(nb + j, 0, 4)
        board = jnp.where(cards[j] >= 0, board.at[pos].set(cards[j]), board)
    return board


class ChanceResult(NamedTuple):
    values: FloatArray  # (2, 1326) pre-chance values per hand (target)
    value_mask: BoolArray  # (2, 1326)


def chance_values(
    game: HoldemGame,
    value_fn: ValueFn,
    street: IntArray,
    board: IntArray,
    chips: FloatArray,
    allin: BoolArray,
    beliefs: FloatArray,
    children: ChanceChildren,
    exact_showdown: bool,
) -> ChanceResult:
    """Value of a PBS at the end of a betting round, before the next cards.

    Chance deals each next-street card set c not on the board, uniformly
    among the cards left after both hands. For player i holding h:
        cfv_pre(h) = sum_c cfv_c(h) / N,  cfv_c(h) = m_c(h) v_c(h),
    with m_c(h) the opponent mass compatible with h and c, and v_c the value
    of the child PBS. Normalising by the same sum of masses gives the
    per-hand value (the weights are P(c | h, opponent beliefs)). With sampled
    flops this is the self-normalised Monte Carlo estimate.

    Children are the start of the next round (value network), or for
    all-in hands the end of the next round (value network) or, once the
    river is dealt, an exact showdown (exact_showdown=True).
    """
    scale = game.reward_per_chip / game.value_scale
    board = jnp.where(board >= 0, board, 0)
    hand_ok = hands_not_blocked(cards_onehot(board, num_board_cards(street)))
    beliefs = normalize(beliefs * hand_ok)
    next_street = street + 1

    def child(cards):
        dealt = cards_onehot(jnp.where(cards >= 0, cards, 0), jnp.sum(cards >= 0))
        keep = hands_not_blocked(dealt)
        x = beliefs * keep
        mass = compatible_mass(x[::-1]) * keep  # (2, 1326)
        new_board = add_cards(board, street, cards)
        if exact_showdown:
            tables = showdown_tables(hand_strengths(rank5_table(), new_board))
            cfv = showdown_value(x[::-1], tables) * keep * (chips * scale)
        else:
            pub = public_features(game, next_street, new_board, allin, allin, chips)
            cfv = value_fn(pub, normalize(x)) * mass
        return cfv, mass

    cfv, mass = jax.vmap(child)(children.cards)
    w = children.ok[:, None, None]
    num = jnp.sum(jnp.where(w, cfv, 0.0), axis=0)
    den = jnp.sum(jnp.where(w, mass, 0.0), axis=0)
    mask = hand_ok[None, :] & (den > MIN_MASS)
    values = jnp.where(mask, num / jnp.where(mask, den, 1.0), 0.0)
    return ChanceResult(values=values, value_mask=mask)
