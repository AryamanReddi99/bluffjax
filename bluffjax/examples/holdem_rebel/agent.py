"""
Test-time ReBeL agent, simple opponents, mirrored matches and checkpoints.

The agent only sees a PublicView of the env state: the betting state, the
board cards dealt so far, and its own hole cards. It plays as in ReBeL
Section 6 (safe search): at the start of every betting round it solves the
subgame rooted at the current PBS, samples one CFR iteration t (linearly
weighted) and plays that iteration's policy for the rest of the round,
updating both players' beliefs with it after every action. If the opponent
takes an action that is not in the subgame (No-Limit raises beyond the tree's
raise cap), it solves a new subgame rooted just before that action, which
contains it, and continues from there.

play_match runs a batch of hands between two players with every deal played
twice, once from each seat (duplicate poker), so luck of the cards largely
cancels in the paired results.
"""

import json
import os
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from jax import lax

from bluffjax.examples.holdem_rebel.cards import NUM_HANDS, hand_index, normalize
from bluffjax.examples.holdem_rebel.game import (
    DECISION,
    PUBLIC_DIM,
    BetState,
    HoldemGame,
    Tree,
    bet_state_from_env,
    build_template,
    make_game,
    num_board_cards,
    slot_action,
)
from bluffjax.examples.holdem_rebel.solver import ValueNetwork, solve_batch
from bluffjax.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray


# =============================================================================
# What a player may see
# =============================================================================


class PublicView(NamedTuple):
    """Public information plus the viewer's own cards."""

    bet: BetState
    board: IntArray  # (5,) dealt board cards, -1 for cards not dealt yet
    stage: IntArray
    current_seat: IntArray
    small_blind: IntArray
    own_hand: IntArray  # hand id of the viewer's hole cards
    done: BoolArray


def public_view(game: HoldemGame, env_state, seat: IntArray) -> PublicView:
    stage = env_state.stage
    board = jnp.concatenate(
        [env_state.flop_cards, jnp.stack([env_state.turn_card, env_state.river_card])]
    ).astype(jnp.int32)
    dealt = jnp.arange(5) < num_board_cards(jnp.minimum(stage, 3))
    return PublicView(
        bet=bet_state_from_env(game, env_state),
        board=jnp.where(dealt, board, -1),
        stage=stage,
        current_seat=env_state.current_player_idx,
        small_blind=env_state.small_blind_idx,
        own_hand=hand_index(env_state.agent_cards[seat].astype(jnp.int32)),
        done=env_state.done,
    )


# =============================================================================
# Players
# =============================================================================


class Player:
    """Interface used by play_match. Methods act on a batch of hands."""

    stateful = False

    def init(self, n: int) -> Any:
        return ()

    def solve(self, params, state, views: PublicView, need: BoolArray, rng):
        return state

    def act(self, params, state, views: PublicView, rng, env_states, avail):
        raise NotImplementedError

    def offtree(self, state, views: PublicView, actions: IntArray) -> BoolArray:
        return jnp.zeros(actions.shape, bool)

    def observe(self, state, views: PublicView, actions: IntArray, go: BoolArray):
        return state


class PolicyPlayer(Player):
    """Stateless player: fn(params, rng, env_state, avail) -> action, per hand."""

    def __init__(self, fn: Callable):
        self.fn = fn

    def act(self, params, state, views, rng, env_states, avail):
        keys = jax.random.split(rng, avail.shape[0])
        return jax.vmap(lambda k, s, a: self.fn(params, k, s, a))(
            keys, env_states, avail
        )


def _heuristic_fns(game: HoldemGame) -> dict[str, Callable]:
    def random_fn(params, rng, s, avail):
        return jax.random.categorical(rng, jnp.where(avail, 0.0, -1e9))

    def first_legal(order):
        def fn(params, rng, s, avail):
            order_a = jnp.asarray(order)
            return order_a[jnp.argmax(avail[order_a])]

        return fn

    fns = {"random": random_fn}
    if game.is_limit:  # 0 call, 1 raise, 2 fold, 3 check
        fns["always_call"] = first_legal([0, 3])
        fns["always_raise"] = first_legal([1, 0, 3])
    else:  # 0 check/call, 1 half pot, 2 pot, 3 all-in, 4 fold
        fns["always_call"] = first_legal([0])
        fns["always_raise"] = first_legal([2, 1, 3, 0])
        fns["always_allin"] = first_legal([3, 0])
    return fns


def heuristic_players(game: HoldemGame) -> dict[str, PolicyPlayer]:
    """Uniform random, always check/call, always raise (and shove in No-Limit)."""
    return {name: PolicyPlayer(fn) for name, fn in _heuristic_fns(game).items()}


def heuristic_switch(game: HoldemGame) -> tuple[PolicyPlayer, list[str]]:
    """One player that plays heuristic number `params` (one compile for all)."""
    names, fns = zip(*_heuristic_fns(game).items())

    def fn(params, rng, s, avail):
        return lax.switch(params, [lambda r, x, a, f=f: f(None, r, x, a) for f in fns],
                          rng, s, avail)

    return PolicyPlayer(fn), list(names)


class RebelState(NamedTuple):
    beliefs: FloatArray  # (n, 2, 1326) by position
    policy: FloatArray  # (n, Ni, S, 1326) policy of the sampled iteration
    tree: Tree  # current subgame, batched
    node: IntArray  # (n,) template node within the current subgame


class RebelPlayer(Player):
    """ReBeL with search at test time.

    by_street=False must be used when the player runs under jax.vmap (e.g.
    inside the vmapped NFSP training loops), see solver.solve_batch.
    """

    stateful = True

    def __init__(
        self,
        game: HoldemGame,
        hidden_dim: int,
        num_layers: int,
        cfr_iters: int,
        solve_chunk: int = 32,
        by_street: bool = True,
    ):
        self.game = game
        self.tpl = build_template(game)
        self.net = ValueNetwork(hidden_dim=hidden_dim, num_layers=num_layers)
        self.cfr_iters = cfr_iters
        self.solve_chunk = solve_chunk
        self.by_street = by_street
        self.children = jnp.asarray(self.tpl.children)
        self.internal_id = jnp.asarray(self.tpl.internal_id)
        self.env_slot = jnp.asarray(game.env_slot)

    def init(self, n: int) -> RebelState:
        ni, a = len(self.tpl.internal), self.game.num_slots
        n_nodes = len(self.tpl.parent)
        tree = Tree(
            kind=jnp.zeros((n, n_nodes), jnp.int32),
            chips=jnp.zeros((n, n_nodes, 2)),
            allin=jnp.zeros((n, n_nodes), bool),
            legal=jnp.zeros((n, ni, a), bool),
            actor=jnp.zeros((n, n_nodes), jnp.int32),
            street=jnp.zeros(n, jnp.int32),
            template_gap=jnp.zeros(n, bool),
        )
        return RebelState(
            beliefs=jnp.full((n, 2, NUM_HANDS), 1.0 / NUM_HANDS, jnp.float32),
            # bfloat16 halves the memory of evaluating many hands at once
            policy=jnp.zeros((n, ni, a, NUM_HANDS), jnp.bfloat16),
            tree=tree,
            node=jnp.zeros(n, jnp.int32),
        )

    def solve(self, params, state: RebelState, views: PublicView, need, rng):
        value_fn = lambda pub, b: self.net.apply(params, pub, b)  # noqa: E731
        sol, root_beliefs = solve_batch(
            self.game, self.tpl, views.bet, views.board, state.beliefs, value_fn,
            rng, self.cfr_iters, self.solve_chunk, by_street=self.by_street,
        )
        new = RebelState(
            beliefs=root_beliefs,
            policy=sol.policy.astype(state.policy.dtype),
            tree=sol.tree,
            node=jnp.zeros_like(state.node),
        )
        return _where(need, new, state)

    def _node_info(self, state: RebelState, slots: IntArray):
        ni = self.internal_id[state.node]
        rows = jnp.arange(state.node.shape[0])
        is_dec = state.tree.kind[rows, state.node] == DECISION
        legal = state.tree.legal[rows, jnp.maximum(ni, 0)]  # (n, S)
        in_tree = is_dec & legal[rows, slots]
        return ni, legal, in_tree

    def action_probs(self, state: RebelState, views: PublicView) -> FloatArray:
        """(n, S) probabilities of each slot for the agent's own hand."""
        n = state.node.shape[0]
        rows = jnp.arange(n)
        ni, legal, _ = self._node_info(state, jnp.zeros(n, jnp.int32))
        probs = state.policy[rows, jnp.maximum(ni, 0), :, views.own_hand]
        probs = jnp.where(legal, probs.astype(jnp.float32), 0.0)
        fallback = jnp.where(legal, 1.0, 0.0)
        probs = jnp.where(probs.sum(-1, keepdims=True) > 0, probs, fallback)
        return probs / probs.sum(-1, keepdims=True)

    def act(self, params, state: RebelState, views, rng, env_states, avail):
        probs = self.action_probs(state, views)
        keys = jax.random.split(rng, probs.shape[0])
        slots = jax.vmap(lambda k, p: jax.random.categorical(k, jnp.log(p)))(keys, probs)
        return jax.vmap(lambda b, k: slot_action(self.game, b, k))(views.bet, slots)

    def offtree(self, state: RebelState, views, actions):
        _, _, in_tree = self._node_info(state, self.env_slot[actions])
        return ~in_tree & (actions != self.game.fold_action)

    def observe(self, state: RebelState, views, actions, go):
        n = state.node.shape[0]
        rows = jnp.arange(n)
        slots = self.env_slot[actions]
        ni, _, in_tree = self._node_info(state, slots)
        upd = go & in_tree
        pos = (views.current_seat - views.small_blind) % 2
        f = state.policy[rows, jnp.maximum(ni, 0), slots].astype(jnp.float32)  # (n, 1326)
        actor_b = state.beliefs[rows, pos] * f
        # An action the policy never takes carries no usable information.
        actor_b = jnp.where(actor_b.sum(-1, keepdims=True) > 0, normalize(actor_b),
                            state.beliefs[rows, pos])
        beliefs = state.beliefs.at[rows, pos].set(actor_b)
        node = self.children[state.node, slots]
        return state._replace(
            beliefs=jnp.where(upd[:, None, None], beliefs, state.beliefs),
            node=jnp.where(upd, node, state.node),
        )


def _where(mask: BoolArray, a, b):
    return jax.tree.map(
        lambda x, y: jnp.where(mask.reshape(mask.shape + (1,) * (x.ndim - 1)), x, y),
        a,
        b,
    )


# =============================================================================
# Mirrored matches
# =============================================================================


class MatchResult(NamedTuple):
    rewards: FloatArray  # (n_deals, 2) player A's reward with A in seat 0 / seat 1
    hand_length: FloatArray  # (n_deals, 2) actions per hand
    unfinished: IntArray  # hands that hit the solve limit (should be 0)
    resolves: IntArray  # opponent actions outside a ReBeL player's subgame


def play_match(
    env,
    game: HoldemGame,
    player_a: Player,
    params_a,
    player_b: Player,
    params_b,
    n_deals: int,
    rng: PRNGKeyArray,
    max_solves: int = 64,
) -> MatchResult:
    """Play n_deals deals twice each, swapping seats; jittable."""
    rng_deal, rng_play = jax.random.split(rng)
    keys = jax.random.split(rng_deal, n_deals)
    keys = jnp.concatenate([keys, keys])
    env_states, _ = jax.vmap(env.reset)(keys)
    n = 2 * n_deals
    seat_a = jnp.concatenate([jnp.zeros(n_deals, jnp.int32), jnp.ones(n_deals, jnp.int32)])
    seat_b = 1 - seat_a
    views_of = lambda s, seat: jax.vmap(lambda e, k: public_view(game, e, k))(s, seat)  # noqa: E731

    class Carry(NamedTuple):
        env_states: Any
        st_a: Any
        st_b: Any
        need_a: BoolArray
        need_b: BoolArray
        pending: IntArray
        done: BoolArray
        rewards: FloatArray
        length: FloatArray
        rng: PRNGKeyArray
        solves: IntArray
        resolves: IntArray

    def inner_cond(c: Carry):
        return jnp.any(~c.done & ~c.need_a & ~c.need_b)

    def inner_body(c: Carry):
        rng, k_a, k_b = jax.random.split(c.rng, 3)
        active = ~c.done & ~c.need_a & ~c.need_b
        va, vb = views_of(c.env_states, seat_a), views_of(c.env_states, seat_b)
        avail = jax.vmap(env.get_avail_actions)(c.env_states)
        act_a = player_a.act(params_a, c.st_a, va, k_a, c.env_states, avail)
        act_b = player_b.act(params_b, c.st_b, vb, k_b, c.env_states, avail)
        seat = c.env_states.current_player_idx
        actions = jnp.where(c.pending >= 0, c.pending, jnp.where(seat == seat_a, act_a, act_b))
        off_a = player_a.offtree(c.st_a, va, actions) & (seat != seat_a)
        off_b = player_b.offtree(c.st_b, vb, actions) & (seat != seat_b)
        hold = active & (off_a | off_b)
        go = active & ~hold
        st_a = player_a.observe(c.st_a, va, actions, go)
        st_b = player_b.observe(c.st_b, vb, actions, go)
        nxt, _, rew, _, nxt_done, _ = jax.vmap(env.step_env)(
            jax.random.split(rng, n), c.env_states, actions
        )
        new_round = go & ~nxt_done & (nxt.stage != c.env_states.stage)
        env_states = _where(go, nxt, c.env_states)
        reward_a = rew[jnp.arange(n), seat_a]
        return Carry(
            env_states=env_states,
            st_a=st_a,
            st_b=st_b,
            need_a=c.need_a | (hold & off_a) | new_round,
            need_b=c.need_b | (hold & off_b) | new_round,
            pending=jnp.where(hold, actions, jnp.where(go, -1, c.pending)),
            done=c.done | (go & nxt_done),
            rewards=jnp.where(go & nxt_done, reward_a, c.rewards),
            length=c.length + go,
            rng=rng,
            solves=c.solves,
            resolves=c.resolves + jnp.sum(hold),
        )

    def outer_cond(c: Carry):
        return jnp.any(~c.done) & (c.solves < max_solves)

    def outer_body(c: Carry):
        rng, k_a, k_b = jax.random.split(c.rng, 3)
        va, vb = views_of(c.env_states, seat_a), views_of(c.env_states, seat_b)
        st_a = player_a.solve(params_a, c.st_a, va, c.need_a & ~c.done, k_a)
        st_b = player_b.solve(params_b, c.st_b, vb, c.need_b & ~c.done, k_b)
        c = c._replace(
            st_a=st_a,
            st_b=st_b,
            need_a=jnp.zeros_like(c.need_a),
            need_b=jnp.zeros_like(c.need_b),
            rng=rng,
            solves=c.solves + 1,
        )
        return lax.while_loop(inner_cond, inner_body, c)

    init = Carry(
        env_states=env_states,
        st_a=player_a.init(n),
        st_b=player_b.init(n),
        need_a=jnp.ones(n, bool),
        need_b=jnp.ones(n, bool),
        pending=-jnp.ones(n, jnp.int32),
        done=jnp.zeros(n, bool),
        rewards=jnp.zeros(n),
        length=jnp.zeros(n),
        rng=rng_play,
        solves=jnp.int32(0),
        resolves=jnp.int32(0),
    )
    out = lax.while_loop(outer_cond, outer_body, init)
    return MatchResult(
        rewards=out.rewards.reshape(2, n_deals).T,
        hand_length=out.length.reshape(2, n_deals).T,
        unfinished=jnp.sum(~out.done),
        resolves=out.resolves,
    )


def summarize(results: list[MatchResult]) -> dict[str, float]:
    """Mean reward per hand of player A and its standard error over deals."""
    pair = np.concatenate([np.asarray(r.rewards).mean(axis=1) for r in results])
    return {
        "mean": float(pair.mean()),
        "se": float(pair.std(ddof=1) / np.sqrt(len(pair))) if len(pair) > 1 else float("nan"),
        "hands": int(pair.size * 2),
        "unfinished": int(sum(int(r.unfinished) for r in results)),
        "resolves": int(sum(int(r.resolves) for r in results)),
    }


# =============================================================================
# Checkpoints
# =============================================================================


def save_checkpoint(path: str, params, meta: dict) -> str:
    """Write params to path (.msgpack) and the model/search config next to it."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))
    with open(_meta_path(path), "w") as f:
        json.dump(meta, f, indent=2)
    return path


def _meta_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".json"


def load_rebel(
    path: str,
    cfr_iters: int | None = None,
    solve_chunk: int = 32,
    by_street: bool = True,
) -> tuple[RebelPlayer, Any, dict]:
    """Load a ReBeL checkpoint as (player, params, meta). Fails loudly."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"ReBeL checkpoint not found: '{path}'")
    meta_file = _meta_path(path)
    if not os.path.isfile(meta_file):
        raise FileNotFoundError(f"ReBeL checkpoint metadata not found: '{meta_file}'")
    with open(meta_file) as f:
        meta = json.load(f)
    game = make_game(meta["game"], meta.get("max_raises"))
    player = RebelPlayer(
        game,
        hidden_dim=meta["value_hidden_dim"],
        num_layers=meta["value_num_layers"],
        cfr_iters=cfr_iters or meta["cfr_iters"],
        solve_chunk=solve_chunk,
        by_street=by_street,
    )
    template = player.net.init(
        jax.random.PRNGKey(0),
        jnp.zeros((PUBLIC_DIM,)),
        jnp.zeros((2, NUM_HANDS)),
    )
    with open(path, "rb") as f:
        params = serialization.from_bytes(template, f.read())
    shapes_ok = jax.tree.map(lambda a, b: a.shape == b.shape, params, template)
    if not all(jax.tree.leaves(shapes_ok)):
        raise ValueError(f"ReBeL checkpoint {path} does not match its metadata")
    return player, params, meta
