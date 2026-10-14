"""
Head-to-head evaluation of the Kuhn Poker and Leduc Hold'em baselines.

Every saved checkpoint plays every other checkpoint; the matrix cell (row A,
column B) is A against B from A's point of view, averaged over all checkpoint
pairs (i of A, j of B), each pair played for 32 hands. Writes
{out_dir}/{kuhn,leduc}.json:

    python -m bluffjax.examples.head_to_head.kuhn_leduc RUNS_ROOT OUT_DIR \
        [--games kuhn leduc] [--seed 0] [--commit 1156562]

RUNS_ROOT holds {kuhn,leduc}/<job_type>_<timestamp>/ (config.yaml, half/,
full/); the run of each baseline is the one directory whose config.yaml has
job_type {game}_{baseline} (an error if there are none or several).

Participants, in this order, each with its half/ and full/ checkpoints:
PPO-NFSP, PQN-NFSP, PPO, PQN, CFR, Deep CFR, ReBeL (Kuhn only), then Random.
Each checkpoint becomes the table of its output policy at every information
set of the game, the table its training script computed the exploitability
on:
    PPO-NFSP  avg_policy_{i}, ActorCriticDiscreteMLP, softmax of the logits
              ({kuhn,leduc}_exploitability.policy_array_from_network)
    PQN-NFSP  avg_policy_{i}, ActorDiscreteMLP, softmax of the logits
    PPO       policy_{i}, ActorCriticDiscreteMLP, softmax of the logits
    PQN       q_network_{i}, QNetworkDiscreteMLP, greedy over the legal
              actions, ties to the first one (policy_array_from_qnetwork)
    CFR       average_policy (one checkpoint per stage), load_average_policy
    Deep CFR  avg_policy_{i}, the script's policy_array (softmax over the
              legal actions)
    ReBeL     value_net_{i} (Kuhn), kuhn_rebel.policy_array of
              test_time_policy: the expected test-time policy of ReBeL's
              search, by enumerating its sampled CFR iterations
    Random    uniform over the legal actions (one checkpoint)
The networks' tables are computed with jax.jit, as in training. Before
playing, the exploitability and value_vs_random of every table are compared
with the values the run logged for the same checkpoint (every full
checkpoint; the half ones where the run evaluated at that point: CFR, Deep
CFR, ReBeL), to EXPLOITABILITY_ATOL (REBEL_ATOL for ReBeL, whose search is
not reproducible to float precision), so a table that is not the evaluated
policy (e.g. a PQN tie broken differently) is caught.

Play: the acting player's information set is looked up from the env (Kuhn:
its observation, which identifies the infoset; Leduc: (position, own card,
public card, both rounds' betting) of the env state, since the observation
only has the ranks while CFR's and Deep CFR's tables are over the physical
cards, as in leduc_cfr's docstring), and its action is sampled from the
table's row (one-hot for PQN). For every checkpoint pair: 16 deals (env reset
keys), each played twice with the seats swapped (the same reset key, so the
same cards and first player, and independent action randomness): in hand 2d
A has env seat 0 and in hand 2d + 1 seat 1, so A plays each deal from both
positions. Every hand is played with env.step_env until it is over.
Outcomes from A's seat: win = return > 0, loss < 0, draw = 0 (a Leduc split
pot), return in chips.

A cell's pairs are all (i, j), and only i != j on the diagonal when A has
more than one checkpoint; its metrics are the means over pairs of the pair
means over 32 hands, with standard errors sd(pair means) / sqrt(#pairs) (NaN
for a cell of one pair). extra holds the exact expectation of the cell, from
the tables (every deal, both of A's positions, mean over the same pairs):
exact_return, exact_win, exact_loss and exact_draw.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import glob
import itertools
import json
import math
import os
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from omegaconf import OmegaConf

from bluffjax import make
from bluffjax.examples.run_utils import checkpoint_path, load_params, read_results
from bluffjax.utils.game_utils import kuhn_exploitability as ke
from bluffjax.utils.game_utils import leduc_exploitability as le

EPISODES_PER_PAIR = 32
NUM_DEALS = EPISODES_PER_PAIR // 2  # each deal is played twice, seats swapped
STAGES = ("half", "full")
# (participant name, job_type suffix), in the matrix order
BASELINES = (
    ("PPO-NFSP", "ppo_nfsp"),
    ("PQN-NFSP", "pqn_nfsp"),
    ("PPO", "ppo"),
    ("PQN", "pqn"),
    ("CFR", "cfr"),
    ("Deep CFR", "deep_cfr"),
    ("ReBeL", "rebel"),
)
RANDOM = "Random"
DEFAULT_COMMIT = "1156562"
# Logged and recomputed exploitability / value_vs_random must agree to this
EXPLOITABILITY_ATOL = 1e-4
# except for ReBeL: its test-time policy (512 iterations of float32 Linear
# CFR-D with the value network at the leaves) changes by up to ~5e-3 per
# probability between compilations of the same computation (one seed vs the
# training's vmap over seeds, TF32 vs full-precision matmuls on the GPU),
# which moves its exploitability and value_vs_random by up to ~1e-3
REBEL_ATOL = 3e-3


# ----------------------------------------------------------------- games
@dataclasses.dataclass(frozen=True)
class Game:
    key: str  # game_key of the output, also the runs' game directory
    name: str
    env_id: str
    num_actions: int
    obs_dim: int
    legal: np.ndarray  # (R, A) legal actions of every infoset (table row)
    row_obs: np.ndarray  # (R, obs_dim) env observation at every infoset
    lookup: np.ndarray  # (num_codes,) table row of an infoset code, -1: none
    code: Callable  # code(state, obs) -> int32 infoset code (jax, one state)
    terminals: dict  # terminal histories, the format of le._terminal_histories
    exploitability: Callable[[np.ndarray], float]
    value_vs_random: Callable[[np.ndarray], float]

    @property
    def num_rows(self) -> int:
        return self.legal.shape[0]

    def rows(self, state, obs):
        """Table rows of the acting players of a batch of env states, -1 for
        an unknown infoset."""
        codes = jax.vmap(self.code)(state, obs)
        return jnp.asarray(self.lookup)[codes]


# Kuhn: the observation (9 binary floats: card, own and opponent chips,
# position) identifies the infoset; its code is the observation's bits.
_KUHN_BITS = 2 ** np.arange(9)


def _kuhn_code(state, obs):
    return jnp.sum(jnp.round(obs).astype(jnp.int32) * jnp.asarray(_KUHN_BITS))


def _kuhn_history(key: str) -> tuple[int, ...]:
    return tuple(ke.ACTION_PASS if c == "p" else ke.ACTION_BET for c in key[1:])


def _kuhn_terminals() -> dict[str, np.ndarray]:
    """Every terminal history of Kuhn (deal and actions) in the format of
    leduc_exploitability._terminal_histories, rows in ke.INFOSETS order."""
    index = {k: i for i, k in enumerate(ke.INFOSETS)}
    deals = list(itertools.permutations(range(3), 2))
    histories = []
    for hands in deals:
        stack = [()]
        while stack:
            hist = stack.pop()
            if ke.is_terminal(hist):
                histories.append((1.0 / len(deals), ke.get_returns(hands, hist), hands, hist))
                continue
            for a in ke.get_legal_actions(hist):
                stack.append(hist + (a,))
    depth = max(len(h) for *_, h in histories)
    shape = (len(histories), depth)
    out = {
        "chance": np.array([p for p, *_ in histories], dtype=np.float64),
        "returns": np.array([r for _, r, *_ in histories], dtype=np.float64),
        "row": np.zeros(shape, dtype=np.int64),
        "action": np.zeros(shape, dtype=np.int64),
        "player": np.full(shape, -1, dtype=np.int64),
        "num_legal": np.ones(shape, dtype=np.int64),
    }
    for z, (_, _, hands, hist) in enumerate(histories):
        for d, a in enumerate(hist):
            player = ke.get_current_player(hist[:d])
            out["row"][z, d] = index[ke.infoset_key(hands[player], hist[:d])]
            out["action"][z, d] = a
            out["player"][z, d] = player
            out["num_legal"][z, d] = 2
    return out


def make_kuhn() -> Game:
    codes = (ke.INFOSET_OBS.astype(np.int64) * _KUHN_BITS).sum(axis=-1)
    lookup = np.full(2 ** 9, -1, dtype=np.int32)
    lookup[codes] = np.arange(len(codes))
    assert len(set(codes.tolist())) == len(ke.INFOSETS)
    return Game(
        key="kuhn",
        name="Kuhn",
        env_id="kuhn_poker",
        num_actions=2,
        obs_dim=9,
        legal=np.ones((len(ke.INFOSETS), 2), dtype=bool),
        row_obs=ke.INFOSET_OBS.astype(np.float32),
        lookup=lookup,
        code=_kuhn_code,
        terminals=_kuhn_terminals(),
        exploitability=lambda t: float(ke.exploitability_from_policy_array(t)),
        value_vs_random=lambda t: float(ke.value_vs_random_from_policy_array(t)),
    )


# Leduc: the infoset is (position, own physical card, public card, both
# rounds' betting sequences), read from the env state; the observation has
# only the ranks. Code: position, card, public card + 1, then the 8 betting
# slots as base-3 digits (0 none, 1 call, 2 raise).
_LEDUC_SLOTS = 2 * le._MAX_ACTIONS_PER_ROUND
_LEDUC_DIGITS = 3 ** np.arange(_LEDUC_SLOTS)
_LEDUC_NUM_CODES = 2 * 6 * 7 * 3 ** _LEDUC_SLOTS


def _leduc_key_code(position, card, public, slots) -> int:
    return int(((position * 6 + card) * 7 + public + 1) * 3 ** _LEDUC_SLOTS
               + np.dot(np.asarray(slots, dtype=np.int64), _LEDUC_DIGITS))


def _leduc_code(state, obs):
    cur = state.current_player_idx
    position = (cur - state.start_player_idx) % 2
    card = state.agent_cards[cur]
    slots = state.history.reshape(-1).astype(jnp.int32)
    return ((position * 6 + card) * 7 + state.public_card + 1) * 3 ** _LEDUC_SLOTS + jnp.sum(
        slots * jnp.asarray(_LEDUC_DIGITS, dtype=jnp.int32)
    )


def make_leduc() -> Game:
    lookup = np.full(_LEDUC_NUM_CODES, -1, dtype=np.int32)
    legal = np.zeros((len(le._INFOSET_LIST), 3), dtype=bool)
    for i, (key, actions) in enumerate(le._INFOSET_LIST):
        position, card, public, round1, round2 = key[:5]
        pad = lambda s: tuple(s) + (0,) * (le._MAX_ACTIONS_PER_ROUND - len(s))
        code = _leduc_key_code(position, card, public, pad(round1) + pad(round2))
        assert lookup[code] == -1, f"two infosets with code {code}"
        lookup[code] = i
        legal[i, list(actions)] = True
    row_obs = np.stack([le._infoset_key_to_obs(k) for k, _ in le._INFOSET_LIST])
    return Game(
        key="leduc",
        name="Leduc",
        env_id="leduc_holdem",
        num_actions=3,
        obs_dim=le._OBS_DIM,
        legal=legal,
        row_obs=row_obs.astype(np.float32),
        lookup=lookup,
        code=_leduc_code,
        terminals=dict(le._terminal_histories()),
        exploitability=lambda t: float(le.exploitability_from_policy_array(t, le._KEY_TO_IDX)),
        value_vs_random=lambda t: float(le.value_vs_random_from_policy_array(t, le._KEY_TO_IDX)),
    )


GAMES: dict[str, Callable[[], Game]] = {"kuhn": make_kuhn, "leduc": make_leduc}


# --------------------------------------------- reachable env states, check
def initial_states(env, game: Game):
    """Every deal and first player of the env as a batch of initial states,
    with their probabilities and the cards (seat order) and public card."""
    base, _ = env.reset(jax.random.PRNGKey(0))
    states, deals = [], []
    if game.key == "kuhn":
        for hands in itertools.permutations(range(3), 2):
            for start in (0, 1):
                states.append(base.replace(
                    agent_hands=jnp.array(hands, jnp.int32),
                    start_player_idx=jnp.int32(start),
                    current_player_idx=jnp.int32(start),
                ))
                deals.append((hands, -1, start))
    else:
        for c0, c1, public in itertools.permutations(range(6), 3):
            rest = [c for c in range(6) if c not in (c0, c1, public)]
            for start in (0, 1):
                states.append(base.replace(
                    agent_cards=jnp.array([c0, c1], jnp.int32),
                    shuffled_deck=jnp.array([c0, c1, public, *rest], jnp.int32),
                    start_player_idx=jnp.int32(start),
                    current_player_idx=jnp.int32(start),
                ))
                deals.append(((c0, c1), public, start))
    batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
    return batch, deals


def walk_env(env, game: Game):
    """Every reachable decision and terminal state of the env, by stepping
    every legal action of every deal and first player (batched per depth).
    Returns (decisions, terminals): decisions has, per decision, deal (index
    into deals), path (the actions so far), row (game.rows), mask (env legal
    actions), obs and seat (the acting env player); terminals has deal,
    path, returns (2,) and steps, the (row, acting seat) of every decision
    along the path. deals: (cards
    of seats 0 and 1, public card or -1, first player), equally likely."""
    states, deals = initial_states(env, game)
    obs = jax.vmap(env.obs_from_state)(states)
    paths = [()] * len(deals)
    deal_idx = np.arange(len(deals))
    rows_along = [()] * len(deals)
    returns = np.zeros((len(deals), 2))
    step = jax.jit(jax.vmap(env.step_env))
    rows_fn = jax.jit(game.rows)
    mask_fn = jax.jit(jax.vmap(env.get_avail_actions))
    decisions = {"deal": [], "path": [], "row": [], "mask": [], "obs": [], "seat": []}
    terminals = {"deal": [], "path": [], "returns": [], "steps": []}
    while len(paths):
        rows = np.asarray(rows_fn(states, obs))
        mask = np.asarray(mask_fn(states))
        seat = np.asarray(states.current_player_idx)
        decisions["deal"] += deal_idx.tolist()
        decisions["path"] += paths
        decisions["row"] += rows.tolist()
        decisions["mask"].append(mask)
        decisions["obs"].append(np.asarray(obs))
        decisions["seat"] += seat.tolist()
        next_parts = []
        for a in range(game.num_actions):
            sel = np.flatnonzero(mask[:, a])
            if not len(sel):
                continue
            sub = jax.tree_util.tree_map(lambda x: x[sel], states)
            keys = jax.random.split(jax.random.PRNGKey(0), len(sel))
            nxt, nobs, reward, _, done, _ = step(keys, sub, jnp.full(len(sel), a, jnp.int32))
            done = np.asarray(done)
            ret = returns[sel] + np.asarray(reward, dtype=np.float64)
            for k, i in enumerate(sel):
                if done[k]:
                    terminals["deal"].append(int(deal_idx[i]))
                    terminals["path"].append(paths[i] + (a,))
                    terminals["returns"].append(ret[k])
                    terminals["steps"].append(rows_along[i] + ((int(rows[i]), int(seat[i])),))
            live = np.flatnonzero(~done)
            next_parts.append((
                jax.tree_util.tree_map(lambda x: x[live], nxt), np.asarray(nobs)[live],
                [paths[sel[k]] + (a,) for k in live], deal_idx[sel[live]],
                [rows_along[sel[k]] + ((int(rows[sel[k]]), int(seat[sel[k]])),) for k in live], ret[live],
            ))
        next_parts = [p for p in next_parts if len(p[2])]
        if not next_parts:
            break
        states = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs), *[p[0] for p in next_parts])
        obs = jnp.asarray(np.concatenate([p[1] for p in next_parts]))
        paths = sum((p[2] for p in next_parts), [])
        deal_idx = np.concatenate([p[3] for p in next_parts])
        rows_along = sum((p[4] for p in next_parts), [])
        returns = np.concatenate([p[5] for p in next_parts])
    decisions["mask"] = np.concatenate(decisions["mask"])
    decisions["obs"] = np.concatenate(decisions["obs"])
    for d in (decisions, terminals):
        for k in ("deal", "row", "seat"):
            if k in d:
                d[k] = np.asarray(d[k])
    terminals["returns"] = np.asarray(terminals["returns"])
    return decisions, terminals, deals


def model_infoset(game: Game, deal, path: tuple[int, ...]):
    """(key, legal actions) of the acting player after `path` in the
    exploitability code's model of the game (player 0 = first to act), for
    an env deal (cards of seats 0 and 1, public card, first player)."""
    cards, public, start = deal
    position = len(path) % 2 if game.key == "kuhn" else None
    if game.key == "kuhn":
        card = cards[(start + position) % 2]
        return ke.infoset_key(card, path), tuple(ke.get_legal_actions(path))
    s = le.initial_state()
    s = s.child(cards[start]).child(cards[1 - start])
    for a in path:
        if s.is_chance_node():
            s = s.child(public)
        s = s.child(a)
    if s.is_chance_node():
        s = s.child(public)
    return s.info_state_key(s.current_player), s.legal_actions()


def check_infosets(env, game: Game) -> str:
    """Checks on every reachable env decision that the table row looked up
    is the acting player's infoset in the exploitability code's model, with
    the same legal actions and the observation the networks' tables were
    computed on, and that the map from reachable infosets to rows is a
    bijection onto all rows. Returns a summary line."""
    decisions, _, deals = walk_env(env, game)
    keys = [k for k, _ in le._INFOSET_LIST] if game.key == "leduc" else list(ke.INFOSETS)
    rows = decisions["row"]
    if (rows < 0).any():
        raise AssertionError(f"{(rows < 0).sum()} reachable decisions at no table row")
    infoset_to_row: dict = {}
    for n, (d, path, row) in enumerate(zip(decisions["deal"], decisions["path"], rows)):
        key, legal = model_infoset(game, deals[d], path)
        if keys[row] != key:
            raise AssertionError(f"deal {deals[d]} path {path}: row {keys[row]}, model {key}")
        if tuple(np.flatnonzero(decisions["mask"][n])) != tuple(legal):
            raise AssertionError(f"deal {deals[d]} path {path}: env legal actions differ")
        if not np.array_equal(decisions["obs"][n], game.row_obs[row]):
            raise AssertionError(f"deal {deals[d]} path {path}: env observation differs")
        if infoset_to_row.setdefault(key, row) != row:
            raise AssertionError(f"infoset {key} at two rows")
    if sorted(infoset_to_row.values()) != list(range(game.num_rows)):
        raise AssertionError("the reachable infosets don't cover every row exactly once")
    return (f"{len(rows)} reachable env decisions, {len(infoset_to_row)} infosets, "
            f"a bijection onto the {game.num_rows} table rows")


# ---------------------------------------------------------------- tables
def normalize_table(game: Game, table: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    """(R, A) float64 probabilities over the legal actions. The table must
    be a policy already (rows summing to 1 with no illegal mass, within tol);
    it is then renormalised over the legal actions, as the exploitability
    code does (leduc_exploitability._normalized_policy_rows)."""
    table = np.asarray(table, dtype=np.float64)
    if table.shape != game.legal.shape:
        raise ValueError(f"table shape {table.shape}, expected {game.legal.shape}")
    if not np.all(np.isfinite(table)) or table.min() < 0:
        raise ValueError("table has negative or non-finite entries")
    illegal = np.where(game.legal, 0.0, table).sum(axis=-1)
    total = np.where(game.legal, table, 0.0).sum(axis=-1)
    if illegal.max() > tol or np.abs(total - 1.0).max() > tol:
        raise ValueError(
            f"not a policy: illegal mass {illegal.max():.2e}, "
            f"row sums off by {np.abs(total - 1.0).max():.2e}"
        )
    table = np.where(game.legal, table, 0.0)
    return table / table.sum(axis=-1, keepdims=True)


def uniform_table(game: Game) -> np.ndarray:
    return game.legal / game.legal.sum(axis=-1, keepdims=True)


def _with_value(apply: Callable) -> Callable:
    """apply returning (logits, value) for a network returning logits only."""

    def wrapped(params, obs):
        out = apply(params, obs)
        return out if isinstance(out, tuple) else (out, None)

    return wrapped


def network_table(game: Game, network, params, greedy: bool) -> np.ndarray:
    """Table of a policy network (softmax of the logits) or of a Q-network
    (greedy over the legal actions, ties to the first one): exactly the
    scripts' policy_array_from_network / policy_array_from_qnetwork, jitted
    as in training."""
    if game.key == "kuhn":
        fn = ke.policy_array_from_qnetwork if greedy else ke.policy_array_from_network
        table = jax.jit(lambda p: fn(network.apply, p))(params)
    else:
        if greedy:
            fn, apply = le.policy_array_from_qnetwork, network.apply
        else:
            # leduc's policy_array_from_network takes (logits, value); the
            # actor-only average policy of PQN-NFSP is wrapped as its script does
            fn, apply = le.policy_array_from_network, _with_value(network.apply)
        table = jax.jit(lambda p: fn(apply, p, le._INFOSET_LIST))(params)
    return np.asarray(table, dtype=np.float64)


@dataclasses.dataclass
class Checkpoint:
    path: str
    table: np.ndarray  # normalized (R, A)
    logged: dict | None = None  # {"exploitability": x, "value_vs_random": y}


@dataclasses.dataclass
class Participant:
    name: str
    checkpoints: list[Checkpoint]


def find_run(runs_root: str, game: Game, suffix: str) -> str | None:
    """The run directory of job_type {game}_{suffix} (None if there is none)."""
    job_type = f"{game.key}_{suffix}"
    found = []
    for config_path in sorted(glob.glob(os.path.join(runs_root, game.key, "*", "config.yaml"))):
        if OmegaConf.load(config_path).get("job_type") == job_type:
            found.append(os.path.dirname(config_path))
    if len(found) > 1:
        raise ValueError(f"several runs of {job_type} in {runs_root}: {found}")
    return found[0] if found else None


def _logged_values(run_dir: str, step_key: str, step: int) -> dict[int, dict]:
    """{seed_index: {exploitability, value_vs_random}} of the records with
    step_key == step."""
    out = {}
    for record in read_results(run_dir):
        if record.get(step_key) == step and "exploitability" in record:
            out[int(record["seed_index"])] = {
                "exploitability": float(record["exploitability"]),
                "value_vs_random": float(record["value_vs_random"]),
            }
    return out


def load_baseline(game: Game, suffix: str, run_dir: str, stage: str) -> list[Checkpoint]:
    """The checkpoints of one baseline and stage as tables, with the values
    its run logged for them (where it evaluated them)."""
    from bluffjax.networks.mlp import (
        ActorCriticDiscreteMLP,
        ActorDiscreteMLP,
        QNetworkDiscreteMLP,
    )

    config = OmegaConf.to_container(OmegaConf.load(os.path.join(run_dir, "config.yaml")))
    num_seeds = int(config["num_seeds"])
    rng = jax.random.PRNGKey(0)
    checkpoints = []

    if suffix in ("ppo_nfsp", "pqn_nfsp", "ppo", "pqn"):
        name, cls, greedy = {
            "ppo_nfsp": ("avg_policy", ActorCriticDiscreteMLP, False),
            "pqn_nfsp": ("avg_policy", ActorDiscreteMLP, False),
            "ppo": ("policy", ActorCriticDiscreteMLP, False),
            "pqn": ("q_network", QNetworkDiscreteMLP, True),
        }[suffix]
        network = cls(action_dim=game.num_actions, hidden_dim=int(config["fc_dim_size"]))
        template = network.init(rng, jnp.zeros(game.obs_dim))
        steps = {"full": int(config["num_update_steps"]) - 1,
                 "half": int(config["half_update_steps"]) - 1}
        logged = _logged_values(run_dir, "update_step", steps[stage])
        for i in range(num_seeds):
            path = checkpoint_path(run_dir, stage, name, i)
            table = network_table(game, network, load_params(path, template), greedy)
            checkpoints.append(Checkpoint(path, normalize_table(game, table), logged.get(i)))
    elif suffix == "cfr":
        if game.key == "kuhn":
            from bluffjax.examples.kuhn import kuhn_cfr as cfr
        else:
            from bluffjax.examples.leduc import leduc_cfr as cfr
        step = int(config["iterations"] if stage == "full" else config["half_iterations"])
        logged = _logged_values(run_dir, "iteration", step)
        path = checkpoint_path(run_dir, stage, "average_policy")
        table = cfr.load_average_policy(path)
        checkpoints.append(Checkpoint(path, normalize_table(game, table), logged.get(0)))
    elif suffix == "deep_cfr":
        hidden = tuple(int(h) for h in config["hidden_sizes"])
        if game.key == "kuhn":
            from bluffjax.examples.kuhn import kuhn_deep_cfr as dcfr

            template = dcfr.MLP(hidden, dcfr.NUM_ACTIONS).init(rng, dcfr.INFOSET_TENSORS[:1])
        else:
            from bluffjax.examples.leduc import leduc_deep_cfr as dcfr

            template = dcfr.policy_network(hidden).init(rng, jnp.zeros((1, dcfr.INFO_STATE_SIZE)))
        step = int(config["iterations"] if stage == "full" else config["half_iterations"])
        logged = _logged_values(run_dir, "iteration", step)
        for i in range(num_seeds):
            path = checkpoint_path(run_dir, stage, "avg_policy", i)
            table = dcfr.policy_array(load_params(path, template), hidden)
            checkpoints.append(Checkpoint(path, normalize_table(game, table), logged.get(i)))
    elif suffix == "rebel":
        if game.key != "kuhn":
            raise ValueError("table ReBeL exists for Kuhn only")
        from bluffjax.examples.kuhn import kuhn_rebel as rebel

        network = rebel.ValueNetwork(hidden_dim=int(config["value_hidden_dim"]))
        template = network.init(rng, jnp.zeros(rebel.PBS_DIM))
        structure = rebel.subgame_structure(int(config["max_depth"]))
        cfr_iters = int(config["cfr_iters"])
        policy_fn = jax.jit(
            lambda p: rebel.test_time_policy(
                rebel.network_value_fn(network, p), structure, cfr_iters
            )
        )
        steps = {"full": int(config["num_update_steps"]), "half": int(config["half_update_steps"])}
        logged = _logged_values(run_dir, "updates_completed", steps[stage])
        for i in range(num_seeds):
            path = checkpoint_path(run_dir, stage, "value_net", i)
            policy = np.asarray(policy_fn(load_params(path, template)))
            table = rebel.policy_array(policy)
            checkpoints.append(Checkpoint(path, normalize_table(game, table), logged.get(i)))
    else:
        raise ValueError(f"unknown baseline {suffix}")
    return checkpoints


def load_participants(runs_root: str, game: Game) -> list[Participant]:
    participants = []
    for label, suffix in BASELINES:
        run_dir = find_run(runs_root, game, suffix)
        if run_dir is None:
            continue
        for stage in STAGES:
            participants.append(
                Participant(f"{label} ({stage})", load_baseline(game, suffix, run_dir, stage))
            )
    participants.append(Participant(RANDOM, [Checkpoint("uniform", uniform_table(game))]))
    return participants


def check_logged(game: Game, participants: list[Participant]) -> list[str]:
    """Recomputes the exploitability and value_vs_random of every checkpoint
    with a logged value; returns one line per participant. Raises if any
    differs by more than EXPLOITABILITY_ATOL."""
    lines, failures = [], []
    for p in participants:
        diffs = []
        atol = REBEL_ATOL if p.name.startswith("ReBeL") else EXPLOITABILITY_ATOL
        for c in p.checkpoints:
            if c.logged is None:
                continue
            got = {"exploitability": game.exploitability(c.table),
                   "value_vs_random": game.value_vs_random(c.table)}
            for k, v in got.items():
                d = abs(v - c.logged[k])
                diffs.append(d)
                if not d <= atol:
                    failures.append(f"{p.name} {c.path} {k}: {v} vs logged {c.logged[k]}")
        checked = sum(c.logged is not None for c in p.checkpoints)
        lines.append(
            f"{p.name}: {checked}/{len(p.checkpoints)} checkpoints checked against the log"
            + (f", max |diff| {max(diffs):.2e} (tolerance {atol:.0e})" if diffs else "")
        )
    if failures:
        raise AssertionError("tables don't match the logged evaluation:\n" + "\n".join(failures))
    return lines


# ------------------------------------------------------------------ play
def mirrored_keys(rng, num_deals: int):
    """Reset keys (2 num_deals,) with hands 2d and 2d + 1 sharing deal d's
    key, and independent action keys (2 num_deals,)."""
    rng_deal, rng_act = jax.random.split(rng)
    deal = jax.random.split(rng_deal, num_deals)
    return jnp.repeat(deal, 2, axis=0), jax.random.split(rng_act, 2 * num_deals)


def play_pairs(env, game: Game, tables, pair_a, pair_b, rng, num_deals: int = NUM_DEALS):
    """Plays 2 num_deals hands for every checkpoint pair (pair_a[p],
    pair_b[p]) (indices into tables (N, R, A)): deal d of pair p is played
    with A in env seat 0 (hand 2d) and seat 1 (hand 2d + 1), from the same
    reset key fold_in(rng, p) -> deal d. Returns per hand (P, 2 num_deals):
    a_return (A's return), length (env steps), done, a_seat, invalid (steps
    at an unknown infoset or whose env legal actions differ from the
    table's) and deal (the hand's initial env state, flattened)."""
    num_games = 2 * num_deals
    pair_a = jnp.asarray(pair_a, dtype=jnp.int32)
    pair_b = jnp.asarray(pair_b, dtype=jnp.int32)
    num_pairs = pair_a.shape[0]
    tables = jnp.asarray(tables, dtype=jnp.float32)
    log_tables = jnp.where(tables > 0, jnp.log(jnp.maximum(tables, 1e-38)), -jnp.inf)
    legal = jnp.asarray(game.legal)

    pair_keys = jax.vmap(lambda p: jax.random.fold_in(rng, p))(jnp.arange(num_pairs))
    reset_keys, act_keys = jax.vmap(lambda k: mirrored_keys(k, num_deals))(pair_keys)
    flat = lambda x: x.reshape((num_pairs * num_games,) + x.shape[2:])
    reset_keys, act_keys = flat(reset_keys), flat(act_keys)
    a_idx = jnp.repeat(pair_a, num_games)
    b_idx = jnp.repeat(pair_b, num_games)
    a_seat = jnp.tile(jnp.arange(num_games, dtype=jnp.int32) % 2, num_pairs)

    states, obs = jax.vmap(env.reset)(reset_keys)
    deal = jnp.concatenate(
        [jnp.asarray(x).reshape(x.shape[0], -1).astype(jnp.float32)
         for x in jax.tree_util.tree_leaves(states)],
        axis=-1,
    )
    total = num_pairs * num_games

    def where_games(games, new, old):
        return jnp.where(games.reshape((-1,) + (1,) * (new.ndim - 1)), new, old)

    def cond(carry):
        return ~carry[2].all()

    def body(carry):
        states, obs, finished, returns, length, invalid, t = carry
        live = ~finished
        rows = game.rows(states, obs)
        mask = jax.vmap(env.get_avail_actions)(states)
        safe_rows = jnp.maximum(rows, 0)
        bad = (rows < 0) | jnp.any(mask != legal[safe_rows], axis=-1)
        table_idx = jnp.where(states.current_player_idx == a_seat, a_idx, b_idx)
        logits = jnp.where(mask, log_tables[table_idx, safe_rows], -jnp.inf)
        step_keys = jax.vmap(lambda k: jax.random.fold_in(k, t))(act_keys)
        split = jax.vmap(jax.random.split)(step_keys)
        k_act, k_env = split[:, 0], split[:, 1]
        action = jax.vmap(jax.random.categorical)(k_act, logits)
        next_states, next_obs, reward, _, done, _ = jax.vmap(env.step_env)(k_env, states, action)
        returns = returns + jnp.where(live[:, None], reward, 0.0)
        length = length + live.astype(jnp.int32)
        invalid = invalid + (live & bad).astype(jnp.int32)
        states = jax.tree_util.tree_map(lambda n, o: where_games(live, n, o), next_states, states)
        obs = where_games(live, next_obs, obs)
        return states, obs, finished | (live & done), returns, length, invalid, t + 1

    init = (
        states,
        obs,
        jnp.zeros(total, dtype=jnp.bool_),
        jnp.zeros((total, env.num_agents), dtype=jnp.float32),
        jnp.zeros(total, dtype=jnp.int32),
        jnp.zeros(total, dtype=jnp.int32),
        jnp.int32(0),
    )
    _, _, done, returns, length, invalid, _ = lax.while_loop(cond, body, init)
    a_return = jnp.take_along_axis(returns, a_seat[:, None], axis=-1)[:, 0]
    shape = (num_pairs, num_games)
    return {
        "a_return": a_return.reshape(shape),
        "length": length.reshape(shape),
        "done": done.reshape(shape),
        "a_seat": a_seat.reshape(shape),
        "invalid": invalid.reshape(shape),
        "deal": deal.reshape(shape + deal.shape[1:]),
    }


# ----------------------------------------------------------------- exact
def own_reach(game: Game, tables: np.ndarray) -> np.ndarray:
    """(N, 2, Z): for every table k and position s, the product of table k's
    probabilities of its own actions along terminal history z when it plays
    position s (0: acts first)."""
    h = game.terminals
    probs = np.asarray(tables, dtype=np.float64)[:, h["row"], h["action"]]  # (N, Z, D)
    out = []
    for s in (0, 1):
        out.append(np.where(h["player"] == s, probs, 1.0).prod(axis=-1))
    return np.stack(out, axis=1)


def exact_outcomes(game: Game, tables: np.ndarray) -> dict[str, np.ndarray]:
    """Exact (N, N) expected return, win, loss and draw probabilities of
    table a (row) against table b (column) per hand, averaged over a's two
    positions and every deal: the expectation of a mirrored pair's mean."""
    h = game.terminals
    reach = own_reach(game, tables)
    out = {}
    for name, f in (("return", lambda r: r), ("win", lambda r: (r > 0) * 1.0),
                    ("loss", lambda r: (r < 0) * 1.0), ("draw", lambda r: (r == 0) * 1.0)):
        total = 0.0
        for s in (0, 1):
            weight = h["chance"] * f(h["returns"][:, s])  # (Z,)
            total = total + (reach[:, s, :] * weight) @ reach[:, 1 - s, :].T
        out[name] = total / 2.0
    return out


# --------------------------------------------------------------- matrix
def cell_pairs(n_a: int, n_b: int, diagonal: bool) -> list[tuple[int, int]]:
    """Checkpoint pairs of a cell: all (i, j), i != j on the diagonal when
    the participant has more than one checkpoint."""
    return [(i, j) for i in range(n_a) for j in range(n_b)
            if not (diagonal and n_a > 1 and i == j)]


def _se(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def evaluate(game: Game, participants: list[Participant], seed: int = 0,
             chunk: int = 4096) -> dict[str, Any]:
    """Plays every cell; returns the cells and the raw per-pair results."""
    env = make(game.env_id)
    tables = np.concatenate([np.stack([c.table for c in p.checkpoints]) for p in participants])
    offsets = np.cumsum([0] + [len(p.checkpoints) for p in participants])
    pairs = []  # (row, col, i, j)
    for r, pa in enumerate(participants):
        for c, pb in enumerate(participants):
            for i, j in cell_pairs(len(pa.checkpoints), len(pb.checkpoints), r == c):
                pairs.append((r, c, i, j))
    pairs = np.array(pairs, dtype=np.int64)
    pair_a = offsets[pairs[:, 0]] + pairs[:, 2]
    pair_b = offsets[pairs[:, 1]] + pairs[:, 3]

    play = jax.jit(lambda ta, a, b, k: play_pairs(env, game, ta, a, b, k))
    rng = jax.random.PRNGKey(seed)
    chunks = []
    for start in range(0, len(pairs), chunk):
        stop = min(start + chunk, len(pairs))
        a = np.zeros(chunk, dtype=np.int32)
        b = np.zeros(chunk, dtype=np.int32)
        a[: stop - start], b[: stop - start] = pair_a[start:stop], pair_b[start:stop]
        out = play(tables, a, b, jax.random.fold_in(rng, start // chunk))
        chunks.append({k: np.asarray(v)[: stop - start] for k, v in out.items()})
    raw = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    check_episodes(raw)

    exact = exact_outcomes(game, tables)
    ret = raw["a_return"].astype(np.float64)
    pair_metrics = {
        "win": (ret > 0).mean(axis=1),
        "loss": (ret < 0).mean(axis=1),
        "draw": (ret == 0).mean(axis=1),
        "return": ret.mean(axis=1),
    }
    pair_exact = {k: v[pair_a, pair_b] for k, v in exact.items()}

    n = len(participants)
    cells = [[None] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            sel = (pairs[:, 0] == r) & (pairs[:, 1] == c)
            m = {k: v[sel] for k, v in pair_metrics.items()}
            e = {k: v[sel] for k, v in pair_exact.items()}
            cells[r][c] = {
                "win": float(m["win"].mean()),
                "loss": float(m["loss"].mean()),
                "draw": float(m["draw"].mean()),
                "return": float(m["return"].mean()),
                "win_se": _se(m["win"]),
                "return_se": _se(m["return"]),
                "pairs": int(sel.sum()),
                "episodes": int(sel.sum()) * EPISODES_PER_PAIR,
                "extra": {f"exact_{k}": float(e[k].mean()) for k in ("return", "win", "loss", "draw")},
            }
    return {"cells": cells, "pairs": pairs, "pair_metrics": pair_metrics,
            "pair_exact": pair_exact, "pair_noise_var": pair_noise_var(ret), "raw": raw}


def pair_noise_var(a_return: np.ndarray) -> np.ndarray:
    """(P,) sampling variance of each pair's mean return: the variance of
    its deal means (the two mirrored hands of a deal together; deals are
    independent) / the number of deals."""
    deal_means = a_return.reshape(a_return.shape[0], -1, 2).mean(axis=-1)
    return deal_means.var(axis=-1, ddof=1) / deal_means.shape[1]


def check_episodes(raw: dict[str, np.ndarray]) -> None:
    """Every pair played EPISODES_PER_PAIR complete hands, at known
    infosets, with A in seat 0 and 1 of every deal and the deal reused."""
    assert raw["a_return"].shape[1] == EPISODES_PER_PAIR
    assert raw["done"].all(), "a hand was not finished"
    assert (raw["length"] > 0).all()
    assert (raw["invalid"] == 0).all(), "a step at an unknown infoset or with other legal actions"
    np.testing.assert_array_equal(raw["a_seat"][:, 0::2], 0)
    np.testing.assert_array_equal(raw["a_seat"][:, 1::2], 1)
    np.testing.assert_array_equal(raw["deal"][:, 0::2], raw["deal"][:, 1::2])


def sanity_report(game: Game, participants: list[Participant], result: dict) -> list[str]:
    """Sanity checks of a matrix, as lines to print. z-scores use the
    sampling noise of the cells' means (from the per-pair deal means,
    pair_noise_var), whose expectations are known: sampled - exact return
    (0), return(A,B) + return(B,A) (0: the same checkpoint pairs, seats
    swapped, zero-sum) and the diagonal's return (0). The antisymmetry is
    also given in units of the cells' return_se."""
    cells, names = result["cells"], [p.name for p in participants]
    n = len(names)
    pairs = result["pairs"]
    diff = result["pair_metrics"]["return"] - result["pair_exact"]["return"]
    noise = result["pair_noise_var"]
    cell_diff = np.zeros((n, n))
    cell_noise_se = np.zeros((n, n))
    for r in range(n):
        for c in range(n):
            sel = (pairs[:, 0] == r) & (pairs[:, 1] == c)
            cell_diff[r, c] = diff[sel].mean()
            cell_noise_se[r, c] = math.sqrt(noise[sel].sum()) / sel.sum()

    def summary(z, label):
        z = np.abs(np.asarray(z))
        return (f"{label}: |z| mean {z.mean():.2f}, share > 2 {(z > 2).mean():.3f}, "
                f"share > 3 {(z > 3).mean():.3f}, max {z.max():.2f} ({len(z)} values)")

    lines = []
    valid = cell_noise_se > 0
    z = cell_diff[valid] / cell_noise_se[valid]
    worst = np.unravel_index(np.argmax(np.where(valid, np.abs(cell_diff) / np.where(valid, cell_noise_se, 1), 0)), (n, n))
    lines.append(summary(z, "sampled - exact return per cell")
                 + f"; worst {names[worst[0]]} vs {names[worst[1]]}")
    total_se = math.sqrt(noise.sum()) / len(diff)
    lines.append(f"sampled - exact return over all {len(diff)} pairs: {diff.mean():+.4f} "
                 f"(z {diff.mean() / total_se:+.2f})")
    z_noise, z_cell, exact_gap = [], [], 0.0
    for r in range(n):
        for c in range(r + 1, n):
            a, b = cells[r][c], cells[c][r]
            gap = a["return"] + b["return"]
            se = math.hypot(cell_noise_se[r, c], cell_noise_se[c, r])
            if se > 0:
                z_noise.append(gap / se)
            se = math.hypot(a["return_se"], b["return_se"])
            if np.isfinite(se) and se > 0:
                z_cell.append(gap / se)
            exact_gap = max(exact_gap, abs(a["extra"]["exact_return"] + b["extra"]["exact_return"]))
    lines.append(summary(z_noise, "return(A,B) + return(B,A), sampling noise"))
    lines.append(summary(z_cell, "return(A,B) + return(B,A), cells' return_se")
                 + f"; exact: max |sum| {exact_gap:.1e}")
    ri = names.index(RANDOM)
    rr = cells[ri][ri]
    lines.append(f"Random vs Random: win {rr['win']:.3f} loss {rr['loss']:.3f} "
                 f"draw {rr['draw']:.3f} return {rr['return']:+.3f} "
                 f"(z {rr['return'] / cell_noise_se[ri, ri]:+.2f}; exact "
                 f"{rr['extra']['exact_return']:+.4f}, win {rr['extra']['exact_win']:.3f} "
                 f"loss {rr['extra']['exact_loss']:.3f})")
    z = [cells[r][r]["return"] / cell_noise_se[r, r] for r in range(n) if cell_noise_se[r, r] > 0]
    lines.append(summary(z, "diagonal return")
                 + f"; exact: max |return| {max(abs(cells[r][r]['extra']['exact_return']) for r in range(n)):.1e}"
                 + f"; max |win - loss| {max(abs(cells[r][r]['win'] - cells[r][r]['loss']) for r in range(n)):.3f}")
    return lines


def to_json(game: Game, participants: list[Participant], cells, runs_root: str,
            commit: str) -> dict:
    return {
        "game": game.name,
        "game_key": game.key,
        "units": "chips per hand",
        "episodes_per_pair": EPISODES_PER_PAIR,
        "protocol": (
            f"Per checkpoint pair, {NUM_DEALS} random deals, each played twice with "
            "the seats swapped (same env reset key, so the same cards and first player; "
            "independent action randomness); every policy plays its table at the acting "
            "player's information set (PQN greedy, ties to the first legal action; the "
            "others sampled)."
        ),
        "participants": [p.name for p in participants],
        "checkpoints": {p.name: len(p.checkpoints) for p in participants},
        "cells": cells,
        "commit": commit,
        "runs_root": runs_root,
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def run_game(game_key: str, runs_root: str, out_dir: str, seed: int, commit: str,
             runs_root_label: str | None = None) -> dict:
    start = time.time()
    game = GAMES[game_key]()
    participants = load_participants(runs_root, game)
    print(f"[{game.name}] participants: "
          + ", ".join(f"{p.name} x{len(p.checkpoints)}" for p in participants), flush=True)
    print(f"[{game.name}] {check_infosets(make(game.env_id), game)}", flush=True)
    for line in check_logged(game, participants):
        print(f"[{game.name}] {line}", flush=True)
    t_play = time.time()
    result = evaluate(game, participants, seed=seed)
    print(f"[{game.name}] played {len(result['pairs'])} pairs in {time.time() - t_play:.1f}s",
          flush=True)
    for line in sanity_report(game, participants, result):
        print(f"[{game.name}] {line}", flush=True)
    data = to_json(game, participants, result["cells"], runs_root_label or runs_root, commit)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{game.key}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    print(f"[{game.name}] wrote {path}; wall time {time.time() - start:.1f}s", flush=True)
    return data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("runs_root", help="directory with kuhn/ and leduc/ run directories")
    parser.add_argument("out_dir", help="where {kuhn,leduc}.json are written")
    parser.add_argument("--games", nargs="+", default=list(GAMES), choices=list(GAMES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--commit", default=DEFAULT_COMMIT,
                        help="code commit of the runs, recorded in the output")
    parser.add_argument("--runs_root_label", default=None,
                        help="runs_root as recorded in the output (default: as given)")
    args = parser.parse_args(argv)
    for key in args.games:
        run_game(key, args.runs_root, args.out_dir, args.seed, args.commit, args.runs_root_label)


if __name__ == "__main__":
    main()
