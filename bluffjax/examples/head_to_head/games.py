"""
Head-to-head matrices of the NFSP baselines in 5-Card Draw, 7-Card Stud,
Goofspiel, Bluff, Kemps and Werewolf.

Participants, in this order: "PPO-NFSP (half)", "PPO-NFSP (full)",
"PQN-NFSP (half)", "PQN-NFSP (full)" (the half/ and full/ checkpoints of the
game's PPO-NFSP and PQN-NFSP runs, seed indices 0..num_seeds-1) and "Random"
(uniformly random legal actions, one checkpoint). Each NFSP participant plays
its average policy, avg_policy_{i}.msgpack, loaded strictly
(run_utils.load_params) into the network its training script documents
(save_checkpoints): ActorCriticDiscreteMLP(action_dim, fc_dim_size) for
PPO-NFSP, ActorDiscreteMLP(action_dim, fc_dim_size) for PQN-NFSP, both
wrapped in chip_scaled(..., env) in 5-Card Draw. Actions are sampled from the
softmax of the logits over the legal actions, as in the in-training
evaluations. The env is built from the runs' config.yaml (env_name,
env_kwargs; both runs of a game must agree), so the observation size is the
one the networks were trained on (load_params checks every shape).

A cell (row A, column B) is the mean over the checkpoint pairs (i of A, j of
B): every pair, except i == j on the diagonal when A has several
checkpoints. Each pair plays EPISODES_PER_PAIR = 32 complete episodes (each
played with env.step_env until done; the env's horizon is the only cut-off)
and its metrics are the means over them. Standard errors are the standard
deviation over the pair means / sqrt(number of pairs) (NaN with one pair).
All numbers are from A's point of view:
- Two-player games (5-Card Draw, 7-Card Stud, Goofspiel): 16 deals, each
  played twice, A in seat 0 and then in seat 1 (mirrored).
- Kemps (2 teams of 2): A's checkpoint plays both partners of one team, B's
  the other team; 16 deals, each played twice with the teams swapped.
- Werewolf (2 werewolves, 4 humans): A's checkpoint controls every player of
  one team, B's the other; 16 role deals, each played twice, A as the
  werewolves and then as the humans. extra: A's win rates as werewolves and
  as villagers (humans).
- Bluff (3 players): A plays one seat, B's checkpoint the other two; A's seat
  is (offset + g) % 3 in episode g from a random offset, with a new deal
  every episode.
A mirrored deal is the same deal key: the same env.reset key (cards, roles,
first player or blinds) and the same key for the env's own randomness at
each step (env.step_env's rng at step t: Kemps' step order, Werewolf's
tie-breaks, 5-Card Draw's reshuffles), while the players' action sampling is
independent in the two games.

Outcomes, as in the training scripts' evaluators:
- 5-Card Draw, 7-Card Stud (play_eval_games): A's net chips; win if
  positive, loss if negative, draw if zero.
- Goofspiel (parallel_nfsp_common.evaluate_goofspiel): win if A is the unique
  top scorer, loss if the opponent scored more, draw on a tie; return = A's
  total reward (its points minus the opponent's).
- Bluff (bluff_nfsp_common.play_games): win if A emptied its hand first, loss
  if another player did, draw if the game was cut off at the horizon; return
  = A's total env reward.
- Kemps (parallel_nfsp_common.evaluate_kemps): the team's reward (+1 win,
  -1 loss, 0 real deal or horizon cut-off).
- Werewolf (play_eval_games): the team's reward (+10 win, -10 loss, 0 no
  winner at the horizon).

The episodes of all pairs of a game are played at once: jax.vmap over the
checkpoint pairs (in chunks of --chunk pairs) and, inside, over the 32
episodes of a pair. One compiled function serves every cell: each
participant is a ParticipantParams (a PPO-NFSP slot, a PQN-NFSP slot and the
kind that says which one plays; Random has zero logits).

    uv run python -m bluffjax.examples.head_to_head.games \
        RUNS_ROOT GAME OUTPUT_DIR [--seed 0] [--chunk 128]

RUNS_ROOT holds {five_card_draw,seven_card_stud,goofspiel,bluff,kemps,
werewolf}/<job_type>_<timestamp>/ (config.yaml, half/, full/); GAME is one of
GAMES (or "all"); the matrix goes to OUTPUT_DIR/<GAME>.json.
"""

import argparse
import datetime
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from bluffjax import make
from bluffjax.environments.werewolf.werewolf import WEREWOLF
from bluffjax.examples.run_utils import checkpoint_path, load_params
from bluffjax.networks.mlp import ActorCriticDiscreteMLP, ActorDiscreteMLP
from bluffjax.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray

EPISODES_PER_PAIR = 32
DEFAULT_COMMIT = "1156562"  # code of the training runs (main)

PPO, PQN, RANDOM = 0, 1, 2  # ParticipantParams.kind
BASELINES = (("PPO-NFSP", "ppo_nfsp", PPO), ("PQN-NFSP", "pqn_nfsp", PQN))
STAGES = ("half", "full")


@dataclass(frozen=True)
class GameSpec:
    key: str  # game_key, directory under the runs root and env name
    name: str
    units: str
    protocol: str
    parallel: bool  # ParallelEnv (every agent acts every step) or AECEnv
    mirrored: bool  # 16 deals x 2 sides; else seat rotation (Bluff)
    chip_scaled: bool  # 5-Card Draw's networks see the chip counts scaled


GAMES = {
    "five_card_draw": GameSpec(
        "five_card_draw",
        "5-Card Draw",
        "chips per hand",
        "Heads-up: 16 deals per pair, each played twice with the seats swapped "
        "(same reset and env-step keys, independent action sampling).",
        parallel=False,
        mirrored=True,
        chip_scaled=True,
    ),
    "seven_card_stud": GameSpec(
        "seven_card_stud",
        "7-Card Stud",
        "chips per hand",
        "Heads-up: 16 deals per pair, each played twice with the seats swapped "
        "(same reset and env-step keys, independent action sampling).",
        parallel=False,
        mirrored=True,
        chip_scaled=False,
    ),
    "goofspiel": GameSpec(
        "goofspiel",
        "Goofspiel",
        "points per game (own points minus the opponent's)",
        "Two players: 16 prize decks per pair, each played twice with the seats "
        "swapped (same reset key, independent action sampling).",
        parallel=True,
        mirrored=True,
        chip_scaled=False,
    ),
    "bluff": GameSpec(
        "bluff",
        "Bluff",
        "env reward per game (+1 per unchallenged card, +1 per challenge won, "
        "-1 per card picked up, +10 for the win)",
        "3 players: the row checkpoint plays one seat and the column checkpoint "
        "the other two; the row's seat is (offset + g) % 3 in episode g from a "
        "random offset, a new deal every episode; draw = cut off at the horizon.",
        parallel=False,
        mirrored=False,
        chip_scaled=False,
    ),
    "kemps": GameSpec(
        "kemps",
        "Kemps",
        "team reward per hand (+1 win, -1 loss)",
        "2 teams of 2: the row checkpoint plays both partners of one team and "
        "the column checkpoint the other; 16 deals per pair, each played twice "
        "with the teams swapped (same reset and env-step keys).",
        parallel=True,
        mirrored=True,
        chip_scaled=False,
    ),
    "werewolf": GameSpec(
        "werewolf",
        "Werewolf",
        "team reward per game (+10 win, -10 loss)",
        "2 werewolves vs 4 villagers: the row checkpoint controls every player "
        "of one team and the column checkpoint the other; 16 role deals per "
        "pair, each played twice with the sides swapped (same reset and "
        "env-step keys).",
        parallel=False,
        mirrored=True,
        chip_scaled=False,
    ),
}


# =============================================================================
# Participants
# =============================================================================


class ParticipantParams(NamedTuple):
    """One participant's checkpoint (or a stack of them, leading axis).
    kind PPO: the PPO-NFSP average policy in `ppo` plays; PQN: the PQN-NFSP
    average policy in `pqn`; RANDOM: zero logits (uniform legal). The unused
    slot holds zeros."""

    ppo: Any
    pqn: Any
    kind: IntArray


class Networks(NamedTuple):
    ppo: Any  # flax module of the PPO-NFSP average policy (actor-critic)
    pqn: Any  # flax module of the PQN-NFSP average policy (actor)


def make_env(env_name: str, env_kwargs: dict):
    return make(env_name, **env_kwargs)


def action_dim_of(env) -> int:
    state, _ = env.reset(jax.random.PRNGKey(0))
    return int(env.get_avail_actions(state).shape[-1])


def make_networks(game: GameSpec, env, hidden_dim: int) -> Networks:
    action_dim = action_dim_of(env)
    ppo = ActorCriticDiscreteMLP(action_dim=action_dim, hidden_dim=hidden_dim)
    pqn = ActorDiscreteMLP(action_dim=action_dim, hidden_dim=hidden_dim)
    if game.chip_scaled:
        from bluffjax.examples.five_card_draw.five_card_draw_ppo_nfsp import (
            chip_scaled,
        )

        ppo, pqn = chip_scaled(ppo, env), chip_scaled(pqn, env)
    return Networks(ppo=ppo, pqn=pqn)


def templates(nets: Networks, obs_dim: int) -> tuple[Any, Any]:
    obs = jnp.zeros((obs_dim,), dtype=jnp.float32)
    key = jax.random.PRNGKey(0)
    return nets.ppo.init(key, obs), nets.pqn.init(key, obs)


def policy_logits(nets: Networks, params: ParticipantParams, obs: FloatArray):
    """Logits of the participant's average policy for obs (..., obs_dim)."""
    ppo_logits, _ = nets.ppo.apply(params.ppo, obs)
    pqn_logits = nets.pqn.apply(params.pqn, obs)
    return jnp.where(
        params.kind == PPO,
        ppo_logits,
        jnp.where(params.kind == PQN, pqn_logits, jnp.zeros_like(pqn_logits)),
    )


def sample_masked(logits: FloatArray, mask: BoolArray, rng: PRNGKeyArray):
    """Sample from the softmax over the legal actions (last axis)."""
    return jax.random.categorical(rng, jnp.where(mask, logits, -jnp.inf))


# =============================================================================
# One episode
# =============================================================================


class EpisodeResult(NamedTuple):
    """One episode, from A's (the row participant's) point of view."""

    win: FloatArray
    loss: FloatArray
    draw: FloatArray
    ret: FloatArray  # A's (team member's) total env reward
    length: IntArray  # env steps
    done: BoolArray  # the episode ended (always, by construction)
    side: IntArray  # A's seat (2 players, Bluff), team (Kemps) or 0 werewolves / 1 humans
    a_seats: BoolArray  # (num_agents,) the seats A's checkpoint plays
    a_moves: IntArray  # (num_agents,) decisions taken with A's policy, per seat
    b_moves: IntArray  # (num_agents,) decisions taken with B's policy


def controlled_by_a(game: GameSpec, env, state, side: IntArray) -> BoolArray:
    """(num_agents,) the seats A's checkpoint plays in this episode."""
    seats = jnp.arange(env.num_agents)
    if game.key == "kemps":
        return env._team_id(seats) == side  # team t = seats {t, t + num_teams}
    if game.key == "werewolf":
        return (state.roles == WEREWOLF) == (side == 0)  # roles fixed at reset
    return seats == side


def outcome(game: GameSpec, final_state, returns: FloatArray, a_seats: BoolArray):
    """(win, loss, draw, ret) of A, from the final state and every seat's
    total reward, with each training script's evaluator's rule."""
    a = a_seats.astype(jnp.float32)
    if game.key == "goofspiel":
        ret = (returns * a).sum()
        points = final_state.points
        win = (final_state.winners & a_seats).any()
        loss = (points * a).sum() < points.max()
        return win, loss, ~win & ~loss, ret
    if game.key == "bluff":
        ret = (returns * a).sum()
        ended = final_state.absorbing[0]  # not set on the horizon cut-off
        won = (final_state.game_winner & a_seats).any()
        return won, ended & ~won, ~ended, ret
    if game.key == "kemps":
        ret = (returns * a).sum() / a.sum()  # both partners get the team's
    elif game.key == "werewolf":
        ret = returns[jnp.argmax(a_seats)]  # a member of A's team
    else:  # poker: net chips of A's seat
        ret = (returns * a).sum()
    return ret > 0, ret < 0, ret == 0, ret


def play_episode(
    game: GameSpec,
    env,
    nets: Networks,
    params_a: ParticipantParams,
    params_b: ParticipantParams,
    deal_key: PRNGKeyArray,
    side: IntArray,
    action_key: PRNGKeyArray,
    return_initial_state: bool = False,
):
    """One complete episode, A's checkpoint in the seats `side` gives it and
    B's in the others. deal_key fixes the deal (reset key) and the env's own
    randomness at every step; action_key the players' sampling."""
    rng_reset, rng_env = jax.random.split(deal_key)
    state0, obs0 = env.reset(rng_reset)
    a_seats = controlled_by_a(game, env, state0, side)
    n = env.num_agents

    def body(carry):
        state, obs, _, returns, t, a_moves, b_moves = carry
        rng_a, rng_b = jax.random.split(jax.random.fold_in(action_key, t))
        mask = env.get_avail_actions(state)
        act_a = sample_masked(policy_logits(nets, params_a, obs), mask, rng_a)
        act_b = sample_masked(policy_logits(nets, params_b, obs), mask, rng_b)
        if game.parallel:  # every seat acts
            action = jnp.where(a_seats, act_a, act_b)
            a_moves = a_moves + a_seats
            b_moves = b_moves + ~a_seats
        else:  # the player to act
            player = jax.nn.one_hot(state.current_player_idx, n, dtype=jnp.int32)
            use_a = a_seats[state.current_player_idx]
            action = jnp.where(use_a, act_a, act_b)
            a_moves = a_moves + player * use_a
            b_moves = b_moves + player * ~use_a
        state, obs, reward, _, done, _ = env.step_env(
            jax.random.fold_in(rng_env, t), state, action
        )
        return state, obs, done, returns + reward, t + 1, a_moves, b_moves

    zeros_n = jnp.zeros((n,), dtype=jnp.int32)
    init = (
        state0,
        obs0,
        jnp.bool_(False),
        jnp.zeros((n,), dtype=jnp.float32),
        jnp.int32(0),
        zeros_n,
        zeros_n,
    )
    state, _, done, returns, length, a_moves, b_moves = lax.while_loop(
        lambda c: ~c[2], body, init
    )
    win, loss, draw, ret = outcome(game, state, returns, a_seats)
    result = EpisodeResult(
        win=win.astype(jnp.float32),
        loss=loss.astype(jnp.float32),
        draw=draw.astype(jnp.float32),
        ret=ret.astype(jnp.float32),
        length=length,
        done=done,
        side=jnp.asarray(side, dtype=jnp.int32),
        a_seats=a_seats,
        a_moves=a_moves,
        b_moves=b_moves,
    )
    if return_initial_state:
        return result, state0
    return result


# =============================================================================
# The 32 episodes of a pair
# =============================================================================


def episode_schedule(
    game: GameSpec, num_agents: int, pair_key: PRNGKeyArray,
    num_episodes: int = EPISODES_PER_PAIR,
):
    """(deal keys, A's sides, action keys) of a pair's episodes.

    Mirrored games: deal d (of num_episodes / 2) is played in episodes 2d (A
    on side 0: seat 0, team 0, the werewolves) and 2d + 1 (side 1) with the
    same deal key. Bluff: a new deal every episode, A's seat
    (offset + g) % num_agents from a random offset."""
    rng_deal, rng_action, rng_seat = jax.random.split(pair_key, 3)
    action_keys = jax.random.split(rng_action, num_episodes)
    games = jnp.arange(num_episodes)
    if game.mirrored:
        if num_episodes % 2:
            raise ValueError("mirrored deals need an even number of episodes")
        deals = jax.random.split(rng_deal, num_episodes // 2)
        deal_keys = jnp.repeat(deals, 2, axis=0)
        sides = games % 2
    else:
        deal_keys = jax.random.split(rng_deal, num_episodes)
        offset = jax.random.randint(rng_seat, (), 0, num_agents)
        sides = (offset + games) % num_agents
    return deal_keys, sides.astype(jnp.int32), action_keys


def play_pair(
    game: GameSpec, env, nets: Networks, params_a, params_b, pair_key,
    num_episodes: int = EPISODES_PER_PAIR, return_initial_state: bool = False,
):
    """The num_episodes episodes of one checkpoint pair, in parallel."""
    deal_keys, sides, action_keys = episode_schedule(
        game, env.num_agents, pair_key, num_episodes
    )
    return jax.vmap(
        lambda d, s, k: play_episode(
            game, env, nets, params_a, params_b, d, s, k, return_initial_state
        )
    )(deal_keys, sides, action_keys)


def make_play_chunk(game: GameSpec, env, nets: Networks, num_episodes: int):
    """jit(bank, a_idx, b_idx, pair_keys) -> EpisodeResult (pairs, episodes):
    pair p is bank[a_idx[p]] (A) against bank[b_idx[p]] (B)."""

    def play_chunk(bank, a_idx, b_idx, pair_keys):
        take = lambda idx: jax.tree_util.tree_map(lambda x: x[idx], bank)  # noqa: E731
        return jax.vmap(
            lambda pa, pb, k: play_pair(game, env, nets, pa, pb, k, num_episodes)
        )(take(a_idx), take(b_idx), pair_keys)

    return jax.jit(play_chunk)


# =============================================================================
# Runs, checkpoints and pairs
# =============================================================================


@dataclass
class Participant:
    name: str
    bank_rows: list[int]  # rows of the bank, one per checkpoint (seed index)


def find_run(runs_root: str, game: GameSpec, job: str) -> str:
    """The single run directory of job_type {game}_{job} under runs_root/game."""
    job_type = f"{game.key}_{job}"
    base = os.path.join(os.path.expanduser(runs_root), game.key)
    runs = sorted(
        os.path.join(base, d)
        for d in os.listdir(base)
        if d.startswith(job_type + "_")
        and os.path.isfile(os.path.join(base, d, "config.yaml"))
    )
    runs = [r for r in runs if OmegaConf.load(os.path.join(r, "config.yaml")).job_type == job_type]
    if len(runs) != 1:
        raise ValueError(f"expected one {job_type} run under {base}, found {runs}")
    return runs[0]


def load_participants(runs_root: str, game: GameSpec):
    """Loads every checkpoint of the game's participants.

    Returns (env, nets, bank, participants, info): bank is a
    ParticipantParams with one row per checkpoint (the last row is Random),
    participants in matrix order."""
    configs = {}
    for _, job, _ in BASELINES:
        run_dir = find_run(runs_root, game, job)
        configs[job] = (run_dir, OmegaConf.to_container(
            OmegaConf.load(os.path.join(run_dir, "config.yaml")), resolve=True
        ))
    (_, first), *rest = configs.values()
    for _, cfg in rest:
        for k in ("env_name", "env_kwargs", "fc_dim_size"):
            if cfg[k] != first[k]:
                raise ValueError(f"runs disagree on {k}: {cfg[k]} vs {first[k]}")
    if first["env_name"] != game.key:
        raise ValueError(f"env_name {first['env_name']} is not {game.key}")
    env = make_env(first["env_name"], first["env_kwargs"])
    nets = make_networks(game, env, first["fc_dim_size"])
    ppo_template, pqn_template = templates(nets, env.obs_dim)
    zeros = lambda t: jax.tree_util.tree_map(jnp.zeros_like, t)  # noqa: E731

    rows, participants, info = [], [], {"runs": {}}
    for label, job, kind in BASELINES:
        run_dir, cfg = configs[job]
        info["runs"][label] = run_dir
        for stage in STAGES:
            p = Participant(f"{label} ({stage})", [])
            for i in range(cfg["num_seeds"]):
                path = checkpoint_path(run_dir, stage, "avg_policy", i)
                if kind == PPO:
                    row = ParticipantParams(
                        load_params(path, ppo_template), zeros(pqn_template), PPO
                    )
                else:
                    row = ParticipantParams(
                        zeros(ppo_template), load_params(path, pqn_template), PQN
                    )
                p.bank_rows.append(len(rows))
                rows.append(row)
            participants.append(p)
    participants.append(Participant("Random", [len(rows)]))
    rows.append(ParticipantParams(zeros(ppo_template), zeros(pqn_template), RANDOM))
    bank = jax.tree_util.tree_map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *rows)
    info.update(env_kwargs=first["env_kwargs"], fc_dim_size=first["fc_dim_size"],
                obs_dim=env.obs_dim, action_dim=action_dim_of(env))
    return env, nets, bank, participants, info


def make_pairs(participants: list[Participant]):
    """[(row, col, bank row of A, bank row of B)]: every checkpoint pair of
    every cell, without i == j on the diagonal when there are several."""
    pairs = []
    for r, pa in enumerate(participants):
        for c, pb in enumerate(participants):
            for i, ra in enumerate(pa.bank_rows):
                for j, rb in enumerate(pb.bank_rows):
                    if r == c and len(pa.bank_rows) > 1 and i == j:
                        continue
                    pairs.append((r, c, ra, rb))
    return pairs


def play_all_pairs(game, env, nets, bank, pairs, seed: int, chunk: int,
                   num_episodes: int = EPISODES_PER_PAIR) -> EpisodeResult:
    """EpisodeResult (num_pairs, num_episodes) as numpy arrays. Pair p's key
    is fold_in(PRNGKey(seed), p)."""
    play_chunk = make_play_chunk(game, env, nets, num_episodes)
    a_rows = np.array([p[2] for p in pairs], dtype=np.int32)
    b_rows = np.array([p[3] for p in pairs], dtype=np.int32)
    keys = jax.vmap(lambda p: jax.random.fold_in(jax.random.PRNGKey(seed), p))(
        jnp.arange(len(pairs))
    )
    out = []
    for start in range(0, len(pairs), chunk):
        stop = min(start + chunk, len(pairs))
        pad = chunk - (stop - start)  # same shapes for every chunk: one compile
        idx = np.concatenate([np.arange(start, stop), np.full(pad, start)])
        res = play_chunk(bank, a_rows[idx], b_rows[idx], keys[idx])
        res = jax.tree_util.tree_map(lambda x: np.asarray(x)[: stop - start], res)
        out.append(res)
        print(f"  pairs {stop}/{len(pairs)}", flush=True)
    return jax.tree_util.tree_map(lambda *xs: np.concatenate(xs), *out)


# =============================================================================
# Matrix
# =============================================================================


def _f(x) -> float:
    return float(x)


def summarize(game: GameSpec, participants, pairs, results: EpisodeResult,
              num_episodes: int = EPISODES_PER_PAIR) -> list[list[dict]]:
    """cells[r][c] from the per-episode results of every pair."""
    if results.win.shape != (len(pairs), num_episodes):
        raise AssertionError(f"results shape {results.win.shape}")
    if not results.done.all():
        raise AssertionError("an episode did not end")
    total = results.win + results.loss + results.draw
    if not np.all(total == 1.0):
        raise AssertionError("every episode must be exactly one of win/loss/draw")
    pair_mean = {k: getattr(results, k).mean(axis=1) for k in ("win", "loss", "draw", "ret")}
    if game.key == "werewolf":
        ww = results.side == 0
        pair_mean["win_as_werewolf"] = (results.win * ww).sum(1) / ww.sum(1)
        pair_mean["win_as_villager"] = (results.win * ~ww).sum(1) / (~ww).sum(1)
    row_of = np.array([p[0] for p in pairs])
    col_of = np.array([p[1] for p in pairs])
    n = len(participants)
    cells = [[None] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            sel = (row_of == r) & (col_of == c)
            k = int(sel.sum())

            def se(x):
                return _f(np.std(x[sel], ddof=1) / math.sqrt(k)) if k > 1 else float("nan")

            cell = {
                "win": _f(pair_mean["win"][sel].mean()),
                "loss": _f(pair_mean["loss"][sel].mean()),
                "draw": _f(pair_mean["draw"][sel].mean()),
                "return": _f(pair_mean["ret"][sel].mean()),
                "win_se": se(pair_mean["win"]),
                "return_se": se(pair_mean["ret"]),
                "pairs": k,
                "episodes": int(k * num_episodes),
            }
            if game.key == "werewolf":
                cell["extra"] = {
                    "win_as_werewolf": _f(pair_mean["win_as_werewolf"][sel].mean()),
                    "win_as_villager": _f(pair_mean["win_as_villager"][sel].mean()),
                }
            cells[r][c] = cell
    return cells


def check_routing(game: GameSpec, env, results: EpisodeResult) -> None:
    """Raises unless, in every episode, A's checkpoint played exactly the seats
    of its side (seat `side`; Kemps: team `side`; Werewolf: the werewolves on
    side 0, the humans on side 1) and took every decision there, and B's
    checkpoint every decision in the other seats."""
    a, b, a_seats = results.a_moves, results.b_moves, results.a_seats
    steps = results.length.sum() * (env.num_agents if game.parallel else 1)
    if (a + b).sum() != steps:
        raise AssertionError("decision counts don't add up to the episode lengths")
    if np.any(a[~a_seats] != 0) or np.any(b[a_seats] != 0):
        raise AssertionError("a checkpoint played a seat that isn't its own")
    seats = np.arange(env.num_agents)
    side = results.side[..., None]
    if game.key == "werewolf":
        size = np.where(results.side == 0, env.num_werewolves,
                        env.num_agents - env.num_werewolves)
        if np.any(a_seats.sum(-1) != size):
            raise AssertionError("A doesn't control exactly one Werewolf team")
        return
    expected = (seats % env.num_teams == side) if game.key == "kemps" else (seats == side)
    if np.any(a_seats != expected):
        raise AssertionError("A's seats don't match its side")


def sanity_report(game: GameSpec, participants, cells) -> list[str]:
    """Human-readable checks: Random vs Random, the diagonal and, in the
    zero-sum games, return(A, B) + return(B, A) in units of their joint SE."""
    names = [p.name for p in participants]
    n = len(names)
    lines = []
    rr = cells[n - 1][n - 1]
    lines.append(
        f"Random vs Random: W/L/D {rr['win']:.3f}/{rr['loss']:.3f}/{rr['draw']:.3f} "
        f"return {rr['return']:+.3f} ({rr['episodes']} episodes)"
    )
    for r in range(n):
        d = cells[r][r]
        z = d["return"] / d["return_se"] if d["return_se"] > 0 else float("nan")
        lines.append(
            f"diagonal {names[r]}: W/L/D {d['win']:.3f}/{d['loss']:.3f}/{d['draw']:.3f} "
            f"return {d['return']:+.3f} +- {d['return_se']:.3f}"
            + ("" if game.key == "bluff" else f" (return / SE = {z:+.1f})")
        )
    if game.key != "bluff":  # zero-sum between the sides
        worst = (0.0, "")
        for r in range(n):
            for c in range(r + 1, n):
                x, y = cells[r][c], cells[c][r]
                se = math.sqrt(np.nan_to_num(x["return_se"]) ** 2 + np.nan_to_num(y["return_se"]) ** 2)
                z = (x["return"] + y["return"]) / se if se > 0 else float("nan")
                if abs(z) > abs(worst[0]):
                    worst = (z, f"{names[r]} / {names[c]}")
        lines.append(f"antisymmetry: largest |return(A,B) + return(B,A)| / SE = {worst[0]:+.2f} ({worst[1]})")
    return lines


def evaluate_game(runs_root: str, game_key: str, output_dir: str, seed: int = 0,
                  chunk: int = 128, commit: str = DEFAULT_COMMIT) -> dict:
    game = GAMES[game_key]
    start = time.time()
    env, nets, bank, participants, info = load_participants(runs_root, game)
    print(f"{game.name}: env_kwargs {info['env_kwargs']}, obs_dim {info['obs_dim']}, "
          f"action_dim {info['action_dim']}, fc_dim_size {info['fc_dim_size']}")
    for label, run in info["runs"].items():
        print(f"  {label}: {run}")
    pairs = make_pairs(participants)
    print(f"  {len(pairs)} pairs x {EPISODES_PER_PAIR} episodes", flush=True)
    results = play_all_pairs(game, env, nets, bank, pairs, seed, chunk)
    check_routing(game, env, results)
    cells = summarize(game, participants, pairs, results)
    wall = time.time() - start
    out = {
        "game": game.name,
        "game_key": game.key,
        "units": game.units,
        "episodes_per_pair": EPISODES_PER_PAIR,
        "protocol": game.protocol,
        "participants": [p.name for p in participants],
        "checkpoints": {p.name: len(p.bank_rows) for p in participants},
        "cells": cells,
        "commit": commit,
        "runs_root": runs_root,
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.expanduser(output_dir), exist_ok=True)
    path = os.path.join(os.path.expanduser(output_dir), f"{game.key}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"  mean episode length {results.length.mean():.1f}, max {results.length.max()}")
    rr = [p for p, (r, c, _, _) in enumerate(pairs) if r == c == len(participants) - 1]
    rr_ret = results.ret[rr].ravel()
    print(f"  Random vs Random episodes: return {rr_ret.mean():+.3f} +- "
          f"{rr_ret.std(ddof=1) / math.sqrt(rr_ret.size):.3f} (episode SE)")
    for line in sanity_report(game, participants, cells):
        print("  " + line)
    print(f"  wrote {path} in {wall:.1f} s", flush=True)
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs_root")
    parser.add_argument("game", choices=list(GAMES) + ["all"])
    parser.add_argument("output_dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=128, help="pairs per vmapped batch")
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    args = parser.parse_args(argv)
    for key in GAMES if args.game == "all" else [args.game]:
        evaluate_game(args.runs_root, key, args.output_dir, args.seed, args.chunk, args.commit)


if __name__ == "__main__":
    main()
