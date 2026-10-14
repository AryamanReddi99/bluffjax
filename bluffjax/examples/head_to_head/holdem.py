"""Head-to-head matrix for heads-up Limit and No-Limit Hold'em.

Participants, in this order: PPO-NFSP (half), PPO-NFSP (full), PQN-NFSP (half),
PQN-NFSP (full), ReBeL (half), ReBeL (full), Random.
    PPO-NFSP / PQN-NFSP: the average policy {stage}/avg_policy_{i}.msgpack of
        the run (i = 0..num_seeds-1), loaded exactly as
        holdem_rebel/eval_nfsp.py does (No-Limit through chip_scaled), actions
        sampled from the softmax over the legal actions.
    ReBeL: holdem_rebel.agent.load_rebel({run_dir}/{stage}/value_net_0.msgpack)
        -> RebelPlayer, with search at every betting round as in its own
        evaluation (CFR iterations from the checkpoint's metadata); one run
        directory per seed, the checkpoint index is the run's `seed`.
    Random: uniformly random legal actions (one checkpoint).

Protocol: a cell (row A, column B) averages over every pair (checkpoint i of
A, checkpoint j of B), on the diagonal only i != j (Random vs Random plays
itself). Every pair plays 16 deals, each from both seats (32 hands), with
holdem_rebel.agent.play_match: the two hands of a deal start from the same
env.reset key (same cards and blinds), with independent action randomness,
and every hand is played to its end with env.step_env. A pair's metrics are
the means over its 32 hands from A's point of view (win: A's reward > 0,
loss: < 0, draw: = 0, a split pot, as in the NFSP scripts' evaluation;
return: A's reward in chips); a cell is the mean over its pairs, with
standard errors sd(pair means) / sqrt(#pairs). Cells (A, B) and (B, A) are
played independently.

Batching (it doesn't change the protocol): all participants without search
are one PoolPlayer whose hand h is played by pool member idx[h], so many pairs
share one play_match call; a call has at most one ReBeL checkpoint per side.

Usage:
    python -m bluffjax.examples.head_to_head.holdem <runs_root> <hul|hunl> <out_dir>
        [--seed S] [--commit C] [--solve-chunk N] [--pool-pairs-per-call N]
Writes <out_dir>/<game>.json (the matrix) and <out_dir>/checks/<game>.json
(sanity checks and wall times).
"""

import argparse
import datetime
import json
import math
import os
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from bluffjax import make
from bluffjax.examples.holdem_rebel.agent import Player, load_rebel, play_match
from bluffjax.examples.holdem_rebel.eval_nfsp import ENV_IDS, avg_network
from bluffjax.examples.run_utils import CHECKPOINT_STAGES, checkpoint_path, load_params

GAMES = {
    "hul": {"dir": "HUL", "name": "Limit Hold'em"},
    "hunl": {"dir": "HUNL", "name": "No-Limit Hold'em"},
}
NFSP = (("PPO-NFSP", "ppo"), ("PQN-NFSP", "pqn"))
N_DEALS = 16  # per pair; each deal is played from both seats
NUM_REBEL_SEEDS = 5
DEFAULT_COMMIT = "1156562"
PROTOCOL = (
    "Each checkpoint pair plays 16 deals twice with the seats swapped (mirrored: the same "
    "env reset key, so the same cards and blinds, with independent action randomness), "
    "every hand played to its end with holdem_rebel.agent.play_match; ReBeL searches at "
    "every betting round, NFSP plays its sampled average policy, Random uniformly random "
    "legal actions."
)


# =============================================================================
# Participants
# =============================================================================


class Participant(NamedTuple):
    name: str
    kind: str  # "ppo", "pqn" (NFSP average networks), "rebel" or "random"
    paths: tuple  # one checkpoint file per checkpoint index (None for Random)


def _load_config(run_dir: str) -> dict:
    return OmegaConf.to_container(OmegaConf.load(os.path.join(run_dir, "config.yaml")))


def find_runs(runs_root: str, game: str) -> dict[str, list[tuple[str, dict]]]:
    """Run directories of the game by job_type suffix ("ppo_nfsp", "pqn_nfsp",
    "rebel"): {suffix: [(run_dir, config)]}."""
    root = os.path.join(os.path.expanduser(runs_root), GAMES[game]["dir"])
    runs: dict[str, list] = {"ppo_nfsp": [], "pqn_nfsp": [], "rebel": []}
    for d in sorted(os.listdir(root)):
        run_dir = os.path.join(root, d)
        if not os.path.isfile(os.path.join(run_dir, "config.yaml")):
            continue
        cfg = _load_config(run_dir)
        suffix = cfg["job_type"].removeprefix(f"{game}_")
        if cfg["job_type"].startswith(f"{game}_") and suffix in runs:
            assert cfg["env_name"] == game, (run_dir, cfg["env_name"])
            runs[suffix].append((run_dir, cfg))
    return runs


def rebel_runs_complete(runs_root: str, game: str, num_seeds: int = NUM_REBEL_SEEDS) -> bool:
    """True when the ReBeL runs of every seed have both checkpoints."""
    runs = find_runs(runs_root, game)["rebel"]
    done = {
        cfg["seed"]
        for run_dir, cfg in runs
        if all(os.path.isfile(os.path.join(run_dir, s, "value_net_0.msgpack"))
               for s in CHECKPOINT_STAGES)
    }
    return done >= set(range(num_seeds))


def participants(
    runs_root: str, game: str, num_rebel_seeds: int = NUM_REBEL_SEEDS
) -> tuple[list[Participant], dict]:
    """The participants in matrix order, and the NFSP run configs by kind.
    Fails loudly unless there is exactly one run per NFSP baseline and one
    run per ReBeL seed 0..num_rebel_seeds-1."""
    runs = find_runs(runs_root, game)
    out, cfgs = [], {}
    for label, kind in NFSP:
        found = runs[f"{kind}_nfsp"]
        if len(found) != 1:
            raise RuntimeError(f"{game}: expected one {kind}_nfsp run, found {[r for r, _ in found]}")
        run_dir, cfg = found[0]
        assert cfg.get("env_kwargs", {}) == {"num_agents": 2}, cfg.get("env_kwargs")
        cfgs[kind] = cfg
        for stage in CHECKPOINT_STAGES:
            paths = tuple(checkpoint_path(run_dir, stage, "avg_policy", i) for i in range(cfg["num_seeds"]))
            out.append(Participant(f"{label} ({stage})", kind, paths))
    by_seed = {}
    for run_dir, cfg in runs["rebel"]:
        if cfg["seed"] in by_seed:
            raise RuntimeError(f"{game}: two ReBeL runs with seed {cfg['seed']}: {by_seed[cfg['seed']]}, {run_dir}")
        by_seed[cfg["seed"]] = run_dir
    if sorted(by_seed) != list(range(num_rebel_seeds)):
        raise RuntimeError(f"{game}: ReBeL seeds {sorted(by_seed)}, expected 0..{num_rebel_seeds - 1}")
    for stage in CHECKPOINT_STAGES:
        paths = tuple(os.path.join(by_seed[i], stage, "value_net_0.msgpack") for i in range(num_rebel_seeds))
        out.append(Participant(f"ReBeL ({stage})", "rebel", paths))
    out.append(Participant("Random", "random", (None,)))
    return out, cfgs


# =============================================================================
# Players
# =============================================================================


class PoolPlayer(Player):
    """Every participant without search as one stateless player.

    params = (stacks, idx): stacks[g] holds the params of network group g
    (`nets[g]`) stacked over its members; idx[h] is the pool member playing
    hand h, numbered group by group, then Random (the last member). Each hand
    samples from the softmax over the legal actions of its member's logits
    (Random: uniform over the legal actions), as eval_nfsp.py does.
    """

    def __init__(self, env, nets: list):
        self.env = env
        self.nets = nets

    def act(self, params, state, views, rng, env_states, avail):
        stacks, idx = params
        n = avail.shape[0]
        obs = jax.vmap(self.env.obs_from_state)(env_states)

        def logits_of(net, p):
            out = net.apply(p, obs)
            return out[0] if isinstance(out, tuple) else out

        groups = [
            jax.vmap(lambda p, net=net: logits_of(net, p))(stack)  # (K_g, n, A)
            for net, stack in zip(self.nets, stacks)
        ]
        groups.append(jnp.zeros((1,) + avail.shape))  # Random
        logits = jnp.concatenate(groups)[idx, jnp.arange(n)]
        keys = jax.random.split(rng, n)
        return jax.vmap(lambda k, l, a: jax.random.categorical(k, jnp.where(a, l, -jnp.inf)))(
            keys, logits, avail
        )


class Pool(NamedTuple):
    player: PoolPlayer
    stacks: tuple
    member: dict  # (participant index, checkpoint index) -> pool member


def build_pool(env, parts: list[Participant], cfgs: dict, game: str) -> Pool:
    """The PoolPlayer of every non-search participant, networks restored as
    in eval_nfsp.py."""
    sample_state, _ = env.reset(jax.random.PRNGKey(0))
    action_dim = int(env.get_avail_actions(sample_state).shape[-1])  # as in the NFSP scripts
    nets, stacks, member, k = [], [], {}, 0
    for _, kind in NFSP:
        net = avg_network(game, kind, env, action_dim, cfgs[kind]["fc_dim_size"])
        template = net.init(jax.random.PRNGKey(0), jnp.zeros((env.obs_dim,), jnp.float32))
        params = []
        for p, part in enumerate(parts):
            if part.kind != kind:
                continue
            for i, path in enumerate(part.paths):
                params.append(load_params(path, template))
                member[(p, i)] = k
                k += 1
        nets.append(net)
        stacks.append(jax.tree.map(lambda *x: jnp.stack(x), *params))
    member[(len(parts) - 1, 0)] = k  # Random, the pool's last member
    assert parts[-1].kind == "random"
    return Pool(PoolPlayer(env, nets), tuple(stacks), member)


# =============================================================================
# Matches
# =============================================================================


class Side(NamedTuple):
    """One side of a play_match call: the pool (per-hand members) or one
    ReBeL checkpoint."""

    kind: str  # "pool" or "rebel"
    rebel: int = -1  # index into the loaded ReBeL params


def pair_rewards(result, n_pairs: int, n_deals: int) -> np.ndarray:
    """(n_pairs, n_deals, 2) A's reward per pair, deal and A's seat from a
    play_match result over n_pairs * n_deals deals (pair p = deals
    p * n_deals ... (p + 1) * n_deals - 1)."""
    rewards = np.asarray(result.rewards, np.float64)
    assert rewards.shape == (n_pairs * n_deals, 2), rewards.shape
    return rewards.reshape(n_pairs, n_deals, 2)


class Matcher:
    """Runs play_match for a batch of pairs with one jitted function per
    (side A kind, side B kind) and shape."""

    def __init__(self, env, game, pool: Pool | None, rebel_player, rebel_params: list, n_deals: int):
        self.env, self.game, self.pool = env, game, pool
        self.rebel_player, self.rebel_params = rebel_player, rebel_params
        self.n_deals = n_deals
        self._fns: dict = {}

    def _player(self, side: Side):
        return self.pool.player if side.kind == "pool" else self.rebel_player

    def _params(self, side: Side, members: list[int] | None):
        if side.kind == "rebel":
            return self.rebel_params[side.rebel]
        per_deal = np.repeat(np.asarray(members, np.int32), self.n_deals)
        return self.pool.stacks, jnp.asarray(np.concatenate([per_deal, per_deal]))  # hand h = deal h % n

    def play(self, a: Side, b: Side, members_a, members_b, n_pairs: int, rng) -> np.ndarray:
        """(n_pairs, n_deals, 2) rewards of side A; members_* are the pool
        members per pair of a pool side (None for a ReBeL side)."""
        key = (a.kind, b.kind, n_pairs)
        if key not in self._fns:
            pa, pb, n = self._player(a), self._player(b), n_pairs * self.n_deals
            self._fns[key] = jax.jit(
                lambda xa, xb, k: play_match(self.env, self.game, pa, xa, pb, xb, n, k)
            )
        res = jax.device_get(self._fns[key](self._params(a, members_a), self._params(b, members_b), rng))
        if int(res.unfinished) != 0:
            raise RuntimeError(f"{int(res.unfinished)} unfinished hands ({a}, {b})")
        return pair_rewards(res, n_pairs, self.n_deals)


def checkpoint_pairs(parts: list[Participant]) -> list[tuple[int, int, int, int]]:
    """Every (row r, ckpt i, column c, ckpt j) to play: all i, j, on the
    diagonal only i != j unless the participant has a single checkpoint."""
    out = []
    for r, pr in enumerate(parts):
        for c, pc in enumerate(parts):
            for i in range(len(pr.paths)):
                for j in range(len(pc.paths)):
                    if r == c and i == j and len(pr.paths) > 1:
                        continue
                    out.append((r, i, c, j))
    return out


def play_all(
    matcher: Matcher, parts: list[Participant], rebel_index: dict, rng, pool_pairs_per_call: int,
    log=print,
) -> tuple[dict, dict]:
    """Plays every checkpoint pair; returns ({(r, i, c, j): (n_deals, 2)
    rewards}, wall seconds by group)."""
    groups: dict = {}
    for r, i, c, j in checkpoint_pairs(parts):
        sa = Side("rebel", rebel_index[(r, i)]) if parts[r].kind == "rebel" else Side("pool")
        sb = Side("rebel", rebel_index[(c, j)]) if parts[c].kind == "rebel" else Side("pool")
        groups.setdefault((sa, sb), []).append((r, i, c, j))
    member = matcher.pool.member
    results, seconds = {}, {}
    order = sorted(groups, key=lambda g: (g[0].kind == "rebel") + (g[1].kind == "rebel"))
    for n_group, (sa, sb) in enumerate(order):
        pairs = groups[(sa, sb)]
        # pool-vs-pool pairs in calls of a fixed size (the last one padded),
        # pairs with a ReBeL side in one call per group
        size = pool_pairs_per_call if sa.kind == sb.kind == "pool" else len(pairs)
        t0 = time.time()
        for start in range(0, len(pairs), size):
            chunk = pairs[start:start + size]
            padded = chunk + [chunk[-1]] * (size - len(chunk))
            ma = [member[(r, i)] for r, i, _, _ in padded] if sa.kind == "pool" else None
            mb = [member[(c, j)] for _, _, c, j in padded] if sb.kind == "pool" else None
            rng, k = jax.random.split(rng)
            rewards = matcher.play(sa, sb, ma, mb, size, k)
            for p, pair in enumerate(chunk):
                results[pair] = rewards[p]
        kind = f"{sa.kind}-{sb.kind}"
        seconds[kind] = seconds.get(kind, 0.0) + time.time() - t0
        log(f"[{n_group + 1}/{len(order)}] {kind} A={sa.rebel} B={sb.rebel}: {len(pairs)} pairs, "
            f"{time.time() - t0:.1f} s", flush=True)
    return results, seconds


# =============================================================================
# Metrics
# =============================================================================


def _se(x: np.ndarray) -> float:
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def pair_metrics(rewards: np.ndarray) -> dict[str, float]:
    r = np.asarray(rewards, np.float64).ravel()
    return {"win": float((r > 0).mean()), "loss": float((r < 0).mean()),
            "draw": float((r == 0).mean()), "return": float(r.mean())}


def cell_metrics(pair_rewards_list: list[np.ndarray]) -> dict[str, float]:
    """Mean over pairs of the pair metrics, SEs over pair means."""
    pm = [pair_metrics(r) for r in pair_rewards_list]
    arr = {k: np.array([m[k] for m in pm]) for k in ("win", "loss", "draw", "return")}
    return {
        "win": float(arr["win"].mean()), "loss": float(arr["loss"].mean()),
        "draw": float(arr["draw"].mean()), "return": float(arr["return"].mean()),
        "win_se": _se(arr["win"]), "return_se": _se(arr["return"]),
        "pairs": len(pm), "episodes": int(sum(np.asarray(r).size for r in pair_rewards_list)),
    }


def matrix(parts: list[Participant], results: dict) -> list[list[dict]]:
    cells = []
    for r in range(len(parts)):
        row = []
        for c in range(len(parts)):
            row.append(cell_metrics([v for (rr, _, cc, _), v in sorted(results.items()) if rr == r and cc == c]))
        cells.append(row)
    return cells


def sanity_checks(names: list[str], cells: list[list[dict]], episodes_per_pair: int) -> dict:
    """Zero-sum and symmetry checks: z = (return(A, B) + return(B, A)) / SE
    and (win(A, B) - loss(B, A)) / SE for every A != B, z = return / SE and
    (win - loss) / SE on the diagonal."""
    n = len(names)

    def z(x, se):
        return float(x / se) if se and np.isfinite(se) and se > 0 else (0.0 if x == 0 else float("nan"))

    anti, diag = [], []
    for a in range(n):
        for b in range(a + 1, n):
            x, y = cells[a][b], cells[b][a]
            anti.append({
                "a": names[a], "b": names[b],
                "return_ab": x["return"], "return_ba": y["return"],
                "z_return": z(x["return"] + y["return"], math.hypot(x["return_se"], y["return_se"])),
                "z_win_vs_loss": z(x["win"] - y["loss"], math.hypot(x["win_se"], y["win_se"])),
            })
    for a in range(n):
        x = cells[a][a]
        diag.append({"participant": names[a], "return": x["return"], "win": x["win"], "loss": x["loss"],
                     "z_return": z(x["return"], x["return_se"])})
    finite = [abs(v["z_return"]) for v in anti if np.isfinite(v["z_return"])]
    return {
        "every_pair_all_episodes_complete": all(
            cells[a][b]["episodes"] == cells[a][b]["pairs"] * episodes_per_pair
            for a in range(n) for b in range(n)),
        "random_vs_random": cells[-1][-1],
        "antisymmetry_max_abs_z_return": max(finite) if finite else float("nan"),
        "antisymmetry_n_abs_z_return_above_3": int(sum(v > 3 for v in finite)),
        "antisymmetry": anti,
        "diagonal": diag,
    }


# =============================================================================
# Main
# =============================================================================


def run(
    runs_root: str, game: str, out_dir: str, seed: int = 0, commit: str = DEFAULT_COMMIT,
    solve_chunk: int | None = None, pool_pairs_per_call: int = 128, n_deals: int = N_DEALS,
    num_rebel_seeds: int = NUM_REBEL_SEEDS,
) -> dict:
    t_start = time.time()
    parts, cfgs = participants(runs_root, game, num_rebel_seeds)
    env = make(ENV_IDS[game], num_agents=2)
    pool = build_pool(env, parts, cfgs, game)

    rebel_params, rebel_index, rebel_player, ref_meta = [], {}, None, None
    for p, part in enumerate(parts):
        if part.kind != "rebel":
            continue
        for i, path in enumerate(part.paths):
            kwargs = {} if solve_chunk is None else {"solve_chunk": solve_chunk}
            player, params, meta = load_rebel(path, **kwargs)
            assert meta["game"] == game, (path, meta["game"])
            arch = {k: meta.get(k) for k in ("game", "max_raises", "value_hidden_dim", "value_num_layers", "cfr_iters")}
            if ref_meta is None:
                rebel_player, ref_meta = player, arch
            assert arch == ref_meta, (path, arch, ref_meta)
            rebel_index[(p, i)] = len(rebel_params)
            rebel_params.append(params)
    print(f"{game}: {[(q.name, len(q.paths)) for q in parts]}; ReBeL {ref_meta}, "
          f"solve_chunk {rebel_player.solve_chunk}", flush=True)

    matcher = Matcher(env, rebel_player.game, pool, rebel_player, rebel_params, n_deals)
    results, seconds = play_all(matcher, parts, rebel_index, jax.random.PRNGKey(seed), pool_pairs_per_call)
    for v in results.values():
        assert v.shape == (n_deals, 2)
    cells = matrix(parts, results)
    names = [q.name for q in parts]
    out = {
        "game": GAMES[game]["name"], "game_key": game, "units": "chips per hand",
        "episodes_per_pair": 2 * n_deals,
        "protocol": PROTOCOL if n_deals == N_DEALS else PROTOCOL.replace("16 deals", f"{n_deals} deals"),
        "participants": names,
        "checkpoints": {q.name: len(q.paths) for q in parts},
        "cells": cells,
        "commit": commit, "runs_root": runs_root,
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    wall = time.time() - t_start
    checks = sanity_checks(names, cells, 2 * n_deals)
    # the matrix's Random vs Random cell is a single pair: also play many
    # (not part of the matrix)
    rand = pool.member[(len(parts) - 1, 0)]
    big = [matcher.play(Side("pool"), Side("pool"), [rand] * pool_pairs_per_call, [rand] * pool_pairs_per_call,
                        pool_pairs_per_call, jax.random.fold_in(jax.random.PRNGKey(seed), 10_000 + t))
           for t in range(4)]
    checks["random_vs_random_many_pairs"] = cell_metrics(list(np.concatenate(big)))
    checks.update({"wall_seconds": wall, "seconds_by_group": seconds, "seed": seed,
                   "rebel": ref_meta, "solve_chunk": rebel_player.solve_chunk,
                   "pairs_total": len(results)})
    os.makedirs(os.path.join(out_dir, "checks"), exist_ok=True)
    with open(os.path.join(out_dir, f"{game}.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(out_dir, "checks", f"{game}.json"), "w") as f:
        json.dump(checks, f, indent=1)
    print_matrix(names, cells)
    print(f"checks: complete {checks['every_pair_all_episodes_complete']}, antisymmetry max |z| "
          f"{checks['antisymmetry_max_abs_z_return']:.2f} ({checks['antisymmetry_n_abs_z_return_above_3']} > 3), "
          f"Random vs Random {cells[-1][-1]['return']:+.3f}; wall {wall:.0f} s", flush=True)
    return out


def print_matrix(names: list[str], cells: list[list[dict]]) -> None:
    w = max(len(n) for n in names)
    print("return (row vs column), chips per hand")
    print(" " * w + "".join(f"{n[:16]:>18}" for n in names))
    for name, row in zip(names, cells):
        print(f"{name:<{w}}" + "".join(f"{c['return']:>+10.3f}±{c['return_se']:<7.3f}" for c in row))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_root")
    ap.add_argument("game", choices=sorted(GAMES))
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--commit", default=DEFAULT_COMMIT)
    ap.add_argument("--solve-chunk", type=int, default=None, help="ReBeL subgames solved at once")
    ap.add_argument("--pool-pairs-per-call", type=int, default=128)
    a = ap.parse_args()
    run(a.runs_root, a.game, a.out_dir, a.seed, a.commit, a.solve_chunk, a.pool_pairs_per_call)


if __name__ == "__main__":
    main()
