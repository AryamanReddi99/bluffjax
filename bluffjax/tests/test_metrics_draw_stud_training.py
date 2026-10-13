"""5-Card Draw and 7-Card Stud training scripts (PPO-NFSP, PQN-NFSP): the game
metrics are counted at every rollout step with their carry kept across
updates and logged every update for both policies, everything logged goes to
results.jsonl, and the half- and full-budget checkpoints are the networks
after exactly half and all of the updates and load strictly.
"""

import functools
import importlib
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.examples.run_utils import (
    ResultsLog,
    checkpoint_path,
    half_update_steps,
    load_params,
    make_run_dir,
    read_results,
)
from bluffjax.metrics import GameMetrics, credit, make_metrics, masked_totals

SCRIPT_NAMES = [
    "five_card_draw_ppo_nfsp",
    "five_card_draw_pqn_nfsp",
    "seven_card_stud_ppo_nfsp",
    "seven_card_stud_pqn_nfsp",
]
KEY = jax.random.PRNGKey(0)


def _game(name: str) -> str:
    return "five_card_draw" if name.startswith("five_card_draw") else "seven_card_stud"


def _script(name: str):
    for dep in ("distrax", "optax", "hydra", "wandb"):
        pytest.importorskip(dep)
    return importlib.import_module(f"bluffjax.examples.{_game(name)}.{name}")


def tiny_config(name: str, **extra) -> dict:
    game, ppo = _game(name), "ppo" in name
    config = dict(
        num_envs=16, num_steps_per_env_per_update=8, num_epochs=2,
        num_minibatches=2, lr=2.5e-4, anneal_lr=True, gamma=0.99,
        max_grad_norm=0.5, fc_dim_size=16, anticipatory_eta=0.3,
        sl_reservoir_capacity=200, sl_batch_size=32, sl_num_steps_per_update=2,
        sl_lr=3e-4, env_name=game, seed=0, num_seeds=2,
        env_kwargs=dict(num_agents=2, horizon=100_000)
        | (dict(init_chips=100) if game == "five_card_draw" else {}),
        compare=False, compare_mode="random", compare_with="",
        compare_network_type="actor_critic" if ppo else "actor",
        compare_episodes=4, compare_interval=2,
        project="bluffjax", job_type=name, wandb=False, save_checkpoints=True,
        num_update_steps=5,
    )
    if ppo:
        config.update(gae_lambda=0.95, clip_eps=0.2, vf_clip=0.2, ent_coef=0.01, vf_coef=1.0)
    else:
        config.update(start_e=0.5, end_e=0.05, exploration_fraction=0.5, q_lambda=0.9)
    config.update(extra)
    config["num_gradient_steps"] = (
        config["num_update_steps"] * config["num_epochs"] * config["num_minibatches"]
    )
    config["half_update_steps"] = half_update_steps(config["num_update_steps"])
    return config


class _Log:
    """Stands in for the run's ResultsLog."""

    def __init__(self):
        self.rows = []

    def log(self, metrics, seed_index=0):
        self.rows.append(dict(metrics, seed_index=int(seed_index)))


class _Probe(GameMetrics):
    """Counts every decision for the acting player and, at the end of a hand,
    the hand and its length (from the carry) for every player. "spanning"
    counts the hands that started before the current rollout (the carry also
    holds the env's step count since training started)."""

    def __init__(self, env, rollout_steps):
        super().__init__(env)
        self.rollout_steps = rollout_steps

    def init_carry(self):
        return (jnp.int32(0), jnp.int32(0))  # steps of this hand, steps so far

    def step(self, carry, state, action, next_state, reward, done, info, probs=None):
        length, t = carry[0] + 1, carry[1]
        everyone = jnp.ones(self.num_agents, jnp.float32) * done
        counts = {
            "decisions": credit(self.num_agents, state.current_player_idx),
            "hands": everyone,
            "length": everyone * length,
            "spanning": everyone * (length > t % self.rollout_steps + 1),
        }
        return (jnp.where(done, 0, length), t + 1), counts

    def summarize(self, totals):
        return dict(totals)


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_game_metrics_plumbing(monkeypatch, name):
    """Per update, the logged game/ and game_br/ counts are the sums over every
    step, env and player of the rollout, split by the policy each player
    follows in its hand, and the metric carry persists across updates (hand
    lengths from the carry equal the env's, also for hands spanning two
    rollouts)."""
    mod = _script(name)
    config = tiny_config(name, num_update_steps=6)
    T, N, P = config["num_steps_per_env_per_update"], config["num_envs"], 2
    monkeypatch.setattr(mod, "make_metrics", lambda env_id, env: _Probe(env, T))
    log = _Log()
    monkeypatch.setattr(mod, "RESULTS_LOG", log)
    jax.block_until_ready(jax.jit(mod.make_train(config))(KEY, jnp.int32(0)))
    jax.effects_barrier()
    assert len(log.rows) == config["num_update_steps"]
    spanning = 0.0
    for u, row in enumerate(log.rows):
        assert row["update_step"] == u and row["env_steps"] == (u + 1) * N * T
        decisions = row["game/decisions"] + row["game_br/decisions"]
        assert decisions == N * T
        assert row["game_br/decisions"] / decisions == pytest.approx(
            row["br_action_frac_rollout"], abs=1e-6
        )
        hands = row["hands_completed"]
        assert hands > 0
        assert row["game/hands"] + row["game_br/hands"] == P * hands
        assert row["game/length"] + row["game_br/length"] == pytest.approx(
            P * row["ep_length_avg"] * hands, rel=1e-5
        )
        spanning += row["game/spanning"] + row["game_br/spanning"]
    assert spanning > 0  # some hands span two rollouts


def _metric_keys(name: str) -> set[str]:
    """Every key the script's game metrics log, from one step of the env."""
    env = make(_game(name), num_agents=2)
    gm = make_metrics(_game(name), env)
    state, _ = env.reset(KEY)
    action = jnp.argmax(env.get_avail_actions(state))
    next_state, _, reward, _, done, info = env.step(KEY, state, action)
    _, counts = gm.step(gm.init_carry(), state, action, next_state, reward, done, info)
    keys = gm.summarize(masked_totals(counts, jnp.ones(2, bool)))
    assert set(gm.rate_names) <= set(keys)
    return {f"{prefix}/{k}" for prefix in ("game", "game_br") for k in keys}


def _check_game_metrics(name: str, records: list[dict]) -> None:
    keys = _metric_keys(name)
    for row in records:
        assert keys <= set(row), sorted(keys - set(row))
        for key in keys:
            value = row[key]
            stat = key.split("/", 1)[1]
            if stat.startswith("n_"):
                assert value >= 0 and value == int(value), (key, value)
            elif stat == "cards_drawn":
                assert math.isnan(value) or 0 <= value <= 5, (key, value)
            else:
                assert math.isnan(value) or 0 <= value <= 1, (key, value)
        hands = row["hands_completed"]
        if _game(name) == "seven_card_stud":
            # every hand is credited to both players, under either policy
            assert row["game/n_hands"] + row["game_br/n_hands"] == 2 * hands
            for prefix in ("game", "game_br"):
                if row[f"{prefix}/n_hands"] > 0:
                    total = row[f"{prefix}/showdown_rate"] + sum(
                        row[f"{prefix}/fold_street_{k}"] for k in range(3, 8)
                    )
                    assert total == pytest.approx(1.0, abs=1e-5)
    # the metrics see events, not only NaN
    for key in ("game/n_draws", "game/n_hands"):
        if key in keys:
            assert sum(row[key] for row in records) > 0


# logged per policy as {stat}_{avg|br}_vs_{random|baseline}
EVAL_STATS = (
    "win_rate", "loss_rate", "draw_rate", "win_rate_decided", "avg_return",
    "avg_episode_length",
)


def _check_eval_metrics(row: dict, opponent: str) -> None:
    """The evaluation metrics of both policies, and nothing else with _vs_."""
    names = [f"{policy}_vs_{opponent}" for policy in ("avg", "br")]
    assert {k for k in row if "_vs_" in k} == {
        f"{stat}_{name}" for stat in EVAL_STATS for name in names
    }
    for name in names:
        total = sum(row[f"{x}_rate_{name}"] for x in ("win", "loss", "draw"))
        assert total == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= row[f"win_rate_decided_{name}"] <= 1.0
        assert row[f"avg_episode_length_{name}"] >= 1.0


def _params_close(a, b, atol=1e-6):
    leaves = zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    return all(np.allclose(np.asarray(x), np.asarray(y), rtol=0, atol=atol) for x, y in leaves)


def _max_diff(a, b):
    leaves = zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    return max(float(np.abs(np.asarray(x) - np.asarray(y)).max()) for x, y in leaves)


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_results_log_half_checkpoint_and_round_trip(monkeypatch, tmp_path, name):
    mod = _script(name)
    U = 5  # half after 2 updates (floor)
    config = tiny_config(name, num_update_steps=U, compare=True, save_dir=str(tmp_path))
    assert config["half_update_steps"] == 2
    run_dir = make_run_dir(config)
    monkeypatch.setattr(mod, "RESULTS_LOG", ResultsLog(run_dir, []))
    train = mod.make_train(config)
    keys = jax.random.split(KEY, 2)
    seeds = jnp.arange(2)
    full = jax.jit(jax.vmap(train))(keys, seeds)
    jax.effects_barrier()

    # results.jsonl: one record per update and seed, with the game metrics
    records = read_results(run_dir)
    assert len(records) == 2 * U
    for i in range(2):
        rows = [r for r in records if r["seed_index"] == i]
        assert [r["update_step"] for r in rows] == list(range(U))
        T, N = config["num_steps_per_env_per_update"], config["num_envs"]
        assert [r["env_steps"] for r in rows] == [(u + 1) * N * T for u in range(U)]
        assert all("wall_time" in r for r in rows)
        # evaluation metrics only where an evaluation ran
        evaluated = [any("_vs_random" in k for k in r) for r in rows]
        assert evaluated == [u % 2 == 0 or u == U - 1 for u in range(U)]
        for r in rows:
            if any("_vs_" in k for k in r):
                _check_eval_metrics(r, "random")
    _check_game_metrics(name, records)

    # the half checkpoint is the state of the run after exactly 2 updates
    partial = {}
    logs = {}
    for n in (1, 2, 3):
        logs[n] = _Log()
        monkeypatch.setattr(mod, "RESULTS_LOG", logs[n])
        partial[n] = jax.jit(jax.vmap(functools.partial(train, num_updates=n)))(keys, seeds)
        jax.effects_barrier()
    assert (np.asarray(full.update_step) == U).all()
    for net, attr in (("avg_params", "avg_train_state"), ("br_params", "br_train_state")):
        half = getattr(full.half, net)
        at = {n: getattr(partial[n], attr).params for n in partial}
        assert _params_close(half, at[2]), _max_diff(half, at[2])
        assert _max_diff(half, at[1]) > 1e-5 and _max_diff(half, at[3]) > 1e-5
        assert _max_diff(half, getattr(full, attr).params) > 1e-5
    # the schedules span the whole run, also in a run stopped early
    E, M = config["num_epochs"], config["num_minibatches"]
    assert (np.asarray(partial[2].br_train_state.step) == 2 * E * M).all()
    for n, log in logs.items():
        assert len(log.rows) == 2 * n
        for row in log.rows:
            u = row["update_step"]
            assert row["br_lr"] == pytest.approx(config["lr"] * (1 - (u + 1) / U), abs=1e-9)
            if "pqn" in name:
                frac = min(1.0, u / (config["exploration_fraction"] * U))
                eps = config["start_e"] + frac * (config["end_e"] - config["start_e"])
                assert row["epsilon"] == pytest.approx(eps, rel=1e-6)

    # saved files and strict loading
    full_ckpt = mod.Checkpoint(full.avg_train_state.params, full.br_train_state.params)
    assert mod.save_checkpoints(run_dir, 2, full.half, full_ckpt) == run_dir
    expected = {"config.yaml", "results.jsonl"} | {
        f"{stage}/{net}_{i}.msgpack"
        for stage in ("half", "full") for net in ("avg_policy", "br") for i in range(2)
    }
    found = {
        os.path.relpath(os.path.join(d, f), run_dir)
        for d, _, files in os.walk(run_dir) for f in files
    }
    assert found == expected

    from bluffjax.networks.mlp import ActorCriticDiscreteMLP, ActorDiscreteMLP, QNetworkDiscreteMLP

    env = make(config["env_name"], **config["env_kwargs"])
    obs = jnp.zeros((env.obs_dim,))
    A, H = env.action_space().n, config["fc_dim_size"]
    assert A == (37 if _game(name) == "five_card_draw" else 4)
    # 5-Card Draw wraps every network in its chip scaling (parameters under "net")
    wrap = (lambda net: mod.chip_scaled(net, env)) if _game(name) == "five_card_draw" else (lambda net: net)
    if "ppo" in name:
        avg_net = br_net = wrap(ActorCriticDiscreteMLP(A, H))
        wrong = wrap(QNetworkDiscreteMLP(A, H))
    else:
        avg_net, br_net = wrap(ActorDiscreteMLP(A, H)), wrap(QNetworkDiscreteMLP(A, H))
        wrong = wrap(ActorCriticDiscreteMLP(A, H))
    templates = {"avg_policy": avg_net.init(KEY, obs), "br": br_net.init(KEY, obs)}
    for stage, ckpt in (("half", full.half), ("full", full_ckpt)):
        for net, params in (("avg_policy", ckpt.avg_params), ("br", ckpt.br_params)):
            for i in range(2):
                loaded = load_params(checkpoint_path(run_dir, stage, net, i), templates[net])
                jax.tree_util.tree_map(
                    lambda g, w: np.testing.assert_array_equal(np.asarray(g), np.asarray(w)[i]),
                    loaded, params,
                )
    with pytest.raises(ValueError):
        load_params(checkpoint_path(run_dir, "full", "br", 0), wrong.init(KEY, obs))
    with pytest.raises(ValueError):
        load_params(
            checkpoint_path(run_dir, "full", "avg_policy", 0),
            wrap((ActorCriticDiscreteMLP if "ppo" in name else ActorDiscreteMLP)(A, 2 * H)).init(KEY, obs),
        )
    if _game(name) == "five_card_draw":  # an unscaled network doesn't load
        with pytest.raises(ValueError):
            load_params(
                checkpoint_path(run_dir, "full", "avg_policy", 0),
                (ActorCriticDiscreteMLP if "ppo" in name else ActorDiscreteMLP)(A, H).init(KEY, obs),
            )

    # a saved network as the evaluation opponent (compare_mode=checkpoint)
    network_type = "actor_critic" if "ppo" in name else "actor"
    log = _Log()
    monkeypatch.setattr(mod, "RESULTS_LOG", log)
    eval_config = tiny_config(
        name, num_update_steps=1, compare=True, compare_mode="checkpoint",
        compare_with=checkpoint_path(run_dir, "full", "avg_policy", 1),
        compare_network_type=network_type,
    )
    jax.block_until_ready(jax.jit(mod.make_train(eval_config))(KEY, jnp.int32(0)))
    jax.effects_barrier()
    _check_eval_metrics(log.rows[0], "baseline")
    with pytest.raises(ValueError):
        mod.make_train(dict(eval_config, compare_network_type="q_network"))


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_main_writes_the_run_directory(monkeypatch, tmp_path, name):
    """main() end to end (hydra's decorator passes a given config through):
    run directory with config.yaml, results.jsonl and the checkpoints."""
    from omegaconf import OmegaConf

    mod = _script(name)
    algo = "ppo" if "ppo" in name else "pqn"
    config = OmegaConf.load(
        os.path.join(os.path.dirname(mod.__file__), f"config_{algo}_nfsp.yaml")
    )
    del config["hydra"]
    tiny = tiny_config(name, compare=True)
    for key in ("num_update_steps", "num_gradient_steps", "half_update_steps"):
        del tiny[key]
    config = OmegaConf.merge(
        config, tiny, {"num_timesteps": 16 * 8 * 3, "save_dir": str(tmp_path)}
    )
    monkeypatch.setattr(mod, "WANDB_RUNS", [])
    monkeypatch.setattr(mod, "RESULTS_LOG", None)  # main() sets both
    mod.main(config)
    (run_dir,) = [os.path.join(tmp_path, d) for d in os.listdir(tmp_path)]
    assert os.path.basename(run_dir).startswith(f"{name}_")
    saved = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    assert saved.num_update_steps == 3 and saved.half_update_steps == 1
    assert saved.run_dir == run_dir and saved.save_checkpoints
    records = read_results(run_dir)
    assert sorted((r["seed_index"], r["update_step"]) for r in records) == [
        (i, u) for i in range(2) for u in range(3)
    ]
    _check_game_metrics(name, records)
    for r in records:  # evaluated every compare_interval (2) updates and after the last
        evaluated = any("_vs_" in k for k in r)
        assert evaluated == (r["update_step"] in (0, 2))
        if evaluated:
            _check_eval_metrics(r, "random")
    for stage in ("half", "full"):
        assert sorted(os.listdir(os.path.join(run_dir, stage))) == [
            "avg_policy_0.msgpack", "avg_policy_1.msgpack", "br_0.msgpack", "br_1.msgpack"
        ]


@pytest.mark.parametrize("name", ["five_card_draw_ppo_nfsp", "five_card_draw_pqn_nfsp"])
def test_five_card_draw_networks_see_chips_divided_by_the_stack(name):
    """chip_scaled divides exactly the two chip counts of the observation (the
    acting player's chips in the pot and the largest) by init_chips."""
    import flax.linen as nn

    class Identity(nn.Module):
        @nn.compact
        def __call__(self, x):
            return x

    mod = _script(name)
    env = make("five_card_draw", num_agents=3, init_chips=80)
    state, obs = env.reset(KEY)
    # put 7 chips in for the player to act and 30 for another
    current = int(state.current_player_idx)
    other = (current + 1) % 3
    state = state.replace(chips_in=state.chips_in.at[current].set(7).at[other].set(30))
    obs = env.obs_from_state(state)
    net = mod.chip_scaled(Identity(), env)
    out = net.apply(net.init(KEY, obs), obs)
    np.testing.assert_allclose(np.asarray(obs[52:54]), [7.0, 30.0])
    np.testing.assert_allclose(np.asarray(out[52:54]), [7.0 / 80, 30.0 / 80])
    keep = np.r_[0:52, 54 : env.obs_dim]
    np.testing.assert_array_equal(np.asarray(out)[keep], np.asarray(obs)[keep])
