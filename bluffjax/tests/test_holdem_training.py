"""Short real runs of the Hold'em training scripts (CPU):

- PPO-NFSP and PQN-NFSP for Limit and No-Limit (bluffjax/examples/HUL,
  HUNL): results.jsonl holds every update of every seed with the game/ and
  game_br/ metrics (split by the policy the credited player follows), the
  evaluation results only when evaluated, the metric carry persists across
  rollouts, the half checkpoint is the networks after exactly
  half_update_steps updates, and half/ and full/ load strictly into the
  scripts' networks (also as the actor-critic evaluation opponent);
- ReBeL (bluffjax/examples/holdem_rebel/train.py) with a tiny budget: run
  directory, results.jsonl with the self-play game metrics at every
  evaluation, half/ and full/ value networks with their metadata.
"""

import functools
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from bluffjax import make
from bluffjax.metrics import make_metrics

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
KEY = jax.random.PRNGKey(0)
NAMES = ["hul_ppo_nfsp", "hul_pqn_nfsp", "hunl_ppo_nfsp", "hunl_pqn_nfsp"]
ENV_IDS = {"hul": "texas_limit_holdem", "hunl": "texas_nolimit_holdem"}


@pytest.fixture(scope="module", autouse=True)
def _baselines():
    """The training code needs the `baselines` extra (distrax, optax, hydra)."""
    for module in ("distrax", "optax", "hydra", "wandb"):
        pytest.importorskip(module)


def _module(name):
    import importlib

    folder = "HUL" if name.startswith("hul") else "HUNL"
    return importlib.import_module(f"bluffjax.examples.{folder}.{name}")


def tiny_config(name, save_dir, **extra):
    """The script's config file with a tiny budget (5 updates, half after 2)."""
    folder = "HUL" if name.startswith("hul") else "HUNL"
    algo = name.split("_")[1]
    config = OmegaConf.to_container(
        OmegaConf.load(os.path.join(EXAMPLES, folder, f"config_{algo}_nfsp.yaml"))
    )
    del config["hydra"]
    config.update(
        num_envs=8, num_steps_per_env_per_update=6, num_epochs=1, num_minibatches=2,
        fc_dim_size=16, sl_reservoir_capacity=256, sl_batch_size=16,
        sl_num_steps_per_update=2, anticipatory_eta=0.4, num_seeds=2,
        compare=True, compare_mode="random", compare_steps=4, compare_interval=2,
        save_dir=str(save_dir), wandb=False, num_update_steps=5,
    )
    config.update(extra)
    config["half_update_steps"] = config["num_update_steps"] // 2
    config["num_gradient_steps"] = (
        config["num_update_steps"] * config["num_epochs"] * config["num_minibatches"]
    )
    return config


def run(mod, config, monkeypatch, num_updates=None, seeds=2):
    """Runs the vmapped seeds as main() does; returns (run_dir, final state)."""
    from bluffjax.examples.run_utils import ResultsLog, make_run_dir

    train = mod.make_train(config)
    run_dir = make_run_dir(config)
    monkeypatch.setattr(mod, "RESULTS_LOG", ResultsLog(run_dir, []))
    keys = jax.random.split(KEY, seeds)
    fn = train if num_updates is None else functools.partial(train, num_updates=num_updates)
    out = jax.block_until_ready(jax.jit(jax.vmap(fn))(keys, jnp.arange(seeds)))
    jax.effects_barrier()
    return run_dir, out


def networks(name, env, hidden):
    """The (average policy, BR) network classes the script documents."""
    from bluffjax.networks.mlp import ActorCriticDiscreteMLP, ActorDiscreteMLP, QNetworkDiscreteMLP

    A = env.num_actions
    if "ppo" in name:
        avg, br = ActorCriticDiscreteMLP(A, hidden), ActorCriticDiscreteMLP(A, hidden)
    else:
        avg, br = ActorDiscreteMLP(A, hidden), QNetworkDiscreteMLP(A, hidden)
    if name.startswith("hunl"):
        wrap = _module(name).ChipScaledInput
        avg, br = wrap(avg, 52, 100.0), wrap(br, 52, 100.0)
    return avg, br


def _leaves_equal(a, b):
    return all(
        np.array_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    )


def _max_diff(a, b):
    return max(
        float(np.abs(np.asarray(x) - np.asarray(y)).max())
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    )


@pytest.mark.parametrize("name", NAMES)
def test_nfsp_run_log_metrics_and_checkpoints(name, monkeypatch, tmp_path):
    from bluffjax.examples.run_utils import checkpoint_path, load_params, read_results

    mod = _module(name)
    config = tiny_config(name, tmp_path)
    U, num_half = config["num_update_steps"], config["half_update_steps"]
    run_dir, full = run(mod, config, monkeypatch)
    env = make(ENV_IDS[name.split("_")[0]], **config["env_kwargs"])
    gm = make_metrics(ENV_IDS[name.split("_")[0]], env)

    # results.jsonl: every update of every seed, with the game metrics
    rows = read_results(run_dir)
    assert len(rows) == 2 * U
    assert sorted((r["seed_index"], r["update_step"]) for r in rows) == [
        (i, u) for i in range(2) for u in range(U)
    ]
    T, N = config["num_steps_per_env_per_update"], config["num_envs"]
    for r in rows:
        assert r["env_steps"] == (r["update_step"] + 1) * T * N
        assert r["wall_time"] >= 0
        evaluated = r["update_step"] % config["compare_interval"] == 0
        assert bool(r["evaluated"]) == evaluated
        assert ("avg_chips_avg_vs_random" in r) == evaluated
        assert "avg_chips_avg_vs_baseline" not in r
        for prefix in ("game", "game_br"):
            for rate in gm.rate_names:
                v = r[f"{prefix}/{rate}"]
                assert math.isnan(v) or 0.0 <= v <= 1.0, (prefix, rate, v)
            for k, v in r.items():
                if k.startswith(f"{prefix}/n_"):
                    assert v >= 0 and v == int(v), (k, v)
    # every hand that ended is in game/ or game_br/ (credited to every player
    # dealt in, by the policy each follows)
    if name.startswith("hul"):
        for r in rows:
            hands = r["game/n_hands"] + r["game_br/n_hands"]
            assert hands == 2 * r["hands_completed"]
            assert r["game/n_hands_sb"] + r["game_br/n_hands_sb"] == r["hands_completed"]
    assert sum(r["game/n_aggressive"] + r["game_br/n_aggressive"] for r in rows) > 0
    assert sum(r["game_br/n_aggressive"] for r in rows) > 0

    # the carry is the hand in progress, kept across rollouts
    carry = full.metric_carry
    state = full.state
    aggressive = np.asarray(carry.bluff.aggressive).sum(-2)  # (seeds, envs, streets)
    if name.startswith("hul"):
        np.testing.assert_array_equal(aggressive, np.asarray(state.raise_nums))
    raised = np.asarray(state.chips_in).max(-1) > 2  # a bet or raise this hand
    np.testing.assert_array_equal(aggressive.sum(-1) > 0, raised)
    assert raised.any() and (np.asarray(state.stage) > 0).any()

    # half checkpoint: the networks after exactly num_half updates
    for n in (num_half - 1, num_half, num_half + 1):
        _, part = run(mod, config, monkeypatch, num_updates=n)
        for net in ("avg_params", "br_params"):
            got = getattr(full.half, net)
            want = getattr(part, "avg_train_state" if net == "avg_params" else "br_train_state").params
            if n == num_half:
                assert _leaves_equal(got, want), net
            else:
                assert _max_diff(got, want) > 1e-6, (net, n)
    assert _max_diff(full.half.avg_params, full.avg_train_state.params) > 1e-6

    # half/ and full/ files, loaded strictly into the documented networks
    mod.save_checkpoints(
        run_dir, 2, full.half,
        mod.Checkpoint(full.avg_train_state.params, full.br_train_state.params),
    )
    found = {
        os.path.relpath(os.path.join(d, f), run_dir)
        for d, _, files in os.walk(run_dir) for f in files
    }
    assert found == {"config.yaml", "results.jsonl"} | {
        f"{stage}/{net}_{i}.msgpack"
        for stage in ("half", "full") for net in ("avg_policy", "br") for i in range(2)
    }
    saved = OmegaConf.to_container(OmegaConf.load(os.path.join(run_dir, "config.yaml")))
    assert saved["half_update_steps"] == num_half and saved["run_dir"] == run_dir
    avg_net, br_net = networks(name, env, config["fc_dim_size"])
    obs = jnp.zeros((env.obs_dim,))
    for stage, ckpt in (("half", full.half), ("full", mod.Checkpoint(
        full.avg_train_state.params, full.br_train_state.params
    ))):
        for i in range(2):
            for net, file, params in ((avg_net, "avg_policy", ckpt.avg_params),
                                      (br_net, "br", ckpt.br_params)):
                loaded = load_params(
                    checkpoint_path(run_dir, stage, file, i), net.init(KEY, obs)
                )
                assert _leaves_equal(loaded, jax.tree_util.tree_map(lambda x: x[i], params))
    wider = networks(name, env, 2 * config["fc_dim_size"])
    for file, net in (("avg_policy", wider[0]), ("br", wider[1])):
        with pytest.raises(ValueError):  # strict: another network doesn't load
            load_params(checkpoint_path(run_dir, "full", file, 0), net.init(KEY, obs))

    # the PPO-NFSP networks as the actor-critic evaluation opponent
    if "ppo" in name:
        for other in (name, name.replace("ppo", "pqn")):
            eval_config = tiny_config(
                other, tmp_path / "eval", num_update_steps=1, compare_mode="checkpoint",
                compare_network_type="actor_critic",
                compare_with=checkpoint_path(run_dir, "full", "avg_policy", 1),
            )
            eval_dir, _ = run(_module(other), eval_config, monkeypatch, seeds=1)
            (row,) = read_results(eval_dir)
            assert np.isfinite(row["avg_chips_avg_vs_baseline"])


@pytest.mark.parametrize("name", ["hul_ppo_nfsp", "hunl_pqn_nfsp"])
def test_nfsp_game_metrics_follow_the_policy(name, monkeypatch, tmp_path):
    """With every player on the average policy (eta 0) game_br/ has no
    events, with every player on the BR (eta 1) game/ has none."""
    from bluffjax.examples.run_utils import read_results

    mod = _module(name)
    for eta, empty, full in ((0.0, "game_br", "game"), (1.0, "game", "game_br")):
        config = tiny_config(name, tmp_path / str(eta), anticipatory_eta=eta, compare=False,
                             num_update_steps=2)
        run_dir, _ = run(mod, config, monkeypatch)
        for r in read_results(run_dir):
            ns = [k for k in r if k.startswith(f"{empty}/n_")]
            assert ns and all(r[k] == 0 for k in ns)
            rates = [k for k in r if k.startswith(f"{empty}/") and not k.startswith(f"{empty}/n_")]
            assert all(math.isnan(r[k]) for k in rates)
            assert r[f"{full}/n_aggressive"] > 0 and r[f"{full}/n_folds"] > 0


# =============================================================================
# ReBeL
# =============================================================================


def rebel_config(game_name, save_dir):
    folder = "HUL" if game_name == "hul" else "HUNL"
    config = OmegaConf.to_container(
        OmegaConf.load(os.path.join(EXAMPLES, folder, "config_rebel.yaml"))
    )
    del config["hydra"]
    config.update(
        num_samples=120, num_games=2, cfr_iters=2, chance_flops=2, solve_chunk=2,
        steps_per_chunk=1, value_hidden_dim=16, value_num_layers=1, batch_size=8,
        replay_capacity=256, eval_every=50, eval_deals=1, final_eval_deals=1,
        eval_batch_deals=2, metric_deals=2, log_every_chunks=1000,
        save_dir=str(save_dir), wandb=False,
    )
    return config


@pytest.mark.parametrize("game_name", ["hul", "hunl"])
def test_rebel_run_log_metrics_and_checkpoints(game_name, tmp_path):
    from bluffjax.examples.holdem_rebel.agent import load_rebel
    from bluffjax.examples.holdem_rebel.cards import NUM_HANDS
    from bluffjax.examples.holdem_rebel.game import PUBLIC_DIM, make_game
    from bluffjax.examples.holdem_rebel.solver import ValueNetwork
    from bluffjax.examples.holdem_rebel.train import run_training
    from bluffjax.examples.run_utils import checkpoint_path, load_params, read_results

    cfg = rebel_config(game_name, tmp_path)
    game = make_game(game_name, cfg.get("max_raises"))
    run_dir = run_training(game, cfg)
    rows = read_results(run_dir)
    assert all("samples" in r and r["seed_index"] == 0 for r in rows)
    samples = [r["samples"] for r in rows if "value_loss" in r]
    assert samples == sorted(samples) and samples[-1] >= cfg["num_samples"]
    gm = make_metrics(game.env_id, make(game.env_id, num_agents=2))
    evals = [r for r in rows if any(k.startswith("game/") for k in r)]
    # every eval_every samples and at the end
    assert len(evals) >= 2 and evals[-1]["samples"] == samples[-1]
    for r in evals:
        assert "eval/random_mean" in r and r["game_eval_seconds"] > 0
        for rate in gm.rate_names:
            v = r[f"game/{rate}"]
            assert math.isnan(v) or 0.0 <= v <= 1.0, (rate, v)
        if game_name == "hul":  # 2 deals from both seats, every hand counted
            hands = 2 * cfg["metric_deals"] - r["game_eval_unfinished"]
            assert r["game/n_hands"] == 2 * hands  # credited to both players
    assert any(k.startswith("final/") for k in rows[-1])

    # half/ (when the samples first reach num_samples // 2) and full/
    template = ValueNetwork(cfg["value_hidden_dim"], cfg["value_num_layers"]).init(
        KEY, jnp.zeros((PUBLIC_DIM,)), jnp.zeros((2, NUM_HANDS))
    )
    metas = {}
    for stage in ("half", "full"):
        path = checkpoint_path(run_dir, stage, "value_net")
        load_params(path, template)
        _, _, metas[stage] = load_rebel(path, solve_chunk=1)
        assert metas[stage]["game"] == game_name
    first_half = min(s for s in samples if s >= cfg["num_samples"] // 2)
    assert metas["half"]["samples"] == first_half
    assert metas["full"]["samples"] == samples[-1]
    assert sorted(os.listdir(run_dir)) == ["config.yaml", "full", "half", "results.jsonl"]


@pytest.mark.parametrize("name", ["hul_pqn_nfsp", "hunl_ppo_nfsp"])
def test_eval_nfsp_reads_the_run_directory(name, monkeypatch, tmp_path):
    """holdem_rebel/eval_nfsp.py plays the average policies of half/ or
    full/ against a ReBeL value network file with its metadata."""
    import json

    from bluffjax.examples.holdem_rebel import eval_nfsp
    from bluffjax.examples.holdem_rebel.agent import save_checkpoint
    from bluffjax.examples.holdem_rebel.cards import NUM_HANDS
    from bluffjax.examples.holdem_rebel.game import PUBLIC_DIM, make_game
    from bluffjax.examples.holdem_rebel.solver import ValueNetwork

    mod = _module(name)
    config = tiny_config(name, tmp_path, num_update_steps=2, compare=False)
    run_dir, full = run(mod, config, monkeypatch)
    mod.save_checkpoints(
        run_dir, 2, full.half,
        mod.Checkpoint(full.avg_train_state.params, full.br_train_state.params),
    )
    game = make_game(name.split("_")[0])
    params = ValueNetwork(16, 1).init(KEY, jnp.zeros((PUBLIC_DIM,)), jnp.zeros((2, NUM_HANDS)))
    rebel_path = save_checkpoint(
        str(tmp_path / "rebel" / "full" / "value_net.msgpack"), params,
        {"game": game.name, "max_raises": game.max_raises, "value_hidden_dim": 16,
         "value_num_layers": 1, "cfr_iters": 2, "samples": 0, "seed": 0},
    )
    for stage in ("half", "full"):
        eval_nfsp.main(run_dir, rebel_path, 2, 2, stage)
        with open(os.path.join(run_dir, f"eval_vs_rebel_{stage}.json")) as f:
            summary = json.load(f)
        assert summary["stage"] == stage and len(summary["per_seed"]) == 2
        assert all(r["hands"] == 4 and r["unfinished"] == 0 for r in summary["per_seed"])
    with pytest.raises(ValueError):
        eval_nfsp.main(run_dir, rebel_path, 2, 2, "quarter")
