"""Short real runs of the Hold'em training scripts (CPU):

- PPO-NFSP and PQN-NFSP for Limit and No-Limit (bluffjax/examples/HUL,
  HUNL): results.jsonl holds every update of every seed with the game/ and
  game_br/ metrics (split by the policy the credited player follows), the
  evaluation results only when evaluated, the metric carry persists across
  rollouts, the half checkpoint is the networks after exactly
  half_update_steps updates, and half/ and full/ load strictly into the
  scripts' networks (also as the actor-critic evaluation opponent); the
  No-Limit networks see the chip counts divided by the starting stack;
- their evaluation (holdem_rebel.agent.play_hands): exactly compare_episodes
  complete hands per policy, every deal from both seats, every
  compare_interval updates and after the last, logged from exactly those
  hands; random against random gives symmetric results; it doesn't change
  training; ReBeL as the opponent;
- ReBeL (bluffjax/examples/holdem_rebel/train.py) with a tiny budget: run
  directory, results.jsonl with the evaluation (avg_return_vs_<opp>,
  avg_return_se_vs_<opp>, final/...) and the self-play game metrics at every
  evaluation, half/value_net_0 and full/value_net_0 value networks with
  their metadata.
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
        compare=True, compare_mode="random", compare_episodes=5, compare_interval=3,
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
        # chip_scaled: the two chip counts (features 52 and 53) divided by the
        # 100-chip stack, the MLP's parameters under "net"
        from bluffjax.networks.scaled_input import ScaledInput

        avg, br = (ScaledInput(net=net, start=52, stop=54, factor=1 / 100) for net in (avg, br))
    return avg, br


EVAL_STATS = (
    "avg_return", "win_rate", "loss_rate", "draw_rate", "win_rate_decided",
    "avg_episode_length",
)


def eval_names(opponent):
    """The evaluation metrics of both policies against `opponent`: the
    average policy's {stat}_vs_{opponent} and the BR's {stat}_br_vs_{opponent}."""
    return {f"{stat}_{p}vs_{opponent}" for stat in EVAL_STATS for p in ("", "br_")} | {
        "eval_unfinished_hands"
    }


def spy_play_hands(mod, monkeypatch, learner=None):
    """Records the hands of every evaluation the script plays (one
    HandsResult per seed, policy and evaluation, in no particular order).
    With `learner`, that player plays instead of the script's policy."""
    from bluffjax.examples.holdem_rebel import agent

    records = []

    def record(*fields):
        records.append(agent.HandsResult(*(np.asarray(x) for x in fields)))

    def spy(env, game, player_a, *args, **kwargs):
        out = agent.play_hands(env, game, learner or player_a, *args, **kwargs)
        jax.debug.callback(record, *out)
        return out

    monkeypatch.setattr(mod, "play_hands", spy)
    return records


def check_eval_hands(records, num_hands):
    """Every evaluation played exactly num_hands complete hands, hands 2k and
    2k + 1 being one deal with the learner in each seat."""
    for h in records:
        assert h.rewards.shape == h.hand_length.shape == (num_hands,)
        assert h.finished.all() and (h.hand_length >= 1).all()
        np.testing.assert_array_equal(h.deal, np.arange(num_hands) // 2)
        np.testing.assert_array_equal(h.seat_a, (h.seat_a[0] + np.arange(num_hands)) % 2)


def eval_summary(h):
    """The logged metrics of one evaluation, computed from its hands."""
    r = h.rewards
    win, loss = (r > 0).mean(), (r < 0).mean()
    decided = win / (win + loss) if win + loss > 0 else np.nan
    return (r.mean(), win, loss, (r == 0).mean(), decided, h.hand_length.mean())


def check_eval_rows(rows, records, opponent, num_hands):
    """The evaluation metrics of the evaluated rows are those of exactly the
    recorded hands (two per row, one per policy)."""
    from_rows = []
    for r in rows:
        assert int(r["eval_unfinished_hands"]) == 0
        for p in ("", "br_"):  # the average policy, the BR
            stats = [r[f"{stat}_{p}vs_{opponent}"] for stat in EVAL_STATS]
            ret, win, loss, draw, decided, length = stats
            for rate in (win, loss, draw):  # a whole number of hands (float32 rates)
                assert rate * num_hands == pytest.approx(round(rate * num_hands), abs=0.01)
            assert win + loss + draw == pytest.approx(1.0)
            if win + loss > 0:
                assert decided == pytest.approx(win / (win + loss))
            else:  # every hand a split pot: no decided hand
                assert math.isnan(decided)
            from_rows.append((ret, win, loss, draw, decided, length))
    from_hands = [eval_summary(h) for h in records]
    assert len(from_hands) == len(from_rows)
    key = lambda t: tuple(-1.0 if math.isnan(x) else round(float(x), 3) for x in t)  # noqa: E731
    for got, want in zip(sorted(from_rows, key=key), sorted(from_hands, key=key)):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


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
    records = spy_play_hands(mod, monkeypatch)
    run_dir, full = run(mod, config, monkeypatch)
    records = list(records)  # this run's evaluations
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
        # every compare_interval updates (0, 3) and after the last one (4)
        evaluated = r["update_step"] in (0, 3, 4)
        assert "evaluated" not in r  # the flag isn't logged
        names = {k for k in r if "_vs_" in k or k.startswith("eval_")}
        assert names == (eval_names("random") if evaluated else set())
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
            assert hands == 2 * r["episodes_completed"]
            assert r["game/n_hands_sb"] + r["game_br/n_hands_sb"] == r["episodes_completed"]
    assert sum(r["game/n_aggressive"] + r["game_br/n_aggressive"] for r in rows) > 0
    assert sum(r["game_br/n_aggressive"] for r in rows) > 0

    # evaluation: 2 policies x 3 evaluations x 2 seeds, compare_episodes hands
    # each, and the logged metrics are those of exactly these hands
    assert len(records) == 2 * 3 * 2
    check_eval_hands(records, config["compare_episodes"])
    check_eval_rows([r for r in rows if "win_rate_vs_random" in r], records, "random",
                    config["compare_episodes"])

    # the evaluation doesn't change training
    _, off = run(mod, tiny_config(name, tmp_path / "off", compare=False), monkeypatch)
    assert _leaves_equal(off.avg_train_state.params, full.avg_train_state.params)
    assert _leaves_equal(off.br_train_state.params, full.br_train_state.params)

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
            assert {k for k in row if "_vs_" in k or k.startswith("eval_")} == eval_names(
                "baseline"
            )
            assert all(np.isfinite(row[k]) for k in eval_names("baseline"))
            assert row["eval_unfinished_hands"] == 0


@pytest.mark.parametrize("name", ["hunl_ppo_nfsp", "hunl_pqn_nfsp"])
def test_hunl_networks_see_chips_divided_by_the_stack(name):
    """chip_scaled divides exactly the two chip counts of the observation
    (features 52 and 53) by the env's starting stack; the card indicators
    pass through."""
    import flax.linen as nn

    class Identity(nn.Module):
        @nn.compact
        def __call__(self, x):
            return x

    env = make("texas_nolimit_holdem", num_agents=2, init_chips=80)
    _, obs = env.reset(KEY)
    obs = obs.at[52:54].set(jnp.array([7.0, 30.0]))
    net = _module(name).chip_scaled(Identity(), env)
    out = np.asarray(net.apply(net.init(KEY, obs), obs))
    assert out.shape == (env.obs_dim,) == (54,)
    np.testing.assert_allclose(out[52:54], [7.0 / 80, 30.0 / 80])
    np.testing.assert_array_equal(out[:52], np.asarray(obs[:52]))


@pytest.mark.parametrize("name", ["hul_ppo_nfsp", "hunl_pqn_nfsp"])
def test_nfsp_game_metrics_follow_the_policy(name, monkeypatch, tmp_path):
    """With every player on the average policy (eta 0) game_br/ has no
    events, with every player on the BR (eta 1) game/ has none."""
    from bluffjax.examples.run_utils import read_results

    mod = _module(name)
    # PQN: a fixed epsilon, so that an untrained greedy BR still folds in
    # every update (the configured schedule is greedy after the first)
    explore = {"start_e": 0.3, "end_e": 0.3} if "pqn" in name else {}
    for eta, empty, full in ((0.0, "game_br", "game"), (1.0, "game", "game_br")):
        config = tiny_config(name, tmp_path / str(eta), anticipatory_eta=eta, compare=False,
                             num_update_steps=2, **explore)
        run_dir, _ = run(mod, config, monkeypatch)
        for r in read_results(run_dir):
            ns = [k for k in r if k.startswith(f"{empty}/n_")]
            assert ns and all(r[k] == 0 for k in ns)
            rates = [k for k in r if k.startswith(f"{empty}/") and not k.startswith(f"{empty}/n_")]
            assert all(math.isnan(r[k]) for k in rates)
            assert r[f"{full}/n_aggressive"] > 0 and r[f"{full}/n_folds"] > 0


# =============================================================================
# In-training evaluation
# =============================================================================


@pytest.mark.parametrize("game_name", ["hul", "hunl"])
def test_play_hands_exact_count_and_mirrored_deals(game_name):
    """play_hands plays exactly num_hands complete hands; hands 2k and 2k + 1
    are one deal from both seats: two copies of a deterministic player get
    exactly opposite results in them."""
    from bluffjax.examples.holdem_rebel.agent import heuristic_players, play_hands
    from bluffjax.examples.holdem_rebel.game import make_game

    game = make_game(game_name)
    env = make(game.env_id, num_agents=2)
    players = heuristic_players(game)
    for n in (1, 6, 7):
        for a, b in (("always_call", "always_call"), ("random", "always_raise")):
            hands = jax.jit(
                lambda k: play_hands(env, game, players[a], None, players[b], None, n, k)
            )(jax.random.PRNGKey(n))
            hands = jax.tree_util.tree_map(np.asarray, hands)
            check_eval_hands([hands], n)
            if a == b:  # the same deal and the same actions from the other seat
                pairs = hands.rewards[: 2 * (n // 2)].reshape(-1, 2)
                np.testing.assert_array_equal(pairs[:, 0], -pairs[:, 1])
                lengths = hands.hand_length[: 2 * (n // 2)].reshape(-1, 2)
                np.testing.assert_array_equal(lengths[:, 0], lengths[:, 1])
    # the seat of a pair's first hand (and of an odd count's last deal) is random
    first_seat = jax.jit(
        lambda k: play_hands(env, game, players["random"], None, players["random"], None,
                             1, k).seat_a[0]
    )
    assert {int(first_seat(jax.random.PRNGKey(i))) for i in range(16)} == {0, 1}


@pytest.mark.parametrize("name", ["hul_ppo_nfsp", "hunl_pqn_nfsp"])
def test_eval_random_against_random_is_symmetric(name, monkeypatch, tmp_path):
    """With random play in place of the learner, the evaluation (every
    update, as with the default compare_interval 1) gives the symmetric
    values of a zero-sum game: win rate = loss rate and return 0, up to
    sampling error."""
    from bluffjax.examples.holdem_rebel.agent import heuristic_players
    from bluffjax.examples.holdem_rebel.game import make_game
    from bluffjax.examples.run_utils import read_results

    mod = _module(name)
    n = 2000
    config = tiny_config(name, tmp_path, num_update_steps=2, compare_episodes=n,
                         compare_interval=1)
    random_player = heuristic_players(make_game(name.split("_")[0]))["random"]
    records = spy_play_hands(mod, monkeypatch, learner=random_player)
    run_dir, _ = run(mod, config, monkeypatch)
    rows = read_results(run_dir)
    assert len(rows) == 4 and all("win_rate_vs_random" in r for r in rows)
    assert len(records) == 2 * 2 * 2  # policies x updates x seeds
    check_eval_hands(records, n)
    check_eval_rows(rows, records, "random", n)
    returns = np.concatenate([h.rewards for h in records])
    se = returns.std() / np.sqrt(returns.size)
    assert abs(returns.mean()) < 4 * se
    win, loss = (returns > 0).mean(), (returns < 0).mean()
    assert abs(win - loss) < 4 * np.sqrt((win + loss) / returns.size)
    for r in rows:
        for p in ("", "br_"):  # the average policy, the BR
            w, l = r[f"win_rate_{p}vs_random"], r[f"loss_rate_{p}vs_random"]
            assert abs(w - l) < 5 * np.sqrt((w + l) / n)
            assert abs(r[f"avg_return_{p}vs_random"]) < 5 * returns.std() / np.sqrt(n)
            assert r[f"win_rate_decided_{p}vs_random"] == pytest.approx(0.5, abs=0.06)


def tiny_rebel_checkpoint(game_name, path):
    """A ReBeL value network file (random weights, 2 CFR iterations)."""
    from bluffjax.examples.holdem_rebel.agent import save_checkpoint
    from bluffjax.examples.holdem_rebel.cards import NUM_HANDS
    from bluffjax.examples.holdem_rebel.game import PUBLIC_DIM, make_game
    from bluffjax.examples.holdem_rebel.solver import ValueNetwork

    game = make_game(game_name)
    params = ValueNetwork(16, 1).init(KEY, jnp.zeros((PUBLIC_DIM,)), jnp.zeros((2, NUM_HANDS)))
    return save_checkpoint(
        str(path), params,
        {"game": game.name, "max_raises": game.max_raises, "value_hidden_dim": 16,
         "value_num_layers": 1, "cfr_iters": 2, "samples": 0, "seed": 0},
    )


@pytest.mark.parametrize("name", ["hul_ppo_nfsp", "hunl_pqn_nfsp"])
def test_eval_against_rebel(name, monkeypatch, tmp_path):
    """compare_mode checkpoint with a ReBeL value network: compare_episodes
    complete hands per policy against the searching opponent."""
    from bluffjax.examples.run_utils import read_results

    mod = _module(name)
    rebel_path = tiny_rebel_checkpoint(
        name.split("_")[0], tmp_path / "rebel" / "full" / "value_net_0.msgpack"
    )
    config = tiny_config(
        name, tmp_path, num_update_steps=1, compare_mode="checkpoint",
        compare_network_type="rebel", compare_with=rebel_path, compare_episodes=3,
        compare_solve_chunk=2,
    )
    records = spy_play_hands(mod, monkeypatch)
    run_dir, _ = run(mod, config, monkeypatch, seeds=1)
    (row,) = read_results(run_dir)
    assert {k for k in row if "_vs_" in k or k.startswith("eval_")} == eval_names("baseline")
    assert len(records) == 2
    check_eval_hands(records, 3)
    check_eval_rows([row], records, "baseline", 3)


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
def test_rebel_run_log_metrics_and_checkpoints(game_name, monkeypatch, tmp_path):
    import re

    from bluffjax.examples.holdem_rebel import train as rebel_train
    from bluffjax.examples.holdem_rebel.agent import heuristic_switch, load_rebel
    from bluffjax.examples.holdem_rebel.cards import NUM_HANDS
    from bluffjax.examples.holdem_rebel.game import PUBLIC_DIM, make_game
    from bluffjax.examples.holdem_rebel.solver import ValueNetwork
    from bluffjax.examples.holdem_rebel.train import run_training
    from bluffjax.examples.run_utils import checkpoint_path, load_params, read_results

    cfg = rebel_config(game_name, tmp_path)
    assert "max_grad_norm" in cfg and "grad_clip" not in cfg
    game = make_game(game_name, cfg.get("max_raises"))
    wandb_inits, real_init = [], rebel_train.wandb.init
    monkeypatch.setattr(
        rebel_train.wandb, "init", lambda **kw: wandb_inits.append(kw) or real_init(**kw)
    )
    run_dir = run_training(game, cfg)
    rows = read_results(run_dir)
    # wandb names as in every other script, seed index 0
    (kw,) = wandb_inits
    assert re.fullmatch(rf"{game_name}_\d{{4}}(-\d\d){{2}}_\d\d(-\d\d){{2}}", kw["group"])
    assert kw["job_type"] == f"{game_name}_rebel_{game_name}"
    assert kw["name"] == f"{cfg['seed']}_0"
    opponents = heuristic_switch(game)[1]
    assert opponents == ["random", "always_call", "always_raise"] + (
        [] if game.is_limit else ["always_allin"]
    )
    eval_keys = {f"avg_return{se}_vs_{o}" for se in ("", "_se") for o in opponents}
    assert not any(k.startswith("eval/") for r in rows for k in r)
    assert all("samples" in r and r["seed_index"] == 0 for r in rows)
    samples = [r["samples"] for r in rows if "value_loss" in r]
    assert samples == sorted(samples) and samples[-1] >= cfg["num_samples"]
    gm = make_metrics(game.env_id, make(game.env_id, num_agents=2))
    evals = [r for r in rows if any(k.startswith("game/") for k in r)]
    # every eval_every samples and at the end
    assert len(evals) >= 2 and evals[-1]["samples"] == samples[-1]
    for r in evals:
        assert {k for k in r if "_vs_" in k} == eval_keys and r["game_eval_seconds"] > 0
        assert all(math.isfinite(r[k]) for k in eval_keys)
        for rate in gm.rate_names:
            v = r[f"game/{rate}"]
            assert math.isnan(v) or 0.0 <= v <= 1.0, (rate, v)
        if game_name == "hul":  # 2 deals from both seats, every hand counted
            hands = 2 * cfg["metric_deals"] - r["game_eval_unfinished"]
            assert r["game/n_hands"] == 2 * hands  # credited to both players
    # the final evaluation: one record after the last training record
    assert set(rows[-1]) == {"seed_index", "wall_time", "samples"} | {
        f"final/{k}" for k in eval_keys
    }
    assert rows[-1]["samples"] == samples[-1]
    assert all(not any(k.startswith("final/") for k in r) for r in rows[:-1])

    # half/ (when the samples first reach num_samples // 2) and full/, as
    # value_net_{i} with seed index i = 0 (one seed per process)
    template = ValueNetwork(cfg["value_hidden_dim"], cfg["value_num_layers"]).init(
        KEY, jnp.zeros((PUBLIC_DIM,)), jnp.zeros((2, NUM_HANDS))
    )
    metas = {}
    for stage in ("half", "full"):
        assert sorted(os.listdir(os.path.join(run_dir, stage))) == [
            "value_net_0.json", "value_net_0.msgpack"
        ]
        path = checkpoint_path(run_dir, stage, "value_net", 0)
        assert path == os.path.join(run_dir, stage, "value_net_0.msgpack")
        load_params(path, template)
        _, _, metas[stage] = load_rebel(path, solve_chunk=1)
        assert metas[stage]["game"] == game_name
        assert metas[stage]["seed"] == cfg["seed"]
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

    mod = _module(name)
    config = tiny_config(name, tmp_path, num_update_steps=2, compare=False)
    run_dir, full = run(mod, config, monkeypatch)
    mod.save_checkpoints(
        run_dir, 2, full.half,
        mod.Checkpoint(full.avg_train_state.params, full.br_train_state.params),
    )
    rebel_path = tiny_rebel_checkpoint(
        name.split("_")[0], tmp_path / "rebel" / "full" / "value_net_0.msgpack"
    )
    for stage in ("half", "full"):
        eval_nfsp.main(run_dir, rebel_path, 2, 2, stage)
        with open(os.path.join(run_dir, f"eval_vs_rebel_{stage}.json")) as f:
            summary = json.load(f)
        assert summary["stage"] == stage and len(summary["per_seed"]) == 2
        assert all(r["hands"] == 4 and r["unfinished"] == 0 for r in summary["per_seed"])
    with pytest.raises(ValueError):
        eval_nfsp.main(run_dir, rebel_path, 2, 2, "quarter")
