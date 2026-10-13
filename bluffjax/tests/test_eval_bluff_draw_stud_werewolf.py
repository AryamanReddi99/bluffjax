"""In-training evaluation of the NFSP scripts of Bluff, 5-Card Draw, 7-Card
Stud and Werewolf (PPO-NFSP and PQN-NFSP).

Every script evaluates its average policy and its best response in
compare_episodes complete games each against uniformly random legal play,
with the learner's seat rotating over the games, every compare_interval
updates and after the last one. Checked here:
- the evaluator plays exactly num_games games, each until it is over: with an
  env that pays every player 1 on the step that ends the game (done) and 0
  otherwise, the learner's return is 1 in every game;
- the learner's seat rotates (game g: seat (offset + g) % num_agents) from a
  random offset, so the seats are used equally up to one game;
- a uniformly random learner against random opponents gets the symmetric
  values: as many wins as losses with two players, and a mean return of 0 in
  the zero-sum games (poker);
- the PPO-NFSP and PQN-NFSP evaluators of a game are the same;
- a tiny training run evaluates in every update with compare_interval 1,
  plays compare_episodes complete games per policy (counted as above) and
  logs the same metric names in the PPO-NFSP and PQN-NFSP scripts.
"""

import copy
import importlib
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make

KEY = jax.random.PRNGKey(0)
GAMES = ("bluff", "five_card_draw", "seven_card_stud", "werewolf")
SCRIPTS = [(game, algo) for game in GAMES for algo in ("ppo", "pqn")]
ENV_KWARGS = {
    "bluff": dict(num_agents=2, horizon=200),
    "five_card_draw": dict(num_agents=2),
    "seven_card_stud": dict(num_agents=2),
    "werewolf": dict(num_agents=6, num_werewolves=2, horizon=200),
}
# logged per policy as {stat}_{avg|br}_vs_{random|baseline}
EVAL_STATS = (
    "win_rate",
    "loss_rate",
    "draw_rate",
    "win_rate_decided",
    "avg_return",
    "avg_episode_length",
)
WEREWOLF_STATS = ("win_rate_as_werewolf", "win_rate_as_human", "learner_werewolf_frac")


def _module(game, algo):
    for dep in ("distrax", "optax", "hydra", "wandb"):
        pytest.importorskip(dep)
    return importlib.import_module(f"bluffjax.examples.{game}.{game}_{algo}_nfsp")


def _evaluator_name(game):
    """The script's module attribute that plays the evaluation games."""
    return "play_games" if game == "bluff" else "play_eval_games"


def _eval_metrics(game, algo):
    return _module(game, algo).eval_metrics


def random_policy(params, obs, action_mask, rng):
    return jax.random.categorical(rng, jnp.where(action_mask, 0.0, -jnp.inf))


def counting_env(env):
    """A copy of env whose step_env pays every player 1 on a step with done
    (the game ended or reached the horizon) and 0 otherwise, so a player's
    return counts the ends of its game."""
    base = type(env)

    class CountingEnv(base):
        def step_env(self, rng, state, action):
            state, obs, reward, absorbing, done, info = base.step_env(
                self, rng, state, action
            )
            reward = jnp.where(done, 1.0, 0.0) * jnp.ones_like(reward)
            return state, obs, reward, absorbing, done, info

    counting = copy.copy(env)
    counting.__class__ = CountingEnv
    return counting


def _play(game, algo, env, num_games, key):
    play = getattr(_module(game, algo), _evaluator_name(game))
    return jax.jit(
        lambda k: play(env, random_policy, None, random_policy, None, num_games, k)
    )(key)


def _np(tree):
    return jax.tree_util.tree_map(np.asarray, tree)


@pytest.mark.parametrize("game,algo", SCRIPTS)
def test_plays_exactly_num_games_complete_games(game, algo):
    env = counting_env(make(game, **ENV_KWARGS[game]))
    for num_games in (1, 10, 37):
        result = _np(_play(game, algo, env, num_games, jax.random.PRNGKey(num_games)))
        assert result.ret.shape == result.length.shape == (num_games,)
        # every game ended exactly once (and nothing was counted after its end)
        np.testing.assert_array_equal(result.ret, np.ones(num_games))
        assert (result.length >= 1).all()
        if game in ("bluff", "werewolf"):
            assert (result.length <= env.horizon).all()
        np.testing.assert_array_equal(result.win + result.loss + result.draw, 1.0)


@pytest.mark.parametrize("game,algo", SCRIPTS)
def test_learner_seat_rotates(game, algo):
    kwargs = dict(ENV_KWARGS[game])
    if game != "werewolf":
        kwargs["num_agents"] = 3
    env = make(game, **kwargs)
    n = env.num_agents
    play = getattr(_module(game, algo), _evaluator_name(game))
    seats = jax.jit(
        jax.vmap(
            lambda k: play(env, random_policy, None, random_policy, None, 10, k).seat
        )
    )(jax.random.split(KEY, 60))
    seats = np.asarray(seats)
    for row in seats:
        np.testing.assert_array_equal(row, (row[0] + np.arange(10)) % n)
        counts = np.bincount(row, minlength=n)
        assert counts.max() - counts.min() <= 1
    assert set(seats[:, 0]) == set(range(n))  # random start seat


@pytest.mark.parametrize("game,algo", [s for s in SCRIPTS if s[0] != "werewolf"])
@pytest.mark.parametrize("num_agents", [2, 3])
def test_random_learner_against_random_is_symmetric(game, algo, num_agents):
    """Uniformly random play against itself, the seats used equally: with two
    players a win of one seat is a loss of the other, so wins and losses are
    equally likely; a decided Bluff game has one winner, so with three players
    the learner wins a third of them; poker is zero-sum, so the mean return
    is 0."""
    kwargs = dict(ENV_KWARGS[game], num_agents=num_agents)
    if game == "bluff":
        kwargs["horizon"] = 4000  # nearly every random game ends
    env = make(game, **kwargs)
    num_games = 1200 if game == "bluff" else 3000  # multiples of 2 and 3
    result = _np(_play(game, algo, env, num_games, jax.random.PRNGKey(7)))
    np.testing.assert_array_equal(np.bincount(result.seat), num_games // num_agents)
    win, loss, draw = result.win.sum(), result.loss.sum(), result.draw.sum()
    assert win + loss + draw == num_games
    if num_agents == 2:
        assert abs(win - loss) < 5 * np.sqrt(win + loss), (win, loss)
        assert draw <= 0.05 * num_games  # Bluff: the horizon; poker: split pots
    elif game == "bluff":
        rate = win / (win + loss)
        assert abs(rate - 1 / 3) < 5 * np.sqrt(2 / 9 / (win + loss)), rate
    if game == "bluff":
        assert draw <= 0.05 * num_games
    else:
        se = result.ret.std() / np.sqrt(num_games)
        assert abs(result.ret.mean()) < 5 * se, (result.ret.mean(), se)
    metrics = _np(_eval_metrics(game, algo)("x", result))
    assert set(metrics) == {f"{stat}_x" for stat in EVAL_STATS}
    np.testing.assert_allclose(metrics["win_rate_x"], win / num_games, rtol=1e-6)
    np.testing.assert_allclose(metrics["win_rate_decided_x"], win / (win + loss), rtol=1e-6)
    np.testing.assert_allclose(metrics["avg_return_x"], result.ret.mean(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(metrics["avg_episode_length_x"], result.length.mean(), rtol=1e-6)


@pytest.mark.parametrize("game", GAMES)
def test_ppo_and_pqn_evaluate_the_same_way(game):
    ppo, pqn = _module(game, "ppo"), _module(game, "pqn")
    env = make(game, **ENV_KWARGS[game])
    a = _np(_play(game, "ppo", env, 40, KEY))
    b = _np(_play(game, "pqn", env, 40, KEY))
    assert a._fields == b._fields
    jax.tree_util.tree_map(np.testing.assert_array_equal, a._asdict(), b._asdict())
    ma = _np(ppo.eval_metrics("avg_vs_random", a))
    mb = _np(pqn.eval_metrics("avg_vs_random", b))
    assert ma.keys() == mb.keys()
    jax.tree_util.tree_map(np.testing.assert_array_equal, ma, mb)


# ---------------------------------------------------------------- training runs


class Recorder:
    """Stands in for the script's ResultsLog."""

    def __init__(self):
        self.rows = []

    def log(self, metrics, seed_index=0):
        self.rows.append(dict(metrics))


def tiny_config(game, algo, num_updates, **overrides):
    """The script's config with a tiny budget of num_updates updates."""
    from omegaconf import OmegaConf

    from bluffjax.examples.run_utils import half_update_steps

    mod = _module(game, algo)
    config = OmegaConf.load(
        os.path.join(os.path.dirname(mod.__file__), f"config_{algo}_nfsp.yaml")
    )
    del config["hydra"]
    config = OmegaConf.to_container(config)  # save_dir stays unresolved (unused)
    config.update(
        num_envs=8,
        num_steps_per_env_per_update=16,
        num_epochs=1,
        num_minibatches=2,
        fc_dim_size=16,
        sl_reservoir_capacity=256,
        sl_batch_size=16,
        sl_num_steps_per_update=2,
        num_seeds=1,
        anticipatory_eta=0.5,
        save_checkpoints=False,
        wandb=False,
    )
    config.update(overrides)
    config["num_update_steps"] = num_updates
    config["half_update_steps"] = half_update_steps(num_updates)
    config["num_gradient_steps"] = (
        num_updates * config["num_epochs"] * config["num_minibatches"]
    )
    return config


def _expected_eval_keys(game):
    stats = EVAL_STATS + (WEREWOLF_STATS if game == "werewolf" else ())
    return {f"{stat}_{policy}_vs_random" for stat in stats for policy in ("avg", "br")}


@pytest.mark.parametrize("game,algo", SCRIPTS)
def test_training_run_evaluates_every_update(game, algo, monkeypatch):
    """compare_interval 1: both policies play compare_episodes complete games
    in every update, and every update logs the evaluation, with the names the
    other script of the game uses. The evaluator is wrapped so that its env
    counts the ends of the games (see counting_env), so the logged return is
    exactly 1 per game."""
    mod = _module(game, algo)
    U, E = 3, 5
    config = tiny_config(game, algo, U, compare=True, compare_episodes=E, compare_interval=1)
    name = _evaluator_name(game)
    real = getattr(mod, name)
    traced, ran = [], []

    def spy(env, learner, learner_params, baseline, baseline_params, num_games, rng):
        traced.append(num_games)
        result = real(
            counting_env(env), learner, learner_params, baseline, baseline_params,
            num_games, rng,
        )
        jax.debug.callback(
            lambda ret, seat: ran.append((np.asarray(ret), np.asarray(seat))),
            result.ret,
            result.seat,
        )
        return result

    monkeypatch.setattr(mod, name, spy)
    log = Recorder()
    monkeypatch.setattr(mod, "RESULTS_LOG", log)
    jax.block_until_ready(jax.jit(mod.make_train(config))(KEY, jnp.int32(0)))
    jax.effects_barrier()

    assert traced and set(traced) == {E}
    assert len(ran) == 2 * U  # the average policy and the BR, every update
    n = ENV_KWARGS[game]["num_agents"]
    for ret, seat in ran:
        np.testing.assert_array_equal(ret, np.ones(E))  # E games, each ended once
        np.testing.assert_array_equal(seat, (seat[0] + np.arange(E)) % n)
    assert [r["update_step"] for r in log.rows] == list(range(U))
    for r in log.rows:
        assert {k for k in r if "_vs_" in k} == _expected_eval_keys(game)
        for policy in ("avg", "br"):
            name = f"{policy}_vs_random"
            assert r[f"avg_return_{name}"] == pytest.approx(1.0)
            total = sum(r[f"{x}_rate_{name}"] for x in ("win", "loss", "draw"))
            assert total == pytest.approx(1.0, abs=1e-6)
            assert r[f"avg_episode_length_{name}"] >= 1.0
