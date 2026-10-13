"""Checks of a tiny training run of a turn-based (AEC) NFSP script, shared by
the game-metric tests of Bluff and Werewolf (test_metrics_bluff.py,
test_metrics_werewolf.py).

check_training_run runs the script's hydra main() on CPU with a tiny config
and checks:
- results.jsonl has one record per update and seed, with env_steps and every
  game/ and game_br/ rate and event count;
- the game/ and game_br/ masks are the policies of the credited players: an
  extra count credited to the player to act sums, under each mask, to the
  decisions of that policy (br_action_frac_rollout), and in a separate run
  the masks the script passes to masked_totals are exactly ~br_mode and
  br_mode of every rollout step (the policy each player follows in the game
  the step belongs to, also on a game's last step);
- half/ and full/ hold avg_policy_{i} and br_{i} of both seeds, which load
  strictly into the script's networks; the half ones equal the networks of
  a run of exactly half_update_steps updates, the full ones those of a run
  of all of them; a saved network is accepted as an evaluation opponent.
"""

import functools
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

KEY = jax.random.PRNGKey(0)
NUM_ENVS, NUM_STEPS = 8, 16


def acting_metrics(metrics_cls):
    """metrics_cls plus `acting`: one count per decision, credited to the
    player to act. Masked by the policy of the credited player it counts the
    decisions of that policy."""

    class ActingMetrics(metrics_cls):
        rate_names = metrics_cls.rate_names + ("acting",)

        def step(self, carry, state, action, next_state, reward, done, info, probs=None):
            carry, c = super().step(carry, state, action, next_state, reward, done, info)
            acting = jnp.zeros(self.num_agents, jnp.float32)
            return carry, {**c, "acting": acting.at[state.current_player_idx].set(1.0)}

        def summarize(self, totals):
            return {**super().summarize(totals), "acting": totals["acting"]}

    return ActingMetrics


class Recorder:
    """Stands in for the script's ResultsLog."""

    def __init__(self):
        self.rows = []

    def log(self, metrics, seed_index=0):
        self.rows.append((seed_index, metrics))


def max_diff(a, b):
    pairs = zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    return max(float(np.abs(np.asarray(x) - np.asarray(y)).max()) for x, y in pairs)


def compose_config(mod, config_name, tmp_path, num_updates, overrides):
    """The script's hydra config with a tiny budget (num_updates updates of
    NUM_ENVS envs x NUM_STEPS steps, two seeds) and `overrides`."""
    from hydra import compose, initialize_config_dir

    from bluffjax.utils.paths import register_resolvers

    register_resolvers()
    overrides = {
        "num_timesteps": num_updates * NUM_ENVS * NUM_STEPS,
        "num_envs": NUM_ENVS,
        "num_steps_per_env_per_update": NUM_STEPS,
        "num_minibatches": 2,
        "fc_dim_size": 16,
        "sl_reservoir_capacity": 256,
        "sl_batch_size": 16,
        "num_seeds": 2,
        "anticipatory_eta": 0.5,
        "wandb": False,
        "save_dir": str(tmp_path),
        "compare_interval": 2,
        **overrides,
    }
    with initialize_config_dir(
        config_dir=os.path.dirname(mod.__file__), version_base=None
    ):
        return compose(
            config_name=config_name,
            overrides=[f"{k}={v}" for k, v in overrides.items()],
        )


def check_masks_follow_game_policies(mod, monkeypatch, config, num_updates):
    """Runs num_updates updates of one seed, recording every rollout step's
    br_mode, is_br, player and done (from the Transition) and the masks the
    script passes to masked_totals. The game/ mask must be ~br_mode and the
    game_br/ one br_mode, at every step and for every player: the policy the
    credited player follows in the game the step belongs to (also on the
    step that ends a game, whose events belong to the game that ended).
    Returns the number of games that ended."""
    seen = {"steps": [], "masks": [], "traced": 0}
    real_transition, real_masked_totals = mod.Transition, mod.masked_totals

    def transition_spy(**fields):
        jax.debug.callback(
            lambda *a: seen["steps"].append([np.asarray(x) for x in a]),
            fields["br_mode"], fields["is_br"], fields["player_idx"], fields["done"],
        )
        return real_transition(**fields)

    def masked_totals_spy(counts, mask):
        # the script calls it for game/, then game_br/ (in program order; the
        # callbacks may run in either order)
        tag = ("game", "game_br")[seen["traced"] % 2]
        seen["traced"] += 1
        jax.debug.callback(lambda m: seen["masks"].append((tag, np.asarray(m))), mask)
        return real_masked_totals(counts, mask)

    monkeypatch.setattr(mod, "Transition", transition_spy)
    monkeypatch.setattr(mod, "masked_totals", masked_totals_spy)
    monkeypatch.setattr(mod, "RESULTS_LOG", Recorder())
    train = mod.make_train(config)
    jax.block_until_ready(
        jax.jit(functools.partial(train, num_updates=num_updates))(KEY, jnp.int32(0))
    )
    jax.effects_barrier()
    monkeypatch.setattr(mod, "Transition", real_transition)
    monkeypatch.setattr(mod, "masked_totals", real_masked_totals)

    T = config["num_steps_per_env_per_update"]
    assert len(seen["steps"]) == num_updates * T
    assert len(seen["masks"]) == 2 * num_updates
    # (U * T, N, P), (U * T, N), (U * T, N), (U * T, N)
    br_mode, is_br, player, done = (np.stack(x) for x in zip(*seen["steps"]))
    # br_mode is the policy of every player in the current game: the acting
    # player follows it, and it is only redrawn after a game ends
    np.testing.assert_array_equal(
        is_br, np.take_along_axis(br_mode, player[..., None], -1)[..., 0]
    )
    changed = (br_mode[1:] != br_mode[:-1]).any(-1)
    assert not (changed & ~done[:-1]).any() and changed.any()
    for u in range(num_updates):
        rollout = br_mode[u * T : (u + 1) * T]
        masks = dict(seen["masks"][2 * u : 2 * u + 2])
        np.testing.assert_array_equal(masks["game"], ~rollout)
        np.testing.assert_array_equal(masks["game_br"], rollout)
    return done.sum()


def check_training_run(
    mod,
    monkeypatch,
    tmp_path,
    *,
    name,
    config_name,
    env_id,
    metrics_cls,
    count_names,
    networks,
    opponent_types,
    overrides,
    short_env_kwargs,
):
    """Runs mod.main() for 3 updates (half after 1) and checks it (see the
    module docstring).

    count_names: the n_ keys summarize() returns; networks(env, config) ->
    (avg network, br network); opponent_types: compare_network_type for the
    avg_policy and br files; short_env_kwargs: env kwargs under which games
    end within the rollouts. Returns the records of results.jsonl."""
    from omegaconf import OmegaConf

    from bluffjax import make
    from bluffjax.examples import run_utils

    monkeypatch.setattr(
        mod, "make_metrics", lambda env_id, env: acting_metrics(metrics_cls)(env)
    )
    U = 3
    mod.main(compose_config(mod, config_name, tmp_path, U, overrides))
    jax.effects_barrier()
    (run_name,) = os.listdir(tmp_path)
    run_dir = os.path.join(tmp_path, run_name)
    assert run_name.startswith(f"{name}_")

    # results.jsonl: one record per update and seed, with the game metrics
    records = run_utils.read_results(run_dir)
    assert sorted((r["seed_index"], r["update_step"]) for r in records) == [
        (i, u) for i in range(2) for u in range(U)
    ]
    decisions = NUM_ENVS * NUM_STEPS
    for r in records:
        assert r["env_steps"] == (r["update_step"] + 1) * decisions
        for prefix in ("game", "game_br"):
            for key in metrics_cls.rate_names + tuple(count_names):
                assert f"{prefix}/{key}" in r, (prefix, key)
            for key in metrics_cls.rate_names:
                value = r[f"{prefix}/{key}"]
                assert math.isnan(value) or value >= 0.0, (prefix, key, value)
        # masked by the credited player's policy: the decisions of each policy
        assert r["game/acting"] + r["game_br/acting"] == decisions
        assert r["game_br/acting"] == pytest.approx(
            r["br_action_frac_rollout"] * decisions, abs=1e-3
        )
    config = OmegaConf.to_container(OmegaConf.load(os.path.join(run_dir, "config.yaml")))
    assert config["run_dir"] == run_dir and config["half_update_steps"] == 1

    # half/ and full/ checkpoints load strictly into the script's networks
    env = make(env_id, **config["env_kwargs"])
    obs = jnp.zeros((env.obs_dim,))
    avg_net, br_net = networks(env, config)
    templates = {
        "avg_policy": avg_net.init(jax.random.PRNGKey(1), obs),
        "br": br_net.init(jax.random.PRNGKey(2), obs),
    }
    found = {
        os.path.relpath(os.path.join(d, f), run_dir)
        for d, _, files in os.walk(run_dir)
        for f in files
    }
    assert found == {"config.yaml", "results.jsonl"} | {
        f"{stage}/{net}_{i}.msgpack"
        for stage in ("half", "full")
        for net in templates
        for i in range(2)
    }
    loaded = {
        (stage, net, i): run_utils.load_params(
            run_utils.checkpoint_path(run_dir, stage, net, i), templates[net]
        )
        for stage in ("half", "full")
        for net in templates
        for i in range(2)
    }

    # the half checkpoint is the networks after exactly half_update_steps
    # updates of the same run; the full one after all of them
    monkeypatch.setattr(mod, "RESULTS_LOG", Recorder())
    train = mod.make_train(config)
    keys = jax.random.split(jax.random.PRNGKey(config["seed"]), 2)
    for num_updates, stage in ((config["half_update_steps"], "half"), (U, "full")):
        state = jax.jit(jax.vmap(functools.partial(train, num_updates=num_updates)))(
            keys, jnp.arange(2)
        )
        jax.effects_barrier()
        for net, attr in (("avg_policy", "avg_train_state"), ("br", "br_train_state")):
            params = getattr(state, attr).params
            for i in range(2):
                want = jax.tree_util.tree_map(lambda x: x[i], params)
                assert max_diff(loaded[(stage, net, i)], want) < 1e-6, (stage, net, i)
    # ... and they differ (by far more than the tolerance above; the Werewolf
    # PQN BR's RAdam makes small first steps)
    for net in templates:
        diff = max_diff(loaded[("half", net, 0)], loaded[("full", net, 0)])
        assert diff > 1e-5, (net, diff)

    # the game/ and game_br/ masks are the policies of the credited players
    short = dict(config, env_kwargs=dict(config["env_kwargs"], **short_env_kwargs))
    assert check_masks_follow_game_policies(mod, monkeypatch, short, 3) > 0

    # a saved network is a valid evaluation opponent
    for net, network_type in zip(("avg_policy", "br"), opponent_types):
        mod.make_train(
            dict(
                config,
                compare_mode="checkpoint",
                compare_with=run_utils.checkpoint_path(run_dir, "full", net, 1),
                compare_network_type=network_type,
            )
        )
    return records
