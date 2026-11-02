"""Self-play PPO for Goofspiel (bluffjax/examples/goofspiel/goofspiel_ppo.py):

- the sweep grid: product order, seeds innermost, defaults filled in,
  unknown keys refused;
- a short vmapped run of a 2-point grid x 2 seeds through the real logging
  path: one record per update and instance with its hp/ values, the exact
  metrics only on exact-evaluation updates, the logged exploitability equal
  to goofspiel_exploitability on the returned parameters, the learning-rate
  and entropy schedules per instance, and different instances of the same
  seed differing only where the swept value differs.
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax.examples.goofspiel import goofspiel_ppo as P
from bluffjax.examples.run_utils import ResultsLog, half_update_steps
from bluffjax.utils.game_utils import goofspiel_exploitability as GE

BASE = dict(
    num_timesteps=8 * 16 * 4,
    num_envs=16,
    num_steps_per_env_per_update=4,
    num_epochs=2,
    num_minibatches=2,
    lr=1e-3,
    lr_final_frac=0.5,
    gamma=1.0,
    gae_lambda=0.95,
    clip_eps=0.2,
    vf_clip=4.0,
    ent_coef=0.1,
    ent_final_frac=0.0,
    vf_coef=1.0,
    max_grad_norm=0.5,
    fc_dim_size=16,
    sweep=None,
    num_seeds=2,
    seed=0,
    env_kwargs={"num_agents": 2, "num_cards": 4},
    exact_eval_interval=3,
    compare=True,
    compare_episodes=4,
    compare_interval=2,
)


def config(**kw):
    c = {**BASE, **kw}
    c["num_update_steps"] = c["num_timesteps"] // c["num_envs"] // c["num_steps_per_env_per_update"]
    c["half_update_steps"] = half_update_steps(c["num_update_steps"])
    return c


def test_instances_grid():
    insts = P.instances(config(sweep={"lr": [1e-4, 1e-3], "ent_coef": [0.0, 0.5, 1.0]}))
    assert len(insts) == 2 * 3 * 2
    assert [(i["hp/lr"], i["hp/ent_coef"], i["hp/seed"]) for i in insts[:4]] == [
        (1e-4, 0.0, 0), (1e-4, 0.0, 1), (1e-4, 0.5, 0), (1e-4, 0.5, 1),
    ]
    assert insts[-1]["hp/grid_index"] == 5 and insts[-1]["hp/lr"] == 1e-3
    assert all(i["hp/gamma"] == 1.0 for i in insts)
    with pytest.raises(ValueError):
        P.instances(config(sweep={"num_epochs": [1, 2]}))


def test_short_run(tmp_path, monkeypatch):
    cfg = config(sweep={"ent_coef": [0.0, 0.1]})
    insts = P.instances(cfg)
    monkeypatch.setattr(P, "INSTANCES", insts)
    monkeypatch.setattr(P, "RESULTS_LOG", ResultsLog(str(tmp_path)))
    monkeypatch.setattr(P, "EVALS", {})
    monkeypatch.setattr(P, "EXACT", {})
    tables = GE.tables_for_training(cfg, P.make("goofspiel", **cfg["env_kwargs"]))
    train = P.make_train(cfg)
    keys = jax.random.split(jax.random.PRNGKey(0), cfg["num_seeds"])
    rngs = keys[jnp.asarray([i["hp/seed"] for i in insts])]
    final = jax.jit(jax.vmap(train, in_axes=(0, 0, 0, None)))(
        rngs, jnp.arange(len(insts)), P.hparams_of(insts), tables
    )
    jax.block_until_ready(final)
    jax.effects_barrier()

    rows = [json.loads(l) for l in open(os.path.join(tmp_path, "results.jsonl"))]
    U = cfg["num_update_steps"]
    assert len(rows) == U * len(insts)
    for r in rows:
        inst = insts[r["seed_index"]]
        assert all(r[k] == pytest.approx(v) for k, v in inst.items())
        u = r["update_step"]
        assert ("exploitability" in r) == (u % 3 == 0 or u == U - 1)
        assert ("win_rate_vs_random" in r) == (u % 2 == 0 or u == U - 1)
        assert "game/bid_entropy" in r
        # schedules after (u + 1) updates of num_epochs x num_minibatches steps
        done = (u + 1) * cfg["num_epochs"] * cfg["num_minibatches"]
        total = U * cfg["num_epochs"] * cfg["num_minibatches"]
        left = 1 - done / total
        assert r["lr"] == pytest.approx(1e-3 * (0.5 + 0.5 * left), rel=1e-5)
        # the entropy coefficient used in update u (before its steps)
        left_u = 1 - u * cfg["num_epochs"] * cfg["num_minibatches"] / total
        assert r["ent_coef"] == pytest.approx(inst["hp/ent_coef"] * left_u, abs=1e-7)

    # The last logged exploitability is that of the returned parameters.
    net = P.ActorCriticDiscreteMLP(action_dim=4, hidden_dim=16)
    for i in range(len(insts)):
        params = jax.tree_util.tree_map(lambda x: x[i], final.train_state.params)
        probs = GE.policy_probs(net.apply(params, tables.obs)[0], tables.legal)
        last = [r for r in rows if r["seed_index"] == i and r["update_step"] == U - 1][0]
        assert last["exploitability"] == pytest.approx(float(GE.exploitability(tables, probs)), rel=1e-4)
        assert last["value_vs_random"] == pytest.approx(float(GE.value_vs_random(tables, probs)), abs=1e-4)

    # Same seed, entropy 0 vs 0.1: same first rollout, different training.
    first = {r["seed_index"]: r for r in rows if r["update_step"] == 0}
    assert first[0]["game/bid_entropy"] == pytest.approx(first[2]["game/bid_entropy"])
    p0 = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x[0], final.train_state.params))
    p2 = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x[2], final.train_state.params))
    assert not all(np.allclose(a, b) for a, b in zip(p0, p2))
