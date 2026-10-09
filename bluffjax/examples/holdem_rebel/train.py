"""
ReBeL self-play training for heads-up Limit / No-Limit Hold'em.

One sample is one value-network training example: a PBS together with both
players' per-hand values (2 x 1,326 numbers) computed by solving a subgame
rooted at it. `num_samples` counts these.

Self-play (ReBeL Algorithm 2, with the sampling of the released code):
  - `num_games` games run in parallel. Each step solves the betting-round
    subgame at every game's current PBS with Linear CFR-D (value network at the
    end-of-round leaves), adds the root values as an example, then samples a
    path to a leaf of the subgame with the policy of a random iteration t,
    starting from a deal drawn from the root PBS. One player, picked at random
    per subgame, takes a uniformly random action with probability
    `explore_prob` at each decision; beliefs follow pi^t either way.
  - At an end-of-round leaf the PBS before the next card(s) is a chance node.
    Its value is the card-removal-weighted average of the network's values
    over the next cards (all turn/river cards, `chance_flops` sampled flops);
    it is added as an example and a card is dealt. When both players are
    all-in the remaining streets have no betting, so the chain of chance nodes
    is followed to the river, where values are exact showdowns.
  - Games whose hand ended restart from the initial PBS.
The value network is trained on a FIFO replay buffer with the Huber loss,
`train_ratio` trained examples per generated example (the released code's
train_gen_ratio).
"""

import datetime
import os
import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
import wandb
from jax import lax

from bluffjax import make
from bluffjax.examples.holdem_rebel.agent import (
    RebelPlayer,
    heuristic_switch,
    load_rebel,
    play_match,
    save_checkpoint,
    summarize,
)
from bluffjax.examples.holdem_rebel.cards import (
    HAND_CARDS,
    NUM_HANDS,
    cards_onehot,
    compatible_mass,
    hands_not_blocked,
    normalize,
)
from bluffjax.examples.holdem_rebel.game import (
    DECISION,
    FOLD,
    PUBLIC_DIM,
    ROUND_END,
    HoldemGame,
    Template,
    num_board_cards,
    public_features,
    round_root,
)
from bluffjax.examples.holdem_rebel.solver import (
    ValueNetwork,
    add_cards,
    chance_children,
    chance_values,
    make_subgame,
    masked_map,
    solve,
)
from bluffjax.utils.paths import REPO_ROOT
from bluffjax.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray


class Games(NamedTuple):
    """Self-play games, each at the start of a betting round."""

    street: IntArray  # (G,)
    board: IntArray  # (G, 5), -1 for cards not dealt
    chips: FloatArray  # (G,) chips per player at the start of the round
    beliefs: FloatArray  # (G, 2, 1326)


class Examples(NamedTuple):
    pub: FloatArray  # (E, PUBLIC_DIM)
    beliefs: FloatArray  # (E, 2, 1326)
    values: FloatArray  # (E, 2, 1326)
    mask: BoolArray  # (E, 2, 1326)
    valid: BoolArray  # (E,)


class Replay(NamedTuple):
    pub: FloatArray
    beliefs: jax.Array  # bfloat16
    values: jax.Array  # bfloat16
    mask: BoolArray
    ptr: IntArray
    size: IntArray


def new_games(n: int) -> Games:
    return Games(
        street=jnp.zeros(n, jnp.int32),
        board=-jnp.ones((n, 5), jnp.int32),
        chips=jnp.zeros(n),
        beliefs=jnp.full((n, 2, NUM_HANDS), 1.0 / NUM_HANDS, jnp.float32),
    )


def sample_deal(beliefs: FloatArray, rng: PRNGKeyArray) -> tuple[IntArray, IntArray]:
    """Draw (h_sb, h_bb) from P(h0, h1) ~ x0(h0) x1(h1) [no shared card]."""
    k0, k1 = jax.random.split(rng)
    p0 = beliefs[0] * compatible_mass(beliefs[1])
    p0 = jnp.where(p0.sum() > 0, p0, beliefs[0])
    h0 = jax.random.categorical(k0, jnp.log(p0))
    blocked = cards_onehot(HAND_CARDS[h0], 2)
    p1 = beliefs[1] * hands_not_blocked(blocked)
    p1 = jnp.where(p1.sum() > 0, p1, hands_not_blocked(blocked) * 1.0)
    h1 = jax.random.categorical(k1, jnp.log(p1))
    return h0, h1


def deal_next_street(
    street: IntArray, board: IntArray, beliefs: FloatArray, rng: PRNGKeyArray
) -> tuple[IntArray, FloatArray]:
    """Deal the next street to a PBS: sample a history, then the card(s)."""
    k_deal, k_cards = jax.random.split(rng)
    h0, h1 = sample_deal(beliefs, k_deal)
    used = cards_onehot(jnp.where(board >= 0, board, 0), num_board_cards(street))
    used = used.at[HAND_CARDS[h0]].set(True).at[HAND_CARDS[h1]].set(True)
    order = jnp.argsort(jax.random.uniform(k_cards, (52,)) + 2.0 * used)
    cards = jnp.where(street == 0, order[:3], jnp.array([order[0], -1, -1]))
    new_board = add_cards(board, street, cards)
    keep = hands_not_blocked(cards_onehot(new_board, num_board_cards(street + 1)))
    return new_board, normalize(beliefs * keep)


def sample_leaf(
    tpl: Template,
    policy: FloatArray,
    tree,
    beliefs: FloatArray,
    explore_prob: float,
    rng: PRNGKeyArray,
) -> tuple[IntArray, FloatArray]:
    """Walk from the root to a leaf with pi^t (+ exploration), updating beliefs."""
    k_deal, k_br, k_walk = jax.random.split(rng, 3)
    hands = jnp.stack(sample_deal(beliefs, k_deal))
    explorer = jax.random.randint(k_br, (), 0, 2)
    children = jnp.asarray(tpl.children)
    internal_id = jnp.asarray(tpl.internal_id)
    node = jnp.int32(0)
    for key in jax.random.split(k_walk, tpl.max_depth):
        k_u, k_a, k_r = jax.random.split(key, 3)
        is_dec = tree.kind[node] == DECISION
        ni = jnp.maximum(internal_id[node], 0)
        legal = tree.legal[ni]
        actor = tree.actor[node]
        probs = jnp.where(legal, policy[ni, :, hands[actor]], 0.0)
        probs = jnp.where(probs.sum() > 0, probs, legal * 1.0)
        explore = (actor == explorer) & (jax.random.uniform(k_u) < explore_prob)
        slot = jnp.where(
            explore,
            jax.random.categorical(k_r, jnp.where(legal, 0.0, -jnp.inf)),
            jax.random.categorical(k_a, jnp.log(probs)),
        )
        # Beliefs follow pi^t whether or not the action was exploratory.
        updated = beliefs[actor] * policy[ni, slot]
        updated = jnp.where(updated.sum() > 0, normalize(updated), beliefs[actor])
        beliefs = jnp.where(is_dec, beliefs.at[actor].set(updated), beliefs)
        node = jnp.where(is_dec, children[node, slot], node)
    return node, beliefs


def make_self_play_step(game: HoldemGame, tpl: Template, net: ValueNetwork, cfg: dict):
    num_flops = int(cfg["chance_flops"])
    explore = float(cfg["explore_prob"])
    cfr_iters = int(cfg["cfr_iters"])
    chunk = int(cfg["solve_chunk"])
    exact_group = 8  # games per group for exact all-in showdowns (52 boards each)

    root0 = round_root(game, jnp.int32(0), jnp.float32(0.0))

    def initial_subgame():
        return make_subgame(
            game, tpl, root0, -jnp.ones(5, jnp.int32),
            jnp.full((2, NUM_HANDS), 1.0 / NUM_HANDS, jnp.float32), with_showdown=False,
        )

    def solve_initial(params):
        """Solve the initial PBS, keeping every iteration's policy.

        It is the same in every hand and depends only on the network, so it is
        solved once per batch of self-play steps (the network is fixed within
        a batch), and each game starting a hand draws its own iteration t,
        which is the same as solving it separately for each game.
        """
        value_fn = lambda pub, b: net.apply(params, pub, b)  # noqa: E731
        sol = solve(game, tpl, initial_subgame(), value_fn, jax.random.PRNGKey(0),
                    cfr_iters, "net", all_policies=True)
        return sol._replace(avg_policy=jnp.zeros((), jnp.float32))

    def solve_games(params, shared, games: Games, rng):
        """Solve every game's betting-round subgame; games at the initial PBS
        use the shared solve, whose example is added once per step."""
        value_fn = lambda pub, b: net.apply(params, pub, b)  # noqa: E731
        k_solve, k_t = jax.random.split(rng)
        n = games.street.shape[0]
        fresh = games.street == 0
        sg0 = initial_subgame()

        def one(root, board, beliefs, key, mode):
            sg = make_subgame(game, tpl, root, board, beliefs, with_showdown=mode != "net")
            sol = solve(game, tpl, sg, value_fn, key, cfr_iters, mode)
            return sol.values, sol.value_mask, sol.policy, sol.tree, sg.beliefs

        def group_fn(a):
            return lax.cond(
                jnp.any(a[0].street == 3),
                lambda: jax.vmap(lambda *x: one(*x, "both"))(*a),
                lambda: jax.vmap(lambda *x: one(*x, "net"))(*a),
            )

        roots = jax.vmap(lambda st, c: round_root(game, st, c))(games.street, games.chips)
        keys = jax.random.split(k_solve, n)
        values, mask, policy, tree, beliefs = masked_map(
            group_fn, ~fresh, (roots, games.board, games.beliefs, keys), chunk,
            sort_key=games.street == 3,
        )
        t = jax.random.categorical(
            k_t, jnp.log(jnp.arange(1, cfr_iters + 1, dtype=jnp.float32)), shape=(n,)
        )
        pick = lambda a, b: jnp.where(  # noqa: E731
            fresh.reshape((n,) + (1,) * (b.ndim - 1)), a, b
        )
        policy = pick(shared.policy[t], policy)
        tree = jax.tree.map(lambda a, b: pick(jnp.broadcast_to(a, b.shape), b), shared.tree, tree)
        beliefs = pick(jnp.broadcast_to(sg0.beliefs, beliefs.shape), beliefs)
        pub = jax.vmap(
            lambda st, b, c: public_features(game, st, b, False, False, c)
        )(games.street, games.board, roots.chips[:, 0])
        ex = Examples(pub, beliefs, values, mask, ~fresh)
        ex_root = Examples(
            public_features(game, jnp.int32(0), -jnp.ones(5, jnp.int32), False, False, root0.chips[0])[None],
            sg0.beliefs[None],
            shared.values[None],
            shared.value_mask[None],
            jnp.any(fresh)[None],
        )
        return ex, ex_root, policy, tree, beliefs, jnp.sum(~fresh) + 1

    def chance_level(params, street, board, chips, allin, beliefs, active, rng, flops):
        """One chance node per active game: an example, then deal the next cards."""
        value_fn = lambda pub, b: net.apply(params, pub, b)  # noqa: E731
        n = street.shape[0]
        k_child, k_deal = jax.random.split(rng)
        keys = jax.random.split(k_child, n)
        board_mask = jax.vmap(
            lambda b, st: hands_not_blocked(cards_onehot(jnp.where(b >= 0, b, 0), num_board_cards(st)))
        )(board, street)
        x = jax.vmap(normalize)(beliefs * board_mask[:, None, :])

        def values_fn(k_flops, exact):
            def group_fn(a):
                st, b, c, al, xx, kk = a
                ch = jax.vmap(lambda s_, b_, k_: chance_children(s_, b_, k_, k_flops))(st, b, kk)
                return jax.vmap(
                    lambda *z: chance_values(game, value_fn, *z, exact_showdown=exact)
                )(st, b, c, al, xx, ch)

            return group_fn

        args = (street, board, chips, allin, x, keys)
        exact = active & allin & (street == 2)
        pre = active & (street == 0)
        later = active & (street > 0) & ~exact
        res = masked_map(values_fn(flops, False), pre, args, chunk)
        res_later = masked_map(values_fn(1, False), later, args, chunk)
        res = jax.tree.map(lambda a, b: jnp.where(later.reshape((n,) + (1,) * (a.ndim - 1)), b, a), res, res_later)
        if not game.is_limit:
            res_exact = masked_map(values_fn(1, True), exact, args, exact_group)
            res = jax.tree.map(
                lambda a, b: jnp.where(exact.reshape((n,) + (1,) * (a.ndim - 1)), b, a),
                res, res_exact,
            )
        pub = jax.vmap(
            lambda s_, b_, a_, c_: public_features(game, s_, b_, True, a_, c_)
        )(street, board, allin, chips)
        ex = Examples(pub, x, res.values, res.value_mask, active)
        new_board, new_beliefs = jax.vmap(deal_next_street)(
            street, board, x, jax.random.split(k_deal, n)
        )
        return ex, new_board, new_beliefs

    def step(params, shared, games: Games, rng):
        k_solve, k_leaf, k_c1, k_c2, k_c3 = jax.random.split(rng, 5)
        n = games.street.shape[0]
        ex0, ex_root, policy, tree, beliefs, n_solved = solve_games(params, shared, games, k_solve)
        leaf, beliefs = jax.vmap(
            lambda p, t, b, k: sample_leaf(tpl, p, t, b, explore, k)
        )(policy, tree, beliefs, jax.random.split(k_leaf, n))
        rows = jnp.arange(n)
        kind = tree.kind[rows, leaf]
        chips = tree.chips[rows, leaf, 0]
        allin = tree.allin[rows, leaf]
        street, board = games.street, games.board
        to_chance = (kind == ROUND_END) & (street < 3)
        ex1, board1, beliefs1 = chance_level(
            params, street, board, chips, allin, beliefs, to_chance, k_c1, num_flops
        )
        examples = [ex0, ex_root, ex1]
        if not game.is_limit:
            # Both all-in: no more betting, follow the chance nodes to the river.
            a2 = to_chance & allin & (street + 1 < 3)
            ex2, board2, beliefs2 = chance_level(
                params, street + 1, board1, chips, allin, beliefs1, a2, k_c2, num_flops
            )
            a3 = a2 & (street + 2 < 3)
            ex3, _, _ = chance_level(
                params, street + 2, board2, chips, allin, beliefs2, a3, k_c3, num_flops
            )
            examples += [ex2, ex3]
        cont = to_chance & ~allin
        fresh = new_games(n)
        nxt = Games(
            street=jnp.where(cont, street + 1, fresh.street),
            board=jnp.where(cont[:, None], board1, fresh.board),
            chips=jnp.where(cont, chips, fresh.chips),
            beliefs=jnp.where(cont[:, None, None], beliefs1, fresh.beliefs),
        )
        examples = jax.tree.map(lambda *xs: jnp.concatenate(xs), *examples)
        stats = {
            "subgames_solved": n_solved,
            "hands_finished": jnp.sum(~cont),
            "folds": jnp.sum(kind == FOLD),
            "allin_chains": jnp.sum(to_chance & allin),
        }
        return nxt, examples, stats

    return solve_initial, step


def new_replay(capacity: int) -> Replay:
    return Replay(
        pub=jnp.zeros((capacity, PUBLIC_DIM)),
        beliefs=jnp.zeros((capacity, 2, NUM_HANDS), jnp.bfloat16),
        values=jnp.zeros((capacity, 2, NUM_HANDS), jnp.bfloat16),
        mask=jnp.zeros((capacity, 2, NUM_HANDS), bool),
        ptr=jnp.int32(0),
        size=jnp.int32(0),
    )


def replay_add(buf: Replay, ex: Examples) -> Replay:
    cap = buf.pub.shape[0]
    rank = jnp.cumsum(ex.valid) - 1
    idx = jnp.where(ex.valid, (buf.ptr + rank) % cap, cap)  # cap = dropped
    added = jnp.sum(ex.valid).astype(jnp.int32)
    return Replay(
        pub=buf.pub.at[idx].set(ex.pub, mode="drop"),
        beliefs=buf.beliefs.at[idx].set(ex.beliefs.astype(jnp.bfloat16), mode="drop"),
        values=buf.values.at[idx].set(ex.values.astype(jnp.bfloat16), mode="drop"),
        mask=buf.mask.at[idx].set(ex.mask, mode="drop"),
        ptr=(buf.ptr + added) % cap,
        size=jnp.minimum(buf.size + added, cap),
    )


def make_trainer(net: ValueNetwork, tx, batch_size: int):
    def loss_fn(params, pub, beliefs, values, mask):
        pred = net.apply(params, pub, beliefs)
        err = optax.huber_loss(pred, values, delta=1.0)
        return jnp.sum(err * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    def train_step(carry, _):
        params, opt_state, buf, rng = carry
        rng, k = jax.random.split(rng)
        idx = jax.random.randint(k, (batch_size,), 0, jnp.maximum(buf.size, 1))
        batch = (
            buf.pub[idx],
            buf.beliefs[idx].astype(jnp.float32),
            buf.values[idx].astype(jnp.float32),
            buf.mask[idx],
        )
        loss, grads = jax.value_and_grad(loss_fn)(params, *batch)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, buf, rng), loss

    def train(params, opt_state, buf, rng, num_steps):
        def body(i, carry):
            c, loss_sum = carry
            c, loss = train_step(c, None)
            return c, loss_sum + loss

        (params, opt_state, _, _), loss_sum = lax.fori_loop(
            0, num_steps, body, ((params, opt_state, buf, rng), 0.0)
        )
        return params, opt_state, loss_sum / jnp.maximum(num_steps, 1)

    return jax.jit(train)


def checkpoint_meta(game: HoldemGame, cfg: dict, samples: int) -> dict:
    return {
        "game": game.name,
        "max_raises": game.max_raises,
        "value_hidden_dim": int(cfg["value_hidden_dim"]),
        "value_num_layers": int(cfg["value_num_layers"]),
        "cfr_iters": int(cfg["cfr_iters"]),
        "samples": int(samples),
        "seed": int(cfg["seed"]),
    }


def run_training(game: HoldemGame, cfg: dict) -> str:
    """Train ReBeL for cfg['num_samples'] samples; returns the checkpoint path."""
    run = wandb.init(
        project=cfg["project"],
        group=f"{cfg['env_name']}_rebel"
        + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S"),
        job_type=f"{cfg['job_type']}_{cfg['env_name']}",
        name=f"{cfg['seed']}_0",
        config=cfg,
        mode=None if cfg["wandb"] else "disabled",
        dir=str(REPO_ROOT),
    )
    try:
        return _train(game, cfg, run)
    finally:
        run.finish()


def _train(game: HoldemGame, cfg: dict, run: wandb.Run) -> str:
    rng = jax.random.PRNGKey(cfg["seed"])
    tpl_player = RebelPlayer(
        game,
        hidden_dim=cfg["value_hidden_dim"],
        num_layers=cfg["value_num_layers"],
        cfr_iters=cfg["cfr_iters"],
        solve_chunk=cfg["solve_chunk"],
    )
    tpl, net = tpl_player.tpl, tpl_player.net
    rng, k_init = jax.random.split(rng)
    params = net.init(k_init, jnp.zeros((PUBLIC_DIM,)), jnp.zeros((2, NUM_HANDS)))
    num_samples = int(float(cfg["num_samples"]))
    halvings = [float(f) for f in cfg["lr_halve_at"]]
    lr = optax.piecewise_constant_schedule(
        cfg["value_lr"],
        {int(f * cfg["train_ratio"] * num_samples / cfg["batch_size"]): 0.5 for f in halvings},
    )
    tx = optax.chain(
        optax.clip_by_global_norm(cfg["grad_clip"]), optax.adam(learning_rate=lr)
    )
    opt_state = tx.init(params)
    train = make_trainer(net, tx, cfg["batch_size"])
    solve_initial, step = make_self_play_step(game, tpl, net, cfg)
    steps_per_chunk = int(cfg["steps_per_chunk"])

    @partial(jax.jit, donate_argnums=(1, 2))
    def generate(params, games, buf, rng):
        shared = solve_initial(params)

        def body(carry, key):
            games, buf = carry
            games, ex, stats = step(params, shared, games, key)
            return (games, replay_add(buf, ex)), (jnp.sum(ex.valid), stats)

        (games, buf), (added, stats) = lax.scan(
            body, (games, buf), jax.random.split(rng, steps_per_chunk)
        )
        return games, buf, jnp.sum(added), jax.tree.map(jnp.sum, stats)

    env = make(game.env_id, num_agents=2)
    heuristics, heuristic_names = heuristic_switch(game)
    eval_batch = int(cfg["eval_batch_deals"])
    play_heuristic = jax.jit(
        lambda p, idx, k: play_match(env, game, tpl_player, p, heuristics, idx, eval_batch, k)
    )
    play_checkpoint = None
    if cfg.get("compare_with"):
        ref_player, ref_params, _ = load_rebel(cfg["compare_with"], solve_chunk=cfg["solve_chunk"])
        play_checkpoint = jax.jit(
            lambda p, k: play_match(env, game, tpl_player, p, ref_player, ref_params, eval_batch, k)
        )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        cfg["save_dir"], f"{game.name}_rebel_seed{cfg['seed']}_{timestamp}.msgpack"
    )

    def evaluate(params, rng, num_deals):
        """Mirrored hands vs each opponent, in batches of eval_batch_deals deals."""
        out = {}
        opponents = [(name, i) for i, name in enumerate(heuristic_names)]
        if play_checkpoint is not None:
            opponents.append(("checkpoint", None))
        for name, idx in opponents:
            results = []
            for _ in range(max(1, num_deals // eval_batch)):
                rng, k = jax.random.split(rng)
                if idx is None:
                    results.append(jax.device_get(play_checkpoint(params, k)))
                else:
                    results.append(jax.device_get(play_heuristic(params, jnp.int32(idx), k)))
            res = summarize(results)
            out[f"eval/{name}_mean"] = res["mean"]
            out[f"eval/{name}_se"] = res["se"]
            print(
                f"  vs {name:13s}: {res['mean']:+.4f} +- {res['se']:.4f} per hand "
                f"({res['hands']} hands, re-solves {res['resolves']}, unfinished {res['unfinished']})"
            )
        return out

    games = new_games(cfg["num_games"])
    buf = new_replay(cfg["replay_capacity"])
    samples, trained, chunk_i = 0, 0, 0
    next_eval = float(cfg["eval_every"])
    t_start = time.time()
    gen_time = 0.0
    while samples < num_samples:
        rng, k_gen, k_train = jax.random.split(rng, 3)
        t0 = time.time()
        games, buf, added, stats = generate(params, games, buf, k_gen)
        added = int(added)
        gen_time += time.time() - t0
        samples += added
        loss = float("nan")
        if int(buf.size) >= 2 * cfg["batch_size"]:
            n_steps = int((cfg["train_ratio"] * samples - trained) // cfg["batch_size"])
            if n_steps > 0:
                params, opt_state, loss = train(params, opt_state, buf, k_train, n_steps)
                loss = float(loss)
                trained += n_steps * cfg["batch_size"]
        chunk_i += 1
        elapsed = time.time() - t_start
        metrics = {
            "samples": samples,
            "value_loss": loss,
            "replay_size": int(buf.size),
            "trained_examples": trained,
            "seconds_per_1e5_samples": elapsed / max(samples, 1) * 1e5,
            "gen_seconds_per_1e5_samples": gen_time / max(samples, 1) * 1e5,
            **{k: int(v) for k, v in jax.device_get(stats).items()},
        }
        if chunk_i % cfg["log_every_chunks"] == 0:
            per_step = steps_per_chunk * cfg["num_games"]
            print(
                f"samples {samples:9d}  loss {loss:.3e}  replay {int(buf.size):7d}  "
                f"{metrics['seconds_per_1e5_samples']:.1f}s/1e5 samples  per game-step: "
                f"solves {metrics['subgames_solved'] / per_step:.2f} "
                f"all-in {metrics['allin_chains'] / per_step:.2f} "
                f"hands done {metrics['hands_finished'] / per_step:.2f}"
            )
        if samples >= next_eval or samples >= num_samples:
            print(f"evaluation at {samples} samples ({elapsed / 60:.1f} min):")
            rng, k_eval = jax.random.split(rng)
            metrics.update(evaluate(params, k_eval, cfg["eval_deals"]))
            next_eval += float(cfg["eval_every"])
            if cfg["save_final"]:
                save_checkpoint(save_path, params, checkpoint_meta(game, cfg, samples))
        run.log(metrics)
    if cfg["save_final"]:
        save_checkpoint(save_path, params, checkpoint_meta(game, cfg, samples))
        print(f"Saved model to {save_path}")
    if cfg["final_eval_deals"]:
        print(f"final evaluation ({cfg['final_eval_deals']} deals, mirrored):")
        rng, k_eval = jax.random.split(rng)
        final = evaluate(params, k_eval, cfg["final_eval_deals"])
        run.log({f"final/{k[5:]}": v for k, v in final.items()})
    print(
        f"total {time.time() - t_start:.0f}s for {samples} samples "
        f"({(time.time() - t_start) / max(samples, 1) * 1e5:.1f}s per 1e5)"
    )
    return save_path
