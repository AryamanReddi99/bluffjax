"""
PPO-NFSP for Werewolf.

Neural Fictitious Self-Play (Heinrich & Silver 2016, arXiv:1603.01121) with
PPO as the best-response (BR) learner. All players share one BR network and
one average-policy network; the observation is relative to the player to act
and holds its role (and, for a werewolf, its teammates).

- Policy mixing: at the start of every game each player independently draws
  the policy it follows for the whole game, the BR with probability
  anticipatory_eta and the average policy otherwise.
- BR learning: PPO is on-policy, so the BR actor and critic train only on
  decisions made while following the BR. Returns follow each player's own
  decisions: a decision's reward is everything the player receives until its
  next decision or the end of the game (the env pays every player its team's
  outcome when the game ends, whoever moves), and it bootstraps from the value
  of that next decision (see per_player_gae). A dead player's last decision is
  credited with the game's outcome.
- Average policy: every BR decision (obs, action) is added to a reservoir
  buffer (Algorithm R) and the average policy is fit to it by cross-entropy.
  The average policy is NFSP's output strategy; the BR is a training device.
- Evaluation: every compare_interval updates and after the last one, the
  average policy and the BR each play compare_episodes complete games, the
  learner's policy controlling every player of one team and the opponent
  (uniformly random legal play or a checkpoint) every player of the other;
  the learner's team alternates over the games from a random one, so it plays
  the werewolves in half of them (see play_eval_games). Logged per policy: win, loss and draw (no winner at the horizon)
  rates, the win rate of the decided games, the return and the game length,
  and the win rates by the learner's team (a rate with no games behind it is
  NaN), as {stat}_vs_{random|baseline} for the average policy and
  {stat}_br_vs_{random|baseline} for the BR.
- Game metrics (bluffjax.metrics.werewolf: villager vote, elimination and
  accusation accuracy, seer information use, doctor save rate, votes received
  by werewolves and villagers) are counted at every rollout step and logged
  every update, as game/<rate> for the decisions of players following the
  average policy and game_br/<rate> for those following the BR (by the policy
  of the player each event is credited to), each with the number of events
  behind it (game/n_villager_votes, ...).

Units: the env pays +10 to the winning team and -10 to the losing one; logged
returns are in these units.

Run directory (bluffjax.examples.run_utils.make_run_dir): config.yaml,
results.jsonl (every logged metric, as sent to wandb) and, with
save_checkpoints, the half- and full-budget networks of every seed, see
save_checkpoints.
"""

import datetime
import time
from typing import Callable, NamedTuple

import distrax
from flax.training.train_state import TrainState
import hydra
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax
import wandb

from bluffjax.utils.typing import (
    Any,
    BoolArray,
    FloatArray,
    IntArray,
    PRNGKeyArray,
)
from bluffjax import make
from bluffjax.environments.werewolf.werewolf import WEREWOLF, WerewolfState
from bluffjax.examples.run_utils import (
    CHECKPOINT_STAGES,
    ResultsLog,
    half_update_steps,
    load_params,
    make_run_dir,
    save_seed_params,
)
from bluffjax.metrics import make_metrics, masked_totals, prefixed, rate
from bluffjax.networks.mlp import (
    ActorCriticDiscreteMLP,
    ActorDiscreteMLP,
    QNetworkDiscreteMLP,
)
from bluffjax.utils.jax_utils import pytree_norm
from bluffjax.utils.paths import REPO_ROOT, register_resolvers

WANDB_RUNS: list = []  # one wandb run per vmapped seed, created in main()
RESULTS_LOG: ResultsLog | None = None  # results.jsonl and wandb, created in main()
# EVALUATIONS[seed] = [(update, env_steps, {metric: value}), ...]
EVALUATIONS: dict[int, list[tuple[int, int, dict[str, float]]]] = {}


class Transition(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    br_log_prob: FloatArray  # log-prob of the action under the BR (BR decisions)
    reward: FloatArray  # (num_agents,) env reward for every player
    done: BoolArray  # the game ended at this step (the env auto-resets)
    value: FloatArray  # BR critic value of obs
    player_idx: IntArray
    is_br: BoolArray  # the acting player follows the BR in this game
    br_mode: BoolArray  # (num_agents,) which players follow the BR in this game
    roles: IntArray  # (num_agents,) roles in this game
    team_won: FloatArray  # (num_agents,) 1 if the player's team won at this step
    game_length: IntArray  # steps of the game so far, after this step
    game_counts: dict  # game metric events of this step, name -> (num_agents,)


class Checkpoint(NamedTuple):
    """The two NFSP networks at one point of training."""

    avg_params: Any  # average policy, NFSP's output strategy
    br_params: Any  # best response


class SLBufferState(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    seen: IntArray
    size: IntArray


class RunnerState(NamedTuple):
    br_train_state: TrainState
    avg_train_state: TrainState
    sl_buffer: SLBufferState
    state: WerewolfState
    obs: FloatArray
    done: BoolArray
    br_mode: BoolArray  # (num_envs, num_agents) policy drawn for the current game
    metric_carry: Any  # (num_envs, ...) game metric carry of the current game
    half: Checkpoint  # networks after half_update_steps updates
    update_step: IntArray
    rng: PRNGKeyArray


class PPOUpdateState(NamedTuple):
    train_state: TrainState
    transitions: Transition
    advantages: FloatArray
    targets: FloatArray
    train_mask: BoolArray
    rng: PRNGKeyArray


class SLBatch(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray


class SLUpdateState(NamedTuple):
    train_state: TrainState
    sl_buffer: SLBufferState
    rng: PRNGKeyArray


def _pick(arr: FloatArray, player: IntArray) -> FloatArray:
    """arr[n, player[n]] for an (N, num_agents) array."""
    return jnp.take_along_axis(arr, player[:, None], axis=-1).squeeze(-1)


def per_player_gae(
    values: FloatArray,
    rewards: FloatArray,
    dones: BoolArray,
    players: IntArray,
    last_value: FloatArray,
    last_player: IntArray,
    gamma: float,
    gae_lambda: float,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """GAE along each player's own decisions in a turn-based rollout.

    Shapes: T steps, N envs, P players.
        values: (T, N) critic value of obs_t for the player acting at step t.
        rewards: (T, N, P) reward vector returned by env.step at step t.
        dones: (T, N) the game ended at step t (the env then auto-resets).
        players: (T, N) player acting at step t.
        last_value: (N,) critic value of the observation after the last step.
        last_player: (N,) player to act after the last step.

    The reward of player p's decision at step t is the sum of p's rewards from
    step t up to p's next decision t' (so rewards paid on other players' steps
    are included), and the decision bootstraps from values[t']. If the game
    ends first the decision is terminal. At the rollout cut-off only the player
    to act has a next decision (bootstrap from last_value). The other players'
    trailing decisions in an unfinished game have no known outcome yet: they
    are marked invalid, and those players' earlier decisions truncate their
    lambda trace there (as GAE does at a cut-off).

    Returns:
        advantages (T, N), value targets (T, N), valid (T, N).
    """
    num_agents = rewards.shape[-1]
    last_onehot = jax.nn.one_hot(last_player, num_agents, dtype=jnp.bool_)
    zeros = jnp.zeros(last_onehot.shape, dtype=jnp.float32)
    # Per player, about its next decision (going backwards): value, advantage
    # (0 if the trace stops there), whether it is known, whether the game ends
    # before it, and the reward collected since the decision being processed.
    init = (
        jnp.where(last_onehot, last_value[:, None], 0.0),
        zeros,
        last_onehot,
        jnp.zeros_like(last_onehot),
        zeros,
    )

    def body(carry, x):
        next_value, next_adv, known, terminal, reward_acc = carry
        value, reward, done, player = x
        # Steps after a game-ending step belong to the next game.
        game_over = done[:, None]
        next_value = jnp.where(game_over, 0.0, next_value)
        next_adv = jnp.where(game_over, 0.0, next_adv)
        known = known | game_over
        terminal = terminal | game_over
        reward_acc = jnp.where(game_over, 0.0, reward_acc) + reward

        not_terminal = 1.0 - _pick(terminal, player).astype(jnp.float32)
        delta = (
            _pick(reward_acc, player)
            + gamma * not_terminal * _pick(next_value, player)
            - value
        )
        adv = delta + gamma * gae_lambda * not_terminal * _pick(next_adv, player)
        valid = _pick(known, player)
        adv = jnp.where(valid, adv, 0.0)

        acting = jax.nn.one_hot(player, num_agents, dtype=jnp.bool_)
        next_value = jnp.where(acting, value[:, None], next_value)
        next_adv = jnp.where(acting, adv[:, None], next_adv)
        known = known | acting
        terminal = terminal & ~acting
        reward_acc = jnp.where(acting, 0.0, reward_acc)
        return (next_value, next_adv, known, terminal, reward_acc), (adv, valid)

    _, (advantages, valid) = lax.scan(
        body, init, (values, rewards, dones, players), reverse=True
    )
    return advantages, advantages + values, valid


def reservoir_append(
    buffer: SLBufferState,
    obs: FloatArray,
    action_mask: BoolArray,
    action: IntArray,
    valid: BoolArray,
    rng: PRNGKeyArray,
) -> SLBufferState:
    """Adds the valid rows to the reservoir exactly as Algorithm R would when
    processing them one at a time in order.

    The k-th valid item of the whole stream (k from 0) goes to slot k while the
    buffer is not full, and otherwise replaces slot j ~ U{0, ..., k} if
    j < capacity. Whether and where an item is written doesn't depend on the
    buffer contents, so the batch is written with one scatter; when several
    items pick the same slot the latest one wins, as in the sequential version.
    """
    capacity = buffer.action.shape[0]
    valid_i = valid.astype(jnp.int32)
    k = buffer.seen + jnp.cumsum(valid_i) - valid_i
    j = jax.random.randint(rng, k.shape, 0, k + 1, dtype=jnp.int32)
    slot = jnp.where(k < capacity, k, j)
    write = valid & (slot < capacity)
    slot = jnp.where(write, slot, capacity)
    last_writer = (
        jnp.full((capacity,), -1, dtype=jnp.int32)
        .at[slot]
        .max(jnp.where(write, k, -1), mode="drop")
    )
    write = write & (last_writer[jnp.minimum(slot, capacity - 1)] == k)
    slot = jnp.where(write, slot, capacity)  # out of range: dropped
    num_new = valid_i.sum()
    return SLBufferState(
        obs=buffer.obs.at[slot].set(obs.astype(buffer.obs.dtype), mode="drop"),
        action_mask=buffer.action_mask.at[slot].set(action_mask, mode="drop"),
        action=buffer.action.at[slot].set(action, mode="drop"),
        seen=buffer.seen + num_new,
        size=jnp.minimum(capacity, buffer.size + num_new),
    )


# act(params, obs, action_mask, rng) -> action, for one observation
Policy = Callable[[Any, FloatArray, BoolArray, PRNGKeyArray], IntArray]


class EvalResult(NamedTuple):
    """Per game, from the learner's point of view."""

    win: FloatArray  # the learner's team won
    loss: FloatArray  # the other team won
    draw: FloatArray  # the env horizon ended the game without a winner
    ret: FloatArray  # the learner's team's total reward (+10, -10 or 0)
    length: FloatArray  # env steps
    werewolf: FloatArray  # the learner played the werewolves (else the humans)


def learner_team_mask(roles: IntArray, werewolf_team: BoolArray) -> BoolArray:
    """(num_agents,) the players the learner controls: every werewolf if it
    plays the werewolves, every other player if it plays the humans."""
    return (roles == WEREWOLF) == werewolf_team


def play_eval_games(
    env: Any,
    learner: Policy,
    learner_params: Any,
    baseline: Policy,
    baseline_params: Any,
    num_games: int,
    rng: PRNGKeyArray,
) -> EvalResult:
    """Plays num_games complete games in parallel, the learner controlling
    every player of one team and the baseline every player of the other.

    The learner's team alternates over the games, starting from a random one
    (game g: the werewolves if (offset + g) % 2 == 0, else the humans), so it
    plays each team in half of the games. Roles, the first speaker and the
    night order are dealt at random at reset, and the learner's policy acts
    for whichever players hold its team's roles. Every game is played with
    env.step_env (no auto-reset) until a team has won or the env horizon ends
    it without a winner. The env pays +10 to every player of the winning team
    and -10 to every player of the losing one when the game ends, so the
    learner's return (its team's) is a win if positive, a loss if negative
    and a draw (no winner) if zero.
    """
    rng_team, rng_games = jax.random.split(rng)
    offset = jax.random.randint(rng_team, (), 0, 2)
    werewolf_team = (offset + jnp.arange(num_games)) % 2 == 0

    def one_game(rng_game: PRNGKeyArray, werewolf_team: BoolArray):
        rng_reset, rng_play = jax.random.split(rng_game)
        state, obs = env.reset(rng_reset)
        on_team = learner_team_mask(state.roles, werewolf_team)  # roles fixed at reset
        member = jnp.argmax(on_team)  # a player of the learner's team

        def body(carry):
            state_s, obs_s, _, ret, length, rng_s = carry
            rng_s, rng_learner, rng_baseline, rng_step = jax.random.split(rng_s, 4)
            action_mask = env.get_avail_actions(state_s)
            action = jnp.where(
                on_team[state_s.current_player_idx],
                learner(learner_params, obs_s, action_mask, rng_learner),
                baseline(baseline_params, obs_s, action_mask, rng_baseline),
            )
            state_s, obs_s, reward, _, done, _ = env.step_env(rng_step, state_s, action)
            return state_s, obs_s, done, ret + reward[member], length + 1, rng_s

        init = (state, obs, jnp.bool_(False), jnp.float32(0.0), jnp.int32(0), rng_play)
        _, _, _, ret, length, _ = lax.while_loop(lambda c: ~c[2], body, init)
        return ret, length

    ret, length = jax.vmap(one_game)(
        jax.random.split(rng_games, num_games), werewolf_team
    )
    return EvalResult(
        win=(ret > 0).astype(jnp.float32),
        loss=(ret < 0).astype(jnp.float32),
        draw=(ret == 0).astype(jnp.float32),
        ret=ret,
        length=length.astype(jnp.float32),
        werewolf=werewolf_team.astype(jnp.float32),
    )


def eval_metrics(name: str, result: EvalResult) -> dict[str, FloatArray]:
    """Means over the evaluation games, as {stat}_{name}; win_rate_decided
    excludes draws, and win_rate_as_werewolf / win_rate_as_human are the win
    rates of the games in which the learner played that team
    (learner_werewolf_frac: the share of games it played the werewolves, 1/2
    for an even number of games). A rate with no games behind it (no decided
    game, or no game on that team) is NaN (bluffjax.metrics.rate). name is
    vs_{opponent} for the average policy and br_vs_{opponent} for the BR."""
    decided = result.win.sum() + result.loss.sum()
    werewolf, human = result.werewolf, 1.0 - result.werewolf
    return {
        f"win_rate_{name}": result.win.mean(),
        f"loss_rate_{name}": result.loss.mean(),
        f"draw_rate_{name}": result.draw.mean(),
        f"win_rate_decided_{name}": rate(result.win.sum(), decided),
        f"avg_return_{name}": result.ret.mean(),
        f"avg_episode_length_{name}": result.length.mean(),
        f"win_rate_as_werewolf_{name}": rate(
            (result.win * werewolf).sum(), werewolf.sum()
        ),
        f"win_rate_as_human_{name}": rate((result.win * human).sum(), human.sum()),
        f"learner_werewolf_frac_{name}": werewolf.mean(),
    }


def format_eval(evals: dict[str, float]) -> str:
    """One line for the policies evaluated, from eval_metrics' metrics."""
    names = sorted(k[len("avg_return_"):] for k in evals if k.startswith("avg_return_"))
    return "; ".join(
        f"{name} W/L/D {evals[f'win_rate_{name}']:.3f}/"
        f"{evals[f'loss_rate_{name}']:.3f}/{evals[f'draw_rate_{name}']:.3f} "
        f"(as werewolf {evals[f'win_rate_as_werewolf_{name}']:.3f}, "
        f"as human {evals[f'win_rate_as_human_{name}']:.3f}), "
        f"return {evals[f'avg_return_{name}']:+.2f}, "
        f"length {evals[f'avg_episode_length_{name}']:.1f}"
        for name in names
    )


def make_train(config: dict) -> Callable[..., RunnerState]:
    env = make("werewolf", **config["env_kwargs"])
    sample_state, _ = env.reset(jax.random.PRNGKey(0))
    action_dim = int(env.get_avail_actions(sample_state).shape[-1])
    num_agents = env.num_agents
    config["batch_shuffle_dim"] = (
        config["num_steps_per_env_per_update"] * config["num_envs"]
    )
    num_half = half_update_steps(config["num_update_steps"])
    game_metrics = make_metrics("werewolf", env)

    network = ActorCriticDiscreteMLP(
        action_dim=action_dim, hidden_dim=config["fc_dim_size"]
    )

    compare_enabled = config["compare"]
    compare_against_random = config["compare_mode"] == "random"
    compare_network_type = config["compare_network_type"]
    opponent = "random" if compare_against_random else "baseline"
    baseline_params = None
    baseline_network = None
    if compare_enabled and not compare_against_random:
        if compare_network_type == "actor_critic":
            baseline_network = ActorCriticDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )
        elif compare_network_type == "q_network":
            baseline_network = QNetworkDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )
        elif compare_network_type == "actor":
            baseline_network = ActorDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )
        else:
            raise ValueError(
                "compare_network_type must be one of "
                "['actor_critic', 'q_network', 'actor'], "
                f"got '{compare_network_type}'"
            )
        template = baseline_network.init(
            jax.random.PRNGKey(0), jnp.zeros((env.obs_dim,), dtype=jnp.float32)
        )
        baseline_params = load_params(config["compare_with"], template)
        print(f"Loaded {compare_network_type} baseline from {config['compare_with']}")

    def linear_decay(count: int) -> float:
        frac = 1.0 - count / config["num_gradient_steps"]
        return config["lr"] * frac

    def init_sl_buffer(obs_dim: int, action_dim: int) -> SLBufferState:
        capacity = config["sl_reservoir_capacity"]
        return SLBufferState(
            obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
            action_mask=jnp.zeros((capacity, action_dim), dtype=jnp.bool_),
            action=jnp.zeros((capacity,), dtype=jnp.int32),
            seen=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
        )

    def sample_sl_batch(
        rng: PRNGKeyArray,
        buffer: SLBufferState,
        batch_size: int,
    ) -> SLBatch:
        max_size = jnp.maximum(buffer.size, 1)
        indices = jax.random.randint(rng, (batch_size,), 0, max_size, dtype=jnp.int32)
        return SLBatch(
            obs=buffer.obs[indices],
            action_mask=buffer.action_mask[indices],
            action=buffer.action[indices],
        )

    def masked_mean(x: FloatArray, mask: FloatArray) -> FloatArray:
        denom = jnp.maximum(mask.sum(), 1.0)
        return (x * mask).sum() / denom

    # Evaluation policies: act(params, obs, action_mask, rng) -> action
    def act_actor_critic(net, params, obs, action_mask, rng):
        logits, _ = net.apply(params, obs)
        logits_masked = jnp.where(action_mask, logits, -jnp.inf)
        return distrax.Categorical(logits=logits_masked).sample(seed=rng)

    def act_actor(net, params, obs, action_mask, rng):
        logits = net.apply(params, obs)
        logits_masked = jnp.where(action_mask, logits, -jnp.inf)
        return distrax.Categorical(logits=logits_masked).sample(seed=rng)

    def act_q_greedy(net, params, obs, action_mask, rng):
        q_vals_masked = jnp.where(action_mask, net.apply(params, obs), -jnp.inf)
        ties = q_vals_masked == jnp.max(q_vals_masked)
        return jax.random.categorical(rng, jnp.where(ties, 0.0, -jnp.inf))

    def act_random(params, obs, action_mask, rng):
        return jax.random.categorical(rng, jnp.where(action_mask, 0.0, -jnp.inf))

    def learner_act(params, obs, action_mask, rng):
        return act_actor_critic(network, params, obs, action_mask, rng)

    if compare_against_random:
        baseline_act = act_random
    else:
        baseline_act_fn = {
            "actor_critic": act_actor_critic,
            "q_network": act_q_greedy,
            "actor": act_actor,
        }[compare_network_type]

        def baseline_act(params, obs, action_mask, rng):
            return baseline_act_fn(baseline_network, params, obs, action_mask, rng)

    def run_compare_eval(
        rng: PRNGKeyArray,
        avg_params: Any,
        br_params: Any,
        opponent_params: Any,
    ) -> dict[str, FloatArray]:
        """Evaluation metrics of the average policy ({stat}_vs_{opponent})
        and the BR ({stat}_br_vs_{opponent})."""
        rng_avg, rng_br = jax.random.split(rng)
        metrics = {}
        for prefix, act, params, rng_eval in (
            ("", learner_act, avg_params, rng_avg),
            ("br_", learner_act, br_params, rng_br),
        ):
            result = play_eval_games(
                env,
                act,
                params,
                baseline_act,
                opponent_params,
                config["compare_episodes"],
                rng_eval,
            )
            metrics.update(eval_metrics(f"{prefix}vs_{opponent}", result))
        return metrics

    def logging_callback(seed_val, update, evaluated, metric_dict) -> None:
        """Logs the metrics; the evaluation ones (keys with "_vs_") only when
        an evaluation ran."""
        seed_i, update = int(seed_val), int(update)
        metrics = {k: np.asarray(v) for k, v in metric_dict.items()}
        eval_keys = [k for k in metrics if "_vs_" in k]
        if bool(evaluated):
            evals = {k: float(metrics[k]) for k in eval_keys}
            env_steps = int(metrics["env_steps"])
            EVALUATIONS.setdefault(seed_i, []).append((update, env_steps, evals))
            print(
                f"seed {seed_i} update {update + 1}/{config['num_update_steps']} "
                f"({env_steps} env steps): {format_eval(evals)}",
                flush=True,
            )
        else:
            for k in eval_keys:
                del metrics[k]
        RESULTS_LOG.log(metrics, seed_i)

    def train(
        rng: PRNGKeyArray, seed: int, num_updates: int | None = None
    ) -> RunnerState:
        """Runs num_updates updates (default: all of them) of one seed. The
        schedules always span config["num_update_steps"] updates."""
        def train_setup(
            rng_inner: PRNGKeyArray,
        ) -> tuple[
            TrainState,
            TrainState,
            WerewolfState,
            FloatArray,
            SLBufferState,
        ]:
            rng_inner, rng_reset = jax.random.split(rng_inner)
            rng_resets = jax.random.split(rng_reset, config["num_envs"])
            state, obs = jax.vmap(env.reset, in_axes=(0))(rng_resets)

            rng_inner, rng_br_init, rng_avg_init = jax.random.split(rng_inner, 3)
            br_params = network.init(rng_br_init, obs)
            # The average policy uses the same actor-critic module so that both
            # networks can be evaluated the same way; only its actor is trained.
            avg_params = network.init(rng_avg_init, obs)

            br_lr = linear_decay if config["anneal_lr"] else config["lr"]
            br_tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=br_lr, eps=1e-5),
            )
            avg_tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=config["sl_lr"], eps=1e-5),
            )

            br_train_state = TrainState.create(
                apply_fn=network.apply, params=br_params, tx=br_tx
            )
            avg_train_state = TrainState.create(
                apply_fn=network.apply, params=avg_params, tx=avg_tx
            )
            sl_buffer = init_sl_buffer(obs.shape[-1], action_dim)
            return (br_train_state, avg_train_state, state, obs, sl_buffer)

        rng, rng_setup, rng_mode = jax.random.split(rng, 3)
        br_train_state, avg_train_state, state, obs, sl_buffer = train_setup(rng_setup)
        br_mode = jax.random.bernoulli(
            rng_mode,
            p=config["anticipatory_eta"],
            shape=(config["num_envs"], num_agents),
        )

        def update_step(
            runner_state: RunnerState,
            unused: None,
        ) -> tuple[RunnerState, None]:
            def step(
                runner_state_inner: RunnerState,
                unused: None,
            ) -> tuple[RunnerState, Transition]:
                br_train_state_s = runner_state_inner.br_train_state
                avg_train_state_s = runner_state_inner.avg_train_state
                state_s = runner_state_inner.state
                obs_s = runner_state_inner.obs
                rng_s = runner_state_inner.rng

                br_mode_s = runner_state_inner.br_mode
                rng_s, rng_mode, rng_action, rng_step = jax.random.split(rng_s, 4)

                br_logits, value = network.apply(br_train_state_s.params, obs_s)
                avg_logits, _ = network.apply(avg_train_state_s.params, obs_s)
                action_mask = jax.vmap(env.get_avail_actions, in_axes=(0))(state_s)

                br_logits_masked = jnp.where(action_mask, br_logits, -jnp.inf)
                avg_logits_masked = jnp.where(action_mask, avg_logits, -jnp.inf)

                # NFSP: the acting player follows the policy it drew for this game.
                player_idx = state_s.current_player_idx
                is_br = _pick(br_mode_s, player_idx)
                acting_logits = jnp.where(
                    is_br[:, None], br_logits_masked, avg_logits_masked
                )

                acting_pi = distrax.Categorical(logits=acting_logits)
                br_pi = distrax.Categorical(logits=br_logits_masked)
                action = acting_pi.sample(seed=rng_action)
                br_log_prob = jnp.where(is_br, br_pi.log_prob(action), 0.0)

                rng_steps = jax.random.split(rng_step, config["num_envs"])
                next_state, next_obs, reward, absorbing, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_steps, state_s, action)
                metric_carry, game_counts = jax.vmap(game_metrics.step)(
                    runner_state_inner.metric_carry,
                    state_s,
                    action,
                    next_state,
                    reward,
                    done,
                    info,
                )

                # A new game starts after done (the env auto-resets): every
                # player draws BR (prob. anticipatory_eta) or average policy.
                new_br_mode = jax.random.bernoulli(
                    rng_mode, p=config["anticipatory_eta"], shape=br_mode_s.shape
                )
                next_br_mode = jnp.where(done[:, None], new_br_mode, br_mode_s)

                transition = Transition(
                    obs=obs_s,
                    action_mask=action_mask,
                    action=action,
                    br_log_prob=br_log_prob,
                    reward=reward,
                    done=done,
                    value=value,
                    player_idx=player_idx,
                    is_br=is_br,
                    br_mode=br_mode_s,
                    roles=state_s.roles,
                    team_won=info["game_winner"],
                    game_length=info["timestep"],
                    game_counts=game_counts,
                )
                runner_state_inner = runner_state_inner._replace(
                    state=next_state,
                    obs=next_obs,
                    done=done,
                    br_mode=next_br_mode,
                    metric_carry=metric_carry,
                    rng=rng_s,
                )
                return runner_state_inner, transition

            runner_state, transitions = lax.scan(
                step,
                runner_state,
                None,
                config["num_steps_per_env_per_update"],
            )

            br_train_state = runner_state.br_train_state
            avg_train_state = runner_state.avg_train_state
            sl_buffer = runner_state.sl_buffer
            last_obs = runner_state.obs
            last_state = runner_state.state
            rng, rng_sl_append, rng_ppo_update, rng_sl_update, rng_compare = (
                jax.random.split(runner_state.rng, 5)
            )

            _, last_val = network.apply(br_train_state.params, last_obs)

            # Per-player GAE. PPO is on-policy: the BR actor and critic train on
            # BR decisions whose outcome is known.
            advantages, targets, valid = per_player_gae(
                transitions.value,
                transitions.reward,
                transitions.done,
                transitions.player_idx,
                last_val,
                last_state.current_player_idx,
                config["gamma"],
                config["gae_lambda"],
            )
            train_mask = valid & transitions.is_br

            # NFSP: every BR decision goes to the reservoir.
            sl_buffer = reservoir_append(
                sl_buffer,
                transitions.obs.reshape(-1, transitions.obs.shape[-1]),
                transitions.action_mask.reshape(-1, transitions.action_mask.shape[-1]),
                transitions.action.reshape(-1),
                transitions.is_br.reshape(-1),
                rng_sl_append,
            )

            def update_ppo_epoch(
                ppo_state: PPOUpdateState,
                unused: None,
            ) -> tuple[PPOUpdateState, dict[str, FloatArray]]:
                rng_s, rng_permute = jax.random.split(ppo_state.rng)
                batch = (
                    ppo_state.transitions,
                    ppo_state.advantages,
                    ppo_state.targets,
                    ppo_state.train_mask,
                )
                # (steps, envs, ...) -> (steps * envs, ...)
                batch_reshaped = jax.tree_util.tree_map(
                    lambda x: x.reshape((-1,) + x.shape[2:]), batch
                )
                permutation = jax.random.permutation(
                    rng_permute, config["batch_shuffle_dim"]
                )
                batch_shuffled = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0),
                    batch_reshaped,
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape(config["num_minibatches"], -1, *x.shape[1:]),
                    batch_shuffled,
                )

                def update_minibatch(
                    train_state: TrainState,
                    minibatch: tuple[Transition, FloatArray, FloatArray, BoolArray],
                ) -> tuple[TrainState, dict[str, FloatArray]]:
                    transitions_mb, advantages_mb, targets_mb, train_mask_mb = minibatch

                    def loss(
                        params,
                        transitions_mb,
                        advantages_mb,
                        targets_mb,
                        train_mask_mb,
                    ) -> tuple[FloatArray, dict[str, FloatArray]]:
                        logits, value = network.apply(params, transitions_mb.obs)
                        logits_masked = jnp.where(
                            transitions_mb.action_mask, logits, -jnp.inf
                        )
                        pi = distrax.Categorical(logits=logits_masked)
                        log_prob = pi.log_prob(transitions_mb.action)

                        # Only BR decisions with a known outcome enter the loss.
                        mask = train_mask_mb.astype(jnp.float32)
                        mask_denom = jnp.maximum(mask.sum(), 1.0)

                        mean_adv = masked_mean(advantages_mb, mask)
                        std_adv = jnp.sqrt(
                            masked_mean(jnp.square(advantages_mb - mean_adv), mask)
                        )
                        gae_normalized = (advantages_mb - mean_adv) / (std_adv + 1e-8)

                        logratio = jnp.where(
                            train_mask_mb, log_prob - transitions_mb.br_log_prob, 0.0
                        )
                        ratio = jnp.exp(logratio)
                        loss_actor_raw = ratio * gae_normalized
                        loss_actor_clipped = (
                            jnp.clip(
                                ratio,
                                1.0 - config["clip_eps"],
                                1.0 + config["clip_eps"],
                            )
                            * gae_normalized
                        )
                        actor_per_item = -jnp.minimum(
                            loss_actor_raw, loss_actor_clipped
                        )
                        loss_actor = (actor_per_item * mask).sum() / mask_denom

                        entropy = (pi.entropy() * mask).sum() / mask_denom

                        value_clipped = transitions_mb.value + jnp.clip(
                            value - transitions_mb.value,
                            -config["vf_clip"],
                            config["vf_clip"],
                        )
                        value_err = jnp.maximum(
                            jnp.square(value - targets_mb),
                            jnp.square(value_clipped - targets_mb),
                        )
                        value_loss = 0.5 * (value_err * mask).sum() / mask_denom

                        total_loss = (
                            loss_actor
                            + config["vf_coef"] * value_loss
                            - config["ent_coef"] * entropy
                        )

                        kl_backward = (
                            ((ratio - 1) - logratio) * mask
                        ).sum() / mask_denom
                        kl_forward = (
                            (ratio * logratio - (ratio - 1)) * mask
                        ).sum() / mask_denom
                        clip_frac = (
                            (jnp.abs(ratio - 1) > config["clip_eps"]).astype(
                                jnp.float32
                            )
                            * mask
                        ).sum() / mask_denom

                        return total_loss, {
                            "total_loss": total_loss,
                            "value_loss": value_loss,
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "ratio_mean": masked_mean(ratio, mask),
                            "ratio_min": jnp.where(train_mask_mb, ratio, jnp.inf).min(),
                            "ratio_max": jnp.where(
                                train_mask_mb, ratio, -jnp.inf
                            ).max(),
                            "gae_mean": mean_adv,
                            "gae_std": std_adv,
                            "mean_target": masked_mean(targets_mb, mask),
                            "value_pred_mean": masked_mean(value, mask),
                            "kl_backward": kl_backward,
                            "kl_forward": kl_forward,
                            "clip_frac": clip_frac,
                            "ppo_sample_frac_minibatch": mask.mean(),
                        }

                    grad_fn = jax.value_and_grad(loss, has_aux=True)
                    (_, aux), grads = grad_fn(
                        train_state.params,
                        transitions_mb,
                        advantages_mb,
                        targets_mb,
                        train_mask_mb,
                    )
                    aux["grad_norm"] = pytree_norm(grads)
                    updated_train_state = train_state.apply_gradients(grads=grads)
                    return updated_train_state, aux

                final_train_state, batch_stats = lax.scan(
                    update_minibatch,
                    ppo_state.train_state,
                    minibatches,
                )
                ppo_state = PPOUpdateState(
                    train_state=final_train_state,
                    transitions=ppo_state.transitions,
                    advantages=ppo_state.advantages,
                    targets=ppo_state.targets,
                    train_mask=ppo_state.train_mask,
                    rng=rng_s,
                )
                return ppo_state, batch_stats

            ppo_state = PPOUpdateState(
                train_state=br_train_state,
                # the game metric counts are not needed in the minibatches
                transitions=transitions._replace(game_counts={}),
                advantages=advantages,
                targets=targets,
                train_mask=train_mask,
                rng=rng_ppo_update,
            )
            final_ppo_state, ppo_loss_info = lax.scan(
                update_ppo_epoch,
                ppo_state,
                None,
                config["num_epochs"],
            )
            br_train_state = final_ppo_state.train_state
            ppo_loss_info = jax.tree_util.tree_map(lambda x: x.mean(), ppo_loss_info)

            def update_sl_step(
                sl_state: SLUpdateState,
                unused: None,
            ) -> tuple[SLUpdateState, dict[str, FloatArray]]:
                rng_s, rng_batch = jax.random.split(sl_state.rng)
                has_data = sl_state.sl_buffer.size > 0

                def _train_step(train_state: TrainState):
                    batch = sample_sl_batch(
                        rng_batch,
                        sl_state.sl_buffer,
                        config["sl_batch_size"],
                    )

                    def sl_loss(params) -> tuple[FloatArray, dict[str, FloatArray]]:
                        logits, _ = network.apply(params, batch.obs)
                        logits_masked = jnp.where(batch.action_mask, logits, -jnp.inf)
                        log_probs = jax.nn.log_softmax(logits_masked, axis=-1)
                        action_log_probs = jnp.take_along_axis(
                            log_probs,
                            batch.action[:, None],
                            axis=-1,
                        ).squeeze(-1)
                        ce_loss = -action_log_probs.mean()
                        acc = (
                            jnp.argmax(logits_masked, axis=-1) == batch.action
                        ).mean()
                        return ce_loss, {
                            "sl_loss": ce_loss,
                            "sl_acc": acc,
                        }

                    grad_fn = jax.value_and_grad(sl_loss, has_aux=True)
                    (_, aux), grads = grad_fn(train_state.params)
                    aux["sl_grad_norm"] = pytree_norm(grads)
                    new_train_state = train_state.apply_gradients(grads=grads)
                    return new_train_state, aux

                def _skip_step(train_state: TrainState):
                    return train_state, {
                        "sl_loss": jnp.array(0.0, dtype=jnp.float32),
                        "sl_acc": jnp.array(0.0, dtype=jnp.float32),
                        "sl_grad_norm": jnp.array(0.0, dtype=jnp.float32),
                    }

                train_state, aux = lax.cond(
                    has_data,
                    _train_step,
                    _skip_step,
                    sl_state.train_state,
                )
                sl_state = SLUpdateState(
                    train_state=train_state,
                    sl_buffer=sl_state.sl_buffer,
                    rng=rng_s,
                )
                return sl_state, aux

            sl_state = SLUpdateState(
                train_state=avg_train_state,
                sl_buffer=sl_buffer,
                rng=rng_sl_update,
            )
            sl_state, sl_loss_info = lax.scan(
                update_sl_step,
                sl_state,
                None,
                config["sl_num_steps_per_update"],
            )
            avg_train_state = sl_state.train_state
            sl_loss_info = jax.tree_util.tree_map(lambda x: x.mean(), sl_loss_info)

            # Self-play metrics over the games that ended in this rollout, per
            # player, in env units.
            game_end = transitions.done[..., None]
            werewolf = transitions.roles == WEREWOLF
            ended_ww = game_end & werewolf
            ended_human = game_end & ~werewolf
            br_games = game_end & transitions.br_mode
            avg_games = game_end & ~transitions.br_mode
            games = jnp.maximum(transitions.done.sum(), 1)

            def per_game(x: FloatArray, mask: BoolArray) -> FloatArray:
                return (x * mask).sum() / jnp.maximum(mask.sum(), 1)

            metric = {
                "update_step": runner_state.update_step,
                "env_steps": (runner_state.update_step + 1)
                * config["num_envs"]
                * config["num_steps_per_env_per_update"],
                "episodes_completed": transitions.done.sum(),
                "ep_length_avg": jnp.where(
                    transitions.done, transitions.game_length, 0
                ).sum()
                / games,
                "win_rate_human_team_rollout": per_game(
                    transitions.team_won, ended_human
                ),
                "win_rate_werewolf_team_rollout": per_game(
                    transitions.team_won, ended_ww
                ),
                "returns_human_team_avg": per_game(transitions.reward, ended_human),
                "returns_werewolf_team_avg": per_game(transitions.reward, ended_ww),
                # Against the mixed opponents the BR should win more often.
                "return_per_episode": per_game(transitions.reward, avg_games),
                "return_per_episode_br": per_game(transitions.reward, br_games),
                "br_action_frac_rollout": transitions.is_br.mean(),
                "ppo_sample_frac": train_mask.mean(),
                "sl_buffer_size": sl_buffer.size.astype(jnp.float32),
                "sl_buffer_seen": sl_buffer.seen.astype(jnp.float32),
                "br_lr": (
                    linear_decay(br_train_state.step)
                    if config["anneal_lr"]
                    else jnp.asarray(config["lr"])
                ),
            }
            # Game metrics, by the policy of the player each event is credited to
            metric.update(
                prefixed(
                    "game",
                    game_metrics.summarize(
                        masked_totals(transitions.game_counts, ~transitions.br_mode)
                    ),
                )
            )
            metric.update(
                prefixed(
                    "game_br",
                    game_metrics.summarize(
                        masked_totals(transitions.game_counts, transitions.br_mode)
                    ),
                )
            )
            metric.update(ppo_loss_info)
            metric.update(sl_loss_info)

            should_compare = compare_enabled & (
                (runner_state.update_step % config["compare_interval"] == 0)
                | (runner_state.update_step == config["num_update_steps"] - 1)
            )
            opponent_params = (
                avg_train_state.params if compare_against_random else baseline_params
            )

            def do_compare(rng_c):
                return run_compare_eval(
                    rng_c,
                    avg_train_state.params,
                    br_train_state.params,
                    opponent_params,
                )

            if compare_enabled:
                compare_metrics = lax.cond(
                    should_compare,
                    do_compare,
                    lambda rng_c: jax.tree_util.tree_map(
                        lambda x: jnp.zeros(x.shape, x.dtype),
                        jax.eval_shape(do_compare, rng_c),
                    ),
                    rng_compare,
                )
                metric.update(compare_metrics)

            jax.experimental.io_callback(
                logging_callback,
                None,
                seed,
                runner_state.update_step,
                should_compare,
                metric,
            )

            # Half-budget checkpoint: the networks after num_half updates.
            take_half = runner_state.update_step + 1 == num_half
            half = jax.tree_util.tree_map(
                lambda new, old: jnp.where(take_half, new, old),
                Checkpoint(avg_train_state.params, br_train_state.params),
                runner_state.half,
            )
            runner_state = runner_state._replace(
                br_train_state=br_train_state,
                avg_train_state=avg_train_state,
                sl_buffer=sl_buffer,
                half=half,
                update_step=runner_state.update_step + 1,
                rng=rng,
            )
            return runner_state, None

        initial_runner_state = RunnerState(
            br_train_state=br_train_state,
            avg_train_state=avg_train_state,
            sl_buffer=sl_buffer,
            state=state,
            obs=obs,
            done=jnp.zeros((config["num_envs"]), dtype=jnp.bool_),
            br_mode=br_mode,
            metric_carry=jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (config["num_envs"],) + jnp.shape(x)),
                game_metrics.init_carry(),
            ),
            # replaced after num_half updates
            half=Checkpoint(avg_train_state.params, br_train_state.params),
            update_step=jnp.array(0, dtype=jnp.int32),
            rng=rng,
        )
        final_runner_state, _ = lax.scan(
            update_step,
            initial_runner_state,
            None,
            config["num_update_steps"] if num_updates is None else num_updates,
        )
        return final_runner_state

    return train


def save_checkpoints(
    run_dir: str, num_seeds: int, half: Checkpoint, full: Checkpoint
) -> str:
    """Saves the networks of every seed at half and at the full budget.

    Layout of the run directory (bluffjax.examples.run_utils.make_run_dir):
        config.yaml                  resolved config of the run
        results.jsonl                everything logged (see run_utils)
        half/avg_policy_{i}.msgpack  after half_update_steps(num_update_steps)
        half/br_{i}.msgpack          (= num_update_steps // 2) updates
        full/avg_policy_{i}.msgpack  after all num_update_steps updates
        full/br_{i}.msgpack
    Both checkpoints come from the same run: the half one is the state after
    exactly that many updates, with the learning-rate schedule running over
    the whole run (config.yaml records num_update_steps and
    half_update_steps). i = 0..num_seeds-1 indexes the vmapped seeds (wandb
    run "{seed}_{i}"). Each file holds the flax params of one network, without
    a seed axis, written with flax.serialization.to_bytes. Both networks are
    ActorCriticDiscreteMLP(action_dim, fc_dim_size) mapping the observation of
    the player to act (env.obs_from_state(state), env.obs_dim floats) to
    (action logits, value in env reward units); action_dim is
    env.num_actions. The value head of the average policy is unused. Load one
    with run_utils.load_params (which checks the tree, shapes and dtypes) and
    a template network.init(rng, obs), and mask illegal actions
    (env.get_avail_actions) before the softmax. Use either file as an evaluation opponent with
    compare_mode=checkpoint compare_network_type=actor_critic.
    """
    for stage, checkpoint in zip(CHECKPOINT_STAGES, (half, full)):
        for name, params in (
            ("avg_policy", checkpoint.avg_params),
            ("br", checkpoint.br_params),
        ):
            save_seed_params(run_dir, stage, name, params, range(num_seeds))
    return run_dir


def print_evaluation_summary() -> None:
    """Last evaluation of every seed, and the mean over seeds."""
    if not EVALUATIONS:
        return
    finals = [EVALUATIONS[s][-1][2] for s in sorted(EVALUATIONS)]
    for key in finals[0]:
        values = np.array([f[key] for f in finals])
        print(
            f"final {key}: mean {values.mean():.3f} std {values.std():.3f} "
            f"over {len(values)} seed(s): {np.round(values, 3).tolist()}"
        )


@hydra.main(version_base=None, config_path="./", config_name="config_ppo_nfsp")
def main(config: dict) -> None:
    global WANDB_RUNS, RESULTS_LOG
    try:
        config = OmegaConf.to_container(config, resolve=True)
        config["num_update_steps"] = int(
            config["num_timesteps"]
            // config["num_envs"]
            // config["num_steps_per_env_per_update"]
        )
        config["half_update_steps"] = half_update_steps(config["num_update_steps"])
        config["num_gradient_steps"] = (
            config["num_update_steps"]
            * config["num_epochs"]
            * config["num_minibatches"]
        )
        env_steps = (
            config["num_update_steps"]
            * config["num_envs"]
            * config["num_steps_per_env_per_update"]
        )

        train = make_train(config)  # completes config (batch_shuffle_dim)

        # The run directory and the results log come first, so that
        # results.jsonl fills while the run trains (wall_time counts from
        # here, compilation included).
        run_dir = make_run_dir(config)
        job_type = f"{config['job_type']}_{config['env_name']}"
        group = f"{config['env_name']}" + datetime.datetime.now().strftime(
            "_%Y-%m-%d_%H-%M-%S"
        )
        WANDB_RUNS = [
            wandb.init(
                project=config["project"],
                group=group,
                job_type=job_type,
                name=f"{config['seed']}_{i}",
                config=config,
                mode=None if config["wandb"] else "disabled",
                dir=str(REPO_ROOT),
                reinit="create_new",
            )
            for i in range(config["num_seeds"])
        ]
        RESULTS_LOG = ResultsLog(run_dir, WANDB_RUNS)
        print(f"Run directory: {run_dir}")

        rng = jax.random.PRNGKey(config["seed"])
        rng_seeds = jax.random.split(rng, config["num_seeds"])
        exp_ids = jnp.arange(config["num_seeds"])

        print("Starting compile...")
        start = time.time()
        train_vjit = jax.jit(jax.vmap(train)).lower(rng_seeds, exp_ids).compile()
        print(f"Compile finished in {time.time() - start:.1f} s")

        print("Running...")
        start = time.time()
        final_runner_state = jax.block_until_ready(train_vjit(rng_seeds, exp_ids))
        jax.effects_barrier()  # every update's record is in results.jsonl
        run_time = time.time() - start
        print(
            f"Trained {config['num_seeds']} seed(s) x {env_steps} env steps in "
            f"{run_time:.1f} s ({run_time / env_steps * 1e6:.2f} s per 1e6 env "
            f"steps, all seeds in parallel)"
        )
        print_evaluation_summary()
        if config["save_checkpoints"]:
            save_checkpoints(
                run_dir,
                config["num_seeds"],
                final_runner_state.half,
                Checkpoint(
                    final_runner_state.avg_train_state.params,
                    final_runner_state.br_train_state.params,
                ),
            )
            print(f"Saved half- and full-budget checkpoints to {run_dir}")
    finally:
        for run in WANDB_RUNS:
            run.finish()
        print("Finished.")


if __name__ == "__main__":
    register_resolvers()
    main()
