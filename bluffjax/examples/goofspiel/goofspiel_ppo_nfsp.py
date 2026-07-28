"""
PPO + NFSP for Goofspiel (Parallel / simultaneous-move).

Unlike the AEC (turn-based) games (e.g. bluff), Goofspiel is a ParallelEnv:
every agent acts every step, so action/reward/is_br/value all carry an extra
num_agents axis: (num_envs, num_agents). There is no current_player_idx to
gather/scatter around, and info is always {} (no game_winner field) -- the
only per-game-outcome signal is state.winners (a (num_agents,) bool array,
only meaningful when done=True, and NEVER all-False: either exactly one agent
is the strict winner, or both agents tie for max cumulative points).
"""

import datetime
import os
from typing import Callable, NamedTuple

import distrax
from flax import serialization
from flax.training.train_state import TrainState
import hydra
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax

from bluffjax.utils.typing import (
    Any,
    BoolArray,
    FloatArray,
    IntArray,
    PRNGKeyArray,
)
from bluffjax import make
from bluffjax.environments.goofspiel.goofspiel import GoofspielState
from bluffjax.networks.mlp import (
    ActorCriticDiscreteMLP,
    ActorDiscreteMLP,
    QNetworkDiscreteMLP,
)
from bluffjax.utils.jax_utils import pytree_norm
from bluffjax.utils.wandb_multilogger import WandbMultiLogger

LOGGER = None


class Transition(NamedTuple):
    obs: FloatArray  # (num_envs, num_agents, obs_dim)
    action_mask: BoolArray  # (num_envs, num_agents, num_actions)
    action: IntArray  # (num_envs, num_agents)
    br_log_prob: FloatArray  # (num_envs, num_agents)
    reward: FloatArray  # (num_envs, num_agents) - this round's prize
    absorbing: BoolArray  # (num_envs, num_agents)
    done: BoolArray  # (num_envs,)
    value: FloatArray  # (num_envs, num_agents)
    is_br: BoolArray  # (num_envs, num_agents)
    winners: BoolArray  # (num_envs, num_agents) - only meaningful when done
    points: FloatArray  # (num_envs, num_agents) - cumulative pts, meaningful when done
    info: dict[str, Any]


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
    state: GoofspielState
    obs: FloatArray
    done: BoolArray
    update_step: IntArray
    rng: PRNGKeyArray


class PPOUpdateState(NamedTuple):
    train_state: TrainState
    transitions: Transition
    advantages: FloatArray
    targets: FloatArray
    rng: PRNGKeyArray


class SLBatch(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray


class SLUpdateState(NamedTuple):
    train_state: TrainState
    sl_buffer: SLBufferState
    rng: PRNGKeyArray


def make_train(config: dict) -> Callable[[PRNGKeyArray, int], RunnerState]:
    env = make(config["env_name"], **config["env_kwargs"])
    sample_state, sample_obs = env.reset(jax.random.PRNGKey(0))
    action_dim = int(env.get_avail_actions(sample_state).shape[-1])
    config["batch_shuffle_dim"] = (
        config["num_steps_per_env_per_update"] * config["num_envs"]
    )
    config["num_update_steps"] = int(config["num_timesteps"]) // config["num_envs"] // config["num_steps_per_env_per_update"]
    config["num_gradient_steps"] = int(config["num_update_steps"]* config["num_epochs"]* config["num_minibatches"])

    num_update_steps = config["num_update_steps"]
    checkpoint_step_to_fracs: dict[int, list[float]] = {}
    for frac in config.get("save_checkpoints", []):
        step = min(max(round(frac * num_update_steps) - 1, 0), num_update_steps - 1)
        checkpoint_step_to_fracs.setdefault(step, []).append(frac)
    checkpoint_steps = sorted(checkpoint_step_to_fracs)

    def checkpoint_callback(
        seed_val: int, update_step_val: int, br_params: Any, avg_params: Any
    ) -> None:
        seed_int = int(seed_val)
        run_dir = os.path.join(
            config["save_dir"],
            f"{config['algorithm']}_{config['group']}_s{seed_int}",
        )
        os.makedirs(run_dir, exist_ok=True)
        for frac in checkpoint_step_to_fracs.get(int(update_step_val), []):
            frac_tag = f"{frac:g}".replace(".", "p")
            for name, params in (("br", br_params), ("avg", avg_params)):
                path = os.path.join(run_dir, f"{name}_{frac_tag}.msgpack")
                with open(path, "wb") as f:
                    f.write(serialization.to_bytes(params))
            print(
                f"[seed {seed_int}] Saved checkpoint at "
                f"{frac * 100:.1f}% training progress to {run_dir}/{{br,avg}}_{frac_tag}.msgpack"
            )

    network = ActorCriticDiscreteMLP(
        action_dim=action_dim, hidden_dim=config["fc_dim_size"]
    )

    def env_step_with_raw(
        rng: PRNGKeyArray,
        state: GoofspielState,
        action: IntArray,
    ) -> tuple[GoofspielState, FloatArray, FloatArray, BoolArray, bool, dict[str, Any], GoofspielState]:
        """Like env.step, but also returns the raw pre-auto-reset state.

        ParallelEnv.step() swaps `state`/`obs` for the reset state/obs
        whenever done=True, which means state.winners/state.points on the
        *returned* state are the RESET state's zeroed values at exactly the
        one moment we need the real terminal values. Goofspiel's info dict
        is always {} (unlike bluff's info["game_winner"]), so we can't rely
        on info to recover the terminal outcome -- we must capture the raw
        state from step_env before the auto-reset swap.
        """
        rng_step, rng_reset = jax.random.split(rng)
        state_reset, obs_reset = env.reset(rng_reset)
        raw_state, raw_obs, reward, absorbing, done, info = env.step_env(
            rng_step, state, action
        )
        final_state, final_obs = lax.cond(
            done, lambda: (state_reset, obs_reset), lambda: (raw_state, raw_obs)
        )
        return final_state, final_obs, reward, absorbing, done, info, raw_state

    compare_mode = config["compare_mode"]
    compare_against_random = compare_mode == "random"
    compare_enabled = config["compare"]
    if not compare_enabled:
        compare_against_random = True
    compare_network_type = config["compare_network_type"]
    baseline_params = None
    baseline_actor_critic_network = None
    baseline_q_network = None
    baseline_actor_network = None
    if compare_enabled and (not compare_against_random):
        checkpoint_path = config["compare_with"]
        template_obs = jnp.zeros_like(sample_obs)

        def _squeeze_leading_batch(x):
            if hasattr(x, "shape") and x.ndim > 0 and x.shape[0] == 1:
                return jnp.squeeze(x, axis=0)
            return x

        if compare_network_type == "actor_critic":
            baseline_actor_critic_network = ActorCriticDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )
            template_params = baseline_actor_critic_network.init(
                jax.random.PRNGKey(0), template_obs
            )
            with open(checkpoint_path, "rb") as f:
                baseline_params = serialization.from_bytes(template_params, f.read())
            baseline_params = jax.tree_util.tree_map(
                _squeeze_leading_batch, baseline_params
            )
            print(f"Loaded actor-critic baseline from {checkpoint_path}")
        elif compare_network_type == "q_network":
            baseline_q_network = QNetworkDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )
            template_params = baseline_q_network.init(
                jax.random.PRNGKey(0), template_obs
            )
            with open(checkpoint_path, "rb") as f:
                baseline_params = serialization.from_bytes(template_params, f.read())
            baseline_params = jax.tree_util.tree_map(
                _squeeze_leading_batch, baseline_params
            )
            print(f"Loaded q-network baseline from {checkpoint_path}")
        elif compare_network_type == "actor":
            baseline_actor_network = ActorDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )
            template_params = baseline_actor_network.init(
                jax.random.PRNGKey(0), template_obs
            )
            with open(checkpoint_path, "rb") as f:
                baseline_params = serialization.from_bytes(template_params, f.read())
            baseline_params = jax.tree_util.tree_map(
                _squeeze_leading_batch, baseline_params
            )
            print(f"Loaded actor baseline from {checkpoint_path}")
        else:
            raise ValueError(
                "compare_network_type must be one of "
                "['actor_critic', 'q_network', 'actor'], "
                f"got '{compare_network_type}'"
            )

    def linear_decay(count: int) -> float:
        frac = 1.0 - (count // config["num_gradient_steps"])
        return config["lr"] * frac

    def wandb_callback(exp_id: int, metrics: dict, info: dict) -> None:
        if LOGGER is None:
            return
        metrics.update(info)
        np_log_dict = {k: np.array(v) for k, v in metrics.items()}
        LOGGER.log(int(exp_id), np_log_dict)

    def init_sl_buffer(obs_dim: int, action_dim_local: int) -> SLBufferState:
        capacity = config["sl_reservoir_capacity"]
        return SLBufferState(
            obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
            action_mask=jnp.zeros((capacity, action_dim_local), dtype=jnp.bool_),
            action=jnp.zeros((capacity,), dtype=jnp.int32),
            seen=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
        )

    def append_sl_samples(
        buffer: SLBufferState,
        obs_batch: FloatArray,
        action_mask_batch: BoolArray,
        action_batch: IntArray,
        valid_batch: BoolArray,
        rng: PRNGKeyArray,
    ) -> tuple[SLBufferState, PRNGKeyArray]:
        capacity = config["sl_reservoir_capacity"]

        def add_one(carry, sample):
            buf, rng_inner = carry
            obs_s, action_mask_s, action_s, valid_s = sample
            rng_inner, rng_i = jax.random.split(rng_inner)

            def _add(cur: SLBufferState) -> SLBufferState:
                k = cur.seen
                j = jax.random.randint(rng_i, (), 0, k + 1, dtype=jnp.int32)
                not_full = k < capacity
                write_idx = jnp.where(not_full, k, j)
                should_write = not_full | (j < capacity)

                def _write(b: SLBufferState) -> SLBufferState:
                    return SLBufferState(
                        obs=b.obs.at[write_idx].set(obs_s),
                        action_mask=b.action_mask.at[write_idx].set(action_mask_s),
                        action=b.action.at[write_idx].set(action_s),
                        seen=b.seen,
                        size=b.size,
                    )

                cur = lax.cond(should_write, _write, lambda b: b, cur)
                return SLBufferState(
                    obs=cur.obs,
                    action_mask=cur.action_mask,
                    action=cur.action,
                    seen=cur.seen + 1,
                    size=jnp.minimum(capacity, cur.size + 1),
                )

            buf = lax.cond(valid_s, _add, lambda b: b, buf)
            return (buf, rng_inner), None

        samples = (obs_batch, action_mask_batch, action_batch, valid_batch)
        (buffer, rng), _ = lax.scan(add_one, (buffer, rng), samples)
        return buffer, rng

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

    def sample_masked_action(
        params: Any,
        obs: FloatArray,
        action_mask: BoolArray,
        rng: PRNGKeyArray,
    ) -> IntArray:
        logits, _ = network.apply(params, obs)
        logits_masked = jnp.where(action_mask, logits, -jnp.inf)
        return distrax.Categorical(logits=logits_masked).sample(seed=rng)

    def sample_random_legal_action(
        action_mask: BoolArray, rng: PRNGKeyArray
    ) -> IntArray:
        probs = action_mask.astype(jnp.float32)
        probs = probs / jnp.maximum(probs.sum(), 1.0)
        return distrax.Categorical(probs=probs).sample(seed=rng)

    def sample_action_q_greedy(
        q_network: QNetworkDiscreteMLP,
        params: Any,
        obs: FloatArray,
        action_mask: BoolArray,
        rng: PRNGKeyArray,
    ) -> IntArray:
        q_vals = q_network.apply(params, obs.astype(jnp.float32))
        q_vals_masked = jnp.where(action_mask, q_vals, -jnp.inf)
        best_val = jnp.max(q_vals_masked)
        ties = (q_vals_masked == best_val) & action_mask
        logits = jnp.where(ties, 0.0, -1e9)
        return jax.random.categorical(rng, logits)

    def sample_action_actor(
        actor_network: ActorDiscreteMLP,
        params: Any,
        obs: FloatArray,
        action_mask: BoolArray,
        rng: PRNGKeyArray,
    ) -> IntArray:
        logits = actor_network.apply(params, obs.astype(jnp.float32))
        logits_masked = jnp.where(action_mask, logits, -jnp.inf)
        return distrax.Categorical(logits=logits_masked).sample(seed=rng)

    def sample_baseline_action(
        baseline_params: Any,
        obs: FloatArray,
        action_mask: BoolArray,
        rng: PRNGKeyArray,
    ) -> IntArray:
        if compare_against_random:
            return sample_random_legal_action(action_mask, rng)
        if compare_network_type == "actor_critic":
            return sample_masked_action(baseline_params, obs, action_mask, rng)
        if compare_network_type == "q_network":
            return sample_action_q_greedy(
                baseline_q_network, baseline_params, obs, action_mask, rng
            )
        return sample_action_actor(
            baseline_actor_network, baseline_params, obs, action_mask, rng
        )

    def compare_single_episode(
        rng: PRNGKeyArray,
        train_params: Any,
        baseline_params: Any,
    ) -> tuple[PRNGKeyArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Seat 0 = train, seat 1 = baseline."""
        rng, rng_reset = jax.random.split(rng)
        state, obs = env.reset(rng_reset)

        def play_cond(carry):
            _, _, _, _, done_flag, _, _ = carry
            return ~done_flag

        def play_body(carry):
            (
                state_s,
                obs_s,
                rng_s,
                step_count,
                _,
                terminal_reward,
                terminal_winners,
            ) = carry
            rng_s, rng_train, rng_base, rng_step = jax.random.split(rng_s, 4)
            action_mask = env.get_avail_actions(state_s)  # (2, 13)

            train_action = sample_masked_action(
                train_params, obs_s[0], action_mask[0], rng_train
            )
            baseline_action = sample_baseline_action(
                baseline_params, obs_s[1], action_mask[1], rng_base
            )
            joint_action = jnp.stack([train_action, baseline_action])  # (2,)

            next_state, next_obs, reward, _, done, info, raw_state = env_step_with_raw(
                rng_step, state_s, joint_action
            )
            new_terminal_reward = jnp.where(done, reward, terminal_reward)
            new_terminal_winners = jnp.where(done, raw_state.winners, terminal_winners)
            return (
                next_state,
                next_obs,
                rng_s,
                step_count + 1,
                done,
                new_terminal_reward,
                new_terminal_winners,
            )

        (
            _,
            _,
            rng,
            step_count,
            done_flag,
            terminal_reward,
            terminal_winners,
        ) = lax.while_loop(
            play_cond,
            play_body,
            (
                state,
                obs,
                rng,
                jnp.array(0, dtype=jnp.int32),
                jnp.bool_(False),
                jnp.zeros(env.num_agents, dtype=jnp.float32),
                jnp.zeros(env.num_agents, dtype=jnp.bool_),
            ),
        )
        train_return = terminal_reward[0]
        win = terminal_winners[0] & (~terminal_winners[1])
        tie = terminal_winners[0] & terminal_winners[1]
        return (
            rng,
            train_return,
            step_count.astype(jnp.float32),
            done_flag.astype(jnp.float32),
            win.astype(jnp.float32),
            tie.astype(jnp.float32),
        )

    def run_compare_eval(
        rng: PRNGKeyArray,
        avg_params: Any,
        br_params: Any,
        checkpoint_params: Any,
    ) -> tuple[
        PRNGKeyArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
    ]:
        def compare_episode(carry, episode_idx):
            rng_s, avg_p, br_p, base_p = carry
            _ = episode_idx

            rng_s, avg_ret, avg_len, avg_done, avg_win, avg_tie = (
                compare_single_episode(rng_s, avg_p, base_p)
            )
            rng_s, br_ret, br_len, br_done, br_win, br_tie = compare_single_episode(
                rng_s, br_p, base_p
            )
            return (rng_s, avg_p, br_p, base_p), (
                avg_ret,
                br_ret,
                avg_len,
                br_len,
                avg_done,
                br_done,
                avg_win,
                br_win,
                avg_tie,
                br_tie,
            )

        (rng, _, _, _), (
            avg_ret_arr,
            br_ret_arr,
            avg_len_arr,
            br_len_arr,
            avg_done_arr,
            br_done_arr,
            avg_win_arr,
            br_win_arr,
            avg_tie_arr,
            br_tie_arr,
        ) = lax.scan(
            compare_episode,
            (rng, avg_params, br_params, checkpoint_params),
            jnp.arange(config["compare_episodes"]),
        )
        avg_done_count = jnp.maximum(avg_done_arr.sum(), 1.0)
        br_done_count = jnp.maximum(br_done_arr.sum(), 1.0)
        avg_ret = avg_ret_arr.sum() / avg_done_count
        br_ret = br_ret_arr.sum() / br_done_count
        avg_win_rate = avg_win_arr.sum() / avg_done_count
        br_win_rate = br_win_arr.sum() / br_done_count
        avg_tie_rate = avg_tie_arr.sum() / avg_done_count
        br_tie_rate = br_tie_arr.sum() / br_done_count
        avg_eval_len = (
            avg_len_arr.sum() / avg_done_count + br_len_arr.sum() / br_done_count
        ) / 2
        return (
            rng,
            avg_ret,
            br_ret,
            avg_eval_len,
            avg_win_rate,
            br_win_rate,
            avg_tie_rate,
            br_tie_rate,
        )

    def train(rng: PRNGKeyArray, seed: int) -> RunnerState:
        def train_setup(
            rng_inner: PRNGKeyArray,
        ) -> tuple[
            TrainState,
            TrainState,
            GoofspielState,
            FloatArray,
            SLBufferState,
        ]:
            rng_inner, rng_reset = jax.random.split(rng_inner)
            rng_resets = jax.random.split(rng_reset, config["num_envs"])
            state, obs = jax.vmap(env.reset, in_axes=(0))(rng_resets)

            rng_inner, rng_network_init = jax.random.split(rng_inner)
            network_params = network.init(rng_network_init, obs)

            br_tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=linear_decay, eps=1e-5),
            )
            avg_tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=config["sl_lr"], eps=1e-5),
            )

            br_train_state = TrainState.create(
                apply_fn=network.apply, params=network_params, tx=br_tx
            )
            avg_train_state = TrainState.create(
                apply_fn=network.apply, params=network_params, tx=avg_tx
            )
            sl_buffer = init_sl_buffer(obs.shape[-1], action_dim)
            return (br_train_state, avg_train_state, state, obs, sl_buffer)

        rng, rng_setup = jax.random.split(rng)
        br_train_state, avg_train_state, state, obs, sl_buffer = train_setup(rng_setup)

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

                rng_s, rng_mix, rng_action, rng_step = jax.random.split(rng_s, 4)

                br_logits, value = network.apply(br_train_state_s.params, obs_s)
                avg_logits, _ = network.apply(avg_train_state_s.params, obs_s)
                action_mask = jax.vmap(env.get_avail_actions, in_axes=(0))(state_s)

                br_logits_masked = jnp.where(action_mask, br_logits, -jnp.inf)
                avg_logits_masked = jnp.where(action_mask, avg_logits, -jnp.inf)

                is_br = jax.random.bernoulli(
                    rng_mix,
                    p=config["anticipatory_eta"],
                    shape=(config["num_envs"], env.num_agents),
                )
                acting_logits = jnp.where(
                    is_br[..., None], br_logits_masked, avg_logits_masked
                )

                acting_pi = distrax.Categorical(logits=acting_logits)
                br_pi = distrax.Categorical(logits=br_logits_masked)
                action = acting_pi.sample(seed=rng_action)
                br_log_prob = jnp.where(is_br, br_pi.log_prob(action), 0.0)

                rng_steps = jax.random.split(rng_step, config["num_envs"])
                next_state, next_obs, reward, absorbing, done, info, raw_state = (
                    jax.vmap(env_step_with_raw, in_axes=(0, 0, 0))(
                        rng_steps, state_s, action
                    )
                )

                transition = Transition(
                    obs=obs_s,
                    action_mask=action_mask,
                    action=action,
                    br_log_prob=br_log_prob,
                    reward=reward,
                    absorbing=absorbing,
                    done=done,
                    value=value,
                    is_br=is_br,
                    winners=raw_state.winners,
                    points=raw_state.points,
                    info=info,
                )
                runner_state_inner = RunnerState(
                    br_train_state=br_train_state_s,
                    avg_train_state=avg_train_state_s,
                    sl_buffer=runner_state_inner.sl_buffer,
                    state=next_state,
                    obs=next_obs,
                    done=done,
                    update_step=runner_state_inner.update_step,
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
            rng, rng_sl_append, rng_ppo_update, rng_sl_update, rng_compare = (
                jax.random.split(runner_state.rng, 5)
            )

            _, last_val = network.apply(br_train_state.params, last_obs)

            def calculate_gae(
                transitions_s: Transition,
                last_val_s: FloatArray,
            ) -> tuple[FloatArray, FloatArray]:
                def get_advantages(carry, transition: Transition):
                    gae, next_value = carry
                    done_b = transition.done[:, None].astype(jnp.float32)
                    delta = (
                        transition.reward
                        + config["gamma"] * next_value * (1 - done_b)
                        - transition.value
                    )
                    gae_new = (
                        delta
                        + config["gamma"] * config["gae_lambda"] * (1 - done_b) * gae
                    )
                    return (gae_new, transition.value), gae_new

                _, advantages_s = lax.scan(
                    get_advantages,
                    (jnp.zeros_like(last_val_s), last_val_s),
                    transitions_s,
                    reverse=True,
                    unroll=16,
                )
                return advantages_s, advantages_s + transitions_s.value

            advantages, targets = calculate_gae(transitions, last_val)

            obs_flat = transitions.obs.reshape(-1, transitions.obs.shape[-1])
            action_mask_flat = transitions.action_mask.reshape(
                -1, transitions.action_mask.shape[-1]
            )
            action_flat = transitions.action.reshape(-1)
            is_br_flat = transitions.is_br.reshape(-1)
            sl_buffer, rng_sl_append = append_sl_samples(
                sl_buffer,
                obs_flat,
                action_mask_flat,
                action_flat,
                is_br_flat,
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
                )

                def _reshape_batch(x):
                    # steps, envs, agents are always exactly the 3 leading
                    # axes for goofspiel (every field gains a num_agents
                    # axis), so this unconditional reshape is safe -- unlike
                    # the AEC templates' ndim-branching version, which would
                    # silently misalign minibatches here.
                    return x.reshape(-1, *x.shape[3:])

                batch_reshaped = jax.tree_util.tree_map(_reshape_batch, batch)
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
                    minibatch: tuple[Transition, FloatArray, FloatArray],
                ) -> tuple[TrainState, dict[str, FloatArray]]:
                    transitions_mb, advantages_mb, targets_mb = minibatch

                    def loss(
                        params,
                        transitions_mb,
                        advantages_mb,
                        targets_mb,
                    ) -> tuple[FloatArray, dict[str, FloatArray]]:
                        logits, value = network.apply(params, transitions_mb.obs)
                        logits_masked = jnp.where(
                            transitions_mb.action_mask, logits, -jnp.inf
                        )
                        pi = distrax.Categorical(logits=logits_masked)
                        log_prob = pi.log_prob(transitions_mb.action)

                        mask = transitions_mb.is_br.astype(jnp.float32)
                        mask_denom = jnp.maximum(mask.sum(), 1.0)

                        mean_adv = masked_mean(advantages_mb, mask)
                        var_adv = masked_mean(
                            jnp.square(advantages_mb - mean_adv), mask
                        )
                        gae_normalized = (advantages_mb - mean_adv) / jnp.sqrt(
                            var_adv + 1e-8
                        )

                        logratio = log_prob - transitions_mb.br_log_prob
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
                        br_sample_frac = mask.mean()

                        return total_loss, {
                            "total_loss": total_loss,
                            "value_loss": value_loss,
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "ratio_mean": ratio.mean(),
                            "ratio_min": ratio.min(),
                            "ratio_max": ratio.max(),
                            "gae_mean": mean_adv,
                            "gae_std": jnp.sqrt(var_adv + 1e-8),
                            "mean_target": targets_mb.mean(),
                            "value_pred_mean": value.mean(),
                            "kl_backward": kl_backward,
                            "kl_forward": kl_forward,
                            "clip_frac": clip_frac,
                            "br_sample_frac_minibatch": br_sample_frac,
                        }

                    grad_fn = jax.value_and_grad(loss, has_aux=True)
                    (_, aux), grads = grad_fn(
                        train_state.params,
                        transitions_mb,
                        advantages_mb,
                        targets_mb,
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
                    rng=rng_s,
                )
                return ppo_state, batch_stats

            ppo_state = PPOUpdateState(
                train_state=br_train_state,
                transitions=transitions,
                advantages=advantages,
                targets=targets,
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

            if checkpoint_steps:
                jax.experimental.io_callback(
                    checkpoint_callback,
                    None,
                    seed,
                    runner_state.update_step,
                    br_train_state.params,
                    avg_train_state.params,
                )

            done_mask = transitions.done  # (steps, envs)
            done_count = jnp.maximum(done_mask.sum(), 1)

            # Cumulative game points (not raw per-round reward) at terminal steps.
            masked_points = jnp.where(done_mask[..., None], transitions.points, 0.0)
            returns_avg = masked_points.sum() / done_count
            returns_avg_agent_one = masked_points[..., 0].sum() / done_count
            returns_avg_agent_two = masked_points[..., 1].sum() / done_count

            winners = transitions.winners  # (steps, envs, agents), valid where done
            win_seat0 = (
                (winners[..., 0] & (~winners[..., 1]) & done_mask)
                .astype(jnp.float32)
                .sum()
                / done_count
            )
            win_seat1 = (
                (winners[..., 1] & (~winners[..., 0]) & done_mask)
                .astype(jnp.float32)
                .sum()
                / done_count
            )
            tie_rate = (
                ((winners.sum(axis=-1) > 1) & done_mask).astype(jnp.float32).sum()
                / done_count
            )

            # Bid-vs-prize-value correlation: first 13 obs columns are the
            # current prize card one-hot (see Goofspiel.obs_from_state).
            prize_value = (
                jnp.argmax(transitions.obs[..., :13], axis=-1).astype(jnp.float32)
                + 1.0
            )
            bid_value = transitions.action.astype(jnp.float32) + 1.0
            bid_prize_corr = jnp.corrcoef(
                bid_value.reshape(-1), prize_value.reshape(-1)
            )[0, 1]

            # Round-tie frequency (non-terminal-round ties), from the joint
            # action logged every step.
            max_bid = jnp.max(transitions.action, axis=-1, keepdims=True)
            round_tie_mask = (transitions.action == max_bid).sum(axis=-1) > 1
            round_tie_rate = round_tie_mask.astype(jnp.float32).mean()

            ep_length_avg = jnp.array(float(env.num_cards), dtype=jnp.float32)

            should_compare = compare_enabled & (
                runner_state.update_step % config["compare_interval"] == 0
            )

            def do_compare(carry_rng):
                checkpoint_params = baseline_params
                if compare_against_random:
                    checkpoint_params = avg_train_state.params
                return run_compare_eval(
                    carry_rng,
                    avg_train_state.params,
                    br_train_state.params,
                    checkpoint_params,
                )

            def skip_compare(carry_rng):
                return (
                    carry_rng,
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                )

            (
                rng_compare,
                avg_ret_avg_vs_baseline,
                avg_ret_br_vs_baseline,
                avg_eval_episode_length,
                win_rate_avg_vs_baseline,
                win_rate_br_vs_baseline,
                tie_rate_avg_vs_baseline_raw,
                tie_rate_br_vs_baseline_raw,
            ) = lax.cond(
                should_compare,
                do_compare,
                skip_compare,
                rng_compare,
            )

            metric = {
                "returns_avg": returns_avg,
                "returns_avg_agent_one": returns_avg_agent_one,
                "returns_avg_agent_two": returns_avg_agent_two,
                "ep_length_avg": ep_length_avg,
                "br_action_frac_rollout": transitions.is_br.mean(),
                "win_rate_seat0": win_seat0,
                "win_rate_seat1": win_seat1,
                "tie_rate": tie_rate,
                "bid_prize_corr": bid_prize_corr,
                "round_tie_rate": round_tie_rate,
                "sl_buffer_size": sl_buffer.size.astype(jnp.float32),
                "sl_buffer_seen": sl_buffer.seen.astype(jnp.float32),
                "update_step": runner_state.update_step,
                "avg_return_avg_vs_baseline": (
                    jnp.array(0.0, dtype=jnp.float32)
                    if compare_against_random
                    else avg_ret_avg_vs_baseline
                ),
                "avg_return_br_vs_baseline": (
                    jnp.array(0.0, dtype=jnp.float32)
                    if compare_against_random
                    else avg_ret_br_vs_baseline
                ),
                "avg_return_avg_vs_random": (
                    avg_ret_avg_vs_baseline
                    if compare_against_random
                    else jnp.array(0.0, dtype=jnp.float32)
                ),
                "avg_return_br_vs_random": (
                    avg_ret_br_vs_baseline
                    if compare_against_random
                    else jnp.array(0.0, dtype=jnp.float32)
                ),
                "avg_eval_episode_length": avg_eval_episode_length,
                "win_rate_avg_vs_baseline": (
                    jnp.array(0.0, dtype=jnp.float32)
                    if compare_against_random
                    else win_rate_avg_vs_baseline
                ),
                "win_rate_br_vs_baseline": (
                    jnp.array(0.0, dtype=jnp.float32)
                    if compare_against_random
                    else win_rate_br_vs_baseline
                ),
                "win_rate_avg_vs_random": (
                    win_rate_avg_vs_baseline
                    if compare_against_random
                    else jnp.array(0.0, dtype=jnp.float32)
                ),
                "win_rate_br_vs_random": (
                    win_rate_br_vs_baseline
                    if compare_against_random
                    else jnp.array(0.0, dtype=jnp.float32)
                ),
                "tie_rate_avg_vs_baseline": (
                    jnp.array(0.0, dtype=jnp.float32)
                    if compare_against_random
                    else tie_rate_avg_vs_baseline_raw
                ),
                "tie_rate_br_vs_baseline": (
                    jnp.array(0.0, dtype=jnp.float32)
                    if compare_against_random
                    else tie_rate_br_vs_baseline_raw
                ),
                "tie_rate_avg_vs_random": (
                    tie_rate_avg_vs_baseline_raw
                    if compare_against_random
                    else jnp.array(0.0, dtype=jnp.float32)
                ),
                "tie_rate_br_vs_random": (
                    tie_rate_br_vs_baseline_raw
                    if compare_against_random
                    else jnp.array(0.0, dtype=jnp.float32)
                ),
            }
            metric.update(ppo_loss_info)
            metric.update(sl_loss_info)

            def logging_callback(seed_val, metric_dict, info):
                wandb_callback(seed_val, dict(metric_dict), info)

            jax.experimental.io_callback(
                logging_callback,
                None,
                seed,
                metric,
                transitions.info,
            )

            runner_state = RunnerState(
                br_train_state=br_train_state,
                avg_train_state=avg_train_state,
                sl_buffer=sl_buffer,
                state=runner_state.state,
                obs=runner_state.obs,
                done=runner_state.done,
                update_step=runner_state.update_step + 1,
                rng=rng_compare,
            )
            return runner_state, None

        initial_runner_state = RunnerState(
            br_train_state=br_train_state,
            avg_train_state=avg_train_state,
            sl_buffer=sl_buffer,
            state=state,
            obs=obs,
            done=jnp.zeros((config["num_envs"]), dtype=jnp.bool_),
            update_step=jnp.array(0, dtype=jnp.int32),
            rng=rng,
        )
        final_runner_state, _ = lax.scan(
            update_step,
            initial_runner_state,
            None,
            config["num_update_steps"],
        )
        return final_runner_state

    return train


@hydra.main(version_base=None, config_path="./", config_name="config_ppo_nfsp")
def main(config: dict) -> None:
    global LOGGER
    try:
        config = OmegaConf.to_container(config)
        group = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config["group"] = group
        if config["use_wandb"]:
            LOGGER = WandbMultiLogger(
                project=config["project"],
                group=group,
                job_type=config["algorithm"] + config.get("custom_name", ""),
                config=config,
                mode="online",
                seed=config["seed"],
                num_seeds=config["num_seeds"],
            )
        else:
            LOGGER = None

        rng = jax.random.PRNGKey(config["seed"])
        rng_seeds = jax.random.split(rng, config["num_seeds"])
        exp_ids = jnp.arange(config["num_seeds"])

        print("Compiling BluffJAX...")
        train_fn = jax.jit(jax.vmap(make_train(config)))
        print("Running...")
        jax.block_until_ready(train_fn(rng_seeds, exp_ids))
    finally:
        if LOGGER is not None:
            LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
