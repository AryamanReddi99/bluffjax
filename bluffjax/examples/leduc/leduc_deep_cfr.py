"""
Deep CFR for Leduc Poker.

Deep Counterfactual Regret Minimization (Brown et al., 2019,
https://arxiv.org/abs/1811.00164), following Algorithms 1 and 2 of the paper
and OpenSpiel's implementations (open_spiel/python/{jax,pytorch}/deep_cfr.py):

- On iteration t, for each player p in turn: K external-sampling traversals
  with p as the traverser, then a new advantage network for p is trained from
  scratch on p's reservoir-sampled advantage memory.
- Strategies come from regret matching on the predicted advantages. When no
  advantage is positive, the max-advantage action is played (paper, Section
  2.1). Before its first training the advantage network predicts 0 everywhere
  (Algorithm 1), which gives the uniform strategy.
- Opponent strategies met during traversals go to a reservoir-sampled strategy
  memory, on which the average-strategy (policy) network is trained.
- Linear CFR: every sample is weighted by the iteration t it was collected on,
  rescaled by 2 / T when training on iteration T (paper, Section 5.3).

OpenSpiel's JAX and PyTorch versions skip training while a memory holds fewer
samples than a batch; here the networks are trained on every iteration, with
minibatches sampled with replacement.

Defaults: 50 iterations of 100 traversals per player, 2 hidden layers of 64
units, 200 advantage-network steps and memories of 100,000 samples, as in the
BluffJAX paper; Adam with learning rate 1e-3 and gradient norms clipped to 1,
as in the Deep CFR paper (Section 5.2); batches of 2048 and 5000
policy-network steps, as in OpenSpiel's Deep CFR examples.

Network inputs are OpenSpiel's leduc_poker information_state_tensor, which is
perfect recall: each of the 936 infosets has a distinct tensor. Exploitability
is exact, over the full game tree.

Evaluation: after iteration 1, every --eval_every iterations, after half of
the iterations and after the last one, a policy network is trained on the
strategy memory (its key depends only on the iteration, so evaluations don't
change the run) and the exact exploitability of its policy and its exact
behaviour metrics in self-play are logged, the latter as game/<name>
(bluffjax.metrics.leduc_holdem: bluff rate and success, correct-fold rate,
round-2 raise rates with and without a pair; definitions there). Every
iteration logs the iteration count, the traversals so far and the advantage
losses.

Seeds: --num_seeds runs (default 5) one after the other in one run directory;
seed index i uses seed --seed + i and has its own wandb run "{seed}_{i}".

Run directory ({save_dir}/{job_type}_{timestamp}/, made when the run starts;
see bluffjax.examples.run_utils):
    config.yaml                  the arguments, with half_iterations
    results.jsonl                every logged record (what wandb gets, plus
                                 seed_index and wall_time); each has iteration
                                 and traversals
    half/avg_policy_{i}.msgpack  policy (average-strategy) network after
                                 half_iterations = half_update_steps(
                                 iterations) iterations of seed index i
    full/avg_policy_{i}.msgpack  after all iterations
(checkpoints unless --no-save_checkpoints). Each file holds the flax params of
MLP(hidden_sizes, 3) (hidden_sizes in config.yaml; policy_network() builds
it), written with run_utils.save_params; load with load_params(path,
policy_network(hidden_sizes).init(rng, jnp.zeros((1, INFO_STATE_SIZE)))).
The network maps information_state_tensor(infoset key) (30 floats, which
needs the physical cards, not the env observation) to action logits over
fold/call/raise; illegal actions are masked before the softmax. That policy is
the average strategy: policy_array(params, hidden_sizes) gives it at every
infoset in leduc_exploitability._INFOSET_LIST order.
"""

import argparse
import datetime
import functools
import json
import time
from typing import Callable, NamedTuple, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from bluffjax.examples.run_utils import (
    ResultsLog,
    half_update_steps,
    make_run_dir,
    save_params,
)
from bluffjax.metrics.leduc_holdem import log_metrics
from bluffjax.utils.game_utils import leduc_exploitability as leduc
from bluffjax.utils.paths import REPO_ROOT
from bluffjax.utils.typing import BoolArray, FloatArray, PRNGKeyArray, PyTree

NUM_ACTIONS = 3  # fold, call, raise
NUM_CARDS = 6
MAX_ACTIONS_PER_ROUND = 4  # with at most 2 raises, e.g. call raise raise call
# Acting player, private card, public card, then [call, raise] bits per action
INFO_STATE_SIZE = 2 + 2 * NUM_CARDS + 2 * 2 * MAX_ACTIONS_PER_ROUND
ILLEGAL_ACTION_LOGITS_PENALTY = jnp.finfo(jnp.float32).min


def information_state_tensor(infoset_key: tuple) -> np.ndarray:
    """OpenSpiel's leduc_poker information_state_tensor for an infoset.

    Layout (30): acting player one-hot (2), private card one-hot (6), public
    card one-hot (6, zeros before the flop), then each round's betting sequence
    as 4 action slots of [call, raise] bits (a fold sets neither).
    """
    player, private_card, public_card, round1, round2 = infoset_key[:5]
    tensor = np.zeros(INFO_STATE_SIZE, dtype=np.float32)
    tensor[player] = 1.0
    tensor[2 + private_card] = 1.0
    if public_card >= 0:
        tensor[2 + NUM_CARDS + public_card] = 1.0
    offset = 2 + 2 * NUM_CARDS
    for sequence in (round1, round2):
        for i, action in enumerate(sequence):
            if action == leduc.ACTION_CALL:
                tensor[offset + 2 * i] = 1.0
            elif action == leduc.ACTION_RAISE:
                tensor[offset + 2 * i + 1] = 1.0
        offset += 2 * MAX_ACTIONS_PER_ROUND
    return tensor


def legal_actions_mask(legal_actions: Sequence[int]) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[list(legal_actions)] = True
    return mask


def regret_matching(advantages: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """Strategy proportional to the positive advantages of the legal actions.

    With no positive advantage, plays the legal action with the highest
    advantage, as in the paper and OpenSpiel. Ties are split evenly, so an
    all-zero prediction gives the uniform strategy.
    """
    positive = np.where(legal_mask, np.maximum(advantages, 0.0), 0.0)
    total = positive.sum()
    if total > 0:
        return positive / total
    masked = np.where(legal_mask, advantages, -np.inf)
    best = (masked == masked.max()).astype(np.float64)
    return best / best.sum()


# Network inputs for every infoset, in leduc_exploitability's infoset order
INFOSET_TENSORS = np.stack(
    [information_state_tensor(k) for k, _ in leduc._INFOSET_LIST]
)
INFOSET_LEGAL_MASKS = np.stack([legal_actions_mask(a) for _, a in leduc._INFOSET_LIST])


# --- Reservoir memories ---
class Memory(NamedTuple):
    info_state: FloatArray  # [capacity, INFO_STATE_SIZE]
    iteration: FloatArray  # [capacity], CFR iteration the sample was collected on
    target: FloatArray  # [capacity, NUM_ACTIONS], sampled regrets or strategy
    legal_mask: BoolArray  # [capacity, NUM_ACTIONS]


class ReservoirBuffer:
    """Reservoir sampling (Vitter, 1985): after n additions, each of the n
    samples is kept with probability min(1, capacity / n)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.add_calls = 0
        self.data = Memory(
            info_state=np.zeros((capacity, INFO_STATE_SIZE), dtype=np.float32),
            iteration=np.zeros(capacity, dtype=np.float32),
            target=np.zeros((capacity, NUM_ACTIONS), dtype=np.float32),
            legal_mask=np.zeros((capacity, NUM_ACTIONS), dtype=bool),
        )

    def __len__(self) -> int:
        return min(self.add_calls, self.capacity)

    def add(self, rng: np.random.Generator, sample: Memory) -> None:
        if self.add_calls < self.capacity:
            idx = self.add_calls
        else:
            idx = rng.integers(self.add_calls + 1)
        self.add_calls += 1
        if idx < self.capacity:
            for buffer, value in zip(self.data, sample):
                buffer[idx] = value


# --- Networks ---
def torch_linear_init(fan_in: int) -> Callable:
    """PyTorch's default nn.Linear initialisation for weights and biases,
    U(-1/sqrt(fan_in), 1/sqrt(fan_in)), as in the Deep CFR paper's network."""
    bound = 1.0 / np.sqrt(fan_in)
    return lambda key, shape, dtype=jnp.float32: jax.random.uniform(
        key, shape, dtype, -bound, bound
    )


class MLP(nn.Module):
    """ReLU MLP with LayerNorm after each hidden layer."""

    hidden_sizes: Sequence[int]
    output_size: int

    @nn.compact
    def __call__(self, x: FloatArray) -> FloatArray:
        for size in self.hidden_sizes:
            init = torch_linear_init(x.shape[-1])
            x = nn.relu(nn.Dense(size, kernel_init=init, bias_init=init)(x))
            x = nn.LayerNorm()(x)
        init = torch_linear_init(x.shape[-1])
        return nn.Dense(self.output_size, kernel_init=init, bias_init=init)(x)


def policy_network(hidden_sizes: Sequence[int]) -> MLP:
    """The advantage and policy network architecture (also of the checkpoints)."""
    return MLP(tuple(hidden_sizes), NUM_ACTIONS)


def policy_probs(logits: np.ndarray) -> np.ndarray:
    """Softmax over the legal actions of every infoset, (936, 3) float64."""
    logits = np.where(INFOSET_LEGAL_MASKS, np.asarray(logits, np.float64), -np.inf)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return probs / probs.sum(axis=-1, keepdims=True)


def policy_array(params: PyTree, hidden_sizes: Sequence[int]) -> np.ndarray:
    """The average strategy of a policy network (e.g. a loaded checkpoint) at
    every infoset, (936, 3) in leduc_exploitability._INFOSET_LIST order."""
    return policy_probs(policy_network(hidden_sizes).apply(params, INFOSET_TENSORS))


def advantage_loss(
    apply_fn: Callable, params: PyTree, batch: Memory, iteration: FloatArray
) -> FloatArray:
    """Linear-CFR weighted squared error between predicted advantages and
    sampled regrets, over the legal actions (Algorithm 1)."""
    preds = apply_fn(params, batch.info_state)
    sq_err = jnp.where(batch.legal_mask, preds - batch.target, 0.0) ** 2
    weights = batch.iteration * 2.0 / iteration
    return jnp.mean(weights * sq_err.sum(axis=-1))


def policy_loss(
    apply_fn: Callable, params: PyTree, batch: Memory, iteration: FloatArray
) -> FloatArray:
    """Linear-CFR weighted squared error between the policy and the stored
    strategies (Algorithm 1). Both are zero on illegal actions."""
    logits = apply_fn(params, batch.info_state)
    logits = jnp.where(batch.legal_mask, logits, ILLEGAL_ACTION_LOGITS_PENALTY)
    sq_err = (jax.nn.softmax(logits) - batch.target) ** 2
    weights = batch.iteration * 2.0 / iteration
    return jnp.mean(weights * sq_err.sum(axis=-1))


def make_train(
    network: nn.Module,
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    num_steps: int,
    batch_size: int,
) -> Callable:
    """Returns a jitted function that initialises `network` from scratch and
    trains it for `num_steps` minibatch steps on the first `memory_size`
    samples of a memory. Minibatches are sampled uniformly with replacement."""
    loss_fn = functools.partial(loss_fn, network.apply)

    def train(
        key: PRNGKeyArray, memory: Memory, memory_size: int, iteration: float
    ) -> tuple[PyTree, FloatArray]:
        init_key, sample_key = jax.random.split(key)
        params = network.init(init_key, memory.info_state[:1])
        opt_state = optimizer.init(params)

        def _update_step(carry, step_key):
            params, opt_state = carry
            idx = jax.random.randint(step_key, (batch_size,), 0, memory_size)
            batch = jax.tree.map(lambda x: x[idx], memory)
            loss, grads = jax.value_and_grad(loss_fn)(params, batch, iteration)
            updates, opt_state = optimizer.update(grads, opt_state)
            return (optax.apply_updates(params, updates), opt_state), loss

        step_keys = jax.random.split(sample_key, num_steps)
        (params, _), losses = jax.lax.scan(_update_step, (params, opt_state), step_keys)
        return params, losses[-1]

    return jax.jit(train)


@functools.lru_cache(maxsize=None)
def _jitted_functions(
    hidden_sizes: tuple[int, ...],
    learning_rate: float,
    max_grad_norm: float,
    advantage_train_steps: int,
    policy_train_steps: int,
    batch_size: int,
) -> tuple[Callable, Callable, Callable]:
    """apply, train_advantage and train_policy, built once per configuration
    so that the seeds of a run share the compiled functions."""
    network = policy_network(hidden_sizes)
    optimizer = optax.adam(learning_rate)
    if max_grad_norm > 0:
        optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optimizer)
    return (
        jax.jit(network.apply),
        make_train(
            network, advantage_loss, optimizer, advantage_train_steps, batch_size
        ),
        make_train(network, policy_loss, optimizer, policy_train_steps, batch_size),
    )


# --- Deep CFR Solver ---
class DeepCFRSolver:
    def __init__(self, config: argparse.Namespace, seed: int | None = None) -> None:
        """seed defaults to config.seed."""
        self._config = config
        self._iteration = 0
        seed = config.seed if seed is None else seed
        # Traversal and reservoir sampling on the host, network training on device
        self._rng = np.random.default_rng(seed)
        self._advantage_key, self._policy_key = jax.random.split(jax.random.key(seed))

        self._advantage_memories = [
            ReservoirBuffer(config.memory_capacity) for _ in range(2)
        ]
        self._strategy_memory = ReservoirBuffer(config.memory_capacity)
        # None stands for the initial advantage network, which predicts 0 everywhere
        self._advantage_params: list[PyTree | None] = [None, None]
        # Regret-matching strategies of each player's current advantage network,
        # keyed by information state
        self._strategy_cache: list[dict[bytes, np.ndarray]] = [{}, {}]

        self._apply, self._train_advantage, self._train_policy = _jitted_functions(
            tuple(config.hidden_sizes),
            config.learning_rate,
            config.max_grad_norm,
            config.advantage_train_steps,
            config.policy_train_steps,
            config.batch_size,
        )

    def run_iteration(self) -> list[float]:
        """One Deep CFR iteration. Returns each player's final advantage loss."""
        self._iteration += 1
        losses = []
        for player in range(2):
            for _ in range(self._config.traversals):
                self._traverse_game_tree(leduc.initial_state(), player)
            losses.append(self._learn_advantage_network(player))
        return losses

    def _strategy(
        self, player: int, info_state: np.ndarray, legal_mask: np.ndarray
    ) -> np.ndarray:
        """Regret matching on `player`'s current advantage network."""
        cache = self._strategy_cache[player]
        key = info_state.tobytes()
        if key not in cache:
            params = self._advantage_params[player]
            if params is None:
                advantages = np.zeros(NUM_ACTIONS)
            else:
                advantages = np.asarray(
                    self._apply(params, info_state), dtype=np.float64
                )
            cache[key] = regret_matching(advantages, legal_mask)
        return cache[key]

    def _traverse_game_tree(self, state: leduc.LeducState, player: int) -> float:
        """External-sampling traversal for `player` (Algorithm 2 of the paper).

        Explores every traverser action and samples one action at chance and
        opponent nodes. Adds the traverser's sampled regrets to its advantage
        memory and the opponent's strategies to the strategy memory. Returns the
        traverser's sampled value of `state`.
        """
        if state.is_terminal():
            return state.returns()[player]
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            action = outcomes[self._rng.choice(len(outcomes), p=probs)]
            return self._traverse_game_tree(state.child(action), player)

        current = state.current_player
        info_state = information_state_tensor(state.info_state_key(current))
        legal_mask = legal_actions_mask(state.legal_actions())
        strategy = self._strategy(current, info_state, legal_mask)
        if current == player:
            values = np.zeros(NUM_ACTIONS)
            for action in state.legal_actions():
                values[action] = self._traverse_game_tree(state.child(action), player)
            value = np.dot(strategy, values)
            regrets = np.where(legal_mask, values - value, 0.0)
            sample = Memory(info_state, self._iteration, regrets, legal_mask)
            self._advantage_memories[player].add(self._rng, sample)
            return value
        sample = Memory(info_state, self._iteration, strategy, legal_mask)
        self._strategy_memory.add(self._rng, sample)
        action = self._rng.choice(NUM_ACTIONS, p=strategy)
        return self._traverse_game_tree(state.child(action), player)

    def _learn_advantage_network(self, player: int) -> float:
        """Trains a new advantage network for `player` from scratch."""
        memory = self._advantage_memories[player]
        key = jax.random.fold_in(
            jax.random.fold_in(self._advantage_key, self._iteration), player
        )
        self._advantage_params[player], loss = self._train_advantage(
            key, memory.data, len(memory), self._iteration
        )
        self._strategy_cache[player].clear()
        return float(loss)

    def learn_policy_network(self) -> tuple[PyTree, float]:
        """Trains the average-strategy network from scratch on the strategy
        memory collected so far. Its key depends only on the iteration, so
        intermediate evaluations don't change the run."""
        key = jax.random.fold_in(self._policy_key, self._iteration)
        params, loss = self._train_policy(
            key, self._strategy_memory.data, len(self._strategy_memory), self._iteration
        )
        return params, float(loss)

    def average_policy(self, policy_params: PyTree) -> np.ndarray:
        """Policy network's action probabilities for every infoset, shape (936, 3)."""
        return policy_probs(self._apply(policy_params, INFOSET_TENSORS))


def exploitability(policy_array: np.ndarray) -> float:
    return leduc.exploitability_from_policy_array(policy_array, leduc._KEY_TO_IDX)


def run_deep_cfr(
    config: argparse.Namespace,
    seed_index: int = 0,
    results_log: ResultsLog | None = None,
    run_dir: str | None = None,
) -> dict:
    """Runs Deep CFR with seed config.seed + seed_index, logging the
    exploitability and game metrics of the average-strategy network after
    iteration 1, every `eval_every` iterations, after half_update_steps(
    iterations) and at the end; with run_dir, saves that network after half
    and all of the iterations as {half,full}/avg_policy_{seed_index}."""
    start = time.time()
    eval_seconds = 0.0
    solver = DeepCFRSolver(config, config.seed + seed_index)
    num_half = half_update_steps(config.iterations)
    if run_dir is not None and num_half < 1:
        raise ValueError("the half checkpoint needs --iterations >= 2")
    results = {
        "iterations": [],
        "exploitability": [],
        "advantage_loss": [],
        "policy_loss": [],
    }

    for it in range(1, config.iterations + 1):
        advantage_losses = solver.run_iteration()
        results["advantage_loss"].append(advantage_losses)
        metrics = {
            "iteration": it,
            "traversals": it * 2 * config.traversals,
            "advantage_loss_p0": advantage_losses[0],
            "advantage_loss_p1": advantage_losses[1],
        }
        is_last = it == config.iterations
        if it == 1 or it % config.eval_every == 0 or it == num_half or is_last:
            eval_start = time.time()
            policy_params, loss = solver.learn_policy_network()
            if is_last:
                # Training the final policy network is part of the algorithm
                eval_start = time.time()
            policy = solver.average_policy(policy_params)
            expl = exploitability(policy)
            metrics.update(exploitability=expl, policy_loss=loss)
            metrics.update(log_metrics("game", policy))
            eval_seconds += time.time() - eval_start
            results["iterations"].append(it)
            results["exploitability"].append(expl)
            results["policy_loss"].append(loss)
            print(
                f"seed index {seed_index} iteration {it:4d}: exploitability = "
                f"{expl:.6f}, bluff rate {metrics['game/bluff_rate']:.4f}, "
                f"correct folds {metrics['game/correct_fold_rate']:.4f}",
                flush=True,
            )
            if run_dir is not None and it in (num_half, config.iterations):
                stage = "full" if is_last else "half"
                save_params(run_dir, stage, "avg_policy", policy_params, seed_index)
        if results_log is not None:
            results_log.log(metrics, seed_index)

    results["final_exploitability"] = results["exploitability"][-1]
    results["runtime_seconds"] = time.time() - start
    # Intermediate policy networks and all exploitability computations, which a
    # run without logging doesn't need
    results["eval_seconds"] = eval_seconds
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep CFR on Leduc Poker.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=50, help="CFR iterations T")
    parser.add_argument(
        "--traversals",
        type=int,
        default=100,
        help="traversals K per player per iteration",
    )
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="0 disables clipping"
    )
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--advantage_train_steps", type=int, default=200)
    parser.add_argument("--policy_train_steps", type=int, default=5000)
    parser.add_argument("--memory_capacity", type=int, default=100_000)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="runs one after the other; seed index i uses seed --seed + i",
    )
    parser.add_argument(
        "--save_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="policy network after half and all of the iterations",
    )
    parser.add_argument(
        "--save_dir", type=str, default=str(REPO_ROOT / "checkpoints" / "leduc")
    )
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--project", type=str, default="bluffjax")
    parser.add_argument("--job_type", type=str, default="leduc_deep_cfr")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="also write every seed's results to this JSON file",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    run_config = {
        **vars(config),
        "env_name": "leduc_poker",
        "half_iterations": half_update_steps(config.iterations),
    }
    run_dir = make_run_dir(run_config)
    group = run_config["env_name"] + datetime.datetime.now().strftime(
        "_%Y-%m-%d_%H-%M-%S"
    )
    wandb_runs = [
        wandb.init(
            project=config.project,
            group=group,
            job_type=f"{config.job_type}_{run_config['env_name']}",
            name=f"{config.seed}_{i}",
            config={**run_config, "seed_index": i},
            mode=None if config.wandb else "disabled",
            dir=str(REPO_ROOT),
            reinit="create_new",
        )
        for i in range(config.num_seeds)
    ]
    try:
        results_log = ResultsLog(run_dir, wandb_runs)
        print(f"Run directory: {run_dir}")
        print("Deep CFR for Leduc Poker")
        all_results = []
        for i in range(config.num_seeds):
            print("=" * 50)
            print(f"seed index {i} (seed {config.seed + i})")
            results = run_deep_cfr(
                config, i, results_log, run_dir if config.save_checkpoints else None
            )
            all_results.append({"seed": config.seed + i, "seed_index": i, **results})
            print(f"Final exploitability: {results['final_exploitability']:.6f}")
            print(
                f"Runtime: {results['runtime_seconds']:.1f}s "
                f"({results['eval_seconds']:.1f}s evaluating)"
            )
        finals = [r["final_exploitability"] for r in all_results]
        print("=" * 50)
        print(
            f"final exploitability: mean {np.mean(finals):.6f} std "
            f"{np.std(finals):.6f} over {len(finals)} seed(s)"
        )
        if config.save_checkpoints:
            print(f"Saved checkpoints to {run_dir}")
        if config.output:
            with open(config.output, "w") as f:
                json.dump(
                    {"game": "leduc_poker", "config": vars(config), "runs": all_results},
                    f,
                    indent=2,
                )
    finally:
        for run in wandb_runs:
            run.finish()


if __name__ == "__main__":
    main()
