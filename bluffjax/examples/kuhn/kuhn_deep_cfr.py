"""
Deep CFR for Kuhn Poker.

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

Network inputs are OpenSpiel's kuhn_poker information_state_tensor, which is
perfect recall: each of the 12 infosets has a distinct tensor. Exploitability
is exact.

Seeds: --num_seeds runs run one after the other in one run directory; seed
index i uses seed + i.

Evaluation: the average-strategy (policy) network is trained after iteration
1, every eval_every iterations, after iteration iterations // 2 (the half
checkpoint) and after the last; each time its exact exploitability, its exact
expected return per hand against uniformly random play averaged over both
seats (value_vs_random, kuhn_exploitability.value_vs_random) and the exact
Kuhn policy metrics (bluffjax.metrics.kuhn_poker) are logged. Its training key
depends only on the iteration, so these intermediate networks don't change the
run.

Logging: main() creates the run directory {save_dir}/{job_type}_{timestamp}/
(run_utils.make_run_dir, with config.yaml holding the flags) before running.
Every iteration writes one record per seed to {run_dir}/results.jsonl (with
seed_index and wall_time), and to the seed's wandb run with --wandb:
iteration, traversals (the traversals so far, both players) and each player's
final advantage-network loss; at the evaluations also exploitability,
value_vs_random, policy_loss and the policy metrics of the policy network,
game/<name> (names and definitions in bluffjax.metrics.kuhn_poker).

Checkpoints (on unless --no-save_checkpoints), for every seed index i:
    {run_dir}/half/avg_policy_{i}.msgpack  policy network trained after
                                           iteration iterations // 2
    {run_dir}/full/avg_policy_{i}.msgpack  policy network trained after the
                                           last iteration
both from the same run. Each holds the flax params of MLP(hidden_sizes, 2)
(hidden_sizes in config.yaml), written with flax.serialization.to_bytes. The
network maps information_state_tensor(card, history) (11 floats; the rows of
INFOSET_TENSORS are the 12 infosets in kuhn_exploitability.INFOSETS order) to
logits over [pass, bet], and Deep CFR's average strategy is softmax(logits);
policy_array(params, hidden_sizes) gives it as kuhn_exploitability's (12, 2)
policy array. Nothing else is needed to play it (the advantage networks only
define the current strategies during training). Restore one with
run_utils.load_params(path, MLP(hidden_sizes, 2).init(rng, INFOSET_TENSORS[:1])).
"""

import argparse
import datetime
import functools
import itertools
import json
import time
from typing import Callable, NamedTuple, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bluffjax.examples.run_utils import (
    ResultsLog,
    half_update_steps,
    make_run_dir,
    save_params,
)
from bluffjax.metrics import prefixed
from bluffjax.metrics.kuhn_poker import policy_metrics
from bluffjax.utils.game_utils.kuhn_exploitability import (
    ACTION_BET,
    ACTION_PASS,
    INFOSETS,
    exploitability,
    get_current_player,
    get_legal_actions,
    get_returns,
    is_terminal,
    policy_from_array,
    value_vs_random,
)
from bluffjax.utils.paths import REPO_ROOT
from bluffjax.utils.typing import BoolArray, FloatArray, PRNGKeyArray, PyTree

NUM_ACTIONS = 2  # pass, bet
NUM_CARDS = 3
MAX_GAME_LENGTH = 3  # pass bet pass/bet
# Acting player, private card, then [pass, bet] bits per action
INFO_STATE_SIZE = 2 + NUM_CARDS + NUM_ACTIONS * MAX_GAME_LENGTH
ILLEGAL_ACTION_LOGITS_PENALTY = jnp.finfo(jnp.float32).min
# Chance deals (P0 card, P1 card), all equally likely
DEALS = tuple(itertools.permutations(range(NUM_CARDS), 2))


def information_state_tensor(card: int, history: tuple[int, ...]) -> np.ndarray:
    """OpenSpiel's kuhn_poker information_state_tensor for the acting player.

    Layout (11): acting player one-hot (2), private card one-hot (3), then 3
    action slots of [pass, bet] bits.
    """
    tensor = np.zeros(INFO_STATE_SIZE, dtype=np.float32)
    tensor[get_current_player(history)] = 1.0
    tensor[2 + card] = 1.0
    for i, action in enumerate(history):
        tensor[2 + NUM_CARDS + NUM_ACTIONS * i + action] = 1.0
    return tensor


def _infoset_tensor(infoset_key: str) -> np.ndarray:
    """Information state tensor for an infoset key such as "1pb"."""
    history = tuple(ACTION_PASS if c == "p" else ACTION_BET for c in infoset_key[1:])
    return information_state_tensor(int(infoset_key[0]), history)


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


# Network inputs for every infoset, in kuhn_exploitability's infoset order.
# Both actions are legal in every Kuhn infoset.
INFOSET_TENSORS = np.stack([_infoset_tensor(k) for k in INFOSETS])


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


def softmax_table(logits: FloatArray) -> np.ndarray:
    """softmax over the last axis, in float64."""
    logits = np.asarray(logits, dtype=np.float64)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return probs / probs.sum(axis=-1, keepdims=True)


def policy_array(params: PyTree, hidden_sizes: Sequence[int]) -> np.ndarray:
    """(12, 2) float64 average strategy softmax(logits) of a policy network
    MLP(hidden_sizes, 2) at every infoset, rows in INFOSETS order."""
    network = MLP(tuple(hidden_sizes), NUM_ACTIONS)
    return softmax_table(network.apply(params, INFOSET_TENSORS))


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


# --- Deep CFR Solver ---
class DeepCFRSolver:
    def __init__(self, config: argparse.Namespace) -> None:
        self._config = config
        self._iteration = 0
        # Traversal and reservoir sampling on the host, network training on device
        self._rng = np.random.default_rng(config.seed)
        self._advantage_key, self._policy_key = jax.random.split(
            jax.random.key(config.seed)
        )

        self._advantage_memories = [
            ReservoirBuffer(config.memory_capacity) for _ in range(2)
        ]
        self._strategy_memory = ReservoirBuffer(config.memory_capacity)
        # None stands for the initial advantage network, which predicts 0 everywhere
        self._advantage_params: list[PyTree | None] = [None, None]
        # Regret-matching strategies of each player's current advantage network,
        # keyed by information state
        self._strategy_cache: list[dict[bytes, np.ndarray]] = [{}, {}]

        network = MLP(tuple(config.hidden_sizes), NUM_ACTIONS)
        optimizer = optax.adam(config.learning_rate)
        if config.max_grad_norm > 0:
            optimizer = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm), optimizer
            )
        self._apply = jax.jit(network.apply)
        self._train_advantage = make_train(
            network,
            advantage_loss,
            optimizer,
            config.advantage_train_steps,
            config.batch_size,
        )
        self._train_policy = make_train(
            network,
            policy_loss,
            optimizer,
            config.policy_train_steps,
            config.batch_size,
        )

    def run_iteration(self) -> list[float]:
        """One Deep CFR iteration. Returns each player's final advantage loss."""
        self._iteration += 1
        losses = []
        for player in range(2):
            for _ in range(self._config.traversals):
                # The deal is Kuhn's only chance event
                hands = DEALS[self._rng.integers(len(DEALS))]
                self._traverse_game_tree(hands, (), player)
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

    def _traverse_game_tree(
        self, hands: tuple[int, int], history: tuple[int, ...], player: int
    ) -> float:
        """External-sampling traversal for `player` (Algorithm 2 of the paper).

        Explores every traverser action and samples one opponent action.
        Adds the traverser's sampled regrets to its advantage memory and the
        opponent's strategies to the strategy memory. Returns the traverser's
        sampled value of the history.
        """
        if is_terminal(history):
            return get_returns(hands, history)[player]

        current = get_current_player(history)
        info_state = information_state_tensor(hands[current], history)
        legal_mask = legal_actions_mask(get_legal_actions(history))
        strategy = self._strategy(current, info_state, legal_mask)
        if current == player:
            values = np.zeros(NUM_ACTIONS)
            for action in get_legal_actions(history):
                values[action] = self._traverse_game_tree(
                    hands, history + (action,), player
                )
            value = np.dot(strategy, values)
            regrets = np.where(legal_mask, values - value, 0.0)
            sample = Memory(info_state, self._iteration, regrets, legal_mask)
            self._advantage_memories[player].add(self._rng, sample)
            return value
        sample = Memory(info_state, self._iteration, strategy, legal_mask)
        self._strategy_memory.add(self._rng, sample)
        action = self._rng.choice(NUM_ACTIONS, p=strategy)
        return self._traverse_game_tree(hands, history + (int(action),), player)

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

    def policy_array(self, policy_params: PyTree) -> np.ndarray:
        """The policy network's (12, 2) average strategy, rows in INFOSETS order."""
        return softmax_table(self._apply(policy_params, INFOSET_TENSORS))

    def average_policy(self, policy_params: PyTree) -> Callable[[str], np.ndarray]:
        """Policy network as infoset key -> [p_pass, p_bet], for exploitability."""
        return policy_from_array(self.policy_array(policy_params))


def run_deep_cfr(
    config: argparse.Namespace,
    results_log: ResultsLog | None = None,
    seed_index: int = 0,
    run_dir: str | None = None,
) -> dict:
    """Runs Deep CFR with config.seed, evaluating the average-strategy network
    after iteration 1, every `eval_every` iterations, after iteration
    iterations // 2 and at the end. Every iteration is logged to results_log
    (if given) as seed index seed_index; with run_dir, the policy networks
    after iterations // 2 and after the last iteration are saved there as
    half/ and full/avg_policy_{seed_index}."""
    start = time.time()
    eval_seconds = 0.0
    solver = DeepCFRSolver(config)
    num_half = half_update_steps(config.iterations)
    if run_dir is not None and num_half < 1:
        raise ValueError("the half checkpoint needs at least 2 iterations")
    results = {
        "iterations": [],
        "exploitability": [],
        "advantage_loss": [],
        "policy_loss": [],
    }

    for it in range(1, config.iterations + 1):
        advantage_losses = solver.run_iteration()
        results["advantage_loss"].append(advantage_losses)
        record = {
            "iteration": it,
            "traversals": it * 2 * config.traversals,
            "advantage_loss_p0": advantage_losses[0],
            "advantage_loss_p1": advantage_losses[1],
        }
        if (
            it == 1
            or it % config.eval_every == 0
            or it == num_half
            or it == config.iterations
        ):
            eval_start = time.time()
            policy_params, loss = solver.learn_policy_network()
            if it == config.iterations:
                # Training the final policy network is part of the algorithm
                eval_start = time.time()
            table = solver.policy_array(policy_params)
            expl = exploitability(policy_from_array(table))
            value = value_vs_random(policy_from_array(table))
            record.update(
                exploitability=expl,
                value_vs_random=value,
                policy_loss=loss,
                **prefixed("game", policy_metrics(table)),
            )
            eval_seconds += time.time() - eval_start
            results["iterations"].append(it)
            results["exploitability"].append(expl)
            results["policy_loss"].append(loss)
            print(
                f"Iteration {it:4d}: exploitability = {expl:.6f}, "
                f"value vs random = {value:+.6f}"
            )
            if run_dir is not None and it in (num_half, config.iterations):
                stage = "half" if it == num_half else "full"
                save_params(run_dir, stage, "avg_policy", policy_params, seed_index)
        if results_log is not None:
            results_log.log(record, seed_index)

    results["final_exploitability"] = results["exploitability"][-1]
    results["runtime_seconds"] = time.time() - start
    # Intermediate policy networks and all exploitability computations, which a
    # run without logging doesn't need
    results["eval_seconds"] = eval_seconds
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep CFR on Kuhn Poker.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="seeds run one after the other; seed index i uses seed + i",
    )
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
        "--output",
        type=str,
        default=None,
        help="also write every seed's results to this JSON file",
    )
    parser.add_argument(
        "--save_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save the half and full policy networks",
    )
    parser.add_argument("--save_dir", type=str, default=str(REPO_ROOT / "checkpoints" / "kuhn"))
    parser.add_argument("--job_type", type=str, default="kuhn_deep_cfr")
    parser.add_argument("--wandb", action="store_true", help="log to wandb")
    parser.add_argument("--project", type=str, default="bluffjax")
    return parser.parse_args(argv)


def run(config: argparse.Namespace) -> str:
    """Runs config.num_seeds seeds one after the other with logging and
    checkpoints; returns the run directory."""
    run_config = vars(config).copy()
    run_config["half_iterations"] = half_update_steps(config.iterations)
    run_dir = make_run_dir(run_config)
    wandb_runs = []
    if config.wandb:
        import wandb

        group = "kuhn_poker" + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        wandb_runs = [
            wandb.init(
                project=config.project,
                group=group,
                job_type=config.job_type,
                name=f"{config.seed}_{i}",
                config=run_config,
                dir=str(REPO_ROOT),
                reinit="create_new",
            )
            for i in range(config.num_seeds)
        ]
    try:
        results_log = ResultsLog(run_dir, wandb_runs)
        print(f"Run directory: {run_dir}")
        print("Deep CFR for Kuhn Poker")
        all_results = []
        for i in range(config.num_seeds):
            seed_config = argparse.Namespace(**{**vars(config), "seed": config.seed + i})
            print("=" * 50)
            print(f"Seed index {i} (seed {seed_config.seed})")
            results = run_deep_cfr(
                seed_config,
                results_log,
                i,
                run_dir if config.save_checkpoints else None,
            )
            print(f"Final exploitability: {results['final_exploitability']:.6f}")
            print(
                f"Runtime: {results['runtime_seconds']:.1f}s "
                f"({results['eval_seconds']:.1f}s evaluating)"
            )
            all_results.append({"seed": seed_config.seed, **results})
        print("=" * 50)
        finals = [r["final_exploitability"] for r in all_results]
        print(
            f"Final exploitability: mean {np.mean(finals):.6f} std "
            f"{np.std(finals):.6f} over {len(finals)} seed(s)"
        )
        if config.save_checkpoints:
            print(f"Saved checkpoints to {run_dir}")
        if config.output:
            with open(config.output, "w") as f:
                json.dump(
                    {"game": "kuhn_poker", "config": run_config, "seeds": all_results},
                    f,
                    indent=2,
                )
    finally:
        for wandb_run in wandb_runs:
            wandb_run.finish()
    return run_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
