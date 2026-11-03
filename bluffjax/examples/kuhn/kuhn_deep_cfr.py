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

Single Deep CFR (--sd_cfr; Steinberger 2019, https://arxiv.org/abs/1901.07621):
also computes, exactly at every infoset, the linearly weighted average of the
advantage networks' regret-matching strategies, each weighted by the player's
own reach probability (bluffjax.examples.sd_cfr; leduc_deep_cfr.py describes
it), and logs it as sd_cfr/exploitability and sd_cfr/value_vs_random and saves
it as {half,full}/sd_cfr_policy_{i}.msgpack (kuhn_cfr.py's average-policy
layout, read with kuhn_cfr.load_average_policy).
--strategy_memory_capacity sets the strategy memory's size separately
(default: --memory_capacity).

OpenSpiel's JAX and PyTorch versions skip training while a memory holds fewer
samples than a batch; here the networks are trained on every iteration, with
minibatches sampled with replacement.

Defaults (the best settings of the 31 Oct 2026 Leduc push): 1,000 iterations of
1,500 traversals per player, 750 advantage-network steps, advantage memories
of 2,000,000 and a strategy memory of 10,000,000 samples (the published Leduc
scale), and Single Deep CFR's exact average strategy as well (--sd_cfr); 2 hidden
layers of 128 units, the network size of every baseline; Adam with learning rate 1e-3 and gradient norms clipped to 1,
as in the Deep CFR paper (Section 5.2); batches of 2048 and 5000
policy-network steps, as in OpenSpiel's Deep CFR examples.

Network inputs are OpenSpiel's kuhn_poker information_state_tensor, which is
perfect recall: each of the 12 infosets has a distinct tensor. Exploitability
is exact.

Seeds: --num_seeds runs (default 5) one after the other in one run
directory; seed index i uses seed --seed + i and has its own wandb run
"{seed}_{i}".

Flags shared with kuhn_cfr.py, leduc_cfr.py and leduc_deep_cfr.py, with the
same meaning: --iterations, --eval_interval, --seed, --num_seeds (Deep CFR),
--save_dir, --job_type, --wandb, --project, --save_checkpoints. With --wandb
the runs log to project --project, group {env_name}_{timestamp}, job type
{job_type}_{env_name} (env_name kuhn_poker), as the RL scripts do.

Evaluation: the average-strategy (policy) network is trained after iteration
1, every --eval_interval iterations, after half_iterations =
half_update_steps(iterations) iterations (the half checkpoint) and after the
last; each time its exact exploitability, its exact expected return per hand
against uniformly random play averaged over both seats (value_vs_random,
kuhn_exploitability.value_vs_random) and the exact Kuhn policy metrics
(bluffjax.metrics.kuhn_poker) are logged. Its training key depends only on the
iteration, so these intermediate networks don't change the run.

Logging: main() creates the run directory {save_dir}/{job_type}_{timestamp}/
(run_utils.make_run_dir) before running; config.yaml holds the flags with
env_name, half_iterations and run_dir. Every iteration writes one record per
seed to {run_dir}/results.jsonl (with seed_index and wall_time) and to the
seed's wandb run: iteration, traversals (the traversals so far, both players)
and each player's final advantage-network loss; at the evaluations also
exploitability, value_vs_random, policy_loss and the policy metrics of the
policy network, game/<name> (names and definitions in
bluffjax.metrics.kuhn_poker).

Checkpoints (on unless --no-save_checkpoints), for every seed index i:
    {run_dir}/half/avg_policy_{i}.msgpack  policy network trained after
                                           half_iterations iterations
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
import wandb

from bluffjax.examples.run_utils import (
    ResultsLog,
    half_update_steps,
    make_run_dir,
    save_params,
)
from bluffjax.examples.kuhn import kuhn_cfr
from bluffjax.examples.sd_cfr import SDCFRAverage, regret_matching_table
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

ENV_NAME = "kuhn_poker"
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
INFOSET_LEGAL_MASKS = np.ones((len(INFOSETS), NUM_ACTIONS), dtype=bool)
# Acting player of every infoset (players alternate: 0 acts at even depths)
INFOSET_PLAYERS = np.array([(len(k) - 1) % 2 for k in INFOSETS])


def _own_parents() -> tuple[np.ndarray, np.ndarray]:
    """For every infoset, the row of its player's previous infoset (two
    actions earlier) and the action taken there; -1 and 0 at the player's
    first decision."""
    row = {k: i for i, k in enumerate(INFOSETS)}
    parent = np.full(len(INFOSETS), -1, dtype=np.int64)
    parent_action = np.zeros(len(INFOSETS), dtype=np.int64)
    for i, k in enumerate(INFOSETS):
        if len(k) >= 3:
            parent[i] = row[k[:-2]]
            parent_action[i] = ACTION_PASS if k[-2] == "p" else ACTION_BET
    return parent, parent_action


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
        strategy_capacity = getattr(config, "strategy_memory_capacity", None)
        self._strategy_memory = ReservoirBuffer(
            config.memory_capacity if strategy_capacity is None else strategy_capacity
        )
        self.sd_cfr = (
            SDCFRAverage(INFOSET_PLAYERS, *_own_parents(), INFOSET_LEGAL_MASKS)
            if getattr(config, "sd_cfr", False)
            else None
        )
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
            if self.sd_cfr is not None:
                # The opponent's strategy, as Deep CFR's strategy memory gets it
                opponent = 1 - player
                self.sd_cfr.add(
                    opponent, self.strategy_table(opponent), self._iteration
                )
            for _ in range(self._config.traversals):
                # The deal is Kuhn's only chance event
                hands = DEALS[self._rng.integers(len(DEALS))]
                self._traverse_game_tree(hands, (), player)
            losses.append(self._learn_advantage_network(player))
        return losses

    def strategy_table(self, player: int) -> np.ndarray:
        """Regret matching on `player`'s current advantage network at every
        infoset, (12, 2) (meaningful on `player`'s rows)."""
        params = self._advantage_params[player]
        if params is None:
            advantages = np.zeros(INFOSET_LEGAL_MASKS.shape)
        else:
            advantages = np.asarray(self._apply(params, INFOSET_TENSORS))
        return regret_matching_table(advantages, INFOSET_LEGAL_MASKS)

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
    seed_index: int = 0,
    results_log: ResultsLog | None = None,
    run_dir: str | None = None,
) -> dict:
    """Runs Deep CFR with seed config.seed + seed_index, evaluating the
    average-strategy network after iteration 1, every `eval_interval`
    iterations, after half_update_steps(iterations) and at the end. Every
    iteration is logged to results_log (if given) as seed index seed_index;
    with run_dir, the policy networks after half and all of the iterations
    are saved there as {half,full}/avg_policy_{seed_index}."""
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
    if solver.sd_cfr is not None:
        results["sd_cfr_exploitability"] = []

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
            or it % config.eval_interval == 0
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
            if solver.sd_cfr is not None:
                sd_table = solver.sd_cfr.policy()
                sd_expl = exploitability(policy_from_array(sd_table))
                record["sd_cfr/exploitability"] = sd_expl
                record["sd_cfr/value_vs_random"] = value_vs_random(
                    policy_from_array(sd_table)
                )
                results["sd_cfr_exploitability"].append(sd_expl)
                print(
                    f"seed index {seed_index} iteration {it:4d}: SD-CFR "
                    f"exploitability = {sd_expl:.6f}",
                    flush=True,
                )
            eval_seconds += time.time() - eval_start
            results["iterations"].append(it)
            results["exploitability"].append(expl)
            results["policy_loss"].append(loss)
            print(
                f"seed index {seed_index} iteration {it:4d}: exploitability = "
                f"{expl:.6f}, value vs random = {value:+.6f}",
                flush=True,
            )
            if run_dir is not None and it in (num_half, config.iterations):
                stage = "half" if it == num_half else "full"
                save_params(run_dir, stage, "avg_policy", policy_params, seed_index)
                if solver.sd_cfr is not None:
                    tree = kuhn_cfr.policy_checkpoint(solver.sd_cfr.policy(), it)
                    save_params(run_dir, stage, "sd_cfr_policy", tree, seed_index)
        if results_log is not None:
            results_log.log(record, seed_index)

    results["final_exploitability"] = results["exploitability"][-1]
    if solver.sd_cfr is not None:
        results["final_sd_cfr_exploitability"] = results["sd_cfr_exploitability"][-1]
    results["runtime_seconds"] = time.time() - start
    # Intermediate policy networks and all exploitability computations, which a
    # run without logging doesn't need
    results["eval_seconds"] = eval_seconds
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep CFR on Kuhn Poker.")
    parser.add_argument("--iterations", type=int, default=1000, help="CFR iterations T")
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=250,
        help="evaluate after iteration 1, every this many iterations, after "
        "half of them and after the last",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="runs one after the other; seed index i uses seed --seed + i",
    )
    parser.add_argument(
        "--traversals",
        type=int,
        default=1500,
        help="traversals K per player per iteration",
    )
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="0 disables clipping"
    )
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--advantage_train_steps", type=int, default=750)
    parser.add_argument("--policy_train_steps", type=int, default=5000)
    parser.add_argument(
        "--memory_capacity",
        type=int,
        default=2_000_000,
        help="size of each advantage memory (and of the strategy memory unless "
        "--strategy_memory_capacity is given)",
    )
    parser.add_argument("--strategy_memory_capacity", type=int, default=10_000_000)
    parser.add_argument(
        "--sd_cfr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also compute Single Deep CFR's average strategy (exact, every "
        "infoset) and log/save it",
    )
    parser.add_argument(
        "--save_dir", type=str, default=str(REPO_ROOT / "checkpoints" / "kuhn")
    )
    parser.add_argument("--job_type", type=str, default="kuhn_deep_cfr")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--project", type=str, default="bluffjax")
    parser.add_argument(
        "--save_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="policy network after half and all of the iterations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="also write every seed's results to this JSON file",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> str:
    """Runs args.num_seeds seeds one after the other with logging and
    checkpoints; returns the run directory."""
    config = {
        **vars(args),
        "env_name": ENV_NAME,
        "half_iterations": half_update_steps(args.iterations),
    }
    run_dir = make_run_dir(config)
    group = f"{ENV_NAME}_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb_runs = [
        wandb.init(
            project=args.project,
            group=group,
            job_type=f"{args.job_type}_{ENV_NAME}",
            name=f"{args.seed}_{i}",
            config=config,
            mode=None if args.wandb else "disabled",
            dir=str(REPO_ROOT),
            reinit="create_new",
        )
        for i in range(args.num_seeds)
    ]
    try:
        results_log = ResultsLog(run_dir, wandb_runs)
        print(f"Run directory: {run_dir}")
        print("Deep CFR for Kuhn Poker")
        all_results = []
        for i in range(args.num_seeds):
            print("=" * 50)
            print(f"seed index {i} (seed {args.seed + i})")
            results = run_deep_cfr(
                args, i, results_log, run_dir if args.save_checkpoints else None
            )
            all_results.append({"seed": args.seed + i, "seed_index": i, **results})
            print(f"Final exploitability: {results['final_exploitability']:.6f}")
            if "final_sd_cfr_exploitability" in results:
                print(
                    "Final SD-CFR exploitability: "
                    f"{results['final_sd_cfr_exploitability']:.6f}"
                )
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
        if args.save_checkpoints:
            print(f"Saved checkpoints to {run_dir}")
        if args.output:
            with open(args.output, "w") as f:
                json.dump(
                    {"game": ENV_NAME, "config": config, "seeds": all_results},
                    f,
                    indent=2,
                )
    finally:
        for wandb_run in wandb_runs:
            wandb_run.finish()
    return run_dir


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
