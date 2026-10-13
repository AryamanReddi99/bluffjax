"""
Counterfactual Regret Minimization (CFR) for Kuhn Poker.

CFR learns a Nash equilibrium by iteratively:
1. Computing the current strategy via regret matching on cumulative regrets
2. Traversing the game tree to compute counterfactual values
3. Updating cumulative regrets and cumulative policy

The average policy converges to a Nash equilibrium. Its exact exploitability
(the OpenSpiel-compatible metric of kuhn_exploitability) and the exact Kuhn
policy metrics (bluffjax.metrics.kuhn_poker) are logged before the first
iteration (uniform policy), every log_interval iterations and after the last.

CFR is deterministic, so a run has no seeds:

    python bluffjax/examples/kuhn/kuhn_cfr.py --iterations 100 [--wandb]

Logging: main() creates the run directory {save_dir}/{job_type}_{timestamp}/
(run_utils.make_run_dir, with config.yaml holding the flags) before running.
Every evaluation writes one record to {run_dir}/results.jsonl (with
seed_index 0 and wall_time), and to the wandb run with --wandb: iteration,
exploitability and the policy metrics of the average policy, game/<name>
(names and definitions in bluffjax.metrics.kuhn_poker).

Checkpoints (on unless --no-save_checkpoints):
    {run_dir}/half/average_policy.msgpack  average strategy after
                                           iterations // 2 iterations
    {run_dir}/full/average_policy.msgpack  after all iterations
both from the same run. Each holds the average strategy as a (12, 2) float64
array of [P(pass), P(bet)] per infoset, rows in kuhn_exploitability.INFOSETS
order (the policy array of kuhn_exploitability and bluffjax.metrics), written
with flax.serialization.to_bytes. Restore one with
run_utils.load_params(path, np.zeros((12, 2))).
"""

import argparse
import datetime
import itertools
from collections import defaultdict
from typing import Callable, NamedTuple, Sequence

import numpy as np

from bluffjax.examples.run_utils import (
    ResultsLog,
    half_update_steps,
    make_run_dir,
    save_params,
)
from bluffjax.metrics import prefixed
from bluffjax.metrics.kuhn_poker import policy_metrics
from bluffjax.utils.game_utils.kuhn_exploitability import (
    INFOSETS,
    ACTION_PASS,
    ACTION_BET,
    infoset_key,
    get_returns,
    is_terminal,
    get_legal_actions,
    get_current_player,
    exploitability,
)
from bluffjax.utils.paths import REPO_ROOT


def _regret_matching(
    cumulative_regret: dict[int, float],
    legal_actions: list[int],
) -> np.ndarray:
    """Regret matching: strategy proportional to positive regrets, else uniform."""
    sum_pos = sum(max(0.0, cumulative_regret.get(a, 0.0)) for a in legal_actions)
    probs = np.zeros(2, dtype=np.float64)
    if sum_pos > 0:
        for a in legal_actions:
            probs[a] = max(0.0, cumulative_regret.get(a, 0.0)) / sum_pos
    else:
        for a in legal_actions:
            probs[a] = 1.0 / len(legal_actions)
    return probs


def _cfr_traversal(
    hands: tuple[int, int],
    history: tuple[int, ...],
    reach_probs: np.ndarray,
    infoset_nodes: dict[str, dict],
    current_policy: dict[str, np.ndarray],
    iteration: int,
    linear_averaging: bool,
    update_player: int | None,
) -> np.ndarray:
    """
    Traverse game tree and update cumulative regrets and policy.
    reach_probs: [reach_p0, reach_p1] (probability each player reached this node)
    update_player: If set, only update regrets for this player (alternating). If None, update both.
    Returns: value for each player at this node.
    """
    if is_terminal(history):
        ret = get_returns(hands, history)
        return np.array([ret[0], ret[1]], dtype=np.float64)

    current = get_current_player(history)
    card = hands[current]
    infok = infoset_key(card, history)
    legal = get_legal_actions(history)

    if infok not in infoset_nodes:
        infoset_nodes[infok] = {
            "cumulative_regret": defaultdict(float),
            "cumulative_policy": defaultdict(float),
        }
    node = infoset_nodes[infok]
    policy_probs = current_policy[infok]

    state_value = np.zeros(2)
    child_values = {}

    for a in legal:
        new_reach = reach_probs.copy()
        new_reach[current] *= policy_probs[a]
        child_val = _cfr_traversal(
            hands,
            history + (a,),
            new_reach,
            infoset_nodes,
            current_policy,
            iteration,
            linear_averaging,
            update_player,
        )
        state_value += policy_probs[a] * child_val
        child_values[a] = child_val

    # Only update if we're updating this player (alternating) or updating all (simultaneous)
    should_update = update_player is None or current == update_player

    if should_update:
        cf_reach = 1.0
        for p in range(2):
            if p != current:
                cf_reach *= reach_probs[p]

        for a in legal:
            regret = cf_reach * (child_values[a][current] - state_value[current])
            node["cumulative_regret"][a] += regret
            if linear_averaging:
                node["cumulative_policy"][a] += (
                    iteration * reach_probs[current] * policy_probs[a]
                )
            else:
                node["cumulative_policy"][a] += reach_probs[current] * policy_probs[a]

    return state_value


def _run_cfr_iteration(
    infoset_nodes: dict[str, dict],
    current_policy: dict[str, np.ndarray],
    iteration: int,
    linear_averaging: bool,
    alternating_updates: bool,
) -> float:
    """One CFR iteration: update policy from regrets, then traverse (alternating or simultaneous)."""
    total_value = 0.0

    if alternating_updates:
        for player in range(2):
            for infok in INFOSETS:
                if infok in infoset_nodes:
                    node = infoset_nodes[infok]
                    legal = [ACTION_PASS, ACTION_BET]
                    current_policy[infok] = _regret_matching(
                        dict(node["cumulative_regret"]),
                        legal,
                    )
            for hands in itertools.permutations([0, 1, 2], 2):
                reach = np.ones(2, dtype=np.float64)
                val = _cfr_traversal(
                    hands,
                    (),
                    reach,
                    infoset_nodes,
                    current_policy,
                    iteration,
                    linear_averaging,
                    player,
                )
                total_value += val[0]
    else:
        for infok in INFOSETS:
            if infok in infoset_nodes:
                node = infoset_nodes[infok]
                legal = [ACTION_PASS, ACTION_BET]
                current_policy[infok] = _regret_matching(
                    dict(node["cumulative_regret"]),
                    legal,
                )
        for hands in itertools.permutations([0, 1, 2], 2):
            reach = np.ones(2, dtype=np.float64)
            val = _cfr_traversal(
                hands,
                (),
                reach,
                infoset_nodes,
                current_policy,
                iteration,
                linear_averaging,
                None,
            )
            total_value += val[0]

    return total_value / 6


def _average_policy_from_nodes(
    infoset_nodes: dict[str, dict],
) -> Callable[[str], np.ndarray]:
    """Build average policy from cumulative policy in infoset nodes."""

    def policy(infok: str) -> np.ndarray:
        if infok not in infoset_nodes:
            return np.array([0.5, 0.5], dtype=np.float64)
        node = infoset_nodes[infok]
        cum = node["cumulative_policy"]
        total = sum(cum.values())
        if total <= 0:
            return np.array([0.5, 0.5], dtype=np.float64)
        probs = np.zeros(2, dtype=np.float64)
        for a in [ACTION_PASS, ACTION_BET]:
            probs[a] = cum.get(a, 0.0) / total
        return probs

    return policy


def policy_array(policy: Callable[[str], np.ndarray]) -> np.ndarray:
    """(12, 2) table [P(pass), P(bet)] of a policy, rows in INFOSETS order."""
    return np.stack([policy(k) for k in INFOSETS])


class CFRResult(NamedTuple):
    average_policy: Callable[[str], np.ndarray]  # after all iterations
    exploitability_log: list[tuple[int, float]]  # (iteration, exploitability)
    half_policy: np.ndarray  # (12, 2) average strategy after num_iterations // 2
    full_policy: np.ndarray  # (12, 2) average strategy after num_iterations


def run_cfr(
    num_iterations: int = 10_000,
    linear_averaging: bool = False,
    alternating_updates: bool = True,
    log_interval: int = 100,
    results_log: ResultsLog | None = None,
) -> CFRResult:
    """
    Run CFR on Kuhn poker.

    Args:
        num_iterations: Number of CFR iterations.
        linear_averaging: If True, use linear (iteration-weighted) averaging.
        alternating_updates: If True, alternate updates between players (faster convergence).
        log_interval: Log exploitability every this many iterations.
        results_log: if given, every evaluation is logged to it (iteration,
            exploitability and the game/ policy metrics).

    Returns:
        CFRResult: the average policy after all iterations, the
        (iteration, exploitability) log, and the average strategy tables after
        half_update_steps(num_iterations) and after all iterations.
    """
    infoset_nodes: dict[str, dict] = {}
    current_policy: dict[str, np.ndarray] = {}

    # Initialize with uniform
    for infok in INFOSETS:
        current_policy[infok] = np.array([0.5, 0.5], dtype=np.float64)
        infoset_nodes[infok] = {
            "cumulative_regret": defaultdict(float),
            "cumulative_policy": defaultdict(float),
        }

    exploitability_log: list[tuple[int, float]] = []
    num_half = half_update_steps(num_iterations)

    def evaluate(it: int) -> float:
        avg_policy = _average_policy_from_nodes(infoset_nodes)
        expl = exploitability(avg_policy)
        exploitability_log.append((it, expl))
        if results_log is not None:
            results_log.log(
                {
                    "iteration": it,
                    "exploitability": expl,
                    **prefixed("game", policy_metrics(policy_array(avg_policy))),
                }
            )
        return expl

    # Log initial (uniform) exploitability
    expl_0 = evaluate(0)
    print(f"Iteration      0: exploitability = {expl_0:.6f} (uniform)")
    half_policy = policy_array(_average_policy_from_nodes(infoset_nodes))

    for it in range(1, num_iterations + 1):
        _run_cfr_iteration(
            infoset_nodes,
            current_policy,
            it,
            linear_averaging,
            alternating_updates,
        )

        if it == num_half:
            half_policy = policy_array(_average_policy_from_nodes(infoset_nodes))
        if it % log_interval == 0 or it == num_iterations:
            expl = evaluate(it)
            print(f"Iteration {it:6d}: exploitability = {expl:.6f}")

    average_policy = _average_policy_from_nodes(infoset_nodes)
    return CFRResult(
        average_policy=average_policy,
        exploitability_log=exploitability_log,
        half_policy=half_policy,
        full_policy=policy_array(average_policy),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CFR on Kuhn Poker.")
    parser.add_argument("--iterations", type=int, default=100, help="CFR iterations")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="evaluate every this many iterations (and after the last)",
    )
    parser.add_argument(
        "--save_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save the half and full average strategy tables",
    )
    parser.add_argument("--save_dir", type=str, default=str(REPO_ROOT / "checkpoints" / "kuhn"))
    parser.add_argument("--job_type", type=str, default="kuhn_cfr")
    parser.add_argument("--wandb", action="store_true", help="log to wandb")
    parser.add_argument("--project", type=str, default="bluffjax")
    return parser.parse_args(argv)


def run(config: argparse.Namespace) -> str:
    """Runs CFR with logging and checkpoints; returns the run directory."""
    run_config = vars(config).copy()
    run_config["half_iterations"] = half_update_steps(config.iterations)
    run_dir = make_run_dir(run_config)
    wandb_runs = []
    if config.wandb:
        import wandb

        wandb_runs = [
            wandb.init(
                project=config.project,
                group="kuhn_poker" + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S"),
                job_type=config.job_type,
                name="cfr",
                config=run_config,
                dir=str(REPO_ROOT),
                reinit="create_new",
            )
        ]
    try:
        results_log = ResultsLog(run_dir, wandb_runs)
        print(f"Run directory: {run_dir}")
        print("CFR for Kuhn Poker")
        print("=" * 50)
        result = run_cfr(
            num_iterations=config.iterations,
            linear_averaging=False,
            log_interval=config.log_interval,
            results_log=results_log,
        )
        final_expl = float(result.exploitability_log[-1][1])
        print("=" * 50)
        print(f"Final exploitability: {final_expl:.6f} (expected ~0 at Nash)")
        if config.save_checkpoints:
            save_params(run_dir, "half", "average_policy", result.half_policy)
            save_params(run_dir, "full", "average_policy", result.full_policy)
            print(f"Saved checkpoints to {run_dir}")
    finally:
        for wandb_run in wandb_runs:
            wandb_run.finish()
    return run_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
