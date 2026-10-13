"""
Counterfactual Regret Minimization (CFR) for Kuhn Poker.

CFR learns a Nash equilibrium by iteratively:
1. Computing the current strategy via regret matching on cumulative regrets
2. Traversing the game tree to compute counterfactual values
3. Updating cumulative regrets and cumulative policy

The average policy converges to a Nash equilibrium. Its exact exploitability
(the OpenSpiel-compatible metric of kuhn_exploitability), its exact expected
return per hand against uniformly random play averaged over both seats
(kuhn_exploitability.value_vs_random) and the exact Kuhn policy metrics
(bluffjax.metrics.kuhn_poker) are logged after iteration 1, every
--eval_interval iterations, after half of the iterations (the half
checkpoint) and after the last one.

CFR is deterministic, so a run is one solve (seed index 0, one wandb run
"{seed}_0"; --seed only names it):

    python bluffjax/examples/kuhn/kuhn_cfr.py --iterations 100 [--wandb]

Flags (the same as leduc_cfr.py; kuhn_deep_cfr.py and leduc_deep_cfr.py add
--num_seeds and their own hyperparameters): --iterations, --eval_interval,
--seed, --save_dir, --job_type, --wandb, --project, --save_checkpoints. With
--wandb the run logs to project --project, group {env_name}_{timestamp}, job
type {job_type}_{env_name} (env_name kuhn_poker), as the RL scripts do.

Logging: main() creates the run directory {save_dir}/{job_type}_{timestamp}/
(run_utils.make_run_dir) before running; config.yaml holds the flags with
env_name, num_seeds (1), half_iterations and run_dir. Every evaluation writes
one record to {run_dir}/results.jsonl (with seed_index 0 and wall_time) and to
the wandb run: iteration, exploitability, value_vs_random and the policy
metrics of the average policy, game/<name> (names and definitions in
bluffjax.metrics.kuhn_poker).

Checkpoints (unless --no-save_checkpoints):
    {run_dir}/half/average_policy.msgpack  average strategy after
                                           half_iterations =
                                           half_update_steps(iterations)
    {run_dir}/full/average_policy.msgpack  after all iterations
both from the same run. Each checkpoint is a dict of numpy arrays, one row per
information set (the 12 infosets, in kuhn_exploitability.INFOSETS order),
written with run_utils.save_params, as leduc_cfr.py's checkpoints:
    policy        float64 (12, 2)  average strategy [P(pass), P(bet)], rows
                                   sum to 1 (uniform where the strategy sum
                                   is 0)
    player        int32 (12,)      the acting player's position, 0 = acts first
    private_card  int32 (12,)      its card 0..2 = J, Q, K
    betting       int32 (12, 2)    the actions so far (0 = pass, 1 = bet,
                                   -1 = none)
    legal         bool (12, 2)     legal actions (pass, bet): all of them
    iterations    int64 ()         CFR iterations behind the strategy
player, private_card and betting identify the infoset (its INFOSETS key is
the card followed by the actions as p / b). load_average_policy(path) reads a
checkpoint back as the (12, 2) policy array in INFOSETS order (the policy
array of kuhn_exploitability and bluffjax.metrics), matching rows by these
fields. In the env, the acting player's infoset is (position
(current_player_idx - start_player_idx) % 2, agent_hands[current_player_idx],
the actions of the hand so far).
"""

import argparse
import datetime
import itertools
from collections import defaultdict
from typing import Callable, NamedTuple, Sequence

import numpy as np
import wandb

from bluffjax.examples.run_utils import (
    ResultsLog,
    half_update_steps,
    load_params,
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
    value_vs_random,
)
from bluffjax.utils.paths import REPO_ROOT

ENV_NAME = "kuhn_poker"
MAX_BETTING = 2  # actions before a decision: at most pass, bet


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


# --- Checkpoints of the average strategy (layout in the module docstring) ---
def _history(infoset: str) -> tuple[int, ...]:
    """The actions of an INFOSETS key, e.g. "1pb" -> (pass, bet)."""
    return tuple(ACTION_PASS if c == "p" else ACTION_BET for c in infoset[1:])


def _infoset_fields() -> dict[str, np.ndarray]:
    """player, private_card and betting of every infoset."""
    n = len(INFOSETS)
    fields = {
        "player": np.zeros(n, dtype=np.int32),
        "private_card": np.zeros(n, dtype=np.int32),
        "betting": np.full((n, MAX_BETTING), -1, dtype=np.int32),
    }
    for i, key in enumerate(INFOSETS):
        history = _history(key)
        fields["player"][i] = get_current_player(history)
        fields["private_card"][i] = int(key[0])
        fields["betting"][i, : len(history)] = history
    return fields


def policy_checkpoint(policy: np.ndarray, iterations: int) -> dict[str, np.ndarray]:
    """The checkpoint tree of a (12, 2) policy in INFOSETS order."""
    legal = np.zeros((len(INFOSETS), 2), dtype=bool)
    for i, key in enumerate(INFOSETS):
        legal[i, get_legal_actions(_history(key))] = True
    return {
        "policy": np.asarray(policy, dtype=np.float64),
        **_infoset_fields(),
        "legal": legal,
        "iterations": np.asarray(iterations, dtype=np.int64),
    }


def load_average_policy(path: str) -> np.ndarray:
    """A checkpoint written by run() as the (12, 2) policy in INFOSETS order,
    matching rows by their infoset fields (strict: every infoset once)."""
    template = policy_checkpoint(np.zeros((len(INFOSETS), 2)), 0)
    tree = load_params(path, template)
    row_of = {
        (get_current_player(_history(key)), int(key[0]), _history(key)): i
        for i, key in enumerate(INFOSETS)
    }
    assert len(row_of) == len(INFOSETS)
    policy = np.full((len(row_of), 2), np.nan)
    for j in range(len(row_of)):
        key = (
            int(tree["player"][j]),
            int(tree["private_card"][j]),
            tuple(int(a) for a in tree["betting"][j] if a >= 0),
        )
        i = row_of.pop(key)  # KeyError for an unknown or repeated infoset
        policy[i] = tree["policy"][j]
    return policy


class CFRResult(NamedTuple):
    average_policy: Callable[[str], np.ndarray]  # after all iterations
    exploitability_log: list[tuple[int, float]]  # (iteration, exploitability)
    half_policy: np.ndarray  # (12, 2) average strategy after half_update_steps
    full_policy: np.ndarray  # (12, 2) average strategy after num_iterations


def run_cfr(
    num_iterations: int = 10_000,
    linear_averaging: bool = False,
    alternating_updates: bool = True,
    eval_interval: int = 100,
    results_log: ResultsLog | None = None,
) -> CFRResult:
    """
    Run CFR on Kuhn poker.

    Args:
        num_iterations: Number of CFR iterations.
        linear_averaging: If True, use linear (iteration-weighted) averaging.
        alternating_updates: If True, alternate updates between players (faster convergence).
        eval_interval: evaluate after iteration 1, every this many
            iterations, after half_update_steps(num_iterations) and after the
            last.
        results_log: if given, every evaluation is logged to it (iteration,
            exploitability, value_vs_random and the game/ policy metrics).

    Returns:
        CFRResult: the average policy after all iterations, the
        (iteration, exploitability) log, and the average strategy tables after
        half_update_steps(num_iterations) (the uniform policy if that is 0)
        and after all iterations.
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

    def evaluate(it: int) -> None:
        avg_policy = _average_policy_from_nodes(infoset_nodes)
        expl = exploitability(avg_policy)
        value = value_vs_random(avg_policy)
        exploitability_log.append((it, expl))
        if results_log is not None:
            results_log.log(
                {
                    "iteration": it,
                    "exploitability": expl,
                    "value_vs_random": value,
                    **prefixed("game", policy_metrics(policy_array(avg_policy))),
                }
            )
        print(
            f"Iteration {it:6d}: exploitability = {expl:.6f}, "
            f"value vs random = {value:+.6f}",
            flush=True,
        )

    # The uniform policy, until num_half >= 1 iterations have run
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
        if (
            it == 1
            or it % eval_interval == 0
            or it == num_half
            or it == num_iterations
        ):
            evaluate(it)

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
        "--eval_interval",
        type=int,
        default=10,
        help="evaluate after iteration 1, every this many iterations, after "
        "half of them and after the last",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="names the run (CFR is deterministic)"
    )
    parser.add_argument(
        "--save_dir", type=str, default=str(REPO_ROOT / "checkpoints" / "kuhn")
    )
    parser.add_argument("--job_type", type=str, default="kuhn_cfr")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--project", type=str, default="bluffjax")
    parser.add_argument(
        "--save_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="average strategy after half and all of the iterations",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> str:
    """Runs CFR with logging and checkpoints; returns the run directory."""
    config = {
        **vars(args),
        "env_name": ENV_NAME,
        "num_seeds": 1,
        "half_iterations": half_update_steps(args.iterations),
    }
    run_dir = make_run_dir(config)
    group = f"{ENV_NAME}_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb_runs = [
        wandb.init(
            project=args.project,
            group=group,
            job_type=f"{args.job_type}_{ENV_NAME}",
            name=f"{args.seed}_0",  # one deterministic run (seed index 0)
            config=config,
            mode=None if args.wandb else "disabled",
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
            num_iterations=args.iterations,
            linear_averaging=False,
            eval_interval=args.eval_interval,
            results_log=results_log,
        )
        final_expl = float(result.exploitability_log[-1][1])
        print("=" * 50)
        print(f"Final exploitability: {final_expl:.6f} (expected ~0 at Nash)")
        final_value = value_vs_random(result.average_policy)
        print(
            f"Final value vs random: {final_value:+.6f} chips per hand (Nash "
            f"equilibria: 1/9 + alpha/6, between 0.1111 and 0.1667)"
        )
        if args.save_checkpoints:
            for stage, policy, iterations in (
                ("half", result.half_policy, config["half_iterations"]),
                ("full", result.full_policy, args.iterations),
            ):
                tree = policy_checkpoint(policy, iterations)
                save_params(run_dir, stage, "average_policy", tree)
            print(f"Saved checkpoints to {run_dir}")
    finally:
        for wandb_run in wandb_runs:
            wandb_run.finish()
    return run_dir


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
