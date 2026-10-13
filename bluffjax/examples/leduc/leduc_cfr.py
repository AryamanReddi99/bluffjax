"""CFR on Leduc Poker.

Vanilla tabular CFR (full-tree traversals) with alternating updates, as
OpenSpiel's CFRSolver: in every iteration player 0 and then player 1 traverse
the tree. Each traversal plays the current strategies (regret matching on the
cumulative regrets at the start of the traversal, so player 1's traversal sees
player 0's strategy updated in the same iteration), updates the traversing
player's regrets and adds its strategy, weighted by its own reach probability,
to its average strategy.

Logs, after iteration 1, every --log_every iterations, after half of the
iterations and after the last one: the exact exploitability of the average
strategy and its exact behaviour metrics in self-play as game/<name>
(bluffjax.metrics.leduc_holdem: bluff rate and success, correct-fold rate,
round-2 raise rates with and without a pair; definitions there). Each record
has the iteration count. CFR is deterministic, so a run is one solve (seed
index 0, one wandb run).

Run directory ({save_dir}/{job_type}_{timestamp}/, made when the run starts;
see bluffjax.examples.run_utils):
    config.yaml                     the arguments, with half_iterations
    results.jsonl                   every logged record (what wandb gets, plus
                                    seed_index and wall_time)
    half/average_policy.msgpack     average strategy after half_iterations =
                                    half_update_steps(iterations) iterations
    full/average_policy.msgpack     average strategy after all iterations
(checkpoints unless --no-save_checkpoints). Each checkpoint is a dict of numpy
arrays, one row per information set (the 936 infosets, in the order of
leduc_exploitability._INFOSET_LIST), written with run_utils.save_params:
    policy        float64 (936, 3)  average strategy: P(fold), P(call),
                                    P(raise), 0 on illegal actions, rows sum
                                    to 1 (uniform where the strategy sum is 0)
    player        int32 (936,)      the acting player's position, 0 = acts
                                    first in both rounds
    private_card  int32 (936,)      its card 0..5 = J1 J2 Q1 Q2 K1 K2 (rank =
                                    card // 2)
    public_card   int32 (936,)      the public card 0..5, -1 in round 1
    betting       int32 (936, 2, 4) the actions so far in rounds 1 and 2
                                    (1 = call/check, 2 = raise/bet, -1 = none)
    legal         bool (936, 3)     legal actions (fold, call, raise)
    iterations    int64 ()          CFR iterations behind the strategy
player, private_card, public_card and betting identify the infoset: they are
the first five entries of its key leduc_exploitability.LeducState.
info_state_key (the rest is derived from them), with both rounds' sequences
padded. load_average_policy(path) reads a checkpoint back as a (936, 3) policy
in _INFOSET_LIST order, matching rows by these fields. In the env, the acting
player's infoset is (position (current_player_idx - start_player_idx) % 2,
agent_cards[current_player_idx], public_card, history[0], history[1]) with the
history's 0 entries (no action) as -1.
"""

from __future__ import annotations

import argparse
import datetime
from collections import defaultdict

import numpy as np
import wandb

from bluffjax.examples.run_utils import (
    ResultsLog,
    half_update_steps,
    load_params,
    make_run_dir,
    save_params,
)
from bluffjax.metrics.leduc_holdem import log_metrics
from bluffjax.utils.game_utils import leduc_exploitability as leduc
from bluffjax.utils.paths import REPO_ROOT


class CFRSolver:
    def __init__(self) -> None:
        # regrets[player][infoset_key][action] -> cumulative regret
        self.regrets = [defaultdict(lambda: defaultdict(float)) for _ in range(2)]
        # strategy_sum[player][infoset_key][action] -> cumulative average-strategy mass
        self.strategy_sum = [defaultdict(lambda: defaultdict(float)) for _ in range(2)]
        # Current strategy of every infoset visited in the ongoing traversal.
        self._strategy_cache: dict[tuple, dict[int, float]] = {}

    @staticmethod
    def _uniform_over(actions: tuple[int, ...]) -> dict[int, float]:
        p = 1.0 / len(actions)
        return {a: p for a in actions}

    def _regret_matching(
        self, player: int, infoset_key: tuple, legal: tuple[int, ...]
    ) -> dict[int, float]:
        regrets = self.regrets[player][infoset_key]
        positive = np.array([max(regrets[a], 0.0) for a in legal], dtype=np.float64)
        if positive.sum() <= 0:
            return self._uniform_over(legal)
        probs = positive / positive.sum()
        return {a: float(probs[i]) for i, a in enumerate(legal)}

    def _current_strategy(
        self, player: int, infoset_key: tuple, legal: tuple[int, ...]
    ) -> dict[int, float]:
        """Regret matching on the regrets at the start of the traversal.

        An infoset is visited once per history in it, and its regrets change
        after the first visit, so the strategy is computed once and cached.
        """
        cache_key = (player, infoset_key)
        if cache_key not in self._strategy_cache:
            self._strategy_cache[cache_key] = self._regret_matching(
                player, infoset_key, legal
            )
        return self._strategy_cache[cache_key]

    def _cfr(
        self, state: leduc.LeducState, update_player: int, reach: tuple[float, float]
    ) -> float:
        """Expected value for update_player of the current strategies at state.

        reach holds both players' reach probabilities without chance. The chance
        probability of reaching a history is the same for every history of an
        infoset in Leduc, so leaving it out scales an infoset's regrets by a
        constant, which regret matching ignores.
        """
        if state.is_terminal():
            return state.returns()[update_player]

        if state.is_chance_node():
            v = 0.0
            for action, prob in state.chance_outcomes():
                v += prob * self._cfr(state.child(action), update_player, reach)
            return v

        current = state.current_player
        legal = state.legal_actions()
        infoset_key = state.info_state_key(current)
        strategy = self._current_strategy(current, infoset_key, legal)

        action_values: dict[int, float] = {}
        node_value = 0.0
        for action in legal:
            next_reach = list(reach)
            next_reach[current] *= strategy[action]
            val = self._cfr(
                state.child(action), update_player, (next_reach[0], next_reach[1])
            )
            action_values[action] = val
            node_value += strategy[action] * val

        if current == update_player:
            opp = 1 - current
            for action in legal:
                regret = action_values[action] - node_value
                self.regrets[current][infoset_key][action] += reach[opp] * regret
                self.strategy_sum[current][infoset_key][action] += (
                    reach[current] * strategy[action]
                )
        return node_value

    def run_iteration(self) -> None:
        root = leduc.initial_state()
        for p in (0, 1):
            self._strategy_cache = {}
            self._cfr(root, p, (1.0, 1.0))
        self._strategy_cache = {}

    def average_policy(self, state: leduc.LeducState) -> dict[int, float]:
        legal = state.legal_actions()
        if not legal:
            return {}
        if state.is_chance_node():
            return {a: p for a, p in state.chance_outcomes()}

        player = state.current_player
        infoset_key = state.info_state_key(player)
        sums = self.strategy_sum[player][infoset_key]
        total = sum(sums[a] for a in legal)
        if total <= 0:
            return self._uniform_over(legal)
        return {a: sums[a] / total for a in legal}

    def average_policy_array(self) -> np.ndarray:
        """The average strategy at every infoset, (936, 3) in
        leduc._INFOSET_LIST order (the representation of
        leduc.exploitability_from_policy_array and bluffjax.metrics.leduc_holdem);
        the same probabilities as average_policy."""
        policy = np.zeros((len(leduc._INFOSET_LIST), 3), dtype=np.float64)
        for i, (key, legal) in enumerate(leduc._INFOSET_LIST):
            sums = self.strategy_sum[key[0]].get(key, {})
            total = sum(sums.get(a, 0.0) for a in legal)
            for a in legal:
                policy[i, a] = sums.get(a, 0.0) / total if total > 0 else 1 / len(legal)
        return policy

    def nash_conv(self) -> float:
        root = leduc.initial_state()
        on_policy = leduc._state_values(root, self.average_policy)
        br0 = leduc.BestResponseSolver(root, 0, self.average_policy).value(root)
        br1 = leduc.BestResponseSolver(root, 1, self.average_policy).value(root)
        return (br0 - on_policy[0]) + (br1 - on_policy[1])

    def exploitability(self) -> float:
        return self.nash_conv() / 2.0


# --- Checkpoints of the average strategy (layout in the module docstring) ---
def _infoset_fields() -> dict[str, np.ndarray]:
    """player, private_card, public_card and betting of every infoset."""
    n = len(leduc._INFOSET_LIST)
    fields = {
        "player": np.zeros(n, dtype=np.int32),
        "private_card": np.zeros(n, dtype=np.int32),
        "public_card": np.zeros(n, dtype=np.int32),
        "betting": np.full((n, 2, 4), -1, dtype=np.int32),
    }
    for i, (key, _) in enumerate(leduc._INFOSET_LIST):
        player, private_card, public_card, round1, round2 = key[:5]
        fields["player"][i] = player
        fields["private_card"][i] = private_card
        fields["public_card"][i] = public_card
        for r, sequence in enumerate((round1, round2)):
            fields["betting"][i, r, : len(sequence)] = sequence
    return fields


def policy_checkpoint(policy: np.ndarray, iterations: int) -> dict[str, np.ndarray]:
    """The checkpoint tree of a (936, 3) policy in _INFOSET_LIST order."""
    legal = np.zeros((len(leduc._INFOSET_LIST), 3), dtype=bool)
    for i, (_, actions) in enumerate(leduc._INFOSET_LIST):
        legal[i, list(actions)] = True
    return {
        "policy": np.asarray(policy, dtype=np.float64),
        **_infoset_fields(),
        "legal": legal,
        "iterations": np.asarray(iterations, dtype=np.int64),
    }


def load_average_policy(path: str) -> np.ndarray:
    """A checkpoint written by run_cfr as a (936, 3) policy in _INFOSET_LIST
    order, matching rows by their infoset fields (strict: every infoset once)."""
    template = policy_checkpoint(np.zeros((len(leduc._INFOSET_LIST), 3)), 0)
    tree = load_params(path, template)
    row_of = {key[:5]: i for i, (key, _) in enumerate(leduc._INFOSET_LIST)}
    assert len(row_of) == len(leduc._INFOSET_LIST)
    policy = np.full((len(row_of), 3), np.nan)
    for j in range(len(row_of)):
        sequences = tuple(
            tuple(int(a) for a in tree["betting"][j, r] if a >= 0) for r in (0, 1)
        )
        key = (
            int(tree["player"][j]),
            int(tree["private_card"][j]),
            int(tree["public_card"][j]),
            *sequences,
        )
        i = row_of.pop(key)  # KeyError for an unknown or repeated infoset
        policy[i] = tree["policy"][j]
    return policy


def run_cfr(
    iterations: int,
    log_every: int,
    results_log: ResultsLog | None = None,
    run_dir: str | None = None,
) -> CFRSolver:
    """Runs CFR, logging after iteration 1, every log_every iterations, after
    half_update_steps(iterations) and at the end; with run_dir, saves the
    average strategy after half and all of the iterations."""
    solver = CFRSolver()
    num_half = half_update_steps(iterations)

    def save(stage: str, it: int) -> None:
        if run_dir is not None:
            tree = policy_checkpoint(solver.average_policy_array(), it)
            save_params(run_dir, stage, "average_policy", tree)

    if num_half == 0:
        save("half", 0)
    for it in range(1, iterations + 1):
        solver.run_iteration()
        if it == num_half:
            save("half", it)
        if it == 1 or it % log_every == 0 or it == num_half or it == iterations:
            policy = solver.average_policy_array()
            expl = leduc.exploitability_from_policy_array(policy, leduc._KEY_TO_IDX)
            metrics = {"iteration": it, "exploitability": expl}
            metrics.update(log_metrics("game", policy))
            if results_log is not None:
                results_log.log(metrics, 0)
            print(
                f"iter={it:6d} exploitability={expl:.6f} "
                f"bluff_rate={metrics['game/bluff_rate']:.4f} "
                f"correct_fold_rate={metrics['game/correct_fold_rate']:.4f}",
                flush=True,
            )
    save("full", iterations)
    return solver


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CFR on Leduc Poker.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--save_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="average strategy after half and all of the iterations",
    )
    parser.add_argument(
        "--save_dir", type=str, default=str(REPO_ROOT / "checkpoints" / "leduc")
    )
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--project", type=str, default="bluffjax")
    parser.add_argument("--job_type", type=str, default="leduc_cfr")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = {
        **vars(args),
        "env_name": "leduc_poker",
        "half_iterations": half_update_steps(args.iterations),
    }
    run_dir = make_run_dir(config)
    wandb_run = wandb.init(
        project=config["project"],
        group=config["env_name"] + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S"),
        job_type=f"{config['job_type']}_{config['env_name']}",
        name="0",  # one deterministic run (seed index 0)
        config=config,
        mode=None if config["wandb"] else "disabled",
        dir=str(REPO_ROOT),
        reinit="create_new",
    )
    try:
        print(f"Run directory: {run_dir}")
        results_log = ResultsLog(run_dir, [wandb_run])
        solver = run_cfr(
            args.iterations,
            args.log_every,
            results_log,
            run_dir if args.save_checkpoints else None,
        )
        final_expl = leduc.exploitability_from_policy_array(
            solver.average_policy_array(), leduc._KEY_TO_IDX
        )
        print(
            f"\nfinal exploitability after {args.iterations} iterations: "
            f"{final_expl:.6f}"
        )
        if args.save_checkpoints:
            print(f"Saved checkpoints to {run_dir}")
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    main()
