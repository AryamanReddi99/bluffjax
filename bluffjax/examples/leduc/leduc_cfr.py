"""CFR on Leduc Poker.

Vanilla tabular CFR (full-tree traversals) with alternating updates, as
OpenSpiel's CFRSolver: in every iteration player 0 and then player 1 traverse
the tree. Each traversal plays the current strategies (regret matching on the
cumulative regrets at the start of the traversal, so player 1's traversal sees
player 0's strategy updated in the same iteration), updates the traversing
player's regrets and adds its strategy, weighted by its own reach probability,
to its average strategy. Logs the exploitability of the average strategy.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from bluffjax.utils.game_utils import leduc_exploitability as leduc


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

    def nash_conv(self) -> float:
        root = leduc.initial_state()
        on_policy = leduc._state_values(root, self.average_policy)
        br0 = leduc.BestResponseSolver(root, 0, self.average_policy).value(root)
        br1 = leduc.BestResponseSolver(root, 1, self.average_policy).value(root)
        return (br0 - on_policy[0]) + (br1 - on_policy[1])

    def exploitability(self) -> float:
        return self.nash_conv() / 2.0


def run_cfr(iterations: int, log_every: int) -> CFRSolver:
    solver = CFRSolver()
    for it in range(1, iterations + 1):
        solver.run_iteration()
        if it == 1 or it % log_every == 0 or it == iterations:
            expl = solver.exploitability()
            print(f"iter={it:6d} exploitability={expl:.6f}")
    return solver


def main() -> None:
    parser = argparse.ArgumentParser(description="CFR on Leduc Poker.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    solver = run_cfr(args.iterations, args.log_every)
    final_expl = solver.exploitability()
    print(
        f"\nfinal exploitability after {args.iterations} iterations: {final_expl:.6f}"
    )


if __name__ == "__main__":
    main()
