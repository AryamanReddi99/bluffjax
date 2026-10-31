"""Quality of an NFSP best response in Leduc Hold'em, computed exactly.

NFSP (Heinrich & Silver 2016) trains its average policy on the actions of a
learned best response (BR). How far that BR is from an exact best response
bounds how fast the average policy can approach a Nash equilibrium, so the
NFSP scripts log it at every exact evaluation (option log_br_gap):

- br_value_vs_avg: value of an exact best response against the average
  policy, averaged over both seats (in a zero-sum game this equals the
  average policy's exploitability).
- greedy_value_vs_avg: value of the learned BR (the given policy table,
  e.g. the greedy policy of a Q-network) against the average policy.
- br_gap_vs_avg: their difference (>= 0; 0 for an exact best response).
- br_value_vs_mix, greedy_value_vs_mix, br_gap_vs_mix: the same against the
  NFSP opponent sigma, the average policy mixed per hand with the learned BR
  (the BR with probability eta for a whole hand), converted to a behaviour
  strategy with reach weights. This is the opponent the BR is trained
  against.

All values are in chips per hand. Policy tables are (num_infosets, 3) rows
over leduc_exploitability._INFOSET_LIST.
"""

import numpy as np

from bluffjax.utils.game_utils import leduc_exploitability as L


def hand_mixture(avg: np.ndarray, br: np.ndarray, eta: float) -> np.ndarray:
    """Behaviour strategy of playing `br` for a whole hand with probability
    eta and `avg` otherwise (independently for each player).

    At an infoset with own reach probabilities r_avg and r_br (products of
    the player's own action probabilities along its perfect-recall history),
    sigma = ((1 - eta) r_avg avg + eta r_br br) / ((1 - eta) r_avg + eta r_br).
    Infosets that neither policy reaches keep the average policy.
    """
    avg = np.asarray(avg, dtype=np.float64)
    br = np.asarray(br, dtype=np.float64)
    mix = avg.copy()
    key_to_idx = L._KEY_TO_IDX
    seen: set[int] = set()

    def visit(state, reach_avg, reach_br):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                visit(state.child(action), reach_avg, reach_br)
            return
        p = state.current_player
        idx = key_to_idx[state.info_state_key(p)]
        if idx not in seen:
            seen.add(idx)
            w_avg = (1.0 - eta) * reach_avg[p]
            w_br = eta * reach_br[p]
            if w_avg + w_br > 0:
                mix[idx] = (w_avg * avg[idx] + w_br * br[idx]) / (w_avg + w_br)
        for action in state.legal_actions():
            next_avg = list(reach_avg)
            next_br = list(reach_br)
            next_avg[p] *= avg[idx, action]
            next_br[p] *= br[idx, action]
            visit(state.child(action), tuple(next_avg), tuple(next_br))

    visit(L.initial_state(), (1.0, 1.0), (1.0, 1.0))
    return mix


def _values(learned: np.ndarray, opponent: np.ndarray) -> tuple[float, float]:
    """(exact BR value, value of `learned`) against `opponent`, both seats
    averaged."""
    root = L.initial_state()
    opp_fn = L._policy_from_array_callable(opponent, L._KEY_TO_IDX)
    own_fn = L._policy_from_array_callable(learned, L._KEY_TO_IDX)
    best = [L.BestResponseSolver(root, i, opp_fn).value(root) for i in (0, 1)]
    got = []
    for i in (0, 1):

        def joint(state, i=i):
            return own_fn(state) if state.current_player == i else opp_fn(state)

        got.append(L._state_values(root, joint)[i])
    return float(np.mean(best)), float(np.mean(got))


def br_gap_metrics(avg: np.ndarray, br: np.ndarray, eta: float) -> dict[str, float]:
    """The metrics of the module docstring for average policy `avg` and
    learned best response `br` (policy tables)."""
    avg = np.asarray(avg, dtype=np.float64)
    br = np.asarray(br, dtype=np.float64)
    out = {}
    for name, opponent in (("avg", avg), ("mix", hand_mixture(avg, br, eta))):
        best, got = _values(br, opponent)
        out[f"br_value_vs_{name}"] = best
        out[f"greedy_value_vs_{name}"] = got
        out[f"br_gap_vs_{name}"] = best - got
    return out
