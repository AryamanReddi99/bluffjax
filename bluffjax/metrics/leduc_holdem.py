"""Behaviour metrics of Leduc Hold'em, computed exactly from the policies.

Like the exploitability (bluffjax.utils.game_utils.leduc_exploitability), the
metrics are expectations over the distribution of play, computed from the
policy tables without sampling: every full deal is enumerated (the first
player's card, the second player's card and the public card, which the env
deals in advance as shuffled_deck[2]; 6 * 5 * 4 equally likely deals) for
both choices of the first player (the env draws it uniformly), and the
betting tree is traversed with both seats' policies, each at its own
information sets. A rate is a ratio of expected counts per hand, which is
what the ratio of the counts summed over many sampled hands converges to.

Policies: a policy is an array (num_infosets, 3) of action probabilities
(fold, call, raise) in the infoset order of leduc_exploitability._INFOSET_LIST,
the representation the exploitability is computed from (e.g.
policy_array_from_network / policy_array_from_qnetwork, or a CFR average
strategy). Rows are renormalised over the legal actions, and a row without
mass on any legal action plays them uniformly, exactly as
leduc_exploitability does. An infoset's player is its position (0 acts first
in both rounds); seat s (the env's player index s) plays position
(s - start) % 2 when the env's start player is `start`, so seat 0 acts first in
half of the hands. policy_metrics(policy_p0, policy_p1) has seat 0 play
policy_p0 and seat 1 play policy_p1 (self-play: policy_p1 = policy_p0).

Events (counted at decisions of the betting tree):
- aggressive action: a raise (action 2), i.e. a bet when not facing one or a
  raise when facing one.
- would lose at showdown: the player's hand (its private card with the deal's
  public card) is strictly worse than the opponent's (a pair beats no pair;
  otherwise, as both hands hold the public card, the higher private rank
  wins; a tie, i.e. equal private ranks, doesn't lose). This uses the public
  card of the deal also for round-1 decisions, before it is shown, and is
  exactly the showdown of the game model (the player's return is negative
  when both players check to the end).

Metrics, every event credited to the player who acts (the bluffer for a
successful bluff), with the expected count per hand of the denominator as
n_<denominator>:
- bluff_rate = E[aggressive actions with a hand that would lose at showdown]
  / E[aggressive actions]                                      (n_aggressive)
- bluff_success = E[bluffs whose next action, the opponent's reply to the
  raise, is a fold] / E[bluffs]                                    (n_bluffs)
- correct_fold_rate = E[folds by a player whose hand would have lost at
  showdown] / E[folds]                                              (n_folds)
- pair_raise_rate = E[raises] / E[decisions], over the round-2 decisions where
  raising is legal and the player's private card pairs the public card
                                                         (n_paired_decisions)
- unpaired_raise_rate: the same over the round-2 decisions where raising is
  legal and the private card doesn't pair the public card
                                                       (n_unpaired_decisions)
A rate with no events (zero expected denominator, e.g. a policy that never
folds) is NaN. policy_metrics returns every metric for both seats together
(<name>) and for each seat's own decisions (<name>_p0, <name>_p1); the
training scripts log the totals (log_metrics).
"""

import functools
from typing import NamedTuple

import numpy as np

from bluffjax.utils.game_utils import leduc_exploitability as leduc

FOLD, CALL, RAISE = leduc.ACTION_FOLD, leduc.ACTION_CALL, leduc.ACTION_RAISE
NUM_ACTIONS = 3

# rate -> the expected count it divides by
DENOMINATORS: dict[str, str] = {
    "bluff_rate": "n_aggressive",
    "bluff_success": "n_bluffs",
    "correct_fold_rate": "n_folds",
    "pair_raise_rate": "n_paired_decisions",
    "unpaired_raise_rate": "n_unpaired_decisions",
}
RATE_NAMES: tuple[str, ...] = tuple(DENOMINATORS)
#: the totals of both seats, in logging order (each rate, then its count)
TOTAL_NAMES: tuple[str, ...] = tuple(
    dict.fromkeys(k for rate in RATE_NAMES for k in (rate, DENOMINATORS[rate]))
)


class _Tables(NamedTuple):
    """Every decision of the betting tree under every deal, one row each.

    A deal is (card of position 0, card of position 1, public card), all 120
    equally likely. Rows are indexed r = 0..R-1; the index R stands for "no
    row" (a padding entry of probability 1).
    """

    infoset: np.ndarray  # (R,) index in leduc._INFOSET_LIST of the acting player's infoset
    position: np.ndarray  # (R,) acting position (0 acts first)
    ancestors: np.ndarray  # (R, D) rows of the decisions leading here, padded with R
    ancestor_actions: np.ndarray  # (R, D) the action taken at each of them (0 for padding)
    raise_reply: np.ndarray  # (R,) the opponent's decision after a raise here (R if illegal)
    would_lose: np.ndarray  # (R,) the acting player's hand would lose at showdown
    paired_opportunity: np.ndarray  # (R,) round 2, raise legal, private card pairs the public card
    unpaired_opportunity: np.ndarray  # (R,) round 2, raise legal, no pair
    legal: np.ndarray  # (num_infosets, 3) legal actions of every infoset
    num_deals: int


def _would_lose(cards: tuple[int, int], public: int) -> tuple[bool, bool]:
    """Whether each position would lose the showdown of this deal (model's own rules)."""
    state = leduc.initial_state().child(cards[0]).child(cards[1])
    state = state.child(CALL).child(CALL).child(public).child(CALL).child(CALL)
    assert state.is_terminal() and state.remaining_players == 2
    returns = state.returns()
    return returns[0] < 0, returns[1] < 0


@functools.lru_cache(maxsize=None)
def _tables() -> _Tables:
    infoset, position, paths, raise_reply = [], [], [], []
    would_lose, paired, unpaired = [], [], []

    def visit(state, path, lose, public):
        """Adds the decision rows below `state`; path = [(row, action), ...]."""
        if state.is_terminal():
            return None
        if state.is_chance_node():  # the flop: this deal's public card only
            return visit(state.child(public), path, lose, public)
        row = len(infoset)
        p = state.current_player
        legal = state.legal_actions()
        infoset.append(leduc._KEY_TO_IDX[state.info_state_key(p)])
        position.append(p)
        paths.append(list(path))
        would_lose.append(lose[p])
        is_pair = state.private_cards[p] // 2 == public // 2
        option = state.round_ == 2 and RAISE in legal
        paired.append(option and is_pair)
        unpaired.append(option and not is_pair)
        raise_reply.append(-1)
        for action in legal:
            child_row = visit(state.child(action), path + [(row, action)], lose, public)
            if action == RAISE:
                # a raise never ends the hand or the round: the opponent replies
                assert child_row is not None and position[child_row] == 1 - p
                raise_reply[row] = child_row
        return row

    num_deals = 0
    for c0 in range(6):
        for c1 in range(6):
            for public in range(6):
                if len({c0, c1, public}) < 3:
                    continue
                num_deals += 1
                root = leduc.initial_state().child(c0).child(c1)
                visit(root, [], _would_lose((c0, c1), public), public)

    num_rows = len(infoset)
    depth = max(len(path) for path in paths)
    ancestors = np.full((num_rows, depth), num_rows, dtype=np.int64)
    ancestor_actions = np.zeros((num_rows, depth), dtype=np.int64)
    for r, path in enumerate(paths):
        for k, (row, action) in enumerate(path):
            ancestors[r, k] = row
            ancestor_actions[r, k] = action
    raise_reply = np.asarray(raise_reply, dtype=np.int64)
    raise_reply[raise_reply < 0] = num_rows
    legal = np.zeros((len(leduc._INFOSET_LIST), NUM_ACTIONS), dtype=bool)
    for i, (_, actions) in enumerate(leduc._INFOSET_LIST):
        legal[i, list(actions)] = True
    return _Tables(
        infoset=np.asarray(infoset, dtype=np.int64),
        position=np.asarray(position, dtype=np.int64),
        ancestors=ancestors,
        ancestor_actions=ancestor_actions,
        raise_reply=raise_reply,
        would_lose=np.asarray(would_lose, dtype=np.float64),
        paired_opportunity=np.asarray(paired, dtype=np.float64),
        unpaired_opportunity=np.asarray(unpaired, dtype=np.float64),
        legal=legal,
        num_deals=num_deals,
    )


def normalize_policy(policy) -> np.ndarray:
    """(num_infosets, 3) float64 policy renormalised over the legal actions,
    uniform over them where a row has no legal mass (as leduc_exploitability)."""
    legal = _tables().legal
    policy = np.asarray(policy, dtype=np.float64)
    if policy.shape != legal.shape:
        raise ValueError(f"policy must have shape {legal.shape}, got {policy.shape}")
    policy = np.where(legal, policy, 0.0)
    total = policy.sum(axis=-1, keepdims=True)
    uniform = legal / legal.sum(axis=-1, keepdims=True)
    return np.where(total > 0, policy / np.where(total > 0, total, 1.0), uniform)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def policy_metrics(policy_p0, policy_p1=None) -> dict[str, float]:
    """Exact metrics (module docstring) of seat 0 playing policy_p0 against
    seat 1 playing policy_p1 (default: self-play, policy_p1 = policy_p0).

    Returns {<name>: both seats, <name>_p0: seat 0's decisions, <name>_p1:
    seat 1's decisions} for every rate and n_<denominator> (expected count
    per hand) in TOTAL_NAMES.
    """
    t = _tables()
    policies = [normalize_policy(policy_p0)]
    policies.append(policies[0] if policy_p1 is None else normalize_policy(policy_p1))
    num_rows = len(t.infoset)
    # start = the seat that plays position 0; position p is seat (start + p) % 2
    seat = np.stack([(start + t.position) % 2 for start in (0, 1)])  # (2, R)
    probs = np.stack(
        [
            np.where(
                (seat[start] == 0)[:, None],
                policies[0][t.infoset],
                policies[1][t.infoset],
            )
            for start in (0, 1)
        ]
    )  # (2, R, 3): the acting seat's policy at every row
    padded = np.concatenate([probs, np.ones((2, 1, NUM_ACTIONS))], axis=1)
    # reach probability of the players' actions (chance is uniform: 1/num_deals)
    reach = padded[:, t.ancestors, t.ancestor_actions].prod(axis=-1)  # (2, R)
    weight = reach / (2 * t.num_deals)  # each start player has probability 1/2

    aggressive = weight * probs[..., RAISE]
    bluffs = aggressive * t.would_lose
    folds = weight * probs[..., FOLD]
    events = {
        "n_aggressive": aggressive,
        "n_bluffs": bluffs,
        "n_successful_bluffs": bluffs * padded[:, t.raise_reply, FOLD],
        "n_folds": folds,
        "n_correct_folds": folds * t.would_lose,
        "n_paired_decisions": weight * t.paired_opportunity,
        "n_paired_raises": weight * t.paired_opportunity * probs[..., RAISE],
        "n_unpaired_decisions": weight * t.unpaired_opportunity,
        "n_unpaired_raises": weight * t.unpaired_opportunity * probs[..., RAISE],
    }
    assert all(e.shape == (2, num_rows) for e in events.values())
    by_seat = {
        suffix: {k: float(v[seat == s].sum()) for k, v in events.items()}
        for s, suffix in ((0, "_p0"), (1, "_p1"))
    }
    by_seat[""] = {k: by_seat["_p0"][k] + by_seat["_p1"][k] for k in events}
    numerators = {
        "bluff_rate": "n_bluffs",
        "bluff_success": "n_successful_bluffs",
        "correct_fold_rate": "n_correct_folds",
        "pair_raise_rate": "n_paired_raises",
        "unpaired_raise_rate": "n_unpaired_raises",
    }
    out = {}
    for suffix in ("", "_p0", "_p1"):
        counts = by_seat[suffix]
        for name in TOTAL_NAMES:
            if name in DENOMINATORS:
                value = _ratio(counts[numerators[name]], counts[DENOMINATORS[name]])
            else:
                value = counts[name]
            out[name + suffix] = value
    return out


def log_metrics(prefix: str, policy_p0, policy_p1=None) -> dict[str, float]:
    """The totals of policy_metrics as {"<prefix>/<name>": value}, as the
    training scripts log them (prefix "game" for the output policy, "game_br"
    for NFSP's best response)."""
    metrics = policy_metrics(policy_p0, policy_p1)
    return {f"{prefix}/{name}": metrics[name] for name in TOTAL_NAMES}
