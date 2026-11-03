"""
DiCE estimators for POLA-DiCE on two-player zero-sum games
(goofspiel_pola.py). Pure jax, no environment code.

Adapted from rrmarl-pola-new, src/rrmarl/jax/experiments/smax/pola/pola_new/
dice.py (accumulate and dice_objective are unchanged; team_dice, which shares
one team advantage between the agents of a cooperative SMAX environment, is
replaced by seat_dice, which gives every choice of self seat its own
advantage).

Notation: T time steps, E environments, N seats (players). A column is a pair
(self seat i, environment e): seat i plays the self parameters, every other
seat the opponent's parameters. `delta[t, c]` is the column's joint
log-probability at step t under the parameters being differentiated, minus the
same quantity under the policy that sampled the data (held constant). At the
sampling parameters it is 0 in value but carries their gradient, which is the
role of DiCE's magic box.
"""

import jax
import jax.numpy as jnp

from bluffjax.utils.typing import BoolArray, FloatArray

# exp() of the accumulated log-ratio is clipped at this magnitude to avoid
# overflow far off-policy
MAX_LOG_RATIO = 20.0


def accumulate(
    delta: FloatArray, starts: BoolArray, lam: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """D_t = delta_t + lam * D_{t-1}, restarted wherever an episode begins
    (starts[t] True).

    Returns (D, D_prev) with D_prev[t] the restarted D_{t-1} that enters step
    t, both shaped like delta.
    """

    def step(prev, x):
        d, s = x
        prev = jnp.where(s, 0.0, prev)
        cur = d + lam * prev
        return cur, (cur, prev)

    _, (D, D_prev) = jax.lax.scan(step, jnp.zeros(delta.shape[1:]), (delta, starts))
    return D, D_prev


def dice_objective(
    delta: FloatArray,
    adv: FloatArray,
    starts: BoolArray,
    lam: FloatArray,
    weight: FloatArray | None = None,
) -> FloatArray:
    """Loaded DiCE surrogate (Farquhar et al., 2019) with an importance
    correction.

        mean_{t,c} [ exp(D_t) - exp(lam * D_{t-1}) ] * adv_t

    At the sampling parameters every exp(.) is 1 in value, the first
    derivative is the policy gradient mean_{t,c} grad(delta_t) * adv_t, and
    higher derivatives estimate those of the expected return, with dependence
    on actions more than k steps back discounted by lam**k (lam = 1: unbiased,
    lam = 0: none). Away from the sampling parameters exp(D_t) is a
    lam-discounted product of importance ratios. `adv` must be held constant
    (stop-gradient) by the caller.

    delta, adv, starts: (T, *C). weight: None (every column counts once) or
    column weights shaped like C; the mean is then weighted over the columns.
    """
    D, D_prev = accumulate(delta, starts, lam)
    D = jnp.clip(D, -MAX_LOG_RATIO, MAX_LOG_RATIO)
    D_prev = jnp.clip(lam * D_prev, -MAX_LOG_RATIO, MAX_LOG_RATIO)
    terms = (jnp.exp(D) - jnp.exp(D_prev)) * adv
    if weight is None:
        return terms.mean()
    w = jnp.broadcast_to(weight, terms.shape[1:])
    return (terms * w).sum() / (terms.shape[0] * jnp.maximum(w.sum(), 1e-8))


def per_seat_deltas(d_self: FloatArray, d_opp: FloatArray) -> FloatArray:
    """Joint log-ratio of each environment with each seat as self.

    d_self[..., i]: seat i's log-ratio under the self parameters; d_opp[..., i]:
    the same under the opponent's parameters. Returns delta[..., i] =
    d_self[..., i] + sum_{j != i} d_opp[..., j], the joint log-ratio when seat
    i plays the self parameters and every other seat the opponent's.
    """
    return d_self + d_opp.sum(axis=-1, keepdims=True) - d_opp


def opponent_advantage(adv: FloatArray) -> FloatArray:
    """adv[..., i]: seat i's advantage. Returns, for each choice of self seat
    i, the advantage of the opponents (the mean over the seats j != i; with two
    players, the other seat's advantage)."""
    n = adv.shape[-1]
    return (adv.sum(axis=-1, keepdims=True) - adv) / (n - 1)


def seat_dice(
    d_self: FloatArray,
    d_opp: FloatArray,
    adv: FloatArray,
    starts: BoolArray,
    lam: FloatArray,
    weight: FloatArray | None = None,
) -> FloatArray:
    """dice_objective averaged over the choices of self seat.

    d_self, d_opp: (T, E, N) log-ratios of every seat's action under the self
    and the opponent's parameters. adv: (T, E, N), the advantage that column
    (self seat i, env e) is credited with: the self seat's own advantage for
    the self objective, opponent_advantage(.) for the opponent's objective.
    starts: (T, E), shared by the seats of an environment. weight: None (every
    choice of self seat) or (E, N) column weights, e.g. only the seat that
    actually played the self parameters.
    """
    T, E, N = d_self.shape
    delta = per_seat_deltas(d_self, d_opp)
    starts = jnp.broadcast_to(starts[..., None], (T, E, N))
    return dice_objective(delta, adv, starts, lam, weight)
