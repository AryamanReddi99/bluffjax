"""DiCE estimators of POLA-DiCE for Goofspiel
(bluffjax/examples/goofspiel/goofspiel_dice.py), adapted from rrmarl-pola-new's
SMAX test_dice.py to the two-seat zero-sum case:

- accumulate's restarts, the weighted column mean, per_seat_deltas and
  opponent_advantage;
- exact (no sampling noise: expectations are enumerated) on a two-step,
  two-player zero-sum game with memory, the parameters indexed by seat as a
  shared network sees the seat in its observation, self at one seat and the
  opponent's parameters at the other, averaged over the self seat as in
  goofspiel_pola.py: value 0, gradient = grad J / T, Hessian = Hess J / T at
  lambda = 1 (with the cross-seat block of the opponent's objective, the term
  POLA's shaping uses, nonzero), lambda = 0 biased at second order;
- the same with a mixed behaviour policy and only the seat that played the self
  parameters as self (a fresh rollout with fresh_self_seats: sampled);
- rock-paper-scissors self-play with one shared logit vector: naive
  simultaneous policy gradient spirals away from the uniform Nash
  equilibrium, LOLA-DiCE and POLA-DiCE (the opponent's lookahead
  differentiated through, as in goofspiel_pola.py) spiral in. Run this file
  as a script to print the trajectories.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax.examples.goofspiel.goofspiel_dice import (
    accumulate,
    dice_objective,
    opponent_advantage,
    per_seat_deltas,
    seat_dice,
)


@pytest.fixture(autouse=True, scope="module")
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


# A two-step, two-player zero-sum game with memory: the second step's state is
# the first step's joint action. Parameters x[seat, state, action]: 2 seats, 5
# states (start + 4 joint outcomes), 2 actions. Seat 0 is paid R[a0, a1] per
# step, seat 1 -R[a0, a1].
N_STATES, N_ACT, T = 5, 2, 2
R = jnp.array([[1.0, -2.0], [3.0, 0.5]])
N_PARAMS = 2 * N_STATES * N_ACT


def payoff(seat):
    return R if seat == 0 else -R


def trajectories():
    return list(itertools.product(range(N_ACT), repeat=2 * T))  # (a0_0, a1_0, a0_1, a1_1)


def log_probs(x, tau):
    """Per-step log-probabilities of each seat's action along tau: (T, 2)."""
    a00, a10, a01, a11 = tau
    s1 = 1 + 2 * a00 + a10
    lp = jax.nn.log_softmax(x, axis=-1)
    return jnp.array([[lp[0, 0, a00], lp[1, 0, a10]], [lp[0, s1, a01], lp[1, s1, a11]]])


def prob(x, tau):
    return jnp.exp(log_probs(x, tau).sum())


def mixed(x, y, s):
    """Seat s plays x, the other seat y."""
    return jnp.stack([x[0] if s == 0 else y[0], x[1] if s == 1 else y[1]])


def exact_J(x, seat):
    """Seat's expected (undiscounted) return with the seats playing x."""
    Rs = payoff(seat)
    return sum(prob(x, tau) * (Rs[tau[0], tau[1]] + Rs[tau[2], tau[3]]) for tau in trajectories())


def exact_advantages(x, seat, tau):
    """A_t = Q(s_t, a_t) - V(s_t) of the joint action along tau for seat's return."""
    Rs = payoff(seat)
    a00, a10, a01, a11 = tau
    p = jax.nn.softmax(x, -1)
    s1 = 1 + 2 * a00 + a10
    A1 = Rs[a01, a11] - p[0, s1] @ Rs @ p[1, s1]
    V1_all = jnp.array(
        [[p[0, 1 + 2 * u + v] @ Rs @ p[1, 1 + 2 * u + v] for v in range(2)] for u in range(2)]
    )
    Q0 = Rs + V1_all
    A0 = Q0[a00, a10] - p[0, 0] @ Q0 @ p[1, 0]
    return jnp.array([A0, A1])


def split(z):
    return z[:N_PARAMS].reshape(2, N_STATES, N_ACT), z[N_PARAMS:].reshape(2, N_STATES, N_ACT)


def expected_seat_dice(z, behaviour, role, lam, weight=None):
    """E_{tau ~ behaviour} of seat_dice at z = (x self, y opponent), one env."""
    x, y = split(z)
    total = 0.0
    for tau in trajectories():
        lp_b = log_probs(behaviour, tau)
        d_self = (log_probs(x, tau) - lp_b)[:, None, :]  # (T, E = 1, N = 2)
        d_opp = (log_probs(y, tau) - lp_b)[:, None, :]
        adv = jnp.stack([exact_advantages(behaviour, 0, tau), exact_advantages(behaviour, 1, tau)], -1)
        adv = adv[:, None, :]
        if role == "opponent":
            adv = opponent_advantage(adv)
        starts = jnp.array([[True], [False]])
        total = total + prob(behaviour, tau) * seat_dice(d_self, d_opp, adv, starts, lam, weight)
    return total


def exact_G(z, role, seats=(0, 1)):
    """Mean over the self seats s of the self seat's (role self) or the other
    seat's (role opponent) return, seat s playing x and the other y."""
    x, y = split(z)
    return sum(exact_J(mixed(x, y, s), s if role == "self" else 1 - s) for s in seats) / len(seats)


def x0_point():
    rng = np.random.default_rng(0)
    x0 = jnp.asarray(rng.normal(size=N_PARAMS))
    return x0, jnp.concatenate([x0, x0])  # self play: both parameter sets at x0


def test_accumulate_restarts():
    delta = jnp.array([[1.0], [2.0], [3.0], [4.0]])
    starts = jnp.array([[True], [False], [True], [False]])
    D, D_prev = accumulate(delta, starts, 0.5)
    np.testing.assert_allclose(D[:, 0], [1.0, 2.5, 3.0, 5.5])
    np.testing.assert_allclose(D_prev[:, 0], [0.0, 1.0, 0.0, 3.0])


def test_weighted_mean_is_mean_over_selected_columns():
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    delta = 0.1 * jax.random.normal(k1, (4, 3, 2))
    adv = jax.random.normal(k2, (4, 3, 2))
    starts = jnp.zeros((4, 3, 2), bool).at[0].set(True)
    w = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    sel = np.asarray(w, bool)
    expect = dice_objective(delta[:, sel], adv[:, sel], starts[:, sel], 0.9)
    np.testing.assert_allclose(dice_objective(delta, adv, starts, 0.9, w), expect, rtol=1e-12)
    np.testing.assert_allclose(
        dice_objective(delta, adv, starts, 0.9, jnp.ones((3, 2))),
        dice_objective(delta, adv, starts, 0.9), rtol=1e-12,
    )


def test_per_seat_deltas_and_opponent_advantage():
    d_self = jnp.arange(12.0).reshape(2, 2, 3)
    d_opp = 10 * jnp.arange(12.0).reshape(2, 2, 3)
    out = per_seat_deltas(d_self, d_opp)
    for idx in itertools.product(range(2), range(2), range(3)):
        *b, i = idx
        expect = d_self[idx] + sum(d_opp[(*b, j)] for j in range(3) if j != i)
        assert float(out[idx]) == float(expect)
    adv = jnp.array([[1.0, -1.0], [2.5, 0.5]])
    np.testing.assert_array_equal(opponent_advantage(adv), adv[:, ::-1])


@pytest.mark.parametrize("role", ["self", "opponent"])
def test_value_and_first_derivative(role):
    """Value 0 (mean advantage), gradient = grad J / T in both parameter sets."""
    x0, z0 = x0_point()
    f = lambda z: expected_seat_dice(z, split(z0)[0], role, 1.0)
    assert abs(float(f(z0))) < 1e-10
    g_exact = jax.grad(lambda z: exact_G(z, role))(z0) / T
    np.testing.assert_allclose(jax.grad(f)(z0), g_exact, atol=1e-10)


@pytest.mark.parametrize("role", ["self", "opponent"])
def test_second_derivative_unbiased_at_lambda_one(role):
    """The expected Hessian equals Hess J / T: own, cross-step and cross-seat terms."""
    x0, z0 = x0_point()
    H = jax.hessian(lambda z: expected_seat_dice(z, split(z0)[0], role, 1.0))(z0)
    H_exact = jax.hessian(lambda z: exact_G(z, role))(z0) / T
    np.testing.assert_allclose(H, H_exact, atol=1e-9)


def test_cross_seat_term_for_the_shaping():
    """d/d theta d/d phi of the opponent's objective, the term POLA's
    lookahead differentiates, is exact at lambda = 1 and nonzero, and (zero
    sum) minus that of the self objective."""
    x0, z0 = x0_point()
    n = N_PARAMS
    H = jax.hessian(lambda z: expected_seat_dice(z, split(z0)[0], "opponent", 1.0))(z0)[n:, :n]
    H_exact = jax.hessian(lambda z: exact_G(z, "opponent"))(z0)[n:, :n] / T
    np.testing.assert_allclose(H, H_exact, atol=1e-9)
    assert np.max(np.abs(H_exact)) > 1e-2
    H_self = jax.hessian(lambda z: exact_G(z, "self"))(z0)[n:, :n] / T
    np.testing.assert_allclose(H_exact, -H_self, atol=1e-12)


def test_lambda_zero_keeps_first_order_biased_second():
    x0, z0 = x0_point()
    f = lambda z: expected_seat_dice(z, split(z0)[0], "opponent", 0.0)
    g_exact = jax.grad(lambda z: exact_G(z, "opponent"))(z0) / T
    np.testing.assert_allclose(jax.grad(f)(z0), g_exact, atol=1e-10)
    H0 = jax.hessian(f)(z0)
    H_exact = jax.hessian(lambda z: exact_G(z, "opponent"))(z0) / T
    assert np.max(np.abs(H0 - H_exact)) > 1e-3


@pytest.mark.parametrize("role", ["self", "opponent"])
def test_mixed_behaviour_sampled_seat(role):
    """A fresh rollout: seat 0 played x0, seat 1 y0 != x0; only seat 0 is
    self (weight). Value 0, first and second derivatives exact."""
    rng = np.random.default_rng(1)
    x0 = jnp.asarray(rng.normal(size=N_PARAMS))
    y0 = jnp.asarray(rng.normal(size=N_PARAMS))
    z0 = jnp.concatenate([x0, y0])
    behaviour = mixed(*split(z0), 0)
    w = jnp.array([[1.0, 0.0]])
    f = lambda z: expected_seat_dice(z, behaviour, role, 1.0, w)
    G = lambda z: exact_G(z, role, seats=(0,))
    assert abs(float(f(z0))) < 1e-10
    np.testing.assert_allclose(jax.grad(f)(z0), jax.grad(G)(z0) / T, atol=1e-10)
    np.testing.assert_allclose(jax.hessian(f)(z0), jax.hessian(G)(z0) / T, atol=1e-9)


# ---------------------------------------------------------------- rock-paper-scissors

RPS = jnp.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])  # seat 0's payoff
RPS_B, RPS_ITERS, RPS_LR, RPS_LR_IN, RPS_OUTER = 4096, 300, 1.0, 1.0, 4


def rps_exploitability(theta):
    """max_a (R pi)_a: what a best response to pi wins (0 at the uniform Nash)."""
    return jnp.max(RPS @ jax.nn.softmax(theta))


def rps_play(key, th_seat0, th_seat1):
    k0, k1 = jax.random.split(key)
    a0 = jax.random.categorical(k0, th_seat0, shape=(RPS_B,))
    a1 = jax.random.categorical(k1, th_seat1, shape=(RPS_B,))
    r0 = RPS[a0, a1]
    return jnp.stack([a0, a1], -1), jnp.stack([r0, -r0], -1)  # (B, 2) each


def rps_objective(th_self, th_opp, actions, rewards, behaviour_lp, role):
    """seat_dice of the one-shot game (T = 1), averaged over both self seats."""
    lp = lambda th: jax.nn.log_softmax(th)[actions][None] - behaviour_lp[None]
    adv = (rewards - rewards.mean(0))[None]
    if role == "opponent":
        adv = opponent_advantage(adv)
    starts = jnp.ones((1, RPS_B), bool)
    return seat_dice(lp(th_self), lp(th_opp), adv, starts, 1.0)


def rps_kl(th, th_old):
    p = jax.nn.softmax(th)
    return (p * (jax.nn.log_softmax(th) - jax.nn.log_softmax(th_old))).sum()


def make_rps_step(method, beta):
    def step(theta, key):
        k0, k1 = jax.random.split(key)
        sg = jax.lax.stop_gradient
        actions, rewards = rps_play(k0, theta, theta)
        blp = jax.nn.log_softmax(theta)[actions]
        if method == "naive":
            g = jax.grad(rps_objective)(theta, sg(theta), actions, rewards, blp, "self")
        else:
            def look(th_self):
                # the opponent's proximal step on its objective, differentiable in th_self
                def inner(phi):
                    return rps_objective(th_self, phi, actions, rewards, blp, "opponent") - beta * rps_kl(
                        phi, sg(theta)
                    )
                return theta + RPS_LR_IN * jax.grad(inner)(theta)

            phi = sg(look(theta))
            # fresh rollout: seat 0 plays theta, seat 1 phi
            acts1, rews1 = rps_play(k1, theta, phi)
            blp1 = jnp.stack(
                [jax.nn.log_softmax(theta)[acts1[:, 0]], jax.nn.log_softmax(phi)[acts1[:, 1]]], -1
            )

            def outer(th):
                return rps_objective(th, look(th), acts1, rews1, blp1, "self") - beta * rps_kl(
                    th, sg(theta)
                )

            # LOLA: one step. POLA: RPS_OUTER steps of RPS_LR / RPS_OUTER from
            # theta on the proximal objective (as goofspiel_pola.py's epochs x
            # minibatches; a single step would not see the KL, whose gradient
            # vanishes at theta).
            n = 1 if method == "lola" else RPS_OUTER
            th = theta
            for _ in range(n):
                th = th + RPS_LR / n * jax.grad(outer)(th)
            return th, rps_exploitability(theta)
        return theta + RPS_LR * g, rps_exploitability(theta)

    return jax.jit(step)


def run_rps(method, beta, seed=0):
    theta = jnp.array([1.5, 0.0, -1.0])
    step = make_rps_step(method, beta)
    out = []
    for k in jax.random.split(jax.random.PRNGKey(seed), RPS_ITERS):
        theta, e = step(theta, k)
        out.append(float(e))
    return np.array(out)


@pytest.mark.parametrize("method,beta", [("naive", 0.0), ("lola", 0.0), ("pola", 0.5)])
def test_rps_self_play(method, beta):
    traj = run_rps(method, beta)
    start, end = traj[0], traj[-30:].mean()
    if method == "naive":
        assert end > start  # spirals out
    else:
        assert end < 0.25 * start  # spirals in


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    for method, beta in (("naive", 0.0), ("lola", 0.0), ("pola", 0.5)):
        finals = [run_rps(method, beta, s) for s in range(4)]
        print(
            f"{method:5s} beta={beta}: RPS exploitability start {finals[0][0]:.3f}, "
            f"mean over the last 30 of {RPS_ITERS} iterations, 4 seeds: "
            f"{np.round([f[-30:].mean() for f in finals], 3)}",
            flush=True,
        )
