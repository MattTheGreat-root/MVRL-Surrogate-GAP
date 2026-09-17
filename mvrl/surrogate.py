"""Exact additive quadratic-increment optimisation on independent returns.
The legacy name varpen denotes a SECOND-MOMENT penalty, not variance.
Signed J differences are not nonnegative regret. See notes/PROOFS.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .market import Market
from .policies import AffinePolicy


class UnboundedSurrogate(RuntimeError):
    """Raised when the additive objective has no finite maximiser.

    This is not a numerical failure.  A reward with no risk term rewards
    unbounded leverage, so the surrogate genuinely has no optimum, even though
    the mean-variance problem it is meant to stand in for is perfectly well
    posed.  Reporting it as a distinct condition rather than a crash is the
    point: it is a property of the reward.
    """


@dataclass
class Reward:
    """An additive per-period reward of the form d - kappa (d - target)^2."""

    name: str
    kappa: float = 0.0
    target: float = 0.0


def additive_objective(
    market: Market, policy: AffinePolicy, reward: Reward, x0: float
) -> float:
    """E[ sum_t r_t ] computed exactly for an affine policy.

    Propagates E[x_t] and E[x_t^2] forward, using independence of the period
    return from current wealth to evaluate the per-period reward in closed
    form.  With d = (s-1) x + P^T u and u = alpha x + beta,

        E[d]   = (s-1) E[x] + m^T u
        E[d^2] = E[x^2] ( (s-1)^2 + 2(s-1) m^T alpha + alpha^T M alpha )
               + 2 E[x] ( (s-1) m^T beta + alpha^T M beta )
               + beta^T M beta
    """
    first = float(x0)
    second = float(x0) ** 2
    total = 0.0

    for t in range(market.horizon):
        s = market.s(t)
        m = market.m(t)
        M = market.M(t)
        a = policy.alpha[t]
        b = policy.beta[t]

        drift = s - 1.0
        mean_d = drift * first + (m @ a) * first + (m @ b)
        second_d = (
            second * (drift**2 + 2.0 * drift * (m @ a) + (a @ M @ a))
            + 2.0 * first * (drift * (m @ b) + (a @ M @ b))
            + (b @ M @ b)
        )

        # r = d - kappa (d - target)^2
        #   = d - kappa (d^2 - 2 target d + target^2)
        total += mean_d - reward.kappa * (
            second_d - 2.0 * reward.target * mean_d + reward.target**2
        )

        # propagate wealth moments one period
        new_first = s * first + m @ (a * first + b)
        new_second = (
            s * s * second
            + 2.0 * s * ((a @ m) * second + (b @ m) * first)
            + (a @ M @ a) * second
            + 2.0 * (a @ M @ b) * first
            + (b @ M @ b)
        )
        first, second = new_first, new_second

    return total


def optimise_surrogate(
    market: Market, reward: Reward, x0: float, n_restarts: int = 8, seed: int = 0
) -> AffinePolicy:
    """Maximise the additive surrogate over affine policies, exactly.

    Multiple restarts are used and the best is returned; convergence is checked
    by the spread across restarts in :mod:`mvrl.verify`.
    """
    T, n = market.horizon, market.n_assets
    rng = np.random.default_rng(seed)

    def unpack(vector: np.ndarray) -> AffinePolicy:
        half = T * n
        return AffinePolicy(
            alpha=vector[:half].reshape(T, n), beta=vector[half:].reshape(T, n)
        )

    def negative(vector: np.ndarray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            value = additive_objective(market, unpack(vector), reward, x0)
        if not np.isfinite(value):
            return np.inf
        return -value

    best_value, best_vector = -np.inf, None
    for attempt in range(n_restarts):
        start = (
            np.zeros(2 * T * n)
            if attempt == 0
            else rng.normal(scale=1.0, size=2 * T * n)
        )
        result = minimize(
            negative, start, method="L-BFGS-B",
            options={"maxiter": 50_000, "maxfun": 200_000,
                     "ftol": 1e-16, "gtol": 1e-12},
        )
        if -result.fun > best_value:
            best_value, best_vector = -result.fun, result.x

    if best_vector is None or not np.isfinite(best_value):
        raise UnboundedSurrogate(
            f"the additive objective for reward '{reward.name}' is unbounded "
            "above on this market: no finite maximiser exists"
        )
    return unpack(best_vector)


def policy_distance(
    market: Market, left: AffinePolicy, right: AffinePolicy, x0: float
) -> float:
    """Root-mean-square distance between two affine policies along the wealth path.

    Distances are measured where the policies are actually used, by weighting
    each period by the wealth distribution the left-hand policy induces.  This
    is the discrete-time analogue of the policy-space discrepancy in the proposal.
    """
    mean, variance = float(x0), 0.0
    total = 0.0
    for t in range(market.horizon):
        da = left.alpha[t] - right.alpha[t]
        db = left.beta[t] - right.beta[t]
        diff = da * mean + db
        total += diff @ diff + (da @ da) * variance
        s, m, M = market.s(t), market.m(t), market.M(t)
        cov = M - np.outer(m, m)
        a, b = left.alpha[t], left.beta[t]
        u = a * mean + b
        variance = (s + m @ a)**2 * variance + u @ cov @ u + variance * (a @ cov @ a)
        mean = s * mean + m @ u
    return float(np.sqrt(max(0.0, total)))


# ---------------------------------------------------------------------------
# Exact backward recursion for the surrogate optimum.
#
# The additive objective is a sum, hence time-consistent, hence subject to
# dynamic programming.  Writing W_t(x) = p_t x^2 + q_t x + c_t and
# g_t = kappa (s-1) - p_{t+1} s, the one-step problem gives
#
#     u*_t(x) = M^{-1} m * [ (1 + q_{t+1}) - 2 x g_t ] / ( 2 (kappa - p_{t+1}) )
#     p_t     = [ kappa p_{t+1} - (1 - nu) g_t^2 ] / (kappa - p_{t+1})
#     1 + q_t = (1 + q_{t+1}) ( s - nu g_t / (kappa - p_{t+1}) )
#
# started from p_T = q_T = 0.  Derived symbolically; the multi-asset case
# reduces to the scalar one with nu = m^T M^{-1} m, every asset-specific
# quantity entering only through nu and the fixed direction M^{-1} m.
#
# This supersedes the numerical optimiser above for the second-moment-penalised
# reward: it is exact, fast, and has no restarts to converge.  The optimiser is
# retained because it is independent, which is what makes it useful as a check.
# ---------------------------------------------------------------------------

def surrogate_backward(market: Market, kappa: float) -> AffinePolicy:
    """Exact surrogate optimum for r = d - kappa d^2, by backward recursion."""
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    T, n = market.horizon, market.n_assets
    alpha = np.zeros((T, n))
    beta = np.zeros((T, n))
    p, q = 0.0, 0.0  # terminal values

    for t in range(T - 1, -1, -1):
        s = market.s(t)
        m = market.m(t)
        M = market.M(t)
        nu = market.nu(t)
        direction = np.linalg.solve(M, m)

        # Concavity is automatic for kappa > 0: p_t <= 0 by backward induction,
        # since the numerator of p_t is non-positive and the denominator is at
        # least kappa.  The guard is kept as an assertion rather than a
        # documented failure mode; it fired in none of 4000 random
        # configurations spanning kappa in [1e-3, 50] and s in [0.8, 1.3].
        denominator = kappa - p
        if denominator <= 0:  # pragma: no cover - unreachable for kappa > 0
            raise UnboundedSurrogate(
                f"at t={t} the one-step surrogate is not concave "
                f"(kappa - p_{{t+1}} = {denominator:.3e}); this should be "
                "unreachable, please report"
            )
        g = kappa * (s - 1.0) - p * s

        alpha[t] = -direction * g / denominator
        beta[t] = direction * (1.0 + q) / (2.0 * denominator)

        p_next = (kappa * p - (1.0 - nu) * g * g) / denominator
        q_next = (1.0 + q) * (s - nu * g / denominator) - 1.0
        p, q = p_next, q_next

    return AffinePolicy(alpha=alpha, beta=beta)
