"""
The surrogate gap, computed exactly.

This is the module that tests the proposal's central claim.  Everything else in
this package reproduces known results; this asks whether the quantity the
project is about is actually non-zero.

The claim under test
--------------------
Practitioners encode risk into a per-period reward and maximise the sum

    J_r(pi) = E[ sum_t r_t ].

The proposal argues that the maximiser ``pi_r`` of this additive surrogate is
neither the pre-committed optimum nor the equilibrium policy, and that the
resulting shortfall

    eps_surr = J(pi_eq) - J(pi_r)

is a distinct source of error that the literature does not account for.

Why no reinforcement learning is needed here
--------------------------------------------
In this market the optimal affine control for an additive quadratic reward can
be found by direct maximisation of an exactly-computed objective.  There is no
sampling error and no optimisation error: ``pi_r`` is obtained to numerical
precision.  Any gap that appears is therefore a property of the reward itself,
not an artefact of a learning algorithm.  This matters for the argument: it
separates ``eps_surr`` from ``eps_learn`` cleanly, which is exactly what the
decomposition claims can be done.

Rewards implemented
-------------------
Writing the per-period wealth increment as ``d_t = x_{t+1} - x_t``:

    plain      r_t = d_t
               Sums to x_T - x_0, so maximising it maximises E[x_T] with no
               risk term at all.  This is the "portfolio return" reward that
               Bai et al. report as the most common choice in the surveyed
               literature.

    varpen     r_t = d_t - kappa * d_t^2
               Per-period variance penalty.  This is the additive stand-in for
               the mean-variance criterion that practitioners actually write
               down, and the main object of interest.

    quadratic  r_t = d_t - kappa * (d_t - target)^2
               A shaped variant, included to check whether the gap is an
               artefact of one particular functional form.
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
    is the discrete-time analogue of the policy-space metric in the proposal.
    """
    first = float(x0)
    second = float(x0) ** 2
    total = 0.0
    for t in range(market.horizon):
        da = left.alpha[t] - right.alpha[t]
        db = left.beta[t] - right.beta[t]
        # E|| da x + db ||^2 = (da.da) E[x^2] + 2 (da.db) E[x] + (db.db)
        total += (da @ da) * second + 2.0 * (da @ db) * first + (db @ db)

        s, m, M = market.s(t), market.m(t), market.M(t)
        a, b = left.alpha[t], left.beta[t]
        new_first = s * first + m @ (a * first + b)
        new_second = (
            s * s * second
            + 2.0 * s * ((a @ m) * second + (b @ m) * first)
            + (a @ M @ a) * second
            + 2.0 * (a @ M @ b) * first
            + (b @ M @ b)
        )
        first, second = new_first, new_second
    return float(np.sqrt(total))
