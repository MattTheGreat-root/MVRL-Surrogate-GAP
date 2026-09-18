"""Frozen-policy stress samplers. Student-t(8); GARCH dependence is explicit."""
from __future__ import annotations

import numpy as np

from .equilibrium import solve_equilibrium
from .market import Market
from .policies import AffinePolicy, li_ng_precommitted
from .surrogate import surrogate_backward


def demo_market(s: float = 1.02, horizon: int = 4) -> Market:
    mu = np.array([0.06, 0.04, 0.02])
    vol = np.array([0.18, 0.12, 0.08])
    corr = np.array([[1.00, 0.30, -0.10], [0.30, 1.00, 0.05], [-0.10, 0.05, 1.00]])
    return Market(mu_excess=mu, cov_excess=np.outer(vol, vol) * corr,
                  riskless=s, horizon=horizon)


def sample_misspecified(
    market: Market, n_paths: int, rng: np.random.Generator, kind: str
) -> np.ndarray:
    """Excess returns from a process the policies were NOT derived under.

    All variants are matched to the market's first two moments, so any
    difference in outcome is attributable to higher moments or to dependence,
    not to a change in the mean or covariance.
    """
    T, n = market.horizon, market.n_assets
    mean = market.mu_excess
    L = np.linalg.cholesky(market.cov_excess)

    if kind == "gaussian":
        z = rng.standard_normal((n_paths, T, n))

    elif kind == "student-t":
        df = 8.0
        z = rng.standard_t(df, size=(n_paths, T, n)) * np.sqrt((df - 2) / df)

    elif kind == "vol-clustering":
        # GARCH(1,1)-like common volatility factor, unit unconditional variance
        omega, alpha, beta = 0.05, 0.15, 0.80
        z = np.empty((n_paths, T, n))
        h = np.ones(n_paths)
        prev = rng.standard_normal(n_paths)
        for t in range(T):
            h = omega + alpha * prev**2 + beta * h
            scale = np.sqrt(h / (omega / (1 - alpha - beta)))
            eps = rng.standard_normal((n_paths, n))
            z[:, t, :] = eps * scale[:, None]
            prev = z[:, t, 0]

    elif kind == "momentum":
        # AR(1) dependence across periods, variance-corrected
        phi_ar = 0.3
        z = np.empty((n_paths, T, n))
        prev = rng.standard_normal((n_paths, n))
        for t in range(T):
            innov = rng.standard_normal((n_paths, n))
            prev = phi_ar * prev + np.sqrt(1 - phi_ar**2) * innov
            z[:, t, :] = prev

    else:  # pragma: no cover
        raise ValueError(kind)

    return mean[None, None, :] + z @ L.T


def evaluate_paths(
    market: Market, policy: AffinePolicy, excess: np.ndarray, x0: float
) -> np.ndarray:
    x = np.full(excess.shape[0], float(x0))
    for t in range(market.horizon):
        u = x[:, None] * policy.alpha[t][None, :] + policy.beta[t][None, :]
        x = market.s(t) * x + np.einsum("pi,pi->p", excess[:, t, :], u)
    return x


def main():
    print("Use python -m mvrl.experiments for matched-batch robustness results.")
    print("Dependent-process evaluations are frozen-policy stress tests, not optimality claims.")

if __name__ == '__main__':
    main()
