"""
Evaluation of the true mean-variance objective

    J(pi) = E[x_T^pi] - phi * Var(x_T^pi).

Two evaluators are provided.  :func:`exact_affine_objective` computes J in
closed form for affine policies by propagating the first two moments of wealth;
it carries no sampling error and is the one to use whenever it applies.
:func:`monte_carlo_objective` simulates and works for any policy, including a
learned one; it is what will be used in later phases of the project.

Having both matters: the Monte Carlo evaluator is validated against the exact
one on affine policies before it is trusted on learned policies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .market import Market
from .policies import AffinePolicy, Policy


@dataclass
class Evaluation:
    mean: float
    variance: float
    objective: float
    std_error: float | None = None

    def __str__(self) -> str:
        se = "" if self.std_error is None else f"  (se {self.std_error:.5f})"
        return (
            f"E[x_T] = {self.mean:9.5f}   Var = {self.variance:9.5f}   "
            f"J = {self.objective:9.5f}{se}"
        )


def exact_affine_objective(
    market: Market, policy: AffinePolicy, phi: float, x0: float
) -> Evaluation:
    """Closed-form J for an affine policy, with no sampling error.

    Propagates E[x_t] and E[x_t^2] forward.  With u_t = alpha_t x + beta_t and
    x_{t+1} = s_t x_t + P_t^T u_t, and writing m = E[P], M = E[P P^T],

        E[x_{t+1}]   = s E[x] + m^T (alpha E[x] + beta)
        E[x_{t+1}^2] = s^2 E[x^2]
                       + 2 s ( alpha^T m E[x^2] + beta^T m E[x] )
                       + alpha^T M alpha E[x^2]
                       + 2 alpha^T M beta E[x]
                       + beta^T M beta

    using independence of P_t from x_t.
    """
    mean, variance = float(x0), 0.0
    for t in range(market.horizon):
        s, m, M = market.s(t), market.m(t), market.M(t)
        cov = M - np.outer(m, m)
        a, b = policy.alpha[t], policy.beta[t]
        u = a * mean + b
        variance = (s + m @ a)**2 * variance + u @ cov @ u + variance * (a @ cov @ a)
        mean = s * mean + m @ u
    return Evaluation(mean=mean, variance=variance, objective=mean-phi*variance)


def simulate(
    market: Market,
    policy: Policy,
    x0: float,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate terminal wealth under any policy.

    Returns
    -------
    (n_paths,) array of terminal wealth.
    """
    excess = market.sample_excess(n_paths, rng)  # (n_paths, T, m)
    wealth = np.full(n_paths, float(x0))
    for t in range(market.horizon):
        actions = policy.act_batch(t, wealth)          # (n_paths, m)
        wealth = market.s(t) * wealth + np.einsum(
            "pi,pi->p", excess[:, t, :], actions
        )
    return wealth


def monte_carlo_objective(
    market: Market,
    policy: Policy,
    phi: float,
    x0: float,
    n_paths: int = 200_000,
    seed: int = 0,
) -> Evaluation:
    """Monte Carlo estimate of J, with a standard error for the mean term."""
    rng = np.random.default_rng(seed)
    terminal = simulate(market, policy, x0, n_paths, rng)
    mean = float(terminal.mean())
    variance = float(terminal.var(ddof=1))
    return Evaluation(
        mean=mean,
        variance=variance,
        objective=mean - phi * variance,
        std_error=float(terminal.std(ddof=1) / np.sqrt(n_paths)),
    )
