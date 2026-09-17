"""Legacy calibration helpers. Corrected proofs and scope: notes/PROOFS.md.
Use design.horizon_constant for every fixed horizon >=2.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .equilibrium import solve_equilibrium
from .market import Market
from .surrogate import policy_distance, surrogate_backward


def single_asset(s: float, mu: float, sd: float, horizon: int = 4) -> Market:
    return Market(
        mu_excess=np.array([mu]), cov_excess=np.array([[sd**2]]),
        riskless=s, horizon=horizon,
    )


def demo_market(s: float = 1.0, horizon: int = 4) -> Market:
    mu = np.array([0.06, 0.04, 0.02])
    vol = np.array([0.18, 0.12, 0.08])
    corr = np.array(
        [[1.00, 0.30, -0.10], [0.30, 1.00, 0.05], [-0.10, 0.05, 1.00]]
    )
    return Market(
        mu_excess=mu, cov_excess=np.outer(vol, vol) * corr,
        riskless=s, horizon=horizon,
    )


def kappa_star(market: Market, phi: float) -> float:
    """kappa* = phi (1 - nu).  Exact when s = 1 (R2)."""
    return phi * (1.0 - market.nu(0))


def c_closed_form(market: Market, phi: float) -> float:
    """The T = 2 constant of R5.  Only valid for horizon 2."""
    nu = market.nu(0)
    d = np.linalg.solve(market.M(0), market.m(0))
    return float(
        np.linalg.norm(d) * np.sqrt((2 - nu) * (1 + nu)) / (2 * phi * (1 - nu))
    )


def minimal_distance(market: Market, phi: float, x0: float = 1.0) -> tuple[float, float]:
    """min over kappa of ||pi_r - pi_eq||, and the minimising kappa."""
    from .design import exact_raw_calibration
    _, k, distance = exact_raw_calibration(market, phi, x0, 'candidate')
    return distance, k


def fit_c(factory, phi: float, drifts=(2e-5, 5e-5, 1e-4, 2e-4)) -> tuple[float, float]:
    """Fit min||pi_r - pi_eq|| = c (s-1)^gamma near s = 1."""
    xs, ys = [], []
    for drift in drifts:
        distance, _ = minimal_distance(factory(1.0 + drift), phi)
        xs.append(drift)
        ys.append(distance)
    slope, intercept = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(np.exp(intercept)), float(slope)


def main():
    from .design import horizon_constant
    print('General-horizon coefficient, proved for fixed T and small |s-1|')
    for T in [2,3,4,6,10,16]:
        print(T, horizon_constant(demo_market(1.,T),1.))
    print('Scope, exceptions and exact redesign: notes/PROOFS.md')

if __name__ == '__main__': main()

def discretised_year(mu_annual: float, sigma_annual: float, rate: float,
                     steps: int) -> Market:
    """One year of a Black--Scholes-like market split into `steps` periods."""
    dt = 1.0 / steps
    return Market(
        mu_excess=np.array([mu_annual * dt]),
        cov_excess=np.array([[sigma_annual**2 * dt]]),
        riskless=1.0 + rate * dt,
        horizon=steps,
    )


def basak_chabakauri_check(mu_annual=0.06, sigma_annual=0.18, rate=0.03,
                           phi=1.0) -> None:
    """Verify the equilibrium recursion against the known continuous-time form.

    Basak and Chabakauri (2010) give, in continuous time,
        u(t, x) = (1/gamma) (alpha - r) / sigma^2 * exp(-r (T - t)),
    with gamma = 2 phi under the variance convention used here.  The discrete
    recursion must converge to this as the rebalancing interval shrinks; that it
    does is an external check on an implementation derived rather than quoted.
    """
    target = (1.0 / (2 * phi)) * (mu_annual / sigma_annual**2) * np.exp(-rate)
    print(f"  {'steps/yr':>9}{'nu':>11}{'discrete':>12}{'continuous':>12}{'rel diff':>10}")
    for steps in (4, 12, 52, 252, 2520):
        market = discretised_year(mu_annual, sigma_annual, rate, steps)
        beta0 = solve_equilibrium(market, phi).policy.beta[0][0]
        print(f"  {steps:>9}{market.nu(0):>11.6f}{beta0:>12.6f}{target:>12.6f}"
              f"{abs(beta0-target)/target:>10.2%}")
