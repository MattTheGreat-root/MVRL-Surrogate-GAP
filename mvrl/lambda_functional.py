"""
Lambda: the structure of the surrogate gap.  Phase 2 result.

This module contains the research contribution.  Everything else in the package
measures; this explains, and it is where Theorem 1 of the proposal is settled
for the i.i.d. market with a variance-penalised reward.

Summary of what is established
------------------------------

R1.  The surrogate optimum satisfies a backward recursion.
     The additive objective is a sum, hence time-consistent, hence subject to
     dynamic programming.  Writing W_t(x) = p_t x^2 + q_t x + c_t and
     g_t = kappa (s - 1) - p_{t+1} s, with d = M^{-1} m and nu = m^T M^{-1} m:

         alpha_t = -d g_t / (kappa - p_{t+1})
         beta_t  =  d (1 + q_{t+1}) / (2 (kappa - p_{t+1}))
         p_t     = [kappa p_{t+1} - (1 - nu) g_t^2] / (kappa - p_{t+1})
         1 + q_t = (1 + q_{t+1}) (s - nu g_t / (kappa - p_{t+1}))

     from p_T = q_T = 0.  Derived symbolically and verified against an
     independent optimiser to 5e-14.  The multi-asset case reduces exactly to
     the scalar one: every asset-specific quantity enters only through nu and
     the fixed direction d.

R2.  At s = 1 the surrogate recovers the equilibrium exactly, at
     kappa* = phi (1 - nu).
     alpha_t = 0 requires g_t = 0, and p_T = 0 forces kappa (s - 1) = 0.  With
     s = 1 the recursion collapses to p_t = 0, alpha_t = 0, beta_t = d/(2 kappa),
     while the equilibrium gives beta_t = d/(2 phi (1 - nu)).  Verified to 1e-16
     in policy distance, for any horizon and any number of assets.

R3.  For T = 1 the gap vanishes at every s.
     With one period the mismatch in alpha multiplies the deterministic x_0 and
     therefore merges with beta, so a single kappa absorbs it.  The surrogate
     gap is a genuinely multi-period phenomenon.

R4.  For s != 1 and T >= 2 the gap is irreducible and linear in (s - 1).
         min over kappa of || pi_r - pi_eq ||  =  c |s - 1| + O((s-1)^2)
     The exponent is 1.0000 across every configuration tested, so the gamma of
     Theorem 1 is 1 rather than a quantity to be fitted.  The reason alpha
     cannot be tuned away is visible in R1: to first order alpha_t = -d (s-1),
     independent of kappa.

R5.  For T = 2 the constant is closed form:

         c = || d ||  sqrt( (2 - nu)(1 + nu) )  /  ( 2 phi (1 - nu) )

     verified to 0.11% worst case across single- and multi-asset markets.

Hence the functional of Theorem 1 is

     Lambda(r) = | kappa - phi (1 - nu) |  +  c |s - 1|,      gamma = 1,

vanishing exactly when the surrogate optimum is the equilibrium policy.

Still open
----------
The constant c for T >= 3.  It is computable numerically and grows close to
linearly in the horizon, but the closed form is not established.  The T = 2
derivation is a first-order perturbation of both recursions and generalises in
principle; the algebra grows quickly.

A correction to the proposal
----------------------------
Theorem 1 as written bounds eps_surr and asserts eps_surr = 0 if and only if
Lambda = 0.  That is false.  eps_surr is a difference of scalar objective values
and changes sign, so it crosses zero at a kappa where the two policies still
differ.  The theorem must be stated for the policy distance, which vanishes
exactly when the policies coincide, with the value gap as a corollary.

Run with ``python -m mvrl.lambda_functional``.
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
    equilibrium = solve_equilibrium(market, phi).policy

    def objective(kappa: float) -> float:
        return policy_distance(
            market, surrogate_backward(market, float(kappa)), equilibrium, x0
        )

    result = minimize_scalar(
        objective, bounds=(0.02, 20.0), method="bounded",
        options={"xatol": 1e-14},
    )
    return float(result.fun), float(result.x)


def fit_c(factory, phi: float, drifts=(2e-5, 5e-5, 1e-4, 2e-4)) -> tuple[float, float]:
    """Fit min||pi_r - pi_eq|| = c (s-1)^gamma near s = 1."""
    xs, ys = [], []
    for drift in drifts:
        distance, _ = minimal_distance(factory(1.0 + drift), phi)
        xs.append(drift)
        ys.append(distance)
    slope, intercept = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(np.exp(intercept)), float(slope)


def main() -> None:
    print("=" * 74)
    print("R2.  s = 1:  kappa* = phi (1 - nu) recovers the equilibrium exactly")
    print("=" * 74)
    print(f"{'assets':>8}{'T':>4}{'phi':>7}{'nu':>10}{'kappa*':>13}"
          f"{'||pi_r - pi_eq||':>19}")
    print("-" * 61)
    for market, label in ((single_asset(1.0, 0.06, 0.18), 1), (demo_market(1.0), 3)):
        for horizon in (2, 4, 6):
            local = Market(
                mu_excess=market.mu_excess, cov_excess=market.cov_excess,
                riskless=1.0, horizon=horizon,
            )
            for phi in (1.0, 3.0):
                k = kappa_star(local, phi)
                distance = policy_distance(
                    local, surrogate_backward(local, k),
                    solve_equilibrium(local, phi).policy, 1.0,
                )
                print(f"{label:>8}{horizon:>4}{phi:>7.1f}{local.nu(0):>10.5f}"
                      f"{k:>13.7f}{distance:>19.2e}")

    print("\n" + "=" * 74)
    print("R3.  T = 1:  the gap vanishes at every s")
    print("=" * 74)
    print(f"{'T':>4}{'s-1':>9}{'min ||pi_r - pi_eq||':>24}")
    print("-" * 37)
    for horizon in (1, 2, 3, 4, 6):
        distance, _ = minimal_distance(
            single_asset(1.02, 0.06, 0.18, horizon), 1.0
        )
        print(f"{horizon:>4}{0.02:>9.3f}{distance:>24.3e}")

    print("\n" + "=" * 74)
    print("R4 and R5.  the gap is linear in (s-1); T = 2 constant in closed form")
    print("=" * 74)
    print(f"{'market':<26}{'phi':>6}{'c predicted':>14}{'c measured':>13}"
          f"{'exponent':>11}{'err':>9}")
    print("-" * 79)
    cases = [
        ("1 asset m=.06 sd=.18", lambda s: single_asset(s, 0.06, 0.18, 2), 1.0),
        ("1 asset m=.04 sd=.25", lambda s: single_asset(s, 0.04, 0.25, 2), 2.0),
        ("1 asset m=.09 sd=.15", lambda s: single_asset(s, 0.09, 0.15, 2), 0.7),
        ("1 asset m=.05 sd=.12", lambda s: single_asset(s, 0.05, 0.12, 2), 0.5),
        ("3 assets demo market", lambda s: demo_market(s, 2), 1.0),
        ("3 assets demo market", lambda s: demo_market(s, 2), 2.5),
    ]
    worst = 0.0
    for name, factory, phi in cases:
        predicted = c_closed_form(factory(1.0), phi)
        measured, exponent = fit_c(factory, phi)
        error = abs(predicted - measured) / measured
        worst = max(worst, error)
        print(f"{name:<26}{phi:>6.1f}{predicted:>14.6f}{measured:>13.6f}"
              f"{exponent:>11.4f}{error:>9.3%}")
    print(f"\n  worst relative error {worst:.3%}")

    print("\n  horizon dependence of c (no closed form beyond T = 2):")
    print(f"  {'T':>4}{'c':>12}{'c/(T-1)':>11}")
    print("  " + "-" * 27)
    for horizon in (2, 3, 4, 5, 6, 8):
        c, _ = fit_c(lambda s, h=horizon: single_asset(s, 0.06, 0.18, h), 1.0)
        print(f"  {horizon:>4}{c:>12.5f}{c/(horizon-1):>11.5f}")

    print("""
=======================================================================
Lambda
=======================================================================

    Lambda(r) = | kappa - phi (1 - nu) |  +  c |s - 1|,      gamma = 1

vanishing exactly when the surrogate optimum is the equilibrium policy.  The
first term is the reward mis-specification, which a practitioner controls; the
second is structural, and no choice of kappa removes it.

Two consequences worth stating plainly.  A practitioner who sets the reward's
risk penalty equal to their risk aversion is making a quantifiable error: the
right value is phi(1-nu), not phi, and for the demo market that is 0.806 rather
than 1.  And even with kappa chosen perfectly, a non-zero riskless rate leaves a
gap that grows linearly in it and in the horizon.""")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# The continuous-time limit.
#
# nu = m^T M^{-1} m = SR^2 * Delta + O(Delta^2), so the (1 - nu) correction is
# O(Delta) and vanishes as the rebalancing interval shrinks.  This explains both
# where the result matters (coarse rebalancing) and why it is absent from the
# equilibrium mean-variance literature, which works in continuous time.
# ---------------------------------------------------------------------------

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
