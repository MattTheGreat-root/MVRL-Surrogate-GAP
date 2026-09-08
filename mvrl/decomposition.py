"""
The error decomposition of the proposal, as far as it can currently be filled in.

The proposal decomposes the shortfall of a learned policy against the
pre-committed optimum as

    J(pi_pre) - J(pi_hat)  =  eps_incons  +  eps_surr  +  eps_learn
                              \\_________/    \\_______/    \\________/
                              known theory     the gap      existing
                                              this project    theory
                                              is about

    eps_incons = J(pi_pre) - J(pi_eq)      price of time-consistency
    eps_surr   = J(pi_eq)  - J(pi_r)       cost of the additive surrogate
    eps_learn  = J(pi_r)   - J(pi_hat)     optimisation and estimation error

This module computes the first term exactly, since both policies it involves
are now available in closed form.  The second and third require ``pi_r``, the
maximiser of the additive surrogate induced by a reward, which needs the
machinery of Steps 2 and 4 and is not implemented yet.  Rather than print a
placeholder, this module reports what is known and states plainly what is not.

Run with ``python -m mvrl.decomposition``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .equilibrium import solve_equilibrium
from .evaluate import exact_affine_objective
from .market import Market
from .policies import MyopicPolicy, li_ng_precommitted


@dataclass
class DecompositionReport:
    J_precommitted: float
    J_equilibrium: float
    eps_incons: float
    J_myopic_mc: float | None = None


def report(market: Market, phi: float, x0: float) -> DecompositionReport:
    precommitted = li_ng_precommitted(market, phi, x0)
    equilibrium = solve_equilibrium(market, phi).policy

    j_pre = exact_affine_objective(market, precommitted, phi, x0).objective
    j_eq = exact_affine_objective(market, equilibrium, phi, x0).objective

    return DecompositionReport(
        J_precommitted=j_pre,
        J_equilibrium=j_eq,
        eps_incons=j_pre - j_eq,
    )


def main() -> None:
    mu = np.array([0.06, 0.04, 0.02])
    vol = np.array([0.18, 0.12, 0.08])
    corr = np.array(
        [[1.00, 0.30, -0.10], [0.30, 1.00, 0.05], [-0.10, 0.05, 1.00]]
    )
    market = Market(
        mu_excess=mu,
        cov_excess=np.outer(vol, vol) * corr,
        riskless=1.02,
        horizon=4,
    )
    phi, x0 = 1.0, 1.0

    precommitted = li_ng_precommitted(market, phi, x0)
    equilibrium = solve_equilibrium(market, phi).policy

    print("Structure of the two reference policies")
    print("=" * 70)
    print("  pre-committed:  u_t(x) = alpha_t x + beta_t   (affine in wealth)")
    for t in range(market.horizon):
        print(f"    t={t}  alpha = "
              f"{np.array2string(precommitted.alpha[t], precision=4)}")
    print("\n  equilibrium:    u_t(x) = beta_t              (wealth-independent)")
    print(f"    max |alpha| over all t = "
          f"{np.abs(equilibrium.alpha).max():.2e}")
    print("""
The two differ structurally, not merely numerically. The pre-committed control
reduces risky holdings as wealth rises, because its multiplier is anchored to
the terminal wealth its time-0 self expected. The equilibrium control holds a
constant amount regardless of wealth. Any statement that a learned policy
"approximates the mean-variance optimum" has to say which of these it means.""")

    result = report(market, phi, x0)

    print("\n" + "=" * 70)
    print("Decomposition terms")
    print("=" * 70)
    print(f"  J(pi_pre)  = {result.J_precommitted:12.9f}")
    print(f"  J(pi_eq)   = {result.J_equilibrium:12.9f}")
    print()
    print(f"  eps_incons = {result.eps_incons:12.9f}"
          f"   ({100*result.eps_incons/abs(result.J_precommitted):.2f}% of J(pi_pre))")
    print(f"  eps_surr   = {'not yet computable':>12s}"
          "   requires pi_r  (Steps 2 and 4)")
    print(f"  eps_learn  = {'not yet computable':>12s}"
          "   requires a trained agent")

    print("""
The first term is now a number rather than a symbol, and it sets the scale the
other two must be compared against. If a trained agent's shortfall against the
pre-committed optimum turns out to be of this order, the reward design is not
the dominant problem; if it is much larger, it is.

That comparison is the point of the project, and it cannot be made from a
backtest, because on historical data J(pi_pre) is unknown.""")

    print("\n" + "=" * 70)
    print("Sensitivity of eps_incons")
    print("=" * 70)
    print(f"{'phi':>8} {'horizon':>9} {'J(pi_pre)':>13} {'J(pi_eq)':>13}"
          f" {'eps_incons':>13}")
    print("-" * 60)
    for phi_value in (0.5, 1.0, 2.0):
        for horizon in (2, 4, 8):
            local = Market(
                mu_excess=market.mu_excess,
                cov_excess=market.cov_excess,
                riskless=market.riskless,
                horizon=horizon,
            )
            r = report(local, phi_value, x0)
            print(f"{phi_value:8.2f} {horizon:9d} {r.J_precommitted:13.6f}"
                  f" {r.J_equilibrium:13.6f} {r.eps_incons:13.6f}")

    print("""
eps_incons grows with the horizon and shrinks with risk aversion. Both are
worth keeping in mind when choosing the experimental configuration for Step 3:
a short horizon makes the term small and the surrogate gap easier to isolate,
while a long one risks the time-consistency price dominating everything else.""")


if __name__ == "__main__":
    main()
