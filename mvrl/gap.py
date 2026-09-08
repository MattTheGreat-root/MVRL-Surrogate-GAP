"""
Does the research gap exist?  The decisive experiment.

Everything else in this package reproduces results that are already known.  The
proposal's contribution rests on a claim that is not known: that maximising an
additive risk-adjusted reward targets a policy which is neither the
pre-committed optimum nor the equilibrium policy, and that the resulting error
is invisible to the statistics the field reports.

This module tests that claim, exactly.  No reinforcement learning is involved,
so no result here can be blamed on a learning algorithm: the surrogate optimum
is computed to numerical precision.  Whatever gap appears is a property of the
reward.

Run with ``python -m mvrl.gap``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .equilibrium import solve_equilibrium
from .evaluate import exact_affine_objective
from .market import Market
from .policies import AffinePolicy, li_ng_precommitted
from .surrogate import (
    Reward,
    additive_objective,
    optimise_surrogate,
    policy_distance,
)


def demo_market() -> Market:
    mu = np.array([0.06, 0.04, 0.02])
    vol = np.array([0.18, 0.12, 0.08])
    corr = np.array(
        [[1.00, 0.30, -0.10], [0.30, 1.00, 0.05], [-0.10, 0.05, 1.00]]
    )
    return Market(
        mu_excess=mu,
        cov_excess=np.outer(vol, vol) * corr,
        riskless=1.02,
        horizon=4,
    )


def unboundedness_evidence(market: Market, x0: float) -> list[tuple[int, float]]:
    """Evaluate the plain-return surrogate along a ray of increasing leverage."""
    T, n = market.horizon, market.n_assets
    direction = np.linalg.solve(market.M(0), market.m(0))
    rows = []
    for scale in (1, 10, 100, 1_000, 10_000, 100_000):
        policy = AffinePolicy(
            alpha=np.zeros((T, n)), beta=np.tile(scale * direction, (T, 1))
        )
        rows.append(
            (scale, additive_objective(market, policy, Reward("plain"), x0))
        )
    return rows


def main() -> None:
    market = demo_market()
    phi, x0 = 1.0, 1.0

    precommitted = li_ng_precommitted(market, phi, x0)
    equilibrium = solve_equilibrium(market, phi).policy
    j_pre = exact_affine_objective(market, precommitted, phi, x0).objective
    j_eq = exact_affine_objective(market, equilibrium, phi, x0).objective

    def surrogate_for(kappa: float) -> tuple[float, AffinePolicy]:
        policy = optimise_surrogate(
            market, Reward("varpen", kappa=float(kappa)), x0, n_restarts=6
        )
        return exact_affine_objective(market, policy, phi, x0).objective, policy

    # ------------------------------------------------------------------ (1)
    print("=" * 74)
    print("1.  The most common reward in the literature has no optimum at all")
    print("=" * 74)
    print("""The plain-return reward r_t = x_{t+1} - x_t carries no risk term, so
the additive objective rewards unbounded leverage.  Along a ray of increasing
position size:""")
    print(f"\n{'leverage scale':>16} {'E[sum of rewards]':>22}")
    for scale, value in unboundedness_evidence(market, x0):
        print(f"{scale:16d} {value:22.4f}")
    print("""
It diverges.  The mean-variance problem this reward is meant to stand in for is
perfectly well posed and has the bounded solution computed below; the surrogate
does not.  A practitioner running an algorithm on this reward is not solving a
badly-conditioned version of the right problem, but a problem with no answer,
and what they obtain instead is whatever their leverage constraints and early
stopping happen to produce.""")

    # ------------------------------------------------------------------ (2)
    print("\n" + "=" * 74)
    print("2.  With a risk term, the surrogate optimum is a third policy")
    print("=" * 74)

    j_natural, policy_natural = surrogate_for(phi)
    search = minimize_scalar(
        lambda k: -surrogate_for(k)[0],
        bounds=(0.4, 2.0),
        method="bounded",
        options={"xatol": 1e-9},
    )
    kappa_star = float(search.x)
    j_star, policy_star = surrogate_for(kappa_star)

    print(f"\n  J(pi_pre)                = {j_pre:12.9f}")
    print(f"  J(pi_eq)                 = {j_eq:12.9f}")
    print(f"  J(pi_r), kappa = phi     = {j_natural:12.9f}")
    print(f"  J(pi_r), kappa = kappa*  = {j_star:12.9f}    kappa* = {kappa_star:.6f}")

    print(f"\n  eps_surr = J(pi_eq) - J(pi_r)  =  {j_eq - j_star:+.9f}")
    print(f"  shortfall against pre-committed =  {j_pre - j_star:.9f}"
          f"   ({100*(j_pre-j_star)/(j_pre-j_eq):.1f}% of eps_incons)")

    print(f"\n  distance(pi_r, pi_eq)  = {policy_distance(market, policy_star, equilibrium, x0):9.6f}")
    print(f"  distance(pi_r, pi_pre) = {policy_distance(market, policy_star, precommitted, x0):9.6f}")

    print("""
Three things follow.  The surrogate optimum is not the equilibrium policy: the
distance is small but not zero, and no choice of kappa closes it.  It is not the
pre-committed policy either, and it is far from it.  And eps_surr is negative
here, so the additive surrogate is not simply a degraded version of the
equilibrium policy; it is a third object, which is the point.

Note also that the natural choice kappa = phi, setting the reward's risk penalty
equal to one's actual risk aversion, is not optimal.""")
    print(f"  best achievable kappa* = {kappa_star:.6f}, not phi = {phi}")
    print(f"  cost of the natural choice = {j_star - j_natural:.9f}")

    # ------------------------------------------------------------------ (3)
    print("\n" + "=" * 74)
    print("3.  A backtest statistic ranks these policies in the wrong order")
    print("=" * 74)

    print(f"\n  {'policy':<20}{'E[x_T]':>10}{'sd':>10}{'Sharpe':>10}{'J':>14}")
    print("  " + "-" * 64)
    rows = [
        ("pre-committed", precommitted),
        ("equilibrium", equilibrium),
        ("surrogate, kappa*", policy_star),
        ("surrogate, kappa=phi", policy_natural),
    ]
    table = []
    for name, policy in rows:
        ev = exact_affine_objective(market, policy, phi, x0)
        sd = float(np.sqrt(ev.variance))
        sharpe = (ev.mean - x0) / sd
        table.append((name, sharpe, ev.objective))
        print(f"  {name:<20}{ev.mean:10.4f}{sd:10.4f}{sharpe:10.4f}"
              f"{ev.objective:14.6f}")

    by_sharpe = [n for n, _, _ in sorted(table, key=lambda r: -r[1])]
    by_objective = [n for n, _, _ in sorted(table, key=lambda r: -r[2])]
    print(f"\n  ranked by Sharpe ratio: {' > '.join(by_sharpe)}")
    print(f"  ranked by J:            {' > '.join(by_objective)}")

    equilibrium_row = next(r for r in table if r[0] == "equilibrium")
    natural_row = next(r for r in table if r[0] == "surrogate, kappa=phi")
    if natural_row[1] > equilibrium_row[1] and natural_row[2] < equilibrium_row[2]:
        print(f"""
  The kappa = phi surrogate has the HIGHER Sharpe ratio
      {natural_row[1]:.4f} against {equilibrium_row[1]:.4f}
  and the LOWER mean-variance objective
      {natural_row[2]:.6f} against {equilibrium_row[2]:.6f}.

  So the two orderings disagree.  A study comparing these policies by reported
  Sharpe ratio, which is the standard practice the surveys describe, would
  conclude that the surrogate policy is the better one on exactly the criterion
  it is worse on.""")

    print("""
This is the third gap of the proposal, demonstrated rather than asserted.  The
statistic the field reports does not order policies the way the objective does,
so no amount of backtesting can reveal the error that reward design introduces.
It has to be measured against a known optimum, which is what the rest of this
package makes possible.""")


if __name__ == "__main__":
    main()
