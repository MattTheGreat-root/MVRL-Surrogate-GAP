"""
Verification of the reference machinery.

Nothing later in the project means anything unless the pre-committed policy is
actually optimal, so it is checked here against methods that do not share its
derivation:

  1. The Monte Carlo evaluator agrees with the exact moment-propagation
     evaluator on affine policies.
  2. The closed-form policy matches a direct numerical maximisation of J over
     all affine policies.  This is the important one: scipy knows nothing about
     the Li--Ng embedding, so agreement is genuine evidence the algebra is right.
  3. The closed-form policy is not beaten by random perturbations of itself.
  4. The embedding fixed point holds: lambda = 1 + 2 phi E[x_T].

Run with ``python -m mvrl.verify``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .evaluate import exact_affine_objective, monte_carlo_objective
from .market import Market
from .equilibrium import deviation_value, solve_equilibrium
from .policies import AffinePolicy, li_ng_precommitted


def _default_market() -> Market:
    """Three risky assets over four periods, loosely calibrated to annual data."""
    mu = np.array([0.06, 0.04, 0.02])
    vol = np.array([0.18, 0.12, 0.08])
    corr = np.array(
        [
            [1.00, 0.30, -0.10],
            [0.30, 1.00, 0.05],
            [-0.10, 0.05, 1.00],
        ]
    )
    cov = np.outer(vol, vol) * corr
    return Market(mu_excess=mu, cov_excess=cov, riskless=1.02, horizon=4)


def _flatten(policy: AffinePolicy) -> np.ndarray:
    return np.concatenate([policy.alpha.ravel(), policy.beta.ravel()])


def _unflatten(vector: np.ndarray, T: int, n: int) -> AffinePolicy:
    half = T * n
    return AffinePolicy(
        alpha=vector[:half].reshape(T, n), beta=vector[half:].reshape(T, n)
    )


def check_evaluators_agree(market, phi, x0, tol=5e-3) -> bool:
    policy = li_ng_precommitted(market, phi, x0)
    exact = exact_affine_objective(market, policy, phi, x0)
    approx = monte_carlo_objective(market, policy, phi, x0, n_paths=400_000, seed=1)
    relative = abs(exact.objective - approx.objective) / max(1.0, abs(exact.objective))
    ok = relative < tol
    print(f"  exact        {exact}")
    print(f"  monte carlo  {approx}")
    print(f"  relative difference {relative:.2e}   {'PASS' if ok else 'FAIL'}")
    return ok


def check_against_direct_optimisation(market, phi, x0, tol=1e-7) -> bool:
    """Maximise J over all affine policies numerically and compare.

    The optimiser is given no knowledge of the Li--Ng embedding, so convergence
    to the same value from independent random starts is genuine evidence that
    the closed form is correct.  We require agreement in both directions: the
    optimiser must not exceed the closed form (it is meant to be optimal) and
    must reach it (otherwise the test is vacuous).
    """
    T, n = market.horizon, market.n_assets
    closed_form = li_ng_precommitted(market, phi, x0)
    target = exact_affine_objective(market, closed_form, phi, x0).objective

    def negative_objective(vector: np.ndarray) -> float:
        candidate = _unflatten(vector, T, n)
        return -exact_affine_objective(market, candidate, phi, x0).objective

    values = []
    rng = np.random.default_rng(7)
    for attempt in range(8):
        start = (
            np.zeros(2 * T * n)
            if attempt == 0
            else rng.normal(scale=1.0, size=2 * T * n)
        )
        result = minimize(
            negative_objective, start, method="L-BFGS-B",
            options={"maxiter": 50_000, "maxfun": 200_000,
                     "ftol": 1e-16, "gtol": 1e-12},
        )
        values.append(-result.fun)
    best = max(values)
    spread = max(values) - min(values)

    exceeds = best - target                       # must be <= tol
    shortfall = target - best                     # must also be <= tol
    ok = (exceeds < tol) and (shortfall < tol)

    print(f"  closed form         J = {target:.12f}")
    print(f"  best of 8 restarts  J = {best:.12f}")
    print(f"  spread across restarts {spread:.2e}")
    print(f"  |difference| = {abs(target - best):.2e}   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_stationarity(market, phi, x0, tol=1e-6) -> bool:
    """The gradient of J must vanish at the closed-form policy."""
    T, n = market.horizon, market.n_assets
    policy = li_ng_precommitted(market, phi, x0)
    flat = _flatten(policy)

    def objective(vector: np.ndarray) -> float:
        return exact_affine_objective(
            market, _unflatten(vector, T, n), phi, x0
        ).objective

    step = 1e-6
    gradient = np.zeros_like(flat)
    for i in range(flat.size):
        forward, backward = flat.copy(), flat.copy()
        forward[i] += step
        backward[i] -= step
        gradient[i] = (objective(forward) - objective(backward)) / (2 * step)

    norm = float(np.abs(gradient).max())
    ok = norm < tol
    print(f"  max |dJ/dparam| at the closed form = {norm:.3e}   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_perturbations(market, phi, x0, n_trials=4000) -> bool:
    T, n = market.horizon, market.n_assets
    policy = li_ng_precommitted(market, phi, x0)
    base = exact_affine_objective(market, policy, phi, x0).objective
    flat = _flatten(policy)
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(n_trials):
        scale = 10 ** rng.uniform(-4, -1)
        perturbed = _unflatten(flat + rng.normal(scale=scale, size=flat.size), T, n)
        value = exact_affine_objective(market, perturbed, phi, x0).objective
        worst = max(worst, value - base)
    ok = worst <= 1e-9
    print(f"  best perturbation improvement {worst:.3e}   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_embedding_fixed_point(market, phi, x0, tol=1e-9) -> bool:
    """lambda* = 1 + 2 phi E[x_T*] must hold at the returned policy."""
    from .policies import _policy_for_lambda, _expected_terminal_wealth

    policy = li_ng_precommitted(market, phi, x0)
    expected = _expected_terminal_wealth(market, policy, x0)
    lam = 1.0 + 2.0 * phi * expected
    rebuilt = _policy_for_lambda(market, phi, lam)
    difference = max(
        np.abs(rebuilt.alpha - policy.alpha).max(),
        np.abs(rebuilt.beta - policy.beta).max(),
    )
    ok = difference < tol
    print(f"  implied lambda = {lam:.10f}")
    print(f"  policy rebuilt from it differs by {difference:.2e}   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_consistency_on_expected_path(market, phi, x0, tol=1e-9) -> bool:
    """The reversal must vanish exactly at the expected wealth path.

    Observed empirically, then promoted to a check: the wealth at which the
    committed and re-solved controls agree equals E[x_1] to machine precision.
    Verified here across several markets so that a future change to the
    embedding cannot silently break it.
    """
    from scipy.optimize import brentq

    from .inconsistency import disagreement_at

    worst = 0.0
    for horizon in (2, 3, 5):
        local = Market(
            mu_excess=market.mu_excess,
            cov_excess=market.cov_excess,
            riskless=market.riskless,
            horizon=horizon,
        )
        policy = li_ng_precommitted(local, phi, x0)
        u0 = policy.alpha[0] * x0 + policy.beta[0]
        expected = float(local.s(0) * x0 + local.m(0) @ u0)

        def gap(x1: float) -> float:
            d = disagreement_at(local, phi, x0, x1)
            return float((d.resolved_action - d.committed_action)[0])

        crossing = brentq(gap, expected - 2.0, expected + 2.0, xtol=1e-13)
        worst = max(worst, abs(crossing - expected))
        print(f"  T={horizon}:  crossing {crossing:.10f}   E[x_1] {expected:.10f}"
              f"   diff {abs(crossing-expected):.2e}")

    ok = worst < tol
    print(f"  worst difference {worst:.2e}   {'PASS' if ok else 'FAIL'}")
    return ok


def check_equilibrium_condition(market, phi, x0, tol=1e-8) -> bool:
    """No single-period deviation may improve on the equilibrium policy.

    This is the definition, equation (10) of the proposal, tested directly:
    at a range of times and wealth levels, L-BFGS-B is asked to beat the
    equilibrium action while the continuation is held fixed.
    """
    solution = solve_equilibrium(market, phi)
    rng = np.random.default_rng(0)
    worst = 0.0
    for t in range(market.horizon):
        for x in (0.7, 1.0, 1.6):
            action = solution.policy.alpha[t] * x + solution.policy.beta[t]
            base = deviation_value(market, solution, phi, t, x, action)
            best = base
            for _ in range(4):
                start = action + rng.normal(scale=1.0, size=market.n_assets)
                result = minimize(
                    lambda u: -deviation_value(market, solution, phi, t, x, u),
                    start, method="L-BFGS-B",
                    options={"ftol": 1e-16, "gtol": 1e-12},
                )
                best = max(best, -result.fun)
            worst = max(worst, best - base)
    ok = worst < tol
    print(f"  best improvement from any deviation {worst:.3e}   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_equilibrium_is_wealth_independent(market, phi, tol=1e-10) -> bool:
    """alpha_t must vanish, equivalently A_t^2 = B_t at every t.

    The equilibrium control holds a constant amount in the risky assets rather
    than a constant fraction.  This reproduces the known structure of the
    equilibrium mean-variance policy and is independent evidence the backward
    recursion is right.
    """
    solution = solve_equilibrium(market, phi)
    max_alpha = float(np.abs(solution.policy.alpha).max())
    max_identity = float(np.abs(solution.A**2 - solution.B).max())
    ok = max_alpha < tol and max_identity < tol
    print(f"  max |alpha_t|      = {max_alpha:.3e}")
    print(f"  max |A_t^2 - B_t|  = {max_identity:.3e}   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_incons_scaling(market, x0, tol=1e-9) -> bool:
    """eps_incons scales as 1/phi and does not depend on x0.

    Both are exact.  The first matters for experimental design: the price of
    time-consistency can be made arbitrarily small relative to the other error
    terms by choosing phi, which is how the surrogate gap will be isolated.
    """
    from .decomposition import report

    products = []
    for phi in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        products.append(report(market, phi, x0).eps_incons * phi)
    spread = (max(products) - min(products)) / abs(float(np.mean(products)))

    values = [report(market, 1.0, w).eps_incons for w in (0.5, 1.0, 2.0)]
    wealth_spread = max(values) - min(values)

    ok = spread < 1e-9 and wealth_spread < tol
    print(f"  relative spread of eps_incons * phi   {spread:.2e}")
    print(f"  spread across x0                      {wealth_spread:.2e}   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_precommitted_dominates(market, phi, x0) -> bool:
    """At time 0 the pre-committed policy must be at least as good."""
    from .decomposition import report

    r = report(market, phi, x0)
    ok = r.eps_incons >= -1e-12
    print(f"  J(pi_pre) = {r.J_precommitted:.9f}")
    print(f"  J(pi_eq)  = {r.J_equilibrium:.9f}")
    print(f"  eps_incons = {r.eps_incons:.9f}   {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    market = _default_market()
    phi, x0 = 1.0, 1.0

    print("Market: 3 risky assets, 4 periods, riskless gross return "
          f"{market.riskless}")
    print(f"nu per period = {market.nu(0):.6f}")
    print(f"phi = {phi},  x0 = {x0}\n")

    results = {}
    print("[1] exact evaluator vs Monte Carlo")
    results["evaluators agree"] = check_evaluators_agree(market, phi, x0)

    print("\n[2] closed form vs direct numerical maximisation over affine policies")
    results["closed form optimal"] = check_against_direct_optimisation(market, phi, x0)

    print("\n[3] stationarity of J at the closed-form policy")
    results["gradient vanishes"] = check_stationarity(market, phi, x0)

    print("\n[4] local optimality under random perturbation")
    results["locally optimal"] = check_perturbations(market, phi, x0)

    print("\n[5] embedding fixed point")
    results["fixed point holds"] = check_embedding_fixed_point(market, phi, x0)

    print("\n[6] time-consistency holds exactly on the expected wealth path")
    results["consistent on mean path"] = check_consistency_on_expected_path(
        market, phi, x0
    )

    print("\n[7] equilibrium condition: no profitable single-period deviation")
    results["equilibrium holds"] = check_equilibrium_condition(market, phi, x0)

    print("\n[8] equilibrium control is wealth-independent")
    results["equilibrium structure"] = check_equilibrium_is_wealth_independent(
        market, phi
    )

    print("\n[9] pre-committed dominates at time 0")
    results["pre-committed wins"] = check_precommitted_dominates(market, phi, x0)

    print("\n[10] eps_incons scales as 1/phi and is independent of x0")
    results["eps_incons scaling"] = check_incons_scaling(market, x0)

    print("\n" + "=" * 62)
    for name, ok in results.items():
        print(f"  {name:24s} {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    print(f"  {'overall':24s} {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
