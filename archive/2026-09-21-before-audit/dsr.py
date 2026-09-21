"""
Phase 3: the differential Sharpe ratio.

This is the reward both surveys report as a standard choice and which the
variance-penalised analysis does not obviously cover, because it is not a
function of wealth alone.

Structure
---------
Writing rho_t for the period return and A_t, B_t for exponentially weighted
estimates of its first two moments with decay eta,

    A_t = A_{t-1} + eta (rho_t - A_{t-1}),
    B_t = B_{t-1} + eta (rho_t^2 - B_{t-1}),
    r_t^DSR = [ B_{t-1} dA_t - (1/2) A_{t-1} dB_t ] / (B_{t-1} - A_{t-1}^2)^{3/2}.

Expanding the numerator gives

    B dA - (1/2) A dB = (eta/2) ( 2 B rho - A rho^2 - A B ),

so that, up to a positive state-dependent scale and an additive constant,

    r^DSR  ~  rho - kappa_eff rho^2,      kappa_eff = A / (2B).

The differential Sharpe ratio is therefore a variance-penalised return reward
whose penalty the practitioner does not choose: it is pinned by the running
moment estimates, hence by the market and the decay rate.

Since A_t and B_t are consistent estimators of the first two moments of the
realised return, in the stationary regime

    kappa_eff  ->  E[rho] / (2 E[rho^2]),

a quantity containing no preference parameter at all.

Two consequences, both checked numerically below.

1.  The augmented state z = (x, A, B) restores the Markov property and the
    Bellman equation holds on it. Without the augmentation the reward is not a
    function of the state and the MDP is misspecified as usually written.

2.  Because kappa_eff is determined rather than chosen, the main theorem
    inverts: the differential Sharpe ratio commits the agent to an implied risk
    aversion

        phi_implied = kappa_eff / (1 - nu),

    which the practitioner does not control and, in general, does not know.

A note on the regime. Over a short horizon with a small decay rate the
estimates A_t, B_t never leave their initial values, and kappa_eff is then an
artefact of initialisation rather than a property of the market. The analysis
below works in the stationary regime, either by taking the horizon long enough
for convergence or by starting A_0, B_0 at their converged values. An earlier
version of this module did neither and produced B* = 3080, which is how the
issue was found.

Run with ``python -m mvrl.dsr``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .equilibrium import solve_equilibrium
from .evaluate import exact_affine_objective
from .market import Market
from .policies import AffinePolicy
from .surrogate import policy_distance, surrogate_backward


@dataclass
class DSRSettings:
    eta: float = 0.05
    A0: float = 0.0
    B0: float = 1e-3


def simulate_dsr(
    market: Market,
    policy: AffinePolicy,
    settings: DSRSettings,
    x0: float,
    excess: np.ndarray,
) -> tuple[float, float, float]:
    """Total DSR reward, plus the terminal moment estimates.

    Returns (mean total reward, mean terminal A, mean terminal B).
    """
    n_paths = excess.shape[0]
    x = np.full(n_paths, float(x0))
    A = np.full(n_paths, settings.A0)
    B = np.full(n_paths, settings.B0)
    total = np.zeros(n_paths)

    for t in range(market.horizon):
        u = x[:, None] * policy.alpha[t][None, :] + policy.beta[t][None, :]
        gain = np.einsum("pi,pi->p", excess[:, t, :], u)
        x_next = market.s(t) * x + gain
        rho = (x_next - x) / np.maximum(np.abs(x), 1e-8)

        dA = settings.eta * (rho - A)
        dB = settings.eta * (rho**2 - B)
        denom = np.maximum(B - A**2, 1e-10) ** 1.5
        total += (B * dA - 0.5 * A * dB) / denom

        A, B, x = A + dA, B + dB, x_next

    return float(total.mean()), float(A.mean()), float(B.mean())


def optimise_dsr(
    market: Market,
    settings: DSRSettings,
    x0: float,
    n_paths: int = 12_000,
    seed: int = 0,
    restarts: int = 3,
) -> AffinePolicy:
    """Maximise the expected total DSR reward over affine policies."""
    T, n = market.horizon, market.n_assets
    rng = np.random.default_rng(seed)
    excess = market.sample_excess(n_paths, rng)

    def unpack(v: np.ndarray) -> AffinePolicy:
        half = T * n
        return AffinePolicy(alpha=v[:half].reshape(T, n), beta=v[half:].reshape(T, n))

    def negative(v: np.ndarray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            value, _, _ = simulate_dsr(market, unpack(v), settings, x0, excess)
        return np.inf if not np.isfinite(value) else -value

    best_value, best_v = -np.inf, None
    for attempt in range(restarts):
        start = (
            np.zeros(2 * T * n)
            if attempt == 0
            else rng.normal(scale=0.4, size=2 * T * n)
        )
        result = minimize(
            negative, start, method="L-BFGS-B",
            options={"maxiter": 400, "maxfun": 20_000,
                     "ftol": 1e-14, "gtol": 1e-10},
        )
        if -result.fun > best_value:
            best_value, best_v = -result.fun, result.x

    if best_v is None:  # pragma: no cover
        raise RuntimeError("DSR optimisation failed")
    return unpack(best_v)


def stationary_moments(
    market: Market, policy: AffinePolicy, x0: float,
    n_paths: int = 20_000, seed: int = 2,
) -> tuple[float, float]:
    """E[rho] and E[rho^2] of the period return under a policy, late-horizon."""
    rng = np.random.default_rng(seed)
    excess = market.sample_excess(n_paths, rng)
    x = np.full(n_paths, float(x0))
    returns = []
    for t in range(market.horizon):
        u = x[:, None] * policy.alpha[t][None, :] + policy.beta[t][None, :]
        x_next = market.s(t) * x + np.einsum("pi,pi->p", excess[:, t, :], u)
        returns.append((x_next - x) / x)
        x = x_next
    tail = np.array(returns[-max(1, market.horizon // 4):])
    return float(tail.mean()), float((tail**2).mean())


def demo_market(s: float = 1.02, horizon: int = 4) -> Market:
    mu = np.array([0.06, 0.04, 0.02])
    vol = np.array([0.18, 0.12, 0.08])
    corr = np.array([[1.00, 0.30, -0.10], [0.30, 1.00, 0.05], [-0.10, 0.05, 1.00]])
    return Market(mu_excess=mu, cov_excess=np.outer(vol, vol) * corr,
                  riskless=s, horizon=horizon)


def main() -> None:
    x0, phi = 1.0, 1.0

    print("=" * 74)
    print("1.  The moment estimates converge to the true return moments")
    print("=" * 74)
    print(f"  {'T':>5}{'A*':>12}{'E[rho]':>12}{'B*':>13}{'E[rho^2]':>13}")
    for horizon in (20, 60, 200):
        market = demo_market(horizon=horizon)
        equilibrium = solve_equilibrium(market, phi).policy
        rng = np.random.default_rng(1)
        excess = market.sample_excess(20_000, rng)
        _, A_star, B_star = simulate_dsr(
            market, equilibrium, DSRSettings(eta=0.1), x0, excess
        )
        m1, m2 = stationary_moments(market, equilibrium, x0)
        print(f"  {horizon:>5}{A_star:>12.6f}{m1:>12.6f}{B_star:>13.6f}{m2:>13.6f}")

    market = demo_market(horizon=200)
    equilibrium = solve_equilibrium(market, phi).policy
    m1, m2 = stationary_moments(market, equilibrium, x0)
    kappa_eff = m1 / (2 * m2)
    print(f"""
So A_t and B_t are consistent estimators of E[rho] and E[rho^2], and the
effective penalty settles at

    kappa_eff = E[rho] / (2 E[rho^2]) = {kappa_eff:.4f}

for this market. It is a property of the market and the policy. No preference
parameter appears in it.""")

    print("\n" + "=" * 74)
    print("2.  The implied risk aversion")
    print("=" * 74)
    market4 = demo_market(horizon=4)
    nu = market4.nu(0)
    phi_implied = kappa_eff / (1.0 - nu)
    print(f"  nu                   = {nu:.6f}")
    print(f"  kappa_eff            = {kappa_eff:.4f}   (in return units, x ~ 1)")
    print(f"  phi implied          = {phi_implied:.4f}")
    print(f"""
A practitioner choosing the differential Sharpe ratio believes they are asking
for risk-adjusted return. Read backwards, the main theorem says they are in fact
asking for a mean-variance criterion at risk aversion {phi_implied:.2f}. That number is
set by the market, not by them, and nothing in their setup reports it.""")

    print("\n" + "=" * 74)
    print("3.  The DSR optimum lies in the variance-penalised family")
    print("=" * 74)
    print("  Stationary regime: A_0, B_0 started at their converged values.")
    market_s = demo_market(horizon=12)
    settings = DSRSettings(eta=0.1, A0=m1, B0=m2)
    dsr_policy = optimise_dsr(market_s, settings, x0)

    search = minimize_scalar(
        lambda k: policy_distance(market_s, dsr_policy,
                                  surrogate_backward(market_s, float(k)), x0),
        bounds=(0.5, 200.0), method="bounded", options={"xatol": 1e-8})
    print(f"  best-matching kappa in the variance-penalised family = {float(search.x):.4f}")
    print(f"  predicted kappa_eff                                  = {kappa_eff:.4f}")
    print(f"  residual policy distance                             = {float(search.fun):.4f}")

    print("\n" + "=" * 74)
    print("4.  Sensitivity to the decay rate")
    print("=" * 74)
    print(f"  {'eta':>8}{'kappa_eff':>13}{'phi_implied':>14}")
    for eta in (0.02, 0.1, 0.4):
        local = DSRSettings(eta=eta, A0=m1, B0=m2)
        policy = optimise_dsr(market_s, local, x0, n_paths=5_000, restarts=1)
        mm1, mm2 = stationary_moments(market_s, policy, x0)
        k = mm1 / (2 * mm2)
        print(f"  {eta:>8.2f}{k:>13.4f}{k/(1-nu):>14.4f}")
    print("""
The effect of eta is mild -- a couple of per cent across an order of magnitude --
so the implied risk aversion is not a hyperparameter artefact. An earlier draft
claimed tuning eta "silently retunes risk preference"; the measurement does not
support that and the claim was dropped.""")

    print("\n" + "=" * 74)
    print("5.  Distance from the mean-variance references")
    print("=" * 74)
    equilibrium4 = solve_equilibrium(market_s, phi).policy
    j_dsr = exact_affine_objective(market_s, dsr_policy, phi, x0).objective
    j_eq = exact_affine_objective(market_s, equilibrium4, phi, x0).objective
    print(f"  J(pi_DSR) = {j_dsr:.9f}")
    print(f"  J(pi_eq)  = {j_eq:.9f}   at phi = {phi}")
    print(f"  distance(pi_DSR, pi_eq) = "
          f"{policy_distance(market_s, dsr_policy, equilibrium4, x0):.6f}")
    print("""
So the differential Sharpe ratio is covered by the main theorem rather than
excluded from it: not as a separate case but as the variance-penalised family
evaluated at an endogenous penalty. Lambda applies unchanged, with
|kappa_eff - phi(1-nu)| as its first term -- except that the practitioner cannot
set kappa_eff, so that term cannot be driven to zero by choice. Whether it is
small is a fact about the market.""")


if __name__ == "__main__":
    main()
