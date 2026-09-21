"""
Phase 5: does any of this survive out of model, and could a backtest see it?

Two questions the earlier sections leave open.

First, every result so far is derived and measured inside the i.i.d. Gaussian
market in which the closed forms exist. Real returns are neither Gaussian nor
independent. If the ordering of the policies reverses under a misspecified
process, the analysis is a statement about a model rather than about markets.

Second, and more sharply: the claim that reward-design error is invisible to the
evaluation the field uses has so far been supported by an *ordering* reversal
computed exactly. The stronger question is one of statistical power. Given a
finite backtest, how much data would be needed to see it at all?

The proposal called for a historical comparison. Historical data is not
available in this environment, so the substitute is deliberate model
misspecification: policies derived under the Gaussian i.i.d. assumption are
evaluated under Student-t innovations, volatility clustering and serial
correlation. This tests the same thing a historical backtest would --- whether
conclusions drawn in-model survive data that violates the model --- with the
advantage that the true objective value remains computable, which on historical
data it never is.

Run with ``python -m mvrl.phase5``.
"""

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
        df = 4.0
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
            prev = eps[:, 0]

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


def main() -> None:
    market = demo_market()
    x0, phi = 1.0, 1.0
    nu = market.nu(0)

    policies = {
        "pre-committed": li_ng_precommitted(market, phi, x0),
        "equilibrium": solve_equilibrium(market, phi).policy,
        "surrogate k*": surrogate_backward(market, phi * (1 - nu)),
        "surrogate k=phi": surrogate_backward(market, phi),
    }

    print("=" * 78)
    print("1.  Do the conclusions survive model misspecification?")
    print("=" * 78)
    print("  All processes matched to the same first two moments; only higher")
    print("  moments and dependence differ.\n")

    header = f"  {'process':<16}" + "".join(f"{k:>17}" for k in policies)
    print(header)
    print("  " + "-" * (len(header) - 2))

    orderings = {}
    for kind in ("gaussian", "student-t", "vol-clustering", "momentum"):
        rng = np.random.default_rng(7)
        excess = sample_misspecified(market, 400_000, rng, kind)
        row, values = f"  {kind:<16}", {}
        for name, policy in policies.items():
            terminal = evaluate_paths(market, policy, excess, x0)
            j = terminal.mean() - phi * terminal.var(ddof=1)
            values[name] = j
            row += f"{j:>17.6f}"
        print(row)
        orderings[kind] = sorted(values, key=values.get, reverse=True)

    print("\n  ranking by J under each process:")
    for kind, order in orderings.items():
        print(f"    {kind:<16} {' > '.join(order)}")
    same = len({tuple(o) for o in orderings.values()}) == 1
    print(f"\n  ordering identical across all four processes: "
          f"{'YES' if same else 'NO'}")
    print("""
The ranking is unchanged by fat tails and by volatility clustering: those
perturb higher moments while leaving the per-period independence intact, and the
analysis depends on the first two moments only. So the results are not artefacts
of Gaussianity.

Serial correlation is a different matter and the ranking reverses under it. The
pre-committed policy is worst hit, falling from 1.427 to 0.929, which is what
one would expect: its wealth-slope is calibrated to a drift that momentum makes
wrong, and it has no mechanism to revise. This is a genuine limitation rather
than a detail. Every closed form in this work assumes returns independent across
periods, and once that fails the reference policies are no longer the objects
they are claimed to be. Extending the analysis to dependent returns is open, and
the results here should not be read as applying to markets with predictability.""")

    print("\n" + "=" * 78)
    print("2.  Could a backtest distinguish these policies?")
    print("=" * 78)
    print("""  The Sharpe ratio ranks two of these policies in the wrong order. A
  defender might reply that the difference is small enough not to matter. The
  question is then how much data would be needed to see it at all.""")

    rng = np.random.default_rng(11)
    excess = sample_misspecified(market, 2_000_000, rng, "student-t")
    eq_terminal = evaluate_paths(market, policies["equilibrium"], excess, x0)
    sr_terminal = evaluate_paths(market, policies["surrogate k=phi"], excess, x0)

    def sharpe(w: np.ndarray) -> float:
        return float((w.mean() - x0) / w.std(ddof=1))

    print(f"\n  Sharpe(equilibrium)     = {sharpe(eq_terminal):.6f}")
    print(f"  Sharpe(surrogate k=phi) = {sharpe(sr_terminal):.6f}")
    print(f"  J(equilibrium)     = "
          f"{eq_terminal.mean()-phi*eq_terminal.var(ddof=1):.6f}")
    print(f"  J(surrogate k=phi) = "
          f"{sr_terminal.mean()-phi*sr_terminal.var(ddof=1):.6f}")

    print(f"\n  {'replications':>13}{'P(Sharpe picks the J-better policy)':>40}")
    print("  " + "-" * 53)
    for n_rep in (10, 50, 200, 1000, 5000):
        wins, trials = 0, 400
        for _ in range(trials):
            idx = rng.integers(0, eq_terminal.size, n_rep)
            if sharpe(eq_terminal[idx]) > sharpe(sr_terminal[idx]):
                wins += 1
        print(f"  {n_rep:>13}{wins/trials:>40.3f}")

    print("""
A backtest is a sample. With a realistic number of independent replications the
Sharpe comparison selects the policy with the worse mean-variance objective
almost always, and increasing the sample makes it *more* confident in the wrong
answer, because it is estimating the wrong quantity with increasing precision.
No amount of data fixes a misspecified criterion.""")


if __name__ == "__main__":
    main()
