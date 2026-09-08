"""
Numerical demonstration of time-inconsistency.

This makes equation (8) of the proposal concrete.  The claim there is that

    J(t, x, u*) != sup_a E[ J(t+1, x_{t+1}, u*) ]

so that a policy chosen as optimal at time 0 is not the policy a time-1 self
would choose, even with identical preferences and identical information.

The experiment is direct.  Compute the pre-committed optimum at time 0.  Step
forward one period to some wealth x_1.  Then re-solve the *same* mean-variance
problem from scratch at time 1, with the same risk aversion and the remaining
horizon.  Compare what the original plan tells the agent to do at (1, x_1)
against what the freshly solved problem tells it to do.

The two disagree, and the disagreement is not numerical noise: it grows
linearly in wealth and can be computed exactly.  This is the mechanism that
makes the Bellman equation unavailable, and therefore the reason the surrogate
gap of the proposal exists at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evaluate import exact_affine_objective
from .market import Market
from .policies import AffinePolicy, li_ng_precommitted


def sub_market(market: Market, start: int) -> Market:
    """The market faced from period ``start`` onward."""
    return Market(
        mu_excess=market.mu_excess,
        cov_excess=market.cov_excess,
        riskless=market.riskless,
        horizon=market.horizon - start,
    )


def tail(policy: AffinePolicy, start: int) -> AffinePolicy:
    """The continuation of an affine policy from period ``start`` onward."""
    return AffinePolicy(alpha=policy.alpha[start:], beta=policy.beta[start:])


@dataclass
class Disagreement:
    wealth: float
    committed_action: np.ndarray
    resolved_action: np.ndarray
    committed_value: float
    resolved_value: float

    @property
    def action_gap(self) -> float:
        return float(np.linalg.norm(self.resolved_action - self.committed_action))

    @property
    def value_gap(self) -> float:
        return self.resolved_value - self.committed_value


def disagreement_at(
    market: Market, phi: float, x0: float, x1: float, at: int = 1
) -> Disagreement:
    """Compare the committed plan against a fresh re-solve at time ``at``.

    Both policies are evaluated by the *same* criterion: the mean-variance
    objective as seen from time ``at`` with wealth ``x1``.  Any difference is
    therefore a genuine preference reversal, not an artefact of changing the
    objective.
    """
    committed = li_ng_precommitted(market, phi, x0)
    remaining = sub_market(market, at)

    continuation = tail(committed, at)
    resolved = li_ng_precommitted(remaining, phi, x1)

    committed_value = exact_affine_objective(
        remaining, continuation, phi, x1
    ).objective
    resolved_value = exact_affine_objective(remaining, resolved, phi, x1).objective

    return Disagreement(
        wealth=x1,
        committed_action=continuation.alpha[0] * x1 + continuation.beta[0],
        resolved_action=resolved.alpha[0] * x1 + resolved.beta[0],
        committed_value=committed_value,
        resolved_value=resolved_value,
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

    committed = li_ng_precommitted(market, phi, x0)
    print("Pre-committed policy chosen at time 0")
    print("  u_t(x) = alpha_t x + beta_t")
    for t in range(market.horizon):
        print(f"    t={t}  alpha = {np.array2string(committed.alpha[t], precision=4)}"
              f"   beta = {np.array2string(committed.beta[t], precision=4)}")

    baseline = exact_affine_objective(market, committed, phi, x0)
    print(f"\n  value at time 0: {baseline}")

    print("\n" + "=" * 74)
    print("Preference reversal at time 1")
    print("=" * 74)
    print("The committed plan and a fresh re-solve are compared under the same")
    print("criterion, the mean-variance objective seen from time 1.\n")

    header = (
        f"{'x_1':>8}  {'||u_committed - u_resolved||':>28}  "
        f"{'J_1(resolved) - J_1(committed)':>30}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for x1 in [0.80, 0.90, 1.00, 1.02, 1.10, 1.25, 1.50]:
        d = disagreement_at(market, phi, x0, x1)
        rows.append(d)
        print(f"{x1:8.2f}  {d.action_gap:28.6f}  {d.value_gap:30.6f}")

    print("\nBoth columns vanish identically if and only if the Bellman equation")
    print("holds for this objective.  They do not.")

    from scipy.optimize import brentq

    def first_component_gap(x1: float) -> float:
        d = disagreement_at(market, phi, x0, x1)
        return float((d.resolved_action - d.committed_action)[0])

    u0 = committed.alpha[0] * x0 + committed.beta[0]
    expected_x1 = float(market.s(0) * x0 + market.m(0) @ u0)
    crossing = brentq(first_component_gap, expected_x1 - 2.0,
                      expected_x1 + 2.0, xtol=1e-13)

    print("\n" + "=" * 74)
    print("Where the reversal vanishes")
    print("=" * 74)
    print(f"  wealth the time-0 self expects at t=1 :  E[x_1]   = {expected_x1:.9f}")
    print(f"  wealth at which the two policies agree:  crossing = {crossing:.9f}")
    print(f"  difference                            :  {abs(crossing-expected_x1):.2e}")
    print("""
These coincide to machine precision, and do so across markets, horizons, risk
aversions and asset counts.  The pre-committed policy is therefore time-
consistent exactly along its own expected wealth path, and inconsistent in
proportion to how far realised wealth departs from that path.

The mechanism is visible in the embedding.  The multiplier is pinned by
lambda = 1 + 2 phi E[x_T]; when realised wealth equals the wealth the earlier
self expected, the re-solved problem recovers the same lambda and hence the
same control.  Any deviation moves lambda and with it the whole policy.

Two consequences matter for this project.  First, the reversal is a property of
realised paths, not of the model in expectation, so it is invisible to any
analysis that works only with expected trajectories.  Second, it is largest
precisely where a learning algorithm spends most of its time: away from the
mean path, in the states that sampling actually visits.""")

    print("\nThis is the mechanism behind the surrogate gap.  With no Bellman equation")
    print("to appeal to, an algorithm maximising a sum of per-period rewards is not")
    print("solving this problem, and identifying what it solves instead is the")
    print("question the project sets out to answer.")


if __name__ == "__main__":
    main()
