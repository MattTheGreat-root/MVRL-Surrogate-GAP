"""
Reference policies for the multi-period mean-variance problem.

The central object is :class:`LiNgPrecommitted`, the pre-committed optimum of

    max  E[x_T] - phi * Var(x_T)

derived by Li and Ng (2000) via embedding in an auxiliary linear-quadratic
problem.  The optimal control is affine in wealth,

    u*_t(x) = alpha_t x + beta_t,

which is what makes ground-truth comparison possible: a learned policy can be
compared against an exact optimum rather than against a benchmark strategy.

Derivation sketch (for the reader checking the code against the algebra)
-----------------------------------------------------------------------
The auxiliary problem is

    A(lambda, phi):  max E[ lambda * x_T - phi * x_T^2 ].

Writing the value function as V_t(x) = -phi a_t x^2 + lambda b_t x + c_t and
solving the one-step optimisation gives

    u*_t(x) = M_t^{-1} m_t * ( gamma_t - s_t x ),
    gamma_t = lambda b_{t+1} / (2 phi a_{t+1}),

with the backward recursions, started from a_T = b_T = 1,

    a_t = (1 - nu_t) s_t^2 a_{t+1},
    b_t = (1 - nu_t) s_t   b_{t+1}.

The multiplier lambda is then pinned by the first-order condition linking the
auxiliary problem to the original one,

    lambda* = 1 + 2 phi E[x_T*],

which is a fixed point.  Since E[x_T] is affine in lambda the fixed point is
solved exactly by evaluating at two values of lambda; no iteration is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .market import Market


class Policy:
    """A deterministic policy mapping (t, wealth) to risky-asset amounts."""

    def act(self, t: int, x: float) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def act_batch(self, t: int, x: np.ndarray) -> np.ndarray:
        """Vectorised form.  Default implementation loops; subclasses override."""
        return np.stack([self.act(t, float(xi)) for xi in np.atleast_1d(x)])


@dataclass
class AffinePolicy(Policy):
    """u_t(x) = alpha_t x + beta_t, stored per period."""

    alpha: np.ndarray  # (T, m)
    beta: np.ndarray   # (T, m)

    def act(self, t: int, x: float) -> np.ndarray:
        return self.alpha[t] * x + self.beta[t]

    def act_batch(self, t: int, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        return x * self.alpha[t][None, :] + self.beta[t][None, :]


def _auxiliary_coefficients(market: Market) -> tuple[np.ndarray, np.ndarray]:
    """Backward recursions a_t and b_t for t = 0 .. T (length T+1)."""
    T = market.horizon
    a = np.ones(T + 1)
    b = np.ones(T + 1)
    for t in range(T - 1, -1, -1):
        nu = market.nu(t)
        s = market.s(t)
        a[t] = (1.0 - nu) * s * s * a[t + 1]
        b[t] = (1.0 - nu) * s * b[t + 1]
    return a, b


def _policy_for_lambda(market: Market, phi: float, lam: float) -> AffinePolicy:
    """Optimal control of the auxiliary problem A(lambda, phi)."""
    T = market.horizon
    a, b = _auxiliary_coefficients(market)
    alpha = np.zeros((T, market.n_assets))
    beta = np.zeros((T, market.n_assets))
    for t in range(T):
        direction = np.linalg.solve(market.M(t), market.m(t))  # M^{-1} m
        gamma = lam * b[t + 1] / (2.0 * phi * a[t + 1])
        alpha[t] = -market.s(t) * direction
        beta[t] = gamma * direction
    return AffinePolicy(alpha=alpha, beta=beta)


def _expected_terminal_wealth(
    market: Market, policy: AffinePolicy, x0: float
) -> float:
    """E[x_T] under an affine policy, computed exactly by forward recursion.

    Under u_t(x) = alpha_t x + beta_t,
        E[x_{t+1}] = s_t E[x_t] + m_t^T (alpha_t E[x_t] + beta_t).
    """
    expectation = float(x0)
    for t in range(market.horizon):
        m = market.m(t)
        expectation = (
            market.s(t) * expectation
            + m @ (policy.alpha[t] * expectation + policy.beta[t])
        )
    return expectation


def li_ng_precommitted(market: Market, phi: float, x0: float) -> AffinePolicy:
    """The pre-committed mean-variance optimal policy.

    Solves the embedding fixed point ``lambda = 1 + 2 phi E[x_T]`` exactly,
    exploiting the fact that ``E[x_T]`` is affine in ``lambda``.
    """
    if phi <= 0:
        raise ValueError("phi must be positive")

    # E[x_T](lambda) = c0 + c1 * lambda, recovered from two evaluations.
    e0 = _expected_terminal_wealth(market, _policy_for_lambda(market, phi, 0.0), x0)
    e1 = _expected_terminal_wealth(market, _policy_for_lambda(market, phi, 1.0), x0)
    c0, c1 = e0, e1 - e0

    # lambda = 1 + 2 phi (c0 + c1 lambda)
    denominator = 1.0 - 2.0 * phi * c1
    if abs(denominator) < 1e-12:
        raise ValueError(
            "degenerate embedding: the lambda fixed point is not well posed "
            "for this market and phi"
        )
    lam = (1.0 + 2.0 * phi * c0) / denominator
    return _policy_for_lambda(market, phi, lam)


@dataclass
class MyopicPolicy(Policy):
    """Single-period mean-variance optimiser applied repeatedly.

    Maximises E[x_{t+1}] - phi Var(x_{t+1}) one step at a time, giving
    u = Sigma^{-1} m / (2 phi), independent of wealth and of the horizon.
    Included as the third notion of optimality discussed in the proposal.
    """

    market: Market
    phi: float

    def act(self, t: int, x: float) -> np.ndarray:
        return np.linalg.solve(self.market.cov_excess, self.market.m(t)) / (
            2.0 * self.phi
        )

    def act_batch(self, t: int, x: np.ndarray) -> np.ndarray:
        single = self.act(t, 0.0)
        return np.tile(single, (np.atleast_1d(x).size, 1))


@dataclass
class ConstantAmountPolicy(Policy):
    """Holds a fixed vector of amounts in the risky assets every period."""

    amounts: np.ndarray

    def act(self, t: int, x: float) -> np.ndarray:
        return np.asarray(self.amounts, dtype=float)

    def act_batch(self, t: int, x: np.ndarray) -> np.ndarray:
        return np.tile(
            np.asarray(self.amounts, dtype=float), (np.atleast_1d(x).size, 1)
        )
