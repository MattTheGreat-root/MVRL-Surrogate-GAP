"""
Synthetic multi-period market with known moments.

The market is deliberately simple: returns are independent across periods with
first two moments specified by the user.  This is the setting in which the
Li--Ng pre-committed optimum is available in closed form, and it is therefore
the setting in which a learned policy can be compared against ground truth.

Notation follows the proposal:

    s_t             gross return of the riskless asset over period t
    e_t             gross returns of the m risky assets over period t
    P_t = e_t - s_t excess gross returns
    m_t = E[P_t]    mean excess return
    M_t = E[P_t P_t^T]  second moment of excess returns (NOT the covariance)

Wealth evolves as

    x_{t+1} = s_t x_t + P_t^T u_t

where u_t is the vector of amounts (not fractions) held in the risky assets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Market:
    """An i.i.d. multi-period market with known first two moments.

    Parameters
    ----------
    mu_excess : (m,) array
        Mean excess gross return of each risky asset per period, i.e. E[e - s].
    cov_excess : (m, m) array
        Covariance of excess gross returns per period.
    riskless : float
        Gross riskless return per period, s.  ``1.0`` means zero interest.
    horizon : int
        Number of decision periods T.
    """

    mu_excess: np.ndarray
    cov_excess: np.ndarray
    riskless: float
    horizon: int

    def __post_init__(self) -> None:
        self.mu_excess = np.asarray(self.mu_excess, dtype=float).ravel()
        self.cov_excess = np.asarray(self.cov_excess, dtype=float)
        n = self.mu_excess.size
        if self.cov_excess.shape != (n, n):
            raise ValueError(
                f"cov_excess must be {(n, n)}, got {self.cov_excess.shape}"
            )
        if not np.allclose(self.cov_excess, self.cov_excess.T):
            raise ValueError("cov_excess must be symmetric")
        eigenvalues = np.linalg.eigvalsh(self.cov_excess)
        if eigenvalues.min() <= 0:
            raise ValueError("cov_excess must be positive definite")
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")

    # ---------------------------------------------------------------- moments

    @property
    def n_assets(self) -> int:
        return self.mu_excess.size

    def m(self, t: int) -> np.ndarray:
        """Mean excess return E[P_t].  Constant in t for this i.i.d. market."""
        return self.mu_excess

    def M(self, t: int) -> np.ndarray:
        """Second moment E[P_t P_t^T] = Cov(P_t) + m_t m_t^T."""
        return self.cov_excess + np.outer(self.mu_excess, self.mu_excess)

    def s(self, t: int) -> float:
        """Gross riskless return over period t."""
        return float(self.riskless)

    def nu(self, t: int) -> float:
        """The scalar nu_t = m_t^T M_t^{-1} m_t.

        This lies in [0, 1) whenever M_t is positive definite, and controls the
        contraction of the Li--Ng value recursion.  It is the multi-period
        analogue of a squared Sharpe ratio.
        """
        m = self.m(t)
        M = self.M(t)
        value = float(m @ np.linalg.solve(M, m))
        if not (0.0 <= value < 1.0):
            raise ValueError(
                f"nu_{t} = {value:.6f} outside [0, 1); the market specification "
                "is degenerate"
            )
        return value

    # --------------------------------------------------------------- sampling

    def sample_excess(
        self, n_paths: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw excess returns for every period.

        Returns
        -------
        (n_paths, horizon, m) array
        """
        return rng.multivariate_normal(
            self.mu_excess, self.cov_excess, size=(n_paths, self.horizon)
        )
