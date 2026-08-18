"""
Cox-Ross-Rubinstein binomial tree, calibrated under the PHYSICAL measure.

Section 5.2 of the paper is explicit that the tree here is calibrated
under P, not under the risk-neutral measure: Q(t,y,S) is an expected-
utility value, not a hedging price, even in the phi != 0 (indifference
pricing) case of Appendix A. This is the one function that builds the
tree consistently for every module in this package, so that the
convention cannot silently drift between the solver, the diagnostics,
and the notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinomialTree:
    """A recombining CRR tree on [0, T], calibrated under the physical
    measure with drift mu, volatility sigma, N steps.

    u = exp(sigma*sqrt(dt)), d = 1/u, p = (exp(mu*dt) - d) / (u - d).
    """

    mu: float
    sigma: float
    T: float
    N: int
    S0: float = 1.0

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        if self.N <= 0:
            raise ValueError("N must be a positive integer.")

    @property
    def dt(self) -> float:
        return self.T / self.N

    @property
    def u(self) -> float:
        return np.exp(self.sigma * np.sqrt(self.dt))

    @property
    def d(self) -> float:
        return 1.0 / self.u

    @property
    def p(self) -> float:
        """Physical-measure up-probability. Raises if it falls outside
        (0, 1) -- a sign that dt is too coarse for the given (mu, sigma),
        rather than silently returning a nonsensical value.
        """
        p = (np.exp(self.mu * self.dt) - self.d) / (self.u - self.d)
        if not (0.0 < p < 1.0):
            raise ValueError(
                f"p={p:.6f} outside (0,1): refine dt (increase N) or "
                f"check mu/sigma."
            )
        return p

    def nodes_at(self, n: int) -> np.ndarray:
        """Stock price at every node of step n, ordered from all-down
        (index 0) to all-up (index n)."""
        j = np.arange(0, n + 1)
        return self.S0 * self.u**j * self.d ** (n - j)

    def terminal_moments(self) -> tuple[float, float]:
        """(E[S_T], Var[S_T]) computed exactly from the tree's binomial
        distribution -- used to check the tree calibration against the
        theoretical GBM moments in ``tests/test_tree.py``."""
        from scipy.stats import binom

        S_N = self.nodes_at(self.N)
        probs = binom.pmf(np.arange(0, self.N + 1), self.N, self.p)
        E = np.sum(probs * S_N)
        Var = np.sum(probs * S_N**2) - E**2
        return E, Var

    @staticmethod
    def theoretical_moments(mu: float, sigma: float, T: float,
                             S0: float = 1.0) -> tuple[float, float]:
        """(E[S_T], Var[S_T]) under the exact GBM law, for comparison."""
        E = S0 * np.exp(mu * T)
        Var = S0**2 * np.exp(2 * mu * T) * (np.exp(sigma**2 * T) - 1)
        return E, Var
