"""
Frictionless benchmarks used throughout the numerical chapter (Section 3.2
of the paper): the constant Merton fraction, and the CARA analogue used to
center the no-transaction-region grid in ``cara_solver.py``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    """One riskless asset at rate r, one risky asset with drift mu and
    volatility sigma, both under the physical measure."""

    mu: float
    r: float
    sigma: float

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")


def merton_fraction(market: Market, gamma: float) -> float:
    """CRRA frictionless optimal fraction of wealth in the risky asset,
    pi* = (mu - r) / (gamma * sigma^2)  (Section 3.2, eq. for pi*)."""
    return (market.mu - market.r) / (gamma * market.sigma**2)


def merton_shares(market: Market, gamma: float, wealth: float,
                   S0: float = 1.0) -> float:
    """Number of shares y* = pi* * wealth / S0 corresponding to the
    Merton fraction -- this is the quantity the CARA solver's y-grid is
    centered around (Section 5.3 uses wealth = S0 = 1, gamma = 1, giving
    y* = 1.5 for the reference parameters)."""
    return merton_fraction(market, gamma) * wealth / S0
