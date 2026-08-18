"""
Small-cost asymptotic benchmarks from Section 4 of the paper, used only
for COMPARISON with the numerical CARA / finite-horizon results of
Section 5 -- these formulas are for a *different* problem (log-utility,
infinite horizon) and must not be mistaken for a ground truth the CARA
solver should reproduce exactly. See ``cara_solver.py`` bug-history and
the module docstring of ``diagnostics.py`` for the precise distinction.
"""
from __future__ import annotations

ROGERS_EXPONENT = 1.0 / 3.0
"""The universal small-cost width-scaling exponent (Rogers 2004): a
generic envelope-theorem argument, applicable regardless of the utility
function, but silent on the leading constant or the rate of convergence
of a finite-horizon problem to this asymptotic value."""


def gerhold_muhle_karbe_schachermayer_leading_constant(theta: float) -> float:
    """Leading constant in the log-utility, infinite-horizon no-transaction
    region width expansion (Corollary 6.2 of Gerhold, Muhle-Karbe &
    Schachermayer 2013):

        width ~ (6 * theta^2 * (1-theta)^2)^(1/3) * lambda^(1/3) + O(lambda)

    where theta = mu/sigma^2 is the frictionless log-optimal fraction
    (r=0 normalization, see Section 4.3 of the paper). Provided here only
    so that a reader can reproduce the exact comparison point; the CARA,
    finite-horizon solver in this package is NOT expected to match it
    (different utility, different horizon) -- see cara_solver.py.
    """
    return (6.0 * theta**2 * (1 - theta) ** 2) ** (1.0 / 3.0)
