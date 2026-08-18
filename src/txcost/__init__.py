"""
txcost: portfolio optimization under proportional transaction costs.

Companion code for the paper's numerical chapter (Section 5): a
finite-horizon CARA solver (Davis-Panas-Zariphopoulou 1993 dimension
reduction) on a binomial tree, with a policy-iteration scheme for the
free boundary, plus the diagnostics used to validate it against the
small-cost asymptotics of Section 4.
"""
from txcost.cara_solver import SolverResult, solve
from txcost.frictionless import Market, merton_fraction, merton_shares
from txcost.tree import BinomialTree

__all__ = [
    "solve",
    "SolverResult",
    "Market",
    "merton_fraction",
    "merton_shares",
    "BinomialTree",
]
