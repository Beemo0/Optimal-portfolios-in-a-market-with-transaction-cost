"""
Validation and diagnostic routines for the CARA transaction-cost solver.

These are the checks behind the numbers quoted in Section 5.3 of the
paper: tree calibration against exact GBM moments, identification and
sanity-checking of the no-transaction region, solver convergence in the
discretization parameters, and the width-vs-lambda scaling exponent
(fit at several horizons T, to compare against the log-utility,
infinite-horizon asymptotics of Section 4.3 -- see the module docstring
of ``asymptotics.py`` for why an exact match is not expected).
"""
from __future__ import annotations

import sys
import time

import numpy as np

from txcost.cara_solver import solve
from txcost.frictionless import Market, merton_fraction
from txcost.tree import BinomialTree

DEFAULT_MARKET = Market(mu=0.08, r=0.02, sigma=0.20)
DEFAULT_GAMMA = 1.0
DEFAULT_LAMS = (0.0005, 0.001, 0.002, 0.004, 0.008)


def check_binomial_tree_calibration(N: int = 150, T: float = 1.0) -> None:
    """E[S_T] and Var[S_T] under the physical measure should match GBM."""
    tree = BinomialTree(mu=DEFAULT_MARKET.mu, sigma=DEFAULT_MARKET.sigma,
                         T=T, N=N)
    E, Var = tree.terminal_moments()
    E_th, Var_th = BinomialTree.theoretical_moments(
        DEFAULT_MARKET.mu, DEFAULT_MARKET.sigma, T)
    print(f"E[S_T]: binomial={E:.6f}  theory={E_th:.6f}")
    print(f"Var[S_T]: binomial={Var:.6f}  theory={Var_th:.6f}")


def check_no_trade_region(lam: float = 0.01) -> None:
    """No-transaction region should be centered near the Merton fraction
    and widen monotonically with lambda -- read off both the grid-snapped
    and the interpolated boundary for comparison."""
    y_star = merton_fraction(DEFAULT_MARKET, DEFAULT_GAMMA)
    res = solve(lam=lam, nu=lam, N=100, M=150, dy=0.02,
                mu=DEFAULT_MARKET.mu, r=DEFAULT_MARKET.r,
                sigma=DEFAULT_MARKET.sigma, gamma=DEFAULT_GAMMA, n_sweeps=4)
    y_lo, y_hi = res.interpolated_boundary()
    print(f"lam={lam}: y* (theory)={y_star:.3f}  "
          f"NT region (interpolated)=[{y_lo:.4f}, {y_hi:.4f}]  "
          f"(grid-snapped=[{res.y_grid[res.idx_NT.min()]:.3f}, "
          f"{res.y_grid[res.idx_NT.max()]:.3f}])")


def check_convergence(lam: float = 0.01, T: float = 1.0) -> None:
    """
    Convergence of the INTERPOLATED width in (dy, N, n_sweeps) at fixed
    T, lam. A check on the solver's discretization, not on the
    theoretical scaling law.
    """
    print(f"--- convergence in dy, fixed N=150, lam={lam} ---")
    for dy in [0.02, 0.01, 0.005, 0.0025]:
        res = solve(lam=lam, nu=lam, N=150, M=int(4.5 / dy), dy=dy,
                    mu=DEFAULT_MARKET.mu, r=DEFAULT_MARKET.r,
                    sigma=DEFAULT_MARKET.sigma, gamma=DEFAULT_GAMMA, n_sweeps=4)
        print(f"  dy={dy:.4f}  interpolated width={res.interpolated_width():.5f}")

    print(f"--- convergence in N, fixed dy=0.01, lam={lam} ---")
    for N in [50, 100, 150, 300, 600]:
        res = solve(lam=lam, nu=lam, N=N, M=450, dy=0.01,
                    mu=DEFAULT_MARKET.mu, r=DEFAULT_MARKET.r,
                    sigma=DEFAULT_MARKET.sigma, gamma=DEFAULT_GAMMA, n_sweeps=4)
        print(f"  N={N}  interpolated width={res.interpolated_width():.5f}")

    print(f"--- convergence in n_sweeps, fixed dy=0.01, N=150, lam={lam} ---")
    for ns in [2, 4, 8, 16]:
        res = solve(lam=lam, nu=lam, N=150, M=450, dy=0.01,
                    mu=DEFAULT_MARKET.mu, r=DEFAULT_MARKET.r,
                    sigma=DEFAULT_MARKET.sigma, gamma=DEFAULT_GAMMA, n_sweeps=ns)
        print(f"  n_sweeps={ns}  interpolated width={res.interpolated_width():.5f}")

    print(f"--- convergence in grid half-width M, fixed dy=0.01, lam={lam} "
          f"(this is the check that caught bug 4 -- see cara_solver.py) ---")
    for M in [200, 300, 450, 600]:
        res = solve(lam=lam, nu=lam, N=150, M=M, dy=0.01,
                    mu=DEFAULT_MARKET.mu, r=DEFAULT_MARKET.r,
                    sigma=DEFAULT_MARKET.sigma, gamma=DEFAULT_GAMMA, n_sweeps=4)
        print(f"  M={M:4d}  interpolated width={res.interpolated_width():.5f}  "
              f"(grid-snapped={res.grid_width():.5f})")


def fit_width_scaling(T_: float = 1.0, dy: float = 0.01, N: int = 150,
                       n_sweeps: int = 2, lams=DEFAULT_LAMS, M: int = None):
    """
    log-log fit of the INTERPOLATED no-transaction-region width vs.
    lambda, at fixed horizon T_. Returns (exponent, widths).

    Uses ``SolverResult.interpolated_width`` rather than the grid-snapped
    width: see cara_solver.py bug 4 for why the grid-snapped version is
    unreliable for this specific purpose (a log-log fit is dominated by
    its extreme points, and the extreme, smallest-lambda point is exactly
    where grid-snapping quantization noise is largest relative to the
    width itself).
    """
    if M is None:
        M = int(4.5 / dy)
    widths = []
    for lam in lams:
        res = solve(lam=lam, nu=lam, N=N, M=M, dy=dy,
                    mu=DEFAULT_MARKET.mu, r=DEFAULT_MARKET.r,
                    sigma=DEFAULT_MARKET.sigma, gamma=DEFAULT_GAMMA, T=T_,
                    n_sweeps=n_sweeps)
        widths.append(res.interpolated_width())
    slope, _ = np.polyfit(np.log(lams), np.log(widths), 1)
    return slope, widths


def check_T_robustness(T_values=(1, 3, 6, 10)) -> None:
    """
    Fits the width-scaling exponent at several horizons T, using the
    interpolated boundary (bug 4 fix) at a grid wide enough
    (M = int(4.5/dy)) that the exponent no longer depends on M -- see
    cara_solver.py bug 4 and tests/test_cara_solver.py for the check
    that established this.

    CURRENT FINDING (superseding the earlier, grid-snapped-width finding
    that showed a spurious rise at T=10): the exponent drops from ~0.38
    at T=1 to a plateau of ~0.348-0.350 from T=3 through T=10, essentially
    flat (spread of 0.0016 across T=3,6,10). This plateau sits about 5%
    above the theoretical 1/3, and does NOT continue decreasing over the
    horizons checked here. See asymptotics.py and the paper's Section 5.3
    for the interpretation (a genuine finite-horizon-vs-infinite-horizon
    and/or CARA-vs-log-utility gap, not a numerical artifact, since it is
    now confirmed converged in dy, N, n_sweeps, and M).
    """
    print("This can take several minutes -- see docstring for expected runtimes.")
    for T_ in T_values:
        N_ = int(150 * T_)
        t0 = time.time()
        slope, widths = fit_width_scaling(T_=T_, dy=0.01, N=N_)
        print(f"  T={T_:2d}  N={N_:5d}  exponent={slope:.4f}  "
              f"widths={[round(w, 4) for w in widths]}  "
              f"time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    print("=== binomial tree calibration ===")
    check_binomial_tree_calibration()

    print("\n=== no-trade region ===")
    for lam in [0.001, 0.005, 0.01, 0.02, 0.05]:
        check_no_trade_region(lam)

    print("\n=== solver convergence in (dy, N, n_sweeps, M) ===")
    check_convergence()

    print("\n=== width vs. lambda scaling, T=1 ===")
    slope, widths = fit_width_scaling()
    print(f"fitted exponent={slope:.4f}  (theory 1/3={1/3:.4f})")
    print("widths:", [round(w, 4) for w in widths])

    if "--full" in sys.argv:
        print("\n=== exponent vs. horizon T (robustness scan) ===")
        check_T_robustness()
    else:
        print("\n(run with --full to add the multi-horizon robustness scan, "
              "several minutes)")
