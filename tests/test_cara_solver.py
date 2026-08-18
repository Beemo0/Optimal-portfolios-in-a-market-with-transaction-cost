"""
Tests for txcost.cara_solver.

Each of the four bugs found during development (see the module docstring
of cara_solver.py) has a corresponding regression test here:
  1/2. test_no_profitable_round_trip           (sign errors)
  3.   test_matches_independent_expspace_reference (overflow / log-space)
  4.   test_interpolated_width_independent_of_grid_halfwidth (M-dependence)
"""
import warnings

import numpy as np
import pytest

from txcost.cara_solver import solve


COMMON = dict(mu=0.08, r=0.02, sigma=0.20, gamma=1.0)


def _reference_solve_expspace(lam, nu, N, M, dy, T, n_sweeps, **kw):
    """
    Independent, deliberately naive exp-space implementation of the same
    recursion, written from scratch (not sharing code with
    cara_solver.solve) as a ground truth for small N/M where overflow
    cannot occur. Intentionally unvectorized over k, to keep the logic
    as transparent as possible for a from-scratch cross-check.
    """
    mu, r, sigma, gamma = kw["mu"], kw["r"], kw["sigma"], kw["gamma"]
    S0 = kw.get("S0", 1.0)
    dt = T / N
    u, d = np.exp(sigma * np.sqrt(dt)), np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(mu * dt) - d) / (u - d)
    y = np.arange(-M, M + 1) * dy
    K = len(y)

    def ell(yy, S):
        return yy * S - S * (nu * max(yy, 0) + lam * max(-yy, 0))

    S_N = S0 * u ** np.arange(N + 1) * d ** (N - np.arange(N + 1))
    Q = np.array([[np.exp(-gamma * ell(yy, S)) for yy in y] for S in S_N])

    for n in range(N - 1, -1, -1):
        disc = np.exp(r * (T - n * dt))
        Sj = S0 * u ** np.arange(n + 1) * d ** (n - np.arange(n + 1))
        Q_next = Q
        Q_NT = p * Q_next[1:n + 2, :] + (1 - p) * Q_next[0:n + 1, :]
        Qcur = Q_NT.copy()
        for _ in range(n_sweeps):
            for jj in range(n + 1):
                for k in range(1, K):
                    cand = np.exp(-gamma * (1 - nu) * Sj[jj] * dy * disc) * Qcur[jj, k - 1]
                    Qcur[jj, k] = min(Qcur[jj, k], cand)
                for k in range(K - 2, -1, -1):
                    cand = np.exp(gamma * (1 + lam) * Sj[jj] * dy * disc) * Qcur[jj, k + 1]
                    Qcur[jj, k] = min(Qcur[jj, k], cand)
        Q = Qcur
    return Q[0, :], y


@pytest.mark.parametrize("N,M,dy,T", [
    (1, 3, 0.1, 1.0),
    (2, 4, 0.1, 1.0),
    (5, 6, 0.05, 0.5),
])
def test_matches_independent_expspace_reference(N, M, dy, T):
    """The log-space solver must reproduce a from-scratch, naive exp-space
    implementation exactly (to float64 precision) wherever the latter
    cannot overflow. This is the regression test for bug 3 -- it does not
    merely check "no crash", it checks numerical agreement with an
    independently written reference."""
    lam = nu = 0.01
    res = solve(lam=lam, nu=nu, N=N, M=M, dy=dy, T=T, n_sweeps=4, **COMMON)
    Q_ref, y_ref = _reference_solve_expspace(
        lam=lam, nu=nu, N=N, M=M, dy=dy, T=T, n_sweeps=4, **COMMON)
    np.testing.assert_allclose(np.exp(res.L_final), Q_ref, rtol=1e-10)
    np.testing.assert_allclose(res.y_grid, y_ref)


def test_no_profitable_round_trip():
    """
    Trading is a candidate at every grid point, and the recursion picks
    the minimum of Q over {no-trade, buy, sell}. Since no-trade is always
    feasible, Q after the sweep can never exceed Q before it: L_final <=
    L_NT pointwise. A sign error in the cost formulas (bugs 1/2 in the
    module docstring) breaks exactly this: it lets a round trip look
    artificially profitable, i.e. Q_final > Q_NT somewhere.
    """
    res = solve(lam=0.01, nu=0.01, N=100, M=150, dy=0.02, T=1.0,
                n_sweeps=4, **COMMON)
    assert np.all(res.L_final <= res.L_NT + 1e-12)


@pytest.mark.parametrize("lam", [0.001, 0.005, 0.01, 0.02])
def test_no_trade_region_widens_with_lambda(lam):
    """Sanity check against the qualitative picture of Section 4.2:
    larger transaction costs should give a wider no-transaction region."""
    widths = {}
    for l in (lam, lam * 2):
        res = solve(lam=l, nu=l, N=100, M=150, dy=0.02, T=1.0,
                    n_sweeps=4, **COMMON)
        widths[l] = res.interpolated_width()
    assert widths[lam * 2] > widths[lam]


def test_interpolated_width_independent_of_grid_halfwidth():
    """
    Regression test for bug 4: the grid-snapped width is sensitive to a
    single grid-cell rounding at the smallest lambda in a scaling fit,
    which is amplified by the log-log fit used elsewhere in this
    package. The interpolated width must be essentially unchanged when
    the grid is widened at fixed dy (here M=250 vs M=400, half-width 2.5
    vs 4.0 in y-units -- both comfortably containing y*=1.5 plus the
    no-transaction region, so any remaining difference reflects the
    boundary measurement, not a too-narrow grid -- a smaller-scale
    version of the M=300 vs M=450 check at T=6 that originally caught
    this bug).
    """
    lam = 0.001
    widths_interp = []
    widths_grid = []
    for M in (250, 400):
        res = solve(lam=lam, nu=lam, N=100, M=M, dy=0.01, T=1.0,
                    n_sweeps=2, **COMMON)
        widths_interp.append(res.interpolated_width())
        widths_grid.append(res.grid_width())
    # interpolated width: agree to a small fraction of one grid cell
    assert abs(widths_interp[0] - widths_interp[1]) < 0.1 * 0.01
    # (not asserting the grid-snapped width is unstable here -- it may or
    # may not flip at this particular lambda/M pair; the point of this
    # test is only that the interpolated version is robust)


def test_interpolated_boundary_warns_when_region_touches_grid_edge():
    """If the no-transaction region reaches the edge of the y-grid, the
    grid is too narrow for the given lambda and the interpolation cannot
    recover a meaningful sub-grid boundary -- this must be surfaced to
    the caller, not silently return a wrong number."""
    # Deliberately tiny, narrow grid relative to lambda=0.05 (whose NT
    # region is wide, per validate/diagnostics output in earlier runs).
    res = solve(lam=0.05, nu=0.05, N=50, M=5, dy=0.3, T=1.0, n_sweeps=4,
                **COMMON)
    with pytest.warns(UserWarning, match="grid edge"):
        res.interpolated_boundary()


def test_raises_on_invalid_probability():
    """dt too coarse for the given (mu, sigma) should raise, not silently
    produce a nonsensical p outside (0,1)."""
    with pytest.raises(ValueError):
        solve(lam=0.01, nu=0.01, N=1, M=10, dy=0.1, T=50.0,
              mu=5.0, r=0.02, sigma=0.01, gamma=1.0)
