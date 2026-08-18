"""
Finite-horizon CARA portfolio choice under proportional transaction costs.
Implements the binomial-tree / policy-iteration scheme of Section 5.2
(dimension reduction of Davis-Panas-Zariphopoulou 1993, phi=0 case).

BUG HISTORY (kept here rather than in a changelog file, because each of
these changes what the *numbers* in Section 5.3 mean, not just the code)
--------------------------------------------------------------------------
1. Two sign errors in the buy/sell cost formulas (direction of the
   recursion, then the sign of the cost exponent), caught by checking
   that a round-trip buy-then-sell can never increase Q -- it did, in
   the buggy versions, which is economically impossible with positive
   costs. See ``tests/test_cara_solver.py::test_no_profitable_round_trip``.

2. A misleading diagnostic, not a code bug: Q(t,y,S) must not be compared
   across different y at face value -- at fixed cash x, larger y means
   more total wealth, so Q decreasing in y is expected. The correct
   no-transaction-region test compares Q *after* the sweep to Q_NT
   *before* it, at the SAME y.

3. Float64 overflow at long horizons (T >~ 4-6 with the grids used here):
   the original exp-space recursion produced +inf at extreme, low-
   probability tree nodes (large |y| combined with an extreme S). This
   is combined with a plain np.minimum() for buy/sell selection, which
   discards a +inf candidate whenever a finite one exists -- so overflow
   at T=3 happened to self-heal before reaching t=0 (correct answer, by
   luck). At T=6 it did not: an inf met a near-zero cost multiplier,
   giving 0*inf=nan, which then contaminates every node back to t=0.
   Fixed by carrying the entire recursion in log-space (L = log Q): the
   terminal condition needs no exp() at all, the no-trade candidate uses
   a numerically stable log-sum-exp, and the min-selection becomes a min
   in log-space. This cannot produce +inf, by construction, not by luck.

4. The no-transaction-region *width*, when read off by snapping to the
   nearest grid point (y_grid[idx.max()] - y_grid[idx.min()]), is
   sensitive to which side of a grid cell the true (continuous) boundary
   falls on. This was invisible until the width-vs-lambda scaling
   exponent was refit at a wider grid (M=300 -> M=450, same dy): a
   single grid-step change in the width at the SMALLEST lambda (the
   point with the most leverage on a log-log fit) moved the fitted
   exponent from 0.364 to 0.349 -- a 4% change from something that
   should be a pure grid-resolution artifact. Fixed by interpolating the
   true sub-grid boundary from the smooth-pasting condition (the point
   where the buy, resp. sell, candidate exactly equals the no-trade
   candidate), rather than snapping to the grid. See
   ``interpolated_boundary`` below and
   ``tests/test_cara_solver.py::test_boundary_independent_of_grid_halfwidth``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SolverResult:
    """Result of `solve` at t=0 (single tree node), packaged so callers
    don't have to remember index conventions."""

    y_grid: np.ndarray
    L_NT: np.ndarray       # log Q_NT, before the buy/sell sweep
    L_final: np.ndarray    # log Q, after the sweep
    idx_NT: np.ndarray     # grid indices classified as no-trade
    log_buy_cost: float    # scalar candidate-cost terms at t=0 (j=0)
    log_sell_cost: float

    @property
    def is_NT(self) -> np.ndarray:
        return self.L_final == self.L_NT

    def grid_width(self) -> float:
        """No-transaction-region width, snapped to the grid. Kept for
        backward compatibility / sanity cross-checks; prefer
        ``interpolated_width`` for anything quoted in the paper (see bug
        4 in the module docstring)."""
        return self.y_grid[self.idx_NT.max()] - self.y_grid[self.idx_NT.min()]

    def interpolated_boundary(self) -> tuple[float, float]:
        """
        Sub-grid location of the lower and upper free boundary, using
        the smooth-pasting condition: the true boundary is where the
        active trade candidate (buy below the region, sell above it)
        exactly equals the no-trade candidate, found by linear
        interpolation of the two using the actual (converged) L values
        at the two grid points straddling the boundary, rather than
        snapping to whichever grid point happens to be classified as
        no-trade.

        Falls back to the raw grid boundary (with a warning) if the
        no-transaction region touches the edge of the grid -- in that
        case the grid itself is too narrow and interpolation cannot
        recover the true boundary; widen M/dy instead.
        """
        import warnings

        y = self.y_grid
        dy = y[1] - y[0]
        k_lo, k_hi = int(self.idx_NT.min()), int(self.idx_NT.max())
        K = len(y)

        if k_lo == 0 or k_hi == K - 1:
            warnings.warn(
                "No-transaction region touches the grid edge: cannot "
                "interpolate the boundary reliably. Widen the grid "
                "(increase M) before trusting this result.",
                stacklevel=2,
            )
            return y[k_lo], y[k_hi]

        # lower boundary: buy candidate at k references L_final[k+1]
        buy_cand_km1 = self.log_buy_cost + self.L_final[k_lo]      # candidate at k_lo-1 uses L(k_lo)
        buy_cand_k = (self.log_buy_cost + self.L_final[k_lo + 1]
                      if k_lo + 1 < K else np.inf)                 # candidate at k_lo uses L(k_lo+1)
        f_km1 = self.L_NT[k_lo - 1] - buy_cand_km1   # > 0: trade preferred at k_lo-1
        f_k = self.L_NT[k_lo] - buy_cand_k           # <= 0: NT preferred at k_lo
        denom = f_km1 - f_k
        frac = 1.0 if denom == 0 else np.clip(f_km1 / denom, 0.0, 1.0)
        y_lower = y[k_lo - 1] + dy * frac

        # upper boundary: sell candidate at k references L_final[k-1]
        sell_cand_k = (self.log_sell_cost + self.L_final[k_hi - 1]
                       if k_hi - 1 >= 0 else np.inf)               # candidate at k_hi uses L(k_hi-1)
        sell_cand_kp1 = self.log_sell_cost + self.L_final[k_hi]    # candidate at k_hi+1 uses L(k_hi)
        g_k = self.L_NT[k_hi] - sell_cand_k          # <= 0: NT preferred at k_hi
        g_kp1 = self.L_NT[k_hi + 1] - sell_cand_kp1  # > 0: trade preferred at k_hi+1
        denom2 = g_kp1 - g_k
        frac2 = 0.0 if denom2 == 0 else np.clip(-g_k / denom2, 0.0, 1.0)
        y_upper = y[k_hi] + dy * frac2

        return y_lower, y_upper

    def interpolated_width(self) -> float:
        y_lower, y_upper = self.interpolated_boundary()
        return y_upper - y_lower


def solve(lam, nu=None, mu=0.08, r=0.02, sigma=0.20, gamma=1.0, T=1.0,
          N=200, S0=1.0, M=150, dy=0.02, n_sweeps=4, phi=None) -> SolverResult:
    """
    Solve the finite-horizon CARA transaction-cost problem by backward
    induction on a CRR binomial tree, returning the no-transaction
    region at t=0 as a ``SolverResult``.

    The whole recursion is carried out in log-space (L = log Q); see bug
    3 in the module docstring for why. Only the fields needed to
    interpolate the sub-grid boundary at t=0 are retained afterwards
    (this function does not keep the L array at every time step, since
    that would be O(N*K) memory for no benefit once t=0 is reached).
    """
    if nu is None:
        nu = lam
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(mu * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError(f"p={p:.6f} outside (0,1): refine dt (increase N).")
    log_p, log_1mp = np.log(p), np.log(1 - p)

    y_grid = np.arange(-M, M + 1) * dy  # shape (2M+1,)
    K = len(y_grid)

    # terminal layer, n = N -- computed directly in log-space, no exp() needed
    S_N = S0 * u ** np.arange(0, N + 1) * d ** (N - np.arange(0, N + 1))
    payoff = phi(S_N) if phi is not None else np.zeros_like(S_N)
    ell_mat = (y_grid[None, :] * S_N[:, None]
               - S_N[:, None] * (nu * np.maximum(y_grid[None, :], 0)
                                  + lam * np.maximum(-y_grid[None, :], 0)))
    L = -gamma * (ell_mat - payoff[:, None])  # log Q at n = N, shape (N+1, K)

    L_NT_0 = None
    L_final_0 = None
    log_buy_cost_0 = None
    log_sell_cost_0 = None

    for n in range(N - 1, -1, -1):
        disc = np.exp(r * (T - n * dt))
        Sj = S0 * u ** np.arange(0, n + 1) * d ** (n - np.arange(0, n + 1))
        L_next = L  # shape (n+2, K)

        a = log_p + L_next[1:n + 2, :]
        b = log_1mp + L_next[0:n + 1, :]
        m = np.maximum(a, b)
        L_NT = m + np.log(np.exp(a - m) + np.exp(b - m))

        Lcur = L_NT.copy()
        log_buy_cost = gamma * (1 + lam) * Sj * dy * disc
        log_sell_cost = -gamma * (1 - nu) * Sj * dy * disc

        for _ in range(n_sweeps):
            for k in range(1, K):
                cand = log_sell_cost + Lcur[:, k - 1]
                Lcur[:, k] = np.minimum(Lcur[:, k], cand)
            for k in range(K - 2, -1, -1):
                cand = log_buy_cost + Lcur[:, k + 1]
                Lcur[:, k] = np.minimum(Lcur[:, k], cand)

        L = Lcur

        if n == 0:
            L_NT_0 = L_NT[0, :].copy()
            L_final_0 = Lcur[0, :].copy()
            log_buy_cost_0 = float(log_buy_cost[0])
            log_sell_cost_0 = float(log_sell_cost[0])

    is_NT = (L_final_0 == L_NT_0)
    idx_NT = np.where(is_NT)[0]
    if len(idx_NT) == 0:
        raise RuntimeError(
            "No no-transaction region found at t=0: grid likely too "
            "coarse or too narrow, or lambda too large for this grid."
        )

    return SolverResult(
        y_grid=y_grid, L_NT=L_NT_0, L_final=L_final_0, idx_NT=idx_NT,
        log_buy_cost=log_buy_cost_0, log_sell_cost=log_sell_cost_0,
    )


if __name__ == "__main__":
    res = solve(lam=0.01, N=150, M=120, dy=0.02)
    print("grid-snapped width      :", res.grid_width())
    print("interpolated width      :", res.interpolated_width())
    print("interpolated boundary   :", res.interpolated_boundary())
