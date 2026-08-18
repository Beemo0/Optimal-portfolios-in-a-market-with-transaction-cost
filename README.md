# txcost

Companion code for the numerical chapter (Section 5) of the paper on
portfolio construction under proportional transaction costs. Implements
the finite-horizon CARA solver of Davis, Panas & Zariphopoulou (1993) on
a binomial tree, via Gauss-Seidel policy iteration, and the diagnostics
used to validate it against the small-cost asymptotics of Section 4.

## Install

```
pip install -e ".[dev]"
```

## Structure

```
src/txcost/
    tree.py          binomial tree calibration (physical measure), moment checks
    frictionless.py  Merton fraction / reference frictionless quantities (Section 3.2)
    cara_solver.py   the solver itself (log-space, interpolated free boundary)
    asymptotics.py   small-cost benchmarks from Section 4 (Rogers 1/3, GMKS 2013),
                     for COMPARISON ONLY -- not the problem this solver solves
    diagnostics.py   validation routines behind every number in Section 5.3
tests/               pytest suite: one file per module above
notebooks/           the executed notebook producing the report's figures
figures/             standalone PNGs for direct inclusion in the report
```

Run the fast checks:
```
python -m txcost.diagnostics
```
Add `--full` for the multi-horizon scan (several minutes):
```
python -m txcost.diagnostics --full
```
Run the tests:
```
pytest
```

## Bugs found and fixed during development

Full details, and their exact regression tests, are in the module
docstrings of `cara_solver.py` and `tests/test_cara_solver.py`. Summary:

1. **Sign errors in the buy/sell cost formulas** (two separate rounds):
   the direction of the Bellman recursion, then the sign of the cost
   exponent, were initially backwards. Caught by the invariant that a
   round trip can never be profitable with positive costs
   (`test_no_profitable_round_trip`).

2. **A misleading diagnostic, not a code bug**: `Q(t,y,S)` must be
   compared to `Q_NT(t,y,S)` at the *same* `y`, not across different
   values of `y` -- `Q` decreasing in `y` at fixed cash is expected
   behavior, not a flaw.

3. **Float64 overflow at long horizons** (`T` beyond ~3-6, depending on
   the grid): the original exp-space recursion produced `inf` at
   extreme, low-probability tree nodes, which sometimes self-healed via
   `np.minimum` (by luck, e.g. at `T=3`) and sometimes contaminated the
   whole tree via `0*inf=nan` (`T=6`). Fixed by carrying the entire
   recursion in log-space, which cannot overflow by construction.
   Cross-checked bit-exact against an independent, deliberately naive
   exp-space reference implementation written from scratch inside the
   test suite (`test_matches_independent_expspace_reference`).

4. **The no-transaction-region width, if read off by snapping to the
   nearest grid point, is sensitive to which side of a grid cell the
   true boundary happens to fall on.** This was invisible until the
   width-vs-lambda scaling exponent was refit at a wider grid
   (`M=300 -> M=450` at fixed `dy`): a single grid-step change in the
   width at the smallest lambda -- the point with the most leverage on
   a log-log fit -- moved the fitted exponent from 0.364 to 0.349, a 4%
   change that should not exist for a converged discretization. Fixed
   by interpolating the true sub-grid boundary from the smooth-pasting
   condition (`SolverResult.interpolated_boundary`), rather than
   snapping to the grid. After the fix, `M=300` and `M=450` agree to
   0.0012 in the fitted exponent (previously 0.0148) --
   `test_interpolated_width_independent_of_grid_halfwidth`.

## Current results (Section 5.3): exponent vs. horizon

Using the interpolated boundary and a grid wide enough (`M=int(4.5/dy)`)
that the result no longer depends on `M`:

| T  | N    | exponent |
|----|------|----------|
| 1  | 150  | 0.3796   |
| 3  | 450  | 0.3483   |
| 6  | 900  | 0.3499   |
| 10 | 1500 | 0.3483   |

This **supersedes** the earlier, grid-snapped-width table, which showed a
spurious rise to 0.375 at `T=10` -- that rise was itself an artifact of
bug 4, not a real feature of the CARA finite-horizon problem.

**Reading**: the exponent drops from ~0.38 at `T=1` and plateaus at
~0.349 (spread of 0.0016) from `T=3` through `T=10`. This plateau sits
about 5% above the theoretical `1/3` (Rogers 2004 / Gerhold-Muhle-Karbe-
Schachermayer 2013 -- derived for **log-utility, infinite horizon**, a
related but different problem from the **CARA, finite-horizon** problem
solved here; see `asymptotics.py`). It does **not** continue decreasing
toward `1/3` over the horizons checked. We do not currently know whether
this reflects (a) a genuine difference in the leading constant between
the CARA finite-horizon and log-utility infinite-horizon problems, or
(b) a convergence to `1/3` that only sets in at horizons well beyond
what is computationally practical here (see "Open questions").
Convergence in `dy`, `N`, `n_sweeps`, and `M` has been checked at this
plateau (`diagnostics.check_convergence`); it is not a discretization
artifact.

## Figures

`fig4_exponent_vs_horizon.png` has been regenerated with the corrected,
M-converged, interpolated-boundary numbers above (it previously showed
the spurious rise at `T=10`). `fig1_nt_region_vs_lambda.png`,
`fig2_nt_region_vs_time.png`, and `fig3_width_scaling_T1.png` were
checked against the fix and found not to need regeneration: they either
plot raw widths (not a log-log-fitted exponent, so not subject to the
extreme-lambda leverage effect of bug 4) or, for `fig3` (the `T=1`
scaling fit), differ only in the third decimal (`0.377` grid-snapped vs.
`0.3796` interpolated). Re-run `notebooks/01_solver_validation_and_figures.ipynb`
end-to-end if you want every figure regenerated from the current code
rather than relying on this spot-check.

## Open questions

- Confirming or refuting whether the plateau near 0.349 eventually
  resumes its decrease toward `1/3` at horizons beyond `T=10` would
  require either a much longer horizon (computationally expensive: cost
  scales like `N^2`, i.e. like `T^2` at fixed `dt`) or a smarter solver
  (e.g. an infinite-horizon formulation directly, avoiding the need to
  take `T` large as a proxy).
- A fully apples-to-apples comparison would replace the log-utility,
  infinite-horizon asymptotics of Section 4.3 with the finite-horizon
  CARA-specific small-cost expansion -- not derived here, and not found
  in the literature reviewed for Section 2.
