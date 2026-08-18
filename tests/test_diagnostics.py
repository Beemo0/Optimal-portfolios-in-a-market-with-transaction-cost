"""Smoke tests for txcost.diagnostics: fast, small-scale versions of the
checks used to produce Section 5.3's numbers, just to catch import/wiring
regressions -- not a substitute for the full multi-horizon scan, which is
run separately (see notebooks/) because it takes several minutes."""
import numpy as np
import pytest

from txcost.diagnostics import fit_width_scaling


def test_fit_width_scaling_gives_plausible_exponent():
    # Small N/M for test speed; a real run for the paper uses N=150*T,
    # M=int(4.5/dy) -- see diagnostics.check_T_robustness.
    slope, widths = fit_width_scaling(
        T_=1.0, dy=0.02, N=60, M=100, n_sweeps=2,
        lams=(0.002, 0.004, 0.008, 0.016),
    )
    assert np.all(np.diff(widths) > 0)          # width increases with lambda
    assert 0.2 < slope < 0.6                     # in the right ballpark of 1/3
