"""Tests for txcost.frictionless: Merton fraction against the closed form
of Section 3.2, and the reference-parameter case (y*=1.5) used throughout
Section 5."""
import pytest

from txcost.frictionless import Market, merton_fraction, merton_shares


def test_reference_parameters_give_y_star_1_5():
    market = Market(mu=0.08, r=0.02, sigma=0.20)
    assert merton_fraction(market, gamma=1.0) == pytest.approx(1.5)
    assert merton_shares(market, gamma=1.0, wealth=1.0, S0=1.0) == pytest.approx(1.5)


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0, 5.0])
def test_fraction_scales_as_inverse_gamma(gamma):
    market = Market(mu=0.08, r=0.02, sigma=0.20)
    assert merton_fraction(market, gamma) == pytest.approx(
        merton_fraction(market, 1.0) / gamma
    )


def test_negative_sigma_rejected():
    with pytest.raises(ValueError):
        Market(mu=0.08, r=0.02, sigma=-0.1)
