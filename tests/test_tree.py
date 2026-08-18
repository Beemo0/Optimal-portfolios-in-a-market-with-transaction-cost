"""Tests for txcost.tree: the CRR tree must reproduce the exact GBM
moments under the physical measure -- this is the calibration check
underlying every downstream result in this package."""
import pytest

from txcost.tree import BinomialTree


@pytest.mark.parametrize("mu,sigma,T,N", [
    (0.08, 0.20, 1.0, 150),
    (0.08, 0.20, 3.0, 450),
    (0.05, 0.35, 0.5, 100),
])
def test_tree_moments_match_gbm(mu, sigma, T, N):
    tree = BinomialTree(mu=mu, sigma=sigma, T=T, N=N)
    E, Var = tree.terminal_moments()
    E_th, Var_th = BinomialTree.theoretical_moments(mu, sigma, T)
    assert E == pytest.approx(E_th, rel=1e-9)
    # Var converges more slowly (it is a second moment of a discretized
    # process); loose but meaningful tolerance for the N used here.
    assert Var == pytest.approx(Var_th, rel=2e-2)


def test_p_out_of_range_raises():
    # dt far too coarse for this (mu, sigma): exp(mu*dt) overtakes u.
    tree = BinomialTree(mu=5.0, sigma=0.01, T=10.0, N=2)
    with pytest.raises(ValueError):
        _ = tree.p


def test_nodes_at_are_ordered_low_to_high():
    tree = BinomialTree(mu=0.08, sigma=0.20, T=1.0, N=10)
    nodes = tree.nodes_at(10)
    assert len(nodes) == 11
    assert (nodes[1:] > nodes[:-1]).all()
