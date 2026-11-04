import numpy as np
import pytest
from scipy import integrate

from tailor_aoi import ServiceDist, optimized_static, simulate, solve, static_cost, update_or_wait, zero_wait
from tailor_aoi.baselines import StaticPolicy


@pytest.mark.parametrize("dist", [
    ServiceDist.lomax(2.5, 1.3),
    ServiceDist.lognormal(0.2, 0.9),
    ServiceDist.exponential(0.7),
])
@pytest.mark.parametrize("a", [0.3, 1.7, 6.0])
def test_partial_moments_match_quadrature(dist, a):
    for n in range(3):
        ref, _ = integrate.quad(lambda s: s**n * float(dist.pdf(np.array(s))), 0, a)
        assert dist.partial_moment(n, a) == pytest.approx(ref, rel=1e-7, abs=1e-10)


def test_pareto2_parameterisation():
    d = ServiceDist.pareto2(mean=2.0, variance=40.0)
    assert d.mean == pytest.approx(2.0)
    assert d.variance == pytest.approx(40.0)


def test_lognormal_from_mean():
    d = ServiceDist.lognormal_from_mean(2.0, 3.0)
    assert d.mean == pytest.approx(2.0)


@pytest.mark.parametrize("D,T", [(0.0, np.inf), (1.2, np.inf), (0.5, 1.5), (2.0, 0.8)])
def test_static_cost_matches_monte_carlo(D, T):
    """Exact renewal-reward cost vs. simulation in a light-tailed case."""
    d = ServiceDist.lognormal(0.0, 0.5)
    exact = static_cost(d, D, T, kappa_s=1.0, kappa_p=2.0)
    mc = simulate(StaticPolicy(D, T), d, 1.0, 2.0, n_attempts=400_000, rng=11).avg_cost
    assert mc == pytest.approx(exact, rel=0.01)


def test_zero_wait_closed_form():
    d = ServiceDist.lognormal_from_mean(2.0, 6.0)
    expected = (d.mean**2 + d.second_moment / 2 + 1.0) / d.mean
    assert zero_wait(d, 1.0, 1.0)[1] == pytest.approx(expected, rel=1e-10)


def test_exponential_theorem2():
    """Theorem 2: v(y) = y / lambda, constant preemption threshold, z* = max(D, rho - 1/lambda)."""
    lam = 1.0
    p = solve(ServiceDist.exponential(lam), 1.0, 1.0, dt=0.01)
    m = p.y <= 10
    assert np.max(np.abs(p.v[m] - p.y[m] / lam)) < 1e-3
    assert np.unique(np.round(p.theta_map[m], 6)).size == 1
    assert p.z_map[0] == pytest.approx(p.rho - 1 / lam, abs=0.011)
    # optimum for k_s = k_p = 1 equals the best static policy: 1 + sqrt(2)
    assert p.rho == pytest.approx(1 + np.sqrt(2), abs=1e-3)


def test_tailor_beats_baselines_and_matches_simulation():
    d = ServiceDist.lomax(2.1, 1.0)
    p = solve(d, 1.0, 5.0)
    assert p.converged
    _, c_static = optimized_static(d, 1.0, 5.0)
    assert p.rho <= c_static + 1e-6
    assert p.rho < update_or_wait(d, 1.0, 5.0)[1]
    mc = simulate(p, d, 1.0, 5.0, n_attempts=400_000, rng=5).avg_cost
    assert mc == pytest.approx(p.rho, rel=0.02)


def test_paper_lognormal_headline():
    """Fig. 2(b): log-normal, E[Y] = 2, sigma^2 = 6, k_s = k_p = 1 -> TAILOR ~ 1.76."""
    d = ServiceDist.lognormal_from_mean(2.0, 6.0)
    p = solve(d, 1.0, 1.0)
    assert p.rho == pytest.approx(1.76, abs=0.01)
