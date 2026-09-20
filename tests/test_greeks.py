import numpy as np
import pytest

from option_pricing_engine.black_scholes import (
    price, delta, gamma, vega, theta, rho, rho_q,
)

CASES = [
    (100, 100, 1.0, 0.04, 0.20, 0.02),
    (100,  90, 0.5, 0.03, 0.35, 0.00),
    ( 80, 120, 2.0, 0.05, 0.15, 0.04),
]


def _fd(f, x, h):
    """numeric derivative: (f(x+h) - f(x-h)) / 2h."""
    return (f(x + h) - f(x - h)) / (2 * h)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S,K,T,r,s,q", CASES)
def test_delta_matches_finite_difference(S, K, T, r, s, q, kind):
    num = _fd(lambda x: price(x, K, T, r, s, q, kind), S, 1e-4)
    assert delta(S, K, T, r, s, q, kind) == pytest.approx(num, rel=1e-6)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S,K,T,r,s,q", CASES)
def test_gamma_matches_finite_difference(S, K, T, r, s, q, kind):
    num = _fd(lambda x: delta(x, K, T, r, s, q, kind), S, 1e-4)
    assert gamma(S, K, T, r, s, q) == pytest.approx(num, rel=1e-6)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S,K,T,r,s,q", CASES)
def test_vega_matches_finite_difference(S, K, T, r, s, q, kind):
    num = _fd(lambda x: price(S, K, T, r, x, q, kind), s, 1e-6)
    assert vega(S, K, T, r, s, q) == pytest.approx(num, rel=1e-6)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S,K,T,r,s,q", CASES)
def test_theta_matches_finite_difference(S, K, T, r, s, q, kind):
    # theta e' dV/dt = -dV/dT: il tempo che passa e' la scadenza che si avvicina
    num = -_fd(lambda x: price(S, K, x, r, s, q, kind), T, 1e-6)
    assert theta(S, K, T, r, s, q, kind) == pytest.approx(num, rel=1e-6)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S,K,T,r,s,q", CASES)
def test_rho_matches_finite_difference(S, K, T, r, s, q, kind):
    num = _fd(lambda x: price(S, K, T, x, s, q, kind), r, 1e-7)
    assert rho(S, K, T, r, s, q, kind) == pytest.approx(num, rel=1e-6)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S,K,T,r,s,q", CASES)
def test_rho_q_matches_finite_difference(S, K, T, r, s, q, kind):
    num = _fd(lambda x: price(S, K, T, r, s, x, kind), q, 1e-7)
    assert rho_q(S, K, T, r, s, q, kind) == pytest.approx(num, rel=1e-6)


@pytest.mark.parametrize("S,K,T,r,s,q", CASES)
def test_gamma_and_vega_are_equal_for_call_and_put(S, K, T, r, s, q):
    """Conseguenza della put-call parity: la differenza C-P e' lineare in S
    e indipendente da sigma, quindi gamma e vega coincidono."""
    assert gamma(S, K, T, r, s, q) == gamma(S, K, T, r, s, q)
    assert vega(S, K, T, r, s, q) == vega(S, K, T, r, s, q)


@pytest.mark.parametrize("S,K,T,r,s,q", CASES)
def test_delta_parity(S, K, T, r, s, q):
    """delta_call - delta_put = exp(-qT)"""
    dc = delta(S, K, T, r, s, q, "call")
    dp = delta(S, K, T, r, s, q, "put")
    assert dc - dp == pytest.approx(np.exp(-q * T), abs=1e-12)