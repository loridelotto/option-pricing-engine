import numpy as np
import pytest

from option_pricing_engine.black_scholes import call_price, put_price


def test_call_known_value():
    """S=K=100, T=1 anno, r=4%, sigma=20% -> 9.9251"""
    assert call_price(100, 100, 1.0, 0.04, 0.20) == pytest.approx(9.9251, abs=1e-4)


@pytest.mark.parametrize("T", [0.25, 0.5, 1.0, 2.0])
def test_put_call_parity(T):
    """C - P = S - K*exp(-rT), for any expiration time"""
    S, K, r, sigma = 100, 90, 0.04, 0.20
    c = call_price(S, K, T, r, sigma)
    p = put_price(S, K, T, r, sigma)
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-10)


def test_call_increases_with_volatility():
    """ as volatility increases, option prices also increase"""
    prices = [call_price(100, 100, 1.0, 0.04, s) for s in (0.10, 0.20, 0.40)]
    assert prices[0] < prices[1] < prices[2]