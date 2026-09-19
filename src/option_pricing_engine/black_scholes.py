# pricing european options, by default q = 0 (non paying dividend)

import numpy as np
from scipy.stats import norm


def _d1(S, K, T, r, sigma, q=0.0):
    return (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, sigma, q=0.0):
    return _d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)

def _check_kind(kind):
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' o 'put', received {kind!r}")


def call_price(S, K, T, r, sigma, q=0.0):
    d1, d2 = _d1(S, K, T, r, sigma, q), _d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, T, r, sigma, q=0.0):
    d1, d2 = _d1(S, K, T, r, sigma, q), _d2(S, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def price(S, K, T, r, sigma, q=0.0, kind="call"):
    _check_kind(kind)
    f = call_price if kind == "call" else put_price
    return f(S, K, T, r, sigma, q)

def delta(S, K, T, r, sigma, q=0.0, kind="call"):
    """dV/dS"""
    _check_kind(kind)
    d1 = _d1(S, K, T, r, sigma, q)
    if kind == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    return -np.exp(-q * T) * norm.cdf(-d1)

def gamma(S, K, T, r, sigma, q=0.0):
    """d2V/dS2 same for call and put, no kind parameter."""
    d1 = _d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma, q=0.0):
    """dV/dsigma same for call and put"""
    d1 = _d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

def theta(S, K, T, r, sigma, q=0.0, kind="call"):
    """dV/dt, annualized"""
    _check_kind(kind)
    d1, d2 = _d1(S, K, T, r, sigma, q), _d2(S, K, T, r, sigma, q)
    if kind == "call":
        return (
            -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            + q * S * np.exp(-q * T) * norm.cdf(d1)
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )
    return (
        -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        - q * S * np.exp(-q * T) * norm.cdf(-d1)
        + r * K * np.exp(-r * T) * norm.cdf(-d2)
    )

def rho(S, K, T, r, sigma, q=0.0, kind="call"):
    """dV/dr"""
    _check_kind(kind)
    d2 = _d2(S, K, T, r, sigma, q)
    if kind == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)

def rho_q(S, K, T, r, sigma, q=0.0, kind="call"):
    """dV/dq sensitivity to the dividend yield (or foreign rate)"""
    _check_kind(kind)
    d1 = _d1(S, K, T, r, sigma, q)
    if kind == "call":
        return -T * S * np.exp(-q * T) * norm.cdf(d1)
    return T * S * np.exp(-q * T) * norm.cdf(-d1)

def greeks(S, K, T, r, sigma, q=0.0, kind="call"):
    """ all the greeks in a dictionary"""
    return {
        "price": price(S, K, T, r, sigma, q, kind),
        "delta": delta(S, K, T, r, sigma, q, kind),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),
        "theta": theta(S, K, T, r, sigma, q, kind),
        "rho": rho(S, K, T, r, sigma, q, kind),
        "rho_q": rho_q(S, K, T, r, sigma, q, kind),
    }