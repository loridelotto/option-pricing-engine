# pricing non-dividend paying european options

import numpy as np
from scipy.stats import norm


def _d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, sigma):
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def call_price(S, K, T, r, sigma):
    """
    S     : Underlying price today
    K     : strike price
    T     : time to maturity (in years)
    r     : annualized risk free rate
    sigma : annualized volatility
    """
    return S * norm.cdf(_d1(S, K, T, r, sigma)) - K * np.exp(-r * T) * norm.cdf(_d2(S, K, T, r, sigma)
    )

def put_price(S, K, T, r, sigma):
    """Stessi parametri di call_price."""
    return K * np.exp(-r * T) * norm.cdf(-_d2(S, K, T, r, sigma)) - S * norm.cdf(
        -_d1(S, K, T, r, sigma)
    )