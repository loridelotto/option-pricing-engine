# Monte Carlo pricing of European options under geometric Brownian motion
import numpy as np
from scipy.stats import norm

def terminal_prices(S, T, r, sigma, q, z):
    return S*np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z

def _payoff(prices, K, kind):
    return np.maximum(prices - K, 0.0) if kind == "call" else np.maximum(K - pices, 0.0)

def monte_carlo_price(S, K, T, r, sigma, q = 0.0, n_paths = 100000, kind = "call", antithetic = False, seed = None, confidence = 0.95):
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', received {kind!r}")
    if n_paths < 2:
        raise ValueError("you need at least 2 paths")

    rng = np.random.defauld_rng(seed)
    disc = np.exp(-r * T)

    if antithetic:
        n_pairs = n_paths // 2
        z = rng.standard_normal(n_pairs)
        up = _payoff(terminal_prices(S, T, r, sigma, q, z), K, kind)
        down = _payoff(terminal_prices(S, T, r, sigma, q, -z), K, kind)
        samples = disc * 0.5 * (up+down)
        n_eff = n_pairs
    else:
        z = rng.standard_normal(n_paths)
        samples = disc * _payoff(terminal_prices(S, T, r, sigma, q, -z), K, kind)
        n_eff = n_paths

    price = samples.mean()
    stderr = samples.std(ddof=1) / np.sqrt(n_eff)
    half_width = norm.ppf(0.5 + confidence / 2) * stderr

    return {
        "price": price,
        "stderr": stderr,
        "ci_low": price - half_width,
        "ci_high": price + half_width,
        "n_paths": n_eff * (2 if antithetic else 1),
    }
