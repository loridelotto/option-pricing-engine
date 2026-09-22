# Monte Carlo pricing of European options under geometric Brownian motion
import numpy as np
from scipy.stats import norm

def terminal_prices(S, T, r, sigma, q, z):
    return S*np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)

def _payoff(prices, K, kind):
    return np.maximum(prices - K, 0.0) if kind == "call" else np.maximum(K - prices, 0.0)

def monte_carlo_price(S, K, T, r, sigma, q = 0.0, n_paths = 100000, kind = "call", antithetic = False, control=False, seed = None, confidence = 0.95):
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', received {kind!r}")
    if n_paths < 2:
        raise ValueError("you need at least 2 paths")

    rng = np.random.defauld_rng(seed)
    disc = np.exp(-r * T)

    if antithetic:
        n_eff = n_paths // 2
        z = rng.standard_normal(n_eff)
        s_up, s_down = (terminal_prices(S, T, r, sigma, q, w) for w in (z, -z))
        samples = disc * 0.5 * (_payoff(s_up, K, kind) +_payoff(s_down, K, kind))
        control_samples = disc * 0.5 * (s_up + s_down)
    else:
        n_eff = n_paths
        s_t = terminal_prices(S, T, r, sigma, q, rng.standard_normal(n_eff))
        samples = disc * _payoff(s_t, K, kind)
        control_samples = disc * s_t

    if control:
        expected = S * np.exp(-q * T)                   
        c = np.cov(samples, control_samples, ddof=1)[0, 1] / np.var(control_samples, ddof=1)
        samples = samples - c * (control_samples - expected)

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