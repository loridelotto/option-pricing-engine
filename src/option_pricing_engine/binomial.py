"""Cox-Ross-Rubinstein binomial tree for european and american options."""

import numpy as np


def _rollback(S, K, T, r, sigma, q, N, kind, exercise):
    """Backward induction. Returns the last three levels of the tree."""
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', received {kind!r}")
    if exercise not in ("european", "american"):
        raise ValueError(f"exercise must be 'european' or 'american', received {exercise!r}")

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)

    def payoff(prices):
        return np.maximum(prices - K, 0.0) if kind == "call" else np.maximum(K - prices, 0.0)

    j = np.arange(N + 1)                       # number of up moves
    V = payoff(S * u**j * d ** (N - j))        # value at expiry

    kept = {}
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        if exercise == "american":
            jj = np.arange(i + 1)
            V = np.maximum(V, payoff(S * u**jj * d ** (i - jj)))
        if i <= 2:
            kept[i] = V.copy()
    return kept, u, d, dt


def binomial_price(S, K, T, r, sigma, q=0.0, N=1000, kind="call", exercise="european"):
    """Option price as a float."""
    kept, _, _, _ = _rollback(S, K, T, r, sigma, q, N, kind, exercise)
    return float(kept[0][0])


def binomial_greeks(S, K, T, r, sigma, q=0.0, N=1000, kind="call", exercise="european"):
    """Price, delta, gamma and theta read from the first levels of the tree."""
    if N < 3:
        raise ValueError("greeks require N >= 3 (levels 0, 1 and 2 are needed)")
    kept, u, d, dt = _rollback(S, K, T, r, sigma, q, N, kind, exercise)

    V0, V1, V2 = kept[0], kept[1], kept[2]
    S1 = S * np.array([d, u])
    S2 = S * np.array([d * d, 1.0, u * u])      # u*d = 1: the middle node is back at S

    delta = (V1[1] - V1[0]) / (S1[1] - S1[0])
    slope_up = (V2[2] - V2[1]) / (S2[2] - S2[1])
    slope_dn = (V2[1] - V2[0]) / (S2[1] - S2[0])
    gamma = (slope_up - slope_dn) / (0.5 * (S2[2] - S2[0]))
    theta = (V2[1] - V0[0]) / (2.0 * dt)        # same S, 2*dt later

    return {"price": float(V0[0]), "delta": float(delta),
            "gamma": float(gamma), "theta": float(theta)}
