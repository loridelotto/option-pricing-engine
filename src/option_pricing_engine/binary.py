"""Cox-Ross-Rubinstein binomial tree for european and american options"""

import numpy as np

def binomial_price(S, K, T, r, sigma, q=0.0, N=1000, kind="call", exercise="european"):
    """

    kind     : "call" o "put"
    exercise : "european" o "american"

    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', received {kind!r}")
    if exercise not in ("european", "american"):
        raise ValueError(
            f"exercise must be 'european' or 'american', {exercise!r}"
        )

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)

    def payoff(prices):
        return (
            np.maximum(prices - K, 0.0) if kind == "call" else np.maximum(K - prices, 0.0)
        )

    # Price at time T: S_j = S * u^j * d^(N-j), j = 0..N
    j = np.arange(N + 1)   # Number of up movements 
    V = payoff(S * u**j * d ** (N - j)) # Option price at time T

    # backwards induction and greeks calculation
    kept = {}
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        if exercise == "american":
            jj = np.arange(i + 1)
            V = np.maximum(V, payoff(S * u**jj * d ** (i - jj)))
        if i <= 2:
            kept[i] = V.copy()

    V0, V1, V2 = kept[0], kept[1], kept[2]
    S1 = S * np.array([d, u])
    S2 = S * np.array([d * d, 1.0, u * u])     

    delta = (V1[1] - V1[0]) / (S1[1] - S1[0])
    slope_up = (V2[2] - V2[1]) / (S2[2] - S2[1])
    slope_dn = (V2[1] - V2[0]) / (S2[1] - S2[0])
    gamma = (slope_up - slope_dn) / (0.5 * (S2[2] - S2[0]))
    theta = (V2[1] - V0[0]) / (2.0 * dt)         

    return {"price": V0[0], "delta": delta, "gamma": gamma, "theta": theta}