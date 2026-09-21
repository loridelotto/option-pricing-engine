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

    # induzione all'indietro: V ha i+1 elementi al passo i
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        if exercise == "american":
            jj = np.arange(i + 1)
            V = np.maximum(V, payoff(S * u**jj * d ** (i - jj)))

    return V[0]