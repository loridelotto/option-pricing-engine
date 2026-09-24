# Option Pricing Engine

A small Python library for pricing European and American options with three different methods:

| Method | Module | Exercise | Output |
|---|---|---|---|
| Black-Scholes (closed form) | `black_scholes.py` | European | price and all the main greeks |
| Cox-Ross-Rubinstein binomial tree | `binomial.py` | European and American | price, delta, gamma, theta |
| Monte Carlo under GBM | `monte_carlo.py` | European | price with standard error and confidence interval, pathwise greeks |

All methods support a continuous dividend yield `q` (default `0.0`) and both calls and puts (`kind="call"` / `kind="put"`).

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and requires Python >= 3.12.

```bash
git clone <repo-url>
cd option-pricing-engine
uv sync
```

## Usage

All functions share the same parameter convention:

- `S`: spot price
- `K`: strike
- `T`: time to maturity in years
- `r`: continuously compounded risk-free rate
- `sigma`: volatility
- `q`: continuous dividend yield (or foreign rate)

### Black-Scholes

```python
from option_pricing_engine import black_scholes as bs

bs.price(S=100, K=100, T=1.0, r=0.05, sigma=0.2, kind="call")   # 10.4506
bs.greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2, kind="put")
# {'price': 5.5735, 'delta': -0.3632, 'gamma': 0.0188, 'vega': 37.524,
#  'theta': -1.6579, 'rho': -41.890, 'rho_q': 36.317}
```

Single greeks are also available as functions: `delta`, `gamma`, `vega`, `theta` (annualized), `rho`, `rho_q` (sensitivity to the dividend yield). The normal CDF uses `scipy.special.ndtr` for speed, so the functions also work on NumPy arrays.

### Binomial tree (CRR)

```python
from option_pricing_engine.binomial import binomial_price, binomial_greeks

binomial_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2, N=1000)            # 10.4486
binomial_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2,
               kind="put", exercise="american")                           # 6.0896
binomial_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2,
                kind="put", exercise="american")
# {'price': 6.0896, 'delta': -0.4111, 'gamma': 0.0230, 'theta': -2.2402}
```

The tree is rolled back in a vectorized way. For American options the early exercise value is checked at every node. Delta, gamma and theta are read directly from the first three levels of the tree (`N >= 3` required), so they come at no extra cost.

### Monte Carlo

```python
from option_pricing_engine.monte_carlo import monte_carlo_price, pathwise_greeks

monte_carlo_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2,
                  antithetic=True, control=True, seed=42)
# {'price': 10.4388, 'stderr': 0.0062, 'ci_low': 10.4266, 'ci_high': 10.4510,
#  'n_paths': 200000}

pathwise_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2, seed=42)
# {'price': ..., 'price_se': ..., 'price_ci': (...),
#  'delta': ..., 'delta_se': ..., 'delta_ci': (...), 'vega': ..., 'rho': ...}
```

- Terminal prices are sampled exactly from geometric Brownian motion (a single step, since the payoff is European).
- Variance reduction:
  - `antithetic=True`: antithetic variates (`z` and `-z`).
  - `control=True`: control variate on the discounted terminal price, whose expectation `S * exp(-q T)` is known, with the optimal coefficient estimated from the sample.
- Every estimate comes with its standard error and a confidence interval (default 95%, set with `confidence`).
- `pathwise_greeks` computes delta, vega and rho with the pathwise derivative method, each with its own standard error and confidence interval.
- `seed` makes the results reproducible.

## Project structure

```
option-pricing-engine/
├── pyproject.toml
├── uv.lock
└── src/option_pricing_engine/
    ├── __init__.py
    ├── black_scholes.py   # closed-form prices and greeks
    ├── binomial.py        # CRR tree, European and American
    └── monte_carlo.py     # MC pricing, variance reduction, pathwise greeks
```

## Future developments

- **Convergence tests of the three methods.** Use Black-Scholes as the exact benchmark for European options and check that the numerical methods converge to it
