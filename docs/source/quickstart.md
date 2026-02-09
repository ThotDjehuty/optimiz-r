# Quick Start Guide

## 1. Verify Installation

```bash
python -c "import optimizr; print(optimizr.__version__)"
```

You should see `0.3.0` (or newer). If the Rust backend is missing, reinstall with `pip install .` from the project root to build the extension module.

## 2. First Optimization (Differential Evolution)

```python
import numpy as np
from optimizr import differential_evolution

def rosenbrock(x: np.ndarray) -> float:
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

best_x, best_fx = differential_evolution(
    objective_fn=rosenbrock,
    bounds=[(-5, 5)] * 5,
    strategy="best1",
    adaptive=True,
    maxiter=500,
)

print(f"Best value: {best_fx:.6f}")
print(f"Best point: {best_x}")
```

## 3. Hidden Markov Model (Regime Detection)

```python
import numpy as np
from optimizr import HMM

returns = np.concatenate([
    np.random.normal(0.01, 0.02, 400),
    np.random.normal(-0.01, 0.03, 400),
])

model = HMM(n_states=2).fit(returns, n_iterations=80)
states = model.predict(returns)
print(np.bincount(states))
```

## 4. MCMC Sampling (Bayesian Inference)

```python
import numpy as np
from optimizr import mcmc_sample

def log_likelihood(params, data):
    mu, sigma = params
    residuals = (data - mu) / sigma
    return -0.5 * np.sum(residuals**2) - len(data) * np.log(sigma)

data = np.random.randn(500) + 1.5
samples = mcmc_sample(
    log_likelihood_fn=log_likelihood,
    data=data,
    initial_params=np.array([0.0, 1.0]),
    param_bounds=[(-5, 5), (0.1, 5.0)],
    n_samples=5000,
    burn_in=500,
)

print(samples.mean(axis=0))
```

## 5. Mean Field Games (1D)

```python
from optimizr import MFGConfig, solve_mfg_1d_rust

config = MFGConfig(
    nx=64,
    nt=40,
    x_min=-3.0,
    x_max=3.0,
    T=1.0,
    epsilon=0.1,
    kappa=1.0,
)

solution = solve_mfg_1d_rust(config)
print(f"Converged: {solution.converged}")
```

## Next Steps

- See [Getting Started](getting-started.md) for environment setup and verification.
- Browse [Examples](examples.md) for code snippets per optimizer.
- Deep dive into algorithms in [Algorithms](algorithms/differential_evolution.md).
