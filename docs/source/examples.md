# Examples

This page lists all available examples and tutorials for OptimizR.

## Jupyter Notebooks

All notebooks are located in the [examples/](https://github.com/ThotDjehuty/optimiz-r/tree/main/examples) directory.

### 1. Differential Evolution

**File**: `01_differential_evolution_tutorial.ipynb`

Learn how to use the Differential Evolution optimizer:
- Basic optimization problems (Rosenbrock, Rastrigin, Ackley)
- Strategy comparison (rand/1, best/1, current-to-best/1)
- Parameter tuning (F, CR, population size)
- Convergence analysis and visualization

### 2. Mean Field Games

**File**: `02_mean_field_games_tutorial.ipynb`

Solve 1D Mean Field Games:
- HJB-Fokker-Planck coupling
- Agent population dynamics
- Nash equilibrium computation
- 3D visualization (time × space × density)

### 3. Hidden Markov Models

**File**: `03_hmm_tutorial.ipynb`

Train and apply HMMs:
- Baum-Welch training algorithm
- Viterbi decoding
- Gaussian emission models
- Real-world applications (regime detection, speech recognition)

### 4. MCMC Sampling

**File**: `04_mcmc_tutorial.ipynb`

Bayesian inference with Metropolis-Hastings:
- Sampling from complex distributions
- Adaptive proposal tuning
- Convergence diagnostics
- Posterior analysis

### 5. Sparse Optimization

**File**: `05_sparse_optimization_tutorial.ipynb`

Sparse methods for high-dimensional data:
- Sparse PCA
- Elastic Net
- ADMM solver
- Feature selection

### 6. Optimal Control

**File**: `06_optimal_control_tutorial.ipynb`

Solve HJB equations:
- Regime-switching jump diffusions (MRSJD)
- Optimal stopping problems
- Dynamic programming
- Financial applications

### 7. Risk Metrics

**File**: `07_risk_metrics_tutorial.ipynb`

Time series analysis:
- Hurst exponent estimation
- Half-life calculation
- Mean reversion testing
- Trading signal generation

## Python Scripts

Quick examples for copy-paste usage:

### Optimize Rosenbrock Function

```python
import numpy as np
from optimizr import DifferentialEvolution

def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

de = DifferentialEvolution(bounds=[(-5, 5)] * 10)
result = de.optimize(rosenbrock, max_iterations=200)

print(f"Minimum: {result.best_fitness}")
```

### Train HMM on Data

```python
import numpy as np
from optimizr import HMMGaussian

# Load your data
observations = np.load("data.npy")

# Train model
hmm = HMMGaussian(n_states=3)
hmm.fit(observations)

# Predict states
states = hmm.decode(observations)
```

### MCMC Sampling

```python
import numpy as np
from optimizr import MetropolisHastings

def log_posterior(x):
    return -0.5 * np.sum((x - 2)**2)

sampler = MetropolisHastings(log_posterior, initial_state=np.zeros(5))
samples = sampler.sample(n_samples=10000)
```

## Running Examples

To run notebooks:

```bash
cd examples/
jupyter notebook
```

To run Python scripts:

```bash
python examples/basic_optimization.py
```

## Contribute Examples

Have a cool use case? Contribute your example:

1. Fork the [repository](https://github.com/ThotDjehuty/optimiz-r)
2. Add your notebook to `examples/`
3. Ensure it runs without errors
4. Submit a pull request

See [Contributing Guide](contributing.md) for details.
