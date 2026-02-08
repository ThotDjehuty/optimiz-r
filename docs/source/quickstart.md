# Quick Start Guide

## Your First Optimization

Let's optimize the classic **Rosenbrock function** using Differential Evolution:

```python
import numpy as np
from optimizr import DifferentialEvolution

# Define the Rosenbrock function
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# Set up optimizer
de = DifferentialEvolution(
    bounds=[(-5, 5)] * 10,  # 10-dimensional problem
    strategy="best/1/bin",
    population_size=50,
    F=0.8,
    CR=0.9
)

# Run optimization
result = de.optimize(rosenbrock, max_iterations=200)

# Print results
print(f"✓ Best fitness: {result.best_fitness:.6f}")
print(f"✓ Best solution: {result.best_solution}")
print(f"✓ Converged in {result.iterations} iterations")
```

**Expected output:**
```
✓ Best fitness: 0.000002
✓ Best solution: [1.0, 1.0, 1.0, ..., 1.0]
✓ Converged in 174 iterations
```

## Mean Field Games Example

Solve a **1D Mean Field Game** (agent population dynamics):

```python
from optimizr import MFGSolver

# Define parameters
solver = MFGSolver(
    nx=100,           # Spatial grid points
    nt=50,            # Time steps
    x_min=-5.0,       
    x_max=5.0,
    T=1.0,            # Terminal time
    epsilon=0.1,      # Noise intensity
    kappa=1.0         # Congestion cost
)

# Solve coupled HJB-Fokker-Planck system
result = solver.solve()

# Access solution
print(f"Value function shape: {result.value_function.shape}")  # (50, 100)
print(f"Density shape: {result.density.shape}")               # (50, 100)
print(f"Converged: {result.converged}")
```

## Hidden Markov Model Example

Train an **HMM** on observed data:

```python
import numpy as np
from optimizr import HMMGaussian

# Generate synthetic data (2 hidden states, 1D observations)
np.random.seed(42)
observations = np.random.randn(1000, 1)

# Initialize HMM
hmm = HMMGaussian(n_states=2, n_features=1)

# Train model
hmm.fit(observations, max_iterations=100, tol=1e-6)

# Decode hidden state sequence
states = hmm.decode(observations)
print(f"Predicted states: {states[:20]}") # First 20 states
```

## MCMC Sampling Example

Sample from a **posterior distribution**:

```python
import numpy as np
from optimizr import MetropolisHastings

# Define log-posterior (unnormalized)
def log_posterior(x):
    # Gaussian prior: N(0, 1)
    prior = -0.5 * np.sum(x**2)
    # Likelihood: N(2, 0.5)
    likelihood = -0.5 * np.sum((x - 2)**2) / 0.25
    return prior + likelihood

# Initialize sampler
sampler = MetropolisHastings(
    log_prob_fn=log_posterior,
    initial_state=np.zeros(5),
    proposal_scale=0.5
)

# Generate samples
samples = sampler.sample(n_samples=10000, burn_in=1000)

print(f"Posterior mean: {samples.mean(axis=0)}")  # ~[1.6, 1.6, ...]
print(f"Acceptance rate: {sampler.acceptance_rate:.2%}")
```

## Next Steps

- **Explore algorithms**: See [Algorithms](algorithms/differential_evolution.md) for detailed guides
- **API reference**: Check [API Reference](api/differential_evolution.md) for all parameters
- **Examples**: Browse [examples/](https://github.com/ThotDjehuty/optimiz-r/tree/main/examples) for Jupyter notebooks
- **Benchmarks**: See [Benchmarks](benchmarks.md) for performance comparisons
