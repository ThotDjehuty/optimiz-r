# MCMC Sampling

Metropolis-Hastings sampler for Bayesian inference with Rust acceleration.

## Usage

```python
import numpy as np
from optimizr import mcmc_sample

# Log-likelihood of a Gaussian model

def log_likelihood(params, data):
    mu, sigma = params
    residuals = (data - mu) / sigma
    return -0.5 * np.sum(residuals**2) - len(data) * np.log(sigma)

observations = np.random.randn(1000) + 1.2
samples = mcmc_sample(
    log_likelihood_fn=log_likelihood,
    data=observations,
    initial_params=np.array([0.0, 1.0]),
    param_bounds=[(-5, 5), (0.1, 5.0)],
    n_samples=8000,
    burn_in=500,
    proposal_std=0.2,
)

print("Posterior mean:", samples.mean(axis=0))
```

## Tips
- Keep `proposal_std` modest to maintain acceptance rate (20–40%).
- `burn_in` should be at least 5–10% of total samples for stable chains.
- Provide tight `param_bounds` to avoid exploring invalid regions.
