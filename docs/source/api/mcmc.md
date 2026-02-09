# API: mcmc_sample

```python
from optimizr import mcmc_sample

samples = mcmc_sample(
    log_likelihood_fn,
    data,
    initial_params,
    param_bounds,
    n_samples=10000,
    burn_in=1000,
    proposal_std=0.1,
)
```

- `log_likelihood_fn(params, data) -> float`
- `data`: np.ndarray passed through to the likelihood
- `initial_params`: np.ndarray starting point
- `param_bounds`: list of `(min, max)` tuples
- Returns `samples: np.ndarray` of shape `(n_samples, n_params)`
