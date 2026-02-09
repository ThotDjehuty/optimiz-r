# Hidden Markov Models

Gaussian HMM for regime detection and sequence modelling.

## Usage

```python
import numpy as np
from optimizr import HMM

returns = np.concatenate([
    np.random.normal(0.01, 0.02, 500),
    np.random.normal(-0.015, 0.03, 500),
])

model = HMM(n_states=2)
model.fit(returns, n_iterations=100)

states = model.predict(returns)
print(np.unique(states, return_counts=True))
```

### With time-series helpers

```python
from optimizr import prepare_for_hmm_py

features = prepare_for_hmm_py(prices, lag_periods=[1, 5, 20])
hmm = HMM(n_states=3).fit(features, n_iterations=120)
```

Use rolling Hurst/half-life from `timeseries_utils` as additional features for richer regime classification.

## Notes
- Uses Rust backend when available; falls back to Python.
- `fit` runs Baum-Welch; `predict` runs Viterbi.
- Call `score(X)` to compute log-likelihood.
