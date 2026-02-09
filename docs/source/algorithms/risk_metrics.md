# Risk Metrics

Time-series utilities for risk analysis, mean-reversion detection, and bootstrapped P&L distributions.

## Quick start

```python
import numpy as np
from optimizr import (
    hurst_exponent_py,
    estimate_half_life_py,
    bootstrap_returns_py,
    compute_risk_metrics_py,
)

returns = np.random.randn(2000) * 0.01
print("Hurst:", hurst_exponent_py(returns))
print("Half-life:", estimate_half_life_py(returns))

metrics = compute_risk_metrics_py(returns)
print(metrics)  # mean, std, skew, kurtosis, sharpe

bootstrapped = bootstrap_returns_py(returns, n_samples=1000)
print("Bootstrap samples:", len(bootstrapped))
```

## Rolling and integration helpers

- Use `rolling_hurst_exponent_py` and `rolling_half_life_py` (from `timeseries_utils`) for sliding-window diagnostics on trading pairs.
- Combine with HMM: feed rolling statistics as features for regime detection.
- Pair with DE/Grid search: optimize strategy thresholds while computing half-life inside the objective.

## Practical guidance
- Input should be 1D NumPy arrays of returns; winsorize extreme tails before estimating Hurst/half-life for stability.
- Half-life helps size holding periods for mean-reversion trades; revisit whenever volatility regime changes.
- Bootstrap outputs can feed VaR/ES estimates; increase `n_samples` for tighter confidence bands.
