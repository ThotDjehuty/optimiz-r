# API: Risk Metrics

```python
from optimizr import (
    hurst_exponent_py,
    compute_risk_metrics_py,
    estimate_half_life_py,
    bootstrap_returns_py,
)

h = hurst_exponent_py(returns)
hl = estimate_half_life_py(returns)
metrics = compute_risk_metrics_py(returns.tolist())
boot = bootstrap_returns_py(returns, n_samples=1000)
```

- `returns`: 1D NumPy array of returns
- `compute_risk_metrics_py` returns a dict with volatility, Sharpe, and drawdown estimates
- `bootstrap_returns_py` resamples the series for uncertainty estimation
