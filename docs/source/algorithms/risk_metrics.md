# Risk Metrics

Time-series utilities for risk analysis and mean-reversion signals.

## Quick Start

```python
import numpy as np
from optimizr import (
    hurst_exponent_py,
    estimate_half_life_py,
    bootstrap_returns_py,
)

returns = np.random.randn(2000) * 0.01
print("Hurst:", hurst_exponent_py(returns))
print("Half-life:", estimate_half_life_py(returns))

bootstrapped = bootstrap_returns_py(returns, n_samples=1000)
print("Bootstrap sample shape:", len(bootstrapped))
```

## Notes
- Input arrays should be 1D NumPy arrays of returns.
- Half-life is useful for calibrating mean-reversion strategies.
- Bootstrap utilities help estimate drawdown and VaR distributions.
