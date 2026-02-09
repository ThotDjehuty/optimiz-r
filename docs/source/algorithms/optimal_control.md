# Optimal Control

High-level bindings for HJB-style optimal control and Kalman filtering utilities.

## Kalman Filter (sensor fusion)

```python
import numpy as np
from optimizr import maths_toolkit

# maths_toolkit is provided by the Rust extension (_core)
F = np.eye(2)  # state transition
H = np.eye(2)  # observation
Q = 0.01 * np.eye(2)
R = 0.1 * np.eye(2)

if maths_toolkit is not None:
    kf = maths_toolkit.init_kalman_filter(F.tolist(), H.tolist(), Q.tolist(), R.tolist())
    state = maths_toolkit.kalman_predict(kf, [0.0, 0.0])
    print(state)
else:
    print("Rust backend not available; install with `pip install .`.")
```

## Notes
- Rust backend (`optimizr._core`) must be present for control utilities.
- The API is thin and intentionally low-level; matrices are passed as lists.
- For 1D Mean Field Games, use the dedicated guide in `mean_field_games.md`.
