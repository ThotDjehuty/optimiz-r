# API: Optimal Control / Kalman

Control utilities are exposed through the Rust extension (`optimizr._core`).

```python
from optimizr import maths_toolkit

if maths_toolkit is None:
    raise RuntimeError("Rust backend missing; reinstall with `pip install .`.")

# Initialize a Kalman filter
kf = maths_toolkit.init_kalman_filter(F, H, Q, R)
state = maths_toolkit.kalman_predict(kf, x0)
state = maths_toolkit.kalman_update(kf, state, observation)
```

Parameters
- `F`: state transition matrix (list of lists)
- `H`: observation matrix
- `Q`: process noise covariance
- `R`: observation noise covariance

Also see the Mean Field Games API in `mean_field_games.md`.
