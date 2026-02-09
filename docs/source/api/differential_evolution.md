# API: differential_evolution

```python
from optimizr import differential_evolution

best_x, best_fx = differential_evolution(
    objective_fn,
    bounds,
    popsize=15,
    maxiter=1000,
    f=None,
    cr=None,
    strategy="rand1",
    seed=None,
    tol=1e-6,
    atol=1e-8,
    track_history=False,
    parallel=False,
    adaptive=False,
    constraint_penalty=1000.0,
)
```

- `objective_fn`: callable `f(x: np.ndarray) -> float`
- `bounds`: list of `(min, max)` tuples
- Strategies: `rand1`, `best1`, `currenttobest1`, `rand2`, `best2`
- Returns `(best_x: np.ndarray, best_fx: float)`
