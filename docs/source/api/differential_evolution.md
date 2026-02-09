# API: differential_evolution

```python
from optimizr import differential_evolution

best_x, best_fx = differential_evolution(
    objective_fn,
    bounds,
    popsize=15,
    maxiter=1000,
    f=None,                 # mutation factor (auto if None)
    cr=None,                # crossover rate (auto if None)
    strategy="rand1",      # rand1, best1, currenttobest1, rand2, best2
    seed=None,
    tol=1e-6,
    atol=1e-8,
    track_history=False,    # keep per-iter best
    parallel=False,         # Python callbacks stay sequential; see Rust path below
    adaptive=False,         # jDE when True
    constraint_penalty=1000.0,
)
```

- `objective_fn`: callable `f(x: np.ndarray) -> float`
- `bounds`: list of `(min, max)` tuples
- Returns `(best_x: np.ndarray, best_fx: float)`

## Parallel Rust entry point

For built-in benchmark objectives (no Python callbacks), use the Rust-native path with Rayon:

```python
from optimizr import parallel_differential_evolution_rust

result = parallel_differential_evolution_rust(
    objective_name="rastrigin",  # sphere, rosenbrock, ackley, griewank
    bounds=[(-5, 5)] * 20,
    maxiter=500,
    parallel=True,
)
```

## Notes
- Adaptive control uses jDE in the current Python API; SHADE/L-SHADE live in Rust and will surface in a future release.
- Use `track_history=True` to export convergence curves for benchmarking.
