# Grid Search

Deterministic hyper-parameter sweeps with optional Rust acceleration.

## Usage

```python
from optimizr import grid_search

# Objective returns a scalar score (lower is better)
def objective(params):
    lr, dropout = params["lr"], params["dropout"]
    return (lr - 0.02)**2 + (dropout - 0.1)**2

best_params, best_score = grid_search(
    objective_fn=objective,
    param_grid={"lr": [0.005, 0.02, 0.05], "dropout": [0.05, 0.1, 0.2]},
)

print(best_params)
print(best_score)
```

## Notes
- The objective receives a dict of parameters.
- Exhaustive search is deterministic; keep grids small for large models.
- Combine with DE for warm-starting a local region.
