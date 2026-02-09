# API: grid_search

```python
from optimizr import grid_search

best_params, best_score = grid_search(
    objective_fn,
    param_grid,
)
```

- `objective_fn`: callable receiving a dict of parameters and returning a scalar loss
- `param_grid`: dict of name -> list of values to enumerate
- Returns `(best_params: dict, best_score: float)`
