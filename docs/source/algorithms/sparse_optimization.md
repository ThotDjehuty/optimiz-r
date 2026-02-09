# Sparse Optimization

Sparse PCA, Elastic Net, and Box–Tao decomposition with Rust speed.

## Sparse PCA

```python
import numpy as np
from optimizr import sparse_pca_py

X = np.random.randn(500, 20)
components = sparse_pca_py(X, n_components=5, l1_ratio=0.15)
print(components.shape)  # (5, 20)
```

- Output: component matrix `(n_components, n_features)`; rows are sparse loadings.
- Tuning: increase `l1_ratio` for harder sparsity; decrease to retain variance.

## Elastic Net

```python
import numpy as np
from optimizr import elastic_net_py

X = np.random.randn(200, 8)
y = np.random.randn(200)
coeffs = elastic_net_py(X, y, l1_ratio=0.3, alpha=0.01)
print(coeffs)
```

- Handles collinearity better than pure Lasso; use for factor shrinkage.
- Sweep `alpha` on a log scale (e.g., $10^{-3}$ to $10^{-1}$) and pick via validation.

## Box–Tao decomposition

```python
from optimizr import box_tao_decomposition_py
solution = box_tao_decomposition_py(X)
```

Useful for constrained sparse decomposition problems; the Rust backend keeps iterations fast.

## Practical notes
- Inputs must be NumPy arrays; standardize features for stable conditioning.
- For high dimensional data, start with fewer components/features to avoid over-regularization.
- Combine with risk metrics: use sparse loadings to build interpretable factors, then evaluate with `compute_risk_metrics_py`.
