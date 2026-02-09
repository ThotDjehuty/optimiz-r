# Sparse Optimization

Sparse PCA and Elastic Net utilities with Rust speed.

## Sparse PCA

```python
import numpy as np
from optimizr import sparse_pca_py

X = np.random.randn(500, 20)
components = sparse_pca_py(X, n_components=5, l1_ratio=0.15)
print(components.shape)  # (5, 20)
```

## Elastic Net

```python
import numpy as np
from optimizr import elastic_net_py

X = np.random.randn(200, 8)
y = np.random.randn(200)
coeffs = elastic_net_py(X, y, l1_ratio=0.3, alpha=0.01)
print(coeffs)
```

## Notes
- Inputs should be NumPy arrays; data is copied to Rust.
- `l1_ratio` balances sparsity vs ridge penalty.
- Standardize features before calling for stable solutions.
