# API: Sparse Optimization

## sparse_pca_py
```python
from optimizr import sparse_pca_py

components = sparse_pca_py(
    X,
    n_components=3,
    l1_ratio=0.2,
)
```
- `X`: 2D NumPy array
- Returns component matrix `(n_components, n_features)`

## box_tao_decomposition_py
```python
from optimizr import box_tao_decomposition_py
solution = box_tao_decomposition_py(X)
```

## elastic_net_py
```python
from optimizr import elastic_net_py
coeffs = elastic_net_py(X, y, l1_ratio=0.3, alpha=0.01)
```
