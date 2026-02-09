# Mean Field Games

Solve 1D Mean Field Games and Mean Field-Type Control problems using the Rust backend.

## Key Concepts
- Coupled HJB–Fokker-Planck PDE system
- Density evolution over time/space
- Congestion and noise parameters control stability

## Usage

```python
from optimizr import MFGConfig, solve_mfg_1d_rust

config = MFGConfig(
    nx=64,
    nt=40,
    x_min=-3.0,
    x_max=3.0,
    T=1.0,
    epsilon=0.1,
    kappa=1.0,
)

solution = solve_mfg_1d_rust(config)
print("Converged:", solution.converged)
print("Value function grid:", solution.value_function.shape)
print("Density grid:", solution.density.shape)
```

## Tips
- Increase `nx`/`nt` for smoother solutions; expect higher compute.
- Reduce `epsilon` for sharper dynamics; increase if unstable.
- Use `kappa` to control congestion cost.
