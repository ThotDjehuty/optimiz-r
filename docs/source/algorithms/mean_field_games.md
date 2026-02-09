# Mean Field Games

Solve 1D Mean Field Games and mean-field–type control problems with the Rust backend. The solver couples a backward HJB equation with a forward Fokker–Planck equation using a fixed-point loop and implicit diffusion for stability.

## What this module provides
- **Rust solver with PyO3 bindings**: `solve_mfg_1d_rust` and `MFGConfig` exposed to Python.
- **Stable numerics**: upwind transport + implicit diffusion, mass renormalization each iteration.
- **Performance**: ~0.4 s for a 100×100 grid on laptop-class CPUs (measured in the tutorial notebook).
- **Coverage**: Congestion term, relaxation `alpha`, configurable domain and viscosity `nu`.

## Usage

```python
import numpy as np
from optimizr import MFGConfig, solve_mfg_1d_rust

x = np.linspace(0, 1, 100)
m0 = np.exp(-50 * (x - 0.3) ** 2)
m0 /= np.trapz(m0, x)

u_terminal = 0.5 * (x - 0.7) ** 2
config = MFGConfig(
    nx=100,
    nt=100,
    x_min=0.0,
    x_max=1.0,
    T=1.0,
    nu=0.01,
    max_iter=50,
    tol=1e-5,
    alpha=0.5,
)

u, m, iters = solve_mfg_1d_rust(m0, u_terminal, config, lambda_congestion=0.5)
print(f"converged in {iters} iterations: u{u.shape}, m{m.shape}")
```

## What to monitor
- Residuals: track `||m^{k+1}-m^k||_1` and `||u^{k+1}-u^k||_inf`; stop when both flatten.
- Mass conservation: integrate `m` after each iteration; values close to 1.0 indicate stable transport.
- CFL sanity: if oscillations appear, reduce `dt` (increase `nt`) or raise `nu` slightly.

## Practical tips
- Grid resolution: start with `nx=64, nt=40`; move to 100×100 for publication-quality plots.
- Congestion: increase `lambda_congestion` to avoid density spikes; decrease for freer flow.
- Relaxation: `alpha=0.5` is a stable default; lower if the fixed-point loop jitters.

## Notebook and audit
- Full walkthrough: `examples/notebooks/mean_field_games_tutorial.ipynb` (all cells validated).
- Audit notes: the notebook renders convergence plots, 3D density/value surfaces, and time-slice snapshots; runs cleanly with the Rust backend (see `docs/MFG_TUTORIAL_COMPLETE.md`).
