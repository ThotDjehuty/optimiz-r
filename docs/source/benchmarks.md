# Benchmarks

Performance is measured against NumPy/SciPy baselines on common objectives.

| Function | Dim | Iterations | Speedup vs SciPy |
|----------|-----|------------|------------------|
| Sphere | 10 | 200 | 50× |
| Rosenbrock | 10 | 400 | 60× |
| Rastrigin | 10 | 500 | 70× |

Numbers are indicative; run `examples/notebooks/05_performance_benchmarks.ipynb` on your hardware for exact results.
