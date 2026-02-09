# Differential Evolution

**Differential Evolution (DE)** is a powerful evolutionary algorithm for global optimization of continuous, non-linear, non-convex functions. It's particularly effective for multimodal optimization landscapes.

## Algorithm Overview

DE works by maintaining a **population** of candidate solutions and iteratively improving them through:

1. **Mutation**: Create mutant vectors by combining existing solutions
2. **Crossover**: Mix mutant with target vector
3. **Selection**: Keep better solution (greedy selection)

### Key Parameters

- **Population Size** (`pop_size`): Number of candidate solutions (typically 10× problem dimension)
- **Mutation Factor** (`F`): Scale factor for difference vectors (0.5-1.0)
- **Crossover Rate** (`CR`): Probability of using mutant component (0.0-1.0)
- **Strategy**: Mutation/crossover strategy (see below)

## Strategies

OptimizR implements 5 DE strategies:

### 1. `rand/1/bin`
```
mutant = x_r1 + F * (x_r2 - x_r3)
```
Most explorative, good for diverse populations.

### 2. `best/1/bin`
```
mutant = x_best + F * (x_r1 - x_r2)
```
Exploitative, fast convergence but may get stuck.

### 3. `current-to-best/1/bin`
```
mutant = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
```
Balanced exploration/exploitation.

### 4. `rand/2/bin`
```
mutant = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)
```
More diversity through two difference vectors.

### 5. `best/2/bin`
```
mutant = x_best + F * (x_r1 - x_r2) + F * (x_r3 - x_r4)
```
Aggressive convergence to best solution.

## Usage Example

```python
import numpy as np
from optimizr import differential_evolution

def rastrigin(x):
    A = 10
    return A * len(x) + sum(x**2 - A * np.cos(2 * np.pi * x))

best_x, best_fx = differential_evolution(
    objective_fn=rastrigin,
    bounds=[(-5.12, 5.12)] * 10,
    strategy="best1",
    popsize=20,
    maxiter=500,
    adaptive=True,
)

print(f"Best fitness: {best_fx:.6f}")
print(f"Best solution: {best_x}")
```

## Advanced Features

### Adaptive jDE

Enable self-adaptive F and CR parameters:

```python
de = DifferentialEvolution(
    bounds=[(-5, 5)] * 20,
    adaptive=True,  # Enable jDE
    tau_F=0.1,      # F adaptation rate
    tau_CR=0.1      # CR adaptation rate
)
```

### Constraint Handling

For constrained optimization:

```python
def constraints(x):
    """Return array of constraint violations (> 0 means violated)"""
    return np.array([
        x[0]**2 + x[1]**2 - 1,  # x0^2 + x1^2 <= 1
        x[0] + x[1] - 2         # x0 + x1 <= 2
    ])

de = DifferentialEvolution(
    bounds=[(-5, 5)] * 2,
    constraints=constraints,
    penalty_factor=1000
)
```

## Performance Tips

1. **Population Size**: Start with `10 × dim`, increase if stuck
2. **F parameter**: 
   - Low (0.4-0.6): Fine-tuning, local search
   - High (0.8-1.0): Exploration, escape local minima
3. **CR parameter**:
   - Low (0.1-0.3): Separable problems
   - High (0.9-1.0): Non-separable, coupled variables
4. **Strategy Selection**:
   - Unknown landscape → `rand/1/bin` or `rand/2/bin`
   - Smooth, unimodal → `best/1/bin`
   - Multimodal, deceptive → `current-to-best/1/bin`

## Benchmarks

Performance on standard test functions (10D, 500 iterations):

| Function | Success Rate | Avg Time | Best Fitness |
|----------|--------------|----------|--------------|
| Sphere | 100% | 12ms | 1e-12 |
| Rosenbrock | 98% | 18ms | 3e-6 |
| Rastrigin | 87% | 22ms | 0.02 |
| Ackley | 95% | 15ms | 2e-8 |

*Compared to SciPy `differential_evolution`: 50-80× faster*

## Mathematical Details

### Mutation Operator

For strategy `rand/1/bin`:

$$
\mathbf{v}_{i,g} = \mathbf{x}_{r_1,g} + F \cdot (\mathbf{x}_{r_2,g} - \mathbf{x}_{r_3,g})
$$

Where:
- $\mathbf{v}_{i,g}$: Mutant vector for individual $i$ at generation $g$
- $\mathbf{x}_{r_j,g}$: Randomly selected individuals ($r_1 \neq r_2 \neq r_3 \neq i$)
- $F \in [0, 2]$: Mutation scaling factor

### Crossover Operator

Binomial crossover:

$$
u_{i,j,g} = \begin{cases}
v_{i,j,g} & \text{if } \text{rand}(0,1) < CR \text{ or } j = j_{rand} \\\\
x_{i,j,g} & \text{otherwise}
\end{cases}
$$

Ensures at least one component from mutant.

### Selection Operator

Greedy selection:

$$
\mathbf{x}_{i,g+1} = \begin{cases}
\mathbf{u}_{i,g} & \text{if } f(\mathbf{u}_{i,g}) \leq f(\mathbf{x}_{i,g}) \\\\
\mathbf{x}_{i,g} & \text{otherwise}
\end{cases}
$$

## References

1. Storn, R., & Price, K. (1997). *Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces*. Journal of global optimization, 11(4), 341-359.

2. Das, S., & Suganthan, P. N. (2011). *Differential evolution: A survey of the state-of-the-art*. IEEE transactions on evolutionary computation, 15(1), 4-31.

3. Brest, J., et al. (2006). *Self-adapting control parameters in differential evolution: A comparative study on numerical benchmark problems*. IEEE transactions on evolutionary computation, 10(6), 646-657.

## See Also

- [API Reference](../api/differential_evolution.md)
- [Jupyter Tutorial](https://github.com/ThotDjehuty/optimiz-r/blob/main/examples/01_differential_evolution_tutorial.ipynb)
- [Benchmarks](../benchmarks.md)
