///! Differential Evolution Global Optimization
///!
///! This module implements the Differential Evolution (DE) algorithm, a population-based
///! stochastic optimization method effective for non-convex, multimodal problems.
///!
///! # Algorithm
///!
///! DE evolves a population of candidate solutions using:
///!
///! 1. **Mutation**: Create mutant vector v = a + F × (b - c)
///!    where a, b, c are randomly selected population members
///!
///! 2. **Crossover**: Mix mutant with target to create trial vector
///!    u[j] = v[j] if rand() < CR or j = j_rand, else u[j] = x[j]
///!
///! 3. **Selection**: Keep trial if it improves objective
///!    x_next = u if f(u) < f(x), else x_next = x
///!
///! # References
///!
///! Storn, R., & Price, K. (1997). Differential evolution–a simple and efficient
///! heuristic for global optimization over continuous spaces. Journal of global
///! optimization, 11(4), 341-359.

use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::Bound;
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

/// Differential Evolution Result
#[pyclass]
#[derive(Clone)]
pub struct DEResult {
    /// Best parameters found
    #[pyo3(get)]
    pub x: Vec<f64>,
    
    /// Best objective value
    #[pyo3(get)]
    pub fun: f64,
    
    /// Number of function evaluations
    #[pyo3(get)]
    pub nfev: usize,
}

#[pymethods]
impl DEResult {
    fn __repr__(&self) -> String {
        format!(
            "DEResult(fun={:.6}, nfev={}, nparams={})",
            self.fun,
            self.nfev,
            self.x.len()
        )
    }
}

/// Differential Evolution Optimizer
///
/// Global optimization algorithm using mutation, crossover, and selection
/// to evolve a population towards the optimum.
///
/// # Arguments
///
/// * `objective_fn` - Python callable to minimize: f(x) -> float
/// * `bounds` - [(min, max), ...] bounds for each parameter
/// * `popsize` - Population size multiplier (total size = popsize × n_params)
/// * `maxiter` - Maximum number of generations
/// * `f` - Mutation factor (typically 0.5-2.0)
/// * `cr` - Crossover probability (typically 0.1-0.9)
///
/// # Returns
///
/// DEResult with best parameters and objective value
///
/// # Example
///
/// ```python
/// import optimizr
///
/// # Minimize Rosenbrock function
/// def rosenbrock(x):
///     return sum(100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2 
///                for i in range(len(x)-1))
///
/// result = optimizr.differential_evolution(
///     objective_fn=rosenbrock,
///     bounds=[(-5, 5)] * 10,
///     popsize=15,
///     maxiter=1000,
///     f=0.8,
///     cr=0.7
/// )
///
/// print(f"Minimum: {result.fun} at {result.x}")
/// ```
#[pyfunction]
#[pyo3(signature = (objective_fn, bounds, popsize=15, maxiter=1000, f=0.8, cr=0.7))]
pub fn differential_evolution(
    objective_fn: &Bound<'_, PyAny>,
    bounds: Vec<(f64, f64)>,
    popsize: usize,
    maxiter: usize,
    f: f64,
    cr: f64,
) -> PyResult<DEResult> {
    let mut rng = thread_rng();
    let n_params = bounds.len();
    let pop_size = popsize * n_params;
    
    if n_params == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "bounds cannot be empty"
        ));
    }
    
    // Initialize population uniformly in bounds
    let mut population: Vec<Vec<f64>> = (0..pop_size)
        .map(|_| {
            bounds
                .iter()
                .map(|(low, high)| {
                    let uniform = Uniform::new(*low, *high);
                    uniform.sample(&mut rng)
                })
                .collect()
        })
        .collect();
    
    // Evaluate initial population
    let mut fitness: Vec<f64> = population
        .iter()
        .map(|ind| {
            objective_fn
                .call1((ind.clone(),))
                .and_then(|r| r.extract::<f64>())
                .unwrap_or(f64::INFINITY)
        })
        .collect();
    
    let mut nfev = pop_size;
    
    // Find initial best
    let mut best_idx = fitness
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    
    // Evolution loop
    for _generation in 0..maxiter {
        for i in 0..pop_size {
            // Select three random distinct individuals
            let indices: Vec<usize> = (0..pop_size).filter(|&idx| idx != i).collect();
            
            if indices.len() < 3 {
                continue;
            }
            
            let uniform_idx = Uniform::new(0, indices.len());
            
            let a_idx = indices[uniform_idx.sample(&mut rng)];
            let b_idx = indices[uniform_idx.sample(&mut rng) % indices.len()];
            let c_idx = indices[uniform_idx.sample(&mut rng) % indices.len()];
            
            // Mutation: v = a + F * (b - c)
            let mutant: Vec<f64> = (0..n_params)
                .map(|j| {
                    let v = population[a_idx][j] + f * (population[b_idx][j] - population[c_idx][j]);
                    v.max(bounds[j].0).min(bounds[j].1) // Clip to bounds
                })
                .collect();
            
            // Crossover: mix mutant with target
            let uniform_prob = Uniform::new(0.0, 1.0);
            let j_rand = uniform_idx.sample(&mut rng) % n_params;
            let mut trial = population[i].clone();
            
            for j in 0..n_params {
                if uniform_prob.sample(&mut rng) < cr || j == j_rand {
                    trial[j] = mutant[j];
                }
            }
            
            // Selection: keep trial if it improves fitness
            let trial_fitness = objective_fn
                .call1((trial.clone(),))
                .and_then(|r| r.extract::<f64>())
                .unwrap_or(f64::INFINITY);
            
            nfev += 1;
            
            if trial_fitness < fitness[i] {
                population[i] = trial;
                fitness[i] = trial_fitness;
                
                if trial_fitness < fitness[best_idx] {
                    best_idx = i;
                }
            }
        }
    }
    
    Ok(DEResult {
        x: population[best_idx].clone(),
        fun: fitness[best_idx],
        nfev,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_de_result() {
        let result = DEResult {
            x: vec![1.0, 2.0],
            fun: 3.14,
            nfev: 1000,
        };
        
        assert_eq!(result.x.len(), 2);
        assert_eq!(result.nfev, 1000);
    }
}
