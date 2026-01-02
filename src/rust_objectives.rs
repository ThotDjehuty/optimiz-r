///! Rust-native objective functions for GIL-free parallelization
///!
///! This module defines a RustObjective trait that enables parallel evaluation
///! of objective functions without Python GIL contention. Useful for:
///! - Benchmark functions (Sphere, Rosenbrock, Rastrigin, etc.)
///! - Pure mathematical functions
///! - High-throughput optimization scenarios
///!
///! Unlike Python callbacks, RustObjective functions can be parallelized
///! using Rayon for 10-100× speedup on multi-core systems.

use pyo3::prelude::*;

/// Trait for Rust-native objective functions
/// 
/// Implementing this trait allows objective functions to be evaluated
/// in parallel without Python GIL contention.
pub trait RustObjective: Send + Sync {
    /// Evaluate the objective function at point x
    fn evaluate(&self, x: &[f64]) -> f64;
    
    /// Optional: Get the dimensionality of the problem
    fn dimension(&self) -> Option<usize> {
        None
    }
    
    /// Optional: Get the known global optimum (for benchmarking)
    fn global_optimum(&self) -> Option<f64> {
        None
    }
    
    /// Optional: Get the known optimal solution (for benchmarking)
    fn optimal_solution(&self) -> Option<Vec<f64>> {
        None
    }
}

// ============================================================================
// Benchmark Functions
// ============================================================================

/// Sphere function: f(x) = sum(x_i^2)
/// Global minimum: f(0, ..., 0) = 0
/// Convex, unimodal, separable
#[pyclass]
#[derive(Clone)]
pub struct Sphere {
    #[pyo3(get)]
    pub dim: usize,
}

#[pymethods]
impl Sphere {
    #[new]
    pub fn new(dim: usize) -> Self {
        Sphere { dim }
    }
    
    pub fn __call__(&self, x: Vec<f64>) -> f64 {
        self.evaluate(&x)
    }
}

impl RustObjective for Sphere {
    fn evaluate(&self, x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }
    
    fn dimension(&self) -> Option<usize> {
        Some(self.dim)
    }
    
    fn global_optimum(&self) -> Option<f64> {
        Some(0.0)
    }
    
    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![0.0; self.dim])
    }
}

/// Rosenbrock function: f(x) = sum(100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
/// Global minimum: f(1, ..., 1) = 0
/// Non-convex, unimodal, non-separable
#[pyclass]
#[derive(Clone)]
pub struct Rosenbrock {
    #[pyo3(get)]
    pub dim: usize,
}

#[pymethods]
impl Rosenbrock {
    #[new]
    pub fn new(dim: usize) -> Self {
        Rosenbrock { dim }
    }
    
    pub fn __call__(&self, x: Vec<f64>) -> f64 {
        self.evaluate(&x)
    }
}

impl RustObjective for Rosenbrock {
    fn evaluate(&self, x: &[f64]) -> f64 {
        (0..x.len() - 1)
            .map(|i| {
                let term1 = x[i + 1] - x[i] * x[i];
                let term2 = 1.0 - x[i];
                100.0 * term1 * term1 + term2 * term2
            })
            .sum()
    }
    
    fn dimension(&self) -> Option<usize> {
        Some(self.dim)
    }
    
    fn global_optimum(&self) -> Option<f64> {
        Some(0.0)
    }
    
    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![1.0; self.dim])
    }
}

/// Rastrigin function: f(x) = 10n + sum(x_i^2 - 10cos(2πx_i))
/// Global minimum: f(0, ..., 0) = 0
/// Highly multimodal, separable
#[pyclass]
#[derive(Clone)]
pub struct Rastrigin {
    #[pyo3(get)]
    pub dim: usize,
}

#[pymethods]
impl Rastrigin {
    #[new]
    pub fn new(dim: usize) -> Self {
        Rastrigin { dim }
    }
    
    pub fn __call__(&self, x: Vec<f64>) -> f64 {
        self.evaluate(&x)
    }
}

impl RustObjective for Rastrigin {
    fn evaluate(&self, x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let pi = std::f64::consts::PI;
        
        10.0 * n + x.iter()
            .map(|xi| xi * xi - 10.0 * (2.0 * pi * xi).cos())
            .sum::<f64>()
    }
    
    fn dimension(&self) -> Option<usize> {
        Some(self.dim)
    }
    
    fn global_optimum(&self) -> Option<f64> {
        Some(0.0)
    }
    
    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![0.0; self.dim])
    }
}

/// Ackley function: f(x) = -20exp(-0.2√(1/n ∑x_i^2)) - exp(1/n ∑cos(2πx_i)) + 20 + e
/// Global minimum: f(0, ..., 0) = 0
/// Highly multimodal, non-separable
#[pyclass]
#[derive(Clone)]
pub struct Ackley {
    #[pyo3(get)]
    pub dim: usize,
}

#[pymethods]
impl Ackley {
    #[new]
    pub fn new(dim: usize) -> Self {
        Ackley { dim }
    }
    
    pub fn __call__(&self, x: Vec<f64>) -> f64 {
        self.evaluate(&x)
    }
}

impl RustObjective for Ackley {
    fn evaluate(&self, x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let pi = std::f64::consts::PI;
        let e = std::f64::consts::E;
        
        let sum_sq = x.iter().map(|xi| xi * xi).sum::<f64>();
        let sum_cos = x.iter().map(|xi| (2.0 * pi * xi).cos()).sum::<f64>();
        
        -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp()
            - (sum_cos / n).exp()
            + 20.0
            + e
    }
    
    fn dimension(&self) -> Option<usize> {
        Some(self.dim)
    }
    
    fn global_optimum(&self) -> Option<f64> {
        Some(0.0)
    }
    
    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![0.0; self.dim])
    }
}

/// Griewank function: f(x) = 1 + (1/4000)∑x_i^2 - ∏cos(x_i/√i)
/// Global minimum: f(0, ..., 0) = 0
/// Multimodal, non-separable
#[pyclass]
#[derive(Clone)]
pub struct Griewank {
    #[pyo3(get)]
    pub dim: usize,
}

#[pymethods]
impl Griewank {
    #[new]
    pub fn new(dim: usize) -> Self {
        Griewank { dim }
    }
    
    pub fn __call__(&self, x: Vec<f64>) -> f64 {
        self.evaluate(&x)
    }
}

impl RustObjective for Griewank {
    fn evaluate(&self, x: &[f64]) -> f64 {
        let sum_sq = x.iter().map(|xi| xi * xi).sum::<f64>();
        let prod_cos = x.iter()
            .enumerate()
            .map(|(i, xi)| (xi / ((i + 1) as f64).sqrt()).cos())
            .product::<f64>();
        
        1.0 + sum_sq / 4000.0 - prod_cos
    }
    
    fn dimension(&self) -> Option<usize> {
        Some(self.dim)
    }
    
    fn global_optimum(&self) -> Option<f64> {
        Some(0.0)
    }
    
    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![0.0; self.dim])
    }
}

// ============================================================================
// Python Bindings
// ============================================================================

pub fn register_benchmark_functions(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Sphere>()?;
    m.add_class::<Rosenbrock>()?;
    m.add_class::<Rastrigin>()?;
    m.add_class::<Ackley>()?;
    m.add_class::<Griewank>()?;
    Ok(())
}
