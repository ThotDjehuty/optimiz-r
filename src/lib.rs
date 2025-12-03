//! OptimizR - High-Performance Optimization Algorithms
//! ===================================================
//!
//! This library provides fast, reliable implementations of advanced optimization
//! and statistical inference algorithms, with Python bindings via PyO3.
//!
//! # Modules
//!
//! - `hmm`: Hidden Markov Model training and inference
//! - `mcmc`: Markov Chain Monte Carlo sampling
//! - `differential_evolution`: Global optimization algorithm
//! - `grid_search`: Exhaustive parameter space search
//! - `information_theory`: Mutual information and entropy calculations

use pyo3::prelude::*;
use pyo3::types::PyModule;

mod hmm;
mod mcmc;
mod differential_evolution;
mod grid_search;
mod information_theory;

/// OptimizR Python module
#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register HMM functions
    m.add_class::<hmm::HMMParams>()?;
    m.add_function(wrap_pyfunction!(hmm::fit_hmm, m)?)?;
    m.add_function(wrap_pyfunction!(hmm::viterbi_decode, m)?)?;
    
    // Register MCMC functions
    m.add_function(wrap_pyfunction!(mcmc::mcmc_sample, m)?)?;
    
    // Register optimization functions
    m.add_function(wrap_pyfunction!(differential_evolution::differential_evolution, m)?)?;
    m.add_function(wrap_pyfunction!(grid_search::grid_search, m)?)?;
    
    // Register information theory functions
    m.add_function(wrap_pyfunction!(information_theory::mutual_information, m)?)?;
    m.add_function(wrap_pyfunction!(information_theory::shannon_entropy, m)?)?;
    
    Ok(())
}
