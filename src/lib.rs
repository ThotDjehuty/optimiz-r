//! OptimizR - High-Performance Optimization Algorithms
//! ===================================================
//!
//! This library provides fast, reliable implementations of advanced optimization
//! and statistical inference algorithms, with Python bindings via PyO3.
//!
//! # Architecture
//!
//! The library is designed with modularity, functional programming patterns,
//! and trait-based abstractions:
//!
//! - `core`: Core traits (Optimizer, Sampler, InformationMeasure) and error types
//! - `functional`: Functional programming utilities (composition, memoization, pipes)
//! - Refactored modules with trait-based design and parallel support
//! - Original modules maintained for backward compatibility
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

// Core modules with trait-based architecture
pub mod core;
pub mod functional;

// New modular structure (recommended)
pub mod hmm;
pub mod mcmc;
pub mod de;

// Legacy modules for backward compatibility
mod hmm_legacy;
mod mcmc_legacy;
mod hmm_refactored;
mod mcmc_refactored;
mod de_refactored;
mod differential_evolution;
mod grid_search;
mod information_theory;

/// OptimizR Python module
#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ===== New Modular API (Recommended) =====
    
    // HMM functions (modular structure)
    m.add_class::<hmm::HMMParams>()?;
    m.add_function(wrap_pyfunction!(hmm::fit_hmm, m)?)?;
    m.add_function(wrap_pyfunction!(hmm::viterbi_decode, m)?)?;
    
    // MCMC functions (modular structure)
    m.add_function(wrap_pyfunction!(mcmc::mcmc_sample, m)?)?;
    m.add_function(wrap_pyfunction!(mcmc::adaptive_mcmc_sample, m)?)?;
    
    // DE functions (modular structure - uses de_refactored for now)
    m.add_class::<de::DEResult>()?;
    m.add_function(wrap_pyfunction!(de::differential_evolution, m)?)?;
    
    // ===== Legacy API (Backward Compatible) =====
    
    // Legacy optimization functions
    m.add_function(wrap_pyfunction!(differential_evolution::differential_evolution, m)?)?;
    m.add_function(wrap_pyfunction!(grid_search::grid_search, m)?)?;
    
    // Information theory functions
    m.add_function(wrap_pyfunction!(information_theory::mutual_information, m)?)?;
    m.add_function(wrap_pyfunction!(information_theory::shannon_entropy, m)?)?;
    
    Ok(())
}
