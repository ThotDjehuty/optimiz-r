//! Markov Chain Monte Carlo (MCMC) module
//!
//! Modular implementation of MCMC sampling algorithms with:
//!
//! - Multiple proposal strategies (Gaussian, Adaptive)
//! - Builder pattern for configuration
//! - Metropolis-Hastings algorithm
//! - Diagnostic tools (acceptance rate, autocorrelation)
//! - Python bindings via PyO3
//!
//! # Example
//!
//! ```rust
//! use optimizr::mcmc::{MCMCConfig, MetropolisHastings, GaussianProposal};
//!
//! // Define your log-likelihood function
//! // Create config and sample
//! ```

mod proposal;
mod config;
mod likelihood;
mod sampler;
mod python_bindings;

// Re-export public API
pub use proposal::{ProposalStrategy, GaussianProposal, AdaptiveProposal};
pub use config::{MCMCConfig, MCMCConfigBuilder};
pub use likelihood::{LogLikelihood, PyLogLikelihood};
pub use sampler::MetropolisHastings;
pub use python_bindings::{mcmc_sample, adaptive_mcmc_sample};
