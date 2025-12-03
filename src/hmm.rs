///! Hidden Markov Model (HMM) Implementation
///!
///! This module provides efficient implementations of:
///! - Baum-Welch algorithm (Expectation-Maximization) for parameter estimation
///! - Forward-Backward algorithm for state probability computation
///! - Viterbi algorithm for most likely state sequence decoding
///!
///! # Mathematical Background
///!
///! A Hidden Markov Model is defined by:
///! - Initial state probabilities: π = [π₁, π₂, ..., πₙ]
///! - Transition probabilities: A = {aᵢⱼ} where aᵢⱼ = P(state_t+1 = j | state_t = i)
///! - Emission probabilities: B = {bⱼ(o)} where bⱼ(o) = P(observation = o | state = j)
///!
///! For continuous observations, we use Gaussian emissions:
///! bⱼ(o) = 𝒩(o | μⱼ, σⱼ²)

use pyo3::prelude::*;
use std::f64;

/// HMM Parameters
///
/// Contains all learned parameters of a Hidden Markov Model with Gaussian emissions.
///
/// # Fields
///
/// * `n_states` - Number of hidden states
/// * `transition_matrix` - State transition probabilities (n_states × n_states)
/// * `emission_means` - Mean of Gaussian emission for each state
/// * `emission_stds` - Standard deviation of Gaussian emission for each state
/// * `initial_probs` - Initial state probabilities
#[pyclass]
#[derive(Clone, Debug)]
pub struct HMMParams {
    #[pyo3(get, set)]
    pub n_states: usize,
    #[pyo3(get, set)]
    pub transition_matrix: Vec<Vec<f64>>,
    #[pyo3(get, set)]
    pub emission_means: Vec<f64>,
    #[pyo3(get, set)]
    pub emission_stds: Vec<f64>,
    #[pyo3(get, set)]
    pub initial_probs: Vec<f64>,
}

#[pymethods]
impl HMMParams {
    /// Create new HMM parameters with uniform initialization
    ///
    /// # Arguments
    ///
    /// * `n_states` - Number of hidden states
    ///
    /// # Returns
    ///
    /// HMMParams with uniform initial, transition probabilities and unit Gaussian emissions
    #[new]
    pub fn new(n_states: usize) -> Self {
        let uniform_prob = 1.0 / n_states as f64;
        HMMParams {
            n_states,
            transition_matrix: vec![vec![uniform_prob; n_states]; n_states],
            emission_means: vec![0.0; n_states],
            emission_stds: vec![1.0; n_states],
            initial_probs: vec![uniform_prob; n_states],
        }
    }
    
    /// String representation of HMM parameters
    fn __repr__(&self) -> String {
        format!(
            "HMMParams(n_states={}, transition_shape={}x{}, emission_params={})",
            self.n_states,
            self.n_states,
            self.n_states,
            self.emission_means.len()
        )
    }
}

/// Fit Hidden Markov Model using Baum-Welch algorithm
///
/// The Baum-Welch algorithm is an Expectation-Maximization (EM) method for learning
/// HMM parameters from observed data.
///
/// # Algorithm Steps
///
/// 1. **E-step**: Compute expected state occupancies using Forward-Backward
/// 2. **M-step**: Update parameters to maximize expected log-likelihood
/// 3. Repeat until convergence
///
/// # Arguments
///
/// * `observations` - Time series of observed values
/// * `n_states` - Number of hidden states to learn
/// * `n_iterations` - Maximum number of EM iterations
/// * `tolerance` - Convergence threshold for log-likelihood change
///
/// # Returns
///
/// Learned `HMMParams` with optimized transition and emission parameters
///
/// # Example
///
/// ```python
/// import optimizr
/// import numpy as np
///
/// # Generate sample data
/// observations = np.random.randn(1000).tolist()
///
/// # Fit HMM with 3 states
/// params = optimizr.fit_hmm(
///     observations=observations,
///     n_states=3,
///     n_iterations=100,
///     tolerance=1e-6
/// )
///
/// print(params.transition_matrix)
/// ```
#[pyfunction]
#[pyo3(signature = (observations, n_states, n_iterations=100, tolerance=1e-6))]
pub fn fit_hmm(
    observations: Vec<f64>,
    n_states: usize,
    n_iterations: usize,
    tolerance: f64,
) -> PyResult<HMMParams> {
    let n_obs = observations.len();
    if n_obs == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Observations cannot be empty"
        ));
    }
    
    let mut params = HMMParams::new(n_states);
    
    // Initialize emission parameters using quantiles
    let mut sorted_obs = observations.clone();
    sorted_obs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    for i in 0..n_states {
        let start_idx = (i * n_obs) / n_states;
        let end_idx = ((i + 1) * n_obs) / n_states;
        let segment = &sorted_obs[start_idx..end_idx];
        
        if !segment.is_empty() {
            params.emission_means[i] = segment.iter().sum::<f64>() / segment.len() as f64;
            let var: f64 = segment
                .iter()
                .map(|x| (x - params.emission_means[i]).powi(2))
                .sum::<f64>()
                / segment.len() as f64;
            params.emission_stds[i] = var.sqrt().max(1e-6);
        }
    }
    
    // EM iterations
    let mut prev_log_likelihood = f64::NEG_INFINITY;
    
    for _iteration in 0..n_iterations {
        // E-step: Forward-Backward algorithm
        let alpha = forward(&observations, &params);
        let beta = backward(&observations, &params);
        let gamma = compute_gamma(&alpha, &beta);
        let xi = compute_xi(&observations, &params, &alpha, &beta);
        
        // M-step: Update parameters
        update_parameters(&observations, &mut params, &gamma, &xi);
        
        // Check convergence
        let log_likelihood = compute_log_likelihood(&alpha);
        
        if (log_likelihood - prev_log_likelihood).abs() < tolerance {
            break;
        }
        
        prev_log_likelihood = log_likelihood;
    }
    
    Ok(params)
}

/// Forward algorithm: Compute α(t, s) = P(o₁, ..., oₜ, qₜ = s | λ)
///
/// Computes the probability of observing the sequence up to time t
/// and being in state s at time t.
fn forward(observations: &[f64], params: &HMMParams) -> Vec<Vec<f64>> {
    let n_obs = observations.len();
    let n_states = params.n_states;
    let mut alpha = vec![vec![0.0; n_states]; n_obs];
    
    // Initialize: α(0, s) = πₛ * bₛ(o₀)
    for s in 0..n_states {
        alpha[0][s] = params.initial_probs[s] * emission_prob(observations[0], params, s);
    }
    
    // Normalize to prevent underflow
    normalize_row(&mut alpha[0]);
    
    // Recursion: α(t, s) = [Σᵢ α(t-1, i) * aᵢₛ] * bₛ(oₜ)
    for t in 1..n_obs {
        for s in 0..n_states {
            let mut sum = 0.0;
            for prev_s in 0..n_states {
                sum += alpha[t - 1][prev_s] * params.transition_matrix[prev_s][s];
            }
            alpha[t][s] = sum * emission_prob(observations[t], params, s);
        }
        normalize_row(&mut alpha[t]);
    }
    
    alpha
}

/// Backward algorithm: Compute β(t, s) = P(oₜ₊₁, ..., oₜ | qₜ = s, λ)
///
/// Computes the probability of observing the remaining sequence
/// given that we are in state s at time t.
fn backward(observations: &[f64], params: &HMMParams) -> Vec<Vec<f64>> {
    let n_obs = observations.len();
    let n_states = params.n_states;
    let mut beta = vec![vec![0.0; n_states]; n_obs];
    
    // Initialize: β(T-1, s) = 1
    for s in 0..n_states {
        beta[n_obs - 1][s] = 1.0;
    }
    
    // Recursion: β(t, s) = Σⱼ aₛⱼ * bⱼ(oₜ₊₁) * β(t+1, j)
    for t in (0..n_obs - 1).rev() {
        for s in 0..n_states {
            let mut sum = 0.0;
            for next_s in 0..n_states {
                sum += params.transition_matrix[s][next_s]
                    * emission_prob(observations[t + 1], params, next_s)
                    * beta[t + 1][next_s];
            }
            beta[t][s] = sum;
        }
        normalize_row(&mut beta[t]);
    }
    
    beta
}

/// Compute γ(t, s) = P(qₜ = s | O, λ)
///
/// State occupation probability: probability of being in state s at time t
/// given the entire observation sequence.
fn compute_gamma(alpha: &[Vec<f64>], beta: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_obs = alpha.len();
    let n_states = alpha[0].len();
    let mut gamma = vec![vec![0.0; n_states]; n_obs];
    
    for t in 0..n_obs {
        let sum: f64 = (0..n_states).map(|s| alpha[t][s] * beta[t][s]).sum();
        
        for s in 0..n_states {
            gamma[t][s] = if sum > 1e-10 {
                alpha[t][s] * beta[t][s] / sum
            } else {
                1.0 / n_states as f64 // Fallback to uniform
            };
        }
    }
    
    gamma
}

/// Compute ξ(t, i, j) = P(qₜ = i, qₜ₊₁ = j | O, λ)
///
/// Transition probability: probability of being in state i at time t
/// and state j at time t+1 given the entire observation sequence.
fn compute_xi(
    observations: &[f64],
    params: &HMMParams,
    alpha: &[Vec<f64>],
    beta: &[Vec<f64>],
) -> Vec<Vec<Vec<f64>>> {
    let n_obs = observations.len();
    let n_states = params.n_states;
    let mut xi = vec![vec![vec![0.0; n_states]; n_states]; n_obs - 1];
    
    for t in 0..n_obs - 1 {
        let mut sum = 0.0;
        
        for i in 0..n_states {
            for j in 0..n_states {
                xi[t][i][j] = alpha[t][i]
                    * params.transition_matrix[i][j]
                    * emission_prob(observations[t + 1], params, j)
                    * beta[t + 1][j];
                sum += xi[t][i][j];
            }
        }
        
        // Normalize
        if sum > 1e-10 {
            for i in 0..n_states {
                for j in 0..n_states {
                    xi[t][i][j] /= sum;
                }
            }
        }
    }
    
    xi
}

/// Update HMM parameters (M-step of Baum-Welch)
///
/// Updates transition and emission parameters to maximize the expected
/// complete data log-likelihood.
fn update_parameters(
    observations: &[f64],
    params: &mut HMMParams,
    gamma: &[Vec<f64>],
    xi: &[Vec<Vec<f64>>],
) {
    let n_obs = observations.len();
    let n_states = params.n_states;
    
    // Update transition matrix: aᵢⱼ = Σₜ ξ(t,i,j) / Σₜ γ(t,i)
    for i in 0..n_states {
        let denom: f64 = gamma[..n_obs - 1].iter().map(|g| g[i]).sum();
        
        for j in 0..n_states {
            let numer: f64 = xi.iter().map(|x| x[i][j]).sum();
            params.transition_matrix[i][j] = if denom > 1e-10 {
                numer / denom
            } else {
                1.0 / n_states as f64
            };
        }
    }
    
    // Update emission parameters: μⱼ = Σₜ γ(t,j)*oₜ / Σₜ γ(t,j)
    for s in 0..n_states {
        let weights: Vec<f64> = gamma.iter().map(|g| g[s]).collect();
        let sum_weights: f64 = weights.iter().sum();
        
        if sum_weights > 1e-10 {
            // Weighted mean
            let mean = observations
                .iter()
                .zip(weights.iter())
                .map(|(obs, w)| obs * w)
                .sum::<f64>()
                / sum_weights;
            
            // Weighted variance
            let var = observations
                .iter()
                .zip(weights.iter())
                .map(|(obs, w)| w * (obs - mean).powi(2))
                .sum::<f64>()
                / sum_weights;
            
            params.emission_means[s] = mean;
            params.emission_stds[s] = var.sqrt().max(1e-6);
        }
    }
}

/// Gaussian emission probability: bₛ(o) = 𝒩(o | μₛ, σₛ²)
///
/// Computes the probability of observing value o from state s
/// using a Gaussian distribution.
fn emission_prob(observation: f64, params: &HMMParams, state: usize) -> f64 {
    let mean = params.emission_means[state];
    let std = params.emission_stds[state];
    
    let z = (observation - mean) / std;
    let coef = 1.0 / (std * (2.0 * f64::consts::PI).sqrt());
    
    (coef * (-0.5 * z * z).exp()).max(1e-10) // Prevent underflow
}

/// Compute log-likelihood of observations
fn compute_log_likelihood(alpha: &[Vec<f64>]) -> f64 {
    let last_alpha = &alpha[alpha.len() - 1];
    let sum: f64 = last_alpha.iter().sum();
    sum.max(1e-10).ln()
}

/// Normalize a probability vector to sum to 1
fn normalize_row(row: &mut [f64]) {
    let sum: f64 = row.iter().sum();
    if sum > 1e-10 {
        for val in row.iter_mut() {
            *val /= sum;
        }
    } else {
        let uniform = 1.0 / row.len() as f64;
        for val in row.iter_mut() {
            *val = uniform;
        }
    }
}

/// Viterbi algorithm: Find most likely state sequence
///
/// Finds the state sequence that maximizes P(Q | O, λ) using dynamic programming.
///
/// # Algorithm
///
/// 1. Initialize: δ(0, s) = πₛ * bₛ(o₀)
/// 2. Recursion: δ(t, s) = maxᵢ[δ(t-1, i) * aᵢₛ] * bₛ(oₜ)
/// 3. Backtrack to find optimal path
///
/// # Arguments
///
/// * `observations` - Time series of observed values
/// * `params` - Learned HMM parameters
///
/// # Returns
///
/// Vector of most likely states at each time step
///
/// # Example
///
/// ```python
/// import optimizr
///
/// # After fitting HMM
/// states = optimizr.viterbi_decode(observations, params)
/// print(f"State sequence: {states}")
/// ```
#[pyfunction]
pub fn viterbi_decode(observations: Vec<f64>, params: HMMParams) -> PyResult<Vec<usize>> {
    let n_obs = observations.len();
    let n_states = params.n_states;
    
    if n_obs == 0 {
        return Ok(Vec::new());
    }
    
    let mut delta = vec![vec![f64::NEG_INFINITY; n_states]; n_obs];
    let mut psi = vec![vec![0usize; n_states]; n_obs];
    
    // Initialize
    for s in 0..n_states {
        delta[0][s] = params.initial_probs[s].ln() + emission_prob(observations[0], &params, s).ln();
    }
    
    // Recursion
    for t in 1..n_obs {
        for s in 0..n_states {
            let mut max_val = f64::NEG_INFINITY;
            let mut max_state = 0;
            
            for prev_s in 0..n_states {
                let val = delta[t - 1][prev_s] + params.transition_matrix[prev_s][s].ln();
                if val > max_val {
                    max_val = val;
                    max_state = prev_s;
                }
            }
            
            psi[t][s] = max_state;
            delta[t][s] = max_val + emission_prob(observations[t], &params, s).ln();
        }
    }
    
    // Backtrack
    let mut path = vec![0usize; n_obs];
    let mut max_val = f64::NEG_INFINITY;
    let mut max_state = 0;
    
    for s in 0..n_states {
        if delta[n_obs - 1][s] > max_val {
            max_val = delta[n_obs - 1][s];
            max_state = s;
        }
    }
    
    path[n_obs - 1] = max_state;
    
    for t in (0..n_obs - 1).rev() {
        path[t] = psi[t + 1][path[t + 1]];
    }
    
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmm_initialization() {
        let params = HMMParams::new(3);
        assert_eq!(params.n_states, 3);
        assert_eq!(params.transition_matrix.len(), 3);
        assert_eq!(params.emission_means.len(), 3);
    }

    #[test]
    fn test_fit_hmm() {
        let observations: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = fit_hmm(observations, 2, 10, 1e-6);
        assert!(result.is_ok());
    }

    #[test]
    fn test_viterbi() {
        let observations: Vec<f64> = vec![1.0, 1.5, 2.0, -1.0, -1.5, -2.0];
        let mut params = HMMParams::new(2);
        params.emission_means = vec![1.5, -1.5];
        params.emission_stds = vec![0.5, 0.5];
        
        let states = viterbi_decode(observations, params).unwrap();
        assert_eq!(states.len(), 6);
    }
}
