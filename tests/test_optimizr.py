"""
Test suite for OptimizR
"""

import pytest
import numpy as np
from optimizr import (
    HMM,
    mcmc_sample,
    differential_evolution,
    grid_search,
    mutual_information,
    shannon_entropy,
)


class TestHMM:
    """Test Hidden Markov Model"""
    
    def test_hmm_initialization(self):
        hmm = HMM(n_states=3)
        assert hmm.n_states == 3
        # After initialization, matrices are zeros (not None)
        assert hmm.transition_matrix_ is not None
        assert hmm.transition_matrix_.shape == (3, 3)
    
    def test_hmm_fit(self):
        np.random.seed(42)
        returns = np.random.randn(1000)
        
        hmm = HMM(n_states=2)
        hmm.fit(returns, n_iterations=10)
        
        assert hmm.transition_matrix_ is not None
        assert hmm.transition_matrix_.shape == (2, 2)
        assert hmm.emission_means_ is not None
        assert len(hmm.emission_means_) == 2
    
    def test_hmm_predict(self):
        np.random.seed(42)
        returns = np.random.randn(100)
        
        hmm = HMM(n_states=2)
        hmm.fit(returns, n_iterations=10)
        states = hmm.predict(returns)
        
        assert len(states) == len(returns)
        assert set(states).issubset({0, 1})


class TestMCMC:
    """Test MCMC Sampling"""
    
    def test_mcmc_basic(self):
        def log_likelihood(params, data):
            mu, sigma = params
            if sigma <= 0:
                return float('-inf')
            residuals = (np.array(data) - mu) / sigma
            return -0.5 * np.sum(residuals**2) - len(data) * np.log(sigma)
        
        np.random.seed(42)
        data = np.random.randn(50) + 2.0
        
        samples = mcmc_sample(
            log_likelihood_fn=log_likelihood,
            data=data,
            initial_params=np.array([0.0, 1.0]),
            param_bounds=[(-10, 10), (0.1, 10)],
            n_samples=100,
            burn_in=10,
            proposal_std=0.1
        )
        
        assert samples.shape == (100, 2)
        assert np.all(samples[:, 1] > 0)  # Sigma should be positive


class TestDifferentialEvolution:
    """Test Differential Evolution"""
    
    def test_de_sphere(self):
        """Test on simple sphere function"""
        def sphere(x):
            return np.sum(np.array(x)**2)
        
        x, fun = differential_evolution(
            objective_fn=sphere,
            bounds=[(-5, 5)] * 3,
            popsize=10,
            maxiter=50
        )
        
        assert len(x) == 3
        assert fun < 1.0  # Should find near-zero minimum
    
    def test_de_rosenbrock(self):
        """Test on Rosenbrock function"""
        def rosenbrock(x):
            x = np.array(x)
            return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        x, fun = differential_evolution(
            objective_fn=rosenbrock,
            bounds=[(-2, 2)] * 5,
            popsize=15,
            maxiter=100
        )
        
        assert len(x) == 5
        assert fun < 10.0  # Should find reasonably good minimum


class TestGridSearch:
    """Test Grid Search"""
    
    def test_grid_search_2d(self):
        """Test on 2D quadratic"""
        def objective(x):
            return -(x[0]**2 + x[1]**2)  # Peak at (0, 0)
        
        x, fun = grid_search(
            objective_fn=objective,
            bounds=[(-5, 5), (-5, 5)],
            n_points=20
        )
        
        assert len(x) == 2
        assert np.allclose(x, [0, 0], atol=0.5)  # Should find near (0, 0)
        assert fun > -1.0  # Should find near-zero maximum


class TestInformationTheory:
    """Test Information Theory Metrics"""
    
    def test_shannon_entropy_uniform(self):
        """Uniform distribution should have high entropy"""
        np.random.seed(42)
        x = np.random.uniform(0, 1, 10000)
        entropy = shannon_entropy(x, n_bins=10)
        
        assert entropy > 2.0  # ln(10) ≈ 2.3
    
    def test_shannon_entropy_constant(self):
        """Constant should have zero entropy"""
        x = np.ones(1000)
        entropy = shannon_entropy(x, n_bins=10)
        
        assert entropy < 0.01
    
    def test_mutual_information_independent(self):
        """Independent variables should have low MI"""
        np.random.seed(42)
        x = np.random.randn(10000)
        y = np.random.randn(10000)
        mi = mutual_information(x, y, n_bins=10)
        
        assert mi >= 0.0
        assert mi < 0.5  # Should be close to zero
    
    def test_mutual_information_dependent(self):
        """Dependent variables should have high MI"""
        np.random.seed(42)
        x = np.random.randn(10000)
        y = 2 * x + np.random.randn(10000) * 0.1
        mi = mutual_information(x, y, n_bins=20)
        
        assert mi > 1.0  # Strong dependence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
