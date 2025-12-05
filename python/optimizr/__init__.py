"""
OptimizR - High-Performance Optimization Algorithms
===================================================

Fast, reliable implementations of advanced optimization and statistical
inference algorithms with Rust acceleration and pure Python fallbacks.

.. moduleauthor:: OptimizR Contributors

"""

from optimizr.hmm import HMM
from optimizr.core import (
    mcmc_sample,
    differential_evolution,
    grid_search,
    mutual_information,
    shannon_entropy,
    sparse_pca_py,
    box_tao_decomposition_py,
    elastic_net_py,
    hurst_exponent_py,
    compute_risk_metrics_py,
    estimate_half_life_py,
    bootstrap_returns_py,
)

__version__ = "0.1.0"
__all__ = [
    "HMM",
    "mcmc_sample",
    "differential_evolution",
    "grid_search",
    "mutual_information",
    "shannon_entropy",
    "sparse_pca_py",
    "box_tao_decomposition_py",
    "elastic_net_py",
    "hurst_exponent_py",
    "compute_risk_metrics_py",
    "estimate_half_life_py",
    "bootstrap_returns_py",
]
