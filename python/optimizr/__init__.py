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
)

__version__ = "0.1.0"
__all__ = [
    "HMM",
    "mcmc_sample",
    "differential_evolution",
    "grid_search",
    "mutual_information",
    "shannon_entropy",
]
