"""
Example: Hidden Markov Model for Regime Detection
=================================================

This example demonstrates using HMM to detect market regimes in synthetic data.
"""

import numpy as np
import matplotlib.pyplot as plt
from optimizr import HMM

# Generate synthetic data with regime changes
np.random.seed(42)

# Regime 1: Bull market (positive drift, low volatility)
bull_returns = np.random.normal(0.01, 0.015, 500)

# Regime 2: Bear market (negative drift, high volatility)
bear_returns = np.random.normal(-0.008, 0.03, 500)

# Regime 3: Sideways (no drift, medium volatility)
sideways_returns = np.random.normal(0.001, 0.02, 500)

# Combine regimes
returns = np.concatenate([bull_returns, bear_returns, sideways_returns])

# True regime labels (for comparison)
true_regimes = np.concatenate([
    np.zeros(500, dtype=int),
    np.ones(500, dtype=int),
    np.full(500, 2, dtype=int)
])

print("="*70)
print("HMM Regime Detection Example")
print("="*70)
print(f"\nGenerated {len(returns)} returns across 3 regimes")
print(f"Regime 1 (Bull): μ=0.01, σ=0.015")
print(f"Regime 2 (Bear): μ=-0.008, σ=0.03")
print(f"Regime 3 (Sideways): μ=0.001, σ=0.02")

# Fit HMM
print("\nFitting HMM with 3 states...")
hmm = HMM(n_states=3)
hmm.fit(returns, n_iterations=100, tolerance=1e-6)

print("\nLearned Parameters:")
print(f"Emission means: {hmm.emission_means_}")
print(f"Emission stds: {hmm.emission_stds_}")
print(f"\nTransition Matrix:")
print(hmm.transition_matrix_)

# Decode states
print("\nDecoding state sequence...")
predicted_states = hmm.predict(returns)

# Compute accuracy (accounting for permutation)
from scipy.stats import mode
best_accuracy = 0
best_mapping = {}

import itertools
for perm in itertools.permutations([0, 1, 2]):
    mapping = {i: perm[i] for i in range(3)}
    mapped_predictions = np.array([mapping[s] for s in predicted_states])
    accuracy = np.mean(mapped_predictions == true_regimes)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_mapping = mapping

print(f"\nBest accuracy: {best_accuracy*100:.1f}%")
print(f"State mapping: {best_mapping}")

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot returns
axes[0].plot(returns, alpha=0.7, linewidth=0.5)
axes[0].set_title("Synthetic Returns", fontsize=14, fontweight='bold')
axes[0].set_ylabel("Return")
axes[0].grid(True, alpha=0.3)

# Plot true regimes
for regime in range(3):
    mask = true_regimes == regime
    axes[1].fill_between(np.where(mask)[0], 0, 1, alpha=0.3, label=f'Regime {regime}')
axes[1].set_title("True Regimes", fontsize=14, fontweight='bold')
axes[1].set_ylabel("State")
axes[1].set_ylim(-0.1, 1.1)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# Plot detected regimes
for regime in range(3):
    mask = predicted_states == regime
    axes[2].fill_between(np.where(mask)[0], 0, 1, alpha=0.3, label=f'State {regime}')
axes[2].set_title("Detected States (HMM)", fontsize=14, fontweight='bold')
axes[2].set_xlabel("Time")
axes[2].set_ylabel("State")
axes[2].set_ylim(-0.1, 1.1)
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("hmm_regime_detection.png", dpi=150, bbox_inches='tight')
print("\n✓ Plot saved to: hmm_regime_detection.png")

plt.show()
