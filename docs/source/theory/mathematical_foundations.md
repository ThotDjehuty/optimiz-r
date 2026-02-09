# Mathematical Foundations

This section provides formulas used across OptimizR algorithms.

## Differential Evolution
- Mutation and crossover follow classic DE/rand/1 and best/1 strategies.
- See Storn & Price (1997) for full derivations.

## MCMC
- Metropolis-Hastings with Gaussian proposals.
- Acceptance probability: $\alpha = \min\left(1, \frac{\pi(x')q(x\mid x')}{\pi(x)q(x'\mid x)}\right)$.

## HMM
- Baum-Welch (EM) for parameter estimation.
- Viterbi for decoding most likely state sequence.
