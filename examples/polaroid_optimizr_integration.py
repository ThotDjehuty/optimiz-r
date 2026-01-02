"""
Polaroid + OptimizR Integration Examples
========================================

Demonstrates workflows combining Polaroid's time-series operations with OptimizR's
optimization and statistical inference capabilities.

Workflows:
1. Regime Detection: Polaroid features → OptimizR HMM
2. Strategy Optimization: Polaroid backtesting → OptimizR DE
3. Risk Analysis: Polaroid data processing → OptimizR risk metrics
4. Pairs Trading: Combined feature engineering and parameter optimization

Prerequisites:
- Polaroid gRPC server running (or data files available)
- OptimizR installed with time-series helpers
"""

import numpy as np
import optimizr
from typing import List, Tuple


def workflow1_regime_detection_with_features():
    """
    Workflow 1: Regime Detection with Feature Engineering
    
    Uses OptimizR's time-series helpers (which could integrate with Polaroid's
    lag/diff/pct_change operations) to prepare features for HMM regime detection.
    """
    print("\n" + "=" * 70)
    print("Workflow 1: Regime Detection with Feature Engineering")
    print("=" * 70)
    
    # Simulate price data (in production, this comes from Polaroid)
    np.random.seed(42)
    
    # Generate regime-switching prices
    prices = [100.0]
    regime = 0  # 0=bull, 1=bear, 2=sideways
    for _ in range(200):
        if np.random.random() < 0.05:  # 5% chance of regime switch
            regime = (regime + 1) % 3
        
        if regime == 0:  # Bull
            ret = np.random.normal(0.001, 0.015)
        elif regime == 1:  # Bear
            ret = np.random.normal(-0.001, 0.02)
        else:  # Sideways
            ret = np.random.normal(0, 0.01)
        
        prices.append(prices[-1] * (1 + ret))
    
    # Step 1: Feature engineering with OptimizR helpers
    print("\n1. Feature Engineering:")
    features = optimizr.prepare_for_hmm_py(prices, lag_periods=[1, 2, 3])
    print(f"   Created feature matrix: {len(features)} rows × {len(features[0])} columns")
    print("   Features: returns, log_returns, volatility, lag1, lag2, lag3")
    
    # Step 2: Train HMM for regime detection
    print("\n2. Training HMM (3 regimes):")
    
    # Extract returns for HMM (first column of feature matrix)
    returns = [row[0] for row in features]
    
    # Initialize and train HMM
    hmm = optimizr.HMM(n_states=3)
    hmm.fit(returns, n_iterations=50, tolerance=1e-4)
    
    print(f"   Training complete after {50} iterations")
    print(f"   Log-likelihood: {hmm.log_likelihood(returns):.2f}")
    
    # Step 3: Predict regimes
    print("\n3. Regime Prediction:")
    states = hmm.predict(returns)
    
    # Analyze regime statistics
    unique_states, counts = np.unique(states, return_counts=True)
    print(f"   Detected {len(unique_states)} regimes:")
    for state, count in zip(unique_states, counts):
        pct = count / len(states) * 100
        print(f"   - Regime {state}: {count} periods ({pct:.1f}%)")
    
    # Step 4: Regime characteristics
    print("\n4. Regime Characteristics:")
    for state in unique_states:
        regime_returns = [r for r, s in zip(returns, states) if s == state]
        mean, std, skew, kurt, sharpe = optimizr.return_statistics_py(regime_returns)
        print(f"   Regime {state}:")
        print(f"     Mean return: {mean*100:.3f}% (annualized: {mean*252*100:.1f}%)")
        print(f"     Volatility:  {std*100:.3f}% (annualized: {std*np.sqrt(252)*100:.1f}%)")
        print(f"     Sharpe:      {sharpe:.2f}")
    
    print("\n✅ Workflow 1 complete! Use regimes for regime-switching strategies.")
    return states, returns


def workflow2_strategy_optimization():
    """
    Workflow 2: Strategy Parameter Optimization
    
    Uses Differential Evolution to optimize trading strategy parameters,
    with Polaroid handling data operations and OptimizR handling optimization.
    """
    print("\n" + "=" * 70)
    print("Workflow 2: Moving Average Crossover Strategy Optimization")
    print("=" * 70)
    
    # Simulate OHLC data
    np.random.seed(42)
    n_days = 500
    prices = [100.0]
    for _ in range(n_days - 1):
        ret = np.random.normal(0.0005, 0.02)
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    def moving_average_strategy(params: List[float], prices: np.ndarray) -> float:
        """
        Simulate MA crossover strategy.
        params = [short_window, long_window, stop_loss]
        Returns: negative Sharpe ratio (for minimization)
        """
        short_win = int(params[0])
        long_win = int(params[1])
        stop_loss = params[2]
        
        # Calculate moving averages
        short_ma = np.convolve(prices, np.ones(short_win)/short_win, mode='valid')
        long_ma = np.convolve(prices, np.ones(long_win)/long_win, mode='valid')
        
        # Align arrays
        n = min(len(short_ma), len(long_ma))
        short_ma = short_ma[-n:]
        long_ma = long_ma[-n:]
        aligned_prices = prices[-n:]
        
        # Generate signals
        position = 0
        returns = []
        entry_price = 0
        
        for i in range(1, n):
            if short_ma[i] > long_ma[i] and short_ma[i-1] <= long_ma[i-1]:
                # Buy signal
                position = 1
                entry_price = aligned_prices[i]
            elif short_ma[i] < long_ma[i] and short_ma[i-1] >= long_ma[i-1]:
                # Sell signal
                position = 0
            
            # Stop loss
            if position == 1 and entry_price > 0:
                drawdown = (aligned_prices[i] - entry_price) / entry_price
                if drawdown < -stop_loss:
                    position = 0
            
            # Calculate returns
            if position == 1:
                ret = (aligned_prices[i] - aligned_prices[i-1]) / aligned_prices[i-1]
                returns.append(ret)
            else:
                returns.append(0)
        
        if len(returns) < 10:
            return 999.0  # Penalty for invalid parameters
        
        # Calculate Sharpe ratio
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 999.0
        
        sharpe = mean_ret / std_ret * np.sqrt(252)
        return -sharpe  # Negative for minimization
    
    print("\n1. Setting up optimization:")
    print("   Parameters: [short_window, long_window, stop_loss]")
    print("   Bounds: short=[5, 50], long=[20, 200], stop_loss=[0.02, 0.15]")
    
    # Define objective function for OptimizR
    def objective(x: List[float]) -> float:
        return moving_average_strategy(x, prices)
    
    # Optimize with Differential Evolution
    print("\n2. Running Differential Evolution:")
    result = optimizr.differential_evolution(
        objective,
        bounds=[(5, 50), (20, 200), (0.02, 0.15)],
        strategy="best1",
        max_iterations=50,
        population_size=20,
        convergence_threshold=1e-6
    )
    
    print(f"   Optimization complete!")
    print(f"   Best parameters:")
    print(f"     Short window:  {int(result['x'][0])} days")
    print(f"     Long window:   {int(result['x'][1])} days")
    print(f"     Stop loss:     {result['x'][2]*100:.1f}%")
    print(f"   Best Sharpe ratio: {-result['fun']:.3f}")
    print(f"   Iterations: {result['nit']}")
    
    print("\n✅ Workflow 2 complete! Optimal strategy parameters found.")
    return result


def workflow3_risk_analysis():
    """
    Workflow 3: Comprehensive Risk Analysis
    
    Combines Polaroid's data processing with OptimizR's risk metrics
    for portfolio risk assessment.
    """
    print("\n" + "=" * 70)
    print("Workflow 3: Portfolio Risk Analysis")
    print("=" * 70)
    
    # Simulate multi-asset portfolio returns
    np.random.seed(42)
    n_days = 252
    n_assets = 3
    
    print("\n1. Simulating 3-asset portfolio (1 year daily data):")
    
    # Generate correlated returns
    corr_matrix = np.array([
        [1.0, 0.6, 0.3],
        [0.6, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ])
    
    # Cholesky decomposition for correlation
    L = np.linalg.cholesky(corr_matrix)
    uncorrelated = np.random.randn(n_days, n_assets) * 0.015
    returns = uncorrelated @ L.T
    
    # Add drift
    returns[:, 0] += 0.0008  # Asset 1: 20% annual
    returns[:, 1] += 0.0004  # Asset 2: 10% annual
    returns[:, 2] += 0.0006  # Asset 3: 15% annual
    
    print(f"   Asset 1: Expected 20% annual return")
    print(f"   Asset 2: Expected 10% annual return")
    print(f"   Asset 3: Expected 15% annual return")
    
    # Step 2: Individual asset statistics
    print("\n2. Individual Asset Analysis:")
    for i in range(n_assets):
        mean, std, skew, kurt, sharpe = optimizr.return_statistics_py(
            returns[:, i].tolist()
        )
        print(f"\n   Asset {i+1}:")
        print(f"     Return (annual):    {mean*252*100:.1f}%")
        print(f"     Volatility (annual): {std*np.sqrt(252)*100:.1f}%")
        print(f"     Skewness:           {skew:.3f}")
        print(f"     Kurtosis:           {kurt:.3f}")
        print(f"     Sharpe ratio:       {sharpe:.3f}")
    
    # Step 3: Mean-reversion analysis
    print("\n3. Mean-Reversion Analysis:")
    prices = [np.cumprod(1 + returns[:, i]) * 100 for i in range(n_assets)]
    
    for i in range(n_assets):
        hurst = optimizr.rolling_hurst_exponent_py(
            returns[:, i].tolist(), 
            window_size=60
        )
        avg_hurst = np.mean(hurst)
        
        half_life = optimizr.rolling_half_life_py(
            prices[i].tolist(),
            window_size=60
        )
        # Filter out infinities
        finite_hl = [hl for hl in half_life if np.isfinite(hl)]
        avg_hl = np.mean(finite_hl) if finite_hl else float('inf')
        
        print(f"\n   Asset {i+1}:")
        print(f"     Hurst exponent: {avg_hurst:.3f}", end="")
        if avg_hurst < 0.45:
            print(" (mean-reverting)")
        elif avg_hurst > 0.55:
            print(" (trending)")
        else:
            print(" (random walk)")
        
        if np.isfinite(avg_hl):
            print(f"     Half-life:      {avg_hl:.1f} days")
    
    # Step 4: Correlation analysis
    print("\n4. Correlation Matrix (rolling 60-day):")
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            corr = optimizr.rolling_correlation_py(
                returns[:, i].tolist(),
                returns[:, j].tolist(),
                window_size=60
            )
            avg_corr = np.mean(corr)
            print(f"   Asset {i+1} ↔ Asset {j+1}: {avg_corr:.3f}")
    
    # Step 5: Portfolio optimization weights (equal risk contribution)
    print("\n5. Portfolio Construction:")
    weights = [1/n_assets] * n_assets
    portfolio_returns = returns @ np.array(weights)
    
    mean, std, skew, kurt, sharpe = optimizr.return_statistics_py(
        portfolio_returns.tolist()
    )
    
    print(f"   Equal-weight portfolio:")
    print(f"     Return (annual):    {mean*252*100:.1f}%")
    print(f"     Volatility (annual): {std*np.sqrt(252)*100:.1f}%")
    print(f"     Sharpe ratio:       {sharpe:.3f}")
    
    print("\n✅ Workflow 3 complete! Comprehensive risk analysis finished.")


def workflow4_pairs_trading_pipeline():
    """
    Workflow 4: Complete Pairs Trading Pipeline
    
    End-to-end pairs trading: cointegration check, parameter optimization,
    and risk management using OptimizR's integrated tools.
    """
    print("\n" + "=" * 70)
    print("Workflow 4: Pairs Trading Pipeline")
    print("=" * 70)
    
    # Generate cointegrated pair
    np.random.seed(42)
    n_days = 500
    
    # Asset 1: Random walk with drift
    returns1 = np.random.normal(0.0003, 0.015, n_days)
    prices1 = 100 * np.cumprod(1 + returns1)
    
    # Asset 2: Cointegrated with Asset 1
    spread_noise = np.random.normal(0, 0.01, n_days)
    prices2 = prices1 * 0.9 + np.cumsum(spread_noise)
    
    # Calculate spread
    spread = prices1 - prices2
    
    print("\n1. Cointegration Analysis:")
    
    # Check mean-reversion
    spread_returns = np.diff(spread) / spread[:-1]
    hurst = optimizr.rolling_hurst_exponent_py(
        spread_returns.tolist(),
        window_size=60
    )
    avg_hurst = np.mean(hurst)
    print(f"   Hurst exponent: {avg_hurst:.3f}", end="")
    
    if avg_hurst < 0.5:
        print(" ✅ Mean-reverting (good for pairs trading)")
    else:
        print(" ⚠️ Not clearly mean-reverting")
    
    # Estimate half-life
    half_lives = optimizr.rolling_half_life_py(
        spread.tolist(),
        window_size=60
    )
    finite_hl = [hl for hl in half_lives if np.isfinite(hl) and hl > 0]
    avg_hl = np.mean(finite_hl) if finite_hl else float('inf')
    
    if np.isfinite(avg_hl):
        print(f"   Half-life: {avg_hl:.1f} days (reversion speed)")
    
    # Correlation check
    returns2 = np.diff(prices2) / prices2[:-1]
    corr = optimizr.rolling_correlation_py(
        returns1[1:].tolist(),
        returns2.tolist(),
        window_size=60
    )
    avg_corr = np.mean(corr)
    print(f"   Correlation: {avg_corr:.3f}", end="")
    
    if avg_corr > 0.7:
        print(" ✅ Strong correlation")
    elif avg_corr > 0.5:
        print(" ⚠️ Moderate correlation")
    else:
        print(" ❌ Weak correlation")
    
    # Step 2: Optimize strategy parameters
    print("\n2. Strategy Parameter Optimization:")
    
    def pairs_strategy(params: List[float]) -> float:
        """
        Pairs trading with mean-reversion.
        params = [entry_z, exit_z, stop_loss]
        Returns: negative Sharpe (for minimization)
        """
        entry_z = params[0]
        exit_z = params[1]
        stop_loss = params[2]
        
        # Calculate z-score
        window = 20
        spread_ma = np.convolve(spread, np.ones(window)/window, mode='valid')
        spread_std = np.array([
            np.std(spread[i:i+window]) 
            for i in range(len(spread) - window + 1)
        ])
        
        aligned_spread = spread[window-1:]
        z_score = (aligned_spread - spread_ma) / (spread_std + 1e-6)
        
        # Trading logic
        position = 0  # 1 = long spread, -1 = short spread
        returns = []
        entry_value = 0
        
        for i in range(1, len(z_score)):
            # Entry signals
            if z_score[i] > entry_z and position == 0:
                position = -1  # Short spread (short asset1, long asset2)
                entry_value = aligned_spread[i]
            elif z_score[i] < -entry_z and position == 0:
                position = 1  # Long spread (long asset1, short asset2)
                entry_value = aligned_spread[i]
            
            # Exit signals
            if abs(z_score[i]) < exit_z and position != 0:
                position = 0
            
            # Stop loss
            if position != 0 and entry_value != 0:
                pnl = position * (aligned_spread[i] - entry_value) / abs(entry_value)
                if pnl < -stop_loss:
                    position = 0
            
            # Calculate returns
            if position != 0:
                spread_ret = (aligned_spread[i] - aligned_spread[i-1]) / aligned_spread[i-1]
                returns.append(position * spread_ret)
            else:
                returns.append(0)
        
        if len(returns) < 10:
            return 999.0
        
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 999.0
        
        sharpe = mean_ret / std_ret * np.sqrt(252)
        return -sharpe
    
    print("   Optimizing: [entry_z, exit_z, stop_loss]")
    
    result = optimizr.differential_evolution(
        pairs_strategy,
        bounds=[(1.5, 3.0), (0.1, 1.0), (0.02, 0.1)],
        strategy="best1",
        max_iterations=30,
        population_size=15
    )
    
    print(f"   Optimal parameters:")
    print(f"     Entry z-score:  {result['x'][0]:.2f}")
    print(f"     Exit z-score:   {result['x'][1]:.2f}")
    print(f"     Stop loss:      {result['x'][2]*100:.1f}%")
    print(f"   Expected Sharpe: {-result['fun']:.3f}")
    
    print("\n✅ Workflow 4 complete! Pairs trading strategy optimized.")
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("Polaroid + OptimizR Integration Examples")
    print("=" * 70)
    print("\nDemonstrating 4 integrated workflows combining time-series")
    print("operations with optimization and statistical inference.")
    
    # Run all workflows
    workflow1_regime_detection_with_features()
    workflow2_strategy_optimization()
    workflow3_risk_analysis()
    workflow4_pairs_trading_pipeline()
    
    print("\n" + "=" * 70)
    print("✅ All integration workflows completed successfully!")
    print("=" * 70)
    print("\nThese examples show how to combine:")
    print("  • Polaroid's time-series operations (lag, diff, pct_change)")
    print("  • OptimizR's optimization (DE, grid search)")
    print("  • OptimizR's inference (HMM, MCMC)")
    print("  • OptimizR's time-series helpers (Hurst, half-life, etc.)")
    print("\nFor production use, connect to Polaroid gRPC for data processing.")
    print("=" * 70)
