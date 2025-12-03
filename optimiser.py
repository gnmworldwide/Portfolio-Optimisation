#!/usr/bin/env python3
"""
Portfolio Optimisation (Mean-Variance via Monte Carlo)
------------------------------------------------------

A minimal Markowitz-style portfolio optimiser:

- Takes historical returns for N assets.
- Generates many random portfolios (long-only, fully invested).
- Finds the portfolio with the highest Sharpe ratio.

This is intentionally simple and self-contained for CV/GitHub purposes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class PortfolioResult:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float


def annualise_stats(daily_returns: np.ndarray, risk_free_rate: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert daily asset returns into annualised mean return & covariance matrix.
    Assumes 252 trading days per year.
    """
    mean_daily = daily_returns.mean(axis=0)
    cov_daily = np.cov(daily_returns, rowvar=False)

    mean_ann = (1 + mean_daily) ** 252 - 1
    cov_ann = cov_daily * 252

    return mean_ann, cov_ann


def random_long_only_weights(n_assets: int) -> np.ndarray:
    """
    Generate random long-only weights that sum to 1.
    """
    w = np.random.rand(n_assets)
    return w / w.sum()


def evaluate_portfolio(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                       risk_free_rate: float = 0.0) -> PortfolioResult:
    """
    Compute expected return, volatility and Sharpe ratio for a given weight vector.
    """
    portfolio_return = float(weights @ mean_returns)
    portfolio_var = float(weights @ cov_matrix @ weights.T)
    portfolio_vol = math.sqrt(portfolio_var) if portfolio_var > 0 else 0.0

    if portfolio_vol == 0:
        sharpe = 0.0
    else:
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol

    return PortfolioResult(
        weights=weights,
        expected_return=portfolio_return,
        volatility=portfolio_vol,
        sharpe_ratio=sharpe,
    )


def max_sharpe_portfolio(
    daily_returns: np.ndarray,
    risk_free_rate: float = 0.0,
    n_portfolios: int = 10_000,
) -> PortfolioResult:
    """
    Brute-force Monte Carlo search for the portfolio with maximum Sharpe ratio.
    """
    mean_ann, cov_ann = annualise_stats(daily_returns, risk_free_rate)
    n_assets = daily_returns.shape[1]

    best_result: PortfolioResult | None = None

    for _ in range(n_portfolios):
        w = random_long_only_weights(n_assets)
        result = evaluate_portfolio(w, mean_ann, cov_ann, risk_free_rate)
        if best_result is None or result.sharpe_ratio > best_result.sharpe_ratio:
            best_result = result

    assert best_result is not None
    return best_result


def demo_synthetic_data(
    n_days: int = 1000,
    n_assets: int = 4,
) -> np.ndarray:
    """
    Create synthetic daily returns for N assets, each with slightly different
    mean/vol profiles. This keeps the project self-contained.
    """
    np.random.seed(42)
    means = np.linspace(0.0002, 0.0008, n_assets)  # ~ 5% to 22% annualised
    vols = np.linspace(0.01, 0.025, n_assets)      # different risk levels

    returns = []
    for i in range(n_assets):
        asset_returns = np.random.normal(loc=means[i], scale=vols[i], size=n_days)
        returns.append(asset_returns)

    return np.column_stack(returns)


def main() -> None:
    # Generate demo data (e.g. 4 synthetic assets)
    daily_returns = demo_synthetic_data()

    best = max_sharpe_portfolio(daily_returns, risk_free_rate=0.0, n_portfolios=5000)

    print("=== Portfolio Optimisation (Monte Carlo Mean-Variance) ===")
    print(f"Number of assets : {len(best.weights)}")
    print(f"Expected return  : {best.expected_return * 100:.2f}% per year")
    print(f"Volatility       : {best.volatility * 100:.2f}% per year")
    print(f"Sharpe ratio     : {best.sharpe_ratio:.2f}")
    print("\nWeights (long-only, sum to 1.0):")
    for i, w in enumerate(best.weights):
        print(f"  Asset {i+1}: {w:.3f}")


if __name__ == "__main__":
    main()
