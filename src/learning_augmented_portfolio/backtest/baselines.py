"""
Financial and Optimization Baselines for Real-Market Backtesting:
  1. Equal Weight (1/N)
  2. Buy and Hold
  3. Unconstrained Continuous Convex QP
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from ..optimization.relaxation import MultiPeriodRelaxedQPOptimizer
from ..optimization.variables import (
    MultiPeriodMarketState,
    MultiPeriodOptimizationResult,
    MultiPeriodPortfolioParameters,
    SolverConfig,
)


class EqualWeightBaseline:
    """
    Standard 1/N financial benchmark.
    Rebalances to uniform allocation across all N assets subject to minimum cash reserve.
    """

    def __init__(self, params: MultiPeriodPortfolioParameters) -> None:
        self.params = params

    def get_target_weights(self, n_assets: int) -> Tuple[np.ndarray, float]:
        min_c = float(self.params.min_cash[0]) if hasattr(self.params, "min_cash") and isinstance(self.params.min_cash, np.ndarray) else 0.10
        investable_budget = 1.0 - min_c
        target_w = np.full(n_assets, investable_budget / float(n_assets))
        target_c = min_c
        return target_w, target_c


class BuyAndHoldBaseline:
    """
    Passive buy-and-hold baseline. Initializes once, executes zero trades thereafter.
    """

    def __init__(self, initial_weights: np.ndarray, initial_cash: float) -> None:
        self.initial_weights = initial_weights
        self.initial_cash = initial_cash
        self.has_initialized = False

    def get_action(self, current_weights: np.ndarray, current_cash: float) -> Tuple[np.ndarray, float]:
        if not self.has_initialized:
            self.has_initialized = True
            return self.initial_weights, self.initial_cash
        # Zero trade: keep pre-trade weights
        return current_weights, current_cash
