"""
Auditable Expected-Return Estimators for Real-Market Portfolio Optimization.
Supports:
  - Historical Rolling Mean
  - Exponentially Weighted Moving Average (EWMA)
  - Simple Cross-Sectional / Time-Series Momentum
Strictly depends on past returns F_t.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
import numpy as np

from ..data.returns import ReturnCalculator


class ReturnEstimatorType(str, Enum):
    HISTORICAL_MEAN = "HISTORICAL_MEAN"
    EWMA = "EWMA"
    MOMENTUM = "MOMENTUM"


class ExpectedReturnEstimator:
    """
    Computes expected-return forecasts mu_t and multi-period horizon forecasts mu_{h|t}.
    """

    def __init__(
        self,
        return_calculator: ReturnCalculator,
        estimator_type: ReturnEstimatorType = ReturnEstimatorType.HISTORICAL_MEAN,
        lookback_window: int = 60,
        ewma_halflife: int = 20,
        annualization_factor: float = 252.0,
    ) -> None:
        self.return_calculator = return_calculator
        self.estimator_type = estimator_type
        self.lookback_window = lookback_window
        self.ewma_halflife = ewma_halflife
        self.annualization_factor = annualization_factor

    def estimate_daily_mean(self, decision_date: str) -> np.ndarray:
        """
        Estimate 1-day ahead expected returns mu_t of shape (N,) using strictly historical data.
        """
        r_mat = self.return_calculator.get_historical_return_window(
            decision_date, self.lookback_window
        )
        W, N = r_mat.shape

        if self.estimator_type == ReturnEstimatorType.HISTORICAL_MEAN:
            return np.mean(r_mat, axis=0)

        elif self.estimator_type == ReturnEstimatorType.EWMA:
            # Exponential decay weights
            decay = 0.5 ** (1.0 / self.ewma_halflife)
            weights = decay ** np.arange(W - 1, -1, -1)
            weights /= np.sum(weights)
            return np.sum(r_mat * weights[:, None], axis=0)

        elif self.estimator_type == ReturnEstimatorType.MOMENTUM:
            # Simple momentum: past 20-day return relative to past 60-day return
            short_w = min(20, W)
            short_ret = np.mean(r_mat[-short_w:], axis=0)
            long_ret = np.mean(r_mat, axis=0)
            return 0.7 * short_ret + 0.3 * long_ret

        else:
            raise ValueError(f"Unknown estimator type: {self.estimator_type}")

    def construct_horizon_forecast(
        self,
        decision_date: str,
        horizon: int,
    ) -> np.ndarray:
        """
        Construct multi-period expected return tensor of shape (H, N).
        Standard real-data benchmark assumption: hold expected returns constant across horizon:
          mu_{h|t} = mu_t for h in {0, ..., H-1}.
        """
        mu_1d = self.estimate_daily_mean(decision_date)
        return np.tile(mu_1d, (horizon, 1))

    def estimate_weekly_mean(self, decision_date: str, days_per_week: int = 5) -> np.ndarray:
        """
        Estimate 5-trading-day ahead expected returns mu_week of shape (N,) using:
          mu_week = (1 + mu_daily)^days_per_week - 1
        """
        mu_daily = self.estimate_daily_mean(decision_date)
        return (1.0 + mu_daily) ** days_per_week - 1.0

    def construct_weekly_horizon_forecast(
        self,
        decision_date: str,
        horizon: int,
        days_per_week: int = 5,
    ) -> np.ndarray:
        """
        Construct multi-period weekly expected return tensor of shape (H, N).
        Tiles conditional weekly forecast across the H-step horizon:
          mu_{h|t} = mu_week for h in {0, ..., H-1}.
        """
        mu_week = self.estimate_weekly_mean(decision_date, days_per_week=days_per_week)
        return np.tile(mu_week, (horizon, 1))

