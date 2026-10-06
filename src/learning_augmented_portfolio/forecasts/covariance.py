"""
Robust Covariance Estimators for Real-Market Portfolio Optimization.
Supports:
  - Ledoit-Wolf Shrinkage (Primary Recommended)
  - Sample Covariance
Strictly guarantees Positive Semi-Definiteness (Sigma_t >= 0).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
import numpy as np
from sklearn.covariance import LedoitWolf

from ..data.returns import ReturnCalculator


class CovarianceEstimatorType(str, Enum):
    LEDOIT_WOLF = "LEDOIT_WOLF"
    SAMPLE = "SAMPLE"


class CovarianceEstimator:
    """
    Computes robust covariance forecasts Sigma_t and multi-period horizon forecasts Sigma_{h|t}.
    """

    def __init__(
        self,
        return_calculator: ReturnCalculator,
        estimator_type: CovarianceEstimatorType = CovarianceEstimatorType.LEDOIT_WOLF,
        lookback_window: int = 60,
        eigenvalue_min: float = 1e-6,
    ) -> None:
        self.return_calculator = return_calculator
        self.estimator_type = estimator_type
        self.lookback_window = lookback_window
        self.eigenvalue_min = eigenvalue_min

    def estimate_daily_covariance(self, decision_date: str) -> np.ndarray:
        """
        Estimate 1-day ahead covariance matrix Sigma_t of shape (N, N) using strictly historical data.
        """
        r_mat = self.return_calculator.get_historical_return_window(
            decision_date, self.lookback_window
        )
        W, N = r_mat.shape

        if self.estimator_type == CovarianceEstimatorType.LEDOIT_WOLF:
            lw = LedoitWolf()
            lw.fit(r_mat)
            sigma = lw.covariance_
        elif self.estimator_type == CovarianceEstimatorType.SAMPLE:
            sigma = np.cov(r_mat, rowvar=False)
        else:
            raise ValueError(f"Unknown estimator type: {self.estimator_type}")

        # Ensure perfect symmetry
        sigma = 0.5 * (sigma + sigma.T)

        # PSD guarantee via eigenvalue floor
        eigvals, eigvecs = np.linalg.eigh(sigma)
        if np.any(eigvals < self.eigenvalue_min):
            eigvals_clipped = np.maximum(eigvals, self.eigenvalue_min)
            sigma = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
            sigma = 0.5 * (sigma + sigma.T)

        return sigma

    def construct_horizon_forecast(
        self,
        decision_date: str,
        horizon: int,
    ) -> np.ndarray:
        """
        Construct multi-period covariance tensor of shape (H, N, N).
        Standard real-data benchmark assumption: hold covariance constant across horizon:
          Sigma_{h|t} = Sigma_t for h in {0, ..., H-1}.
        """
        sigma_1d = self.estimate_daily_covariance(decision_date)
        return np.tile(sigma_1d, (horizon, 1, 1))

    def estimate_weekly_covariance(self, decision_date: str, days_per_week: int = 5) -> np.ndarray:
        """
        Estimate 5-trading-day ahead covariance matrix Sigma_week of shape (N, N) using
        the explicit i.i.d. daily-return aggregation:
          Sigma_week = days_per_week * Sigma_daily
        Ensures positive semi-definiteness via eigenvalue floor.
        """
        sigma_daily = self.estimate_daily_covariance(decision_date)
        sigma_week = float(days_per_week) * sigma_daily
        sigma_week = 0.5 * (sigma_week + sigma_week.T)
        eigvals, eigvecs = np.linalg.eigh(sigma_week)
        if np.any(eigvals < self.eigenvalue_min):
            eigvals_clipped = np.maximum(eigvals, self.eigenvalue_min)
            sigma_week = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
            sigma_week = 0.5 * (sigma_week + sigma_week.T)
        return sigma_week

    def construct_weekly_horizon_forecast(
        self,
        decision_date: str,
        horizon: int,
        days_per_week: int = 5,
    ) -> np.ndarray:
        """
        Construct multi-period weekly covariance tensor of shape (H, N, N).
        Tiles conditional weekly covariance across the H-step horizon:
          Sigma_{h|t} = Sigma_week for h in {0, ..., H-1}.
        """
        sigma_week = self.estimate_weekly_covariance(decision_date, days_per_week=days_per_week)
        return np.tile(sigma_week, (horizon, 1, 1))

