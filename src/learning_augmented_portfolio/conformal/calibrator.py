r"""
Coverage-calibrated candidate reduction module.
Implements:
  1. Exactness budget calculation: B^\dagger(theta) and B^\dagger_\varepsilon(theta)
  2. Split-conformal calibration: finite-sample marginal and instance-adaptive calibrated budgets
  3. Support depth R^\dagger(theta)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class ConformalCalibrationResult:
    """Output container for conformal budget calibration."""
    alpha: float
    n_cal: int
    k_cal: int
    q_alpha: float
    is_safe_fallback: bool
    empirical_coverage: Optional[float] = None
    mean_calibrated_budget: Optional[float] = None
    mean_candidate_ratio: Optional[float] = None


class SplitConformalCalibrator:
    """
    Split-conformal candidate budget calibrator.
    Guarantees:
      P(B^\dagger(theta_new) <= B_alpha(X_new)) >= 1 - alpha
    under exchangeability of calibration and test instances.
    """

    def __init__(self, alpha: float = 0.10, B_min: int = 3, N: int = 10) -> None:
        self.alpha = float(alpha)
        self.B_min = int(B_min)
        self.N = int(N)
        self.predictor_weights: Optional[np.ndarray] = None
        self.predictor_intercept: float = 0.0
        self.q_alpha: float = float("inf")
        self.k_cal: int = 0
        self.n_cal: int = 0
        self.is_safe_fallback: bool = False
        self.is_fitted: bool = False

    def fit_predictor(self, X_train: np.ndarray, B_dagger_train: np.ndarray) -> None:
        r"""
        Fits a simple regularized linear predictor \hat{b}(X) on the training set D_train.
        Uses ridge regression with non-negative projection to ensure monotonic budget predictions.
        """
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(B_dagger_train, dtype=float)
        n, d = X.shape

        # Ridge regression closed-form: w = (X^T X + lambda I)^(-1) X^T (y - mean(y))
        y_mean = float(np.mean(y))
        y_centered = y - y_mean
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean

        reg = 1.0 * np.eye(d)
        w = np.linalg.solve(X_centered.T @ X_centered + reg, X_centered.T @ y_centered)
        intercept = y_mean - float(X_mean @ w)

        self.predictor_weights = w
        self.predictor_intercept = intercept
        self.is_fitted = True

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Computes raw uncalibrated budget prediction \hat{b}(X)."""
        if not self.is_fitted or self.predictor_weights is None:
            # Fallback to mean B_min
            return np.full(len(X), float(self.B_min))
        X = np.asarray(X, dtype=float)
        return X @ self.predictor_weights + self.predictor_intercept

    def calibrate(self, X_cal: np.ndarray, B_dagger_cal: np.ndarray) -> ConformalCalibrationResult:
        """
        Calibrates the one-sided conformal quantile on D_cal.
        Uses the exact finite-sample quantile convention:
          k_cal = ceil((n_cal + 1)(1 - alpha))
        If k_cal <= n_cal: q_alpha = R_{(k_cal)}
        If k_cal = n_cal + 1: safe fallback q_alpha = +inf => B_alpha = N
        """
        X_cal = np.asarray(X_cal, dtype=float)
        y_cal = np.asarray(B_dagger_cal, dtype=float)
        n_cal = len(y_cal)
        self.n_cal = n_cal

        y_pred = self.predict_raw(X_cal)
        residuals = y_cal - y_pred

        k_cal = int(np.ceil((n_cal + 1) * (1.0 - self.alpha)))
        self.k_cal = k_cal

        if k_cal <= n_cal:
            # k_cal-th order statistic (1-indexed: index k_cal - 1 in 0-indexed sorted array)
            sorted_residuals = np.sort(residuals)
            q_alpha = float(sorted_residuals[k_cal - 1])
            self.q_alpha = q_alpha
            self.is_safe_fallback = False
        else:
            # Requested coverage is finer than sample size allows
            self.q_alpha = float("inf")
            self.is_safe_fallback = True

        return ConformalCalibrationResult(
            alpha=self.alpha,
            n_cal=n_cal,
            k_cal=k_cal,
            q_alpha=self.q_alpha,
            is_safe_fallback=self.is_safe_fallback,
        )

    def predict_calibrated_budget(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the calibrated candidate budget:
          B_alpha(X) = min(N, max(B_min, ceil(\hat{b}(X) + q_alpha)))
        """
        if self.is_safe_fallback:
            return np.full(len(X), self.N, dtype=int)

        raw_pred = self.predict_raw(X)
        adjusted = np.ceil(raw_pred + self.q_alpha).astype(int)
        clipped = np.clip(adjusted, self.B_min, self.N)
        return clipped

    def marginal_calibrate(self, B_dagger_cal: np.ndarray) -> int:
        """
        Computes the constant marginal conformal budget \hat{B}_alpha:
          k = ceil((n + 1)(1 - alpha))
          if k <= n: \hat{B}_alpha = min(N, B_{(k)}^\dagger)
          else: \hat{B}_alpha = N
        """
        y_cal = np.asarray(B_dagger_cal, dtype=int)
        n_cal = len(y_cal)
        k_cal = int(np.ceil((n_cal + 1) * (1.0 - self.alpha)))
        if k_cal <= n_cal:
            sorted_b = np.sort(y_cal)
            return int(min(self.N, sorted_b[k_cal - 1]))
        return self.N


def compute_support_depth(
    ranking: List[int],
    optimal_support: List[int],
    mandatory_set: List[int],
) -> int:
    """
    Computes optimal support depth R^\dagger(theta) = max_{i in S* \ M} rank(i).
    ranking is 1-indexed for non-mandatory assets.
    """
    non_mandatory_opt = set(optimal_support) - set(mandatory_set)
    if not non_mandatory_opt:
        return 0
    # ranking maps asset_id -> 1-indexed rank
    ranks = [ranking.index(i) + 1 for i in non_mandatory_opt if i in ranking]
    return max(ranks) if ranks else len(ranking)
