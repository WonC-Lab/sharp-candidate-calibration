"""
Abstract interfaces and data structures for market forecasting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class ForecastBundle:
    """
    Bundle of planning parameters and forecasts over horizon h = 0, ..., H-1.
    All inputs represent information known strictly at or before decision time t.
    """
    expected_returns: np.ndarray       # shape (H, n)
    covariance_matrices: np.ndarray    # shape (H, n, n)
    risk_free_rates: np.ndarray        # shape (H,)
    weight_upper_limits: np.ndarray    # shape (H, n)
    max_turnovers: np.ndarray          # shape (H,)
    min_cash: np.ndarray               # shape (H,)
    prop_cost_coeffs: np.ndarray       # shape (H, n)
    gross_returns: Optional[np.ndarray] = None          # shape (H-1, n)
    dollar_adv: Optional[np.ndarray] = None             # shape (H, n) in dollars
    participation_rates: Optional[np.ndarray] = None    # shape (n,)
    liquidity_limits: Optional[np.ndarray] = None       # shape (H, n) normalized
    dollar_fixed_costs: Optional[np.ndarray] = None     # shape (H, n) in dollars
    fixed_cost_coeffs: Optional[np.ndarray] = None      # shape (H, n) normalized
    cardinality_limits: Optional[np.ndarray] = None     # shape (H,)
    min_holding_weights: Optional[np.ndarray] = None    # shape (H, n)
    min_trade_sizes: Optional[np.ndarray] = None        # shape (H, n)
    discount_factor: float = 1.0
    sector_indices: Dict[str, List[int]] = field(default_factory=dict)
    sector_bounds: Dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return len(self.expected_returns)

    @property
    def n_assets(self) -> int:
        return self.expected_returns.shape[1]


class BaseForecastProvider(ABC):
    """Abstract base class for information-bounded forecast providers."""

    @abstractmethod
    def forecast(self, current_time: int, horizon: int) -> ForecastBundle:
        """
        Generate forecast bundle for the planning horizon [current_time, current_time + horizon - 1].
        Must NOT access any information from calendar dates t' > current_time.
        """
        pass
