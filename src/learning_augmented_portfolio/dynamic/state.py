"""
Dynamic portfolio state representation for Rolling-Horizon MPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass(frozen=True)
class PortfolioState:
    """
    Physical portfolio state at actual calendar/rebalancing date t.

    Attributes:
        actual_wealth: V_t > 0, total portfolio wealth in dollars (numeraire base).
        normalized_weights: w_bar_t in R_+^n, pre-trade risky weights (sum_i w_bar_t + c_bar_t == 1.0).
        normalized_cash: c_bar_t in R, pre-trade cash fraction.
        time_index: t >= 0, integer calendar time index.
        asset_names: Optional string labels for the n assets.
    """
    actual_wealth: float
    normalized_weights: np.ndarray
    normalized_cash: float
    time_index: int = 0
    asset_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.actual_wealth <= 0:
            raise ValueError(f"actual_wealth V_t must be strictly positive, got {self.actual_wealth}")
        if np.any(self.normalized_weights < -1e-7):
            raise ValueError(
                f"normalized_weights must be non-negative for long-only portfolios, min={np.min(self.normalized_weights)}"
            )
        total_normalized = np.sum(self.normalized_weights) + self.normalized_cash
        if not np.isclose(total_normalized, 1.0, atol=1e-5):
            raise ValueError(
                f"State normalization identity violated: 1^T w_bar + c_bar = {total_normalized:.6f} != 1.0"
            )

    @property
    def n_assets(self) -> int:
        return len(self.normalized_weights)

    @property
    def dollar_holdings(self) -> np.ndarray:
        """Physical dollar value of risky assets: x_dollar = V_t * w_bar."""
        return self.actual_wealth * self.normalized_weights

    @property
    def dollar_cash(self) -> float:
        """Physical dollar value of cash balance: b_dollar = V_t * c_bar."""
        return self.actual_wealth * self.normalized_cash

    @classmethod
    def from_dollar_amounts(
        cls,
        dollar_holdings: np.ndarray,
        dollar_cash: float,
        time_index: int = 0,
        asset_names: Optional[List[str]] = None,
    ) -> PortfolioState:
        """Construct PortfolioState from raw physical dollar holdings and cash."""
        total_wealth = float(np.sum(dollar_holdings) + dollar_cash)
        if total_wealth <= 0:
            raise ValueError(f"Total physical wealth must be positive, got {total_wealth}")
        normalized_weights = dollar_holdings / total_wealth
        normalized_cash = float(dollar_cash / total_wealth)
        return cls(
            actual_wealth=total_wealth,
            normalized_weights=normalized_weights,
            normalized_cash=normalized_cash,
            time_index=time_index,
            asset_names=asset_names,
        )
