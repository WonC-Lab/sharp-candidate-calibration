"""
Liquidity and volume telemetry computation.
Computes strictly backward-looking Average Daily Volume (ADV) and Dollar ADV (DADV).
Dynamically scales normalized liquidity limits L_{i,t} with actual portfolio wealth V_t:
  L_{i,t} = (alpha_i * DADV_{i,t}) / V_t
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import pandas as pd

from .market_data import MarketDataStore


class LiquidityCalculator:
    """
    Computes rolling ADV and dynamically rescaled liquidity constraints
    without lookahead bias.
    """

    def __init__(
        self,
        data_store: MarketDataStore,
        default_lookback: int = 20,
        default_participation_rate: float = 0.05,  # 5% of daily volume
    ) -> None:
        self.data_store = data_store
        self.default_lookback = default_lookback
        self.default_participation_rate = default_participation_rate
        self.volume_pivot = data_store.volume_pivot
        self.dates = data_store.trading_dates
        self.date_to_idx = data_store.calendar.date_to_index

    def compute_adv(
        self,
        decision_date: str,
        lookback_window: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute rolling ADV in shares over past W trading days up to decision_date (inclusive):
          ADV_{i,t} = (1 / W) * sum_{tau=t-W+1}^t Volume_{i,tau}
        """
        w = lookback_window or self.default_lookback
        t_idx = self.date_to_idx[decision_date]
        if t_idx + 1 < w:
            raise ValueError(
                f"Insufficient volume history for lookback {w} at date {decision_date}."
            )

        start_idx = t_idx - w + 1
        window_vol = self.volume_pivot.iloc[start_idx : t_idx + 1]
        return window_vol.mean(axis=0).to_numpy(dtype=float)

    def compute_dadv(
        self,
        decision_date: str,
        lookback_window: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Dollar ADV:
          DADV_{i,t} = P_{i,t} * ADV_{i,t}
        """
        p = self.data_store.get_prices(decision_date)
        adv = self.compute_adv(decision_date, lookback_window)
        return p * adv

    def compute_normalized_liquidity_limits(
        self,
        decision_date: str,
        portfolio_wealth: float,
        participation_rates: Optional[np.ndarray | float] = None,
        lookback_window: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute dimensionless normalized liquidity trade limits L_{i,t} for optimizer:
          L_{i,t} = (alpha_i * DADV_{i,t}) / V_t
        """
        if portfolio_wealth <= 0.0:
            raise ValueError(f"Portfolio wealth must be positive, got {portfolio_wealth}")

        dadv = self.compute_dadv(decision_date, lookback_window)
        N = len(dadv)

        if participation_rates is None:
            alphas = np.full(N, self.default_participation_rate)
        elif isinstance(participation_rates, (int, float)):
            alphas = np.full(N, float(participation_rates))
        else:
            alphas = np.asarray(participation_rates, dtype=float)

        dollar_limit = alphas * dadv
        norm_limit = dollar_limit / portfolio_wealth

        # Enforce realistic ceiling (e.g. at most 1.0 or 0.5 per period)
        return np.clip(norm_limit, 0.01, 1.0)
