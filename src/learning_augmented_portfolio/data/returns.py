"""
Return calculation and historical realized return accounting.
Strictly separates realized returns r^{real}_{i, t+1} from forecast returns r^{hat}_{i, h|t}.
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import pandas as pd

from .market_data import MarketDataStore


class ReturnCalculator:
    """
    Computes corporate-action-adjusted returns and strictly isolates
    realized transition returns from optimization forecast inputs.
    """

    def __init__(self, data_store: MarketDataStore) -> None:
        self.data_store = data_store
        self.adj_close = data_store.adj_close_pivot
        # Daily arithmetic returns r_{i,t} = (P_{i,t} / P_{i,t-1}) - 1
        self.daily_returns = self.adj_close.pct_change(fill_method=None).dropna()
        self.return_dates = list(self.daily_returns.index)
        self.date_to_idx = {d: idx for idx, d in enumerate(self.return_dates)}

    @property
    def num_return_dates(self) -> int:
        return len(self.return_dates)

    def get_realized_single_day_return(self, date: str) -> np.ndarray:
        """
        Return realized return vector r_{t} = (P_t / P_{t-1}) - 1 of shape (N,).
        Used exclusively in actual portfolio state transitions:
          V_{t+1} = V_t * (1 + sum(w_i * r_{i, t+1}) + c_t * r_f)
        """
        if date not in self.date_to_idx:
            raise KeyError(f"Date {date} has no realized return in calendar.")
        return self.daily_returns.loc[date].to_numpy(dtype=float)

    def get_realized_multi_day_return(
        self,
        start_date: str,
        end_date: str,
    ) -> np.ndarray:
        """
        Return compound realized return between start_date and end_date:
          R_{start, end} = (P_{end} / P_{start}) - 1
        """
        p_start = self.data_store.get_prices(start_date)
        p_end = self.data_store.get_prices(end_date)
        return (p_end / p_start) - 1.0

    def get_historical_return_window(
        self,
        decision_date: str,
        lookback_window: int,
    ) -> np.ndarray:
        """
        Retrieve lookback return matrix of shape (W, N) strictly up to decision_date (inclusive):
          tau in {t - W + 1, ..., t}.
        Strictly forbids future returns (t+1, ...).
        """
        if decision_date not in self.date_to_idx:
            raise KeyError(f"Decision date {decision_date} not found.")

        t_idx = self.date_to_idx[decision_date]
        if t_idx + 1 < lookback_window:
            raise ValueError(
                f"Insufficient historical observations for window {lookback_window} at date {decision_date} (available: {t_idx + 1})."
            )

        start_idx = t_idx - lookback_window + 1
        window_df = self.daily_returns.iloc[start_idx : t_idx + 1]
        return window_df.to_numpy(dtype=float)
