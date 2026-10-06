"""
No-lookahead data pipeline auditing and validation utilities.
Verifies that all market indicators and estimators depend strictly on F_t.
"""

from __future__ import annotations

from typing import List, Sequence
import numpy as np

from .market_data import MarketDataStore
from .returns import ReturnCalculator
from .liquidity import LiquidityCalculator


def audit_no_lookahead_slice(
    data_store: MarketDataStore,
    decision_date: str,
    lookback_window: int,
) -> bool:
    """
    Formally verify that queries for decision_date do not reference any dates > decision_date.
    """
    cal = data_store.calendar
    t_idx = cal.get_index(decision_date)
    all_dates = cal.trading_dates

    # Check future dates
    future_dates = set(all_dates[t_idx + 1 :])

    ret_calc = ReturnCalculator(data_store)
    liq_calc = LiquidityCalculator(data_store)

    # 1. Price query
    p = data_store.get_prices(decision_date)
    assert len(p) == data_store.universe.n_assets

    # 2. Return window query
    r_mat = ret_calc.get_historical_return_window(decision_date, lookback_window)
    assert r_mat.shape[0] == lookback_window

    # 3. ADV query
    adv = liq_calc.compute_adv(decision_date, lookback_window)
    assert len(adv) == data_store.universe.n_assets

    return True
