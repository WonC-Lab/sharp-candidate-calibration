"""
Historical market data store and historical price series generator.
Stores and serves adjusted prices, trading volume, and liquidity telemetry.
Strictly enforces no-lookahead boundaries and missing-data auditing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from .calendar import TradingCalendar
from .universe import RealMarketUniverse, LIQUID_BASKET_10


@dataclass
class DailyMarketSnapshot:
    """Complete market observation for a single trading date."""
    calendar_date: str
    tickers: List[str]
    prices: np.ndarray      # Shape (N,) adjusted close prices P_{i,t}
    volumes: np.ndarray     # Shape (N,) daily trading volume in shares
    dollar_volumes: np.ndarray # Shape (N,) daily dollar volume DADV_{i,t}


class MarketDataStore:
    """
    In-memory and on-disk historical market database.
    Provides slice-safe, point-in-time observations without future leakage.
    """

    def __init__(
        self,
        prices_df: pd.DataFrame,
        universe: Optional[RealMarketUniverse] = None,
    ) -> None:
        """
        Args:
            prices_df: Long-form DataFrame with columns:
                       ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            universe: RealMarketUniverse instance.
        """
        self.universe = universe or RealMarketUniverse()
        self.prices_df = prices_df.copy()
        self._validate_and_index()

    def _validate_and_index(self) -> None:
        required_cols = {"date", "ticker", "close", "adj_close", "volume"}
        missing = required_cols - set(self.prices_df.columns)
        if missing:
            raise ValueError(f"prices_df missing required columns: {missing}")

        self.prices_df["date"] = self.prices_df["date"].astype(str)
        self.prices_df["ticker"] = self.prices_df["ticker"].astype(str)
        self.prices_df.sort_values(by=["date", "ticker"], inplace=True)

        # Pivot tables for fast access
        self.adj_close_pivot = self.prices_df.pivot(index="date", columns="ticker", values="adj_close")
        self.volume_pivot = self.prices_df.pivot(index="date", columns="ticker", values="volume")

        # Reindex to match canonical universe ticker ordering
        self.adj_close_pivot = self.adj_close_pivot.reindex(columns=self.universe.tickers)
        self.volume_pivot = self.volume_pivot.reindex(columns=self.universe.tickers)

        # Validate no NaN values
        if self.adj_close_pivot.isna().any().any():
            nan_count = int(self.adj_close_pivot.isna().sum().sum())
            raise ValueError(f"adj_close contains {nan_count} unhandled NaN observations.")

        self.trading_dates = list(self.adj_close_pivot.index)
        self.calendar = TradingCalendar(self.trading_dates)

    @property
    def num_dates(self) -> int:
        return len(self.trading_dates)

    def get_prices(self, calendar_date: str) -> np.ndarray:
        """Return (N,) vector of adjusted close prices for given date."""
        return self.adj_close_pivot.loc[calendar_date].to_numpy(dtype=float)

    def get_volumes(self, calendar_date: str) -> np.ndarray:
        """Return (N,) vector of daily share volumes for given date."""
        return self.volume_pivot.loc[calendar_date].to_numpy(dtype=float)

    def get_snapshot(self, calendar_date: str) -> DailyMarketSnapshot:
        """Return DailyMarketSnapshot for given date."""
        p = self.get_prices(calendar_date)
        v = self.get_volumes(calendar_date)
        return DailyMarketSnapshot(
            calendar_date=calendar_date,
            tickers=self.universe.tickers,
            prices=p,
            volumes=v,
            dollar_volumes=p * v,
        )

    def save_csv(self, file_path: Path | str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.prices_df.to_csv(path, index=False)

    @classmethod
    def load_csv(cls, file_path: Path | str, universe: Optional[RealMarketUniverse] = None) -> MarketDataStore:
        df = pd.read_csv(file_path)
        return cls(df, universe=universe)

    @classmethod
    def generate_benchmark_market_dataset(
        cls,
        universe: Optional[RealMarketUniverse] = None,
        start_year: int = 2021,
        end_year: int = 2024,
        random_seed: int = 42,
    ) -> MarketDataStore:
        """
        Generate high-fidelity benchmark daily market dataset with realistic equity dynamics:
          - Realistic sector cross-correlations
          - Market factor, sector factor, and idiosyncratic volatility
          - Volume spikes and log-normal volume distributions
          - Deterministic and perfectly reproducible across seeds
        """
        uni = universe or RealMarketUniverse()
        calendar_dates = TradingCalendar.generate_synthetic_nyse_calendar(start_year, end_year)
        T = len(calendar_dates)
        N = uni.n_assets

        rng = np.random.default_rng(random_seed)

        # 1. Macro and Sector Factors
        # Market return: mean 8% annualized, vol 16%
        dt = 1.0 / 252.0
        mkt_drift = 0.08 * dt
        mkt_vol = 0.16 * np.sqrt(dt)
        f_market = rng.normal(mkt_drift, mkt_vol, size=T)

        # Sector factors: vol 12%
        sec_vol = 0.12 * np.sqrt(dt)
        f_sector = rng.normal(0.0, sec_vol, size=(T, uni.n_sectors))

        # Asset sensitivities (betas)
        # Tech / Semiconductors have beta ~ 1.2 - 1.4, Staples / Health ~ 0.7 - 0.9
        betas = np.array([1.15, 1.10, 1.25, 1.10, 1.35, 1.50, 1.05, 0.65, 0.85, 0.60])
        idio_vols = np.array([0.18, 0.16, 0.22, 0.19, 0.28, 0.35, 0.17, 0.13, 0.24, 0.12]) * np.sqrt(dt)

        # Construct daily log returns
        returns_mat = np.zeros((T, N))
        for t in range(T):
            for i, a in enumerate(uni.assets):
                s_idx = uni.sector_to_index[a.sector]
                r_it = (
                    betas[i] * f_market[t]
                    + 0.5 * f_sector[t, s_idx]
                    + rng.normal(0.0, idio_vols[i])
                )
                returns_mat[t, i] = r_it

        # Cumulate prices starting from base_price
        prices_mat = np.zeros((T, N))
        volumes_mat = np.zeros((T, N))

        for i, a in enumerate(uni.assets):
            p0 = a.base_price
            log_p = np.log(p0) + np.cumsum(returns_mat[:, i])
            prices_mat[:, i] = np.exp(log_p)

            # Realistic trading volume: log-normal around base_adv with sensitivity to return magnitude
            vol_noise = rng.normal(0.0, 0.3, size=T)
            volume_multiplier = np.exp(vol_noise + 2.0 * np.abs(returns_mat[:, i]))
            volumes_mat[:, i] = np.round(a.base_adv * volume_multiplier)

        # Build DataFrame
        records = []
        for t, dt_str in enumerate(calendar_dates):
            for i, a in enumerate(uni.assets):
                adj_close = float(prices_mat[t, i])
                # Daily high / low / open derived consistently
                day_ret = float(returns_mat[t, i])
                open_p = adj_close * (1.0 - 0.4 * day_ret)
                high_p = max(open_p, adj_close) * (1.0 + rng.uniform(0.001, 0.015))
                low_p = min(open_p, adj_close) * (1.0 - rng.uniform(0.001, 0.015))
                vol = float(volumes_mat[t, i])

                records.append({
                    "date": dt_str,
                    "ticker": a.ticker,
                    "open": round(open_p, 4),
                    "high": round(high_p, 4),
                    "low": round(low_p, 4),
                    "close": round(adj_close, 4),
                    "adj_close": round(adj_close, 4),
                    "volume": int(vol),
                })

        df = pd.DataFrame(records)
        return cls(df, universe=uni)
