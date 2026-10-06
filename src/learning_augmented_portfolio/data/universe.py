"""
Deterministic asset universe configuration for Phase 9 Real-Market Backtesting.
Universe: Fixed Historical Liquid-Equity Basket (N=10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class AssetInfo:
    """Metadata for an equity asset in the liquid basket."""
    ticker: str
    company_name: str
    sector: str
    industry: str
    base_price: float
    base_adv: float  # Shares per day
    base_spread_bps: float
    description: str


# Fixed 10-asset basket across 6 major sectors
LIQUID_BASKET_10: List[AssetInfo] = [
    AssetInfo("AAPL", "Apple Inc.", "Technology", "Consumer Electronics", 150.0, 80_000_000.0, 1.5, "Mega-cap technology leader"),
    AssetInfo("MSFT", "Microsoft Corp.", "Technology", "Software", 280.0, 30_000_000.0, 1.5, "Cloud and enterprise software"),
    AssetInfo("AMZN", "Amazon.com Inc.", "Consumer", "Broadline Retail", 140.0, 50_000_000.0, 2.0, "E-commerce and cloud provider"),
    AssetInfo("GOOGL", "Alphabet Inc.", "Communication", "Interactive Media", 120.0, 25_000_000.0, 2.0, "Digital advertising and AI"),
    AssetInfo("META", "Meta Platforms Inc.", "Communication", "Interactive Media", 200.0, 30_000_000.0, 2.5, "Social platforms and advertising"),
    AssetInfo("NVDA", "NVIDIA Corp.", "Technology", "Semiconductors", 220.0, 45_000_000.0, 2.5, "Accelerated computing and AI chips"),
    AssetInfo("JPM", "JPMorgan Chase & Co.", "Financials", "Diversified Banks", 140.0, 15_000_000.0, 1.8, "Global investment and commercial banking"),
    AssetInfo("JNJ", "Johnson & Johnson", "Healthcare", "Pharmaceuticals", 160.0, 10_000_000.0, 1.5, "Healthcare and pharmaceuticals"),
    AssetInfo("XOM", "Exxon Mobil Corp.", "Energy", "Integrated Oil & Gas", 90.0, 20_000_000.0, 2.0, "Global energy and petrochemicals"),
    AssetInfo("PG", "Procter & Gamble Co.", "Consumer", "Household Products", 145.0, 8_000_000.0, 1.5, "Consumer staples and household goods"),
]


class RealMarketUniverse:
    """
    Manages deterministic asset universe definitions, sector classifications,
    and survivorship-bias documentation.
    """

    def __init__(self, assets: Optional[List[AssetInfo]] = None) -> None:
        self.assets = assets or LIQUID_BASKET_10
        self.tickers = [a.ticker for a in self.assets]
        self.ticker_to_index = {t: idx for idx, t in enumerate(self.tickers)}
        self.sectors = sorted(list({a.sector for a in self.assets}))
        self.sector_to_index = {s: idx for idx, s in enumerate(self.sectors)}

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    @property
    def n_sectors(self) -> int:
        return len(self.sectors)

    def get_sector_matrix(self) -> np.ndarray:
        """
        Build binary sector indicator matrix M of shape (n_sectors, n_assets)
        such that M_{s, i} = 1 if asset i belongs to sector s.
        """
        M = np.zeros((self.n_sectors, self.n_assets), dtype=float)
        for i, a in enumerate(self.assets):
            s_idx = self.sector_to_index[a.sector]
            M[s_idx, i] = 1.0
        return M

    @property
    def survivorship_bias_limitation(self) -> str:
        """Explicit scientific disclosure regarding survivorship bias."""
        return (
            "LIMITATION DISCLOSURE: The universe is a fixed historical liquid-equity basket "
            "comprising 10 high-liquidity US equities selected based on current market representation. "
            "This configuration does not incorporate point-in-time index constituent additions/deletions. "
            "Consequently, it provides a rigorous testbed for mathematical optimization, solver acceleration, "
            "and liquidity-constrained rebalancing on genuine market series, but should not be interpreted "
            "as an un-survivorship-biased historical backtest of the broader S&P 500 index."
        )
