"""
Trading calendar and rebalancing schedule management.
Enforces strict chronological sequence and configurable rebalancing frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Sequence
import pandas as pd


class TradingCalendar:
    """
    Manages US equity market trading dates and configurable rebalancing schedules.
    """

    def __init__(self, trading_dates: Sequence[str]) -> None:
        """
        Args:
            trading_dates: Sorted list of ISO format date strings ('YYYY-MM-DD').
        """
        self.trading_dates = sorted(list(trading_dates))
        self.date_to_index = {d: idx for idx, d in enumerate(self.trading_dates)}

    @property
    def num_trading_days(self) -> int:
        return len(self.trading_dates)

    def get_index(self, calendar_date: str) -> int:
        if calendar_date not in self.date_to_index:
            raise KeyError(f"Date {calendar_date} is not a valid trading day in calendar.")
        return self.date_to_index[calendar_date]

    def get_rebalancing_dates(
        self,
        start_date: str,
        end_date: str,
        frequency_days: int = 5,
    ) -> List[str]:
        """
        Generate rebalancing decision dates spaced every `frequency_days` trading days.

        Args:
            start_date: Inclusive start date.
            end_date: Inclusive end date.
            frequency_days: Number of trading days between rebalances (e.g., 5 for weekly).

        Returns:
            List of rebalancing dates.
        """
        valid_dates = [d for d in self.trading_dates if start_date <= d <= end_date]
        if not valid_dates:
            return []

        rebal_dates = [valid_dates[i] for i in range(0, len(valid_dates), frequency_days)]
        return rebal_dates

    @staticmethod
    def generate_synthetic_nyse_calendar(
        start_year: int = 2021,
        end_year: int = 2024,
    ) -> List[str]:
        """
        Generate approximate NYSE trading days (excluding weekends and major US holidays).
        """
        start = date(start_year, 1, 1)
        end = date(end_year, 12, 31)
        cur = start
        trading_days: List[str] = []

        # Standard holidays (month, day)
        fixed_holidays = {
            (1, 1),    # New Year's Day
            (7, 4),    # Independence Day
            (12, 25),  # Christmas
        }

        while cur <= end:
            # Skip Saturday (5) and Sunday (6)
            if cur.weekday() < 5:
                if (cur.month, cur.day) not in fixed_holidays:
                    trading_days.append(cur.isoformat())
            cur += timedelta(days=1)

        return trading_days
