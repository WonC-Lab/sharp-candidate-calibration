"""
Deterministic forecast providers for testing and controlled simulations.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from .base import BaseForecastProvider, ForecastBundle


class ConstantForecastProvider(BaseForecastProvider):
    """Provides a fixed, unchanging forecast bundle at every decision date."""

    def __init__(self, bundle: ForecastBundle) -> None:
        self.bundle = bundle

    def forecast(self, current_time: int, horizon: int) -> ForecastBundle:
        if self.bundle.horizon != horizon:
            raise ValueError(
                f"Requested horizon {horizon} does not match bundle horizon {self.bundle.horizon}"
            )
        return self.bundle


class PredefinedTrajectoryForecastProvider(BaseForecastProvider):
    """Provides predetermined forecast bundles indexed by calendar time t."""

    def __init__(self, schedule: Dict[int, ForecastBundle]) -> None:
        self.schedule = schedule

    def forecast(self, current_time: int, horizon: int) -> ForecastBundle:
        if current_time not in self.schedule:
            raise KeyError(f"No forecast scheduled for calendar time t={current_time}")
        bundle = self.schedule[current_time]
        if bundle.horizon != horizon:
            raise ValueError(
                f"Scheduled bundle horizon {bundle.horizon} != requested horizon {horizon}"
            )
        return bundle


class InstrumentedForecastProvider(BaseForecastProvider):
    """
    Monitoring wrapper that records all queries to audit against lookahead bias.
    """

    def __init__(self, underlying_provider: BaseForecastProvider) -> None:
        self.underlying_provider = underlying_provider
        self.query_log: List[Tuple[int, int]] = []  # List of (current_time, horizon)

    def forecast(self, current_time: int, horizon: int) -> ForecastBundle:
        self.query_log.append((current_time, horizon))
        return self.underlying_provider.forecast(current_time, horizon)

    def verify_no_lookahead(self, max_allowed_time: int) -> bool:
        """Verify that no query was ever made for time t > max_allowed_time."""
        return all(t <= max_allowed_time for t, _ in self.query_log)
