"""
Real-Market Backtesting package for Phase 9.
"""

from .records import (
    BacktestStepRecord,
    FinancialMetricsSummary,
    OptimizationMetricsSummary,
    CertificationMetricsSummary,
    compute_financial_metrics,
)
from .baselines import EqualWeightBaseline, BuyAndHoldBaseline
from .audit import BacktestAuditor, BacktestAuditResult
from .engine import RealMarketBacktestEngine, BacktestExecutionResult

__all__ = [
    "BacktestStepRecord",
    "FinancialMetricsSummary",
    "OptimizationMetricsSummary",
    "CertificationMetricsSummary",
    "compute_financial_metrics",
    "EqualWeightBaseline",
    "BuyAndHoldBaseline",
    "BacktestAuditor",
    "BacktestAuditResult",
    "RealMarketBacktestEngine",
    "BacktestExecutionResult",
]
