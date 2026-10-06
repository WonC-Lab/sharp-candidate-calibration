"""
Backtest execution records and metric reporting containers.
Strictly separates Financial Quality, Optimization/Solver Quality, and Certification Quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class BacktestStepRecord:
    """Detailed telemetry for a single backtest rebalancing date."""
    date: str
    rebalance_index: int
    portfolio_wealth_pre: float
    portfolio_wealth_post: float
    cash_pre: float
    cash_post: float
    pre_trade_weights: np.ndarray
    post_trade_weights: np.ndarray
    trades: np.ndarray
    turnover: float
    transaction_cost: float
    realized_period_return: float
    num_holdings: int
    num_trades: int
    solve_time_ms: float
    node_count: int
    work: float
    mip_gap: float
    objective_value: Optional[float]
    certificate_source: str             # "FULL_RELAXATION", "FULL_MIP_BOUND", "NONE"
    relaxation_certified: bool
    full_mip_certified: bool
    any_valid_certificate: bool
    certificate_gap: Optional[float]
    stages_executed: int
    final_budget: int
    full_fallback_occurred: bool


@dataclass
class FinancialMetricsSummary:
    """Summary of financial portfolio performance."""
    strategy_name: str
    annualized_return: float                     # Compound Annual Growth Rate (CAGR)
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    total_transaction_cost: float
    mean_turnover: float
    mean_holdings: float
    mean_cash_weight: float
    final_wealth: float
    total_wealth_return: float = 0.0             # R_wealth = V_T / V_0 - 1
    cagr: float = 0.0                            # Realized CAGR = (V_T / V_0)**(252 / N_trading) - 1
    annualized_arithmetic_mean_return: float = 0.0 # Arithmetic periodic mean * annualization factor
    risk_free_rate_used: float = 0.0
    sharpe_convention: str = "annualized_excess_cagr_over_annualized_vol"


@dataclass
class OptimizationMetricsSummary:
    """Summary of solver and acceleration performance."""
    strategy_name: str
    mean_runtime_ms: float
    median_runtime_ms: float
    speedup_vs_full: float
    mean_node_count: float
    mean_work: float
    mean_mip_gap: float
    mean_objective_degradation: float
    exact_recovery_rate: float
    avg_recovery_stages: float
    full_fallback_rate: float


@dataclass
class CertificationMetricsSummary:
    """Summary of solution-quality certification performance under corrected semantics."""
    strategy_name: str
    relaxation_certified_rate: float
    full_mip_certified_rate: float
    any_valid_certificate_rate: float
    mean_relaxation_gap: Optional[float]


def compute_financial_metrics(
    strategy_name: str,
    wealth_series: np.ndarray,
    turnovers: np.ndarray,
    transaction_costs: np.ndarray,
    holdings_counts: np.ndarray,
    cash_weights: np.ndarray,
    annualization_factor: float = 52.0,  # Weekly rebalancing (52 weeks per year)
    risk_free_rate: float = 0.02,        # 2% standard benchmark for Sharpe
    n_trading_days: Optional[int] = None,
) -> FinancialMetricsSummary:
    """
    Compute rigorous financial performance metrics with strict reconciliation between
    realized wealth trajectory and annualized CAGR.
    """
    period_returns = wealth_series[1:] / wealth_series[:-1] - 1.0
    T = len(period_returns)
    v0 = float(wealth_series[0])
    vt = float(wealth_series[-1])
    total_wealth_return = vt / v0 - 1.0

    if T == 0:
        return FinancialMetricsSummary(
            strategy_name=strategy_name,
            annualized_return=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            total_transaction_cost=0.0,
            mean_turnover=0.0,
            mean_holdings=0.0,
            mean_cash_weight=0.0,
            final_wealth=vt,
            total_wealth_return=total_wealth_return,
            cagr=0.0,
            annualized_arithmetic_mean_return=0.0,
            risk_free_rate_used=risk_free_rate,
        )

    # 1. Realized CAGR
    if vt > 0 and v0 > 0:
        if n_trading_days is not None:
            cagr = (vt / v0) ** (252.0 / float(n_trading_days)) - 1.0
        else:
            cagr = (vt / v0) ** (float(annualization_factor) / float(T)) - 1.0
    else:
        cagr = -1.0

    # 2. Arithmetic Periodic Mean Annualized
    mean_ret = float(np.mean(period_returns))
    ann_arithmetic_ret = mean_ret * annualization_factor

    # 3. Volatility
    vol_ret = float(np.std(period_returns, ddof=1)) if T > 1 else 1e-6
    ann_vol = vol_ret * np.sqrt(annualization_factor)

    # 4. Documented Sharpe Convention: (CAGR - R_f) / ann_vol
    excess_ret = cagr - risk_free_rate
    sharpe = excess_ret / ann_vol if ann_vol > 1e-8 else 0.0

    # 5. Max Drawdown
    peaks = np.maximum.accumulate(wealth_series)
    drawdowns = (wealth_series - peaks) / peaks
    max_dd = float(np.min(drawdowns))

    return FinancialMetricsSummary(
        strategy_name=strategy_name,
        annualized_return=cagr,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        total_transaction_cost=float(np.sum(transaction_costs)),
        mean_turnover=float(np.mean(turnovers)),
        mean_holdings=float(np.mean(holdings_counts)),
        mean_cash_weight=float(np.mean(cash_weights)),
        final_wealth=vt,
        total_wealth_return=total_wealth_return,
        cagr=cagr,
        annualized_arithmetic_mean_return=ann_arithmetic_ret,
        risk_free_rate_used=risk_free_rate,
        sharpe_convention="annualized_excess_cagr_over_annualized_vol",
    )
