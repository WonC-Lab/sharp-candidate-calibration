"""
Real-Market Multi-Period MPC Backtesting Engine.
Reuses the validated Phase 4 rolling-horizon simulator semantics:
  - First-action-only execution
  - Realized return vs forecast return strict isolation
  - Portfolio wealth evolution V_{t+1} = V_t * R^V_{t+1}
  - Dynamic liquidity rescaling L_{i,t} = alpha * DADV_{i,t} / V_t
  - Dynamic transaction cost rescaling f_{i,t} = F_i / V_t
  - Numeraire renormalization: sum(w_bar) + c_bar = 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from ..data.calendar import TradingCalendar
from ..data.liquidity import LiquidityCalculator
from ..data.market_data import MarketDataStore
from ..data.returns import ReturnCalculator
from ..data.universe import RealMarketUniverse
from ..dynamic.state import PortfolioState
from ..forecasts.covariance import CovarianceEstimator
from ..forecasts.expected_returns import ExpectedReturnEstimator
from ..learning.active_set import SupportPredictorModel
from ..optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from ..optimization.reduced_miqp import MultiPeriodReducedMIQPOptimizer
from ..optimization.relaxation import MultiPeriodRelaxedQPOptimizer
from ..optimization.support_restriction import CandidateUniverse
from ..optimization.variables import (
    MultiPeriodMarketState,
    MultiPeriodOptimizationResult,
    MultiPeriodPortfolioParameters,
    SolverConfig,
)
from ..recovery.certificate import CertificateSource
from ..recovery.recovery_loop import NestedRecoveryOptimizer, RecoveryLoopResult
from .baselines import BuyAndHoldBaseline, EqualWeightBaseline
from .records import BacktestStepRecord


@dataclass
class BacktestExecutionResult:
    """Complete output for a backtested strategy."""
    strategy_name: str
    rebalance_dates: List[str]
    wealth_series: np.ndarray
    cash_series: np.ndarray
    step_records: List[BacktestStepRecord]


class RealMarketBacktestEngine:
    """
    Executes chronological real-market multi-period portfolio backtests.
    """

    def __init__(
        self,
        data_store: MarketDataStore,
        params: MultiPeriodPortfolioParameters,
        solver_config: Optional[SolverConfig] = None,
        initial_wealth: float = 1_000_000.0,
        fixed_cost_dollars: float = 10.0,
        proportional_cost_bps: float = 5.0,  # 5 bps
        participation_rate: float = 0.05,
        rebalance_frequency_days: int = 5,
        risk_free_rate_annual: float = 0.03,
        use_weekly_forecasts: bool = True,
    ) -> None:
        self.data_store = data_store
        self.params = params
        self.solver_config = solver_config or SolverConfig(time_limit=15.0, mip_gap=1e-4, threads=1)
        self.initial_wealth = initial_wealth
        self.fixed_cost_dollars = fixed_cost_dollars
        self.proportional_cost = proportional_cost_bps * 1e-4
        self.participation_rate = participation_rate
        self.rebalance_frequency_days = rebalance_frequency_days
        self.risk_free_rate_annual = risk_free_rate_annual
        self.dt_rebalance = rebalance_frequency_days / 252.0
        self.r_f_period = (1.0 + risk_free_rate_annual) ** self.dt_rebalance - 1.0
        self.use_weekly_forecasts = use_weekly_forecasts

        # Analytics helpers
        self.ret_calc = ReturnCalculator(data_store)
        self.liq_calc = LiquidityCalculator(data_store, default_participation_rate=participation_rate)
        self.exp_ret_est = ExpectedReturnEstimator(self.ret_calc)
        self.cov_est = CovarianceEstimator(self.ret_calc)

    def get_step_parameters(self, t_date: str, portfolio_wealth: float) -> MultiPeriodPortfolioParameters:
        """Construct dynamically rescaled MultiPeriodPortfolioParameters for a given step date and wealth."""
        N = self.data_store.universe.n_assets
        H = self.params.horizon
        norm_liq = self.liq_calc.compute_normalized_liquidity_limits(
            t_date, portfolio_wealth=portfolio_wealth, participation_rates=self.participation_rate
        )
        norm_f = np.full(N, self.fixed_cost_dollars / portfolio_wealth)
        norm_a = np.full(N, self.proportional_cost)
        return MultiPeriodPortfolioParameters(
            risk_aversion=self.params.risk_aversion,
            horizon=H,
            risk_free_rates=self.params.risk_free_rates,
            max_turnovers=self.params.max_turnovers,
            min_cash=self.params.min_cash,
            weight_upper_limits=self.params.weight_upper_limits,
            liquidity_limits=np.tile(norm_liq, (H, 1)),
            prop_cost_coeffs=np.tile(norm_a, (H, 1)),
            discount_factor=self.params.discount_factor,
            sector_indices=self.params.sector_indices,
            sector_bounds=self.params.sector_bounds,
            cardinality_limits=self.params.cardinality_limits,
            min_holding_weights=self.params.min_holding_weights,
            min_trade_sizes=self.params.min_trade_sizes,
            fixed_cost_coeffs=np.tile(norm_f, (H, 1)),
        )

    def run_backtest(
        self,
        strategy_name: str,
        start_date: str,
        end_date: str,
        strategy_fn: Callable[[MultiPeriodMarketState, PortfolioState, str], Tuple[np.ndarray, float, Dict[str, Any]]],
    ) -> BacktestExecutionResult:
        """
        Run backtest over specified chronological dates.
        """
        cal = self.data_store.calendar
        rebal_dates = cal.get_rebalancing_dates(
            start_date, end_date, frequency_days=self.rebalance_frequency_days
        )
        if len(rebal_dates) < 2:
            raise ValueError(f"Need at least 2 rebalancing dates, found {len(rebal_dates)}")

        N = self.data_store.universe.n_assets
        curr_wealth = self.initial_wealth
        curr_w = np.zeros(N)
        curr_cash = 1.0  # 100% cash initially

        wealth_history = [curr_wealth]
        cash_history = [curr_cash]
        step_records: List[BacktestStepRecord] = []

        for step_idx in range(len(rebal_dates) - 1):
            t_date = rebal_dates[step_idx]
            t_next_date = rebal_dates[step_idx + 1]

            # 1. State at decision date
            state = PortfolioState(
                actual_wealth=curr_wealth,
                normalized_weights=curr_w,
                normalized_cash=curr_cash,
                time_index=step_idx,
                asset_names=self.data_store.universe.tickers,
            )

            # 2. Rescale liquidity and transaction costs
            norm_liq = self.liq_calc.compute_normalized_liquidity_limits(
                t_date, portfolio_wealth=curr_wealth, participation_rates=self.participation_rate
            )
            norm_f = np.full(N, self.fixed_cost_dollars / curr_wealth)
            norm_a = np.full(N, self.proportional_cost)

            # 3. Forecast expected returns and covariance
            H = self.params.horizon
            if self.use_weekly_forecasts:
                mu_mat = self.exp_ret_est.construct_weekly_horizon_forecast(
                    t_date, H, days_per_week=self.rebalance_frequency_days
                )
                cov_mat = self.cov_est.construct_weekly_horizon_forecast(
                    t_date, H, days_per_week=self.rebalance_frequency_days
                )
            else:
                mu_mat = self.exp_ret_est.construct_horizon_forecast(t_date, H)
                cov_mat = self.cov_est.construct_horizon_forecast(t_date, H)
            gross_mat = 1.0 + mu_mat[:-1]

            mkt_state = MultiPeriodMarketState(
                initial_holdings=curr_w,
                initial_cash=curr_cash,
                expected_returns=mu_mat,
                covariance_matrices=cov_mat,
                gross_returns=gross_mat,
                asset_names=self.data_store.universe.tickers,
            )

            # 4. Invoke Strategy Solver
            target_x0, target_b0, meta = strategy_fn(mkt_state, state, t_date)

            # 5. Execute First Action Trades
            trades = target_x0 - curr_w
            abs_trades = np.abs(trades)
            turnover = float(0.5 * np.sum(abs_trades))
            min_trade_thresh = float(np.min(self.params.min_trade_sizes[0])) if self.params.min_trade_sizes is not None else 0.01
            t_indicators = (abs_trades > min_trade_thresh * 0.5).astype(float)
            step_cost = float(np.sum(norm_a * abs_trades + norm_f * t_indicators))
            wealth_post_cost = curr_wealth * (1.0 - step_cost)

            # 6. Realized Asset Return over rebalancing interval
            price_ret = self.ret_calc.get_realized_multi_day_return(t_date, t_next_date)
            gross_returns = 1.0 + price_ret

            # Portfolio value growth applied to post-cost capital
            period_growth = float(np.sum(target_x0 * gross_returns) + target_b0 * (1.0 + self.r_f_period))
            wealth_next = wealth_post_cost * period_growth

            # Renormalize post-transition weights
            curr_w = (target_x0 * gross_returns) / period_growth
            curr_cash = (target_b0 * (1.0 + self.r_f_period)) / period_growth

            # Record step telemetry
            rec = BacktestStepRecord(
                date=t_date,
                rebalance_index=step_idx,
                portfolio_wealth_pre=curr_wealth,
                portfolio_wealth_post=wealth_post_cost,
                cash_pre=state.normalized_cash,
                cash_post=target_b0,
                pre_trade_weights=state.normalized_weights,
                post_trade_weights=target_x0,
                trades=trades,
                turnover=turnover,
                transaction_cost=step_cost * wealth_history[-1],
                realized_period_return=period_growth - 1.0,
                num_holdings=int(np.sum(target_x0 > 1e-5)),
                num_trades=int(np.sum(t_indicators > 0.5)),
                solve_time_ms=meta.get("solve_time_ms", 0.0),
                node_count=meta.get("node_count", 0),
                work=meta.get("work", 0.0),
                mip_gap=meta.get("mip_gap", 0.0),
                objective_value=meta.get("objective_value"),
                certificate_source=meta.get("certificate_source", "NONE"),
                relaxation_certified=meta.get("relaxation_certified", False),
                full_mip_certified=meta.get("full_mip_certified", False),
                any_valid_certificate=meta.get("any_valid_certificate", False),
                certificate_gap=meta.get("certificate_gap"),
                stages_executed=meta.get("stages_executed", 1),
                final_budget=meta.get("final_budget", N),
                full_fallback_occurred=meta.get("full_fallback_occurred", False),
            )
            step_records.append(rec)
            wealth_history.append(wealth_next)
            cash_history.append(curr_cash)
            curr_wealth = wealth_next

        return BacktestExecutionResult(
            strategy_name=strategy_name,
            rebalance_dates=rebal_dates,
            wealth_series=np.array(wealth_history),
            cash_series=np.array(cash_history),
            step_records=step_records,
        )
