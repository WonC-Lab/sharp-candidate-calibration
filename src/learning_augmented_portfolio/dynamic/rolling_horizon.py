"""
Rolling-Horizon Policy Engine for MPC execution.
Constructs optimization instances, enforces numeraire rescaling, and extracts first-period decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from ..forecasts.base import ForecastBundle
from ..optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from ..optimization.variables import (
    MultiPeriodMarketState,
    MultiPeriodOptimizationResult,
    MultiPeriodPortfolioParameters,
    SolverConfig,
)
from .state import PortfolioState
from .transitions import rescale_fixed_costs, rescale_liquidity_limits


class OptimizationFailureError(RuntimeError):
    """Raised when the optimization solver fails to produce a certified or feasible incumbent."""
    pass


@dataclass(frozen=True)
class FirstAction:
    """The executed decision corresponding strictly to planning index h = 0."""
    trade: np.ndarray               # u^*_{0|t}
    post_trade_holdings: np.ndarray # x^*_{0|t}
    post_trade_cash: float          # b^*_{0|t}
    absolute_trade: np.ndarray      # q^*_{0|t}
    transaction_cost: float         # kappa^*_{0|t}
    holding_indicator: np.ndarray   # y^*_{0|t}
    trade_indicator: np.ndarray     # z^*_{0|t}


class RollingHorizonPolicy:
    """
    Model Predictive Control (MPC) policy wrapper.
    Formulates and solves the multi-period MIQP at time t, then extracts only the h=0 action.
    """

    def __init__(
        self,
        solver_config: Optional[SolverConfig] = None,
        allow_fallback_on_failure: bool = False,
    ) -> None:
        self.solver_config = solver_config or SolverConfig()
        self.allow_fallback_on_failure = allow_fallback_on_failure

    def plan_and_extract_action(
        self,
        state: PortfolioState,
        forecast: ForecastBundle,
    ) -> tuple[FirstAction, MultiPeriodOptimizationResult]:
        """
        Formulate normalized multi-period problem, solve via Gurobi, and extract h=0 action.

        Args:
            state: Current PortfolioState at calendar date t.
            forecast: ForecastBundle containing predictions for [t, t + H - 1].

        Returns:
            Tuple of (FirstAction, full MultiPeriodOptimizationResult).
        """
        H = forecast.horizon
        n = state.n_assets
        V_t = state.actual_wealth

        # 1. Rescale dollar-denominated constraints into V_t numeraire units
        if forecast.liquidity_limits is not None:
            norm_liquidity = forecast.liquidity_limits
        elif forecast.dollar_adv is not None and forecast.participation_rates is not None:
            norm_liquidity = np.zeros((H, n))
            for h in range(H):
                norm_liquidity[h] = rescale_liquidity_limits(
                    dollar_adv=forecast.dollar_adv[h],
                    participation_rates=forecast.participation_rates,
                    actual_wealth=V_t,
                )
        else:
            raise ValueError("ForecastBundle must specify either liquidity_limits or (dollar_adv, participation_rates)")

        if forecast.fixed_cost_coeffs is not None:
            norm_fixed_costs = forecast.fixed_cost_coeffs
        elif forecast.dollar_fixed_costs is not None:
            norm_fixed_costs = np.zeros((H, n))
            for h in range(H):
                norm_fixed_costs[h] = rescale_fixed_costs(
                    dollar_fixed_costs=forecast.dollar_fixed_costs[h],
                    actual_wealth=V_t,
                )
        else:
            norm_fixed_costs = np.zeros((H, n))

        # 2. Build MultiPeriodPortfolioParameters and MultiPeriodMarketState
        params = MultiPeriodPortfolioParameters(
            risk_aversion=2.0,  # Or configured in policy
            horizon=H,
            risk_free_rates=forecast.risk_free_rates,
            max_turnovers=forecast.max_turnovers,
            min_cash=forecast.min_cash,
            weight_upper_limits=forecast.weight_upper_limits,
            liquidity_limits=norm_liquidity,
            prop_cost_coeffs=forecast.prop_cost_coeffs,
            discount_factor=forecast.discount_factor,
            sector_indices=forecast.sector_indices,
            sector_bounds=forecast.sector_bounds,
            cardinality_limits=forecast.cardinality_limits,
            min_holding_weights=forecast.min_holding_weights,
            min_trade_sizes=forecast.min_trade_sizes,
            fixed_cost_coeffs=norm_fixed_costs,
        )

        market_state = MultiPeriodMarketState(
            initial_holdings=state.normalized_weights,
            initial_cash=state.normalized_cash,
            expected_returns=forecast.expected_returns,
            covariance_matrices=forecast.covariance_matrices,
            gross_returns=forecast.gross_returns,
            asset_names=state.asset_names,
        )

        # 3. Solve Phase 3 MIQP
        optimizer = MultiPeriodMIQPOptimizer(params, self.solver_config)
        result = optimizer.solve(market_state)

        # 4. Enforce Solver Failure Policy
        if not result.is_feasible or result.post_trade_holdings is None:
            if self.allow_fallback_on_failure:
                # Deterministic fallback: execute zero trade (u=0)
                first_action = FirstAction(
                    trade=np.zeros(n),
                    post_trade_holdings=state.normalized_weights.copy(),
                    post_trade_cash=state.normalized_cash,
                    absolute_trade=np.zeros(n),
                    transaction_cost=0.0,
                    holding_indicator=np.where(state.normalized_weights > 1e-6, 1, 0),
                    trade_indicator=np.zeros(n, dtype=int),
                )
                return first_action, result
            else:
                raise OptimizationFailureError(
                    f"Solver failed at time t={state.time_index} with status {result.status}. "
                    "No feasible incumbent available."
                )

        # 5. Extract ONLY the first action (h = 0)
        first_action = FirstAction(
            trade=result.trades[0],
            post_trade_holdings=result.post_trade_holdings[0],
            post_trade_cash=float(result.post_trade_cash[0]),
            absolute_trade=result.absolute_trades[0],
            transaction_cost=float(result.transaction_costs[0]),
            holding_indicator=result.holding_indicators[0],
            trade_indicator=result.trade_indicators[0],
        )

        return first_action, result
