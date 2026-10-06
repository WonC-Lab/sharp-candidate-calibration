"""
Rolling-Horizon Simulator for Model Predictive Control (MPC) portfolio rebalancing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from ..forecasts.base import BaseForecastProvider, ForecastBundle
from ..optimization.variables import MultiPeriodOptimizationResult, SolverConfig
from .rolling_horizon import FirstAction, RollingHorizonPolicy
from .state import PortfolioState
from .transitions import (
    realized_market_transition,
    renormalize_state,
    validate_first_action_execution,
)


@dataclass
class SimulatorStepRecord:
    """Complete diagnostic record of a single rolling-horizon MPC decision step."""
    time_step: int
    actual_wealth_pre: float
    pre_trade_weights: np.ndarray
    pre_trade_cash: float
    forecast_bundle: ForecastBundle
    planned_trades_horizon: np.ndarray
    executed_trade: np.ndarray
    post_trade_holdings: np.ndarray
    post_trade_cash: float
    transaction_cost: float
    realized_risky_returns: np.ndarray
    realized_risk_free_rate: float
    realized_wealth_growth: float
    actual_wealth_post: float
    next_normalized_weights: np.ndarray
    next_normalized_cash: float
    solver_status: str
    objective_value: Optional[float]
    best_bound: Optional[float]
    mip_gap: Optional[float]
    node_count: Optional[int]
    solve_time: float


@dataclass
class SimulationHistory:
    """Historical container for full simulation trajectory and summary statistics."""
    records: List[SimulatorStepRecord] = field(default_factory=list)

    @property
    def num_steps(self) -> int:
        return len(self.records)

    @property
    def initial_wealth(self) -> float:
        return self.records[0].actual_wealth_pre if self.records else 0.0

    @property
    def final_wealth(self) -> float:
        return self.records[-1].actual_wealth_post if self.records else 0.0

    @property
    def cumulative_return(self) -> float:
        if not self.records or self.initial_wealth == 0:
            return 0.0
        return (self.final_wealth / self.initial_wealth) - 1.0


class RollingHorizonSimulator:
    """
    Simulation engine executing the receding-horizon loop.
    Enforces strict information boundaries, execution isolation, and wealth accounting.
    """

    def __init__(
        self,
        policy: Optional[RollingHorizonPolicy] = None,
        solver_config: Optional[SolverConfig] = None,
    ) -> None:
        self.policy = policy or RollingHorizonPolicy(solver_config=solver_config)

    def run(
        self,
        initial_state: PortfolioState,
        horizon: int,
        num_steps: int,
        forecast_provider: BaseForecastProvider,
        realized_risky_returns: np.ndarray,      # shape (num_steps, n)
        realized_risk_free_rates: np.ndarray,    # shape (num_steps,)
    ) -> SimulationHistory:
        """
        Run the rolling-horizon simulation for num_steps periods.

        Args:
            initial_state: State at calendar time t = 0.
            horizon: MPC planning horizon H.
            num_steps: Total number of execution steps T.
            forecast_provider: Provider yielding ForecastBundle for given (t, H).
            realized_risky_returns: Matrix of actual realized returns, shape (num_steps, n).
            realized_risk_free_rates: Vector of actual realized risk-free rates, shape (num_steps,).

        Returns:
            SimulationHistory containing all step records.
        """
        n = initial_state.n_assets
        if realized_risky_returns.shape != (num_steps, n):
            raise ValueError(
                f"realized_risky_returns shape {realized_risky_returns.shape} != ({num_steps}, {n})"
            )
        if realized_risk_free_rates.shape != (num_steps,):
            raise ValueError(
                f"realized_risk_free_rates shape {realized_risk_free_rates.shape} != ({num_steps},)"
            )

        current_state = initial_state
        history = SimulationHistory()

        for t in range(num_steps):
            # 1. Obtain information-bounded forecast bundle for horizon [t, t + H - 1]
            forecast = forecast_provider.forecast(current_time=t, horizon=horizon)

            # 2. Optimize multi-period problem and extract ONLY the h=0 action
            first_action, opt_result = self.policy.plan_and_extract_action(
                state=current_state,
                forecast=forecast,
            )

            # 3. Validate post-trade execution budget identity
            validate_first_action_execution(
                post_trade_holdings=first_action.post_trade_holdings,
                post_trade_cash=first_action.post_trade_cash,
                transaction_cost=first_action.transaction_cost,
            )

            # 4. Observe actual realized market move between t and t+1
            r_real_risky = realized_risky_returns[t]
            r_real_rf = float(realized_risk_free_rates[t])

            # 5. Propagate physical holdings through realized returns (in V_t units)
            x_tilde_minus, b_tilde_minus, wealth_growth = realized_market_transition(
                post_trade_holdings=first_action.post_trade_holdings,
                post_trade_cash=first_action.post_trade_cash,
                realized_risky_returns=r_real_risky,
                realized_risk_free_rate=r_real_rf,
            )

            # 6. Renormalize state for the next decision date t+1 (V_{t+1} numeraire)
            next_state = renormalize_state(
                x_tilde_minus=x_tilde_minus,
                b_tilde_minus=b_tilde_minus,
                wealth_growth_factor=wealth_growth,
                current_wealth=current_state.actual_wealth,
                current_time_index=t,
            )

            # 7. Record complete step telemetry
            step_record = SimulatorStepRecord(
                time_step=t,
                actual_wealth_pre=current_state.actual_wealth,
                pre_trade_weights=current_state.normalized_weights.copy(),
                pre_trade_cash=current_state.normalized_cash,
                forecast_bundle=forecast,
                planned_trades_horizon=opt_result.trades if opt_result.trades is not None else np.zeros((horizon, n)),
                executed_trade=first_action.trade.copy(),
                post_trade_holdings=first_action.post_trade_holdings.copy(),
                post_trade_cash=first_action.post_trade_cash,
                transaction_cost=first_action.transaction_cost,
                realized_risky_returns=r_real_risky.copy(),
                realized_risk_free_rate=r_real_rf,
                realized_wealth_growth=wealth_growth,
                actual_wealth_post=next_state.actual_wealth,
                next_normalized_weights=next_state.normalized_weights.copy(),
                next_normalized_cash=next_state.normalized_cash,
                solver_status=opt_result.status,
                objective_value=opt_result.objective_value,
                best_bound=opt_result.best_bound,
                mip_gap=opt_result.mip_gap,
                node_count=opt_result.node_count,
                solve_time=opt_result.solve_time,
            )
            history.records.append(step_record)

            # Advance state
            current_state = next_state

        return history
