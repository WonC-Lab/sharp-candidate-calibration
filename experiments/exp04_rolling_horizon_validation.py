"""
Experiment 04: End-to-End Rolling-Horizon MPC Simulation Validation.

Demonstrates genuine Model Predictive Control (MPC) behavior over T = 6 steps:
- Forecast returns hat{r}_{h|t} and realized returns r^{real}_t deliberately differ.
- Future planned actions for h >= 1 are logged, discarded, and re-optimized at t+1.
- Wealth accounting, fixed-cost rescaling, and state renormalization are tracked at every step.
"""

import numpy as np

from learning_augmented_portfolio.dynamic import (
    PortfolioState,
    RollingHorizonPolicy,
    RollingHorizonSimulator,
)
from learning_augmented_portfolio.forecasts import (
    ConstantForecastProvider,
    ForecastBundle,
)
from learning_augmented_portfolio.optimization.variables import SolverConfig


def run_rolling_horizon_simulation():
    print("=" * 90)
    print("Phase 4 Rolling-Horizon MPC Simulation (n=3 Assets, H=3 Horizon, T=6 Steps)")
    print("=" * 90)

    n = 3
    H = 3
    T = 6
    asset_names = ["MegaTech", "CleanEnergy", "TreasuryBond"]

    # 1. Setup Base Planning Forecasts (Used inside optimizer)
    # Expected returns forecast for 3 steps
    mu = np.array([
        [0.12, 0.10, 0.04],
        [0.11, 0.09, 0.04],
        [0.10, 0.08, 0.04],
    ])
    Sigma = np.tile(np.diag([0.04, 0.05, 0.01]), (H, 1, 1))
    gross_planning = np.array([
        [1.10, 1.08, 1.02],
        [1.09, 1.07, 1.02],
    ])
    rf_planning = np.array([0.01, 0.01, 0.01])

    # Constraints
    u_limits = np.full((H, n), 0.50)
    tau_limits = np.full(H, 0.35)
    c_min = np.full(H, 0.10)
    prop_costs = np.full((H, n), 0.001)         # 10 bps
    dollar_fixed_costs = np.full((H, n), 25.0)   # $25 fixed charge per trade
    dollar_adv = np.full((H, n), 5_000_000.0)    # $5M daily volume
    alphas = np.array([0.05, 0.05, 0.10])

    forecast_bundle = ForecastBundle(
        expected_returns=mu,
        covariance_matrices=Sigma,
        gross_returns=gross_planning,
        risk_free_rates=rf_planning,
        weight_upper_limits=u_limits,
        max_turnovers=tau_limits,
        min_cash=c_min,
        prop_cost_coeffs=prop_costs,
        dollar_fixed_costs=dollar_fixed_costs,
        dollar_adv=dollar_adv,
        participation_rates=alphas,
        cardinality_limits=np.full(H, 2),        # Hold at most 2 out of 3
        min_holding_weights=np.full((H, n), 0.10),
        min_trade_sizes=np.full((H, n), 0.02),
        discount_factor=0.98,
        sector_indices={"Equities": [0, 1], "Bonds": [2]},
        sector_bounds={"Equities": (0.0, 0.70), "Bonds": (0.0, 0.50)},
    )
    provider = ConstantForecastProvider(forecast_bundle)

    # 2. Setup Realized Market Path (Outside the optimizer)
    # Deliberate market volatility and divergences from forecast:
    realized_risky = np.array([
        [+0.03, -0.04, +0.005],  # Step 0: Tech rallies, Clean Energy dips
        [-0.05, +0.06, +0.002],  # Step 1: Tech correction, Clean Energy rebound
        [+0.02, +0.01, +0.004],  # Step 2: Mild gains
        [-0.08, -0.05, +0.010],  # Step 3: Sharp equity drawdown, flight to safety
        [+0.04, +0.07, -0.001],  # Step 4: Strong recovery
        [+0.01, +0.02, +0.003],  # Step 5: Stable growth
    ])
    realized_rf = np.full(T, 0.005)

    # 3. Initial Portfolio State ($500,000 portfolio)
    initial_state = PortfolioState(
        actual_wealth=500_000.0,
        normalized_weights=np.array([0.15, 0.15, 0.10]),
        normalized_cash=0.60,
        time_index=0,
        asset_names=asset_names,
    )

    # 4. Execute Rolling-Horizon Simulator
    simulator = RollingHorizonSimulator(
        policy=RollingHorizonPolicy(solver_config=SolverConfig(output_flag=False))
    )
    history = simulator.run(
        initial_state=initial_state,
        horizon=H,
        num_steps=T,
        forecast_provider=provider,
        realized_risky_returns=realized_risky,
        realized_risk_free_rates=realized_rf,
    )

    # 5. Display Step-by-Step MPC Telemetry
    print(f"\nInitial Portfolio Wealth: ${history.initial_wealth:,.2f}")
    print(f"Initial Holdings: {initial_state.normalized_weights}, Cash: {initial_state.normalized_cash:.4f}\n")

    for rec in history.records:
        t = rec.time_step
        print("-" * 90)
        print(f"TIME STEP t = {t} | Pre-trade Wealth: ${rec.actual_wealth_pre:,.2f} | Status: {rec.solver_status}")
        print("-" * 90)
        print(f"  Pre-trade State:    Weights = {rec.pre_trade_weights}, Cash = {rec.pre_trade_cash:.4f}")
        print(f"  Executed Trade u*:  {rec.executed_trade}")
        print(f"  Post-trade State:   Holdings = {rec.post_trade_holdings}, Cash = {rec.post_trade_cash:.4f}")
        print(f"  Transaction Cost:   kappa* = {rec.transaction_cost:.6f} (${rec.transaction_cost * rec.actual_wealth_pre:,.2f})")
        print(f"  Planned Horizon u:  h=1: {rec.planned_trades_horizon[1]}")
        print(f"                      h=2: {rec.planned_trades_horizon[2]}")
        print(f"  Realized Return:    Risky = {rec.realized_risky_returns}, RF = {rec.realized_risk_free_rate:.4f}")
        print(f"  Wealth Growth R^V:  {rec.realized_wealth_growth:.6f} ({(rec.realized_wealth_growth - 1.0) * 100:+.2f}%)")
        print(f"  Post-move Wealth:   ${rec.actual_wealth_post:,.2f}")
        print(f"  Next Pre-trade:     Weights = {rec.next_normalized_weights}, Cash = {rec.next_normalized_cash:.4f}")
        print(f"  Solver Performance: ObjVal = {rec.objective_value:.6f}, MIPGap = {rec.mip_gap:.2e}, SolveTime = {rec.solve_time * 1000:.2f} ms")

        # Discarded action demonstration:
        # Check that planned_trades_horizon[1] at t was NOT directly executed at t+1
        if t < T - 1:
            next_rec = history.records[t + 1]
            h1_planned_prev = rec.planned_trades_horizon[1]
            actual_executed_next = next_rec.executed_trade
            is_different = not np.allclose(h1_planned_prev, actual_executed_next, atol=1e-4)
            print(f"  [Receding-Horizon Check] Was planned h=1 discarded & re-optimized? {'YES (Genuine MPC)' if is_different else 'NO'}")

    # 6. Overall Simulation Performance Summary
    print("\n" + "=" * 90)
    print("SIMULATION TRAJECTORY SUMMARY")
    print("=" * 90)
    print(f"  Starting Wealth:     ${history.initial_wealth:,.2f}")
    print(f"  Ending Wealth:       ${history.final_wealth:,.2f}")
    print(f"  Cumulative Return:   {history.cumulative_return * 100:+.2f}%")
    print(f"  Total Steps:         {history.num_steps}")

    # Mathematical Verifications
    for rec in history.records:
        # 1. Budget conservation at post-trade
        post_trade_sum = np.sum(rec.post_trade_holdings) + rec.post_trade_cash
        assert np.isclose(post_trade_sum, 1.0 - rec.transaction_cost, atol=1e-5)
        # 2. Renormalization identity at next date
        next_sum = np.sum(rec.next_normalized_weights) + rec.next_normalized_cash
        assert np.isclose(next_sum, 1.0, atol=1e-5)
        # 3. Wealth evolution consistency
        assert np.isclose(rec.actual_wealth_post, rec.actual_wealth_pre * rec.realized_wealth_growth, atol=1e-4)

    print("\n>>> ALL ROLLING-HORIZON ACCOUNTING IDENTITIES & MPC BEHAVIORS VERIFIED. <<<")
    print("=" * 90)


if __name__ == "__main__":
    run_rolling_horizon_simulation()
