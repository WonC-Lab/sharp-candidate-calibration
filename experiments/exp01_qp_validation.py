"""
Experiment 01: Validation of Phase 1 Convex QP on Minimal Synthetic Benchmark.

This script executes the canonical 2-asset benchmark defined in the Phase 1 specification,
verifying that Gurobi's solution matches the analytically expected KKT binding conditions.
"""

import numpy as np

from learning_augmented_portfolio.optimization import (
    MarketState,
    PortfolioParameters,
    SinglePeriodQPOptimizer,
    SolverConfig,
)


def run_minimal_benchmark_validation():
    print("=" * 70)
    print("Phase 1 Minimal Synthetic Benchmark Validation")
    print("=" * 70)

    # 1. Problem Setup
    n = 2
    mu = np.array([0.12, 0.04])
    r_f = 0.01
    Sigma = np.array([[0.04, 0.00], [0.00, 0.04]])
    gamma = 2.0

    w_bar = np.array([0.10, 0.10])
    c_bar = 0.80

    state = MarketState(
        expected_returns=mu,
        covariance_matrix=Sigma,
        current_weights=w_bar,
        current_cash=c_bar,
        asset_names=["Asset_1 (High Return)", "Asset_2 (Low Return)"],
    )

    params = PortfolioParameters(
        risk_aversion=gamma,
        risk_free_rate=r_f,
        max_turnover=0.30,
        min_cash=0.10,
        weight_upper_limits=np.array([0.60, 0.60]),
        liquidity_limits=np.array([0.30, 0.30]),
        prop_cost_coeffs=np.array([0.002, 0.002]),  # 20 bps
        sector_indices={"Sector_1": [0], "Sector_2": [1]},
        sector_bounds={"Sector_1": (0.0, 0.50), "Sector_2": (0.0, 0.50)},
    )

    solver_config = SolverConfig(output_flag=True, time_limit=10.0)
    optimizer = SinglePeriodQPOptimizer(params, solver_config=solver_config)

    print("\n[Input State]")
    print(f"  Asset Names:     {state.asset_names}")
    print(f"  Expected Returns: {mu}")
    print(f"  Pre-trade Weights: {w_bar}")
    print(f"  Pre-trade Cash:    {c_bar:.4f}")
    print(f"  Liquidity Bounds:  {params.liquidity_limits}")
    print(f"  Max Turnover:      {params.max_turnover:.4f}")

    # 2. Optimization Solve
    result = optimizer.solve(state)

    print("\n[Optimization Result]")
    print(f"  Status:            {result.status}")
    print(f"  Is Optimal:        {result.is_optimal}")
    print(f"  Solve Time:        {result.solve_time * 1000:.3f} ms")
    print(f"  Min Objective Val: {result.objective_value:.6f}")
    print(f"  Post-trade Weights (w*): {result.post_trade_weights}")
    print(f"  Trades (u*):             {result.trades}")
    print(f"  Absolute Trades (q*):    {result.absolute_trades}")
    print(f"  Post-trade Cash (c*):    {result.post_trade_cash:.6f}")
    print(f"  Transaction Cost (k*):   {result.transaction_cost:.6f}")

    # 3. KKT Analytical Predictions Comparison
    expected_u1 = 0.30
    expected_u2 = 0.00
    expected_w1 = 0.40
    expected_w2 = 0.10
    expected_kappa = 0.0006
    expected_c = 0.4994

    print("\n[Analytical KKT Verification]")
    print(f"  u1*: actual = {result.trades[0]:.6f}, expected = {expected_u1:.6f}")
    print(f"  u2*: actual = {result.trades[1]:.6f}, expected = {expected_u2:.6f}")
    print(f"  w1*: actual = {result.post_trade_weights[0]:.6f}, expected = {expected_w1:.6f}")
    print(f"  w2*: actual = {result.post_trade_weights[1]:.6f}, expected = {expected_w2:.6f}")
    print(f"  c*:  actual = {result.post_trade_cash:.6f}, expected = {expected_c:.6f}")
    print(f"  k*:  actual = {result.transaction_cost:.6f}, expected = {expected_kappa:.6f}")

    # Numerical assertions
    np.testing.assert_allclose(result.trades, [expected_u1, expected_u2], atol=1e-5)
    np.testing.assert_allclose(result.post_trade_weights, [expected_w1, expected_w2], atol=1e-5)
    assert np.isclose(result.post_trade_cash, expected_c, atol=1e-5)
    assert np.isclose(result.transaction_cost, expected_kappa, atol=1e-5)

    # Constraint verifications
    assert np.allclose(result.absolute_trades, np.abs(result.trades), atol=1e-7)
    total_budget = np.sum(result.post_trade_weights) + result.post_trade_cash + result.transaction_cost
    assert np.isclose(total_budget, 1.0, atol=1e-7)

    print("\n>>> ALL MATHEMATICAL & NUMERICAL SANITY CHECKS PASSED SUCCESSFULLY. <<<")
    print("=" * 70)


if __name__ == "__main__":
    run_minimal_benchmark_validation()
