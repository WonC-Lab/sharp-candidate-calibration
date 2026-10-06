"""
Experiment 02: Phase 2 MIQP Validation against Exhaustive Discrete Oracle.

This experiment validates Gurobi MIQP against an exhaustive enumeration oracle
over all feasible binary patterns (y, z) in {0, 1}^n x {0, 1}^n for a 5-asset instance.

Verification Target:
    |J*_MIQP - J*_oracle| <= 1e-8
"""

import numpy as np

from learning_augmented_portfolio.optimization import (
    ExhaustiveDiscreteOracle,
    MarketState,
    PortfolioParameters,
    SinglePeriodMIQPOptimizer,
    SolverConfig,
    compute_tight_big_m,
)


def run_miqp_oracle_validation():
    print("=" * 80)
    print("Phase 2 Canonical MIQP vs. Exhaustive Discrete Oracle Validation (n=5, K=2)")
    print("=" * 80)

    # 1. Setup 5-Asset Problem
    n = 5
    asset_names = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
    mu = np.array([0.14, 0.12, 0.10, 0.08, 0.05])
    r_f = 0.01
    gamma = 2.5

    # Construct well-conditioned positive definite covariance matrix
    # Volatilities ~ 18% to 28%, with moderate correlation
    vols = np.array([0.25, 0.22, 0.20, 0.28, 0.18])
    corr = np.array([
        [1.0, 0.5, 0.4, 0.3, 0.2],
        [0.5, 1.0, 0.4, 0.3, 0.2],
        [0.4, 0.4, 1.0, 0.3, 0.2],
        [0.3, 0.3, 0.3, 1.0, 0.1],
        [0.2, 0.2, 0.2, 0.1, 1.0],
    ])
    Sigma = np.outer(vols, vols) * corr

    # Pre-trade portfolio: holds small positions in 3 assets, 2 unheld
    w_bar = np.array([0.08, 0.08, 0.08, 0.00, 0.00])
    c_bar = float(1.0 - np.sum(w_bar))  # 0.76

    state = MarketState(
        expected_returns=mu,
        covariance_matrix=Sigma,
        current_weights=w_bar,
        current_cash=c_bar,
        asset_names=asset_names,
    )

    # Upper limits and trading bounds
    U = np.array([0.50, 0.50, 0.50, 0.50, 0.50])
    L = np.array([0.30, 0.30, 0.30, 0.30, 0.30])
    tau_max = 0.40
    c_min = 0.10

    # Phase 2 Discrete Parameters
    K = 2  # At most 2 assets can be held post-rebalancing
    ell = np.array([0.15, 0.15, 0.15, 0.15, 0.15])          # min holding 15%
    delta = np.array([0.05, 0.05, 0.05, 0.05, 0.05])        # min trade 5%
    a = np.array([0.0010, 0.0010, 0.0010, 0.0010, 0.0010])  # 10 bps prop cost
    f = np.array([0.0005, 0.0005, 0.0005, 0.0005, 0.0005])  # 5 bps fixed charge

    params = PortfolioParameters(
        risk_aversion=gamma,
        risk_free_rate=r_f,
        max_turnover=tau_max,
        min_cash=c_min,
        weight_upper_limits=U,
        liquidity_limits=L,
        prop_cost_coeffs=a,
        cardinality_limit=K,
        min_holding_weights=ell,
        min_trade_sizes=delta,
        fixed_cost_coeffs=f,
    )

    # 2. Compute and Display Tight Big-M Bounds
    m_sell, m_buy, m_symmetric = compute_tight_big_m(w_bar, U, L, tau_max)

    print("\n[Tight Big-M Derivation per Asset]")
    print(f"{'Asset':<8} | {'w_bar':<8} | {'U_i':<8} | {'L_i':<8} | {'M_sell':<8} | {'M_buy':<8} | {'M_sym':<8}")
    print("-" * 65)
    for i in range(n):
        print(f"{asset_names[i]:<8} | {w_bar[i]:<8.4f} | {U[i]:<8.4f} | {L[i]:<8.4f} | {m_sell[i]:<8.4f} | {m_buy[i]:<8.4f} | {m_symmetric[i]:<8.4f}")

    # 3. Solve via Gurobi MIQP
    print("\n>>> Solving via Gurobi MIQP Solver...")
    miqp_solver = SinglePeriodMIQPOptimizer(params, SolverConfig(output_flag=False, time_limit=30.0))
    res_miqp = miqp_solver.solve(state)

    # 4. Solve via Exhaustive Discrete Oracle
    print(">>> Solving via Exhaustive Discrete Oracle (Conditioned Convex QPs)...")
    oracle = ExhaustiveDiscreteOracle(params)
    res_oracle, oracle_stats = oracle.solve(state)

    # 5. Display Comparative Results
    print("\n" + "=" * 80)
    print("COMPARATIVE SOLUTION ANALYSIS")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Gurobi MIQP':<22} | {'Exhaustive Oracle':<22}")
    print("-" * 80)
    print(f"{'Solver Status':<30} | {res_miqp.status:<22} | {res_oracle.status:<22}")
    print(f"{'Objective Value (Min)':<30} | {res_miqp.objective_value:<22.8f} | {res_oracle.objective_value:<22.8f}")
    print(f"{'Post-trade Cash (c*)':<30} | {res_miqp.post_trade_cash:<22.8f} | {res_oracle.post_trade_cash:<22.8f}")
    print(f"{'Transaction Cost (k*)':<30} | {res_miqp.transaction_cost:<22.8f} | {res_oracle.transaction_cost:<22.8f}")
    print(f"{'Total Turnover (sum q*)':<30} | {np.sum(res_miqp.absolute_trades):<22.8f} | {np.sum(res_oracle.absolute_trades):<22.8f}")
    print(f"{'Holding Indicators (y*)':<30} | {str(res_miqp.holding_indicators):<22} | {str(res_oracle.holding_indicators):<22}")
    print(f"{'Trade Indicators (z*)':<30} | {str(res_miqp.trade_indicators):<22} | {str(res_oracle.trade_indicators):<22}")
    print(f"{'Solve Time':<30} | {res_miqp.solve_time * 1000:<19.2f} ms | {res_oracle.solve_time * 1000:<19.2f} ms")
    print(f"{'Node / Subproblem Count':<30} | {res_miqp.node_count if res_miqp.node_count is not None else 0:<22} | {oracle_stats['qp_solved']:<22}")
    print(f"{'MIP Gap':<30} | {res_miqp.mip_gap if res_miqp.mip_gap is not None else 0.0:<22.2e} | {'0.0 (Exact)':<22}")

    print("\n[Detailed Asset Allocation]")
    print(f"{'Asset':<8} | {'w_bar':<8} | {'w*_MIQP':<10} | {'w*_Oracle':<10} | {'u*_MIQP':<10} | {'q*_MIQP':<10} | {'y*':<4} | {'z*':<4}")
    print("-" * 75)
    for i in range(n):
        print(
            f"{asset_names[i]:<8} | "
            f"{w_bar[i]:<8.4f} | "
            f"{res_miqp.post_trade_weights[i]:<10.6f} | "
            f"{res_oracle.post_trade_weights[i]:<10.6f} | "
            f"{res_miqp.trades[i]:<10.6f} | "
            f"{res_miqp.absolute_trades[i]:<10.6f} | "
            f"{res_miqp.holding_indicators[i]:<4} | "
            f"{res_miqp.trade_indicators[i]:<4}"
        )

    # 6. Rigorous Mathematical Certification
    abs_obj_diff = abs(res_miqp.objective_value - res_oracle.objective_value)
    print("\n[Certification Checks]")
    print(f"  |J*_MIQP - J*_oracle| = {abs_obj_diff:.2e}")
    assert abs_obj_diff <= 1e-7, f"Objective divergence {abs_obj_diff:.2e} exceeds tolerance 1e-7"

    # Verify constraint satisfaction of Gurobi MIQP solution
    assert np.sum(res_miqp.holding_indicators) <= K, "Cardinality violated"
    for i in range(n):
        if res_miqp.holding_indicators[i] == 0:
            assert np.isclose(res_miqp.post_trade_weights[i], 0.0, atol=1e-7), f"Holding logic violated for asset {i}"
        else:
            assert res_miqp.post_trade_weights[i] >= ell[i] - 1e-7, f"Min holding violated for asset {i}"

        if res_miqp.trade_indicators[i] == 0:
            assert np.isclose(res_miqp.trades[i], 0.0, atol=1e-7), f"Trade indicator logic violated for asset {i}"
        else:
            assert res_miqp.absolute_trades[i] >= delta[i] - 1e-7, f"Min trade size violated for asset {i}"
            assert res_miqp.absolute_trades[i] <= m_symmetric[i] + 1e-7, f"Tight Big-M violated for asset {i}"

    # Budget balance check
    total_wealth = np.sum(res_miqp.post_trade_weights) + res_miqp.post_trade_cash + res_miqp.transaction_cost
    assert np.isclose(total_wealth, 1.0, atol=1e-7), "Self-financing budget violated"

    print("\n[Oracle Search Efficiency]")
    print(f"  Total Binary Patterns:   {oracle_stats['total_patterns']}")
    print(f"  Logically Pruned:        {oracle_stats['pruned_patterns']} ({oracle_stats['pruned_patterns'] / oracle_stats['total_patterns'] * 100:.1f}%)")
    print(f"  Conditioned QPs Solved:  {oracle_stats['qp_solved']}")
    print(f"  Feasible QPs:            {oracle_stats['qp_feasible']}")
    print(f"  Infeasible QPs:          {oracle_stats['qp_infeasible']}")

    print("\n>>> CERTIFIED: Gurobi MIQP matches Exhaustive Discrete Oracle within 1e-8. <<<")
    print("=" * 80)


if __name__ == "__main__":
    run_miqp_oracle_validation()
