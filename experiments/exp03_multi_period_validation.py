"""
Experiment 03: Multi-Period MIQP Validation against Exhaustive Multi-Period Discrete Oracle.

Validates Gurobi Multi-Period MIQP (H=2, n=2) against an exhaustive enumeration oracle
over all binary patterns (y_{0:H-1}, z_{0:H-1}) in {0, 1}^{2nH}.

Verification Target:
    |J*_MIQP - J*_oracle| <= 1e-8
"""

import itertools
import time
from typing import Dict, Tuple
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from learning_augmented_portfolio.optimization import (
    MultiPeriodMarketState,
    MultiPeriodMIQPOptimizer,
    MultiPeriodOptimizationResult,
    MultiPeriodPortfolioParameters,
    SolverConfig,
    compute_dynamic_tight_big_m,
)


def solve_multi_period_oracle(
    params: MultiPeriodPortfolioParameters,
    state: MultiPeriodMarketState,
) -> Tuple[MultiPeriodOptimizationResult, Dict[str, int]]:
    """
    Exhaustive binary pattern oracle for small multi-period MIQP instances.
    Enumerates all (y, z) trajectories across h=0, ..., H-1 and solves conditioned QPs.
    """
    H = params.horizon
    n = state.n_assets
    total_binary_vars = 2 * n * H

    m_sell, m_buy, m_symmetric = compute_dynamic_tight_big_m(
        initial_holdings=state.initial_holdings,
        weight_upper_limits=params.weight_upper_limits,
        liquidity_limits=params.liquidity_limits,
        max_turnovers=params.max_turnovers,
        gross_returns=state.gross_returns,
    )

    ell = params.min_holding_weights if params.min_holding_weights is not None else np.zeros((H, n))
    delta = params.min_trade_sizes if params.min_trade_sizes is not None else np.zeros((H, n))
    f_cost = params.fixed_cost_coeffs if params.fixed_cost_coeffs is not None else np.zeros((H, n))
    k_card = params.cardinality_limits if params.cardinality_limits is not None else np.full(H, n)

    stats = {
        "total_patterns": 0,
        "pruned_patterns": 0,
        "qp_solved": 0,
        "qp_feasible": 0,
        "qp_infeasible": 0,
    }

    best_obj = float("inf")
    best_sol = None
    best_y = None
    best_z = None

    start_time = time.perf_counter()

    # Enumerate all combinations of y in {0, 1}^{H x n} and z in {0, 1}^{H x n}
    for bit_seq in itertools.product([0, 1], repeat=total_binary_vars):
        stats["total_patterns"] += 1
        arr = np.array(bit_seq, dtype=int)
        y_cand = arr[: n * H].reshape((H, n))
        z_cand = arr[n * H :].reshape((H, n))

        # Pruning Rule 1: Cardinality limit per period
        if np.any(np.sum(y_cand, axis=1) > k_card):
            stats["pruned_patterns"] += 1
            continue

        # Pruning Rule 2: Minimum trade turnover sum_{i: z_hi=1} delta_hi <= tau_h^max
        prune_turnover = False
        for h in range(H):
            if np.sum(delta[h, z_cand[h] == 1]) > params.max_turnovers[h] + 1e-8:
                prune_turnover = True
                break
        if prune_turnover:
            stats["pruned_patterns"] += 1
            continue

        # Solve Conditioned Multi-Period QP with fixed y and z
        stats["qp_solved"] += 1
        qp_obj, qp_data = _solve_conditioned_multi_period_qp(
            params, state, y_cand, z_cand, ell, delta, f_cost, m_symmetric
        )

        if qp_obj is not None:
            stats["qp_feasible"] += 1
            if qp_obj < best_obj:
                best_obj = qp_obj
                best_sol = qp_data
                best_y = y_cand
                best_z = z_cand
        else:
            stats["qp_infeasible"] += 1

    total_time = time.perf_counter() - start_time

    oracle_res = MultiPeriodOptimizationResult(
        status="OPTIMAL" if best_sol is not None else "INFEASIBLE",
        is_optimal=(best_sol is not None),
        is_feasible=(best_sol is not None),
        solve_time=total_time,
        objective_value=best_obj if best_sol is not None else None,
        pre_trade_holdings=best_sol["x_minus"] if best_sol else None,
        post_trade_holdings=best_sol["x"] if best_sol else None,
        trades=best_sol["u"] if best_sol else None,
        absolute_trades=best_sol["q"] if best_sol else None,
        pre_trade_cash=best_sol["b_minus"] if best_sol else None,
        post_trade_cash=best_sol["b"] if best_sol else None,
        transaction_costs=best_sol["kappa"] if best_sol else None,
        holding_indicators=best_y,
        trade_indicators=best_z,
        best_bound=best_obj if best_sol is not None else None,
        mip_gap=0.0,
        node_count=stats["qp_solved"],
        metadata={"oracle_stats": stats},
    )
    return oracle_res, stats


def _solve_conditioned_multi_period_qp(
    params: MultiPeriodPortfolioParameters,
    state: MultiPeriodMarketState,
    y: np.ndarray,
    z: np.ndarray,
    ell: np.ndarray,
    delta: np.ndarray,
    f_cost: np.ndarray,
    m_sym: np.ndarray,
):
    H = params.horizon
    n = state.n_assets
    gross_returns = state.gross_returns if state.gross_returns is not None else (
        np.ones((H - 1, n)) if H > 1 else None
    )

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()

        with gp.Model("Conditioned_MP_QP", env=env) as model:
            model.Params.TimeLimit = 5.0
            model.Params.Threads = 1

            x_minus = model.addVars(H, n, lb=0.0, ub=GRB.INFINITY, name="xm")
            x = {}
            u = {}
            q = {}

            for h in range(H):
                for i in range(n):
                    if y[h, i] == 0:
                        x[h, i] = model.addVar(lb=0.0, ub=0.0, name=f"x_{h}_{i}")
                    else:
                        lb_x = ell[h, i] if ell[h, i] > 0 else 0.0
                        ub_x = params.weight_upper_limits[h, i]
                        x[h, i] = model.addVar(lb=lb_x, ub=ub_x, name=f"x_{h}_{i}")

                    if z[h, i] == 0:
                        u[h, i] = model.addVar(lb=0.0, ub=0.0, name=f"u_{h}_{i}")
                        q[h, i] = model.addVar(lb=0.0, ub=0.0, name=f"q_{h}_{i}")
                    else:
                        u[h, i] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"u_{h}_{i}")
                        lb_q = delta[h, i] if delta[h, i] > 0 else 0.0
                        ub_q = min(m_sym[h, i], params.liquidity_limits[h, i])
                        q[h, i] = model.addVar(lb=lb_q, ub=ub_q, name=f"q_{h}_{i}")

            b_minus = model.addVars(H, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="bm")
            b = model.addVars(H, lb=params.min_cash, ub=GRB.INFINITY, name="b")
            kappa = model.addVars(H, lb=0.0, ub=GRB.INFINITY, name="kap")

            model.update()

            # Objective
            obj = gp.QuadExpr()
            for h in range(H):
                disc = params.discount_factor ** h
                cov_h = state.covariance_matrices[h]
                for i in range(n):
                    for j in range(n):
                        if abs(cov_h[i, j]) > 1e-12:
                            obj.addTerms(disc * 0.5 * params.risk_aversion * cov_h[i, j], x[h, i], x[h, j])
                    obj.addTerms(-disc * state.expected_returns[h, i], x[h, i])
                obj.addTerms(-disc * float(params.risk_free_rates[h]), b[h])
                obj.addTerms(disc, kappa[h])
            model.setObjective(obj, GRB.MINIMIZE)

            # Initial state
            for i in range(n):
                model.addConstr(x_minus[0, i] == float(state.initial_holdings[i]))
            model.addConstr(b_minus[0] == float(state.initial_cash))

            # Intra-period constraints
            for h in range(H):
                for i in range(n):
                    model.addConstr(x[h, i] - u[h, i] == x_minus[h, i])
                    if z[h, i] == 1:
                        model.addConstr(q[h, i] - u[h, i] >= 0.0)
                        model.addConstr(q[h, i] + u[h, i] >= 0.0)

                # Budget
                model.addConstr(
                    gp.quicksum(x[h, i] for i in range(n)) + b[h]
                    == gp.quicksum(x_minus[h, i] for i in range(n)) + b_minus[h] - kappa[h]
                )

                # Fixed + Prop cost
                fixed_h = float(np.sum(f_cost[h, z[h] == 1]))
                model.addConstr(
                    kappa[h] - gp.quicksum(params.prop_cost_coeffs[h, i] * q[h, i] for i in range(n)) == fixed_h
                )

                # Turnover
                model.addConstr(gp.quicksum(q[h, i] for i in range(n)) <= float(params.max_turnovers[h]))

            # Transitions
            if H > 1 and gross_returns is not None:
                for h in range(H - 1):
                    for i in range(n):
                        model.addConstr(x_minus[h + 1, i] == float(gross_returns[h, i]) * x[h, i])
                    rf_fac = 1.0 + float(params.risk_free_rates[h])
                    model.addConstr(b_minus[h + 1] == rf_fac * b[h])

            model.optimize()

            if model.Status == GRB.OPTIMAL:
                sol = {
                    "x_minus": np.array([[x_minus[h, i].X for i in range(n)] for h in range(H)]),
                    "x": np.array([[x[h, i].X for i in range(n)] for h in range(H)]),
                    "u": np.array([[u[h, i].X for i in range(n)] for h in range(H)]),
                    "q": np.array([[q[h, i].X for i in range(n)] for h in range(H)]),
                    "b_minus": np.array([b_minus[h].X for h in range(H)]),
                    "b": np.array([b[h].X for h in range(H)]),
                    "kappa": np.array([kappa[h].X for h in range(H)]),
                }
                return float(model.ObjVal), sol
            else:
                return None, None


def run_multi_period_validation():
    print("=" * 80)
    print("Phase 3 Multi-Period MIQP vs. Exhaustive Oracle Validation (H=2, n=2)")
    print("=" * 80)

    H = 2
    n = 2
    asset_names = ["GrowthStock", "ValueStock"]

    state = MultiPeriodMarketState(
        initial_holdings=np.array([0.15, 0.10]),
        initial_cash=0.75,
        expected_returns=np.array([
            [0.12, 0.06],
            [0.10, 0.05],
        ]),
        covariance_matrices=np.array([
            np.diag([0.04, 0.03]),
            np.diag([0.04, 0.03]),
        ]),
        gross_returns=np.array([
            [1.10, 1.02],  # Growth grows 10%, Value grows 2%
        ]),
        asset_names=asset_names,
    )

    params = MultiPeriodPortfolioParameters(
        risk_aversion=2.0,
        horizon=H,
        risk_free_rates=np.array([0.01, 0.01]),
        max_turnovers=np.array([0.35, 0.35]),
        min_cash=np.array([0.10, 0.10]),
        weight_upper_limits=np.array([
            [0.50, 0.50],
            [0.50, 0.50],
        ]),
        liquidity_limits=np.array([
            [0.25, 0.25],
            [0.25, 0.25],
        ]),
        prop_cost_coeffs=np.array([
            [0.001, 0.001],
            [0.001, 0.001],
        ]),
        discount_factor=0.98,
        cardinality_limits=np.array([2, 2]),
        min_holding_weights=np.array([
            [0.05, 0.05],
            [0.05, 0.05],
        ]),
        min_trade_sizes=np.array([
            [0.02, 0.02],
            [0.02, 0.02],
        ]),
        fixed_cost_coeffs=np.array([
            [0.0003, 0.0003],
            [0.0003, 0.0003],
        ]),
    )

    # 1. Solve via Gurobi Multi-Period MIQP
    print(">>> Solving via Gurobi Multi-Period MIQP Solver...")
    solver = MultiPeriodMIQPOptimizer(params, SolverConfig(output_flag=False))
    res_miqp = solver.solve(state)

    # 2. Solve via Exhaustive Multi-Period Oracle
    print(">>> Solving via Exhaustive Multi-Period Oracle (256 Binary Patterns)...")
    res_oracle, stats = solve_multi_period_oracle(params, state)

    # 3. Comparative Diagnostics
    print("\n" + "=" * 80)
    print("COMPARATIVE SOLUTION ANALYSIS")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Gurobi MP-MIQP':<22} | {'Exhaustive Oracle':<22}")
    print("-" * 80)
    print(f"{'Solver Status':<30} | {res_miqp.status:<22} | {res_oracle.status:<22}")
    print(f"{'Objective Value (Min)':<30} | {res_miqp.objective_value:<22.8f} | {res_oracle.objective_value:<22.8f}")
    print(f"{'Solve Time':<30} | {res_miqp.solve_time * 1000:<19.2f} ms | {res_oracle.solve_time * 1000:<19.2f} ms")
    print(f"{'MIP Gap':<30} | {res_miqp.mip_gap if res_miqp.mip_gap is not None else 0.0:<22.2e} | {'0.0 (Exact)':<22}")
    print(f"{'B&B Node / QP Count':<30} | {res_miqp.node_count if res_miqp.node_count is not None else 0:<22} | {stats['qp_solved']:<22}")

    # Model Size Breakdown
    meta = res_miqp.metadata
    print("\n[Model Structural Complexity]")
    print(f"  Total Decision Variables:  {meta['num_vars']}")
    print(f"  Binary Variables (2*n*H):  {meta['num_bin_vars']} (expected 2*2*2 = 8)")
    print(f"  Continuous Variables:      {meta['num_vars'] - meta['num_bin_vars']}")
    print(f"  Linear Constraints:        {meta['num_constrs']}")

    # Trajectory comparison across horizon
    print("\n[Optimal Multi-Period Allocation Trajectories]")
    for h in range(H):
        print(f"\n--- Horizon Step h = {h} ---")
        print(f"  Pre-trade Holdings (x_minus): {res_miqp.pre_trade_holdings[h]}")
        print(f"  Post-trade Holdings (x*):     {res_miqp.post_trade_holdings[h]}")
        print(f"  Trades (u*):                  {res_miqp.trades[h]}")
        print(f"  Pre-trade Cash (b_minus):     {res_miqp.pre_trade_cash[h]:.6f}")
        print(f"  Post-trade Cash (b*):         {res_miqp.post_trade_cash[h]:.6f}")
        print(f"  Transaction Cost (k*):        {res_miqp.transaction_costs[h]:.6f}")
        print(f"  Holding Indicators (y*):      {res_miqp.holding_indicators[h]}")
        print(f"  Trade Indicators (z*):        {res_miqp.trade_indicators[h]}")

    # Verification Checks
    abs_diff = abs(res_miqp.objective_value - res_oracle.objective_value)
    print("\n[Certification Checks]")
    print(f"  |J*_MIQP - J*_oracle| = {abs_diff:.2e}")
    assert abs_diff <= 1e-8, f"Objective divergence {abs_diff:.2e} > 1e-8"

    np.testing.assert_allclose(res_miqp.post_trade_holdings, res_oracle.post_trade_holdings, atol=1e-6)
    np.testing.assert_allclose(res_miqp.trades, res_oracle.trades, atol=1e-6)
    np.testing.assert_allclose(res_miqp.post_trade_cash, res_oracle.post_trade_cash, atol=1e-6)
    np.testing.assert_allclose(res_miqp.transaction_costs, res_oracle.transaction_costs, atol=1e-6)

    print("\n>>> CERTIFIED: Gurobi Multi-Period MIQP matches Exhaustive Oracle within 1e-8. <<<")
    print("=" * 80)


if __name__ == "__main__":
    run_multi_period_validation()
