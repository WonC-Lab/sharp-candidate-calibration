"""
Exhaustive discrete oracle for small MIQP instances (n=5~8, K=2~3).

Solves:
    J*_MIQP = min_{(y, z) in B_feasible} J*_QP(y, z)
by enumerating all feasible binary patterns (y, z) with rigorous logical pruning,
solving the conditioned continuous QP for each pattern, and returning the global optimum.
"""

from __future__ import annotations

import itertools
import time
from typing import Dict, List, Optional, Tuple
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from .big_m import compute_tight_big_m
from .variables import MarketState, OptimizationResult, PortfolioParameters, SolverConfig


class ExhaustiveDiscreteOracle:
    """
    Exhaustive binary pattern enumerator and conditioned QP solver.
    Provides an independent, certified ground-truth oracle for small MIQP instances.
    """

    def __init__(
        self,
        params: PortfolioParameters,
        solver_config: Optional[SolverConfig] = None,
    ) -> None:
        self.params = params
        self.solver_config = solver_config or SolverConfig()

    def solve(self, state: MarketState) -> Tuple[OptimizationResult, Dict[str, int]]:
        """
        Enumerate all feasible binary vectors (y, z), solve conditioned QPs,
        and return the best solution and search statistics.
        """
        n = state.n_assets
        self.params.validate(n)
        state.validate()

        start_time = time.perf_counter()

        m_sell, m_buy, m_symmetric = compute_tight_big_m(
            current_weights=state.current_weights,
            weight_upper_limits=self.params.weight_upper_limits,
            liquidity_limits=self.params.liquidity_limits,
            max_turnover=self.params.max_turnover,
        )

        ell = self.params.min_holding_weights if self.params.min_holding_weights is not None else np.zeros(n)
        delta = self.params.min_trade_sizes if self.params.min_trade_sizes is not None else np.zeros(n)
        f_cost = self.params.fixed_cost_coeffs if self.params.fixed_cost_coeffs is not None else np.zeros(n)
        k_cardinality = self.params.cardinality_limit if self.params.cardinality_limit is not None else n

        stats = {
            "total_patterns": 0,
            "pruned_patterns": 0,
            "qp_solved": 0,
            "qp_feasible": 0,
            "qp_infeasible": 0,
        }

        best_obj = float("inf")
        best_res: Optional[OptimizationResult] = None
        best_y: Optional[np.ndarray] = None
        best_z: Optional[np.ndarray] = None

        # Enumerate y in {0, 1}^n, z in {0, 1}^n
        for y_tuple in itertools.product([0, 1], repeat=n):
            y_arr = np.array(y_tuple, dtype=int)
            stats["total_patterns"] += (1 << n)

            # Pruning Rule 1: Cardinality limit sum_i y_i <= K
            if np.sum(y_arr) > k_cardinality:
                stats["pruned_patterns"] += (1 << n)
                continue

            for z_tuple in itertools.product([0, 1], repeat=n):
                z_arr = np.array(z_tuple, dtype=int)

                # Pruning Rule 2: Minimum trade turnover sum_{i: z_i=1} delta_i <= tau_max
                if np.sum(delta[z_arr == 1]) > self.params.max_turnover + 1e-8:
                    stats["pruned_patterns"] += 1
                    continue

                # Pruning Rule 3: Fixed cost feasibility sum_i f_i z_i <= 1 - c_min
                if np.sum(f_cost[z_arr == 1]) > (1.0 - self.params.min_cash + 1e-8):
                    stats["pruned_patterns"] += 1
                    continue

                # Pruning Rule 4: Per-asset consistency between y_i, z_i, and w_bar_i
                is_pruned = False
                for i in range(n):
                    w_bar_i = state.current_weights[i]
                    yi = y_arr[i]
                    zi = z_arr[i]

                    if zi == 0:
                        # No trade: w_i must equal w_bar_i
                        if w_bar_i > 1e-7 and yi == 0:
                            is_pruned = True
                            break
                        if w_bar_i < 1e-7 and yi == 1 and ell[i] > 0.0:
                            is_pruned = True
                            break
                        if w_bar_i > 0.0 and yi == 1 and w_bar_i < ell[i] - 1e-8:
                            is_pruned = True
                            break
                        if w_bar_i > self.params.weight_upper_limits[i] + 1e-8:
                            is_pruned = True
                            break
                    else:
                        # Active trade: q_i >= delta_i
                        if yi == 0:
                            # Must liquidate completely: u_i = -w_bar_i
                            if w_bar_i < 1e-7:
                                is_pruned = True
                                break
                            if w_bar_i > self.params.liquidity_limits[i] + 1e-8:
                                is_pruned = True
                                break
                            if w_bar_i < delta[i] - 1e-8:
                                is_pruned = True
                                break
                            if w_bar_i > self.params.max_turnover + 1e-8:
                                is_pruned = True
                                break

                if is_pruned:
                    stats["pruned_patterns"] += 1
                    continue

                # -------------------------------------------------------------
                # Solve Conditioned Convex QP for fixed (y, z)
                # -------------------------------------------------------------
                stats["qp_solved"] += 1
                qp_res = self._solve_conditioned_qp(state, y_arr, z_arr, ell, delta, f_cost, m_symmetric)

                if qp_res.is_feasible and qp_res.objective_value is not None:
                    stats["qp_feasible"] += 1
                    if qp_res.objective_value < best_obj:
                        best_obj = qp_res.objective_value
                        best_res = qp_res
                        best_y = y_arr
                        best_z = z_arr
                else:
                    stats["qp_infeasible"] += 1

        total_time = time.perf_counter() - start_time

        if best_res is not None and best_y is not None and best_z is not None:
            final_res = OptimizationResult(
                status="OPTIMAL",
                is_optimal=True,
                is_feasible=True,
                solve_time=total_time,
                objective_value=best_res.objective_value,
                post_trade_weights=best_res.post_trade_weights,
                trades=best_res.trades,
                absolute_trades=best_res.absolute_trades,
                post_trade_cash=best_res.post_trade_cash,
                transaction_cost=best_res.transaction_cost,
                holding_indicators=best_y,
                trade_indicators=best_z,
                best_bound=best_res.objective_value,
                mip_gap=0.0,
                node_count=stats["qp_solved"],
                metadata={"oracle_stats": stats},
            )
            return final_res, stats
        else:
            final_res = OptimizationResult(
                status="INFEASIBLE",
                is_optimal=False,
                is_feasible=False,
                solve_time=total_time,
                metadata={"oracle_stats": stats},
            )
            return final_res, stats

    def _solve_conditioned_qp(
        self,
        state: MarketState,
        y_arr: np.ndarray,
        z_arr: np.ndarray,
        ell: np.ndarray,
        delta: np.ndarray,
        f_cost: np.ndarray,
        m_symmetric: np.ndarray,
    ) -> OptimizationResult:
        """Solve continuous QP with fixed binary indicators y and z."""
        n = state.n_assets

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()

            with gp.Model("Conditioned_QP", env=env) as model:
                model.Params.TimeLimit = 5.0
                model.Params.Threads = 1

                # Variables with bounds directly conditioned on y and z
                w = {}
                u = {}
                q = {}
                for i in range(n):
                    if y_arr[i] == 0:
                        w[i] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"w_{i}")
                    else:
                        lb_w = ell[i] if ell[i] > 0 else 0.0
                        ub_w = self.params.weight_upper_limits[i]
                        w[i] = model.addVar(lb=lb_w, ub=ub_w, vtype=GRB.CONTINUOUS, name=f"w_{i}")

                    if z_arr[i] == 0:
                        u[i] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"u_{i}")
                        q[i] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"q_{i}")
                    else:
                        u[i] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"u_{i}")
                        lb_q = delta[i] if delta[i] > 0 else 0.0
                        ub_q = min(m_symmetric[i], self.params.liquidity_limits[i])
                        q[i] = model.addVar(lb=lb_q, ub=ub_q, vtype=GRB.CONTINUOUS, name=f"q_{i}")

                c = model.addVar(lb=self.params.min_cash, ub=1.0, vtype=GRB.CONTINUOUS, name="c")
                kappa = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="kappa")

                model.update()

                # Objective
                quad_expr = gp.QuadExpr()
                for i in range(n):
                    for j in range(n):
                        cov_ij = state.covariance_matrix[i, j]
                        if abs(cov_ij) > 1e-12:
                            quad_expr.addTerms(0.5 * self.params.risk_aversion * cov_ij, w[i], w[j])
                for i in range(n):
                    quad_expr.addTerms(-state.expected_returns[i], w[i])
                quad_expr.addTerms(-self.params.risk_free_rate, c)
                quad_expr.addTerms(1.0, kappa)
                model.setObjective(quad_expr, GRB.MINIMIZE)

                # Constraints
                for i in range(n):
                    model.addConstr(w[i] - u[i] == state.current_weights[i])
                    if z_arr[i] == 1:
                        model.addConstr(q[i] - u[i] >= 0.0)
                        model.addConstr(q[i] + u[i] >= 0.0)

                # Budget
                model.addConstr(gp.quicksum(w[i] for i in range(n)) + c + kappa == 1.0)

                # Fixed + Proportional Cost
                fixed_charge = float(np.sum(f_cost[z_arr == 1]))
                model.addConstr(
                    kappa - gp.quicksum(self.params.prop_cost_coeffs[i] * q[i] for i in range(n)) == fixed_charge
                )

                # Turnover
                model.addConstr(gp.quicksum(q[i] for i in range(n)) <= self.params.max_turnover)

                # Sectors
                for sector_name, asset_indices in self.params.sector_indices.items():
                    if sector_name in self.params.sector_bounds:
                        e_min, e_max = self.params.sector_bounds[sector_name]
                        sector_sum = gp.quicksum(w[idx] for idx in asset_indices)
                        model.addConstr(sector_sum >= e_min)
                        model.addConstr(sector_sum <= e_max)

                model.optimize()

                if model.Status == GRB.OPTIMAL:
                    opt_w = np.array([w[i].X for i in range(n)])
                    opt_u = np.array([u[i].X for i in range(n)])
                    opt_q = np.array([q[i].X for i in range(n)])
                    opt_c = float(c.X)
                    opt_kappa = float(kappa.X)
                    return OptimizationResult(
                        status="OPTIMAL",
                        is_optimal=True,
                        is_feasible=True,
                        solve_time=0.0,
                        objective_value=float(model.ObjVal),
                        post_trade_weights=opt_w,
                        trades=opt_u,
                        absolute_trades=opt_q,
                        post_trade_cash=opt_c,
                        transaction_cost=opt_kappa,
                    )
                else:
                    return OptimizationResult(
                        status="INFEASIBLE",
                        is_optimal=False,
                        is_feasible=False,
                        solve_time=0.0,
                    )
