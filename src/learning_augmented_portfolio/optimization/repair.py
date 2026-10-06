"""
Warm-Start Feasibility Repair module.
Solves a continuous quadratic program conditional on a proposed discrete binary structure (y_hat, z_hat).
Constructs a complete, certified feasible primal starting point (x_tilde, u_tilde, q_tilde, b_tilde, y_hat, z_hat)
for Gurobi MIP warm-starting.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Dict, Any, Tuple
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from .big_m import compute_dynamic_tight_big_m
from .variables import (
    MultiPeriodMarketState,
    MultiPeriodPortfolioParameters,
    MultiPeriodOptimizationResult,
    SolverConfig,
)
from ..recovery.feasibility import FullProblemFeasibilityChecker


@dataclass
class WarmStartRepairResult:
    """Result of continuous QP feasibility repair."""
    is_feasible: bool
    repaired_solution: Optional[MultiPeriodOptimizationResult]
    repair_time: float
    objective_value: Optional[float]
    sanitized_y: np.ndarray
    sanitized_z: np.ndarray


class ContinuousRepairOptimizer:
    """
    Solves continuous QP with discrete binaries fixed to proposed assignments.
    Produces complete Gurobi primal warm starts.
    """

    def __init__(
        self,
        params: MultiPeriodPortfolioParameters,
        solver_config: Optional[SolverConfig] = None,
    ) -> None:
        self.params = params
        self.solver_config = solver_config or SolverConfig(time_limit=5.0)

    def repair(
        self,
        state: MultiPeriodMarketState,
        proposed_y: np.ndarray,
        proposed_z: np.ndarray,
        probabilities_y: Optional[np.ndarray] = None,
    ) -> WarmStartRepairResult:
        """
        Attempt to repair and complete a proposed binary assignment into a full feasible primal solution.

        Args:
            state: MultiPeriodMarketState.
            proposed_y: Binary holding pattern of shape (H, N).
            proposed_z: Binary trade pattern of shape (H, N).
            probabilities_y: Optional confidence probabilities for y of shape (H, N).

        Returns:
            WarmStartRepairResult.
        """
        t0 = time.perf_counter()
        H = self.params.horizon
        N = state.n_assets

        # 1. Sanitize proposed binaries
        clean_y = np.clip(np.round(proposed_y), 0, 1).astype(int)
        clean_z = np.clip(np.round(proposed_z), 0, 1).astype(int)

        # Enforce cardinality per period
        if self.params.cardinality_limits is not None:
            for h in range(H):
                k_lim = int(self.params.cardinality_limits[h])
                active_count = int(np.sum(clean_y[h]))
                if active_count > k_lim:
                    if probabilities_y is not None:
                        # Keep top k_lim probabilities
                        active_indices = np.where(clean_y[h] == 1)[0]
                        scores = probabilities_y[h, active_indices]
                        top_sub = active_indices[np.argsort(-scores)[:k_lim]]
                        new_row = np.zeros(N, dtype=int)
                        new_row[top_sub] = 1
                        clean_y[h] = new_row
                    else:
                        active_indices = np.where(clean_y[h] == 1)[0]
                        new_row = np.zeros(N, dtype=int)
                        new_row[active_indices[:k_lim]] = 1
                        clean_y[h] = new_row

        # Enforce holding/trade logical consistency
        # If clean_y[h, i] == 0 and initial holding was 0 (or h > 0 and clean_y[h-1, i] == 0), then clean_z should be 0 unless traded
        # Build continuous QP with fixed y and z
        gross_returns = state.gross_returns if state.gross_returns is not None else (
            np.ones((H - 1, N)) if H > 1 else None
        )
        m_sell, m_buy, m_symmetric = compute_dynamic_tight_big_m(
            initial_holdings=state.initial_holdings,
            weight_upper_limits=self.params.weight_upper_limits,
            liquidity_limits=self.params.liquidity_limits,
            max_turnovers=self.params.max_turnovers,
            gross_returns=gross_returns,
        )

        ell = self.params.min_holding_weights if self.params.min_holding_weights is not None else np.zeros((H, N))
        delta = self.params.min_trade_sizes if self.params.min_trade_sizes is not None else np.zeros((H, N))
        f_cost = self.params.fixed_cost_coeffs if self.params.fixed_cost_coeffs is not None else np.zeros((H, N))

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()

            with gp.Model("Continuous_Repair_QP", env=env) as model:
                model.Params.TimeLimit = self.solver_config.time_limit
                model.Params.Threads = self.solver_config.threads
                model.Params.FeasibilityTol = self.solver_config.feasibility_tol
                model.Params.OptimalityTol = self.solver_config.optimality_tol

                # Continuous variables
                x_minus = model.addVars(H, N, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_minus")
                x = model.addVars(H, N, lb=0.0, ub=self.params.weight_upper_limits, vtype=GRB.CONTINUOUS, name="x")
                u = model.addVars(H, N, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u")
                q = model.addVars(H, N, lb=0.0, ub=self.params.liquidity_limits, vtype=GRB.CONTINUOUS, name="q")
                b_minus = model.addVars(H, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b_minus")
                b = model.addVars(H, lb=self.params.min_cash, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b")
                kappa = model.addVars(H, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="kappa")

                # Objective
                total_obj = gp.QuadExpr()
                for h in range(H):
                    discount = (self.params.discount_factor ** h)
                    cov_h = state.covariance_matrices[h]
                    for i in range(N):
                        for j in range(N):
                            cov_ij = cov_h[i, j]
                            if abs(cov_ij) > 1e-12:
                                total_obj.addTerms(discount * 0.5 * self.params.risk_aversion * cov_ij, x[h, i], x[h, j])
                        total_obj.addTerms(-discount * state.expected_returns[h, i], x[h, i])
                    total_obj.addTerms(-discount * float(self.params.risk_free_rates[h]), b[h])
                    total_obj.addTerms(discount, kappa[h])
                model.setObjective(total_obj, GRB.MINIMIZE)

                # Initial conditions
                for i in range(N):
                    model.addConstr(x_minus[0, i] == float(state.initial_holdings[i]))
                model.addConstr(b_minus[0] == float(state.initial_cash))

                # Intra-period constraints conditional on clean_y, clean_z
                for h in range(H):
                    for i in range(N):
                        model.addConstr(x[h, i] - u[h, i] == x_minus[h, i])
                        model.addConstr(q[h, i] - u[h, i] >= 0.0)
                        model.addConstr(q[h, i] + u[h, i] >= 0.0)

                    model.addConstr(
                        gp.quicksum(x[h, i] for i in range(N)) + b[h]
                        == gp.quicksum(x_minus[h, i] for i in range(N)) + b_minus[h] - kappa[h]
                    )

                    model.addConstr(
                        kappa[h]
                        >= gp.quicksum(
                            self.params.prop_cost_coeffs[h, i] * q[h, i] + f_cost[h, i] * float(clean_z[h, i])
                            for i in range(N)
                        )
                    )

                    for i in range(N):
                        # Conditional upper and lower holding bounds
                        model.addConstr(x[h, i] <= self.params.weight_upper_limits[h, i] * float(clean_y[h, i]))
                        if ell[h, i] > 0.0 and clean_y[h, i] == 1:
                            model.addConstr(x[h, i] >= ell[h, i])

                        # Conditional trade upper and lower bounds
                        model.addConstr(q[h, i] <= m_symmetric[h, i] * float(clean_z[h, i]))
                        if delta[h, i] > 0.0 and clean_z[h, i] == 1:
                            model.addConstr(q[h, i] >= delta[h, i])

                    model.addConstr(gp.quicksum(q[h, i] for i in range(N)) <= float(self.params.max_turnovers[h]))

                    if self.params.sector_bounds and self.params.sector_indices:
                        for sector_name, asset_indices in self.params.sector_indices.items():
                            if sector_name in self.params.sector_bounds:
                                e_min, e_max = self.params.sector_bounds[sector_name]
                                sector_sum = gp.quicksum(x[h, idx] for idx in asset_indices)
                                model.addConstr(sector_sum >= e_min)
                                model.addConstr(sector_sum <= e_max)

                # Transitions
                if H > 1 and gross_returns is not None:
                    for h in range(H - 1):
                        for i in range(N):
                            model.addConstr(x_minus[h + 1, i] == float(gross_returns[h, i]) * x[h, i])
                        rf_fac = 1.0 + float(self.params.risk_free_rates[h])
                        model.addConstr(b_minus[h + 1] == rf_fac * b[h])

                model.optimize()
                t_repair = time.perf_counter() - t0

                if model.Status == GRB.OPTIMAL:
                    opt_x_minus = np.array([[x_minus[h, i].X for i in range(N)] for h in range(H)])
                    opt_x = np.array([[x[h, i].X for i in range(N)] for h in range(H)])
                    opt_u = np.array([[u[h, i].X for i in range(N)] for h in range(H)])
                    opt_q = np.array([[q[h, i].X for i in range(N)] for h in range(H)])
                    opt_b_minus = np.array([b_minus[h].X for h in range(H)])
                    opt_b = np.array([b[h].X for h in range(H)])
                    opt_kappa = np.array([kappa[h].X for h in range(H)])

                    repaired_sol = MultiPeriodOptimizationResult(
                        status="OPTIMAL",
                        is_optimal=True,
                        is_feasible=True,
                        solve_time=t_repair,
                        objective_value=float(model.ObjVal),
                        pre_trade_holdings=opt_x_minus,
                        post_trade_holdings=opt_x,
                        trades=opt_u,
                        absolute_trades=opt_q,
                        pre_trade_cash=opt_b_minus,
                        post_trade_cash=opt_b,
                        transaction_costs=opt_kappa,
                        holding_indicators=clean_y,
                        trade_indicators=clean_z,
                    )

                    # Validate with independent checker
                    checker = FullProblemFeasibilityChecker(self.params, state)
                    audit = checker.audit_solution(repaired_sol)
                    if audit.is_feasible:
                        return WarmStartRepairResult(
                            is_feasible=True,
                            repaired_solution=repaired_sol,
                            repair_time=t_repair,
                            objective_value=audit.objective_value,
                            sanitized_y=clean_y,
                            sanitized_z=clean_z,
                        )

                return WarmStartRepairResult(
                    is_feasible=False,
                    repaired_solution=None,
                    repair_time=t_repair,
                    objective_value=None,
                    sanitized_y=clean_y,
                    sanitized_z=clean_z,
                )
