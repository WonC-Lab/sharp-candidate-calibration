"""
Phase 1: Convex Single-Period Portfolio Optimization Formulation.
"""

from __future__ import annotations

import time
from typing import Optional
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from .variables import MarketState, OptimizationResult, PortfolioParameters, SolverConfig


class SinglePeriodQPOptimizer:
    """
    Builder and solver for the canonical Phase 1 Convex Quadratic Program (QP).

    Mathematical Formulation (Minimization Form):
        min_{w, u, q, c, kappa}
            (gamma / 2) * w^T Sigma w - mu^T w - r_f * c + kappa
        s.t.
            w_i - u_i = w_bar_i,          forall i = 1, ..., n  (Trade Definition)
            sum_i w_i + c + kappa = 1,                          (Self-Financing Budget)
            kappa - sum_i a_i q_i = 0,                          (Transaction Cost Accounting)
            q_i - u_i >= 0,               forall i = 1, ..., n  (Linearization +)
            q_i + u_i >= 0,               forall i = 1, ..., n  (Linearization -)
            sum_i q_i <= tau_max,                               (Turnover Limit)
            E_g^min <= sum_{i in S_g} w_i <= E_g^max,           (Sector Exposures)
            0 <= w_i <= U_i,              forall i = 1, ..., n  (Holding Limits)
            0 <= q_i <= L_i,              forall i = 1, ..., n  (Liquidity Limits)
            c >= c_min                                          (Minimum Cash)
    """

    def __init__(
        self,
        params: PortfolioParameters,
        solver_config: Optional[SolverConfig] = None,
    ) -> None:
        self.params = params
        self.solver_config = solver_config or SolverConfig()

    def solve(self, state: MarketState) -> OptimizationResult:
        """
        Construct and solve the Phase 1 Convex QP for the given market state.

        Args:
            state: MarketState containing mu, Sigma, w_bar, c_bar.

        Returns:
            OptimizationResult containing optimal decision variables and diagnostic info.
        """
        # Validate inputs
        n = state.n_assets
        self.params.validate(n)
        state.validate()

        start_time = time.perf_counter()

        with gp.Env(empty=True) as env:
            if not self.solver_config.output_flag:
                env.setParam("OutputFlag", 0)
            env.start()

            with gp.Model("Phase1_Convex_QP", env=env) as model:
                # Apply solver configuration
                model.Params.TimeLimit = self.solver_config.time_limit
                model.Params.Threads = self.solver_config.threads
                model.Params.FeasibilityTol = self.solver_config.feasibility_tol
                model.Params.OptimalityTol = self.solver_config.optimality_tol

                # -------------------------------------------------------------
                # 1. Decision Variables
                # -------------------------------------------------------------
                # Post-trade risky weights: 0 <= w_i <= U_i
                w = model.addVars(
                    n,
                    lb=0.0,
                    ub=self.params.weight_upper_limits,
                    vtype=GRB.CONTINUOUS,
                    name="w",
                )

                # Signed trades: u_i in R (unbounded below and above)
                u = model.addVars(
                    n,
                    lb=-GRB.INFINITY,
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name="u",
                )

                # Absolute trades: 0 <= q_i <= L_i
                q = model.addVars(
                    n,
                    lb=0.0,
                    ub=self.params.liquidity_limits,
                    vtype=GRB.CONTINUOUS,
                    name="q",
                )

                # Post-trade cash: c_min <= c <= 1.0
                c = model.addVar(
                    lb=self.params.min_cash,
                    ub=1.0,
                    vtype=GRB.CONTINUOUS,
                    name="c",
                )

                # Total proportional transaction cost: kappa >= 0
                kappa = model.addVar(
                    lb=0.0,
                    ub=1.0,
                    vtype=GRB.CONTINUOUS,
                    name="kappa",
                )

                model.update()

                # -------------------------------------------------------------
                # 2. Objective Function (Minimization)
                # -------------------------------------------------------------
                # (gamma / 2) * w^T Sigma w - mu^T w - r_f * c + kappa
                quad_expr = gp.QuadExpr()
                # Quadratic risk term: 0.5 * gamma * sum_{i,j} Sigma_{i,j} * w_i * w_j
                for i in range(n):
                    for j in range(n):
                        cov_ij = state.covariance_matrix[i, j]
                        if abs(cov_ij) > 1e-12:
                            quad_expr.addTerms(0.5 * self.params.risk_aversion * cov_ij, w[i], w[j])

                # Linear expected return & risk-free return terms
                for i in range(n):
                    quad_expr.addTerms(-state.expected_returns[i], w[i])
                quad_expr.addTerms(-self.params.risk_free_rate, c)

                # Transaction cost term
                quad_expr.addTerms(1.0, kappa)

                model.setObjective(quad_expr, GRB.MINIMIZE)

                # -------------------------------------------------------------
                # 3. Constraints
                # -------------------------------------------------------------
                # (a) Trade definition: w_i - u_i = w_bar_i
                for i in range(n):
                    model.addConstr(w[i] - u[i] == state.current_weights[i], name=f"trade_def_{i}")

                # (b) Self-financing budget: sum_i w_i + c + kappa = 1.0
                model.addConstr(
                    gp.quicksum(w[i] for i in range(n)) + c + kappa == 1.0,
                    name="self_financing_budget",
                )

                # (c) Transaction cost definition: kappa - sum_i a_i * q_i = 0
                model.addConstr(
                    kappa - gp.quicksum(self.params.prop_cost_coeffs[i] * q[i] for i in range(n)) == 0.0,
                    name="transaction_cost_def",
                )

                # (d) Absolute trade linearization: q_i >= u_i and q_i >= -u_i
                for i in range(n):
                    model.addConstr(q[i] - u[i] >= 0.0, name=f"abs_trade_pos_{i}")
                    model.addConstr(q[i] + u[i] >= 0.0, name=f"abs_trade_neg_{i}")

                # (e) Maximum turnover limit: sum_i q_i <= tau_max
                model.addConstr(
                    gp.quicksum(q[i] for i in range(n)) <= self.params.max_turnover,
                    name="turnover_limit",
                )

                # (f) Sector exposure constraints: E_g^min <= sum_{i in S_g} w_i <= E_g^max
                for sector_name, asset_indices in self.params.sector_indices.items():
                    if sector_name in self.params.sector_bounds:
                        e_min, e_max = self.params.sector_bounds[sector_name]
                        sector_sum = gp.quicksum(w[idx] for idx in asset_indices)
                        model.addConstr(sector_sum >= e_min, name=f"sector_min_{sector_name}")
                        model.addConstr(sector_sum <= e_max, name=f"sector_max_{sector_name}")

                # -------------------------------------------------------------
                # 4. Solve and Diagnostic Processing
                # -------------------------------------------------------------
                model.optimize()
                solve_time = time.perf_counter() - start_time

                status_code = model.Status
                status_str = {
                    GRB.OPTIMAL: "OPTIMAL",
                    GRB.INFEASIBLE: "INFEASIBLE",
                    GRB.UNBOUNDED: "UNBOUNDED",
                    GRB.INF_OR_UNBD: "INF_OR_UNBD",
                    GRB.TIME_LIMIT: "TIME_LIMIT",
                    GRB.SUBOPTIMAL: "SUBOPTIMAL",
                }.get(status_code, f"UNKNOWN_{status_code}")

                is_optimal = (status_code == GRB.OPTIMAL)
                is_feasible = (model.SolCount > 0)

                if is_feasible:
                    opt_w = np.array([w[i].X for i in range(n)])
                    opt_u = np.array([u[i].X for i in range(n)])
                    opt_q = np.array([q[i].X for i in range(n)])
                    opt_c = float(c.X)
                    opt_kappa = float(kappa.X)
                    obj_val = float(model.ObjVal)

                    return OptimizationResult(
                        status=status_str,
                        is_optimal=is_optimal,
                        is_feasible=True,
                        solve_time=solve_time,
                        objective_value=obj_val,
                        post_trade_weights=opt_w,
                        trades=opt_u,
                        absolute_trades=opt_q,
                        post_trade_cash=opt_c,
                        transaction_cost=opt_kappa,
                        metadata={
                            "gurobi_status_code": status_code,
                            "iter_count": model.IterCount,
                            "bar_iter_count": getattr(model, "BarIterCount", 0),
                        },
                    )
                else:
                    return OptimizationResult(
                        status=status_str,
                        is_optimal=False,
                        is_feasible=False,
                        solve_time=solve_time,
                        metadata={"gurobi_status_code": status_code},
                    )
