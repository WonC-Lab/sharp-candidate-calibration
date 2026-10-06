"""
Phase 2: Single-Period Mixed-Integer Quadratic Programming (MIQP) Formulation.
"""

from __future__ import annotations

import time
from typing import Optional
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from .big_m import compute_tight_big_m, validate_big_m
from .variables import MarketState, OptimizationResult, PortfolioParameters, SolverConfig


class SinglePeriodMIQPOptimizer:
    """
    Builder and solver for the canonical Phase 2 Mixed-Integer Quadratic Program (MIQP).

    Mathematical Formulation (Minimization Form):
        min_{w, u, q, c, kappa, y, z}
            (gamma / 2) * w^T Sigma w - mu^T w - r_f * c + kappa
        s.t.
            w_i - u_i = w_bar_i,                        forall i = 1, ..., n  (Trade Definition)
            sum_i w_i + c + kappa = 1,                                        (Self-Financing Budget)
            kappa - sum_i (a_i q_i + f_i z_i) = 0,                            (Fixed + Prop Cost)
            q_i - u_i >= 0,                             forall i = 1, ..., n  (Linearization +)
            q_i + u_i >= 0,                             forall i = 1, ..., n  (Linearization -)
            ell_i * y_i <= w_i <= U_i * y_i,            forall i = 1, ..., n  (Holding Semi-Continuous)
            sum_i y_i <= K,                                                   (Cardinality Limit)
            delta_i * z_i <= q_i <= M_i * z_i,          forall i = 1, ..., n  (Trade Semi-Continuous)
            q_i <= L_i,                                 forall i = 1, ..., n  (Liquidity Limits)
            sum_i q_i <= tau_max,                                             (Turnover Limit)
            E_g^min <= sum_{i in S_g} w_i <= E_g^max,                         (Sector Exposures)
            c >= c_min,                                                       (Minimum Cash)
            y_i in {0, 1}, z_i in {0, 1},               forall i = 1, ..., n  (Binary Decisions)
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
        Construct and solve the Phase 2 MIQP for the given market state.

        Args:
            state: MarketState containing mu, Sigma, w_bar, c_bar.

        Returns:
            OptimizationResult containing optimal decision variables, binary vectors, and diagnostics.
        """
        n = state.n_assets
        self.params.validate(n)
        state.validate()

        # Compute tight, non-arbitrary Big-M bounds
        m_sell, m_buy, m_symmetric = compute_tight_big_m(
            current_weights=state.current_weights,
            weight_upper_limits=self.params.weight_upper_limits,
            liquidity_limits=self.params.liquidity_limits,
            max_turnover=self.params.max_turnover,
        )

        # Validate that minimum trade sizes are feasible under tight Big-M
        validate_big_m(m_symmetric, self.params.min_trade_sizes, state.asset_names)

        # Resolve optional parameters with default fallbacks
        ell = self.params.min_holding_weights if self.params.min_holding_weights is not None else np.zeros(n)
        delta = self.params.min_trade_sizes if self.params.min_trade_sizes is not None else np.zeros(n)
        f_cost = self.params.fixed_cost_coeffs if self.params.fixed_cost_coeffs is not None else np.zeros(n)
        k_cardinality = self.params.cardinality_limit if self.params.cardinality_limit is not None else n

        start_time = time.perf_counter()

        with gp.Env(empty=True) as env:
            if not self.solver_config.output_flag:
                env.setParam("OutputFlag", 0)
            env.start()

            with gp.Model("Phase2_Canonical_MIQP", env=env) as model:
                # Solver parameter configuration
                model.Params.TimeLimit = self.solver_config.time_limit
                model.Params.MIPGap = self.solver_config.mip_gap
                model.Params.Threads = self.solver_config.threads
                model.Params.FeasibilityTol = self.solver_config.feasibility_tol
                model.Params.OptimalityTol = self.solver_config.optimality_tol

                # -------------------------------------------------------------
                # 1. Decision Variables
                # -------------------------------------------------------------
                # Continuous Variables
                w = model.addVars(
                    n,
                    lb=0.0,
                    ub=self.params.weight_upper_limits,
                    vtype=GRB.CONTINUOUS,
                    name="w",
                )
                u = model.addVars(
                    n,
                    lb=-GRB.INFINITY,
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name="u",
                )
                q = model.addVars(
                    n,
                    lb=0.0,
                    ub=self.params.liquidity_limits,
                    vtype=GRB.CONTINUOUS,
                    name="q",
                )
                c = model.addVar(
                    lb=self.params.min_cash,
                    ub=1.0,
                    vtype=GRB.CONTINUOUS,
                    name="c",
                )
                kappa = model.addVar(
                    lb=0.0,
                    ub=1.0,
                    vtype=GRB.CONTINUOUS,
                    name="kappa",
                )

                # Binary Decision Variables
                y = model.addVars(n, vtype=GRB.BINARY, name="y")  # Holding indicators
                z = model.addVars(n, vtype=GRB.BINARY, name="z")  # Trade indicators

                model.update()

                # -------------------------------------------------------------
                # 2. Objective Function (Minimization)
                # -------------------------------------------------------------
                # (gamma / 2) * w^T Sigma w - mu^T w - r_f * c + kappa
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

                # (c) Fixed + Proportional transaction cost accounting:
                # kappa - sum_i (a_i * q_i + f_i * z_i) == 0
                model.addConstr(
                    kappa - gp.quicksum(self.params.prop_cost_coeffs[i] * q[i] + f_cost[i] * z[i] for i in range(n)) == 0.0,
                    name="fixed_prop_transaction_cost_def",
                )

                # (d) Absolute trade linearization: q_i >= u_i and q_i >= -u_i
                for i in range(n):
                    model.addConstr(q[i] - u[i] >= 0.0, name=f"abs_trade_pos_{i}")
                    model.addConstr(q[i] + u[i] >= 0.0, name=f"abs_trade_neg_{i}")

                # (e) Semi-continuous holding bounds: ell_i * y_i <= w_i <= U_i * y_i
                for i in range(n):
                    model.addConstr(w[i] <= self.params.weight_upper_limits[i] * y[i], name=f"holding_ub_{i}")
                    if ell[i] > 0.0:
                        model.addConstr(w[i] >= ell[i] * y[i], name=f"holding_lb_{i}")

                # (f) Portfolio cardinality: sum_i y_i <= K
                model.addConstr(
                    gp.quicksum(y[i] for i in range(n)) <= k_cardinality,
                    name="portfolio_cardinality",
                )

                # (g) Semi-continuous trade bounds & Tight Big-M: delta_i * z_i <= q_i <= M_i * z_i
                for i in range(n):
                    model.addConstr(q[i] <= m_symmetric[i] * z[i], name=f"trade_ub_big_m_{i}")
                    if delta[i] > 0.0:
                        model.addConstr(q[i] >= delta[i] * z[i], name=f"trade_lb_{i}")

                # (h) Maximum turnover limit: sum_i q_i <= tau_max
                model.addConstr(
                    gp.quicksum(q[i] for i in range(n)) <= self.params.max_turnover,
                    name="turnover_limit",
                )

                # (i) Sector exposure bounds: E_g^min <= sum_{i in S_g} w_i <= E_g^max
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

                best_bound = None
                mip_gap = None
                node_count = None

                try:
                    best_bound = float(model.ObjBound)
                except (gp.GurobiError, AttributeError):
                    pass

                try:
                    mip_gap = float(model.MIPGap)
                except (gp.GurobiError, AttributeError):
                    pass

                try:
                    node_count = int(model.NodeCount)
                except (gp.GurobiError, AttributeError):
                    pass

                if is_feasible:
                    opt_w = np.array([w[i].X for i in range(n)])
                    opt_u = np.array([u[i].X for i in range(n)])
                    opt_q = np.array([q[i].X for i in range(n)])
                    opt_c = float(c.X)
                    opt_kappa = float(kappa.X)
                    opt_y = np.array([round(y[i].X) for i in range(n)], dtype=int)
                    opt_z = np.array([round(z[i].X) for i in range(n)], dtype=int)
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
                        holding_indicators=opt_y,
                        trade_indicators=opt_z,
                        best_bound=best_bound,
                        mip_gap=mip_gap,
                        node_count=node_count,
                        metadata={
                            "gurobi_status_code": status_code,
                            "sol_count": model.SolCount,
                            "tight_big_m": m_symmetric,
                        },
                    )
                else:
                    return OptimizationResult(
                        status=status_str,
                        is_optimal=False,
                        is_feasible=False,
                        solve_time=solve_time,
                        best_bound=best_bound,
                        node_count=node_count,
                        metadata={
                            "gurobi_status_code": status_code,
                            "tight_big_m": m_symmetric,
                        },
                    )
