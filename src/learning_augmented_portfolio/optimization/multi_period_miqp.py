"""
Phase 3: Deterministic Multi-Period Mixed-Integer Quadratic Programming (MIQP) Formulation.
"""

from __future__ import annotations

import time
from typing import Optional
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from .big_m import compute_dynamic_tight_big_m, validate_big_m
from .variables import (
    MultiPeriodMarketState,
    MultiPeriodOptimizationResult,
    MultiPeriodPortfolioParameters,
    SolverConfig,
)


class MultiPeriodMIQPOptimizer:
    """
    Builder and solver for the canonical Phase 3 Deterministic Multi-Period MIQP.

    Planning index: h in {0, ..., H-1}.
    Numeraire: Decision-time portfolio wealth V_t = 1.0.

    Mathematical Formulation (Minimization Form):
        min sum_{h=0}^{H-1} beta^h [ (gamma/2) * x_h^T Sigma_h x_h - mu_h^T x_h - r_{f,h} * b_h + kappa_h ]
        s.t.
            At h = 0:
                x_0^- = initial_holdings, b_0^- = initial_cash
            For all h in {0, ..., H-1}:
                x_{i,h} - u_{i,h} = x_{i,h}^-,                forall i          (Trade Definition)
                1^T x_h + b_h = 1^T x_h^- + b_h^- - kappa_h,                    (Self-Financing Budget)
                kappa_h - sum_i (a_{i,h} q_{i,h} + f_{i,h} z_{i,h}) = 0,        (Fixed + Prop Cost)
                q_{i,h} - u_{i,h} >= 0, q_{i,h} + u_{i,h} >= 0, forall i       (Absolute Trade Linearization)
                ell_{i,h} * y_{i,h} <= x_{i,h} <= U_{i,h} * y_{i,h}, forall i   (Holding Semi-Continuous)
                sum_i y_{i,h} <= K_h,                                           (Cardinality Limit)
                delta_{i,h} * z_{i,h} <= q_{i,h} <= M_{i,h} * z_{i,h}, forall i (Trade Semi-Continuous)
                q_{i,h} <= L_{i,h},                           forall i          (Liquidity Limits)
                sum_i q_{i,h} <= tau_h^max,                                     (Turnover Limit)
                b_h >= b_h^min,                                                 (Minimum Cash Reserve)
                E_{g,h}^min <= sum_{i in S_g} x_{i,h} <= E_{g,h}^max,           (Sector Exposures)
            For h in {0, ..., H-2}:
                x_{i,h+1}^- = g_{i,h} * x_{i,h},              forall i          (Risky Holding Transition)
                b_{h+1}^-   = (1 + r_{f,h}) * b_h                               (Cash Transition)
    """

    def __init__(
        self,
        params: MultiPeriodPortfolioParameters,
        solver_config: Optional[SolverConfig] = None,
    ) -> None:
        self.params = params
        self.solver_config = solver_config or SolverConfig()

    def solve(
        self,
        state: MultiPeriodMarketState,
        callback: Optional[Any] = None,
        warm_start: Optional[Any] = None,
    ) -> MultiPeriodOptimizationResult:
        """
        Construct and solve the deterministic multi-period MIQP over horizon H.

        Args:
            state: MultiPeriodMarketState containing initial state, returns, covariances.
            callback: Optional Gurobi callback (e.g. OptimizationProgressLogger).

        Returns:
            MultiPeriodOptimizationResult containing decision trajectories and solver diagnostics.
        """
        H = self.params.horizon
        n = state.n_assets

        # Input validations
        self.params.validate(n)
        state.validate()

        if state.horizon != H:
            raise ValueError(f"State horizon {state.horizon} does not match parameter horizon {H}")

        # Compute dynamic tight Big-M bounds across all periods h in {0, ..., H-1}
        gross_returns = state.gross_returns if state.gross_returns is not None else (
            np.ones((H - 1, n)) if H > 1 else None
        )
        m_sell, m_buy, m_symmetric = compute_dynamic_tight_big_m(
            initial_holdings=state.initial_holdings,
            weight_upper_limits=self.params.weight_upper_limits,
            liquidity_limits=self.params.liquidity_limits,
            max_turnovers=self.params.max_turnovers,
            gross_returns=gross_returns,
        )

        validate_big_m(m_symmetric, self.params.min_trade_sizes, state.asset_names)

        # Prepare default parameters if not provided
        ell = self.params.min_holding_weights if self.params.min_holding_weights is not None else np.zeros((H, n))
        delta = self.params.min_trade_sizes if self.params.min_trade_sizes is not None else np.zeros((H, n))
        f_cost = self.params.fixed_cost_coeffs if self.params.fixed_cost_coeffs is not None else np.zeros((H, n))
        k_cardinality = self.params.cardinality_limits if self.params.cardinality_limits is not None else np.full(H, n)

        start_time = time.perf_counter()

        with gp.Env(empty=True) as env:
            if not self.solver_config.output_flag:
                env.setParam("OutputFlag", 0)
            env.start()

            with gp.Model("Phase3_Multi_Period_MIQP", env=env) as model:
                model.Params.TimeLimit = self.solver_config.time_limit
                model.Params.MIPGap = self.solver_config.mip_gap
                model.Params.Threads = self.solver_config.threads
                model.Params.FeasibilityTol = self.solver_config.feasibility_tol
                model.Params.OptimalityTol = self.solver_config.optimality_tol

                # -------------------------------------------------------------
                # 1. Decision Variables
                # -------------------------------------------------------------
                # Pre-trade risky holdings: x_h^- in R_+^n
                x_minus = model.addVars(
                    H,
                    n,
                    lb=0.0,
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name="x_minus",
                )

                # Post-trade risky holdings: 0 <= x_h <= U_{i,h}
                x = model.addVars(
                    H,
                    n,
                    lb=0.0,
                    ub=self.params.weight_upper_limits,
                    vtype=GRB.CONTINUOUS,
                    name="x",
                )

                # Signed trades: u_h in R^n
                u = model.addVars(
                    H,
                    n,
                    lb=-GRB.INFINITY,
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name="u",
                )

                # Absolute trades: 0 <= q_h <= L_{i,h}
                q = model.addVars(
                    H,
                    n,
                    lb=0.0,
                    ub=self.params.liquidity_limits,
                    vtype=GRB.CONTINUOUS,
                    name="q",
                )

                # Pre-trade cash: b_h^- in R
                b_minus = model.addVars(
                    H,
                    lb=-GRB.INFINITY,
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name="b_minus",
                )

                # Post-trade cash: b_h >= b_h^min
                b = model.addVars(
                    H,
                    lb=self.params.min_cash,
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name="b",
                )

                # Transaction costs: kappa_h >= 0
                kappa = model.addVars(
                    H,
                    lb=0.0,
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name="kappa",
                )

                # Binary Decision Variables: y_{i,h} in {0, 1}, z_{i,h} in {0, 1}
                y = model.addVars(H, n, vtype=GRB.BINARY, name="y")
                z = model.addVars(H, n, vtype=GRB.BINARY, name="z")

                model.update()

                # -------------------------------------------------------------
                # 2. Objective Function (Discounted Minimization)
                # -------------------------------------------------------------
                # sum_{h=0}^{H-1} beta^h [ (gamma/2) * x_h^T Sigma_h x_h - mu_h^T x_h - r_{f,h} * b_h + kappa_h ]
                total_obj = gp.QuadExpr()
                for h in range(H):
                    discount = (self.params.discount_factor ** h)

                    # Quadratic risk term
                    cov_h = state.covariance_matrices[h]
                    for i in range(n):
                        for j in range(n):
                            cov_ij = cov_h[i, j]
                            if abs(cov_ij) > 1e-12:
                                coeff = discount * 0.5 * self.params.risk_aversion * cov_ij
                                total_obj.addTerms(coeff, x[h, i], x[h, j])

                    # Linear expected return term
                    for i in range(n):
                        total_obj.addTerms(-discount * state.expected_returns[h, i], x[h, i])

                    # Risk-free cash return term
                    total_obj.addTerms(-discount * float(self.params.risk_free_rates[h]), b[h])

                    # Transaction cost penalty term
                    total_obj.addTerms(discount, kappa[h])

                model.setObjective(total_obj, GRB.MINIMIZE)

                # -------------------------------------------------------------
                # 3. Initial Conditions at h = 0
                # -------------------------------------------------------------
                for i in range(n):
                    model.addConstr(
                        x_minus[0, i] == float(state.initial_holdings[i]),
                        name=f"init_holdings_{i}",
                    )
                model.addConstr(
                    b_minus[0] == float(state.initial_cash),
                    name="init_cash",
                )

                # -------------------------------------------------------------
                # 4. Intra-Period Constraints for every h in {0, ..., H-1}
                # -------------------------------------------------------------
                for h in range(H):
                    # (a) Trade definition: u_{i,h} = x_{i,h} - x_{i,h}^-
                    for i in range(n):
                        model.addConstr(
                            x[h, i] - u[h, i] == x_minus[h, i],
                            name=f"trade_def_h{h}_{i}",
                        )

                    # (b) Self-financing budget: sum_i x_{i,h} + b_h = sum_i x_{i,h}^- + b_h^- - kappa_h
                    model.addConstr(
                        gp.quicksum(x[h, i] for i in range(n)) + b[h]
                        == gp.quicksum(x_minus[h, i] for i in range(n)) + b_minus[h] - kappa[h],
                        name=f"self_financing_budget_h{h}",
                    )

                    # (c) Fixed + Proportional transaction costs
                    # kappa_h - sum_i (a_{i,h} q_{i,h} + f_{i,h} z_{i,h}) == 0
                    model.addConstr(
                        kappa[h]
                        - gp.quicksum(
                            self.params.prop_cost_coeffs[h, i] * q[h, i] + f_cost[h, i] * z[h, i]
                            for i in range(n)
                        )
                        == 0.0,
                        name=f"transaction_cost_def_h{h}",
                    )

                    # (d) Absolute trade linearization: q_{i,h} >= u_{i,h} and q_{i,h} >= -u_{i,h}
                    for i in range(n):
                        model.addConstr(q[h, i] - u[h, i] >= 0.0, name=f"abs_trade_pos_h{h}_{i}")
                        model.addConstr(q[h, i] + u[h, i] >= 0.0, name=f"abs_trade_neg_h{h}_{i}")

                    # (e) Holding semi-continuous bounds: ell_{i,h} * y_{i,h} <= x_{i,h} <= U_{i,h} * y_{i,h}
                    for i in range(n):
                        model.addConstr(
                            x[h, i] <= self.params.weight_upper_limits[h, i] * y[h, i],
                            name=f"holding_ub_h{h}_{i}",
                        )
                        if ell[h, i] > 0.0:
                            model.addConstr(
                                x[h, i] >= ell[h, i] * y[h, i],
                                name=f"holding_lb_h{h}_{i}",
                            )

                    # (f) Portfolio cardinality: sum_i y_{i,h} <= K_h
                    model.addConstr(
                        gp.quicksum(y[h, i] for i in range(n)) <= int(k_cardinality[h]),
                        name=f"cardinality_h{h}",
                    )

                    # (g) Trade semi-continuous bounds & Dynamic Big-M:
                    # delta_{i,h} * z_{i,h} <= q_{i,h} <= M_{i,h} * z_{i,h}
                    for i in range(n):
                        model.addConstr(
                            q[h, i] <= m_symmetric[h, i] * z[h, i],
                            name=f"trade_ub_big_m_h{h}_{i}",
                        )
                        if delta[h, i] > 0.0:
                            model.addConstr(
                                q[h, i] >= delta[h, i] * z[h, i],
                                name=f"trade_lb_h{h}_{i}",
                            )

                    # (h) Turnover limit: sum_i q_{i,h} <= tau_h^max
                    model.addConstr(
                        gp.quicksum(q[h, i] for i in range(n)) <= float(self.params.max_turnovers[h]),
                        name=f"turnover_limit_h{h}",
                    )

                    # (i) Sector exposure bounds: E_{g,h}^min <= sum_{i in S_g} x_{i,h} <= E_{g,h}^max
                    for sector_name, asset_indices in self.params.sector_indices.items():
                        if sector_name in self.params.sector_bounds:
                            e_min, e_max = self.params.sector_bounds[sector_name]
                            sector_sum = gp.quicksum(x[h, idx] for idx in asset_indices)
                            model.addConstr(sector_sum >= e_min, name=f"sector_min_h{h}_{sector_name}")
                            model.addConstr(sector_sum <= e_max, name=f"sector_max_h{h}_{sector_name}")

                # -------------------------------------------------------------
                # 5. Inter-Period Transition Constraints (for h = 0, ..., H-2)
                # -------------------------------------------------------------
                if H > 1 and gross_returns is not None:
                    for h in range(H - 1):
                        # Risky holding transition: x_{i,h+1}^- = g_{i,h} * x_{i,h}
                        for i in range(n):
                            model.addConstr(
                                x_minus[h + 1, i] == float(gross_returns[h, i]) * x[h, i],
                                name=f"holding_transition_h{h}_{i}",
                            )

                        # Cash transition: b_{h+1}^- = (1 + r_{f,h}) * b_h
                        rf_factor = 1.0 + float(self.params.risk_free_rates[h])
                        model.addConstr(
                            b_minus[h + 1] == rf_factor * b[h],
                            name=f"cash_transition_h{h}",
                        )

                # -------------------------------------------------------------
                # 6. Solve and Extract Diagnostics
                # -------------------------------------------------------------
                if warm_start is not None:
                    from ..learning.warm_start import inject_warm_start_into_model
                    inject_warm_start_into_model(
                        model=model,
                        start_data=warm_start,
                        var_y=y,
                        var_z=z,
                        var_x=x,
                        var_b=b,
                    )

                if callback is not None:
                    if hasattr(callback, "start"):
                        callback.start()
                    model.optimize(callback)
                else:
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
                sol_count = model.SolCount
                is_feasible = (sol_count > 0)

                best_bound = None
                mip_gap = None
                node_count = None
                work = None

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

                try:
                    work = float(model.Work)
                except (gp.GurobiError, AttributeError):
                    pass

                if is_feasible:
                    opt_x_minus = np.array([[x_minus[h, i].X for i in range(n)] for h in range(H)])
                    opt_x = np.array([[x[h, i].X for i in range(n)] for h in range(H)])
                    opt_u = np.array([[u[h, i].X for i in range(n)] for h in range(H)])
                    opt_q = np.array([[q[h, i].X for i in range(n)] for h in range(H)])
                    opt_b_minus = np.array([b_minus[h].X for h in range(H)])
                    opt_b = np.array([b[h].X for h in range(H)])
                    opt_kappa = np.array([kappa[h].X for h in range(H)])
                    opt_y = np.array([[round(y[h, i].X) for i in range(n)] for h in range(H)], dtype=int)
                    opt_z = np.array([[round(z[h, i].X) for i in range(n)] for h in range(H)], dtype=int)
                    obj_val = float(model.ObjVal)

                    return MultiPeriodOptimizationResult(
                        status=status_str,
                        is_optimal=is_optimal,
                        is_feasible=True,
                        solve_time=solve_time,
                        objective_value=obj_val,
                        pre_trade_holdings=opt_x_minus,
                        post_trade_holdings=opt_x,
                        trades=opt_u,
                        absolute_trades=opt_q,
                        pre_trade_cash=opt_b_minus,
                        post_trade_cash=opt_b,
                        transaction_costs=opt_kappa,
                        holding_indicators=opt_y,
                        trade_indicators=opt_z,
                        best_bound=best_bound,
                        mip_gap=mip_gap,
                        node_count=node_count,
                        work=work,
                        sol_count=sol_count,
                        metadata={
                            "gurobi_status_code": status_code,
                            "sol_count": sol_count,
                            "work": work,
                            "num_vars": model.NumVars,
                            "num_bin_vars": model.NumBinVars,
                            "num_constrs": model.NumConstrs,
                            "dynamic_big_m": m_symmetric,
                        },
                    )
                else:
                    return MultiPeriodOptimizationResult(
                        status=status_str,
                        is_optimal=False,
                        is_feasible=False,
                        solve_time=solve_time,
                        best_bound=best_bound,
                        node_count=node_count,
                        work=work,
                        sol_count=sol_count,
                        metadata={
                            "gurobi_status_code": status_code,
                            "work": work,
                            "dynamic_big_m": m_symmetric,
                        },
                    )
