"""
Continuous relaxation solver for multi-period portfolio optimization.
Relaxes integer constraints y_{i,h}, z_{i,h} in [0, 1], yielding a valid convex QP lower bound.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Dict, Any
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from .big_m import compute_dynamic_tight_big_m
from .variables import (
    MultiPeriodMarketState,
    MultiPeriodPortfolioParameters,
    SolverConfig,
)


@dataclass
class ContinuousRelaxationResult:
    """Output container for continuous relaxation solve."""
    status: str
    is_feasible: bool
    objective_value: Optional[float]
    solve_time: float
    relaxed_x: Optional[np.ndarray]       # shape (H, n)
    relaxed_y: Optional[np.ndarray]       # shape (H, n) fractional in [0, 1]
    relaxed_z: Optional[np.ndarray]       # shape (H, n) fractional in [0, 1]
    reduced_costs_x: Optional[np.ndarray] = None
    slacks: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = None


class MultiPeriodRelaxedQPOptimizer:
    """
    Constructs and solves the continuous QP relaxation of the Phase 3 multi-period MIQP.
    Replaces binary variables with continuous bounds y in [0, 1], z in [0, 1].
    """

    def __init__(
        self,
        params: MultiPeriodPortfolioParameters,
        solver_config: Optional[SolverConfig] = None,
    ) -> None:
        self.params = params
        self.solver_config = solver_config or SolverConfig()

    def solve(self, state: MultiPeriodMarketState) -> ContinuousRelaxationResult:
        H = self.params.horizon
        n = state.n_assets

        self.params.validate(n)
        state.validate()

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

        ell = self.params.min_holding_weights if self.params.min_holding_weights is not None else np.zeros((H, n))
        delta = self.params.min_trade_sizes if self.params.min_trade_sizes is not None else np.zeros((H, n))
        f_cost = self.params.fixed_cost_coeffs if self.params.fixed_cost_coeffs is not None else np.zeros((H, n))
        k_cardinality = self.params.cardinality_limits if self.params.cardinality_limits is not None else np.full(H, n)

        start_time = time.perf_counter()

        with gp.Env(empty=True) as env:
            if not self.solver_config.output_flag:
                env.setParam("OutputFlag", 0)
            env.start()

            with gp.Model("Multi_Period_Continuous_Relaxation_QP", env=env) as model:
                model.Params.TimeLimit = self.solver_config.time_limit
                model.Params.Threads = self.solver_config.threads
                model.Params.FeasibilityTol = self.solver_config.feasibility_tol
                model.Params.OptimalityTol = self.solver_config.optimality_tol

                # Continuous variables
                x_minus = model.addVars(H, n, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_minus")
                x = model.addVars(H, n, lb=0.0, ub=self.params.weight_upper_limits, vtype=GRB.CONTINUOUS, name="x")
                u = model.addVars(H, n, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u")
                q = model.addVars(H, n, lb=0.0, ub=self.params.liquidity_limits, vtype=GRB.CONTINUOUS, name="q")
                b_minus = model.addVars(H, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b_minus")
                b = model.addVars(H, lb=self.params.min_cash, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b")
                kappa = model.addVars(H, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="kappa")

                # RELAXED BINARIES: y in [0, 1], z in [0, 1] as continuous variables
                y = model.addVars(H, n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y_relaxed")
                z = model.addVars(H, n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="z_relaxed")

                model.update()

                # Objective
                total_obj = gp.QuadExpr()
                for h in range(H):
                    discount = (self.params.discount_factor ** h)
                    cov_h = state.covariance_matrices[h]
                    for i in range(n):
                        for j in range(n):
                            cov_ij = cov_h[i, j]
                            if abs(cov_ij) > 1e-12:
                                total_obj.addTerms(discount * 0.5 * self.params.risk_aversion * cov_ij, x[h, i], x[h, j])
                        total_obj.addTerms(-discount * state.expected_returns[h, i], x[h, i])
                    total_obj.addTerms(-discount * float(self.params.risk_free_rates[h]), b[h])
                    total_obj.addTerms(discount, kappa[h])
                model.setObjective(total_obj, GRB.MINIMIZE)

                # Initial conditions
                for i in range(n):
                    model.addConstr(x_minus[0, i] == float(state.initial_holdings[i]))
                model.addConstr(b_minus[0] == float(state.initial_cash))

                # Intra-period constraints
                for h in range(H):
                    for i in range(n):
                        model.addConstr(x[h, i] - u[h, i] == x_minus[h, i])
                        model.addConstr(q[h, i] - u[h, i] >= 0.0)
                        model.addConstr(q[h, i] + u[h, i] >= 0.0)

                    model.addConstr(
                        gp.quicksum(x[h, i] for i in range(n)) + b[h]
                        == gp.quicksum(x_minus[h, i] for i in range(n)) + b_minus[h] - kappa[h]
                    )

                    model.addConstr(
                        kappa[h]
                        - gp.quicksum(
                            self.params.prop_cost_coeffs[h, i] * q[h, i] + f_cost[h, i] * z[h, i]
                            for i in range(n)
                        )
                        == 0.0
                    )

                    for i in range(n):
                        model.addConstr(x[h, i] <= self.params.weight_upper_limits[h, i] * y[h, i])
                        if ell[h, i] > 0.0:
                            model.addConstr(x[h, i] >= ell[h, i] * y[h, i])

                    model.addConstr(gp.quicksum(y[h, i] for i in range(n)) <= int(k_cardinality[h]))

                    for i in range(n):
                        model.addConstr(q[h, i] <= m_symmetric[h, i] * z[h, i])
                        if delta[h, i] > 0.0:
                            model.addConstr(q[h, i] >= delta[h, i] * z[h, i])

                    model.addConstr(gp.quicksum(q[h, i] for i in range(n)) <= float(self.params.max_turnovers[h]))

                    for sector_name, asset_indices in self.params.sector_indices.items():
                        if sector_name in self.params.sector_bounds:
                            e_min, e_max = self.params.sector_bounds[sector_name]
                            sector_sum = gp.quicksum(x[h, idx] for idx in asset_indices)
                            model.addConstr(sector_sum >= e_min)
                            model.addConstr(sector_sum <= e_max)

                # Transitions
                if H > 1 and gross_returns is not None:
                    for h in range(H - 1):
                        for i in range(n):
                            model.addConstr(x_minus[h + 1, i] == float(gross_returns[h, i]) * x[h, i])
                        rf_fac = 1.0 + float(self.params.risk_free_rates[h])
                        model.addConstr(b_minus[h + 1] == rf_fac * b[h])

                model.optimize()
                solve_time = time.perf_counter() - start_time

                if model.Status == GRB.OPTIMAL:
                    rx = np.array([[x[h, i].X for i in range(n)] for h in range(H)])
                    ry = np.array([[y[h, i].X for i in range(n)] for h in range(H)])
                    rz = np.array([[z[h, i].X for i in range(n)] for h in range(H)])

                    # Extract reduced costs if available (for continuous QP)
                    reduced_costs = None
                    try:
                        reduced_costs = np.array([[x[h, i].RC for i in range(n)] for h in range(H)])
                    except (gp.GurobiError, AttributeError):
                        pass

                    return ContinuousRelaxationResult(
                        status="OPTIMAL",
                        is_feasible=True,
                        objective_value=float(model.ObjVal),
                        solve_time=solve_time,
                        relaxed_x=rx,
                        relaxed_y=ry,
                        relaxed_z=rz,
                        reduced_costs_x=reduced_costs,
                        metadata={"iter_count": model.IterCount},
                    )
                else:
                    return ContinuousRelaxationResult(
                        status=f"STATUS_{model.Status}",
                        is_feasible=False,
                        objective_value=None,
                        solve_time=solve_time,
                        relaxed_x=None,
                        relaxed_y=None,
                        relaxed_z=None,
                        metadata={"status_code": model.Status},
                    )


class MultiPeriodPerspectiveRelaxedOptimizer:
    """
    Constructs and solves the perspective-strengthened continuous relaxation of the multi-period MIQP.
    Decomposes Sigma = D + R where D = rho_persp * lambda_min(Sigma) * I >= 0 and R >= 0.
    Replaces d_i * x_i^2 with perspective variable t_i satisfying rotated SOC:
      t_i * y_i >= x_i^2,  t_i >= 0,  y_i in [0, 1].
    Provably dominates standard relaxation: LB^{basic} <= LB^{persp} <= J^*.
    """

    def __init__(
        self,
        params: MultiPeriodPortfolioParameters,
        solver_config: Optional[SolverConfig] = None,
        rho_persp: float = 0.95,
    ) -> None:
        self.params = params
        self.solver_config = solver_config or SolverConfig()
        self.rho_persp = float(min(1.0, max(0.0, rho_persp)))

    def solve(self, state: MultiPeriodMarketState) -> ContinuousRelaxationResult:
        H = self.params.horizon
        n = state.n_assets

        self.params.validate(n)
        state.validate()

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

        ell = self.params.min_holding_weights if self.params.min_holding_weights is not None else np.zeros((H, n))
        delta = self.params.min_trade_sizes if self.params.min_trade_sizes is not None else np.zeros((H, n))
        f_cost = self.params.fixed_cost_coeffs if self.params.fixed_cost_coeffs is not None else np.zeros((H, n))
        k_cardinality = self.params.cardinality_limits if self.params.cardinality_limits is not None else np.full(H, n)

        # Decompose Sigma = D + R for each horizon step
        D_diags = []
        R_mats = []
        for h in range(H):
            cov_h = state.covariance_matrices[h]
            evals, evecs = np.linalg.eigh(cov_h)
            lam_min = max(0.0, float(np.min(evals)))
            d_val = self.rho_persp * lam_min
            D_diags.append(np.full(n, d_val))
            # R = cov - d * I
            R_h = cov_h - d_val * np.eye(n)
            # Ensure PSD numerically
            r_evals, r_evecs = np.linalg.eigh(R_h)
            r_evals = np.maximum(0.0, r_evals)
            R_h_psd = r_evecs @ np.diag(r_evals) @ r_evecs.T
            R_mats.append(R_h_psd)

        start_time = time.perf_counter()

        with gp.Env(empty=True) as env:
            if not self.solver_config.output_flag:
                env.setParam("OutputFlag", 0)
            env.start()

            with gp.Model("Multi_Period_Perspective_Relaxation_SOCP", env=env) as model:
                model.Params.TimeLimit = self.solver_config.time_limit
                model.Params.Threads = self.solver_config.threads
                model.Params.FeasibilityTol = self.solver_config.feasibility_tol
                model.Params.OptimalityTol = self.solver_config.optimality_tol

                # Variables
                x_minus = model.addVars(H, n, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_minus")
                x = model.addVars(H, n, lb=0.0, ub=self.params.weight_upper_limits, vtype=GRB.CONTINUOUS, name="x")
                u = model.addVars(H, n, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u")
                q = model.addVars(H, n, lb=0.0, ub=self.params.liquidity_limits, vtype=GRB.CONTINUOUS, name="q")
                b_minus = model.addVars(H, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b_minus")
                b = model.addVars(H, lb=self.params.min_cash, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b")
                kappa = model.addVars(H, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="kappa")

                # Continuous relaxations of binary indicators
                y = model.addVars(H, n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y_relaxed")
                z = model.addVars(H, n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="z_relaxed")

                # Auxiliary perspective variables: t[h, i] >= x[h, i]^2 / y[h, i]
                t_persp = model.addVars(H, n, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t_persp")

                model.update()

                # Objective: sum_h beta^h [ gamma/2 (x^T R x + sum d_i t_i) - mu^T x - r_f b + kappa ]
                total_obj = gp.QuadExpr()
                for h in range(H):
                    discount = (self.params.discount_factor ** h)
                    R_h = R_mats[h]
                    d_diag = D_diags[h]
                    for i in range(n):
                        for j in range(n):
                            rij = R_h[i, j]
                            if abs(rij) > 1e-12:
                                total_obj.addTerms(discount * 0.5 * self.params.risk_aversion * rij, x[h, i], x[h, j])
                        if d_diag[i] > 1e-12:
                            total_obj.addTerms(discount * 0.5 * self.params.risk_aversion * d_diag[i], t_persp[h, i])
                        total_obj.addTerms(-discount * state.expected_returns[h, i], x[h, i])
                    total_obj.addTerms(-discount * float(self.params.risk_free_rates[h]), b[h])
                    total_obj.addTerms(discount, kappa[h])
                model.setObjective(total_obj, GRB.MINIMIZE)

                # Initial conditions
                for i in range(n):
                    model.addConstr(x_minus[0, i] == float(state.initial_holdings[i]))
                model.addConstr(b_minus[0] == float(state.initial_cash))

                # Intra-period constraints
                for h in range(H):
                    for i in range(n):
                        model.addConstr(x[h, i] - u[h, i] == x_minus[h, i])
                        model.addConstr(q[h, i] - u[h, i] >= 0.0)
                        model.addConstr(q[h, i] + u[h, i] >= 0.0)

                    model.addConstr(
                        gp.quicksum(x[h, i] for i in range(n)) + b[h]
                        == gp.quicksum(x_minus[h, i] for i in range(n)) + b_minus[h] - kappa[h]
                    )

                    model.addConstr(
                        kappa[h]
                        - gp.quicksum(
                            self.params.prop_cost_coeffs[h, i] * q[h, i] + f_cost[h, i] * z[h, i]
                            for i in range(n)
                        )
                        == 0.0
                    )

                    for i in range(n):
                        model.addConstr(x[h, i] <= self.params.weight_upper_limits[h, i] * y[h, i])
                        if ell[h, i] > 0.0:
                            model.addConstr(x[h, i] >= ell[h, i] * y[h, i])
                        # Rotated second-order cone: x[h, i]^2 <= t_persp[h, i] * y[h, i]
                        if D_diags[h][i] > 1e-12:
                            model.addQConstr(x[h, i] * x[h, i] <= t_persp[h, i] * y[h, i], name=f"persp_{h}_{i}")

                    model.addConstr(gp.quicksum(y[h, i] for i in range(n)) <= int(k_cardinality[h]))

                    for i in range(n):
                        model.addConstr(q[h, i] <= m_symmetric[h, i] * z[h, i])
                        if delta[h, i] > 0.0:
                            model.addConstr(q[h, i] >= delta[h, i] * z[h, i])

                    model.addConstr(gp.quicksum(q[h, i] for i in range(n)) <= float(self.params.max_turnovers[h]))

                    for sector_name, asset_indices in self.params.sector_indices.items():
                        if sector_name in self.params.sector_bounds:
                            e_min, e_max = self.params.sector_bounds[sector_name]
                            sector_sum = gp.quicksum(x[h, idx] for idx in asset_indices)
                            model.addConstr(sector_sum >= e_min)
                            model.addConstr(sector_sum <= e_max)

                # Transitions
                if H > 1 and gross_returns is not None:
                    for h in range(H - 1):
                        for i in range(n):
                            model.addConstr(x_minus[h + 1, i] == float(gross_returns[h, i]) * x[h, i])
                        rf_fac = 1.0 + float(self.params.risk_free_rates[h])
                        model.addConstr(b_minus[h + 1] == rf_fac * b[h])

                model.optimize()
                solve_time = time.perf_counter() - start_time

                if model.Status == GRB.OPTIMAL:
                    rx = np.array([[x[h, i].X for i in range(n)] for h in range(H)])
                    ry = np.array([[y[h, i].X for i in range(n)] for h in range(H)])
                    rz = np.array([[z[h, i].X for i in range(n)] for h in range(H)])

                    return ContinuousRelaxationResult(
                        status="OPTIMAL",
                        is_feasible=True,
                        objective_value=float(model.ObjVal),
                        solve_time=solve_time,
                        relaxed_x=rx,
                        relaxed_y=ry,
                        relaxed_z=rz,
                        metadata={"iter_count": model.IterCount, "rho_persp": self.rho_persp},
                    )
                else:
                    return ContinuousRelaxationResult(
                        status=f"STATUS_{model.Status}",
                        is_feasible=False,
                        objective_value=None,
                        solve_time=solve_time,
                        relaxed_x=None,
                        relaxed_y=None,
                        relaxed_z=None,
                        metadata={"status_code": model.Status},
                    )

