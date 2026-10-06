"""
Candidate-Universe Reduced MIQP Solvers.
Implements:
  1. Fixing Form (Reference): Full-dimension N model with y_{i,h} = 0 for i not in C_t.
  2. Physically Sliced Form (Acceleration): Actual smaller B^{eff} model over C_t.
Validates numerical equivalence between fixing and slicing representations.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Dict, Any
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from .big_m import compute_dynamic_tight_big_m, validate_big_m
from .multi_period_miqp import MultiPeriodMIQPOptimizer
from .support_restriction import (
    CandidateUniverse,
    slice_market_state,
    slice_parameters,
    expand_reduced_solution_to_full,
)
from .variables import (
    MultiPeriodMarketState,
    MultiPeriodOptimizationResult,
    MultiPeriodPortfolioParameters,
    SolverConfig,
)


@dataclass
class ReducedMIQPExecutionResult:
    """
    Detailed telemetry container for a reduced MIQP solve.

    Attributes:
        optimization_result: MultiPeriodOptimizationResult (expanded to full N if requested).
        candidate_universe: The CandidateUniverse used for restriction.
        build_time: Model construction time in seconds.
        solver_time: Gurobi solver execution time in seconds.
        total_time: Combined build_time + solver_time.
        num_vars: Total variables in model.
        num_bin_vars: Binary variables in model.
        num_constrs: Linear constraints in model.
        is_fixing_form: True if fixing form P^{fix}, False if physically sliced P^{slice}.
    """
    optimization_result: MultiPeriodOptimizationResult
    candidate_universe: CandidateUniverse
    build_time: float
    solver_time: float
    total_time: float
    num_vars: int
    num_bin_vars: int
    num_constrs: int
    is_fixing_form: bool

    @property
    def objective_value(self) -> Optional[float]:
        return self.optimization_result.objective_value

    @property
    def is_optimal(self) -> bool:
        return self.optimization_result.is_optimal

    @property
    def is_feasible(self) -> bool:
        return self.optimization_result.is_feasible

    @property
    def mip_gap(self) -> Optional[float]:
        return self.optimization_result.mip_gap

    @property
    def node_count(self) -> Optional[int]:
        return self.optimization_result.node_count

    @property
    def work(self) -> Optional[float]:
        return self.optimization_result.work


class MultiPeriodReducedMIQPOptimizer:
    """
    Physically Sliced Reduced MIQP Optimizer P^{slice}(C_t).
    Slices market state and parameters to the candidate universe C_t of size B^{eff},
    building and solving a genuinely smaller optimization problem.
    """

    def __init__(
        self,
        full_params: MultiPeriodPortfolioParameters,
        solver_config: Optional[SolverConfig] = None,
    ) -> None:
        self.full_params = full_params
        self.solver_config = solver_config or SolverConfig()

    def solve(
        self,
        full_state: MultiPeriodMarketState,
        candidate: CandidateUniverse,
        expand_solution: bool = True,
        callback: Optional[Any] = None,
        warm_start: Optional[Any] = None,
    ) -> ReducedMIQPExecutionResult:
        """
        Build and solve the physically sliced reduced MIQP.

        Args:
            full_state: Original full-universe market state.
            candidate: Candidate universe C_t.
            expand_solution: Whether to expand the output solution back to full dimension N.
            callback: Optional Gurobi progress callback.
            warm_start: Optional MIPStartData for primal warm starting.

        Returns:
            ReducedMIQPExecutionResult with execution telemetry.
        """
        N = full_state.n_assets
        t0_build = time.perf_counter()

        # Check if candidate contains all non-zero initial holdings
        initial_holdings = full_state.initial_holdings
        omitted_holdings = [
            i for i in range(N)
            if initial_holdings[i] > 1e-6 and not candidate.contains(i)
        ]
        if omitted_holdings:
            # Physical slicing cannot preserve wealth conservation when assets with initial holdings are omitted.
            # Delegate to full-dimension fixing optimizer P_t^{fix}(C_t) to evaluate candidate restriction.
            fixed_opt = MultiPeriodFixedMIQPOptimizer(self.full_params, self.solver_config)
            return fixed_opt.solve(full_state, candidate, callback=callback, warm_start=warm_start)

        # 1. Slice parameters and market state
        sliced_params = slice_parameters(self.full_params, candidate)
        sliced_state = slice_market_state(full_state, candidate)

        # Slice warm start if provided in full dimension
        sliced_warm_start = warm_start
        if warm_start is not None and getattr(warm_start, "y_start", None) is not None:
            if warm_start.y_start.shape[1] > candidate.candidate_size:
                idx = list(candidate.candidate_indices)
                from ..learning.warm_start import MIPStartData
                sliced_warm_start = MIPStartData(
                    start_type=warm_start.start_type,
                    y_start=warm_start.y_start[:, idx],
                    z_start=warm_start.z_start[:, idx] if warm_start.z_start is not None else None,
                    x_start=warm_start.x_start[:, idx] if warm_start.x_start is not None else None,
                    u_start=warm_start.u_start[:, idx] if warm_start.u_start is not None else None,
                    b_start=warm_start.b_start,
                    is_complete=warm_start.is_complete,
                    construction_time=warm_start.construction_time,
                    repair_time=warm_start.repair_time,
                )

        t_build = time.perf_counter() - t0_build

        # 2. Build and solve smaller MIQP using existing MultiPeriodMIQPOptimizer
        optimizer = MultiPeriodMIQPOptimizer(sliced_params, self.solver_config)
        raw_res = optimizer.solve(sliced_state, callback=callback, warm_start=sliced_warm_start)

        solver_time = raw_res.solve_time
        num_vars = int(raw_res.metadata.get("num_vars", 0))
        num_bin = int(raw_res.metadata.get("num_bin_vars", 0))
        num_constrs = int(raw_res.metadata.get("num_constrs", 0))

        # 3. Optionally expand solution to full dimension N
        out_res = expand_reduced_solution_to_full(raw_res, candidate, N) if expand_solution else raw_res

        return ReducedMIQPExecutionResult(
            optimization_result=out_res,
            candidate_universe=candidate,
            build_time=t_build,
            solver_time=solver_time,
            total_time=t_build + solver_time,
            num_vars=num_vars,
            num_bin_vars=num_bin,
            num_constrs=num_constrs,
            is_fixing_form=False,
        )


class MultiPeriodFixedMIQPOptimizer:
    """
    Full-Dimension Fixing Form Optimizer P^{fix}(C_t).
    Maintains full universe N but adds explicit constraints y_{i,h} = 0 for all i not in C_t.
    Serves as an authoritative correctness reference against the physically sliced solver.
    """

    def __init__(
        self,
        full_params: MultiPeriodPortfolioParameters,
        solver_config: Optional[SolverConfig] = None,
    ) -> None:
        self.params = full_params
        self.solver_config = solver_config or SolverConfig()

    def solve(
        self,
        full_state: MultiPeriodMarketState,
        candidate: CandidateUniverse,
        callback: Optional[Any] = None,
        warm_start: Optional[Any] = None,
    ) -> ReducedMIQPExecutionResult:
        """
        Build and solve the full-dimension MIQP with non-candidate binaries fixed to 0.
        """
        H = self.params.horizon
        N = full_state.n_assets

        self.params.validate(N)
        full_state.validate()

        t0_build = time.perf_counter()

        gross_returns = full_state.gross_returns if full_state.gross_returns is not None else (
            np.ones((H - 1, N)) if H > 1 else None
        )
        _, _, m_symmetric = compute_dynamic_tight_big_m(
            initial_holdings=full_state.initial_holdings,
            weight_upper_limits=self.params.weight_upper_limits,
            liquidity_limits=self.params.liquidity_limits,
            max_turnovers=self.params.max_turnovers,
            gross_returns=gross_returns,
        )
        validate_big_m(m_symmetric, self.params.min_trade_sizes, full_state.asset_names)

        ell = self.params.min_holding_weights if self.params.min_holding_weights is not None else np.zeros((H, N))
        delta = self.params.min_trade_sizes if self.params.min_trade_sizes is not None else np.zeros((H, N))
        f_cost = self.params.fixed_cost_coeffs if self.params.fixed_cost_coeffs is not None else np.zeros((H, N))
        k_card = self.params.cardinality_limits if self.params.cardinality_limits is not None else np.full(H, N)

        non_candidate_indices = [i for i in range(N) if not candidate.contains(i)]

        with gp.Env(empty=True) as env:
            if not self.solver_config.output_flag:
                env.setParam("OutputFlag", 0)
            env.start()

            with gp.Model("Phase7_Fixing_Reduced_MIQP", env=env) as model:
                model.Params.TimeLimit = self.solver_config.time_limit
                model.Params.MIPGap = self.solver_config.mip_gap
                model.Params.Threads = self.solver_config.threads
                model.Params.FeasibilityTol = self.solver_config.feasibility_tol
                model.Params.OptimalityTol = self.solver_config.optimality_tol

                # Variables
                x_minus = model.addVars(H, N, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_minus")
                x = model.addVars(H, N, lb=0.0, ub=self.params.weight_upper_limits, vtype=GRB.CONTINUOUS, name="x")
                u = model.addVars(H, N, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u")
                q = model.addVars(H, N, lb=0.0, ub=self.params.liquidity_limits, vtype=GRB.CONTINUOUS, name="q")
                b_minus = model.addVars(H, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b_minus")
                b = model.addVars(H, lb=self.params.min_cash, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b")
                kappa = model.addVars(H, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="kappa")
                y = model.addVars(H, N, vtype=GRB.BINARY, name="y")
                z = model.addVars(H, N, vtype=GRB.BINARY, name="z")

                # Objective
                total_obj = gp.QuadExpr()
                for h in range(H):
                    discount = (self.params.discount_factor ** h)
                    cov_h = full_state.covariance_matrices[h]
                    for i in range(N):
                        for j in range(N):
                            cov_ij = cov_h[i, j]
                            if abs(cov_ij) > 1e-12:
                                total_obj.addTerms(discount * 0.5 * self.params.risk_aversion * cov_ij, x[h, i], x[h, j])
                        total_obj.addTerms(-discount * full_state.expected_returns[h, i], x[h, i])
                    total_obj.addTerms(-discount * float(self.params.risk_free_rates[h]), b[h])
                    total_obj.addTerms(discount, kappa[h])
                model.setObjective(total_obj, GRB.MINIMIZE)

                # Initial conditions
                for i in range(N):
                    model.addConstr(x_minus[0, i] == float(full_state.initial_holdings[i]))
                model.addConstr(b_minus[0] == float(full_state.initial_cash))

                # Intra-period constraints
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
                        - gp.quicksum(
                            self.params.prop_cost_coeffs[h, i] * q[h, i] + f_cost[h, i] * z[h, i]
                            for i in range(N)
                        )
                        == 0.0
                    )

                    for i in range(N):
                        model.addConstr(x[h, i] <= self.params.weight_upper_limits[h, i] * y[h, i])
                        if ell[h, i] > 0.0:
                            model.addConstr(x[h, i] >= ell[h, i] * y[h, i])

                    model.addConstr(gp.quicksum(y[h, i] for i in range(N)) <= int(k_card[h]))

                    for i in range(N):
                        model.addConstr(q[h, i] <= m_symmetric[h, i] * z[h, i])
                        if delta[h, i] > 0.0:
                            model.addConstr(q[h, i] >= delta[h, i] * z[h, i])

                    model.addConstr(gp.quicksum(q[h, i] for i in range(N)) <= float(self.params.max_turnovers[h]))

                    for sector_name, asset_indices in self.params.sector_indices.items():
                        if sector_name in self.params.sector_bounds:
                            e_min, e_max = self.params.sector_bounds[sector_name]
                            sector_sum = gp.quicksum(x[h, idx] for idx in asset_indices)
                            model.addConstr(sector_sum >= e_min)
                            model.addConstr(sector_sum <= e_max)

                    # ---------------------------------------------------------
                    # Candidate Universe Fixing: y_{i,h} = 0 for i not in C_t
                    # ---------------------------------------------------------
                    for i_out in non_candidate_indices:
                        model.addConstr(y[h, i_out] == 0, name=f"fix_y_h{h}_{i_out}")

                # Transitions
                if H > 1 and gross_returns is not None:
                    for h in range(H - 1):
                        for i in range(N):
                            model.addConstr(x_minus[h + 1, i] == float(gross_returns[h, i]) * x[h, i])
                        rf_fac = 1.0 + float(self.params.risk_free_rates[h])
                        model.addConstr(b_minus[h + 1] == rf_fac * b[h])

                model.update()
                t_build = time.perf_counter() - t0_build

                num_vars = model.NumVars
                num_bin = model.NumBinVars
                num_constrs = model.NumConstrs

                # Solve
                t0_solve = time.perf_counter()
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
                solve_time = time.perf_counter() - t0_solve

                status_code = model.Status
                status_str = {
                    GRB.OPTIMAL: "OPTIMAL",
                    GRB.INFEASIBLE: "INFEASIBLE",
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
                    opt_x_minus = np.array([[x_minus[h, i].X for i in range(N)] for h in range(H)])
                    opt_x = np.array([[x[h, i].X for i in range(N)] for h in range(H)])
                    opt_u = np.array([[u[h, i].X for i in range(N)] for h in range(H)])
                    opt_q = np.array([[q[h, i].X for i in range(N)] for h in range(H)])
                    opt_b_minus = np.array([b_minus[h].X for h in range(H)])
                    opt_b = np.array([b[h].X for h in range(H)])
                    opt_kappa = np.array([kappa[h].X for h in range(H)])
                    opt_y = np.array([[round(y[h, i].X) for i in range(N)] for h in range(H)], dtype=int)
                    opt_z = np.array([[round(z[h, i].X) for i in range(N)] for h in range(H)], dtype=int)

                    res = MultiPeriodOptimizationResult(
                        status=status_str,
                        is_optimal=is_optimal,
                        is_feasible=True,
                        solve_time=solve_time,
                        objective_value=float(model.ObjVal),
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
                    )
                else:
                    res = MultiPeriodOptimizationResult(
                        status=status_str,
                        is_optimal=False,
                        is_feasible=False,
                        solve_time=solve_time,
                        best_bound=best_bound,
                        node_count=node_count,
                        work=work,
                        sol_count=sol_count,
                    )

                return ReducedMIQPExecutionResult(
                    optimization_result=res,
                    candidate_universe=candidate,
                    build_time=t_build,
                    solver_time=solve_time,
                    total_time=t_build + solve_time,
                    num_vars=num_vars,
                    num_bin_vars=num_bin,
                    num_constrs=num_constrs,
                    is_fixing_form=True,
                )
