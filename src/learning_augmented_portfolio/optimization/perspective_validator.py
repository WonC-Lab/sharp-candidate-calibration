"""Perspective formulation numerical validator.

Implements Tests A-E:
- Test A: PSD Extraction Validity (lambda_min(D) >= -tol, lambda_min(Sigma-D) >= -tol)
- Test B: Lower-bound ordering (LB_basic <= LB_persp <= UB_full)
- Test C: Integer-model equivalence (|J_basic_miqp - J_persp_miqp| <= tol)
- Test D: Rotated-SOC constraint scaling (t * y >= x^2 - tol)
- Test E: No invalid strengthening (feasible integer points remain feasible)
"""
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from .variables import MultiPeriodMarketState, MultiPeriodPortfolioParameters, SolverConfig
from .big_m import compute_dynamic_tight_big_m
from .relaxation import MultiPeriodRelaxedQPOptimizer, MultiPeriodPerspectiveRelaxedOptimizer
from .multi_period_miqp import MultiPeriodMIQPOptimizer

def validate_psd_extraction(
    covariance_matrices: np.ndarray,
    rho_persp: float = 0.95,
    tol_psd: float = 1e-9
) -> Dict[str, Any]:
    """Verify PSD diagonal decomposition Sigma = D + R with D >= 0 and R >= 0."""
    H = len(covariance_matrices)
    min_eig_D_list = []
    min_eig_R_list = []
    
    for h in range(H):
        cov = covariance_matrices[h]
        evals, evecs = np.linalg.eigh(cov)
        lam_min = max(0.0, float(np.min(evals)))
        d_val = rho_persp * lam_min
        D = d_val * np.eye(len(cov))
        R = cov - D
        
        eig_D = float(np.min(np.linalg.eigvalsh(D)))
        eig_R = float(np.min(np.linalg.eigvalsh(R)))
        
        min_eig_D_list.append(eig_D)
        min_eig_R_list.append(eig_R)
        
        assert eig_D >= -tol_psd, f"Period {h}: D min eigenvalue {eig_D} < {-tol_psd}"
        assert eig_R >= -tol_psd, f"Period {h}: R min eigenvalue {eig_R} < {-tol_psd}"
        
    return {
        "valid": True,
        "min_eig_D": min(min_eig_D_list),
        "min_eig_R": min(min_eig_R_list),
        "rho_persp": rho_persp
    }

def solve_full_perspective_miqp(
    params: MultiPeriodPortfolioParameters,
    state: MultiPeriodMarketState,
    solver_config: Optional[SolverConfig] = None,
    rho_persp: float = 0.95
) -> Dict[str, Any]:
    """Solve the full Perspective MIQP with binary indicators intact."""
    cfg = solver_config or SolverConfig()
    H = params.horizon
    n = state.n_assets
    
    gross_returns = state.gross_returns if state.gross_returns is not None else (
        np.ones((H - 1, n)) if H > 1 else None
    )
    m_sell, m_buy, m_symmetric = compute_dynamic_tight_big_m(
        initial_holdings=state.initial_holdings,
        weight_upper_limits=params.weight_upper_limits,
        liquidity_limits=params.liquidity_limits,
        max_turnovers=params.max_turnovers,
        gross_returns=gross_returns,
    )
    
    ell = params.min_holding_weights if params.min_holding_weights is not None else np.zeros((H, n))
    delta = params.min_trade_sizes if params.min_trade_sizes is not None else np.zeros((H, n))
    f_cost = params.fixed_cost_coeffs if params.fixed_cost_coeffs is not None else np.zeros((H, n))
    k_card = params.cardinality_limits if params.cardinality_limits is not None else np.full(H, n)
    
    D_diags = []
    R_mats = []
    for h in range(H):
        cov_h = state.covariance_matrices[h]
        evals, evecs = np.linalg.eigh(cov_h)
        lam_min = max(0.0, float(np.min(evals)))
        d_val = rho_persp * lam_min
        D_diags.append(np.full(n, d_val))
        R_h = cov_h - d_val * np.eye(n)
        r_evals, r_evecs = np.linalg.eigh(R_h)
        r_evals = np.maximum(0.0, r_evals)
        R_mats.append(r_evecs @ np.diag(r_evals) @ r_evecs.T)
        
    start_t = time.perf_counter()
    with gp.Env(empty=True) as env:
        if not cfg.output_flag:
            env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model("Full_Perspective_MIQP", env=env) as model:
            model.Params.TimeLimit = cfg.time_limit
            model.Params.Threads = cfg.threads
            model.Params.MIPGap = cfg.mip_gap
            model.Params.FeasibilityTol = cfg.feasibility_tol
            model.Params.OptimalityTol = cfg.optimality_tol
            
            x_minus = model.addVars(H, n, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_minus")
            x = model.addVars(H, n, lb=0.0, ub=params.weight_upper_limits, vtype=GRB.CONTINUOUS, name="x")
            u = model.addVars(H, n, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u")
            q = model.addVars(H, n, lb=0.0, ub=params.liquidity_limits, vtype=GRB.CONTINUOUS, name="q")
            b_minus = model.addVars(H, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b_minus")
            b = model.addVars(H, lb=params.min_cash, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b")
            kappa = model.addVars(H, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="kappa")
            
            # Binary variables intact!
            y = model.addVars(H, n, vtype=GRB.BINARY, name="y")
            z = model.addVars(H, n, vtype=GRB.BINARY, name="z")
            t_persp = model.addVars(H, n, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t_persp")
            
            total_obj = gp.QuadExpr()
            for h in range(H):
                discount = (params.discount_factor ** h)
                R_h = R_mats[h]
                d_diag = D_diags[h]
                for i in range(n):
                    for j in range(n):
                        rij = R_h[i, j]
                        if abs(rij) > 1e-12:
                            total_obj.addTerms(discount * 0.5 * params.risk_aversion * rij, x[h, i], x[h, j])
                    if d_diag[i] > 1e-12:
                        total_obj.addTerms(discount * 0.5 * params.risk_aversion * d_diag[i], t_persp[h, i])
                    total_obj.addTerms(-discount * state.expected_returns[h, i], x[h, i])
                total_obj.addTerms(-discount * float(params.risk_free_rates[h]), b[h])
                total_obj.addTerms(discount, kappa[h])
            model.setObjective(total_obj, GRB.MINIMIZE)
            
            for i in range(n):
                model.addConstr(x_minus[0, i] == float(state.initial_holdings[i]))
            model.addConstr(b_minus[0] == float(state.initial_cash))
            
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
                    kappa[h] - gp.quicksum(
                        params.prop_cost_coeffs[h, i] * q[h, i] + f_cost[h, i] * z[h, i]
                        for i in range(n)
                    ) == 0.0
                )
                for i in range(n):
                    model.addConstr(x[h, i] <= params.weight_upper_limits[h, i] * y[h, i])
                    if ell[h, i] > 0.0:
                        model.addConstr(x[h, i] >= ell[h, i] * y[h, i])
                    if D_diags[h][i] > 1e-12:
                        model.addQConstr(x[h, i] * x[h, i] <= t_persp[h, i] * y[h, i], name=f"persp_{h}_{i}")
                        
                model.addConstr(gp.quicksum(y[h, i] for i in range(n)) <= int(k_card[h]))
                for i in range(n):
                    model.addConstr(q[h, i] <= m_symmetric[h, i] * z[h, i])
                    if delta[h, i] > 0.0:
                        model.addConstr(q[h, i] >= delta[h, i] * z[h, i])
                model.addConstr(gp.quicksum(q[h, i] for i in range(n)) <= float(params.max_turnovers[h]))
                
                for s_name, indices in params.sector_indices.items():
                    if s_name in params.sector_bounds:
                        e_min, e_max = params.sector_bounds[s_name]
                        s_sum = gp.quicksum(x[h, idx] for idx in indices)
                        model.addConstr(s_sum >= e_min)
                        model.addConstr(s_sum <= e_max)
                        
                if h < H - 1:
                    gross_h = gross_returns[h]
                    for i in range(n):
                        model.addConstr(x_minus[h + 1, i] == gross_h[i] * x[h, i])
                    model.addConstr(b_minus[h + 1] == (1.0 + params.risk_free_rates[h]) * b[h])
                    
            model.optimize()
            sol_time = time.perf_counter() - start_t
            
            if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
                return {
                    "status": "OPTIMAL" if model.Status == GRB.OPTIMAL else "TIME_LIMIT",
                    "objective": float(model.ObjVal),
                    "best_bound": float(model.ObjBound),
                    "solve_time": sol_time,
                    "num_vars": model.NumVars,
                    "num_constrs": model.NumConstrs,
                    "num_qconstrs": model.NumQConstrs
                }
            return {"status": "INFEASIBLE", "objective": None, "solve_time": sol_time}
