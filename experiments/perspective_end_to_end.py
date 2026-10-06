"""End-to-end comparison between standard continuous relaxation and perspective relaxation.

Audit Requirements:
- Number of LB-ordering violations (must be 0)
- Number of integer-equivalence violations (must be 0)
- Minimum eigenvalue of R
- Mean / Median LB tightening: LB_persp - LB_basic
- Early certification frequency (stage 1 success)
- Mean L_term and full-fallback rate
- Relaxation solve overhead
- End-to-end runtime and runtime reduction percentage
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from learning_augmented_portfolio.optimization.variables import (
    MultiPeriodMarketState,
    MultiPeriodPortfolioParameters,
    SolverConfig,
)
from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from learning_augmented_portfolio.optimization.reduced_miqp import MultiPeriodReducedMIQPOptimizer
from learning_augmented_portfolio.optimization.support_restriction import CandidateUniverse
from learning_augmented_portfolio.optimization.relaxation import (
    MultiPeriodRelaxedQPOptimizer,
    MultiPeriodPerspectiveRelaxedOptimizer,
)
from learning_augmented_portfolio.optimization.certification import (
    relaxation_certificate,
    full_mip_certificate,
)
from learning_augmented_portfolio.optimization.recovery_schedule import (
    StrictlyIncreasingRecoveryLadder,
    compute_monotone_ub_envelope,
    evaluate_recovery_stages,
)
from learning_augmented_portfolio.optimization.perspective_validator import (
    validate_psd_extraction,
    solve_full_perspective_miqp,
)


def run_perspective_end_to_end(
    num_instances: int = 10,
    epsilon_certs: list = [1e-4, 5e-4, 1e-3],
    rho_persp: float = 0.95
):
    print("=" * 75)
    print("RUNNING PERSPECTIVE RIGOROUS PRE-EXPERIMENT VALIDATION AUDIT")
    print("=" * 75)
    
    manifest_path = "artifacts/instances_medium/manifest.csv"
    if not os.path.exists(manifest_path):
        print("Manifest not found:", manifest_path)
        return
        
    df_manifest = pd.read_csv(manifest_path)
    solver_cfg = SolverConfig(output_flag=False, time_limit=15.0, mip_gap=1e-5)
    
    records = []
    lb_ordering_violations = 0
    integer_equiv_violations = 0
    min_eig_R_overall = float("inf")
    
    for idx in range(min(num_instances, len(df_manifest))):
        row = df_manifest.iloc[idx]
        npz_path = os.path.join("artifacts/instances_medium", row["filename"])
        if not os.path.exists(npz_path):
            continue
            
        data = np.load(npz_path, allow_pickle=True)
        N = 5
        H = 3
        
        cov_full = data["covariance_matrices"][:H, :N, :N].copy()
        for h in range(H):
            cov_full[h] = 0.5 * (cov_full[h] + cov_full[h].T)
            min_eig = np.min(np.linalg.eigvalsh(cov_full[h]))
            if min_eig < 1e-6:
                cov_full[h] += (1e-6 - min_eig + 1e-6) * np.eye(N)
                
        init_holdings = data["initial_holdings"][:N].copy()
        if np.sum(init_holdings) > 0.90:
            init_holdings = init_holdings / np.sum(init_holdings) * 0.85
        init_cash = float(max(0.05, 1.0 - np.sum(init_holdings)))
        
        mu = data["expected_returns"][:H, :N].copy()
        gross_ret = data["gross_returns"][:H-1, :N].copy() if "gross_returns" in data else np.ones((H-1, N))
        
        params = MultiPeriodPortfolioParameters(
            horizon=H,
            risk_aversion=5.0,
            discount_factor=0.99,
            min_cash=np.full(H, 0.05),
            weight_upper_limits=np.full((H, N), 0.50),
            min_holding_weights=np.full((H, N), 0.08),
            liquidity_limits=np.full((H, N), 0.30),
            min_trade_sizes=np.full((H, N), 0.02),
            fixed_cost_coeffs=np.full((H, N), 1e-4),
            prop_cost_coeffs=np.full((H, N), 5e-4),
            max_turnovers=np.full(H, 0.40),
            cardinality_limits=np.full(H, 2),
            risk_free_rates=np.full(H, 0.001),
        )
        
        state = MultiPeriodMarketState(
            expected_returns=mu,
            covariance_matrices=cov_full,
            initial_holdings=init_holdings,
            initial_cash=init_cash,
            gross_returns=gross_ret,
        )
        
        # Test A: PSD Extraction Validation
        psd_audit = validate_psd_extraction(cov_full, rho_persp=rho_persp, tol_psd=1e-9)
        min_eig_R_overall = min(min_eig_R_overall, psd_audit["min_eig_R"])
        
        # 1. Full reference canonical MIP solve
        full_opt = MultiPeriodMIQPOptimizer(params, solver_cfg)
        full_res = full_opt.solve(state)
        if not full_res.is_feasible:
            continue
        ub_full = full_res.objective_value
        lb_full = full_res.best_bound if full_res.best_bound is not None else ub_full - 1e-5
        
        # Test C: Full Perspective MIP solve with binary variables
        persp_miqp_res = solve_full_perspective_miqp(params, state, solver_cfg, rho_persp=rho_persp)
        if persp_miqp_res["status"] in ("OPTIMAL", "TIME_LIMIT") and persp_miqp_res["objective"] is not None:
            diff_miqp = abs(ub_full - persp_miqp_res["objective"])
            if diff_miqp > 1e-4:
                integer_equiv_violations += 1
        
        # 2. Standard Continuous Relaxation
        t0 = time.perf_counter()
        basic_opt = MultiPeriodRelaxedQPOptimizer(params, solver_cfg)
        basic_res = basic_opt.solve(state)
        t_basic_ms = (time.perf_counter() - t0) * 1000.0
        lb_basic = basic_res.objective_value
        
        # 3. Perspective Continuous Relaxation
        t0 = time.perf_counter()
        persp_opt = MultiPeriodPerspectiveRelaxedOptimizer(params, solver_cfg, rho_persp=rho_persp)
        persp_res = persp_opt.solve(state)
        t_persp_ms = (time.perf_counter() - t0) * 1000.0
        lb_persp = persp_res.objective_value
        
        # Test B: LB Ordering Check
        if lb_basic > lb_persp + 1e-6:
            lb_ordering_violations += 1
        if lb_persp > ub_full + 1e-6:
            lb_ordering_violations += 1
            
        tightening = lb_persp - lb_basic
        
        # Candidate ranking and subproblem solving
        mandatory = [i for i in range(N) if state.initial_holdings[i] > 1e-6]
        B_min_C = max(2, len(mandatory))
        
        scores = np.mean(state.expected_returns, axis=0)
        non_mand_ranked = [i for i in np.argsort(-scores) if i not in mandatory]
        
        cand_family = {}
        cand_upper_bounds = {}
        cand_runtimes_ms = {}
        for B in range(B_min_C, N + 1):
            cand_indices = mandatory + non_mand_ranked[:(B - len(mandatory))]
            cand_family[B] = set(cand_indices)
            
            cand_univ = CandidateUniverse(candidate_indices=tuple(sorted(cand_indices)), full_universe_size=N)
            red_opt = MultiPeriodReducedMIQPOptimizer(params, solver_cfg)
            exec_res = red_opt.solve(state, cand_univ)
            r = exec_res.optimization_result
            if r.is_feasible and r.objective_value is not None:
                cand_upper_bounds[B] = r.objective_value
                cand_runtimes_ms[B] = r.solve_time * 1000.0
                
        # Recovery evaluation for each epsilon_cert
        for eps in epsilon_certs:
            ladder_gen = StrictlyIncreasingRecoveryLadder(gamma=1.5, N=N)
            ladder = ladder_gen.generate_ladder(B_1=B_min_C, B_min=B_min_C)
            
            # --- Baseline Recovery with LB_basic ---
            L_term_basic = len(ladder)
            cert_status_basic = "SOLVER_CERTIFIED"
            time_basic_rec = t_basic_ms
            early_cert_basic = False
            raw_ubs = []
            
            for s_idx, B_curr in enumerate(ladder, start=1):
                ub_c = cand_upper_bounds.get(B_curr, ub_full)
                raw_ubs.append(ub_c)
                time_basic_rec += cand_runtimes_ms.get(B_curr, 0.0)
                cert_eval = relaxation_certificate(ub_c, lb_basic, epsilon_cert=eps)
                if cert_eval["certified"]:
                    L_term_basic = s_idx
                    cert_status_basic = "RELAXATION_CERTIFIED"
                    if s_idx == 1:
                        early_cert_basic = True
                    break
                    
            if cert_status_basic != "RELAXATION_CERTIFIED":
                time_basic_rec += full_res.solve_time * 1000.0
                
            # --- Perspective Recovery with LB_persp ---
            L_term_persp = len(ladder)
            cert_status_persp = "SOLVER_CERTIFIED"
            time_persp_rec = t_persp_ms
            early_cert_persp = False
            
            for s_idx, B_curr in enumerate(ladder, start=1):
                ub_c = cand_upper_bounds.get(B_curr, ub_full)
                time_persp_rec += cand_runtimes_ms.get(B_curr, 0.0)
                cert_eval = relaxation_certificate(ub_c, lb_persp, epsilon_cert=eps)
                if cert_eval["certified"]:
                    L_term_persp = s_idx
                    cert_status_persp = "RELAXATION_CERTIFIED"
                    if s_idx == 1:
                        early_cert_persp = True
                    break
                    
            if cert_status_persp != "RELAXATION_CERTIFIED":
                time_persp_rec += full_res.solve_time * 1000.0
                
            runtime_ratio = time_persp_rec / max(1e-3, time_basic_rec)
            
            records.append({
                "instance_id": row["filename"],
                "epsilon_cert": eps,
                "lb_basic": lb_basic,
                "lb_persp": lb_persp,
                "ub_full": ub_full,
                "tightening": tightening,
                "t_basic_ms": t_basic_ms,
                "t_persp_ms": t_persp_ms,
                "relaxation_overhead_ms": t_persp_ms - t_basic_ms,
                "gap_basic": ub_full - lb_basic,
                "gap_persp": ub_full - lb_persp,
                "L_term_basic": L_term_basic,
                "L_term_persp": L_term_persp,
                "cert_status_basic": cert_status_basic,
                "cert_status_persp": cert_status_persp,
                "early_cert_basic": early_cert_basic,
                "early_cert_persp": early_cert_persp,
                "time_basic_total_ms": time_basic_rec,
                "time_persp_total_ms": time_persp_rec,
                "runtime_ratio": runtime_ratio,
                "runtime_reduction_pct": max(0.0, (1.0 - runtime_ratio) * 100.0)
            })
            
    df = pd.DataFrame(records)
    out_csv = "artifacts/perspective_end_to_end_results.csv"
    df.to_csv(out_csv, index=False)
    
    print(f"\nAudit complete across {len(df)} configurations. Saved to {out_csv}")
    print("-" * 75)
    print(f"LB-ordering violations:         {lb_ordering_violations}")
    print(f"Integer-equivalence violations: {integer_equiv_violations}")
    print(f"Minimum eigenvalue of R:        {min_eig_R_overall:.6e}")
    print(f"Mean LB tightening:             +{df['tightening'].mean():.5f}")
    print(f"Median LB tightening:           +{df['tightening'].median():.5f}")
    print(f"Mean Relaxation Overhead:       +{df['relaxation_overhead_ms'].mean():.2f} ms")
    
    for eps, sub in df.groupby("epsilon_cert"):
        early_b = sub["early_cert_basic"].mean() * 100
        early_p = sub["early_cert_persp"].mean() * 100
        mean_l_b = sub["L_term_basic"].mean()
        mean_l_p = sub["L_term_persp"].mean()
        fallback_b = (sub["cert_status_basic"] == "SOLVER_CERTIFIED").mean() * 100
        fallback_p = (sub["cert_status_persp"] == "SOLVER_CERTIFIED").mean() * 100
        mean_t_b = sub["time_basic_total_ms"].mean()
        mean_t_p = sub["time_persp_total_ms"].mean()
        speedup = (1.0 - mean_t_p / mean_t_b) * 100
        
        print(f"\n[Epsilon_cert = {eps:.0e}]")
        print(f"  Early Certification Rate: Basic {early_b:.1f}% vs Persp {early_p:.1f}%")
        print(f"  Mean Recovery Depth L_term: Basic {mean_l_b:.2f} vs Persp {mean_l_p:.2f}")
        print(f"  Full-Fallback Rate: Basic {fallback_b:.1f}% vs Persp {fallback_p:.1f}%")
        print(f"  End-to-End Runtime: Basic {mean_t_b:.1f}ms vs Persp {mean_t_p:.1f}ms (Speedup: {speedup:.1f}%)")
    print("=" * 75)
    
    assert lb_ordering_violations == 0, f"Invariant failed: {lb_ordering_violations} LB-ordering violations"
    assert integer_equiv_violations == 0, f"Invariant failed: {integer_equiv_violations} integer-equivalence violations"


if __name__ == "__main__":
    run_perspective_end_to_end()
