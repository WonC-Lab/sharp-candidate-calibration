"""Small-scale smoke test suite executing prior to large Monte Carlo experiments.

Tests:
1. Portfolio MIQP: instances covering N=5, H=3 and N=10, H=1.
2. CFLP: IID instances covering N=8-12, M=15-30.
Validates:
- No budget below B_{min, C} is ever queried.
- Reference-match and certified-epsilon exactness budgets are computed and separated.
- Certifications are evaluated based on numerical bound gaps, not status alone.
- Monotone UB envelope UB_best(B_{l+1}) <= UB_best(B_l) is tracked.
- Strictly increasing recovery schedule terminates.
- Complete per-instance logging schema is populated without error.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from learning_augmented_portfolio.dataset.serialization import load_instance
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
from learning_augmented_portfolio.theory.exactness_budget import (
    compute_reference_match_budget,
    compute_certified_epsilon_budget,
    compute_certified_epsilon_budget_record,
    compute_entity_entry_budgets,
    compute_support_entry_depth,
)
from learning_augmented_portfolio.conformal.conformal_budget import (
    calibrate_marginal_budget,
    calibrate_adaptive_budget,
)
from learning_augmented_portfolio.optimization.recovery_schedule import (
    StrictlyIncreasingRecoveryLadder,
    compute_monotone_ub_envelope,
    evaluate_recovery_stages,
)
from learning_augmented_portfolio.optimization.certification import (
    relaxation_certificate,
    full_mip_certificate,
)
from learning_augmented_portfolio.benchmarks.repeated_cflp import (
    generate_iid_cflp_instances,
    build_cflp_candidate_family,
    solve_cflp,
)


def run_portfolio_smoke_test():
    print("\n--- Running Portfolio MIQP Smoke Test ---")
    files_n5 = glob.glob("artifacts/instances/train/instance_SP500_SUB5_*.npz")[:3]
    files_n10 = glob.glob("artifacts/instances_medium/train/instance_UNIV_N10_*.npz")[:3]
    
    test_files = files_n5 + files_n10
    solver_cfg = SolverConfig(output_flag=False, time_limit=5.0, mip_gap=1e-5)
    logs = []
    
    for npz_path in test_files:
        inst = load_instance(npz_path)
        params = inst.params
        state = inst.market_state
        N = inst.metadata.n_assets
        H = inst.metadata.horizon
        
        # Solve Full Reference
        full_opt = MultiPeriodMIQPOptimizer(params, solver_cfg)
        full_res = full_opt.solve(state)
        assert full_res.is_feasible, f"Portfolio instance {npz_path} infeasible"
        ub_full = full_res.objective_value
        lb_full = full_res.best_bound if full_res.best_bound is not None else ub_full - 1e-5
        
        mandatory = [i for i in range(N) if state.initial_holdings[i] > 1e-6]
        B_min_C = max(1, len(mandatory))
        
        # Continuous relaxation lower bound
        relax_opt = MultiPeriodRelaxedQPOptimizer(params, solver_cfg)
        relax_res = relax_opt.solve(state)
        lb_relax = relax_res.objective_value if (relax_res.is_feasible and relax_res.objective_value is not None) else lb_full - 1e-3
        
        # Candidate ranking
        scores = np.mean(state.expected_returns, axis=0)
        non_mand_ranked = [i for i in np.argsort(-scores) if i not in mandatory]
        
        cand_family = {}
        cand_upper_bounds = {}
        for B in range(B_min_C, N + 1):
            cand_indices = mandatory + non_mand_ranked[:(B - len(mandatory))]
            cand_family[B] = set(cand_indices)
            
            cand_univ = CandidateUniverse(candidate_indices=tuple(sorted(cand_indices)), full_universe_size=N)
            red_opt = MultiPeriodReducedMIQPOptimizer(params, solver_cfg)
            exec_res = red_opt.solve(state, cand_univ)
            r = exec_res.optimization_result
            if r.is_feasible and r.objective_value is not None:
                cand_upper_bounds[B] = r.objective_value
                
        # Compute budgets
        ref_dict = compute_reference_match_budget(cand_upper_bounds, ub_full, epsilon_ref=1e-4, B_min=B_min_C, N=N)
        b_ref = ref_dict["budget"]
        reduced_better = ref_dict["reduced_better_than_reference"]
        
        cert_record = compute_certified_epsilon_budget_record(
            cand_upper_bounds, full_upper_bound=ub_full, full_lower_bound=lb_relax, epsilon_cert=1e-3, B_min=B_min_C, N=N
        )
        b_cert = cert_record["b_dagger_cert_eps"]
        
        # Conformal calibration
        conformal_res = calibrate_marginal_budget(
            calibration_budgets=[b_ref, b_ref + 1, b_ref],
            alpha=0.10,
            N=N,
            B_min_new=B_min_C
        )
        B_alpha = conformal_res["B_alpha"]
        assert B_alpha >= B_min_C
        
        # Recovery schedule
        ladder_gen = StrictlyIncreasingRecoveryLadder(gamma=1.5, N=N)
        ladder = ladder_gen.generate_ladder(B_1=B_alpha, B_min=B_min_C)
        assert ladder[0] >= B_min_C
        
        L_term = len(ladder)
        L_cert = float("inf")
        cert_status = "UNCERTIFIED"
        term_reason = "MAX_DEPTH"
        raw_ubs = []
        
        for stage_idx, B_curr in enumerate(ladder, start=1):
            ub_cand = cand_upper_bounds.get(B_curr, ub_full)
            raw_ubs.append(ub_cand)
            cert_eval = relaxation_certificate(ub_cand, lb_relax, epsilon_cert=1e-3)
            if cert_eval["certified"]:
                L_term = stage_idx
                L_cert = stage_idx
                cert_status = "RELAXATION_CERTIFIED"
                term_reason = "CERTIFIED"
                break
                
        ub_envelope = compute_monotone_ub_envelope(raw_ubs)
        assert all(ub_envelope[i+1] <= ub_envelope[i] for i in range(len(ub_envelope)-1))
        
        stages = evaluate_recovery_stages(ladder, b_ref, b_cert, L_term, L_cert, raw_upper_bounds=raw_ubs)
        
        inst_id = os.path.basename(npz_path).replace(".npz", "")
        log_entry = {
            "instance_id": inst_id,
            "domain": "portfolio",
            "candidate_family": "expected_return",
            "ranking_method": "mean_return",
            "B_min_C": B_min_C,
            "B_alpha": B_alpha,
            "B_dagger_ref_eps": b_ref,
            "B_dagger_cert_eps": b_cert,
            "cert_budget_exists": cert_record["cert_budget_exists"],
            "cert_budget_value": cert_record["cert_budget_value"],
            "full_universe_certified": cert_record["full_universe_certified"],
            "full_universe_abs_gap": cert_record["full_universe_abs_gap"],
            "full_universe_rel_gap": cert_record["full_universe_rel_gap"],
            "reduced_better_than_reference": reduced_better,
            "epsilon_ref": 1e-4,
            "epsilon_cert": 1e-3,
            "L_exact_ref": stages["L_exact_ref"],
            "L_exact_cert": stages["L_exact_cert"],
            "L_term": L_term,
            "L_cert": L_cert,
            "certification_status": cert_status,
            "certificate_type": "RELAXATION_CERTIFICATE" if cert_status != "UNCERTIFIED" else "NONE",
            "termination_reason": term_reason,
            "UB_candidate": raw_ubs[-1],
            "UB_best_envelope": ub_envelope[-1],
            "LB_relax": lb_relax,
            "UB_full": ub_full,
            "LB_full": lb_full,
            "mip_abs_gap": full_res.abs_gap if hasattr(full_res, "abs_gap") and full_res.abs_gap is not None else 0.0,
            "mip_rel_gap": full_res.rel_gap if hasattr(full_res, "rel_gap") and full_res.rel_gap is not None else 0.0,
            "cumulative_dimension_to_exact_ref": stages["cum_dim_to_exact_ref"],
            "cumulative_dimension_to_exact_cert": stages["cum_dim_to_exact_cert"],
            "cumulative_dimension_to_term": stages["cum_dim_to_term"],
            "full_fallback": (ladder[L_term - 1] == N),
            "solver_status": full_res.status,
            "runtime_ms": full_res.solve_time * 1000.0
        }
        logs.append(log_entry)
        print(f"Portfolio {inst_id} (N={N}, H={H}): B_min={B_min_C}, B_ref={b_ref}, B_cert={b_cert}, L_term={L_term}, L_cert={L_cert}")
        
    return logs


def run_cflp_smoke_test(num_instances: int = 5):
    print("\n--- Running CFLP Smoke Test ---")
    # CFLP covering N=8-12, M=15-30
    instances_8 = generate_iid_cflp_instances(num_instances=3, N=8, M=15, seed=42)
    instances_12 = generate_iid_cflp_instances(num_instances=3, N=12, M=25, seed=84)
    instances = instances_8 + instances_12
    logs = []
    
    for inst in instances:
        cand_family, B_min_C, ranking = build_cflp_candidate_family(inst, ranking_method="efficiency")
        
        full_res = solve_cflp(inst, candidate_set=None, time_limit=5.0)
        assert full_res["feasible"], f"CFLP {inst.instance_id} was infeasible"
        ub_full = full_res["upper_bound"]
        lb_full = full_res["lower_bound"]
        
        cand_upper_bounds = {}
        for B in range(B_min_C, inst.N + 1):
            cand_res = solve_cflp(inst, candidate_set=cand_family[B], time_limit=3.0)
            if cand_res["feasible"]:
                cand_upper_bounds[B] = cand_res["upper_bound"]
                
        ref_dict = compute_reference_match_budget(cand_upper_bounds, ub_full, epsilon_ref=1e-4, B_min=B_min_C, N=inst.N)
        b_ref = ref_dict["budget"]
        reduced_better = ref_dict["reduced_better_than_reference"]
        
        cert_record = compute_certified_epsilon_budget_record(
            cand_upper_bounds, full_upper_bound=ub_full, full_lower_bound=lb_full, epsilon_cert=1e-3, B_min=B_min_C, N=inst.N
        )
        b_cert = cert_record["b_dagger_cert_eps"]
        
        conformal_res = calibrate_marginal_budget(
            calibration_budgets=[b_ref, b_ref + 1, b_ref],
            alpha=0.10,
            N=inst.N,
            B_min_new=B_min_C
        )
        B_alpha = conformal_res["B_alpha"]
        assert B_alpha >= B_min_C
        
        ladder_gen = StrictlyIncreasingRecoveryLadder(gamma=1.5, N=inst.N)
        ladder = ladder_gen.generate_ladder(B_1=B_alpha, B_min=B_min_C)
        assert ladder[0] >= B_min_C
        
        L_term = len(ladder)
        L_cert = float("inf")
        cert_status = "UNCERTIFIED"
        term_reason = "MAX_DEPTH"
        raw_ubs = []
        
        for stage_idx, B_curr in enumerate(ladder, start=1):
            ub_cand = cand_upper_bounds.get(B_curr, ub_full)
            raw_ubs.append(ub_cand)
            cert_eval = relaxation_certificate(ub_cand, lb_full, epsilon_cert=1e-3)
            if cert_eval["certified"]:
                L_term = stage_idx
                L_cert = stage_idx
                cert_status = "RELAXATION_CERTIFIED"
                term_reason = "CERTIFIED"
                break
                
        ub_envelope = compute_monotone_ub_envelope(raw_ubs)
        assert all(ub_envelope[i+1] <= ub_envelope[i] for i in range(len(ub_envelope)-1))
        
        stages = evaluate_recovery_stages(ladder, b_ref, b_cert, L_term, L_cert, raw_upper_bounds=raw_ubs)
        
        log_entry = {
            "instance_id": inst.instance_id,
            "domain": "cflp",
            "candidate_family": "efficiency",
            "ranking_method": "efficiency",
            "B_min_C": B_min_C,
            "B_alpha": B_alpha,
            "B_dagger_ref_eps": b_ref,
            "B_dagger_cert_eps": b_cert,
            "cert_budget_exists": cert_record["cert_budget_exists"],
            "cert_budget_value": cert_record["cert_budget_value"],
            "full_universe_certified": cert_record["full_universe_certified"],
            "full_universe_abs_gap": cert_record["full_universe_abs_gap"],
            "full_universe_rel_gap": cert_record["full_universe_rel_gap"],
            "reduced_better_than_reference": reduced_better,
            "epsilon_ref": 1e-4,
            "epsilon_cert": 1e-3,
            "L_exact_ref": stages["L_exact_ref"],
            "L_exact_cert": stages["L_exact_cert"],
            "L_term": L_term,
            "L_cert": L_cert,
            "certification_status": cert_status,
            "certificate_type": "RELAXATION_CERTIFICATE" if cert_status != "UNCERTIFIED" else "NONE",
            "termination_reason": term_reason,
            "UB_candidate": raw_ubs[-1],
            "UB_best_envelope": ub_envelope[-1],
            "LB_relax": lb_full,
            "UB_full": ub_full,
            "LB_full": lb_full,
            "mip_abs_gap": full_res["abs_gap"],
            "mip_rel_gap": full_res["rel_gap"],
            "cumulative_dimension_to_exact_ref": stages["cum_dim_to_exact_ref"],
            "cumulative_dimension_to_exact_cert": stages["cum_dim_to_exact_cert"],
            "cumulative_dimension_to_term": stages["cum_dim_to_term"],
            "full_fallback": (ladder[L_term - 1] == inst.N),
            "solver_status": "OPTIMAL" if full_res["abs_gap"] <= 1e-4 else "SUBOPTIMAL",
            "runtime_ms": full_res["runtime_ms"]
        }
        logs.append(log_entry)
        print(f"CFLP {inst.instance_id} (N={inst.N}, M={inst.M}): B_min={B_min_C}, B_ref={b_ref}, B_cert={b_cert}, L_term={L_term}, L_cert={L_cert}")
        
    return logs


def main():
    portfolio_logs = run_portfolio_smoke_test()
    cflp_logs = run_cflp_smoke_test()
    all_logs = portfolio_logs + cflp_logs
    
    if len(all_logs) > 0:
        df = pd.DataFrame(all_logs)
        out_path = "artifacts/smoke_test_logs.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSmoke test complete: saved {len(df)} logs to {out_path}")
        print("Logged fields count:", len(df.columns))
        print("Columns:", list(df.columns))
        
        # Verify invariants
        assert (df["B_alpha"] < df["B_min_C"]).sum() == 0, "Invariant violation: B_alpha < B_min_C"
        print("Invariant check PASSED: B_alpha >= B_min_C for all instances.")
    else:
        print("Error: No logs generated!")
        sys.exit(1)


if __name__ == "__main__":
    main()
