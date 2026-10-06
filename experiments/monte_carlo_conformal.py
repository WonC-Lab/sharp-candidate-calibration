"""Monte Carlo conformal calibration experiment suite.

Executes:
1. M = 100 independent Monte Carlo replications on repeated IID CFLP instances.
2. Evaluates separately:
   A. Empirical reference-match diagnostic target B^{dagger, ref}
   B. Certified epsilon-quality operational target B^{dagger, cert}
3. Across alpha in {0.20, 0.10, 0.05} with n_cal >= 60 to guarantee non-trivial calibration.
4. Computes replication-level coverage, candidate ratio B_alpha / N, Monte Carlo standard error,
   and 95% bootstrap confidence intervals over independent replications.
"""
import os
import sys
import math
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from learning_augmented_portfolio.benchmarks.repeated_cflp import (
    generate_iid_cflp_instances,
    build_cflp_candidate_family,
    solve_cflp,
)
from learning_augmented_portfolio.theory.exactness_budget import (
    compute_reference_match_budget,
    compute_certified_epsilon_budget,
    compute_certified_epsilon_budget_record,
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


def run_monte_carlo_conformal_cflp(
    M_replications: int = 100,
    n_train: int = 20,
    n_cal: int = 65,
    n_test: int = 20,
    N_facilities: int = 8,
    M_customers: int = 15,
    alphas: list = [0.20, 0.10, 0.05],
    seed: int = 2026
):
    print("=" * 80)
    print(f"RUNNING INDEPENDENT MONTE CARLO CONFORMAL EXPERIMENT (M = {M_replications} REPLICATIONS)")
    print(f"Configuration: N={N_facilities}, M_cust={M_customers}, n_train={n_train}, n_cal={n_cal} (n_cal>=60), n_test={n_test}")
    print("=" * 80)
    
    total_per_rep = n_train + n_cal + n_test
    rep_records = []
    detailed_instance_logs = []
    
    rng = np.random.default_rng(seed)
    
    t_start_total = time.perf_counter()
    
    for rep in range(1, M_replications + 1):
        rep_seed = int(rng.integers(1, 10000000))
        instances = generate_iid_cflp_instances(
            num_instances=total_per_rep,
            N=N_facilities,
            M=M_customers,
            seed=rep_seed
        )
        
        train_insts = instances[:n_train]
        cal_insts = instances[n_train:n_train + n_cal]
        test_insts = instances[n_train + n_cal:]
        
        def get_instance_budgets(inst):
            cand_family, B_min_C, ranking = build_cflp_candidate_family(inst, ranking_method="efficiency")
            full_res = solve_cflp(inst, candidate_set=None, time_limit=5.0)
            ub_full = full_res["upper_bound"]
            lb_full = full_res["lower_bound"]
            
            c_bounds = {}
            cand_runtimes = {}
            for B in range(B_min_C, inst.N + 1):
                sub_res = solve_cflp(inst, candidate_set=cand_family[B], time_limit=3.0)
                if sub_res["feasible"]:
                    c_bounds[B] = sub_res["upper_bound"]
                    cand_runtimes[B] = sub_res["runtime_ms"]
                    
            ref_dict = compute_reference_match_budget(c_bounds, ub_full, epsilon_ref=1e-4, B_min=B_min_C, N=inst.N)
            b_ref = ref_dict["budget"]
            
            cert_rec = compute_certified_epsilon_budget_record(
                c_bounds, full_upper_bound=ub_full, full_lower_bound=lb_full, epsilon_cert=1e-3, B_min=B_min_C, N=inst.N
            )
            b_cert = cert_rec["b_dagger_cert_eps"]
            
            return {
                "instance": inst,
                "B_min_C": B_min_C,
                "ub_full": ub_full,
                "lb_full": lb_full,
                "c_bounds": c_bounds,
                "cand_runtimes": cand_runtimes,
                "b_ref": b_ref,
                "b_cert": b_cert,
                "cert_exists": cert_rec["cert_budget_exists"],
                "full_certified": cert_rec["full_universe_certified"],
                "runtime_ms": full_res["runtime_ms"]
            }

        train_data = [get_instance_budgets(inst) for inst in train_insts]
        cal_data = [get_instance_budgets(inst) for inst in cal_insts]
        test_data = [get_instance_budgets(inst) for inst in test_insts]
        
        # Fit predictor b_hat(X) on train_data for reference target
        X_train = np.array([td["instance"].features for td in train_data])
        y_train_ref = np.array([td["b_ref"] for td in train_data], dtype=float)
        
        X_design = np.column_stack([np.ones(len(X_train)), X_train])
        reg_eye = 1.0 * np.eye(X_design.shape[1])
        reg_eye[0, 0] = 0.0
        weights_ref = np.linalg.solve(X_design.T @ X_design + reg_eye, X_design.T @ y_train_ref)
        
        def predict_b_ref(feats):
            x_vec = np.concatenate([[1.0], feats])
            return float(np.dot(weights_ref, x_vec))
            
        cal_residuals_ref = [cd["b_ref"] - predict_b_ref(cd["instance"].features) for cd in cal_data]
        cal_budgets_ref = [cd["b_ref"] for cd in cal_data]
        cal_budgets_cert = [cd["b_cert"] for cd in cal_data]
        
        # Uncertified label rate on calibration set
        uncert_cal_rate = np.mean([1.0 if math.isinf(b) else 0.0 for b in cal_budgets_cert])
        
        for alpha in alphas:
            cal_marg_ref = calibrate_marginal_budget(cal_budgets_ref, alpha=alpha, N=N_facilities, target_type="reference")
            cal_marg_cert = calibrate_marginal_budget(cal_budgets_cert, alpha=alpha, N=N_facilities, target_type="certified_epsilon")
            
            cov_marg_ref = 0
            cov_adapt_ref = 0
            cov_marg_cert = 0
            
            b_alpha_marg_ref_list = []
            b_alpha_adapt_ref_list = []
            b_alpha_marg_cert_list = []
            
            l_exact_ref_list = []
            l_exact_cert_list = []
            l_term_ref_list = []
            l_cert_ref_list = []
            runtime_ref_list = []
            
            for td in test_data:
                inst = td["instance"]
                b_min_c = td["B_min_C"]
                
                # A. Marginal Reference Target
                b_marg_ref = max(b_min_c, cal_marg_ref["B_alpha"])
                hit_marg_ref = (td["b_ref"] <= b_marg_ref)
                cov_marg_ref += int(hit_marg_ref)
                b_alpha_marg_ref_list.append(b_marg_ref)
                
                # Adaptive Reference Target
                b_pred = predict_b_ref(inst.features)
                cal_adapt_ref = calibrate_adaptive_budget(
                    calibration_residuals=cal_residuals_ref,
                    predicted_budget_new=b_pred,
                    alpha=alpha,
                    N=N_facilities,
                    B_min_new=b_min_c,
                    target_type="reference"
                )
                b_adapt_ref = cal_adapt_ref["B_alpha"]
                hit_adapt_ref = (td["b_ref"] <= b_adapt_ref)
                cov_adapt_ref += int(hit_adapt_ref)
                b_alpha_adapt_ref_list.append(b_adapt_ref)
                
                # B. Certified Epsilon Target
                b_marg_cert = max(b_min_c, cal_marg_cert["B_alpha"])
                hit_marg_cert = (td["b_cert"] <= b_marg_cert) if not math.isinf(td["b_cert"]) else False
                cov_marg_cert += int(hit_marg_cert)
                b_alpha_marg_cert_list.append(b_marg_cert)
                
                # Recovery simulation on adaptive ref
                ladder_gen = StrictlyIncreasingRecoveryLadder(gamma=1.5, N=N_facilities)
                ladder = ladder_gen.generate_ladder(B_1=b_adapt_ref, B_min=b_min_c)
                
                raw_ubs = [td["c_bounds"].get(B, td["ub_full"]) for B in ladder]
                envelope = compute_monotone_ub_envelope(raw_ubs)
                
                l_term = len(ladder)
                l_cert = float("inf")
                run_t = td["runtime_ms"]
                
                for s_idx, B_curr in enumerate(ladder, start=1):
                    ub_c = td["c_bounds"].get(B_curr, td["ub_full"])
                    run_t += td["cand_runtimes"].get(B_curr, 0.0)
                    if ub_c - td["lb_full"] <= 1e-3 + 1e-9:
                        l_term = s_idx
                        l_cert = s_idx
                        break
                        
                stages = evaluate_recovery_stages(
                    ladder=ladder,
                    B_dagger_ref=td["b_ref"],
                    B_dagger_cert=td["b_cert"],
                    L_term=l_term,
                    L_cert=l_cert,
                    raw_upper_bounds=raw_ubs
                )
                
                l_exact_ref_list.append(stages["L_exact_ref"])
                l_exact_cert_list.append(stages["L_exact_cert"] if not math.isinf(stages["L_exact_cert"]) else N_facilities)
                l_term_ref_list.append(l_term)
                l_cert_ref_list.append(l_cert if not math.isinf(l_cert) else float(len(ladder)))
                runtime_ref_list.append(run_t)
                
                detailed_instance_logs.append({
                    "replication": rep,
                    "alpha": alpha,
                    "instance_id": inst.instance_id,
                    "domain": "cflp",
                    "B_min_C": b_min_c,
                    "B_dagger_ref": td["b_ref"],
                    "B_dagger_cert": td["b_cert"],
                    "B_alpha_adaptive_ref": b_adapt_ref,
                    "B_alpha_marginal_cert": b_marg_cert,
                    "hit_adaptive_ref": hit_adapt_ref,
                    "hit_marginal_cert": hit_marg_cert,
                    "L_exact_ref": stages["L_exact_ref"],
                    "L_exact_cert": stages["L_exact_cert"],
                    "L_term": l_term,
                    "L_cert": l_cert,
                    "runtime_ms": run_t
                })
                
            cov_marg_ref_rate = cov_marg_ref / n_test
            cov_adapt_ref_rate = cov_adapt_ref / n_test
            cov_marg_cert_rate = cov_marg_cert / n_test
            
            fallback_rate = np.mean([1.0 if b == N_facilities else 0.0 for b in b_alpha_adapt_ref_list])
            
            rep_records.append({
                "replication": rep,
                "alpha": alpha,
                "nominal_coverage": 1.0 - alpha,
                "cov_marginal_ref": cov_marg_ref_rate,
                "cov_adaptive_ref": cov_adapt_ref_rate,
                "cov_marginal_cert": cov_marg_cert_rate,
                "mean_B_adaptive_ref": float(np.mean(b_alpha_adapt_ref_list)),
                "candidate_ratio_adaptive": float(np.mean(b_alpha_adapt_ref_list)) / N_facilities,
                "mean_B_marginal_cert": float(np.mean(b_alpha_marg_cert_list)),
                "candidate_ratio_cert": float(np.mean(b_alpha_marg_cert_list)) / N_facilities,
                "fallback_to_N_rate": fallback_rate,
                "uncertified_label_rate": uncert_cal_rate,
                "mean_L_exact_ref": float(np.mean(l_exact_ref_list)),
                "mean_L_exact_cert": float(np.mean(l_exact_cert_list)),
                "mean_L_term": float(np.mean(l_term_ref_list)),
                "mean_L_cert": float(np.mean(l_cert_ref_list)),
                "mean_runtime_ms": float(np.mean(runtime_ref_list))
            })
            
        if rep % 10 == 0 or rep == M_replications:
            elapsed = time.perf_counter() - t_start_total
            print(f"Completed replication {rep:3d}/{M_replications} | Elapsed: {elapsed:.1f}s")
            
    df_reps = pd.DataFrame(rep_records)
    df_details = pd.DataFrame(detailed_instance_logs)
    
    summary_list = []
    for alpha in alphas:
        sub = df_reps[df_reps["alpha"] == alpha]
        
        covs = sub["cov_adaptive_ref"].values
        mean_cov = float(np.mean(covs))
        se_cov = float(np.std(covs, ddof=1) / math.sqrt(len(covs)))
        
        boot_means = [np.mean(rng.choice(covs, size=len(covs), replace=True)) for _ in range(3000)]
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        
        cov_cert_vals = sub["cov_marginal_cert"].values
        mean_cov_cert = float(np.mean(cov_cert_vals))
        se_cov_cert = float(np.std(cov_cert_vals, ddof=1) / math.sqrt(len(cov_cert_vals)))
        
        ratios = sub["candidate_ratio_adaptive"].values
        mean_ratio = float(np.mean(ratios))
        
        summary_list.append({
            "alpha": alpha,
            "nominal_coverage": 1.0 - alpha,
            "mc_mean_coverage_ref": mean_cov,
            "mc_se_coverage_ref": se_cov,
            "ci_95_lower_ref": ci_lower,
            "ci_95_upper_ref": ci_upper,
            "cov_ref_p05": float(np.percentile(covs, 5)),
            "cov_ref_p50": float(np.percentile(covs, 50)),
            "cov_ref_p95": float(np.percentile(covs, 95)),
            "mc_mean_coverage_cert": mean_cov_cert,
            "mc_se_coverage_cert": se_cov_cert,
            "mean_candidate_ratio_B_over_N": mean_ratio,
            "fallback_to_N_rate": float(sub["fallback_to_N_rate"].mean()),
            "uncertified_label_rate": float(sub["uncertified_label_rate"].mean()),
            "mean_L_exact_ref": float(sub["mean_L_exact_ref"].mean()),
            "mean_L_exact_cert": float(sub["mean_L_exact_cert"].mean()),
            "mean_L_term": float(sub["mean_L_term"].mean()),
            "mean_L_cert": float(sub["mean_L_cert"].mean()),
            "mean_runtime_ms": float(sub["mean_runtime_ms"].mean()),
            "mc_replications": M_replications,
            "n_cal": n_cal,
            "n_test_per_rep": n_test
        })
        
    df_summary = pd.DataFrame(summary_list)
    
    os.makedirs("artifacts", exist_ok=True)
    summary_path = "artifacts/monte_carlo_conformal_summary.csv"
    reps_path = "artifacts/monte_carlo_conformal_replications.csv"
    details_path = "artifacts/monte_carlo_conformal_details.csv"
    
    df_summary.to_csv(summary_path, index=False)
    df_reps.to_csv(reps_path, index=False)
    df_details.to_csv(details_path, index=False)
    
    total_time = time.perf_counter() - t_start_total
    print("\n" + "=" * 80)
    print(f"MONTE CARLO CONFORMAL EXPERIMENT COMPLETE (M={M_replications}, Total Time: {total_time:.1f}s)")
    print("=" * 80)
    print(df_summary[["alpha", "nominal_coverage", "mc_mean_coverage_ref", "mc_se_coverage_ref", "ci_95_lower_ref", "ci_95_upper_ref", "mc_mean_coverage_cert", "mean_candidate_ratio_B_over_N", "fallback_to_N_rate"]].to_string(index=False))
    return df_summary


if __name__ == "__main__":
    run_monte_carlo_conformal_cflp(
        M_replications=100,
        n_train=20,
        n_cal=65,
        n_test=20,
        N_facilities=8,
        M_customers=15,
        seed=2026
    )
