"""
Phase 10 Experiment 10D: Reduced Solver Scaling & E2E Acceleration (Tables C & D).
Evaluates paired execution on identical synthetic instances:
  - M0: Full MIQP
  - M1: Fast MarketOnly Candidate MIQP
  - M2: ContinuousRelaxation Ranking + Reduced MIQP
  - M5: Certified Recovery
  - M6: Oracle Support Diagnostic (Offline diagnostic only)
Measures:
  - Component timers (T_rank, T_relax, T_build, T_solver, T_E2E)
  - Speedup and Runtime Reduction
  - Exact Objective Recovery and Relative Degradation
  - Nodes and Gurobi Work
Produces Table C (Reduced Solver Scaling) and Table D (End-to-End Acceleration).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from learning_augmented_portfolio.scalability.instance_generator import ScalableInstanceGenerator
from learning_augmented_portfolio.scalability.runner import ScalabilityRunner
from learning_augmented_portfolio.scalability.aggregation import compute_robust_stats
from learning_augmented_portfolio.scalability.experiment_grid import (
    CardinalityRegime,
    compute_cardinality,
    compute_candidate_budget,
    generate_nested_budget_sequence,
)
from learning_augmented_portfolio.evaluation.scalability_metrics import (
    compute_speedup,
    compute_runtime_reduction,
    compute_objective_quality,
)
from learning_augmented_portfolio.optimization.variables import SolverConfig


def run_exp10d_reduced_scaling() -> None:
    print("=" * 80)
    print("PHASE 10D: REDUCED SOLVER SCALING & END-TO-END ACCELERATION (Tables C & D)")
    print("=" * 80)

    gen = ScalableInstanceGenerator()
    runner = ScalabilityRunner(SolverConfig(time_limit=15.0, mip_gap=1e-4, threads=1))

    # Supported evaluation configurations for empirical paired comparison
    eval_configs = [
        # (H, N)
        (1, 10),
        (1, 25),
        (1, 30),
        (3, 8),
        (3, 10),
        (5, 5),
    ]

    ratios_b = [0.20, 0.40, 0.60]
    seeds = [42, 43, 44, 45, 46]  # R=5 paired replicates

    c_rows = []  # Table C: Reduced Solver Scaling
    d_rows = []  # Table D: E2E Acceleration Comparison

    print(f"\n[1] Running paired evaluations across {len(eval_configs)} (H, N) configs and candidate budgets...")

    for H, N in eval_configs:
        K_port = compute_cardinality(N, CardinalityRegime.FIXED, fixed_k=4)

        for seed in seeds:
            inst = gen.generate_instance(n_assets=N, horizon=H, cardinality_limit=K_port, seed=seed)

            # 1. Full MIQP (Baseline M0)
            res_full = runner.solve_full_miqp(inst, record_anytime=False)
            t_full_e2e = res_full.timers.total_e2e_sec
            j_full = res_full.objective_value
            full_opt = bool(res_full.status.value == "OPTIMAL")

            # True Teacher Solution Indicators for Oracle diagnostic (M6)
            teacher_indicators_y = res_full.holding_indicators if (full_opt and res_full.holding_indicators is not None) else None

            # M6: True Oracle Diagnostic (evaluated once per instance on true support C_t^oracle)
            res_m6 = runner.solve_oracle_support(inst, teacher_support_or_y=teacher_indicators_y if teacher_indicators_y is not None else {0})

            for rho_b in ratios_b:
                n_held = int(np.sum(inst.actual_state.normalized_weights > 1e-6))
                B = compute_candidate_budget(N, rho_b, cardinality_k=K_port, n_held=n_held)

                # M1: Fast MarketOnly
                res_m1 = runner.solve_fast_market_only(inst, candidate_budget_B=B)

                # M2: Continuous Relaxation
                res_m2 = runner.solve_continuous_relaxation_ranking(inst, candidate_budget_B=B)

                # M5: Certified Recovery
                ladder = generate_nested_budget_sequence(N, [rho_b, min(rho_b + 0.20, 1.0), 1.0], K_port, n_held)
                res_m5 = runner.solve_certified_recovery(inst, nested_budgets=ladder)

                for m_res in [res_m1, res_m2, res_m5, res_m6]:
                    t_e2e = m_res.timers.total_e2e_sec
                    speedup = compute_speedup(t_full_e2e, t_e2e)
                    reduction = compute_runtime_reduction(t_full_e2e, t_e2e)
                    q = compute_objective_quality(m_res.objective_value, j_full, full_opt)

                    row_entry = {
                        "H": H,
                        "N": N,
                        "Candidate_B": m_res.candidate_size,
                        "CandidateRatio": f"{m_res.candidate_size / N:.1%}",
                        "Method": m_res.method_name,
                        "Seed": seed,
                        "T_solver_ms": m_res.timers.t_solver_sec * 1000.0,
                        "T_e2e_ms": t_e2e * 1000.0,
                        "T_full_e2e_ms": t_full_e2e * 1000.0,
                        "Speedup": speedup,
                        "RuntimeReduction": reduction,
                        "Nodes": m_res.node_count,
                        "Work": m_res.work,
                        "ObjDegradation": q["abs_degradation"],
                        "ExactRecovery": 1.0 if q["is_exact_recovery"] else 0.0,
                        "Stages": m_res.recovery_stages,
                        "IsDeployable": m_res.is_deployable,
                    }
                    c_rows.append(row_entry)

    raw_df = pd.DataFrame(c_rows)

    # 2. Aggregate into Table C (Reduced Solver Scaling) and Table D (E2E Acceleration)
    print("\n[2] Aggregating results into summary tables...")

    # Table C: Grouped by H, N, Method, CandidateRatio
    c_summary = raw_df.groupby(["H", "N", "Method", "CandidateRatio"]).agg({
        "Candidate_B": "first",
        "T_solver_ms": ["median", "mean"],
        "T_e2e_ms": ["median", "mean"],
        "Nodes": "median",
        "Work": "median",
        "ExactRecovery": "mean",
        "ObjDegradation": "mean",
    }).reset_index()

    c_summary.columns = [
        "H", "N", "Method", "CandidateRatio", "Candidate_B",
        "Median_Solver_ms", "Mean_Solver_ms",
        "Median_E2E_ms", "Mean_E2E_ms",
        "Median_Nodes", "Median_Work",
        "ExactRecoveryRate", "Mean_ObjDegradation",
    ]

    # Table D: Acceleration summary vs Full MIQP
    d_summary = raw_df.groupby(["H", "N", "Method"]).agg({
        "CandidateRatio": "first",
        "Speedup": ["median", "mean"],
        "RuntimeReduction": "median",
        "ExactRecovery": "mean",
        "ObjDegradation": "mean",
        "IsDeployable": "first",
    }).reset_index()

    d_summary.columns = [
        "H", "N", "Method", "SampleRatio",
        "Median_Speedup", "Mean_Speedup",
        "Median_RuntimeReduction", "ExactRecoveryRate",
        "Mean_ObjDegradation", "IsDeployable",
    ]

    out_dir = Path("artifacts/scalability")
    out_dir.mkdir(parents=True, exist_ok=True)
    c_summary.to_csv(out_dir / "table_c_reduced_scaling.csv", index=False)
    d_summary.to_csv(out_dir / "table_d_e2e_acceleration.csv", index=False)

    print(f"    Saved Table C to {out_dir / 'table_c_reduced_scaling.csv'}")
    print(f"    Saved Table D to {out_dir / 'table_d_e2e_acceleration.csv'}")

    print("\nSAMPLE ROWS FROM TABLE D (E2E Acceleration Summary):")
    print(d_summary[["H", "N", "Method", "Median_Speedup", "Median_RuntimeReduction", "ExactRecoveryRate", "Mean_ObjDegradation", "IsDeployable"]].to_string(index=False))


if __name__ == "__main__":
    run_exp10d_reduced_scaling()
