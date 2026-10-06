"""
Phase 10F: Certification & Nested Recovery Scaling (Table F).
Measures how continuous relaxation time, certificate gap, recovery stages,
and full fallback rates scale with problem dimension (N, H).
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from learning_augmented_portfolio.optimization.variables import SolverConfig
from learning_augmented_portfolio.scalability.instance_generator import ScalableInstanceGenerator
from learning_augmented_portfolio.scalability.runner import ScalabilityRunner
from learning_augmented_portfolio.scalability.experiment_grid import generate_nested_budget_sequence


def main():
    print("=" * 80)
    print("PHASE 10F: CERTIFICATION & NESTED RECOVERY SCALING (Table F)")
    print("=" * 80)

    artifacts_dir = Path("artifacts/scalability")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Valid configurations within Gurobi community license
    test_configs = [
        {"H": 1, "N": 10, "K": 3},
        {"H": 1, "N": 25, "K": 5},
        {"H": 1, "N": 30, "K": 6},
        {"H": 3, "N": 8, "K": 3},
        {"H": 3, "N": 10, "K": 3},
        {"H": 5, "N": 5, "K": 2},
    ]

    ratios = [0.20, 0.40, 0.60, 1.00]
    n_replicates = 5
    generator = ScalableInstanceGenerator()
    solver_config = SolverConfig(mip_gap=1e-4, time_limit=30.0)
    runner = ScalabilityRunner(solver_config=solver_config)

    records = []
    print(f"\n[1] Running certified recovery scaling across {len(test_configs)} configurations (R={n_replicates})...")

    for cfg in test_configs:
        H, N, K = cfg["H"], cfg["N"], cfg["K"]
        nested_budgets = generate_nested_budget_sequence(n_assets=N, ratios=ratios, cardinality_k=K)
        print(f"\n--> Config: H={H}, N={N}, K={K}, Budgets={nested_budgets}")

        for seed in range(42, 42 + n_replicates):
            instance = generator.generate_instance(
                n_assets=N,
                horizon=H,
                cardinality_limit=K,
                seed=seed,
            )

            # Solve Certified Recovery
            res = runner.solve_certified_recovery(instance, nested_budgets)

            # Check if fell back to full universe (last budget)
            fell_back_to_full = (res.candidate_size >= N)

            records.append({
                "H": H,
                "N": N,
                "K": K,
                "Seed": seed,
                "Status": res.status.value,
                "RelaxationTime_sec": res.timers.t_relax_sec,
                "SolverTime_sec": res.timers.t_solver_sec,
                "E2E_Time_sec": res.timers.total_e2e_sec,
                "RelaxationCertGap": res.relaxation_cert_gap,
                "RelaxationCertified": res.relaxation_certified,
                "FullMIPCertified": res.full_mip_certified,
                "AnyValidCertificate": res.any_valid_certificate,
                "RecoveryStages": res.recovery_stages,
                "FinalCandidateBudget": res.candidate_size,
                "FullFallback": fell_back_to_full,
                "Work": res.work,
                "NodeCount": res.node_count,
            })

    df_records = pd.DataFrame(records)

    # Aggregate into Table F
    summary_rows = []
    for (H, N, K), group in df_records.groupby(["H", "N", "K"]):
        n_inst = len(group)
        valid_gaps = group["RelaxationCertGap"].dropna()

        summary_rows.append({
            "H": H,
            "N": N,
            "K": K,
            "Replicates": n_inst,
            "Mean_RelaxationTime_sec": group["RelaxationTime_sec"].mean(),
            "Median_RelaxationTime_sec": group["RelaxationTime_sec"].median(),
            "Mean_RelaxationCertGap": valid_gaps.mean() if len(valid_gaps) > 0 else np.nan,
            "Median_RelaxationCertGap": valid_gaps.median() if len(valid_gaps) > 0 else np.nan,
            "RelaxationCertifiedRate": group["RelaxationCertified"].mean(),
            "AnyCertifiedRate": group["AnyValidCertificate"].mean(),
            "Mean_RecoveryStages": group["RecoveryStages"].mean(),
            "Median_RecoveryStages": group["RecoveryStages"].median(),
            "Mean_FinalBudget": group["FinalCandidateBudget"].mean(),
            "FullFallbackRate": group["FullFallback"].mean(),
            "Median_E2E_Time_sec": group["E2E_Time_sec"].median(),
        })

    df_table_f = pd.DataFrame(summary_rows)
    table_f_path = artifacts_dir / "table_f_certification_scaling.csv"
    df_table_f.to_csv(table_f_path, index=False)
    print(f"\n[2] Saved Table F to {table_f_path}")

    print("\nSUMMARY TABLE F (Certification & Recovery Scaling):")
    cols_to_print = [
        "H", "N", "K", "Median_RelaxationTime_sec", "Median_RelaxationCertGap",
        "RelaxationCertifiedRate", "Mean_RecoveryStages", "FullFallbackRate", "Median_E2E_Time_sec"
    ]
    print(df_table_f[cols_to_print].to_string(index=False))


if __name__ == "__main__":
    main()
