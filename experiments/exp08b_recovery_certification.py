"""
Phase 8 Experiment 8B: Nested Candidate Recovery and Full-Problem Certification.
Evaluates:
  - Nested candidate sequences B_1 < B_2 < ... < B_m = N
  - Continuous relaxation lower bounds LB_t and lifted feasible upper bounds UB_t
  - Full-problem certificate gap Gap^{cert} = UB_t - LB_t
  - Eventual full-universe fallback and recovery stage counts
  - Fast Mode vs Certified Mode end-to-end timing
  - Reuses relaxation cost without double counting
Generates:
  - Plot 1: Certificate gap vs total runtime
  - Plot 2: Initial candidate reduction vs average recovery stages
  - Plot 3: Fast-mode speedup vs certified-mode speedup
  - Plot 5: Certificate tolerance epsilon_{cert} vs runtime and final candidate size
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import List, Dict, Any, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from learning_augmented_portfolio.dataset.instance import OptimizationInstance
from learning_augmented_portfolio.dataset.manifest import DatasetManifest
from learning_augmented_portfolio.dataset.serialization import load_instance
from learning_augmented_portfolio.learning.active_set import (
    build_logistic_regression_predictor,
    build_hist_gradient_boosting_predictor,
)
from learning_augmented_portfolio.learning.baselines import (
    ExpectedReturnBaseline,
    ContinuousRelaxationYBaseline,
)
from learning_augmented_portfolio.learning.dataset import (
    SupervisedSplitDataset,
    build_supervised_datasets_by_split,
)
from learning_augmented_portfolio.optimization.variables import SolverConfig
from learning_augmented_portfolio.recovery.recovery_loop import NestedRecoveryOptimizer
from learning_augmented_portfolio.evaluation.certification_metrics import (
    CertificationEvaluationRecord,
    summarize_certification_evaluations,
)


def run_exp08b_recovery_certification() -> None:
    print("=" * 80)
    print("PHASE 8B: NESTED RECOVERY & CERTIFICATION EXPERIMENT")
    print("=" * 80)

    manifest_path = Path("artifacts/instances_medium/manifest.csv")
    manifest_df = pd.read_csv(manifest_path)
    base_dir = manifest_path.parent

    train_rows = manifest_df[manifest_df["split"] == "train"]
    test_rows = manifest_df[manifest_df["split"] == "test"]

    print(f"\n[1] Loading train instances ({len(train_rows)}) and test instances ({len(test_rows)})...")
    train_instances = [load_instance(base_dir / row["filename"]) for _, row in train_rows.iterrows()]
    test_instances = [load_instance(base_dir / row["filename"]) for _, row in test_rows.iterrows()]
    test_instances.sort(key=lambda inst: inst.metadata.time_index)

    print("\n[2] Training Machine Learning Predictors...")
    from learning_augmented_portfolio.learning.features import MARKET_ONLY_GROUPS, ALL_FEATURE_GROUPS
    train_ds_mo = build_supervised_datasets_by_split(train_instances, feature_groups=MARKET_ONLY_GROUPS)["train"]
    train_ds_ff = build_supervised_datasets_by_split(train_instances, feature_groups=ALL_FEATURE_GROUPS)["train"]

    market_only_model = build_logistic_regression_predictor(feature_groups=MARKET_ONLY_GROUPS, random_state=42)
    market_only_model.fit(train_ds_mo.X, train_ds_mo.y_union)

    full_feature_model = build_logistic_regression_predictor(feature_groups=ALL_FEATURE_GROUPS, random_state=42)
    full_feature_model.fit(train_ds_ff.X, train_ds_ff.y_union)

    methods = {
        "ExpectedReturn": (ExpectedReturnBaseline(), False, False),
        "ContinuousRelaxation": (ContinuousRelaxationYBaseline(), True, False),
        "MarketOnlyML": (market_only_model, False, True),
        "FullFeatureML": (full_feature_model, True, True),
    }

    budget_sequences = {
        3: [3, 4, 6, 10],
        4: [4, 6, 10],
        5: [5, 8, 10],
    }

    solver_cfg = SolverConfig(time_limit=15.0, mip_gap=1e-4, threads=1)
    recovery_opt = NestedRecoveryOptimizer(
        params=test_instances[0].params,
        solver_config=solver_cfg,
        epsilon_cert=1e-4,
        use_recovery_warm_starts=True,
    )

    all_records: List[CertificationEvaluationRecord] = []

    print("\n[3] Running Recovery & Certification evaluations across methods, budgets, and operating modes...")
    for init_b, seq in budget_sequences.items():
        for method_name, (ranker, req_relax, req_ml) in methods.items():
            for fast_mode in [True, False]:
                for inst in test_instances:
                    t_full = inst.metadata.runtime
                    j_star = inst.solution.objective_value if inst.solution else None

                    # Score prediction
                    scores = ranker.predict_instance_scores(inst)

                    # Determine reusable relaxation lower bound
                    if req_relax and inst.relaxation and inst.relaxation.is_feasible:
                        pre_lb = inst.relaxation.objective_value
                        pre_t_relax = inst.relaxation.solve_time
                    else:
                        pre_lb = None
                        pre_t_relax = 0.0

                    res = recovery_opt.solve(
                        instance=inst,
                        scores=scores,
                        budgets=seq,
                        precomputed_lower_bound=pre_lb,
                        precomputed_relaxation_time=pre_t_relax,
                        fast_mode=fast_mode,
                    )

                    true_deg = (res.final_objective - j_star) if (res.final_objective is not None and j_star is not None) else None
                    speedup = t_full / max(1e-4, res.total_runtime)

                    rec = CertificationEvaluationRecord(
                        instance_id=inst.metadata.instance_id,
                        method=method_name,
                        fast_mode=fast_mode,
                        initial_B=init_b,
                        final_B=res.final_budget,
                        initial_reduction=res.initial_reduction,
                        final_reduction=res.final_reduction,
                        stages_executed=res.stages_executed,
                        full_fallback_occurred=res.full_fallback_occurred,
                        is_certified=res.relaxation_certified,
                        relaxation_certified=res.relaxation_certified,
                        full_mip_certified=res.full_mip_certified,
                        any_valid_certificate=res.any_valid_certificate,
                        certificate_source=res.certificate_source.value,
                        lower_bound=res.lower_bound,
                        lower_bound_best=res.lower_bound_best,
                        upper_bound=res.upper_bound,
                        certificate_gap=res.final_certificate_gap,
                        relaxation_certificate_gap=res.relaxation_certificate_gap,
                        full_mip_certificate_gap=res.full_mip_certificate_gap,
                        relative_gap=res.relative_certificate_gap,
                        true_degradation=true_deg,
                        total_runtime=res.total_runtime,
                        relaxation_time=res.relaxation_time,
                        full_runtime=t_full,
                        speedup_vs_full=speedup,
                    )
                    all_records.append(rec)

    # Save detailed CSV
    df_evals = pd.DataFrame([r.__dict__ for r in all_records])
    csv_path = Path("artifacts/exp08b_recovery_certification.csv")
    df_evals.to_csv(csv_path, index=False)
    print(f"\n[4] Saved detailed evaluations to {csv_path}")

    # Summary Table
    summary_df = summarize_certification_evaluations(all_records)
    print("\n" + "=" * 80)
    print("RECOVERY & CERTIFICATION EVALUATION SUMMARY TABLE")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    plot_dir = Path("artifacts/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Certificate gap vs total runtime
    fig, ax = plt.subplots(figsize=(8, 5))
    df_cert = df_evals[~df_evals["fast_mode"] & df_evals["certificate_gap"].notna()]
    for method, grp in df_cert.groupby("method"):
        ax.scatter(grp["certificate_gap"], grp["total_runtime"] * 1000, label=method, alpha=0.8, s=40)
    ax.set_xlabel("Certificate Gap (Gap^{cert} = UB - LB)")
    ax.set_ylabel("Total End-to-End Runtime (ms)")
    ax.set_title("Plot 1: Full-Problem Certificate Gap vs Total Pipeline Runtime")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plot1_path = plot_dir / "phase8_plot1_certificate_gap_vs_runtime.png"
    plt.tight_layout()
    plt.savefig(plot1_path, dpi=300)
    plt.close()

    # Plot 2: Initial candidate reduction vs average recovery stages
    fig, ax = plt.subplots(figsize=(8, 5))
    df_stages = df_cert.groupby(["method", "initial_reduction"])["stages_executed"].mean().reset_index()
    for method, grp in df_stages.groupby("method"):
        ax.plot(grp["initial_reduction"], grp["stages_executed"], marker="o", label=method, linewidth=2)
    ax.set_xlabel("Initial Candidate Reduction (1 - B_1 / N)")
    ax.set_ylabel("Average Recovery Stages Executed")
    ax.set_title("Plot 2: Initial Candidate Reduction vs Average Recovery Stages")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plot2_path = plot_dir / "phase8_plot2_reduction_vs_recovery_stages.png"
    plt.tight_layout()
    plt.savefig(plot2_path, dpi=300)
    plt.close()

    # Plot 3: Fast-mode speedup vs certified-mode speedup
    fig, ax = plt.subplots(figsize=(8, 5))
    df_speedup = df_evals.groupby(["method", "fast_mode", "initial_B"])["speedup_vs_full"].mean().reset_index()
    methods_list = sorted(df_evals["method"].unique())
    x = np.arange(len(methods_list))
    width = 0.35

    sp_fast = [df_speedup[(df_speedup["method"] == m) & (df_speedup["fast_mode"]) & (df_speedup["initial_B"] == 3)]["speedup_vs_full"].values[0] for m in methods_list]
    sp_cert = [df_speedup[(df_speedup["method"] == m) & (~df_speedup["fast_mode"]) & (df_speedup["initial_B"] == 3)]["speedup_vs_full"].values[0] for m in methods_list]

    ax.bar(x - width/2, sp_fast, width, label="Fast Mode (B=3)", color="#3182CE", alpha=0.85)
    ax.bar(x + width/2, sp_cert, width, label="Certified Mode (B=3)", color="#DD6B20", alpha=0.85)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Full MIQP Parity (1.0x)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods_list)
    ax.set_ylabel("Speedup Factor vs Full MIQP")
    ax.set_title("Plot 3: Fast-Mode Speedup vs Certified-Mode Speedup (Initial B=3)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plot3_path = plot_dir / "phase8_plot3_fast_vs_certified_speedup.png"
    plt.tight_layout()
    plt.savefig(plot3_path, dpi=300)
    plt.close()

    # Plot 5: Certificate tolerance sensitivity
    print("\n[5] Running Tolerance Sensitivity Analysis for Plot 5...")
    eps_grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    eps_records = []
    mo_inst = test_instances[0]
    scores_mo = market_only_model.predict_instance_scores(mo_inst)

    for eps in eps_grid:
        opt_eps = NestedRecoveryOptimizer(mo_inst.params, solver_cfg, epsilon_cert=eps)
        res_eps = opt_eps.solve(mo_inst, scores=scores_mo, budgets=[3, 4, 6, 10], fast_mode=False)
        j_opt = mo_inst.solution.objective_value if mo_inst.solution else None
        deg = (res_eps.final_objective - j_opt) if (res_eps.final_objective is not None and j_opt is not None) else 0.0
        eps_records.append({
            "epsilon": eps,
            "runtime_ms": res_eps.total_runtime * 1000,
            "final_B": res_eps.final_budget,
            "true_degradation": deg,
        })
    df_eps = pd.DataFrame(eps_records)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "#3182CE"
    ax1.set_xscale("log")
    ax1.plot(df_eps["epsilon"], df_eps["runtime_ms"], marker="o", color=color, linewidth=2, label="Runtime (ms)")
    ax1.set_xlabel("Certificate Tolerance Epsilon (log scale)")
    ax1.set_ylabel("Total Runtime (ms)", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = "#DD6B20"
    ax2.plot(df_eps["epsilon"], df_eps["final_B"], marker="s", color=color, linewidth=2, linestyle="--", label="Final B")
    ax2.set_ylabel("Final Candidate Budget B", color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Plot 5: Certificate Tolerance Epsilon vs Runtime and Final Candidate Budget")
    plot5_path = plot_dir / "phase8_plot5_tolerance_sensitivity.png"
    plt.tight_layout()
    plt.savefig(plot5_path, dpi=300)
    plt.close()
    print(f"\n[6] Saved Plots 1, 2, 3, 5 to {plot_dir}")


if __name__ == "__main__":
    run_exp08b_recovery_certification()
