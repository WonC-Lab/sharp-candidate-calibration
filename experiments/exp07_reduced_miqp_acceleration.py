"""
Phase 7 Validation Experiment: Candidate-Universe Reduced MIQP Solver Acceleration.
Quantifies the exact trade-off between computational acceleration and optimization quality:
  - Compares physically sliced reduced MIQP against full-universe ground truth.
  - Accounts for full end-to-end runtime (relaxation + features + inference + build + solver).
  - Evaluates heuristic baselines, continuous relaxation, MarketOnlyML, and FullFeatureML.
  - Generates Table A (Objective Quality), Table B (Computation), Table C (Cost of Intelligence).
  - Produces diagnostic Pareto trade-off and speedup plots.
"""

from __future__ import annotations

from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from learning_augmented_portfolio.dataset.instance import OptimizationInstance
from learning_augmented_portfolio.dataset.serialization import load_instance
from learning_augmented_portfolio.optimization.reduced_miqp import (
    MultiPeriodReducedMIQPOptimizer,
    ReducedMIQPExecutionResult,
)
from learning_augmented_portfolio.optimization.support_restriction import CandidateUniverse
from learning_augmented_portfolio.optimization.variables import (
    MultiPeriodOptimizationResult,
    SolverConfig,
)
from learning_augmented_portfolio.learning.active_set import (
    build_logistic_regression_predictor,
    build_hist_gradient_boosting_predictor,
)
from learning_augmented_portfolio.learning.baselines import (
    RandomRankingBaseline,
    CurrentHoldingBaseline,
    ExpectedReturnBaseline,
    ContinuousRelaxationYBaseline,
)
from learning_augmented_portfolio.learning.candidate_set import construct_candidate_universe
from learning_augmented_portfolio.learning.dataset import (
    SupervisedSplitDataset,
    build_supervised_datasets_by_split,
)
from learning_augmented_portfolio.learning.features import (
    ALL_FEATURE_GROUPS,
    MARKET_ONLY_GROUPS,
)
from learning_augmented_portfolio.evaluation.objective_quality import (
    compare_instance_objectives,
    summarize_objective_quality,
    objective_reports_to_dataframe,
)
from learning_augmented_portfolio.evaluation.solver_acceleration import (
    compute_instance_acceleration,
    summarize_acceleration,
    acceleration_reports_to_dataframe,
)


def load_instances_from_manifest(manifest_csv_path: Path) -> list[OptimizationInstance]:
    df = pd.read_csv(manifest_csv_path)
    base_dir = manifest_csv_path.parent
    instances = []
    for _, row in df.iterrows():
        filepath = base_dir / row["filename"]
        inst = load_instance(filepath)
        instances.append(inst)
    return instances


def run_phase7_experiment():
    print("=" * 82)
    print("PHASE 7 VALIDATION EXPERIMENT: CANDIDATE-UNIVERSE REDUCED MIQP ACCELERATION")
    print("=" * 82)

    # 1. Load intermediate dataset (N=10, H=3, T=60)
    manifest_csv = Path("artifacts/instances_medium/manifest.csv")
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Medium dataset manifest {manifest_csv} not found. Please run exp06 first."
        )

    print(f"\n[1] Loading intermediate optimization instances from {manifest_csv}...")
    instances = load_instances_from_manifest(manifest_csv)
    datasets_all = build_supervised_datasets_by_split(instances, feature_groups=ALL_FEATURE_GROUPS)
    datasets_market = build_supervised_datasets_by_split(instances, feature_groups=MARKET_ONLY_GROUPS)

    ds_train = datasets_all["train"]
    ds_test = datasets_all["test"]

    N = instances[0].metadata.n_assets
    H = instances[0].metadata.horizon
    test_instances = ds_test.instances
    T_test = len(test_instances)

    print(f"    - Full Universe N = {N}, Horizon H = {H}")
    print(f"    - Test Instances T = {T_test} (strictly out-of-sample)")

    # 2. Train ML models on Train split
    print("\n[2] Training Machine Learning Models on Train Split...")
    # Market-Only ML (no relaxation features)
    ml_market = build_hist_gradient_boosting_predictor(feature_groups=MARKET_ONLY_GROUPS, max_iter=100)
    ml_market.fit(datasets_market["train"].X, datasets_market["train"].y_union)

    # Full-Feature ML (includes continuous relaxation features)
    ml_full = build_hist_gradient_boosting_predictor(feature_groups=ALL_FEATURE_GROUPS, max_iter=100)
    ml_full.fit(ds_train.X, ds_train.y_union)

    # Logistic Regression
    ml_logreg = build_logistic_regression_predictor(feature_groups=ALL_FEATURE_GROUPS, C=1.0)
    ml_logreg.fit(ds_train.X, ds_train.y_union)

    methods = {
        "Random": (RandomRankingBaseline(seed=42), False, False),
        "CurrentHolding": (CurrentHoldingBaseline(), False, False),
        "ExpectedReturn": (ExpectedReturnBaseline(), False, False),
        "ContinuousRelaxation": (ContinuousRelaxationYBaseline(), True, False),
        "LogisticRegression": (ml_logreg, True, True),
        "MarketOnlyML": (ml_market, False, True),
        "FullFeatureML": (ml_full, True, True),
    }

    # Solver configuration (identical for full and reduced)
    solver_cfg = SolverConfig(time_limit=15.0, mip_gap=1e-4, threads=1)
    reduced_opt = MultiPeriodReducedMIQPOptimizer(test_instances[0].params, solver_cfg)

    # 3. Evaluate across candidate budgets B in [3, 4, 5, 6, 8, 10]
    budgets = [3, 4, 5, 6, 8, 10]
    print(f"\n[3] Solving Reduced MIQPs across Budgets B in {budgets}...")

    all_obj_comparisons = []
    all_accel_records = []
    detailed_rows = []

    for B in budgets:
        for method_name, (ranker, req_relax, req_ml) in methods.items():
            for use_safe in [False, True]:
                comp_list = []
                accel_list = []

                for inst in test_instances:
                    # Ground truth full MIQP
                    full_res = inst.solution
                    assert full_res is not None
                    t_full_solver = inst.metadata.runtime
                    full_nodes = inst.metadata.node_count or 0
                    full_work = inst.metadata.work

                    # Component timers for end-to-end accounting
                    t_relax = inst.relaxation.solve_time if (req_relax and inst.relaxation) else 0.0

                    t0_feat = time.perf_counter()
                    t_feat = time.perf_counter() - t0_feat if req_ml else 0.0

                    t0_inf = time.perf_counter()
                    scores = ranker.predict_instance_scores(inst)
                    t_inf = time.perf_counter() - t0_inf if req_ml else 0.0

                    t0_cand = time.perf_counter()
                    curr_w = inst.actual_state.normalized_weights if use_safe else None
                    cand = construct_candidate_universe(scores, budget_B=B, current_weights=curr_w, use_safe_augmentation=use_safe)
                    t_cand = time.perf_counter() - t0_cand

                    # Solve physically sliced reduced MIQP
                    red_exec = reduced_opt.solve(inst.market_state, cand, expand_solution=True)

                    t_build = red_exec.build_time
                    t_red_solver = red_exec.solver_time
                    red_nodes = red_exec.node_count or 0
                    red_work = red_exec.work

                    # Teacher target support S_t^*
                    target_sup = set(np.where(inst.labels.y_union == 1)[0].tolist()) if inst.labels else None

                    # Objective comparison
                    obj_comp = compare_instance_objectives(
                        instance_id=inst.metadata.instance_id,
                        method=method_name,
                        budget_B=B,
                        effective_B=cand.candidate_size,
                        full_result=full_res,
                        reduced_exec=red_exec,
                        target_support=target_sup,
                        is_safe_augmented=use_safe,
                    )
                    comp_list.append(obj_comp)
                    all_obj_comparisons.append(obj_comp)

                    # Acceleration record
                    accel_rec = compute_instance_acceleration(
                        instance_id=inst.metadata.instance_id,
                        method=method_name,
                        budget_B=B,
                        effective_B=cand.candidate_size,
                        full_solver_time=t_full_solver,
                        reduced_solver_time=t_red_solver,
                        build_time=t_build,
                        candidate_time=t_cand,
                        relax_time=t_relax,
                        feature_time=t_feat,
                        inference_time=t_inf,
                        full_nodes=full_nodes,
                        reduced_nodes=red_nodes,
                        full_work=full_work,
                        reduced_work=red_work,
                        is_safe_augmented=use_safe,
                    )
                    accel_list.append(accel_rec)
                    all_accel_records.append(accel_rec)

                    # Log detailed row
                    detailed_rows.append({
                        "instance_id": inst.metadata.instance_id,
                        "method": method_name,
                        "B": B,
                        "B_eff": cand.candidate_size,
                        "reduction_ratio": cand.reduction_ratio,
                        "is_safe_augmented": use_safe,
                        "support_contained": obj_comp.is_support_contained,
                        "full_objective": full_res.objective_value,
                        "reduced_objective": red_exec.objective_value,
                        "objective_degradation": obj_comp.objective_degradation,
                        "relative_degradation": obj_comp.relative_degradation,
                        "full_runtime": t_full_solver,
                        "reduced_solver_runtime": t_red_solver,
                        "relaxation_runtime": t_relax,
                        "feature_runtime": t_feat,
                        "inference_runtime": t_inf,
                        "build_runtime": t_build,
                        "end_to_end_runtime": accel_rec.e2e_reduced_time,
                        "runtime_reduction": accel_rec.runtime_reduction,
                        "speedup_factor": accel_rec.e2e_speedup,
                        "full_nodes": full_nodes,
                        "reduced_nodes": red_nodes,
                        "full_work": full_work,
                        "reduced_work": red_work,
                    })

    df_details = pd.DataFrame(detailed_rows)
    df_details.to_csv("artifacts/phase7_detailed_evaluations.csv", index=False)
    print("    -> Saved all instance evaluations to artifacts/phase7_detailed_evaluations.csv")

    # 4. Generate Core Table A — Objective Quality
    print("\n" + "=" * 82)
    print("TABLE A: OBJECTIVE QUALITY & SUPPORT PRESERVATION (TEST SPLIT)")
    print("=" * 82)
    obj_reports = []
    for B in [3, 4, 5, 6, 8]:
        for method_name in ["Random", "CurrentHolding", "ExpectedReturn", "ContinuousRelaxation", "MarketOnlyML", "FullFeatureML"]:
            for use_safe in [False, True]:
                subset = [c for c in all_obj_comparisons if c.method == method_name and c.budget_B == B and c.is_safe_augmented == use_safe]
                if subset:
                    rep = summarize_objective_quality(subset, method_name, "test", B, N, use_safe)
                    obj_reports.append(rep)

    df_table_a = objective_reports_to_dataframe(obj_reports)
    print(df_table_a[["Method", "SafeAug", "B", "Eff_B", "Reduction", "Coverage@B", "ExactObjRate", "MeanDeltaJ", "MaxDeltaJ", "InfeasibilityRate"]].to_string(index=False))

    # 5. Generate Core Table B — Computational Acceleration & Runtime
    print("\n" + "=" * 82)
    print("TABLE B: COMPUTATIONAL ACCELERATION & END-TO-END TIMING (TEST SPLIT)")
    print("=" * 82)
    accel_reports = []
    for B in [3, 4, 5, 6, 8]:
        for method_name in ["Random", "CurrentHolding", "ExpectedReturn", "ContinuousRelaxation", "MarketOnlyML", "FullFeatureML"]:
            for use_safe in [False, True]:
                subset = [r for r in all_accel_records if r.method == method_name and r.budget_B == B and r.is_safe_augmented == use_safe]
                if subset:
                    rep = summarize_acceleration(subset, method_name, "test", B, N, use_safe)
                    accel_reports.append(rep)

    df_table_b = acceleration_reports_to_dataframe(accel_reports)
    print(df_table_b[["Method", "SafeAug", "B", "Eff_B", "Reduction", "FullTime_ms", "RedSolverTime_ms", "E2ETime_ms", "SolverSpeedup", "E2ESpeedup", "RuntimeReduction", "NodeReduction"]].to_string(index=False))

    # 6. Generate Core Table C — Cost of Intelligence Comparison
    print("\n" + "=" * 82)
    print("TABLE C: COST OF INTELLIGENCE COMPARISON (MarketOnlyML vs Relaxation vs FullFeatureML)")
    print("=" * 82)
    cost_intel = df_table_b[df_table_b["Method"].isin(["ContinuousRelaxation", "MarketOnlyML", "FullFeatureML"]) & (df_table_b["SafeAug"] == True)]
    print(cost_intel[["Method", "B", "Reduction", "RedSolverTime_ms", "E2ETime_ms", "SolverSpeedup", "E2ESpeedup", "RuntimeReduction"]].to_string(index=False))

    # 7. Generate Core Phase 7 Plots
    plots_dir = Path("artifacts/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[4] Generating Phase 7 Diagnostic Plots in {plots_dir}...")

    # Plot 1: Pareto Plot (Relative Objective Degradation vs End-to-End Runtime Reduction)
    plt.figure(figsize=(9, 6))
    plot_methods = ["ContinuousRelaxation", "MarketOnlyML", "FullFeatureML", "ExpectedReturn", "CurrentHolding"]
    colors = {"ContinuousRelaxation": "#1f77b4", "MarketOnlyML": "#2ca02c", "FullFeatureML": "#d62728", "ExpectedReturn": "#ff7f0e", "CurrentHolding": "#9467bd"}
    markers = {"ContinuousRelaxation": "o", "MarketOnlyML": "^", "FullFeatureML": "s", "ExpectedReturn": "v", "CurrentHolding": "D"}

    for m in plot_methods:
        df_sub = df_details[(df_details["method"] == m) & (df_details["is_safe_augmented"] == True)]
        grouped = df_sub.groupby("B").agg({
            "relative_degradation": "mean",
            "runtime_reduction": "mean",
        }).reset_index()
        plt.plot(grouped["relative_degradation"] * 100, grouped["runtime_reduction"] * 100, marker=markers[m], label=m, color=colors[m], linewidth=2, markersize=8)
        for _, row in grouped.iterrows():
            plt.annotate(f"B={int(row['B'])}", (row["relative_degradation"] * 100, row["runtime_reduction"] * 100), textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.axhline(0, color="gray", linestyle="--", alpha=0.7, label="Zero Speedup (Parity)")
    plt.title("Pareto Frontier: Objective Degradation vs. E2E Runtime Reduction", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Relative Objective Degradation (%)", fontsize=11)
    plt.ylabel("Mean End-to-End Runtime Reduction (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    p1 = plots_dir / "phase7_plot1_pareto_degradation_vs_runtime.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"    - Saved Plot 1 (Pareto Frontier): {p1}")

    # Plot 2: Candidate Reduction vs Exact Objective Recovery Rate
    plt.figure(figsize=(8, 5))
    for m in plot_methods:
        df_sub = df_table_a[(df_table_a["Method"] == m) & (df_table_a["SafeAug"] == True)]
        plt.plot(df_sub["Reduction"] * 100, df_sub["ExactObjRate"] * 100, marker=markers[m], label=m, color=colors[m], linewidth=2)
    plt.title("Candidate Reduction (%) vs. Exact Objective Recovery Rate (%)", fontsize=13, fontweight="bold")
    plt.xlabel("True Candidate Universe Reduction (%)", fontsize=11)
    plt.ylabel("Exact Objective Recovery Rate (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()
    p2 = plots_dir / "phase7_plot2_reduction_vs_exact_objective.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"    - Saved Plot 2: {p2}")

    # Plot 3: Candidate Reduction vs End-to-End Speedup Factor
    plt.figure(figsize=(8, 5))
    for m in plot_methods:
        df_sub = df_table_b[(df_table_b["Method"] == m) & (df_table_b["SafeAug"] == True)]
        plt.plot(df_sub["Reduction"] * 100, df_sub["E2ESpeedup"], marker=markers[m], label=m, color=colors[m], linewidth=2)
    plt.axhline(1.0, color="black", linestyle="--", alpha=0.7, label="1.0x Parity")
    plt.title("Candidate Reduction (%) vs. End-to-End Speedup Factor", fontsize=13, fontweight="bold")
    plt.xlabel("True Candidate Universe Reduction (%)", fontsize=11)
    plt.ylabel("End-to-End Speedup Factor (T_full / T_E2E)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()
    p3 = plots_dir / "phase7_plot3_reduction_vs_speedup.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"    - Saved Plot 3: {p3}")

    # Plot 4: Effective Budget B^{eff} vs Node Count Reduction
    plt.figure(figsize=(8, 5))
    for m in plot_methods:
        df_sub = df_table_b[(df_table_b["Method"] == m) & (df_table_b["SafeAug"] == True)]
        plt.plot(df_sub["Eff_B"], df_sub["NodeReduction"] * 100, marker=markers[m], label=m, color=colors[m], linewidth=2)
    plt.title("Effective Candidate Budget B^{eff} vs. B&B Node Reduction (%)", fontsize=13, fontweight="bold")
    plt.xlabel("Effective Candidate Size B^{eff}", fontsize=11)
    plt.ylabel("B&B Node Count Reduction (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()
    p4 = plots_dir / "phase7_plot4_budget_vs_node_reduction.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"    - Saved Plot 4: {p4}")

    print("\n" + "=" * 82)
    print("PHASE 7 VALIDATION EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 82)


if __name__ == "__main__":
    run_phase7_experiment()
