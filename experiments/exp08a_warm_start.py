"""
Phase 8 Experiment 8A: Discrete MIP Warm Starts and Feasibility Repair.
Compares:
  - W0: No Start (Standard Gurobi cold solve)
  - W1: Previous MPC Solution Shift
  - W2: ML Discrete Binary Start (Market-Only)
  - W3: ML + Continuous Repaired Complete Start
Evaluates:
  - Construction & repair times
  - Primal feasibility rate
  - Time to first feasible solution (T_{first}) and Delta T_{first}
  - First incumbent quality, final runtime, Work, NodeCount, final MIP gap
Generates Plot 4: Warm-start method vs time-to-first-feasible.
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
from learning_augmented_portfolio.dataset.manifest import DatasetManifest
from learning_augmented_portfolio.dataset.serialization import load_instance
from learning_augmented_portfolio.learning.warm_start import (
    MIPStartData,
    WarmStartType,
    build_previous_solution_warm_start,
)
from learning_augmented_portfolio.learning.warm_start_model import HorizonWarmStartPredictor
from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from learning_augmented_portfolio.optimization.repair import ContinuousRepairOptimizer
from learning_augmented_portfolio.optimization.variables import (
    MultiPeriodOptimizationResult,
    SolverConfig,
)
from learning_augmented_portfolio.evaluation.warm_start_metrics import (
    WarmStartEvaluationRecord,
    summarize_warm_start_evaluations,
)
from learning_augmented_portfolio.solver.callbacks import OptimizationProgressLogger


def run_exp08a_warm_start() -> None:
    print("=" * 80)
    print("PHASE 8A: WARM-START & FEASIBILITY REPAIR EXPERIMENT")
    print("=" * 80)

    manifest_path = Path("artifacts/instances_medium/manifest.csv")
    manifest_df = pd.read_csv(manifest_path)
    base_dir = manifest_path.parent

    train_rows = manifest_df[manifest_df["split"] == "train"]
    test_rows = manifest_df[manifest_df["split"] == "test"]

    print(f"\n[1] Loading train instances ({len(train_rows)}) and test instances ({len(test_rows)})...")
    train_instances = [load_instance(base_dir / row["filename"]) for _, row in train_rows.iterrows()]
    test_instances = [load_instance(base_dir / row["filename"]) for _, row in test_rows.iterrows()]

    # Sort test instances chronologically
    test_instances.sort(key=lambda inst: inst.metadata.time_index)

    print("\n[2] Training HorizonWarmStartPredictor on train split (Market-Only)...")
    ws_predictor = HorizonWarmStartPredictor(include_relaxation=False, random_state=42)
    ws_predictor.fit(train_instances)

    cfg = SolverConfig(time_limit=15.0, mip_gap=1e-4, threads=1)
    repairer = ContinuousRepairOptimizer(test_instances[0].params, solver_config=SolverConfig(time_limit=5.0, threads=1))

    all_records: List[WarmStartEvaluationRecord] = []
    plot_data: Dict[str, List[float]] = {"W0_NoStart": [], "W1_PrevSol": [], "W2_MLDiscrete": [], "W3_MLRepaired": []}

    print("\n[3] Evaluating Warm-Start Strategies on test instances...")
    for idx, inst in enumerate(test_instances):
        inst_id = inst.metadata.instance_id
        opt = MultiPeriodMIQPOptimizer(inst.params, cfg)

        # Baseline: Cold solve (W0)
        cb_w0 = OptimizationProgressLogger()
        res_w0 = opt.solve(inst.market_state, callback=cb_w0)
        t_first_w0 = cb_w0.first_feasible_time or res_w0.solve_time
        plot_data["W0_NoStart"].append(t_first_w0 * 1000)

        all_records.append(WarmStartEvaluationRecord(
            instance_id=inst_id,
            start_type="W0_NoStart",
            construction_time=0.0,
            repair_time=0.0,
            partial_start_provided=False,
            complete_start_feasible=None,
            repair_success_rate=None,
            first_feasible_time=t_first_w0,
            delta_t_first=0.0,
            first_incumbent_obj=cb_w0.first_feasible_obj,
            final_runtime=res_w0.solve_time,
            work=res_w0.work or 0.0,
            node_count=res_w0.node_count or 0,
            final_objective=res_w0.objective_value or 0.0,
            best_bound=res_w0.best_bound or 0.0,
            mip_gap=res_w0.mip_gap or 0.0,
        ))

        # W1: Previous Solution Shift
        prev_inst = test_instances[idx - 1] if idx > 0 else train_instances[-1]
        prev_sol = prev_inst.solution
        if prev_sol is not None and prev_sol.is_feasible:
            w1_data = build_previous_solution_warm_start(prev_sol, inst.params.horizon, inst.market_state.n_assets)
            cb_w1 = OptimizationProgressLogger()
            res_w1 = opt.solve(inst.market_state, callback=cb_w1, warm_start=w1_data)
            t_first_w1 = cb_w1.first_feasible_time or res_w1.solve_time
            plot_data["W1_PrevSol"].append(t_first_w1 * 1000)

            all_records.append(WarmStartEvaluationRecord(
                instance_id=inst_id,
                start_type="W1_PrevSol",
                construction_time=w1_data.construction_time,
                repair_time=0.0,
                partial_start_provided=True,
                complete_start_feasible=True,
                repair_success_rate=None,
                first_feasible_time=t_first_w1,
                delta_t_first=t_first_w1 - t_first_w0,
                first_incumbent_obj=cb_w1.first_feasible_obj,
                final_runtime=res_w1.solve_time,
                work=res_w1.work or 0.0,
                node_count=res_w1.node_count or 0,
                final_objective=res_w1.objective_value or 0.0,
                best_bound=res_w1.best_bound or 0.0,
                mip_gap=res_w1.mip_gap or 0.0,
            ))

        # W2: ML Discrete Binary Start
        w2_data, probs_y, probs_z = ws_predictor.predict_warm_start(inst)
        cb_w2 = OptimizationProgressLogger()
        res_w2 = opt.solve(inst.market_state, callback=cb_w2, warm_start=w2_data)
        t_first_w2 = cb_w2.first_feasible_time or res_w2.solve_time
        plot_data["W2_MLDiscrete"].append(t_first_w2 * 1000)

        all_records.append(WarmStartEvaluationRecord(
            instance_id=inst_id,
            start_type="W2_MLDiscrete",
            construction_time=w2_data.construction_time,
            repair_time=0.0,
            partial_start_provided=True,
            complete_start_feasible=False,
            repair_success_rate=None,
            first_feasible_time=t_first_w2,
            delta_t_first=t_first_w2 - t_first_w0,
            first_incumbent_obj=cb_w2.first_feasible_obj,
            final_runtime=res_w2.solve_time,
            work=res_w2.work or 0.0,
            node_count=res_w2.node_count or 0,
            final_objective=res_w2.objective_value or 0.0,
            best_bound=res_w2.best_bound or 0.0,
            mip_gap=res_w2.mip_gap or 0.0,
        ))

        # W3: ML + Continuous Repair
        t0_rep = time.perf_counter()
        repair_res = repairer.repair(inst.market_state, w2_data.y_start, w2_data.z_start, probabilities_y=probs_y)
        t_rep = time.perf_counter() - t0_rep

        if repair_res.is_feasible and repair_res.repaired_solution is not None:
            sol_rep = repair_res.repaired_solution
            w3_data = MIPStartData(
                start_type=WarmStartType.ML_REPAIRED,
                y_start=sol_rep.holding_indicators,
                z_start=sol_rep.trade_indicators,
                x_start=sol_rep.post_trade_holdings,
                b_start=sol_rep.post_trade_cash,
                is_complete=True,
                construction_time=w2_data.construction_time,
                repair_time=t_rep,
            )
            is_feasible_start = True
        else:
            w3_data = w2_data
            is_feasible_start = False

        cb_w3 = OptimizationProgressLogger()
        res_w3 = opt.solve(inst.market_state, callback=cb_w3, warm_start=w3_data)
        t_first_w3 = cb_w3.first_feasible_time or res_w3.solve_time
        plot_data["W3_MLRepaired"].append(t_first_w3 * 1000)

        all_records.append(WarmStartEvaluationRecord(
            instance_id=inst_id,
            start_type="W3_MLRepaired",
            construction_time=w2_data.construction_time,
            repair_time=t_rep,
            partial_start_provided=True,
            complete_start_feasible=is_feasible_start,
            repair_success_rate=1.0 if is_feasible_start else 0.0,
            first_feasible_time=t_first_w3,
            delta_t_first=t_first_w3 - t_first_w0,
            first_incumbent_obj=cb_w3.first_feasible_obj,
            final_runtime=res_w3.solve_time,
            work=res_w3.work or 0.0,
            node_count=res_w3.node_count or 0,
            final_objective=res_w3.objective_value or 0.0,
            best_bound=res_w3.best_bound or 0.0,
            mip_gap=res_w3.mip_gap or 0.0,
        ))

    # Save detailed CSV
    df_records = pd.DataFrame([r.__dict__ for r in all_records])
    csv_path = Path("artifacts/exp08a_warm_start_evaluations.csv")
    df_records.to_csv(csv_path, index=False)
    print(f"\n[4] Saved detailed evaluations to {csv_path}")

    # Summary Table
    summary_df = summarize_warm_start_evaluations(all_records)
    print("\n" + "=" * 80)
    print("WARM-START EVALUATION SUMMARY TABLE (FULL MIQP N=10, H=3)")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Plot 4: Warm-start method vs time-to-first-feasible
    plot_dir = Path("artifacts/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(plot_data.keys())
    means = [float(np.mean(plot_data[k])) for k in labels]
    stds = [float(np.std(plot_data[k])) for k in labels]

    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=["#4A5568", "#3182CE", "#DD6B20", "#38A169"], alpha=0.85)
    ax.set_ylabel("Time to First Feasible Incumbent (ms)")
    ax.set_title("Plot 4: Warm-Start Method vs Time-to-First-Feasible Incumbent (T_first)")
    ax.grid(True, linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f} ms", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plot_file = plot_dir / "phase8_plot4_warm_start_time_to_first.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"\n[5] Saved Plot 4 to {plot_file}")


if __name__ == "__main__":
    run_exp08a_warm_start()
