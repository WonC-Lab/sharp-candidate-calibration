"""
Phase 8 Experiment 8C: Integrated Hybrid Optimizer.
Combines:
  State -> Ranking -> CandidateSet -> WarmStart -> ReducedMIQP -> Lift -> Certificate -> Recovery
Compares:
  1. Full MIQP (Standard baseline)
  2. Fast MarketOnly Reduced MIQP (B=3, no certificate)
  3. Certified MarketOnly + Cold Recovery
  4. Certified MarketOnly + Warm Recovery (ML repaired initial start + inter-stage warm start)
  5. Certified ContinuousRelaxation + Warm Recovery
  6. Certified FullFeatureML + Warm Recovery
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from learning_augmented_portfolio.dataset.instance import OptimizationInstance
from learning_augmented_portfolio.dataset.manifest import DatasetManifest
from learning_augmented_portfolio.dataset.serialization import load_instance
from learning_augmented_portfolio.learning.active_set import build_logistic_regression_predictor
from learning_augmented_portfolio.learning.baselines import ContinuousRelaxationYBaseline
from learning_augmented_portfolio.learning.dataset import build_supervised_datasets_by_split
from learning_augmented_portfolio.learning.warm_start import MIPStartData, WarmStartType
from learning_augmented_portfolio.learning.warm_start_model import HorizonWarmStartPredictor
from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from learning_augmented_portfolio.optimization.repair import ContinuousRepairOptimizer
from learning_augmented_portfolio.optimization.variables import SolverConfig
from learning_augmented_portfolio.recovery.recovery_loop import NestedRecoveryOptimizer


def run_exp08c_hybrid() -> None:
    print("=" * 80)
    print("PHASE 8C: INTEGRATED HYBRID OPTIMIZER EXPERIMENT")
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

    print("\n[2] Training Ranking Models and Warm-Start Models...")
    # Ranking models
    from learning_augmented_portfolio.learning.features import MARKET_ONLY_GROUPS, ALL_FEATURE_GROUPS
    train_ds_mo = build_supervised_datasets_by_split(train_instances, feature_groups=MARKET_ONLY_GROUPS)["train"]
    train_ds_ff = build_supervised_datasets_by_split(train_instances, feature_groups=ALL_FEATURE_GROUPS)["train"]

    mo_ranker = build_logistic_regression_predictor(feature_groups=MARKET_ONLY_GROUPS, random_state=42)
    mo_ranker.fit(train_ds_mo.X, train_ds_mo.y_union)

    ff_ranker = build_logistic_regression_predictor(feature_groups=ALL_FEATURE_GROUPS, random_state=42)
    ff_ranker.fit(train_ds_ff.X, train_ds_ff.y_union)

    cr_ranker = ContinuousRelaxationYBaseline()

    # Warm-start predictor
    ws_predictor = HorizonWarmStartPredictor(include_relaxation=False, random_state=42)
    ws_predictor.fit(train_instances)

    solver_cfg = SolverConfig(time_limit=15.0, mip_gap=1e-4, threads=1)
    repairer = ContinuousRepairOptimizer(test_instances[0].params, solver_config=SolverConfig(time_limit=5.0, threads=1))

    budget_seq = [3, 4, 6, 10]
    hybrid_records: List[Dict[str, Any]] = []

    print("\n[3] Evaluating Integrated Hybrid Pipelines on Test Instances...")
    for inst in test_instances:
        inst_id = inst.metadata.instance_id
        t_full = inst.metadata.runtime
        j_star = inst.solution.objective_value if inst.solution else None

        # 1. Full MIQP (Cold)
        full_opt = MultiPeriodMIQPOptimizer(inst.params, solver_cfg)
        res_full = full_opt.solve(inst.market_state)
        hybrid_records.append({
            "instance_id": inst_id,
            "pipeline": "1_FullMIQP",
            "runtime_ms": res_full.solve_time * 1000,
            "speedup": 1.0,
            "objective": res_full.objective_value,
            "true_degradation": 0.0,
            "cert_gap": 0.0,
            "stages": 1,
            "is_certified": True,
            "node_count": res_full.node_count or 0,
        })

        # Precompute ML rankings and warm starts
        mo_scores = mo_ranker.predict_instance_scores(inst)
        ff_scores = ff_ranker.predict_instance_scores(inst)
        cr_scores = cr_ranker.predict_instance_scores(inst)

        # Build repaired warm start
        raw_ws, probs_y, probs_z = ws_predictor.predict_warm_start(inst)
        rep_res = repairer.repair(inst.market_state, raw_ws.y_start, raw_ws.z_start, probabilities_y=probs_y)
        if rep_res.is_feasible and rep_res.repaired_solution is not None:
            repaired_ws = MIPStartData(
                start_type=WarmStartType.ML_REPAIRED,
                y_start=rep_res.repaired_solution.holding_indicators,
                z_start=rep_res.repaired_solution.trade_indicators,
                x_start=rep_res.repaired_solution.post_trade_holdings,
                b_start=rep_res.repaired_solution.post_trade_cash,
                is_complete=True,
            )
        else:
            repaired_ws = raw_ws

        # 2. Fast MarketOnly (B=3, no certificate)
        rec_fast = NestedRecoveryOptimizer(inst.params, solver_cfg)
        res_fast = rec_fast.solve(inst, mo_scores, budget_seq, fast_mode=True)
        deg_fast = (res_fast.final_objective - j_star) if (res_fast.final_objective is not None and j_star is not None) else 0.0
        hybrid_records.append({
            "instance_id": inst_id,
            "pipeline": "2_Fast_MarketOnly",
            "runtime_ms": res_fast.total_runtime * 1000,
            "speedup": t_full / max(1e-4, res_fast.total_runtime),
            "objective": res_fast.final_objective,
            "true_degradation": deg_fast,
            "cert_gap": None,
            "stages": 1,
            "is_certified": False,
            "node_count": res_fast.stage_telemetry[0].node_count,
        })

        # 3. Certified MarketOnly + Cold Recovery
        rec_mo_cold = NestedRecoveryOptimizer(inst.params, solver_cfg, epsilon_cert=1e-4, use_recovery_warm_starts=False)
        res_mo_cold = rec_mo_cold.solve(inst, mo_scores, budget_seq, fast_mode=False)
        deg_mo_cold = (res_mo_cold.final_objective - j_star) if (res_mo_cold.final_objective is not None and j_star is not None) else 0.0
        hybrid_records.append({
            "instance_id": inst_id,
            "pipeline": "3_Cert_MarketOnly_Cold",
            "runtime_ms": res_mo_cold.total_runtime * 1000,
            "speedup": t_full / max(1e-4, res_mo_cold.total_runtime),
            "objective": res_mo_cold.final_objective,
            "true_degradation": deg_mo_cold,
            "cert_gap": res_mo_cold.final_certificate_gap,
            "stages": res_mo_cold.stages_executed,
            "is_certified": res_mo_cold.is_certified,
            "node_count": sum(st.node_count for st in res_mo_cold.stage_telemetry),
        })

        # 4. Certified MarketOnly + Warm Recovery
        rec_mo_warm = NestedRecoveryOptimizer(inst.params, solver_cfg, epsilon_cert=1e-4, use_recovery_warm_starts=True)
        res_mo_warm = rec_mo_warm.solve(inst, mo_scores, budget_seq, initial_warm_start=repaired_ws, fast_mode=False)
        deg_mo_warm = (res_mo_warm.final_objective - j_star) if (res_mo_warm.final_objective is not None and j_star is not None) else 0.0
        hybrid_records.append({
            "instance_id": inst_id,
            "pipeline": "4_Cert_MarketOnly_Warm",
            "runtime_ms": res_mo_warm.total_runtime * 1000,
            "speedup": t_full / max(1e-4, res_mo_warm.total_runtime),
            "objective": res_mo_warm.final_objective,
            "true_degradation": deg_mo_warm,
            "cert_gap": res_mo_warm.final_certificate_gap,
            "stages": res_mo_warm.stages_executed,
            "is_certified": res_mo_warm.is_certified,
            "node_count": sum(st.node_count for st in res_mo_warm.stage_telemetry),
        })

        # 5. Certified ContinuousRelaxation + Warm Recovery
        pre_lb = inst.relaxation.objective_value if inst.relaxation else None
        pre_t = inst.relaxation.solve_time if inst.relaxation else 0.0
        rec_cr_warm = NestedRecoveryOptimizer(inst.params, solver_cfg, epsilon_cert=1e-4, use_recovery_warm_starts=True)
        res_cr_warm = rec_cr_warm.solve(inst, cr_scores, budget_seq, precomputed_lower_bound=pre_lb, precomputed_relaxation_time=pre_t, fast_mode=False)
        deg_cr_warm = (res_cr_warm.final_objective - j_star) if (res_cr_warm.final_objective is not None and j_star is not None) else 0.0
        hybrid_records.append({
            "instance_id": inst_id,
            "pipeline": "5_Cert_Relaxation_Warm",
            "runtime_ms": res_cr_warm.total_runtime * 1000,
            "speedup": t_full / max(1e-4, res_cr_warm.total_runtime),
            "objective": res_cr_warm.final_objective,
            "true_degradation": deg_cr_warm,
            "cert_gap": res_cr_warm.final_certificate_gap,
            "stages": res_cr_warm.stages_executed,
            "is_certified": res_cr_warm.is_certified,
            "node_count": sum(st.node_count for st in res_cr_warm.stage_telemetry),
        })

        # 6. Certified FullFeatureML + Warm Recovery
        rec_ff_warm = NestedRecoveryOptimizer(inst.params, solver_cfg, epsilon_cert=1e-4, use_recovery_warm_starts=True)
        res_ff_warm = rec_ff_warm.solve(inst, ff_scores, budget_seq, precomputed_lower_bound=pre_lb, precomputed_relaxation_time=pre_t, fast_mode=False)
        deg_ff_warm = (res_ff_warm.final_objective - j_star) if (res_ff_warm.final_objective is not None and j_star is not None) else 0.0
        hybrid_records.append({
            "instance_id": inst_id,
            "pipeline": "6_Cert_FullFeature_Warm",
            "runtime_ms": res_ff_warm.total_runtime * 1000,
            "speedup": t_full / max(1e-4, res_ff_warm.total_runtime),
            "objective": res_ff_warm.final_objective,
            "true_degradation": deg_ff_warm,
            "cert_gap": res_ff_warm.final_certificate_gap,
            "stages": res_ff_warm.stages_executed,
            "is_certified": res_ff_warm.is_certified,
            "node_count": sum(st.node_count for st in res_ff_warm.stage_telemetry),
        })

    # Save detailed CSV
    df_hybrid = pd.DataFrame(hybrid_records)
    csv_path = Path("artifacts/exp08c_hybrid_evaluations.csv")
    df_hybrid.to_csv(csv_path, index=False)
    print(f"\n[4] Saved detailed evaluations to {csv_path}")

    # Summary Table
    summary_df = df_hybrid.groupby("pipeline").agg(
        Instances=("instance_id", "count"),
        MeanRuntime_ms=("runtime_ms", "mean"),
        MeanSpeedup=("speedup", "mean"),
        MeanTrueDegradation=("true_degradation", "mean"),
        MeanCertGap=("cert_gap", lambda s: float(s.dropna().mean()) if len(s.dropna()) > 0 else np.nan),
        AvgStages=("stages", "mean"),
        CertifiedRate=("is_certified", "mean"),
        MeanNodeCount=("node_count", "mean"),
    ).reset_index()

    print("\n" + "=" * 80)
    print("INTEGRATED HYBRID PIPELINE EVALUATION SUMMARY TABLE")
    print("=" * 80)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_exp08c_hybrid()
