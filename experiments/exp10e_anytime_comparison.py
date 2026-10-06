"""
Phase 10E: Common Time-Budget Experiments & Callback-Based Anytime Curves (Table E).
Evaluates Full MIQP and candidate-reduced methods under identical time budgets
and tracks anytime incumbent UB(t), best bound LB(t), and MIP gap evolution.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import gurobipy as gp
import numpy as np
import pandas as pd

from learning_augmented_portfolio.dataset.instance import OptimizationInstance
from learning_augmented_portfolio.learning.baselines import (
    ContinuousRelaxationYBaseline,
    ExpectedReturnBaseline,
)
from learning_augmented_portfolio.learning.candidate_set import construct_candidate_universe
from learning_augmented_portfolio.optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from learning_augmented_portfolio.optimization.reduced_miqp import MultiPeriodReducedMIQPOptimizer
from learning_augmented_portfolio.optimization.relaxation import MultiPeriodRelaxedQPOptimizer
from learning_augmented_portfolio.optimization.variables import SolverConfig
from learning_augmented_portfolio.scalability.anytime import AnytimeCallback
from learning_augmented_portfolio.scalability.instance_generator import ScalableInstanceGenerator


def evaluate_fixed_time_limit(
    instance: OptimizationInstance,
    time_limit: float,
    method_type: str = "full",
    candidate_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Solves an instance under a strict wall-clock time limit and records solver telemetry.
    """
    t_start = time.perf_counter()
    N = instance.market_state.n_assets
    H = instance.params.horizon

    # Determine candidate set if reduced
    cand = None
    elapsed_overhead = 0.0
    current_w = instance.actual_state.normalized_weights
    n_held = int(np.sum(current_w > 1e-6))

    if method_type == "market_only":
        t0_ov = time.perf_counter()
        B = candidate_size if candidate_size is not None else max(instance.params.cardinality_limit, int(0.20 * N))
        eff_B = max(B, n_held)
        scores = ExpectedReturnBaseline().predict_instance_scores(instance)
        cand = construct_candidate_universe(
            scores=scores,
            budget_B=eff_B,
            current_weights=current_w,
            use_safe_augmentation=True,
        )
        elapsed_overhead = time.perf_counter() - t0_ov
    elif method_type == "relaxation":
        t0_ov = time.perf_counter()
        B = candidate_size if candidate_size is not None else max(instance.params.cardinality_limit, int(0.20 * N))
        eff_B = max(B, n_held)
        # Solve continuous relaxation
        rel_opt = MultiPeriodRelaxedQPOptimizer(instance.params, SolverConfig(time_limit=max(0.01, time_limit)))
        rel_sol = rel_opt.solve(instance.market_state)
        inst_with_rel = OptimizationInstance(
            metadata=instance.metadata,
            actual_state=instance.actual_state,
            params=instance.params,
            market_state=instance.market_state,
            relaxation=rel_sol,
        )
        scores = ContinuousRelaxationYBaseline().predict_instance_scores(inst_with_rel)
        cand = construct_candidate_universe(
            scores=scores,
            budget_B=eff_B,
            current_weights=current_w,
            use_safe_augmentation=True,
        )
        elapsed_overhead = time.perf_counter() - t0_ov

    rem_time_limit = max(0.005, time_limit - elapsed_overhead)
    cfg = SolverConfig(time_limit=rem_time_limit, mip_gap=1e-4)

    cb = AnytimeCallback(target_tol=1e-4)
    cb.start()

    def gurobi_cb(model, where):
        cb(model, where)

    if cand is not None:
        optimizer = MultiPeriodReducedMIQPOptimizer(
            full_params=instance.params,
            solver_config=cfg,
        )
        exec_res = optimizer.solve(
            instance.market_state,
            cand,
            callback=gurobi_cb,
            expand_solution=True,
        )
        res = exec_res.optimization_result
    else:
        optimizer = MultiPeriodMIQPOptimizer(
            params=instance.params,
            solver_config=cfg,
        )
        res = optimizer.solve(
            instance.market_state,
            callback=gurobi_cb,
        )
    t_e2e = time.perf_counter() - t_start

    has_feasible = res.is_feasible
    incumbent = float(res.objective_value) if res.objective_value is not None else None
    bound = float(res.best_bound) if res.best_bound is not None else None
    gap = float(res.mip_gap) if res.mip_gap is not None else None

    return {
        "time_budget": time_limit,
        "method": method_type,
        "e2e_time": t_e2e,
        "has_feasible": has_feasible,
        "incumbent": incumbent,
        "best_bound": bound,
        "gap": gap,
        "node_count": getattr(res, "node_count", 0),
        "work": getattr(res, "work", 0.0),
        "trajectory": [
            {
                "t": pt.elapsed_seconds + elapsed_overhead,
                "ub": pt.incumbent,
                "lb": pt.best_bound,
                "gap": pt.gap,
                "nodes": pt.node_count,
            }
            for pt in cb.trajectory.points
        ],
        "time_to_first_feasible": cb.trajectory.time_to_first_feasible,
        "time_to_gap_10pct": cb.trajectory.time_to_gap_10pct,
        "time_to_gap_5pct": cb.trajectory.time_to_gap_5pct,
        "time_to_gap_1pct": cb.trajectory.time_to_gap_1pct,
        "time_to_gap_tol": cb.trajectory.time_to_gap_tol,
    }


def main():
    print("=" * 80)
    print("PHASE 10E: ANYTIME PERFORMANCE & COMMON TIME-BUDGET EXPERIMENTS (Table E)")
    print("=" * 80)

    artifacts_dir = Path("artifacts/scalability")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Configurations within license limits
    test_configs = [
        {"H": 3, "N": 10, "K": 3, "B": 5},
        {"H": 1, "N": 25, "K": 5, "B": 10},
    ]

    time_budgets = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00]
    n_replicates = 5
    methods = ["full", "market_only", "relaxation"]

    records = []
    saved_trajectories = {}

    print(f"\n[1] Evaluating methods across time budgets: {time_budgets} (R={n_replicates})...")
    generator = ScalableInstanceGenerator()

    for cfg in test_configs:
        H, N, K, B = cfg["H"], cfg["N"], cfg["K"], cfg["B"]
        print(f"\n--> Config: H={H}, N={N}, K={K}, B={B}")

        for seed in range(42, 42 + n_replicates):
            instance = generator.generate_instance(
                n_assets=N,
                horizon=H,
                cardinality_limit=K,
                seed=seed,
            )

            for t_budget in time_budgets:
                for method in methods:
                    res_eval = evaluate_fixed_time_limit(
                        instance,
                        time_limit=t_budget,
                        method_type=method,
                        candidate_size=B,
                    )
                    records.append({
                        "H": H,
                        "N": N,
                        "K": K,
                        "B": B,
                        "Seed": seed,
                        "TimeBudget_sec": t_budget,
                        "Method": method,
                        "HasFeasible": res_eval["has_feasible"],
                        "Incumbent": res_eval["incumbent"],
                        "BestBound": res_eval["best_bound"],
                        "MIPGap": res_eval["gap"],
                        "E2E_Time": res_eval["e2e_time"],
                        "Work": res_eval["work"],
                        "TimeToFirstFeasible": res_eval["time_to_first_feasible"],
                        "TimeToGap10pct": res_eval["time_to_gap_10pct"],
                        "TimeToGap5pct": res_eval["time_to_gap_5pct"],
                        "TimeToGap1pct": res_eval["time_to_gap_1pct"],
                        "TimeToGapTol": res_eval["time_to_gap_tol"],
                    })

                    # Save trajectory for one representative instance (seed=42)
                    if seed == 42 and t_budget == 5.00:
                        traj_key = f"H{H}_N{N}_{method}"
                        saved_trajectories[traj_key] = res_eval["trajectory"]

    df_records = pd.DataFrame(records)

    # Aggregate into Table E
    summary_rows = []
    for (H, N, method, t_budget), group in df_records.groupby(["H", "N", "Method", "TimeBudget_sec"]):
        feas_rate = group["HasFeasible"].mean()
        valid_incumbents = group["Incumbent"].dropna()
        valid_gaps = group["MIPGap"].dropna()
        valid_t_first = group["TimeToFirstFeasible"].dropna()

        summary_rows.append({
            "H": H,
            "N": N,
            "Method": method,
            "TimeBudget_sec": t_budget,
            "FeasibleRate": feas_rate,
            "Median_Incumbent": valid_incumbents.median() if len(valid_incumbents) > 0 else np.nan,
            "Median_MIPGap": valid_gaps.median() if len(valid_gaps) > 0 else np.nan,
            "P90_MIPGap": valid_gaps.quantile(0.90) if len(valid_gaps) > 0 else np.nan,
            "Median_TimeToFirstFeasible": valid_t_first.median() if len(valid_t_first) > 0 else np.nan,
            "Mean_E2E_Time": group["E2E_Time"].mean(),
        })

    df_table_e = pd.DataFrame(summary_rows)
    table_e_path = artifacts_dir / "table_e_anytime_quality.csv"
    df_table_e.to_csv(table_e_path, index=False)
    print(f"\n[2] Saved Table E to {table_e_path}")

    # Save trajectories JSON
    traj_path = artifacts_dir / "anytime_trajectories.json"
    with open(traj_path, "w") as f:
        json.dump(saved_trajectories, f, indent=2)
    print(f"    Saved anytime trajectories to {traj_path}")

    print("\nSAMPLE ROWS FROM TABLE E (Anytime Quality):")
    sample_df = df_table_e.sort_values(by=["H", "N", "TimeBudget_sec", "Method"]).head(18)
    print(sample_df[["H", "N", "Method", "TimeBudget_sec", "FeasibleRate", "Median_Incumbent", "Median_MIPGap"]].to_string(index=False))


if __name__ == "__main__":
    main()
