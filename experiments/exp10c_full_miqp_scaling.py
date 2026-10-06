"""
Phase 10 Experiment 10C: Full MIQP Empirical Scaling (Table B).
Evaluates empirical computational burden across supported (N, H) configurations
with R=5 independent replicates per configuration.
Records:
  - Median, IQR, Mean, P90 wall-clock solve time
  - Gurobi Work
  - Node count
  - MIP Gap
  - Timeout rate
  - Time to first feasible
Marks unsupported configurations explicitly as NOT_EXECUTED_LICENSE_LIMIT.
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
)
from learning_augmented_portfolio.optimization.variables import SolverConfig


def run_exp10c_full_miqp_scaling() -> None:
    print("=" * 80)
    print("PHASE 10C: FULL MIQP EMPIRICAL SCALING (Table B)")
    print("=" * 80)

    gen = ScalableInstanceGenerator()
    runner = ScalabilityRunner(SolverConfig(time_limit=15.0, mip_gap=1e-4, threads=1))

    # Supported and license-limited configurations
    eval_grid = [
        # (H, N, is_supported)
        (1, 10, True),
        (1, 20, True),
        (1, 25, True),
        (1, 30, True),
        (1, 32, True),
        (1, 50, False),
        (1, 100, False),
        (1, 250, False),
        (1, 500, False),
        (3, 5, True),
        (3, 8, True),
        (3, 10, True),
        (3, 25, False),
        (3, 50, False),
        (3, 100, False),
        (5, 5, True),
        (5, 10, False),
        (5, 25, False),
        (5, 50, False),
    ]

    replicate_seeds = [42, 43, 44, 45, 46]  # R=5
    rows = []

    print(f"\n[1] Evaluating Full MIQP across {len(eval_grid)} configurations (R={len(replicate_seeds)} replicates)...")

    for H, N, supported in eval_grid:
        K_port = compute_cardinality(N, CardinalityRegime.FIXED, fixed_k=4)

        if not supported:
            # Explicitly mark as NOT_EXECUTED_LICENSE_LIMIT
            rows.append({
                "H": H,
                "N": N,
                "Status": "NOT_EXECUTED_LICENSE_LIMIT",
                "Replicates": 0,
                "MedianTime_ms": None,
                "IQR_Time_ms": None,
                "MeanTime_ms": None,
                "P90_Time_ms": None,
                "MedianWork": None,
                "MeanWork": None,
                "MedianNodes": None,
                "MeanNodes": None,
                "MeanMIPGap": None,
                "TimeoutRate": None,
                "TimeToFirstFeas_ms": None,
            })
            print(f"    - H={H}, N={N:3d} : NOT_EXECUTED_LICENSE_LIMIT")
            continue

        runtimes = []
        works = []
        nodes = []
        gaps = []
        timeouts = []
        first_times = []

        for seed in replicate_seeds:
            inst = gen.generate_instance(n_assets=N, horizon=H, cardinality_limit=K_port, seed=seed)
            res = runner.solve_full_miqp(inst, record_anytime=True)

            if res.status.value in ["OPTIMAL", "TIME_LIMIT_WITH_INCUMBENT"]:
                t_ms = res.timers.t_solver_sec * 1000.0
                runtimes.append(t_ms)
                works.append(res.work)
                nodes.append(res.node_count)
                gaps.append(res.mip_gap or 0.0)
                timeouts.append(1.0 if "TIME_LIMIT" in res.status.value else 0.0)
                if res.anytime_trajectory and res.anytime_trajectory.time_to_first_feasible:
                    first_times.append(res.anytime_trajectory.time_to_first_feasible * 1000.0)
                else:
                    first_times.append(t_ms)

        stats_time = compute_robust_stats(runtimes)
        stats_work = compute_robust_stats(works)
        stats_node = compute_robust_stats(nodes)
        first_time_val = float(np.median(first_times)) if first_times else None

        rows.append({
            "H": H,
            "N": N,
            "Status": "OPTIMAL",
            "Replicates": len(runtimes),
            "MedianTime_ms": stats_time.median,
            "IQR_Time_ms": stats_time.iqr,
            "MeanTime_ms": stats_time.mean,
            "P90_Time_ms": stats_time.p90,
            "MedianWork": stats_work.median,
            "MeanWork": stats_work.mean,
            "MedianNodes": stats_node.median,
            "MeanNodes": stats_node.mean,
            "MeanMIPGap": float(np.mean(gaps)) if gaps else 0.0,
            "TimeoutRate": float(np.mean(timeouts)) if timeouts else 0.0,
            "TimeToFirstFeas_ms": first_time_val,
        })
        print(f"    - H={H}, N={N:3d} : Median Time={stats_time.median:.2f} ms (IQR={stats_time.iqr:.2f} ms, Nodes={stats_node.median:.0f}, Work={stats_work.median:.4f})")

    df = pd.DataFrame(rows)
    out_dir = Path("artifacts/scalability")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "table_b_full_miqp_scaling.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[2] Saved Table B to {csv_path}")


if __name__ == "__main__":
    run_exp10c_full_miqp_scaling()
