"""
Computational acceleration and end-to-end runtime accounting for reduced MIQPs.
Implements:
  - Solver-only speedup factor: T^{full,solver} / T^{red,solver}
  - Strict End-to-End Runtime Accounting:
      T^{E2E} = T^{relax} + T^{feature} + T^{preprocess} + T^{inference} + T^{candidate} + T^{build} + T^{solver}
    (Only components required by each specific method are included)
  - End-to-End Runtime Reduction: 1 - T^{E2E,red} / T^{full}
  - End-to-End Speedup Factor: T^{full} / T^{E2E,red}
  - Branch-and-Bound node count reduction and Gurobi Work reduction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InstanceAccelerationRecord:
    """Detailed computational runtime and solver telemetry record for a single instance."""
    instance_id: str
    method: str
    budget_B: int
    effective_B: int
    is_safe_augmented: bool
    # Timers (seconds)
    full_solver_time: float
    reduced_solver_time: float
    build_time: float
    candidate_time: float
    relax_time: float
    feature_time: float
    inference_time: float
    e2e_reduced_time: float
    # Telemetry
    full_nodes: int
    reduced_nodes: int
    full_work: Optional[float]
    reduced_work: Optional[float]
    # Metrics
    solver_speedup: float
    e2e_speedup: float
    runtime_reduction: float


@dataclass(frozen=True)
class SolverAccelerationReport:
    """Macro-aggregated acceleration report across an evaluation split."""
    method: str
    split: str
    budget_B: int
    mean_effective_B: float
    mean_reduction: float
    mean_full_solver_time: float
    mean_reduced_solver_time: float
    mean_e2e_time: float
    mean_solver_speedup: float
    mean_e2e_speedup: float
    mean_runtime_reduction: float
    node_count_reduction: float
    work_reduction: float
    num_instances: int
    is_safe_augmented: bool


def compute_instance_acceleration(
    instance_id: str,
    method: str,
    budget_B: int,
    effective_B: int,
    full_solver_time: float,
    reduced_solver_time: float,
    build_time: float = 0.0,
    candidate_time: float = 0.0,
    relax_time: float = 0.0,
    feature_time: float = 0.0,
    inference_time: float = 0.0,
    full_nodes: int = 0,
    reduced_nodes: int = 0,
    full_work: Optional[float] = None,
    reduced_work: Optional[float] = None,
    is_safe_augmented: bool = False,
) -> InstanceAccelerationRecord:
    """
    Compute solver-only and end-to-end acceleration metrics for an optimization instance.
    Strictly accounts for required overheads.
    """
    # Total E2E time: sum of required components
    e2e_time = (
        relax_time
        + feature_time
        + inference_time
        + candidate_time
        + build_time
        + reduced_solver_time
    )

    # Solver speedup
    solver_sp = (
        full_solver_time / reduced_solver_time
        if reduced_solver_time > 1e-9 else 1.0
    )

    # E2E speedup and reduction
    e2e_sp = (
        full_solver_time / e2e_time
        if e2e_time > 1e-9 else 1.0
    )

    # Runtime reduction: 1 - T^{E2E} / T^{full} (can be negative if overhead exceeds solver savings)
    runtime_red = (
        1.0 - (e2e_time / full_solver_time)
        if full_solver_time > 1e-9 else 0.0
    )

    return InstanceAccelerationRecord(
        instance_id=instance_id,
        method=method,
        budget_B=budget_B,
        effective_B=effective_B,
        is_safe_augmented=is_safe_augmented,
        full_solver_time=full_solver_time,
        reduced_solver_time=reduced_solver_time,
        build_time=build_time,
        candidate_time=candidate_time,
        relax_time=relax_time,
        feature_time=feature_time,
        inference_time=inference_time,
        e2e_reduced_time=e2e_time,
        full_nodes=full_nodes,
        reduced_nodes=reduced_nodes,
        full_work=full_work,
        reduced_work=reduced_work,
        solver_speedup=solver_sp,
        e2e_speedup=e2e_sp,
        runtime_reduction=runtime_red,
    )


def summarize_acceleration(
    records: List[InstanceAccelerationRecord],
    method: str,
    split: str,
    budget_B: int,
    full_universe_size: int,
    is_safe_augmented: bool = False,
) -> SolverAccelerationReport:
    """
    Macro-aggregate instance acceleration records into a summary report.
    """
    T = len(records)
    if T == 0:
        return SolverAccelerationReport(
            method=method,
            split=split,
            budget_B=budget_B,
            mean_effective_B=float(budget_B),
            mean_reduction=1.0 - (budget_B / full_universe_size),
            mean_full_solver_time=0.0,
            mean_reduced_solver_time=0.0,
            mean_e2e_time=0.0,
            mean_solver_speedup=1.0,
            mean_e2e_speedup=1.0,
            mean_runtime_reduction=0.0,
            node_count_reduction=0.0,
            work_reduction=0.0,
            num_instances=0,
            is_safe_augmented=is_safe_augmented,
        )

    mean_eff_B = float(np.mean([r.effective_B for r in records]))
    mean_reduction = 1.0 - (mean_eff_B / full_universe_size)

    mean_full_time = float(np.mean([r.full_solver_time for r in records]))
    mean_red_time = float(np.mean([r.reduced_solver_time for r in records]))
    mean_e2e_time = float(np.mean([r.e2e_reduced_time for r in records]))

    mean_solver_sp = float(np.mean([r.solver_speedup for r in records]))
    mean_e2e_sp = float(np.mean([r.e2e_speedup for r in records]))
    mean_rt_red = float(np.mean([r.runtime_reduction for r in records]))

    # Node count reduction
    tot_full_nodes = sum(r.full_nodes for r in records)
    tot_red_nodes = sum(r.reduced_nodes for r in records)
    node_red = (
        1.0 - (tot_red_nodes / tot_full_nodes)
        if tot_full_nodes > 0 else 0.0
    )

    # Work reduction if available
    work_pairs = [(r.full_work, r.reduced_work) for r in records if r.full_work is not None and r.reduced_work is not None]
    if work_pairs:
        tot_full_work = sum(p[0] for p in work_pairs)
        tot_red_work = sum(p[1] for p in work_pairs)
        work_red = (
            1.0 - (tot_red_work / tot_full_work)
            if tot_full_work > 1e-9 else 0.0
        )
    else:
        work_red = 0.0

    return SolverAccelerationReport(
        method=method,
        split=split,
        budget_B=budget_B,
        mean_effective_B=mean_eff_B,
        mean_reduction=mean_reduction,
        mean_full_solver_time=mean_full_time,
        mean_reduced_solver_time=mean_red_time,
        mean_e2e_time=mean_e2e_time,
        mean_solver_speedup=mean_solver_sp,
        mean_e2e_speedup=mean_e2e_sp,
        mean_runtime_reduction=mean_rt_red,
        node_count_reduction=node_red,
        work_reduction=work_red,
        num_instances=T,
        is_safe_augmented=is_safe_augmented,
    )


def acceleration_reports_to_dataframe(reports: List[SolverAccelerationReport]) -> pd.DataFrame:
    """Convert a list of SolverAccelerationReport objects into a clean pandas DataFrame."""
    records = []
    for r in reports:
        records.append({
            "Method": r.method,
            "Split": r.split,
            "SafeAug": r.is_safe_augmented,
            "B": r.budget_B,
            "Eff_B": round(r.mean_effective_B, 2),
            "Reduction": round(r.mean_reduction, 4),
            "FullTime_ms": round(r.mean_full_solver_time * 1000, 2),
            "RedSolverTime_ms": round(r.mean_reduced_solver_time * 1000, 2),
            "E2ETime_ms": round(r.mean_e2e_time * 1000, 2),
            "SolverSpeedup": round(r.mean_solver_speedup, 3),
            "E2ESpeedup": round(r.mean_e2e_speedup, 3),
            "RuntimeReduction": round(r.mean_runtime_reduction, 4),
            "NodeReduction": round(r.node_count_reduction, 4),
            "WorkReduction": round(r.work_reduction, 4),
            "NumInstances": r.num_instances,
        })
    return pd.DataFrame(records)
