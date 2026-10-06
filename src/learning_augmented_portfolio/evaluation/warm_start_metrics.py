"""
Warm-start evaluation and benchmarking metrics.
Tracks:
  - Start construction & repair times
  - Primal feasibility rate
  - Time to first feasible solution (T_{first}) and Delta T_{first}
  - First incumbent quality
  - Node count, work, runtime, and final MIP gap
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


@dataclass
class WarmStartEvaluationRecord:
    """Benchmark record for a single warm-start evaluation."""
    instance_id: str
    start_type: str
    construction_time: float
    repair_time: float
    partial_start_provided: bool
    complete_start_feasible: Optional[bool]  # None for NoStart (N/A)
    repair_success_rate: Optional[float]     # None if repair not attempted
    first_feasible_time: Optional[float]
    delta_t_first: Optional[float]
    first_incumbent_obj: Optional[float]
    final_runtime: float
    work: float
    node_count: int
    final_objective: float
    best_bound: float
    mip_gap: float

    @property
    def is_start_feasible(self) -> bool:
        """Backward-compatible boolean property."""
        return bool(self.complete_start_feasible) if self.complete_start_feasible is not None else False


def summarize_warm_start_evaluations(
    records: List[WarmStartEvaluationRecord],
) -> pd.DataFrame:
    """
    Summarize warm-start performance across test instances.
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            **r.__dict__,
            "complete_start_feasible": r.complete_start_feasible,
            "repair_success_rate": r.repair_success_rate,
        }
        for r in records
    ])

    grouped = df.groupby("start_type").agg(
        Instances=("instance_id", "count"),
        PartialStartProvidedRate=("partial_start_provided", "mean"),
        CompleteFeasibleRate=("complete_start_feasible", lambda s: float(s.dropna().mean()) if len(s.dropna()) > 0 else np.nan),
        RepairSuccessRate=("repair_success_rate", lambda s: float(s.dropna().mean()) if len(s.dropna()) > 0 else np.nan),
        MeanConstructTime_ms=("construction_time", lambda s: float(s.mean() * 1000)),
        MeanRepairTime_ms=("repair_time", lambda s: float(s.mean() * 1000)),
        MeanTFirst_ms=("first_feasible_time", lambda s: float(s.dropna().mean() * 1000) if len(s.dropna()) > 0 else np.nan),
        MeanRuntime_ms=("final_runtime", lambda s: float(s.mean() * 1000)),
        MeanNodeCount=("node_count", "mean"),
        MeanWork=("work", "mean"),
        MeanObjective=("final_objective", "mean"),
        MeanGap=("mip_gap", "mean"),
    ).reset_index()
    return grouped
