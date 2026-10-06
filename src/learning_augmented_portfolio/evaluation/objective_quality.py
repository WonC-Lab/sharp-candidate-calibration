"""
Objective Quality and Solution Degradation Evaluation for Reduced MIQPs.
Measures:
  - Absolute Objective Degradation: Delta J_t = J_t^{red} - J_t^{full}
  - Relative Objective Degradation: Delta J_t^{rel} = (J_t^{red} - J_t^{full}) / max(1, |J_t^{full}|)
  - Exact Objective Recovery Rate: (1/T) sum_t 1{|J_t^{red} - J_t^{full}| <= eps_J}
  - Infeasibility Rate: (# infeasible reduced instances) / T
  - Full-Support Coverage vs Exact Objective Recovery comparison
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set
import numpy as np
import pandas as pd

from ..optimization.reduced_miqp import ReducedMIQPExecutionResult
from ..optimization.variables import MultiPeriodOptimizationResult


@dataclass(frozen=True)
class InstanceObjectiveComparison:
    """Instance-level objective comparison between full MIQP and reduced MIQP."""
    instance_id: str
    method: str
    budget_B: int
    effective_B: int
    is_safe_augmented: bool
    full_objective: Optional[float]
    reduced_objective: Optional[float]
    full_status: str
    reduced_status: str
    is_reduced_feasible: bool
    is_support_contained: bool
    objective_degradation: float
    relative_degradation: float
    is_exact_recovery: bool


@dataclass(frozen=True)
class ObjectiveQualityReport:
    """Dataset-level macro report of objective quality metrics across instances."""
    method: str
    split: str
    budget_B: int
    mean_effective_B: float
    mean_reduction: float
    coverage: float
    exact_objective_rate: float
    mean_objective_degradation: float
    median_objective_degradation: float
    p90_objective_degradation: float
    max_objective_degradation: float
    zero_degradation_ratio: float
    infeasibility_rate: float
    num_instances: int
    is_safe_augmented: bool


def compare_instance_objectives(
    instance_id: str,
    method: str,
    budget_B: int,
    effective_B: int,
    full_result: MultiPeriodOptimizationResult,
    reduced_exec: ReducedMIQPExecutionResult,
    target_support: Optional[Set[int]] = None,
    eps_J: float = 1e-5,
    is_safe_augmented: bool = False,
) -> InstanceObjectiveComparison:
    """
    Compare optimization objective of reduced MIQP against full-universe ground truth.
    """
    cand = reduced_exec.candidate_universe
    red_res = reduced_exec.optimization_result

    J_full = full_result.objective_value
    J_red = red_res.objective_value

    is_support_contained = False
    if target_support is not None:
        is_support_contained = cand.contains_support(target_support)

    if not red_res.is_feasible or J_red is None:
        return InstanceObjectiveComparison(
            instance_id=instance_id,
            method=method,
            budget_B=budget_B,
            effective_B=effective_B,
            is_safe_augmented=is_safe_augmented,
            full_objective=J_full,
            reduced_objective=None,
            full_status=full_result.status,
            reduced_status=red_res.status,
            is_reduced_feasible=False,
            is_support_contained=is_support_contained,
            objective_degradation=float("inf"),
            relative_degradation=float("inf"),
            is_exact_recovery=False,
        )

    if J_full is None:
        raise ValueError(f"Full result for instance {instance_id} has no objective value.")

    # In minimization notation: Delta J = J_red - J_full >= 0 (up to numerical tolerance)
    abs_deg = float(J_red - J_full)
    # Clip tiny negative numerical noise from solver tolerances
    if abs(abs_deg) < 1e-7:
        abs_deg = 0.0

    denom = max(1.0, abs(J_full))
    rel_deg = abs_deg / denom

    is_exact = bool(abs(J_red - J_full) <= eps_J)

    return InstanceObjectiveComparison(
        instance_id=instance_id,
        method=method,
        budget_B=budget_B,
        effective_B=effective_B,
        is_safe_augmented=is_safe_augmented,
        full_objective=J_full,
        reduced_objective=J_red,
        full_status=full_result.status,
        reduced_status=red_res.status,
        is_reduced_feasible=True,
        is_support_contained=is_support_contained,
        objective_degradation=abs_deg,
        relative_degradation=rel_deg,
        is_exact_recovery=is_exact,
    )


def summarize_objective_quality(
    comparisons: List[InstanceObjectiveComparison],
    method: str,
    split: str,
    budget_B: int,
    full_universe_size: int,
    is_safe_augmented: bool = False,
) -> ObjectiveQualityReport:
    """
    Summarize instance objective comparisons into a macro quality report.
    """
    T = len(comparisons)
    if T == 0:
        return ObjectiveQualityReport(
            method=method,
            split=split,
            budget_B=budget_B,
            mean_effective_B=float(budget_B),
            mean_reduction=1.0 - (budget_B / full_universe_size),
            coverage=0.0,
            exact_objective_rate=0.0,
            mean_objective_degradation=0.0,
            median_objective_degradation=0.0,
            p90_objective_degradation=0.0,
            max_objective_degradation=0.0,
            zero_degradation_ratio=0.0,
            infeasibility_rate=0.0,
            num_instances=0,
            is_safe_augmented=is_safe_augmented,
        )

    eff_Bs = [c.effective_B for c in comparisons]
    mean_eff_B = float(np.mean(eff_Bs))
    mean_reduction = 1.0 - (mean_eff_B / full_universe_size)

    coverage = float(np.mean([1.0 if c.is_support_contained else 0.0 for c in comparisons]))
    exact_rate = float(np.mean([1.0 if c.is_exact_recovery else 0.0 for c in comparisons]))

    feasible_comps = [c for c in comparisons if c.is_reduced_feasible]
    infeasibility_rate = float(T - len(feasible_comps)) / float(T)

    if feasible_comps:
        degs = [c.objective_degradation for c in feasible_comps]
        mean_deg = float(np.mean(degs))
        median_deg = float(np.median(degs))
        p90_deg = float(np.percentile(degs, 90))
        max_deg = float(np.max(degs))
        zero_ratio = float(np.mean([1.0 if abs(d) <= 1e-6 else 0.0 for d in degs]))
    else:
        mean_deg = float("inf")
        median_deg = float("inf")
        p90_deg = float("inf")
        max_deg = float("inf")
        zero_ratio = 0.0

    return ObjectiveQualityReport(
        method=method,
        split=split,
        budget_B=budget_B,
        mean_effective_B=mean_eff_B,
        mean_reduction=mean_reduction,
        coverage=coverage,
        exact_objective_rate=exact_rate,
        mean_objective_degradation=mean_deg,
        median_objective_degradation=median_deg,
        p90_objective_degradation=p90_deg,
        max_objective_degradation=max_deg,
        zero_degradation_ratio=zero_ratio,
        infeasibility_rate=infeasibility_rate,
        num_instances=T,
        is_safe_augmented=is_safe_augmented,
    )


def objective_reports_to_dataframe(reports: List[ObjectiveQualityReport]) -> pd.DataFrame:
    """Convert a list of ObjectiveQualityReport objects into a clean pandas DataFrame."""
    records = []
    for r in reports:
        records.append({
            "Method": r.method,
            "Split": r.split,
            "SafeAug": r.is_safe_augmented,
            "B": r.budget_B,
            "Eff_B": round(r.mean_effective_B, 2),
            "Reduction": round(r.mean_reduction, 4),
            "Coverage@B": round(r.coverage, 4),
            "ExactObjRate": round(r.exact_objective_rate, 4),
            "MeanDeltaJ": round(r.mean_objective_degradation, 6),
            "MedianDeltaJ": round(r.median_objective_degradation, 6),
            "P90DeltaJ": round(r.p90_objective_degradation, 6),
            "MaxDeltaJ": round(r.max_objective_degradation, 6),
            "ZeroDegradationRatio": round(r.zero_degradation_ratio, 4),
            "InfeasibilityRate": round(r.infeasibility_rate, 4),
            "NumInstances": r.num_instances,
        })
    return pd.DataFrame(records)
