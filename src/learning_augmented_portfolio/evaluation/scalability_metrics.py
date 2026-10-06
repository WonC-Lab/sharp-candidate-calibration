"""
Scalability Metrics and Timer Accounting Module.
Implements:
  - Exact Component Timer Accounting: T_E2E = T_data + T_feature + T_ranking + T_relax + T_build + T_solver
  - Speedup and Runtime Reduction
  - Dual Objective Quality Metrics:
      * Exact Degradation (when J* is known)
      * Incumbent Comparison & Bound Interval (when Full MIQP times out)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class E2ETimerBreakdown:
    """Component timers for complete end-to-end accounting."""
    t_data_sec: float = 0.0
    t_feature_sec: float = 0.0
    t_ranking_sec: float = 0.0
    t_relax_sec: float = 0.0
    t_build_sec: float = 0.0
    t_solver_sec: float = 0.0

    @property
    def total_e2e_sec(self) -> float:
        return (
            self.t_data_sec
            + self.t_feature_sec
            + self.t_ranking_sec
            + self.t_relax_sec
            + self.t_build_sec
            + self.t_solver_sec
        )

    @property
    def total_e2e_ms(self) -> float:
        return self.total_e2e_sec * 1000.0


def compute_speedup(t_full_e2e: float, t_method_e2e: float) -> float:
    """
    Compute Speedup = T_full / T_method.
    Never truncates negative speedup (Speedup < 1.0 means method is slower).
    """
    if t_method_e2e <= 1e-9:
        return 1.0
    return float(t_full_e2e / t_method_e2e)


def compute_runtime_reduction(t_full_e2e: float, t_method_e2e: float) -> float:
    """Compute RuntimeReduction = 1 - T_method / T_full."""
    if t_full_e2e <= 1e-9:
        return 0.0
    return float(1.0 - (t_method_e2e / t_full_e2e))


def compute_objective_quality(
    obj_method: Optional[float],
    obj_full: Optional[float],
    full_is_optimal: bool,
    ub_full: Optional[float] = None,
    lb_best: Optional[float] = None,
) -> dict[str, Optional[float]]:
    """
    Compute objective quality handling both certified optima and timeout cases.
    """
    if obj_method is None:
        return {
            "abs_degradation": None,
            "rel_degradation": None,
            "incumbent_diff": None,
            "bound_gap": None,
            "is_exact_recovery": False,
        }

    # Case 1: Full solve certified optimal
    if full_is_optimal and obj_full is not None:
        abs_deg = obj_method - obj_full
        rel_deg = abs_deg / max(1.0, abs(obj_full))
        return {
            "abs_degradation": abs_deg,
            "rel_degradation": rel_deg,
            "incumbent_diff": abs_deg,
            "bound_gap": None,
            "is_exact_recovery": bool(abs(abs_deg) <= 1e-4),
        }

    # Case 2: Full solve timed out
    incumbent_diff = (obj_method - ub_full) if ub_full is not None else None
    bound_gap = (obj_method - lb_best) if lb_best is not None else None

    return {
        "abs_degradation": None,
        "rel_degradation": None,
        "incumbent_diff": incumbent_diff,
        "bound_gap": bound_gap,
        "is_exact_recovery": False,
    }
