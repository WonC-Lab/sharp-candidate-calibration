"""
Gurobi callback utilities for tracking anytime B&B solver trajectories.
"""

from __future__ import annotations

import time
from typing import List, Tuple, Any
import gurobipy as gp
from gurobipy import GRB


class OptimizationProgressLogger:
    """
    Gurobi callback logger that records progress snapshots:
        (elapsed_time, incumbent_obj, best_bound, mip_gap, node_count)
    """

    def __init__(self) -> None:
        self.trajectory: List[Tuple[float, float, float, float, int]] = []
        self.start_time: float = 0.0

    def start(self) -> None:
        """Reset and start the timer."""
        self.start_time = time.perf_counter()
        self.trajectory.clear()

    def __call__(self, model: gp.Model, where: int) -> None:
        """Callback entry point invoked by Gurobi during B&B execution."""
        if where in (GRB.Callback.MIP, GRB.Callback.MIPSOL):
            try:
                node_cnt = int(model.cbGet(GRB.Callback.MIP_NODCNT))
                obj_bst = float(model.cbGet(GRB.Callback.MIP_OBJBST))
                obj_bnd = float(model.cbGet(GRB.Callback.MIP_OBJBND))
                elapsed = time.perf_counter() - self.start_time

                # Compute robust relative gap
                if abs(obj_bst) < 1e-10 and abs(obj_bnd) < 1e-10:
                    gap = 0.0
                elif abs(obj_bst) >= 1e10:
                    gap = 1.0
                else:
                    denom = max(1.0, abs(obj_bst), abs(obj_bnd))
                    gap = max(0.0, (obj_bst - obj_bnd) / denom)

                self.trajectory.append((elapsed, obj_bst, obj_bnd, gap, node_cnt))
            except (gp.GurobiError, AttributeError):
                pass

    @property
    def first_feasible_time(self) -> float | None:
        """Return the elapsed time in seconds when the first feasible incumbent was discovered."""
        for elapsed, obj_bst, _, _, _ in self.trajectory:
            if abs(obj_bst) < 1e9:
                return elapsed
        return None

    @property
    def first_feasible_obj(self) -> float | None:
        """Return the objective value of the first discovered feasible incumbent."""
        for _, obj_bst, _, _, _ in self.trajectory:
            if abs(obj_bst) < 1e9:
                return obj_bst
        return None
