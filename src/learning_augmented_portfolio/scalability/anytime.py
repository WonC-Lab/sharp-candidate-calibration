"""
Anytime Solver Telemetry and Callback Module.
Tracks the evolution of upper bound UB(t), lower bound LB(t), MIPGap(t),
nodes, and work over wall-clock elapsed time during branch-and-bound optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import List, Optional, Tuple
import gurobipy as gp
from gurobipy import GRB
import numpy as np


@dataclass
class AnytimePoint:
    """Snapshot of solver state at a specific elapsed time."""
    elapsed_seconds: float
    incumbent: Optional[float]
    best_bound: Optional[float]
    gap: Optional[float]
    node_count: int
    work: float


@dataclass
class AnytimeTrajectory:
    """Complete anytime trajectory across solver execution."""
    points: List[AnytimePoint] = field(default_factory=list)
    time_to_first_feasible: Optional[float] = None
    time_to_gap_10pct: Optional[float] = None
    time_to_gap_5pct: Optional[float] = None
    time_to_gap_1pct: Optional[float] = None
    time_to_gap_tol: Optional[float] = None

    def add_point(self, point: AnytimePoint, target_tol: float = 1e-4) -> None:
        self.points.append(point)
        t = point.elapsed_seconds

        # First feasible
        if point.incumbent is not None and self.time_to_first_feasible is None:
            self.time_to_first_feasible = t

        # Gap thresholds
        if point.gap is not None:
            if point.gap <= 0.10 and self.time_to_gap_10pct is None:
                self.time_to_gap_10pct = t
            if point.gap <= 0.05 and self.time_to_gap_5pct is None:
                self.time_to_gap_5pct = t
            if point.gap <= 0.01 and self.time_to_gap_1pct is None:
                self.time_to_gap_1pct = t
            if point.gap <= target_tol and self.time_to_gap_tol is None:
                self.time_to_gap_tol = t


class AnytimeCallback:
    """
    Gurobi callback that samples solver progress and records AnytimeTrajectory.
    """

    def __init__(self, target_tol: float = 1e-4, min_sample_interval_sec: float = 0.005) -> None:
        self.target_tol = target_tol
        self.min_sample_interval = min_sample_interval_sec
        self.trajectory = AnytimeTrajectory()
        self.start_time: float = 0.0
        self.last_sample_time: float = -1.0

    def start(self) -> None:
        self.start_time = time.perf_counter()
        self.last_sample_time = -1.0
        self.trajectory = AnytimeTrajectory()

    def __call__(self, model: gp.Model, where: int) -> None:
        t_now = time.perf_counter()
        elapsed = t_now - self.start_time

        # Check sampling throttle
        if self.last_sample_time >= 0 and (elapsed - self.last_sample_time) < self.min_sample_interval:
            return

        if where in (GRB.Callback.MIP, GRB.Callback.MIPNODE):
            try:
                obj_bst = model.cbGet(GRB.Callback.MIP_OBJBST)
                # Gurobi infinity marker for no incumbent
                incumbent = float(obj_bst) if abs(obj_bst) < 1e20 else None
            except gp.GurobiError:
                incumbent = None

            try:
                obj_bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                bound = float(obj_bnd) if abs(obj_bnd) < 1e20 else None
            except gp.GurobiError:
                bound = None

            try:
                nodes = int(model.cbGet(GRB.Callback.MIP_NODCNT))
            except gp.GurobiError:
                nodes = 0

            try:
                work = float(model.cbGet(GRB.Callback.RUNTIME))
            except gp.GurobiError:
                work = 0.0

            gap = None
            if incumbent is not None and bound is not None:
                denom = max(1.0, abs(incumbent))
                gap = max(0.0, float(incumbent - bound) / denom)

            pt = AnytimePoint(
                elapsed_seconds=elapsed,
                incumbent=incumbent,
                best_bound=bound,
                gap=gap,
                node_count=nodes,
                work=work,
            )
            self.trajectory.add_point(pt, self.target_tol)
            self.last_sample_time = elapsed
