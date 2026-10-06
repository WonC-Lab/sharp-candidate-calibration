"""
Gurobi solver wrapper and utilities.
"""

from __future__ import annotations

from typing import Dict, Any
import gurobipy as gp
from gurobipy import GRB

from ..optimization.variables import SolverConfig


class GurobiSolverWrapper:
    """Utility wrapper for creating and configuring Gurobi models and environments."""

    STATUS_MAP: Dict[int, str] = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INPROGRESS: "INPROGRESS",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }

    @staticmethod
    def get_status_string(status_code: int) -> str:
        """Map Gurobi integer status code to human-readable string."""
        return GurobiSolverWrapper.STATUS_MAP.get(status_code, f"UNKNOWN_STATUS_{status_code}")

    @staticmethod
    def configure_params(model: gp.Model, config: SolverConfig) -> None:
        """Apply SolverConfig parameters to a Gurobi model instance."""
        model.Params.TimeLimit = config.time_limit
        model.Params.MIPGap = config.mip_gap
        model.Params.Threads = config.threads
        model.Params.OutputFlag = 1 if config.output_flag else 0
        model.Params.FeasibilityTol = config.feasibility_tol
        model.Params.OptimalityTol = config.optimality_tol
