"""
Solver Exit and Failure Classification Module.
Strictly distinguishes optimal termination, timeouts with/without incumbent,
infeasibility, numerical/memory failures, and license-induced limits.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class SolverExitStatus(str, Enum):
    """Fine-grained classification of solver execution outcomes."""
    OPTIMAL = "OPTIMAL"
    TIME_LIMIT_WITH_INCUMBENT = "TIME_LIMIT_WITH_INCUMBENT"
    TIME_LIMIT_NO_INCUMBENT = "TIME_LIMIT_NO_INCUMBENT"
    INFEASIBLE = "INFEASIBLE"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    MEMORY_FAILURE = "MEMORY_FAILURE"
    LICENSE_LIMIT = "LICENSE_LIMIT"
    BUILD_FAILURE = "BUILD_FAILURE"
    NOT_EXECUTED_LICENSE_LIMIT = "NOT_EXECUTED_LICENSE_LIMIT"


def classify_solver_outcome(
    is_optimal: bool,
    is_feasible: bool,
    has_incumbent: bool,
    gurobi_status: int,
    error_message: Optional[str] = None,
) -> SolverExitStatus:
    """
    Classify raw Gurobi termination status into standardized SolverExitStatus.
    """
    if error_message is not None:
        err_lower = error_message.lower()
        if "size-limited license" in err_lower or "license" in err_lower:
            return SolverExitStatus.LICENSE_LIMIT
        if "out of memory" in err_lower or "memory" in err_lower:
            return SolverExitStatus.MEMORY_FAILURE
        return SolverExitStatus.BUILD_FAILURE

    # Gurobi status codes:
    # 2: OPTIMAL
    # 3: INFEASIBLE
    # 4: INF_OR_UNBD
    # 9: TIME_LIMIT
    # 10: NODE_LIMIT
    # 11: ITERATION_LIMIT
    # 12: NUMERIC
    if is_optimal or gurobi_status == 2:
        return SolverExitStatus.OPTIMAL

    if gurobi_status in (9, 10, 11):
        if has_incumbent or is_feasible:
            return SolverExitStatus.TIME_LIMIT_WITH_INCUMBENT
        return SolverExitStatus.TIME_LIMIT_NO_INCUMBENT

    if gurobi_status in (3, 4):
        return SolverExitStatus.INFEASIBLE

    if gurobi_status == 12:
        return SolverExitStatus.NUMERICAL_FAILURE

    if has_incumbent or is_feasible:
        return SolverExitStatus.TIME_LIMIT_WITH_INCUMBENT

    return SolverExitStatus.TIME_LIMIT_NO_INCUMBENT
