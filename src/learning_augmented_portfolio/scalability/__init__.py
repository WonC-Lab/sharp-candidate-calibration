"""
Scalability & Large-Universe Solver Acceleration Package.
"""

from .failure import SolverExitStatus, classify_solver_outcome
from .instance_generator import FactorModelSpec, ScalableInstanceGenerator
from .model_stats import ModelStructuralStats, compute_theoretical_model_stats
from .anytime import AnytimePoint, AnytimeTrajectory, AnytimeCallback
from .experiment_grid import (
    CardinalityRegime,
    ScalabilityGridConfig,
    compute_cardinality,
    compute_candidate_budget,
    generate_nested_budget_sequence,
)
from .runner import MethodSolveResult, ScalabilityRunner
from .aggregation import RobustDistributionStats, compute_robust_stats

__all__ = [
    "SolverExitStatus",
    "classify_solver_outcome",
    "FactorModelSpec",
    "ScalableInstanceGenerator",
    "ModelStructuralStats",
    "compute_theoretical_model_stats",
    "AnytimePoint",
    "AnytimeTrajectory",
    "AnytimeCallback",
    "CardinalityRegime",
    "ScalabilityGridConfig",
    "compute_cardinality",
    "compute_candidate_budget",
    "generate_nested_budget_sequence",
    "MethodSolveResult",
    "ScalabilityRunner",
    "RobustDistributionStats",
    "compute_robust_stats",
]
