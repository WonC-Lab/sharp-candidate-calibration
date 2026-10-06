"""
Evaluation package for learning-augmented portfolio optimization.
Provides:
  - Objective quality and degradation metrics
  - Exact-objective recovery rates
  - End-to-end runtime accounting and acceleration factors
"""

from .objective_quality import (
    InstanceObjectiveComparison,
    ObjectiveQualityReport,
    compare_instance_objectives,
    summarize_objective_quality,
    objective_reports_to_dataframe,
)
from .solver_acceleration import (
    InstanceAccelerationRecord,
    SolverAccelerationReport,
    compute_instance_acceleration,
    summarize_acceleration,
    acceleration_reports_to_dataframe,
)

__all__ = [
    "InstanceObjectiveComparison",
    "ObjectiveQualityReport",
    "compare_instance_objectives",
    "summarize_objective_quality",
    "objective_reports_to_dataframe",
    "InstanceAccelerationRecord",
    "SolverAccelerationReport",
    "compute_instance_acceleration",
    "summarize_acceleration",
    "acceleration_reports_to_dataframe",
]
