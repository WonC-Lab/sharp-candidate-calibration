"""
Recovery module initialization.
Provides nested candidate sequence generation, solution lifting, independent feasibility checking,
full-problem relaxation certificates, and recovery loops.
"""

from .candidate_sets import generate_nested_candidate_sequence
from .feasibility import FullProblemFeasibilityChecker, FeasibilityAuditReport
from .certificate import FullProblemCertificate, compute_full_problem_certificate
from .recovery_loop import NestedRecoveryOptimizer, RecoveryLoopResult

__all__ = [
    "generate_nested_candidate_sequence",
    "FullProblemFeasibilityChecker",
    "FeasibilityAuditReport",
    "FullProblemCertificate",
    "compute_full_problem_certificate",
    "NestedRecoveryOptimizer",
    "RecoveryLoopResult",
]
