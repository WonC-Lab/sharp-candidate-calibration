"""
Conformal calibration module for coverage-calibrated candidate reduction.
"""

from .calibrator import (
    ConformalCalibrationResult,
    SplitConformalCalibrator,
    compute_support_depth,
)

__all__ = [
    "ConformalCalibrationResult",
    "SplitConformalCalibrator",
    "compute_support_depth",
]
