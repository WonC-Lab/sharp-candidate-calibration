"""
Statistical Aggregation and Paired Comparison Module.
Computes robust non-parametric summary statistics (Median, IQR, P90, Mean)
for heavy-tailed mixed-integer programming runtime distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import numpy as np


@dataclass
class RobustDistributionStats:
    """Robust summary statistics for heavy-tailed optimization metrics."""
    count: int
    mean: float
    std: float
    median: float
    p25: float
    p75: float
    iqr: float
    p90: float
    min_val: float
    max_val: float


def compute_robust_stats(values: Sequence[float]) -> RobustDistributionStats:
    """
    Compute robust statistics over an array of numerical observations.
    """
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return RobustDistributionStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    arr = np.array(clean, dtype=float)
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    return RobustDistributionStats(
        count=len(arr),
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        median=float(np.median(arr)),
        p25=p25,
        p75=p75,
        iqr=float(p75 - p25),
        p90=float(np.percentile(arr, 90)),
        min_val=float(np.min(arr)),
        max_val=float(np.max(arr)),
    )
