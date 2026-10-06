"""
Recovery and Certification evaluation metrics.
Tracks:
  - Initial vs final candidate reduction
  - Average recovery stages and full fallback rate
  - Certificate success rate and final certificate gap
  - End-to-end runtime (Fast Mode vs Certified Mode) and Certified Speedup
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


@dataclass
class CertificationEvaluationRecord:
    """Benchmark record for a single recovery & certification evaluation."""
    instance_id: str
    method: str
    fast_mode: bool
    initial_B: int
    final_B: int
    initial_reduction: float
    final_reduction: float
    stages_executed: int
    full_fallback_occurred: bool
    is_certified: bool                     # Relaxation-certified
    relaxation_certified: bool
    full_mip_certified: bool
    any_valid_certificate: bool
    certificate_source: str
    lower_bound: Optional[float]
    lower_bound_best: Optional[float]
    upper_bound: Optional[float]
    certificate_gap: Optional[float]       # Relaxation gap
    relaxation_certificate_gap: Optional[float]
    full_mip_certificate_gap: Optional[float]
    relative_gap: Optional[float]
    true_degradation: Optional[float]
    total_runtime: float
    relaxation_time: float
    full_runtime: float
    speedup_vs_full: float


def summarize_certification_evaluations(
    records: List[CertificationEvaluationRecord],
) -> pd.DataFrame:
    """
    Summarize certification and recovery loop metrics by method and operating mode.
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame([r.__dict__ for r in records])
    grouped = df.groupby(["method", "fast_mode", "initial_B"]).agg(
        Instances=("instance_id", "count"),
        InitialReduction=("initial_reduction", "mean"),
        FinalReduction=("final_reduction", "mean"),
        AvgStages=("stages_executed", "mean"),
        FullFallbackRate=("full_fallback_occurred", "mean"),
        RelaxationCertRate=("relaxation_certified", "mean"),
        FullMIPCertRate=("full_mip_certified", "mean"),
        AnyCertRate=("any_valid_certificate", "mean"),
        MeanRelaxGap=("relaxation_certificate_gap", lambda s: float(s.dropna().mean()) if len(s.dropna()) > 0 else np.nan),
        MeanTrueDegradation=("true_degradation", lambda s: float(s.dropna().mean()) if len(s.dropna()) > 0 else np.nan),
        MeanE2ERuntime_ms=("total_runtime", lambda s: float(s.mean() * 1000)),
        MeanSpeedup=("speedup_vs_full", "mean"),
    ).reset_index()
    return grouped
