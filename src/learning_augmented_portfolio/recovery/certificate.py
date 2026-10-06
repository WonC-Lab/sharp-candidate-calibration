"""
Full-Problem Quality Certificate module.
Computes mathematically valid certification bounds:
  LB_t <= J_t^* <= UB_t
  0 <= UB_t - J_t^* <= Gap_t^{cert} = UB_t - LB_t

Strictly distinguishes:
  1. Relaxation-based Certificate: Gap^{relax-cert} = UB - LB^{relax} <= epsilon_{cert}
  2. Full-MIP Solver Certificate: Gap^{MIP-cert} = ObjVal_{full} - ObjBound_{full} <= epsilon_{mip}
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class CertificateSource(str, Enum):
    """Source of the mathematical solution-quality certificate."""
    FULL_RELAXATION = "FULL_RELAXATION"
    FULL_MIP_BOUND = "FULL_MIP_BOUND"
    NONE = "NONE"


@dataclass
class FullProblemCertificate:
    """Full-problem solution quality certificate."""
    instance_id: str
    certificate_source: CertificateSource
    relaxation_certificate_gap: Optional[float]
    relaxation_certified: bool
    full_mip_bound: Optional[float]
    full_mip_certificate_gap: Optional[float]
    full_mip_certified: bool
    any_valid_certificate: bool
    lower_bound_relax: float              # LB^{relax} = J^{relax}
    lower_bound_best: float               # LB^{best} = max(LB^{relax}, LB^{MIP})
    upper_bound: Optional[float]          # UB = J(\hat{x}) from lifted full-feasible solution
    reduced_mip_gap: Optional[float]      # MIPGap_{red} (certifies ONLY reduced problem)
    candidate_budget: int                 # Budget B
    candidate_size: int                   # Effective B^{eff}
    tolerance_met: bool                   # Whether certificate_source != NONE

    @property
    def is_certified(self) -> bool:
        """Backward-compatible alias for relaxation_certified."""
        return self.relaxation_certified

    @property
    def lower_bound(self) -> float:
        """Backward-compatible alias for lower_bound_relax."""
        return self.lower_bound_relax

    @property
    def absolute_gap(self) -> Optional[float]:
        """Backward-compatible alias for relaxation_certificate_gap."""
        return self.relaxation_certificate_gap

    @property
    def relative_gap(self) -> Optional[float]:
        """Relative gap with respect to relaxation lower bound."""
        if self.relaxation_certificate_gap is None or self.upper_bound is None:
            return None
        norm_denom = max(1.0, abs(self.upper_bound), abs(self.lower_bound_relax))
        return self.relaxation_certificate_gap / norm_denom


def compute_full_problem_certificate(
    instance_id: str,
    lower_bound: float,
    upper_bound: Optional[float],
    reduced_mip_gap: Optional[float],
    candidate_budget: int,
    candidate_size: int,
    epsilon_cert: float = 1e-4,
    full_mip_bound: Optional[float] = None,
    full_mip_gap: Optional[float] = None,
    epsilon_mip: float = 1e-4,
    is_full_universe: bool = False,
) -> FullProblemCertificate:
    """
    Compute rigorous full-problem certificate distinguishing relaxation vs full-MIP certificates.
    """
    if upper_bound is None or not np.isfinite(upper_bound) or not np.isfinite(lower_bound):
        return FullProblemCertificate(
            instance_id=instance_id,
            certificate_source=CertificateSource.NONE,
            relaxation_certificate_gap=None,
            relaxation_certified=False,
            full_mip_bound=full_mip_bound,
            full_mip_certificate_gap=full_mip_gap,
            full_mip_certified=False,
            any_valid_certificate=False,
            lower_bound_relax=lower_bound,
            lower_bound_best=lower_bound,
            upper_bound=upper_bound,
            reduced_mip_gap=reduced_mip_gap,
            candidate_budget=candidate_budget,
            candidate_size=candidate_size,
            tolerance_met=False,
        )

    # 1. Relaxation Certificate Evaluation
    raw_relax_gap = float(upper_bound - lower_bound)
    if raw_relax_gap < -1e-5:
        # Severe lower bound violation
        relax_gap = None
        relax_certified = False
    else:
        relax_gap = max(0.0, raw_relax_gap)
        relax_certified = (relax_gap <= epsilon_cert)

    # 2. Full-MIP Solver Certificate Evaluation (Valid only at full universe B = N)
    mip_certified = False
    mip_gap_val = None
    if is_full_universe and full_mip_bound is not None and np.isfinite(full_mip_bound):
        raw_mip_gap = float(upper_bound - full_mip_bound)
        mip_gap_val = max(0.0, raw_mip_gap)
        # Check against solver MIP tolerance or user epsilon
        tol_threshold = max(epsilon_mip, (reduced_mip_gap or 0.0))
        mip_certified = (mip_gap_val <= tol_threshold + 1e-6)

    # Best combined lower bound
    lb_best = lower_bound
    if full_mip_bound is not None and np.isfinite(full_mip_bound):
        lb_best = max(lb_best, full_mip_bound)

    # 3. Determine Certificate Source
    if relax_certified:
        source = CertificateSource.FULL_RELAXATION
    elif mip_certified:
        source = CertificateSource.FULL_MIP_BOUND
    else:
        source = CertificateSource.NONE

    any_cert = (source != CertificateSource.NONE)

    return FullProblemCertificate(
        instance_id=instance_id,
        certificate_source=source,
        relaxation_certificate_gap=relax_gap,
        relaxation_certified=relax_certified,
        full_mip_bound=full_mip_bound,
        full_mip_certificate_gap=mip_gap_val,
        full_mip_certified=mip_certified,
        any_valid_certificate=any_cert,
        lower_bound_relax=lower_bound,
        lower_bound_best=lb_best,
        upper_bound=upper_bound,
        reduced_mip_gap=reduced_mip_gap,
        candidate_budget=candidate_budget,
        candidate_size=candidate_size,
        tolerance_met=any_cert,
    )
