"""Deterministic continuous-relaxation and full-MIP certificate evaluators."""
from typing import Dict, Any, Optional

def relaxation_certificate(
    upper_bound: float,
    lower_bound_relax: float,
    epsilon_cert: float,
    solver_status: int = 2
) -> Dict[str, Any]:
    """Evaluate whether candidate solution is certified by a continuous relaxation lower bound.
    
    Condition: UB - LB_relax <= epsilon_cert.
    Since LB_relax <= J^* <= UB, this guarantees:
        0 <= UB - J^* <= epsilon_cert.
    """
    abs_gap = upper_bound - lower_bound_relax
    # Scale-safe relative gap
    denom = max(1.0, abs(upper_bound))
    rel_gap = max(0.0, abs_gap) / denom
    
    is_certified = (abs_gap <= epsilon_cert + 1e-9)
    
    return {
        "certified": is_certified,
        "certificate_type": "RELAXATION_CERTIFICATE" if is_certified else "NONE",
        "upper_bound": float(upper_bound),
        "lower_bound": float(lower_bound_relax),
        "absolute_gap": float(abs_gap),
        "relative_gap": float(rel_gap),
        "tolerance": float(epsilon_cert),
        "solver_status": int(solver_status)
    }

def full_mip_certificate(
    upper_bound: float,
    lower_bound_mip: float,
    epsilon_abs: float = 1e-5,
    epsilon_rel: float = 1e-4,
    solver_status: int = 2
) -> Dict[str, Any]:
    """Evaluate whether full-universe MIP solve satisfies mathematical optimality certification.
    
    A run is certified if and only if the numerical gap UB_mip - LB_mip satisfies either
    the absolute or relative tolerance criteria.
    Note: A run that hits TimeLimit with an incumbent having gap <= tolerance IS certified.
    A run with status OPTIMAL but having numerical gap > tolerance is NOT certified.
    """
    abs_gap = upper_bound - lower_bound_mip
    denom = max(1.0, abs(upper_bound))
    rel_gap = max(0.0, abs_gap) / denom
    
    certified_abs = (abs_gap <= epsilon_abs + 1e-9)
    certified_rel = (rel_gap <= epsilon_rel + 1e-9)
    is_certified = certified_abs or certified_rel
    
    return {
        "certified": is_certified,
        "certificate_type": "FULL_MIP_CERTIFICATE" if is_certified else "NONE",
        "upper_bound": float(upper_bound),
        "lower_bound": float(lower_bound_mip),
        "absolute_gap": float(abs_gap),
        "relative_gap": float(rel_gap),
        "tolerance": float(epsilon_abs),
        "solver_status": int(solver_status)
    }
