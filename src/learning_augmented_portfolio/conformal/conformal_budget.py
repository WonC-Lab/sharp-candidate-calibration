"""Conformal calibration for candidate budgets under candidate-family-dependent B_{min, C}(theta).

Supports zero-budget exactness, instance-dependent lower clamping, and explicit
handling of infinite/uncertified budgets without silent truncation.
"""
import math
from typing import Dict, List, Optional, Any, Union
import numpy as np

def calibrate_marginal_budget(
    calibration_budgets: Union[List[Union[int, float]], np.ndarray],
    alpha: float,
    N: int,
    B_min_new: int = 1,
    target_type: str = "certified_epsilon"
) -> Dict[str, Any]:
    r"""Calibrate marginal conformal candidate budget with instance-dependent clamping.
    
    Q_alpha is the k_alpha-th order statistic of calibration budgets, where
    k_alpha = ceil((n + 1) * (1 - alpha)).
    
    If Q_alpha is infinite (e.g. uncertified budgets under time limits), returns
    explicit ABSTAIN / FULL-CERTIFICATION-UNAVAILABLE status rather than falsely
    claiming B=N guarantees epsilon quality.
    
    If Q_alpha is finite, the deployed budget for a new instance is clipped to:
    B_hat_alpha = min(N, max(B_{min, C}(theta_new), Q_alpha)).
    
    Args:
        calibration_budgets: Array of exactness budgets on calibration set (size n).
        alpha: Target error rate in (0, 1).
        N: Full universe size.
        B_min_new: Minimum feasible budget for the test instance B_{min, C}(theta_new) \in {0, ..., N}.
        target_type: "certified_epsilon" or "reference".
        
    Returns:
        Dict containing calibrated budget, quantile, order statistic index, and fallback/abstain flags.
    """
    cal_list = [float(x) for x in calibration_budgets]
    n = len(cal_list)
    if n == 0:
        return {
            "B_alpha": N,
            "k_alpha": 1,
            "quantile": float("inf"),
            "fallback_to_full": True,
            "certification_guaranteed": False,
            "status": "INSUFFICIENT_CALIBRATION_DATA",
            "B_min_C": B_min_new,
            "alpha": alpha,
            "n_cal": 0,
            "target_type": target_type
        }
        
    k_alpha = int(math.ceil((n + 1) * (1.0 - alpha)))
    
    if k_alpha > n:
        # Insufficient granularity: safe fallback to full universe
        return {
            "B_alpha": N,
            "k_alpha": k_alpha,
            "quantile": float("inf"),
            "fallback_to_full": True,
            "certification_guaranteed": False,
            "status": "FALLBACK_GRANULARITY_LIMIT",
            "B_min_C": B_min_new,
            "alpha": alpha,
            "n_cal": n,
            "target_type": target_type
        }
        
    sorted_budgets = sorted(cal_list)
    Q_alpha = sorted_budgets[k_alpha - 1]
    
    if math.isinf(Q_alpha):
        return {
            "B_alpha": N,
            "k_alpha": k_alpha,
            "quantile": float("inf"),
            "fallback_to_full": True,
            "certification_guaranteed": False,
            "status": "ABSTAIN_UNCERTIFIED_QUANTILE",
            "B_min_C": B_min_new,
            "alpha": alpha,
            "n_cal": n,
            "target_type": target_type
        }
        
    # Enforce instance-adaptive lower clamping: B_hat = min(N, max(B_min_new, int(Q_alpha)))
    Q_int = int(Q_alpha)
    B_hat = min(N, max(B_min_new, Q_int))
    
    return {
        "B_alpha": B_hat,
        "k_alpha": k_alpha,
        "quantile": Q_int,
        "fallback_to_full": (B_hat == N),
        "certification_guaranteed": True,
        "status": "CALIBRATED",
        "B_min_C": B_min_new,
        "alpha": alpha,
        "n_cal": n,
        "target_type": target_type
    }

def calibrate_adaptive_budget(
    calibration_residuals: Union[List[float], np.ndarray],
    predicted_budget_new: float,
    alpha: float,
    N: int,
    B_min_new: int = 1,
    target_type: str = "certified_epsilon"
) -> Dict[str, Any]:
    r"""Calibrate instance-adaptive split-conformal candidate budget with clamping.
    
    Given one-sided residuals R_j = B_j^dagger - \hat{b}(X_j) on D_cal, compute
    the conformal quantile q_alpha = R_{(k_cal)} where k_cal = ceil((n_cal + 1)(1 - alpha)).
    Then B_alpha(X) = clip(ceil(predicted_budget + q_alpha), B_{min, C}(theta_new), N).
    
    Args:
        calibration_residuals: One-sided residuals on calibration set.
        predicted_budget_new: Raw point prediction \hat{b}(X_new) for test instance.
        alpha: Target error rate.
        N: Full universe size.
        B_min_new: Feasibility floor B_{min, C}(theta_new) \in {0, ..., N}.
        target_type: "certified_epsilon" or "reference".
        
    Returns:
        Dict containing adaptive budget, conformal quantile, and diagnostics.
    """
    res_list = [float(x) for x in calibration_residuals]
    n = len(res_list)
    if n == 0:
        return {
            "B_alpha": N,
            "k_cal": 1,
            "quantile": float("inf"),
            "fallback_to_full": True,
            "certification_guaranteed": False,
            "status": "INSUFFICIENT_CALIBRATION_DATA",
            "B_min_C": B_min_new,
            "alpha": alpha,
            "n_cal": 0,
            "target_type": target_type
        }
        
    k_cal = int(math.ceil((n + 1) * (1.0 - alpha)))
    
    if k_cal > n:
        return {
            "B_alpha": N,
            "k_cal": k_cal,
            "quantile": float("inf"),
            "fallback_to_full": True,
            "certification_guaranteed": False,
            "status": "FALLBACK_GRANULARITY_LIMIT",
            "B_min_C": B_min_new,
            "alpha": alpha,
            "n_cal": n,
            "target_type": target_type
        }
        
    sorted_res = sorted(res_list)
    q_alpha = sorted_res[k_cal - 1]
    
    if math.isinf(q_alpha):
        return {
            "B_alpha": N,
            "k_cal": k_cal,
            "quantile": float("inf"),
            "fallback_to_full": True,
            "certification_guaranteed": False,
            "status": "ABSTAIN_UNCERTIFIED_QUANTILE",
            "B_min_C": B_min_new,
            "alpha": alpha,
            "n_cal": n,
            "target_type": target_type
        }
        
    raw_budget = math.ceil(predicted_budget_new + q_alpha)
    clamped_budget = min(N, max(B_min_new, raw_budget))
    
    return {
        "B_alpha": clamped_budget,
        "k_cal": k_cal,
        "quantile": q_alpha,
        "fallback_to_full": (clamped_budget == N),
        "certification_guaranteed": True,
        "status": "CALIBRATED",
        "B_min_C": B_min_new,
        "alpha": alpha,
        "n_cal": n,
        "target_type": target_type
    }
