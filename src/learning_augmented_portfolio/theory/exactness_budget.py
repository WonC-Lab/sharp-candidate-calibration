"""Theory utilities for Exactness Budget and Support-Entry Depth.

Implements mathematically rigorous oracle targets, empirical diagnostic targets,
and solver-certified epsilon-quality targets with zero-budget consistency.
"""
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import math
import numpy as np

def compute_family_feasibility_floor_cflp(
    capacities: np.ndarray,
    demands: np.ndarray,
    ranking: List[int],
    total_demand: float
) -> int:
    r"""Compute B_{min, C}(theta) \in {0, ..., N} for CFLP given a candidate ranking.
    
    In a complete bipartite transportation graph, a candidate subset of facilities
    is feasible if and only if the cumulative capacity of included facilities
    is at least the total demand.
    
    If total_demand <= 1e-9 (zero demand), zero facilities are feasible (B=0).
    
    Args:
        capacities: Array of shape (N,) with capacity for each facility.
        demands: Array of shape (M,) with customer demands.
        ranking: Ordered list of facility indices [i_1, i_2, ..., i_N].
        total_demand: Sum of demands.
        
    Returns:
        Minimal integer B \in {0, ..., N} such that sum_{k=0}^{B-1} capacities[ranking[k]] >= total_demand.
    """
    if total_demand <= 1e-9:
        return 0
    cum_cap = 0.0
    for B, fac_idx in enumerate(ranking, start=1):
        cum_cap += capacities[fac_idx]
        if cum_cap >= total_demand - 1e-9:
            return B
    return len(ranking)

def compute_entity_entry_budgets(
    candidate_family: Dict[int, Set[int]],
    B_min: int,
    N: int
) -> Dict[int, int]:
    """Compute e_{C, theta}(i) = min { B in {B_min, ..., N} : i in C_theta(B) }.
    
    Args:
        candidate_family: Mapping from candidate size B to set of entity indices.
        B_min: Minimum feasible budget B_{min, C}(theta) in {0, ..., N}.
        N: Full universe size.
        
    Returns:
        Mapping from entity index i to its entry budget.
    """
    entry_budgets = {}
    for B in range(B_min, N + 1):
        c_set = candidate_family.get(B, set())
        for entity in c_set:
            if entity not in entry_budgets:
                entry_budgets[entity] = B
    # Any entity not yet entered entered at N
    for entity in range(N):
        if entity not in entry_budgets:
            entry_budgets[entity] = N
    return entry_budgets

def compute_support_entry_depth(
    support: Set[int],
    entry_budgets: Dict[int, int],
    B_min: int,
    mandatory_set: Optional[Set[int]] = None
) -> int:
    """Compute tau_{C, theta}(S) = max(B_min, max_{i in S} e_{C, theta}(i)).
    
    Args:
        support: Subset of entities forming the active support.
        entry_budgets: Entity entry budgets e_{C, theta}(i).
        B_min: Minimum feasible candidate size in {0, ..., N}.
        mandatory_set: Optional set of mandatory baseline entities.
        
    Returns:
        Candidate entry depth tau.
    """
    if mandatory_set is not None and support.issubset(mandatory_set):
        return B_min
    if len(support) == 0:
        return B_min
    max_e = max(entry_budgets[i] for i in support)
    return max(B_min, max_e)

def compute_reference_match_budget(
    candidate_objectives: Dict[int, float],
    reference_objective: float,
    epsilon_ref: float,
    B_min: int,
    N: int
) -> Dict[str, Any]:
    """Compute empirical reference-fidelity budget B^{dagger, ref}_{C, epsilon}(theta).
    
    One-sided criterion:
    B^{dagger, ref}_{C, epsilon}(theta) = min { B >= B_min : UB_C(B) - UB_ref <= epsilon_ref }.
    
    A candidate solution is fidelity-acceptable if it is no worse than UB_ref by more than epsilon_ref.
    If UB_C(B) < UB_ref - tol, reduced_better_than_reference is flagged.
    
    Args:
        candidate_objectives: Dict mapping B -> candidate restricted optimal objective.
        reference_objective: Full-universe reference objective J^{ref} (UB_ref).
        epsilon_ref: One-sided reference fidelity tolerance.
        B_min: Minimum feasible budget B_{min, C}(theta).
        N: Full universe size.
        
    Returns:
        Dict with 'budget': int, 'reduced_better_than_reference': bool, 'best_candidate_obj': float.
    """
    best_cand_obj = float('inf')
    reduced_better = False
    ref_match_b = N
    
    for B in range(B_min, N + 1):
        if B in candidate_objectives:
            obj_b = candidate_objectives[B]
            if obj_b < best_cand_obj:
                best_cand_obj = obj_b
            if obj_b < reference_objective - 1e-7:
                reduced_better = True
            # One-sided reference-fidelity criterion
            if obj_b - reference_objective <= epsilon_ref + 1e-9:
                if ref_match_b == N:
                    ref_match_b = B
                    
    return {
        "budget": ref_match_b,
        "reduced_better_than_reference": reduced_better,
        "best_candidate_obj": best_cand_obj,
        "reference_objective": reference_objective,
        "epsilon_ref": epsilon_ref
    }

def compute_certified_epsilon_budget(
    candidate_upper_bounds: Dict[int, float],
    full_lower_bound: float,
    epsilon_cert: float,
    B_min: int,
    N: int
) -> float:
    """Compute B^{dagger, cert}_{C, epsilon}(theta) = min { B >= B_min : UB_C(B) - LB_full <= epsilon_cert }.
    
    Since LB_full <= J^* <= UB_C(B), this certifies 0 <= UB_C(B) - J^* <= epsilon_cert.
    If no B satisfies the criterion, returns float('inf') (never silently replaces with N).
    
    Args:
        candidate_upper_bounds: Dict mapping B -> upper bound (feasible candidate objective).
        full_lower_bound: Valid continuous relaxation lower bound of the full-universe program.
        epsilon_cert: Absolute optimality certification tolerance.
        B_min: Minimum feasible candidate size.
        N: Full universe size.
        
    Returns:
        Certified epsilon-quality budget, or float('inf') if uncertified.
    """
    for B in range(B_min, N + 1):
        if B in candidate_upper_bounds:
            if candidate_upper_bounds[B] - full_lower_bound <= epsilon_cert + 1e-9:
                return float(B)
    return float('inf')

def compute_certified_epsilon_budget_record(
    candidate_upper_bounds: Dict[int, float],
    full_upper_bound: float,
    full_lower_bound: float,
    epsilon_cert: float,
    B_min: int,
    N: int
) -> Dict[str, Any]:
    """Compute certified epsilon budget with full metadata logging.
    
    Tracks whether B_dagger_cert exists, whether the full universe itself
    is certified, and the exact absolute/relative bounds.
    """
    abs_gap = full_upper_bound - full_lower_bound
    denom = max(1.0, abs(full_upper_bound))
    rel_gap = abs_gap / denom
    
    full_certified = (abs_gap <= epsilon_cert + 1e-9)
    b_cert = compute_certified_epsilon_budget(
        candidate_upper_bounds=candidate_upper_bounds,
        full_lower_bound=full_lower_bound,
        epsilon_cert=epsilon_cert,
        B_min=B_min,
        N=N
    )
    
    cert_exists = not math.isinf(b_cert)
    
    return {
        "cert_budget_exists": cert_exists,
        "cert_budget_value": int(b_cert) if cert_exists else None,
        "b_dagger_cert_eps": b_cert,
        "full_universe_certified": full_certified,
        "full_universe_abs_gap": abs_gap,
        "full_universe_rel_gap": rel_gap
    }

def compute_oracle_exactness_budget_enumeration(
    candidate_objectives: Dict[int, float],
    oracle_optimal_objective: float,
    B_min: int,
    N: int
) -> int:
    """Compute true mathematical B^dagger_C(theta) when exact oracle objective is known.
    
    Only used when exhaustive enumeration or an analytical solution guarantees
    the exact oracle global optimum.
    """
    for B in range(B_min, N + 1):
        if B in candidate_objectives:
            if abs(candidate_objectives[B] - oracle_optimal_objective) < 1e-12:
                return B
    return N
