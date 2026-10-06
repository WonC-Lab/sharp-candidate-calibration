"""Strictly increasing geometric recovery schedule, monotone upper bound envelope, and stage tracking."""
import math
from typing import Dict, List, Optional, Any, Tuple, Union

class StrictlyIncreasingRecoveryLadder:
    """Strictly increasing geometric candidate ladder: B_{ell+1} = min(N, max(B_ell + 1, ceil(gamma B_ell)))."""
    
    def __init__(self, gamma: float = 1.5, N: int = 10):
        if gamma <= 1.0:
            raise ValueError(f"Expansion factor gamma must be strictly greater than 1.0, got {gamma}")
        self.gamma = gamma
        self.N = N
        
    def generate_ladder(self, B_1: int, B_min: int = 1) -> List[int]:
        """Generate the full candidate budget ladder starting from B_1.
        
        Enforces B_1 >= max(1, B_min), where B_min \in {0, ..., N}.
        If B_1 == 0 and zero-budget exactness was already handled, ladder is empty or starts at 1.
        """
        init_b = max(1, max(B_min, B_1))
        init_b = min(self.N, init_b)
        
        ladder = [init_b]
        curr = init_b
        while curr < self.N:
            nxt = min(self.N, max(curr + 1, math.ceil(self.gamma * curr)))
            ladder.append(nxt)
            curr = nxt
        return ladder
        
    def next_budget(self, B_curr: int) -> int:
        """Compute the next budget in the ladder."""
        if B_curr >= self.N:
            return self.N
        return min(self.N, max(B_curr + 1, math.ceil(self.gamma * B_curr)))

def compute_monotone_ub_envelope(raw_upper_bounds: List[float]) -> List[float]:
    """Compute the monotone feasible upper-bound envelope along the ladder.
    
    UB_best(B_l) := min { UB_candidate(B_1), ..., UB_candidate(B_l) }.
    
    Because each previous candidate incumbent remains feasible for all later nested candidate sets,
    this envelope is provably a valid feasible upper bound at each stage:
    UB_best(B_{l+1}) <= UB_best(B_l).
    
    Args:
        raw_upper_bounds: Sequence of raw incumbent upper bounds [UB(B_1), ..., UB(B_L)].
        
    Returns:
        Monotonically non-increasing envelope sequence.
    """
    envelope = []
    current_best = float('inf')
    for ub in raw_upper_bounds:
        if ub < current_best:
            current_best = ub
        envelope.append(current_best)
    return envelope

def evaluate_recovery_stages(
    ladder: List[int],
    B_dagger_ref: Union[int, float],
    B_dagger_cert: Union[int, float, None],
    L_term: int,
    L_cert: float,
    raw_upper_bounds: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Compute decoupled structural and operational stage metrics.
    
    Args:
        ladder: Full sequence of budgets [B_1, B_2, ..., B_L].
        B_dagger_ref: Empirical reference-match exactness budget.
        B_dagger_cert: Certified epsilon-quality budget (finite int, float('inf'), or None).
        L_term: Stage at which algorithm terminated (1-indexed, in 1..len(ladder)).
        L_cert: Stage at which mathematical certificate was proven (1-indexed), or float('inf').
        raw_upper_bounds: Optional sequence of raw solver incumbents at each ladder stage.
        
    Returns:
        Dict with decoupled L_exact_ref, L_exact_cert, cumulative dimensions, and envelope diagnostics.
    """
    # L_exact_ref
    if B_dagger_ref == 0:
        L_exact_ref = 0
        cum_exact_ref = 0
    elif math.isinf(B_dagger_ref):
        L_exact_ref = float('inf')
        cum_exact_ref = float('inf')
    else:
        L_exact_ref = len(ladder)
        for idx, B in enumerate(ladder, start=1):
            if B >= B_dagger_ref:
                L_exact_ref = idx
                break
        cum_exact_ref = sum(ladder[:L_exact_ref])
        
    # L_exact_cert
    if B_dagger_cert is None or math.isinf(B_dagger_cert):
        L_exact_cert = float('inf')
        cum_exact_cert = float('inf')
    elif B_dagger_cert == 0:
        L_exact_cert = 0
        cum_exact_cert = 0
    else:
        L_exact_cert = float('inf')
        for idx, B in enumerate(ladder, start=1):
            if B >= B_dagger_cert:
                L_exact_cert = idx
                break
        if not math.isinf(L_exact_cert):
            cum_exact_cert = sum(ladder[:L_exact_cert])
        else:
            cum_exact_cert = float('inf')
        
    # Cumulative dimension to termination
    term_idx = min(len(ladder), max(1, L_term))
    cum_term = sum(ladder[:term_idx])
    
    # Monotone UB envelope
    ub_envelope = None
    if raw_upper_bounds is not None:
        ub_envelope = compute_monotone_ub_envelope(raw_upper_bounds)
    
    return {
        "L_exact_ref": L_exact_ref,
        "L_exact_cert": L_exact_cert,
        "L_term": L_term,
        "L_cert": L_cert,
        "cum_dim_to_exact_ref": cum_exact_ref,
        "cum_dim_to_exact_cert": cum_exact_cert,
        "cum_dim_to_term": cum_term,
        "ladder": ladder,
        "UB_raw": raw_upper_bounds,
        "UB_best_envelope": ub_envelope
    }
