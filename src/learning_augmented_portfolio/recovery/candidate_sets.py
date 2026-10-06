"""
Nested candidate sets generator for recovery sequences.
Guarantees strict set inclusion: C_t(B_1) subset C_t(B_2) subset ... subset C_t(B_m) = {0, ..., N-1}
using Semantics B: C_t(B) = M_t cup Top_{B - |M_t|}(p_t | i not in M_t).
"""

from __future__ import annotations

from typing import List, Sequence, Optional
import numpy as np

from ..optimization.support_restriction import CandidateUniverse
from ..learning.candidate_set import construct_candidate_universe


def generate_nested_candidate_sequence(
    scores: np.ndarray,
    current_weights: np.ndarray,
    budgets: Sequence[int],
) -> List[CandidateUniverse]:
    """
    Construct a sequence of nested CandidateUniverse objects for recovery.

    Guarantees:
      1. |C_t(B_k)| = B_k exactly for each stage.
      2. C_t(B_1) subset C_t(B_2) subset ... subset C_t(B_m).
      3. C_t(B_m) = {0, 1, ..., N - 1} when B_m = N.
      4. Rejects any B_k < |M_t|.

    Args:
        scores: Ranking scores of shape (N,).
        current_weights: Pre-trade portfolio weights w_bar_t of shape (N,).
        budgets: Monotonically increasing candidate budgets B_1 < B_2 < ... < B_m.

    Returns:
        List of CandidateUniverse objects satisfying strict nested inclusion.
    """
    N = len(scores)
    if len(budgets) == 0:
        raise ValueError("Budget sequence cannot be empty.")

    # Validate monotonicity of budgets
    for k in range(len(budgets) - 1):
        if budgets[k] >= budgets[k + 1]:
            raise ValueError(
                f"Budgets must be strictly increasing. Got B_{k}={budgets[k]} >= B_{k+1}={budgets[k+1]}"
            )

    held_indices = np.where(current_weights > 1e-6)[0].tolist()
    n_held = len(held_indices)

    if budgets[0] < n_held:
        raise ValueError(
            f"Smallest candidate budget B_1={budgets[0]} is strictly less than mandatory set size |M_t|={n_held}. "
            f"Initial candidate budget must satisfy B >= |M_t|."
        )

    # If final budget is not N, ensure fallback is possible or check
    candidate_list: List[CandidateUniverse] = []
    for B in budgets:
        cand = construct_candidate_universe(
            scores=scores,
            budget_B=B,
            current_weights=current_weights,
            use_safe_augmentation=True,
            semantics="fixed_budget",
        )
        candidate_list.append(cand)

    # Rigorous nestedness verification
    for k in range(len(candidate_list) - 1):
        c_k = set(candidate_list[k].candidate_indices)
        c_next = set(candidate_list[k + 1].candidate_indices)
        if not c_k.issubset(c_next):
            raise AssertionError(
                f"Nestedness violation: Stage {k} (B={budgets[k]}) is not a subset of Stage {k+1} (B={budgets[k+1]}). "
                f"Missing elements: {c_k - c_next}"
            )
        if len(c_k) >= len(c_next):
            raise AssertionError(
                f"Strict nestedness violation: Stage {k} size {len(c_k)} >= Stage {k+1} size {len(c_next)}"
            )

    # Check if last stage is full universe
    if budgets[-1] == N:
        assert candidate_list[-1].candidate_size == N, "Final stage budget is N but candidate size != N"

    return candidate_list
