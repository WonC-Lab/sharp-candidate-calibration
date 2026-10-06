"""
Candidate Universe Construction module.
Translates predicted continuous scores into discrete candidate subsets.
Implements:
  - Raw Top-B candidate sets: A_t(B) = TopB(p_{i,t})
  - Conservative mandatory safe set: M_t = {i : w_bar_{i,t} > 0}
  - Safe-augmented candidate sets: C_t(B) = M_t cup A_t(B)
  - Effective candidate count B^{eff} and true reduction ratio 1 - B^{eff} / N
"""

from __future__ import annotations

from typing import Optional, Set
import numpy as np

from ..optimization.support_restriction import CandidateUniverse


def construct_candidate_universe(
    scores: np.ndarray,
    budget_B: int,
    current_weights: Optional[np.ndarray] = None,
    use_safe_augmentation: bool = True,
    semantics: str = "fixed_budget",
) -> CandidateUniverse:
    """
    Construct a validated CandidateUniverse object from model predicted scores.

    Candidate Semantics:
      - Raw Top-B (use_safe_augmentation=False):
          A_t(B) = TopB(p_{i,t})
      - Semantics B: Fixed Total Candidate Budget (semantics='fixed_budget', default):
          C_t(B) = M_t cup Top_{B - |M_t|}(p_t | i not in M_t)
          Guarantees exact budget |C_t(B)| = B.
          Requires B >= |M_t|. Rejects B < |M_t| with ValueError.
          Guarantees strict nesting: B_1 < B_2 => C_t(B_1) subset C_t(B_2).
      - Semantics A: True Union (semantics='true_union'):
          C_t(B) = M_t cup TopB(p_t)
          Effective size |C_t(B)| >= B. Adding M_t cannot decrease coverage.

    Args:
        scores: Predicted scores or probabilities of shape (N,).
        budget_B: Requested candidate budget B in [1, N].
        current_weights: Pre-trade risky weights w_bar_t of shape (N,).
        use_safe_augmentation: If True, enforce inclusion of M_t = {i : w_bar_{i,t} > 1e-6}.
        semantics: Either 'fixed_budget' (Semantics B) or 'true_union' (Semantics A).

    Returns:
        CandidateUniverse instance with sorted unique indices and effective sizing.
    """
    N = len(scores)
    B = min(max(1, budget_B), N)

    # 1. Rank indices by descending score
    ranked_indices = np.argsort(-scores).tolist()

    if not use_safe_augmentation or current_weights is None:
        # Raw Top-B selection
        selected = set(ranked_indices[:B])
        return CandidateUniverse.from_indices(selected, full_universe_size=N)

    held_indices = set(np.where(current_weights > 1e-6)[0].tolist())
    n_held = len(held_indices)

    if semantics == "fixed_budget":
        # Semantics B: Fixed total candidate budget |C_t(B)| = B
        if B < n_held:
            raise ValueError(
                f"Candidate budget B={budget_B} is strictly less than mandatory set size |M_t|={n_held}. "
                f"Fixed-budget safe candidate construction requires B >= |M_t|."
            )
        selected = set(held_indices)
        needed = B - n_held
        for idx in ranked_indices:
            if needed <= 0:
                break
            if idx not in selected:
                selected.add(idx)
                needed -= 1
        assert len(selected) == B, f"Expected exactly {B} candidates, got {len(selected)}"
        return CandidateUniverse.from_indices(selected, full_universe_size=N)

    elif semantics == "true_union":
        # Semantics A: True union M_t cup TopB(p_t)
        top_b = set(ranked_indices[:B])
        selected = held_indices.union(top_b)
        return CandidateUniverse.from_indices(selected, full_universe_size=N)
    else:
        raise ValueError(f"Unknown candidate semantics: '{semantics}'. Choose 'fixed_budget' or 'true_union'.")

