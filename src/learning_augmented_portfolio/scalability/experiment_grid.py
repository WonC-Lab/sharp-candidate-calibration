"""
Scalability Experiment Grid and Cardinality/Candidate Regimes Module.
Defines:
  - N in {10, 25, 50, 100, 250, 500}
  - H in {1, 3, 5}
  - Cardinality Regime A (Fixed K) vs Regime B (Proportional K)
  - Candidate Budget Ratios rho_B in {0.10, 0.20, 0.30, 0.50, 1.00}
  - Deterministic nested candidate sequences
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence, Tuple


class CardinalityRegime(str, Enum):
    """Cardinality scaling regimes."""
    FIXED = "FIXED"               # K_port = min(10, N)
    PROPORTIONAL = "PROPORTIONAL" # K_port = max(2, floor(rho_K * N))


@dataclass(frozen=True)
class ScalabilityGridConfig:
    """Configuration for a single scalability evaluation point."""
    n_assets: int
    horizon: int
    cardinality_regime: CardinalityRegime
    cardinality_k: int
    candidate_ratios: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.50, 1.00)


def compute_cardinality(
    n_assets: int,
    regime: CardinalityRegime,
    fixed_k: int = 4,
    proportional_rho_k: float = 0.10,
) -> int:
    """Compute K_port under the specified regime."""
    if regime == CardinalityRegime.FIXED:
        return min(n_assets, fixed_k)
    elif regime == CardinalityRegime.PROPORTIONAL:
        return min(n_assets, max(2, int(proportional_rho_k * n_assets)))
    raise ValueError(f"Unknown regime: {regime}")


def compute_candidate_budget(
    n_assets: int,
    ratio_rho_b: float,
    cardinality_k: int,
    n_held: int = 0,
) -> int:
    """
    Compute discrete candidate budget B = |C_t| satisfying:
      B >= K_port
      B >= |M_t| (held assets)
      1 <= B <= N
    """
    raw_b = int(round(ratio_rho_b * n_assets))
    b = max(raw_b, cardinality_k, n_held, 1)
    return min(b, n_assets)


def generate_nested_budget_sequence(
    n_assets: int,
    ratios: Sequence[float],
    cardinality_k: int,
    n_held: int = 0,
) -> List[int]:
    """
    Generate deterministic strictly non-decreasing nested budget sequence B_1 <= B_2 <= ... <= N.
    """
    budgets = []
    for r in ratios:
        b = compute_candidate_budget(n_assets, r, cardinality_k, n_held)
        budgets.append(b)
    # Deduplicate while preserving order
    dedup = []
    for b in budgets:
        if not dedup or b > dedup[-1]:
            dedup.append(b)
    if dedup[-1] != n_assets:
        dedup.append(n_assets)
    return dedup
