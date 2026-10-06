"""
Model-Size Structural Statistics and Formulas Module.
Computes and verifies exact mathematical dimensions:
  - Continuous variables
  - Binary variables
  - Linear constraints
  - Quadratic objective nonzeros
for both Full MIQP P(N) and Physically Sliced Reduced MIQP P(B).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelStructuralStats:
    """Exact dimensions of a Multi-Period Portfolio MIQP."""
    n_assets: int
    horizon: int
    is_reduced: bool
    candidate_size: int
    n_vars: int
    n_continuous: int
    n_binary: int
    n_constraints: int
    n_quadratic_terms: int

    @property
    def reduction_ratio(self) -> float:
        if self.n_assets == 0:
            return 0.0
        return 1.0 - (self.candidate_size / float(self.n_assets))


def compute_theoretical_model_stats(
    n_assets: int,
    horizon: int,
    candidate_size: int | None = None,
) -> ModelStructuralStats:
    """
    Compute theoretical dimensions based on the canonical MultiPeriodMIQP formulation:
      Variables per horizon step h:
        - x_minus: n (continuous)
        - x:       n (continuous)
        - u:       n (continuous)
        - q:       n (continuous)
        - b_minus: 1 (continuous)
        - b:       1 (continuous)
        - kappa:   1 (continuous)
        - y:       n (binary)
        - z:       n (binary)
        Total per step = 6n + 3. Grand total = H * (6n + 3).
        Continuous = H * (4n + 3).
        Binary = 2 * n * H.

      Constraints:
        - Init conditions: n holdings + 1 cash = n + 1
        - Per-step constraints:
            * trade_def:        n
            * budget:           1
            * cost_accounting:  1
            * abs_trade_pos:    n
            * abs_trade_neg:    n
            * holding_upper:    n
            * holding_lower:    n
            * cardinality:      1
            * trade_upper:      n
            * trade_lower:      n
            * turnover:         1
            Subtotal per step = 7n + 4.
        - Inter-step transitions (H-1 steps):
            * risky holdings: (H-1) * n
            * cash transition: (H-1) * 1
        Grand total constraints = (n + 1) + H * (7n + 4) + (H - 1) * (n + 1) = H * (8n + 5).

      Quadratic objective terms:
        - Dense covariance: H * n^2 (or H * n * (n + 1) / 2 unique symmetric pairs).
    """
    H = horizon
    effective_n = candidate_size if candidate_size is not None else n_assets
    is_reduced = bool(candidate_size is not None and candidate_size < n_assets)

    n_cont = H * (4 * effective_n + 3)
    n_bin = 2 * effective_n * H
    n_tot_vars = n_cont + n_bin
    n_constrs = H * (8 * effective_n + 5)
    n_quad = H * effective_n * (effective_n + 1) // 2

    return ModelStructuralStats(
        n_assets=n_assets,
        horizon=horizon,
        is_reduced=is_reduced,
        candidate_size=effective_n,
        n_vars=n_tot_vars,
        n_continuous=n_cont,
        n_binary=n_bin,
        n_constraints=n_constrs,
        n_quadratic_terms=n_quad,
    )
