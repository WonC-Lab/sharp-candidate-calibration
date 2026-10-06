"""
Support restriction and candidate universe slicing utilities for reduced MIQP.
Maps indices between original universe and candidate subset, slices market state and
parameters, and expands reduced decision vectors back to full-universe dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

from .variables import (
    MultiPeriodMarketState,
    MultiPeriodPortfolioParameters,
    MultiPeriodOptimizationResult,
)


@dataclass(frozen=True)
class CandidateUniverse:
    """
    Representation of an admissible candidate risky asset universe C_t subseteq {0, ..., N-1}.

    Attributes:
        candidate_indices: Strictly ascending, unique list of original asset indices.
        full_universe_size: Total universe size N.
    """
    candidate_indices: Tuple[int, ...]
    full_universe_size: int

    def __post_init__(self) -> None:
        if self.full_universe_size < 1:
            raise ValueError(f"full_universe_size must be >= 1, got {self.full_universe_size}")
        if not self.candidate_indices:
            raise ValueError("Candidate universe cannot be empty.")
        # Check uniqueness, ordering, and bounds
        for i, idx in enumerate(self.candidate_indices):
            if idx < 0 or idx >= self.full_universe_size:
                raise ValueError(
                    f"Candidate index {idx} out of bounds [0, {self.full_universe_size - 1}]"
                )
            if i > 0 and idx <= self.candidate_indices[i - 1]:
                raise ValueError(
                    f"Candidate indices must be strictly increasing: {self.candidate_indices}"
                )

    @property
    def candidate_size(self) -> int:
        """Effective candidate size B^{eff} = |C_t|."""
        return len(self.candidate_indices)

    @property
    def reduction_ratio(self) -> float:
        """Candidate reduction ratio: 1 - B^{eff} / N."""
        return 1.0 - (self.candidate_size / self.full_universe_size)

    @property
    def index_map_full_to_red(self) -> Dict[int, int]:
        """Mapping from original universe index -> reduced universe index."""
        return {orig: red for red, orig in enumerate(self.candidate_indices)}

    @property
    def index_map_red_to_full(self) -> List[int]:
        """Mapping from reduced universe index -> original universe index."""
        return list(self.candidate_indices)

    def contains(self, orig_idx: int) -> bool:
        """Check if an original asset index is present in the candidate universe."""
        return orig_idx in self.index_map_full_to_red

    def contains_support(self, support_indices: Set[int]) -> bool:
        """Check if an entire target support S_t^* is contained in C_t."""
        cand_set = set(self.candidate_indices)
        return support_indices.issubset(cand_set)

    @classmethod
    def from_indices(cls, indices: Sequence[int], full_universe_size: int) -> CandidateUniverse:
        """Construct CandidateUniverse from arbitrary iterable, sorting and deduplicating."""
        sorted_unique = tuple(sorted(list(set(indices))))
        return cls(candidate_indices=sorted_unique, full_universe_size=full_universe_size)


def slice_market_state(
    state: MultiPeriodMarketState,
    candidate: CandidateUniverse,
) -> MultiPeriodMarketState:
    """
    Physically slice MultiPeriodMarketState to the candidate universe C_t.
    A principal submatrix of a symmetric PSD covariance matrix is guaranteed PSD.
    """
    idx = list(candidate.candidate_indices)
    H = state.horizon
    n_red = candidate.candidate_size

    # 1. Initial risky holdings (B_eff,)
    init_holdings_red = state.initial_holdings[idx].copy()

    # 2. Expected returns (H, B_eff)
    mu_red = state.expected_returns[:, idx].copy()

    # 3. Covariance matrices: principal submatrices (H, B_eff, B_eff)
    sigma_red = state.covariance_matrices[:, idx, :][:, :, idx].copy()

    # 4. Gross returns (H-1, B_eff) if present
    g_red = None
    if state.gross_returns is not None:
        g_red = state.gross_returns[:, idx].copy()

    # 5. Asset names
    names_red = None
    if state.asset_names is not None:
        names_red = [state.asset_names[i] for i in idx]

    sliced_state = MultiPeriodMarketState(
        initial_holdings=init_holdings_red,
        initial_cash=float(state.initial_cash),
        expected_returns=mu_red,
        covariance_matrices=sigma_red,
        gross_returns=g_red,
        asset_names=names_red,
    )
    sliced_state.validate()
    return sliced_state


def slice_parameters(
    params: MultiPeriodPortfolioParameters,
    candidate: CandidateUniverse,
) -> MultiPeriodPortfolioParameters:
    """
    Physically slice MultiPeriodPortfolioParameters to candidate universe C_t.
    Preserves all risk parameters, solver settings, and sector constraints on retained assets.
    """
    idx = list(candidate.candidate_indices)
    H = params.horizon
    n_red = candidate.candidate_size
    f2r = candidate.index_map_full_to_red

    # 1. Per-asset constraint matrices
    u_red = params.weight_upper_limits[:, idx].copy()
    l_red = params.liquidity_limits[:, idx].copy()
    a_red = params.prop_cost_coeffs[:, idx].copy()

    f_red = params.fixed_cost_coeffs[:, idx].copy() if params.fixed_cost_coeffs is not None else None
    ell_red = params.min_holding_weights[:, idx].copy() if params.min_holding_weights is not None else None
    delta_red = params.min_trade_sizes[:, idx].copy() if params.min_trade_sizes is not None else None

    # 2. Cardinality limit: capped by candidate universe size n_red
    k_red = None
    if params.cardinality_limits is not None:
        k_red = np.minimum(params.cardinality_limits, n_red)

    # 3. Sector indices: map retained original indices to reduced indices
    sec_indices_red: Dict[str, List[int]] = {}
    for sec_name, orig_indices in params.sector_indices.items():
        retained = [f2r[orig] for orig in orig_indices if orig in f2r]
        if retained:
            sec_indices_red[sec_name] = retained

    sliced_params = MultiPeriodPortfolioParameters(
        risk_aversion=float(params.risk_aversion),
        horizon=H,
        risk_free_rates=params.risk_free_rates.copy(),
        max_turnovers=params.max_turnovers.copy(),
        min_cash=params.min_cash.copy(),
        weight_upper_limits=u_red,
        liquidity_limits=l_red,
        prop_cost_coeffs=a_red,
        discount_factor=float(params.discount_factor),
        sector_indices=sec_indices_red,
        sector_bounds=dict(params.sector_bounds),
        cardinality_limits=k_red,
        min_holding_weights=ell_red,
        min_trade_sizes=delta_red,
        fixed_cost_coeffs=f_red,
        allow_degenerate=params.allow_degenerate,
    )
    sliced_params.validate(n_red)
    return sliced_params


def expand_reduced_solution_to_full(
    reduced_result: MultiPeriodOptimizationResult,
    candidate: CandidateUniverse,
    full_n: int,
) -> MultiPeriodOptimizationResult:
    """
    Expand a solved reduced-universe result of dimension B^{eff} back to original dimension N.
    Non-candidate assets i not in C_t are set to zero holding, trade, and indicators.
    """
    if not reduced_result.is_feasible:
        return reduced_result

    H = len(reduced_result.post_trade_holdings)
    idx = list(candidate.candidate_indices)

    def _expand_2d(arr_red: Optional[np.ndarray], dtype=float) -> Optional[np.ndarray]:
        if arr_red is None:
            return None
        full_arr = np.zeros((H, full_n), dtype=dtype)
        full_arr[:, idx] = arr_red
        return full_arr

    x_minus_full = _expand_2d(reduced_result.pre_trade_holdings, float)
    x_full = _expand_2d(reduced_result.post_trade_holdings, float)
    u_full = _expand_2d(reduced_result.trades, float)
    q_full = _expand_2d(reduced_result.absolute_trades, float)
    y_full = _expand_2d(reduced_result.holding_indicators, int)
    z_full = _expand_2d(reduced_result.trade_indicators, int)

    return MultiPeriodOptimizationResult(
        status=reduced_result.status,
        is_optimal=reduced_result.is_optimal,
        is_feasible=reduced_result.is_feasible,
        solve_time=reduced_result.solve_time,
        objective_value=reduced_result.objective_value,
        pre_trade_holdings=x_minus_full,
        post_trade_holdings=x_full,
        trades=u_full,
        absolute_trades=q_full,
        pre_trade_cash=reduced_result.pre_trade_cash.copy() if reduced_result.pre_trade_cash is not None else None,
        post_trade_cash=reduced_result.post_trade_cash.copy() if reduced_result.post_trade_cash is not None else None,
        transaction_costs=reduced_result.transaction_costs.copy() if reduced_result.transaction_costs is not None else None,
        holding_indicators=y_full,
        trade_indicators=z_full,
        best_bound=reduced_result.best_bound,
        mip_gap=reduced_result.mip_gap,
        node_count=reduced_result.node_count,
        work=reduced_result.work,
        sol_count=reduced_result.sol_count,
        metadata={
            **reduced_result.metadata,
            "expanded_from_candidate_size": candidate.candidate_size,
            "full_universe_size": full_n,
        },
    )
