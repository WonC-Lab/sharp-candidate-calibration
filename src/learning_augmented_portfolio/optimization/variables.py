"""
Core data structures, parameters, and result types for mathematical optimization.
Supports Phase 1 (Convex QP), Phase 2 (MIQP), and Phase 3 (Deterministic Multi-Period MIQP).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class PortfolioParameters:
    """
    Immutable specification of portfolio constraints and optimization parameters (Single Period).
    
    Phase 1 (Convex QP) Attributes:
        risk_aversion: γ > 0, scaling factor for quadratic risk penalty.
        risk_free_rate: r_f, return on post-trade cash balance.
        max_turnover: τ_max >= 0, maximum allowable two-way turnover sum_i q_i.
        min_cash: c_min >= 0, minimum post-trade cash reserve fraction.
        weight_upper_limits: U in R^n, per-asset upper bounds on weights (0 <= w_i <= U_i).
        liquidity_limits: L in R^n, per-asset trading limits (0 <= q_i <= L_i).
        prop_cost_coeffs: a in R^n, proportional transaction cost rates (kappa = sum_i a_i q_i + ...).
        sector_indices: Mapping from sector id/name to list of asset indices.
        sector_bounds: Mapping from sector id/name to tuple (E_min, E_max).

    Phase 2 (MIQP) Discrete Attributes:
        cardinality_limit: K in {1, ..., n}, maximum number of assets held (sum_i y_i <= K).
        min_holding_weights: ℓ in R^n, minimum holding threshold if held (ℓ_i y_i <= w_i <= U_i y_i).
        min_trade_sizes: δ in R^n, minimum trade threshold if traded (δ_i z_i <= q_i <= M_i z_i).
        fixed_cost_coeffs: f in R^n, fixed transaction charge per executed trade (kappa = sum_i (a_i q_i + f_i z_i)).
        allow_degenerate: If True, allows ℓ_i=0 or δ_i=0 for Phase 1 regression verification.
    """
    risk_aversion: float
    risk_free_rate: float
    max_turnover: float
    min_cash: float
    weight_upper_limits: np.ndarray
    liquidity_limits: np.ndarray
    prop_cost_coeffs: np.ndarray
    sector_indices: Dict[str, List[int]] = field(default_factory=dict)
    sector_bounds: Dict[str, tuple[float, float]] = field(default_factory=dict)

    # Phase 2 Discrete Extensions
    cardinality_limit: Optional[int] = None
    min_holding_weights: Optional[np.ndarray] = None
    min_trade_sizes: Optional[np.ndarray] = None
    fixed_cost_coeffs: Optional[np.ndarray] = None
    allow_degenerate: bool = False

    def validate(self, n_assets: int) -> None:
        """Validate dimensions and domain validity of parameters."""
        if self.risk_aversion <= 0:
            raise ValueError(f"risk_aversion must be strictly positive, got {self.risk_aversion}")
        if self.max_turnover < 0:
            raise ValueError(f"max_turnover must be non-negative, got {self.max_turnover}")
        if self.min_cash < 0 or self.min_cash > 1:
            raise ValueError(f"min_cash must be in [0, 1], got {self.min_cash}")

        for name, arr in [
            ("weight_upper_limits", self.weight_upper_limits),
            ("liquidity_limits", self.liquidity_limits),
            ("prop_cost_coeffs", self.prop_cost_coeffs),
        ]:
            if arr.shape != (n_assets,):
                raise ValueError(f"{name} must have shape ({n_assets},), got {arr.shape}")
            if np.any(arr < 0):
                raise ValueError(f"{name} elements must be non-negative")

        for sector, (e_min, e_max) in self.sector_bounds.items():
            if e_min > e_max:
                raise ValueError(f"Sector {sector}: E_min ({e_min}) cannot exceed E_max ({e_max})")
            if sector not in self.sector_indices:
                raise ValueError(f"Sector {sector} defined in bounds but missing in sector_indices")
            for idx in self.sector_indices[sector]:
                if not (0 <= idx < n_assets):
                    raise IndexError(f"Asset index {idx} in sector {sector} is out of bounds [0, {n_assets})")

        # Phase 2 Validations
        if self.cardinality_limit is not None:
            if not (1 <= self.cardinality_limit <= n_assets):
                raise ValueError(
                    f"cardinality_limit K must satisfy 1 <= K <= {n_assets}, got {self.cardinality_limit}"
                )

        if self.min_holding_weights is not None:
            if self.min_holding_weights.shape != (n_assets,):
                raise ValueError(f"min_holding_weights shape must be ({n_assets},), got {self.min_holding_weights.shape}")
            if np.any(self.min_holding_weights < 0):
                raise ValueError("min_holding_weights elements must be non-negative")
            if not self.allow_degenerate and np.any(self.min_holding_weights <= 0):
                raise ValueError("In non-degenerate Phase 2 model, min_holding_weights ℓ_i must be strictly positive")
            if np.any(self.min_holding_weights > self.weight_upper_limits):
                raise ValueError("min_holding_weights ℓ_i cannot exceed weight_upper_limits U_i")

        if self.min_trade_sizes is not None:
            if self.min_trade_sizes.shape != (n_assets,):
                raise ValueError(f"min_trade_sizes shape must be ({n_assets},), got {self.min_trade_sizes.shape}")
            if np.any(self.min_trade_sizes < 0):
                raise ValueError("min_trade_sizes elements must be non-negative")
            if not self.allow_degenerate and np.any(self.min_trade_sizes <= 0):
                raise ValueError("In non-degenerate Phase 2 model, min_trade_sizes δ_i must be strictly positive")

        if self.fixed_cost_coeffs is not None:
            if self.fixed_cost_coeffs.shape != (n_assets,):
                raise ValueError(f"fixed_cost_coeffs shape must be ({n_assets},), got {self.fixed_cost_coeffs.shape}")
            if np.any(self.fixed_cost_coeffs < 0):
                raise ValueError("fixed_cost_coeffs elements must be non-negative")


@dataclass(frozen=True)
class MarketState:
    """
    State of the market and current portfolio before rebalancing (Single Period).
    
    Attributes:
        expected_returns: μ in R^n, forecast expected return vector.
        covariance_matrix: Σ in S_+^n, forecast covariance matrix (must be PSD).
        current_weights: w_bar in R^n, pre-trade risky weights.
        current_cash: c_bar in R, pre-trade cash balance (1 - 1^T w_bar).
        asset_names: Optional string identifiers for the n assets.
    """
    expected_returns: np.ndarray
    covariance_matrix: np.ndarray
    current_weights: np.ndarray
    current_cash: float
    asset_names: Optional[List[str]] = None

    @property
    def n_assets(self) -> int:
        return len(self.expected_returns)

    def validate(self) -> None:
        """Validate vector shapes, non-negativity, cash conservation, and covariance PSD."""
        n = self.n_assets
        if self.expected_returns.shape != (n,):
            raise ValueError(f"expected_returns shape {self.expected_returns.shape} != ({n},)")
        if self.covariance_matrix.shape != (n, n):
            raise ValueError(f"covariance_matrix shape {self.covariance_matrix.shape} != ({n}, {n})")
        if self.current_weights.shape != (n,):
            raise ValueError(f"current_weights shape {self.current_weights.shape} != ({n},)")

        if np.any(self.current_weights < -1e-8):
            raise ValueError(f"current_weights w_bar must be non-negative for long-only portfolios, min={np.min(self.current_weights)}")

        # Verify symmetry of covariance matrix
        if not np.allclose(self.covariance_matrix, self.covariance_matrix.T, atol=1e-8):
            raise ValueError("covariance_matrix must be symmetric")

        # Verify positive semi-definiteness: minimum eigenvalue >= -1e-8
        eigenvalues = np.linalg.eigvalsh(self.covariance_matrix)
        min_eig = np.min(eigenvalues)
        if min_eig < -1e-8:
            raise ValueError(
                f"covariance_matrix must be positive semi-definite (min eigenvalue = {min_eig:.2e} < 0)"
            )

        # Verify pre-trade budget identity: 1^T w_bar + c_bar == 1.0 within numerical precision
        total_initial_wealth = np.sum(self.current_weights) + self.current_cash
        if not np.isclose(total_initial_wealth, 1.0, atol=1e-5):
            raise ValueError(
                f"Pre-trade wealth violation: 1^T w_bar + c_bar = {total_initial_wealth:.6f} != 1.0"
            )


@dataclass(frozen=True)
class MultiPeriodPortfolioParameters:
    """
    Immutable specification of portfolio constraints across planning horizon h = 0, ..., H-1.
    All bounds and constraints are expressed in the decision-time numeraire V_t = 1.0.
    """
    risk_aversion: float
    horizon: int
    risk_free_rates: np.ndarray       # shape (H,)
    max_turnovers: np.ndarray         # shape (H,)
    min_cash: np.ndarray              # shape (H,)
    weight_upper_limits: np.ndarray   # shape (H, n)
    liquidity_limits: np.ndarray      # shape (H, n)
    prop_cost_coeffs: np.ndarray      # shape (H, n)
    discount_factor: float = 1.0      # beta in (0, 1]
    sector_indices: Dict[str, List[int]] = field(default_factory=dict)
    sector_bounds: Dict[str, tuple[float, float]] = field(default_factory=dict)
    cardinality_limits: Optional[np.ndarray] = None    # shape (H,)
    min_holding_weights: Optional[np.ndarray] = None   # shape (H, n)
    min_trade_sizes: Optional[np.ndarray] = None       # shape (H, n)
    fixed_cost_coeffs: Optional[np.ndarray] = None     # shape (H, n)
    allow_degenerate: bool = False

    def validate(self, n_assets: int) -> None:
        """Validate multi-period parameters across all periods h = 0, ..., H-1."""
        H = self.horizon
        n = n_assets
        if H < 1:
            raise ValueError(f"horizon H must be >= 1, got {H}")
        if self.risk_aversion <= 0:
            raise ValueError(f"risk_aversion must be strictly positive, got {self.risk_aversion}")
        if self.discount_factor <= 0 or self.discount_factor > 1.0:
            raise ValueError(f"discount_factor beta must be in (0, 1], got {self.discount_factor}")

        if self.risk_free_rates.shape != (H,):
            raise ValueError(f"risk_free_rates shape must be ({H},), got {self.risk_free_rates.shape}")
        if self.max_turnovers.shape != (H,):
            raise ValueError(f"max_turnovers shape must be ({H},), got {self.max_turnovers.shape}")
        if np.any(self.max_turnovers < 0):
            raise ValueError("max_turnovers elements must be non-negative")
        if self.min_cash.shape != (H,):
            raise ValueError(f"min_cash shape must be ({H},), got {self.min_cash.shape}")
        if np.any(self.min_cash < 0) or np.any(self.min_cash > 1.0):
            raise ValueError("min_cash elements must be in [0, 1]")

        for name, arr in [
            ("weight_upper_limits", self.weight_upper_limits),
            ("liquidity_limits", self.liquidity_limits),
            ("prop_cost_coeffs", self.prop_cost_coeffs),
        ]:
            if arr.shape != (H, n):
                raise ValueError(f"{name} must have shape ({H}, {n}), got {arr.shape}")
            if np.any(arr < 0):
                raise ValueError(f"{name} elements must be non-negative")

        if self.cardinality_limits is not None:
            if self.cardinality_limits.shape != (H,):
                raise ValueError(f"cardinality_limits shape must be ({H},), got {self.cardinality_limits.shape}")
            if np.any(self.cardinality_limits < 1) or np.any(self.cardinality_limits > n):
                raise ValueError(f"cardinality_limits elements must be in [1, {n}]")

        if self.min_holding_weights is not None:
            if self.min_holding_weights.shape != (H, n):
                raise ValueError(f"min_holding_weights shape must be ({H}, {n}), got {self.min_holding_weights.shape}")
            if np.any(self.min_holding_weights < 0):
                raise ValueError("min_holding_weights elements must be non-negative")
            if not self.allow_degenerate and np.any(self.min_holding_weights <= 0):
                raise ValueError("min_holding_weights ell_{i,h} must be strictly positive in non-degenerate model")
            if np.any(self.min_holding_weights > self.weight_upper_limits):
                raise ValueError("min_holding_weights ell_{i,h} cannot exceed weight_upper_limits U_{i,h}")

        if self.min_trade_sizes is not None:
            if self.min_trade_sizes.shape != (H, n):
                raise ValueError(f"min_trade_sizes shape must be ({H}, {n}), got {self.min_trade_sizes.shape}")
            if np.any(self.min_trade_sizes < 0):
                raise ValueError("min_trade_sizes elements must be non-negative")
            if not self.allow_degenerate and np.any(self.min_trade_sizes <= 0):
                raise ValueError("min_trade_sizes delta_{i,h} must be strictly positive in non-degenerate model")

        if self.fixed_cost_coeffs is not None:
            if self.fixed_cost_coeffs.shape != (H, n):
                raise ValueError(f"fixed_cost_coeffs shape must be ({H}, {n}), got {self.fixed_cost_coeffs.shape}")
            if np.any(self.fixed_cost_coeffs < 0):
                raise ValueError("fixed_cost_coeffs elements must be non-negative")

    @classmethod
    def from_single_period(
        cls,
        params: PortfolioParameters,
        horizon: int = 1,
        discount_factor: float = 1.0,
    ) -> MultiPeriodPortfolioParameters:
        """Create multi-period parameters by broadcasting single-period parameters across H steps."""
        H = horizon
        n = len(params.weight_upper_limits)
        rf_arr = np.full(H, params.risk_free_rate)
        tau_arr = np.full(H, params.max_turnover)
        c_min_arr = np.full(H, params.min_cash)
        u_arr = np.tile(params.weight_upper_limits, (H, 1))
        l_arr = np.tile(params.liquidity_limits, (H, 1))
        a_arr = np.tile(params.prop_cost_coeffs, (H, 1))

        k_arr = np.full(H, params.cardinality_limit) if params.cardinality_limit is not None else None
        ell_arr = np.tile(params.min_holding_weights, (H, 1)) if params.min_holding_weights is not None else None
        delta_arr = np.tile(params.min_trade_sizes, (H, 1)) if params.min_trade_sizes is not None else None
        f_arr = np.tile(params.fixed_cost_coeffs, (H, 1)) if params.fixed_cost_coeffs is not None else None

        return cls(
            risk_aversion=params.risk_aversion,
            horizon=H,
            risk_free_rates=rf_arr,
            max_turnovers=tau_arr,
            min_cash=c_min_arr,
            weight_upper_limits=u_arr,
            liquidity_limits=l_arr,
            prop_cost_coeffs=a_arr,
            discount_factor=discount_factor,
            sector_indices=params.sector_indices,
            sector_bounds=params.sector_bounds,
            cardinality_limits=k_arr,
            min_holding_weights=ell_arr,
            min_trade_sizes=delta_arr,
            fixed_cost_coeffs=f_arr,
            allow_degenerate=params.allow_degenerate,
        )


@dataclass(frozen=True)
class MultiPeriodMarketState:
    """
    Market state across planning horizon h = 0, ..., H-1.
    
    Attributes:
        initial_holdings: x_0^- in R^n, pre-trade risky holdings at h=0.
        initial_cash: b_0^- in R, pre-trade cash at h=0 (1^T x_0^- + b_0^- == 1.0).
        expected_returns: mu of shape (H, n), forecast returns.
        covariance_matrices: Sigma of shape (H, n, n), forecast covariances (each must be PSD).
        gross_returns: g of shape (H-1, n), deterministic return transitions g_{i,h} = 1 + r_hat_{i,h} >= 0.
        asset_names: Optional asset identifiers.
    """
    initial_holdings: np.ndarray
    initial_cash: float
    expected_returns: np.ndarray
    covariance_matrices: np.ndarray
    gross_returns: Optional[np.ndarray] = None
    asset_names: Optional[List[str]] = None

    @property
    def n_assets(self) -> int:
        return len(self.initial_holdings)

    @property
    def horizon(self) -> int:
        return len(self.expected_returns)

    def validate(self) -> None:
        """Validate state dimensions, positive semi-definiteness, and return non-negativity."""
        H = self.horizon
        n = self.n_assets

        if self.initial_holdings.shape != (n,):
            raise ValueError(f"initial_holdings shape {self.initial_holdings.shape} != ({n},)")
        if np.any(self.initial_holdings < -1e-8):
            raise ValueError("initial_holdings must be non-negative for long-only portfolios")

        init_wealth = np.sum(self.initial_holdings) + self.initial_cash
        if not np.isclose(init_wealth, 1.0, atol=1e-5):
            raise ValueError(f"Initial budget violation: 1^T x_0^- + b_0^- = {init_wealth:.6f} != 1.0")

        if self.expected_returns.shape != (H, n):
            raise ValueError(f"expected_returns shape {self.expected_returns.shape} != ({H}, {n})")
        if self.covariance_matrices.shape != (H, n, n):
            raise ValueError(f"covariance_matrices shape {self.covariance_matrices.shape} != ({H}, {n}, {n})")

        for h in range(H):
            cov_h = self.covariance_matrices[h]
            if not np.allclose(cov_h, cov_h.T, atol=1e-8):
                raise ValueError(f"covariance_matrices at h={h} must be symmetric")
            min_eig = np.min(np.linalg.eigvalsh(cov_h))
            if min_eig < -1e-8:
                raise ValueError(f"covariance_matrices at h={h} is not PSD (min eigenvalue = {min_eig:.2e} < 0)")

        if H > 1:
            if self.gross_returns is not None:
                if self.gross_returns.shape != (H - 1, n):
                    raise ValueError(f"gross_returns shape {self.gross_returns.shape} != ({H - 1}, {n})")
                if np.any(self.gross_returns < 0):
                    raise ValueError("gross_returns elements must be non-negative (g_{i,h} >= 0)")

    @classmethod
    def from_single_period(
        cls,
        state: MarketState,
        horizon: int = 1,
        gross_returns: Optional[np.ndarray] = None,
    ) -> MultiPeriodMarketState:
        """Create multi-period market state by broadcasting single-period state across H steps."""
        H = horizon
        n = state.n_assets
        mu_h = np.tile(state.expected_returns, (H, 1))
        sigma_h = np.tile(state.covariance_matrix, (H, 1, 1))
        g_h = gross_returns if gross_returns is not None else (np.ones((H - 1, n)) if H > 1 else None)

        return cls(
            initial_holdings=state.current_weights,
            initial_cash=state.current_cash,
            expected_returns=mu_h,
            covariance_matrices=sigma_h,
            gross_returns=g_h,
            asset_names=state.asset_names,
        )


@dataclass
class SolverConfig:
    """Configuration for Gurobi solver execution."""
    time_limit: float = 60.0
    mip_gap: float = 1e-4
    threads: int = 1
    output_flag: bool = False
    feasibility_tol: float = 1e-7
    optimality_tol: float = 1e-7


@dataclass
class OptimizationResult:
    """Container for the output and diagnostic metrics of a single-period optimization run (QP and MIQP)."""
    status: str
    is_optimal: bool
    is_feasible: bool
    solve_time: float
    objective_value: Optional[float] = None
    post_trade_weights: Optional[np.ndarray] = None
    trades: Optional[np.ndarray] = None
    absolute_trades: Optional[np.ndarray] = None
    post_trade_cash: Optional[float] = None
    transaction_cost: Optional[float] = None
    holding_indicators: Optional[np.ndarray] = None
    trade_indicators: Optional[np.ndarray] = None
    best_bound: Optional[float] = None
    mip_gap: Optional[float] = None
    node_count: Optional[int] = None
    dual_values: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiPeriodOptimizationResult:
    """Container for the output and diagnostic metrics of a multi-period MIQP run."""
    status: str
    is_optimal: bool
    is_feasible: bool
    solve_time: float
    objective_value: Optional[float] = None
    # Decision Variables across Horizon (H, n) or (H,)
    pre_trade_holdings: Optional[np.ndarray] = None   # x_h^-, shape (H, n)
    post_trade_holdings: Optional[np.ndarray] = None  # x_h,   shape (H, n)
    trades: Optional[np.ndarray] = None               # u_h,   shape (H, n)
    absolute_trades: Optional[np.ndarray] = None      # q_h,   shape (H, n)
    pre_trade_cash: Optional[np.ndarray] = None       # b_h^-, shape (H,)
    post_trade_cash: Optional[np.ndarray] = None      # b_h,   shape (H,)
    transaction_costs: Optional[np.ndarray] = None    # kappa_h, shape (H,)
    holding_indicators: Optional[np.ndarray] = None   # y_{i,h}, shape (H, n)
    trade_indicators: Optional[np.ndarray] = None     # z_{i,h}, shape (H, n)
    best_bound: Optional[float] = None
    mip_gap: Optional[float] = None
    node_count: Optional[int] = None
    work: Optional[float] = None
    sol_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
