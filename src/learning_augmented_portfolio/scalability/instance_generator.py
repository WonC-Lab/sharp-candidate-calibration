"""
Controlled Scalable Instance Generator Module.
Generates synthetic portfolio optimization instances across arbitrary (N, H, K)
using a low-rank factor model with mathematically guaranteed positive definite covariance:
    Sigma = B_f * Sigma_f * B_f^T + D_epsilon
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from ..dynamic.state import PortfolioState
from ..optimization.variables import (
    MultiPeriodMarketState,
    MultiPeriodPortfolioParameters,
    PortfolioParameters,
)
from ..dataset.instance import (
    OptimizationInstance,
    OptimizationInstanceMetadata,
)


@dataclass
class FactorModelSpec:
    """Specification for low-rank factor covariance model."""
    n_factors: int = 3
    factor_vol_annual: float = 0.16
    idiosyncratic_vol_annual: float = 0.20
    market_factor_loading_mean: float = 1.0
    market_factor_loading_std: float = 0.20


class ScalableInstanceGenerator:
    """
    Generates controlled synthetic portfolio optimization instances for scalability benchmarking.
    Guarantees PSD covariance and realistic heterogeneous market structures.
    """

    def __init__(self, factor_spec: Optional[FactorModelSpec] = None) -> None:
        self.factor_spec = factor_spec or FactorModelSpec()

    def generate_covariance_matrix(
        self,
        n_assets: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate guaranteed positive-definite covariance matrix via factor model:
            Sigma = B_f * Sigma_f * B_f^T + D_eps
        """
        K_f = self.factor_spec.n_factors
        dt = 5.0 / 252.0  # Weekly rebalancing period

        # 1. Factor covariance Sigma_f (K_f x K_f), strictly PD
        factor_vols = rng.uniform(0.12, 0.20, size=K_f) * np.sqrt(dt)
        # Random orthogonal rotation for factor correlations
        A = rng.normal(size=(K_f, K_f))
        Q, _ = np.linalg.qr(A)
        Sigma_f = Q @ np.diag(factor_vols ** 2) @ Q.T

        # 2. Factor loadings B_f (N x K_f)
        # Factor 0: Market factor with mean ~ 1.0
        b_market = rng.normal(
            self.factor_spec.market_factor_loading_mean,
            self.factor_spec.market_factor_loading_std,
            size=(n_assets, 1),
        )
        # Remaining factors: industry / style factors
        b_style = rng.normal(0.0, 0.40, size=(n_assets, K_f - 1)) if K_f > 1 else np.empty((n_assets, 0))
        B_f = np.hstack([b_market, b_style])

        # 3. Idiosyncratic diagonal variance D_eps (N x N)
        idio_vols = rng.uniform(0.15, 0.35, size=n_assets) * np.sqrt(dt)
        D_eps = np.diag(idio_vols ** 2)

        # 4. Total covariance Sigma = B_f * Sigma_f * B_f^T + D_eps
        Sigma = B_f @ Sigma_f @ B_f.T + D_eps

        # Symmetrize to eliminate numerical asymmetry
        Sigma = 0.5 * (Sigma + Sigma.T)

        # Verify minimal eigenvalue strictly positive
        min_eig = np.min(np.linalg.eigvalsh(Sigma))
        assert min_eig > 0, f"Covariance must be strictly positive definite, got min_eig={min_eig}"
        return Sigma

    def generate_instance(
        self,
        n_assets: int,
        horizon: int,
        cardinality_limit: int,
        seed: int,
        initial_held_count: Optional[int] = None,
        risk_aversion: float = 5.0,
        fixed_cost_dollars: float = 10.0,
        prop_cost_bps: float = 5.0,
        portfolio_wealth: float = 1_000_000.0,
    ) -> OptimizationInstance:
        """
        Generate a fully configured, mathematically valid OptimizationInstance.
        """
        rng = np.random.default_rng(seed)
        N = n_assets
        H = horizon
        dt = 5.0 / 252.0

        # Asset tickers
        asset_names = [f"SYNTH_{i:04d}" for i in range(N)]

        # 1. Expected Returns across horizon (H, N)
        # Calibrated around 8% annualized mean return with cross-sectional dispersion
        base_annual_ret = rng.normal(0.08, 0.12, size=N)
        mu_period = base_annual_ret * dt
        # Horizon expected returns with mild mean-reverting dynamics
        mu_mat = np.zeros((H, N))
        for h in range(H):
            noise = rng.normal(0.0, 0.002, size=N)
            mu_mat[h] = mu_period * (0.95 ** h) + noise

        # 2. Covariance Matrices across horizon (H, N, N)
        base_cov = self.generate_covariance_matrix(N, rng)
        cov_mat = np.tile(base_cov, (H, 1, 1))

        # 3. Gross returns for transitions (max(1, H-1), N)
        gross_mat = 1.0 + mu_mat[:-1] if H > 1 else np.ones((1, N))

        # 4. Current Holdings (Initial State)
        # Ensure individual holdings do not exceed weight upper limits (0.40) or violate liquidity bounds
        n_held = initial_held_count if initial_held_count is not None else min(cardinality_limit, max(1, int(0.5 * cardinality_limit)))
        curr_w = np.zeros(N)
        if n_held > 0 and n_held <= N:
            held_indices = rng.choice(N, size=n_held, replace=False)
            max_holding_cap = 0.30
            raw_weights = rng.uniform(0.10, 0.25, size=n_held)
            norm_weights = (raw_weights / np.sum(raw_weights)) * min(0.70, n_held * max_holding_cap)
            curr_w[held_indices] = norm_weights
        curr_cash = 1.0 - float(np.sum(curr_w))

        state = PortfolioState(
            actual_wealth=portfolio_wealth,
            normalized_weights=curr_w,
            normalized_cash=curr_cash,
            time_index=0,
            asset_names=asset_names,
        )

        mkt_state = MultiPeriodMarketState(
            initial_holdings=curr_w.copy(),
            initial_cash=curr_cash,
            expected_returns=mu_mat,
            covariance_matrices=cov_mat,
            gross_returns=gross_mat,
            asset_names=asset_names,
        )

        # 5. Parameters
        norm_f = fixed_cost_dollars / portfolio_wealth
        norm_a = prop_cost_bps * 1e-4

        single_p = PortfolioParameters(
            risk_aversion=risk_aversion,
            risk_free_rate=0.03 * dt,
            max_turnover=0.40,
            min_cash=0.10,
            weight_upper_limits=np.full(N, 0.40),
            liquidity_limits=np.full(N, 0.20),
            prop_cost_coeffs=np.full(N, norm_a),
            cardinality_limit=cardinality_limit,
            min_holding_weights=np.full(N, 0.04 if N > 20 else 0.08),
            min_trade_sizes=np.full(N, 0.01),
            fixed_cost_coeffs=np.full(N, norm_f),
        )
        step_params = MultiPeriodPortfolioParameters.from_single_period(single_p, horizon=H)

        meta = OptimizationInstanceMetadata(
            instance_id=f"synth_N{N}_H{H}_K{cardinality_limit}_seed{seed}",
            calendar_date="2024-06-01",
            time_index=0,
            split="synthetic_scalability",
            universe_id=f"SYNTH_{N}",
            n_assets=N,
            horizon=H,
            dataset_version="1.0",
            config_hash=f"synth_seed_{seed}",
            git_commit=None,
            gurobi_version="13.0",
            python_version="3.13",
            solver_params={},
            random_seed=seed,
            solver_status=2,
            label_quality="SYNTHETIC",
            is_trainable_label=False,
            objective=0.0,
            best_bound=0.0,
            mip_gap=0.0,
            runtime=0.0,
            work=0.0,
            node_count=0,
            sol_count=1,
            used_fallback=False,
        )

        return OptimizationInstance(
            metadata=meta,
            actual_state=state,
            params=step_params,
            market_state=mkt_state,
        )
