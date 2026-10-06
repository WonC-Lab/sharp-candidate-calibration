"""
Historical Optimization Instance Dataset Generator.
Executes the full-universe Phase 3 MIQP teacher policy inside the Phase 4 MPC simulator,
producing an expert-trajectory supervised learning dataset with continuous relaxation bounds
and strict chronological partitioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import subprocess
from typing import Dict, List, Optional, Tuple, Union, Any
import gurobipy as gp
import numpy as np

from ..dynamic.rolling_horizon import FirstAction, OptimizationFailureError
from ..dynamic.state import PortfolioState
from ..dynamic.transitions import (
    realized_market_transition,
    renormalize_state,
    rescale_fixed_costs,
    rescale_liquidity_limits,
    validate_first_action_execution,
)
from ..forecasts.base import BaseForecastProvider, ForecastBundle
from ..optimization.big_m import compute_dynamic_tight_big_m
from ..optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from ..optimization.relaxation import MultiPeriodRelaxedQPOptimizer
from ..optimization.variables import (
    MultiPeriodMarketState,
    MultiPeriodOptimizationResult,
    MultiPeriodPortfolioParameters,
    SolverConfig,
)
from ..solver.callbacks import OptimizationProgressLogger
from .instance import (
    OptimizationInstance,
    OptimizationInstanceMetadata,
    OptimizationLabels,
    ContinuousRelaxationRecord,
    compute_config_hash,
)
from .manifest import DatasetManifest
from .quality import LabelQuality, LabelQualityConfig, classify_label_quality
from .schema import DATASET_SCHEMA_VERSION
from .serialization import save_instance
from .split import DatasetSplit, DatasetSplitConfig


def _get_git_commit() -> Optional[str]:
    """Retrieve current Git HEAD SHA if available."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return None


@dataclass
class DatasetGeneratorConfig:
    """Configuration for historical dataset generation."""
    universe_id: str = "CANONICAL_UNIVERSE"
    dataset_version: str = DATASET_SCHEMA_VERSION
    random_seed: Optional[int] = 42
    output_dir: Union[Path, str] = "artifacts/instances"
    solver_config: Optional[SolverConfig] = None
    label_quality_config: Optional[LabelQualityConfig] = None
    allow_fallback_on_failure: bool = False
    compute_relaxations: bool = True
    save_json_metadata: bool = True


class HistoricalDatasetGenerator:
    """
    Expert-trajectory optimization instance generator.
    Solves full-MIQP MPC along an actual execution path and persists instances.
    """

    def __init__(self, config: Optional[DatasetGeneratorConfig] = None) -> None:
        self.config = config or DatasetGeneratorConfig()
        self.solver_config = self.config.solver_config or SolverConfig()
        self.quality_config = self.config.label_quality_config or LabelQualityConfig()
        self.git_commit = _get_git_commit()
        self.gurobi_ver = ".".join(str(x) for x in gp.gurobi.version())
        self.python_ver = platform.python_version()

    def generate(
        self,
        initial_state: PortfolioState,
        horizon: int,
        num_steps: int,
        forecast_provider: BaseForecastProvider,
        realized_risky_returns: np.ndarray,      # (num_steps, n)
        realized_risk_free_rates: np.ndarray,    # (num_steps,)
        split_config: DatasetSplitConfig,
        calendar_dates: Optional[List[Union[int, str]]] = None,
    ) -> DatasetManifest:
        """
        Execute full-MIQP MPC simulation, serialize all instances, and write manifest.

        Args:
            initial_state: Initial physical PortfolioState at t=0.
            horizon: Multi-period planning horizon H.
            num_steps: Total decision steps T.
            forecast_provider: Source of information-bounded forecasts.
            realized_risky_returns: Matrix of actual realized returns, shape (num_steps, n).
            realized_risk_free_rates: Vector of actual risk-free rates, shape (num_steps,).
            split_config: Chronological train/validation/test partitioning configuration.
            calendar_dates: Optional list of calendar date labels for each step t in 0..num_steps-1.

        Returns:
            DatasetManifest indexing all generated instances.
        """
        n = initial_state.n_assets
        if realized_risky_returns.shape != (num_steps, n):
            raise ValueError(
                f"realized_risky_returns shape {realized_risky_returns.shape} != ({num_steps}, {n})"
            )
        if realized_risk_free_rates.shape != (num_steps,):
            raise ValueError(
                f"realized_risk_free_rates shape {realized_risk_free_rates.shape} != ({num_steps},)"
            )

        base_out = Path(self.config.output_dir)
        base_out.mkdir(parents=True, exist_ok=True)

        manifest = DatasetManifest()
        current_state = initial_state

        for t in range(num_steps):
            date_label = calendar_dates[t] if calendar_dates is not None else f"T_{t:04d}"
            split = split_config.assign_split(date_label if calendar_dates is not None else t)
            split_dir = base_out / split.value
            split_dir.mkdir(parents=True, exist_ok=True)

            instance_id = f"{self.config.universe_id}_{t:04d}"

            # 1. Obtain information-bounded forecast at decision time t
            forecast = forecast_provider.forecast(current_time=t, horizon=horizon)

            # 2. Rescale dollar-denominated bounds to current wealth V_t
            V_t = current_state.actual_wealth
            if forecast.liquidity_limits is not None:
                norm_liquidity = forecast.liquidity_limits
            elif forecast.dollar_adv is not None and forecast.participation_rates is not None:
                norm_liquidity = np.zeros((horizon, n))
                for h in range(horizon):
                    norm_liquidity[h] = rescale_liquidity_limits(
                        dollar_adv=forecast.dollar_adv[h],
                        participation_rates=forecast.participation_rates,
                        actual_wealth=V_t,
                    )
            else:
                norm_liquidity = np.full((horizon, n), 1.0)

            if forecast.fixed_cost_coeffs is not None:
                norm_fixed_costs = forecast.fixed_cost_coeffs
            elif forecast.dollar_fixed_costs is not None:
                norm_fixed_costs = np.zeros((horizon, n))
                for h in range(horizon):
                    norm_fixed_costs[h] = rescale_fixed_costs(
                        dollar_fixed_costs=forecast.dollar_fixed_costs[h],
                        actual_wealth=V_t,
                    )
            else:
                norm_fixed_costs = np.zeros((horizon, n))

            # 3. Build normalized parameters and market state
            risk_aversion = 2.0
            params = MultiPeriodPortfolioParameters(
                risk_aversion=risk_aversion,
                horizon=horizon,
                risk_free_rates=forecast.risk_free_rates,
                max_turnovers=forecast.max_turnovers,
                min_cash=forecast.min_cash,
                weight_upper_limits=forecast.weight_upper_limits,
                liquidity_limits=norm_liquidity,
                prop_cost_coeffs=forecast.prop_cost_coeffs,
                discount_factor=forecast.discount_factor,
                sector_indices=forecast.sector_indices,
                sector_bounds=forecast.sector_bounds,
                cardinality_limits=forecast.cardinality_limits,
                min_holding_weights=forecast.min_holding_weights,
                min_trade_sizes=forecast.min_trade_sizes,
                fixed_cost_coeffs=norm_fixed_costs,
            )

            market_state = MultiPeriodMarketState(
                initial_holdings=current_state.normalized_weights.copy(),
                initial_cash=current_state.normalized_cash,
                expected_returns=forecast.expected_returns,
                covariance_matrices=forecast.covariance_matrices,
                gross_returns=forecast.gross_returns,
                asset_names=current_state.asset_names,
            )

            # 4. Continuous relaxation solve (if enabled)
            relaxation_record = None
            if self.config.compute_relaxations:
                relax_opt = MultiPeriodRelaxedQPOptimizer(params, self.solver_config)
                relax_res = relax_opt.solve(market_state)
                relaxation_record = ContinuousRelaxationRecord.from_result(relax_res)

            # 5. Full MIQP teacher solve with callback tracking
            miqp_opt = MultiPeriodMIQPOptimizer(params, self.solver_config)
            cb_logger = OptimizationProgressLogger()
            miqp_res = miqp_opt.solve(market_state, callback=cb_logger)

            # 6. Fallback handling and label classification
            used_fallback = False
            first_action = None
            labels = None

            if miqp_res.is_feasible and miqp_res.post_trade_holdings is not None:
                first_action = FirstAction(
                    trade=miqp_res.trades[0],
                    post_trade_holdings=miqp_res.post_trade_holdings[0],
                    post_trade_cash=float(miqp_res.post_trade_cash[0]),
                    absolute_trade=miqp_res.absolute_trades[0],
                    transaction_cost=float(miqp_res.transaction_costs[0]),
                    holding_indicator=miqp_res.holding_indicators[0],
                    trade_indicator=miqp_res.trade_indicators[0],
                )
                labels = OptimizationLabels.from_miqp_result(miqp_res)
            else:
                if self.config.allow_fallback_on_failure:
                    used_fallback = True
                    # Zero-trade fallback
                    first_action = FirstAction(
                        trade=np.zeros(n),
                        post_trade_holdings=current_state.normalized_weights.copy(),
                        post_trade_cash=current_state.normalized_cash,
                        absolute_trade=np.zeros(n),
                        transaction_cost=0.0,
                        holding_indicator=np.where(current_state.normalized_weights > 1e-6, 1, 0),
                        trade_indicator=np.zeros(n, dtype=int),
                    )
                else:
                    raise OptimizationFailureError(
                        f"Teacher MIQP failed at t={t} with status {miqp_res.status}."
                    )

            quality, is_trainable = classify_label_quality(
                solver_status=miqp_res.status,
                is_optimal=miqp_res.is_optimal,
                is_feasible=miqp_res.is_feasible,
                mip_gap=miqp_res.mip_gap,
                used_fallback=used_fallback,
                config=self.quality_config,
            )

            # 7. Compute config hash
            cfg_hash = compute_config_hash(
                universe_id=self.config.universe_id,
                n_assets=n,
                horizon=horizon,
                risk_aversion=risk_aversion,
                discount_factor=params.discount_factor,
                solver_config=self.solver_config,
                dataset_version=self.config.dataset_version,
            )

            # 8. Create and validate instance
            metadata = OptimizationInstanceMetadata(
                instance_id=instance_id,
                calendar_date=date_label,
                time_index=t,
                split=split.value,
                universe_id=self.config.universe_id,
                n_assets=n,
                horizon=horizon,
                dataset_version=self.config.dataset_version,
                config_hash=cfg_hash,
                git_commit=self.git_commit,
                gurobi_version=self.gurobi_ver,
                python_version=self.python_ver,
                solver_params={
                    "time_limit": self.solver_config.time_limit,
                    "mip_gap": self.solver_config.mip_gap,
                    "threads": self.solver_config.threads,
                },
                random_seed=self.config.random_seed,
                solver_status=miqp_res.status,
                label_quality=quality.value,
                is_trainable_label=is_trainable,
                objective=miqp_res.objective_value,
                best_bound=miqp_res.best_bound,
                mip_gap=miqp_res.mip_gap,
                runtime=miqp_res.solve_time,
                work=miqp_res.work,
                node_count=miqp_res.node_count,
                sol_count=miqp_res.sol_count if miqp_res.sol_count is not None else (1 if miqp_res.is_feasible else 0),
                used_fallback=used_fallback,
            )

            instance = OptimizationInstance(
                metadata=metadata,
                actual_state=current_state,
                params=params,
                market_state=market_state,
                solution=miqp_res if miqp_res.is_feasible else None,
                labels=labels,
                relaxation=relaxation_record,
                progress_trajectory=cb_logger.trajectory if cb_logger.trajectory else None,
            )

            # 9. Save instance to disk
            npz_path = save_instance(
                instance=instance,
                output_dir=split_dir,
                save_json_metadata=self.config.save_json_metadata,
            )

            # 10. Catalog in manifest
            rel_filename = f"{split.value}/{npz_path.name}"
            manifest.add_instance(instance=instance, filename=rel_filename)

            # 11. Advance simulation state
            validate_first_action_execution(
                post_trade_holdings=first_action.post_trade_holdings,
                post_trade_cash=first_action.post_trade_cash,
                transaction_cost=first_action.transaction_cost,
            )

            r_real_risky = realized_risky_returns[t]
            r_real_rf = float(realized_risk_free_rates[t])

            x_tilde, b_tilde, w_growth = realized_market_transition(
                post_trade_holdings=first_action.post_trade_holdings,
                post_trade_cash=first_action.post_trade_cash,
                realized_risky_returns=r_real_risky,
                realized_risk_free_rate=r_real_rf,
            )

            current_state = renormalize_state(
                x_tilde_minus=x_tilde,
                b_tilde_minus=b_tilde,
                wealth_growth_factor=w_growth,
                current_wealth=current_state.actual_wealth,
                current_time_index=t,
            )

        # 12. Save manifest to disk
        manifest_csv_path = base_out / "manifest.csv"
        manifest.save_csv(manifest_csv_path)

        return manifest
