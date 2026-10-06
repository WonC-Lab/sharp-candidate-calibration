"""
Serialization and deserialization for OptimizationInstance.
Provides atomic .npz storage and exact MIQP replay execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import numpy as np

from ..dynamic.state import PortfolioState
from ..optimization.big_m import compute_dynamic_tight_big_m
from ..optimization.multi_period_miqp import MultiPeriodMIQPOptimizer
from ..optimization.variables import (
    MultiPeriodMarketState,
    MultiPeriodOptimizationResult,
    MultiPeriodPortfolioParameters,
    SolverConfig,
)
from .instance import (
    OptimizationInstance,
    OptimizationInstanceMetadata,
    OptimizationLabels,
    ContinuousRelaxationRecord,
)


def save_instance(
    instance: OptimizationInstance,
    output_dir: Union[Path, str],
    save_json_metadata: bool = True,
) -> Path:
    """
    Save an OptimizationInstance to an atomic .npz file and optional companion .json metadata.

    Args:
        instance: The OptimizationInstance to serialize.
        output_dir: Target directory (e.g. artifacts/instances/train/).
        save_json_metadata: Whether to write a human-readable companion JSON file.

    Returns:
        Path to the saved .npz file.
    """
    instance.validate()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filename = f"instance_{instance.metadata.instance_id}.npz"
    npz_filepath = out_path / filename

    # Collect arrays
    arrays_to_save: Dict[str, np.ndarray] = {
        # Actual state
        "normalized_weights": instance.actual_state.normalized_weights,
        "initial_holdings": instance.market_state.initial_holdings,
        # Inputs across horizon
        "expected_returns": instance.market_state.expected_returns,
        "covariance_matrices": instance.market_state.covariance_matrices,
        "risk_free_rates": instance.params.risk_free_rates,
        "max_turnovers": instance.params.max_turnovers,
        "min_cash": instance.params.min_cash,
        "weight_upper_limits": instance.params.weight_upper_limits,
        "liquidity_limits": instance.params.liquidity_limits,
        "prop_cost_coeffs": instance.params.prop_cost_coeffs,
    }

    if instance.market_state.gross_returns is not None:
        arrays_to_save["gross_returns"] = instance.market_state.gross_returns

    if instance.params.cardinality_limits is not None:
        arrays_to_save["cardinality_limits"] = instance.params.cardinality_limits
    if instance.params.min_holding_weights is not None:
        arrays_to_save["min_holding_weights"] = instance.params.min_holding_weights
    if instance.params.min_trade_sizes is not None:
        arrays_to_save["min_trade_sizes"] = instance.params.min_trade_sizes
    if instance.params.fixed_cost_coeffs is not None:
        arrays_to_save["fixed_cost_coeffs"] = instance.params.fixed_cost_coeffs

    # Dynamic Big-M
    gross_ret = instance.market_state.gross_returns if instance.market_state.gross_returns is not None else (
        np.ones((instance.params.horizon - 1, instance.metadata.n_assets)) if instance.params.horizon > 1 else None
    )
    _, _, m_symmetric = compute_dynamic_tight_big_m(
        initial_holdings=instance.market_state.initial_holdings,
        weight_upper_limits=instance.params.weight_upper_limits,
        liquidity_limits=instance.params.liquidity_limits,
        max_turnovers=instance.params.max_turnovers,
        gross_returns=gross_ret,
    )
    arrays_to_save["dynamic_big_m"] = m_symmetric

    # Solution arrays if available
    if instance.solution is not None and instance.solution.is_feasible:
        if instance.solution.pre_trade_holdings is not None:
            arrays_to_save["pre_trade_holdings"] = instance.solution.pre_trade_holdings
        if instance.solution.post_trade_holdings is not None:
            arrays_to_save["post_trade_holdings"] = instance.solution.post_trade_holdings
        if instance.solution.trades is not None:
            arrays_to_save["trades"] = instance.solution.trades
        if instance.solution.absolute_trades is not None:
            arrays_to_save["absolute_trades"] = instance.solution.absolute_trades
        if instance.solution.pre_trade_cash is not None:
            arrays_to_save["pre_trade_cash"] = instance.solution.pre_trade_cash
        if instance.solution.post_trade_cash is not None:
            arrays_to_save["post_trade_cash"] = instance.solution.post_trade_cash
        if instance.solution.transaction_costs is not None:
            arrays_to_save["transaction_costs"] = instance.solution.transaction_costs
        if instance.solution.holding_indicators is not None:
            arrays_to_save["holding_indicators"] = instance.solution.holding_indicators
        if instance.solution.trade_indicators is not None:
            arrays_to_save["trade_indicators"] = instance.solution.trade_indicators

    # Explicit labels
    if instance.labels is not None:
        arrays_to_save["y_first"] = instance.labels.y_first
        arrays_to_save["y_union"] = instance.labels.y_union
        arrays_to_save["z_first"] = instance.labels.z_first
        arrays_to_save["z_union"] = instance.labels.z_union

    # Relaxation arrays
    if instance.relaxation is not None and instance.relaxation.is_feasible:
        if instance.relaxation.relaxed_x is not None:
            arrays_to_save["relaxed_x"] = instance.relaxation.relaxed_x
        if instance.relaxation.relaxed_y is not None:
            arrays_to_save["relaxed_y"] = instance.relaxation.relaxed_y
        if instance.relaxation.relaxed_z is not None:
            arrays_to_save["relaxed_z"] = instance.relaxation.relaxed_z
        if instance.relaxation.reduced_costs_x is not None:
            arrays_to_save["reduced_costs_x"] = instance.relaxation.reduced_costs_x

    # Progress trajectory
    if instance.progress_trajectory:
        arrays_to_save["progress_trajectory"] = np.array(instance.progress_trajectory, dtype=float)

    # Asset names
    if instance.actual_state.asset_names is not None:
        arrays_to_save["asset_names"] = np.array(instance.actual_state.asset_names, dtype=str)

    # Encode metadata and scalar parameter dict
    meta_dict = {
        "metadata": {
            "instance_id": instance.metadata.instance_id,
            "calendar_date": instance.metadata.calendar_date,
            "time_index": instance.metadata.time_index,
            "split": instance.metadata.split,
            "universe_id": instance.metadata.universe_id,
            "n_assets": instance.metadata.n_assets,
            "horizon": instance.metadata.horizon,
            "dataset_version": instance.metadata.dataset_version,
            "config_hash": instance.metadata.config_hash,
            "git_commit": instance.metadata.git_commit,
            "gurobi_version": instance.metadata.gurobi_version,
            "python_version": instance.metadata.python_version,
            "solver_params": instance.metadata.solver_params,
            "random_seed": instance.metadata.random_seed,
            "solver_status": instance.metadata.solver_status,
            "label_quality": instance.metadata.label_quality,
            "is_trainable_label": bool(instance.metadata.is_trainable_label),
            "objective": instance.metadata.objective,
            "best_bound": instance.metadata.best_bound,
            "mip_gap": instance.metadata.mip_gap,
            "runtime": instance.metadata.runtime,
            "work": instance.metadata.work,
            "node_count": instance.metadata.node_count,
            "sol_count": int(instance.metadata.sol_count),
            "used_fallback": bool(instance.metadata.used_fallback),
        },
        "actual_state": {
            "actual_wealth": float(instance.actual_state.actual_wealth),
            "normalized_cash": float(instance.actual_state.normalized_cash),
            "time_index": int(instance.actual_state.time_index),
        },
        "params": {
            "risk_aversion": float(instance.params.risk_aversion),
            "horizon": int(instance.params.horizon),
            "discount_factor": float(instance.params.discount_factor),
            "sector_indices": instance.params.sector_indices,
            "sector_bounds": {k: [float(b[0]), float(b[1])] for k, b in instance.params.sector_bounds.items()},
            "allow_degenerate": bool(instance.params.allow_degenerate),
        },
        "market_state": {
            "initial_cash": float(instance.market_state.initial_cash),
        },
        "relaxation_metadata": {
            "status": instance.relaxation.status if instance.relaxation else None,
            "is_feasible": instance.relaxation.is_feasible if instance.relaxation else False,
            "objective_value": instance.relaxation.objective_value if instance.relaxation else None,
            "solve_time": instance.relaxation.solve_time if instance.relaxation else 0.0,
        } if instance.relaxation else None,
    }

    meta_json_str = json.dumps(meta_dict, indent=2)
    arrays_to_save["meta_json"] = np.array([meta_json_str], dtype=object)

    np.savez_compressed(npz_filepath, **arrays_to_save)

    if save_json_metadata:
        json_filepath = out_path / f"instance_{instance.metadata.instance_id}.json"
        with open(json_filepath, "w", encoding="utf-8") as f:
            f.write(meta_json_str)

    return npz_filepath


def load_instance(filepath: Union[Path, str]) -> OptimizationInstance:
    """
    Load an OptimizationInstance from an atomic .npz file.
    Reconstructs the full instance without any hidden external state.
    """
    npz_path = Path(filepath)
    if not npz_path.exists():
        raise FileNotFoundError(f"Instance file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        meta_json_str = str(data["meta_json"][0])
        meta_dict = json.loads(meta_json_str)

        raw_meta = meta_dict["metadata"]
        raw_actual = meta_dict["actual_state"]
        raw_params = meta_dict["params"]
        raw_mkt = meta_dict["market_state"]
        raw_relax = meta_dict.get("relaxation_metadata")

        metadata = OptimizationInstanceMetadata(
            instance_id=raw_meta["instance_id"],
            calendar_date=raw_meta["calendar_date"],
            time_index=int(raw_meta["time_index"]),
            split=raw_meta["split"],
            universe_id=raw_meta["universe_id"],
            n_assets=int(raw_meta["n_assets"]),
            horizon=int(raw_meta["horizon"]),
            dataset_version=raw_meta["dataset_version"],
            config_hash=raw_meta["config_hash"],
            git_commit=raw_meta.get("git_commit"),
            gurobi_version=raw_meta["gurobi_version"],
            python_version=raw_meta["python_version"],
            solver_params=raw_meta["solver_params"],
            random_seed=raw_meta.get("random_seed"),
            solver_status=raw_meta["solver_status"],
            label_quality=raw_meta["label_quality"],
            is_trainable_label=bool(raw_meta["is_trainable_label"]),
            objective=raw_meta.get("objective"),
            best_bound=raw_meta.get("best_bound"),
            mip_gap=raw_meta.get("mip_gap"),
            runtime=float(raw_meta["runtime"]),
            work=raw_meta.get("work"),
            node_count=raw_meta.get("node_count"),
            sol_count=int(raw_meta.get("sol_count", 0)),
            used_fallback=bool(raw_meta.get("used_fallback", False)),
        )

        asset_names = [str(x) for x in data["asset_names"]] if "asset_names" in data else None

        actual_state = PortfolioState(
            actual_wealth=float(raw_actual["actual_wealth"]),
            normalized_weights=data["normalized_weights"],
            normalized_cash=float(raw_actual["normalized_cash"]),
            time_index=int(raw_actual["time_index"]),
            asset_names=asset_names,
        )

        sec_bounds = {k: (v[0], v[1]) for k, v in raw_params.get("sector_bounds", {}).items()}
        params = MultiPeriodPortfolioParameters(
            risk_aversion=float(raw_params["risk_aversion"]),
            horizon=int(raw_params["horizon"]),
            risk_free_rates=data["risk_free_rates"],
            max_turnovers=data["max_turnovers"],
            min_cash=data["min_cash"],
            weight_upper_limits=data["weight_upper_limits"],
            liquidity_limits=data["liquidity_limits"],
            prop_cost_coeffs=data["prop_cost_coeffs"],
            discount_factor=float(raw_params["discount_factor"]),
            sector_indices=raw_params.get("sector_indices", {}),
            sector_bounds=sec_bounds,
            cardinality_limits=data["cardinality_limits"] if "cardinality_limits" in data else None,
            min_holding_weights=data["min_holding_weights"] if "min_holding_weights" in data else None,
            min_trade_sizes=data["min_trade_sizes"] if "min_trade_sizes" in data else None,
            fixed_cost_coeffs=data["fixed_cost_coeffs"] if "fixed_cost_coeffs" in data else None,
            allow_degenerate=bool(raw_params.get("allow_degenerate", False)),
        )

        gross_ret = data["gross_returns"] if "gross_returns" in data else None
        market_state = MultiPeriodMarketState(
            initial_holdings=data["initial_holdings"],
            initial_cash=float(raw_mkt["initial_cash"]),
            expected_returns=data["expected_returns"],
            covariance_matrices=data["covariance_matrices"],
            gross_returns=gross_ret,
            asset_names=asset_names,
        )

        solution = None
        if metadata.sol_count > 0 and "post_trade_holdings" in data:
            solution = MultiPeriodOptimizationResult(
                status=metadata.solver_status,
                is_optimal=(metadata.solver_status == "OPTIMAL"),
                is_feasible=True,
                solve_time=metadata.runtime,
                objective_value=metadata.objective,
                pre_trade_holdings=data.get("pre_trade_holdings"),
                post_trade_holdings=data.get("post_trade_holdings"),
                trades=data.get("trades"),
                absolute_trades=data.get("absolute_trades"),
                pre_trade_cash=data.get("pre_trade_cash"),
                post_trade_cash=data.get("post_trade_cash"),
                transaction_costs=data.get("transaction_costs"),
                holding_indicators=data.get("holding_indicators"),
                trade_indicators=data.get("trade_indicators"),
                best_bound=metadata.best_bound,
                mip_gap=metadata.mip_gap,
                node_count=metadata.node_count,
                work=metadata.work,
                sol_count=metadata.sol_count,
                metadata={"dynamic_big_m": data.get("dynamic_big_m")},
            )

        labels = None
        if "y_first" in data:
            labels = OptimizationLabels(
                y_first=data["y_first"],
                y_union=data["y_union"],
                z_first=data["z_first"],
                z_union=data["z_union"],
            )

        relaxation = None
        if raw_relax is not None:
            relaxation = ContinuousRelaxationRecord(
                status=raw_relax["status"],
                is_feasible=bool(raw_relax["is_feasible"]),
                objective_value=raw_relax.get("objective_value"),
                solve_time=float(raw_relax.get("solve_time", 0.0)),
                relaxed_x=data.get("relaxed_x"),
                relaxed_y=data.get("relaxed_y"),
                relaxed_z=data.get("relaxed_z"),
                reduced_costs_x=data.get("reduced_costs_x"),
            )

        progress_trajectory = None
        if "progress_trajectory" in data:
            arr_traj = data["progress_trajectory"]
            progress_trajectory = [
                (float(r[0]), float(r[1]), float(r[2]), float(r[3]), int(r[4]))
                for r in arr_traj
            ]

    instance = OptimizationInstance(
        metadata=metadata,
        actual_state=actual_state,
        params=params,
        market_state=market_state,
        solution=solution,
        labels=labels,
        relaxation=relaxation,
        progress_trajectory=progress_trajectory,
    )
    instance.validate()
    return instance


def rebuild_and_solve_miqp(
    instance: OptimizationInstance,
    solver_config: Optional[SolverConfig] = None,
) -> MultiPeriodOptimizationResult:
    """
    Rebuild and re-solve the Phase 3 multi-period MIQP from a loaded OptimizationInstance.
    Verifies that the serialized instance is completely self-contained and reproducible.
    """
    cfg = solver_config or SolverConfig(
        time_limit=float(instance.metadata.solver_params.get("time_limit", 60.0)),
        mip_gap=float(instance.metadata.solver_params.get("mip_gap", 1e-4)),
        threads=int(instance.metadata.solver_params.get("threads", 1)),
        output_flag=False,
    )
    optimizer = MultiPeriodMIQPOptimizer(instance.params, cfg)
    return optimizer.solve(instance.market_state)
