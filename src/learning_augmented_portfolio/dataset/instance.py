"""
OptimizationInstance and OptimizationLabels dataclasses.
Encapsulates complete instance inputs, actual state, full teacher solution, explicit labels,
solver diagnostics, anytime trajectory, and continuous relaxation features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import platform
from typing import Dict, List, Optional, Tuple, Any, Union
import gurobipy as gp
import numpy as np

from ..dynamic.state import PortfolioState
from ..optimization.relaxation import ContinuousRelaxationResult
from ..optimization.variables import (
    MultiPeriodMarketState,
    MultiPeriodOptimizationResult,
    MultiPeriodPortfolioParameters,
    SolverConfig,
)
from .quality import LabelQuality
from .schema import DATASET_SCHEMA_VERSION, validate_array_shapes


def compute_config_hash(
    universe_id: str,
    n_assets: int,
    horizon: int,
    risk_aversion: float,
    discount_factor: float,
    solver_config: SolverConfig,
    dataset_version: str = DATASET_SCHEMA_VERSION,
) -> str:
    """Compute a deterministic SHA-256 hash uniquely characterizing the instance configuration."""
    canonical_dict = {
        "universe_id": str(universe_id),
        "n_assets": int(n_assets),
        "horizon": int(horizon),
        "risk_aversion": round(float(risk_aversion), 6),
        "discount_factor": round(float(discount_factor), 6),
        "solver_time_limit": float(solver_config.time_limit),
        "solver_mip_gap": float(solver_config.mip_gap),
        "solver_threads": int(solver_config.threads),
        "dataset_version": str(dataset_version),
    }
    encoded = json.dumps(canonical_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class OptimizationLabels:
    """
    Explicit supervised learning labels constructed from the full MIQP teacher solution.
    
    Attributes:
        y_first: Y^{first}_{i,t} = y^*_{i, 0|t}, first-period holding indicator in {0, 1}^n.
        y_union: Y^{union}_{i,t} = max_{h=0..H-1} y^*_{i, h|t}, horizon-union holding indicator in {0, 1}^n.
        z_first: Z^{first}_{i,t} = z^*_{i, 0|t}, first-period trade indicator in {0, 1}^n.
        z_union: Z^{union}_{i,t} = max_{h=0..H-1} z^*_{i, h|t}, horizon-union trade indicator in {0, 1}^n.
    """
    y_first: np.ndarray
    y_union: np.ndarray
    z_first: np.ndarray
    z_union: np.ndarray

    def validate(self, n_assets: int) -> None:
        """Validate shapes and binary indicator integrity."""
        for name, arr in [
            ("y_first", self.y_first),
            ("y_union", self.y_union),
            ("z_first", self.z_first),
            ("z_union", self.z_union),
        ]:
            if arr.shape != (n_assets,):
                raise ValueError(f"Label '{name}' shape {arr.shape} != ({n_assets},)")
            if not np.all(np.isin(arr, [0, 1])):
                raise ValueError(f"Label '{name}' contains non-binary values: {arr}")

    @classmethod
    def from_miqp_result(
        cls,
        result: MultiPeriodOptimizationResult,
    ) -> OptimizationLabels:
        """Construct canonical labels from a solved MultiPeriodOptimizationResult."""
        if result.holding_indicators is None or result.trade_indicators is None:
            raise ValueError("Cannot construct labels from an optimization result without indicators.")
        y_all = result.holding_indicators   # (H, n)
        z_all = result.trade_indicators     # (H, n)

        y_first = y_all[0].copy().astype(int)
        y_union = np.max(y_all, axis=0).astype(int)
        z_first = z_all[0].copy().astype(int)
        z_union = np.max(z_all, axis=0).astype(int)

        return cls(
            y_first=y_first,
            y_union=y_union,
            z_first=z_first,
            z_union=z_union,
        )


@dataclass
class ContinuousRelaxationRecord:
    """
    Continuous relaxation record storing relaxed decision variables and lower bound features.
    """
    status: str
    is_feasible: bool
    objective_value: Optional[float]
    solve_time: float
    relaxed_x: Optional[np.ndarray] = None          # (H, n)
    relaxed_y: Optional[np.ndarray] = None          # (H, n) fractional in [0, 1]
    relaxed_z: Optional[np.ndarray] = None          # (H, n) fractional in [0, 1]
    reduced_costs_x: Optional[np.ndarray] = None    # (H, n)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(cls, res: ContinuousRelaxationResult) -> ContinuousRelaxationRecord:
        return cls(
            status=res.status,
            is_feasible=res.is_feasible,
            objective_value=res.objective_value,
            solve_time=res.solve_time,
            relaxed_x=res.relaxed_x,
            relaxed_y=res.relaxed_y,
            relaxed_z=res.relaxed_z,
            reduced_costs_x=res.reduced_costs_x,
            metadata=res.metadata or {},
        )


@dataclass
class OptimizationInstanceMetadata:
    """Scalar and string metadata uniquely identifying an optimization instance."""
    instance_id: str
    calendar_date: Union[int, str]
    time_index: int
    split: str
    universe_id: str
    n_assets: int
    horizon: int
    dataset_version: str
    config_hash: str
    git_commit: Optional[str]
    gurobi_version: str
    python_version: str
    solver_params: Dict[str, Any]
    random_seed: Optional[int]
    solver_status: str
    label_quality: str
    is_trainable_label: bool
    objective: Optional[float]
    best_bound: Optional[float]
    mip_gap: Optional[float]
    runtime: float
    work: Optional[float]
    node_count: Optional[int]
    sol_count: int
    used_fallback: bool


@dataclass
class OptimizationInstance:
    """
    Full canonical optimization instance.
    Completely self-contained: contains all information required to reconstruct,
    re-solve, and analyze the instance without any hidden external state.
    """
    metadata: OptimizationInstanceMetadata
    actual_state: PortfolioState
    params: MultiPeriodPortfolioParameters
    market_state: MultiPeriodMarketState
    solution: Optional[MultiPeriodOptimizationResult] = None
    labels: Optional[OptimizationLabels] = None
    relaxation: Optional[ContinuousRelaxationRecord] = None
    progress_trajectory: Optional[List[Tuple[float, float, float, float, int]]] = None

    def validate(self) -> None:
        """Rigorously validate self-consistency across all components."""
        n = self.metadata.n_assets
        H = self.metadata.horizon

        # 1. State normalization identity: 1^T w_bar + c_bar == 1.0
        tot_wealth = np.sum(self.actual_state.normalized_weights) + self.actual_state.normalized_cash
        if not np.isclose(tot_wealth, 1.0, atol=1e-5):
            raise ValueError(f"Actual state normalization violated: {tot_wealth} != 1.0")

        # 2. Asset count and name consistency
        if self.actual_state.n_assets != n:
            raise ValueError(f"State n_assets {self.actual_state.n_assets} != metadata {n}")

        if self.actual_state.asset_names is not None:
            if len(self.actual_state.asset_names) != n:
                raise ValueError("Asset names count does not match n_assets")
            if self.market_state.asset_names is not None:
                if self.market_state.asset_names != self.actual_state.asset_names:
                    raise ValueError("Asset names in market_state do not match actual_state")

        # 3. Parameters and Market State
        self.params.validate(n)
        self.market_state.validate()

        # 4. Consistency between actual_state and market_state initial holdings
        if not np.allclose(self.actual_state.normalized_weights, self.market_state.initial_holdings, atol=1e-7):
            raise ValueError("actual_state.normalized_weights does not match market_state.initial_holdings")
        if not np.isclose(self.actual_state.normalized_cash, self.market_state.initial_cash, atol=1e-7):
            raise ValueError("actual_state.normalized_cash does not match market_state.initial_cash")

        # 5. Label validation
        if self.labels is not None:
            self.labels.validate(n)
