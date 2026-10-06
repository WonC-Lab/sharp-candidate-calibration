"""
Warm-Start Data Containers and Gurobi Injection Interface.
Supports:
  - W0: No Start
  - W1: Previous MPC Solution Horizon Shift
  - W2: ML Discrete Binary Start
  - W3: ML + Continuous Repaired Complete Start
Strictly isolates warm-start injection from optimization model constraints and feasible sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Dict, Optional, Tuple, Any
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from ..optimization.support_restriction import CandidateUniverse
from ..optimization.variables import MultiPeriodOptimizationResult


class WarmStartType(str, Enum):
    NO_START = "NO_START"
    PREVIOUS_SOLUTION = "PREVIOUS_SOLUTION"
    ML_DISCRETE = "ML_DISCRETE"
    ML_REPAIRED = "ML_REPAIRED"


@dataclass
class MIPStartData:
    """Primal MIP start data for multi-period portfolio optimization."""
    start_type: WarmStartType
    y_start: Optional[np.ndarray] = None       # shape (H, N) or (H, B)
    z_start: Optional[np.ndarray] = None       # shape (H, N) or (H, B)
    x_start: Optional[np.ndarray] = None       # shape (H, N) or (H, B)
    u_start: Optional[np.ndarray] = None       # shape (H, N) or (H, B)
    b_start: Optional[np.ndarray] = None       # shape (H,)
    is_complete: bool = False
    construction_time: float = 0.0
    repair_time: float = 0.0
    metadata: Dict[str, Any] = None


def build_previous_solution_warm_start(
    prev_sol: MultiPeriodOptimizationResult,
    horizon: int,
    n_assets: int,
) -> MIPStartData:
    """
    Construct W1 Previous-Solution warm start by shifting horizon forward:
      y_{h|t} = y^*_{h+1|t-1} for h in {0, ..., H-2}, y_{H-1|t} = y^*_{H-1|t-1}
      z_{h|t} = z^*_{h+1|t-1} for h in {0, ..., H-2}, z_{H-1|t} = 0 (no trade)
    """
    t0 = time.perf_counter()
    y_prev = prev_sol.holding_indicators
    z_prev = prev_sol.trade_indicators
    x_prev = prev_sol.post_trade_holdings
    b_prev = prev_sol.post_trade_cash

    y_shift = np.zeros((horizon, n_assets), dtype=int)
    z_shift = np.zeros((horizon, n_assets), dtype=int)
    x_shift = np.zeros((horizon, n_assets), dtype=float) if x_prev is not None else None
    b_shift = np.zeros(horizon, dtype=float) if b_prev is not None else None

    # Shift periods 1..H-1 to 0..H-2
    copy_len = min(horizon - 1, len(y_prev) - 1)
    if copy_len > 0:
        y_shift[:copy_len] = y_prev[1:copy_len + 1]
        z_shift[:copy_len] = z_prev[1:copy_len + 1]
        if x_shift is not None and x_prev is not None:
            x_shift[:copy_len] = x_prev[1:copy_len + 1]
        if b_shift is not None and b_prev is not None:
            b_shift[:copy_len] = b_prev[1:copy_len + 1]

    # Final period completion rule: repeat holding, no trade
    y_shift[horizon - 1] = y_prev[-1]
    z_shift[horizon - 1] = 0
    if x_shift is not None and x_prev is not None:
        x_shift[horizon - 1] = x_prev[-1]
    if b_shift is not None and b_prev is not None:
        b_shift[horizon - 1] = b_prev[-1]

    t_construct = time.perf_counter() - t0
    return MIPStartData(
        start_type=WarmStartType.PREVIOUS_SOLUTION,
        y_start=y_shift,
        z_start=z_shift,
        x_start=x_shift,
        b_start=b_shift,
        is_complete=(x_shift is not None),
        construction_time=t_construct,
    )


def inject_warm_start_into_model(
    model: gp.Model,
    start_data: Optional[MIPStartData],
    var_y: Any,
    var_z: Any,
    var_x: Optional[Any] = None,
    var_b: Optional[Any] = None,
    candidate: Optional[CandidateUniverse] = None,
) -> int:
    """
    Inject MIPStartData into Gurobi model variables via the .Start attribute.
    Does NOT alter constraints, bounds, or objectives.

    Args:
        model: Gurobi Model object.
        start_data: MIPStartData to inject (or None for NO_START).
        var_y: Gurobi variables for y (holding).
        var_z: Gurobi variables for z (trading).
        var_x: Optional Gurobi variables for continuous x.
        var_b: Optional Gurobi variables for cash b.
        candidate: Optional CandidateUniverse if model is physically sliced.

    Returns:
        Number of variables initialized.
    """
    if start_data is None or start_data.start_type == WarmStartType.NO_START:
        return 0

    assigned_count = 0
    H = len(start_data.y_start) if start_data.y_start is not None else 0

    if candidate is not None:
        # Model is physically sliced to candidate universe C_t
        # start_data might be in full dimension N or sliced dimension B
        cand_indices = candidate.candidate_indices
        B = candidate.candidate_size

        if start_data.y_start is not None:
            full_N = start_data.y_start.shape[1]
            for h in range(H):
                for red_idx, full_idx in enumerate(cand_indices):
                    val_y = start_data.y_start[h, full_idx] if full_N > B else start_data.y_start[h, red_idx]
                    var_y[h, red_idx].Start = float(val_y)
                    assigned_count += 1

                    if start_data.z_start is not None:
                        val_z = start_data.z_start[h, full_idx] if full_N > B else start_data.z_start[h, red_idx]
                        var_z[h, red_idx].Start = float(val_z)
                        assigned_count += 1

                    if var_x is not None and start_data.x_start is not None:
                        val_x = start_data.x_start[h, full_idx] if full_N > B else start_data.x_start[h, red_idx]
                        var_x[h, red_idx].Start = float(val_x)
                        assigned_count += 1

                if var_b is not None and start_data.b_start is not None:
                    var_b[h].Start = float(start_data.b_start[h])
                    assigned_count += 1
    else:
        # Full-dimension model
        N = start_data.y_start.shape[1] if start_data.y_start is not None else 0
        for h in range(H):
            for i in range(N):
                if start_data.y_start is not None:
                    var_y[h, i].Start = float(start_data.y_start[h, i])
                    assigned_count += 1
                if start_data.z_start is not None:
                    var_z[h, i].Start = float(start_data.z_start[h, i])
                    assigned_count += 1
                if var_x is not None and start_data.x_start is not None:
                    var_x[h, i].Start = float(start_data.x_start[h, i])
                    assigned_count += 1

            if var_b is not None and start_data.b_start is not None:
                var_b[h].Start = float(start_data.b_start[h])
                assigned_count += 1

    return assigned_count
