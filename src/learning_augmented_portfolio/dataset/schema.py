"""
Dataset schema definitions, array dimensions, versioning constants, and validation rules.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

DATASET_SCHEMA_VERSION = "1.0.0"

MANIFEST_COLUMNS: List[str] = [
    "instance_id",
    "calendar_date",
    "time_index",
    "split",
    "filename",
    "universe_id",
    "n_assets",
    "horizon",
    "solver_status",
    "label_quality",
    "is_trainable_label",
    "objective",
    "best_bound",
    "mip_gap",
    "runtime",
    "work",
    "node_count",
    "sol_count",
    "used_fallback",
    "config_hash",
    "dataset_version",
]

# Required dense array fields for an optimization instance
INSTANCE_ARRAY_KEYS: List[str] = [
    # Actual state at decision time t (in V_t numeraire)
    "normalized_weights",      # (n,)
    "initial_holdings",        # (n,)
    # Forecast inputs across horizon H
    "expected_returns",        # (H, n)
    "covariance_matrices",     # (H, n, n)
    "gross_returns",           # (H-1, n) or empty if H=1
    "risk_free_rates",         # (H,)
    "max_turnovers",           # (H,)
    "min_cash",                # (H,)
    "weight_upper_limits",     # (H, n)
    "liquidity_limits",        # (H, n)
    "prop_cost_coeffs",        # (H, n)
    "dynamic_big_m",           # (H, n)
]

# Full teacher solution arrays across horizon H
SOLUTION_ARRAY_KEYS: List[str] = [
    "pre_trade_holdings",      # (H, n)
    "post_trade_holdings",     # (H, n)
    "trades",                  # (H, n)
    "absolute_trades",         # (H, n)
    "pre_trade_cash",          # (H,)
    "post_trade_cash",         # (H,)
    "transaction_costs",       # (H,)
    "holding_indicators",      # (H, n)
    "trade_indicators",        # (H, n)
]

# Explicit supervised learning labels
LABEL_ARRAY_KEYS: List[str] = [
    "y_first",                 # (n,)  First-period holding indicator: y^*_{i, 0|t}
    "y_union",                 # (n,)  Horizon-union holding indicator: max_h y^*_{i, h|t}
    "z_first",                 # (n,)  First-period trade indicator: z^*_{i, 0|t}
    "z_union",                 # (n,)  Horizon-union trade indicator: max_h z^*_{i, h|t}
]

# Continuous relaxation arrays
RELAXATION_ARRAY_KEYS: List[str] = [
    "relaxed_x",               # (H, n)
    "relaxed_y",               # (H, n)
    "relaxed_z",               # (H, n)
]


def validate_array_shapes(
    arrays: Dict[str, np.ndarray],
    n_assets: int,
    horizon: int,
) -> None:
    """
    Validate that all serialized arrays conform exactly to the mathematical schema.
    """
    n = n_assets
    H = horizon

    expected_shapes: Dict[str, Tuple[int, ...]] = {
        "normalized_weights": (n,),
        "initial_holdings": (n,),
        "expected_returns": (H, n),
        "covariance_matrices": (H, n, n),
        "risk_free_rates": (H,),
        "max_turnovers": (H,),
        "min_cash": (H,),
        "weight_upper_limits": (H, n),
        "liquidity_limits": (H, n),
        "prop_cost_coeffs": (H, n),
        "dynamic_big_m": (H, n),
        # Labels
        "y_first": (n,),
        "y_union": (n,),
        "z_first": (n,),
        "z_union": (n,),
    }

    if H > 1 and "gross_returns" in arrays and arrays["gross_returns"] is not None:
        expected_shapes["gross_returns"] = (H - 1, n)

    for key, shape in expected_shapes.items():
        if key in arrays and arrays[key] is not None:
            arr = arrays[key]
            if arr.shape != shape:
                raise ValueError(
                    f"Array '{key}' shape {arr.shape} does not match expected schema shape {shape}"
                )
