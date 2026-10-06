"""
Modular feature extraction for asset-level support prediction.
Extracts:
  - Group A: Current State Features
  - Group B: Expected-Return Features
  - Group C: Risk / Covariance Features
  - Group D: Optimization-Constraint Features
  - Group E: Continuous-Relaxation Features
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from ..dataset.instance import OptimizationInstance


class FeatureGroup(str, Enum):
    """Modular feature group categorization."""
    STATE = "state"
    RETURN = "return"
    RISK = "risk"
    CONSTRAINTS = "constraints"
    RELAXATION = "relaxation"


ALL_FEATURE_GROUPS: Set[FeatureGroup] = {
    FeatureGroup.STATE,
    FeatureGroup.RETURN,
    FeatureGroup.RISK,
    FeatureGroup.CONSTRAINTS,
    FeatureGroup.RELAXATION,
}

MARKET_ONLY_GROUPS: Set[FeatureGroup] = {
    FeatureGroup.STATE,
    FeatureGroup.RETURN,
    FeatureGroup.RISK,
}

RELAXATION_ONLY_GROUPS: Set[FeatureGroup] = {
    FeatureGroup.RELAXATION,
}

MARKET_AND_CONSTRAINTS_GROUPS: Set[FeatureGroup] = {
    FeatureGroup.STATE,
    FeatureGroup.RETURN,
    FeatureGroup.RISK,
    FeatureGroup.CONSTRAINTS,
}


def extract_asset_features(
    instance: OptimizationInstance,
    asset_idx: int,
    feature_groups: Optional[Set[FeatureGroup]] = None,
) -> Dict[str, float]:
    """
    Extract a modular dictionary of features for a specific asset at decision time t.

    Args:
        instance: The OptimizationInstance at time t.
        asset_idx: Asset index i in {0, ..., n-1}.
        feature_groups: Set of FeatureGroup enums to include (defaults to all).

    Returns:
        Dict mapping feature_name -> float value.
    """
    groups = feature_groups if feature_groups is not None else ALL_FEATURE_GROUPS
    features: Dict[str, float] = {}

    n = instance.metadata.n_assets
    H = instance.metadata.horizon
    i = asset_idx

    # -------------------------------------------------------------------------
    # Group A: Current State Features
    # -------------------------------------------------------------------------
    if FeatureGroup.STATE in groups:
        w_bar_i = float(instance.actual_state.normalized_weights[i])
        c_bar = float(instance.actual_state.normalized_cash)
        features["state_weight"] = w_bar_i
        features["state_is_held"] = 1.0 if w_bar_i > 1e-6 else 0.0
        features["state_cash"] = c_bar

    # -------------------------------------------------------------------------
    # Group B: Expected-Return Features
    # -------------------------------------------------------------------------
    if FeatureGroup.RETURN in groups:
        mu_ih = instance.market_state.expected_returns[:, i]  # shape (H,)
        for h in range(H):
            features[f"ret_mu_h{h}"] = float(mu_ih[h])
        features["ret_mu_mean"] = float(np.mean(mu_ih))
        features["ret_mu_max"] = float(np.max(mu_ih))
        features["ret_mu_min"] = float(np.min(mu_ih))
        features["ret_mu_spread"] = float(np.max(mu_ih) - np.min(mu_ih))

    # -------------------------------------------------------------------------
    # Group C: Risk / Covariance Features
    # -------------------------------------------------------------------------
    if FeatureGroup.RISK in groups:
        # Asset volatilities across horizon: sqrt(Sigma_{h, ii})
        sigmas_h = [
            float(np.sqrt(max(0.0, instance.market_state.covariance_matrices[h, i, i])))
            for h in range(H)
        ]
        for h, s in enumerate(sigmas_h):
            features[f"risk_vol_h{h}"] = s
        features["risk_vol_mean"] = float(np.mean(sigmas_h))
        features["risk_vol_max"] = float(np.max(sigmas_h))

        # First-period covariance and correlation summary with market universe
        sigma_0 = instance.market_state.covariance_matrices[0]
        row_i = sigma_0[i, :]
        features["risk_cov_row_abs_sum"] = float(np.sum(np.abs(row_i)))
        other_mask = np.ones(n, dtype=bool)
        other_mask[i] = False
        if n > 1:
            features["risk_cov_mean_other"] = float(np.mean(row_i[other_mask]))
            # Correlations with other assets at h=0
            denom = sigmas_h[0] * np.sqrt(np.maximum(1e-8, np.diag(sigma_0)[other_mask]))
            corrs = row_i[other_mask] / denom
            features["risk_corr_mean_other"] = float(np.mean(corrs))
            features["risk_corr_max_abs_other"] = float(np.max(np.abs(corrs)))
        else:
            features["risk_cov_mean_other"] = 0.0
            features["risk_corr_mean_other"] = 0.0
            features["risk_corr_max_abs_other"] = 0.0

    # -------------------------------------------------------------------------
    # Group D: Optimization-Constraint Features
    # -------------------------------------------------------------------------
    if FeatureGroup.CONSTRAINTS in groups:
        # Per-asset constraints
        features["constr_weight_upper_h0"] = float(instance.params.weight_upper_limits[0, i])
        features["constr_liquidity_h0"] = float(instance.params.liquidity_limits[0, i])
        features["constr_prop_cost_h0"] = float(instance.params.prop_cost_coeffs[0, i])

        if instance.params.fixed_cost_coeffs is not None:
            features["constr_fixed_cost_h0"] = float(instance.params.fixed_cost_coeffs[0, i])
        else:
            features["constr_fixed_cost_h0"] = 0.0

        if instance.params.min_holding_weights is not None:
            features["constr_min_holding_h0"] = float(instance.params.min_holding_weights[0, i])
        else:
            features["constr_min_holding_h0"] = 0.0

        if instance.params.min_trade_sizes is not None:
            features["constr_min_trade_h0"] = float(instance.params.min_trade_sizes[0, i])
        else:
            features["constr_min_trade_h0"] = 0.0

        # Global problem parameters
        features["constr_risk_aversion"] = float(instance.params.risk_aversion)
        features["constr_discount_factor"] = float(instance.params.discount_factor)
        features["constr_max_turnover_h0"] = float(instance.params.max_turnovers[0])
        features["constr_min_cash_h0"] = float(instance.params.min_cash[0])
        features["constr_cardinality_h0"] = float(
            instance.params.cardinality_limits[0] if instance.params.cardinality_limits is not None else n
        )
        features["constr_n_assets"] = float(n)
        features["constr_horizon"] = float(H)

    # -------------------------------------------------------------------------
    # Group E: Continuous-Relaxation Features
    # -------------------------------------------------------------------------
    if FeatureGroup.RELAXATION in groups:
        if instance.relaxation is not None and instance.relaxation.is_feasible:
            rx = instance.relaxation.relaxed_x[:, i] if instance.relaxation.relaxed_x is not None else np.zeros(H)
            ry = instance.relaxation.relaxed_y[:, i] if instance.relaxation.relaxed_y is not None else np.zeros(H)
            rz = instance.relaxation.relaxed_z[:, i] if instance.relaxation.relaxed_z is not None else np.zeros(H)

            for h in range(H):
                features[f"relax_x_h{h}"] = float(rx[h])
                features[f"relax_y_h{h}"] = float(ry[h])
                features[f"relax_z_h{h}"] = float(rz[h])

            features["relax_y_max"] = float(np.max(ry))
            features["relax_x_max"] = float(np.max(rx))
            features["relax_x_sum"] = float(np.sum(rx))
            features["relax_z_max"] = float(np.max(rz))

            if instance.relaxation.reduced_costs_x is not None:
                rc_i = instance.relaxation.reduced_costs_x[:, i]
                features["relax_rc_mean"] = float(np.mean(rc_i))
                features["relax_rc_h0"] = float(rc_i[0])
            else:
                features["relax_rc_mean"] = 0.0
                features["relax_rc_h0"] = 0.0
        else:
            # Fallback zeros if relaxation is absent
            for h in range(H):
                features[f"relax_x_h{h}"] = 0.0
                features[f"relax_y_h{h}"] = 0.0
                features[f"relax_z_h{h}"] = 0.0
            features["relax_y_max"] = 0.0
            features["relax_x_max"] = 0.0
            features["relax_x_sum"] = 0.0
            features["relax_z_max"] = 0.0
            features["relax_rc_mean"] = 0.0
            features["relax_rc_h0"] = 0.0

    return features


def extract_instance_feature_matrix(
    instance: OptimizationInstance,
    feature_groups: Optional[Set[FeatureGroup]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract feature matrix of shape (n, D) for all assets in the given instance.

    Returns:
        Tuple of (X_matrix, feature_names_list).
    """
    n = instance.metadata.n_assets
    row_0 = extract_asset_features(instance, 0, feature_groups)
    feature_names = sorted(row_0.keys())
    D = len(feature_names)

    X = np.zeros((n, D), dtype=float)
    X[0, :] = [row_0[k] for k in feature_names]

    for i in range(1, n):
        row_i = extract_asset_features(instance, i, feature_groups)
        X[i, :] = [row_i[k] for k in feature_names]

    return X, feature_names
