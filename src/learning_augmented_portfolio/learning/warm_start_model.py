"""
Supervised Learning Models for Multi-Period Discrete MIP Warm Starts.
Predicts per-period binary holding decisions y_{i,h} and trade decisions z_{i,h}.
Strictly isolates Market-Only features from Relaxation-Assisted features.
Guarantees no leakage of teacher labels into inference features.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..dataset.instance import OptimizationInstance
from .warm_start import MIPStartData, WarmStartType


class HorizonWarmStartPredictor:
    """
    Learns to predict per-period discrete decisions y^*_{i,h} and z^*_{i,h}
    across the multi-period optimization horizon.
    """

    def __init__(
        self,
        include_relaxation: bool = False,
        random_state: int = 42,
    ) -> None:
        self.include_relaxation = include_relaxation
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model_y = LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=500)
        self.model_z = LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=500)
        self.is_fitted = False

    def _extract_instance_features_and_labels(
        self,
        inst: OptimizationInstance,
        is_training: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract tabular features for all (asset i, horizon h) pairs in an instance.
        """
        H = inst.params.horizon
        N = inst.market_state.n_assets
        feature_rows: List[np.ndarray] = []
        labels_y: List[int] = []
        labels_z: List[int] = []

        # Current state features
        curr_w = inst.actual_state.normalized_weights
        c_prop = inst.params.prop_cost_coeffs[0]
        u_lim = inst.params.weight_upper_limits[0]

        for h in range(H):
            mu_h = inst.market_state.expected_returns[h]
            cov_h = inst.market_state.covariance_matrices[h]
            var_h = np.diag(cov_h)
            std_h = np.sqrt(np.maximum(1e-8, var_h))
            sharpe_h = mu_h / std_h

            rel_y = inst.relaxation.relaxed_y[h] if (self.include_relaxation and inst.relaxation and inst.relaxation.relaxed_y is not None) else None
            rel_z = inst.relaxation.relaxed_z[h] if (self.include_relaxation and inst.relaxation and inst.relaxation.relaxed_z is not None) else None

            for i in range(N):
                row = [
                    float(curr_w[i]),
                    float(curr_w[i] > 1e-6),
                    float(mu_h[i]),
                    float(std_h[i]),
                    float(sharpe_h[i]),
                    float(h) / float(max(1, H - 1)),
                    float(c_prop[i]),
                    float(u_lim[i]),
                ]
                if self.include_relaxation:
                    val_ry = float(rel_y[i]) if rel_y is not None else 0.0
                    val_rz = float(rel_z[i]) if rel_z is not None else 0.0
                    row.extend([val_ry, val_rz])

                feature_rows.append(row)

                if is_training and inst.solution is not None:
                    y_star = inst.solution.holding_indicators[h, i]
                    z_star = inst.solution.trade_indicators[h, i]
                    labels_y.append(int(y_star))
                    labels_z.append(int(z_star))

        X = np.array(feature_rows, dtype=float)
        y_arr = np.array(labels_y, dtype=int) if is_training else None
        z_arr = np.array(labels_z, dtype=int) if is_training else None
        return X, y_arr, z_arr

    def fit(self, instances: List[OptimizationInstance]) -> None:
        """Fit supervised classifiers on historical optimization instances."""
        X_all: List[np.ndarray] = []
        y_all: List[np.ndarray] = []
        z_all: List[np.ndarray] = []

        for inst in instances:
            if inst.solution is None or not inst.solution.is_feasible:
                continue
            X_inst, y_inst, z_inst = self._extract_instance_features_and_labels(inst, is_training=True)
            X_all.append(X_inst)
            y_all.append(y_inst)
            z_all.append(z_inst)

        X_mat = np.vstack(X_all)
        y_vec = np.concatenate(y_all)
        z_vec = np.concatenate(z_all)

        X_scaled = self.scaler.fit_transform(X_mat)
        self.model_y.fit(X_scaled, y_vec)
        self.model_z.fit(X_scaled, z_vec)
        self.is_fitted = True

    def predict_warm_start(
        self,
        inst: OptimizationInstance,
        threshold_y: float = 0.5,
        threshold_z: float = 0.5,
    ) -> Tuple[MIPStartData, np.ndarray, np.ndarray]:
        """
        Predict discrete MIP start for a new optimization instance.

        Returns:
            Tuple of (MIPStartData, probs_y, probs_z).
        """
        t0 = time.perf_counter()
        if not self.is_fitted:
            raise RuntimeError("HorizonWarmStartPredictor must be fitted before prediction.")

        H = inst.params.horizon
        N = inst.market_state.n_assets

        X_inst, _, _ = self._extract_instance_features_and_labels(inst, is_training=False)
        X_scaled = self.scaler.transform(X_inst)

        probs_y_flat = self.model_y.predict_proba(X_scaled)[:, 1]
        probs_z_flat = self.model_z.predict_proba(X_scaled)[:, 1]

        probs_y = probs_y_flat.reshape((H, N))
        probs_z = probs_z_flat.reshape((H, N))

        pred_y = (probs_y >= threshold_y).astype(int)
        pred_z = (probs_z >= threshold_z).astype(int)

        t_inf = time.perf_counter() - t0

        start_data = MIPStartData(
            start_type=WarmStartType.ML_DISCRETE,
            y_start=pred_y,
            z_start=pred_z,
            is_complete=False,
            construction_time=t_inf,
        )
        return start_data, probs_y, probs_z
