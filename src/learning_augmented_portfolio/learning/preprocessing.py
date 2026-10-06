"""
Train-only feature preprocessing and standardization pipeline.
Enforces the invariant: scaler.fit is called strictly on X_train.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeaturePreprocessor:
    """
    Standardizes feature distributions, fitted strictly on training data.
    Safely handles constant features (variance zero) and numerical edge cases.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self.scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        self.is_fitted: bool = False
        self.feature_names: Optional[List[str]] = None

    def fit(self, X_train: np.ndarray, feature_names: Optional[List[str]] = None) -> FeaturePreprocessor:
        """Fit scaler strictly on training features."""
        if len(X_train) == 0:
            raise ValueError("Cannot fit preprocessor on empty X_train.")
        # Replace non-finite values if any
        X_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler.fit(X_clean)
        self.is_fitted = True
        self.feature_names = feature_names
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted training statistics."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePreprocessor must be fitted before transforming.")
        if len(X) == 0:
            return X.copy()
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_clean)
        return np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, X_train: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit on X_train and return transformed X_train."""
        return self.fit(X_train, feature_names).transform(X_train)
