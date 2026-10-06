"""
Active-Set Support Predictor Models and Candidate Universe Selection.
Implements:
  - Top-B Candidate Universe Selector (Raw vs Safe-Augmented)
  - Scikit-learn model wrappers:
      * Model 1: Logistic Regression
      * Model 2: HistGradientBoostingClassifier
      * Model 3: Random Forest
      * Model 4: Small MLP
"""

from __future__ import annotations

from typing import List, Optional, Set, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from ..dataset.instance import OptimizationInstance
from .baselines import BaseRanker
from .features import FeatureGroup, extract_instance_feature_matrix
from .preprocessing import FeaturePreprocessor


def select_candidate_universe(
    scores: np.ndarray,
    budget_B: int,
    mandatory_assets: Optional[Set[int]] = None,
) -> List[int]:
    """
    Select candidate asset indices A_t(B) using predicted scores.

    Args:
        scores: Vector of predicted scores/probabilities p_i for each asset, shape (n,).
        budget_B: Integer candidate budget B in [1, n].
        mandatory_assets: Optional set of asset indices M_t that MUST be included
                          (e.g. currently held assets for safe augmentation).

    Returns:
        List of selected asset indices, sorted in deterministic ascending order.
    """
    n = len(scores)
    B = min(max(1, budget_B), n)

    # 1. Base ranking: order assets by descending predicted score
    ranked_indices = np.argsort(-scores).tolist()

    if mandatory_assets is None or len(mandatory_assets) == 0:
        # Raw Top-B selection
        selected = set(ranked_indices[:B])
    else:
        # Safe augmentation: mandatory assets M_t union Top-(B - |M_t|) remaining assets
        selected = set(mandatory_assets)
        for idx in ranked_indices:
            if len(selected) >= B:
                break
            selected.add(idx)

    return sorted(list(selected))


class SupportPredictorModel(BaseRanker):
    """
    Wrapper around an estimator providing fit, probability scoring, and instance inference.
    """

    def __init__(
        self,
        model_name: str,
        estimator: Any,
        preprocessor: Optional[FeaturePreprocessor] = None,
        feature_groups: Optional[Set[FeatureGroup]] = None,
    ) -> None:
        self.model_name = model_name
        self.estimator = estimator
        self.preprocessor = preprocessor or FeaturePreprocessor()
        self.feature_groups = feature_groups
        self.is_fitted: bool = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> SupportPredictorModel:
        """Fit preprocessor and classifier strictly on training data."""
        X_scaled = self.preprocessor.fit_transform(X_train)
        self.estimator.fit(X_scaled, y_train)
        self.is_fitted = True
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Predict positive class probabilities P(Y=1) for an input feature matrix (M, D)."""
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.model_name}' must be fitted before predict_scores.")
        X_scaled = self.preprocessor.transform(X)
        if hasattr(self.estimator, "predict_proba"):
            probs = self.estimator.predict_proba(X_scaled)
            # If binary classifier with classes [0, 1]
            if probs.shape[1] >= 2:
                return probs[:, 1]
            return probs[:, 0]
        elif hasattr(self.estimator, "decision_function"):
            raw_scores = self.estimator.decision_function(X_scaled)
            # Sigmoid normalization
            return 1.0 / (1.0 + np.exp(-raw_scores))
        else:
            return self.estimator.predict(X_scaled).astype(float)

    def predict_instance_scores(self, instance: OptimizationInstance) -> np.ndarray:
        """Extract features for single instance and predict scores of shape (n,)."""
        X_inst, _ = extract_instance_feature_matrix(instance, self.feature_groups)
        return self.predict_scores(X_inst)


def build_logistic_regression_predictor(
    feature_groups: Optional[Set[FeatureGroup]] = None,
    C: float = 1.0,
    random_state: int = 42,
) -> SupportPredictorModel:
    """Model 1: Interpretable linear baseline with balanced class weights."""
    est = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )
    return SupportPredictorModel("LogisticRegression", est, feature_groups=feature_groups)


def build_hist_gradient_boosting_predictor(
    feature_groups: Optional[Set[FeatureGroup]] = None,
    max_iter: int = 100,
    random_state: int = 42,
) -> SupportPredictorModel:
    """Model 2: Nonlinear tabular boosting baseline with balanced class weights."""
    est = HistGradientBoostingClassifier(
        max_iter=max_iter,
        class_weight="balanced",
        random_state=random_state,
    )
    return SupportPredictorModel("HistGradientBoosting", est, feature_groups=feature_groups)


def build_random_forest_predictor(
    feature_groups: Optional[Set[FeatureGroup]] = None,
    n_estimators: int = 100,
    random_state: int = 42,
) -> SupportPredictorModel:
    """Model 3: Random Forest ensemble baseline."""
    est = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )
    return SupportPredictorModel("RandomForest", est, feature_groups=feature_groups)


def build_mlp_predictor(
    feature_groups: Optional[Set[FeatureGroup]] = None,
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    max_iter: int = 500,
    random_state: int = 42,
) -> SupportPredictorModel:
    """Model 4: Small Multi-Layer Perceptron neural baseline."""
    est = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
    )
    return SupportPredictorModel("SmallMLP", est, feature_groups=feature_groups)
