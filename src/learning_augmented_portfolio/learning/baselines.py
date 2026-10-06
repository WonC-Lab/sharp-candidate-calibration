"""
Heuristic and Optimization-Relaxation Ranking Baselines for candidate universe selection.
Includes:
  - Baseline 0: Random Ranking
  - Baseline 1: Current-Holding Priority
  - Baseline 2: Expected-Return Ranking
  - Baseline 3: Continuous-Relaxation Ranking (Relaxation-Y and Relaxation-X)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from ..dataset.instance import OptimizationInstance


class BaseRanker(ABC):
    """Abstract base interface for asset support rankers / predictors."""

    @abstractmethod
    def predict_instance_scores(self, instance: OptimizationInstance) -> np.ndarray:
        """
        Compute predicted scores p_i in R for each asset i in {0, ..., n-1} for a single instance.
        Higher scores indicate greater predicted probability of being in optimal support S_t^*.
        """
        pass

    def predict_dataset_scores(self, instances: list[OptimizationInstance]) -> np.ndarray:
        """Compute concatenated score vector of shape (sum_t n_t,) across instances."""
        scores_list = [self.predict_instance_scores(inst) for inst in instances]
        return np.concatenate(scores_list) if scores_list else np.empty(0)


class RandomRankingBaseline(BaseRanker):
    """Baseline 0: Uniform random score assignment under a deterministic seed."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def predict_instance_scores(self, instance: OptimizationInstance) -> np.ndarray:
        n = instance.metadata.n_assets
        return self.rng.uniform(low=0.0, high=1.0, size=n)


class CurrentHoldingBaseline(BaseRanker):
    """Baseline 1: Prioritize currently held assets (w_bar_i > 0) with tie-breaker by weight."""

    def predict_instance_scores(self, instance: OptimizationInstance) -> np.ndarray:
        return instance.actual_state.normalized_weights.copy()


class ExpectedReturnBaseline(BaseRanker):
    """Baseline 2: Rank assets by maximum expected return across the planning horizon."""

    def predict_instance_scores(self, instance: OptimizationInstance) -> np.ndarray:
        # max_h mu_{i, h}
        return np.max(instance.market_state.expected_returns, axis=0)


class ContinuousRelaxationYBaseline(BaseRanker):
    """
    Baseline 3A: Critical OR Baseline.
    Ranks assets by continuous relaxation holding indicator: max_h y^{relax}_{i, h}.
    """

    def predict_instance_scores(self, instance: OptimizationInstance) -> np.ndarray:
        n = instance.metadata.n_assets
        if instance.relaxation is not None and instance.relaxation.relaxed_y is not None:
            return np.max(instance.relaxation.relaxed_y, axis=0)
        return np.zeros(n)


class ContinuousRelaxationXBaseline(BaseRanker):
    """
    Baseline 3B: Continuous relaxation position ranking: max_h x^{relax}_{i, h}.
    """

    def predict_instance_scores(self, instance: OptimizationInstance) -> np.ndarray:
        n = instance.metadata.n_assets
        if instance.relaxation is not None and instance.relaxation.relaxed_x is not None:
            return np.max(instance.relaxation.relaxed_x, axis=0)
        return np.zeros(n)
