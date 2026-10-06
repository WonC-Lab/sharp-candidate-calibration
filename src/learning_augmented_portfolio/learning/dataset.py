"""
Asset-level supervised dataset generation with strict chronological split enforcement.
Guarantees: Instances -> Chronological Split -> Asset-Level Rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import numpy as np

from ..dataset.instance import OptimizationInstance
from .features import FeatureGroup, extract_instance_feature_matrix


@dataclass
class SupervisedSplitDataset:
    """
    Asset-level tabular dataset for a single chronological partition (train, validation, or test).
    
    Attributes:
        split: Partition identifier ('train', 'validation', 'test').
        X: Feature matrix of shape (M, D) where M = num_instances * n_assets.
        y_union: Binary target vector Y^{union}_{i,t} of shape (M,).
        y_first: Binary secondary target vector Y^{first}_{i,t} of shape (M,).
        instance_ids: List of unique instance IDs in this partition.
        instance_indices: Vector of shape (M,) mapping row -> instance index.
        asset_indices: Vector of shape (M,) mapping row -> asset index i in {0..n-1}.
        current_weights: Vector of shape (M,) containing pre-trade weight w_bar_{i,t}.
        max_optimal_holdings: Vector of shape (M,) containing max_h x^*_{i,h|t} (evaluation-only).
        feature_names: List of D feature names.
        instances: Original list of OptimizationInstance objects for evaluation.
    """
    split: str
    X: np.ndarray
    y_union: np.ndarray
    y_first: np.ndarray
    instance_ids: List[str]
    instance_indices: np.ndarray
    asset_indices: np.ndarray
    current_weights: np.ndarray
    max_optimal_holdings: np.ndarray
    feature_names: List[str]
    instances: List[OptimizationInstance]

    @property
    def num_samples(self) -> int:
        return len(self.y_union)

    @property
    def num_instances(self) -> int:
        return len(self.instances)

    @property
    def positive_ratio(self) -> float:
        if self.num_samples == 0:
            return 0.0
        return float(np.mean(self.y_union))


def build_supervised_datasets_by_split(
    instances: List[OptimizationInstance],
    feature_groups: Optional[Set[FeatureGroup]] = None,
) -> Dict[str, SupervisedSplitDataset]:
    """
    Partition instances chronologically first, then flatten each partition into asset-level rows.
    Guarantees no instance or asset appears across multiple splits.

    Args:
        instances: Sequence of chronologically generated OptimizationInstances.
        feature_groups: Optional feature groups to include.

    Returns:
        Dict mapping split_name ('train', 'validation', 'test') -> SupervisedSplitDataset.
    """
    # 1. Group instances strictly by split first
    split_buckets: Dict[str, List[OptimizationInstance]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for inst in instances:
        sp = inst.metadata.split.lower()
        if sp in split_buckets:
            split_buckets[sp].append(inst)
        else:
            split_buckets.setdefault(sp, []).append(inst)

    datasets: Dict[str, SupervisedSplitDataset] = {}

    for split_name, split_insts in split_buckets.items():
        if not split_insts:
            continue

        X_rows: List[np.ndarray] = []
        y_union_rows: List[int] = []
        y_first_rows: List[int] = []
        inst_indices: List[int] = []
        asset_indices: List[int] = []
        curr_weights: List[float] = []
        max_opt_holdings: List[float] = []
        inst_ids: List[str] = []
        feature_names: List[str] = []

        for inst_idx, inst in enumerate(split_insts):
            inst_ids.append(inst.metadata.instance_id)
            n = inst.metadata.n_assets

            # Extract (n, D) matrix for this instance
            X_inst, f_names = extract_instance_feature_matrix(inst, feature_groups)
            if not feature_names:
                feature_names = f_names

            X_rows.append(X_inst)

            # Targets
            if inst.labels is not None:
                y_union_rows.extend(inst.labels.y_union.tolist())
                y_first_rows.extend(inst.labels.y_first.tolist())
            else:
                y_union_rows.extend([0] * n)
                y_first_rows.extend([0] * n)

            # Bookkeeping
            inst_indices.extend([inst_idx] * n)
            asset_indices.extend(list(range(n)))
            curr_weights.extend(inst.actual_state.normalized_weights.tolist())

            # Max optimal holding (evaluation-only)
            if inst.solution is not None and inst.solution.post_trade_holdings is not None:
                max_h_holding = np.max(inst.solution.post_trade_holdings, axis=0)
                max_opt_holdings.extend(max_h_holding.tolist())
            else:
                max_opt_holdings.extend([0.0] * n)

        X_mat = np.vstack(X_rows) if X_rows else np.empty((0, 0))
        y_union_arr = np.array(y_union_rows, dtype=int)
        y_first_arr = np.array(y_first_rows, dtype=int)

        datasets[split_name] = SupervisedSplitDataset(
            split=split_name,
            X=X_mat,
            y_union=y_union_arr,
            y_first=y_first_arr,
            instance_ids=inst_ids,
            instance_indices=np.array(inst_indices, dtype=int),
            asset_indices=np.array(asset_indices, dtype=int),
            current_weights=np.array(curr_weights, dtype=float),
            max_optimal_holdings=np.array(max_opt_holdings, dtype=float),
            feature_names=feature_names,
            instances=split_insts,
        )

    return datasets
