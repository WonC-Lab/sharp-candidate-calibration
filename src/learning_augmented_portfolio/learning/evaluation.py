"""
Evaluation orchestrator across candidate budgets and comparison reporting.
Evaluates predictors, baselines, and ablations across budgets B.
"""

from __future__ import annotations

from typing import List, Optional, Set
import numpy as np
import pandas as pd

from .active_set import select_candidate_universe
from .baselines import BaseRanker
from .dataset import SupervisedSplitDataset
from .metrics import (
    CandidateEvaluationResult,
    compute_instance_metrics,
    aggregate_dataset_metrics,
)


def evaluate_ranker_across_budgets(
    ranker: BaseRanker,
    dataset: SupervisedSplitDataset,
    budgets: List[int],
    use_safe_augmentation: bool = False,
    model_name: Optional[str] = None,
) -> List[CandidateEvaluationResult]:
    """
    Evaluate a ranker/predictor across a grid of candidate universe budgets B.

    Args:
        ranker: BaseRanker object (baseline or SupportPredictorModel).
        dataset: SupervisedSplitDataset (train, validation, or test).
        budgets: List of integer candidate budgets B in [1, n].
        use_safe_augmentation: If True, include mandatory currently held assets M_t.
        model_name: Optional display name for reporting.

    Returns:
        List of CandidateEvaluationResult objects, one for each budget B.
    """
    name = model_name or getattr(ranker, "model_name", ranker.__class__.__name__)
    n_assets = dataset.instances[0].metadata.n_assets if dataset.instances else 0

    results: List[CandidateEvaluationResult] = []

    for B in budgets:
        inst_metrics = []

        for inst_idx, inst in enumerate(dataset.instances):
            # 1. Predict scores for all assets in this instance
            scores = ranker.predict_instance_scores(inst)

            # 2. Extract true teacher support S_t^*
            if inst.labels is not None:
                true_support = set(np.where(inst.labels.y_union == 1)[0].tolist())
            else:
                true_support = set()

            # 3. Mandatory assets for safe augmentation (w_bar_i > 0)
            mandatory = None
            if use_safe_augmentation:
                held_indices = np.where(inst.actual_state.normalized_weights > 1e-6)[0].tolist()
                mandatory = set(held_indices)

            # 4. Form candidate universe A_t(B)
            candidates = set(select_candidate_universe(scores, budget_B=B, mandatory_assets=mandatory))

            # 5. Extract max optimal holding for evaluation-only missed mass
            opt_max = None
            if inst.solution is not None and inst.solution.post_trade_holdings is not None:
                opt_max = np.max(inst.solution.post_trade_holdings, axis=0)

            # 6. Compute instance-level metrics
            m = compute_instance_metrics(
                instance_id=inst.metadata.instance_id,
                true_support=true_support,
                predicted_candidates=candidates,
                n_assets=inst.metadata.n_assets,
                budget_B=B,
                optimal_holdings_max=opt_max,
            )
            inst_metrics.append(m)

        # 7. Macro-aggregate across instances
        agg = aggregate_dataset_metrics(
            instance_metrics=inst_metrics,
            model_name=name,
            split=dataset.split,
            budget_B=B,
            is_safe_augmented=use_safe_augmentation,
        )
        results.append(agg)

    return results


def results_to_dataframe(results: List[CandidateEvaluationResult]) -> pd.DataFrame:
    """Convert a list of CandidateEvaluationResults into a clean pandas DataFrame."""
    records = []
    for r in results:
        records.append({
            "Model": r.model_name,
            "Split": r.split,
            "SafeAugmented": r.is_safe_augmented,
            "B": r.budget_B,
            "CandidateReduction": r.candidate_reduction,
            "Recall@B": r.macro_recall,
            "FNR@B": r.macro_fnr,
            "Precision@B": r.macro_precision,
            "Coverage@B": r.full_support_coverage,
            "MissedCount": r.mean_missed_count,
            "MissedHoldingMass": r.mean_missed_holding_mass,
            "NumInstances": r.num_instances,
        })
    return pd.DataFrame(records)
