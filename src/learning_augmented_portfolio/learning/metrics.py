r"""
Evaluation metrics for candidate universe support prediction.
Includes:
  - Recall@B: Macro-average |S_t^* cap A_t(B)| / |S_t^*|
  - False-Negative Rate: FNR@B = 1 - Recall@B
  - Precision@B: |S_t^* cap A_t(B)| / B
  - Candidate Reduction: Reduction@B = 1 - B / N
  - Full-Support Coverage: Coverage@B = (1/T) sum_t 1{S_t^* subseteq A_t(B)}
  - Missed Count: |S_t^* \ A_t(B)|
  - Missed Holding Mass: sum_{i notin A_t(B)} max_h x^*_{i, h|t} (evaluation-only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set
import numpy as np


@dataclass(frozen=True)
class InstanceEvaluationMetrics:
    """Evaluation metrics for a single optimization instance at budget B."""
    instance_id: str
    budget_B: int
    n_assets: int
    teacher_support_size: int
    predicted_candidate_size: int
    recall: float
    precision: float
    false_negative_rate: float
    full_support_covered: bool
    missed_count: int
    missed_holding_mass: float
    candidate_reduction: float


@dataclass(frozen=True)
class CandidateEvaluationResult:
    """Macro-aggregated evaluation metrics across an entire dataset split at budget B."""
    model_name: str
    split: str
    budget_B: int
    candidate_reduction: float
    macro_recall: float
    macro_precision: float
    macro_fnr: float
    full_support_coverage: float
    mean_missed_count: float
    mean_missed_holding_mass: float
    num_instances: int
    is_safe_augmented: bool


def compute_instance_metrics(
    instance_id: str,
    true_support: Set[int],
    predicted_candidates: Set[int],
    n_assets: int,
    budget_B: int,
    optimal_holdings_max: Optional[np.ndarray] = None,
) -> InstanceEvaluationMetrics:
    """Compute support prediction metrics for a single optimization instance."""
    S_star = set(true_support)
    A_B = set(predicted_candidates)
    intersection = S_star.intersection(A_B)
    missed = S_star.difference(A_B)

    card_S = len(S_star)
    card_A = len(A_B)

    # Recall
    recall = len(intersection) / card_S if card_S > 0 else 1.0
    fnr = 1.0 - recall

    # Precision
    precision = len(intersection) / card_A if card_A > 0 else 0.0

    # Coverage: S_t^* subseteq A_t(B) <=> len(missed) == 0
    full_support_covered = (len(missed) == 0)

    # Missed count
    missed_count = len(missed)

    # Missed holding mass (evaluation-only)
    missed_mass = 0.0
    if optimal_holdings_max is not None and len(missed) > 0:
        for m_idx in missed:
            if m_idx < len(optimal_holdings_max):
                missed_mass += float(optimal_holdings_max[m_idx])

    # Candidate reduction
    reduction = 1.0 - (budget_B / n_assets)

    return InstanceEvaluationMetrics(
        instance_id=instance_id,
        budget_B=budget_B,
        n_assets=n_assets,
        teacher_support_size=card_S,
        predicted_candidate_size=card_A,
        recall=recall,
        precision=precision,
        false_negative_rate=fnr,
        full_support_covered=full_support_covered,
        missed_count=missed_count,
        missed_holding_mass=missed_mass,
        candidate_reduction=reduction,
    )


def aggregate_dataset_metrics(
    instance_metrics: List[InstanceEvaluationMetrics],
    model_name: str,
    split: str,
    budget_B: int,
    is_safe_augmented: bool = False,
) -> CandidateEvaluationResult:
    """Aggregate individual instance evaluation metrics into dataset-level macro statistics."""
    T = len(instance_metrics)
    if T == 0:
        return CandidateEvaluationResult(
            model_name=model_name,
            split=split,
            budget_B=budget_B,
            candidate_reduction=0.0,
            macro_recall=0.0,
            macro_precision=0.0,
            macro_fnr=1.0,
            full_support_coverage=0.0,
            mean_missed_count=0.0,
            mean_missed_holding_mass=0.0,
            num_instances=0,
            is_safe_augmented=is_safe_augmented,
        )

    macro_rec = float(np.mean([m.recall for m in instance_metrics]))
    macro_prec = float(np.mean([m.precision for m in instance_metrics]))
    macro_fnr = float(np.mean([m.false_negative_rate for m in instance_metrics]))
    coverage = float(np.mean([1.0 if m.full_support_covered else 0.0 for m in instance_metrics]))
    mean_missed_cnt = float(np.mean([m.missed_count for m in instance_metrics]))
    mean_missed_mass = float(np.mean([m.missed_holding_mass for m in instance_metrics]))
    reduction = instance_metrics[0].candidate_reduction

    return CandidateEvaluationResult(
        model_name=model_name,
        split=split,
        budget_B=budget_B,
        candidate_reduction=reduction,
        macro_recall=macro_rec,
        macro_precision=macro_prec,
        macro_fnr=macro_fnr,
        full_support_coverage=coverage,
        mean_missed_count=mean_missed_cnt,
        mean_missed_holding_mass=mean_missed_mass,
        num_instances=T,
        is_safe_augmented=is_safe_augmented,
    )
