"""
Machine learning package for active-set / candidate support prediction.
"""

from .features import (
    FeatureGroup,
    ALL_FEATURE_GROUPS,
    MARKET_ONLY_GROUPS,
    RELAXATION_ONLY_GROUPS,
    MARKET_AND_CONSTRAINTS_GROUPS,
    extract_asset_features,
    extract_instance_feature_matrix,
)
from .dataset import (
    SupervisedSplitDataset,
    build_supervised_datasets_by_split,
)
from .preprocessing import FeaturePreprocessor
from .baselines import (
    BaseRanker,
    RandomRankingBaseline,
    CurrentHoldingBaseline,
    ExpectedReturnBaseline,
    ContinuousRelaxationYBaseline,
    ContinuousRelaxationXBaseline,
)
from .active_set import (
    select_candidate_universe,
    SupportPredictorModel,
    build_logistic_regression_predictor,
    build_hist_gradient_boosting_predictor,
    build_random_forest_predictor,
    build_mlp_predictor,
)
from .candidate_set import construct_candidate_universe
from .metrics import (
    InstanceEvaluationMetrics,
    CandidateEvaluationResult,
    compute_instance_metrics,
    aggregate_dataset_metrics,
)
from .evaluation import (
    evaluate_ranker_across_budgets,
    results_to_dataframe,
)

__all__ = [
    "FeatureGroup",
    "ALL_FEATURE_GROUPS",
    "MARKET_ONLY_GROUPS",
    "RELAXATION_ONLY_GROUPS",
    "MARKET_AND_CONSTRAINTS_GROUPS",
    "extract_asset_features",
    "extract_instance_feature_matrix",
    "SupervisedSplitDataset",
    "build_supervised_datasets_by_split",
    "FeaturePreprocessor",
    "BaseRanker",
    "RandomRankingBaseline",
    "CurrentHoldingBaseline",
    "ExpectedReturnBaseline",
    "ContinuousRelaxationYBaseline",
    "ContinuousRelaxationXBaseline",
    "select_candidate_universe",
    "construct_candidate_universe",
    "SupportPredictorModel",
    "build_logistic_regression_predictor",
    "build_hist_gradient_boosting_predictor",
    "build_random_forest_predictor",
    "build_mlp_predictor",
    "InstanceEvaluationMetrics",
    "CandidateEvaluationResult",
    "compute_instance_metrics",
    "aggregate_dataset_metrics",
    "evaluate_ranker_across_budgets",
    "results_to_dataframe",
]
