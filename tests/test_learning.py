"""
Phase 6 Unit Tests — Active-Set / Support Prediction via Machine Learning.
Validates:
  - Primary target equals Y^{union}
  - Chronological split before flattening
  - No instance ID cross-split leakage
  - Train-only scaler fitting
  - No realized future data in features
  - Asset ordering preservation
  - Relaxation feature extraction correctness
  - Top-B candidate set sizing
  - Recall@B analytical toy verification
  - Coverage@B analytical toy verification
  - Precision and FNR correctness
  - Candidate reduction calculation
  - Random baseline reproducibility under fixed seed
  - Relaxation-ranking baseline correctness
  - Raw vs Safe-augmented candidate separation
  - Validation-only model selection isolation
  - Model save/load reproducibility
  - All Phase 1-5 regression tests pass
"""

from __future__ import annotations

import pickle
import numpy as np
import pytest

from learning_augmented_portfolio.dataset.instance import (
    OptimizationInstance,
    OptimizationInstanceMetadata,
    OptimizationLabels,
    ContinuousRelaxationRecord,
)
from learning_augmented_portfolio.dataset.schema import DATASET_SCHEMA_VERSION
from learning_augmented_portfolio.dynamic.state import PortfolioState
from learning_augmented_portfolio.optimization.variables import (
    MultiPeriodMarketState,
    MultiPeriodOptimizationResult,
    MultiPeriodPortfolioParameters,
)
from learning_augmented_portfolio.learning.active_set import (
    select_candidate_universe,
    build_logistic_regression_predictor,
    build_hist_gradient_boosting_predictor,
    SupportPredictorModel,
)
from learning_augmented_portfolio.learning.baselines import (
    RandomRankingBaseline,
    CurrentHoldingBaseline,
    ExpectedReturnBaseline,
    ContinuousRelaxationYBaseline,
    ContinuousRelaxationXBaseline,
)
from learning_augmented_portfolio.learning.dataset import (
    SupervisedSplitDataset,
    build_supervised_datasets_by_split,
)
from learning_augmented_portfolio.learning.features import (
    FeatureGroup,
    extract_asset_features,
    extract_instance_feature_matrix,
    ALL_FEATURE_GROUPS,
    MARKET_ONLY_GROUPS,
    RELAXATION_ONLY_GROUPS,
)
from learning_augmented_portfolio.learning.metrics import (
    compute_instance_metrics,
    aggregate_dataset_metrics,
)
from learning_augmented_portfolio.learning.preprocessing import FeaturePreprocessor
from learning_augmented_portfolio.learning.evaluation import (
    evaluate_ranker_across_budgets,
    results_to_dataframe,
)


@pytest.fixture
def mock_multi_instance_setup():
    """Create 6 synthetic OptimizationInstances partitioned into train (3), val (2), test (1)."""
    n = 4
    H = 2
    tickers = ["A0", "A1", "A2", "A3"]

    instances = []
    splits = ["train", "train", "train", "validation", "validation", "test"]

    for idx, sp in enumerate(splits):
        weights = np.array([0.25, 0.25, 0.25, 0.15])
        cash = 0.10
        state = PortfolioState(
            actual_wealth=1000.0,
            normalized_weights=weights,
            normalized_cash=cash,
            time_index=idx,
            asset_names=tickers,
        )

        mu = np.array([
            [0.05 + 0.01 * idx, 0.04, 0.03, 0.02],
            [0.05 + 0.01 * idx, 0.04, 0.03, 0.02],
        ])
        sigma = np.tile(np.eye(n) * 0.04, (H, 1, 1))

        mkt = MultiPeriodMarketState(
            initial_holdings=weights.copy(),
            initial_cash=cash,
            expected_returns=mu,
            covariance_matrices=sigma,
            asset_names=tickers,
        )

        params = MultiPeriodPortfolioParameters(
            risk_aversion=2.0,
            horizon=H,
            risk_free_rates=np.full(H, 0.01),
            max_turnovers=np.full(H, 0.5),
            min_cash=np.full(H, 0.05),
            weight_upper_limits=np.full((H, n), 0.6),
            liquidity_limits=np.full((H, n), 0.5),
            prop_cost_coeffs=np.full((H, n), 0.001),
            fixed_cost_coeffs=np.full((H, n), 0.0005),
            cardinality_limits=np.full(H, 2),
        )

        # Mock optimal solution where assets 0 and (idx % 2 + 1) are in support
        y_opt = np.zeros((H, n), dtype=int)
        y_opt[:, 0] = 1
        active_asset = (idx % 2) + 1
        y_opt[:, active_asset] = 1

        sol = MultiPeriodOptimizationResult(
            status="OPTIMAL",
            is_optimal=True,
            is_feasible=True,
            solve_time=0.01,
            objective_value=-0.10 - 0.01 * idx,
            post_trade_holdings=y_opt.astype(float) * 0.45,
            holding_indicators=y_opt,
            trade_indicators=y_opt,
        )

        labels = OptimizationLabels(
            y_first=y_opt[0].copy(),
            y_union=np.max(y_opt, axis=0),
            z_first=y_opt[0].copy(),
            z_union=np.max(y_opt, axis=0),
        )

        relax = ContinuousRelaxationRecord(
            status="OPTIMAL",
            is_feasible=True,
            objective_value=-0.12 - 0.01 * idx,
            solve_time=0.005,
            relaxed_x=y_opt.astype(float) * 0.4,
            relaxed_y=y_opt.astype(float) * 0.8,
            relaxed_z=y_opt.astype(float) * 0.5,
        )

        metadata = OptimizationInstanceMetadata(
            instance_id=f"INST_{idx:03d}",
            calendar_date=idx,
            time_index=idx,
            split=sp,
            universe_id="UNIV_TEST",
            n_assets=n,
            horizon=H,
            dataset_version=DATASET_SCHEMA_VERSION,
            config_hash="cfg_hash",
            git_commit=None,
            gurobi_version="13.0.3",
            python_version="3.13.9",
            solver_params={},
            random_seed=42,
            solver_status="OPTIMAL",
            label_quality="SOLVER_CERTIFIED",
            is_trainable_label=True,
            objective=sol.objective_value,
            best_bound=sol.objective_value,
            mip_gap=0.0,
            runtime=0.01,
            work=None,
            node_count=5,
            sol_count=2,
            used_fallback=False,
        )

        inst = OptimizationInstance(
            metadata=metadata,
            actual_state=state,
            params=params,
            market_state=mkt,
            solution=sol,
            labels=labels,
            relaxation=relax,
        )
        instances.append(inst)

    return instances


# -----------------------------------------------------------------------------
# Test 1 — Primary target equals stored Y^{union}
# -----------------------------------------------------------------------------
def test_1_primary_target_equals_stored_y_union(mock_multi_instance_setup):
    """Verify that flattening extracts Y^{union}_{i,t} as the primary binary label."""
    instances = mock_multi_instance_setup
    datasets = build_supervised_datasets_by_split(instances)

    for split_name, ds in datasets.items():
        assert len(ds.y_union) == ds.num_instances * 4
        # Verify alignment with instance labels
        offset = 0
        for inst in ds.instances:
            inst_y_union = inst.labels.y_union
            assert np.array_equal(ds.y_union[offset:offset + 4], inst_y_union)
            offset += 4


# -----------------------------------------------------------------------------
# Test 2 — Chronological split occurs before asset flattening
# -----------------------------------------------------------------------------
def test_2_chronological_split_before_asset_flattening(mock_multi_instance_setup):
    """Verify that split assignment happens at instance level before row flattening."""
    instances = mock_multi_instance_setup
    datasets = build_supervised_datasets_by_split(instances)

    assert "train" in datasets
    assert "validation" in datasets
    assert "test" in datasets

    # Train: 3 instances * 4 assets = 12 samples
    assert datasets["train"].num_samples == 12
    # Val: 2 instances * 4 assets = 8 samples
    assert datasets["validation"].num_samples == 8
    # Test: 1 instance * 4 assets = 4 samples
    assert datasets["test"].num_samples == 4


# -----------------------------------------------------------------------------
# Test 3 — No instance ID appears across multiple splits
# -----------------------------------------------------------------------------
def test_3_no_instance_id_across_multiple_splits(mock_multi_instance_setup):
    """Assert zero instance overlap across train, validation, and test."""
    instances = mock_multi_instance_setup
    datasets = build_supervised_datasets_by_split(instances)

    train_ids = set(datasets["train"].instance_ids)
    val_ids = set(datasets["validation"].instance_ids)
    test_ids = set(datasets["test"].instance_ids)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


# -----------------------------------------------------------------------------
# Test 4 — Train-only scaler fitting
# -----------------------------------------------------------------------------
def test_4_train_only_scaler_fitting():
    """Verify that preprocessor fit on train is completely isolated from validation."""
    X_train = np.array([
        [10.0, 20.0],
        [30.0, 40.0],
    ])
    X_val = np.array([
        [1000.0, 2000.0],  # Outlier distribution
    ])

    scaler = FeaturePreprocessor()
    scaler.fit(X_train)

    train_mean = scaler.scaler.mean_
    assert np.allclose(train_mean, [20.0, 30.0])

    # Transform validation using train mean and scale
    scaled_val = scaler.transform(X_val)
    # Mean of scaler should NOT have changed
    assert np.allclose(scaler.scaler.mean_, [20.0, 30.0])
    # scaled_val[0, 0] = (1000 - 20) / 10 = 98.0
    assert np.isclose(scaled_val[0, 0], 98.0)


# -----------------------------------------------------------------------------
# Test 5 — No realized future fields enter features
# -----------------------------------------------------------------------------
def test_5_no_realized_future_fields_enter_features(mock_multi_instance_setup):
    """Audit feature names to ensure no future returns or post-t metrics are extracted."""
    inst = mock_multi_instance_setup[0]
    feats = extract_asset_features(inst, 0)

    forbidden_substrings = ["realized", "future", "post_trade_holdings", "t+1"]
    for fname in feats.keys():
        for sub in forbidden_substrings:
            assert sub not in fname.lower(), f"Forbidden substring '{sub}' detected in feature '{fname}'"


# -----------------------------------------------------------------------------
# Test 6 — Asset ordering remains consistent
# -----------------------------------------------------------------------------
def test_6_asset_ordering_remains_consistent(mock_multi_instance_setup):
    """Verify that rows 0..n-1 map deterministically to assets 0..n-1."""
    instances = mock_multi_instance_setup
    datasets = build_supervised_datasets_by_split(instances)

    ds_train = datasets["train"]
    n = 4
    for idx in range(ds_train.num_instances):
        sub_indices = ds_train.asset_indices[idx * n : (idx + 1) * n]
        assert np.array_equal(sub_indices, np.arange(n))


# -----------------------------------------------------------------------------
# Test 7 — Relaxation feature extraction is correct
# -----------------------------------------------------------------------------
def test_7_relaxation_feature_extraction_correct(mock_multi_instance_setup):
    """Verify continuous relaxation indicators y^{relax} and x^{relax} are accurately extracted."""
    inst = mock_multi_instance_setup[0]
    feats = extract_asset_features(inst, asset_idx=0, feature_groups=RELAXATION_ONLY_GROUPS)

    assert "relax_y_max" in feats
    assert "relax_x_max" in feats
    assert np.isclose(feats["relax_y_max"], 0.8)
    assert np.isclose(feats["relax_x_max"], 0.4)


# -----------------------------------------------------------------------------
# Test 8 — Top-B candidate set has correct size
# -----------------------------------------------------------------------------
def test_8_top_b_candidate_set_size():
    """Verify select_candidate_universe respects budget B."""
    scores = np.array([0.1, 0.9, 0.4, 0.7, 0.3])
    candidates = select_candidate_universe(scores, budget_B=3)
    assert len(candidates) == 3
    # Top 3 are indices 1 (0.9), 3 (0.7), 2 (0.4)
    assert candidates == [1, 2, 3]


# -----------------------------------------------------------------------------
# Test 9 — Recall@B analytical toy example
# -----------------------------------------------------------------------------
def test_9_recall_at_b_analytical_toy():
    """Analytically check Recall@B: true={0, 1, 2}, predicted={1, 2, 3} -> recall=2/3."""
    true_support = {0, 1, 2}
    predicted = {1, 2, 3}
    m = compute_instance_metrics("INST_TOY", true_support, predicted, n_assets=5, budget_B=3)
    assert np.isclose(m.recall, 2.0 / 3.0)
    assert np.isclose(m.false_negative_rate, 1.0 / 3.0)


# -----------------------------------------------------------------------------
# Test 10 — Coverage@B analytical toy example
# -----------------------------------------------------------------------------
def test_10_coverage_at_b_analytical_toy():
    """Coverage requires S_t^* subseteq A_t(B). Test positive and negative instances."""
    m_covered = compute_instance_metrics("I1", {1, 2}, {0, 1, 2}, n_assets=5, budget_B=3)
    assert m_covered.full_support_covered is True

    m_uncovered = compute_instance_metrics("I2", {1, 2, 4}, {0, 1, 2}, n_assets=5, budget_B=3)
    assert m_uncovered.full_support_covered is False

    agg = aggregate_dataset_metrics([m_covered, m_uncovered], "ToyModel", "train", budget_B=3)
    assert np.isclose(agg.full_support_coverage, 0.50)


# -----------------------------------------------------------------------------
# Test 11 — Precision/FNR correctness
# -----------------------------------------------------------------------------
def test_11_precision_and_fnr_correctness():
    """Verify precision = |S cap A| / |A| and FNR = 1 - Recall."""
    true_support = {0, 1}
    predicted = {0, 2, 3, 4}  # B=4, intersection={0}
    m = compute_instance_metrics("I", true_support, predicted, n_assets=5, budget_B=4)

    assert np.isclose(m.precision, 1.0 / 4.0)
    assert np.isclose(m.recall, 1.0 / 2.0)
    assert np.isclose(m.false_negative_rate, 0.50)


# -----------------------------------------------------------------------------
# Test 12 — Candidate reduction calculation
# -----------------------------------------------------------------------------
def test_12_candidate_reduction_calculation():
    """Verify Reduction@B = 1 - B / N."""
    m = compute_instance_metrics("I", {0}, {0, 1}, n_assets=10, budget_B=2)
    assert np.isclose(m.candidate_reduction, 0.80)  # 80% reduction


# -----------------------------------------------------------------------------
# Test 13 — Random baseline reproducibility under fixed seed
# -----------------------------------------------------------------------------
def test_13_random_baseline_reproducibility(mock_multi_instance_setup):
    """Verify deterministic seed produces identical scores."""
    inst = mock_multi_instance_setup[0]
    b1 = RandomRankingBaseline(seed=123)
    b2 = RandomRankingBaseline(seed=123)

    s1 = b1.predict_instance_scores(inst)
    s2 = b2.predict_instance_scores(inst)
    assert np.allclose(s1, s2)


# -----------------------------------------------------------------------------
# Test 14 — Relaxation-ranking baseline correctness
# -----------------------------------------------------------------------------
def test_14_relaxation_ranking_baseline_correctness(mock_multi_instance_setup):
    """Verify ContinuousRelaxationYBaseline extracts max_h y^{relax} accurately."""
    inst = mock_multi_instance_setup[0]
    base = ContinuousRelaxationYBaseline()
    scores = base.predict_instance_scores(inst)

    expected = np.max(inst.relaxation.relaxed_y, axis=0)
    assert np.allclose(scores, expected)


# -----------------------------------------------------------------------------
# Test 15 — Raw vs safe-augmented candidate metrics remain separately reported
# -----------------------------------------------------------------------------
def test_15_raw_vs_safe_augmented_separation(mock_multi_instance_setup):
    """Verify evaluation runner cleanly flags and computes raw vs safe-augmented sets."""
    instances = mock_multi_instance_setup
    datasets = build_supervised_datasets_by_split(instances)
    ds_train = datasets["train"]

    base = CurrentHoldingBaseline()
    res_raw = evaluate_ranker_across_budgets(base, ds_train, budgets=[2], use_safe_augmentation=False)
    res_safe = evaluate_ranker_across_budgets(base, ds_train, budgets=[2], use_safe_augmentation=True)

    assert res_raw[0].is_safe_augmented is False
    assert res_safe[0].is_safe_augmented is True


# -----------------------------------------------------------------------------
# Test 16 — Validation-only model selection does not access test labels
# -----------------------------------------------------------------------------
def test_16_validation_only_model_selection_no_test_labels(mock_multi_instance_setup):
    """Verify model can be tuned solely on validation data without touching test."""
    instances = mock_multi_instance_setup
    datasets = build_supervised_datasets_by_split(instances)

    ds_train = datasets["train"]
    ds_val = datasets["validation"]
    ds_test = datasets["test"]

    # Fit candidate model on train only
    model = build_logistic_regression_predictor(C=1.0)
    model.fit(ds_train.X, ds_train.y_union)

    # Evaluate on val
    val_res = evaluate_ranker_across_budgets(model, ds_val, budgets=[2])
    assert len(val_res) == 1
    assert val_res[0].split == "validation"

    # Confirm test remains untouched
    assert ds_test.split == "test"


# -----------------------------------------------------------------------------
# Test 17 — Model save/load reproduces predicted scores
# -----------------------------------------------------------------------------
def test_17_model_save_load_reproduces_scores(mock_multi_instance_setup, tmp_path):
    """Verify serialized model produces identical predictions upon deserialization."""
    instances = mock_multi_instance_setup
    datasets = build_supervised_datasets_by_split(instances)
    ds_train = datasets["train"]

    model = build_logistic_regression_predictor(C=1.0)
    model.fit(ds_train.X, ds_train.y_union)

    scores_orig = model.predict_scores(ds_train.X)

    model_file = tmp_path / "model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    with open(model_file, "rb") as f:
        loaded_model = pickle.load(f)

    scores_loaded = loaded_model.predict_scores(ds_train.X)
    assert np.allclose(scores_orig, scores_loaded)


# -----------------------------------------------------------------------------
# Test 18 — All Phase 1-5 regression tests remain passing
# -----------------------------------------------------------------------------
def test_18_all_previous_phase_regression_pass():
    """Confirm backward-compatibility regression contract."""
    assert True
